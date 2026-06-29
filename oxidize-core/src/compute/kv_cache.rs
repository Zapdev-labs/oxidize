use crate::tensor::DType;
use crate::turboquant::TURBOQUANT_BLOCK_SIZE;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Quantization scheme for I8/I16 KV cache storage.
///
/// `Asymmetric` keeps the original per-token (scale, min) layout: one pair of
/// floats per (layer, position). `TurboQuant` switches to per-block symmetric
/// scales using 32-element blocks (see [`crate::turboquant`]). The block scheme
/// is more accurate at long context because each 32-channel slice gets its own
/// scale, at the cost of `blocks_per_token` extra f32 scales per token.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum KvQuantization {
    Asymmetric,
    #[default]
    TurboQuant,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvCacheConfig {
    pub layer_count: usize,
    pub context_size: usize,
    pub head_count: usize,
    pub head_dim: usize,
    pub dtype: DType,
    #[serde(default)]
    pub quantization: KvQuantization,
}

impl KvCacheConfig {
    pub fn token_size(&self) -> usize {
        self.head_count.saturating_mul(self.head_dim)
    }

    pub fn layer_size(&self) -> usize {
        self.context_size.saturating_mul(self.token_size())
    }

    pub fn element_count(&self) -> usize {
        self.layer_count.saturating_mul(self.layer_size())
    }

    /// Number of TurboQuant scale entries per (layer, position) token.
    pub(crate) fn blocks_per_token(&self) -> usize {
        self.token_size().div_ceil(TURBOQUANT_BLOCK_SIZE)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum KvCacheEvictionStrategy {
    SlidingWindow,
    StopAtCapacity,
}

#[derive(Debug, Clone, PartialEq)]
pub enum KvCacheError {
    UnsupportedDType {
        dtype: DType,
    },
    LayerOutOfBounds {
        layer: usize,
        layer_count: usize,
    },
    PositionEvicted {
        position: usize,
        oldest_available: usize,
        newest_available: usize,
    },
    CacheFull {
        requested_position: usize,
        oldest_available: usize,
        newest_available: usize,
        capacity: usize,
    },
    ValueLengthMismatch {
        expected: usize,
        actual: usize,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum KvCachePersistenceError {
    #[error("failed to read or write cache file: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to serialize or deserialize cache: {0}")]
    Serde(#[from] serde_json::Error),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ContinuousBatchError {
    SequenceAlreadyExists {
        sequence_id: u64,
    },
    SequenceNotFound {
        sequence_id: u64,
    },
    SequenceCapacityExceeded {
        max_sequences: usize,
    },
    TokenIndexOutOfBounds {
        sequence_id: u64,
        token_index: usize,
        token_count: usize,
    },
    KvCache(KvCacheError),
}

const KV_CACHE_STORAGE_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
enum KvCacheStorageLayout {
    /// Storage is grouped by layer, then position: `[layer][position][head][head_dim]`.
    LayerMajor,
    /// Legacy serialized storage grouped by position, then layer.
    PositionMajor,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
struct KvCacheStorageMetadata {
    version: u32,
    layout: KvCacheStorageLayout,
}

impl Default for KvCacheStorageMetadata {
    fn default() -> Self {
        // Missing metadata means a legacy persisted cache. Older cache files used
        // position-major storage, while the runtime layout is now layer-major so
        // layer prefixes can be borrowed without copying.
        Self {
            version: 0,
            layout: KvCacheStorageLayout::PositionMajor,
        }
    }
}

fn current_storage_metadata() -> KvCacheStorageMetadata {
    KvCacheStorageMetadata {
        version: KV_CACHE_STORAGE_VERSION,
        layout: KvCacheStorageLayout::LayerMajor,
    }
}

impl From<KvCacheError> for ContinuousBatchError {
    fn from(value: KvCacheError) -> Self {
        Self::KvCache(value)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
enum KvStorage {
    F32(Vec<f32>),
    F16(Vec<u16>),
    Q8 {
        data: Vec<u8>,
        scales: Vec<f32>,
        mins: Vec<f32>,
    },
    Q4 {
        data: Vec<u8>,
        scales: Vec<f32>,
        mins: Vec<f32>,
    },
    /// TurboQuant INT8: per-block (32 channels) symmetric signed scale,
    /// stored as `q + 127` so the on-disk byte is unsigned.
    TurboQ8 {
        data: Vec<u8>,
        scales: Vec<f32>,
    },
    /// TurboQuant INT4: per-block (32 channels) symmetric signed scale,
    /// two 4-bit values packed per byte. Each nibble stores `q + 7`.
    TurboQ4 {
        data: Vec<u8>,
        scales: Vec<f32>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KvCache {
    #[serde(default)]
    storage_metadata: KvCacheStorageMetadata,
    config: KvCacheConfig,
    key: KvStorage,
    value: KvStorage,
    eviction_strategy: KvCacheEvictionStrategy,
    oldest_position: Option<usize>,
    newest_position: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct SequenceState {
    positions: Vec<usize>,
    last_active_step: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContinuousBatchKvCache {
    kv_cache: KvCache,
    max_sequences: usize,
    current_step: usize,
    next_position: usize,
    sequences: HashMap<u64, SequenceState>,
    #[serde(skip)]
    pooled_positions: Vec<Vec<usize>>,
}

impl KvCache {
    pub fn new(config: KvCacheConfig) -> Result<Self, KvCacheError> {
        Self::with_eviction_strategy(config, KvCacheEvictionStrategy::SlidingWindow)
    }

    pub fn with_eviction_strategy(
        config: KvCacheConfig,
        eviction_strategy: KvCacheEvictionStrategy,
    ) -> Result<Self, KvCacheError> {
        let size = config.element_count();
        let token_slots = config.layer_count.saturating_mul(config.context_size);
        let tq_scale_slots = token_slots.saturating_mul(config.blocks_per_token());
        let (key, value) = match (config.dtype, config.quantization) {
            (DType::F32, _) => (
                KvStorage::F32(vec![0.0; size]),
                KvStorage::F32(vec![0.0; size]),
            ),
            (DType::F16, _) => (
                KvStorage::F16(vec![f32_to_f16_bits(0.0); size]),
                KvStorage::F16(vec![f32_to_f16_bits(0.0); size]),
            ),
            (DType::I8, KvQuantization::Asymmetric) => (
                KvStorage::Q8 {
                    data: vec![0; size],
                    scales: vec![0.0; token_slots],
                    mins: vec![0.0; token_slots],
                },
                KvStorage::Q8 {
                    data: vec![0; size],
                    scales: vec![0.0; token_slots],
                    mins: vec![0.0; token_slots],
                },
            ),
            (DType::I16, KvQuantization::Asymmetric) => (
                KvStorage::Q4 {
                    data: vec![0; size.div_ceil(2)],
                    scales: vec![0.0; token_slots],
                    mins: vec![0.0; token_slots],
                },
                KvStorage::Q4 {
                    data: vec![0; size.div_ceil(2)],
                    scales: vec![0.0; token_slots],
                    mins: vec![0.0; token_slots],
                },
            ),
            (DType::I8, KvQuantization::TurboQuant) => (
                KvStorage::TurboQ8 {
                    data: vec![0; size],
                    scales: vec![0.0; tq_scale_slots],
                },
                KvStorage::TurboQ8 {
                    data: vec![0; size],
                    scales: vec![0.0; tq_scale_slots],
                },
            ),
            (DType::I16, KvQuantization::TurboQuant) => (
                KvStorage::TurboQ4 {
                    data: vec![0; size.div_ceil(2)],
                    scales: vec![0.0; tq_scale_slots],
                },
                KvStorage::TurboQ4 {
                    data: vec![0; size.div_ceil(2)],
                    scales: vec![0.0; tq_scale_slots],
                },
            ),
            (dtype, _) => return Err(KvCacheError::UnsupportedDType { dtype }),
        };

        Ok(Self {
            storage_metadata: current_storage_metadata(),
            config,
            key,
            value,
            eviction_strategy,
            oldest_position: None,
            newest_position: None,
        })
    }

    pub fn config(&self) -> KvCacheConfig {
        self.config
    }

    pub fn set(
        &mut self,
        layer: usize,
        position: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<(), KvCacheError> {
        self.validate_write(layer, key, value)?;
        self.validate_write_position(position)?;
        let physical_position = self.physical_position(position);
        write_storage(&mut self.key, &self.config, layer, physical_position, key);
        write_storage(
            &mut self.value,
            &self.config,
            layer,
            physical_position,
            value,
        );
        self.record_position(position);
        Ok(())
    }

    pub fn get_key(
        &self,
        layer: usize,
        position: usize,
        out: &mut [f32],
    ) -> Result<(), KvCacheError> {
        self.validate_read(layer, position, out)?;
        let physical_position = self.physical_position(position);
        read_storage(&self.key, &self.config, layer, physical_position, out);
        Ok(())
    }

    pub fn get_value(
        &self,
        layer: usize,
        position: usize,
        out: &mut [f32],
    ) -> Result<(), KvCacheError> {
        self.validate_read(layer, position, out)?;
        let physical_position = self.physical_position(position);
        read_storage(&self.value, &self.config, layer, physical_position, out);
        Ok(())
    }

    /// Copy all keys for positions [0, seq_len) in a layer into a contiguous output buffer.
    /// Output layout: [position][head][head_dim] — seq_len * token_size elements.
    pub fn copy_layer_keys(
        &self,
        layer: usize,
        seq_len: usize,
        out: &mut [f32],
    ) -> Result<(), KvCacheError> {
        self.validate_layer(layer)?;
        let token_size = self.config.token_size();
        let expected = seq_len.saturating_mul(token_size);
        if out.len() != expected {
            return Err(KvCacheError::ValueLengthMismatch {
                expected,
                actual: out.len(),
            });
        }
        for p in 0..seq_len {
            if !self.position_available(p) {
                let newest_available = self.newest_position.unwrap_or(0);
                let oldest_available = self.oldest_available_position().unwrap_or(0);
                return Err(KvCacheError::PositionEvicted {
                    position: p,
                    oldest_available,
                    newest_available,
                });
            }
            let physical_position = self.physical_position(p);
            let out_off = p * token_size;
            read_storage(
                &self.key,
                &self.config,
                layer,
                physical_position,
                &mut out[out_off..out_off + token_size],
            );
        }
        Ok(())
    }

    /// Copy all values for positions [0, seq_len) in a layer into a contiguous output buffer.
    /// Output layout: [position][head][head_dim] — seq_len * token_size elements.
    pub fn copy_layer_values(
        &self,
        layer: usize,
        seq_len: usize,
        out: &mut [f32],
    ) -> Result<(), KvCacheError> {
        self.validate_layer(layer)?;
        let token_size = self.config.token_size();
        let expected = seq_len.saturating_mul(token_size);
        if out.len() != expected {
            return Err(KvCacheError::ValueLengthMismatch {
                expected,
                actual: out.len(),
            });
        }
        for p in 0..seq_len {
            if !self.position_available(p) {
                let newest_available = self.newest_position.unwrap_or(0);
                let oldest_available = self.oldest_available_position().unwrap_or(0);
                return Err(KvCacheError::PositionEvicted {
                    position: p,
                    oldest_available,
                    newest_available,
                });
            }
            let physical_position = self.physical_position(p);
            let out_off = p * token_size;
            read_storage(
                &self.value,
                &self.config,
                layer,
                physical_position,
                &mut out[out_off..out_off + token_size],
            );
        }
        Ok(())
    }

    /// Borrow all F32 keys for positions [0, seq_len) in a layer when they are
    /// already contiguous in the cache storage.
    ///
    /// Returns `Ok(None)` instead of copying when the cache dtype is not F32, the
    /// requested logical prefix is not fully available, or the sliding-window
    /// mapping has wrapped and the prefix no longer maps to a contiguous storage
    /// range. The returned layout is `[position][head][head_dim]`.
    pub fn f32_layer_key_prefix(
        &self,
        layer: usize,
        seq_len: usize,
    ) -> Result<Option<&[f32]>, KvCacheError> {
        self.f32_layer_prefix(&self.key, layer, seq_len)
    }

    /// Borrow all F32 values for positions [0, seq_len) in a layer when they are
    /// already contiguous in the cache storage.
    ///
    /// See [`Self::f32_layer_key_prefix`] for validity requirements.
    pub fn f32_layer_value_prefix(
        &self,
        layer: usize,
        seq_len: usize,
    ) -> Result<Option<&[f32]>, KvCacheError> {
        self.f32_layer_prefix(&self.value, layer, seq_len)
    }

    /// Borrow all F16 keys (raw half bits) for positions [0, seq_len) in a
    /// layer when they are already contiguous in the cache storage. Same
    /// validity rules as [`Self::f32_layer_key_prefix`], for `DType::F16`.
    pub fn f16_layer_key_prefix(
        &self,
        layer: usize,
        seq_len: usize,
    ) -> Result<Option<&[u16]>, KvCacheError> {
        self.f16_layer_prefix(&self.key, layer, seq_len)
    }

    /// See [`Self::f16_layer_key_prefix`].
    pub fn f16_layer_value_prefix(
        &self,
        layer: usize,
        seq_len: usize,
    ) -> Result<Option<&[u16]>, KvCacheError> {
        self.f16_layer_prefix(&self.value, layer, seq_len)
    }

    pub fn bytes_per_tensor(&self) -> usize {
        match &self.key {
            KvStorage::F32(data) => data.len() * std::mem::size_of::<f32>(),
            KvStorage::F16(data) => data.len() * std::mem::size_of::<u16>(),
            KvStorage::Q8 { data, scales, mins } => {
                data.len()
                    + (scales.len() * std::mem::size_of::<f32>())
                    + (mins.len() * std::mem::size_of::<f32>())
            }
            KvStorage::Q4 { data, scales, mins } => {
                data.len()
                    + (scales.len() * std::mem::size_of::<f32>())
                    + (mins.len() * std::mem::size_of::<f32>())
            }
            KvStorage::TurboQ8 { data, scales } => {
                data.len() + (scales.len() * std::mem::size_of::<f32>())
            }
            KvStorage::TurboQ4 { data, scales } => {
                data.len() + (scales.len() * std::mem::size_of::<f32>())
            }
        }
    }

    /// Quantize the KV cache from F32/F16 to Q8_0 (8-bit asymmetric quantization).
    /// This reduces memory usage by ~4x compared to F32 and ~2x compared to F16.
    pub fn quantize_to_q8(&mut self) -> Result<(), KvCacheError> {
        if matches!(self.config.dtype, DType::I8 | DType::I16) {
            return Ok(()); // Already quantized
        }

        let size = self.config.element_count();
        let token_slots = self
            .config
            .layer_count
            .saturating_mul(self.config.context_size);

        // Convert key storage
        let key_data = match &self.key {
            KvStorage::F32(data) => data.clone(),
            KvStorage::F16(data) => data.iter().map(|&b| f16_bits_to_f32(b)).collect(),
            _ => return Ok(()),
        };

        let mut q8_key_data = vec![0u8; size];
        let mut q8_key_scales = vec![0.0f32; token_slots];
        let mut q8_key_mins = vec![0.0f32; token_slots];

        for layer in 0..self.config.layer_count {
            for position in 0..self.config.context_size {
                let range = token_range(&self.config, layer, position);
                let token_index = token_slot_index(&self.config, layer, position);
                let src = &key_data[range.clone()];
                let (min, max) = min_max(src);
                let scale = if max <= min { 0.0 } else { (max - min) / 255.0 };
                q8_key_scales[token_index] = scale;
                q8_key_mins[token_index] = min;
                if scale == 0.0 {
                    q8_key_data[range].fill(0);
                } else {
                    for (dst, &value) in q8_key_data[range.clone()].iter_mut().zip(src.iter()) {
                        let q = ((value - min) / scale).round().clamp(0.0, 255.0) as u8;
                        *dst = q;
                    }
                }
            }
        }

        // Convert value storage
        let value_data = match &self.value {
            KvStorage::F32(data) => data.clone(),
            KvStorage::F16(data) => data.iter().map(|&b| f16_bits_to_f32(b)).collect(),
            _ => return Ok(()),
        };

        let mut q8_value_data = vec![0u8; size];
        let mut q8_value_scales = vec![0.0f32; token_slots];
        let mut q8_value_mins = vec![0.0f32; token_slots];

        for layer in 0..self.config.layer_count {
            for position in 0..self.config.context_size {
                let range = token_range(&self.config, layer, position);
                let token_index = token_slot_index(&self.config, layer, position);
                let src = &value_data[range.clone()];
                let (min, max) = min_max(src);
                let scale = if max <= min { 0.0 } else { (max - min) / 255.0 };
                q8_value_scales[token_index] = scale;
                q8_value_mins[token_index] = min;
                if scale == 0.0 {
                    q8_value_data[range].fill(0);
                } else {
                    for (dst, &value) in q8_value_data[range.clone()].iter_mut().zip(src.iter()) {
                        let q = ((value - min) / scale).round().clamp(0.0, 255.0) as u8;
                        *dst = q;
                    }
                }
            }
        }

        self.key = KvStorage::Q8 {
            data: q8_key_data,
            scales: q8_key_scales,
            mins: q8_key_mins,
        };
        self.value = KvStorage::Q8 {
            data: q8_value_data,
            scales: q8_value_scales,
            mins: q8_value_mins,
        };
        self.config.dtype = DType::I8;
        self.config.quantization = KvQuantization::Asymmetric;

        Ok(())
    }

    /// Returns the compression ratio compared to F32 storage.
    pub fn compression_ratio(&self) -> f32 {
        let f32_size = self.config.element_count() * std::mem::size_of::<f32>();
        let current_size = self.bytes_per_tensor();
        if current_size == 0 {
            1.0
        } else {
            f32_size as f32 / current_size as f32
        }
    }

    pub fn availability_window(&self) -> Option<(usize, usize)> {
        Some((self.oldest_available_position()?, self.newest_position?))
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), KvCachePersistenceError> {
        let payload = serde_json::to_vec(self)?;
        std::fs::write(path, payload)?;
        Ok(())
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, KvCachePersistenceError> {
        let payload = std::fs::read(path)?;
        let mut cache: Self = serde_json::from_slice(&payload)?;
        cache.migrate_legacy_storage_layout();
        Ok(cache)
    }

    fn migrate_legacy_storage_layout(&mut self) {
        if self.storage_metadata.layout != KvCacheStorageLayout::PositionMajor {
            self.storage_metadata = current_storage_metadata();
            return;
        }

        migrate_storage_from_position_major(&mut self.key, &self.config);
        migrate_storage_from_position_major(&mut self.value, &self.config);
        self.storage_metadata = current_storage_metadata();
    }

    fn validate_write(&self, layer: usize, key: &[f32], value: &[f32]) -> Result<(), KvCacheError> {
        self.validate_layer(layer)?;
        let expected = self.config.token_size();
        if key.len() != expected {
            return Err(KvCacheError::ValueLengthMismatch {
                expected,
                actual: key.len(),
            });
        }
        if value.len() != expected {
            return Err(KvCacheError::ValueLengthMismatch {
                expected,
                actual: value.len(),
            });
        }
        Ok(())
    }

    fn validate_read(
        &self,
        layer: usize,
        position: usize,
        out: &[f32],
    ) -> Result<(), KvCacheError> {
        self.validate_layer(layer)?;
        if !self.position_available(position) {
            let newest_available = self.newest_position.unwrap_or(0);
            let oldest_available = self.oldest_available_position().unwrap_or(0);
            return Err(KvCacheError::PositionEvicted {
                position,
                oldest_available,
                newest_available,
            });
        }
        let expected = self.config.token_size();
        if out.len() != expected {
            return Err(KvCacheError::ValueLengthMismatch {
                expected,
                actual: out.len(),
            });
        }
        Ok(())
    }

    fn validate_layer(&self, layer: usize) -> Result<(), KvCacheError> {
        if layer >= self.config.layer_count {
            return Err(KvCacheError::LayerOutOfBounds {
                layer,
                layer_count: self.config.layer_count,
            });
        }
        Ok(())
    }

    fn f32_layer_prefix<'a>(
        &self,
        storage: &'a KvStorage,
        layer: usize,
        seq_len: usize,
    ) -> Result<Option<&'a [f32]>, KvCacheError> {
        self.validate_layer(layer)?;
        if seq_len == 0 {
            return match storage {
                KvStorage::F32(data) => Ok(Some(&data[0..0])),
                _ => Ok(None),
            };
        }
        if self.config.dtype != DType::F32 || !self.prefix_is_contiguous_and_available(seq_len) {
            return Ok(None);
        }

        let KvStorage::F32(data) = storage else {
            return Ok(None);
        };
        let token_size = self.config.token_size();
        let start = token_range(&self.config, layer, 0).start;
        let end = start + seq_len.saturating_mul(token_size);
        Ok(data.get(start..end))
    }

    fn f16_layer_prefix<'a>(
        &self,
        storage: &'a KvStorage,
        layer: usize,
        seq_len: usize,
    ) -> Result<Option<&'a [u16]>, KvCacheError> {
        self.validate_layer(layer)?;
        if seq_len == 0 {
            return match storage {
                KvStorage::F16(data) => Ok(Some(&data[0..0])),
                _ => Ok(None),
            };
        }
        if self.config.dtype != DType::F16 || !self.prefix_is_contiguous_and_available(seq_len) {
            return Ok(None);
        }

        let KvStorage::F16(data) = storage else {
            return Ok(None);
        };
        let token_size = self.config.token_size();
        let start = token_range(&self.config, layer, 0).start;
        let end = start + seq_len.saturating_mul(token_size);
        Ok(data.get(start..end))
    }

    fn prefix_is_contiguous_and_available(&self, seq_len: usize) -> bool {
        if seq_len > self.config.context_size {
            return false;
        }
        let Some(oldest) = self.oldest_available_position() else {
            return false;
        };
        let Some(newest) = self.newest_position else {
            return false;
        };
        oldest == 0 && newest >= seq_len - 1
    }

    fn position_available(&self, position: usize) -> bool {
        match (self.oldest_available_position(), self.newest_position) {
            (Some(oldest), Some(newest)) => (oldest..=newest).contains(&position),
            _ => false,
        }
    }

    fn oldest_available_position(&self) -> Option<usize> {
        self.oldest_position
    }

    fn validate_write_position(&self, position: usize) -> Result<(), KvCacheError> {
        if self.eviction_strategy != KvCacheEvictionStrategy::StopAtCapacity {
            return Ok(());
        }
        let (Some(oldest), Some(newest)) = (self.oldest_position, self.newest_position) else {
            return Ok(());
        };
        if position <= newest {
            return Ok(());
        }
        let used = newest.saturating_sub(oldest).saturating_add(1);
        if used >= self.config.context_size {
            return Err(KvCacheError::CacheFull {
                requested_position: position,
                oldest_available: oldest,
                newest_available: newest,
                capacity: self.config.context_size,
            });
        }
        Ok(())
    }

    /// Drop KV entries after `position` (inclusive). Used to roll back speculative verify runs.
    pub fn rewind_to(&mut self, position: usize) -> Result<(), KvCacheError> {
        match self.newest_position {
            None => Ok(()),
            Some(newest) if position > newest => Err(KvCacheError::PositionEvicted {
                position,
                oldest_available: self.oldest_available_position().unwrap_or(0),
                newest_available: newest,
            }),
            Some(newest) => {
                if let Some(oldest) = self.oldest_available_position()
                    && position < oldest
                {
                    return Err(KvCacheError::PositionEvicted {
                        position,
                        oldest_available: oldest,
                        newest_available: newest,
                    });
                }
                self.newest_position = Some(position);
                Ok(())
            }
        }
    }

    fn record_position(&mut self, position: usize) {
        let newest = self
            .newest_position
            .map_or(position, |current| current.max(position));
        self.newest_position = Some(newest);
        let oldest = match self.eviction_strategy {
            KvCacheEvictionStrategy::SlidingWindow => newest
                .saturating_add(1)
                .saturating_sub(self.config.context_size),
            KvCacheEvictionStrategy::StopAtCapacity => self.oldest_position.unwrap_or(position),
        };
        self.oldest_position = Some(oldest);
    }

    fn physical_position(&self, position: usize) -> usize {
        position % self.config.context_size
    }
}

impl ContinuousBatchKvCache {
    pub fn new(kv_cache: KvCache, max_sequences: usize) -> Self {
        Self {
            kv_cache,
            max_sequences,
            current_step: 0,
            next_position: 0,
            sequences: HashMap::new(),
            pooled_positions: Vec::new(),
        }
    }

    pub fn begin_step(&mut self) {
        self.current_step = self.current_step.saturating_add(1);
    }

    pub fn add_sequence(&mut self, sequence_id: u64) -> Result<(), ContinuousBatchError> {
        if self.sequences.contains_key(&sequence_id) {
            return Err(ContinuousBatchError::SequenceAlreadyExists { sequence_id });
        }
        if self.sequences.len() >= self.max_sequences {
            return Err(ContinuousBatchError::SequenceCapacityExceeded {
                max_sequences: self.max_sequences,
            });
        }
        let positions = self.pooled_positions.pop().unwrap_or_default();
        self.sequences.insert(
            sequence_id,
            SequenceState {
                positions,
                last_active_step: self.current_step,
            },
        );
        Ok(())
    }

    pub fn remove_sequence(&mut self, sequence_id: u64) -> Result<(), ContinuousBatchError> {
        let state = self
            .sequences
            .remove(&sequence_id)
            .ok_or(ContinuousBatchError::SequenceNotFound { sequence_id })?;
        self.recycle_positions(state.positions);
        Ok(())
    }

    pub fn evict_inactive_sequences(&mut self, max_idle_steps: usize) {
        let eviction_step = self.current_step.saturating_sub(max_idle_steps);
        let evicted_ids = self
            .sequences
            .iter()
            .filter_map(|(sequence_id, state)| {
                (state.last_active_step < eviction_step).then_some(*sequence_id)
            })
            .collect::<Vec<_>>();
        for sequence_id in evicted_ids {
            if let Some(state) = self.sequences.remove(&sequence_id) {
                self.recycle_positions(state.positions);
            }
        }
    }

    pub fn append_token(
        &mut self,
        sequence_id: u64,
        layer: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<usize, ContinuousBatchError> {
        let state = self
            .sequences
            .get_mut(&sequence_id)
            .ok_or(ContinuousBatchError::SequenceNotFound { sequence_id })?;
        let position = self.next_position;
        self.kv_cache.set(layer, position, key, value)?;
        state.positions.push(position);
        state.last_active_step = self.current_step;
        self.next_position = self.next_position.saturating_add(1);
        Ok(position)
    }

    pub fn get_sequence_key(
        &self,
        sequence_id: u64,
        layer: usize,
        token_index: usize,
        out: &mut [f32],
    ) -> Result<(), ContinuousBatchError> {
        let position = self.position_for(sequence_id, token_index)?;
        self.kv_cache.get_key(layer, position, out)?;
        Ok(())
    }

    pub fn get_sequence_value(
        &self,
        sequence_id: u64,
        layer: usize,
        token_index: usize,
        out: &mut [f32],
    ) -> Result<(), ContinuousBatchError> {
        let position = self.position_for(sequence_id, token_index)?;
        self.kv_cache.get_value(layer, position, out)?;
        Ok(())
    }

    pub fn sequence_count(&self) -> usize {
        self.sequences.len()
    }

    pub fn cache(&self) -> &KvCache {
        &self.kv_cache
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), KvCachePersistenceError> {
        let payload = serde_json::to_vec(self)?;
        std::fs::write(path, payload)?;
        Ok(())
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, KvCachePersistenceError> {
        let payload = std::fs::read(path)?;
        let mut cache: Self = serde_json::from_slice(&payload)?;
        cache.kv_cache.migrate_legacy_storage_layout();
        Ok(cache)
    }

    fn position_for(
        &self,
        sequence_id: u64,
        token_index: usize,
    ) -> Result<usize, ContinuousBatchError> {
        let state = self
            .sequences
            .get(&sequence_id)
            .ok_or(ContinuousBatchError::SequenceNotFound { sequence_id })?;
        state.positions.get(token_index).copied().ok_or(
            ContinuousBatchError::TokenIndexOutOfBounds {
                sequence_id,
                token_index,
                token_count: state.positions.len(),
            },
        )
    }

    fn recycle_positions(&mut self, mut positions: Vec<usize>) {
        if self.pooled_positions.len() >= self.max_sequences {
            return;
        }
        positions.clear();
        self.pooled_positions.push(positions);
    }

    #[cfg(test)]
    fn pooled_position_buffer_count(&self) -> usize {
        self.pooled_positions.len()
    }
}

fn write_storage(
    storage: &mut KvStorage,
    config: &KvCacheConfig,
    layer: usize,
    position: usize,
    src: &[f32],
) {
    let range = token_range(config, layer, position);
    let token_index = token_slot_index(config, layer, position);
    match storage {
        KvStorage::F32(data) => data[range].copy_from_slice(src),
        KvStorage::F16(data) => {
            for (dst, value) in data[range].iter_mut().zip(src.iter()) {
                *dst = f32_to_f16_bits(*value);
            }
        }
        KvStorage::Q8 { data, scales, mins } => {
            let (min, max) = min_max(src);
            let scale = if max <= min { 0.0 } else { (max - min) / 255.0 };
            scales[token_index] = scale;
            mins[token_index] = min;
            if scale == 0.0 {
                data[range].fill(0);
            } else {
                for (dst, value) in data[range].iter_mut().zip(src.iter()) {
                    let q = ((*value - min) / scale).round().clamp(0.0, 255.0) as u8;
                    *dst = q;
                }
            }
        }
        KvStorage::Q4 { data, scales, mins } => {
            let (min, max) = min_max(src);
            let scale = if max <= min { 0.0 } else { (max - min) / 15.0 };
            scales[token_index] = scale;
            mins[token_index] = min;
            let quantize = |value: f32| -> u8 {
                if scale == 0.0 {
                    0
                } else {
                    ((value - min) / scale).round().clamp(0.0, 15.0) as u8
                }
            };
            let packed_start = range.start / 2;
            if range.start.is_multiple_of(2) {
                for (pair_index, pair) in src.chunks(2).enumerate() {
                    let low = quantize(pair[0]);
                    let high = if pair.len() == 2 {
                        quantize(pair[1])
                    } else {
                        (data[packed_start + pair_index] >> 4) & 0x0F
                    };
                    data[packed_start + pair_index] = (high << 4) | (low & 0x0F);
                }
            } else {
                let first_high = quantize(src[0]) << 4;
                data[packed_start] = (data[packed_start] & 0x0F) | first_high;
                for (pair_index, pair) in src[1..].chunks(2).enumerate() {
                    let low = quantize(pair[0]);
                    let high = if pair.len() == 2 {
                        quantize(pair[1])
                    } else {
                        0
                    };
                    data[packed_start + 1 + pair_index] = (high << 4) | (low & 0x0F);
                }
            }
        }
        KvStorage::TurboQ8 { data, scales } => {
            write_turboquant_token::<8>(data, scales, config, range.start, token_index, src);
        }
        KvStorage::TurboQ4 { data, scales } => {
            write_turboquant_token::<4>(data, scales, config, range.start, token_index, src);
        }
    }
}

/// Encode one token into TurboQuant block-quantized form.
///
/// `BITS` is 4 or 8. Each 32-element block of `src` produces one scale
/// (`max_abs / max_val`) and a stream of unsigned codes `q + max_val` written
/// into `data` starting at element offset `data_element_start`. For `BITS=4`
/// the codes are packed two-per-byte (low nibble first); for `BITS=8` they are
/// written one-byte-per-code.
///
/// Requires `data_element_start` to be even when `BITS=4` (guaranteed by
/// `token_range` because `token_size` is always head_count * head_dim, both even).
fn write_turboquant_token<const BITS: u8>(
    data: &mut [u8],
    scales: &mut [f32],
    config: &KvCacheConfig,
    data_element_start: usize,
    token_index: usize,
    src: &[f32],
) {
    let block_size = TURBOQUANT_BLOCK_SIZE;
    let max_val = ((1u32 << (BITS - 1)) - 1) as f32;
    let blocks_per_token = config.blocks_per_token();
    let scale_off = token_index * blocks_per_token;

    for b in 0..blocks_per_token {
        let s = b * block_size;
        let e = (s + block_size).min(src.len());
        let chunk = &src[s..e];
        let mut max_abs = 0.0_f32;
        for &v in chunk {
            max_abs = max_abs.max(v.abs());
        }
        let scale = if max_abs > 0.0 {
            max_abs / max_val
        } else {
            1.0
        };
        scales[scale_off + b] = scale;

        if BITS == 4 {
            let packed_start = (data_element_start + s) / 2;
            for (i, &v) in chunk.iter().enumerate() {
                let q = (v / scale).round().clamp(-max_val, max_val) as i32;
                let uq = (q + max_val as i32) as u8 & 0x0F;
                let byte_idx = packed_start + (i / 2);
                if i % 2 == 0 {
                    data[byte_idx] = (data[byte_idx] & 0xF0) | uq;
                } else {
                    data[byte_idx] = (data[byte_idx] & 0x0F) | (uq << 4);
                }
            }
        } else {
            let byte_start = data_element_start + s;
            for (i, &v) in chunk.iter().enumerate() {
                let q = (v / scale).round().clamp(-max_val, max_val) as i32;
                data[byte_start + i] = (q + max_val as i32) as u8;
            }
        }
    }
}

fn read_storage(
    storage: &KvStorage,
    config: &KvCacheConfig,
    layer: usize,
    position: usize,
    dst: &mut [f32],
) {
    let range = token_range(config, layer, position);
    let token_index = token_slot_index(config, layer, position);
    match storage {
        KvStorage::F32(data) => dst.copy_from_slice(&data[range]),
        KvStorage::F16(data) => {
            for (out, value) in dst.iter_mut().zip(data[range].iter()) {
                *out = f16_bits_to_f32(*value);
            }
        }
        KvStorage::Q8 { data, scales, mins } => {
            let scale = scales[token_index];
            let min = mins[token_index];
            for (out, value) in dst.iter_mut().zip(data[range].iter()) {
                *out = (*value as f32) * scale + min;
            }
        }
        KvStorage::Q4 { data, scales, mins } => {
            let scale = scales[token_index];
            let min = mins[token_index];
            let packed_start = range.start / 2;
            if range.start.is_multiple_of(2) {
                for (pair_index, pair) in dst.chunks_mut(2).enumerate() {
                    let byte = data[packed_start + pair_index];
                    pair[0] = ((byte & 0x0F) as f32) * scale + min;
                    if pair.len() == 2 {
                        pair[1] = (((byte >> 4) & 0x0F) as f32) * scale + min;
                    }
                }
            } else {
                let first_byte = data[packed_start];
                dst[0] = (((first_byte >> 4) & 0x0F) as f32) * scale + min;
                for (pair_index, pair) in dst[1..].chunks_mut(2).enumerate() {
                    let byte = data[packed_start + 1 + pair_index];
                    pair[0] = ((byte & 0x0F) as f32) * scale + min;
                    if pair.len() == 2 {
                        pair[1] = (((byte >> 4) & 0x0F) as f32) * scale + min;
                    }
                }
            }
        }
        KvStorage::TurboQ8 { data, scales } => {
            read_turboquant_token::<8>(data, scales, config, range.start, token_index, dst);
        }
        KvStorage::TurboQ4 { data, scales } => {
            read_turboquant_token::<4>(data, scales, config, range.start, token_index, dst);
        }
    }
}

fn read_turboquant_token<const BITS: u8>(
    data: &[u8],
    scales: &[f32],
    config: &KvCacheConfig,
    data_element_start: usize,
    token_index: usize,
    dst: &mut [f32],
) {
    let block_size = TURBOQUANT_BLOCK_SIZE;
    let max_val = ((1u32 << (BITS - 1)) - 1) as f32;
    let blocks_per_token = config.blocks_per_token();
    let scale_off = token_index * blocks_per_token;

    for b in 0..blocks_per_token {
        let s = b * block_size;
        let e = (s + block_size).min(dst.len());
        let scale = scales[scale_off + b];
        if BITS == 4 {
            let packed_start = (data_element_start + s) / 2;
            for i in 0..(e - s) {
                let byte = data[packed_start + (i / 2)];
                let nibble = if i % 2 == 0 {
                    byte & 0x0F
                } else {
                    (byte >> 4) & 0x0F
                };
                dst[s + i] = (nibble as f32 - max_val) * scale;
            }
        } else {
            let byte_start = data_element_start + s;
            for i in 0..(e - s) {
                dst[s + i] = (data[byte_start + i] as f32 - max_val) * scale;
            }
        }
    }
}

fn migrate_storage_from_position_major(storage: &mut KvStorage, config: &KvCacheConfig) {
    match storage {
        KvStorage::F32(data) => migrate_flat_elements_from_position_major(data, config),
        KvStorage::F16(data) => migrate_flat_elements_from_position_major(data, config),
        KvStorage::Q8 { data, scales, mins } => {
            migrate_flat_elements_from_position_major(data, config);
            migrate_token_slots_from_position_major(scales, config);
            migrate_token_slots_from_position_major(mins, config);
        }
        KvStorage::Q4 { data, scales, mins } => {
            migrate_q4_elements_from_position_major(data, config);
            migrate_token_slots_from_position_major(scales, config);
            migrate_token_slots_from_position_major(mins, config);
        }
        // TurboQuant variants were introduced after the layer-major migration,
        // so no legacy position-major data ever exists for them.
        KvStorage::TurboQ8 { .. } | KvStorage::TurboQ4 { .. } => {}
    }
}

fn migrate_flat_elements_from_position_major<T: Copy>(data: &mut [T], config: &KvCacheConfig) {
    let token_size = config.token_size();
    let expected = config.element_count();
    if data.len() != expected || token_size == 0 {
        return;
    }

    let old = data.to_owned();
    for layer in 0..config.layer_count {
        for position in 0..config.context_size {
            let old_start = (position * config.layer_count + layer) * token_size;
            let new_start = token_range(config, layer, position).start;
            data[new_start..new_start + token_size]
                .copy_from_slice(&old[old_start..old_start + token_size]);
        }
    }
}

fn migrate_token_slots_from_position_major<T: Copy>(data: &mut [T], config: &KvCacheConfig) {
    let expected = config.layer_count.saturating_mul(config.context_size);
    if data.len() != expected {
        return;
    }

    let old = data.to_owned();
    for layer in 0..config.layer_count {
        for position in 0..config.context_size {
            let old_index = position * config.layer_count + layer;
            let new_index = token_slot_index(config, layer, position);
            data[new_index] = old[old_index];
        }
    }
}

fn migrate_q4_elements_from_position_major(data: &mut [u8], config: &KvCacheConfig) {
    let expected_elements = config.element_count();
    if data.len() != expected_elements.div_ceil(2) {
        return;
    }

    let old_nibbles = unpack_q4_nibbles(data, expected_elements);
    let mut new_nibbles = vec![0_u8; expected_elements];
    let token_size = config.token_size();
    for layer in 0..config.layer_count {
        for position in 0..config.context_size {
            let old_start = (position * config.layer_count + layer) * token_size;
            let new_start = token_range(config, layer, position).start;
            new_nibbles[new_start..new_start + token_size]
                .copy_from_slice(&old_nibbles[old_start..old_start + token_size]);
        }
    }
    pack_q4_nibbles(&new_nibbles, data);
}

fn unpack_q4_nibbles(data: &[u8], element_count: usize) -> Vec<u8> {
    let mut nibbles = Vec::with_capacity(element_count);
    for byte in data {
        nibbles.push(byte & 0x0F);
        if nibbles.len() < element_count {
            nibbles.push((byte >> 4) & 0x0F);
        }
    }
    nibbles
}

fn pack_q4_nibbles(nibbles: &[u8], data: &mut [u8]) {
    for (index, byte) in data.iter_mut().enumerate() {
        let low = nibbles.get(index * 2).copied().unwrap_or(0) & 0x0F;
        let high = nibbles.get(index * 2 + 1).copied().unwrap_or(0) & 0x0F;
        *byte = (high << 4) | low;
    }
}

fn token_range(config: &KvCacheConfig, layer: usize, position: usize) -> std::ops::Range<usize> {
    let token_size = config.token_size();
    let offset = token_slot_index(config, layer, position) * token_size;
    offset..offset + token_size
}

fn token_slot_index(config: &KvCacheConfig, layer: usize, position: usize) -> usize {
    layer * config.context_size + position
}

fn min_max(values: &[f32]) -> (f32, f32) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for value in values {
        min = min.min(*value);
        max = max.max(*value);
    }
    (min, max)
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x03FF) as u32;

    let f32_bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            let mut frac_norm = frac;
            let mut e = -14_i32;
            while (frac_norm & 0x0400) == 0 {
                frac_norm <<= 1;
                e -= 1;
            }
            frac_norm &= 0x03FF;
            (sign << 31) | (((e + 127) as u32) << 23) | (frac_norm << 13)
        }
    } else if exp == 0x1F {
        (sign << 31) | 0x7F80_0000 | (frac << 13)
    } else {
        let e = exp as i32 - 15 + 127;
        (sign << 31) | ((e as u32) << 23) | (frac << 13)
    };

    f32::from_bits(f32_bits)
}

pub(crate) fn f32_to_f16_bits(value: f32) -> u16 {
    let x = value.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xFF) as i32;
    let frac = x & 0x7F_FFFF;

    if exp == 0xFF {
        if frac == 0 {
            return sign | 0x7C00;
        }
        let nan = (frac >> 13) as u16;
        return sign | 0x7C00 | nan | 1;
    }

    let exp16 = exp - 127 + 15;
    if exp16 <= 0 {
        if exp16 < -10 {
            return sign;
        }
        let mant = frac | 0x80_0000;
        let shift = (14 - exp16) as u32;
        let mut half_frac = (mant >> shift) as u16;
        if ((mant >> (shift - 1)) & 1) != 0 {
            half_frac = half_frac.wrapping_add(1);
        }
        return sign | half_frac;
    }

    if exp16 >= 0x1F {
        return sign | 0x7C00;
    }

    let mut half_exp = (exp16 as u16) << 10;
    let mut half_frac = (frac >> 13) as u16;
    if (frac & 0x1000) != 0 {
        half_frac = half_frac.wrapping_add(1);
        if (half_frac & 0x0400) != 0 {
            half_frac = 0;
            half_exp = half_exp.wrapping_add(0x0400);
            if half_exp >= 0x7C00 {
                return sign | 0x7C00;
            }
        }
    }
    sign | half_exp | (half_frac & 0x03FF)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::flash_attention_decode_f32;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_path(prefix: &str) -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{unique}.json"))
    }

    #[test]
    fn allocates_kv_cache_with_requested_dtype() {
        let f32_cache = KvCache::new(KvCacheConfig {
            layer_count: 2,
            context_size: 4,
            head_count: 2,
            head_dim: 8,
            dtype: DType::F32,
            quantization: Default::default(),
        })
        .expect("f32 kv cache should be supported");
        let f16_cache = KvCache::new(KvCacheConfig {
            layer_count: 2,
            context_size: 4,
            head_count: 2,
            head_dim: 8,
            dtype: DType::F16,
            quantization: Default::default(),
        })
        .expect("f16 kv cache should be supported");
        let q8_cache = KvCache::new(KvCacheConfig {
            layer_count: 2,
            context_size: 4,
            head_count: 2,
            head_dim: 8,
            dtype: DType::I8,
            quantization: Default::default(),
        })
        .expect("q8 kv cache should be supported");
        let q4_cache = KvCache::new(KvCacheConfig {
            layer_count: 2,
            context_size: 4,
            head_count: 2,
            head_dim: 8,
            dtype: DType::I16,
            quantization: Default::default(),
        })
        .expect("q4 kv cache should be supported");

        assert_eq!(f32_cache.bytes_per_tensor(), 2 * 4 * 2 * 8 * 4);
        assert_eq!(f16_cache.bytes_per_tensor(), 2 * 4 * 2 * 8 * 2);
        let token_slots: usize = 2 * 4;
        let token_size: usize = 2 * 8;
        let turbo_blocks_per_token = token_size.div_ceil(TURBOQUANT_BLOCK_SIZE);
        let turbo_scale_bytes = token_slots * turbo_blocks_per_token * std::mem::size_of::<f32>();

        assert_eq!(q8_cache.bytes_per_tensor(), (token_slots * token_size) + turbo_scale_bytes);
        assert_eq!(
            q4_cache.bytes_per_tensor(),
            (token_slots * token_size).div_ceil(2) + turbo_scale_bytes
        );
    }

    #[test]
    fn stores_and_reads_back_f32_kv_vectors() {
        let mut cache = KvCache::new(KvCacheConfig {
            layer_count: 1,
            context_size: 3,
            head_count: 1,
            head_dim: 4,
            dtype: DType::F32,
            quantization: Default::default(),
        })
        .expect("f32 kv cache should be supported");

        let key = [0.25_f32, -1.0, 2.0, 3.5];
        let value = [1.5_f32, -0.5, 0.75, 4.0];
        cache
            .set(0, 2, &key, &value)
            .expect("writing kv entry should succeed");

        let mut loaded_key = [0.0_f32; 4];
        let mut loaded_value = [0.0_f32; 4];
        cache
            .get_key(0, 2, &mut loaded_key)
            .expect("reading key should succeed");
        cache
            .get_value(0, 2, &mut loaded_value)
            .expect("reading value should succeed");

        assert_eq!(loaded_key, key);
        assert_eq!(loaded_value, value);
    }

    #[test]
    fn borrows_contiguous_f32_layer_prefixes() {
        let mut cache = KvCache::new(KvCacheConfig {
            layer_count: 2,
            context_size: 3,
            head_count: 1,
            head_dim: 2,
            dtype: DType::F32,
            quantization: Default::default(),
        })
        .expect("f32 kv cache should be supported");

        cache
            .set(0, 0, &[1.0, 2.0], &[10.0, 20.0])
            .expect("first layer position 0 should write");
        cache
            .set(1, 0, &[3.0, 4.0], &[30.0, 40.0])
            .expect("second layer position 0 should write");
        cache
            .set(0, 1, &[5.0, 6.0], &[50.0, 60.0])
            .expect("first layer position 1 should write");
        cache
            .set(1, 1, &[7.0, 8.0], &[70.0, 80.0])
            .expect("second layer position 1 should write");

        let layer_zero_keys = cache
            .f32_layer_key_prefix(0, 2)
            .expect("borrow should validate")
            .expect("f32 prefix should be contiguous");
        let layer_one_values = cache
            .f32_layer_value_prefix(1, 2)
            .expect("borrow should validate")
            .expect("f32 prefix should be contiguous");

        assert_eq!(layer_zero_keys, &[1.0, 2.0, 5.0, 6.0]);
        assert_eq!(layer_one_values, &[30.0, 40.0, 70.0, 80.0]);
    }

    #[test]
    fn borrowed_layer_prefix_matches_copy_and_flash_attention_output() {
        let mut cache = KvCache::new(KvCacheConfig {
            layer_count: 1,
            context_size: 3,
            head_count: 2,
            head_dim: 2,
            dtype: DType::F32,
            quantization: Default::default(),
        })
        .expect("f32 kv cache should be supported");

        cache
            .set(0, 0, &[1.0, 2.0, 3.0, 4.0], &[0.1, 0.2, 0.3, 0.4])
            .expect("position 0 should write");
        cache
            .set(0, 1, &[5.0, 6.0, 7.0, 8.0], &[0.5, 0.6, 0.7, 0.8])
            .expect("position 1 should write");
        cache
            .set(0, 2, &[9.0, 10.0, 11.0, 12.0], &[0.9, 1.0, 1.1, 1.2])
            .expect("position 2 should write");

        let borrowed_keys = cache
            .f32_layer_key_prefix(0, 3)
            .expect("borrow should validate")
            .expect("keys should be borrowable");
        let borrowed_values = cache
            .f32_layer_value_prefix(0, 3)
            .expect("borrow should validate")
            .expect("values should be borrowable");

        let mut copied_keys = vec![0.0_f32; 12];
        let mut copied_values = vec![0.0_f32; 12];
        cache
            .copy_layer_keys(0, 3, &mut copied_keys)
            .expect("keys should copy");
        cache
            .copy_layer_values(0, 3, &mut copied_values)
            .expect("values should copy");

        assert_eq!(borrowed_keys, copied_keys.as_slice());
        assert_eq!(borrowed_values, copied_values.as_slice());

        let query = [0.25_f32, -0.5];
        let mut borrowed_output = [0.0_f32; 2];
        let mut copied_output = [0.0_f32; 2];
        flash_attention_decode_f32(
            &query,
            borrowed_keys,
            borrowed_values,
            3,
            2,
            4,
            1,
            &mut borrowed_output,
        )
        .expect("borrowed cache should be valid flash attention input");
        flash_attention_decode_f32(
            &query,
            &copied_keys,
            &copied_values,
            3,
            2,
            4,
            1,
            &mut copied_output,
        )
        .expect("copied cache should be valid flash attention input");

        assert_eq!(borrowed_output, copied_output);
    }

    #[test]
    fn refuses_to_borrow_when_f32_prefix_has_wrapped() {
        let mut cache = KvCache::new(KvCacheConfig {
            layer_count: 1,
            context_size: 2,
            head_count: 1,
            head_dim: 2,
            dtype: DType::F32,
            quantization: Default::default(),
        })
        .expect("f32 kv cache should be supported");

        cache
            .set(0, 0, &[1.0, 2.0], &[3.0, 4.0])
            .expect("position 0 should write");
        cache
            .set(0, 1, &[5.0, 6.0], &[7.0, 8.0])
            .expect("position 1 should write");
        cache
            .set(0, 2, &[9.0, 10.0], &[11.0, 12.0])
            .expect("position 2 should wrap");

        assert_eq!(
            cache
                .f32_layer_key_prefix(0, 2)
                .expect("borrow should validate"),
            None
        );
    }

    #[test]
    fn refuses_to_borrow_non_f32_layer_prefixes() {
        let mut cache = KvCache::new(KvCacheConfig {
            layer_count: 1,
            context_size: 2,
            head_count: 1,
            head_dim: 2,
            dtype: DType::F16,
            quantization: Default::default(),
        })
        .expect("f16 kv cache should be supported");

        cache
            .set(0, 0, &[1.0, 2.0], &[3.0, 4.0])
            .expect("position 0 should write");

        assert_eq!(
            cache
                .f32_layer_key_prefix(0, 1)
                .expect("borrow should validate"),
            None
        );
    }

    #[test]
    fn stores_f16_kv_vectors_with_expected_quantization_error() {
        let mut cache = KvCache::new(KvCacheConfig {
            layer_count: 1,
            context_size: 1,
            head_count: 1,
            head_dim: 4,
            dtype: DType::F16,
            quantization: Default::default(),
        })
        .expect("f16 kv cache should be supported");

        let key = [0.3333_f32, -1.111, 123.75, 0.00003];
        let value = [2.5_f32, -0.75, 0.125, 9.0];
        cache
            .set(0, 0, &key, &value)
            .expect("writing kv entry should succeed");

        let mut loaded_key = [0.0_f32; 4];
        let mut loaded_value = [0.0_f32; 4];
        cache
            .get_key(0, 0, &mut loaded_key)
            .expect("reading key should succeed");
        cache
            .get_value(0, 0, &mut loaded_value)
            .expect("reading value should succeed");

        for (actual, expected) in loaded_key.iter().zip(key.iter()) {
            assert!((actual - expected).abs() < 1e-2);
        }
        for (actual, expected) in loaded_value.iter().zip(value.iter()) {
            assert!((actual - expected).abs() < 1e-3);
        }
    }

    #[test]
    fn stores_i8_kv_vectors_with_quantization_error_bound() {
        let mut cache = KvCache::new(KvCacheConfig {
            layer_count: 1,
            context_size: 1,
            head_count: 1,
            head_dim: 8,
            dtype: DType::I8,
            quantization: Default::default(),
        })
        .expect("i8 kv cache should be supported");

        let key = [-1.0_f32, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0];
        let value = [0.15_f32, 0.3, 0.45, 0.6, 0.75, -0.15, -0.3, -0.45];
        cache
            .set(0, 0, &key, &value)
            .expect("writing kv entry should succeed");

        let mut loaded_key = [0.0_f32; 8];
        let mut loaded_value = [0.0_f32; 8];
        cache
            .get_key(0, 0, &mut loaded_key)
            .expect("reading key should succeed");
        cache
            .get_value(0, 0, &mut loaded_value)
            .expect("reading value should succeed");

        for (actual, expected) in loaded_key.iter().zip(key.iter()) {
            assert!((actual - expected).abs() < 0.01);
        }
        for (actual, expected) in loaded_value.iter().zip(value.iter()) {
            assert!((actual - expected).abs() < 0.01);
        }
    }

    #[test]
    fn stores_i16_kv_vectors_with_4bit_quantization_error_bound() {
        let mut cache = KvCache::new(KvCacheConfig {
            layer_count: 1,
            context_size: 1,
            head_count: 1,
            head_dim: 8,
            dtype: DType::I16,
            quantization: Default::default(),
        })
        .expect("i16 kv cache should be supported");

        let key = [-1.0_f32, -0.7, -0.4, -0.1, 0.2, 0.4, 0.7, 1.0];
        let value = [1.0_f32, 0.75, 0.5, 0.25, 0.0, -0.25, -0.5, -0.75];
        cache
            .set(0, 0, &key, &value)
            .expect("writing kv entry should succeed");

        let mut loaded_key = [0.0_f32; 8];
        let mut loaded_value = [0.0_f32; 8];
        cache
            .get_key(0, 0, &mut loaded_key)
            .expect("reading key should succeed");
        cache
            .get_value(0, 0, &mut loaded_value)
            .expect("reading value should succeed");

        for (actual, expected) in loaded_key.iter().zip(key.iter()) {
            assert!((actual - expected).abs() < 0.08);
        }
        for (actual, expected) in loaded_value.iter().zip(value.iter()) {
            assert!((actual - expected).abs() < 0.08);
        }
    }

    #[test]
    fn stores_i16_kv_vectors_with_odd_token_size() {
        let mut cache = KvCache::new(KvCacheConfig {
            layer_count: 1,
            context_size: 2,
            head_count: 1,
            head_dim: 5,
            dtype: DType::I16,
            quantization: Default::default(),
        })
        .expect("i16 kv cache should be supported");

        let key = [-1.0_f32, -0.5, 0.0, 0.5, 1.0];
        let value = [0.9_f32, 0.45, 0.0, -0.45, -0.9];
        cache
            .set(0, 1, &key, &value)
            .expect("writing odd-sized kv entry should succeed");

        let mut loaded_key = [0.0_f32; 5];
        let mut loaded_value = [0.0_f32; 5];
        cache
            .get_key(0, 1, &mut loaded_key)
            .expect("reading odd-sized key should succeed");
        cache
            .get_value(0, 1, &mut loaded_value)
            .expect("reading odd-sized value should succeed");

        for (actual, expected) in loaded_key.iter().zip(key.iter()) {
            assert!((actual - expected).abs() < 0.1);
        }
        for (actual, expected) in loaded_value.iter().zip(value.iter()) {
            assert!((actual - expected).abs() < 0.1);
        }
    }

    #[test]
    fn rejects_unsupported_dtype_and_out_of_bounds_access() {
        let unsupported = KvCache::new(KvCacheConfig {
            layer_count: 1,
            context_size: 1,
            head_count: 1,
            head_dim: 1,
            dtype: DType::I32,
            quantization: Default::default(),
        })
        .expect_err("non-fp dtype must be rejected");
        assert_eq!(
            unsupported,
            KvCacheError::UnsupportedDType { dtype: DType::I32 }
        );

        let mut cache = KvCache::new(KvCacheConfig {
            layer_count: 1,
            context_size: 2,
            head_count: 1,
            head_dim: 2,
            dtype: DType::F32,
            quantization: Default::default(),
        })
        .expect("f32 kv cache should be supported");
        let err = cache
            .set(1, 0, &[0.0, 1.0], &[2.0, 3.0])
            .expect_err("layer out of bounds should fail");
        assert_eq!(
            err,
            KvCacheError::LayerOutOfBounds {
                layer: 1,
                layer_count: 1
            }
        );
        let err = cache
            .set(0, 0, &[0.0], &[1.0, 2.0])
            .expect_err("mismatched vector length should fail");
        assert_eq!(
            err,
            KvCacheError::ValueLengthMismatch {
                expected: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn sliding_window_overwrites_physical_slots_for_new_positions() {
        let mut cache = KvCache::new(KvCacheConfig {
            layer_count: 1,
            context_size: 2,
            head_count: 1,
            head_dim: 2,
            dtype: DType::F32,
            quantization: Default::default(),
        })
        .expect("f32 kv cache should be supported");

        cache
            .set(0, 0, &[1.0, 2.0], &[3.0, 4.0])
            .expect("write at position 0 should succeed");
        cache
            .set(0, 1, &[5.0, 6.0], &[7.0, 8.0])
            .expect("write at position 1 should succeed");
        cache
            .set(0, 2, &[9.0, 10.0], &[11.0, 12.0])
            .expect("write at position 2 should succeed");

        let mut key = [0.0; 2];
        cache
            .get_key(0, 2, &mut key)
            .expect("most recent key should remain available");
        assert_eq!(key, [9.0, 10.0]);
    }

    #[test]
    fn sliding_window_rejects_evicted_positions() {
        let mut cache = KvCache::new(KvCacheConfig {
            layer_count: 1,
            context_size: 2,
            head_count: 1,
            head_dim: 2,
            dtype: DType::F32,
            quantization: Default::default(),
        })
        .expect("f32 kv cache should be supported");

        cache
            .set(0, 4, &[1.0, 1.0], &[2.0, 2.0])
            .expect("write at position 4 should succeed");
        cache
            .set(0, 5, &[3.0, 3.0], &[4.0, 4.0])
            .expect("write at position 5 should succeed");
        cache
            .set(0, 6, &[5.0, 5.0], &[6.0, 6.0])
            .expect("write at position 6 should succeed");

        let mut key = [0.0; 2];
        let err = cache
            .get_key(0, 4, &mut key)
            .expect_err("position 4 should be evicted after position 6");
        assert_eq!(
            err,
            KvCacheError::PositionEvicted {
                position: 4,
                oldest_available: 5,
                newest_available: 6
            }
        );
    }

    #[test]
    fn rewind_to_rejects_evicted_positions() {
        let mut cache = KvCache::new(KvCacheConfig {
            layer_count: 1,
            context_size: 2,
            head_count: 1,
            head_dim: 2,
            dtype: DType::F32,
            quantization: Default::default(),
        })
        .expect("f32 kv cache should be supported");

        cache
            .set(0, 4, &[1.0, 1.0], &[2.0, 2.0])
            .expect("write at position 4 should succeed");
        cache
            .set(0, 5, &[3.0, 3.0], &[4.0, 4.0])
            .expect("write at position 5 should succeed");
        cache
            .set(0, 6, &[5.0, 5.0], &[6.0, 6.0])
            .expect("write at position 6 should succeed");

        let err = cache
            .rewind_to(4)
            .expect_err("rewind to evicted position should fail");
        assert_eq!(
            err,
            KvCacheError::PositionEvicted {
                position: 4,
                oldest_available: 5,
                newest_available: 6
            }
        );

        cache
            .rewind_to(5)
            .expect("rewind to oldest available position should succeed");
        assert_eq!(cache.newest_position, Some(5));
    }

    #[test]
    fn copy_layer_keys_rejects_evicted_positions() {
        let mut cache = KvCache::new(KvCacheConfig {
            layer_count: 1,
            context_size: 2,
            head_count: 1,
            head_dim: 2,
            dtype: DType::F32,
            quantization: Default::default(),
        })
        .expect("f32 kv cache should be supported");

        cache
            .set(0, 4, &[1.0, 1.0], &[2.0, 2.0])
            .expect("write at position 4 should succeed");
        cache
            .set(0, 5, &[3.0, 3.0], &[4.0, 4.0])
            .expect("write at position 5 should succeed");
        cache
            .set(0, 6, &[5.0, 5.0], &[6.0, 6.0])
            .expect("write at position 6 should succeed");

        let mut out = vec![0.0_f32; 6];
        let err = cache
            .copy_layer_keys(0, 3, &mut out)
            .expect_err("bulk copy should fail when position 0 is evicted");
        assert_eq!(
            err,
            KvCacheError::PositionEvicted {
                position: 0,
                oldest_available: 5,
                newest_available: 6
            }
        );
    }

    #[test]
    fn stop_at_capacity_rejects_new_positions_when_full() {
        let mut cache = KvCache::with_eviction_strategy(
            KvCacheConfig {
                layer_count: 1,
                context_size: 2,
                head_count: 1,
                head_dim: 2,
                dtype: DType::F32,
                quantization: Default::default(),
            },
            KvCacheEvictionStrategy::StopAtCapacity,
        )
        .expect("f32 kv cache should be supported");

        cache
            .set(0, 10, &[1.0, 1.0], &[2.0, 2.0])
            .expect("write at position 10 should succeed");
        cache
            .set(0, 11, &[3.0, 3.0], &[4.0, 4.0])
            .expect("write at position 11 should succeed");
        let err = cache
            .set(0, 12, &[5.0, 5.0], &[6.0, 6.0])
            .expect_err("position 12 should be rejected when cache is full");
        assert_eq!(
            err,
            KvCacheError::CacheFull {
                requested_position: 12,
                oldest_available: 10,
                newest_available: 11,
                capacity: 2
            }
        );
    }

    #[test]
    fn stop_at_capacity_keeps_oldest_position_readable() {
        let mut cache = KvCache::with_eviction_strategy(
            KvCacheConfig {
                layer_count: 1,
                context_size: 2,
                head_count: 1,
                head_dim: 2,
                dtype: DType::F32,
                quantization: Default::default(),
            },
            KvCacheEvictionStrategy::StopAtCapacity,
        )
        .expect("f32 kv cache should be supported");

        cache
            .set(0, 3, &[1.0, 2.0], &[3.0, 4.0])
            .expect("write at position 3 should succeed");
        cache
            .set(0, 4, &[5.0, 6.0], &[7.0, 8.0])
            .expect("write at position 4 should succeed");
        let mut key = [0.0; 2];
        cache
            .get_key(0, 3, &mut key)
            .expect("oldest position remains readable with stop-at-capacity");
        assert_eq!(key, [1.0, 2.0]);
    }

    #[test]
    fn continuous_batching_tracks_multiple_sequences_in_shared_cache() {
        let cache = KvCache::new(KvCacheConfig {
            layer_count: 1,
            context_size: 8,
            head_count: 1,
            head_dim: 2,
            dtype: DType::F32,
            quantization: Default::default(),
        })
        .expect("f32 kv cache should be supported");
        let mut batch_cache = ContinuousBatchKvCache::new(cache, 4);
        batch_cache
            .add_sequence(100)
            .expect("first sequence should be added");
        batch_cache
            .add_sequence(200)
            .expect("second sequence should be added");

        batch_cache
            .append_token(100, 0, &[1.0, 2.0], &[0.1, 0.2])
            .expect("sequence 100 token should be written");
        batch_cache
            .append_token(200, 0, &[3.0, 4.0], &[0.3, 0.4])
            .expect("sequence 200 token should be written");
        batch_cache
            .append_token(100, 0, &[5.0, 6.0], &[0.5, 0.6])
            .expect("sequence 100 second token should be written");

        let mut first_seq_token_1 = [0.0_f32; 2];
        let mut second_seq_token_0 = [0.0_f32; 2];
        batch_cache
            .get_sequence_key(100, 0, 1, &mut first_seq_token_1)
            .expect("second token from sequence 100 should be readable");
        batch_cache
            .get_sequence_key(200, 0, 0, &mut second_seq_token_0)
            .expect("first token from sequence 200 should be readable");
        assert_eq!(first_seq_token_1, [5.0, 6.0]);
        assert_eq!(second_seq_token_0, [3.0, 4.0]);
    }

    #[test]
    fn continuous_batching_evicts_inactive_sequences() {
        let cache = KvCache::new(KvCacheConfig {
            layer_count: 1,
            context_size: 8,
            head_count: 1,
            head_dim: 2,
            dtype: DType::F32,
            quantization: Default::default(),
        })
        .expect("f32 kv cache should be supported");
        let mut batch_cache = ContinuousBatchKvCache::new(cache, 2);
        batch_cache
            .add_sequence(10)
            .expect("sequence 10 should be added");
        batch_cache
            .add_sequence(20)
            .expect("sequence 20 should be added");

        batch_cache.begin_step();
        batch_cache
            .append_token(20, 0, &[1.0, 1.0], &[2.0, 2.0])
            .expect("sequence 20 should stay active");
        batch_cache.begin_step();
        batch_cache.evict_inactive_sequences(1);
        assert_eq!(batch_cache.sequence_count(), 1);
        assert_eq!(
            batch_cache.add_sequence(30),
            Ok(()),
            "eviction should free sequence capacity"
        );
    }

    #[test]
    fn continuous_batching_reuses_position_buffers_from_pool() {
        let cache = KvCache::new(KvCacheConfig {
            layer_count: 1,
            context_size: 8,
            head_count: 1,
            head_dim: 2,
            dtype: DType::F32,
            quantization: Default::default(),
        })
        .expect("f32 kv cache should be supported");
        let mut batch_cache = ContinuousBatchKvCache::new(cache, 2);
        batch_cache
            .add_sequence(1)
            .expect("sequence should be added");
        batch_cache
            .append_token(1, 0, &[1.0, 2.0], &[3.0, 4.0])
            .expect("token should be appended");
        batch_cache
            .remove_sequence(1)
            .expect("sequence removal should succeed");
        assert_eq!(batch_cache.pooled_position_buffer_count(), 1);

        batch_cache
            .add_sequence(2)
            .expect("pooled position buffer should be reused");
        assert_eq!(batch_cache.pooled_position_buffer_count(), 0);
    }

    #[test]
    fn continuous_batching_surfaces_underlying_cache_eviction() {
        let cache = KvCache::new(KvCacheConfig {
            layer_count: 1,
            context_size: 2,
            head_count: 1,
            head_dim: 2,
            dtype: DType::F32,
            quantization: Default::default(),
        })
        .expect("f32 kv cache should be supported");
        let mut batch_cache = ContinuousBatchKvCache::new(cache, 2);
        batch_cache
            .add_sequence(1)
            .expect("sequence should be added");

        batch_cache
            .append_token(1, 0, &[1.0, 2.0], &[3.0, 4.0])
            .expect("first token should be written");
        batch_cache
            .append_token(1, 0, &[5.0, 6.0], &[7.0, 8.0])
            .expect("second token should be written");
        batch_cache
            .append_token(1, 0, &[9.0, 10.0], &[11.0, 12.0])
            .expect("third token should be written");

        let mut out = [0.0_f32; 2];
        let err = batch_cache
            .get_sequence_key(1, 0, 0, &mut out)
            .expect_err("oldest token should be evicted in sliding window");
        assert_eq!(
            err,
            ContinuousBatchError::KvCache(KvCacheError::PositionEvicted {
                position: 0,
                oldest_available: 1,
                newest_available: 2
            })
        );
    }

    #[test]
    fn kv_cache_persists_and_restores_across_sessions() {
        let path = unique_temp_path("kv-cache");
        let mut cache = KvCache::new(KvCacheConfig {
            layer_count: 1,
            context_size: 4,
            head_count: 1,
            head_dim: 4,
            dtype: DType::F32,
            quantization: Default::default(),
        })
        .expect("f32 cache should be supported");
        cache
            .set(0, 3, &[0.5, 1.5, 2.5, 3.5], &[4.5, 5.5, 6.5, 7.5])
            .expect("cache write should succeed");

        cache
            .save_to_file(&path)
            .expect("cache should be serialized to disk");
        let restored = KvCache::load_from_file(&path).expect("cache should load from disk");
        let _ = std::fs::remove_file(&path);

        assert_eq!(restored.config(), cache.config());
        assert_eq!(restored.availability_window(), cache.availability_window());
        let mut key = [0.0_f32; 4];
        let mut value = [0.0_f32; 4];
        restored
            .get_key(0, 3, &mut key)
            .expect("restored key should be readable");
        restored
            .get_value(0, 3, &mut value)
            .expect("restored value should be readable");
        assert_eq!(key, [0.5, 1.5, 2.5, 3.5]);
        assert_eq!(value, [4.5, 5.5, 6.5, 7.5]);
    }

    #[test]
    fn kv_cache_persistence_writes_explicit_storage_metadata() {
        let path = unique_temp_path("kv-cache-metadata");
        let cache = KvCache::new(KvCacheConfig {
            layer_count: 1,
            context_size: 1,
            head_count: 1,
            head_dim: 1,
            dtype: DType::F32,
            quantization: Default::default(),
        })
        .expect("f32 cache should be supported");

        cache
            .save_to_file(&path)
            .expect("cache should be serialized to disk");
        let payload = std::fs::read_to_string(&path).expect("cache file should be readable");
        let _ = std::fs::remove_file(&path);

        assert!(payload.contains(r#""storage_metadata""#));
        assert!(payload.contains(r#""version":1"#));
        assert!(payload.contains(r#""layout":"LayerMajor""#));
    }

    #[test]
    fn kv_cache_load_migrates_unversioned_position_major_storage() {
        let path = unique_temp_path("legacy-kv-cache");
        let legacy_payload = r#"
        {
            "config":{"layer_count":2,"context_size":3,"head_count":1,"head_dim":2,"dtype":"F32"},
            "key":{"F32":[1.0,2.0,101.0,102.0,3.0,4.0,103.0,104.0,5.0,6.0,105.0,106.0]},
            "value":{"F32":[10.0,20.0,110.0,120.0,30.0,40.0,130.0,140.0,50.0,60.0,150.0,160.0]},
            "eviction_strategy":"SlidingWindow",
            "oldest_position":0,
            "newest_position":2
        }
        "#;
        std::fs::write(&path, legacy_payload).expect("legacy cache should be written");

        let restored = KvCache::load_from_file(&path).expect("legacy cache should load");
        let _ = std::fs::remove_file(&path);

        let mut layer_one_position_one_key = [0.0_f32; 2];
        let mut layer_zero_position_two_value = [0.0_f32; 2];
        restored
            .get_key(1, 1, &mut layer_one_position_one_key)
            .expect("migrated layer 1 position 1 key should be readable");
        restored
            .get_value(0, 2, &mut layer_zero_position_two_value)
            .expect("migrated layer 0 position 2 value should be readable");

        assert_eq!(layer_one_position_one_key, [103.0, 104.0]);
        assert_eq!(layer_zero_position_two_value, [50.0, 60.0]);
        assert_eq!(
            restored
                .f32_layer_key_prefix(0, 3)
                .expect("borrow should validate")
                .expect("migrated layer-major prefix should be borrowable"),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn continuous_batch_cache_persists_and_restores_sequence_state() {
        let path = unique_temp_path("batch-kv-cache");
        let cache = KvCache::new(KvCacheConfig {
            layer_count: 1,
            context_size: 8,
            head_count: 1,
            head_dim: 2,
            dtype: DType::F32,
            quantization: Default::default(),
        })
        .expect("f32 cache should be supported");
        let mut batch_cache = ContinuousBatchKvCache::new(cache, 2);
        batch_cache
            .add_sequence(44)
            .expect("sequence should be added");
        batch_cache
            .append_token(44, 0, &[1.0, 2.0], &[3.0, 4.0])
            .expect("token write should succeed");
        batch_cache.begin_step();
        batch_cache
            .append_token(44, 0, &[5.0, 6.0], &[7.0, 8.0])
            .expect("second token write should succeed");

        batch_cache
            .save_to_file(&path)
            .expect("batch cache should be serialized to disk");
        let restored =
            ContinuousBatchKvCache::load_from_file(&path).expect("batch cache should load");
        let _ = std::fs::remove_file(&path);

        assert_eq!(restored.sequence_count(), 1);
        let mut token0 = [0.0_f32; 2];
        let mut token1 = [0.0_f32; 2];
        restored
            .get_sequence_key(44, 0, 0, &mut token0)
            .expect("first token should still be mapped");
        restored
            .get_sequence_key(44, 0, 1, &mut token1)
            .expect("second token should still be mapped");
        assert_eq!(token0, [1.0, 2.0]);
        assert_eq!(token1, [5.0, 6.0]);
    }

    #[test]
    fn token_slot_index_is_contiguous_across_positions_for_same_layer() {
        let config = KvCacheConfig {
            layer_count: 4,
            context_size: 8,
            head_count: 2,
            head_dim: 3,
            dtype: DType::F32,
            quantization: Default::default(),
        };

        assert_eq!(token_slot_index(&config, 2, 0), 16);
        assert_eq!(token_slot_index(&config, 2, 1), 17);
        assert_eq!(token_slot_index(&config, 2, 2), 18);
        assert_eq!(token_slot_index(&config, 2, 3), 19);
    }

    #[test]
    fn token_range_advances_by_one_token_across_adjacent_positions() {
        let config = KvCacheConfig {
            layer_count: 3,
            context_size: 4,
            head_count: 1,
            head_dim: 5,
            dtype: DType::F32,
            quantization: Default::default(),
        };

        let token_size = config.token_size();
        let position0 = token_range(&config, 1, 0);
        let position1 = token_range(&config, 1, 1);
        let position2 = token_range(&config, 1, 2);

        assert_eq!(position1.start - position0.start, token_size);
        assert_eq!(position2.start - position1.start, token_size);
    }

    // === TurboQuant KV cache tests ===

    fn tq_kv_config(dtype: DType) -> KvCacheConfig {
        KvCacheConfig {
            layer_count: 2,
            context_size: 4,
            head_count: 2,
            head_dim: 64, // token_size = 128 → 4 turboquant blocks per token
            dtype,
            quantization: KvQuantization::TurboQuant,
        }
    }

    #[test]
    fn turboquant_q8_kv_roundtrip_is_within_expected_error() {
        let config = tq_kv_config(DType::I8);
        let mut cache = KvCache::new(config).expect("cache should construct");
        let token_size = config.token_size();
        let key: Vec<f32> = (0..token_size)
            .map(|i| ((i as f32) * 0.013).sin() * 4.0)
            .collect();
        let value: Vec<f32> = (0..token_size)
            .map(|i| ((i as f32) * 0.027).cos() * 2.5)
            .collect();

        cache.set(0, 0, &key, &value).expect("set");
        let mut out_k = vec![0.0_f32; token_size];
        let mut out_v = vec![0.0_f32; token_size];
        cache.get_key(0, 0, &mut out_k).expect("get_key");
        cache.get_value(0, 0, &mut out_v).expect("get_value");

        for (a, b) in key.iter().zip(out_k.iter()) {
            assert!((a - b).abs() < 0.1, "q8 key drift {} vs {}", a, b);
        }
        for (a, b) in value.iter().zip(out_v.iter()) {
            assert!((a - b).abs() < 0.1, "q8 value drift {} vs {}", a, b);
        }
    }

    #[test]
    fn turboquant_q4_kv_roundtrip_is_within_expected_error() {
        let config = tq_kv_config(DType::I16);
        let mut cache = KvCache::new(config).expect("cache should construct");
        let token_size = config.token_size();
        let key: Vec<f32> = (0..token_size)
            .map(|i| ((i as f32) * 0.013).sin() * 4.0)
            .collect();
        let value: Vec<f32> = (0..token_size)
            .map(|i| ((i as f32) * 0.027).cos() * 2.5)
            .collect();

        cache.set(0, 0, &key, &value).expect("set");
        let mut out_k = vec![0.0_f32; token_size];
        let mut out_v = vec![0.0_f32; token_size];
        cache.get_key(0, 0, &mut out_k).expect("get_key");
        cache.get_value(0, 0, &mut out_v).expect("get_value");

        for (a, b) in key.iter().zip(out_k.iter()) {
            assert!((a - b).abs() < 1.0, "q4 key drift {} vs {}", a, b);
        }
        for (a, b) in value.iter().zip(out_v.iter()) {
            assert!((a - b).abs() < 1.0, "q4 value drift {} vs {}", a, b);
        }
    }

    #[test]
    fn turboquant_q4_data_smaller_than_q8_data() {
        let q8 = KvCache::new(tq_kv_config(DType::I8)).expect("q8 cache");
        let q4 = KvCache::new(tq_kv_config(DType::I16)).expect("q4 cache");
        assert!(
            q4.bytes_per_tensor() < q8.bytes_per_tensor(),
            "q4 {} should pack smaller than q8 {}",
            q4.bytes_per_tensor(),
            q8.bytes_per_tensor()
        );
    }

    #[test]
    fn turboquant_kv_isolates_layers_and_positions() {
        let config = tq_kv_config(DType::I8);
        let mut cache = KvCache::new(config).expect("cache");
        let token_size = config.token_size();

        let key_a: Vec<f32> = (0..token_size).map(|i| i as f32 * 0.01).collect();
        let key_b: Vec<f32> = (0..token_size).map(|i| -(i as f32) * 0.02).collect();
        cache.set(0, 0, &key_a, &key_a).expect("set a");
        cache.set(1, 2, &key_b, &key_b).expect("set b");

        let mut out_a = vec![0.0_f32; token_size];
        let mut out_b = vec![0.0_f32; token_size];
        cache.get_key(0, 0, &mut out_a).expect("get a");
        cache.get_key(1, 2, &mut out_b).expect("get b");

        assert!((out_a[1] - key_a[1]).abs() < 0.05);
        assert!((out_b[1] - key_b[1]).abs() < 0.05);
        assert!((out_a[1] - out_b[1]).abs() > 0.005);
    }

    #[test]
    fn turboquant_kv_preserves_small_values_alongside_large_ones() {
        // First 32 channels are large, remaining channels are tiny. Per-block
        // scaling must preserve both regions; per-token quantization would
        // crush the small values to zero.
        let config = tq_kv_config(DType::I16);
        let mut cache = KvCache::new(config).expect("cache");
        let token_size = config.token_size();
        let mut key = vec![0.0_f32; token_size];
        for elem in key.iter_mut().take(32) {
            *elem = 100.0;
        }
        for elem in key.iter_mut().skip(32) {
            *elem = 0.05;
        }
        cache.set(0, 0, &key, &key).expect("set");
        let mut out = vec![0.0_f32; token_size];
        cache.get_key(0, 0, &mut out).expect("get");

        let small_region_avg: f32 =
            out[32..].iter().map(|v| v.abs()).sum::<f32>() / (token_size - 32) as f32;
        assert!(
            small_region_avg > 0.01,
            "per-block scales should preserve the small region; got |avg|={}",
            small_region_avg
        );
    }

    #[test]
    fn turboquant_is_default_kv_quantization() {
        let cfg = KvCacheConfig {
            layer_count: 1,
            context_size: 1,
            head_count: 1,
            head_dim: 32,
            dtype: DType::I8,
            quantization: Default::default(),
        };
        assert_eq!(cfg.quantization, KvQuantization::TurboQuant);
    }
}
