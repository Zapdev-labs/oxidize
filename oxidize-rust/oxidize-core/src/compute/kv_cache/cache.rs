use super::*;
use crate::tensor::DType;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KvCache {
    #[serde(default)]
    pub(super) storage_metadata: KvCacheStorageMetadata,
    pub(super) config: KvCacheConfig,
    pub(super) key: KvStorage,
    pub(super) value: KvStorage,
    pub(super) eviction_strategy: KvCacheEvictionStrategy,
    pub(super) oldest_position: Option<usize>,
    pub(super) newest_position: Option<usize>,
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

    pub fn copy_layer_key_prefix_values(
        &self,
        layer: usize,
        seq_len: usize,
        logical_token_size: usize,
        out: &mut [f32],
    ) -> Result<(), KvCacheError> {
        self.copy_layer_prefix_values(&self.key, layer, seq_len, logical_token_size, out)
    }

    pub fn copy_layer_value_prefix_values(
        &self,
        layer: usize,
        seq_len: usize,
        logical_token_size: usize,
        out: &mut [f32],
    ) -> Result<(), KvCacheError> {
        self.copy_layer_prefix_values(&self.value, layer, seq_len, logical_token_size, out)
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

    pub(super) fn migrate_legacy_storage_layout(&mut self) {
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

    fn copy_layer_prefix_values(
        &self,
        storage: &KvStorage,
        layer: usize,
        seq_len: usize,
        logical_token_size: usize,
        out: &mut [f32],
    ) -> Result<(), KvCacheError> {
        self.validate_layer(layer)?;
        let token_size = self.config.token_size();
        if logical_token_size > token_size {
            return Err(KvCacheError::ValueLengthMismatch {
                expected: token_size,
                actual: logical_token_size,
            });
        }
        let expected = seq_len.saturating_mul(logical_token_size);
        if out.len() != expected {
            return Err(KvCacheError::ValueLengthMismatch {
                expected,
                actual: out.len(),
            });
        }
        let mut row = vec![0.0_f32; token_size];
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
            read_storage(storage, &self.config, layer, physical_position, &mut row);
            let out_off = p * logical_token_size;
            out[out_off..out_off + logical_token_size].copy_from_slice(&row[..logical_token_size]);
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

    pub(super) fn position_available(&self, position: usize) -> bool {
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
