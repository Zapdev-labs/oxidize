use crate::tensor::DType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheConfig {
    pub layer_count: usize,
    pub context_size: usize,
    pub head_count: usize,
    pub head_dim: usize,
    pub dtype: DType,
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheEvictionStrategy {
    SlidingWindow,
    StopAtCapacity,
}

#[derive(Debug, Clone, PartialEq)]
pub enum KvCacheError {
    UnsupportedDType { dtype: DType },
    LayerOutOfBounds { layer: usize, layer_count: usize },
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
    ValueLengthMismatch { expected: usize, actual: usize },
}

#[derive(Debug, Clone, PartialEq)]
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
}

#[derive(Debug, Clone, PartialEq)]
pub struct KvCache {
    config: KvCacheConfig,
    key: KvStorage,
    value: KvStorage,
    eviction_strategy: KvCacheEvictionStrategy,
    oldest_position: Option<usize>,
    newest_position: Option<usize>,
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
        let (key, value) = match config.dtype {
            DType::F32 => (
                KvStorage::F32(vec![0.0; size]),
                KvStorage::F32(vec![0.0; size]),
            ),
            DType::F16 => (
                KvStorage::F16(vec![f32_to_f16_bits(0.0); size]),
                KvStorage::F16(vec![f32_to_f16_bits(0.0); size]),
            ),
            DType::I8 => (
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
            DType::I16 => (
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
            dtype => return Err(KvCacheError::UnsupportedDType { dtype }),
        };

        Ok(Self {
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

    pub fn set(&mut self, layer: usize, position: usize, key: &[f32], value: &[f32]) -> Result<(), KvCacheError> {
        self.validate_write(layer, key, value)?;
        self.validate_write_position(position)?;
        let physical_position = self.physical_position(position);
        write_storage(&mut self.key, &self.config, layer, physical_position, key);
        write_storage(&mut self.value, &self.config, layer, physical_position, value);
        self.record_position(position);
        Ok(())
    }

    pub fn get_key(&self, layer: usize, position: usize, out: &mut [f32]) -> Result<(), KvCacheError> {
        self.validate_read(layer, position, out)?;
        let physical_position = self.physical_position(position);
        read_storage(&self.key, &self.config, layer, physical_position, out);
        Ok(())
    }

    pub fn get_value(&self, layer: usize, position: usize, out: &mut [f32]) -> Result<(), KvCacheError> {
        self.validate_read(layer, position, out)?;
        let physical_position = self.physical_position(position);
        read_storage(&self.value, &self.config, layer, physical_position, out);
        Ok(())
    }

    pub fn bytes_per_tensor(&self) -> usize {
        match &self.key {
            KvStorage::F32(data) => data.len() * std::mem::size_of::<f32>(),
            KvStorage::F16(data) => data.len() * std::mem::size_of::<u16>(),
            KvStorage::Q8 { data, scales, mins } => {
                data.len() + (scales.len() * std::mem::size_of::<f32>()) + (mins.len() * std::mem::size_of::<f32>())
            }
            KvStorage::Q4 { data, scales, mins } => {
                data.len() + (scales.len() * std::mem::size_of::<f32>()) + (mins.len() * std::mem::size_of::<f32>())
            }
        }
    }

    fn validate_write(
        &self,
        layer: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<(), KvCacheError> {
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

    fn record_position(&mut self, position: usize) {
        let newest = self.newest_position.map_or(position, |current| current.max(position));
        self.newest_position = Some(newest);
        let oldest = match self.eviction_strategy {
            KvCacheEvictionStrategy::SlidingWindow => {
                newest
                    .saturating_add(1)
                    .saturating_sub(self.config.context_size)
            }
            KvCacheEvictionStrategy::StopAtCapacity => self.oldest_position.unwrap_or(position),
        };
        self.oldest_position = Some(oldest);
    }

    fn physical_position(&self, position: usize) -> usize {
        position % self.config.context_size
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
    let token_index = layer * config.context_size + position;
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
            for (idx, value) in src.iter().enumerate() {
                let q = if scale == 0.0 {
                    0
                } else {
                    ((*value - min) / scale).round().clamp(0.0, 15.0) as u8
                };
                let packed_idx = (range.start + idx) / 2;
                if (range.start + idx).is_multiple_of(2) {
                    data[packed_idx] = (data[packed_idx] & 0xF0) | (q & 0x0F);
                } else {
                    data[packed_idx] = (data[packed_idx] & 0x0F) | ((q & 0x0F) << 4);
                }
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
    let token_index = layer * config.context_size + position;
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
            for (idx, out) in dst.iter_mut().enumerate() {
                let packed_idx = (range.start + idx) / 2;
                let packed = data[packed_idx];
                let q = if (range.start + idx).is_multiple_of(2) {
                    packed & 0x0F
                } else {
                    (packed >> 4) & 0x0F
                };
                *out = (q as f32) * scale + min;
            }
        }
    }
}

fn token_range(config: &KvCacheConfig, layer: usize, position: usize) -> std::ops::Range<usize> {
    let token_size = config.token_size();
    let offset = layer * config.layer_size() + position * token_size;
    offset..offset + token_size
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

fn f32_to_f16_bits(value: f32) -> u16 {
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

    #[test]
    fn allocates_kv_cache_with_requested_dtype() {
        let f32_cache = KvCache::new(KvCacheConfig {
            layer_count: 2,
            context_size: 4,
            head_count: 2,
            head_dim: 8,
            dtype: DType::F32,
        })
        .expect("f32 kv cache should be supported");
        let f16_cache = KvCache::new(KvCacheConfig {
            layer_count: 2,
            context_size: 4,
            head_count: 2,
            head_dim: 8,
            dtype: DType::F16,
        })
        .expect("f16 kv cache should be supported");
        let q8_cache = KvCache::new(KvCacheConfig {
            layer_count: 2,
            context_size: 4,
            head_count: 2,
            head_dim: 8,
            dtype: DType::I8,
        })
        .expect("q8 kv cache should be supported");
        let q4_cache = KvCache::new(KvCacheConfig {
            layer_count: 2,
            context_size: 4,
            head_count: 2,
            head_dim: 8,
            dtype: DType::I16,
        })
        .expect("q4 kv cache should be supported");

        assert_eq!(f32_cache.bytes_per_tensor(), 2 * 4 * 2 * 8 * 4);
        assert_eq!(f16_cache.bytes_per_tensor(), 2 * 4 * 2 * 8 * 2);
        assert_eq!(q8_cache.bytes_per_tensor(), (2 * 4 * 2 * 8) + (2 * 4 * 4) + (2 * 4 * 4));
        assert_eq!(q4_cache.bytes_per_tensor(), (2_usize * 4 * 2 * 8).div_ceil(2) + (2 * 4 * 4) + (2 * 4 * 4));
    }

    #[test]
    fn stores_and_reads_back_f32_kv_vectors() {
        let mut cache = KvCache::new(KvCacheConfig {
            layer_count: 1,
            context_size: 3,
            head_count: 1,
            head_dim: 4,
            dtype: DType::F32,
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
    fn stores_f16_kv_vectors_with_expected_quantization_error() {
        let mut cache = KvCache::new(KvCacheConfig {
            layer_count: 1,
            context_size: 1,
            head_count: 1,
            head_dim: 4,
            dtype: DType::F16,
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
    fn rejects_unsupported_dtype_and_out_of_bounds_access() {
        let unsupported = KvCache::new(KvCacheConfig {
            layer_count: 1,
            context_size: 1,
            head_count: 1,
            head_dim: 1,
            dtype: DType::I32,
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
    fn stop_at_capacity_rejects_new_positions_when_full() {
        let mut cache = KvCache::with_eviction_strategy(
            KvCacheConfig {
                layer_count: 1,
                context_size: 2,
                head_count: 1,
                head_dim: 2,
                dtype: DType::F32,
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
}
