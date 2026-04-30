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

#[derive(Debug, Clone, PartialEq)]
pub enum KvCacheError {
    UnsupportedDType { dtype: DType },
    PositionOutOfBounds { position: usize, context_size: usize },
    LayerOutOfBounds { layer: usize, layer_count: usize },
    ValueLengthMismatch { expected: usize, actual: usize },
}

#[derive(Debug, Clone, PartialEq)]
enum KvStorage {
    F32(Vec<f32>),
    F16(Vec<u16>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct KvCache {
    config: KvCacheConfig,
    key: KvStorage,
    value: KvStorage,
}

impl KvCache {
    pub fn new(config: KvCacheConfig) -> Result<Self, KvCacheError> {
        let size = config.element_count();
        let (key, value) = match config.dtype {
            DType::F32 => (
                KvStorage::F32(vec![0.0; size]),
                KvStorage::F32(vec![0.0; size]),
            ),
            DType::F16 => (
                KvStorage::F16(vec![f32_to_f16_bits(0.0); size]),
                KvStorage::F16(vec![f32_to_f16_bits(0.0); size]),
            ),
            dtype => return Err(KvCacheError::UnsupportedDType { dtype }),
        };

        Ok(Self {
            config,
            key,
            value,
        })
    }

    pub fn config(&self) -> KvCacheConfig {
        self.config
    }

    pub fn set(&mut self, layer: usize, position: usize, key: &[f32], value: &[f32]) -> Result<(), KvCacheError> {
        self.validate_write(layer, position, key, value)?;
        let range = self.token_range(layer, position);
        write_storage(&mut self.key, range.clone(), key);
        write_storage(&mut self.value, range, value);
        Ok(())
    }

    pub fn get_key(&self, layer: usize, position: usize, out: &mut [f32]) -> Result<(), KvCacheError> {
        self.validate_read(layer, position, out)?;
        let range = self.token_range(layer, position);
        read_storage(&self.key, range, out);
        Ok(())
    }

    pub fn get_value(&self, layer: usize, position: usize, out: &mut [f32]) -> Result<(), KvCacheError> {
        self.validate_read(layer, position, out)?;
        let range = self.token_range(layer, position);
        read_storage(&self.value, range, out);
        Ok(())
    }

    pub fn bytes_per_tensor(&self) -> usize {
        match &self.key {
            KvStorage::F32(data) => data.len() * std::mem::size_of::<f32>(),
            KvStorage::F16(data) => data.len() * std::mem::size_of::<u16>(),
        }
    }

    fn validate_write(
        &self,
        layer: usize,
        position: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<(), KvCacheError> {
        self.validate_indices(layer, position)?;
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
        self.validate_indices(layer, position)?;
        let expected = self.config.token_size();
        if out.len() != expected {
            return Err(KvCacheError::ValueLengthMismatch {
                expected,
                actual: out.len(),
            });
        }
        Ok(())
    }

    fn validate_indices(&self, layer: usize, position: usize) -> Result<(), KvCacheError> {
        if layer >= self.config.layer_count {
            return Err(KvCacheError::LayerOutOfBounds {
                layer,
                layer_count: self.config.layer_count,
            });
        }
        if position >= self.config.context_size {
            return Err(KvCacheError::PositionOutOfBounds {
                position,
                context_size: self.config.context_size,
            });
        }
        Ok(())
    }

    fn token_range(&self, layer: usize, position: usize) -> std::ops::Range<usize> {
        let token_size = self.config.token_size();
        let offset = layer * self.config.layer_size() + position * token_size;
        offset..offset + token_size
    }
}

fn write_storage(storage: &mut KvStorage, range: std::ops::Range<usize>, src: &[f32]) {
    match storage {
        KvStorage::F32(data) => data[range].copy_from_slice(src),
        KvStorage::F16(data) => {
            for (dst, value) in data[range].iter_mut().zip(src.iter()) {
                *dst = f32_to_f16_bits(*value);
            }
        }
    }
}

fn read_storage(storage: &KvStorage, range: std::ops::Range<usize>, dst: &mut [f32]) {
    match storage {
        KvStorage::F32(data) => dst.copy_from_slice(&data[range]),
        KvStorage::F16(data) => {
            for (out, value) in dst.iter_mut().zip(data[range].iter()) {
                *out = f16_bits_to_f32(*value);
            }
        }
    }
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

        assert_eq!(f32_cache.bytes_per_tensor(), 2 * 4 * 2 * 8 * 4);
        assert_eq!(f16_cache.bytes_per_tensor(), 2 * 4 * 2 * 8 * 2);
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
    fn rejects_unsupported_dtype_and_out_of_bounds_access() {
        let unsupported = KvCache::new(KvCacheConfig {
            layer_count: 1,
            context_size: 1,
            head_count: 1,
            head_dim: 1,
            dtype: DType::I8,
        })
        .expect_err("non-fp dtype must be rejected");
        assert_eq!(
            unsupported,
            KvCacheError::UnsupportedDType { dtype: DType::I8 }
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
            .set(0, 3, &[0.0, 1.0], &[2.0, 3.0])
            .expect_err("position out of bounds should fail");
        assert_eq!(
            err,
            KvCacheError::PositionOutOfBounds {
                position: 3,
                context_size: 2
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
}
