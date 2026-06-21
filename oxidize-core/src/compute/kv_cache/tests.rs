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
        quantization: KvQuantization::Asymmetric,
    })
    .expect("q8 kv cache should be supported");
    let q4_cache = KvCache::new(KvCacheConfig {
        layer_count: 2,
        context_size: 4,
        head_count: 2,
        head_dim: 8,
        dtype: DType::I16,
        quantization: KvQuantization::Asymmetric,
    })
    .expect("q4 kv cache should be supported");

    assert_eq!(f32_cache.bytes_per_tensor(), 2 * 4 * 2 * 8 * 4);
    assert_eq!(f16_cache.bytes_per_tensor(), 2 * 4 * 2 * 8 * 2);
    assert_eq!(
        q8_cache.bytes_per_tensor(),
        (2 * 4 * 2 * 8) + (2 * 4 * 4) + (2 * 4 * 4)
    );
    assert_eq!(
        q4_cache.bytes_per_tensor(),
        (2_usize * 4 * 2 * 8).div_ceil(2) + (2 * 4 * 4) + (2 * 4 * 4)
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
    let restored = ContinuousBatchKvCache::load_from_file(&path).expect("batch cache should load");
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
