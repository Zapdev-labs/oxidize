#[path = "helix_cache/codec.rs"]
mod codec;
#[path = "helix_cache/config.rs"]
mod config;
#[path = "helix_cache/encode.rs"]
mod encode;
#[path = "helix_cache/pack.rs"]
mod pack;
#[path = "helix_cache/runtime.rs"]
mod runtime;

pub use codec::HelixCache;
pub use config::{HelixCacheConfig, HelixCacheError, HelixCacheStats};

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual} expected={expected} tolerance={tolerance}"
        );
    }

    #[test]
    fn cold_page_attention_matches_rope_polar_reference() {
        let mut cache = HelixCache::new(HelixCacheConfig {
            page_size: 4,
            head_dim: 8,
            inactive_threshold: 0.05,
            ..HelixCacheConfig::default()
        })
        .expect("valid helix config should construct");

        let keys = [
            1.0, 0.0, 0.02, 0.01, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.03, 0.01, 2.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.01, 0.01, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.02, 0.01, 2.0, 0.0, 0.0, 0.0,
        ];
        let values = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 2.0,
            3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,
        ];
        cache
            .store_cold_page(0, 0, 0, &keys, &values, &[0, 1, 2, 3])
            .expect("cold page should store");

        let query = [1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0];
        let logits = cache
            .logits(0, 0, &query, 3, 10000.0)
            .expect("logits should compute");
        assert_eq!(logits.len(), 4);
        for (token, logit) in logits.iter().enumerate() {
            let relative = token as f32 - 3.0;
            let expected = relative.cos() + (0.01 * relative).cos();
            assert_close(*logit, expected, 0.001);
        }

        let output = cache
            .attention(0, 0, &query, 3, 10000.0)
            .expect("attention should compute");
        assert_eq!(output.len(), 8);
        assert_close(output[0], 1.75, 0.35);

        let stats = cache.stats();
        assert_eq!(stats.cold_pages, 1);
        assert!(stats.key_bits_per_coord > 0.0);
        assert!(stats.value_bits_per_coord > 3.0);
        assert!(stats.compression_ratio_vs_f32() > 1.0);
    }

    #[test]
    fn malformed_dimensions_are_rejected() {
        let err = HelixCache::new(HelixCacheConfig {
            page_size: 4,
            head_dim: 7,
            ..HelixCacheConfig::default()
        })
        .expect_err("odd head_dim should fail");
        assert!(matches!(err, HelixCacheError::InvalidConfig(_)));
    }

    #[test]
    fn promotion_state_marks_uncertain_pages() {
        let mut cache = HelixCache::new(HelixCacheConfig {
            page_size: 2,
            head_dim: 8,
            promotion_budget: 2,
            ..HelixCacheConfig::default()
        })
        .expect("valid helix config should construct");
        let keys = [
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ];
        let values = [1.0; 16];
        cache
            .store_cold_page(1, 2, 3, &keys, &values, &[8, 9])
            .expect("cold page should store");
        cache
            .bump_uncertainty(1, 2, 3, 0.25)
            .expect("first uncertainty bump should store");
        assert!(
            !cache
                .should_promote(1, 2, 3)
                .expect("page should exist after first bump")
        );
        cache
            .bump_uncertainty(1, 2, 3, 0.25)
            .expect("second uncertainty bump should store");
        assert!(
            cache
                .should_promote(1, 2, 3)
                .expect("page should exist after second bump")
        );
    }
}
