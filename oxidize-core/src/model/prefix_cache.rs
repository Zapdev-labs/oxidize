//! Prefix caching for common prompt prefixes.
//!
//! Caches KV cache entries for common prompt prefixes (system prompts, few-shot
//! examples) so subsequent requests with the same prefix can skip prefill.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::kv_cache::{KvCache, KvCacheConfig};
use crate::model::Token;

/// Hashed representation of a token sequence for cache lookup.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PrefixHash(u64);

impl PrefixHash {
    pub fn from_tokens(tokens: &[Token]) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        tokens.hash(&mut hasher);
        Self(hasher.finish())
    }
}

/// Cached prefix entry containing the KV cache state up to a certain position.
pub struct CachedPrefix {
    pub hash: PrefixHash,
    pub token_count: usize,
    pub kv_cache_snapshot: KvCache,
    pub hit_count: usize,
}

/// Prefix cache that stores KV cache entries for common prompt prefixes.
pub struct PrefixCache {
    #[allow(dead_code)]
    config: KvCacheConfig,
    cache: HashMap<PrefixHash, CachedPrefix>,
    max_entries: usize,
    min_prefix_length: usize,
    total_hits: usize,
    total_misses: usize,
}

impl PrefixCache {
    pub fn new(config: KvCacheConfig, max_entries: usize, min_prefix_length: usize) -> Self {
        Self {
            config,
            cache: HashMap::new(),
            max_entries,
            min_prefix_length,
            total_hits: 0,
            total_misses: 0,
        }
    }

    /// Try to find a cached prefix matching the start of the given tokens.
    pub fn lookup(&self, tokens: &[Token]) -> Option<(&CachedPrefix, usize)> {
        if tokens.len() < self.min_prefix_length {
            return None;
        }

        // Try longest prefix first
        for length in (self.min_prefix_length..=tokens.len()).rev() {
            let prefix = &tokens[..length];
            let hash = PrefixHash::from_tokens(prefix);
            if let Some(entry) = self.cache.get(&hash) {
                return Some((entry, length));
            }
        }

        None
    }

    /// Store a prefix in the cache.
    pub fn store(&mut self, tokens: &[Token], kv_cache: KvCache) -> Result<(), PrefixCacheError> {
        if tokens.len() < self.min_prefix_length {
            return Ok(());
        }

        if self.cache.len() >= self.max_entries {
            self.evict_lru();
        }

        let hash = PrefixHash::from_tokens(tokens);
        let entry = CachedPrefix {
            hash: hash.clone(),
            token_count: tokens.len(),
            kv_cache_snapshot: kv_cache,
            hit_count: 0,
        };

        self.cache.insert(hash, entry);
        Ok(())
    }

    /// Record a cache hit.
    pub fn record_hit(&mut self, hash: &PrefixHash) {
        self.total_hits += 1;
        if let Some(entry) = self.cache.get_mut(hash) {
            entry.hit_count += 1;
        }
    }

    /// Record a cache miss.
    pub fn record_miss(&mut self) {
        self.total_misses += 1;
    }

    /// Get cache statistics.
    pub fn stats(&self) -> PrefixCacheStats {
        let total = self.total_hits + self.total_misses;
        PrefixCacheStats {
            entries: self.cache.len(),
            total_hits: self.total_hits,
            total_misses: self.total_misses,
            hit_ratio: if total > 0 {
                self.total_hits as f32 / total as f32
            } else {
                0.0
            },
        }
    }

    fn evict_lru(&mut self) {
        if let Some(oldest) = self
            .cache
            .iter()
            .min_by_key(|(_, entry)| entry.hit_count)
            .map(|(hash, _)| hash.clone())
        {
            self.cache.remove(&oldest);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PrefixCacheStats {
    pub entries: usize,
    pub total_hits: usize,
    pub total_misses: usize,
    pub hit_ratio: f32,
}

#[derive(Debug, thiserror::Error)]
pub enum PrefixCacheError {
    #[error("cache is full")]
    CacheFull,
    #[error("prefix too short: {0} < {1}")]
    PrefixTooShort(usize, usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> KvCacheConfig {
        KvCacheConfig {
            layer_count: 1,
            context_size: 16,
            head_count: 1,
            head_dim: 4,
            dtype: crate::tensor::DType::F32,
            quantization: Default::default(),
        }
    }

    #[test]
    fn prefix_hash_is_deterministic() {
        let tokens = vec![1, 2, 3, 4, 5];
        let hash1 = PrefixHash::from_tokens(&tokens);
        let hash2 = PrefixHash::from_tokens(&tokens);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn cache_stores_and_looks_up_prefix() {
        let config = test_config();
        let mut cache = PrefixCache::new(config, 10, 3);
        let tokens = vec![1, 2, 3, 4, 5];
        let kv = KvCache::new(config).unwrap();

        cache.store(&tokens, kv).unwrap();

        let (entry, matched_len) = cache.lookup(&tokens).unwrap();
        assert_eq!(matched_len, 5);
        assert_eq!(entry.token_count, 5);
    }

    #[test]
    fn cache_returns_longest_match() {
        let config = test_config();
        let mut cache = PrefixCache::new(config, 10, 2);
        let short = vec![1, 2, 3];
        let long = vec![1, 2, 3, 4, 5];
        let kv = KvCache::new(config).unwrap();

        cache.store(&short, kv.clone()).unwrap();
        cache.store(&long, kv).unwrap();

        let query = vec![1, 2, 3, 4, 5, 6, 7];
        let (entry, matched_len) = cache.lookup(&query).unwrap();
        assert_eq!(matched_len, 5);
        assert_eq!(entry.token_count, 5);
    }

    #[test]
    fn cache_misses_short_prefix() {
        let config = test_config();
        let cache = PrefixCache::new(config, 10, 5);
        let tokens = vec![1, 2, 3];

        assert!(cache.lookup(&tokens).is_none());
    }

    #[test]
    fn cache_evicts_when_full() {
        let config = test_config();
        let mut cache = PrefixCache::new(config, 2, 2);
        let kv = KvCache::new(config).unwrap();

        cache.store(&[1, 2], kv.clone()).unwrap();
        cache.store(&[3, 4], kv.clone()).unwrap();
        cache.store(&[5, 6], kv.clone()).unwrap();

        assert_eq!(cache.cache.len(), 2);
    }

    #[test]
    fn cache_tracks_hits_and_misses() {
        let config = test_config();
        let mut cache = PrefixCache::new(config, 10, 2);
        let tokens = vec![1, 2, 3];
        let kv = KvCache::new(config).unwrap();

        cache.store(&tokens, kv).unwrap();

        cache.record_miss();
        cache.record_miss();

        if let Some((entry, _)) = cache.lookup(&tokens) {
            let hash = entry.hash.clone();
            cache.record_hit(&hash);
        }

        let stats = cache.stats();
        assert_eq!(stats.total_hits, 1);
        assert_eq!(stats.total_misses, 2);
        assert!((stats.hit_ratio - 0.333).abs() < 0.01);
    }
}
