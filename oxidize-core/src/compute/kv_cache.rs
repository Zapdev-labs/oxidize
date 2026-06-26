use crate::tensor::DType;
use crate::turboquant::TURBOQUANT_BLOCK_SIZE;
use serde::{Deserialize, Serialize};

#[path = "kv_cache/batch.rs"]
mod batch;
#[path = "kv_cache/cache.rs"]
mod cache;
#[path = "kv_cache/storage.rs"]
mod storage;

pub use batch::ContinuousBatchKvCache;
pub use cache::KvCache;

pub(crate) use storage::f32_to_f16_bits;
use storage::{
    f16_bits_to_f32, migrate_storage_from_position_major, min_max, read_storage, token_range,
    token_slot_index, write_storage,
};

#[cfg(test)]
#[path = "kv_cache/tests.rs"]
mod tests;

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
pub(super) enum KvCacheStorageLayout {
    /// Storage is grouped by layer, then position: `[layer][position][head][head_dim]`.
    LayerMajor,
    /// Legacy serialized storage grouped by position, then layer.
    PositionMajor,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct KvCacheStorageMetadata {
    pub(super) version: u32,
    pub(super) layout: KvCacheStorageLayout,
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

pub(super) fn current_storage_metadata() -> KvCacheStorageMetadata {
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
pub(super) enum KvStorage {
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
