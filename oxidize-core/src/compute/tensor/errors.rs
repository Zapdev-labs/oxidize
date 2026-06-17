use crate::gguf::GgufQuantizationType;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    I8,
    I16,
    I32,
    I64,
}

impl DType {
    /// Return the size of a single element in bytes.
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::I8 => 1,
            DType::I16 => 2,
            DType::I32 => 4,
            DType::I64 => 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GemvError {
    InvalidMatrixLength {
        expected: usize,
        actual: usize,
    },
    InvalidVectorLength {
        expected: usize,
        actual: usize,
    },
    InvalidOutputLength {
        expected: usize,
        actual: usize,
    },
    UnsupportedQuantizationType {
        quantization: GgufQuantizationType,
    },
    #[cfg(feature = "cuda")]
    Cuda(String),
    #[cfg(feature = "metal")]
    Metal(String),
    #[cfg(feature = "webgpu")]
    WebGpu(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GemmError {
    InvalidLeftMatrixLength {
        expected: usize,
        actual: usize,
    },
    InvalidRightMatrixLength {
        expected: usize,
        actual: usize,
    },
    InvalidOutputLength {
        expected: usize,
        actual: usize,
    },
    #[cfg(feature = "cuda")]
    Cuda(String),
    #[cfg(feature = "metal")]
    Metal(String),
    #[cfg(feature = "webgpu")]
    WebGpu(String),
    InvalidTensorParallelShardCount {
        shared_dim: usize,
        shard_count: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttentionError {
    ZeroHeadDim,
    InvalidQueryLength { expected: usize, actual: usize },
    InvalidKeyLength { expected: usize, actual: usize },
    InvalidValueLength { expected: usize, actual: usize },
    InvalidOutputLength { expected: usize, actual: usize },
    InvalidKvHead { kv_head: usize, kv_heads: usize },
    InvalidHeadGrouping { num_heads: usize, kv_heads: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RopeError {
    InvalidInputLength { expected: usize, actual: usize },
    InvalidOutputLength { expected: usize, actual: usize },
    OddHeadDim { head_dim: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SwiGluError {
    InvalidGateLength { expected: usize, actual: usize },
    InvalidUpLength { expected: usize, actual: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationFn {
    Relu,
    Gelu,
    Silu,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinearActivationError {
    InvalidMatrixLength { expected: usize, actual: usize },
    InvalidVectorLength { expected: usize, actual: usize },
    InvalidOutputLength { expected: usize, actual: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RmsNormError {
    ZeroDimension,
    InvalidInputLength { expected: usize, actual: usize },
    InvalidWeightLength { expected: usize, actual: usize },
    InvalidOutputLength { expected: usize, actual: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerNormError {
    InvalidInputLength { expected: usize, actual: usize },
    InvalidWeightLength { expected: usize, actual: usize },
    InvalidBiasLength { expected: usize, actual: usize },
    InvalidOutputLength { expected: usize, actual: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SoftmaxError {
    InvalidInputLength { expected: usize, actual: usize },
}
