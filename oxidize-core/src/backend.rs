//! Backend selection and platform-aware fallback logic.

use crate::tensor::DType;

/// Supported compute backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Metal,
    Cuda,
    Mlx,
    Vulkan,
}

impl std::str::FromStr for Backend {
    type Err = ();

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "cpu" => Ok(Backend::Cpu),
            "metal" => Ok(Backend::Metal),
            "cuda" => Ok(Backend::Cuda),
            "mlx" => Ok(Backend::Mlx),
            "vulkan" => Ok(Backend::Vulkan),
            _ => Err(()),
        }
    }
}

impl Backend {
    /// Return the canonical name of this backend.
    pub fn as_str(&self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Metal => "metal",
            Backend::Cuda => "cuda",
            Backend::Mlx => "mlx",
            Backend::Vulkan => "vulkan",
        }
    }

    /// Determine the effective backend for the current platform.
    ///
    /// On non-macOS platforms, `Mlx` is downgraded to `Cpu` and a warning
    /// message is returned.
    pub fn effective(self) -> (Self, Option<&'static str>) {
        match self {
            Backend::Mlx if !cfg!(target_os = "macos") => (
                Backend::Cpu,
                Some("MLX backend requested but unavailable on Linux; falling back to CPU"),
            ),
            Backend::Vulkan => (Backend::Vulkan, None),
            other => (other, None),
        }
    }
}

/// Trait that abstracts the core compute operations needed by the inference
/// engine.  Each backend (CPU, CUDA, Metal, MLX) provides an implementation.
pub trait ComputeBackend: Send + Sync {
    /// A backend-specific tensor handle.
    type Tensor: Clone + Send + Sync;

    /// A backend-specific weight storage handle.
    type WeightStorage: Clone + Send + Sync;

    /// Human-readable backend name.
    fn name(&self) -> &'static str;

    /// Create a 1-D tensor from a slice of `f32` values.
    fn tensor_from_f32(&self, data: &[f32]) -> Result<Self::Tensor, String>;

    /// Create a 2-D tensor from a slice of `f32` values.
    fn tensor_from_f32_2d(&self, data: &[f32], rows: usize, cols: usize) -> Result<Self::Tensor, String>;

    /// Copy tensor data back to host as `f32`.  Returns the number of elements copied.
    fn tensor_to_f32(&self, tensor: &Self::Tensor, out: &mut [f32]) -> Result<usize, String>;

    /// Return the shape of the tensor as a vector of dimensions.
    fn tensor_shape(&self, tensor: &Self::Tensor) -> Vec<usize>;

    /// Return the element dtype of the tensor.
    fn tensor_dtype(&self, tensor: &Self::Tensor) -> DType;

    /// RMS normalization: `output = input / sqrt(mean(input^2) + eps) * weight`.
    fn rms_norm(
        &self,
        input: &Self::Tensor,
        weight: &Self::Tensor,
        eps: f32,
    ) -> Result<Self::Tensor, String>;

    /// Rotary Position Embedding (RoPE) applied to `input` at `position`.
    fn apply_rope(
        &self,
        input: &Self::Tensor,
        position: usize,
        head_dim: usize,
        theta: f32,
    ) -> Result<Self::Tensor, String>;

    /// Scaled dot-product attention for a single query attending to cached keys/values.
    fn attention_decode(
        &self,
        query: &Self::Tensor,
        key_cache: &Self::Tensor,
        value_cache: &Self::Tensor,
        seq_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Self::Tensor, String>;

    /// Matrix-vector multiplication: `output = matrix * vector`.
    fn gemv(
        &self,
        matrix: &Self::WeightStorage,
        vector: &Self::Tensor,
        rows: usize,
        cols: usize,
    ) -> Result<Self::Tensor, String>;

    /// Matrix-matrix multiplication: `output = a * b`.
    fn gemm(
        &self,
        a: &Self::Tensor,
        b: &Self::Tensor,
        rows: usize,
        shared_dim: usize,
        cols: usize,
    ) -> Result<Self::Tensor, String>;

    /// Element-wise addition.
    fn add(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, String>;

    /// Element-wise multiplication (used for SwiGLU gate).
    fn mul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, String>;

    /// Sigmoid activation: `1 / (1 + exp(-x))`.
    fn sigmoid(&self, x: &Self::Tensor) -> Result<Self::Tensor, String>;

    /// Softmax along the last axis.
    fn softmax(&self, x: &Self::Tensor) -> Result<Self::Tensor, String>;

    /// Evaluate / synchronize any pending lazy operations.
    fn synchronize(&self) -> Result<(), String>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn backend_parses_all_variants() {
        assert_eq!(Backend::from_str("cpu"), Ok(Backend::Cpu));
        assert_eq!(Backend::from_str("metal"), Ok(Backend::Metal));
        assert_eq!(Backend::from_str("cuda"), Ok(Backend::Cuda));
        assert_eq!(Backend::from_str("mlx"), Ok(Backend::Mlx));
        assert_eq!(Backend::from_str("vulkan"), Ok(Backend::Vulkan));
        assert_eq!(Backend::from_str("unknown"), Err(()));
    }

    #[test]
    fn backend_roundtrips_through_str() {
        for backend in [Backend::Cpu, Backend::Metal, Backend::Cuda, Backend::Mlx, Backend::Vulkan] {
            assert_eq!(Backend::from_str(backend.as_str()), Ok(backend));
        }
    }

    #[test]
    fn mlx_fallback_on_linux() {
        if !cfg!(target_os = "macos") {
            let (effective, warning) = Backend::Mlx.effective();
            assert_eq!(effective, Backend::Cpu);
            assert!(
                warning.is_some(),
                "expected a warning when requesting MLX on non-macOS"
            );
            assert_eq!(
                warning.unwrap(),
                "MLX backend requested but unavailable on Linux; falling back to CPU"
            );
        }
    }

    #[test]
    fn cpu_always_effective() {
        let (effective, warning) = Backend::Cpu.effective();
        assert_eq!(effective, Backend::Cpu);
        assert!(warning.is_none());
    }

    #[test]
    fn metal_always_effective() {
        let (effective, warning) = Backend::Metal.effective();
        assert_eq!(effective, Backend::Metal);
        assert!(warning.is_none());
    }

    #[test]
    fn cuda_always_effective() {
        let (effective, warning) = Backend::Cuda.effective();
        assert_eq!(effective, Backend::Cuda);
        assert!(warning.is_none());
    }

    #[test]
    fn vulkan_always_effective() {
        let (effective, warning) = Backend::Vulkan.effective();
        assert_eq!(effective, Backend::Vulkan);
        assert!(warning.is_none());
    }
}
