#[cfg(any(feature = "cuda", feature = "rocm"))]
use super::*;

/// Returns `true` when GPU activation kernels should be preferred over the CPU
/// implementation.  See module-level comment for full policy.
pub fn should_use_gpu_activations(hidden_size: usize) -> bool {
    // Feature gate: at least one GPU backend must be compiled in.
    #[cfg(not(any(feature = "cuda", feature = "rocm")))]
    {
        let _ = hidden_size;
        return false;
    }
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    {
        if crate::gpu_dispatch::active_gpu().is_none() {
            return false;
        }
        // Safety limit: avoid surprise OOM on very large hidden sizes
        // (e.g. 65 536-dim future models) before the pool supports them.
        const GPU_ACTIVATION_MAX_HIDDEN: usize = 8192;
        hidden_size <= GPU_ACTIVATION_MAX_HIDDEN
    }
}

/// GPU-routed RMSNorm.  Falls back gracefully to [`rms_norm_f32`] when
/// `should_use_gpu_activations` returns `false`.
///
/// # Future optimisation
/// Once `ws.x` is kept on-device, the upload of `x` and download of `output`
/// can be removed — only the `weight` upload (cached) will remain.
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub fn gpu_rms_norm_f32(
    x: &[f32],
    weight: &[f32],
    output: &mut [f32],
    eps: f32,
) -> Result<(), RmsNormError> {
    // Current implementation: use the existing CPU path.
    // Weight is already in CPU memory and the RMSNorm kernel is lightweight;
    // this stub is here so call sites can be switched to GPU without API churn.
    // TODO: replace body with a GPU kernel once hidden-state residency lands.
    rms_norm_f32(x, weight, eps, output)
}

/// GPU-routed SwiGLU element-wise activation (`gate *= sigmoid(gate) * up`).
///
/// # Future optimisation
/// When both `gate` and `up` live in device memory this reduces to a single
/// in-place kernel launch with no host traffic at all.
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub fn gpu_silu_mul_f32(gate: &mut [f32], up: &[f32]) {
    // Current implementation: CPU path.  Stub exists to centralise the
    // routing decision once device-resident buffers are available.
    apply_swiglu_inplace_f32(gate, up);
}

/// GPU-routed residual add (`x[i] += delta[i]`).
///
/// # Future optimisation
/// When both buffers are device-resident this is a single `axpy` cuBLAS call
/// (or equivalent ROCm/rocBLAS) with zero host traffic.
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub fn gpu_residual_add_f32(x: &mut [f32], delta: &[f32]) {
    // Current implementation: CPU path.
    debug_assert_eq!(x.len(), delta.len(), "residual add: length mismatch");
    for (xi, di) in x.iter_mut().zip(delta.iter()) {
        *xi += di;
    }
}
