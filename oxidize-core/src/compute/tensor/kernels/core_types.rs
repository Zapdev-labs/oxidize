use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub dtype: DType,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, strides: Vec<usize>, dtype: DType) -> Self {
        assert_eq!(
            shape.len(),
            strides.len(),
            "shape and strides must have the same rank"
        );
        Self {
            shape,
            strides,
            dtype,
        }
    }
}

// ---------------------------------------------------------------------------
// GPU-native activation stubs
//
// These functions provide GPU-routed versions of the most common per-token
// operations (RMSNorm, SwiGLU activation, residual add).  Today each call
// still performs a CPU↔GPU upload/download round-trip, so the gain is only
// that the RMSNorm *weight* and the SwiGLU math run on the GPU rather than
// the CPU.  The real win will come once the hidden state (`ws.x`) is kept
// resident on the GPU across layers, eliminating all intermediate D2H copies.
// The function signatures are intentionally identical to their CPU counterparts
// to make that future substitution a one-line change at each call site.
//
// Routing policy (`should_use_gpu_activations`):
//   • CUDA or ROCm feature must be compiled in.
//   • A GPU must have been detected at runtime (`active_gpu().is_some()`).
//   • `hidden_size` must be ≤ 8192 — the GPU buffer pool only pre-allocates
//     up to that size; larger hidden states fall back to CPU to avoid
//     unbounded allocation on the first call.
// ---------------------------------------------------------------------------
