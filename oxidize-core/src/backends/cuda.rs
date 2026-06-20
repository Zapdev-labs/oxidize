use crate::gguf::GgufQuantizationType;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;

const QK8_0: usize = 32;
const BLOCK_Q8_0_SIZE: usize = 2 + QK8_0;
const QK_K: usize = 256;
const BLOCK_Q4_K_SIZE: usize = 144;
const BLOCK_Q8_K_BYTES: usize = 4 + QK_K + 32;

pub const GEMV_KERNEL_NAME: &str = "gemv_f32_kernel";
pub const GEMV_Q8_0_KERNEL_NAME: &str = "gemv_q8_0_f32_kernel";
pub const GEMV_F16_KERNEL_NAME: &str = "gemv_f16_kernel";
/// On-the-fly Q8_0 GEMV (no f16 materialization).
pub const GEMV_Q8_0_DIRECT_KERNEL_NAME: &str = "gemv_q8_0_kernel";
/// On-the-fly Q4_0 GEMV (no f16 materialization).
pub const GEMV_Q4_0_DIRECT_KERNEL_NAME: &str = "gemv_q4_0_kernel";
/// On-the-fly Q4_K × Q8_K GEMV (no f16 materialization; OXK GPU path).
pub const GEMV_Q4_K_DIRECT_KERNEL_NAME: &str = "gemv_q4_k_kernel";
/// Q4_K × F32 GEMV — GPU-native activation path (input stays on GPU, no Q8K quantization).
pub const GEMV_Q4K_F32IN_KERNEL_NAME: &str = "gemv_q4k_f32in_kernel";
/// Q6_K × F32 GEMV — GPU-native activation path for Q6_K weight matrices.
pub const GEMV_Q6K_F32IN_KERNEL_NAME: &str = "gemv_q6k_f32in_kernel";
pub const GEMV_IQ1_S_KERNEL_NAME: &str = "gemv_iq1_s_kernel";
pub const GEMV_IQ1_M_KERNEL_NAME: &str = "gemv_iq1_m_kernel";
pub const GEMV_NVFP4_KERNEL_NAME: &str = "gemv_nvfp4_kernel";
pub const RMS_NORM_KERNEL_NAME: &str = "rms_norm_f32_kernel";
pub const RESIDUAL_ADD_KERNEL_NAME: &str = "residual_add_f32_kernel";
pub const SILU_MUL_KERNEL_NAME: &str = "silu_mul_f32_kernel";
pub const CAST_F32_TO_F16_KERNEL_NAME: &str = "cast_f32_to_f16_kernel";
pub const CAST_F16_TO_F32_KERNEL_NAME: &str = "cast_f16_to_f32_kernel";
/// On-device attention (OX_GPU_ATTN): cast post-RoPE F32 K/V into the F16 KV cache row.
pub const KV_APPEND_F16_KERNEL_NAME: &str = "kv_append_f16_kernel";
/// On-device attention (OX_GPU_ATTN): in-place partial NeoX RoPE on F32 Q and K.
pub const ROPE_F32_KERNEL_NAME: &str = "rope_f32_kernel";
/// On-device attention (OX_GPU_ATTN): GQA decode flash attention (F16 cache -> F32 out).
pub const FLASH_ATTN_DECODE_KERNEL_NAME: &str = "flash_attn_decode_kernel";
/// Split-K decode attention: emits one online-softmax state per head and KV split.
pub const FLASH_ATTN_DECODE_SPLITK_KERNEL_NAME: &str = "flash_attn_decode_splitk_kernel";
/// Split-K decode attention: merges per-split online-softmax states exactly.
pub const FLASH_ATTN_DECODE_REDUCE_KERNEL_NAME: &str = "flash_attn_decode_reduce_kernel";

// PTX is generated from `kernels/gemv_f32.cu` by `build.rs` (nvcc) into OUT_DIR.
#[cfg(feature = "cuda")]
const GEMV_F32_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/gemv_f32.ptx"));

#[cfg(feature = "cuda")]
// Weight cache key: (ptr, len). Model weights are mmap'd and immutable for
// the lifetime of inference — the pointer is a stable unique identity. No
// content hashing needed; hashing MB-sized tensors on every GEMV call was
// the primary throughput bottleneck (400MB+ hashed per token on 1B models).
type WeightCacheKey = (usize, usize);

#[cfg(feature = "cuda")]
#[inline(always)]
fn f32_cache_key(slice: &[f32]) -> WeightCacheKey {
    (slice.as_ptr() as usize, slice.len())
}

#[cfg(feature = "cuda")]
#[inline(always)]
fn bytes_cache_key(slice: &[u8]) -> WeightCacheKey {
    (slice.as_ptr() as usize, slice.len())
}

pub fn cuda_build_info() -> CudaBuildInfo {
    CudaBuildInfo {
        detected_at_build: cfg!(cuda_available),
        cuda_path: option_env!("OXIDIZE_CUDA_PATH"),
    }
}

#[cfg(feature = "cuda")]
pub fn initialize_cuda() -> Result<cust::context::Context, cust::error::CudaError> {
    cust::quick_init()
}

// ---------------------------------------------------------------------------
// Persistent per-thread GPU state
//
// The previous implementation created a fresh CUDA context, JIT-compiled the
// PTX module, and created a new cuBLAS handle on *every* matmul. Across a
// transformer forward pass that is thousands of PTX JIT compilations per token
// — the dominant cost, far larger than the actual math. We now build all of
// these once and reuse them, and keep static (quantized) weight matrices
// resident in VRAM so they are uploaded a single time instead of per token.
// ---------------------------------------------------------------------------

#[path = "cuda/types.rs"]
mod types;
pub use types::*;

#[path = "cuda/gpu_state.rs"]
mod gpu_state;
#[allow(unused_imports)]
pub use gpu_state::*;

#[path = "cuda/gemv_f32.rs"]
mod gemv_f32;
pub use gemv_f32::*;

#[path = "cuda/gemv_quantized.rs"]
mod gemv_quantized;
pub use gemv_quantized::*;

#[path = "cuda/gemm.rs"]
mod gemm;
pub use gemm::*;

#[path = "cuda/gpu_kernels.rs"]
mod gpu_kernels;
#[allow(unused_imports)]
pub use gpu_kernels::*;

#[path = "cuda/gpu_native_forward.rs"]
mod gpu_native_forward;
#[allow(unused_imports)]
pub use gpu_native_forward::*;

#[path = "cuda/flash_decode.rs"]
mod flash_decode;
pub use flash_decode::*;

#[cfg(feature = "cuda")]
#[path = "cuda/flash_decode_launch.rs"]
mod flash_decode_launch;
#[cfg(feature = "cuda")]
use flash_decode_launch::*;

#[cfg(test)]
#[path = "cuda/tests.rs"]
mod tests;

#[cfg(all(test, feature = "cuda"))]
#[path = "cuda/flash_decode_cuda_tests.rs"]
mod flash_decode_cuda_tests;

#[cfg(all(test, feature = "cuda"))]
#[path = "cuda/flash_decode_cuda_fixture.rs"]
mod flash_decode_cuda_fixture;
#[cfg(all(test, feature = "cuda"))]
use flash_decode_cuda_fixture::*;
