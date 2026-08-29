use crate::gguf::GgufQuantizationType;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;

const QK8_0: usize = 32;
const BLOCK_Q8_0_SIZE: usize = 2 + QK8_0;
const QK_K: usize = 256;
const BLOCK_Q4_K_SIZE: usize = 144;
const BLOCK_Q8_K_BYTES: usize = 4 + QK_K + 32;

/// Stamp a cached env-flag getter: `ox_env_flag!(name, "OX_ENV", default)`
/// reads `OX_ENV` once per process. "0" (or empty) disables; any other value
/// enables; unset uses `default`. This is the semantics every OX_GPU_* flag
/// in the CUDA backends already used.
#[allow(unused_macros)]
macro_rules! ox_env_flag {
    ($name:ident, $env:literal, $default:literal) => {
        pub(super) fn $name() -> bool {
            static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ENABLED.get_or_init(|| {
                std::env::var($env)
                    .map(|v| v != "0" && !v.is_empty())
                    .unwrap_or($default)
            })
        }
    };
}
pub(crate) use ox_env_flag;

pub const GEMV_KERNEL_NAME: &str = "gemv_f32_kernel";
pub const GEMV_Q8_0_KERNEL_NAME: &str = "gemv_q8_0_f32_kernel";
pub const GEMV_F16_KERNEL_NAME: &str = "gemv_f16_kernel";
/// On-the-fly Q8_0 GEMV (no f16 materialization).
pub const GEMV_Q8_0_DIRECT_KERNEL_NAME: &str = "gemv_q8_0_kernel";
/// On-the-fly Q4_0 GEMV (no f16 materialization).
pub const GEMV_Q4_0_DIRECT_KERNEL_NAME: &str = "gemv_q4_0_kernel";
/// On-the-fly Q4_K × Q8_K GEMV (no f16 materialization; OXK GPU path).
pub const GEMV_Q4_K_DIRECT_KERNEL_NAME: &str = "gemv_q4_k_kernel";
/// F32 activation → Q8_K on device (one CUDA block per 256-value super-block).
pub const QUANTIZE_F32_TO_Q8K_KERNEL_NAME: &str = "quantize_f32_to_q8k_kernel";
/// Q4_K × Q8_K GEMV with DP4A — device-resident activation (OXK GPU path).
pub const GEMV_Q4K_Q8KIN_KERNEL_NAME: &str = "gemv_q4k_q8kin_kernel";
/// Multi-warp Q4_K × Q8_K GEMV against a pre-quantized global xq8k buffer.
pub const GEMV_Q4K_Q8KIN_MW_KERNEL_NAME: &str = "gemv_q4k_q8kin_mw_kernel";
/// Fused Q4_K × Q8_K MMQ GEMV: activation quantized to shared Q8_K in-kernel,
/// then a tile of rows reuses it with DP4A — one launch, no VRAM round-trip.
pub const GEMV_Q4K_Q8K_FUSED_KERNEL_NAME: &str = "gemv_q4k_q8k_fused_kernel";
/// Fused MMQ + 4 warps/row (H100 HBM); shared activation, 2 rows/block.
pub const GEMV_Q4K_Q8K_FUSED_MW_KERNEL_NAME: &str = "gemv_q4k_q8k_fused_mw_kernel";
/// One launch: in-kernel shared quantize + Q/K/V multi-warp MMQ (legacy).
pub const GEMV_Q4K_Q8K_FUSED_QKV_MW_KERNEL_NAME: &str = "gemv_q4k_q8k_fused_qkv_mw_kernel";
/// One launch: global xq8k + Q/K/V multi-warp DP4A (shared-quantize QKV path).
pub const GEMV_Q4K_Q8KIN_QKV_MW_KERNEL_NAME: &str = "gemv_q4k_q8kin_qkv_mw_kernel";
/// Multi-row Q4_K × F32 lm_head GEMV: ROWS_PER_BLK rows share one cached
/// activation tile (the 19k×hidden output projection is the largest GEMV).
pub const GEMV_Q4K_F32IN_MULTIROW_KERNEL_NAME: &str = "gemv_q4k_f32in_multirow_kernel";
/// Q4_K × F32 GEMV — fallback when weights are Q6_K or DP4A path unavailable.
pub const GEMV_Q4K_F32IN_KERNEL_NAME: &str = "gemv_q4k_f32in_kernel";
/// Multi-warp-per-row Q4_K × F32 GEMV (OX_GPU_GEMV_MW): NWARPS warps cooperate on
/// one output row for more in-flight HBM requests/row (H100 bandwidth). Output is
/// numerically identical to [`GEMV_Q4K_F32IN_KERNEL_NAME`].
pub const GEMV_Q4K_F32IN_MW_KERNEL_NAME: &str = "gemv_q4k_f32in_mw_kernel";
pub const GEMV_Q4K_F32IN_2ROW_KERNEL_NAME: &str = "gemv_q4k_f32in_2row_kernel";
pub const GEMV_Q4K_F32IN_RESIDUAL_KERNEL_NAME: &str = "gemv_q4k_f32in_residual_kernel";
pub const GEMV_Q6K_F32IN_RESIDUAL_KERNEL_NAME: &str = "gemv_q6k_f32in_residual_kernel";
/// Batched Q4_K × Q8_K GEMV: `ncols` (batch B) activation vectors share each
/// weight stream (B GEMVs → one weight pass; weights reused via warm L2). The
/// continuous-batching GPU decode lever. Output row-major `[rows, ncols]`; each
/// column matches [`GEMV_Q4_K_DIRECT_KERNEL_NAME`] on that activation alone.
pub const GEMV_Q4K_Q8KIN_BN_KERNEL_NAME: &str = "gemv_q4k_q8kin_bN_kernel";
/// Transpose the bN GEMV output `[rows, ncols]` -> per-row activation
/// `[ncols, rows]` so each batched sequence's vector is contiguous for the
/// per-seq attention/quantize steps. Pure index gather (no arithmetic).
pub const TRANSPOSE_ROWB_KERNEL_NAME: &str = "transpose_rowB_to_Brow_kernel";
/// Q6_K × F32 GEMV — GPU-native activation path for Q6_K weight matrices.
pub const GEMV_Q6K_F32IN_KERNEL_NAME: &str = "gemv_q6k_f32in_kernel";
pub const GEMV_IQ1_S_KERNEL_NAME: &str = "gemv_iq1_s_kernel";
pub const GEMV_IQ1_M_KERNEL_NAME: &str = "gemv_iq1_m_kernel";
pub const GEMV_NVFP4_KERNEL_NAME: &str = "gemv_nvfp4_kernel";
pub const RMS_NORM_KERNEL_NAME: &str = "rms_norm_f32_kernel";
pub const RESIDUAL_ADD_KERNEL_NAME: &str = "residual_add_f32_kernel";
pub const SILU_MUL_KERNEL_NAME: &str = "silu_mul_f32_kernel";
/// Fused SwiGLU FFN gate/up: out[r] = silu(dot(gate_w[r],x)) * dot(up_w[r],x).
/// One launch replaces gate GEMV + up GEMV + silu_mul (Q4_K gate/up only).
pub const GEMV_Q4K_F32IN_GATE_UP_SILU_KERNEL_NAME: &str = "gemv_q4k_f32in_gate_up_silu_kernel";
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
/// F32 embedding row lookup into device activation buffer.
pub const EMBED_F32_ROW_KERNEL_NAME: &str = "embed_f32_row_kernel";
/// Q4_K embedding row dequant + lookup into device activation buffer.
pub const EMBED_Q4K_F32_ROW_KERNEL_NAME: &str = "embed_q4k_f32_row_kernel";

// CUDA-graph device-scalar decode kernels (OX_GPU_CUDA_GRAPH). These read the
// per-token state (pos, context, token_id) from a 3-word device buffer instead
// of kernel args, so one captured graph stays valid as the sequence grows. The
// math is byte-identical to the eager kernels above; they are inert until the
// graph capture harness launches them.
pub const ROPE_F32_GPH_KERNEL_NAME: &str = "rope_f32_gph_kernel";
pub const KV_APPEND_F16_GPH_KERNEL_NAME: &str = "kv_append_f16_gph_kernel";
pub const FLASH_ATTN_DECODE_GPH_KERNEL_NAME: &str = "flash_attn_decode_gph_kernel";
pub const EMBED_F32_ROW_GPH_KERNEL_NAME: &str = "embed_f32_row_gph_kernel";
pub const EMBED_Q4K_F32_ROW_GPH_KERNEL_NAME: &str = "embed_q4k_f32_row_gph_kernel";

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

/// Env gate for the device-side batched decode forward (`OX_GPU_BATCHED=1`).
///
/// Default OFF: when unset the single-token GPU forward path is byte-identical.
/// Consulted ONLY inside the batched forward orchestration (never by the
/// single-token `layers.rs` driver), so the shipped default forward is untouched.
#[cfg(feature = "cuda")]
pub fn ox_gpu_batched_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("OX_GPU_BATCHED")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
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

#[cfg(feature = "cuda")]
#[path = "cuda/cuda_decode_graph.rs"]
mod cuda_decode_graph;
#[cfg(feature = "cuda")]
pub use cuda_decode_graph::{
    gpu_decode_graph_eligible, gpu_decode_graph_reset, gpu_decode_graph_set_token,
    gpu_decode_layer_graph_begin, gpu_decode_layer_graph_end, ox_gpu_cuda_graph_enabled,
};

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

#[cfg(all(test, feature = "cuda"))]
#[path = "cuda/ffn_fusion_cuda_fixture.rs"]
mod ffn_fusion_cuda_fixture;

#[cfg(all(test, feature = "cuda"))]
#[path = "cuda/ffn_fusion_cuda_tests.rs"]
mod ffn_fusion_cuda_tests;
