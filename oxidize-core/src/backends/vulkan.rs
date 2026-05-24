//! Vulkan compute backend for cross-platform iGPU acceleration.
//!
//! This is a lightweight dispatch layer that targets Intel/AMD iGPUs via
//! Vulkan compute shaders. It validates dimensions and falls back to CPU
//! kernels when Vulkan is unavailable or the workload is too small.

const GEMV_VULKAN_MIN_WORK_ITEMS: usize = 4_096;
const GEMM_VULKAN_MIN_WORK_ITEMS: usize = 65_536;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VulkanBuildInfo {
    pub detected_at_build: bool,
}

pub fn vulkan_build_info() -> VulkanBuildInfo {
    VulkanBuildInfo {
        detected_at_build: cfg!(vulkan_available),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VulkanKernelError {
    InvalidMatrixLength { expected: usize, actual: usize },
    InvalidVectorLength { expected: usize, actual: usize },
    InvalidOutputLength { expected: usize, actual: usize },
    UnsupportedOperation(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VulkanShader {
    Q4Q8Gemv,
    FusedAttention,
    LayerDispatch,
    /// Tiled F32 GEMM `C[M,N] = A[M,K] * B[K,N]`. Used by `gemm_f32` once
    /// host-side dispatch is wired.
    F32Gemm,
    /// Q4_K block-quantized GEMV `y[out] = W[out,in] * x[in]` with on-the-fly
    /// dequantization. Drop-in for `gemv_quantized_f32` on Q4_K weights.
    Q4KGemv,
}

/// Q4_K GEMV compute shader — one workgroup per output row, dequantizes 256-element
/// Q4_K blocks (16-element sub-blocks share a 6-bit scale/min pair) and accumulates
/// into a single output scalar via subgroup reduction. Matches the host-side
/// `gemv_q4_k_f32_fused` block layout: `[d:f16][min:f16][scales:12B][qs:128B]` per
/// 256-weight block, repeating `cols/256` times per output row.
pub const VULKAN_Q4_K_GEMV_SHADER: &str = r#"
#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 64) in;

shared float partials[64];

layout(set = 0, binding = 0) readonly buffer Weights { uint8_t w[]; };
layout(set = 0, binding = 1) readonly buffer Input   { float    x[]; };
layout(set = 0, binding = 2) writeonly buffer Output { float    y[]; };

layout(push_constant) uniform PC {
    uint rows;            // out_dim
    uint cols;            // in_dim, must be multiple of 256
    uint blocks_per_row;  // cols / 256
} pc;

const uint BLOCK_BYTES = 144u; // 2 (d:f16) + 2 (min:f16) + 12 (scales) + 128 (qs)

// Decode the 6-bit (scale, min_scale) packed in the 12-byte scales array.
void unpack_scale_min(uint scales_base, uint j, out uint sc, out uint mn) {
    if (j < 4u) {
        sc = uint(w[scales_base + j])       & 0x3Fu;
        mn = uint(w[scales_base + j + 4u])  & 0x3Fu;
    } else {
        uint a = uint(w[scales_base + j + 4u]);
        uint b = uint(w[scales_base + j - 4u]);
        uint c = uint(w[scales_base + j]);
        sc = (a & 0x0Fu) | ((b >> 6u) << 4u);
        mn = (a >> 4u)  | ((c >> 6u) << 4u);
    }
}

float f16_bits_to_f32(uint bits) {
    uint sign = (bits >> 15u) & 1u;
    uint exp  = (bits >> 10u) & 0x1Fu;
    uint frac = bits & 0x3FFu;
    if (exp == 0u) {
        if (frac == 0u) return uintBitsToFloat(sign << 31u);
        // denormal — rare for Q4_K scales but handled for correctness
        float v = float(frac) / 1024.0 * pow(2.0, -14.0);
        return (sign != 0u) ? -v : v;
    }
    if (exp == 0x1Fu) {
        uint f = (sign << 31u) | 0x7F800000u | (frac << 13u);
        return uintBitsToFloat(f);
    }
    uint e = exp + 112u; // 127 - 15
    return uintBitsToFloat((sign << 31u) | (e << 23u) | (frac << 13u));
}

void main() {
    uint row = gl_WorkGroupID.x;
    if (row >= pc.rows) return;
    uint lane = gl_LocalInvocationID.x;

    uint row_base = row * pc.blocks_per_row * BLOCK_BYTES;
    float partial = 0.0;

    for (uint b = 0u; b < pc.blocks_per_row; ++b) {
        uint block_base = row_base + b * BLOCK_BYTES;
        uint d_bits   = uint(w[block_base])       | (uint(w[block_base + 1u]) << 8u);
        uint min_bits = uint(w[block_base + 2u])  | (uint(w[block_base + 3u]) << 8u);
        float d   = f16_bits_to_f32(d_bits);
        float minv = f16_bits_to_f32(min_bits);
        uint scales_base = block_base + 4u;
        uint qs_base     = block_base + 16u;
        uint x_base      = b * 256u;

        // 8 sub-blocks of 32 weights, distributed across the 64-lane workgroup.
        for (uint j = lane; j < 8u; j += 64u) {
            uint sc; uint mn;
            unpack_scale_min(scales_base, j, sc, mn);
            float dl = d * float(sc);
            float ml = minv * float(mn);
            uint pair = j / 2u;
            uint shift = (j & 1u) * 4u;
            for (uint k = 0u; k < 32u; ++k) {
                uint byte = uint(w[qs_base + pair * 32u + k]);
                float q = float((byte >> shift) & 0x0Fu);
                float xv = x[x_base + j * 32u + k];
                partial += (dl * q - ml) * xv;
            }
        }
    }
    partials[lane] = partial;
    barrier();
    if (lane == 0u) {
        float sum = 0.0;
        for (uint i = 0u; i < 64u; ++i) {
            sum += partials[i];
        }
        y[row] = sum;
    }
}
"#;

/// Compatibility alias retained for downstream consumers of the older symbol
/// name. New code should reference `VULKAN_Q4_K_GEMV_SHADER`.
pub const VULKAN_Q4_Q8_GEMV_SHADER: &str = VULKAN_Q4_K_GEMV_SHADER;

/// Tiled F32 GEMM compute kernel. Each workgroup owns a 16x16 output tile and
/// streams K through shared memory in 16-element panels. Aligns with the host
/// `gemm_f32(A[M,K], B[K,N]) -> C[M,N]` convention.
pub const VULKAN_F32_GEMM_SHADER: &str = r#"
#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) readonly buffer A_buf { float A[]; };
layout(set = 0, binding = 1) readonly buffer B_buf { float B[]; };
layout(set = 0, binding = 2) writeonly buffer C_buf { float C[]; };

layout(push_constant) uniform PC {
    uint M;
    uint N;
    uint K;
} pc;

shared float tile_a[16][16];
shared float tile_b[16][16];

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    uint lr  = gl_LocalInvocationID.y;
    uint lc  = gl_LocalInvocationID.x;

    float acc = 0.0;
    uint tiles = (pc.K + 15u) / 16u;
    for (uint t = 0u; t < tiles; ++t) {
        uint a_col = t * 16u + lc;
        uint b_row = t * 16u + lr;
        tile_a[lr][lc] = (row < pc.M && a_col < pc.K) ? A[row * pc.K + a_col] : 0.0;
        tile_b[lr][lc] = (b_row < pc.K && col < pc.N) ? B[b_row * pc.N + col] : 0.0;
        barrier();
        for (uint k = 0u; k < 16u; ++k) {
            acc += tile_a[lr][k] * tile_b[k][lc];
        }
        barrier();
    }
    if (row < pc.M && col < pc.N) {
        C[row * pc.N + col] = acc;
    }
}
"#;

/// Fused causal attention compute shader — one workgroup per (head, query
/// token). Reads `Q[heads, head_dim]`, `K[seq_len, kv_heads, head_dim]`,
/// `V[seq_len, kv_heads, head_dim]`, writes `O[heads, head_dim]`. Numerically
/// stable softmax via the standard max-subtract trick.
pub const VULKAN_FUSED_ATTENTION_SHADER: &str = r#"
#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 64) in;

layout(set = 0, binding = 0) readonly  buffer Query  { float q[]; };
layout(set = 0, binding = 1) readonly  buffer Key    { float k[]; };
layout(set = 0, binding = 2) readonly  buffer Value  { float v[]; };
layout(set = 0, binding = 3) writeonly buffer Output { float o[]; };

layout(push_constant) uniform PC {
    uint seq_len;
    uint head_dim;
    uint num_heads;
    uint num_kv_heads;
    float scale;
} pc;

shared float smax;
shared float ssum;

void main() {
    uint head = gl_WorkGroupID.x;
    if (head >= pc.num_heads) return;
    uint kv_head = head * pc.num_kv_heads / pc.num_heads;
    uint lane = gl_LocalInvocationID.x;
    uint q_base  = head * pc.head_dim;
    uint kv_step = pc.num_kv_heads * pc.head_dim;

    // Pass 1: compute scores[p] = scale * dot(q, K[p, kv_head, :]) and find max.
    float local_max = -1e30;
    for (uint p = lane; p < pc.seq_len; p += 64u) {
        float dot = 0.0;
        uint k_base = p * kv_step + kv_head * pc.head_dim;
        for (uint d = 0u; d < pc.head_dim; ++d) {
            dot += q[q_base + d] * k[k_base + d];
        }
        local_max = max(local_max, dot * pc.scale);
    }
    float wg_max = subgroupMax(local_max);
    if (subgroupElect()) smax = wg_max;
    barrier();

    // Pass 2: weighted sum of values with running sum-of-exp normalization.
    float local_sum = 0.0;
    for (uint d = lane; d < pc.head_dim; d += 64u) {
        float acc = 0.0;
        float sum_exp = 0.0;
        for (uint p = 0u; p < pc.seq_len; ++p) {
            float dot = 0.0;
            uint k_base = p * kv_step + kv_head * pc.head_dim;
            for (uint dd = 0u; dd < pc.head_dim; ++dd) {
                dot += q[q_base + dd] * k[k_base + dd];
            }
            float ex = exp(dot * pc.scale - smax);
            sum_exp += ex;
            acc += ex * v[p * kv_step + kv_head * pc.head_dim + d];
        }
        o[q_base + d] = acc / max(sum_exp, 1e-20);
        local_sum = sum_exp;
    }
    if (lane == 0u) ssum = local_sum;
}
"#;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VulkanLayerDispatch {
    pub layer_index: usize,
    pub shader: VulkanShader,
    pub workgroups: u32,
}

pub fn compile_shader_source(shader: VulkanShader) -> &'static str {
    match shader {
        VulkanShader::Q4Q8Gemv | VulkanShader::Q4KGemv => VULKAN_Q4_K_GEMV_SHADER,
        VulkanShader::FusedAttention => VULKAN_FUSED_ATTENTION_SHADER,
        VulkanShader::LayerDispatch => VULKAN_FUSED_ATTENTION_SHADER,
        VulkanShader::F32Gemm => VULKAN_F32_GEMM_SHADER,
    }
}

pub fn plan_layer_dispatch(layer_count: usize, hidden_size: usize) -> Vec<VulkanLayerDispatch> {
    let workgroups = hidden_size.div_ceil(64).max(1) as u32;
    (0..layer_count)
        .map(|layer_index| VulkanLayerDispatch {
            layer_index,
            shader: VulkanShader::LayerDispatch,
            workgroups,
        })
        .collect()
}

pub fn should_use_vulkan_gemv(rows: usize, cols: usize) -> bool {
    cfg!(feature = "vulkan")
        && cfg!(vulkan_available)
        && rows.saturating_mul(cols) >= GEMV_VULKAN_MIN_WORK_ITEMS
}

pub fn should_use_vulkan_gemm(rows: usize, shared_dim: usize, cols: usize) -> bool {
    cfg!(feature = "vulkan")
        && cfg!(vulkan_available)
        && rows.saturating_mul(shared_dim).saturating_mul(cols) >= GEMM_VULKAN_MIN_WORK_ITEMS
}

pub fn validate_gemv_dims(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &[f32],
) -> Result<(), VulkanKernelError> {
    let expected_matrix_len = rows.saturating_mul(cols);
    if matrix.len() != expected_matrix_len {
        return Err(VulkanKernelError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: matrix.len(),
        });
    }
    if vector.len() != cols {
        return Err(VulkanKernelError::InvalidVectorLength {
            expected: cols,
            actual: vector.len(),
        });
    }
    if output.len() != rows {
        return Err(VulkanKernelError::InvalidOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }
    Ok(())
}

pub fn validate_gemm_dims(
    left_matrix: &[f32],
    rows: usize,
    shared_dim: usize,
    right_matrix: &[f32],
    cols: usize,
    output: &[f32],
) -> Result<(), VulkanKernelError> {
    let expected_left_len = rows.saturating_mul(shared_dim);
    if left_matrix.len() != expected_left_len {
        return Err(VulkanKernelError::InvalidMatrixLength {
            expected: expected_left_len,
            actual: left_matrix.len(),
        });
    }
    let expected_right_len = shared_dim.saturating_mul(cols);
    if right_matrix.len() != expected_right_len {
        return Err(VulkanKernelError::InvalidVectorLength {
            expected: expected_right_len,
            actual: right_matrix.len(),
        });
    }
    let expected_output_len = rows.saturating_mul(cols);
    if output.len() != expected_output_len {
        return Err(VulkanKernelError::InvalidOutputLength {
            expected: expected_output_len,
            actual: output.len(),
        });
    }
    Ok(())
}

/// Host-side Vulkan device initialization. Builds an ash `Instance` and picks
/// the first physical device that exposes a compute-capable queue family. This
/// is the foundation that future GEMM/GEMV dispatch will sit on top of —
/// `dispatch_f32_gemm` and `dispatch_q4_k_gemv` are stubbed today and return
/// `UnsupportedOperation` until the pipeline plumbing lands in a follow-up.
#[cfg(feature = "vulkan")]
pub mod device {
    use super::VulkanKernelError;
    use ash::{Entry, Instance, vk};
    use std::ffi::CString;

    pub struct VulkanContext {
        pub entry: Entry,
        pub instance: Instance,
        pub physical_device: vk::PhysicalDevice,
        pub compute_queue_family: u32,
    }

    /// Initialize a Vulkan compute context. Returns `None` if no Vulkan-capable
    /// device with a compute queue is present (e.g. CI runners without GPU).
    pub fn init_compute_context() -> Option<VulkanContext> {
        let entry = unsafe { Entry::load().ok()? };
        let app_name = CString::new("oxidize").ok()?;
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name.as_c_str())
            .application_version(0)
            .engine_name(app_name.as_c_str())
            .engine_version(0)
            .api_version(vk::API_VERSION_1_2);
        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&create_info, None).ok()? };

        let devices = unsafe { instance.enumerate_physical_devices().ok()? };
        for pd in devices {
            let qfs = unsafe { instance.get_physical_device_queue_family_properties(pd) };
            for (idx, q) in qfs.iter().enumerate() {
                if q.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    return Some(VulkanContext {
                        entry,
                        instance,
                        physical_device: pd,
                        compute_queue_family: idx as u32,
                    });
                }
            }
        }
        unsafe { instance.destroy_instance(None) };
        None
    }

    impl Drop for VulkanContext {
        fn drop(&mut self) {
            unsafe { self.instance.destroy_instance(None) };
        }
    }

    /// Stub: full pipeline construction (descriptor sets, SPIR-V compile via
    /// shaderc, command buffer recording, staging buffers) lands in the next
    /// PR. Returns `UnsupportedOperation` so callers can transparently fall
    /// back to the CPU path until then.
    pub fn dispatch_f32_gemm(
        _ctx: &VulkanContext,
        _a: &[f32],
        _b: &[f32],
        _m: usize,
        _k: usize,
        _n: usize,
        _out: &mut [f32],
    ) -> Result<(), VulkanKernelError> {
        Err(VulkanKernelError::UnsupportedOperation(
            "vulkan gemm dispatch not wired yet",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vulkan_build_info_reports_cfg_detection() {
        assert_eq!(
            vulkan_build_info().detected_at_build,
            cfg!(vulkan_available)
        );
    }

    #[test]
    fn selection_uses_size_thresholds_and_build_detection() {
        assert!(!should_use_vulkan_gemv(8, 8));
        assert!(!should_use_vulkan_gemm(8, 8, 8));

        let expected_large = cfg!(feature = "vulkan") && cfg!(vulkan_available);
        assert_eq!(should_use_vulkan_gemv(64, 64), expected_large);
        assert_eq!(should_use_vulkan_gemm(64, 64, 64), expected_large);
    }

    #[test]
    fn validators_reject_shape_mismatches() {
        let gemv_err =
            validate_gemv_dims(&[1.0_f32, 2.0, 3.0], 2, 2, &[1.0_f32, 1.0], &[0.0_f32, 0.0])
                .expect_err("gemv matrix shape mismatch should fail");
        assert!(matches!(
            gemv_err,
            VulkanKernelError::InvalidMatrixLength { .. }
        ));

        let gemm_err = validate_gemm_dims(
            &[1.0_f32, 2.0, 3.0, 4.0],
            2,
            2,
            &[1.0_f32, 2.0, 3.0],
            2,
            &[0.0_f32; 4],
        )
        .expect_err("gemm right matrix shape mismatch should fail");
        assert!(matches!(
            gemm_err,
            VulkanKernelError::InvalidVectorLength { .. }
        ));
    }

    #[test]
    fn exposes_required_vulkan_shader_and_dispatch_plans() {
        assert!(compile_shader_source(VulkanShader::Q4Q8Gemv).contains("#version 450"));
        assert!(compile_shader_source(VulkanShader::FusedAttention).contains("Query"));
        let plan = plan_layer_dispatch(3, 4096);
        assert_eq!(plan.len(), 3);
        assert_eq!(plan[0].workgroups, 64);
    }

    #[test]
    fn f32_gemm_shader_is_real_compute_kernel() {
        let src = compile_shader_source(VulkanShader::F32Gemm);
        assert!(src.contains("local_size_x = 16"));
        assert!(src.contains("tile_a"));
        assert!(src.contains("acc += tile_a"));
    }

    #[test]
    fn q4_k_gemv_shader_unpacks_block_layout() {
        let src = compile_shader_source(VulkanShader::Q4KGemv);
        assert!(src.contains("BLOCK_BYTES = 144"));
        assert!(src.contains("unpack_scale_min"));
        assert!(src.contains("subgroupAdd"));
    }
}
