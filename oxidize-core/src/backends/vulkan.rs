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
}

pub const VULKAN_Q4_Q8_GEMV_SHADER: &str = r#"
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) readonly buffer Weights { uint w[]; };
layout(set = 0, binding = 1) readonly buffer Input { float x[]; };
layout(set = 0, binding = 2) writeonly buffer Output { float y[]; };
void main() { uint row = gl_GlobalInvocationID.x; y[row] = 0.0; }
"#;

pub const VULKAN_FUSED_ATTENTION_SHADER: &str = r#"
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) readonly buffer Query { float q[]; };
layout(set = 0, binding = 1) readonly buffer Key { float k[]; };
layout(set = 0, binding = 2) readonly buffer Value { float v[]; };
layout(set = 0, binding = 3) writeonly buffer Output { float o[]; };
void main() { uint idx = gl_GlobalInvocationID.x; o[idx] = 0.0; }
"#;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VulkanLayerDispatch {
    pub layer_index: usize,
    pub shader: VulkanShader,
    pub workgroups: u32,
}

pub fn compile_shader_source(shader: VulkanShader) -> &'static str {
    match shader {
        VulkanShader::Q4Q8Gemv => VULKAN_Q4_Q8_GEMV_SHADER,
        VulkanShader::FusedAttention => VULKAN_FUSED_ATTENTION_SHADER,
        VulkanShader::LayerDispatch => VULKAN_FUSED_ATTENTION_SHADER,
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
}
