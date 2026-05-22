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
}
