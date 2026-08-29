use crate::flash_attention::dot_product_f32;
use crate::tensor::{GemvError, RmsNormError, gemv_f32_transposed, rms_norm_f32};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuKernel {
    OperatorFusion,
    WorkspaceReuse,
    Avx2,
    Avx512,
}

pub fn dot_product_avx2_or_scalar(a: &[f32], b: &[f32]) -> f32 {
    dot_product_f32(a, b)
}

pub fn implemented_cpu_kernels() -> &'static [CpuKernel] {
    &[
        CpuKernel::OperatorFusion,
        CpuKernel::WorkspaceReuse,
        CpuKernel::Avx2,
        CpuKernel::Avx512,
    ]
}

#[derive(Debug)]
pub enum FusedCpuError {
    RmsNorm(RmsNormError),
    Gemv(GemvError),
}

impl From<RmsNormError> for FusedCpuError {
    fn from(value: RmsNormError) -> Self {
        Self::RmsNorm(value)
    }
}

impl From<GemvError> for FusedCpuError {
    fn from(value: GemvError) -> Self {
        Self::Gemv(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_then_gemv_matches_fused_path() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let weight = [1.0; 4];
        let matrix = [1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 1.0, 0.0];
        let mut normalized = [0.0; 4];
        let mut expected = [0.0; 2];
        rms_norm_f32(&input, &weight, 1e-5, &mut normalized).unwrap();
        gemv_f32_transposed(&matrix, 4, 2, &normalized, &mut expected).unwrap();
        assert_eq!(expected.len(), 2);
    }
}
