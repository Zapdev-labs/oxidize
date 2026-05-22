use crate::flash_attention::dot_product_f32;
use crate::tensor::{
    GemmError, GemvError, RmsNormError, gemm_f32, gemv_f32_transposed, rms_norm_f32,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuKernel {
    OperatorFusion,
    WorkspaceReuse,
    Avx2,
    Avx512,
}

#[derive(Debug, Default, Clone)]
pub struct CpuWorkspace {
    scratch: Vec<f32>,
}

impl CpuWorkspace {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            scratch: Vec::with_capacity(capacity),
        }
    }

    pub fn get(&mut self, len: usize) -> &mut [f32] {
        self.scratch.resize(len, 0.0);
        &mut self.scratch
    }

    pub fn capacity(&self) -> usize {
        self.scratch.capacity()
    }
}

pub fn fused_rms_norm_gemv_f32_transposed(
    params: FusedRmsNormGemv<'_>,
    workspace: &mut CpuWorkspace,
    output: &mut [f32],
) -> Result<(), FusedCpuError> {
    let normalized = workspace.get(params.input.len());
    rms_norm_f32(params.input, params.norm_weight, params.eps, normalized)?;
    gemv_f32_transposed(params.matrix, params.rows, params.cols, normalized, output)?;
    Ok(())
}

pub struct FusedRmsNormGemv<'a> {
    pub input: &'a [f32],
    pub norm_weight: &'a [f32],
    pub eps: f32,
    pub matrix: &'a [f32],
    pub rows: usize,
    pub cols: usize,
}

pub fn matmul_reuse_workspace<'a>(
    left: &[f32],
    rows: usize,
    shared_dim: usize,
    right: &[f32],
    cols: usize,
    workspace: &'a mut CpuWorkspace,
) -> Result<&'a [f32], GemmError> {
    let out = workspace.get(rows.saturating_mul(cols));
    gemm_f32(left, rows, shared_dim, right, cols, out)?;
    Ok(out)
}

pub fn dot_product_avx2_or_scalar(a: &[f32], b: &[f32]) -> f32 {
    dot_product_f32(a, b)
}

pub fn dot_product_avx512_or_scalar(a: &[f32], b: &[f32]) -> f32 {
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
    fn fused_norm_gemv_matches_unfused_path() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let weight = [1.0; 4];
        let matrix = [1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 1.0, 0.0];
        let mut workspace = CpuWorkspace::default();
        let mut fused = [0.0; 2];
        fused_rms_norm_gemv_f32_transposed(
            FusedRmsNormGemv {
                input: &input,
                norm_weight: &weight,
                eps: 1e-5,
                matrix: &matrix,
                rows: 4,
                cols: 2,
            },
            &mut workspace,
            &mut fused,
        )
        .unwrap();

        let mut normalized = [0.0; 4];
        let mut expected = [0.0; 2];
        rms_norm_f32(&input, &weight, 1e-5, &mut normalized).unwrap();
        gemv_f32_transposed(&matrix, 4, 2, &normalized, &mut expected).unwrap();
        assert_eq!(fused, expected);
    }
}
