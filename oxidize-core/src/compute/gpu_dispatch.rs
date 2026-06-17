//! Unified GPU backend dispatch (CUDA + ROCm/HIP).

use crate::gguf::GgufQuantizationType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveGpu {
    Cuda,
    Rocm,
}

pub fn active_gpu() -> Option<ActiveGpu> {
    #[cfg(feature = "cuda")]
    if crate::cuda::cuda_build_info().detected_at_build {
        return Some(ActiveGpu::Cuda);
    }
    #[cfg(feature = "rocm")]
    if crate::rocm::rocm_build_info().detected_at_build {
        return Some(ActiveGpu::Rocm);
    }
    None
}

pub fn gemv_f32(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), String> {
    match active_gpu() {
        #[cfg(feature = "cuda")]
        Some(ActiveGpu::Cuda) => crate::cuda::gemv_f32_cuda(matrix, rows, cols, vector, output)
            .map_err(|e| format!("{e:?}")),
        #[cfg(feature = "rocm")]
        Some(ActiveGpu::Rocm) => crate::rocm::gemv_f32_rocm(matrix, rows, cols, vector, output)
            .map_err(|e| format!("{e:?}")),
        #[cfg(not(any(feature = "cuda", feature = "rocm")))]
        _ => {
            let _ = (matrix, rows, cols, vector, output);
            Err("no GPU backend available".to_string())
        }
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        None => Err("no GPU backend available".to_string()),
    }
}

pub fn gemv_quantized(
    quantization: GgufQuantizationType,
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), String> {
    match active_gpu() {
        #[cfg(feature = "cuda")]
        Some(ActiveGpu::Cuda) => dispatch_cuda_quant(
            quantization,
            quantized_matrix,
            rows,
            cols,
            vector,
            output,
        ),
        #[cfg(feature = "rocm")]
        Some(ActiveGpu::Rocm) => dispatch_rocm_quant(
            quantization,
            quantized_matrix,
            rows,
            cols,
            vector,
            output,
        ),
        #[cfg(not(any(feature = "cuda", feature = "rocm")))]
        _ => {
            let _ = (
                quantization,
                quantized_matrix,
                rows,
                cols,
                vector,
                output,
            );
            Err("no GPU backend available".to_string())
        }
        #[cfg(any(feature = "cuda", feature = "rocm"))]
        None => Err("no GPU backend available".to_string()),
    }
}

#[cfg(feature = "cuda")]
fn dispatch_cuda_quant(
    quantization: GgufQuantizationType,
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), String> {
    use crate::compute::quantization::{BLOCK_Q8_K_BYTES, QK_K};
    use crate::tensor::quantize_vector_q8_k_into;

    match quantization {
        GgufQuantizationType::Q8_0 => crate::cuda::gemv_q8_0_direct_cuda(
            quantized_matrix,
            rows,
            cols,
            vector,
            output,
        )
        .map_err(|e| format!("{e:?}")),
        GgufQuantizationType::Q4_0 => crate::cuda::gemv_q4_0_direct_cuda(
            quantized_matrix,
            rows,
            cols,
            vector,
            output,
        )
        .map_err(|e| format!("{e:?}")),
        GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M if cols.is_multiple_of(QK_K) => {
            let blocks_per_row = cols / QK_K;
            let mut q8k = vec![0_u8; blocks_per_row * BLOCK_Q8_K_BYTES];
            quantize_vector_q8_k_into(vector, blocks_per_row, &mut q8k);
            crate::cuda::gemv_q4_k_direct_cuda(quantized_matrix, rows, cols, &q8k, output)
                .map_err(|e| format!("{e:?}"))
        }
        GgufQuantizationType::IQ1_S if cols.is_multiple_of(QK_K) => {
            crate::cuda::gemv_iq1_s_direct_cuda(quantized_matrix, rows, cols, vector, output)
                .map_err(|e| format!("{e:?}"))
        }
        GgufQuantizationType::IQ1_M if cols.is_multiple_of(QK_K) => {
            crate::cuda::gemv_iq1_m_direct_cuda(quantized_matrix, rows, cols, vector, output)
                .map_err(|e| format!("{e:?}"))
        }
        GgufQuantizationType::NVFP4 => crate::cuda::gemv_nvfp4_direct_cuda(
            quantized_matrix,
            rows,
            cols,
            vector,
            output,
        )
        .map_err(|e| format!("{e:?}")),
        _ => crate::cuda::gemv_quantized_cuda(
            quantization,
            quantized_matrix,
            rows,
            cols,
            vector,
            output,
        )
        .map_err(|e| format!("{e:?}")),
    }
}

#[cfg(feature = "rocm")]
fn dispatch_rocm_quant(
    quantization: GgufQuantizationType,
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), String> {
    crate::rocm::gemv_quantized_rocm(
        quantization,
        quantized_matrix,
        rows,
        cols,
        vector,
        output,
    )
    .map_err(|e| format!("{e:?}"))
}
