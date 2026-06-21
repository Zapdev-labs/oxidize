use super::*;

/// Whether [`gemv_quantized_cuda`] has a GPU dequant kernel for this type.
/// Callers should fall back to the CPU quantized path when this is `false`.
#[cfg(feature = "cuda")]
pub fn supports_quantized_gpu(quantization: GgufQuantizationType) -> bool {
    dequant_kernel_for(quantization).is_some()
}

/// GPU dequantization kernel name + raw block size in bytes + decoded values
/// per block, for a quantization type. Returns `None` for types without a GPU
/// dequant kernel (callers fall back to the CPU quantized path).
#[cfg(feature = "cuda")]
fn dequant_kernel_for(quantization: GgufQuantizationType) -> Option<(&'static str, usize, usize)> {
    match quantization {
        GgufQuantizationType::Q8_0 => Some(("dequant_q8_0_kernel", 34, 32)),
        GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M => {
            Some(("dequant_q4_k_kernel", 144, 256))
        }
        GgufQuantizationType::Q6_K => Some(("dequant_q6_k_kernel", 210, 256)),
        GgufQuantizationType::Q2_K => Some(("dequant_q2_k_kernel", 84, 256)),
        GgufQuantizationType::NVFP4 => Some(("dequant_nvfp4_kernel", 36, 64)),
        _ => None,
    }
}

/// Q8_0 on-the-fly GEMV: read quantized blocks [scale_f16 + 32 int8]
/// directly in the dot-product loop.  No dequantization pass.
#[cfg(feature = "cuda")]
pub fn gemv_q8_0_direct_cuda(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvCudaError> {
    validate_q8_0_gemv_dims(quantized_matrix, rows, cols, vector, output)?;

    let rows_u32 = u32::try_from(rows).map_err(|_| GemvCudaError::InvalidOutputLength {
        expected: u32::MAX as usize,
        actual: rows,
    })?;
    let cols_u32 = u32::try_from(cols).map_err(|_| GemvCudaError::InvalidVectorLength {
        expected: u32::MAX as usize,
        actual: cols,
    })?;

    with_gpu(|gpu| {
        let key = bytes_cache_key(quantized_matrix);
        gpu.ensure_resident_quant(key, quantized_matrix)?;
        let matrix_ptr = gpu
            .resident_quant
            .get(&key)
            .ok_or_else(|| "Q8_0 weight missing from resident cache".to_string())?
            .as_device_ptr();

        let vector_device = cust::memory::DeviceBuffer::from_slice(vector).map_err(stringify)?;
        let output_device = gpu.get_f32_buffer(rows).map_err(stringify)?;

        let block_size = 256_u32;
        let grid_size = rows_u32.saturating_mul(32).div_ceil(block_size);
        let function = gpu
            .module
            .get_function(GEMV_Q8_0_DIRECT_KERNEL_NAME)
            .map_err(stringify)?;
        let stream = &gpu.stream;
        unsafe {
            cust::launch!(
                function<<<grid_size, block_size, 0, stream>>>(
                    matrix_ptr,
                    vector_device.as_device_ptr(),
                    output_device.as_device_ptr(),
                    rows_u32,
                    cols_u32
                )
            )
            .map_err(stringify)?;
        }
        output_device.copy_to(output).map_err(stringify)?;
        gpu.return_f32_buffer(output_device);
        Ok(())
    })
    .map_err(GemvCudaError::Cuda)
}

/// Q4_0 on-the-fly GEMV: read 18-byte blocks [scale_f16 + 16 bytes of
/// nibbles] directly.  Only 0.56 bytes per parameter.
#[cfg(feature = "cuda")]
pub fn gemv_q4_0_direct_cuda(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvCudaError> {
    use crate::quantization::{BLOCK_Q4_0_SIZE, QK4_0};

    if !cols.is_multiple_of(QK4_0) {
        return Err(GemvCudaError::InvalidVectorLength {
            expected: cols.div_ceil(QK4_0) * QK4_0,
            actual: cols,
        });
    }
    let blocks_per_row = cols / QK4_0;
    let expected_matrix_len = rows
        .saturating_mul(blocks_per_row)
        .saturating_mul(BLOCK_Q4_0_SIZE);
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvCudaError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
        });
    }
    if vector.len() != cols {
        return Err(GemvCudaError::InvalidVectorLength {
            expected: cols,
            actual: vector.len(),
        });
    }
    if output.len() != rows {
        return Err(GemvCudaError::InvalidOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }

    let rows_u32 = u32::try_from(rows).map_err(|_| GemvCudaError::InvalidOutputLength {
        expected: u32::MAX as usize,
        actual: rows,
    })?;
    let cols_u32 = u32::try_from(cols).map_err(|_| GemvCudaError::InvalidVectorLength {
        expected: u32::MAX as usize,
        actual: cols,
    })?;

    with_gpu(|gpu| {
        let key = bytes_cache_key(quantized_matrix);
        gpu.ensure_resident_quant(key, quantized_matrix)?;
        let matrix_ptr = gpu
            .resident_quant
            .get(&key)
            .ok_or_else(|| "Q4_0 weight missing from resident cache".to_string())?
            .as_device_ptr();

        let vector_device = cust::memory::DeviceBuffer::from_slice(vector).map_err(stringify)?;
        let output_device = gpu.get_f32_buffer(rows).map_err(stringify)?;

        let block_size = 256_u32;
        let grid_size = rows_u32.saturating_mul(32).div_ceil(block_size);
        let function = gpu
            .module
            .get_function(GEMV_Q4_0_DIRECT_KERNEL_NAME)
            .map_err(stringify)?;
        let stream = &gpu.stream;
        unsafe {
            cust::launch!(
                function<<<grid_size, block_size, 0, stream>>>(
                    matrix_ptr,
                    vector_device.as_device_ptr(),
                    output_device.as_device_ptr(),
                    rows_u32,
                    cols_u32
                )
            )
            .map_err(stringify)?;
        }
        output_device.copy_to(output).map_err(stringify)?;
        gpu.return_f32_buffer(output_device);
        Ok(())
    })
    .map_err(GemvCudaError::Cuda)
}

pub fn validate_q4_k_gemv_dims(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    q8k: &[u8],
    output: &[f32],
) -> Result<(), GemvCudaError> {
    if !cols.is_multiple_of(QK_K) {
        return Err(GemvCudaError::InvalidVectorLength {
            expected: cols.div_ceil(QK_K) * QK_K,
            actual: cols,
        });
    }
    let blocks_per_row = cols / QK_K;
    let expected_matrix_len = rows
        .saturating_mul(blocks_per_row)
        .saturating_mul(BLOCK_Q4_K_SIZE);
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvCudaError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
        });
    }
    let expected_q8k_len = blocks_per_row * BLOCK_Q8_K_BYTES;
    if q8k.len() != expected_q8k_len {
        return Err(GemvCudaError::InvalidVectorLength {
            expected: expected_q8k_len,
            actual: q8k.len(),
        });
    }
    if output.len() != rows {
        return Err(GemvCudaError::InvalidOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }
    Ok(())
}

/// Q4_K on-the-fly GEMV via Q4_K × Q8_K dot products (OXK GPU path).
/// Weights stay compressed in VRAM; the input vector is quantized to Q8_K
/// once per token on the CPU (same layout as the OXK CPU kernels).
#[cfg(feature = "cuda")]
pub fn gemv_q4_k_direct_cuda(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    q8k: &[u8],
    output: &mut [f32],
) -> Result<(), GemvCudaError> {
    validate_q4_k_gemv_dims(quantized_matrix, rows, cols, q8k, output)?;

    let blocks_per_row = cols / QK_K;
    let rows_u32 = u32::try_from(rows).map_err(|_| GemvCudaError::InvalidOutputLength {
        expected: u32::MAX as usize,
        actual: rows,
    })?;
    let blocks_u32 =
        u32::try_from(blocks_per_row).map_err(|_| GemvCudaError::InvalidVectorLength {
            expected: u32::MAX as usize,
            actual: blocks_per_row,
        })?;

    with_gpu(|gpu| {
        let key = bytes_cache_key(quantized_matrix);
        gpu.ensure_resident_quant(key, quantized_matrix)?;
        let matrix_ptr = gpu
            .resident_quant
            .get(&key)
            .ok_or_else(|| "Q4_K weight missing from resident cache".to_string())?
            .as_device_ptr();

        let mut q8k_device = gpu.get_q8k_buffer(q8k.len()).map_err(stringify)?;
        q8k_device.copy_from(q8k).map_err(stringify)?;
        let output_device = gpu.get_f32_buffer(rows).map_err(stringify)?;

        let block_size = 256_u32;
        let grid_size = rows_u32.saturating_mul(32).div_ceil(block_size);
        let function = gpu
            .module
            .get_function(GEMV_Q4_K_DIRECT_KERNEL_NAME)
            .map_err(stringify)?;
        let stream = &gpu.stream;
        unsafe {
            cust::launch!(
                function<<<grid_size, block_size, 0, stream>>>(
                    matrix_ptr,
                    q8k_device.as_device_ptr(),
                    output_device.as_device_ptr(),
                    rows_u32,
                    blocks_u32
                )
            )
            .map_err(stringify)?;
        }
        output_device.copy_to(output).map_err(stringify)?;
        gpu.return_f32_buffer(output_device);
        gpu.return_q8k_buffer(q8k_device);
        Ok(())
    })
    .map_err(GemvCudaError::Cuda)
}

/// Q4_K × F32 GEMV where the input vector already lives in a GPU buffer
/// (`d_input`) and the result is written into `d_output` without any D2H copy.
///
/// This is the hot path for the GPU-native forward pass: the hidden state stays
/// on the GPU across all FFN operations, eliminating the per-layer CPU↔GPU
/// round-trips that otherwise dominate decode latency.
#[cfg(feature = "cuda")]
pub fn gemv_q4k_f32in_to_device_cuda(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    d_input: &cust::memory::DeviceBuffer<f32>,
    d_output: &cust::memory::DeviceBuffer<f32>,
) -> Result<(), GemvCudaError> {
    if !cols.is_multiple_of(256) {
        return Err(GemvCudaError::InvalidVectorLength {
            expected: cols.div_ceil(256) * 256,
            actual: cols,
        });
    }
    let blocks_per_row = cols / 256;
    let expected_len = rows.saturating_mul(blocks_per_row).saturating_mul(144);
    if quantized_matrix.len() != expected_len {
        return Err(GemvCudaError::InvalidMatrixLength {
            expected: expected_len,
            actual: quantized_matrix.len(),
        });
    }
    if d_input.len() < cols || d_output.len() < rows {
        return Err(GemvCudaError::InvalidOutputLength {
            expected: rows,
            actual: d_output.len(),
        });
    }
    let rows_u32 = rows as u32;
    let blocks_u32 = blocks_per_row as u32;

    with_gpu(|gpu| {
        let key = bytes_cache_key(quantized_matrix);
        gpu.ensure_resident_quant(key, quantized_matrix)?;
        let matrix_ptr = gpu
            .resident_quant
            .get(&key)
            .ok_or_else(|| "Q4K weight missing from resident cache".to_string())?
            .as_device_ptr();

        let block_size = 256_u32;
        let grid_size = rows_u32.saturating_mul(32).div_ceil(block_size);
        let function = gpu
            .module
            .get_function(GEMV_Q4K_F32IN_KERNEL_NAME)
            .map_err(stringify)?;
        let stream = &gpu.stream;
        unsafe {
            cust::launch!(
                function<<<grid_size, block_size, 0, stream>>>(
                    matrix_ptr,
                    d_input.as_device_ptr(),
                    d_output.as_device_ptr(),
                    rows_u32,
                    blocks_u32
                )
            )
            .map_err(stringify)?;
        }
        // No D2H copy — result stays on GPU in d_output.
        Ok(())
    })
    .map_err(GemvCudaError::Cuda)
}

/// True when a quantized weight matrix uses Q6_K blocks (210 B) rather than Q4_K (144 B).
#[cfg(feature = "cuda")]
pub(crate) fn weight_block_bytes_is_q6k(w: &[u8], rows: usize, blocks_per_row: usize) -> bool {
    blocks_per_row > 0 && rows > 0 && w.len() / (rows * blocks_per_row) >= 200
}

/// Quantize a device-resident F32 vector into a pooled Q8_K buffer (292 B / 256 values).
#[cfg(feature = "cuda")]
pub(crate) fn launch_quantize_f32_to_q8k_device(
    gpu: &mut super::GpuState,
    d_input: cust::memory::DevicePointer<f32>,
    d_q8k: &cust::memory::DeviceBuffer<u8>,
    blocks_per_row: u32,
) -> Result<(), String> {
    let fn_q = gpu
        .module
        .get_function(QUANTIZE_F32_TO_Q8K_KERNEL_NAME)
        .map_err(stringify)?;
    let stream = &gpu.stream;
    unsafe {
        cust::launch!(fn_q<<<blocks_per_row, 256u32, 0, stream>>>(
            d_input,
            d_q8k.as_device_ptr(),
            blocks_per_row
        ))
        .map_err(stringify)?;
    }
    Ok(())
}

/// Q4_K × Q8_K GEMV: `d_q8k` holds `blocks_per_row` Q8_K blocks; output → `d_output`.
#[cfg(feature = "cuda")]
pub(crate) fn launch_gemv_q4k_q8kin_device(
    gpu: &mut super::GpuState,
    w_ptr: cust::memory::DevicePointer<u8>,
    rows: u32,
    blocks_per_row: u32,
    d_q8k: cust::memory::DevicePointer<u8>,
    d_output: cust::memory::DevicePointer<f32>,
    _n_splits: u32,
) -> Result<(), String> {
    let block_size = 256_u32;
    let grid = rows.saturating_mul(32).div_ceil(block_size);
    let fn_gemv = gpu
        .module
        .get_function(GEMV_Q4_K_DIRECT_KERNEL_NAME)
        .map_err(stringify)?;
    let stream = &gpu.stream;
    unsafe {
        cust::launch!(fn_gemv<<<grid, block_size, 0, stream>>>(
            w_ptr,
            d_q8k,
            d_output,
            rows,
            blocks_per_row
        ))
        .map_err(stringify)?;
    }
    Ok(())
}

/// Launch F32-in GEMV (Q4_K or Q6_K) for device-resident activations.
#[cfg(feature = "cuda")]
pub(crate) fn launch_gemv_f32in_device(
    gpu: &mut super::GpuState,
    kern_name: &str,
    w_ptr: cust::memory::DevicePointer<u8>,
    d_input: cust::memory::DevicePointer<f32>,
    d_output: cust::memory::DevicePointer<f32>,
    rows: u32,
    blocks_per_row: u32,
) -> Result<(), String> {
    let block_size = 256_u32;
    let grid = rows.saturating_mul(32).div_ceil(block_size);
    let fn_gemv = gpu.module.get_function(kern_name).map_err(stringify)?;
    let stream = &gpu.stream;
    unsafe {
        cust::launch!(fn_gemv<<<grid, block_size, 0, stream>>>(
            w_ptr,
            d_input,
            d_output,
            rows,
            blocks_per_row
        ))
        .map_err(stringify)?;
    }
    Ok(())
}

/// Opt-in: fuse activation Q8_K quantization into the Q4_K GEMV kernel
/// (`gemv_q4k_q8k_fused_kernel`), eliminating the separate quantize launch and
/// the Q8_K VRAM round-trip. OFF by default (`OX_GPU_FUSED_MMQ=1` to enable)
/// so the shipped two-kernel path stays byte-identical until validated on GPU.
#[cfg(feature = "cuda")]
pub(super) fn ox_gpu_fused_mmq_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("OX_GPU_FUSED_MMQ")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false)
    })
}

/// Per-block dynamic shared-memory budget for the fused MMQ kernel. The fused
/// kernel holds the whole Q8_K activation (blocks_per_row × 292 B) in shared
/// memory; above this it falls back to the two-kernel path. 44 KB stays under
/// the 48 KB default per-block limit on every supported arch (no opt-in launch
/// attribute needed).
#[cfg(feature = "cuda")]
const FUSED_MMQ_MAX_SHMEM: usize = 44 * 1024;

/// Single-launch fused Q4_K × Q8_K MMQ GEMV. `blockDim.x` must be 256 (one
/// thread per activation value during the in-kernel quantization phase); the
/// block then computes 256/OX_WAVE rows (8 on CUDA) from the shared activation.
#[cfg(feature = "cuda")]
pub(crate) fn launch_gemv_q4k_q8k_fused_device(
    gpu: &mut super::GpuState,
    w_ptr: cust::memory::DevicePointer<u8>,
    d_input: cust::memory::DevicePointer<f32>,
    d_output: cust::memory::DevicePointer<f32>,
    rows: u32,
    blocks_per_row: u32,
) -> Result<(), String> {
    // The fused kernel's phase-1 quantization assigns one thread per Q8_K value,
    // so blockDim.x MUST be 256 (one full super-block). Phase 2 then derives
    // 256/OX_WAVE = 8 rows/block from it. Do not change without updating the
    // kernel's amax/bsum reduction.
    let block_size = 256_u32;
    debug_assert_eq!(
        block_size, 256,
        "fused MMQ kernel requires blockDim.x == 256"
    );
    let grid = rows.saturating_mul(32).div_ceil(block_size);
    let shmem = blocks_per_row.saturating_mul(292) as u32;
    let fn_gemv = gpu
        .module
        .get_function(GEMV_Q4K_Q8K_FUSED_KERNEL_NAME)
        .map_err(stringify)?;
    let stream = &gpu.stream;
    unsafe {
        cust::launch!(fn_gemv<<<grid, block_size, shmem, stream>>>(
            w_ptr,
            d_input,
            d_output,
            rows,
            blocks_per_row
        ))
        .map_err(stringify)?;
    }
    Ok(())
}

/// Device GEMV dispatch: Q4_K weights use quantize→Q8_K→DP4A; Q6_K falls back to F32-in.
#[cfg(feature = "cuda")]
pub(super) fn launch_gemv_q4k_q8kin_or_q6k_f32in_device(
    gpu: &mut super::GpuState,
    w: &[u8],
    rows: u32,
    blocks_per_row: u32,
    d_input: cust::memory::DevicePointer<f32>,
    d_q8k: &cust::memory::DeviceBuffer<u8>,
    d_output: cust::memory::DevicePointer<f32>,
    w_ptr: cust::memory::DevicePointer<u8>,
) -> Result<(), String> {
    let rows_usize = rows as usize;
    let bpr = blocks_per_row as usize;
    if weight_block_bytes_is_q6k(w, rows_usize, bpr) {
        let kern = GEMV_Q6K_F32IN_KERNEL_NAME;
        launch_gemv_f32in_device(gpu, kern, w_ptr, d_input, d_output, rows, blocks_per_row)
    } else if ox_gpu_fused_mmq_enabled() && bpr.saturating_mul(292) <= FUSED_MMQ_MAX_SHMEM {
        // One launch: quantize activation to shared Q8_K + DP4A GEMV.
        launch_gemv_q4k_q8k_fused_device(gpu, w_ptr, d_input, d_output, rows, blocks_per_row)
    } else {
        launch_quantize_f32_to_q8k_device(gpu, d_input, d_q8k, blocks_per_row)?;
        launch_gemv_q4k_q8kin_device(
            gpu,
            w_ptr,
            rows,
            blocks_per_row,
            d_q8k.as_device_ptr(),
            d_output,
            1,
        )
    }
}

#[cfg(feature = "cuda")]
fn gemv_superblock_direct_cuda(
    kernel_name: &str,
    block_bytes: usize,
    vals_per_block: usize,
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvCudaError> {
    if !cols.is_multiple_of(vals_per_block) {
        return Err(GemvCudaError::InvalidVectorLength {
            expected: cols.div_ceil(vals_per_block) * vals_per_block,
            actual: cols,
        });
    }
    let blocks_per_row = cols / vals_per_block;
    let expected_matrix_len = rows
        .saturating_mul(blocks_per_row)
        .saturating_mul(block_bytes);
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvCudaError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
        });
    }
    if vector.len() != cols {
        return Err(GemvCudaError::InvalidVectorLength {
            expected: cols,
            actual: vector.len(),
        });
    }
    if output.len() != rows {
        return Err(GemvCudaError::InvalidOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }

    let rows_u32 = u32::try_from(rows).map_err(|_| GemvCudaError::InvalidOutputLength {
        expected: u32::MAX as usize,
        actual: rows,
    })?;
    let blocks_u32 =
        u32::try_from(blocks_per_row).map_err(|_| GemvCudaError::InvalidVectorLength {
            expected: u32::MAX as usize,
            actual: blocks_per_row,
        })?;

    with_gpu(|gpu| {
        let key = bytes_cache_key(quantized_matrix);
        gpu.ensure_resident_quant(key, quantized_matrix)?;
        let matrix_ptr = gpu
            .resident_quant
            .get(&key)
            .ok_or_else(|| "quant weight missing from resident cache".to_string())?
            .as_device_ptr();

        let vector_device = cust::memory::DeviceBuffer::from_slice(vector).map_err(stringify)?;
        let output_device = gpu.get_f32_buffer(rows).map_err(stringify)?;

        let block_size = 256_u32;
        let grid_size = rows_u32.saturating_mul(32).div_ceil(block_size);
        let function = gpu.module.get_function(kernel_name).map_err(stringify)?;
        let stream = &gpu.stream;
        unsafe {
            cust::launch!(
                function<<<grid_size, block_size, 0, stream>>>(
                    matrix_ptr,
                    vector_device.as_device_ptr(),
                    output_device.as_device_ptr(),
                    rows_u32,
                    blocks_u32
                )
            )
            .map_err(stringify)?;
        }
        output_device.copy_to(output).map_err(stringify)?;
        gpu.return_f32_buffer(output_device);
        Ok(())
    })
    .map_err(GemvCudaError::Cuda)
}

#[cfg(feature = "cuda")]
pub fn gemv_iq1_s_direct_cuda(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvCudaError> {
    gemv_superblock_direct_cuda(
        GEMV_IQ1_S_KERNEL_NAME,
        50,
        256,
        quantized_matrix,
        rows,
        cols,
        vector,
        output,
    )
}

#[cfg(feature = "cuda")]
pub fn gemv_iq1_m_direct_cuda(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvCudaError> {
    gemv_superblock_direct_cuda(
        GEMV_IQ1_M_KERNEL_NAME,
        56,
        256,
        quantized_matrix,
        rows,
        cols,
        vector,
        output,
    )
}

#[cfg(feature = "cuda")]
pub fn gemv_nvfp4_direct_cuda(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvCudaError> {
    gemv_superblock_direct_cuda(
        GEMV_NVFP4_KERNEL_NAME,
        36,
        64,
        quantized_matrix,
        rows,
        cols,
        vector,
        output,
    )
}

pub fn validate_q8_0_gemv_dims(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &[f32],
) -> Result<(), GemvCudaError> {
    if !cols.is_multiple_of(QK8_0) {
        return Err(GemvCudaError::InvalidVectorLength {
            expected: cols.div_ceil(QK8_0) * QK8_0,
            actual: cols,
        });
    }

    let blocks_per_row = cols / QK8_0;
    let expected_matrix_len = rows
        .saturating_mul(blocks_per_row)
        .saturating_mul(BLOCK_Q8_0_SIZE);
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvCudaError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
        });
    }
    if vector.len() != cols {
        return Err(GemvCudaError::InvalidVectorLength {
            expected: cols,
            actual: vector.len(),
        });
    }
    if output.len() != rows {
        return Err(GemvCudaError::InvalidOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }

    Ok(())
}

#[cfg(feature = "cuda")]
pub fn gemv_quantized_cuda(
    quantization: GgufQuantizationType,
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvCudaError> {
    // Map the quantization type to its GPU dequant kernel + block geometry.
    // Types without a GPU kernel are reported so the caller can fall back to the
    // CPU quantized path.
    let (dequant_kernel, block_bytes, vals_per_block) = dequant_kernel_for(quantization)
        .ok_or(GemvCudaError::UnsupportedQuantizationType { quantization })?;

    // Validate the quantized matrix / vector / output geometry.
    if quantized_matrix.len() % block_bytes != 0 {
        return Err(GemvCudaError::InvalidMatrixLength {
            expected: quantized_matrix.len().next_multiple_of(block_bytes),
            actual: quantized_matrix.len(),
        });
    }
    let n_blocks = quantized_matrix.len() / block_bytes;
    let expected_elems = rows.saturating_mul(cols);
    if n_blocks.saturating_mul(vals_per_block) != expected_elems {
        return Err(GemvCudaError::InvalidMatrixLength {
            expected: expected_elems,
            actual: n_blocks * vals_per_block,
        });
    }
    if vector.len() != cols {
        return Err(GemvCudaError::InvalidVectorLength {
            expected: cols,
            actual: vector.len(),
        });
    }
    if output.len() != rows {
        return Err(GemvCudaError::InvalidOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }

    let rows_i32 = i32::try_from(rows).map_err(|_| GemvCudaError::InvalidOutputLength {
        expected: i32::MAX as usize,
        actual: rows,
    })?;
    let cols_i32 = i32::try_from(cols).map_err(|_| GemvCudaError::InvalidVectorLength {
        expected: i32::MAX as usize,
        actual: cols,
    })?;
    let n_blocks_u32 = u32::try_from(n_blocks).map_err(|_| GemvCudaError::InvalidMatrixLength {
        expected: u32::MAX as usize,
        actual: n_blocks,
    })?;

    with_gpu(|gpu| {
        // First use: upload the raw quantized weight, dequantize it on the GPU
        // to resident f16 (stored as u16 bits), and cache it. Every later token
        // reuses the resident half-precision weight — no re-upload, no CPU work.
        let key = bytes_cache_key(quantized_matrix);
        if !gpu.resident_f16.contains_key(&key) {
            let weight_bytes = expected_elems * std::mem::size_of::<u16>();
            gpu.ensure_vram_headroom(weight_bytes);

            let raw =
                cust::memory::DeviceBuffer::from_slice(quantized_matrix).map_err(stringify)?;
            let weight =
                cust::memory::DeviceBuffer::<u16>::zeroed(expected_elems).map_err(stringify)?;

            let block_size = 256_u32;
            let grid_size = n_blocks_u32.div_ceil(block_size);
            let function = gpu.module.get_function(dequant_kernel).map_err(stringify)?;
            let stream = &gpu.stream;
            // SAFETY: device buffers are valid; nblocks bounds the kernel.
            unsafe {
                cust::launch!(
                    function<<<grid_size, block_size, 0, stream>>>(
                        raw.as_device_ptr(),
                        weight.as_device_ptr(),
                        n_blocks_u32
                    )
                )
                .map_err(stringify)?;
            }
            stream.synchronize().map_err(stringify)?;
            gpu.resident_bytes += weight_bytes;
            gpu.orphan_f16_keys.push_back(key);
            gpu.resident_f16.insert(key, weight);
            gpu.enforce_budget();
        } else {
            gpu.touch_orphan_f16(key);
        }

        let matrix_ptr = gpu
            .resident_f16
            .get(&key)
            .ok_or_else(|| "quantized weight missing from resident cache after insert".to_string())?
            .as_device_ptr()
            .as_raw();

        // Upload vector (pooled buffer reused when size matches).
        let vector_device = cust::memory::DeviceBuffer::from_slice(vector).map_err(stringify)?;
        let output_device = gpu.get_f32_buffer(rows).map_err(stringify)?;

        // Use the custom f16 GEMV kernel (not cuBLAS Hgemm) because the kernel
        // accumulates dot-products in f32 precision before writing f32 output.
        // cuBLAS Hgemm accumulates in f16, which causes unacceptable numerical
        // drift for LLM inference.
        let rows_u32 = rows_i32 as u32;
        let cols_u32 = cols_i32 as u32;
        let block_size = 256_u32;
        let grid_size = rows_u32.saturating_mul(32).div_ceil(block_size);
        let function = gpu
            .module
            .get_function(GEMV_F16_KERNEL_NAME)
            .map_err(stringify)?;
        let stream = &gpu.stream;
        unsafe {
            cust::launch!(
                function<<<grid_size, block_size, 0, stream>>>(
                    matrix_ptr,
                    vector_device.as_device_ptr(),
                    output_device.as_device_ptr(),
                    rows_u32,
                    cols_u32
                )
            )
            .map_err(stringify)?;
        }

        output_device.copy_to(output).map_err(stringify)?;
        gpu.return_f32_buffer(output_device);
        Ok(())
    })
    .map_err(GemvCudaError::Cuda)
}
