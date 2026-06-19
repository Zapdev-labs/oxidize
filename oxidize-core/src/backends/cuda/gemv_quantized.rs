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
    let blocks_u32 = u32::try_from(blocks_per_row).map_err(|_| GemvCudaError::InvalidVectorLength {
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
    let blocks_u32 = u32::try_from(blocks_per_row).map_err(|_| GemvCudaError::InvalidVectorLength {
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
