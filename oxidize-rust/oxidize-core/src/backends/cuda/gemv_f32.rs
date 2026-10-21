use super::*;

#[cfg(feature = "cuda")]
pub fn gemv_f32_cuda(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvCudaError> {
    validate_gemv_dims(matrix, rows, cols, vector, output)?;

    let rows_i32 = i32::try_from(rows).map_err(|_| GemvCudaError::InvalidOutputLength {
        expected: i32::MAX as usize,
        actual: rows,
    })?;
    let cols_i32 = i32::try_from(cols).map_err(|_| GemvCudaError::InvalidVectorLength {
        expected: i32::MAX as usize,
        actual: cols,
    })?;

    with_gpu(|gpu| {
        // The matrix argument to a gemv is a model weight (output = W · x), so
        // it is kept resident in VRAM and uploaded only once; activations flow
        // through the small `vector`/`output` buffers.
        let key = f32_cache_key(matrix);
        if !gpu.resident_f32.contains_key(&key) {
            let buffer = cust::memory::DeviceBuffer::from_slice(matrix).map_err(stringify)?;
            gpu.resident_f32.insert(key, buffer);
        }
        let matrix_ptr = gpu.resident_f32.get(&key).unwrap().as_device_ptr().as_raw();

        // Upload vector (pooled buffer reused when size matches).
        let mut vector_device = gpu.get_f32_buffer(cols).map_err(stringify)?;
        vector_device.copy_from(vector).map_err(stringify)?;
        let output_device = gpu.get_f32_buffer(rows).map_err(stringify)?;

        let alpha = 1.0_f32;
        let beta = 0.0_f32;

        // Our data is row-major.  In cuBLAS (column-major) the transpose trick
        // means we pass trans=CUBLAS_OP_T, m=cols, n=rows, lda=cols so that
        // cuBLAS interprets the memory as the transpose of a cols×rows matrix,
        // which is exactly our rows×cols row-major matrix.
        let status = unsafe {
            cublas_sys::cublasSgemv_v2(
                gpu.cublas,
                cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                cols_i32,
                rows_i32,
                &alpha,
                matrix_ptr as *const f32,
                cols_i32,
                vector_device.as_device_ptr().as_raw() as *const f32,
                1,
                &beta,
                output_device.as_device_ptr().as_raw() as *mut f32,
                1,
            )
        };
        if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(format!("cublasSgemv_v2 failed with status {status:?}"));
        }

        output_device.copy_to(output).map_err(stringify)?;
        gpu.return_f32_buffer(vector_device);
        gpu.return_f32_buffer(output_device);
        Ok(())
    })
    .map_err(GemvCudaError::Cuda)
}

pub fn validate_gemv_dims(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &[f32],
) -> Result<(), GemvCudaError> {
    let expected_matrix_len = rows.saturating_mul(cols);
    if matrix.len() != expected_matrix_len {
        return Err(GemvCudaError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: matrix.len(),
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

/// Transposed GEMV: `output = matrix^T * vector` where `matrix` is
/// `rows × cols` row-major.  In cuBLAS column-major terms this is a plain
/// `y = A * x` with A = cols × rows, so no transpose flag is needed.
#[cfg(feature = "cuda")]
pub fn gemv_f32_transposed_cuda(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvCudaError> {
    let expected_matrix_len = rows.saturating_mul(cols);
    if matrix.len() != expected_matrix_len {
        return Err(GemvCudaError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: matrix.len(),
        });
    }
    if vector.len() != rows {
        return Err(GemvCudaError::InvalidVectorLength {
            expected: rows,
            actual: vector.len(),
        });
    }
    if output.len() != cols {
        return Err(GemvCudaError::InvalidOutputLength {
            expected: cols,
            actual: output.len(),
        });
    }

    let cols_i32 = i32::try_from(cols).map_err(|_| GemvCudaError::InvalidOutputLength {
        expected: i32::MAX as usize,
        actual: cols,
    })?;
    let rows_i32 = i32::try_from(rows).map_err(|_| GemvCudaError::InvalidVectorLength {
        expected: i32::MAX as usize,
        actual: rows,
    })?;

    with_gpu(|gpu| {
        let key = f32_cache_key(matrix);
        if !gpu.resident_f32.contains_key(&key) {
            let buffer = cust::memory::DeviceBuffer::from_slice(matrix).map_err(stringify)?;
            gpu.resident_f32.insert(key, buffer);
        }
        let matrix_ptr = gpu.resident_f32.get(&key).unwrap().as_device_ptr().as_raw();

        let vector_device = cust::memory::DeviceBuffer::from_slice(vector).map_err(stringify)?;
        let output_device = gpu.get_f32_buffer(cols).map_err(stringify)?;

        let alpha = 1.0_f32;
        let beta = 0.0_f32;

        let status = unsafe {
            cublas_sys::cublasSgemv_v2(
                gpu.cublas,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cols_i32,
                rows_i32,
                &alpha,
                matrix_ptr as *const f32,
                cols_i32,
                vector_device.as_device_ptr().as_raw() as *const f32,
                1,
                &beta,
                output_device.as_device_ptr().as_raw() as *mut f32,
                1,
            )
        };
        if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(format!(
                "cublasSgemv_v2 (transposed) failed with status {status:?}"
            ));
        }

        output_device.copy_to(output).map_err(stringify)?;
        gpu.return_f32_buffer(output_device);
        Ok(())
    })
    .map_err(GemvCudaError::Cuda)
}
