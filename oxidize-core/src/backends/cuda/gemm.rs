use super::*;

pub fn validate_gemm_dims(
    left_matrix: &[f32],
    rows: usize,
    shared_dim: usize,
    right_matrix: &[f32],
    cols: usize,
    output: &[f32],
) -> Result<(), GemmCudaError> {
    let expected_left_len = rows.saturating_mul(shared_dim);
    if left_matrix.len() != expected_left_len {
        return Err(GemmCudaError::InvalidLeftMatrixLength {
            expected: expected_left_len,
            actual: left_matrix.len(),
        });
    }

    let expected_right_len = shared_dim.saturating_mul(cols);
    if right_matrix.len() != expected_right_len {
        return Err(GemmCudaError::InvalidRightMatrixLength {
            expected: expected_right_len,
            actual: right_matrix.len(),
        });
    }

    let expected_output_len = rows.saturating_mul(cols);
    if output.len() != expected_output_len {
        return Err(GemmCudaError::InvalidOutputLength {
            expected: expected_output_len,
            actual: output.len(),
        });
    }

    Ok(())
}

#[cfg(feature = "cuda")]
pub fn gemm_f32_cuda(
    left_matrix: &[f32],
    rows: usize,
    shared_dim: usize,
    right_matrix: &[f32],
    cols: usize,
    output: &mut [f32],
) -> Result<(), GemmCudaError> {
    validate_gemm_dims(left_matrix, rows, shared_dim, right_matrix, cols, output)?;

    let m = i32::try_from(cols).map_err(|_| GemmCudaError::InvalidOutputLength {
        expected: i32::MAX as usize,
        actual: cols,
    })?;
    let n = i32::try_from(rows).map_err(|_| GemmCudaError::InvalidOutputLength {
        expected: i32::MAX as usize,
        actual: rows,
    })?;
    let k = i32::try_from(shared_dim).map_err(|_| GemmCudaError::InvalidOutputLength {
        expected: i32::MAX as usize,
        actual: shared_dim,
    })?;

    with_gpu(|gpu| {
        // Cache left matrix (model weights) in VRAM.
        let left_key = f32_cache_key(left_matrix);
        if !gpu.resident_f32.contains_key(&left_key) {
            let bytes = left_matrix.len() * std::mem::size_of::<f32>();
            gpu.ensure_vram_headroom(bytes);
            let buffer = cust::memory::DeviceBuffer::from_slice(left_matrix).map_err(stringify)?;
            gpu.resident_bytes += buffer.len();
            gpu.resident_f32.insert(left_key, buffer);
            gpu.enforce_budget_protecting(Some(left_key));
        }
        let left_ptr = gpu
            .resident_f32
            .get(&left_key)
            .unwrap()
            .as_device_ptr()
            .as_raw();

        // Right matrix is an activation (not a static weight), so we always
        // upload a fresh copy to avoid stale-cache bugs when the host buffer
        // is reused or mutated between calls.
        let right_device =
            cust::memory::DeviceBuffer::from_slice(right_matrix).map_err(stringify)?;
        let right_ptr = right_device.as_device_ptr().as_raw();

        let output_device =
            cust::memory::DeviceBuffer::<f32>::zeroed(output.len()).map_err(stringify)?;

        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        let lda = m;
        let ldb = k;
        let ldc = m;

        // SAFETY: device buffers are allocated and valid; dimensions and
        // leading dimensions are consistent; the cuBLAS handle is cached and
        // valid for the lifetime of this thread's GPU state.
        let status = unsafe {
            cublas_sys::cublasSgemm_v2(
                gpu.cublas,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                m,
                n,
                k,
                &alpha,
                right_ptr as *const f32,
                lda,
                left_ptr as *const f32,
                ldb,
                &beta,
                output_device.as_device_ptr().as_raw() as *mut f32,
                ldc,
            )
        };
        if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(format!("cublasSgemm_v2 failed with status {status:?}"));
        }
        output_device.copy_to(output).map_err(stringify)?;
        Ok(())
    })
    .map_err(GemmCudaError::Cuda)
}
