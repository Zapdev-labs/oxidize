use crate::gguf::GgufQuantizationType;

const QK8_0: usize = 32;
const BLOCK_Q8_0_SIZE: usize = 2 + QK8_0;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaBuildInfo {
    pub detected_at_build: bool,
    pub cuda_path: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryDevice {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryError {
    SizeMismatch {
        expected: usize,
        actual: usize,
    },
    #[cfg(feature = "cuda")]
    Cuda(String),
}

#[cfg(feature = "cuda")]
impl From<cust::error::CudaError> for MemoryError {
    fn from(error: cust::error::CudaError) -> Self {
        Self::Cuda(error.to_string())
    }
}

pub struct DeviceBuffer {
    device: MemoryDevice,
    len: usize,
    host_bytes: Vec<u8>,
    #[cfg(feature = "cuda")]
    cuda_bytes: Option<cust::memory::DeviceBuffer<u8>>,
}

impl DeviceBuffer {
    pub fn allocate(device: MemoryDevice, len: usize) -> Result<Self, MemoryError> {
        let host_bytes = vec![0_u8; len];
        #[cfg(feature = "cuda")]
        let cuda_bytes = match device {
            MemoryDevice::Cpu => None,
            MemoryDevice::Cuda => Some(cust::memory::DeviceBuffer::zeroed(len)?),
        };

        Ok(Self {
            device,
            len,
            host_bytes,
            #[cfg(feature = "cuda")]
            cuda_bytes,
        })
    }

    pub fn device(&self) -> MemoryDevice {
        self.device
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn copy_from_host(&mut self, host: &[u8]) -> Result<(), MemoryError> {
        if host.len() != self.len {
            return Err(MemoryError::SizeMismatch {
                expected: self.len,
                actual: host.len(),
            });
        }

        self.host_bytes.copy_from_slice(host);
        #[cfg(feature = "cuda")]
        if let Some(cuda_buffer) = self.cuda_bytes.as_mut() {
            cuda_buffer.copy_from(host)?;
        }

        Ok(())
    }

    pub fn copy_to_host(&self, host: &mut [u8]) -> Result<(), MemoryError> {
        if host.len() != self.len {
            return Err(MemoryError::SizeMismatch {
                expected: self.len,
                actual: host.len(),
            });
        }

        #[cfg(feature = "cuda")]
        if let Some(cuda_buffer) = self.cuda_bytes.as_ref() {
            cuda_buffer.copy_to(host)?;
            return Ok(());
        }

        host.copy_from_slice(&self.host_bytes);
        Ok(())
    }
}

pub fn cuda_build_info() -> CudaBuildInfo {
    CudaBuildInfo {
        detected_at_build: cfg!(cuda_available),
        cuda_path: option_env!("LLAMAS_CUDA_PATH"),
    }
}

#[cfg(feature = "cuda")]
pub fn initialize_cuda() -> Result<cust::context::Context, cust::error::CudaError> {
    cust::quick_init()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GemvCudaError {
    InvalidMatrixLength {
        expected: usize,
        actual: usize,
    },
    InvalidVectorLength {
        expected: usize,
        actual: usize,
    },
    InvalidOutputLength {
        expected: usize,
        actual: usize,
    },
    UnsupportedQuantizationType {
        quantization: GgufQuantizationType,
    },
    #[cfg(feature = "cuda")]
    Cuda(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GemmCudaError {
    InvalidLeftMatrixLength {
        expected: usize,
        actual: usize,
    },
    InvalidRightMatrixLength {
        expected: usize,
        actual: usize,
    },
    InvalidOutputLength {
        expected: usize,
        actual: usize,
    },
    #[cfg(feature = "cuda")]
    Cuda(String),
}

#[cfg(feature = "cuda")]
impl From<cust::error::CudaError> for GemvCudaError {
    fn from(error: cust::error::CudaError) -> Self {
        Self::Cuda(error.to_string())
    }
}

#[cfg(feature = "cuda")]
impl From<cust::error::CudaError> for GemmCudaError {
    fn from(error: cust::error::CudaError) -> Self {
        Self::Cuda(error.to_string())
    }
}

pub const GEMV_KERNEL_NAME: &str = "gemv_f32_kernel";
pub const GEMV_Q8_0_KERNEL_NAME: &str = "gemv_q8_0_f32_kernel";

#[cfg(feature = "cuda")]
const GEMV_F32_PTX: &str = include_str!("../kernels/gemv_f32.ptx");

#[cfg(feature = "cuda")]
pub fn gemv_f32_cuda(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvCudaError> {
    validate_gemv_dims(matrix, rows, cols, vector, output)?;

    let _context = initialize_cuda()?;
    let module = cust::module::Module::from_ptx(GEMV_F32_PTX, &[])?;
    let stream = cust::stream::Stream::new(cust::stream::StreamFlags::DEFAULT, None)?;

    let mut matrix_device = cust::memory::DeviceBuffer::from_slice(matrix)?;
    let mut vector_device = cust::memory::DeviceBuffer::from_slice(vector)?;
    let mut output_device = cust::memory::DeviceBuffer::zeroed(rows)?;

    let block_size = 128_u32;
    let grid_size = (rows as u32).div_ceil(block_size);
    let rows_u32 = u32::try_from(rows).map_err(|_| GemvCudaError::InvalidOutputLength {
        expected: u32::MAX as usize,
        actual: rows,
    })?;
    let cols_u32 = u32::try_from(cols).map_err(|_| GemvCudaError::InvalidVectorLength {
        expected: u32::MAX as usize,
        actual: cols,
    })?;

    let function = module.get_function(GEMV_KERNEL_NAME)?;
    // SAFETY: Kernel parameters are valid device buffers and scalar dimensions.
    unsafe {
        cust::launch!(
            function<<<grid_size, block_size, 0, stream>>>(
                matrix_device.as_device_ptr(),
                vector_device.as_device_ptr(),
                output_device.as_device_ptr(),
                rows_u32,
                cols_u32
            )
        )?;
    }
    stream.synchronize()?;
    output_device.copy_to(output)?;
    Ok(())
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
    if quantization != GgufQuantizationType::Q8_0 {
        return Err(GemvCudaError::UnsupportedQuantizationType { quantization });
    }
    validate_q8_0_gemv_dims(quantized_matrix, rows, cols, vector, output)?;

    let _context = initialize_cuda()?;
    let module = cust::module::Module::from_ptx(GEMV_F32_PTX, &[])?;
    let stream = cust::stream::Stream::new(cust::stream::StreamFlags::DEFAULT, None)?;

    let matrix_device = cust::memory::DeviceBuffer::from_slice(quantized_matrix)?;
    let vector_device = cust::memory::DeviceBuffer::from_slice(vector)?;
    let output_device = cust::memory::DeviceBuffer::zeroed(rows)?;

    let block_size = 128_u32;
    let grid_size = (rows as u32).div_ceil(block_size);
    let rows_u32 = u32::try_from(rows).map_err(|_| GemvCudaError::InvalidOutputLength {
        expected: u32::MAX as usize,
        actual: rows,
    })?;
    let cols_u32 = u32::try_from(cols).map_err(|_| GemvCudaError::InvalidVectorLength {
        expected: u32::MAX as usize,
        actual: cols,
    })?;

    let function = module.get_function(GEMV_Q8_0_KERNEL_NAME)?;
    // SAFETY: Kernel parameters are valid device buffers and scalar dimensions.
    unsafe {
        cust::launch!(
            function<<<grid_size, block_size, 0, stream>>>(
                matrix_device.as_device_ptr(),
                vector_device.as_device_ptr(),
                output_device.as_device_ptr(),
                rows_u32,
                cols_u32
            )
        )?;
    }
    stream.synchronize()?;
    output_device.copy_to(output)?;
    Ok(())
}

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
fn cublas_status_to_error(
    status: cublas_sys::cublasStatus_t,
    op: &str,
) -> Result<(), GemmCudaError> {
    if status == cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(GemmCudaError::Cuda(format!(
            "{op} failed with status {status:?}"
        )))
    }
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

    let _context = initialize_cuda()?;
    let left_device = cust::memory::DeviceBuffer::from_slice(left_matrix)?;
    let right_device = cust::memory::DeviceBuffer::from_slice(right_matrix)?;
    let output_device = cust::memory::DeviceBuffer::<f32>::zeroed(output.len())?;

    let mut handle: cublas_sys::cublasHandle_t = std::ptr::null_mut();
    // SAFETY: cuBLAS expects a valid out-pointer for handle creation.
    unsafe { cublas_status_to_error(cublas_sys::cublasCreate_v2(&mut handle), "cublasCreate_v2")? };

    let alpha = 1.0_f32;
    let beta = 0.0_f32;
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

    let lda = m;
    let ldb = k;
    let ldc = m;

    // SAFETY: device buffers are allocated and valid, dimensions and leading dimensions are consistent.
    let gemm_result = unsafe {
        cublas_status_to_error(
            cublas_sys::cublasSgemm_v2(
                handle,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                m,
                n,
                k,
                &alpha,
                right_device.as_device_ptr().as_raw() as *const f32,
                lda,
                left_device.as_device_ptr().as_raw() as *const f32,
                ldb,
                &beta,
                output_device.as_device_ptr().as_raw_mut() as *mut f32,
                ldc,
            ),
            "cublasSgemm_v2",
        )
    };

    // SAFETY: handle is either null or created by cublasCreate_v2.
    let destroy_result =
        unsafe { cublas_status_to_error(cublas_sys::cublasDestroy_v2(handle), "cublasDestroy_v2") };
    gemm_result?;
    destroy_result?;
    output_device.copy_to(output)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_cpu_buffer_with_requested_size() {
        let buffer = DeviceBuffer::allocate(MemoryDevice::Cpu, 32).expect("buffer allocates");
        assert_eq!(buffer.device(), MemoryDevice::Cpu);
        assert_eq!(buffer.len(), 32);
    }

    #[test]
    fn supports_host_to_device_and_back_for_cpu_buffer() {
        let mut buffer = DeviceBuffer::allocate(MemoryDevice::Cpu, 4).expect("buffer allocates");
        let input = [1_u8, 2, 3, 4];
        buffer
            .copy_from_host(&input)
            .expect("host to device copy succeeds");

        let mut output = [0_u8; 4];
        buffer
            .copy_to_host(&mut output)
            .expect("device to host copy succeeds");
        assert_eq!(output, input);
    }

    #[test]
    fn rejects_mismatched_transfer_lengths() {
        let mut buffer = DeviceBuffer::allocate(MemoryDevice::Cpu, 3).expect("buffer allocates");

        let h2d_error = buffer
            .copy_from_host(&[1_u8, 2])
            .expect_err("h2d mismatch should fail");
        assert_eq!(
            h2d_error,
            MemoryError::SizeMismatch {
                expected: 3,
                actual: 2
            }
        );

        let mut host = [0_u8; 2];
        let d2h_error = buffer
            .copy_to_host(&mut host)
            .expect_err("d2h mismatch should fail");
        assert_eq!(
            d2h_error,
            MemoryError::SizeMismatch {
                expected: 3,
                actual: 2
            }
        );
    }

    #[test]
    fn validates_gemv_cuda_dimensions() {
        let matrix = [1.0_f32, 2.0, 3.0, 4.0];
        let vector = [1.0_f32, 1.0];
        let output = [0.0_f32; 2];
        validate_gemv_dims(&matrix, 2, 2, &vector, &output).expect("dimensions should be valid");
    }

    #[test]
    fn rejects_gemv_cuda_dimension_mismatch() {
        let err = validate_gemv_dims(&[1.0_f32, 2.0, 3.0], 2, 2, &[1.0_f32, 1.0], &[0.0_f32; 2])
            .expect_err("matrix size mismatch should fail");
        assert!(matches!(err, GemvCudaError::InvalidMatrixLength { .. }));
    }

    #[test]
    fn validates_q8_0_gemv_cuda_dimensions() {
        let rows = 2;
        let cols = 32;
        let matrix = vec![0_u8; rows * BLOCK_Q8_0_SIZE];
        let vector = vec![1.0_f32; cols];
        let output = vec![0.0_f32; rows];
        validate_q8_0_gemv_dims(&matrix, rows, cols, &vector, &output)
            .expect("dimensions should be valid");
    }

    #[test]
    fn rejects_q8_0_gemv_cols_not_aligned() {
        let rows = 1;
        let cols = 31;
        let matrix = vec![0_u8; BLOCK_Q8_0_SIZE];
        let vector = vec![1.0_f32; cols];
        let output = vec![0.0_f32; rows];
        let err = validate_q8_0_gemv_dims(&matrix, rows, cols, &vector, &output)
            .expect_err("non-aligned columns should fail");
        assert!(matches!(err, GemvCudaError::InvalidVectorLength { .. }));
    }

    #[test]
    fn validates_gemm_cuda_dimensions() {
        let left = [1.0_f32, 2.0, 3.0, 4.0];
        let right = [1.0_f32, 2.0, 3.0, 4.0];
        let output = [0.0_f32; 4];
        validate_gemm_dims(&left, 2, 2, &right, 2, &output).expect("dimensions should be valid");
    }

    #[test]
    fn rejects_gemm_cuda_dimension_mismatch() {
        let err = validate_gemm_dims(
            &[1.0_f32, 2.0, 3.0],
            2,
            2,
            &[1.0_f32, 2.0, 3.0, 4.0],
            2,
            &[0.0_f32; 4],
        )
        .expect_err("left matrix size mismatch should fail");
        assert!(matches!(err, GemmCudaError::InvalidLeftMatrixLength { .. }));
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn gemv_cuda_kernel_name_matches_ptx_entry() {
        assert!(GEMV_F32_PTX.contains(".entry gemv_f32_kernel"));
        assert_eq!(GEMV_KERNEL_NAME, "gemv_f32_kernel");
    }
}
