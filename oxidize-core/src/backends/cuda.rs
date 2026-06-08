use crate::gguf::GgufQuantizationType;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;

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
        cuda_path: option_env!("OXIDIZE_CUDA_PATH"),
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
pub const GEMV_F16_KERNEL_NAME: &str = "gemv_f16_kernel";

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
fn dequant_kernel_for(
    quantization: GgufQuantizationType,
) -> Option<(&'static str, usize, usize)> {
    match quantization {
        GgufQuantizationType::Q8_0 => Some(("dequant_q8_0_kernel", 34, 32)),
        GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M => {
            Some(("dequant_q4_k_kernel", 144, 256))
        }
        GgufQuantizationType::Q6_K => Some(("dequant_q6_k_kernel", 210, 256)),
        _ => None,
    }
}

// PTX is generated from `kernels/gemv_f32.cu` by `build.rs` (nvcc) into OUT_DIR.
#[cfg(feature = "cuda")]
const GEMV_F32_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/gemv_f32.ptx"));

// ---------------------------------------------------------------------------
// Persistent per-thread GPU state
//
// The previous implementation created a fresh CUDA context, JIT-compiled the
// PTX module, and created a new cuBLAS handle on *every* matmul. Across a
// transformer forward pass that is thousands of PTX JIT compilations per token
// — the dominant cost, far larger than the actual math. We now build all of
// these once and reuse them, and keep static (quantized) weight matrices
// resident in VRAM so they are uploaded a single time instead of per token.
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
struct GpuState {
    // Held to keep the CUDA context current for this thread; never read.
    _ctx: cust::context::Context,
    module: cust::module::Module,
    stream: cust::stream::Stream,
    cublas: cublas_sys::cublasHandle_t,
    /// Quantized weights dequantized once on the GPU to resident f16 (stored as
    /// raw u16 bits), keyed by the host slice's `(pointer, len)`. Quantized
    /// matrices are always static model weights with a stable backing
    /// allocation for the run, so pointer identity is a safe cache key.
    resident_f16: std::collections::HashMap<(usize, usize), cust::memory::DeviceBuffer<u16>>,
    /// Resident f32 weight matrices for the dense gemv path, same keying.
    resident_f32: std::collections::HashMap<(usize, usize), cust::memory::DeviceBuffer<f32>>,
    /// Pool of reusable f32 device buffers keyed by length.
    f32_pool: std::collections::HashMap<usize, Vec<cust::memory::DeviceBuffer<f32>>>,
    /// Pool of reusable u16 device buffers keyed by length.
    u16_pool: std::collections::HashMap<usize, Vec<cust::memory::DeviceBuffer<u16>>>,
}

#[cfg(feature = "cuda")]
impl GpuState {
    fn get_f32_buffer(
        &mut self,
        len: usize,
    ) -> Result<cust::memory::DeviceBuffer<f32>, String> {
        if let Some(pool) = self.f32_pool.get_mut(&len) {
            if let Some(buf) = pool.pop() {
                return Ok(buf);
            }
        }
        cust::memory::DeviceBuffer::<f32>::zeroed(len).map_err(stringify)
    }

    fn return_f32_buffer(&mut self,
        buf: cust::memory::DeviceBuffer<f32>,
    ) {
        let len = buf.len();
        self.f32_pool.entry(len).or_default().push(buf);
    }

    fn get_u16_buffer(
        &mut self,
        len: usize,
    ) -> Result<cust::memory::DeviceBuffer<u16>, String> {
        if let Some(pool) = self.u16_pool.get_mut(&len) {
            if let Some(buf) = pool.pop() {
                return Ok(buf);
            }
        }
        cust::memory::DeviceBuffer::<u16>::zeroed(len).map_err(stringify)
    }

    fn return_u16_buffer(&mut self,
        buf: cust::memory::DeviceBuffer<u16>,
    ) {
        let len = buf.len();
        self.u16_pool.entry(len).or_default().push(buf);
    }
}

#[cfg(feature = "cuda")]
thread_local! {
    static GPU_STATE: std::cell::RefCell<Option<GpuState>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(feature = "cuda")]
fn gpu_init() -> Result<GpuState, String> {
    let _ctx = cust::quick_init().map_err(|e| e.to_string())?;
    let module = cust::module::Module::from_ptx(GEMV_F32_PTX, &[]).map_err(|e| e.to_string())?;
    let stream = cust::stream::Stream::new(cust::stream::StreamFlags::DEFAULT, None)
        .map_err(|e| e.to_string())?;
    let mut cublas: cublas_sys::cublasHandle_t = std::ptr::null_mut();
    // SAFETY: cuBLAS expects a valid out-pointer for handle creation.
    let status = unsafe { cublas_sys::cublasCreate_v2(&mut cublas) };
    if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(format!("cublasCreate_v2 failed with status {status:?}"));
    }
    Ok(GpuState {
        _ctx,
        module,
        stream,
        cublas,
        resident_f16: std::collections::HashMap::new(),
        resident_f32: std::collections::HashMap::new(),
        f32_pool: std::collections::HashMap::new(),
        u16_pool: std::collections::HashMap::new(),
    })
}

/// Run `f` with the thread-local GPU state, initializing it on first use.
#[cfg(feature = "cuda")]
fn with_gpu<R>(f: impl FnOnce(&mut GpuState) -> Result<R, String>) -> Result<R, String> {
    GPU_STATE.with(|cell| {
        let mut guard = cell.borrow_mut();
        if guard.is_none() {
            *guard = Some(gpu_init()?);
        }
        f(guard.as_mut().expect("gpu state initialized"))
    })
}

/// Map any `Display` error (e.g. `cust::error::CudaError`) into a `String`,
/// the common error currency used inside [`with_gpu`] closures.
#[cfg(feature = "cuda")]
fn stringify<E: std::fmt::Display>(error: E) -> String {
    error.to_string()
}

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
        let key = (matrix.as_ptr() as usize, matrix.len());
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
    let (dequant_kernel, block_bytes, vals_per_block) =
        dequant_kernel_for(quantization)
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
        let key = (quantized_matrix.as_ptr() as usize, quantized_matrix.len());
        if !gpu.resident_f16.contains_key(&key) {
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
            gpu.resident_f16.insert(key, weight);
        }

        let matrix_device = gpu
            .resident_f16
            .get(&key)
            .expect("weight just dequantized into resident cache");

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
                    matrix_device.as_device_ptr(),
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
        let left_key = (left_matrix.as_ptr() as usize, left_matrix.len());
        if !gpu.resident_f32.contains_key(&left_key) {
            let buffer = cust::memory::DeviceBuffer::from_slice(left_matrix).map_err(stringify)?;
            gpu.resident_f32.insert(left_key, buffer);
        }
        let left_ptr = gpu.resident_f32.get(&left_key).unwrap().as_device_ptr().as_raw();

        // Cache right matrix (also often static / reused).
        let right_key = (right_matrix.as_ptr() as usize, right_matrix.len());
        if !gpu.resident_f32.contains_key(&right_key) {
            let buffer = cust::memory::DeviceBuffer::from_slice(right_matrix).map_err(stringify)?;
            gpu.resident_f32.insert(right_key, buffer);
        }
        let right_ptr = gpu.resident_f32.get(&right_key).unwrap().as_device_ptr().as_raw();

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
        let err = validate_gemv_dims(&[1.0_f32, 2.0, 3.0], 2, 2, &[1.0_f32, 1.0], &[0.0_f32; 2],
        )
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
        let err = validate_q8_0_gemv_dims(&matrix, rows, cols, &vector, &output
        )
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
