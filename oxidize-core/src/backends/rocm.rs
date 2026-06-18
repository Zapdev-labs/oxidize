//! AMD ROCm / HIP GPU backend.
//!
//! Compiles the same `kernels/gemv_f32.cu` sources with `hipcc` at build time and
//! loads the resulting code object at runtime. Mirrors the CUDA direct-GEMV paths
//! for Q8_0, Q4_0, Q4_K, IQ1_S, IQ1_M (TQ1), and NVFP4.

use crate::gguf::GgufQuantizationType;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RocmBuildInfo {
    pub detected_at_build: bool,
    pub rocm_path: Option<&'static str>,
}

pub fn rocm_build_info() -> RocmBuildInfo {
    RocmBuildInfo {
        detected_at_build: cfg!(rocm_available),
        rocm_path: option_env!("OXIDIZE_ROCM_PATH"),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GemvRocmError {
    InvalidMatrixLength { expected: usize, actual: usize },
    InvalidVectorLength { expected: usize, actual: usize },
    InvalidOutputLength { expected: usize, actual: usize },
    UnsupportedQuantizationType { quantization: GgufQuantizationType },
    Hip(String),
}

#[cfg(all(feature = "rocm", rocm_available))]
mod hip_rt {
    use libloading::{Library, Symbol};
    use std::ffi::{CStr, CString};
    use std::os::raw::{c_char, c_int, c_uint, c_void};
    use std::path::PathBuf;
    use std::ptr;
    use std::sync::OnceLock;

    pub type hipError_t = c_int;
    pub type hipStream_t = *mut c_void;
    pub type hipModule_t = *mut c_void;
    pub type hipFunction_t = *mut c_void;
    pub type hipDeviceptr_t = *mut c_void;

    const HIP_SUCCESS: hipError_t = 0;
    const HIP_MEMCPY_HOST_TO_DEVICE: c_uint = 1;
    const HIP_MEMCPY_DEVICE_TO_HOST: c_uint = 2;

    struct HipApi {
        _lib: Library,
        hipInit: Symbol<'static, unsafe extern "C" fn(c_uint) -> hipError_t>,
        hipSetDevice: Symbol<'static, unsafe extern "C" fn(c_int) -> hipError_t>,
        hipStreamCreate: Symbol<'static, unsafe extern "C" fn(*mut hipStream_t) -> hipError_t>,
        hipStreamSynchronize: Symbol<'static, unsafe extern "C" fn(hipStream_t) -> hipError_t>,
        hipMalloc: Symbol<'static, unsafe extern "C" fn(*mut hipDeviceptr_t, usize) -> hipError_t>,
        hipFree: Symbol<'static, unsafe extern "C" fn(hipDeviceptr_t) -> hipError_t>,
        hipMemcpy: Symbol<
            'static,
            unsafe extern "C" fn(hipDeviceptr_t, *const c_void, usize, c_uint) -> hipError_t,
        >,
        hipModuleLoad: Symbol<'static, unsafe extern "C" fn(*mut hipModule_t, *const c_char) -> hipError_t>,
        hipModuleGetFunction:
            Symbol<'static, unsafe extern "C" fn(*mut hipFunction_t, hipModule_t, *const c_char) -> hipError_t>,
        hipModuleLaunchKernel: Symbol<
            'static,
            unsafe extern "C" fn(
                hipFunction_t,
                c_uint,
                c_uint,
                c_uint,
                c_uint,
                c_uint,
                c_uint,
                c_uint,
                hipStream_t,
                *mut *mut c_void,
                *mut *mut c_void,
            ) -> hipError_t,
        >,
        hipModuleUnload: Symbol<'static, unsafe extern "C" fn(hipModule_t) -> hipError_t>,
    }

    static HIP: OnceLock<Result<HipApi, String>> = OnceLock::new();

    fn load() -> Result<&'static HipApi, String> {
        HIP.get_or_init(|| {
            let paths = [
                "libamdhip64.so.6",
                "libamdhip64.so",
                "/opt/rocm/lib/libamdhip64.so.6",
            ];
            let mut last_err = String::from("libamdhip64 not found");
            for path in paths {
                match unsafe { Library::new(path) } {
                    Ok(lib) => {
                        // SAFETY: symbols match ROCm HIP ABI.
                        let api = unsafe {
                            HipApi {
                                hipInit: lib.get(b"hipInit\0")?,
                                hipSetDevice: lib.get(b"hipSetDevice\0")?,
                                hipStreamCreate: lib.get(b"hipStreamCreate\0")?,
                                hipStreamSynchronize: lib.get(b"hipStreamSynchronize\0")?,
                                hipMalloc: lib.get(b"hipMalloc\0")?,
                                hipFree: lib.get(b"hipFree\0")?,
                                hipMemcpy: lib.get(b"hipMemcpy\0")?,
                                hipModuleLoad: lib.get(b"hipModuleLoad\0")?,
                                hipModuleGetFunction: lib.get(b"hipModuleGetFunction\0")?,
                                hipModuleLaunchKernel: lib.get(b"hipModuleLaunchKernel\0")?,
                                hipModuleUnload: lib.get(b"hipModuleUnload\0")?,
                                _lib: lib,
                            }
                        };
                        return Ok(api);
                    }
                    Err(e) => last_err = e.to_string(),
                }
            }
            Err(last_err)
        })
        .as_ref()
        .map_err(|e| e.clone())
    }

    fn check(code: hipError_t, ctx: &str) -> Result<(), String> {
        if code == HIP_SUCCESS {
            Ok(())
        } else {
            Err(format!("{ctx}: hip error {code}"))
        }
    }

    pub struct DeviceBuffer {
        ptr: hipDeviceptr_t,
        len: usize,
    }

    impl DeviceBuffer {
        pub fn alloc(len: usize) -> Result<Self, String> {
            let api = load()?;
            let mut ptr: hipDeviceptr_t = ptr::null_mut();
            unsafe {
                check((api.hipMalloc)(&mut ptr, len), "hipMalloc")?;
            }
            Ok(Self { ptr, len })
        }

        pub fn from_slice(data: &[u8]) -> Result<Self, String> {
            let mut buf = Self::alloc(data.len())?;
            buf.copy_from_host(data)?;
            Ok(buf)
        }

        pub fn copy_from_host(&mut self, data: &[u8]) -> Result<(), String> {
            if data.len() != self.len {
                return Err("host slice length mismatch".to_string());
            }
            let api = load()?;
            unsafe {
                check(
                    (api.hipMemcpy)(
                        self.ptr,
                        data.as_ptr() as *const c_void,
                        self.len,
                        HIP_MEMCPY_HOST_TO_DEVICE,
                    ),
                    "hipMemcpy H2D",
                )
            }
        }

        pub fn copy_to_host(&self, out: &mut [u8]) -> Result<(), String> {
            if out.len() != self.len {
                return Err("host slice length mismatch".to_string());
            }
            let api = load()?;
            unsafe {
                check(
                    (api.hipMemcpy)(
                        out.as_mut_ptr() as hipDeviceptr_t,
                        self.ptr,
                        self.len,
                        HIP_MEMCPY_DEVICE_TO_HOST,
                    ),
                    "hipMemcpy D2H",
                )
            }
        }

        pub fn ptr(&self) -> hipDeviceptr_t {
            self.ptr
        }
    }

    impl Drop for DeviceBuffer {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                if let Ok(api) = load() {
                    unsafe {
                        let _ = (api.hipFree)(self.ptr);
                    }
                }
            }
        }
    }

    pub struct HipState {
        stream: hipStream_t,
        module: hipModule_t,
        resident_quant: std::collections::HashMap<(usize, usize, u64), DeviceBuffer>,
    }

    impl Drop for HipState {
        fn drop(&mut self) {
            if let Ok(api) = load() {
                unsafe {
                    if !self.module.is_null() {
                        let _ = (api.hipModuleUnload)(self.module);
                    }
                }
            }
        }
    }

    impl HipState {
        pub fn init(co_path: &str) -> Result<Self, String> {
            let api = load()?;
            unsafe {
                check((api.hipInit)(0), "hipInit")?;
                check((api.hipSetDevice)(0), "hipSetDevice")?;
            }
            let mut stream: hipStream_t = ptr::null_mut();
            unsafe {
                check((api.hipStreamCreate)(&mut stream), "hipStreamCreate")?;
            }
            let c_path = CString::new(co_path).map_err(|e| e.to_string())?;
            let mut module: hipModule_t = ptr::null_mut();
            unsafe {
                check(
                    (api.hipModuleLoad)(&mut module, c_path.as_ptr()),
                    "hipModuleLoad",
                )?;
            }
            Ok(Self {
                stream,
                module,
                resident_quant: std::collections::HashMap::new(),
            })
        }

        pub fn function(&self, name: &str) -> Result<hipFunction_t, String> {
            let api = load()?;
            let c_name = CString::new(name).map_err(|e| e.to_string())?;
            let mut func: hipFunction_t = ptr::null_mut();
            unsafe {
                check(
                    (api.hipModuleGetFunction)(&mut func, self.module, c_name.as_ptr()),
                    "hipModuleGetFunction",
                )?;
            }
            Ok(func)
        }

        pub fn launch(
            &self,
            func: hipFunction_t,
            grid: (u32, u32, u32),
            block: (u32, u32, u32),
            args: &mut [*mut c_void],
        ) -> Result<(), String> {
            let api = load()?;
            unsafe {
                check(
                    (api.hipModuleLaunchKernel)(
                        func,
                        grid.0,
                        grid.1,
                        grid.2,
                        block.0,
                        block.1,
                        block.2,
                        0,
                        self.stream,
                        args.as_mut_ptr(),
                        ptr::null_mut(),
                    ),
                    "hipModuleLaunchKernel",
                )?;
                check((api.hipStreamSynchronize)(self.stream), "hipStreamSynchronize")
            }
        }

        pub fn ensure_quant(&mut self, key: (usize, usize, u64), host: &[u8]) -> Result<(), String> {
            if !self.resident_quant.contains_key(&key) {
                self.resident_quant
                    .insert(key, DeviceBuffer::from_slice(host)?);
            }
            Ok(())
        }

        pub fn quant_ptr(&self, key: (usize, usize, u64)) -> Result<hipDeviceptr_t, String> {
            self.resident_quant
                .get(&key)
                .map(|b| b.ptr())
                .ok_or_else(|| "quant buffer missing".to_string())
        }
    }

    pub fn co_path() -> PathBuf {
        PathBuf::from(env!("OUT_DIR")).join("gemv_f32.co")
    }
}

#[cfg(all(feature = "rocm", rocm_available))]
type WeightCacheKey = (usize, usize, u64);

#[cfg(all(feature = "rocm", rocm_available))]
fn hash_bytes(data: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(all(feature = "rocm", rocm_available))]
fn bytes_cache_key(slice: &[u8]) -> WeightCacheKey {
    (slice.as_ptr() as usize, slice.len(), hash_bytes(slice))
}

#[cfg(all(feature = "rocm", rocm_available))]
thread_local! {
    static HIP_STATE: std::cell::RefCell<Option<hip_rt::HipState>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(all(feature = "rocm", rocm_available))]
fn with_hip<R>(f: impl FnOnce(&mut hip_rt::HipState) -> Result<R, String>) -> Result<R, String> {
    HIP_STATE.with(|cell| {
        let mut guard = cell.borrow_mut();
        if guard.is_none() {
            let path = hip_rt::co_path();
            let path_str = path.to_str().ok_or("invalid OUT_DIR path")?;
            *guard = Some(hip_rt::HipState::init(path_str)?);
        }
        f(guard.as_mut().expect("hip state initialized"))
    })
}

#[cfg(all(feature = "rocm", rocm_available))]
fn launch_gemv_rows_cols(
    gpu: &mut hip_rt::HipState,
    kernel: &str,
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), String> {
    use std::os::raw::c_void;

    let key = bytes_cache_key(quantized_matrix);
    gpu.ensure_quant(key, quantized_matrix)?;

    let vector_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            vector.as_ptr() as *const u8,
            vector.len() * std::mem::size_of::<f32>(),
        )
    };
    let vector_dev = hip_rt::DeviceBuffer::from_slice(vector_bytes)?;
    let mut output_dev = hip_rt::DeviceBuffer::alloc(rows * std::mem::size_of::<f32>())?;

    let mut rows_u32 = u32::try_from(rows).map_err(|_| "rows overflow")?;
    let mut cols_u32 = u32::try_from(cols).map_err(|_| "cols overflow")?;
    let mut matrix_ptr = gpu.quant_ptr(key)?;
    let mut vector_ptr = vector_dev.ptr();
    let mut output_ptr = output_dev.ptr();

    let mut args: [*mut c_void; 5] = [
        &mut matrix_ptr as *mut _ as *mut c_void,
        &mut vector_ptr as *mut _ as *mut c_void,
        &mut output_ptr as *mut _ as *mut c_void,
        &mut rows_u32 as *mut _ as *mut c_void,
        &mut cols_u32 as *mut _ as *mut c_void,
    ];

    let func = gpu.function(kernel)?;
    let grid = (rows_u32.saturating_mul(32).div_ceil(256), 1, 1);
    gpu.launch(func, grid, (256, 1, 1), &mut args)?;

    let out_bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(
            output.as_mut_ptr() as *mut u8,
            output.len() * std::mem::size_of::<f32>(),
        )
    };
    output_dev.copy_to_host(out_bytes)?;
    Ok(())
}

#[cfg(all(feature = "rocm", rocm_available))]
fn launch_gemv_superblock(
    gpu: &mut hip_rt::HipState,
    kernel: &str,
    block_bytes: usize,
    quantized_matrix: &[u8],
    rows: usize,
    blocks_per_row: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), String> {
    use std::os::raw::c_void;

    let key = bytes_cache_key(quantized_matrix);
    gpu.ensure_quant(key, quantized_matrix)?;

    let vector_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            vector.as_ptr() as *const u8,
            vector.len() * std::mem::size_of::<f32>(),
        )
    };
    let vector_dev = hip_rt::DeviceBuffer::from_slice(vector_bytes)?;
    let mut output_dev = hip_rt::DeviceBuffer::alloc(rows * std::mem::size_of::<f32>())?;

    let mut rows_u32 = u32::try_from(rows).map_err(|_| "rows overflow")?;
    let mut blocks_u32 = u32::try_from(blocks_per_row).map_err(|_| "blocks overflow")?;
    let mut matrix_ptr = gpu.quant_ptr(key)?;
    let mut vector_ptr = vector_dev.ptr();
    let mut output_ptr = output_dev.ptr();

    let mut args: [*mut c_void; 5] = [
        &mut matrix_ptr as *mut _ as *mut c_void,
        &mut vector_ptr as *mut _ as *mut c_void,
        &mut output_ptr as *mut _ as *mut c_void,
        &mut rows_u32 as *mut _ as *mut c_void,
        &mut blocks_u32 as *mut _ as *mut c_void,
    ];

    let func = gpu.function(kernel)?;
    let grid = (rows_u32.saturating_mul(32).div_ceil(256), 1, 1);
    gpu.launch(func, grid, (256, 1, 1), &mut args)?;

    let out_bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(
            output.as_mut_ptr() as *mut u8,
            output.len() * std::mem::size_of::<f32>(),
        )
    };
    output_dev.copy_to_host(out_bytes)?;
    let _ = block_bytes;
    Ok(())
}

#[cfg(feature = "rocm")]
pub fn gemv_f32_rocm(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvRocmError> {
    #[cfg(not(rocm_available))]
    {
        let _ = (matrix, rows, cols, vector, output);
        return Err(GemvRocmError::Hip("ROCm not available at build time".into()));
    }

    #[cfg(rocm_available)]
    {
        if matrix.len() != rows * cols || vector.len() != cols || output.len() != rows {
            return Err(GemvRocmError::InvalidOutputLength {
                expected: rows,
                actual: output.len(),
            });
        }
        // Dense f32 GEMV: dequant path not needed; use CPU fallback via HIP memcpy loop
        // is wasteful — run a simple host fallback for rare f32 weights on ROCm.
        for (row_idx, out) in output.iter_mut().enumerate().take(rows) {
            let row = &matrix[row_idx * cols..(row_idx + 1) * cols];
            *out = row.iter().zip(vector.iter()).map(|(w, v)| w * v).sum();
        }
        Ok(())
    }
}

#[cfg(feature = "rocm")]
pub fn gemv_quantized_rocm(
    quantization: GgufQuantizationType,
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvRocmError> {
    #[cfg(not(rocm_available))]
    {
        let _ = (quantization, quantized_matrix, rows, cols, vector, output);
        return Err(GemvRocmError::Hip("ROCm not available at build time".into()));
    }

    #[cfg(rocm_available)]
    {
        use crate::compute::quantization::{BLOCK_Q8_K_BYTES, QK_K};
        use crate::tensor::quantize_vector_q8_k_into;

        let map_err = |e: String| GemvRocmError::Hip(e);

        match quantization {
            GgufQuantizationType::Q8_0 => with_hip(|gpu| {
                launch_gemv_rows_cols(
                    gpu,
                    "gemv_q8_0_kernel",
                    quantized_matrix,
                    rows,
                    cols,
                    vector,
                    output,
                )
            })
            .map_err(map_err),
            GgufQuantizationType::Q4_0 => with_hip(|gpu| {
                launch_gemv_rows_cols(
                    gpu,
                    "gemv_q4_0_kernel",
                    quantized_matrix,
                    rows,
                    cols,
                    vector,
                    output,
                )
            })
            .map_err(map_err),
            GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M
                if cols.is_multiple_of(QK_K) =>
            {
                let blocks_per_row = cols / QK_K;
                let mut q8k = vec![0_u8; blocks_per_row * BLOCK_Q8_K_BYTES];
                quantize_vector_q8_k_into(vector, blocks_per_row, &mut q8k);
                with_hip(|gpu| {
                    use std::os::raw::c_void;

                    let key = bytes_cache_key(quantized_matrix);
                    gpu.ensure_quant(key, quantized_matrix)?;
                    let q8k_dev = hip_rt::DeviceBuffer::from_slice(&q8k)?;
                    let mut output_dev =
                        hip_rt::DeviceBuffer::alloc(rows * std::mem::size_of::<f32>())?;
                    let mut rows_u32 = u32::try_from(rows).map_err(|_| "rows overflow".to_string())?;
                    let mut blocks_u32 =
                        u32::try_from(blocks_per_row).map_err(|_| "blocks overflow".to_string())?;
                    let mut matrix_ptr = gpu.quant_ptr(key)?;
                    let mut q8k_ptr = q8k_dev.ptr();
                    let mut output_ptr = output_dev.ptr();
                    let mut args: [*mut c_void; 5] = [
                        &mut matrix_ptr as *mut _ as *mut c_void,
                        &mut q8k_ptr as *mut _ as *mut c_void,
                        &mut output_ptr as *mut _ as *mut c_void,
                        &mut rows_u32 as *mut _ as *mut c_void,
                        &mut blocks_u32 as *mut _ as *mut c_void,
                    ];
                    let func = gpu.function("gemv_q4_k_kernel")?;
                    gpu.launch(
                        func,
                        (rows_u32.saturating_mul(32).div_ceil(256), 1, 1),
                        (256, 1, 1),
                        &mut args,
                    )?;
                    output_dev.copy_to_host(unsafe {
                        std::slice::from_raw_parts_mut(
                            output.as_mut_ptr() as *mut u8,
                            output.len() * 4,
                        )
                    })?;
                    Ok(())
                })
                .map_err(map_err)
            }
            GgufQuantizationType::IQ1_S if cols.is_multiple_of(QK_K) => with_hip(|gpu| {
                launch_gemv_superblock(
                    gpu,
                    "gemv_iq1_s_kernel",
                    50,
                    quantized_matrix,
                    rows,
                    cols / QK_K,
                    vector,
                    output,
                )
            })
            .map_err(map_err),
            GgufQuantizationType::IQ1_M if cols.is_multiple_of(QK_K) => with_hip(|gpu| {
                launch_gemv_superblock(
                    gpu,
                    "gemv_iq1_m_kernel",
                    56,
                    quantized_matrix,
                    rows,
                    cols / QK_K,
                    vector,
                    output,
                )
            })
            .map_err(map_err),
            GgufQuantizationType::NVFP4 if cols.is_multiple_of(64) => with_hip(|gpu| {
                launch_gemv_superblock(
                    gpu,
                    "gemv_nvfp4_kernel",
                    36,
                    quantized_matrix,
                    rows,
                    cols / 64,
                    vector,
                    output,
                )
            })
            .map_err(map_err),
            other => Err(GemvRocmError::UnsupportedQuantizationType {
                quantization: other,
            }),
        }
    }
}

#[cfg(not(feature = "rocm"))]
pub fn gemv_f32_rocm(
    _matrix: &[f32],
    _rows: usize,
    _cols: usize,
    _vector: &[f32],
    _output: &mut [f32],
) -> Result<(), GemvRocmError> {
    Err(GemvRocmError::Hip("rocm feature disabled".into()))
}

#[cfg(not(feature = "rocm"))]
pub fn gemv_quantized_rocm(
    quantization: GgufQuantizationType,
    _quantized_matrix: &[u8],
    _rows: usize,
    _cols: usize,
    _vector: &[f32],
    _output: &mut [f32],
) -> Result<(), GemvRocmError> {
    Err(GemvRocmError::UnsupportedQuantizationType { quantization })
}
