use crate::gguf::GgufQuantizationType;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;

const QK8_0: usize = 32;
const BLOCK_Q8_0_SIZE: usize = 2 + QK8_0;
const QK_K: usize = 256;
const BLOCK_Q4_K_SIZE: usize = 144;
const BLOCK_Q8_K_BYTES: usize = 4 + QK_K + 32;

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
/// On-the-fly Q8_0 GEMV (no f16 materialization).
pub const GEMV_Q8_0_DIRECT_KERNEL_NAME: &str = "gemv_q8_0_kernel";
/// On-the-fly Q4_0 GEMV (no f16 materialization).
pub const GEMV_Q4_0_DIRECT_KERNEL_NAME: &str = "gemv_q4_0_kernel";
/// On-the-fly Q4_K × Q8_K GEMV (no f16 materialization; OXK GPU path).
pub const GEMV_Q4_K_DIRECT_KERNEL_NAME: &str = "gemv_q4_k_kernel";
/// Q4_K × F32 GEMV — GPU-native activation path (input stays on GPU, no Q8K quantization).
pub const GEMV_Q4K_F32IN_KERNEL_NAME: &str = "gemv_q4k_f32in_kernel";
/// Q6_K × F32 GEMV — GPU-native activation path for Q6_K weight matrices.
pub const GEMV_Q6K_F32IN_KERNEL_NAME: &str = "gemv_q6k_f32in_kernel";
pub const GEMV_IQ1_S_KERNEL_NAME: &str = "gemv_iq1_s_kernel";
pub const GEMV_IQ1_M_KERNEL_NAME: &str = "gemv_iq1_m_kernel";
pub const GEMV_NVFP4_KERNEL_NAME: &str = "gemv_nvfp4_kernel";
pub const RMS_NORM_KERNEL_NAME: &str = "rms_norm_f32_kernel";
pub const RESIDUAL_ADD_KERNEL_NAME: &str = "residual_add_f32_kernel";
pub const SILU_MUL_KERNEL_NAME: &str = "silu_mul_f32_kernel";
pub const CAST_F32_TO_F16_KERNEL_NAME: &str = "cast_f32_to_f16_kernel";
pub const CAST_F16_TO_F32_KERNEL_NAME: &str = "cast_f16_to_f32_kernel";

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

// PTX is generated from `kernels/gemv_f32.cu` by `build.rs` (nvcc) into OUT_DIR.
#[cfg(feature = "cuda")]
const GEMV_F32_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/gemv_f32.ptx"));

#[cfg(feature = "cuda")]
// Weight cache key: (ptr, len). Model weights are mmap'd and immutable for
// the lifetime of inference — the pointer is a stable unique identity. No
// content hashing needed; hashing MB-sized tensors on every GEMV call was
// the primary throughput bottleneck (400MB+ hashed per token on 1B models).
type WeightCacheKey = (usize, usize);

#[cfg(feature = "cuda")]
#[inline(always)]
fn f32_cache_key(slice: &[f32]) -> WeightCacheKey {
    (slice.as_ptr() as usize, slice.len())
}

#[cfg(feature = "cuda")]
#[inline(always)]
fn bytes_cache_key(slice: &[u8]) -> WeightCacheKey {
    (slice.as_ptr() as usize, slice.len())
}

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

/// Opaque handle used to tag a group of weight matrices as belonging to the
/// same model layer.  The inference engine calls [`preload_layer`] before a
/// forward pass and [`evict_layer`] when the layer is no longer needed.
pub type LayerId = usize;

/// Configuration for layer-by-layer VRAM management (AirLLM-style).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CudaLayerConfig {
    /// Maximum number of layers to keep resident in VRAM at once.
    /// 0 = unlimited (default, loads all layers).
    pub max_resident_layers: usize,
    /// Maximum VRAM bytes to use for weight caching.
    /// 0 = unlimited (default).
    pub max_vram_bytes: usize,
}

#[cfg(feature = "cuda")]
struct LayerEntry {
    /// Pointer keys of f32 weights owned by this layer.
    f32_keys: Vec<WeightCacheKey>,
    /// Pointer keys of f16 weights owned by this layer.
    f16_keys: Vec<WeightCacheKey>,
    /// Approximate bytes consumed by this layer's device buffers.
    bytes: usize,
}

/// GPU-resident activation buffers for a single decode step.
///
/// Allocated once (sized to the model's `hidden_size` / `intermediate_size`)
/// and reused across tokens to avoid per-token device allocations.
#[cfg(feature = "cuda")]
pub struct GpuActivationBuffer {
    /// Current residual hidden state `[hidden_size]`.
    pub hidden: cust::memory::DeviceBuffer<f32>,
    /// RMS-normed copy of `hidden` `[hidden_size]`.
    pub normed: cust::memory::DeviceBuffer<f32>,
    /// FFN gate-projection output `[intermediate_size]`.
    pub ffn_gate: cust::memory::DeviceBuffer<f32>,
    /// FFN up-projection output `[intermediate_size]`.
    pub ffn_up: cust::memory::DeviceBuffer<f32>,
    /// SiLU(gate) * up result fed into the down-projection `[intermediate_size]`.
    pub ffn_down_in: cust::memory::DeviceBuffer<f32>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
}

#[cfg(feature = "cuda")]
struct GpuState {
    // Held to keep the CUDA context current for this thread; never read.
    _ctx: cust::context::Context,
    module: cust::module::Module,
    stream: cust::stream::Stream,
    cublas: cublas_sys::cublasHandle_t,
    /// Quantized weights dequantized once on the GPU to resident f16 (stored as
    /// raw u16 bits), keyed by `(pointer, len, content_hash)`.
    resident_f16: std::collections::HashMap<WeightCacheKey, cust::memory::DeviceBuffer<u16>>,
    /// Resident f32 weight matrices for the dense gemv path, same keying.
    resident_f32: std::collections::HashMap<WeightCacheKey, cust::memory::DeviceBuffer<f32>>,
    /// Pool of reusable f32 device buffers keyed by length.
    f32_pool: std::collections::HashMap<usize, Vec<cust::memory::DeviceBuffer<f32>>>,
    /// Layer-by-layer VRAM management.
    layer_config: CudaLayerConfig,
    /// Which layers are currently resident and in what order (front = MRU).
    layer_lru: std::collections::VecDeque<LayerId>,
    /// Mapping from layer id to the weight keys it owns.
    layer_map: std::collections::HashMap<LayerId, LayerEntry>,
    /// Current bytes used by resident weights (excludes pools / scratch).
    resident_bytes: usize,
    /// Keys resident in `resident_f16` that are NOT owned by any layer.
    /// These are lazily cached by `gemv_quantized_cuda` and must be
    /// subject to the same budget enforcement as layer-managed weights.
    orphan_f16_keys: std::collections::VecDeque<WeightCacheKey>,
    /// Raw quantized weights for on-the-fly GEMV (Q8_0, Q4_0, Q4_K).
    resident_quant: std::collections::HashMap<WeightCacheKey, cust::memory::DeviceBuffer<u8>>,
    orphan_quant_keys: std::collections::VecDeque<WeightCacheKey>,
    /// Reusable Q8_K activation buffers keyed by byte length.
    q8k_pool: std::collections::HashMap<usize, Vec<cust::memory::DeviceBuffer<u8>>>,
    /// Optional GPU-resident activation buffers (hidden state, normed, FFN
    /// gate/up/down_in).  `None` until `gpu_init_activation_buffers` is called.
    activation: Option<GpuActivationBuffer>,
}

#[cfg(feature = "cuda")]
impl Drop for GpuState {
    fn drop(&mut self) {
        // The cuBLAS handle (from `cublasCreate_v2`) is a raw resource the other
        // RAII fields don't release. `Drop::drop` runs before the struct's
        // fields are dropped, so the CUDA context (`_ctx`) is still current.
        if !self.cublas.is_null() {
            unsafe {
                cublas_sys::cublasDestroy_v2(self.cublas);
            }
        }
    }
}

#[cfg(feature = "cuda")]
impl GpuState {
    fn get_f32_buffer(&mut self, len: usize) -> Result<cust::memory::DeviceBuffer<f32>, String> {
        if let Some(pool) = self.f32_pool.get_mut(&len) {
            if let Some(buf) = pool.pop() {
                return Ok(buf);
            }
        }
        cust::memory::DeviceBuffer::<f32>::zeroed(len).map_err(stringify)
    }

    fn return_f32_buffer(&mut self, buf: cust::memory::DeviceBuffer<f32>) {
        let len = buf.len();
        self.f32_pool.entry(len).or_default().push(buf);
    }

    /// Ensure the given layer is marked as most-recently-used, evicting LRU
    /// layers if we are over the configured budget.
    fn touch_layer(&mut self, layer: LayerId) {
        if self.layer_config.max_resident_layers == 0 && self.layer_config.max_vram_bytes == 0 {
            return; // unlimited
        }
        // Remove from current position if present.
        if let Some(pos) = self.layer_lru.iter().position(|&id| id == layer) {
            self.layer_lru.remove(pos);
        }
        self.layer_lru.push_back(layer);
        self.enforce_budget();
    }

    fn enforce_budget(&mut self) {
        self.enforce_budget_protecting(None);
    }

    /// Like [`Self::enforce_budget`], but never evicts `protect` (the orphan
    /// quant entry a caller is about to use this turn).
    fn enforce_budget_protecting(&mut self, protect: Option<WeightCacheKey>) {
        let max_layers = self.layer_config.max_resident_layers;
        let max_bytes = self.layer_config.max_vram_bytes;

        // First evict LRU layers.
        loop {
            let over_layer_limit = max_layers > 0 && self.layer_lru.len() > max_layers;
            let over_byte_limit = max_bytes > 0 && self.resident_bytes > max_bytes;
            if !over_layer_limit && !over_byte_limit {
                break;
            }
            if let Some(evict_id) = self.layer_lru.pop_front() {
                self.evict_layer_internal(evict_id);
            } else {
                break;
            }
        }

        // If still over byte budget, evict orphan (non-layer) f16 entries LRU-style.
        while max_bytes > 0 && self.resident_bytes > max_bytes {
            if let Some(key) = self.orphan_f16_keys.pop_front()
                && let Some(buf) = self.resident_f16.remove(&key)
            {
                self.resident_bytes -= buf.len() * std::mem::size_of::<u16>();
                drop(buf);
                continue;
            }
            if let Some(key) = self.orphan_quant_keys.pop_front() {
                if Some(key) == protect {
                    // Don't evict the entry the caller still needs; re-queue it
                    // at the front and stop (everything else is already gone).
                    self.orphan_quant_keys.push_front(key);
                    break;
                }
                if let Some(buf) = self.resident_quant.remove(&key) {
                    self.resident_bytes -= buf.len();
                    drop(buf);
                    continue;
                }
            }
            break;
        }
    }

    /// Evict LRU layers/orphans until `resident_bytes + additional_bytes` fits
    /// within the configured VRAM budget. Call before inserting new weights so
    /// the freshly inserted entry is not evicted in the same operation.
    fn ensure_vram_headroom(&mut self, additional_bytes: usize) {
        let max_bytes = self.layer_config.max_vram_bytes;
        if max_bytes == 0 {
            return;
        }
        while self.resident_bytes.saturating_add(additional_bytes) > max_bytes {
            if let Some(evict_id) = self.layer_lru.pop_front() {
                self.evict_layer_internal(evict_id);
                continue;
            }
            if let Some(key) = self.orphan_f16_keys.pop_front()
                && let Some(buf) = self.resident_f16.remove(&key)
            {
                self.resident_bytes -= buf.len() * std::mem::size_of::<u16>();
                drop(buf);
                continue;
            }
            if let Some(key) = self.orphan_quant_keys.pop_front()
                && let Some(buf) = self.resident_quant.remove(&key)
            {
                self.resident_bytes -= buf.len();
                drop(buf);
                continue;
            }
            break;
        }
    }

    fn touch_orphan_f16(&mut self, key: WeightCacheKey) {
        if let Some(pos) = self.orphan_f16_keys.iter().position(|&k| k == key) {
            self.orphan_f16_keys.remove(pos);
        }
        self.orphan_f16_keys.push_back(key);
    }

    fn touch_orphan_quant(&mut self, key: WeightCacheKey) {
        if let Some(pos) = self.orphan_quant_keys.iter().position(|&k| k == key) {
            self.orphan_quant_keys.remove(pos);
        }
        self.orphan_quant_keys.push_back(key);
    }

    fn get_q8k_buffer(&mut self, len: usize) -> Result<cust::memory::DeviceBuffer<u8>, String> {
        if let Some(pool) = self.q8k_pool.get_mut(&len) {
            if let Some(buf) = pool.pop() {
                return Ok(buf);
            }
        }
        cust::memory::DeviceBuffer::<u8>::zeroed(len).map_err(stringify)
    }

    fn return_q8k_buffer(&mut self, buf: cust::memory::DeviceBuffer<u8>) {
        let len = buf.len();
        self.q8k_pool.entry(len).or_default().push(buf);
    }

    /// Upload quantized weights once; reuse the device buffer on later tokens.
    fn ensure_resident_quant(&mut self, key: WeightCacheKey, host: &[u8]) -> Result<(), String> {
        if !self.resident_quant.contains_key(&key) {
            self.ensure_vram_headroom(host.len());
            let buf = cust::memory::DeviceBuffer::from_slice(host).map_err(stringify)?;
            self.resident_bytes += buf.len();
            self.resident_quant.insert(key, buf);
            self.orphan_quant_keys.push_back(key);
            // Protect the entry we just made resident: the caller is about to
            // `get(&key)` it, so it must not be evicted in this same budget
            // pass even if `ensure_vram_headroom` could not free enough room.
            self.enforce_budget_protecting(Some(key));
        } else {
            self.touch_orphan_quant(key);
        }
        Ok(())
    }

    fn evict_layer_internal(&mut self, layer: LayerId) {
        if let Some(entry) = self.layer_map.remove(&layer) {
            for key in &entry.f32_keys {
                // Only remove if no other resident layer still references this key.
                let other_refs = self.layer_map.values().any(|e| e.f32_keys.contains(key));
                if !other_refs {
                    if let Some(buf) = self.resident_f32.remove(key) {
                        self.resident_bytes -= buf.len() * std::mem::size_of::<f32>();
                        drop(buf);
                    }
                }
            }
            for key in &entry.f16_keys {
                let other_refs = self.layer_map.values().any(|e| e.f16_keys.contains(key));
                if !other_refs {
                    if let Some(buf) = self.resident_f16.remove(key) {
                        self.resident_bytes -= buf.len() * std::mem::size_of::<u16>();
                        drop(buf);
                    }
                }
            }
        }
    }
}

/// Thread-local GPU state.
///
/// **Limitation:** Each thread that calls a CUDA kernel gets its own
/// `GpuState` (CUDA context, module, caches, etc.). In a multi-threaded
/// program this means weight caches are duplicated per thread, which can
/// multiply VRAM usage. For best efficiency run GPU work from a single
/// thread (e.g. one dedicated inference thread).
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
        layer_config: CudaLayerConfig::default(),
        layer_lru: std::collections::VecDeque::new(),
        layer_map: std::collections::HashMap::new(),
        resident_bytes: 0,
        orphan_f16_keys: std::collections::VecDeque::new(),
        resident_quant: std::collections::HashMap::new(),
        orphan_quant_keys: std::collections::VecDeque::new(),
        q8k_pool: std::collections::HashMap::new(),
        activation: None,
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

// ---------------------------------------------------------------------------
// Layer-by-layer VRAM management public API
// ---------------------------------------------------------------------------

/// Configure the layer cache budget.  Must be called before any kernels run.
#[cfg(feature = "cuda")]
pub fn set_layer_config(config: CudaLayerConfig) -> Result<(), String> {
    with_gpu(|gpu| {
        gpu.layer_config = config;
        gpu.enforce_budget();
        Ok(())
    })
}

/// Mark a layer as needed and upload its f32 weights if they are not already
/// resident.  Evicts LRU layers when over budget.
///
/// * `f32_weights` – slice of `(matrix_data, rows, cols)` for each f32 weight
///   matrix belonging to this layer.
#[cfg(feature = "cuda")]
pub fn preload_layer(layer: LayerId, f32_weights: &[(&[f32], usize, usize)]) -> Result<(), String> {
    with_gpu(|gpu| {
        if gpu.layer_map.contains_key(&layer) {
            // Already resident — just bump to MRU.
            gpu.touch_layer(layer);
            return Ok(());
        }

        let mut entry = LayerEntry {
            f32_keys: Vec::new(),
            f16_keys: Vec::new(),
            bytes: 0,
        };

        for (matrix, _rows, _cols) in f32_weights {
            let key = f32_cache_key(matrix);
            if !gpu.resident_f32.contains_key(&key) {
                let buf = cust::memory::DeviceBuffer::from_slice(*matrix).map_err(stringify)?;
                entry.bytes += buf.len() * std::mem::size_of::<f32>();
                gpu.resident_f32.insert(key, buf);
            }
            entry.f32_keys.push(key);
        }

        gpu.resident_bytes += entry.bytes;
        gpu.layer_map.insert(layer, entry);
        gpu.touch_layer(layer);
        Ok(())
    })
}

/// Explicitly evict a layer from VRAM, freeing its device buffers.
#[cfg(feature = "cuda")]
pub fn evict_layer(layer: LayerId) -> Result<(), String> {
    with_gpu(|gpu| {
        gpu.layer_lru.retain(|&id| id != layer);
        gpu.evict_layer_internal(layer);
        Ok(())
    })
}

/// Return how many bytes of weight data are currently resident on the GPU.
#[cfg(feature = "cuda")]
pub fn resident_vram_bytes() -> usize {
    with_gpu(|gpu| Ok(gpu.resident_bytes)).unwrap_or(0)
}

/// Clear all resident weight caches (f16, f32, and layer entries).
///
/// Call this when loading a new model or when host weight buffers have been
/// mutated, to ensure stale GPU copies are not reused.
#[cfg(feature = "cuda")]
pub fn clear_resident_cache() -> Result<(), String> {
    with_gpu(|gpu| {
        gpu.resident_f16.clear();
        gpu.resident_f32.clear();
        gpu.layer_map.clear();
        gpu.layer_lru.clear();
        gpu.orphan_f16_keys.clear();
        gpu.resident_bytes = 0;
        Ok(())
    })
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

// ---------------------------------------------------------------------------
// On-the-fly quantized GEMV (no f16 materialization)
//
// These paths keep weights in their compressed form on the GPU and
// dequantize inside the GEMV kernel one block at a time.  This is the
// key to fitting 70B models on 4GB GPUs:
//   Q8_0: 1.06 bytes/param  -> 70B = ~74 GB in CPU RAM
//   Q4_0: 0.56 bytes/param  -> 70B = ~39 GB in CPU RAM
// Each layer (~2-4 GB compressed) is streamed to GPU, computed, then
// evicted — only activations stay resident.
// ---------------------------------------------------------------------------

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
            let buffer = cust::memory::DeviceBuffer::from_slice(left_matrix).map_err(stringify)?;
            gpu.resident_f32.insert(left_key, buffer);
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

// ---------------------------------------------------------------------------
// GPU-resident activation buffer management and per-layer kernel wrappers
// ---------------------------------------------------------------------------

/// Allocate (or reallocate) the GPU activation buffers for a given model size.
///
/// Safe to call multiple times; existing buffers are replaced only if the
/// dimensions change.
#[cfg(feature = "cuda")]
pub fn gpu_init_activation_buffers(
    hidden_size: usize,
    intermediate_size: usize,
) -> Result<(), String> {
    with_gpu(|gpu| {
        if let Some(ref ab) = gpu.activation {
            if ab.hidden_size == hidden_size && ab.intermediate_size == intermediate_size {
                return Ok(());
            }
        }
        let hidden =
            cust::memory::DeviceBuffer::<f32>::zeroed(hidden_size).map_err(stringify)?;
        let normed =
            cust::memory::DeviceBuffer::<f32>::zeroed(hidden_size).map_err(stringify)?;
        let ffn_gate =
            cust::memory::DeviceBuffer::<f32>::zeroed(intermediate_size).map_err(stringify)?;
        let ffn_up =
            cust::memory::DeviceBuffer::<f32>::zeroed(intermediate_size).map_err(stringify)?;
        let ffn_down_in =
            cust::memory::DeviceBuffer::<f32>::zeroed(intermediate_size).map_err(stringify)?;
        gpu.activation = Some(GpuActivationBuffer {
            hidden,
            normed,
            ffn_gate,
            ffn_up,
            ffn_down_in,
            hidden_size,
            intermediate_size,
        });
        Ok(())
    })
}

/// Upload a CPU f32 hidden-state slice into `activation.hidden` on the GPU.
///
/// The activation buffers must have been initialised via
/// [`gpu_init_activation_buffers`] before calling this function.
#[cfg(feature = "cuda")]
pub fn gpu_upload_hidden(hidden: &[f32]) -> Result<(), String> {
    with_gpu(|gpu| {
        let ab = gpu
            .activation
            .as_mut()
            .ok_or_else(|| "activation buffers not initialised".to_string())?;
        if hidden.len() != ab.hidden_size {
            return Err(format!(
                "gpu_upload_hidden: slice len {} != hidden_size {}",
                hidden.len(),
                ab.hidden_size
            ));
        }
        ab.hidden.copy_from(hidden).map_err(stringify)
    })
}

/// Run RMS-norm on the GPU: reads `activation.hidden`, writes `activation.normed`.
///
/// `weight` is a per-element scale vector of length `hidden_size`; it is
/// cached in `resident_f32` (mmap-stable pointer identity, same as ordinary
/// weight matrices).
#[cfg(feature = "cuda")]
pub fn gpu_rms_norm(weight: &[f32], eps: f32) -> Result<(), String> {
    with_gpu(|gpu| {
        let ab = gpu
            .activation
            .as_ref()
            .ok_or_else(|| "activation buffers not initialised".to_string())?;
        let hidden_size = ab.hidden_size;
        if weight.len() != hidden_size {
            return Err(format!(
                "gpu_rms_norm: weight len {} != hidden_size {}",
                weight.len(),
                hidden_size
            ));
        }

        // Cache norm weight in resident_f32 (same pointer-stable identity pattern).
        let key = f32_cache_key(weight);
        if !gpu.resident_f32.contains_key(&key) {
            let buf = cust::memory::DeviceBuffer::from_slice(weight).map_err(stringify)?;
            gpu.resident_f32.insert(key, buf);
        }

        let ab = gpu
            .activation
            .as_ref()
            .ok_or_else(|| "activation buffers not initialised".to_string())?;
        let hidden_ptr = ab.hidden.as_device_ptr();
        let normed_ptr = ab.normed.as_device_ptr();
        let weight_ptr = gpu
            .resident_f32
            .get(&key)
            .unwrap()
            .as_device_ptr();

        // One warp-reduction block per token; blockDim capped at 512 for the
        // parallel-reduction within the block (must be power-of-two).
        let block_size = hidden_size.next_power_of_two().min(512) as u32;
        let grid_size = 1_u32;
        let n = hidden_size as u32;

        let function = gpu
            .module
            .get_function(RMS_NORM_KERNEL_NAME)
            .map_err(stringify)?;
        let stream = &gpu.stream;
        let shmem_bytes = block_size * 4; // rms_norm kernel uses block_size floats of dynamic shared mem
        unsafe {
            cust::launch!(
                function<<<grid_size, block_size, shmem_bytes, stream>>>(
                    hidden_ptr,
                    weight_ptr,
                    normed_ptr,
                    n,
                    eps
                )
            )
            .map_err(stringify)?;
        }
        Ok(())
    })
}

/// GPU residual add: `activation.hidden[i] += delta[i]` for all i.
///
/// `delta` must be a device buffer of exactly `hidden_size` f32 elements.
#[cfg(feature = "cuda")]
pub fn gpu_residual_add(
    delta: &cust::memory::DeviceBuffer<f32>,
) -> Result<(), String> {
    with_gpu(|gpu| {
        let ab = gpu
            .activation
            .as_ref()
            .ok_or_else(|| "activation buffers not initialised".to_string())?;
        let hidden_size = ab.hidden_size;
        if delta.len() != hidden_size {
            return Err(format!(
                "gpu_residual_add: delta len {} != hidden_size {}",
                delta.len(),
                hidden_size
            ));
        }

        let hidden_ptr = ab.hidden.as_device_ptr();
        let delta_ptr = delta.as_device_ptr();
        let n = hidden_size as u32;

        let block_size = 256_u32;
        let grid_size = n.div_ceil(block_size);

        let function = gpu
            .module
            .get_function(RESIDUAL_ADD_KERNEL_NAME)
            .map_err(stringify)?;
        let stream = &gpu.stream;
        unsafe {
            cust::launch!(
                function<<<grid_size, block_size, 0, stream>>>(
                    hidden_ptr,
                    delta_ptr,
                    n
                )
            )
            .map_err(stringify)?;
        }
        Ok(())
    })
}

/// GPU SiLU-mul: `activation.ffn_down_in[i] = silu(ffn_gate[i]) * ffn_up[i]`.
///
/// Reads `activation.ffn_gate` and `activation.ffn_up`, writes
/// `activation.ffn_down_in`.  `intermediate_size` must equal the value used
/// when the buffers were allocated.
#[cfg(feature = "cuda")]
pub fn gpu_silu_mul(intermediate_size: usize) -> Result<(), String> {
    with_gpu(|gpu| {
        let ab = gpu
            .activation
            .as_ref()
            .ok_or_else(|| "activation buffers not initialised".to_string())?;
        if intermediate_size != ab.intermediate_size {
            return Err(format!(
                "gpu_silu_mul: intermediate_size {} != buffer size {}",
                intermediate_size, ab.intermediate_size
            ));
        }

        let gate_ptr = ab.ffn_gate.as_device_ptr();
        let up_ptr = ab.ffn_up.as_device_ptr();
        let out_ptr = ab.ffn_down_in.as_device_ptr();
        let n = intermediate_size as u32;

        let block_size = 256_u32;
        let grid_size = n.div_ceil(block_size);

        let function = gpu
            .module
            .get_function(SILU_MUL_KERNEL_NAME)
            .map_err(stringify)?;
        let stream = &gpu.stream;
        unsafe {
            cust::launch!(
                function<<<grid_size, block_size, 0, stream>>>(
                    gate_ptr,
                    up_ptr,
                    out_ptr,
                    n
                )
            )
            .map_err(stringify)?;
        }
        Ok(())
    })
}

/// Synchronise the GPU stream so that all preceding asynchronous kernel launches
/// have completed before the caller reads any device buffers.
#[cfg(feature = "cuda")]
pub fn gpu_sync_stream() -> Result<(), String> {
    with_gpu(|gpu| gpu.stream.synchronize().map_err(stringify))
}

/// GPU-native attention block: RMS-norm the hidden state, run Q/K/V projections,
/// then download Q/K/V to CPU for the attention computation.
///
/// The three GEMV results stay on the GPU until the combined D2H sync at the
/// end, keeping the GPU busy and reducing the CPU↔GPU round-trips from 3 to 1
/// per attention block.
///
/// Only supports Q4_K_S / Q4_K_M quantised weight matrices (which covers the
/// vast majority of GGUF files in the wild).  Returns `Err` if called without
/// having first called [`gpu_init_activation_buffers`].
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn gpu_attn_rms_and_qkv_q4k(
    attn_norm: &[f32],
    eps: f32,
    wq: &[u8],
    q_len: usize,
    hidden_size: usize,
    wk: &[u8],
    kv_len: usize,
    wv: &[u8],
    q_out: &mut [f32],
    k_out: &mut [f32],
    v_out: &mut [f32],
) -> Result<(), String> {
    if !hidden_size.is_multiple_of(256) {
        return Err(format!("gpu_attn_rms_and_qkv: hidden_size {hidden_size} not multiple of 256"));
    }
    let blocks_per_row = hidden_size / 256;
    let bpr_u32 = blocks_per_row as u32;
    let q_u32 = q_len as u32;
    let kv_u32 = kv_len as u32;

    with_gpu(|gpu| {
        // --- RMS-norm: hidden → normed ---
        let norm_key = f32_cache_key(attn_norm);
        if !gpu.resident_f32.contains_key(&norm_key) {
            let buf = cust::memory::DeviceBuffer::from_slice(attn_norm).map_err(stringify)?;
            gpu.resident_f32.insert(norm_key, buf);
        }

        let ab = gpu.activation.as_ref()
            .ok_or_else(|| "activation buffers not initialised".to_string())?;
        let hidden_ptr = ab.hidden.as_device_ptr();
        let normed_ptr = ab.normed.as_device_ptr();
        let hidden_size_u32 = ab.hidden_size as u32;
        let weight_ptr = gpu.resident_f32[&norm_key].as_device_ptr();

        let block_dim = ab.hidden_size.next_power_of_two().min(512) as u32;
        let shmem_attn = block_dim * 4;
        let function_norm = gpu.module.get_function(RMS_NORM_KERNEL_NAME).map_err(stringify)?;
        let stream = &gpu.stream;
        unsafe {
            cust::launch!(function_norm<<<1, block_dim, shmem_attn, stream>>>(
                hidden_ptr, weight_ptr, normed_ptr, hidden_size_u32, eps
            )).map_err(stringify)?;
        }

        // Detect Q4K vs Q6K from block byte-size (Q4K=144, Q6K=210 bytes per 256-value block).
        let quant_kern_name = |w: &[u8], rows: usize| -> &'static str {
            let bsz = if rows > 0 && blocks_per_row > 0 { w.len() / (rows * blocks_per_row) } else { 144 };
            if bsz >= 200 { GEMV_Q6K_F32IN_KERNEL_NAME } else { GEMV_Q4K_F32IN_KERNEL_NAME }
        };
        let qname_q = quant_kern_name(wq, q_len);
        let qname_k = quant_kern_name(wk, kv_len);
        let qname_v = quant_kern_name(wv, kv_len);

        // --- Upload weight matrices (Q4K or Q6K) ---
        let q_key = bytes_cache_key(wq);
        gpu.ensure_resident_quant(q_key, wq)?;
        let k_key = bytes_cache_key(wk);
        gpu.ensure_resident_quant(k_key, wk)?;
        let v_key = bytes_cache_key(wv);
        gpu.ensure_resident_quant(v_key, wv)?;

        // --- Allocate temporary GPU output buffers ---
        let d_q = gpu.get_f32_buffer(q_len)?;
        let d_k = gpu.get_f32_buffer(kv_len)?;
        let d_v = gpu.get_f32_buffer(kv_len)?;

        let block_size = 256_u32;
        let stream = &gpu.stream;

        let ab = gpu.activation.as_ref()
            .ok_or_else(|| "activation buffers not initialised".to_string())?;
        let normed_ptr = ab.normed.as_device_ptr();

        let q_grid = q_u32.saturating_mul(32).div_ceil(block_size);
        let k_grid = kv_u32.saturating_mul(32).div_ceil(block_size);

        let wq_ptr = gpu.resident_quant[&q_key].as_device_ptr();
        let wk_ptr = gpu.resident_quant[&k_key].as_device_ptr();
        let wv_ptr = gpu.resident_quant[&v_key].as_device_ptr();

        let fn_q = gpu.module.get_function(qname_q).map_err(stringify)?;
        let fn_k = gpu.module.get_function(qname_k).map_err(stringify)?;
        let fn_v = gpu.module.get_function(qname_v).map_err(stringify)?;

        unsafe {
            cust::launch!(fn_q<<<q_grid, block_size, 0, stream>>>(
                wq_ptr, normed_ptr, d_q.as_device_ptr(), q_u32, bpr_u32
            )).map_err(stringify)?;
            cust::launch!(fn_k<<<k_grid, block_size, 0, stream>>>(
                wk_ptr, normed_ptr, d_k.as_device_ptr(), kv_u32, bpr_u32
            )).map_err(stringify)?;
            cust::launch!(fn_v<<<k_grid, block_size, 0, stream>>>(
                wv_ptr, normed_ptr, d_v.as_device_ptr(), kv_u32, bpr_u32
            )).map_err(stringify)?;
        }

        // --- Single combined D2H sync: download Q, K, V ---
        gpu.stream.synchronize().map_err(stringify)?;
        d_q.copy_to(q_out).map_err(stringify)?;
        d_k.copy_to(k_out).map_err(stringify)?;
        d_v.copy_to(v_out).map_err(stringify)?;

        gpu.return_f32_buffer(d_q);
        gpu.return_f32_buffer(d_k);
        gpu.return_f32_buffer(d_v);
        Ok(())
    })
}

/// GPU-native O-projection + attention residual add.
///
/// Uploads the CPU attention output to the GPU, runs the wo Q4K GEMV, then
/// does an in-place residual add on the GPU hidden state.  The hidden state
/// remains on the GPU after this call; no D2H copy is performed.
#[cfg(feature = "cuda")]
pub fn gpu_wo_residual_q4k(
    attn_out: &[f32],
    wo: &[u8],
    rows: usize,
    cols: usize,
) -> Result<(), String> {
    if !cols.is_multiple_of(256) {
        return Err(format!("gpu_wo_residual_q4k: cols {cols} not multiple of 256"));
    }
    let bpr = (cols / 256) as u32;
    let rows_u32 = rows as u32;

    with_gpu(|gpu| {
        // Upload attention output to a temporary GPU buffer.
        let mut d_attn = gpu.get_f32_buffer(attn_out.len())?;
        d_attn.copy_from(attn_out).map_err(stringify)?;

        let wo_key = bytes_cache_key(wo);
        gpu.ensure_resident_quant(wo_key, wo)?;

        // Reuse `activation.normed` as the wo-projection output buffer — it is
        // no longer needed after the attention QKV step.
        let ab = gpu.activation.as_ref()
            .ok_or_else(|| "activation buffers not initialised".to_string())?;
        let normed_ptr = ab.normed.as_device_ptr();
        let hidden_ptr = ab.hidden.as_device_ptr();
        let hidden_n = ab.hidden_size as u32;

        let block_size = 256_u32;
        let wo_grid = rows_u32.saturating_mul(32).div_ceil(block_size);
        let res_grid = hidden_n.div_ceil(block_size);
        let wo_ptr = gpu.resident_quant[&wo_key].as_device_ptr();

        let bpr_usize = (cols / 256) as usize;
        let wo_kern_name = if bpr_usize > 0 && rows > 0 && wo.len() / (rows * bpr_usize) >= 200 {
            GEMV_Q6K_F32IN_KERNEL_NAME
        } else {
            GEMV_Q4K_F32IN_KERNEL_NAME
        };
        let fn_wo = gpu.module.get_function(wo_kern_name).map_err(stringify)?;
        let fn_res = gpu.module.get_function(RESIDUAL_ADD_KERNEL_NAME).map_err(stringify)?;
        let stream = &gpu.stream;

        unsafe {
            cust::launch!(fn_wo<<<wo_grid, block_size, 0, stream>>>(
                wo_ptr, d_attn.as_device_ptr(), normed_ptr, rows_u32, bpr
            )).map_err(stringify)?;
            cust::launch!(fn_res<<<res_grid, block_size, 0, stream>>>(
                hidden_ptr, normed_ptr, hidden_n
            )).map_err(stringify)?;
        }

        gpu.return_f32_buffer(d_attn);
        Ok(())
    })
}

/// GPU-native FFN block: RMS-norm + gate + up + SiLU-mul + down + residual.
///
/// Runs the complete feed-forward network on the GPU without any CPU↔GPU data
/// movement.  Reads `activation.hidden`, writes back to `activation.hidden`.
/// Intermediate results (gate, up, silu output) stay in the activation buffers.
///
/// The hidden state is **not** downloaded; the caller owns the GPU-resident
/// hidden state across layers and only downloads it once (for logit sampling).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn gpu_ffn_q4k(
    ffn_norm: &[f32],
    eps: f32,
    gate_w: &[u8],
    gate_rows: usize,
    gate_cols: usize,
    up_w: &[u8],
    up_rows: usize,
    down_w: &[u8],
    down_rows: usize,
    down_cols: usize,
) -> Result<(), String> {
    if !gate_cols.is_multiple_of(256) || !down_cols.is_multiple_of(256) {
        return Err(format!(
            "gpu_ffn_q4k: gate_cols {gate_cols} or down_cols {down_cols} not multiple of 256"
        ));
    }
    let gate_bpr = (gate_cols / 256) as u32;
    let down_bpr = (down_cols / 256) as u32;
    let gate_u32 = gate_rows as u32;
    let up_u32 = up_rows as u32;
    let down_u32 = down_rows as u32;

    with_gpu(|gpu| {
        // --- FFN RMS-norm: hidden → normed ---
        let norm_key = f32_cache_key(ffn_norm);
        if !gpu.resident_f32.contains_key(&norm_key) {
            let buf = cust::memory::DeviceBuffer::from_slice(ffn_norm).map_err(stringify)?;
            gpu.resident_f32.insert(norm_key, buf);
        }
        let ab = gpu.activation.as_ref()
            .ok_or_else(|| "activation buffers not initialised".to_string())?;
        let hidden_ptr = ab.hidden.as_device_ptr();
        let normed_ptr = ab.normed.as_device_ptr();
        let hidden_n = ab.hidden_size as u32;
        let norm_weight_ptr = gpu.resident_f32[&norm_key].as_device_ptr();
        let block_dim = ab.hidden_size.next_power_of_two().min(512) as u32;

        let shmem_ffn = block_dim * 4;
        let fn_norm = gpu.module.get_function(RMS_NORM_KERNEL_NAME).map_err(stringify)?;
        let stream = &gpu.stream;
        unsafe {
            cust::launch!(fn_norm<<<1, block_dim, shmem_ffn, stream>>>(
                hidden_ptr, norm_weight_ptr, normed_ptr, hidden_n, eps
            )).map_err(stringify)?;
        }

        // --- Upload FFN weight matrices ---
        let gate_key = bytes_cache_key(gate_w);
        gpu.ensure_resident_quant(gate_key, gate_w)?;
        let up_key = bytes_cache_key(up_w);
        gpu.ensure_resident_quant(up_key, up_w)?;
        let down_key = bytes_cache_key(down_w);
        gpu.ensure_resident_quant(down_key, down_w)?;

        let ab = gpu.activation.as_ref()
            .ok_or_else(|| "activation buffers not initialised".to_string())?;
        let normed_ptr = ab.normed.as_device_ptr();
        let gate_buf_ptr = ab.ffn_gate.as_device_ptr();
        let up_buf_ptr = ab.ffn_up.as_device_ptr();
        let ffn_down_in_ptr = ab.ffn_down_in.as_device_ptr();
        let inter_n = ab.intermediate_size as u32;

        let block_size = 256_u32;
        let gate_grid = gate_u32.saturating_mul(32).div_ceil(block_size);
        let up_grid = up_u32.saturating_mul(32).div_ceil(block_size);
        let silu_grid = inter_n.div_ceil(block_size);
        let down_grid = down_u32.saturating_mul(32).div_ceil(block_size);
        let res_grid = hidden_n.div_ceil(block_size);

        let gate_ptr = gpu.resident_quant[&gate_key].as_device_ptr();
        let up_ptr = gpu.resident_quant[&up_key].as_device_ptr();
        let down_ptr = gpu.resident_quant[&down_key].as_device_ptr();

        // Auto-detect Q4K vs Q6K for each FFN weight by block byte-size.
        let gate_bpr_usize = gate_bpr as usize;
        let down_bpr_usize = down_bpr as usize;
        let gate_kern = if gate_bpr_usize > 0 && gate_rows > 0 && gate_w.len() / (gate_rows * gate_bpr_usize) >= 200 {
            GEMV_Q6K_F32IN_KERNEL_NAME
        } else { GEMV_Q4K_F32IN_KERNEL_NAME };
        let up_kern = if gate_bpr_usize > 0 && up_rows > 0 && up_w.len() / (up_rows * gate_bpr_usize) >= 200 {
            GEMV_Q6K_F32IN_KERNEL_NAME
        } else { GEMV_Q4K_F32IN_KERNEL_NAME };
        let down_kern = if down_bpr_usize > 0 && down_rows > 0 && down_w.len() / (down_rows * down_bpr_usize) >= 200 {
            GEMV_Q6K_F32IN_KERNEL_NAME
        } else { GEMV_Q4K_F32IN_KERNEL_NAME };
        let fn_gate = gpu.module.get_function(gate_kern).map_err(stringify)?;
        let fn_up   = gpu.module.get_function(up_kern).map_err(stringify)?;
        let fn_down = gpu.module.get_function(down_kern).map_err(stringify)?;
        let fn_silu = gpu.module.get_function(SILU_MUL_KERNEL_NAME).map_err(stringify)?;
        let fn_res  = gpu.module.get_function(RESIDUAL_ADD_KERNEL_NAME).map_err(stringify)?;
        let stream = &gpu.stream;

        // Reuse `normed` as the down-projection output buffer (safe: gate/up
        // GEMVs only READ normed; by the time down runs, normed is free).
        let down_out_ptr = normed_ptr;
        let hidden_ptr = ab.hidden.as_device_ptr();

        unsafe {
            // gate × normed → ffn_gate
            cust::launch!(fn_gate<<<gate_grid, block_size, 0, stream>>>(
                gate_ptr, normed_ptr, gate_buf_ptr, gate_u32, gate_bpr
            )).map_err(stringify)?;
            // up × normed → ffn_up
            cust::launch!(fn_up<<<up_grid, block_size, 0, stream>>>(
                up_ptr, normed_ptr, up_buf_ptr, up_u32, gate_bpr
            )).map_err(stringify)?;
            // silu(ffn_gate) * ffn_up → ffn_down_in
            cust::launch!(fn_silu<<<silu_grid, block_size, 0, stream>>>(
                gate_buf_ptr, up_buf_ptr, ffn_down_in_ptr, inter_n
            )).map_err(stringify)?;
            // down × ffn_down_in → normed (reused as temp)
            cust::launch!(fn_down<<<down_grid, block_size, 0, stream>>>(
                down_ptr, ffn_down_in_ptr, down_out_ptr, down_u32, down_bpr
            )).map_err(stringify)?;
            // hidden += normed (ffn delta)
            cust::launch!(fn_res<<<res_grid, block_size, 0, stream>>>(
                hidden_ptr, down_out_ptr, hidden_n
            )).map_err(stringify)?;
        }
        // No D2H — hidden remains GPU-resident.
        Ok(())
    })
}

/// Download the current `activation.hidden` back to a CPU slice.
///
/// Triggers a device-to-host DMA transfer; the stream is synchronised before
/// the copy returns so the caller receives a consistent snapshot.
#[cfg(feature = "cuda")]
pub fn gpu_download_hidden(out: &mut [f32]) -> Result<(), String> {
    with_gpu(|gpu| {
        // Flush any pending kernel work on the stream before reading back.
        gpu.stream.synchronize().map_err(stringify)?;
        let ab = gpu
            .activation
            .as_ref()
            .ok_or_else(|| "activation buffers not initialised".to_string())?;
        if out.len() != ab.hidden_size {
            return Err(format!(
                "gpu_download_hidden: out len {} != hidden_size {}",
                out.len(),
                ab.hidden_size
            ));
        }
        ab.hidden.copy_to(out).map_err(stringify)
    })
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
        assert!(GEMV_F32_PTX.contains(".entry gemv_q4_k_kernel"));
        assert_eq!(GEMV_KERNEL_NAME, "gemv_f32_kernel");
        assert_eq!(GEMV_Q4_K_DIRECT_KERNEL_NAME, "gemv_q4_k_kernel");
    }
}
