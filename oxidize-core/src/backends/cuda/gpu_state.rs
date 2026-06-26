#[allow(unused_imports)]
use super::*;

/// Thread-local GPU state.
///
/// **Limitation:** Each thread that calls a CUDA kernel gets its own
/// `GpuState` (CUDA context, module, caches, etc.). In a multi-threaded
/// program this means weight caches are duplicated per thread, which can
/// multiply VRAM usage. For best efficiency run GPU work from a single
/// thread (e.g. one dedicated inference thread).
#[cfg(feature = "cuda")]
thread_local! {
    pub(super) static GPU_STATE: std::cell::RefCell<Option<GpuState>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(feature = "cuda")]
pub(super) fn gpu_init() -> Result<GpuState, String> {
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
    const CONSERVATIVE_UNKNOWN_SM_COUNT: u32 = 1;
    let sm_count = cust::device::Device::get_device(0)
        .and_then(|dev| dev.get_attribute(cust::device::DeviceAttribute::MultiprocessorCount))
        .map(|n| (n.max(1)) as u32)
        .unwrap_or(CONSERVATIVE_UNKNOWN_SM_COUNT);
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
        kv_k_cache: Vec::new(),
        kv_v_cache: Vec::new(),
        kv_layers: 0,
        kv_len: 0,
        kv_context: 0,
        kv_seq_len: Vec::new(),
        kv_k_batched: Vec::new(),
        kv_v_batched: Vec::new(),
        kv_batched_b: 0,
        kv_batched_seq_len: Vec::new(),
        batched_activation: None,
        sm_count,
        decode_d_state: None,
        decode_layer_graphs: Vec::new(),
        decode_graph_layers: 0,
    })
}

/// Run `f` with the thread-local GPU state, initializing it on first use.
#[cfg(feature = "cuda")]
pub(super) fn with_gpu<R>(f: impl FnOnce(&mut GpuState) -> Result<R, String>) -> Result<R, String> {
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
pub(super) fn stringify<E: std::fmt::Display>(error: E) -> String {
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
        // Drop the device-resident KV cache too, so a model reload starts fresh.
        gpu.kv_k_cache.clear();
        gpu.kv_v_cache.clear();
        gpu.kv_seq_len.clear();
        gpu.kv_layers = 0;
        gpu.kv_len = 0;
        gpu.kv_context = 0;
        // Drop the batched device state too (separate from the single-stream cache).
        gpu.kv_k_batched.clear();
        gpu.kv_v_batched.clear();
        gpu.kv_batched_b = 0;
        gpu.kv_batched_seq_len.clear();
        gpu.batched_activation = None;
        Ok(())
    })
}
