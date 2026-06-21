use super::*;

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
pub(super) struct LayerEntry {
    /// Pointer keys of f32 weights owned by this layer.
    pub(super) f32_keys: Vec<WeightCacheKey>,
    /// Pointer keys of f16 weights owned by this layer.
    pub(super) f16_keys: Vec<WeightCacheKey>,
    /// Approximate bytes consumed by this layer's device buffers.
    pub(super) bytes: usize,
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

/// GPU-resident activation buffers for a batched (`B`-row) decode step
/// (`OX_GPU_BATCHED`). Kept entirely separate from [`GpuActivationBuffer`] so
/// the single-token path stays byte-identical. All row-major `[b][feature]`.
///
/// `xq8k` is laid out column-major over the B activations exactly as the bN
/// GEMV kernel expects: activation column `j` starts at byte offset
/// `j * blocks_per_row * BLOCK_Q8_K_BYTES`.
#[cfg(feature = "cuda")]
pub struct GpuBatchedActivation {
    /// Residual hidden state `[B * hidden_size]`.
    pub hidden: cust::memory::DeviceBuffer<f32>,
    /// RMS-normed copy `[B * hidden_size]`.
    pub normed: cust::memory::DeviceBuffer<f32>,
    /// Q projection `[B * q_len]` (per-seq contiguous after transpose).
    pub q: cust::memory::DeviceBuffer<f32>,
    /// K projection `[B * kv_len]`.
    pub k: cust::memory::DeviceBuffer<f32>,
    /// V projection `[B * kv_len]`.
    pub v: cust::memory::DeviceBuffer<f32>,
    /// Per-seq attention output `[B * q_len]`.
    pub attn: cust::memory::DeviceBuffer<f32>,
    /// Wo projection / attention residual delta `[B * hidden_size]`.
    pub attn_proj: cust::memory::DeviceBuffer<f32>,
    /// FFN gate `[B * intermediate_size]`.
    pub ffn_gate: cust::memory::DeviceBuffer<f32>,
    /// FFN up `[B * intermediate_size]`.
    pub ffn_up: cust::memory::DeviceBuffer<f32>,
    /// SiLU(gate)*up fed into the down projection `[B * intermediate_size]`.
    pub ffn_down: cust::memory::DeviceBuffer<f32>,
    /// bN output scratch `[max(q_len,kv_len,intermediate_size,vocab_chunk) * B]`
    /// (row-major `[rows, B]`) before transpose to per-seq layout.
    pub bn_out: cust::memory::DeviceBuffer<f32>,
    /// Column-major Q8_K activations for projection inputs
    /// `[B * blocks_per_row_hidden * BLOCK_Q8_K_BYTES]`.
    pub xq8k: cust::memory::DeviceBuffer<u8>,
    /// Column-major Q8_K activations for the FFN-down input
    /// `[B * blocks_per_row_inter * BLOCK_Q8_K_BYTES]`.
    pub xq8k_ffn: cust::memory::DeviceBuffer<u8>,
    pub batch: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub q_len: usize,
    pub kv_len: usize,
    /// Row capacity of `bn_out` (== `bn_out.len() / batch`).
    pub bn_rows: usize,
}

#[cfg(feature = "cuda")]
pub(super) struct GpuState {
    // Held to keep the CUDA context current for this thread; never read.
    pub(super) _ctx: cust::context::Context,
    pub(super) module: cust::module::Module,
    pub(super) stream: cust::stream::Stream,
    pub(super) cublas: cublas_sys::cublasHandle_t,
    /// Quantized weights dequantized once on the GPU to resident f16 (stored as
    /// raw u16 bits), keyed by `(pointer, len, content_hash)`.
    pub(super) resident_f16:
        std::collections::HashMap<WeightCacheKey, cust::memory::DeviceBuffer<u16>>,
    /// Resident f32 weight matrices for the dense gemv path, same keying.
    pub(super) resident_f32:
        std::collections::HashMap<WeightCacheKey, cust::memory::DeviceBuffer<f32>>,
    /// Pool of reusable f32 device buffers keyed by length.
    pub(super) f32_pool: std::collections::HashMap<usize, Vec<cust::memory::DeviceBuffer<f32>>>,
    /// Layer-by-layer VRAM management.
    pub(super) layer_config: CudaLayerConfig,
    /// Which layers are currently resident and in what order (front = MRU).
    pub(super) layer_lru: std::collections::VecDeque<LayerId>,
    /// Mapping from layer id to the weight keys it owns.
    pub(super) layer_map: std::collections::HashMap<LayerId, LayerEntry>,
    /// Current bytes used by resident weights (excludes pools / scratch).
    pub(super) resident_bytes: usize,
    /// Keys resident in `resident_f16` that are NOT owned by any layer.
    /// These are lazily cached by `gemv_quantized_cuda` and must be
    /// subject to the same budget enforcement as layer-managed weights.
    pub(super) orphan_f16_keys: std::collections::VecDeque<WeightCacheKey>,
    /// Raw quantized weights for on-the-fly GEMV (Q8_0, Q4_0, Q4_K).
    pub(super) resident_quant:
        std::collections::HashMap<WeightCacheKey, cust::memory::DeviceBuffer<u8>>,
    pub(super) orphan_quant_keys: std::collections::VecDeque<WeightCacheKey>,
    /// Reusable Q8_K activation buffers keyed by byte length.
    pub(super) q8k_pool: std::collections::HashMap<usize, Vec<cust::memory::DeviceBuffer<u8>>>,
    /// Optional GPU-resident activation buffers (hidden state, normed, FFN
    /// gate/up/down_in).  `None` until `gpu_init_activation_buffers` is called.
    pub(super) activation: Option<GpuActivationBuffer>,
    // --- On-device attention (OX_GPU_ATTN) ---
    /// Per-layer device KV cache (raw F16 bits). Index = kv_layer_idx.
    /// Each buffer length = `kv_context * kv_len` (== context_size * token_size).
    pub(super) kv_k_cache: Vec<cust::memory::DeviceBuffer<u16>>,
    pub(super) kv_v_cache: Vec<cust::memory::DeviceBuffer<u16>>,
    /// Geometry of the resident KV cache; (0,0,0) until initialised.
    pub(super) kv_layers: usize,
    /// `token_size` = kv_heads*head_dim (per-token row stride within a layer).
    pub(super) kv_len: usize,
    /// `context_size` (row capacity per layer).
    pub(super) kv_context: usize,
    /// Number of tokens written so far, per layer (0 = empty).
    pub(super) kv_seq_len: Vec<usize>,
    // --- Batched device decode (OX_GPU_BATCHED), kept fully separate from the
    //     single-stream KV cache above so the single-token path is untouched. ---
    /// Per-layer batched KV cache (raw F16 bits). Each buffer length =
    /// `kv_batched_b * kv_context * kv_len`; sequence `s` owns the contiguous
    /// block `[s*kv_context*kv_len, (s+1)*kv_context*kv_len)`.
    pub(super) kv_k_batched: Vec<cust::memory::DeviceBuffer<u16>>,
    pub(super) kv_v_batched: Vec<cust::memory::DeviceBuffer<u16>>,
    /// Number of sequence regions allocated per layer (0 = uninitialised).
    pub(super) kv_batched_b: usize,
    /// Tokens written so far per `[layer][seq]`.
    pub(super) kv_batched_seq_len: Vec<Vec<usize>>,
    /// Optional batched activation buffers; `None` until first batched forward.
    pub(super) batched_activation: Option<GpuBatchedActivation>,
    /// Streaming-multiprocessor count of the active device (probed at init,
    /// default 132 = H100). Used by the split-K decode-attention heuristic to
    /// size the number of KV partitions so the grid saturates all SMs.
    pub(super) sm_count: u32,
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
    pub(super) fn get_f32_buffer(
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

    pub(super) fn return_f32_buffer(&mut self, buf: cust::memory::DeviceBuffer<f32>) {
        let len = buf.len();
        self.f32_pool.entry(len).or_default().push(buf);
    }

    /// Ensure the given layer is marked as most-recently-used, evicting LRU
    /// layers if we are over the configured budget.
    pub(super) fn touch_layer(&mut self, layer: LayerId) {
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

    pub(super) fn enforce_budget(&mut self) {
        self.enforce_budget_protecting(None);
    }

    /// Like [`Self::enforce_budget`], but never evicts `protect` (the orphan
    /// quant entry a caller is about to use this turn).
    pub(super) fn enforce_budget_protecting(&mut self, protect: Option<WeightCacheKey>) {
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
    pub(super) fn ensure_vram_headroom(&mut self, additional_bytes: usize) {
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

    pub(super) fn touch_orphan_f16(&mut self, key: WeightCacheKey) {
        if let Some(pos) = self.orphan_f16_keys.iter().position(|&k| k == key) {
            self.orphan_f16_keys.remove(pos);
        }
        self.orphan_f16_keys.push_back(key);
    }

    pub(super) fn touch_orphan_quant(&mut self, key: WeightCacheKey) {
        if let Some(pos) = self.orphan_quant_keys.iter().position(|&k| k == key) {
            self.orphan_quant_keys.remove(pos);
        }
        self.orphan_quant_keys.push_back(key);
    }

    pub(super) fn get_q8k_buffer(
        &mut self,
        len: usize,
    ) -> Result<cust::memory::DeviceBuffer<u8>, String> {
        if let Some(pool) = self.q8k_pool.get_mut(&len) {
            if let Some(buf) = pool.pop() {
                return Ok(buf);
            }
        }
        cust::memory::DeviceBuffer::<u8>::zeroed(len).map_err(stringify)
    }

    pub(super) fn return_q8k_buffer(&mut self, buf: cust::memory::DeviceBuffer<u8>) {
        let len = buf.len();
        self.q8k_pool.entry(len).or_default().push(buf);
    }

    /// Upload quantized weights once; reuse the device buffer on later tokens.
    pub(super) fn ensure_resident_quant(
        &mut self,
        key: WeightCacheKey,
        host: &[u8],
    ) -> Result<(), String> {
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

    pub(super) fn evict_layer_internal(&mut self, layer: LayerId) {
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
