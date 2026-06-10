use crate::conversion::normalize_gguf_tensor_name;
use crate::flash_attention::flash_attention_decode_f32;
use crate::gguf::{GgufQuantizationType, MappedGgufFile};
use crate::inference::{
    InferenceConfig, MoeFfnWeights, WeightStorage, lookup_quantized_embedding,
    moe_ffn_forward_weights,
};
use crate::kv_cache::KvCache;
use crate::model::{Logits, Model, ModelError, Session, Token};
use crate::quantization::{dequantize_scalar, quantized_size};
use crate::tensor::{
    apply_rope_f32, apply_swiglu_f32, gemm_quantized_f32, gemv_f32, gemv_quantized_f32,
    rms_norm_f32,
};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct LayerWiseModel {
    config: InferenceConfig,
    mmap: Arc<MappedGgufFile>,
    layer_tensors: Vec<HashMap<String, GgufTensorRef>>,
    tok_embeddings: WeightStorage,
    tok_embeddings_cols: usize,
    norm_weight: Vec<f32>,
    output_weight: WeightStorage,
    kv_cache: KvCache,
    ssm_states: Vec<Vec<f32>>,
    ssm_conv_buffers: Vec<ConvHistoryRing>,
    /// Number of tokens applied to the recurrent (GDN) state so far.
    ssm_pos: usize,
    /// Snapshots of (position, ssm_states, conv rings) for speculative
    /// rollback: unlike the KV cache, recurrent state is not
    /// position-addressable, so rewinding requires restoring a checkpoint.
    /// Two entries are live per speculative round (the rollback target set at
    /// the pre-verify rewind, plus the forward_many entry position).
    ssm_checkpoints: Vec<(usize, Vec<Vec<f32>>, Vec<ConvHistoryRing>)>,
    cache: LayerCache,
}

#[derive(Debug, Clone, PartialEq)]
struct GgufTensorRef {
    qtype: GgufQuantizationType,
    offset: usize,
    size: usize,
    value_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct LayerCache {
    capacity: usize,
    entries: Vec<Option<LayerWeights>>,
    access_count: Vec<u64>,
    generation: u64,
}

enum AttentionCacheSlice<'a> {
    Borrowed(&'a [f32]),
    Owned(Vec<f32>),
}

impl<'a> AttentionCacheSlice<'a> {
    fn as_slice(&'a self) -> &'a [f32] {
        match self {
            Self::Borrowed(data) => data,
            Self::Owned(data) => data,
        }
    }
}

impl LayerCache {
    fn new(capacity: usize, layer_count: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: vec![None; layer_count],
            access_count: vec![0; layer_count],
            generation: 0,
        }
    }
    fn get(&mut self, layer_idx: usize) -> Option<LayerWeights> {
        self.generation += 1;
        self.access_count[layer_idx] = self.generation;
        self.entries[layer_idx].take()
    }
    fn put(&mut self, layer_idx: usize, weights: LayerWeights) {
        if self.entries[layer_idx].is_some() {
            self.entries[layer_idx] = Some(weights);
            return;
        }
        let occupied = self.entries.iter().filter(|e| e.is_some()).count();
        if occupied < self.capacity {
            self.entries[layer_idx] = Some(weights);
            return;
        }
        let mut min_gen = u64::MAX;
        let mut evict_idx = 0;
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.is_some() && self.access_count[i] < min_gen {
                min_gen = self.access_count[i];
                evict_idx = i;
            }
        }
        self.entries[evict_idx] = None;
        self.entries[layer_idx] = Some(weights);
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
struct LayerWeights {
    attn_norm: Vec<f32>,
    attn_q: WeightStorage,
    attn_q_bias: Vec<f32>,
    attn_k: WeightStorage,
    attn_k_bias: Vec<f32>,
    attn_v: WeightStorage,
    attn_v_bias: Vec<f32>,
    attn_output: WeightStorage,
    attn_output_bias: Vec<f32>,
    ffn_norm: Vec<f32>,
    post_attention_norm: Vec<f32>,
    ffn_gate: WeightStorage,
    ffn_up: WeightStorage,
    ffn_down: WeightStorage,
    ffn_down_bias: Vec<f32>,
    ffn_gate_exps: WeightStorage,
    ffn_up_exps: WeightStorage,
    ffn_down_exps: WeightStorage,
    ffn_gate_inp: WeightStorage,
    ffn_exp_probs_b: Vec<f32>,
    ffn_gate_shexp: WeightStorage,
    ffn_gate_inp_shexp: WeightStorage,
    ffn_up_shexp: WeightStorage,
    ffn_down_shexp: WeightStorage,
    attn_qkv: WeightStorage,
    attn_gate: WeightStorage,
    ssm_a: Vec<f32>,
    ssm_alpha: WeightStorage,
    ssm_beta: WeightStorage,
    ssm_conv1d: Vec<f32>,
    ssm_dt_bias: Vec<f32>,
    ssm_norm: Vec<f32>,
    ssm_out: WeightStorage,
    attn_q_norm: Vec<f32>,
    attn_k_norm: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
struct ConvHistoryRing {
    slots: Vec<f32>,
    dim: usize,
    capacity: usize,
    head: usize,
    len: usize,
}

impl ConvHistoryRing {
    fn checksum(&self) -> f64 {
        self.slots.iter().map(|v| *v as f64).sum::<f64>() + self.head as f64 * 1e-3 + self.len as f64 * 1e-6
    }

    fn new(capacity: usize, dim: usize) -> Self {
        Self {
            slots: vec![0.0_f32; capacity.saturating_mul(dim)],
            dim,
            capacity: capacity.max(1),
            head: 0,
            len: 0,
        }
    }

    fn push(&mut self, frame: &[f32]) {
        if self.dim == 0 || frame.len() != self.dim {
            return;
        }
        let start = self.head * self.dim;
        self.slots[start..start + self.dim].copy_from_slice(frame);
        self.head = (self.head + 1) % self.capacity;
        self.len = (self.len + 1).min(self.capacity);
    }

    fn past_frame(&self, steps_back: usize) -> Option<&[f32]> {
        if steps_back == 0 || steps_back > self.len {
            return None;
        }
        let idx = (self.head + self.capacity - steps_back) % self.capacity;
        let start = idx * self.dim;
        Some(&self.slots[start..start + self.dim])
    }
}

fn quant_block_info(qtype: GgufQuantizationType) -> (usize, usize) {
    match qtype {
        GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M => (256, 144),
        GgufQuantizationType::Q6_K => (256, 210),
        GgufQuantizationType::Q8_0 => (32, 34),
        _ => (1, 4),
    }
}

fn weight_is_empty(ws: &WeightStorage) -> bool {
    match ws {
        WeightStorage::F32(v) => v.is_empty(),
        WeightStorage::Quantized(_, v) => v.is_empty(),
        WeightStorage::MmapQuantized(_, _, _, size) => *size == 0,
    }
}

fn weight_output_dim(ws: &WeightStorage, input_dim: usize) -> usize {
    match ws {
        WeightStorage::F32(v) => v.len() / input_dim,
        WeightStorage::Quantized(qtype, v) => {
            let (block_width, block_size) = quant_block_info(*qtype);
            let bytes_per_row = (input_dim / block_width) * block_size;
            if bytes_per_row == 0 {
                return 0;
            }
            v.len() / bytes_per_row
        }
        WeightStorage::MmapQuantized(qtype, _, _, size) => {
            let (block_width, block_size) = quant_block_info(*qtype);
            let bytes_per_row = (input_dim / block_width) * block_size;
            if bytes_per_row == 0 {
                return 0;
            }
            size / bytes_per_row
        }
    }
}

fn weight_input_dim(ws: &WeightStorage, output_dim: usize) -> usize {
    if output_dim == 0 {
        return 0;
    }
    match ws {
        WeightStorage::F32(v) => v.len() / output_dim,
        WeightStorage::Quantized(qtype, v) => {
            let (block_width, block_size) = quant_block_info(*qtype);
            let bytes_per_row = v.len() / output_dim;
            if block_size == 0 {
                return 0;
            }
            (bytes_per_row / block_size) * block_width
        }
        WeightStorage::MmapQuantized(qtype, _, _, size) => {
            let (block_width, block_size) = quant_block_info(*qtype);
            let bytes_per_row = size / output_dim;
            if block_size == 0 {
                return 0;
            }
            (bytes_per_row / block_size) * block_width
        }
    }
}

fn gemv_weight(
    storage: &WeightStorage,
    rows: usize,
    cols: usize,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), String> {
    match storage {
        WeightStorage::F32(data) => {
            gemv_f32(data, rows, cols, input, output).map_err(|e| format!("{:?}", e))
        }
        WeightStorage::Quantized(qtype, data) => {
            gemv_quantized_f32(*qtype, data, rows, cols, input, output)
                .map_err(|e| format!("{:?}", e))
        }
        WeightStorage::MmapQuantized(qtype, mmap, offset, size) => {
            let data = &mmap[*offset..*offset + *size];
            gemv_quantized_f32(*qtype, data, rows, cols, input, output)
                .map_err(|e| format!("{:?}", e))
        }
    }
}

/// Batched [`gemv_weight`]: `inputs` is `batch` row-major vectors of `cols`,
/// `outputs` is `batch` vectors of `rows`. Quantized storage goes through the
/// fused GEMM kernels, which decode each weight block once and reuse it across
/// the whole batch — this is what amortizes weight reads during prefill and
/// speculative verification.
fn gemm_weight(
    storage: &WeightStorage,
    rows: usize,
    cols: usize,
    inputs: &[f32],
    outputs: &mut [f32],
    batch: usize,
) -> Result<(), String> {
    match storage {
        WeightStorage::F32(data) => {
            for t in 0..batch {
                gemv_f32(
                    data,
                    rows,
                    cols,
                    &inputs[t * cols..(t + 1) * cols],
                    &mut outputs[t * rows..(t + 1) * rows],
                )
                .map_err(|e| format!("{:?}", e))?;
            }
            Ok(())
        }
        WeightStorage::Quantized(qtype, data) => {
            gemm_quantized_f32(*qtype, data, rows, cols, inputs, outputs, batch)
                .map_err(|e| format!("{:?}", e))
        }
        WeightStorage::MmapQuantized(qtype, mmap, offset, size) => {
            let data = &mmap[*offset..*offset + *size];
            gemm_quantized_f32(*qtype, data, rows, cols, inputs, outputs, batch)
                .map_err(|e| format!("{:?}", e))
        }
    }
}

fn l2_normalize(v: &mut [f32]) {
    let mut sum = 0.0_f32;
    for x in v.iter() {
        sum += x * x;
    }
    let inv = 1.0_f32 / (sum + 1e-6_f32).sqrt();
    for x in v.iter_mut() {
        *x *= inv;
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0_f32 / (1.0_f32 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    if x > 20.0_f32 {
        x
    } else {
        (1.0_f32 + x.exp()).ln()
    }
}

fn gated_rms_norm(x: &mut [f32], weight: &[f32], gate: &[f32], eps: f32) {
    let n = x.len();
    if n == 0 {
        return;
    }
    let mut var = 0.0_f32;
    for val in x.iter() {
        var += val * val;
    }
    var /= n as f32;
    let inv = 1.0_f32 / (var + eps).sqrt();
    for i in 0..n {
        let w = weight.get(i).copied().unwrap_or(1.0_f32);
        let g = gate.get(i).copied().unwrap_or(0.0_f32);
        let silu = g * (1.0_f32 / (1.0_f32 + (-g).exp()));
        x[i] = x[i] * inv * w * silu;
    }
}

fn rms_norm_model(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    output: &mut [f32],
    cfg: &InferenceConfig,
) -> Result<(), ModelError> {
    if cfg.rms_norm_weight_plus_one {
        let hidden_dim = output.len();
        if hidden_dim == 0 || input.len() != hidden_dim || weight.len() != hidden_dim {
            return Err(ModelError::InferenceFailed(
                "rms_norm: dimension mismatch".to_owned(),
            ));
        }
        let sum_sq: f32 = input.iter().map(|v| v * v).sum();
        let inv_rms = 1.0_f32 / (sum_sq / hidden_dim as f32 + eps).sqrt();
        for ((value, w), out) in input.iter().zip(weight.iter()).zip(output.iter_mut()) {
            *out = value * inv_rms * (1.0_f32 + w);
        }
        Ok(())
    } else {
        rms_norm_f32(input, weight, eps, output)
            .map_err(|e| ModelError::InferenceFailed(format!("rms_norm: {:?}", e)))
    }
}

fn split_gated_query_proj(q_full: &[f32], head_dim: usize) -> Option<(Vec<f32>, Vec<f32>)> {
    if head_dim == 0 || !q_full.len().is_multiple_of(2 * head_dim) {
        return None;
    }
    let num_heads = q_full.len() / (2 * head_dim);
    let mut query = vec![0.0_f32; num_heads * head_dim];
    let mut gate = vec![0.0_f32; num_heads * head_dim];
    for head in 0..num_heads {
        let base = head * 2 * head_dim;
        query[head * head_dim..(head + 1) * head_dim]
            .copy_from_slice(&q_full[base..base + head_dim]);
        gate[head * head_dim..(head + 1) * head_dim]
            .copy_from_slice(&q_full[base + head_dim..base + 2 * head_dim]);
    }
    Some((query, gate))
}

fn debug_vec(label: &str, x: &[f32]) {
    if !std::env::var("OXIDIZE_DEBUG_LAYERS").is_ok() {
        return;
    }
    let nan_count = x.iter().filter(|v| v.is_nan()).count();
    let inf_count = x.iter().filter(|v| v.is_infinite()).count();
    let max_abs = x
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| v.abs())
        .fold(0.0_f32, f32::max);
    let large = x.iter().filter(|v| v.is_finite() && v.abs() > 1000.0).count();
    eprintln!("{label} nan={nan_count} inf={inf_count} max_abs={max_abs} gt1k={large}");
}


/// Per-layer hidden-state checksum tracing (OXIDIZE_TRACE_FWD=1) for
/// diffing the batched window path against the per-token path.
fn trace_fwd(path: &str, pos: usize, layer: usize, x: &[f32]) {
    if std::env::var_os("OXIDIZE_TRACE_FWD").is_some() {
        let sum: f64 = x.iter().map(|v| *v as f64).sum();
        eprintln!("TRACE {path} pos={pos} layer={layer} sum={sum:.9e}");
    }
}

fn debug_hidden(label: &str, pos: usize, x: &[f32]) {
    if pos == 0 {
        debug_vec(label, x);
    }
}


impl LayerWiseModel {
    fn trace_state(&self, label: &str, pos: usize) {
        if std::env::var_os("OXIDIZE_TRACE_FWD").is_some() {
            let s0: f64 = self.ssm_states.first().map(|s| s.iter().map(|v| *v as f64).sum()).unwrap_or(0.0);
            let r0: f64 = self
                .ssm_conv_buffers
                .first()
                .map(|b| b.checksum())
                .unwrap_or(0.0);
            eprintln!("STATE {label} pos={pos} ssm_pos={} s0={s0:.9e} r0={r0:.9e}", self.ssm_pos);
        }
    }
}

impl LayerWiseModel {
    /// Tokens processed per batched layer-major pass in `forward`/`forward_many`.
    /// Larger windows amortize weight reads further but grow activation scratch
    /// linearly; 16 keeps scratch in the tens of MB for typical models.
    const FORWARD_WINDOW: usize = 16;

    /// Record a recurrent-state checkpoint at `pos`, keeping at most the two
    /// most recent distinct positions (one speculative round needs two: the
    /// rollback target and the verify-window entry).
    fn push_ssm_checkpoint(&mut self, pos: usize) {
        self.trace_state("push", pos);
        self.ssm_checkpoints.retain(|(p, _, _)| *p != pos);
        self.ssm_checkpoints.push((
            pos,
            self.ssm_states.clone(),
            self.ssm_conv_buffers.clone(),
        ));
        if self.ssm_checkpoints.len() > 2 {
            self.ssm_checkpoints.remove(0);
        }
    }

    /// Window size for batched forward; `OXIDIZE_WINDOW_BISECT=off` forces the
    /// per-token path everywhere, `=fwd` only in `forward`, `=many` only in
    /// `forward_many` (debugging / A-B reference).
    fn forward_window_size_for(caller: &str) -> usize {
        let v = std::env::var("OXIDIZE_WINDOW_BISECT").unwrap_or_default();
        if v.contains("off") || v.contains(caller) {
            1
        } else {
            Self::FORWARD_WINDOW
        }
    }

    /// Access the model's inference configuration.
    pub fn config(&self) -> &InferenceConfig {
        &self.config
    }

    pub fn load_from_gguf(
        mapped: &MappedGgufFile,
        config: InferenceConfig,
        layer_cache_size: usize,
    ) -> Result<Self, String> {
        let mut tok_embeddings: Option<WeightStorage> = None;
        let mut tok_embeddings_cols: usize = config.hidden_size;
        let mut norm_weight: Option<Vec<f32>> = None;
        let mut output_weight: Option<WeightStorage> = None;
        let mut layer_tensors: Vec<HashMap<String, GgufTensorRef>> =
            vec![HashMap::new(); config.layer_count];

        let is_supported_quant_gemv = |qtype: GgufQuantizationType| {
            matches!(
                qtype,
                GgufQuantizationType::Q8_0
                    | GgufQuantizationType::Q4_K_S
                    | GgufQuantizationType::Q4_K_M
                    | GgufQuantizationType::Q6_K
            )
        };

        for tensor in mapped.mapped_tensor_infos().iter() {
            let qtype = GgufQuantizationType::from_ggml_type(tensor.ggml_type);
            let value_count: usize = tensor.dimensions.iter().map(|&d| d as usize).product();
            let qsize = quantized_size(qtype, value_count)
                .map_err(|e| format!("quantized_size: {:?}", e))?;
            let offset = tensor.absolute_offset as usize;

            let Some(tensor_name) = normalize_gguf_tensor_name(&tensor.name) else {
                continue;
            };
            match tensor_name.as_str() {
                "tok_embeddings.weight" | "token_embd.weight" => {
                    tok_embeddings_cols = tensor
                        .dimensions
                        .get(1)
                        .copied()
                        .unwrap_or(config.hidden_size as u64)
                        as usize;
                    if is_supported_quant_gemv(qtype) {
                        tok_embeddings = Some(WeightStorage::MmapQuantized(
                            qtype,
                            mapped.mmap(),
                            offset,
                            qsize,
                        ));
                    } else {
                        let qdata = &mapped.bytes()[offset..offset + qsize];
                        let mut f32_data = vec![0.0_f32; value_count];
                        dequantize_scalar(qtype, qdata, &mut f32_data)
                            .map_err(|e| format!("dequantize: {:?}", e))?;
                        tok_embeddings = Some(WeightStorage::F32(f32_data));
                    }
                }
                "norm.weight" | "output_norm.weight" => {
                    let qdata = &mapped.bytes()[offset..offset + qsize];
                    let mut f32_data = vec![0.0_f32; value_count];
                    dequantize_scalar(qtype, qdata, &mut f32_data)
                        .map_err(|e| format!("dequantize: {:?}", e))?;
                    norm_weight = Some(f32_data);
                }
                "output.weight" => {
                    if is_supported_quant_gemv(qtype) {
                        output_weight = Some(WeightStorage::MmapQuantized(
                            qtype,
                            mapped.mmap(),
                            offset,
                            qsize,
                        ));
                    } else {
                        let qdata = &mapped.bytes()[offset..offset + qsize];
                        let mut f32_data = vec![0.0_f32; value_count];
                        dequantize_scalar(qtype, qdata, &mut f32_data)
                            .map_err(|e| format!("dequantize: {:?}", e))?;
                        output_weight = Some(WeightStorage::F32(f32_data));
                    }
                }
                name if name.starts_with("blk.") => {
                    let parts: Vec<&str> = name.split('.').collect();
                    if parts.len() < 4 {
                        continue;
                    }
                    let layer_idx: usize = parts[1]
                        .parse()
                        .map_err(|_| format!("bad layer index: {}", name))?;
                    if layer_idx >= config.layer_count {
                        continue;
                    }
                    let key = parts[2..].join(".");
                    layer_tensors[layer_idx].insert(
                        key,
                        GgufTensorRef {
                            qtype,
                            offset,
                            size: qsize,
                            value_count,
                        },
                    );
                }
                _ => {}
            }
        }

        let tok_embeddings = tok_embeddings.ok_or("missing tok_embeddings.weight")?;
        let norm_weight = norm_weight.ok_or("missing norm.weight")?;
        let output_weight = output_weight.unwrap_or_else(|| tok_embeddings.clone());

        let kv_cache_config = crate::kv_cache::KvCacheConfig {
            layer_count: config.layer_count,
            context_size: config.context_size,
            head_count: config.num_key_value_heads,
            head_dim: config.kv_head_dim(),
            dtype: config.kv_cache_dtype,
            quantization: config.kv_quantization,
        };
        let kv_cache = KvCache::new(kv_cache_config).map_err(|e| format!("kv_cache: {:?}", e))?;

        let mut ssm_states = Vec::with_capacity(config.layer_count);
        let mut ssm_conv_buffers = Vec::with_capacity(config.layer_count);
        let layer_count = config.layer_count;
        for _ in 0..layer_count {
            ssm_states.push(vec![0.0_f32; 1]);
            ssm_conv_buffers.push(ConvHistoryRing::new(4, 0));
        }

        let effective_cache = if layer_cache_size == 0 {
            layer_count
        } else {
            layer_cache_size
        };
        if effective_cache < layer_count {
            eprintln!(
                "layer-wise: layer_cache={effective_cache} < {layer_count} layers — expect low TPS from cache thrashing; use --layer-cache 0 or --layer-cache {layer_count} when RAM allows"
            );
        }

        Ok(Self {
            config,
            mmap: Arc::new(mapped.clone()),
            layer_tensors,
            tok_embeddings,
            tok_embeddings_cols,
            norm_weight,
            output_weight,
            kv_cache,
            ssm_states,
            ssm_conv_buffers,
            ssm_pos: 0,
            ssm_checkpoints: Vec::new(),
            cache: LayerCache::new(effective_cache, layer_count),
        })
    }

    /// Preload layer weights into the LRU cache so decode does not reload tensors each token.
    pub fn warm_layer_cache(&mut self) -> Result<(), String> {
        let layer_count = self.config.layer_count;
        let capacity = self.cache.capacity;
        let warm_count = capacity.min(layer_count);
        if warm_count == 0 {
            return Ok(());
        }
        eprintln!("layer-wise: warming {warm_count}/{layer_count} layer slots...");
        let started = std::time::Instant::now();
        for layer_idx in 0..warm_count {
            if self.cache.entries[layer_idx].is_none() {
                let layer = self.load_layer_weights(layer_idx)?;
                self.cache.put(layer_idx, layer);
            }
        }
        eprintln!(
            "layer-wise: cache warm done in {:.1}s",
            started.elapsed().as_secs_f64()
        );
        Ok(())
    }

    fn load_quant_weight(
        &self,
        qtype: GgufQuantizationType,
        offset: usize,
        size: usize,
        qdata: &[u8],
        prefer_mmap: bool,
    ) -> WeightStorage {
        if prefer_mmap {
            WeightStorage::MmapQuantized(
                qtype,
                self.mmap.mmap(),
                offset,
                size,
            )
        } else {
            WeightStorage::Quantized(qtype, qdata.to_vec())
        }
    }

    fn load_layer_weights(&self, layer_idx: usize) -> Result<LayerWeights, String> {
        let refs = &self.layer_tensors[layer_idx];
        let mut layer = LayerWeights::default();
        let bytes = self.mmap.bytes();
        let is_supported_quant_gemv = |qtype: GgufQuantizationType| {
            matches!(
                qtype,
                GgufQuantizationType::Q8_0
                    | GgufQuantizationType::Q4_K_S
                    | GgufQuantizationType::Q4_K_M
                    | GgufQuantizationType::Q6_K
            )
        };
        let prefer_mmap = |key: &str, size: usize| {
            size > 16 * 1024 * 1024
                || key.starts_with("ffn_gate_exps")
                || key.starts_with("ffn_up_exps")
                || key.starts_with("ffn_down_exps")
                || key.starts_with("ffn_gate_inp")
                || key.starts_with("ffn_gate_shexp")
                || key.starts_with("ffn_gate_inp_shexp")
                || key.starts_with("ffn_up_shexp")
                || key.starts_with("ffn_down_shexp")
        };

        for (key, tensor_ref) in refs.iter() {
            let qdata = &bytes[tensor_ref.offset..tensor_ref.offset + tensor_ref.size];
            let count = tensor_ref.value_count;
            let qtype = tensor_ref.qtype;
            let weight_key = key.as_str();
            let mmap_this = prefer_mmap(weight_key, tensor_ref.size);
            let is_ssm_vec = matches!(weight_key, "ssm_a.weight" | "ssm_conv1d.weight");

            if weight_key.ends_with(".weight")
                && !weight_key.contains("norm")
                && !weight_key.contains("bias")
                && !is_ssm_vec
                && is_supported_quant_gemv(qtype)
            {
                let ws = self.load_quant_weight(
                    qtype,
                    tensor_ref.offset,
                    tensor_ref.size,
                    qdata,
                    mmap_this,
                );
                match weight_key {
                    "attn_q.weight" => layer.attn_q = ws,
                    "attn_k.weight" => layer.attn_k = ws,
                    "attn_v.weight" => layer.attn_v = ws,
                    "attn_output.weight" => layer.attn_output = ws,
                    "ffn_gate.weight" => layer.ffn_gate = ws,
                    "ffn_up.weight" => layer.ffn_up = ws,
                    "ffn_down.weight" => layer.ffn_down = ws,
                    "ffn_gate_exps.weight" => layer.ffn_gate_exps = ws,
                    "ffn_up_exps.weight" => layer.ffn_up_exps = ws,
                    "ffn_down_exps.weight" => layer.ffn_down_exps = ws,
                    "ffn_gate_inp.weight" => layer.ffn_gate_inp = ws,
                    "ffn_gate_inp_shexp.weight" => layer.ffn_gate_inp_shexp = ws,
                    "ffn_gate_shexp.weight" => layer.ffn_gate_shexp = ws,
                    "ffn_up_shexp.weight" => layer.ffn_up_shexp = ws,
                    "ffn_down_shexp.weight" => layer.ffn_down_shexp = ws,
                    "attn_qkv.weight" => layer.attn_qkv = ws,
                    "attn_gate.weight" => layer.attn_gate = ws,
                    "ssm_alpha.weight" => layer.ssm_alpha = ws,
                    "ssm_beta.weight" => layer.ssm_beta = ws,
                    "ssm_out.weight" => layer.ssm_out = ws,
                    _ => {}
                }
            } else {
                let mut v = vec![0.0_f32; count];
                dequantize_scalar(qtype, qdata, &mut v)
                    .map_err(|e| format!("dequantize: {:?}", e))?;
                match weight_key {
                    "attn_norm.weight" => layer.attn_norm = v,
                    "attn_q.weight" => layer.attn_q = WeightStorage::F32(v),
                    "attn_q.bias" => layer.attn_q_bias = v,
                    "attn_k.weight" => layer.attn_k = WeightStorage::F32(v),
                    "attn_k.bias" => layer.attn_k_bias = v,
                    "attn_v.weight" => layer.attn_v = WeightStorage::F32(v),
                    "attn_v.bias" => layer.attn_v_bias = v,
                    "attn_output.weight" => layer.attn_output = WeightStorage::F32(v),
                    "attn_output.bias" => layer.attn_output_bias = v,
                    "ffn_norm.weight" => layer.ffn_norm = v,
                    "post_attention_norm.weight" => layer.post_attention_norm = v,
                    "ffn_gate.weight" => layer.ffn_gate = WeightStorage::F32(v),
                    "ffn_up.weight" => layer.ffn_up = WeightStorage::F32(v),
                    "ffn_down.weight" => layer.ffn_down = WeightStorage::F32(v),
                    "ffn_down.bias" => layer.ffn_down_bias = v,
                    "ffn_gate_exps.weight" => layer.ffn_gate_exps = WeightStorage::F32(v),
                    "ffn_up_exps.weight" => layer.ffn_up_exps = WeightStorage::F32(v),
                    "ffn_down_exps.weight" => layer.ffn_down_exps = WeightStorage::F32(v),
                    "ffn_gate_inp.weight" => layer.ffn_gate_inp = WeightStorage::F32(v),
                    "ffn_gate_inp_shexp.weight" => layer.ffn_gate_inp_shexp = WeightStorage::F32(v),
                    "ffn_gate_shexp.weight" => layer.ffn_gate_shexp = WeightStorage::F32(v),
                    "ffn_up_shexp.weight" => layer.ffn_up_shexp = WeightStorage::F32(v),
                    "ffn_down_shexp.weight" => layer.ffn_down_shexp = WeightStorage::F32(v),
                    "ffn_exp_probs_b.bias" => layer.ffn_exp_probs_b = v,
                    "attn_qkv.weight" => layer.attn_qkv = WeightStorage::F32(v),
                    "attn_gate.weight" => layer.attn_gate = WeightStorage::F32(v),
                    "ssm_a.weight" => layer.ssm_a = v,
                    "ssm_alpha.weight" => layer.ssm_alpha = WeightStorage::F32(v),
                    "ssm_beta.weight" => layer.ssm_beta = WeightStorage::F32(v),
                    "ssm_conv1d.weight" => layer.ssm_conv1d = v,
                    "ssm_dt.bias" => layer.ssm_dt_bias = v,
                    "ssm_norm.weight" => layer.ssm_norm = v,
                    "ssm_out.weight" => layer.ssm_out = WeightStorage::F32(v),
                    "attn_q_norm.weight" => layer.attn_q_norm = v,
                    "attn_k_norm.weight" => layer.attn_k_norm = v,
                    _ => {}
                }
            }
        }
        Ok(layer)
    }

    fn ensure_layer_loaded(&mut self, layer_idx: usize) -> Result<(), String> {
        if self.cache.entries[layer_idx].is_none() {
            let layer = self.load_layer_weights(layer_idx)?;
            self.cache.put(layer_idx, layer);
        }
        self.cache.generation += 1;
        self.cache.access_count[layer_idx] = self.cache.generation;
        Ok(())
    }

    fn layer_ref(&self, layer_idx: usize) -> &LayerWeights {
        self.cache.entries[layer_idx]
            .as_ref()
            .expect("layer must be loaded before layer_ref")
    }

    fn get_or_load_layer(&mut self, layer_idx: usize) -> Result<LayerWeights, String> {
        if let Some(cached) = self.cache.get(layer_idx) {
            return Ok(cached);
        }
        self.load_layer_weights(layer_idx)
    }

    fn return_layer(&mut self, layer_idx: usize, weights: LayerWeights) {
        self.cache.put(layer_idx, weights);
    }

    fn forward_single(&mut self, token: Token, pos: usize) -> Result<Logits, ModelError> {
        self.trace_state("fwd1-entry", pos);
        let cfg = self.config.clone();
        let h = cfg.hidden_size;

        let mut x = vec![0.0_f32; h];
        let token_idx = (token as usize).min(cfg.vocab_size.saturating_sub(1));
        if std::env::var("OXIDIZE_DEBUG_LAYERS").is_ok() && pos == 0 {
            eprintln!("token_idx={token_idx}");
        }
        match &self.tok_embeddings {
            WeightStorage::F32(data) => {
                x.copy_from_slice(&data[token_idx * h..(token_idx + 1) * h]);
            }
            WeightStorage::Quantized(qtype, data) => {
                lookup_quantized_embedding(h, *qtype, data, token_idx, &mut x);
            }
            WeightStorage::MmapQuantized(qtype, mmap, offset, size) => {
                let data = &mmap[*offset..*offset + *size];
                lookup_quantized_embedding(h, *qtype, data, token_idx, &mut x);
            }
        }

        debug_hidden("embed", pos, &x);

        for layer_idx in 0..cfg.layer_count {
            self.ensure_layer_loaded(layer_idx)
                .map_err(|e| ModelError::InferenceFailed(format!("layer load: {}", e)))?;
            // SAFETY: layer weights live in a fixed cache slot; we only touch kv/ssm state below.
            let layer = unsafe {
                let layer_ptr = self.layer_ref(layer_idx) as *const LayerWeights;
                &*layer_ptr
            };
            let is_mamba = !weight_is_empty(&layer.attn_qkv) && layer.attn_q.is_empty();
            let ffn_norm_weight: &[f32] = if !layer.post_attention_norm.is_empty() {
                &layer.post_attention_norm
            } else if !layer.ffn_norm.is_empty() {
                &layer.ffn_norm
            } else {
                &[]
            };

            if is_mamba {
                let mamba_out = self.run_mamba_layer(layer_idx, layer, &x, &cfg)?;
                for (xi, out) in x.iter_mut().zip(mamba_out.iter()).take(h) {
                    *xi += out;
                }
                debug_hidden(&format!("layer {layer_idx} after gdn"), pos, &x);
            } else if !weight_is_empty(&layer.attn_q) {
                let attn_out = self.run_attention_layer(layer_idx, layer, &x, pos, &cfg)?;
                for (xi, out) in x.iter_mut().zip(attn_out.iter()).take(h) {
                    *xi += out;
                }
                debug_hidden(&format!("layer {layer_idx} after attn"), pos, &x);
            }

            let has_dense_ffn = !weight_is_empty(&layer.ffn_gate)
                && !weight_is_empty(&layer.ffn_up)
                && !weight_is_empty(&layer.ffn_down)
                && !ffn_norm_weight.is_empty();
            let has_moe = cfg.num_experts > 0
                && !weight_is_empty(&layer.ffn_gate_exps)
                && !weight_is_empty(&layer.ffn_up_exps)
                && !weight_is_empty(&layer.ffn_down_exps)
                && !weight_is_empty(&layer.ffn_gate_inp)
                && !ffn_norm_weight.is_empty();
            if has_dense_ffn || has_moe {
                let mut ffn_out = vec![0.0_f32; h];
                {
                    let mut normed = vec![0.0_f32; h];
                    rms_norm_model(&x, ffn_norm_weight, cfg.rms_norm_eps, &mut normed, &cfg)?;
                    if has_moe {
                        let moe_i = if cfg.expert_intermediate_size > 0 {
                            cfg.expert_intermediate_size
                        } else {
                            cfg.intermediate_size
                        };
                        let n_sel = cfg.num_experts_per_tok.max(1).min(cfg.num_experts);
                        let mut gate_scratch = vec![0.0_f32; n_sel * moe_i];
                        let mut up_scratch = vec![0.0_f32; n_sel * moe_i];
                        let mut expert_out = vec![0.0_f32; n_sel * h];
                        let mut router_logits = vec![0.0_f32; cfg.num_experts];
                        let mut expert_scores = vec![(0usize, 0.0_f32); cfg.num_experts];
                        let moe_weights = MoeFfnWeights {
                            gate_inp: &layer.ffn_gate_inp,
                            gate_exps: &layer.ffn_gate_exps,
                            up_exps: &layer.ffn_up_exps,
                            down_exps: &layer.ffn_down_exps,
                            exp_probs_b: &layer.ffn_exp_probs_b,
                        };
                        moe_ffn_forward_weights(
                            &moe_weights,
                            &cfg,
                            &normed,
                            &mut ffn_out,
                            &mut gate_scratch,
                            &mut up_scratch,
                            &mut expert_out,
                            &mut router_logits,
                            &mut expert_scores,
                        )?;
                        if !weight_is_empty(&layer.ffn_gate_shexp)
                            && !weight_is_empty(&layer.ffn_up_shexp)
                            && !weight_is_empty(&layer.ffn_down_shexp)
                        {
                            let shexp_i = moe_i;
                            let mut gate = vec![0.0_f32; shexp_i];
                            let mut up = vec![0.0_f32; shexp_i];
                            let mut shexp_out = vec![0.0_f32; h];
                            gemv_weight(&layer.ffn_gate_shexp, shexp_i, h, &normed, &mut gate)
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!("shexp gate: {:?}", e))
                                })?;
                            gemv_weight(&layer.ffn_up_shexp, shexp_i, h, &normed, &mut up)
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!("shexp up: {:?}", e))
                                })?;
                            let mut swiglu = vec![0.0_f32; shexp_i];
                            apply_swiglu_f32(&gate, &up, &mut swiglu).map_err(|e| {
                                ModelError::InferenceFailed(format!("shexp swiglu: {:?}", e))
                            })?;
                            gemv_weight(
                                &layer.ffn_down_shexp,
                                h,
                                shexp_i,
                                &swiglu,
                                &mut shexp_out,
                            )
                            .map_err(|e| {
                                ModelError::InferenceFailed(format!("shexp down: {:?}", e))
                            })?;
                            if !weight_is_empty(&layer.ffn_gate_inp_shexp) {
                                let mut gate_logit = vec![0.0_f32; 1];
                                gemv_weight(
                                    &layer.ffn_gate_inp_shexp,
                                    1,
                                    h,
                                    &normed,
                                    &mut gate_logit,
                                )
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!(
                                        "shexp router gate: {:?}",
                                        e
                                    ))
                                })?;
                                let scale = sigmoid(gate_logit[0]);
                                for val in shexp_out.iter_mut() {
                                    *val *= scale;
                                }
                            }
                            for (out, val) in ffn_out.iter_mut().zip(shexp_out.iter()) {
                                *out += val;
                            }
                        }
                    } else {
                        let mut gate = vec![0.0_f32; cfg.intermediate_size];
                        let mut up = vec![0.0_f32; cfg.intermediate_size];
                        gemv_weight(
                            &layer.ffn_gate,
                            cfg.intermediate_size,
                            h,
                            &normed,
                            &mut gate,
                        )
                        .map_err(|e| ModelError::InferenceFailed(format!("ffn_gate: {:?}", e)))?;
                        gemv_weight(&layer.ffn_up, cfg.intermediate_size, h, &normed, &mut up)
                            .map_err(|e| ModelError::InferenceFailed(format!("ffn_up: {:?}", e)))?;
                        let mut swiglu = vec![0.0_f32; cfg.intermediate_size];
                        apply_swiglu_f32(&gate, &up, &mut swiglu)
                            .map_err(|e| ModelError::InferenceFailed(format!("swiglu: {:?}", e)))?;
                        gemv_weight(
                            &layer.ffn_down,
                            h,
                            cfg.intermediate_size,
                            &swiglu,
                            &mut ffn_out,
                        )
                        .map_err(|e| ModelError::InferenceFailed(format!("ffn_down: {:?}", e)))?;
                        if !layer.ffn_down_bias.is_empty() {
                            for (i, out) in ffn_out.iter_mut().enumerate() {
                                *out += layer.ffn_down_bias[i % layer.ffn_down_bias.len()];
                            }
                        }
                    }
                }
                for (xi, out) in x.iter_mut().zip(ffn_out.iter()).take(h) {
                    *xi += out;
                }
            }

            debug_hidden(&format!("layer {layer_idx}"), pos, &x);
            trace_fwd("single", pos, layer_idx, &x);
        }

        let mut normed = vec![0.0_f32; h];
        rms_norm_model(&x, &self.norm_weight, cfg.rms_norm_eps, &mut normed, &cfg)?;
        let mut logits = vec![0.0_f32; cfg.vocab_size];
        gemv_weight(&self.output_weight, cfg.vocab_size, h, &normed, &mut logits)
            .map_err(|e| ModelError::InferenceFailed(format!("output: {:?}", e)))?;

        if std::env::var("OXIDIZE_DEBUG_LOGITS").is_ok() {
            let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
            eprintln!(
                "OXIDIZE_DEBUG pos={pos} top5: {:?}",
                indexed
                    .iter()
                    .take(5)
                    .map(|(id, val)| (id, val))
                    .collect::<Vec<_>>()
            );
        }

        self.ssm_pos = pos + 1;
        Ok(logits)
    }

    /// Process a window of consecutive tokens layer-major with batched weight
    /// passes (TileRT/MTP-style): every dense projection reads its weights once
    /// per window instead of once per token. Recurrent pieces (conv ring,
    /// delta-rule state, KV attention) run sequentially per token inside each
    /// layer, so results match the token-by-token path.
    ///
    /// Returns logits for every token when `want_all_logits` (speculative
    /// verification), or just the last token's logits otherwise (prefill —
    /// skipping the ~1B-param LM head for all non-final tokens).
    fn forward_window(
        &mut self,
        tokens: &[Token],
        start_pos: usize,
        want_all_logits: bool,
    ) -> Result<Vec<Logits>, ModelError> {
        let kk = tokens.len();
        if kk == 0 {
            return Err(ModelError::EmptyInput);
        }
        if kk == 1 {
            let logits = self.forward_single(tokens[0], start_pos)?;
            return Ok(vec![logits]);
        }
        let cfg = self.config.clone();
        let h = cfg.hidden_size;

        let mut xs = vec![0.0_f32; kk * h];
        for (t, &token) in tokens.iter().enumerate() {
            let token_idx = (token as usize).min(cfg.vocab_size.saturating_sub(1));
            let x_t = &mut xs[t * h..(t + 1) * h];
            match &self.tok_embeddings {
                WeightStorage::F32(data) => {
                    x_t.copy_from_slice(&data[token_idx * h..(token_idx + 1) * h]);
                }
                WeightStorage::Quantized(qtype, data) => {
                    lookup_quantized_embedding(h, *qtype, data, token_idx, x_t);
                }
                WeightStorage::MmapQuantized(qtype, mmap, offset, size) => {
                    let data = &mmap[*offset..*offset + *size];
                    lookup_quantized_embedding(h, *qtype, data, token_idx, x_t);
                }
            }
        }

        for layer_idx in 0..cfg.layer_count {
            self.ensure_layer_loaded(layer_idx)
                .map_err(|e| ModelError::InferenceFailed(format!("layer load: {}", e)))?;
            // SAFETY: layer weights live in a fixed cache slot; we only touch
            // kv/ssm state below (same invariant as forward_single).
            let layer = unsafe {
                let layer_ptr = self.layer_ref(layer_idx) as *const LayerWeights;
                &*layer_ptr
            };
            let is_mamba = !weight_is_empty(&layer.attn_qkv) && layer.attn_q.is_empty();
            let ffn_norm_weight: &[f32] = if !layer.post_attention_norm.is_empty() {
                &layer.post_attention_norm
            } else if !layer.ffn_norm.is_empty() {
                &layer.ffn_norm
            } else {
                &[]
            };

            if is_mamba {
                let residual = self.run_mamba_layer_batch(layer_idx, layer, &xs, kk, &cfg)?;
                for (xi, out) in xs.iter_mut().zip(residual.iter()) {
                    *xi += out;
                }
            } else if !weight_is_empty(&layer.attn_q) {
                // Full-attention layers stay per token: KV append order matters.
                for t in 0..kk {
                    let x_t = xs[t * h..(t + 1) * h].to_vec();
                    let attn_out =
                        self.run_attention_layer(layer_idx, layer, &x_t, start_pos + t, &cfg)?;
                    for (xi, out) in xs[t * h..(t + 1) * h].iter_mut().zip(attn_out.iter()) {
                        *xi += out;
                    }
                }
            }

            let has_dense_ffn = !weight_is_empty(&layer.ffn_gate)
                && !weight_is_empty(&layer.ffn_up)
                && !weight_is_empty(&layer.ffn_down)
                && !ffn_norm_weight.is_empty();
            let has_moe = cfg.num_experts > 0
                && !weight_is_empty(&layer.ffn_gate_exps)
                && !weight_is_empty(&layer.ffn_up_exps)
                && !weight_is_empty(&layer.ffn_down_exps)
                && !weight_is_empty(&layer.ffn_gate_inp)
                && !ffn_norm_weight.is_empty();
            if !(has_dense_ffn || has_moe) {
                continue;
            }

            let mut normed_all = vec![0.0_f32; kk * h];
            for t in 0..kk {
                let mut normed = vec![0.0_f32; h];
                rms_norm_model(
                    &xs[t * h..(t + 1) * h],
                    ffn_norm_weight,
                    cfg.rms_norm_eps,
                    &mut normed,
                    &cfg,
                )?;
                normed_all[t * h..(t + 1) * h].copy_from_slice(&normed);
            }
            let mut ffn_all = vec![0.0_f32; kk * h];

            if has_moe {
                let moe_i = if cfg.expert_intermediate_size > 0 {
                    cfg.expert_intermediate_size
                } else {
                    cfg.intermediate_size
                };
                let n_sel = cfg.num_experts_per_tok.max(1).min(cfg.num_experts);
                let mut gate_scratch = vec![0.0_f32; n_sel * moe_i];
                let mut up_scratch = vec![0.0_f32; n_sel * moe_i];
                let mut expert_out = vec![0.0_f32; n_sel * h];
                let mut router_logits = vec![0.0_f32; cfg.num_experts];
                let mut expert_scores = vec![(0usize, 0.0_f32); cfg.num_experts];
                let moe_weights = MoeFfnWeights {
                    gate_inp: &layer.ffn_gate_inp,
                    gate_exps: &layer.ffn_gate_exps,
                    up_exps: &layer.ffn_up_exps,
                    down_exps: &layer.ffn_down_exps,
                    exp_probs_b: &layer.ffn_exp_probs_b,
                };
                for t in 0..kk {
                    moe_ffn_forward_weights(
                        &moe_weights,
                        &cfg,
                        &normed_all[t * h..(t + 1) * h],
                        &mut ffn_all[t * h..(t + 1) * h],
                        &mut gate_scratch,
                        &mut up_scratch,
                        &mut expert_out,
                        &mut router_logits,
                        &mut expert_scores,
                    )?;
                }
                if !weight_is_empty(&layer.ffn_gate_shexp)
                    && !weight_is_empty(&layer.ffn_up_shexp)
                    && !weight_is_empty(&layer.ffn_down_shexp)
                {
                    let shexp_i = moe_i;
                    let mut gate_all = vec![0.0_f32; kk * shexp_i];
                    let mut up_all = vec![0.0_f32; kk * shexp_i];
                    gemm_weight(
                        &layer.ffn_gate_shexp,
                        shexp_i,
                        h,
                        &normed_all,
                        &mut gate_all,
                        kk,
                    )
                    .map_err(|e| ModelError::InferenceFailed(format!("shexp gate: {:?}", e)))?;
                    gemm_weight(&layer.ffn_up_shexp, shexp_i, h, &normed_all, &mut up_all, kk)
                        .map_err(|e| ModelError::InferenceFailed(format!("shexp up: {:?}", e)))?;
                    let mut swiglu_all = vec![0.0_f32; kk * shexp_i];
                    for t in 0..kk {
                        let mut swiglu = vec![0.0_f32; shexp_i];
                        apply_swiglu_f32(
                            &gate_all[t * shexp_i..(t + 1) * shexp_i],
                            &up_all[t * shexp_i..(t + 1) * shexp_i],
                            &mut swiglu,
                        )
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("shexp swiglu: {:?}", e))
                        })?;
                        swiglu_all[t * shexp_i..(t + 1) * shexp_i].copy_from_slice(&swiglu);
                    }
                    let mut shexp_out_all = vec![0.0_f32; kk * h];
                    gemm_weight(
                        &layer.ffn_down_shexp,
                        h,
                        shexp_i,
                        &swiglu_all,
                        &mut shexp_out_all,
                        kk,
                    )
                    .map_err(|e| ModelError::InferenceFailed(format!("shexp down: {:?}", e)))?;
                    if !weight_is_empty(&layer.ffn_gate_inp_shexp) {
                        let mut gate_logit_all = vec![0.0_f32; kk];
                        gemm_weight(
                            &layer.ffn_gate_inp_shexp,
                            1,
                            h,
                            &normed_all,
                            &mut gate_logit_all,
                            kk,
                        )
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("shexp router gate: {:?}", e))
                        })?;
                        for t in 0..kk {
                            let scale = sigmoid(gate_logit_all[t]);
                            for val in shexp_out_all[t * h..(t + 1) * h].iter_mut() {
                                *val *= scale;
                            }
                        }
                    }
                    for (out, val) in ffn_all.iter_mut().zip(shexp_out_all.iter()) {
                        *out += val;
                    }
                }
            } else {
                let i_size = cfg.intermediate_size;
                let mut gate_all = vec![0.0_f32; kk * i_size];
                let mut up_all = vec![0.0_f32; kk * i_size];
                gemm_weight(&layer.ffn_gate, i_size, h, &normed_all, &mut gate_all, kk)
                    .map_err(|e| ModelError::InferenceFailed(format!("ffn_gate: {:?}", e)))?;
                gemm_weight(&layer.ffn_up, i_size, h, &normed_all, &mut up_all, kk)
                    .map_err(|e| ModelError::InferenceFailed(format!("ffn_up: {:?}", e)))?;
                let mut swiglu_all = vec![0.0_f32; kk * i_size];
                for t in 0..kk {
                    let mut swiglu = vec![0.0_f32; i_size];
                    apply_swiglu_f32(
                        &gate_all[t * i_size..(t + 1) * i_size],
                        &up_all[t * i_size..(t + 1) * i_size],
                        &mut swiglu,
                    )
                    .map_err(|e| ModelError::InferenceFailed(format!("swiglu: {:?}", e)))?;
                    swiglu_all[t * i_size..(t + 1) * i_size].copy_from_slice(&swiglu);
                }
                gemm_weight(&layer.ffn_down, h, i_size, &swiglu_all, &mut ffn_all, kk)
                    .map_err(|e| ModelError::InferenceFailed(format!("ffn_down: {:?}", e)))?;
                if !layer.ffn_down_bias.is_empty() {
                    for t in 0..kk {
                        for (i, out) in ffn_all[t * h..(t + 1) * h].iter_mut().enumerate() {
                            *out += layer.ffn_down_bias[i % layer.ffn_down_bias.len()];
                        }
                    }
                }
            }

            for (xi, out) in xs.iter_mut().zip(ffn_all.iter()) {
                *xi += out;
            }
            for t in 0..kk {
                trace_fwd("window", start_pos + t, layer_idx, &xs[t * h..(t + 1) * h]);
            }
        }

        // Final norm + LM head, batched over the tokens that need logits.
        let needed: Vec<usize> = if want_all_logits {
            (0..kk).collect()
        } else {
            vec![kk - 1]
        };
        let nb = needed.len();
        let mut normed_all = vec![0.0_f32; nb * h];
        for (j, &t) in needed.iter().enumerate() {
            let mut normed = vec![0.0_f32; h];
            rms_norm_model(
                &xs[t * h..(t + 1) * h],
                &self.norm_weight,
                cfg.rms_norm_eps,
                &mut normed,
                &cfg,
            )?;
            normed_all[j * h..(j + 1) * h].copy_from_slice(&normed);
        }
        let mut logits_all = vec![0.0_f32; nb * cfg.vocab_size];
        gemm_weight(
            &self.output_weight,
            cfg.vocab_size,
            h,
            &normed_all,
            &mut logits_all,
            nb,
        )
        .map_err(|e| ModelError::InferenceFailed(format!("output: {:?}", e)))?;
        self.ssm_pos = start_pos + kk;
        Ok(needed
            .iter()
            .enumerate()
            .map(|(j, _)| logits_all[j * cfg.vocab_size..(j + 1) * cfg.vocab_size].to_vec())
            .collect())
    }

    fn run_mamba_layer(
        &mut self,
        layer_idx: usize,
        layer: &LayerWeights,
        x: &[f32],
        cfg: &InferenceConfig,
    ) -> Result<Vec<f32>, ModelError> {
        let h = cfg.hidden_size;
        let mut normed = vec![0.0_f32; h];
        rms_norm_model(x, &layer.attn_norm, cfg.rms_norm_eps, &mut normed, cfg)?;

        let qkv_out_len = weight_output_dim(&layer.attn_qkv, h);
        let value_dim = weight_output_dim(&layer.attn_gate, h);
        if qkv_out_len == 0 || value_dim == 0 {
            return Ok(vec![0.0_f32; h]);
        }
        let key_dim = qkv_out_len.saturating_sub(value_dim) / 2;
        let num_v_heads = layer.ssm_a.len().max(1);
        let head_v_dim = if layer.ssm_norm.len() > 1 {
            layer.ssm_norm.len()
        } else if value_dim.is_multiple_of(num_v_heads) {
            value_dim / num_v_heads
        } else {
            return Err(ModelError::InferenceFailed(format!(
                "layer {layer_idx} GDN: cannot infer head_v_dim (value_dim={value_dim}, num_v_heads={num_v_heads})"
            )));
        };
        if weight_is_empty(&layer.ssm_alpha) {
            return Err(ModelError::InferenceFailed(format!(
                "layer {layer_idx} GDN missing ssm_alpha (in_proj_a) weights"
            )));
        }
        let num_k_heads = if head_v_dim > 0 && key_dim >= head_v_dim {
            key_dim / head_v_dim
        } else {
            1
        };
        let head_k_dim = if num_k_heads > 0 {
            key_dim / num_k_heads
        } else {
            head_v_dim
        };
        let head_repeat = num_v_heads / num_k_heads.max(1);

        let mut mixed_qkv = vec![0.0_f32; qkv_out_len];
        gemv_weight(
            &layer.attn_qkv,
            qkv_out_len,
            h,
            &normed,
            &mut mixed_qkv,
        )
        .map_err(|e| ModelError::InferenceFailed(format!("attn_qkv: {:?}", e)))?;

        let conv_kernel = 4_usize;
        let mut conv_out = vec![0.0_f32; qkv_out_len];
        if !layer.ssm_conv1d.is_empty() && layer.ssm_conv1d.len() == conv_kernel * qkv_out_len {
            if self.ssm_conv_buffers[layer_idx].dim != qkv_out_len {
                self.ssm_conv_buffers[layer_idx] = ConvHistoryRing::new(conv_kernel, qkv_out_len);
            }
            let buffer = &self.ssm_conv_buffers[layer_idx];
            for c in 0..qkv_out_len {
                let mut sum = 0.0_f32;
                // Weights are tap-major [kernel, channels]; newest input uses the last tap.
                sum += layer.ssm_conv1d[(conv_kernel - 1) * qkv_out_len + c] * mixed_qkv[c];
                for b in 1..conv_kernel {
                    if let Some(prev) = buffer.past_frame(b) {
                        let weight_idx = (conv_kernel - 1 - b) * qkv_out_len + c;
                        sum += layer.ssm_conv1d[weight_idx] * prev[c];
                    }
                }
                conv_out[c] = sum;
            }
            self.ssm_conv_buffers[layer_idx].push(&mixed_qkv);
        } else {
            conv_out.copy_from_slice(&mixed_qkv);
        }

        for val in conv_out.iter_mut() {
            *val *= 1.0_f32 / (1.0_f32 + (-*val).exp());
        }

        debug_vec(&format!("layer {layer_idx} gdn conv"), &conv_out);

        let mut b = vec![0.0_f32; num_v_heads];
        if !weight_is_empty(&layer.ssm_beta) {
            gemv_weight(&layer.ssm_beta, num_v_heads, h, &normed, &mut b)
                .map_err(|e| ModelError::InferenceFailed(format!("ssm_beta: {:?}", e)))?;
        }
        let mut a = vec![0.0_f32; num_v_heads];
        gemv_weight(&layer.ssm_alpha, num_v_heads, h, &normed, &mut a)
            .map_err(|e| ModelError::InferenceFailed(format!("ssm_alpha: {:?}", e)))?;

        let mut z = vec![0.0_f32; value_dim];
        gemv_weight(&layer.attn_gate, value_dim, h, &normed, &mut z)
            .map_err(|e| ModelError::InferenceFailed(format!("attn_gate: {:?}", e)))?;

        let state_elems = num_v_heads * head_k_dim * head_v_dim;
        if self.ssm_states[layer_idx].len() != state_elems {
            self.ssm_states[layer_idx] = vec![0.0_f32; state_elems];
        }
        let state = &mut self.ssm_states[layer_idx];

        let q_scale = 1.0_f32 / (head_k_dim as f32).sqrt();
        let mut core_out = vec![0.0_f32; value_dim];

        // Heads are independent (disjoint state and output chunks), so run the
        // delta-rule update in parallel across v_heads. Body is numerically
        // identical to the previous sequential loop.
        let head_state = head_k_dim * head_v_dim;
        let conv_out_ref = &conv_out;
        state
            .par_chunks_mut(head_state)
            .zip(core_out.par_chunks_mut(head_v_dim))
            .enumerate()
            .for_each(|(v_head, (state_h, out_h))| {
                let k_head = v_head / head_repeat;
                let q_off = k_head * head_k_dim;
                let k_off = key_dim + k_head * head_k_dim;
                let v_off = key_dim * 2 + v_head * head_v_dim;

                let mut q = conv_out_ref[q_off..q_off + head_k_dim].to_vec();
                let mut k = conv_out_ref[k_off..k_off + head_k_dim].to_vec();
                l2_normalize(&mut q);
                l2_normalize(&mut k);
                for x in q.iter_mut() {
                    *x *= q_scale;
                }

                let v = &conv_out_ref[v_off..v_off + head_v_dim];
                let beta = sigmoid(b[v_head]);
                let a_val = a[v_head];
                let dt = if v_head < layer.ssm_dt_bias.len() {
                    softplus(a_val + layer.ssm_dt_bias[v_head])
                } else {
                    softplus(a_val)
                };
                let a_log = if v_head < layer.ssm_a.len() {
                    layer.ssm_a[v_head]
                } else {
                    0.0_f32
                };
                let g = -(a_log.exp()) * dt;
                let decay = g.exp();

                for s in state_h.iter_mut() {
                    *s *= decay;
                }

                let mut kv_mem = vec![0.0_f32; head_v_dim];
                for j in 0..head_v_dim {
                    let mut sum = 0.0_f32;
                    for i in 0..head_k_dim {
                        sum += state_h[i * head_v_dim + j] * k[i];
                    }
                    kv_mem[j] = sum;
                }

                let mut delta = vec![0.0_f32; head_v_dim];
                for j in 0..head_v_dim {
                    delta[j] = (v[j] - kv_mem[j]) * beta;
                }

                for i in 0..head_k_dim {
                    for j in 0..head_v_dim {
                        state_h[i * head_v_dim + j] += k[i] * delta[j];
                    }
                }

                for j in 0..head_v_dim {
                    let mut sum = 0.0_f32;
                    for i in 0..head_k_dim {
                        sum += state_h[i * head_v_dim + j] * q[i];
                    }
                    out_h[j] = sum;
                }
            });

        debug_vec(&format!("layer {layer_idx} gdn core"), &core_out);

        if !layer.ssm_norm.is_empty() && layer.ssm_norm.len() == head_v_dim {
            for head in 0..num_v_heads {
                let start = head * head_v_dim;
                let end = start + head_v_dim;
                gated_rms_norm(
                    &mut core_out[start..end],
                    &layer.ssm_norm,
                    &z[start..end],
                    cfg.rms_norm_eps,
                );
            }
        }

        debug_vec(&format!("layer {layer_idx} gdn normed"), &core_out);

        let mut residual = vec![0.0_f32; h];
        if !weight_is_empty(&layer.ssm_out) {
            let out_len = weight_output_dim(&layer.ssm_out, value_dim);
            if out_len > 0 {
                let mut projected = vec![0.0_f32; out_len];
                gemv_weight(
                    &layer.ssm_out,
                    out_len,
                    value_dim,
                    &core_out,
                    &mut projected,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("ssm_out: {:?}", e)))?;
                let copy_len = h.min(projected.len());
                residual[..copy_len].copy_from_slice(&projected[..copy_len]);
            }
        } else {
            let copy_len = h.min(core_out.len());
            residual[..copy_len].copy_from_slice(&core_out[..copy_len]);
        }
        Ok(residual)
    }

    /// Batched [`Self::run_mamba_layer`]: processes `kk` consecutive tokens
    /// through one GDN layer with the dense projections (qkv / beta / alpha /
    /// gate / out) computed as fused batch GEMMs so each weight block is read
    /// once per window instead of once per token. The conv ring and delta-rule
    /// recurrence stay sequential over tokens (per head), preserving the exact
    /// math of the single-token path.
    fn run_mamba_layer_batch(
        &mut self,
        layer_idx: usize,
        layer: &LayerWeights,
        xs: &[f32],
        kk: usize,
        cfg: &InferenceConfig,
    ) -> Result<Vec<f32>, ModelError> {
        let h = cfg.hidden_size;
        let mut normed_all = vec![0.0_f32; kk * h];
        for t in 0..kk {
            let mut normed = vec![0.0_f32; h];
            rms_norm_model(
                &xs[t * h..(t + 1) * h],
                &layer.attn_norm,
                cfg.rms_norm_eps,
                &mut normed,
                cfg,
            )?;
            normed_all[t * h..(t + 1) * h].copy_from_slice(&normed);
        }

        let qkv_out_len = weight_output_dim(&layer.attn_qkv, h);
        let value_dim = weight_output_dim(&layer.attn_gate, h);
        if qkv_out_len == 0 || value_dim == 0 {
            return Ok(vec![0.0_f32; kk * h]);
        }
        let key_dim = qkv_out_len.saturating_sub(value_dim) / 2;
        let num_v_heads = layer.ssm_a.len().max(1);
        let head_v_dim = if layer.ssm_norm.len() > 1 {
            layer.ssm_norm.len()
        } else if value_dim.is_multiple_of(num_v_heads) {
            value_dim / num_v_heads
        } else {
            return Err(ModelError::InferenceFailed(format!(
                "layer {layer_idx} GDN: cannot infer head_v_dim (value_dim={value_dim}, num_v_heads={num_v_heads})"
            )));
        };
        if weight_is_empty(&layer.ssm_alpha) {
            return Err(ModelError::InferenceFailed(format!(
                "layer {layer_idx} GDN missing ssm_alpha (in_proj_a) weights"
            )));
        }
        let num_k_heads = if head_v_dim > 0 && key_dim >= head_v_dim {
            key_dim / head_v_dim
        } else {
            1
        };
        let head_k_dim = key_dim.checked_div(num_k_heads).unwrap_or(head_v_dim);
        let head_repeat = num_v_heads / num_k_heads.max(1);

        // Batched dense projections — all share `normed_all` as input.
        let mut mixed_all = vec![0.0_f32; kk * qkv_out_len];
        gemm_weight(
            &layer.attn_qkv,
            qkv_out_len,
            h,
            &normed_all,
            &mut mixed_all,
            kk,
        )
        .map_err(|e| ModelError::InferenceFailed(format!("attn_qkv: {:?}", e)))?;
        let mut b_all = vec![0.0_f32; kk * num_v_heads];
        if !weight_is_empty(&layer.ssm_beta) {
            gemm_weight(&layer.ssm_beta, num_v_heads, h, &normed_all, &mut b_all, kk)
                .map_err(|e| ModelError::InferenceFailed(format!("ssm_beta: {:?}", e)))?;
        }
        let mut a_all = vec![0.0_f32; kk * num_v_heads];
        gemm_weight(&layer.ssm_alpha, num_v_heads, h, &normed_all, &mut a_all, kk)
            .map_err(|e| ModelError::InferenceFailed(format!("ssm_alpha: {:?}", e)))?;
        let mut z_all = vec![0.0_f32; kk * value_dim];
        gemm_weight(&layer.attn_gate, value_dim, h, &normed_all, &mut z_all, kk)
            .map_err(|e| ModelError::InferenceFailed(format!("attn_gate: {:?}", e)))?;

        // Causal conv + SiLU: sequential over tokens (ring buffer recurrence).
        let conv_kernel = 4_usize;
        let mut conv_all = vec![0.0_f32; kk * qkv_out_len];
        for t in 0..kk {
            let mixed = &mixed_all[t * qkv_out_len..(t + 1) * qkv_out_len];
            let conv_out = &mut conv_all[t * qkv_out_len..(t + 1) * qkv_out_len];
            if !layer.ssm_conv1d.is_empty() && layer.ssm_conv1d.len() == conv_kernel * qkv_out_len
            {
                if self.ssm_conv_buffers[layer_idx].dim != qkv_out_len {
                    self.ssm_conv_buffers[layer_idx] =
                        ConvHistoryRing::new(conv_kernel, qkv_out_len);
                }
                let buffer = &self.ssm_conv_buffers[layer_idx];
                for c in 0..qkv_out_len {
                    let mut sum =
                        layer.ssm_conv1d[(conv_kernel - 1) * qkv_out_len + c] * mixed[c];
                    for b in 1..conv_kernel {
                        if let Some(prev) = buffer.past_frame(b) {
                            let weight_idx = (conv_kernel - 1 - b) * qkv_out_len + c;
                            sum += layer.ssm_conv1d[weight_idx] * prev[c];
                        }
                    }
                    conv_out[c] = sum;
                }
                self.ssm_conv_buffers[layer_idx].push(mixed);
            } else {
                conv_out.copy_from_slice(mixed);
            }
            for val in conv_out.iter_mut() {
                *val *= 1.0_f32 / (1.0_f32 + (-*val).exp());
            }
        }

        let state_elems = num_v_heads * head_k_dim * head_v_dim;
        if self.ssm_states[layer_idx].len() != state_elems {
            self.ssm_states[layer_idx] = vec![0.0_f32; state_elems];
        }
        let state = &mut self.ssm_states[layer_idx];

        let q_scale = 1.0_f32 / (head_k_dim as f32).sqrt();

        // Delta-rule recurrence: parallel over heads, sequential over tokens
        // within each head. Output is head-major scratch, transposed below.
        let head_state = head_k_dim * head_v_dim;
        let conv_ref = &conv_all;
        let b_ref = &b_all;
        let a_ref = &a_all;
        let mut core_head_major = vec![0.0_f32; num_v_heads * kk * head_v_dim];
        state
            .par_chunks_mut(head_state)
            .zip(core_head_major.par_chunks_mut(kk * head_v_dim))
            .enumerate()
            .for_each(|(v_head, (state_h, out_h))| {
                let k_head = v_head / head_repeat;
                let q_off = k_head * head_k_dim;
                let k_off = key_dim + k_head * head_k_dim;
                let v_off = key_dim * 2 + v_head * head_v_dim;
                let a_log = if v_head < layer.ssm_a.len() {
                    layer.ssm_a[v_head]
                } else {
                    0.0_f32
                };

                for t in 0..kk {
                    let conv_out = &conv_ref[t * qkv_out_len..(t + 1) * qkv_out_len];
                    let mut q = conv_out[q_off..q_off + head_k_dim].to_vec();
                    let mut k = conv_out[k_off..k_off + head_k_dim].to_vec();
                    l2_normalize(&mut q);
                    l2_normalize(&mut k);
                    for x in q.iter_mut() {
                        *x *= q_scale;
                    }

                    let v = &conv_out[v_off..v_off + head_v_dim];
                    let beta = sigmoid(b_ref[t * num_v_heads + v_head]);
                    let a_val = a_ref[t * num_v_heads + v_head];
                    let dt = if v_head < layer.ssm_dt_bias.len() {
                        softplus(a_val + layer.ssm_dt_bias[v_head])
                    } else {
                        softplus(a_val)
                    };
                    let g = -(a_log.exp()) * dt;
                    let decay = g.exp();

                    for s in state_h.iter_mut() {
                        *s *= decay;
                    }

                    let mut kv_mem = vec![0.0_f32; head_v_dim];
                    for j in 0..head_v_dim {
                        let mut sum = 0.0_f32;
                        for i in 0..head_k_dim {
                            sum += state_h[i * head_v_dim + j] * k[i];
                        }
                        kv_mem[j] = sum;
                    }

                    let mut delta = vec![0.0_f32; head_v_dim];
                    for j in 0..head_v_dim {
                        delta[j] = (v[j] - kv_mem[j]) * beta;
                    }

                    for i in 0..head_k_dim {
                        for j in 0..head_v_dim {
                            state_h[i * head_v_dim + j] += k[i] * delta[j];
                        }
                    }

                    let out_t = &mut out_h[t * head_v_dim..(t + 1) * head_v_dim];
                    for j in 0..head_v_dim {
                        let mut sum = 0.0_f32;
                        for i in 0..head_k_dim {
                            sum += state_h[i * head_v_dim + j] * q[i];
                        }
                        out_t[j] = sum;
                    }
                }
            });

        // Transpose head-major scratch into token-major core output.
        let mut core_all = vec![0.0_f32; kk * value_dim];
        for v_head in 0..num_v_heads {
            for t in 0..kk {
                let src = v_head * kk * head_v_dim + t * head_v_dim;
                let dst = t * value_dim + v_head * head_v_dim;
                core_all[dst..dst + head_v_dim]
                    .copy_from_slice(&core_head_major[src..src + head_v_dim]);
            }
        }

        if !layer.ssm_norm.is_empty() && layer.ssm_norm.len() == head_v_dim {
            for t in 0..kk {
                for head in 0..num_v_heads {
                    let start = t * value_dim + head * head_v_dim;
                    let end = start + head_v_dim;
                    gated_rms_norm(
                        &mut core_all[start..end],
                        &layer.ssm_norm,
                        &z_all[start..end],
                        cfg.rms_norm_eps,
                    );
                }
            }
        }

        let mut residual_all = vec![0.0_f32; kk * h];
        if !weight_is_empty(&layer.ssm_out) {
            let out_len = weight_output_dim(&layer.ssm_out, value_dim);
            if out_len > 0 {
                let mut proj_all = vec![0.0_f32; kk * out_len];
                gemm_weight(
                    &layer.ssm_out,
                    out_len,
                    value_dim,
                    &core_all,
                    &mut proj_all,
                    kk,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("ssm_out: {:?}", e)))?;
                let copy_len = h.min(out_len);
                for t in 0..kk {
                    residual_all[t * h..t * h + copy_len]
                        .copy_from_slice(&proj_all[t * out_len..t * out_len + copy_len]);
                }
            }
        } else {
            let copy_len = h.min(value_dim);
            for t in 0..kk {
                residual_all[t * h..t * h + copy_len]
                    .copy_from_slice(&core_all[t * value_dim..t * value_dim + copy_len]);
            }
        }
        Ok(residual_all)
    }

    fn run_attention_layer(
        &mut self,
        layer_idx: usize,
        layer: &LayerWeights,
        x: &[f32],
        pos: usize,
        cfg: &InferenceConfig,
    ) -> Result<Vec<f32>, ModelError> {
        let h = cfg.hidden_size;
        let n = cfg.num_attention_heads;
        let k = cfg.num_key_value_heads;
        let mut attn_out = vec![0.0_f32; h];

        let mut normed = vec![0.0_f32; h];
        rms_norm_model(x, &layer.attn_norm, cfg.rms_norm_eps, &mut normed, cfg)?;

        let q_len = weight_output_dim(&layer.attn_q, h);
        let kv_len = if !weight_is_empty(&layer.attn_k) {
            weight_output_dim(&layer.attn_k, h)
        } else {
            0
        };
        let attn_output_input_len = if !weight_is_empty(&layer.attn_output) {
            weight_output_dim(&layer.attn_output, h)
        } else {
            0
        };

        // Run Q, K, V projections concurrently (one fork instead of three
        // sequential parallel regions) — same pattern as InferenceModel.
        let mut q_full = vec![0.0_f32; q_len];
        let mut k_vec = vec![0.0_f32; kv_len];
        let mut v_vec = vec![0.0_f32; kv_len];
        {
            let normed_ref = &normed;
            let ((qr, kr), vr) = rayon::join(
                || {
                    rayon::join(
                        || gemv_weight(&layer.attn_q, q_len, h, normed_ref, &mut q_full),
                        || {
                            if weight_is_empty(&layer.attn_k) {
                                Ok(())
                            } else {
                                gemv_weight(&layer.attn_k, kv_len, h, normed_ref, &mut k_vec)
                            }
                        },
                    )
                },
                || {
                    if weight_is_empty(&layer.attn_v) {
                        Ok(())
                    } else {
                        gemv_weight(&layer.attn_v, kv_len, h, normed_ref, &mut v_vec)
                    }
                },
            );
            qr.map_err(|e| ModelError::InferenceFailed(format!("attn_q: {:?}", e)))?;
            kr.map_err(|e| ModelError::InferenceFailed(format!("attn_k: {:?}", e)))?;
            vr.map_err(|e| ModelError::InferenceFailed(format!("attn_v: {:?}", e)))?;
        }
        if !layer.attn_q_bias.is_empty() {
            for (i, q) in q_full.iter_mut().enumerate() {
                *q += layer.attn_q_bias[i % layer.attn_q_bias.len()];
            }
        }
        if !layer.attn_k_bias.is_empty() {
            for (i, k) in k_vec.iter_mut().enumerate() {
                *k += layer.attn_k_bias[i % layer.attn_k_bias.len()];
            }
        }
        if !layer.attn_v_bias.is_empty() {
            for (i, v) in v_vec.iter_mut().enumerate() {
                *v += layer.attn_v_bias[i % layer.attn_v_bias.len()];
            }
        }

        let kv_head_dim = if k > 0 && kv_len % k == 0 {
            kv_len / k
        } else if kv_len > 0 {
            kv_len
        } else {
            cfg.kv_head_dim()
        };

        let q_len_used_guess = if attn_output_input_len > 0 && q_len == 2 * attn_output_input_len {
            q_len / 2
        } else if attn_output_input_len > 0 {
            q_len.min(attn_output_input_len)
        } else if q_len > h {
            h
        } else {
            q_len
        };
        let q_heads_guess = if n > 0 && q_len_used_guess.is_multiple_of(n) {
            n
        } else {
            1
        };
        let q_head_dim_guess = if q_heads_guess > 0 {
            q_len_used_guess / q_heads_guess
        } else {
            q_len_used_guess
        };

        let (mut q, attn_gate) = if attn_output_input_len > 0 && q_len == 2 * attn_output_input_len {
            let (query, gate) = split_gated_query_proj(&q_full, q_head_dim_guess).ok_or_else(|| {
                ModelError::InferenceFailed("gated q_proj split failed".to_owned())
            })?;
            (query, Some(gate))
        } else {
            (q_full[..q_len_used_guess].to_vec(), None)
        };



        if std::env::var_os("OXIDIZE_TRACE_FWD").is_some() {
            let s = |v: &[f32]| v.iter().map(|x| *x as f64).sum::<f64>();
            eprintln!(
                "STAGE lw pos={pos} layer={layer_idx} normed={:.6e} q={:.6e} k={:.6e} v={:.6e} x={:.6e} nw_len={} nw={:.6e}",
                s(&normed), s(&q), s(&k_vec), s(&v_vec), s(x), layer.attn_norm.len(), s(&layer.attn_norm)
            );
        }
        let q_len_used = q.len();
        let q_head_dim = if n > 0 && q_len_used.is_multiple_of(n) {
            q_len_used / n
        } else {
            q_len_used
        };
        let q_heads = q_len_used.checked_div(q_head_dim).unwrap_or(1);
        let kv_heads = kv_len.checked_div(kv_head_dim).unwrap_or(1);

        if !layer.attn_q_norm.is_empty() && q_head_dim == layer.attn_q_norm.len() {
            for head in 0..q_heads {
                let start = head * q_head_dim;
                let end = start + q_head_dim;
                if end > q.len() {
                    break;
                }
                let mut normed_head = vec![0.0_f32; q_head_dim];
                rms_norm_model(
                    &q[start..end],
                    &layer.attn_q_norm,
                    cfg.rms_norm_eps,
                    &mut normed_head,
                    cfg,
                )?;
                q[start..end].copy_from_slice(&normed_head);
            }
        }
        if !layer.attn_k_norm.is_empty() && kv_head_dim == layer.attn_k_norm.len() {
            for head in 0..kv_heads {
                let start = head * kv_head_dim;
                let end = start + kv_head_dim;
                if end > k_vec.len() {
                    break;
                }
                let mut normed_head = vec![0.0_f32; kv_head_dim];
                rms_norm_model(
                    &k_vec[start..end],
                    &layer.attn_k_norm,
                    cfg.rms_norm_eps,
                    &mut normed_head,
                    cfg,
                )?;
                k_vec[start..end].copy_from_slice(&normed_head);
            }
        }

        for head in 0..q_heads {
            let off = head * q_head_dim;
            if off + q_head_dim > q.len() {
                break;
            }
            let q_rope_len = cfg.effective_rope_dim().min(q_head_dim);
            let mut rotated = vec![0.0_f32; q_rope_len];
            apply_rope_f32(
                &q[off..off + q_rope_len],
                pos,
                q_rope_len,
                cfg.rope_theta,
                &mut rotated,
            )
            .map_err(|e| ModelError::InferenceFailed(format!("rope q: {:?}", e)))?;
            q[off..off + q_rope_len].copy_from_slice(&rotated);
        }
        for head in 0..kv_heads {
            let off = head * kv_head_dim;
            if off + kv_head_dim > k_vec.len() {
                break;
            }
            let k_rope_len = cfg.effective_rope_dim().min(kv_head_dim);
            let mut rotated = vec![0.0_f32; k_rope_len];
            apply_rope_f32(
                &k_vec[off..off + k_rope_len],
                pos,
                k_rope_len,
                cfg.rope_theta,
                &mut rotated,
            )
            .map_err(|e| ModelError::InferenceFailed(format!("rope k: {:?}", e)))?;
            k_vec[off..off + k_rope_len].copy_from_slice(&rotated);
        }

        self.kv_cache
            .set(layer_idx, pos, &k_vec, &v_vec)
            .map_err(|e| ModelError::InferenceFailed(format!("kv set: {:?}", e)))?;

        let seq_len = pos + 1;
        let borrowed_key_cache = self
            .kv_cache
            .f32_layer_key_prefix(layer_idx, seq_len)
            .map_err(|e| ModelError::InferenceFailed(format!("kv borrow keys: {:?}", e)))?;
        let borrowed_value_cache = self
            .kv_cache
            .f32_layer_value_prefix(layer_idx, seq_len)
            .map_err(|e| ModelError::InferenceFailed(format!("kv borrow values: {:?}", e)))?;
        let (key_cache, value_cache) = match (borrowed_key_cache, borrowed_value_cache) {
            (Some(keys), Some(values)) => (
                AttentionCacheSlice::Borrowed(keys),
                AttentionCacheSlice::Borrowed(values),
            ),
            _ => {
                let mut key_cache = vec![0.0_f32; seq_len * kv_len];
                let mut value_cache = vec![0.0_f32; seq_len * kv_len];
                self.kv_cache
                    .copy_layer_keys(layer_idx, seq_len, &mut key_cache)
                    .map_err(|e| ModelError::InferenceFailed(format!("kv copy keys: {:?}", e)))?;
                self.kv_cache
                    .copy_layer_values(layer_idx, seq_len, &mut value_cache)
                    .map_err(|e| ModelError::InferenceFailed(format!("kv copy values: {:?}", e)))?;
                (
                    AttentionCacheSlice::Owned(key_cache),
                    AttentionCacheSlice::Owned(value_cache),
                )
            }
        };
        let key_cache = key_cache.as_slice();
        let value_cache = value_cache.as_slice();

        let mut attn_result = vec![0.0_f32; q_len_used];
        let actual_kv_group_size = q_heads
            .checked_div(kv_heads)
            .filter(|g| *g > 0)
            .unwrap_or(1);
        for head in 0..q_heads {
            let kv_head = head / actual_kv_group_size;
            let q_head_start = head * q_head_dim;
            let q_head_end = q_head_start + q_head_dim;
            if q_head_end > q.len() {
                break;
            }
            let q_head = &q[q_head_start..q_head_end];

            let q_head_for_attn = if q_head_dim > kv_head_dim {
                &q_head[..kv_head_dim]
            } else {
                q_head
            };
            let mut out_head = vec![0.0_f32; kv_head_dim];
            flash_attention_decode_f32(
                q_head_for_attn,
                key_cache,
                value_cache,
                seq_len,
                kv_head_dim,
                kv_len,
                kv_head,
                &mut out_head,
            )
            .map_err(|e| ModelError::InferenceFailed(format!("flash attention: {:?}", e)))?;

            let write_start = head * kv_head_dim;
            if write_start + out_head.len() <= attn_result.len() {
                attn_result[write_start..write_start + out_head.len()].copy_from_slice(&out_head);
            }
        }

        let mut attn_input = if attn_output_input_len > 0 && attn_result.len() != attn_output_input_len
        {
            if attn_result.len() >= attn_output_input_len {
                attn_result[..attn_output_input_len].to_vec()
            } else {
                let mut padded = vec![0.0_f32; attn_output_input_len];
                padded[..attn_result.len()].copy_from_slice(&attn_result);
                padded
            }
        } else {
            attn_result
        };

        if let Some(gate) = attn_gate {
            for (out, g) in attn_input.iter_mut().zip(gate.iter()) {
                *out *= sigmoid(*g);
            }
        }

        if !weight_is_empty(&layer.attn_output) && attn_output_input_len > 0 {
            gemv_weight(
                &layer.attn_output,
                h,
                attn_output_input_len,
                &attn_input,
                &mut attn_out,
            )
            .map_err(|e| ModelError::InferenceFailed(format!("attn_output: {:?}", e)))?;
            if !layer.attn_output_bias.is_empty() {
                for (i, out) in attn_out.iter_mut().enumerate() {
                    *out += layer.attn_output_bias[i % layer.attn_output_bias.len()];
                }
            }
        }

        Ok(attn_out)
    }
}

impl Model for LayerWiseModel {
    fn rewind_to(&mut self, consumed_tokens: usize) -> Result<(), ModelError> {
        let position = if consumed_tokens == 0 {
            0
        } else {
            consumed_tokens.saturating_sub(1)
        };
        if consumed_tokens == 0 {
            // Fresh session: recurrent GDN state is not position-addressable
            // like the KV cache, so it must be cleared explicitly or every new
            // request is conditioned on leftover state from the previous one.
            for state in self.ssm_states.iter_mut() {
                state.fill(0.0_f32);
            }
            for buffer in self.ssm_conv_buffers.iter_mut() {
                let dim = buffer.dim;
                if dim > 0 {
                    *buffer = ConvHistoryRing::new(4, dim);
                }
            }
            self.ssm_pos = 0;
            self.ssm_checkpoints.clear();
        } else if consumed_tokens == self.ssm_pos {
            // State already sits at the target position. Capture a checkpoint:
            // the speculative loop rewinds here, forwards the pending token,
            // verifies a draft window, then rolls back to this position.
            self.push_ssm_checkpoint(consumed_tokens);
        } else if let Some(idx) = self
            .ssm_checkpoints
            .iter()
            .position(|(p, _, _)| *p == consumed_tokens)
        {
            // Speculative rollback: restore the recurrent state snapshot taken
            // before the rejected draft window was processed.
            let (_, states, bufs) = &self.ssm_checkpoints[idx];
            self.ssm_states = states.clone();
            self.ssm_conv_buffers = bufs.clone();
            self.ssm_pos = consumed_tokens;
            self.trace_state("restore", consumed_tokens);
        } else if self.ssm_states.iter().any(|s| s.len() > 1) {
            // GDN model rewinding to a position we have no snapshot for: the
            // recurrent state cannot be reconstructed. Warn instead of failing
            // so non-speculative callers keep the (previous) best-effort
            // behavior, but generation quality may degrade past this point.
            eprintln!(
                "layer-wise: rewind_to({consumed_tokens}) without a GDN checkpoint (state at {}); recurrent state may be stale",
                self.ssm_pos
            );
        }
        self.kv_cache
            .rewind_to(position)
            .map_err(|e| ModelError::InferenceFailed(format!("{e:?}")))
    }

    fn forward(&mut self, tokens: &[Token], session: &mut Session) -> Result<Logits, ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::EmptyInput);
        }
        let requested_total = session.consumed_tokens().saturating_add(tokens.len());
        if requested_total > self.config.context_size {
            return Err(ModelError::ContextExceeded {
                context_size: self.config.context_size,
                requested_total_tokens: requested_total,
            });
        }
        let start_pos = session.consumed_tokens();
        let window = Self::forward_window_size_for("fwd");
        let mut logits = Vec::new();
        let mut offset = 0;
        while offset < tokens.len() {
            let end = (offset + window).min(tokens.len());
            logits = self
                .forward_window(&tokens[offset..end], start_pos + offset, false)?
                .pop()
                .unwrap_or_default();
            offset = end;
        }
        session.record_tokens(tokens.len());
        Ok(logits)
    }

    fn forward_many(
        &mut self,
        tokens: &[Token],
        session: &mut Session,
    ) -> Result<Vec<Logits>, ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::EmptyInput);
        }
        let requested_total = session.consumed_tokens().saturating_add(tokens.len());
        if requested_total > self.config.context_size {
            return Err(ModelError::ContextExceeded {
                context_size: self.config.context_size,
                requested_total_tokens: requested_total,
            });
        }
        let start_pos = session.consumed_tokens();
        // forward_many is the speculative-verification entry point: checkpoint
        // the recurrent state here so a rejected draft can rewind to start_pos.
        self.push_ssm_checkpoint(start_pos);
        let window = Self::forward_window_size_for("many");
        let mut all_logits = Vec::with_capacity(tokens.len());
        let mut offset = 0;
        while offset < tokens.len() {
            let end = (offset + window).min(tokens.len());
            all_logits
                .extend(self.forward_window(&tokens[offset..end], start_pos + offset, true)?);
            offset = end;
        }
        session.record_tokens(tokens.len());
        Ok(all_logits)
    }
    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
    fn context_size(&self) -> usize {
        self.config.context_size
    }
    fn layer_count(&self) -> usize {
        self.config.layer_count
    }
}
