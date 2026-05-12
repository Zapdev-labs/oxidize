use crate::gguf::{GgufQuantizationType, MappedGgufFile};
use crate::inference::{InferenceConfig, WeightStorage};
use crate::kv_cache::KvCache;
use crate::model::{Logits, Model, ModelError, Session, Token};
use crate::quantization::{dequantize_scalar, quantized_size};
use crate::tensor::{
    apply_rope_f32, apply_swiglu_f32, f16_le_to_f32, gemv_f32_transposed,
    gemv_quantized_f32_transposed, rms_norm_f32, scaled_dot_product_attention_f32,
};
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
    ssm_conv_buffers: Vec<Vec<Vec<f32>>>,
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
    attn_qkv: WeightStorage,
    attn_gate: WeightStorage,
    ssm_a: Vec<f32>,
    ssm_alpha: Vec<f32>,
    ssm_beta: Vec<f32>,
    ssm_conv1d: Vec<f32>,
    ssm_dt_bias: Vec<f32>,
    ssm_norm: Vec<f32>,
    ssm_out: WeightStorage,
    attn_q_norm: Vec<f32>,
    attn_k_norm: Vec<f32>,
}

fn weight_is_empty(ws: &WeightStorage) -> bool {
    match ws {
        WeightStorage::F32(v) => v.is_empty(),
        WeightStorage::Quantized(_, v) => v.is_empty(),
    }
}

fn weight_output_dim(ws: &WeightStorage, input_dim: usize) -> usize {
    match ws {
        WeightStorage::F32(v) => v.len() / input_dim,
        WeightStorage::Quantized(qtype, v) => {
            let (block_width, block_size) = match qtype {
                GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M => (256, 130),
                GgufQuantizationType::Q6_K => (256, 194),
                GgufQuantizationType::Q8_0 => (32, 34),
                _ => (1, 4),
            };
            let bytes_per_row = (input_dim / block_width) * block_size;
            if bytes_per_row == 0 { return 0; }
            v.len() / bytes_per_row
        }
    }
}

fn gemv_weight(storage: &WeightStorage, rows: usize, cols: usize, input: &[f32], output: &mut [f32]) -> Result<(), String> {
    match storage {
        WeightStorage::F32(data) => {
            gemv_f32_transposed(data, cols, rows, input, output).map_err(|e| format!("{:?}", e))
        }
        WeightStorage::Quantized(qtype, data) => {
            gemv_quantized_f32_transposed(*qtype, data, cols, rows, input, output)
                .map_err(|e| format!("{:?}", e))
        }
    }
}

impl LayerWiseModel {
    pub fn load_from_gguf(mapped: &MappedGgufFile, config: InferenceConfig, layer_cache_size: usize) -> Result<Self, String> {
        let mut tok_embeddings: Option<WeightStorage> = None;
        let mut tok_embeddings_cols: usize = config.hidden_size;
        let mut norm_weight: Option<Vec<f32>> = None;
        let mut output_weight: Option<WeightStorage> = None;
        let mut layer_tensors: Vec<HashMap<String, GgufTensorRef>> = vec![HashMap::new(); config.layer_count];

        let is_supported_quant_gemv = |qtype: GgufQuantizationType| {
            matches!(qtype, GgufQuantizationType::Q8_0 | GgufQuantizationType::Q4_0 | GgufQuantizationType::Q4_1 | GgufQuantizationType::Q5_0 | GgufQuantizationType::Q5_1)
        };

        for tensor in mapped.mapped_tensor_infos().iter() {
            let qtype = GgufQuantizationType::from_ggml_type(tensor.ggml_type);
            let value_count: usize = tensor.dimensions.iter().map(|&d| d as usize).product();
            let qsize = quantized_size(qtype, value_count).map_err(|e| format!("quantized_size: {:?}", e))?;
            let offset = tensor.absolute_offset as usize;

            match tensor.name.as_str() {
                "tok_embeddings.weight" | "token_embd.weight" => {
                    tok_embeddings_cols = tensor.dimensions.get(1).copied().unwrap_or(config.hidden_size as u64) as usize;
                    let qdata = &mapped.bytes()[offset..offset + qsize];
                    if is_supported_quant_gemv(qtype) {
                        tok_embeddings = Some(WeightStorage::Quantized(qtype, qdata.to_vec()));
                    } else {
                        let mut f32_data = vec![0.0_f32; value_count];
                        dequantize_scalar(qtype, qdata, &mut f32_data).map_err(|e| format!("dequantize: {:?}", e))?;
                        tok_embeddings = Some(WeightStorage::F32(f32_data));
                    }
                }
                "norm.weight" | "output_norm.weight" => {
                    let qdata = &mapped.bytes()[offset..offset + qsize];
                    let mut f32_data = vec![0.0_f32; value_count];
                    dequantize_scalar(qtype, qdata, &mut f32_data).map_err(|e| format!("dequantize: {:?}", e))?;
                    norm_weight = Some(f32_data);
                }
                "output.weight" => {
                    let qdata = &mapped.bytes()[offset..offset + qsize];
                    if is_supported_quant_gemv(qtype) {
                        output_weight = Some(WeightStorage::Quantized(qtype, qdata.to_vec()));
                    } else {
                        let mut f32_data = vec![0.0_f32; value_count];
                        dequantize_scalar(qtype, qdata, &mut f32_data).map_err(|e| format!("dequantize: {:?}", e))?;
                        output_weight = Some(WeightStorage::F32(f32_data));
                    }
                }
                name if name.starts_with("blk.") => {
                    let parts: Vec<&str> = name.split('.').collect();
                    if parts.len() < 4 { continue; }
                    let layer_idx: usize = parts[1].parse().map_err(|_| format!("bad layer index: {}", name))?;
                    if layer_idx >= config.layer_count { continue; }
                    let key = parts[2..].join(".");
                    layer_tensors[layer_idx].insert(key, GgufTensorRef { qtype, offset, size: qsize, value_count });
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
            dtype: crate::tensor::DType::F32,
        };
        let kv_cache = KvCache::new(kv_cache_config).map_err(|e| format!("kv_cache: {:?}", e))?;

        let mut ssm_states = Vec::with_capacity(config.layer_count);
        let mut ssm_conv_buffers = Vec::with_capacity(config.layer_count);
        for _ in 0..config.layer_count {
            ssm_states.push(vec![0.0_f32; 1]);
            ssm_conv_buffers.push(Vec::new());
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
            cache: LayerCache::new(layer_cache_size, config.layer_count),
        })
    }

    fn load_layer_weights(&self, layer_idx: usize) -> Result<LayerWeights, String> {
        let refs = &self.layer_tensors[layer_idx];
        let mut layer = LayerWeights::default();
        let bytes = self.mmap.bytes();
        let is_supported_quant_gemv = |qtype: GgufQuantizationType| {
            matches!(qtype, GgufQuantizationType::Q8_0 | GgufQuantizationType::Q4_0 | GgufQuantizationType::Q4_1 | GgufQuantizationType::Q5_0 | GgufQuantizationType::Q5_1)
        };

        for (key, tensor_ref) in refs.iter() {
            let qdata = &bytes[tensor_ref.offset..tensor_ref.offset + tensor_ref.size];
            let count = tensor_ref.value_count;
            let qtype = tensor_ref.qtype;
            let weight_key = key.as_str();

            if weight_key.ends_with(".weight") && !weight_key.contains("norm") && !weight_key.contains("bias") && is_supported_quant_gemv(qtype) {
                match weight_key {
                    "attn_q.weight" => layer.attn_q = WeightStorage::Quantized(qtype, qdata.to_vec()),
                    "attn_k.weight" => layer.attn_k = WeightStorage::Quantized(qtype, qdata.to_vec()),
                    "attn_v.weight" => layer.attn_v = WeightStorage::Quantized(qtype, qdata.to_vec()),
                    "attn_output.weight" => layer.attn_output = WeightStorage::Quantized(qtype, qdata.to_vec()),
                    "ffn_gate.weight" => layer.ffn_gate = WeightStorage::Quantized(qtype, qdata.to_vec()),
                    "ffn_up.weight" => layer.ffn_up = WeightStorage::Quantized(qtype, qdata.to_vec()),
                    "ffn_down.weight" => layer.ffn_down = WeightStorage::Quantized(qtype, qdata.to_vec()),
                    "attn_qkv.weight" => layer.attn_qkv = WeightStorage::Quantized(qtype, qdata.to_vec()),
                    "attn_gate.weight" => layer.attn_gate = WeightStorage::Quantized(qtype, qdata.to_vec()),
                    "ssm_out.weight" => layer.ssm_out = WeightStorage::Quantized(qtype, qdata.to_vec()),
                    _ => {}
                }
            } else {
                let mut v = vec![0.0_f32; count];
                dequantize_scalar(qtype, qdata, &mut v).map_err(|e| format!("dequantize: {:?}", e))?;
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
                    "attn_qkv.weight" => layer.attn_qkv = WeightStorage::F32(v),
                    "attn_gate.weight" => layer.attn_gate = WeightStorage::F32(v),
                    "ssm_a.weight" => layer.ssm_a = v,
                    "ssm_alpha.weight" => layer.ssm_alpha = v,
                    "ssm_beta.weight" => layer.ssm_beta = v,
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

    fn get_or_load_layer(&mut self, layer_idx: usize) -> Result<LayerWeights, String> {
        if let Some(cached) = self.cache.get(layer_idx) { return Ok(cached); }
        self.load_layer_weights(layer_idx)
    }

    fn return_layer(&mut self, layer_idx: usize, weights: LayerWeights) {
        self.cache.put(layer_idx, weights);
    }

    fn forward_single(&mut self, token: Token, pos: usize) -> Result<Logits, ModelError> {
        let cfg = &self.config.clone();
        let h = cfg.hidden_size;
        let n = cfg.num_attention_heads;
        let k = cfg.num_key_value_heads;

        let mut x = vec![0.0_f32; h];
        let token_idx = (token as usize).min(cfg.vocab_size.saturating_sub(1));
        match &self.tok_embeddings {
            WeightStorage::F32(data) => {
                for i in 0..h { x[i] = data[i * self.tok_embeddings_cols + token_idx]; }
            }
            WeightStorage::Quantized(qtype, data) => {
                let cols = self.tok_embeddings_cols;
                let (block_width, block_size, bits, zero_point) = match qtype {
                    GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M => (256, 130, 4, 8.0),
                    GgufQuantizationType::Q6_K => (256, 194, 6, 32.0),
                    GgufQuantizationType::Q8_0 => (32, 34, 8, 0.0),
                    _ => (256, 130, 4, 8.0),
                };
                let blocks_per_row = cols / block_width;
                let block_idx = token_idx / block_width;
                let pos_in_block = token_idx % block_width;
                for i in 0..h {
                    let row_start = i * blocks_per_row * block_size;
                    let block_start = row_start + block_idx * block_size;
                    let block = &data[block_start..block_start + block_size];
                    let d = f16_le_to_f32([block[0], block[1]]);
                    if *qtype == GgufQuantizationType::Q8_0 {
                        x[i] = (block[2 + pos_in_block] as i8) as f32 * d;
                    } else {
                        let q = if bits == 4 {
                            let byte = block[2 + pos_in_block / 2];
                            if pos_in_block % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F }
                        } else {
                            block[2 + pos_in_block]
                        };
                        x[i] = (q as f32 - zero_point) * d;
                    }
                }
            }
        }

        for layer_idx in 0..cfg.layer_count {
            let layer = self.get_or_load_layer(layer_idx).map_err(|e| ModelError::InferenceFailed(format!("layer load: {}", e)))?;
            let is_mamba = !weight_is_empty(&layer.attn_qkv) && layer.attn_q.is_empty();
            let ffn_norm_weight: &[f32] = if !layer.post_attention_norm.is_empty() { &layer.post_attention_norm } else if !layer.ffn_norm.is_empty() { &layer.ffn_norm } else { &[] };

            if is_mamba {
                let mamba_out = self.run_mamba_layer(layer_idx, &layer, &x, cfg)?;
                for i in 0..h { x[i] += mamba_out[i]; }
            } else if !weight_is_empty(&layer.attn_q) {
                let attn_out = self.run_attention_layer(layer_idx, &layer, &x, pos, cfg)?;
                for i in 0..h { x[i] += attn_out[i]; }
            }

            let has_ffn = !weight_is_empty(&layer.ffn_gate) && !weight_is_empty(&layer.ffn_up) && !weight_is_empty(&layer.ffn_down) && !ffn_norm_weight.is_empty();
            if has_ffn {
                let mut ffn_out = vec![0.0_f32; h];
                {
                    let mut normed = vec![0.0_f32; h];
                    rms_norm_f32(&x, ffn_norm_weight, cfg.rms_norm_eps, &mut normed).map_err(|e| ModelError::InferenceFailed(format!("ffn_norm: {:?}", e)))?;
                    let mut gate = vec![0.0_f32; cfg.intermediate_size];
                    let mut up = vec![0.0_f32; cfg.intermediate_size];
                    gemv_weight(&layer.ffn_gate, cfg.intermediate_size, h, &normed, &mut gate).map_err(|e| ModelError::InferenceFailed(format!("ffn_gate: {:?}", e)))?;
                    gemv_weight(&layer.ffn_up, cfg.intermediate_size, h, &normed, &mut up).map_err(|e| ModelError::InferenceFailed(format!("ffn_up: {:?}", e)))?;
                    let mut swiglu = vec![0.0_f32; cfg.intermediate_size];
                    apply_swiglu_f32(&gate, &up, &mut swiglu).map_err(|e| ModelError::InferenceFailed(format!("swiglu: {:?}", e)))?;
                    gemv_weight(&layer.ffn_down, h, cfg.intermediate_size, &swiglu, &mut ffn_out).map_err(|e| ModelError::InferenceFailed(format!("ffn_down: {:?}", e)))?;
                    if !layer.ffn_down_bias.is_empty() { for i in 0..ffn_out.len() { ffn_out[i] += layer.ffn_down_bias[i % layer.ffn_down_bias.len()]; } }
                }
                for i in 0..h { x[i] += ffn_out[i]; }
            }

            self.return_layer(layer_idx, layer);
        }

        let mut normed = vec![0.0_f32; h];
        rms_norm_f32(&x, &self.norm_weight, cfg.rms_norm_eps, &mut normed).map_err(|e| ModelError::InferenceFailed(format!("final_norm: {:?}", e)))?;
        let mut logits = vec![0.0_f32; cfg.vocab_size];
        gemv_weight(&self.output_weight, cfg.vocab_size, h, &normed, &mut logits).map_err(|e| ModelError::InferenceFailed(format!("output: {:?}", e)))?;
        Ok(logits)
    }

    fn run_mamba_layer(&mut self, _layer_idx: usize, layer: &LayerWeights, x: &[f32], cfg: &InferenceConfig) -> Result<Vec<f32>, ModelError> {
        let h = cfg.hidden_size;
        let mut normed = vec![0.0_f32; h];
        rms_norm_f32(x, &layer.attn_norm, cfg.rms_norm_eps, &mut normed).map_err(|e| ModelError::InferenceFailed(format!("mamba_norm: {:?}", e)))?;

        let mut gate = Vec::new();
        if !weight_is_empty(&layer.attn_gate) {
            let gate_len = weight_output_dim(&layer.attn_gate, h);
            if gate_len > 0 {
                gate = vec![0.0_f32; gate_len];
                gemv_weight(&layer.attn_gate, gate_len, h, &normed, &mut gate).map_err(|e| ModelError::InferenceFailed(format!("attn_gate: {:?}", e)))?;
            }
        }

        let qkv_out_len = weight_output_dim(&layer.attn_qkv, h);
        let mut x_proj = vec![0.0_f32; qkv_out_len];
        gemv_weight(&layer.attn_qkv, qkv_out_len, h, &normed, &mut x_proj).map_err(|e| ModelError::InferenceFailed(format!("attn_qkv: {:?}", e)))?;

        let half = qkv_out_len / 2;
        let mut mamba_out = vec![0.0_f32; half];
        let copy_len = h.min(mamba_out.len());
        mamba_out[..copy_len].copy_from_slice(&x[..copy_len]);

        if gate.len() == mamba_out.len() {
            for i in 0..mamba_out.len() { mamba_out[i] *= gate[i] * (1.0_f32 / (1.0_f32 + (-gate[i]).exp())); }
        }

        if !weight_is_empty(&layer.ssm_out) {
            let out_len = layer.ssm_out.output_dim(mamba_out.len());
            if out_len > 0 {
                let mut projected = vec![0.0_f32; out_len];
                gemv_weight(&layer.ssm_out, out_len, mamba_out.len(), &mamba_out, &mut projected).map_err(|e| ModelError::InferenceFailed(format!("ssm_out: {:?}", e)))?;
                let mut residual = vec![0.0_f32; h];
                let copy_len = h.min(projected.len());
                residual[..copy_len].copy_from_slice(&projected[..copy_len]);
                return Ok(residual);
            }
        }
        Ok(mamba_out)
    }

    fn run_attention_layer(&mut self, layer_idx: usize, layer: &LayerWeights, x: &[f32], pos: usize, cfg: &InferenceConfig) -> Result<Vec<f32>, ModelError> {
        let h = cfg.hidden_size;
        let n = cfg.num_attention_heads;
        let k = cfg.num_key_value_heads;
        let mut attn_out = vec![0.0_f32; h];

        let mut normed = vec![0.0_f32; h];
        rms_norm_f32(x, &layer.attn_norm, cfg.rms_norm_eps, &mut normed).map_err(|e| ModelError::InferenceFailed(format!("rms_norm: {:?}", e)))?;

        let q_len = weight_output_dim(&layer.attn_q, h);
        let kv_len = if !weight_is_empty(&layer.attn_k) { weight_output_dim(&layer.attn_k, h) } else { 0 };
        let attn_output_input_len = if !weight_is_empty(&layer.attn_output) { weight_output_dim(&layer.attn_output, h) } else { 0 };

        let mut q_full = vec![0.0_f32; q_len];
        gemv_weight(&layer.attn_q, q_len, h, &normed, &mut q_full).map_err(|e| ModelError::InferenceFailed(format!("attn_q: {:?}", e)))?;
        if !layer.attn_q_bias.is_empty() { for i in 0..q_full.len() { q_full[i] += layer.attn_q_bias[i % layer.attn_q_bias.len()]; } }

        let q_len_actual = if attn_output_input_len > 0 { q_len.min(attn_output_input_len) } else if q_len > h { h } else { q_len };
        let mut q = q_full[..q_len_actual].to_vec();

        let mut k_vec = vec![0.0_f32; kv_len];
        let mut v_vec = vec![0.0_f32; kv_len];
        if !weight_is_empty(&layer.attn_k) {
            gemv_weight(&layer.attn_k, kv_len, h, &normed, &mut k_vec).map_err(|e| ModelError::InferenceFailed(format!("attn_k: {:?}", e)))?;
            if !layer.attn_k_bias.is_empty() { for i in 0..k_vec.len() { k_vec[i] += layer.attn_k_bias[i % layer.attn_k_bias.len()]; } }
        }
        if !weight_is_empty(&layer.attn_v) {
            gemv_weight(&layer.attn_v, kv_len, h, &normed, &mut v_vec).map_err(|e| ModelError::InferenceFailed(format!("attn_v: {:?}", e)))?;
            if !layer.attn_v_bias.is_empty() { for i in 0..v_vec.len() { v_vec[i] += layer.attn_v_bias[i % layer.attn_v_bias.len()]; } }
        }

        let q_len_used = q.len();
        let q_head_dim = if n > 0 && q_len_used % n == 0 { q_len_used / n } else { q_len_used };
        let q_heads = q_len_used.checked_div(q_head_dim).unwrap_or(1);
        let kv_head_dim = if k > 0 && kv_len % k == 0 { kv_len / k } else if kv_len > 0 { kv_len } else { q_head_dim };
        let kv_heads = kv_len.checked_div(kv_head_dim).unwrap_or(1);

        if !layer.attn_q_norm.is_empty() && q_head_dim == layer.attn_q_norm.len() {
            for head in 0..q_heads {
                let start = head * q_head_dim;
                let end = start + q_head_dim;
                if end > q.len() { break; }
                let mut normed_head = vec![0.0_f32; q_head_dim];
                rms_norm_f32(&q[start..end], &layer.attn_q_norm, cfg.rms_norm_eps, &mut normed_head).map_err(|e| ModelError::InferenceFailed(format!("q_norm: {:?}", e)))?;
                q[start..end].copy_from_slice(&normed_head);
            }
        }
        if !layer.attn_k_norm.is_empty() && kv_head_dim == layer.attn_k_norm.len() {
            for head in 0..kv_heads {
                let start = head * kv_head_dim;
                let end = start + kv_head_dim;
                if end > k_vec.len() { break; }
                let mut normed_head = vec![0.0_f32; kv_head_dim];
                rms_norm_f32(&k_vec[start..end], &layer.attn_k_norm, cfg.rms_norm_eps, &mut normed_head).map_err(|e| ModelError::InferenceFailed(format!("k_norm: {:?}", e)))?;
                k_vec[start..end].copy_from_slice(&normed_head);
            }
        }

        for head in 0..q_heads {
            let off = head * q_head_dim;
            if off + q_head_dim > q.len() { break; }
            let mut rotated = vec![0.0_f32; q_head_dim];
            apply_rope_f32(&q[off..off + q_head_dim], pos, q_head_dim, cfg.rope_theta, &mut rotated).map_err(|e| ModelError::InferenceFailed(format!("rope q: {:?}", e)))?;
            q[off..off + q_head_dim].copy_from_slice(&rotated);
        }
        for head in 0..kv_heads {
            let off = head * kv_head_dim;
            if off + kv_head_dim > k_vec.len() { break; }
            let mut rotated = vec![0.0_f32; kv_head_dim];
            apply_rope_f32(&k_vec[off..off + kv_head_dim], pos, kv_head_dim, cfg.rope_theta, &mut rotated).map_err(|e| ModelError::InferenceFailed(format!("rope k: {:?}", e)))?;
            k_vec[off..off + kv_head_dim].copy_from_slice(&rotated);
        }

        self.kv_cache.set(layer_idx, pos, &k_vec, &v_vec).map_err(|e| ModelError::InferenceFailed(format!("kv set: {:?}", e)))?;

        let seq_len = pos + 1;
        let mut key_cache = vec![0.0_f32; seq_len * kv_len];
        let mut value_cache = vec![0.0_f32; seq_len * kv_len];
        for p in 0..seq_len {
            let k_off = p * kv_len;
            self.kv_cache.get_key(layer_idx, p, &mut key_cache[k_off..k_off + kv_len]).map_err(|e| ModelError::InferenceFailed(format!("kv get_key: {:?}", e)))?;
            self.kv_cache.get_value(layer_idx, p, &mut value_cache[k_off..k_off + kv_len]).map_err(|e| ModelError::InferenceFailed(format!("kv get_value: {:?}", e)))?;
        }

        let mut attn_result = vec![0.0_f32; q_len_actual];
        let actual_kv_group_size = q_heads.checked_div(kv_heads).filter(|g| *g > 0).unwrap_or(1);
        for head in 0..q_heads {
            let kv_head = head / actual_kv_group_size;
            let q_head_start = head * q_head_dim;
            let q_head_end = q_head_start + q_head_dim;
            if q_head_end > q.len() { break; }
            let q_head = &q[q_head_start..q_head_end];

            let mut key_seq = vec![0.0_f32; seq_len * kv_head_dim];
            let mut value_seq = vec![0.0_f32; seq_len * kv_head_dim];
            for p in 0..seq_len {
                let src_off = p * kv_len + kv_head * kv_head_dim;
                let dst_off = p * kv_head_dim;
                if src_off + kv_head_dim <= key_cache.len() {
                    key_seq[dst_off..dst_off + kv_head_dim].copy_from_slice(&key_cache[src_off..src_off + kv_head_dim]);
                    value_seq[dst_off..dst_off + kv_head_dim].copy_from_slice(&value_cache[src_off..src_off + kv_head_dim]);
                }
            }

            let q_head_for_attn = if q_head_dim > kv_head_dim { &q_head[..kv_head_dim] } else { q_head };
            let mut out_head = vec![0.0_f32; kv_head_dim];
            scaled_dot_product_attention_f32(q_head_for_attn, &key_seq, &value_seq, seq_len, kv_head_dim, &mut out_head).map_err(|e| ModelError::InferenceFailed(format!("attention: {:?}", e)))?;

            let write_start = head * kv_head_dim;
            if write_start + out_head.len() <= attn_result.len() {
                attn_result[write_start..write_start + out_head.len()].copy_from_slice(&out_head);
            }
        }

        let attn_input = if attn_output_input_len > 0 && attn_result.len() != attn_output_input_len {
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

        if !weight_is_empty(&layer.attn_output) && attn_output_input_len > 0 {
            gemv_weight(&layer.attn_output, h, attn_output_input_len, &attn_input, &mut attn_out).map_err(|e| ModelError::InferenceFailed(format!("attn_output: {:?}", e)))?;
            if !layer.attn_output_bias.is_empty() { for i in 0..attn_out.len() { attn_out[i] += layer.attn_output_bias[i % layer.attn_output_bias.len()]; } }
        }

        Ok(attn_out)
    }
}

impl Model for LayerWiseModel {
    fn forward(&mut self, tokens: &[Token], session: &mut Session) -> Result<Logits, ModelError> {
        if tokens.is_empty() { return Err(ModelError::EmptyInput); }
        let requested_total = session.consumed_tokens().saturating_add(tokens.len());
        if requested_total > self.config.context_size {
            return Err(ModelError::ContextExceeded { context_size: self.config.context_size, requested_total_tokens: requested_total });
        }
        let start_pos = session.consumed_tokens();
        let mut logits = Vec::new();
        for (i, &token) in tokens.iter().enumerate() {
            let pos = start_pos + i;
            logits = self.forward_single(token, pos)?;
        }
        session.record_tokens(tokens.len());
        Ok(logits)
    }
    fn vocab_size(&self) -> usize { self.config.vocab_size }
    fn context_size(&self) -> usize { self.config.context_size }
    fn layer_count(&self) -> usize { self.config.layer_count }
}
