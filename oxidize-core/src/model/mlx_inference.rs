//! MLX-backed inference model (macOS only).
//!
//! Implements the `Model` trait using `MlxComputeBackend` for all compute
//! operations.  Weights are loaded into `MlxWeightStorage` for unified-memory
//! execution on Apple Silicon.

#[cfg(target_os = "macos")]
use crate::backends::mlx::{MlxComputeBackend, MlxTensor, MlxWeightStorage};
#[cfg(target_os = "macos")]
use crate::gguf::{GgufQuantizationType, MappedGgufFile};
#[cfg(target_os = "macos")]
use crate::inference::InferenceConfig;
#[cfg(target_os = "macos")]
use crate::model::{Logits, Model, ModelError, Session, Token};
#[cfg(target_os = "macos")]
use crate::quantization::{dequantize_scalar, quantized_size};
#[cfg(target_os = "macos")]
use crate::tensor::{apply_rope_f32, rms_norm_f32};

// ---------------------------------------------------------------------------
//  macOS-only: MlxInferenceModel
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
pub struct MlxInferenceModel {
    config: InferenceConfig,
    backend: MlxComputeBackend,
    tok_embeddings: Vec<f32>,
    tok_embeddings_cols: usize,
    norm_weight: Vec<f32>,
    output_weight: MlxWeightStorage,
    layers: Vec<MlxLayerWeights>,
    kv_cache: MlxKvCache,
    workspace: MlxWorkspace,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
struct MlxLayerWeights {
    attn_norm: Vec<f32>,
    attn_q: MlxWeightStorage,
    attn_q_bias: Vec<f32>,
    attn_k: MlxWeightStorage,
    attn_k_bias: Vec<f32>,
    attn_v: MlxWeightStorage,
    attn_v_bias: Vec<f32>,
    attn_output: MlxWeightStorage,
    attn_output_bias: Vec<f32>,
    ffn_norm: Vec<f32>,
    post_attention_norm: Vec<f32>,
    ffn_gate: MlxWeightStorage,
    ffn_up: MlxWeightStorage,
    ffn_down: MlxWeightStorage,
    ffn_down_bias: Vec<f32>,
    attn_qkv: MlxWeightStorage,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
struct MlxWorkspace {
    x: Vec<f32>,
    hidden_a: Vec<f32>,
    hidden_b: Vec<f32>,
    intermediate_a: Vec<f32>,
    intermediate_b: Vec<f32>,
    q_full: Vec<f32>,
    k_vec: Vec<f32>,
    v_vec: Vec<f32>,
    attn_result: Vec<f32>,
    head_scratch: Vec<f32>,
    logits: Vec<f32>,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
struct MlxKvCache {
    config: InferenceConfig,
    keys: Vec<f32>,
    values: Vec<f32>,
}

#[cfg(target_os = "macos")]
impl MlxKvCache {
    fn new(config: &InferenceConfig) -> Self {
        let max_kv_len = config.num_key_value_heads * config.kv_head_dim();
        let size = config.layer_count * config.context_size * max_kv_len;
        Self {
            config: config.clone(),
            keys: vec![0.0_f32; size],
            values: vec![0.0_f32; size],
        }
    }

    fn token_size(&self) -> usize {
        self.config.num_key_value_heads * self.config.kv_head_dim()
    }

    fn set(&mut self, layer: usize, position: usize, key: &[f32], value: &[f32]) {
        let token_size = self.token_size();
        let layer_offset = layer * self.config.context_size * token_size;
        let pos_offset = position * token_size;
        let start = layer_offset + pos_offset;
        self.keys[start..start + token_size].copy_from_slice(key);
        self.values[start..start + token_size].copy_from_slice(value);
    }

    fn layer_key_prefix(&self, layer: usize, seq_len: usize) -> &[f32] {
        let token_size = self.token_size();
        let layer_offset = layer * self.config.context_size * token_size;
        let end = layer_offset + seq_len * token_size;
        &self.keys[layer_offset..end]
    }

    fn layer_value_prefix(&self, layer: usize, seq_len: usize) -> &[f32] {
        let token_size = self.token_size();
        let layer_offset = layer * self.config.context_size * token_size;
        let end = layer_offset + seq_len * token_size;
        &self.values[layer_offset..end]
    }

    fn rewind_to(&mut self, position: usize) {
        let token_size = self.token_size();
        for layer in 0..self.config.layer_count {
            let layer_offset = layer * self.config.context_size * token_size;
            let start = layer_offset + (position + 1) * token_size;
            let end = layer_offset + self.config.context_size * token_size;
            self.keys[start..end].fill(0.0_f32);
            self.values[start..end].fill(0.0_f32);
        }
    }
}

#[cfg(target_os = "macos")]
impl MlxInferenceModel {
    pub fn load_from_gguf(
        mapped: &MappedGgufFile,
        config: InferenceConfig,
    ) -> Result<Self, String> {
        let backend = MlxComputeBackend::new();
        let mut tok_embeddings: Option<Vec<f32>> = None;
        let mut tok_embeddings_cols: usize = config.hidden_size;
        let mut norm_weight: Option<Vec<f32>> = None;
        let mut output_weight: Option<MlxWeightStorage> = None;
        let mut layers: Vec<MlxLayerWeights> =
            vec![MlxLayerWeights::default_weights(); config.layer_count];

        for tensor in mapped.mapped_tensor_infos().iter() {
            let qtype = GgufQuantizationType::from_ggml_type(tensor.ggml_type);
            let value_count: usize = tensor.dimensions.iter().map(|&d| d as usize).product();
            let qsize = quantized_size(qtype, value_count)
                .map_err(|e| format!("quantized_size: {:?}", e))?;
            let offset = tensor.absolute_offset as usize;
            let qdata = &mapped.bytes()[offset..offset + qsize];

            let load_weight = |_name: &str,
                               qtype: GgufQuantizationType,
                               qdata: &[u8],
                               count: usize|
             -> Result<MlxWeightStorage, String> {
                let is_supported_quant = matches!(
                    qtype,
                    GgufQuantizationType::Q8_0
                        | GgufQuantizationType::Q4_K_S
                        | GgufQuantizationType::Q4_K_M
                        | GgufQuantizationType::Q6_K
                );
                if is_supported_quant {
                    let shape: Vec<usize> =
                        tensor.dimensions.iter().map(|&d| d as usize).collect();
                    MlxWeightStorage::from_gguf_tensor_quantized(qtype, qdata, &shape, 64, 4)
                        .or_else(|_| MlxWeightStorage::from_gguf_tensor(qtype, qdata, &shape))
                } else {
                    let mut f32_data = vec![0.0_f32; count];
                    dequantize_scalar(qtype, qdata, &mut f32_data)
                        .map_err(|e| format!("dequantize: {:?}", e))?;
                    Ok(MlxWeightStorage::F32(
                        backend.tensor_from_f32_2d(&f32_data, 1, count)?.array,
                    ))
                }
            };

            let load_vec =
                |qtype: GgufQuantizationType, qdata: &[u8], count: usize| -> Result<Vec<f32>, String> {
                    let mut f32_data = vec![0.0_f32; count];
                    dequantize_scalar(qtype, qdata, &mut f32_data)
                        .map_err(|e| format!("dequantize: {:?}", e))?;
                    Ok(f32_data)
                };

            match tensor.name.as_str() {
                "tok_embeddings.weight" | "token_embd.weight" => {
                    tok_embeddings_cols = tensor
                        .dimensions
                        .get(1)
                        .copied()
                        .unwrap_or(config.hidden_size as u64)
                        as usize;
                    tok_embeddings = Some(load_vec(qtype, qdata, value_count)?);
                }
                "norm.weight" | "output_norm.weight" => {
                    norm_weight = Some(load_vec(qtype, qdata, value_count)?);
                }
                "output.weight" => {
                    output_weight = Some(load_weight("output.weight", qtype, qdata, value_count)?);
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
                    let weight_name = parts[2];
                    let suffix = parts.get(3).copied();
                    match (weight_name, suffix) {
                        ("attn_norm", _) => {
                            layers[layer_idx].attn_norm =
                                load_vec(qtype, qdata, value_count)?;
                        }
                        ("attn_q", Some("weight")) => {
                            layers[layer_idx].attn_q =
                                load_weight(name, qtype, qdata, value_count)?;
                        }
                        ("attn_q", Some("bias")) => {
                            layers[layer_idx].attn_q_bias =
                                load_vec(qtype, qdata, value_count)?;
                        }
                        ("attn_k", Some("weight")) => {
                            layers[layer_idx].attn_k =
                                load_weight(name, qtype, qdata, value_count)?;
                        }
                        ("attn_k", Some("bias")) => {
                            layers[layer_idx].attn_k_bias =
                                load_vec(qtype, qdata, value_count)?;
                        }
                        ("attn_v", Some("weight")) => {
                            layers[layer_idx].attn_v =
                                load_weight(name, qtype, qdata, value_count)?;
                        }
                        ("attn_v", Some("bias")) => {
                            layers[layer_idx].attn_v_bias =
                                load_vec(qtype, qdata, value_count)?;
                        }
                        ("attn_output", Some("weight")) => {
                            layers[layer_idx].attn_output =
                                load_weight(name, qtype, qdata, value_count)?;
                        }
                        ("attn_output", Some("bias")) => {
                            layers[layer_idx].attn_output_bias =
                                load_vec(qtype, qdata, value_count)?;
                        }
                        ("ffn_norm", _) => {
                            layers[layer_idx].ffn_norm =
                                load_vec(qtype, qdata, value_count)?;
                        }
                        ("post_attention_norm", _) => {
                            layers[layer_idx].post_attention_norm =
                                load_vec(qtype, qdata, value_count)?;
                        }
                        ("ffn_gate", _) => {
                            layers[layer_idx].ffn_gate =
                                load_weight(name, qtype, qdata, value_count)?;
                        }
                        ("ffn_up", _) => {
                            layers[layer_idx].ffn_up =
                                load_weight(name, qtype, qdata, value_count)?;
                        }
                        ("ffn_down", Some("weight")) => {
                            layers[layer_idx].ffn_down =
                                load_weight(name, qtype, qdata, value_count)?;
                        }
                        ("ffn_down", Some("bias")) => {
                            layers[layer_idx].ffn_down_bias =
                                load_vec(qtype, qdata, value_count)?;
                        }
                        ("attn_qkv", _) => {
                            layers[layer_idx].attn_qkv =
                                load_weight(name, qtype, qdata, value_count)?;
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        let tok_embeddings = tok_embeddings.ok_or("missing tok_embeddings.weight")?;
        let norm_weight = norm_weight.ok_or("missing norm.weight")?;
        let output_weight = output_weight.unwrap_or_else(|| {
            MlxWeightStorage::F32(
                backend
                    .tensor_from_f32_2d(&tok_embeddings, 1, tok_embeddings.len())
                    .unwrap()
                    .array,
            )
        });

        let kv_cache = MlxKvCache::new(&config);
        let workspace = MlxWorkspace::for_config(&config);

        Ok(Self {
            config,
            backend,
            tok_embeddings,
            tok_embeddings_cols,
            norm_weight,
            output_weight,
            layers,
            kv_cache,
            workspace,
        })
    }

    fn forward_single(
        &mut self,
        token: Token,
        pos: usize,
        need_logits: bool,
    ) -> Result<Option<Logits>, ModelError> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let n = cfg.num_attention_heads;
        let k = cfg.num_key_value_heads;
        let backend = &self.backend;
        let ws = &mut self.workspace;

        // embedding lookup
        let x = &mut ws.x[..h];
        x.fill(0.0_f32);
        let token_idx = (token as usize).min(cfg.vocab_size.saturating_sub(1));
        for (i, value) in x.iter_mut().enumerate().take(h) {
            *value = self.tok_embeddings[i * self.tok_embeddings_cols + token_idx];
        }

        for layer_idx in 0..cfg.layer_count {
            let layer = &self.layers[layer_idx];

            // Standard attention
            let attn_out = &mut ws.hidden_a[..h];
            attn_out.fill(0.0_f32);
            {
                let normed = &mut ws.hidden_b[..h];
                normed.fill(0.0_f32);
                rms_norm_f32(x, &layer.attn_norm, cfg.rms_norm_eps, normed)
                    .map_err(|e| ModelError::InferenceFailed(format!("rms_norm: {:?}", e)))?;

                let normed_tensor = backend
                    .tensor_from_f32(&normed[..h])
                    .map_err(|e| ModelError::InferenceFailed(e))?;

                let q_len = if !layer.attn_qkv.is_empty() {
                    layer.attn_qkv.shape()[0]
                } else {
                    layer.attn_q.shape()[0]
                };
                let kv_len = if !layer.attn_k.is_empty() {
                    layer.attn_k.shape()[0]
                } else {
                    0
                };
                let attn_output_input_len = if !layer.attn_output.is_empty() {
                    layer.attn_output.shape()[0]
                } else {
                    0
                };

                let q_full = &mut ws.q_full[..q_len];
                q_full.fill(0.0_f32);

                // Q projection via MLX
                if !layer.attn_qkv.is_empty() {
                    let qkv_out = backend
                        .gemv(&layer.attn_qkv, &normed_tensor, q_len, h)
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("attn_qkv: {}", e))
                        })?;
                    let copied = backend
                        .tensor_to_f32(&qkv_out, q_full)
                        .map_err(|e| ModelError::InferenceFailed(e))?;
                    q_full[copied..].fill(0.0_f32);
                } else {
                    let q_out = backend
                        .gemv(&layer.attn_q, &normed_tensor, q_len, h)
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("attn_q: {}", e))
                        })?;
                    let copied = backend
                        .tensor_to_f32(&q_out, q_full)
                        .map_err(|e| ModelError::InferenceFailed(e))?;
                    q_full[copied..].fill(0.0_f32);
                    if !layer.attn_q_bias.is_empty() {
                        for (i, q) in q_full.iter_mut().enumerate() {
                            *q += layer.attn_q_bias[i % layer.attn_q_bias.len()];
                        }
                    }
                }

                let q_len_actual = if attn_output_input_len > 0 {
                    q_len.min(attn_output_input_len)
                } else if q_len > h {
                    h
                } else {
                    q_len
                };
                let q = &mut ws.q_full[..q_len_actual];

                let k_vec = &mut ws.k_vec[..kv_len];
                k_vec.fill(0.0_f32);
                let v_vec = &mut ws.v_vec[..kv_len];
                v_vec.fill(0.0_f32);

                if !layer.attn_k.is_empty() {
                    let k_out = backend
                        .gemv(&layer.attn_k, &normed_tensor, kv_len, h)
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("attn_k: {}", e))
                        })?;
                    backend
                        .tensor_to_f32(&k_out, k_vec)
                        .map_err(|e| ModelError::InferenceFailed(e))?;
                    if !layer.attn_k_bias.is_empty() {
                        for (i, k_val) in k_vec.iter_mut().enumerate() {
                            *k_val += layer.attn_k_bias[i % layer.attn_k_bias.len()];
                        }
                    }
                }
                if !layer.attn_v.is_empty() {
                    let v_out = backend
                        .gemv(&layer.attn_v, &normed_tensor, kv_len, h)
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("attn_v: {}", e))
                        })?;
                    backend
                        .tensor_to_f32(&v_out, v_vec)
                        .map_err(|e| ModelError::InferenceFailed(e))?;
                    if !layer.attn_v_bias.is_empty() {
                        for (i, v_val) in v_vec.iter_mut().enumerate() {
                            *v_val += layer.attn_v_bias[i % layer.attn_v_bias.len()];
                        }
                    }
                }

                let q_head_dim = if n > 0 && q_len_actual.is_multiple_of(n) {
                    q_len_actual / n
                } else {
                    q_len_actual
                };
                let q_heads = q_len_actual.checked_div(q_head_dim).unwrap_or(1);

                let kv_head_dim = if k > 0 && kv_len.is_multiple_of(k) {
                    kv_len / k
                } else if kv_len > 0 {
                    kv_len
                } else {
                    q_head_dim
                };
                let kv_heads = kv_len.checked_div(kv_head_dim).unwrap_or(1);

                // Apply RoPE to Q
                for head in 0..q_heads {
                    let off = head * q_head_dim;
                    if off + q_head_dim > q.len() {
                        break;
                    }
                    let rotated = &mut ws.head_scratch[..q_head_dim];
                    rotated.fill(0.0_f32);
                    apply_rope_f32(
                        &q[off..off + q_head_dim],
                        pos,
                        q_head_dim,
                        cfg.rope_theta,
                        rotated,
                    )
                    .map_err(|e| ModelError::InferenceFailed(format!("rope q: {:?}", e)))?;
                    q[off..off + q_head_dim].copy_from_slice(rotated);
                }

                // Apply RoPE to K
                for head in 0..kv_heads {
                    let off = head * kv_head_dim;
                    if off + kv_head_dim > k_vec.len() {
                        break;
                    }
                    let rotated = &mut ws.head_scratch[..kv_head_dim];
                    rotated.fill(0.0_f32);
                    apply_rope_f32(
                        &k_vec[off..off + kv_head_dim],
                        pos,
                        kv_head_dim,
                        cfg.rope_theta,
                        rotated,
                    )
                    .map_err(|e| ModelError::InferenceFailed(format!("rope k: {:?}", e)))?;
                    k_vec[off..off + kv_head_dim].copy_from_slice(rotated);
                }

                // Store in KV cache
                self.kv_cache.set(layer_idx, pos, k_vec, v_vec);

                // Attention via MLX fast_scaled_dot_product_attention
                let seq_len = pos + 1;
                let key_cache = self.kv_cache.layer_key_prefix(layer_idx, seq_len);
                let value_cache = self.kv_cache.layer_value_prefix(layer_idx, seq_len);

                let attn_result = &mut ws.attn_result[..q_len_actual];
                attn_result.fill(0.0_f32);

                // For each head, run MLX attention
                let scale = 1.0_f32 / (kv_head_dim as f32).sqrt();
                for head in 0..q_heads.min(kv_heads) {
                    let q_off = head * q_head_dim;
                    let kv_off = head * kv_head_dim;
                    let q_slice = &q[q_off..q_off + q_head_dim.min(kv_head_dim)];

                    let q_tensor = backend
                        .tensor_from_f32(q_slice)
                        .map_err(|e| ModelError::InferenceFailed(e))?;
                    let k_tensor = backend
                        .tensor_from_f32(&key_cache[kv_off..kv_off + seq_len * kv_head_dim])
                        .map_err(|e| ModelError::InferenceFailed(e))?;
                    let v_tensor = backend
                        .tensor_from_f32(&value_cache[kv_off..kv_off + seq_len * kv_head_dim])
                        .map_err(|e| ModelError::InferenceFailed(e))?;

                    let out_tensor = backend
                        .attention_decode(
                            &q_tensor,
                            &k_tensor,
                            &v_tensor,
                            seq_len,
                            kv_head_dim,
                            scale,
                        )
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("mlx attention: {}", e))
                        })?;

                    let out_slice =
                        &mut attn_result[q_off..q_off + q_head_dim.min(kv_head_dim)];
                    backend
                        .tensor_to_f32(&out_tensor, out_slice)
                        .map_err(|e| ModelError::InferenceFailed(e))?;
                }

                // Output projection
                let attn_input = if attn_output_input_len > 0
                    && attn_result.len() != attn_output_input_len
                {
                    if attn_result.len() >= attn_output_input_len {
                        &attn_result[..attn_output_input_len]
                    } else {
                        let padded = &mut ws.q_full[..attn_output_input_len];
                        padded.fill(0.0_f32);
                        padded[..attn_result.len()].copy_from_slice(attn_result);
                        &padded[..attn_output_input_len]
                    }
                } else {
                    &attn_result[..]
                };

                if !layer.attn_output.is_empty() && attn_output_input_len > 0 {
                    let attn_tensor = backend
                        .tensor_from_f32(attn_input)
                        .map_err(|e| ModelError::InferenceFailed(e))?;
                    let out = backend
                        .gemv(&layer.attn_output, &attn_tensor, h, attn_output_input_len)
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("attn_output: {}", e))
                        })?;
                    backend
                        .tensor_to_f32(&out, attn_out)
                        .map_err(|e| ModelError::InferenceFailed(e))?;
                    if !layer.attn_output_bias.is_empty() {
                        for (i, out) in attn_out.iter_mut().enumerate() {
                            *out += layer.attn_output_bias[i % layer.attn_output_bias.len()];
                        }
                    }
                }
            }

            for i in 0..h {
                x[i] += attn_out[i];
            }

            // FFN
            let has_ffn = !layer.ffn_gate.is_empty()
                && !layer.ffn_up.is_empty()
                && !layer.ffn_down.is_empty();
            if has_ffn {
                let ffn_out = &mut ws.hidden_a[..h];
                ffn_out.fill(0.0_f32);
                {
                    let normed = &mut ws.hidden_b[..h];
                    normed.fill(0.0_f32);
                    let ffn_norm_weight = if !layer.post_attention_norm.is_empty() {
                        &layer.post_attention_norm
                    } else {
                        &layer.ffn_norm
                    };
                    rms_norm_f32(x, ffn_norm_weight, cfg.rms_norm_eps, normed)
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("ffn_norm: {:?}", e))
                        })?;

                    let normed_tensor = backend
                        .tensor_from_f32(&normed[..h])
                        .map_err(|e| ModelError::InferenceFailed(e))?;

                    let gate = &mut ws.intermediate_a[..cfg.intermediate_size];
                    gate.fill(0.0_f32);
                    let up = &mut ws.intermediate_b[..cfg.intermediate_size];
                    up.fill(0.0_f32);

                    let gate_out = backend
                        .gemv(&layer.ffn_gate, &normed_tensor, cfg.intermediate_size, h)
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("ffn_gate: {}", e))
                        })?;
                    backend
                        .tensor_to_f32(&gate_out, gate)
                        .map_err(|e| ModelError::InferenceFailed(e))?;

                    let up_out = backend
                        .gemv(&layer.ffn_up, &normed_tensor, cfg.intermediate_size, h)
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("ffn_up: {}", e))
                        })?;
                    backend
                        .tensor_to_f32(&up_out, up)
                        .map_err(|e| ModelError::InferenceFailed(e))?;

                    // SwiGLU
                    for (g, u) in gate.iter_mut().zip(up.iter()) {
                        let sigmoid = 1.0_f32 / (1.0 + (-*g).exp());
                        *g = *g * sigmoid * *u;
                    }

                    let gate_tensor = backend
                        .tensor_from_f32(gate)
                        .map_err(|e| ModelError::InferenceFailed(e))?;
                    let down_out = backend
                        .gemv(&layer.ffn_down, &gate_tensor, h, cfg.intermediate_size)
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("ffn_down: {}", e))
                        })?;
                    backend
                        .tensor_to_f32(&down_out, ffn_out)
                        .map_err(|e| ModelError::InferenceFailed(e))?;
                    if !layer.ffn_down_bias.is_empty() {
                        for (i, out) in ffn_out.iter_mut().enumerate() {
                            *out += layer.ffn_down_bias[i % layer.ffn_down_bias.len()];
                        }
                    }
                }

                for i in 0..h {
                    x[i] += ffn_out[i];
                }
            }
        }

        if !need_logits {
            return Ok(None);
        }

        let normed = &mut ws.hidden_a[..h];
        normed.fill(0.0_f32);
        rms_norm_f32(x, &self.norm_weight, cfg.rms_norm_eps, normed)
            .map_err(|e| ModelError::InferenceFailed(format!("final_norm: {:?}", e)))?;

        let logits = &mut ws.logits[..cfg.vocab_size];
        logits.fill(0.0_f32);
        let normed_tensor = backend
            .tensor_from_f32(&normed[..h])
            .map_err(|e| ModelError::InferenceFailed(e))?;
        let out = backend
            .gemv(&self.output_weight, &normed_tensor, cfg.vocab_size, h)
            .map_err(|e| ModelError::InferenceFailed(format!("output: {}", e)))?;
        backend
            .tensor_to_f32(&out, logits)
            .map_err(|e| ModelError::InferenceFailed(e))?;

        Ok(Some(logits.to_vec()))
    }
}

#[cfg(target_os = "macos")]
impl MlxLayerWeights {
    fn default_weights() -> Self {
        Self {
            attn_norm: Vec::new(),
            attn_q: MlxWeightStorage::F32(MlxTensor::from_f32(&[]).array),
            attn_q_bias: Vec::new(),
            attn_k: MlxWeightStorage::F32(MlxTensor::from_f32(&[]).array),
            attn_k_bias: Vec::new(),
            attn_v: MlxWeightStorage::F32(MlxTensor::from_f32(&[]).array),
            attn_v_bias: Vec::new(),
            attn_output: MlxWeightStorage::F32(MlxTensor::from_f32(&[]).array),
            attn_output_bias: Vec::new(),
            ffn_norm: Vec::new(),
            post_attention_norm: Vec::new(),
            ffn_gate: MlxWeightStorage::F32(MlxTensor::from_f32(&[]).array),
            ffn_up: MlxWeightStorage::F32(MlxTensor::from_f32(&[]).array),
            ffn_down: MlxWeightStorage::F32(MlxTensor::from_f32(&[]).array),
            ffn_down_bias: Vec::new(),
            attn_qkv: MlxWeightStorage::F32(MlxTensor::from_f32(&[]).array),
        }
    }
}

#[cfg(target_os = "macos")]
impl MlxWorkspace {
    fn for_config(config: &InferenceConfig) -> Self {
        let h = config.hidden_size;
        let inter = config.intermediate_size;
        let max_kv_len = config.num_key_value_heads * config.kv_head_dim();
        let max_qkv = (h * 3).max(inter);
        let head_dim = config.head_dim().max(config.kv_head_dim());

        Self {
            x: vec![0.0_f32; h],
            hidden_a: vec![0.0_f32; h],
            hidden_b: vec![0.0_f32; h],
            intermediate_a: vec![0.0_f32; inter],
            intermediate_b: vec![0.0_f32; inter],
            q_full: vec![0.0_f32; max_qkv],
            k_vec: vec![0.0_f32; max_kv_len],
            v_vec: vec![0.0_f32; max_kv_len],
            attn_result: vec![0.0_f32; h],
            head_scratch: vec![0.0_f32; head_dim],
            logits: vec![0.0_f32; config.vocab_size],
        }
    }
}

#[cfg(target_os = "macos")]
impl Model for MlxInferenceModel {
    fn rewind_to(&mut self, consumed_tokens: usize) -> Result<(), ModelError> {
        let position = if consumed_tokens == 0 {
            0
        } else {
            consumed_tokens.saturating_sub(1)
        };
        self.kv_cache.rewind_to(position);
        Ok(())
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
        let mut logits = Vec::new();
        for (i, &token) in tokens.iter().enumerate() {
            let pos = start_pos + i;
            let need_logits = i + 1 == tokens.len();
            if let Some(final_logits) = self.forward_single(token, pos, need_logits)? {
                logits = final_logits;
            }
        }
        session.record_tokens(tokens.len());
        Ok(logits)
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

// ---------------------------------------------------------------------------
//  Linux stub
// ---------------------------------------------------------------------------

#[cfg(not(target_os = "macos"))]
#[derive(Debug, Clone)]
pub struct MlxInferenceModel;

#[cfg(not(target_os = "macos"))]
impl MlxInferenceModel {
    pub fn load_from_gguf(
        _mapped: &crate::gguf::MappedGgufFile,
        _config: crate::inference::InferenceConfig,
    ) -> Result<Self, String> {
        Err("MlxInferenceModel is only available on macOS".to_string())
    }
}

#[cfg(not(target_os = "macos"))]
impl crate::model::Model for MlxInferenceModel {
    fn forward(
        &mut self,
        _tokens: &[crate::model::Token],
        _session: &mut crate::model::Session,
    ) -> Result<crate::model::Logits, crate::model::ModelError> {
        Err(crate::model::ModelError::InferenceFailed(
            "MlxInferenceModel is only available on macOS".to_string(),
        ))
    }

    fn vocab_size(&self) -> usize {
        0
    }

    fn context_size(&self) -> usize {
        0
    }

    fn layer_count(&self) -> usize {
        0
    }
}

// ---------------------------------------------------------------------------
//  Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #[test]
    fn mlx_inference_model_exists() {
        // Compilation test: MlxInferenceModel is declared and load_from_gguf
        // has the correct signature on both macOS and Linux.
        #[cfg(not(target_os = "macos"))]
        {
            let _ = super::MlxInferenceModel::load_from_gguf;
        }
    }
}
