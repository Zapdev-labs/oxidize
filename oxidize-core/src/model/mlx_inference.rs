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
use crate::inference::{InferenceConfig, ModelArchitecture};
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
    /// Precomputed Alibi slopes [num_heads], constant per model.
    alibi_slopes: Vec<f32>,
}

#[cfg(target_os = "macos")]
impl MlxInferenceModel {
    /// Access the model's inference configuration.
    pub fn config(&self) -> &InferenceConfig {
        &self.config
    }
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
    // --- Architecture-specific fields ---
    // Mixtral MoE: router gate + per-expert weights
    moe_gate: MlxWeightStorage,
    moe_ffn_gate: Vec<MlxWeightStorage>,
    moe_ffn_up: Vec<MlxWeightStorage>,
    moe_ffn_down: Vec<MlxWeightStorage>,
    // DeepSeek MLA: compressed latent projection weights
    mla_latent: MlxWeightStorage,
    mla_q_up: MlxWeightStorage,
    mla_kv_up: MlxWeightStorage,
    mla_out: MlxWeightStorage,
    // Qwen sliding window: nothing extra, driven by config.sliding_window
    // Gemma/Phi parallel attention/FFN: nothing extra, driven by dispatch
    // Falcon/GPT Alibi: nothing extra, driven by dispatch
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
    // Architecture-specific scratch
    /// MoE expert gate scores [num_experts]
    moe_scores: Vec<f32>,
    /// MLA latent vector [latent_dim]
    mla_latent: Vec<f32>,
    /// Alibi slope buffer [num_heads]
    alibi_slopes: Vec<f32>,
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
        mut config: InferenceConfig,
    ) -> Result<Self, String> {
        let backend = MlxComputeBackend::new();

        // Architecture detection from GGUF metadata
        config.architecture = ModelArchitecture::from_gguf(mapped);
        if config.alibi_num_heads == 0 {
            config.alibi_num_heads = config.num_attention_heads;
        }

        let metadata = &mapped.parsed().metadata;
        if config.sliding_window == 0 {
            config.sliding_window =
                metadata_u32(metadata, "llama.attention.sliding_window").unwrap_or(0) as usize;
        }
        if config.num_experts == 0 {
            config.num_experts = metadata_u32(metadata, "llama.expert_count").unwrap_or(0) as usize;
        }
        if config.num_experts_per_tok == 0 {
            config.num_experts_per_tok =
                metadata_u32(metadata, "llama.expert_used_count").unwrap_or(0) as usize;
        }

        let metadata_u32 = |metadata: &std::collections::BTreeMap<
            String,
            crate::gguf::GgufMetadataValue,
        >,
                            key: &str|
         -> Option<u32> {
            match metadata.get(key) {
                Some(crate::gguf::GgufMetadataValue::Uint8(v)) => Some((*v).into()),
                Some(crate::gguf::GgufMetadataValue::Uint16(v)) => Some((*v).into()),
                Some(crate::gguf::GgufMetadataValue::Uint32(v)) => Some(*v),
                Some(crate::gguf::GgufMetadataValue::Uint64(v)) => (*v).try_into().ok(),
                Some(crate::gguf::GgufMetadataValue::Int8(v)) if *v >= 0 => Some((*v as u8).into()),
                Some(crate::gguf::GgufMetadataValue::Int16(v)) if *v >= 0 => {
                    Some((*v as u16).into())
                }
                Some(crate::gguf::GgufMetadataValue::Int32(v)) if *v >= 0 => (*v).try_into().ok(),
                Some(crate::gguf::GgufMetadataValue::Int64(v)) if *v >= 0 => (*v).try_into().ok(),
                _ => None,
            }
        };

        let is_moe = config.architecture.uses_moe();
        let uses_mla = config.architecture.uses_mla();
        let _ = is_moe;
        let _ = uses_mla;

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
            let qdata = mapped.tensor_bytes(tensor, qsize);

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
                    let shape: Vec<usize> = tensor.dimensions.iter().map(|&d| d as usize).collect();
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

            let load_vec = |qtype: GgufQuantizationType,
                            qdata: &[u8],
                            count: usize|
             -> Result<Vec<f32>, String> {
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
                            layers[layer_idx].attn_norm = load_vec(qtype, qdata, value_count)?;
                        }
                        ("attn_q", Some("weight")) => {
                            layers[layer_idx].attn_q =
                                load_weight(name, qtype, qdata, value_count)?;
                        }
                        ("attn_q", Some("bias")) => {
                            layers[layer_idx].attn_q_bias = load_vec(qtype, qdata, value_count)?;
                        }
                        ("attn_k", Some("weight")) => {
                            layers[layer_idx].attn_k =
                                load_weight(name, qtype, qdata, value_count)?;
                        }
                        ("attn_k", Some("bias")) => {
                            layers[layer_idx].attn_k_bias = load_vec(qtype, qdata, value_count)?;
                        }
                        ("attn_v", Some("weight")) => {
                            layers[layer_idx].attn_v =
                                load_weight(name, qtype, qdata, value_count)?;
                        }
                        ("attn_v", Some("bias")) => {
                            layers[layer_idx].attn_v_bias = load_vec(qtype, qdata, value_count)?;
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
                            layers[layer_idx].ffn_norm = load_vec(qtype, qdata, value_count)?;
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
                            layers[layer_idx].ffn_down_bias = load_vec(qtype, qdata, value_count)?;
                        }
                        ("attn_qkv", _) => {
                            layers[layer_idx].attn_qkv =
                                load_weight(name, qtype, qdata, value_count)?;
                        }
                        // Mixtral MoE router gate
                        ("ffn_gate_inp", Some("weight")) => {
                            layers[layer_idx].moe_gate =
                                load_weight(name, qtype, qdata, value_count)?;
                        }
                        // Mixtral MoE per-expert weights
                        ("ffn_gate", Some(expert_and_weight))
                            if expert_and_weight
                                .split_once('.')
                                .and_then(|(prefix, _)| prefix.parse::<usize>().ok())
                                .is_some() =>
                        {
                            let expert_idx = expert_and_weight
                                .split_once('.')
                                .unwrap()
                                .0
                                .parse::<usize>()
                                .unwrap();
                            while layers[layer_idx].moe_ffn_gate.len() <= expert_idx {
                                layers[layer_idx]
                                    .moe_ffn_gate
                                    .push(MlxWeightStorage::F32(MlxTensor::from_f32(&[]).array));
                            }
                            layers[layer_idx].moe_ffn_gate[expert_idx] =
                                load_weight(name, qtype, qdata, value_count)?;
                        }
                        ("ffn_up", Some(expert_and_weight))
                            if expert_and_weight
                                .split_once('.')
                                .and_then(|(prefix, _)| prefix.parse::<usize>().ok())
                                .is_some() =>
                        {
                            let expert_idx = expert_and_weight
                                .split_once('.')
                                .unwrap()
                                .0
                                .parse::<usize>()
                                .unwrap();
                            while layers[layer_idx].moe_ffn_up.len() <= expert_idx {
                                layers[layer_idx]
                                    .moe_ffn_up
                                    .push(MlxWeightStorage::F32(MlxTensor::from_f32(&[]).array));
                            }
                            layers[layer_idx].moe_ffn_up[expert_idx] =
                                load_weight(name, qtype, qdata, value_count)?;
                        }
                        ("ffn_down", Some(expert_and_weight))
                            if expert_and_weight
                                .split_once('.')
                                .and_then(|(prefix, _)| prefix.parse::<usize>().ok())
                                .is_some() =>
                        {
                            let expert_idx = expert_and_weight
                                .split_once('.')
                                .unwrap()
                                .0
                                .parse::<usize>()
                                .unwrap();
                            while layers[layer_idx].moe_ffn_down.len() <= expert_idx {
                                layers[layer_idx]
                                    .moe_ffn_down
                                    .push(MlxWeightStorage::F32(MlxTensor::from_f32(&[]).array));
                            }
                            layers[layer_idx].moe_ffn_down[expert_idx] =
                                load_weight(name, qtype, qdata, value_count)?;
                        }
                        // DeepSeek MLA compressed weights
                        ("attn_q_a", _) => {
                            layers[layer_idx].mla_latent =
                                load_weight(name, qtype, qdata, value_count)?;
                        }
                        ("attn_q_b", _) => {
                            layers[layer_idx].mla_q_up =
                                load_weight(name, qtype, qdata, value_count)?;
                        }
                        ("attn_kv_a_mqa", _) => {
                            layers[layer_idx].mla_latent =
                                load_weight(name, qtype, qdata, value_count)?;
                        }
                        ("attn_kv_b", _) => {
                            layers[layer_idx].mla_kv_up =
                                load_weight(name, qtype, qdata, value_count)?;
                        }
                        ("attn_out_mla", _) => {
                            layers[layer_idx].mla_out =
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

        // Precompute Alibi slopes once at load time if needed.
        let alibi_slopes = if config.architecture.uses_alibi() {
            let alibi_num = n.min(config.alibi_num_heads);
            let base: f32 = 2.0_f32.powf(-(8.0_f32 / alibi_num as f32));
            (0..alibi_num)
                .map(|h| -(base.powf(h as f32 + 1.0)))
                .collect()
        } else {
            Vec::new()
        };

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
            alibi_slopes,
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

            let uses_parallel = cfg.architecture.uses_parallel_attn_ffn();
            let uses_mla = cfg.architecture.uses_mla();

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

                // --- DeepSeek MLA compressed attention path ---
                if uses_mla && !layer.mla_latent.is_empty() {
                    let latent_dim = layer.mla_latent.shape()[0].max(1);
                    let q_out_dim = if !layer.mla_q_up.is_empty() {
                        layer.mla_q_up.shape()[0]
                    } else {
                        n * cfg.head_dim()
                    };
                    let kv_out_dim = if !layer.mla_kv_up.is_empty() {
                        layer.mla_kv_up.shape()[0]
                    } else {
                        k * cfg.kv_head_dim()
                    };
                    let (q_host, k_host, v_host) = backend
                        .mla_project_qkv(
                            &normed_tensor,
                            &layer.mla_latent,
                            &layer.mla_q_up,
                            &layer.mla_kv_up,
                            latent_dim,
                            q_out_dim,
                            kv_out_dim,
                            h,
                        )
                        .map_err(|e| ModelError::InferenceFailed(format!("mla_qkv: {}", e)))?;

                    let q_len_actual = q_host.len();
                    let kv_len = k_host.len();
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

                    let q = &mut ws.q_full[..q_len_actual];
                    q.copy_from_slice(&q_host);
                    let k_vec = &mut ws.k_vec[..kv_len];
                    k_vec.copy_from_slice(&k_host);
                    let v_vec = &mut ws.v_vec[..kv_len];
                    v_vec.copy_from_slice(&v_host);

                    // Apply RoPE to Q and K (MLA still uses RoPE on expanded heads)
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
                        .map_err(|e| ModelError::InferenceFailed(format!("mla rope q: {:?}", e)))?;
                        q[off..off + q_head_dim].copy_from_slice(rotated);
                    }
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
                        .map_err(|e| ModelError::InferenceFailed(format!("mla rope k: {:?}", e)))?;
                        k_vec[off..off + kv_head_dim].copy_from_slice(rotated);
                    }

                    self.kv_cache.set(layer_idx, pos, k_vec, v_vec);

                    let seq_len = pos + 1;
                    let key_cache = self.kv_cache.layer_key_prefix(layer_idx, seq_len);
                    let value_cache = self.kv_cache.layer_value_prefix(layer_idx, seq_len);
                    let attn_result = &mut ws.attn_result[..q_len_actual];
                    attn_result.fill(0.0_f32);
                    let scale = 1.0_f32 / (kv_head_dim as f32).sqrt();
                    for head in 0..q_heads.min(kv_heads) {
                        let q_off = head * q_head_dim;
                        let kv_off = head * kv_head_dim;
                        let q_slice = &q[q_off..q_off + q_head_dim.min(kv_head_dim)];
                        let q_tensor = backend.tensor_from_f32(q_slice).map_err(|e| {
                            ModelError::InferenceFailed(format!("mla q tensor: {}", e))
                        })?;
                        let k_tensor = backend
                            .tensor_from_f32(&key_cache[kv_off..kv_off + seq_len * kv_head_dim])
                            .map_err(|e| {
                                ModelError::InferenceFailed(format!("mla k tensor: {}", e))
                            })?;
                        let v_tensor = backend
                            .tensor_from_f32(&value_cache[kv_off..kv_off + seq_len * kv_head_dim])
                            .map_err(|e| {
                                ModelError::InferenceFailed(format!("mla v tensor: {}", e))
                            })?;
                        let out_tensor = if cfg.sliding_window > 0 {
                            backend
                                .sliding_window_attention_decode(
                                    &q_tensor,
                                    &k_tensor,
                                    &v_tensor,
                                    seq_len,
                                    kv_head_dim,
                                    scale,
                                    cfg.sliding_window,
                                )
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!(
                                        "mla sliding attention: {}",
                                        e
                                    ))
                                })?
                        } else {
                            backend
                                .attention_decode(
                                    &q_tensor,
                                    &k_tensor,
                                    &v_tensor,
                                    seq_len,
                                    kv_head_dim,
                                    scale,
                                )
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!("mla attention: {}", e))
                                })?
                        };
                        let out_slice =
                            &mut attn_result[q_off..q_off + q_head_dim.min(kv_head_dim)];
                        backend.tensor_to_f32(&out_tensor, out_slice).map_err(|e| {
                            ModelError::InferenceFailed(format!("mla attn to host: {}", e))
                        })?;
                    }

                    let attn_input = if !layer.mla_out.is_empty() {
                        let mla_out_dim = layer.mla_out.shape()[0];
                        if attn_result.len() >= mla_out_dim {
                            &attn_result[..mla_out_dim]
                        } else {
                            let padded = &mut ws.q_full[..mla_out_dim];
                            padded.fill(0.0_f32);
                            padded[..attn_result.len()].copy_from_slice(attn_result);
                            &padded[..mla_out_dim]
                        }
                    } else {
                        &attn_result[..]
                    };

                    if !layer.mla_out.is_empty() {
                        let attn_tensor = backend
                            .tensor_from_f32(attn_input)
                            .map_err(|e| ModelError::InferenceFailed(e))?;
                        let out = backend
                            .gemv(&layer.mla_out, &attn_tensor, h, attn_input.len())
                            .map_err(|e| ModelError::InferenceFailed(format!("mla_out: {}", e)))?;
                        backend
                            .tensor_to_f32(&out, attn_out)
                            .map_err(|e| ModelError::InferenceFailed(e))?;
                    } else if !layer.attn_output.is_empty() {
                        let attn_tensor = backend
                            .tensor_from_f32(attn_input)
                            .map_err(|e| ModelError::InferenceFailed(e))?;
                        let out = backend
                            .gemv(&layer.attn_output, &attn_tensor, h, attn_input.len())
                            .map_err(|e| {
                                ModelError::InferenceFailed(format!("attn_output: {}", e))
                            })?;
                        backend
                            .tensor_to_f32(&out, attn_out)
                            .map_err(|e| ModelError::InferenceFailed(e))?;
                    }
                } else {
                    // --- Standard attention path (Llama / Mistral / Qwen / Falcon / GPT / Gemma / Phi) ---
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

                    if !layer.attn_qkv.is_empty() {
                        let qkv_out = backend
                            .gemv(&layer.attn_qkv, &normed_tensor, q_len, h)
                            .map_err(|e| ModelError::InferenceFailed(format!("attn_qkv: {}", e)))?;
                        let copied = backend
                            .tensor_to_f32(&qkv_out, q_full)
                            .map_err(|e| ModelError::InferenceFailed(e))?;
                        q_full[copied..].fill(0.0_f32);
                    } else {
                        let q_out = backend
                            .gemv(&layer.attn_q, &normed_tensor, q_len, h)
                            .map_err(|e| ModelError::InferenceFailed(format!("attn_q: {}", e)))?;
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
                            .map_err(|e| ModelError::InferenceFailed(format!("attn_k: {}", e)))?;
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
                            .map_err(|e| ModelError::InferenceFailed(format!("attn_v: {}", e)))?;
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

                    if cfg.architecture.uses_alibi() {
                        // Alibi: no RoPE; copy precomputed slopes into workspace.
                        let alibi_num = self.alibi_slopes.len();
                        if alibi_num > 0 {
                            let copy_len = alibi_num.min(ws.alibi_slopes.len());
                            ws.alibi_slopes[..copy_len]
                                .copy_from_slice(&self.alibi_slopes[..copy_len]);
                        }
                        // Q and K remain unrotated.
                    } else {
                        // RoPE for non-Alibi architectures
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
                    }

                    self.kv_cache.set(layer_idx, pos, k_vec, v_vec);

                    let seq_len = pos + 1;
                    let key_cache = self.kv_cache.layer_key_prefix(layer_idx, seq_len);
                    let value_cache = self.kv_cache.layer_value_prefix(layer_idx, seq_len);

                    let attn_result = &mut ws.attn_result[..q_len_actual];
                    attn_result.fill(0.0_f32);

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

                        let out_tensor = if cfg.sliding_window > 0 {
                            backend
                                .sliding_window_attention_decode(
                                    &q_tensor,
                                    &k_tensor,
                                    &v_tensor,
                                    seq_len,
                                    kv_head_dim,
                                    scale,
                                    cfg.sliding_window,
                                )
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!(
                                        "sliding_window attention: {}",
                                        e
                                    ))
                                })?
                        } else {
                            backend
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
                                })?
                        };

                        let out_slice =
                            &mut attn_result[q_off..q_off + q_head_dim.min(kv_head_dim)];
                        backend
                            .tensor_to_f32(&out_tensor, out_slice)
                            .map_err(|e| ModelError::InferenceFailed(e))?;
                    }

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
            }

            // --- Parallel attention + FFN for Gemma / Phi ---
            if uses_parallel {
                let ffn_out = &mut ws.hidden_b[..h];
                ffn_out.fill(0.0_f32);
                {
                    let normed = &mut ws.intermediate_a[..h];
                    normed.fill(0.0_f32);
                    let ffn_norm_weight = if !layer.post_attention_norm.is_empty() {
                        &layer.post_attention_norm
                    } else {
                        &layer.ffn_norm
                    };
                    rms_norm_f32(x, ffn_norm_weight, cfg.rms_norm_eps, normed).map_err(|e| {
                        ModelError::InferenceFailed(format!("parallel ffn_norm: {:?}", e))
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
                            ModelError::InferenceFailed(format!("parallel ffn_gate: {}", e))
                        })?;
                    backend
                        .tensor_to_f32(&gate_out, gate)
                        .map_err(|e| ModelError::InferenceFailed(e))?;

                    let up_out = backend
                        .gemv(&layer.ffn_up, &normed_tensor, cfg.intermediate_size, h)
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("parallel ffn_up: {}", e))
                        })?;
                    backend
                        .tensor_to_f32(&up_out, up)
                        .map_err(|e| ModelError::InferenceFailed(e))?;

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
                            ModelError::InferenceFailed(format!("parallel ffn_down: {}", e))
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
                // Parallel: add attention + ffn simultaneously to x
                for i in 0..h {
                    x[i] += attn_out[i] + ffn_out[i];
                }
            } else {
                // Standard sequential residual
                for i in 0..h {
                    x[i] += attn_out[i];
                }

                // --- Mixtral MoE FFN ---
                if cfg.architecture.uses_moe() && !layer.moe_gate.is_empty() && cfg.num_experts > 0
                {
                    let moe_out = &mut ws.hidden_a[..h];
                    moe_out.fill(0.0_f32);
                    {
                        let normed_tensor = backend
                            .tensor_from_f32(&ws.hidden_b[..h])
                            .map_err(|e| ModelError::InferenceFailed(e))?;
                        let (experts, weights) = backend
                            .moe_topk(
                                &normed_tensor,
                                &layer.moe_gate,
                                cfg.num_experts,
                                cfg.num_experts_per_tok.max(1),
                            )
                            .map_err(|e| ModelError::InferenceFailed(format!("moe_topk: {}", e)))?;

                        let mut expert_outputs: Vec<Vec<f32>> = Vec::new();
                        for &expert_idx in &experts {
                            if expert_idx >= layer.moe_ffn_gate.len()
                                || expert_idx >= layer.moe_ffn_up.len()
                                || expert_idx >= layer.moe_ffn_down.len()
                            {
                                continue;
                            }
                            let gate = &mut ws.intermediate_a[..cfg.intermediate_size];
                            gate.fill(0.0_f32);
                            let up = &mut ws.intermediate_b[..cfg.intermediate_size];
                            up.fill(0.0_f32);

                            let gate_out = backend
                                .gemv(
                                    &layer.moe_ffn_gate[expert_idx],
                                    &normed_tensor,
                                    cfg.intermediate_size,
                                    h,
                                )
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!(
                                        "moe expert {} gate: {}",
                                        expert_idx, e
                                    ))
                                })?;
                            backend
                                .tensor_to_f32(&gate_out, gate)
                                .map_err(|e| ModelError::InferenceFailed(e))?;

                            let up_out = backend
                                .gemv(
                                    &layer.moe_ffn_up[expert_idx],
                                    &normed_tensor,
                                    cfg.intermediate_size,
                                    h,
                                )
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!(
                                        "moe expert {} up: {}",
                                        expert_idx, e
                                    ))
                                })?;
                            backend
                                .tensor_to_f32(&up_out, up)
                                .map_err(|e| ModelError::InferenceFailed(e))?;

                            for (g, u) in gate.iter_mut().zip(up.iter()) {
                                let sigmoid = 1.0_f32 / (1.0 + (-*g).exp());
                                *g = *g * sigmoid * *u;
                            }

                            let gate_tensor = backend
                                .tensor_from_f32(gate)
                                .map_err(|e| ModelError::InferenceFailed(e))?;
                            let down_out = backend
                                .gemv(
                                    &layer.moe_ffn_down[expert_idx],
                                    &gate_tensor,
                                    h,
                                    cfg.intermediate_size,
                                )
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!(
                                        "moe expert {} down: {}",
                                        expert_idx, e
                                    ))
                                })?;
                            let mut expert_out = vec![0.0_f32; h];
                            backend
                                .tensor_to_f32(&down_out, &mut expert_out)
                                .map_err(|e| ModelError::InferenceFailed(e))?;
                            expert_outputs.push(expert_out);
                        }

                        moe_out.fill(0.0_f32);
                        for (expert_out, &w) in expert_outputs.iter().zip(weights.iter()) {
                            for (m, &e) in moe_out.iter_mut().zip(expert_out.iter()) {
                                *m += e * w;
                            }
                        }
                    }
                    for i in 0..h {
                        x[i] += moe_out[i];
                    }
                } else {
                    // Dense FFN (standard)
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
                            rms_norm_f32(x, ffn_norm_weight, cfg.rms_norm_eps, normed).map_err(
                                |e| ModelError::InferenceFailed(format!("ffn_norm: {:?}", e)),
                            )?;

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
            moe_gate: MlxWeightStorage::F32(MlxTensor::from_f32(&[]).array),
            moe_ffn_gate: Vec::new(),
            moe_ffn_up: Vec::new(),
            moe_ffn_down: Vec::new(),
            mla_latent: MlxWeightStorage::F32(MlxTensor::from_f32(&[]).array),
            mla_q_up: MlxWeightStorage::F32(MlxTensor::from_f32(&[]).array),
            mla_kv_up: MlxWeightStorage::F32(MlxTensor::from_f32(&[]).array),
            mla_out: MlxWeightStorage::F32(MlxTensor::from_f32(&[]).array),
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
            moe_scores: vec![0.0_f32; config.num_experts.max(1)],
            mla_latent: vec![0.0_f32; h.max(1)],
            alibi_slopes: vec![0.0_f32; config.alibi_num_heads.max(1)],
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
pub struct MlxInferenceModel(crate::inference::InferenceConfig);

#[cfg(not(target_os = "macos"))]
impl MlxInferenceModel {
    pub fn load_from_gguf(
        _mapped: &crate::gguf::MappedGgufFile,
        config: crate::inference::InferenceConfig,
    ) -> Result<Self, String> {
        // We never reach the Ok path on non-macOS, but store config for consistency.
        Ok(Self(config))
    }

    /// Access the model's inference configuration (stub on non-macOS).
    pub fn config(&self) -> &crate::inference::InferenceConfig {
        &self.0
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
    use crate::inference::ModelArchitecture;

    #[test]
    fn mlx_inference_model_exists() {
        // Compilation test: MlxInferenceModel is declared and load_from_gguf
        // has the correct signature on both macOS and Linux.
        #[cfg(not(target_os = "macos"))]
        {
            let _ = super::MlxInferenceModel::load_from_gguf;
        }
    }

    #[test]
    fn model_architecture_detects_all_families_from_gguf() {
        use crate::gguf::{GgufFile, GgufMetadataValue, GgufTensorInfo, MappedGgufFile};
        use std::collections::BTreeMap;

        fn file_with_arch(arch: &str) -> MappedGgufFile {
            let parsed = GgufFile {
                version: 3,
                tensor_count: 1,
                metadata: BTreeMap::from([(
                    "general.architecture".to_owned(),
                    GgufMetadataValue::String(arch.to_owned()),
                )]),
                tensor_infos: vec![GgufTensorInfo {
                    name: "tok_embeddings.weight".to_owned(),
                    dimensions: vec![4096, 32000],
                    ggml_type: 0,
                    relative_offset: 0,
                    absolute_offset: 0,
                    mmap_index: 0,
                }],
                alignment: 32,
                data_section_start: 0,
            };
            MappedGgufFile::from_parsed_for_test(parsed)
        }

        let archs = vec![
            ("llama", ModelArchitecture::Llama),
            ("mistral", ModelArchitecture::Mistral),
            ("mixtral", ModelArchitecture::Mixtral),
            ("deepseek", ModelArchitecture::DeepSeek),
            ("qwen", ModelArchitecture::Qwen),
            ("gemma", ModelArchitecture::Gemma),
            ("phi", ModelArchitecture::Phi),
            ("falcon", ModelArchitecture::Falcon),
            ("gpt2", ModelArchitecture::Gpt2),
            ("gptj", ModelArchitecture::GptJ),
            ("gptneox", ModelArchitecture::GptNeoX),
        ];
        for (name, expected) in archs {
            let mapped = file_with_arch(name);
            let detected = ModelArchitecture::from_gguf(&mapped);
            assert_eq!(
                detected, expected,
                "architecture detection mismatch for '{name}'"
            );
        }
    }

    #[test]
    fn architecture_flags_are_consistent() {
        assert!(ModelArchitecture::Falcon.uses_alibi());
        assert!(ModelArchitecture::Gpt2.uses_alibi());
        assert!(!ModelArchitecture::Llama.uses_alibi());
        assert!(!ModelArchitecture::Mixtral.uses_alibi());

        assert!(ModelArchitecture::Mixtral.uses_moe());
        assert!(!ModelArchitecture::Llama.uses_moe());

        assert!(ModelArchitecture::DeepSeek.uses_mla());
        assert!(!ModelArchitecture::Llama.uses_mla());

        assert!(ModelArchitecture::Qwen.uses_sliding_window());
        assert!(ModelArchitecture::Mistral.uses_sliding_window());
        assert!(!ModelArchitecture::Llama.uses_sliding_window());

        assert!(ModelArchitecture::Gemma.uses_parallel_attn_ffn());
        assert!(ModelArchitecture::Phi.uses_parallel_attn_ffn());
        assert!(!ModelArchitecture::Llama.uses_parallel_attn_ffn());
    }

    #[cfg(target_os = "macos")]
    mod macos_architecture_tests {
        use super::super::{
            MlxComputeBackend, MlxInferenceModel, MlxKvCache, MlxLayerWeights, MlxTensor,
            MlxWeightStorage, MlxWorkspace,
        };
        use super::ModelArchitecture;
        use crate::inference::InferenceConfig;
        use crate::model::{Model, Session};

        fn tiny_mlx_model_config(arch: ModelArchitecture) -> InferenceConfig {
            InferenceConfig {
                vocab_size: 4,
                context_size: 8,
                layer_count: 1,
                hidden_size: 4,
                intermediate_size: 8,
                num_attention_heads: 2,
                num_key_value_heads: 1,
                key_value_head_dim: 2,
                kv_cache_dtype: crate::tensor::DType::F32,
                rms_norm_eps: 1e-6,
                rope_theta: 10_000.0,
                architecture: arch,
                sliding_window: if arch.uses_sliding_window() { 4 } else { 0 },
                num_experts: if arch.uses_moe() { 2 } else { 0 },
                num_experts_per_tok: if arch.uses_moe() { 1 } else { 0 },
                alibi_num_heads: 2,
                ..InferenceConfig::default()
            }
        }

        #[test]
        fn tiny_llama_forward_runs() {
            let cfg = tiny_mlx_model_config(ModelArchitecture::Llama);
            let mut model = build_tiny_mlx_model(cfg);
            let mut session = Session::new();
            let logits = model
                .forward(&[1], &mut session)
                .expect("llama forward should succeed");
            assert_eq!(logits.len(), 4);
        }

        #[test]
        fn tiny_mixtral_moe_forward_runs() {
            let cfg = tiny_mlx_model_config(ModelArchitecture::Mixtral);
            let mut model = build_tiny_mlx_model(cfg);
            let mut session = Session::new();
            let logits = model
                .forward(&[1], &mut session)
                .expect("mixtral forward should succeed");
            assert_eq!(logits.len(), 4);
        }

        #[test]
        fn tiny_deepseek_mla_forward_runs() {
            let cfg = tiny_mlx_model_config(ModelArchitecture::DeepSeek);
            let mut model = build_tiny_mlx_model(cfg);
            let mut session = Session::new();
            let logits = model
                .forward(&[1], &mut session)
                .expect("deepseek forward should succeed");
            assert_eq!(logits.len(), 4);
        }

        #[test]
        fn tiny_falcon_alibi_forward_runs() {
            let cfg = tiny_mlx_model_config(ModelArchitecture::Falcon);
            let mut model = build_tiny_mlx_model(cfg);
            let mut session = Session::new();
            let logits = model
                .forward(&[1], &mut session)
                .expect("falcon forward should succeed");
            assert_eq!(logits.len(), 4);
        }

        #[test]
        fn tiny_qwen_sliding_window_forward_runs() {
            let cfg = tiny_mlx_model_config(ModelArchitecture::Qwen);
            let mut model = build_tiny_mlx_model(cfg);
            let mut session = Session::new();
            let logits = model
                .forward(&[1], &mut session)
                .expect("qwen forward should succeed");
            assert_eq!(logits.len(), 4);
        }

        #[test]
        fn tiny_gemma_parallel_forward_runs() {
            let cfg = tiny_mlx_model_config(ModelArchitecture::Gemma);
            let mut model = build_tiny_mlx_model(cfg);
            let mut session = Session::new();
            let logits = model
                .forward(&[1], &mut session)
                .expect("gemma forward should succeed");
            assert_eq!(logits.len(), 4);
        }

        #[test]
        fn tiny_phi_parallel_forward_runs() {
            let cfg = tiny_mlx_model_config(ModelArchitecture::Phi);
            let mut model = build_tiny_mlx_model(cfg);
            let mut session = Session::new();
            let logits = model
                .forward(&[1], &mut session)
                .expect("phi forward should succeed");
            assert_eq!(logits.len(), 4);
        }

        fn build_tiny_mlx_model(config: InferenceConfig) -> MlxInferenceModel {
            let backend = MlxComputeBackend::new();
            let h = config.hidden_size;
            let inter = config.intermediate_size;
            let head_dim = config.head_dim();
            let q_len = h;
            let kv_len = config.num_key_value_heads * config.kv_head_dim();

            let mut layers = Vec::with_capacity(config.layer_count);
            for _ in 0..config.layer_count {
                let mut lw = MlxLayerWeights::default_weights();
                lw.attn_norm = vec![1.0_f32; h];
                lw.attn_q = MlxWeightStorage::F32(
                    backend
                        .tensor_from_f32_2d(&vec![0.01_f32; q_len * h], q_len, h)
                        .unwrap()
                        .array,
                );
                lw.attn_k = MlxWeightStorage::F32(
                    backend
                        .tensor_from_f32_2d(&vec![0.01_f32; kv_len * h], kv_len, h)
                        .unwrap()
                        .array,
                );
                lw.attn_v = MlxWeightStorage::F32(
                    backend
                        .tensor_from_f32_2d(&vec![0.01_f32; kv_len * h], kv_len, h)
                        .unwrap()
                        .array,
                );
                lw.attn_output = MlxWeightStorage::F32(
                    backend
                        .tensor_from_f32_2d(&vec![0.01_f32; h * q_len], h, q_len)
                        .unwrap()
                        .array,
                );
                lw.ffn_norm = vec![1.0_f32; h];
                lw.ffn_gate = MlxWeightStorage::F32(
                    backend
                        .tensor_from_f32_2d(&vec![0.01_f32; inter * h], inter, h)
                        .unwrap()
                        .array,
                );
                lw.ffn_up = MlxWeightStorage::F32(
                    backend
                        .tensor_from_f32_2d(&vec![0.01_f32; inter * h], inter, h)
                        .unwrap()
                        .array,
                );
                lw.ffn_down = MlxWeightStorage::F32(
                    backend
                        .tensor_from_f32_2d(&vec![0.01_f32; h * inter], h, inter)
                        .unwrap()
                        .array,
                );

                if config.architecture.uses_moe() {
                    lw.moe_gate = MlxWeightStorage::F32(
                        backend
                            .tensor_from_f32_2d(
                                &vec![0.01_f32; config.num_experts * h],
                                config.num_experts,
                                h,
                            )
                            .unwrap()
                            .array,
                    );
                    for _ in 0..config.num_experts {
                        lw.moe_ffn_gate.push(MlxWeightStorage::F32(
                            backend
                                .tensor_from_f32_2d(&vec![0.01_f32; inter * h], inter, h)
                                .unwrap()
                                .array,
                        ));
                        lw.moe_ffn_up.push(MlxWeightStorage::F32(
                            backend
                                .tensor_from_f32_2d(&vec![0.01_f32; inter * h], inter, h)
                                .unwrap()
                                .array,
                        ));
                        lw.moe_ffn_down.push(MlxWeightStorage::F32(
                            backend
                                .tensor_from_f32_2d(&vec![0.01_f32; h * inter], h, inter)
                                .unwrap()
                                .array,
                        ));
                    }
                }

                if config.architecture.uses_mla() {
                    let latent_dim = h / 2;
                    let q_out = q_len;
                    let kv_out = kv_len * 2;
                    lw.mla_latent = MlxWeightStorage::F32(
                        backend
                            .tensor_from_f32_2d(&vec![0.01_f32; latent_dim * h], latent_dim, h)
                            .unwrap()
                            .array,
                    );
                    lw.mla_q_up = MlxWeightStorage::F32(
                        backend
                            .tensor_from_f32_2d(
                                &vec![0.01_f32; q_out * latent_dim],
                                q_out,
                                latent_dim,
                            )
                            .unwrap()
                            .array,
                    );
                    lw.mla_kv_up = MlxWeightStorage::F32(
                        backend
                            .tensor_from_f32_2d(
                                &vec![0.01_f32; kv_out * latent_dim],
                                kv_out,
                                latent_dim,
                            )
                            .unwrap()
                            .array,
                    );
                    lw.mla_out = MlxWeightStorage::F32(
                        backend
                            .tensor_from_f32_2d(&vec![0.01_f32; h * q_out], h, q_out)
                            .unwrap()
                            .array,
                    );
                }

                layers.push(lw);
            }

            MlxInferenceModel {
                config: config.clone(),
                backend,
                tok_embeddings: vec![0.1_f32; h * config.vocab_size],
                tok_embeddings_cols: config.vocab_size,
                norm_weight: vec![1.0_f32; h],
                output_weight: MlxWeightStorage::F32(MlxTensor::from_f32(&vec![
                    0.01_f32;
                    config.vocab_size
                        * h
                ])),
                layers,
                kv_cache: MlxKvCache::new(&config),
                workspace: MlxWorkspace::for_config(&config),
                alibi_slopes: vec![0.0_f32; config.alibi_num_heads.max(1)],
            }
        }
    }
}
