//! EAGLE3 speculative draft model (encoder/decoder).
//!
//! EAGLE3 is not a quantization format — it is a speculative-decoding technique:
//! a small draft network extracts features from three target-model layers, fuses
//! them through `fc`, then a single decoder layer proposes draft tokens that the
//! target model verifies in parallel.

use crate::dflash::{DFlashKvLayerCache, F32Weight};
use crate::flash_attention::flash_attention_decode_heads_f32;
use crate::gguf::{GgufQuantizationType, MappedGgufFile};
use crate::model::{Logits, Model, ModelError, Session, Token};
use crate::quantization::{dequantize_scalar, quantized_size};
use crate::tensor::{apply_rope_f32, rms_norm_f32};

/// EAGLE3 draft configuration from GGUF metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct Eagle3Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    /// Three target layer indices whose hidden states are concatenated for `fc`.
    pub extract_layers: Vec<usize>,
    pub target_hidden_size: usize,
    pub norm_before_residual: bool,
    pub vocab_size: usize,
    pub draft_vocab_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

impl Eagle3Config {
    pub fn encoder_input_width(&self) -> usize {
        self.extract_layers.len() * self.target_hidden_size
    }

    pub fn head_dim(&self) -> usize {
        if self.num_attention_heads == 0 {
            return 0;
        }
        self.hidden_size / self.num_attention_heads
    }

    pub fn from_gguf(mapped: &MappedGgufFile) -> Self {
        use crate::gguf::GgufMetadataValue;
        let metadata = &mapped.parsed().metadata;
        let arch = mapped.parsed().architecture().unwrap_or("eagle3");
        let namespaced_key = |namespace: &str, suffix: &str| format!("{namespace}.{suffix}");
        let arch_key = |suffix: &str| namespaced_key(arch, suffix);

        let arch_u32 = |suffix: &str| -> Option<u32> {
            for key in [
                arch_key(suffix),
                namespaced_key("eagle3", suffix),
                namespaced_key("eagle", suffix),
            ] {
                if let Some(value) = metadata.get(&key).and_then(|v| match v {
                    GgufMetadataValue::Uint8(x) => Some(*x as u32),
                    GgufMetadataValue::Uint16(x) => Some(*x as u32),
                    GgufMetadataValue::Uint32(x) => Some(*x),
                    GgufMetadataValue::Uint64(x) => (*x).try_into().ok(),
                    GgufMetadataValue::Int32(x) if *x >= 0 => Some(*x as u32),
                    _ => None,
                }) {
                    return Some(value);
                }
            }
            None
        };

        let arch_f32 = |suffix: &str| -> Option<f32> {
            for key in [arch_key(suffix), namespaced_key("eagle3", suffix)] {
                if let Some(value) = metadata.get(&key).and_then(|v| match v {
                    GgufMetadataValue::Float32(x) => Some(*x),
                    GgufMetadataValue::Float64(x) => Some(*x as f32),
                    GgufMetadataValue::Int32(x) => Some(*x as f32),
                    _ => None,
                }) {
                    return Some(value);
                }
            }
            None
        };

        let parse_extract_layers = |key: &str| -> Option<Vec<usize>> {
            metadata.get(key).and_then(|v| match v {
                GgufMetadataValue::Array(arr) => arr
                    .values
                    .iter()
                    .map(|elem| match elem {
                        GgufMetadataValue::Int32(x) if *x >= 0 => (*x).try_into().ok(),
                        GgufMetadataValue::Int64(x) if *x >= 0 => (*x).try_into().ok(),
                        GgufMetadataValue::Uint32(x) => Some(*x as usize),
                        GgufMetadataValue::Uint64(x) => (*x).try_into().ok(),
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>(),
                _ => None,
            })
        };

        let extract_layers = parse_extract_layers(&arch_key("extract_layers"))
            .or_else(|| parse_extract_layers(&namespaced_key("eagle3", "extract_layers")))
            .or_else(|| parse_extract_layers(&arch_key("target_layers")))
            .unwrap_or_else(|| vec![1, 16, 32]);

        let hidden_size = arch_u32("hidden_size")
            .or_else(|| arch_u32("embedding_length"))
            .unwrap_or(4096) as usize;
        let target_hidden_size = arch_u32("target_hidden_size")
            .or_else(|| arch_u32("n_embd_tgt"))
            .unwrap_or(hidden_size as u32) as usize;
        let num_hidden_layers = arch_u32("block_count")
            .or_else(|| arch_u32("num_hidden_layers"))
            .unwrap_or(1) as usize;
        let vocab_size = arch_u32("vocab_size").unwrap_or(128_256) as usize;
        let draft_vocab_size = mapped
            .mapped_tensor_infos()
            .iter()
            .find(|t| t.name == "d2t")
            .map(|t| t.dimensions.first().copied().unwrap_or(vocab_size as u64) as usize)
            .unwrap_or(vocab_size);
        let num_attention_heads = arch_u32("attention.head_count")
            .or_else(|| arch_u32("num_attention_heads"))
            .unwrap_or(32) as usize;
        let num_key_value_heads = arch_u32("attention.head_count_kv")
            .or_else(|| arch_u32("num_key_value_heads"))
            .unwrap_or(num_attention_heads as u32) as usize;
        let intermediate_size = arch_u32("feed_forward_length")
            .or_else(|| arch_u32("intermediate_size"))
            .unwrap_or(hidden_size as u32 * 4) as usize;

        Self {
            hidden_size,
            num_hidden_layers,
            extract_layers,
            target_hidden_size,
            norm_before_residual: metadata
                .get(&arch_key("norm_before_residual"))
                .and_then(|v| match v {
                    GgufMetadataValue::Bool(b) => Some(*b),
                    GgufMetadataValue::Uint8(x) => Some(*x != 0),
                    _ => None,
                })
                .unwrap_or(false),
            vocab_size,
            draft_vocab_size,
            num_attention_heads,
            num_key_value_heads,
            intermediate_size,
            rms_norm_eps: arch_f32("attention.layer_norm_rms_epsilon")
                .or_else(|| arch_f32("rms_norm_eps"))
                .unwrap_or(1e-5),
            rope_theta: arch_f32("rope.freq_base")
                .or_else(|| arch_f32("rope_theta"))
                .unwrap_or(10_000.0),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Eagle3Attention {
    q_proj: F32Weight,
    k_proj: F32Weight,
    v_proj: F32Weight,
    o_proj: F32Weight,
}

#[derive(Debug, Clone, PartialEq)]
struct Eagle3DecoderLayer {
    attn_norm: Vec<f32>,
    attn_norm_2: Vec<f32>,
    attention: Eagle3Attention,
    ffn_norm: Vec<f32>,
    mlp_gate: F32Weight,
    mlp_up: F32Weight,
    mlp_down: F32Weight,
}

/// EAGLE3 draft model: feature fusion (`fc`) + single decoder layer.
#[derive(Debug, Clone, PartialEq)]
pub struct Eagle3DraftModel {
    pub config: Eagle3Config,
    pub fc: F32Weight,
    pub d2t: Vec<u64>,
    pub layer: Eagle3DecoderLayer,
    pub output_norm: Vec<f32>,
    pub output: F32Weight,
    pub tok_embeddings: F32Weight,
    /// Encoder output (`g_embeddings`) carried across autoregressive draft steps.
    pub g_embeddings: Vec<f32>,
    pub kv_cache: Vec<DFlashKvLayerCache>,
    pub position_offset: usize,
}

impl Eagle3DraftModel {
    pub fn new(config: Eagle3Config) -> Self {
        let h = config.hidden_size;
        Self {
            config: config.clone(),
            fc: F32Weight::from_slice(Vec::new(), 0, 0),
            d2t: Vec::new(),
            layer: Eagle3DecoderLayer {
                attn_norm: vec![1.0; h],
                attn_norm_2: vec![1.0; h],
                attention: Eagle3Attention {
                    q_proj: F32Weight::from_slice(Vec::new(), 0, 0),
                    k_proj: F32Weight::from_slice(Vec::new(), 0, 0),
                    v_proj: F32Weight::from_slice(Vec::new(), 0, 0),
                    o_proj: F32Weight::from_slice(Vec::new(), 0, 0),
                },
                ffn_norm: vec![1.0; h],
                mlp_gate: F32Weight::from_slice(Vec::new(), 0, 0),
                mlp_up: F32Weight::from_slice(Vec::new(), 0, 0),
                mlp_down: F32Weight::from_slice(Vec::new(), 0, 0),
            },
            output_norm: vec![1.0; h],
            output: F32Weight::from_slice(Vec::new(), 0, 0),
            tok_embeddings: F32Weight::from_slice(Vec::new(), 0, 0),
            g_embeddings: vec![0.0; h],
            kv_cache: vec![DFlashKvLayerCache::new(); config.num_hidden_layers.max(1)],
            position_offset: 0,
        }
    }

    /// Fuse concatenated target hidden states into `g_embeddings`.
    pub fn encode_features(&mut self, target_features: &[f32]) -> Result<(), String> {
        let expected = self.config.encoder_input_width();
        if target_features.len() != expected {
            return Err(format!(
                "EAGLE3 encoder input width mismatch: expected {expected}, got {}",
                target_features.len()
            ));
        }
        if !self.fc.is_loaded() {
            return Err("EAGLE3 draft model missing fc.weight".to_string());
        }
        let h = self.config.hidden_size;
        self.g_embeddings.resize(h, 0.0);
        self.fc.gemv(target_features, &mut self.g_embeddings)?;
        Ok(())
    }

    pub fn reset_cache(&mut self) {
        for layer_cache in &mut self.kv_cache {
            layer_cache.keys.clear();
            layer_cache.values.clear();
            layer_cache.seq_len = 0;
        }
        self.position_offset = 0;
    }

    pub fn reserve_cache_tokens(&mut self, tokens: usize) {
        let kv_len = self.config.num_key_value_heads * self.config.head_dim();
        let additional = tokens.saturating_mul(kv_len);
        for layer_cache in &mut self.kv_cache {
            layer_cache.keys.reserve(additional);
            layer_cache.values.reserve(additional);
        }
    }

    fn weight_output_dim(weight: &F32Weight) -> usize {
        weight
            .quant
            .as_ref()
            .map(|q| q.out_dim)
            .unwrap_or(weight.rows)
    }

    fn fill_token_embedding(&self, token: u32, output: &mut [f32]) -> Result<(), String> {
        if !self.tok_embeddings.is_loaded() {
            return Err("EAGLE3 draft model missing token embeddings".to_string());
        }
        let vocab_size = Self::weight_output_dim(&self.tok_embeddings).max(1);
        let idx = (token as usize).min(vocab_size - 1);
        self.tok_embeddings.row(idx, output)
    }

    fn scatter_d2t_logits(&self, draft_logits: &[f32]) -> Result<Vec<f32>, String> {
        if self.d2t.is_empty() {
            return Ok(draft_logits.to_vec());
        }
        let vocab = self.config.vocab_size;
        let mut logits = vec![f32::NEG_INFINITY; vocab];
        for (draft_idx, &target_idx) in self.d2t.iter().enumerate() {
            if draft_idx < draft_logits.len() {
                let ti = target_idx as usize;
                if ti < vocab {
                    logits[ti] = draft_logits[draft_idx];
                }
            }
        }
        Ok(logits)
    }

    pub fn logits_from_hidden(&self, hidden: &[f32]) -> Result<Vec<f32>, String> {
        let h = self.config.hidden_size;
        let mut normed = vec![0.0_f32; h];
        rms_norm_f32(hidden, &self.output_norm, self.config.rms_norm_eps, &mut normed)
            .map_err(|e| format!("output_norm: {e:?}"))?;

        let head = if self.output.is_loaded() {
            &self.output
        } else {
            return Err("EAGLE3 draft model missing output projection".to_string());
        };
        let draft_vocab = Self::weight_output_dim(head);
        let mut draft_logits = vec![0.0_f32; draft_vocab];
        head.gemv(&normed, &mut draft_logits)?;
        self.scatter_d2t_logits(&draft_logits)
    }

    /// Decoder step: token embedding + `g_embeddings` -> hidden + logits.
    pub fn forward_decoder(&mut self, token: u32) -> Result<(Vec<f32>, Vec<f32>), String> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let layer = &self.layer;
        let pos = self.position_offset;

        let mut embd = vec![0.0_f32; h];
        self.fill_token_embedding(token, &mut embd)?;

        let mut embd_norm = vec![0.0_f32; h];
        rms_norm_f32(&embd, &layer.attn_norm, cfg.rms_norm_eps, &mut embd_norm)
            .map_err(|e| format!("attn_norm: {e:?}"))?;

        let mut g_norm = vec![0.0_f32; h];
        rms_norm_f32(&self.g_embeddings, &layer.attn_norm_2, cfg.rms_norm_eps, &mut g_norm)
            .map_err(|e| format!("attn_norm_2: {e:?}"))?;

        let residual = if cfg.norm_before_residual {
            g_norm.clone()
        } else {
            self.g_embeddings.clone()
        };

        let mut concat = vec![0.0_f32; h * 2];
        concat[..h].copy_from_slice(&embd_norm);
        concat[h..].copy_from_slice(&g_norm);

        let head_dim = cfg.head_dim();
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let q_size = num_heads * head_dim;
        let kv_len = num_kv_heads * head_dim;

        let mut q = vec![0.0_f32; q_size];
        let mut k = vec![0.0_f32; kv_len];
        let mut v = vec![0.0_f32; kv_len];
        layer.attention.q_proj.gemv(&concat, &mut q)?;
        layer.attention.k_proj.gemv(&concat, &mut k)?;
        layer.attention.v_proj.gemv(&concat, &mut v)?;

        let mut head_scratch = vec![0.0_f32; head_dim];
        for h_idx in 0..num_heads {
            let start = h_idx * head_dim;
            apply_rope_f32(
                &q[start..start + head_dim],
                pos,
                head_dim,
                cfg.rope_theta,
                &mut head_scratch,
            )
            .map_err(|e| format!("rope q: {e:?}"))?;
            q[start..start + head_dim].copy_from_slice(&head_scratch);
        }
        for kv_h in 0..num_kv_heads {
            let start = kv_h * head_dim;
            apply_rope_f32(
                &k[start..start + head_dim],
                pos,
                head_dim,
                cfg.rope_theta,
                &mut head_scratch,
            )
            .map_err(|e| format!("rope k: {e:?}"))?;
            k[start..start + head_dim].copy_from_slice(&head_scratch);
        }

        let layer_cache = &mut self.kv_cache[0];
        layer_cache.keys.extend_from_slice(&k);
        layer_cache.values.extend_from_slice(&v);
        layer_cache.seq_len += 1;

        let mut attn_out = vec![0.0_f32; q_size];
        flash_attention_decode_heads_f32(
            &q,
            &layer_cache.keys,
            &layer_cache.values,
            layer_cache.seq_len,
            head_dim,
            kv_len,
            num_heads,
            num_kv_heads,
            &mut attn_out,
        )
        .map_err(|e| format!("attention: {e:?}"))?;

        let mut hidden = vec![0.0_f32; h];
        if layer.attention.o_proj.is_loaded() {
            layer.attention.o_proj.gemv(&attn_out, &mut hidden)?;
        } else if q_size == h {
            hidden.copy_from_slice(&attn_out);
        }
        for i in 0..h {
            hidden[i] += residual[i];
        }

        let mut normed_ffn = vec![0.0_f32; h];
        rms_norm_f32(&hidden, &layer.ffn_norm, cfg.rms_norm_eps, &mut normed_ffn)
            .map_err(|e| format!("ffn_norm: {e:?}"))?;

        let inter = cfg.intermediate_size;
        let mut gate = vec![0.0_f32; inter];
        let mut up = vec![0.0_f32; inter];
        layer.mlp_gate.gemv(&normed_ffn, &mut gate)?;
        layer.mlp_up.gemv(&normed_ffn, &mut up)?;
        for i in 0..inter {
            let g = gate[i];
            gate[i] = g * (1.0 / (1.0 + (-g).exp())) * up[i];
        }
        let mut mlp_out = vec![0.0_f32; h];
        layer.mlp_down.gemv(&gate, &mut mlp_out)?;
        for i in 0..h {
            hidden[i] += mlp_out[i];
        }

        self.g_embeddings.copy_from_slice(&hidden);
        self.position_offset += 1;

        let logits = self.logits_from_hidden(&hidden)?;
        Ok((hidden, logits))
    }

    pub fn load_from_gguf(
        mapped: &MappedGgufFile,
        config: Eagle3Config,
    ) -> Result<Self, String> {
        let mut model = Self::new(config.clone());
        let tensor_infos = mapped.mapped_tensor_infos();

        let load_f32_with_dims = |name: &str| -> Result<Option<(Vec<f32>, Vec<u64>)>, String> {
            let info = match tensor_infos.iter().find(|t| t.name == name) {
                Some(i) => i,
                None => return Ok(None),
            };
            let qtype = GgufQuantizationType::from_ggml_type(info.ggml_type);
            let value_count: usize = info.dimensions.iter().map(|&d| d as usize).product();
            let qsize = quantized_size(qtype, value_count)
                .map_err(|e| format!("quantized_size for {name}: {e:?}"))?;
            let offset = info.absolute_offset as usize;
            let shard = mapped.tensor_mmap(info);
            let end = offset
                .checked_add(qsize)
                .ok_or_else(|| format!("offset overflow for {name}"))?;
            if end > shard.len() {
                return Err(format!("tensor {name} out of bounds"));
            }
            let mut f32_data = vec![0.0_f32; value_count];
            dequantize_scalar(qtype, &shard[offset..end], &mut f32_data)
                .map_err(|e| format!("dequantize {name}: {e:?}"))?;
            Ok(Some((f32_data, info.dimensions.clone())))
        };

        let load_proj = |name: &str| -> Result<F32Weight, String> {
            let info = match tensor_infos.iter().find(|t| t.name == name) {
                Some(i) => i,
                None => return Ok(F32Weight::from_slice(Vec::new(), 0, 0)),
            };
            if info.dimensions.len() != 2 {
                return load_f32_with_dims(name).map(|opt| match opt {
                    Some((data, dims)) => {
                        let r = dims[0] as usize;
                        let c = dims[1] as usize;
                        F32Weight::from_slice(transpose_f32(&data, r, c), c, r)
                    }
                    None => F32Weight::from_slice(Vec::new(), 0, 0),
                });
            }
            let qtype = GgufQuantizationType::from_ggml_type(info.ggml_type);
            let in_dim = info.dimensions[0] as usize;
            let out_dim = info.dimensions[1] as usize;
            if matches!(
                qtype,
                GgufQuantizationType::Q4_K_S
                    | GgufQuantizationType::Q4_K_M
                    | GgufQuantizationType::Q6_K
                    | GgufQuantizationType::Q8_0
                    | GgufQuantizationType::NVFP4
            ) {
                let value_count = out_dim * in_dim;
                let qsize = quantized_size(qtype, value_count)
                    .map_err(|e| format!("quantized_size: {e:?}"))?;
                let offset = info.absolute_offset as usize;
                let shard = mapped.tensor_mmap(info);
                return Ok(F32Weight::from_quantized(
                    shard[offset..offset + qsize].to_vec(),
                    qtype,
                    out_dim,
                    in_dim,
                ));
            }
            load_f32_with_dims(name).map(|opt| match opt {
                Some((data, _)) => F32Weight::from_slice(transpose_f32(&data, in_dim, out_dim), out_dim, in_dim),
                None => F32Weight::from_slice(Vec::new(), 0, 0),
            })
        };

        model.fc = load_proj("fc.weight")?;
        if let Some((data, dims)) = load_f32_with_dims("d2t")? {
            let count = dims.first().copied().unwrap_or(data.len() as u64) as usize;
            model.d2t = data.into_iter().take(count).map(|v| v as u64).collect();
        }

        let prefix = "blk.0";
        if let Some((data, _)) = load_f32_with_dims(&format!("{prefix}.attn_norm.weight"))? {
            model.layer.attn_norm = data;
        }
        if let Some((data, _)) = load_f32_with_dims(&format!("{prefix}.attn_norm_2.weight"))? {
            model.layer.attn_norm_2 = data;
        }
        if let Some((data, _)) = load_f32_with_dims(&format!("{prefix}.ffn_norm.weight"))? {
            model.layer.ffn_norm = data;
        }
        model.layer.attention.q_proj = load_proj(&format!("{prefix}.attn_q.weight"))?;
        model.layer.attention.k_proj = load_proj(&format!("{prefix}.attn_k.weight"))?;
        model.layer.attention.v_proj = load_proj(&format!("{prefix}.attn_v.weight"))?;
        model.layer.attention.o_proj = load_proj(&format!("{prefix}.attn_output.weight"))?;
        model.layer.mlp_gate = load_proj(&format!("{prefix}.ffn_gate.weight"))?;
        model.layer.mlp_up = load_proj(&format!("{prefix}.ffn_up.weight"))?;
        model.layer.mlp_down = load_proj(&format!("{prefix}.ffn_down.weight"))?;

        if let Some((data, _)) = load_f32_with_dims("output_norm.weight")? {
            model.output_norm = data;
        }
        model.output = load_proj("output.weight")?;
        model.tok_embeddings = load_proj("token_embd.weight")?;

        Ok(model)
    }

    pub fn load_external_io_from_gguf(&mut self, mapped: &MappedGgufFile) -> Result<(), String> {
        let borrowed = Self::load_from_gguf(mapped, self.config.clone())?;
        if !self.tok_embeddings.is_loaded() && borrowed.tok_embeddings.is_loaded() {
            self.tok_embeddings = borrowed.tok_embeddings;
        }
        if !self.output.is_loaded() && borrowed.output.is_loaded() {
            self.output = borrowed.output;
        }
        Ok(())
    }
}

fn transpose_f32(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

impl Model for Eagle3DraftModel {
    fn forward(&mut self, tokens: &[Token], session: &mut Session) -> Result<Logits, ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::EmptyInput);
        }
        let mut last_logits = Vec::new();
        for &token in tokens {
            let (_, logits) = self
                .forward_decoder(token)
                .map_err(ModelError::InferenceFailed)?;
            last_logits = logits;
        }
        session.record_tokens(tokens.len());
        Ok(last_logits)
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn context_size(&self) -> usize {
        8192
    }

    fn layer_count(&self) -> usize {
        self.config.num_hidden_layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eagle3_config_encoder_width() {
        let cfg = Eagle3Config {
            hidden_size: 4096,
            num_hidden_layers: 1,
            extract_layers: vec![2, 16, 30],
            target_hidden_size: 4096,
            norm_before_residual: false,
            vocab_size: 128_256,
            draft_vocab_size: 64_000,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            intermediate_size: 14_336,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
        };
        assert_eq!(cfg.encoder_input_width(), 4096 * 3);
    }
}
