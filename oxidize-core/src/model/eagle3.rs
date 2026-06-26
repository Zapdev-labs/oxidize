//! EAGLE3 speculative draft model (encoder/decoder).
//!
//! EAGLE3 is not a quantization format — it is a speculative-decoding technique:
//! a small draft network extracts features from three target-model layers, fuses
//! them through `fc`, then a single decoder layer proposes draft tokens that the
//! target model verifies in parallel.

use crate::dflash::{DFlashKvLayerCache, F32Weight};
use crate::flash_attention::flash_attention_decode_heads_f32;
use crate::safetensors::{
    load_mapped_safetensors, tensor_bytes_to_f32, tensor_bytes_to_i64, MappedSafeTensorsFile,
};
use crate::gguf::{GgufQuantizationType, MappedGgufFile};
use crate::model::{Logits, Model, ModelError, Session, Token};
use crate::quantization::{dequantize_scalar, quantized_size};
use crate::tensor::{apply_rope_f32, rms_norm_f32};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

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
    pub head_dim: Option<usize>,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

impl Eagle3Config {
    pub fn encoder_input_width(&self) -> usize {
        self.extract_layers.len() * self.target_hidden_size
    }

    pub fn head_dim(&self) -> usize {
        if let Some(hd) = self.head_dim.filter(|&d| d > 0) {
            return hd;
        }
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
            head_dim: arch_u32("head_dim").map(|v| v as usize),
            intermediate_size,
            rms_norm_eps: arch_f32("attention.layer_norm_rms_epsilon")
                .or_else(|| arch_f32("rms_norm_eps"))
                .unwrap_or(1e-5),
            rope_theta: arch_f32("rope.freq_base")
                .or_else(|| arch_f32("rope_theta"))
                .unwrap_or(10_000.0),
        }
    }

    /// Target-model hints used when loading HF SafeTensors drafts (layer indices, hidden size).
    pub fn from_hf_json(
        draft: &serde_json::Value,
        target: Option<&Eagle3TargetHints>,
    ) -> Self {
        let hidden_size = draft
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(4096) as usize;
        let num_hidden_layers = draft
            .get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;
        let draft_vocab_size = draft
            .get("draft_vocab_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or_else(|| {
                draft
                    .get("vocab_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(128_256) as usize
            });
        let vocab_size = target
            .and_then(|t| t.target_vocab_size)
            .or_else(|| draft.get("vocab_size").and_then(|v| v.as_u64()).map(|v| v as usize))
            .unwrap_or(draft_vocab_size);
        let target_hidden_size = draft
            .get("target_hidden_size")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .or_else(|| target.map(|t| t.target_hidden_size))
            .unwrap_or(hidden_size);
        let extract_layers = parse_hf_extract_layers(draft).or_else(|| {
            target.map(|t| default_eagle3_extract_layers(t.target_layer_count))
        }).unwrap_or_else(|| vec![1, 16, 32]);
        let num_attention_heads = draft
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;
        let num_key_value_heads = draft
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(num_attention_heads as u64) as usize;
        let intermediate_size = draft
            .get("intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or((hidden_size * 4) as u64) as usize;
        let rms_norm_eps = draft
            .get("rms_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5) as f32;
        let rope_theta = draft
            .get("rope_parameters")
            .and_then(|rp| rp.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .or_else(|| draft.get("rope_theta").and_then(|v| v.as_f64()))
            .unwrap_or(10_000.0) as f32;
        let norm_before_residual = draft
            .get("norm_before_residual")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let head_dim = draft
            .get("head_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        Self {
            hidden_size,
            num_hidden_layers,
            extract_layers,
            target_hidden_size,
            norm_before_residual,
            vocab_size,
            draft_vocab_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            intermediate_size,
            rms_norm_eps,
            rope_theta,
        }
    }
}

/// Hints from the target (full) model when loading an HF EAGLE3 draft checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Eagle3TargetHints {
    pub target_hidden_size: usize,
    pub target_layer_count: usize,
    pub target_vocab_size: Option<usize>,
}

pub fn default_eagle3_extract_layers(target_layer_count: usize) -> Vec<usize> {
    if target_layer_count < 4 {
        return vec![0, 0, 0];
    }
    vec![
        2,
        target_layer_count / 2,
        target_layer_count.saturating_sub(3),
    ]
}

fn parse_hf_extract_layers(draft: &serde_json::Value) -> Option<Vec<usize>> {
    let from_array = |value: &serde_json::Value| -> Option<Vec<usize>> {
        let arr = value.as_array()?;
        arr.iter()
            .map(|elem| elem.as_u64().map(|v| v as usize))
            .collect()
    };
    draft
        .get("extract_layers")
        .and_then(from_array)
        .or_else(|| {
            draft
                .get("eagle_config")
                .and_then(|ec| ec.get("eagle_aux_hidden_state_layer_ids"))
                .and_then(from_array)
        })
        .or_else(|| {
            draft
                .get("target_layers")
                .and_then(from_array)
        })
}

pub fn is_eagle3_safetensors_path(path: &Path) -> bool {
    if path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
    {
        return true;
    }
    if path.is_dir() {
        let config_path = path.join("config.json");
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                if json.get("draft_vocab_size").is_some() {
                    return true;
                }
                if json.get("eagle_config").is_some() || json.get("extract_layers").is_some() {
                    return true;
                }
            }
        }
        return std::fs::read_dir(path).ok().is_some_and(|entries| {
            entries.filter_map(Result::ok).any(|entry| {
                entry
                    .path()
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
            })
        });
    }
    false
}

pub fn resolve_eagle3_safetensors_paths(path: &Path) -> Result<(Vec<PathBuf>, PathBuf), String> {
    if path.is_file() {
        if !path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
        {
            return Err(format!(
                "expected .safetensors file, got {}",
                path.display()
            ));
        }
        let config = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("config.json");
        if !config.is_file() {
            return Err(format!(
                "missing config.json beside {}",
                path.display()
            ));
        }
        return Ok((vec![path.to_path_buf()], config));
    }

    if path.is_dir() {
        let config = path.join("config.json");
        if !config.is_file() {
            return Err(format!("missing config.json in {}", path.display()));
        }
        let preferred = path.join("model.safetensors");
        if preferred.is_file() {
            return Ok((vec![preferred], config));
        }
        let mut shards: Vec<PathBuf> = std::fs::read_dir(path)
            .map_err(|e| format!("read {}: {e}", path.display()))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|p| {
                p.extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
            })
            .collect();
        shards.sort();
        if shards.is_empty() {
            return Err(format!("no .safetensors weights found in {}", path.display()));
        }
        return Ok((shards, config));
    }

    Err(format!("EAGLE3 draft path not found: {}", path.display()))
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
                let end = offset
                    .checked_add(qsize)
                    .ok_or_else(|| format!("offset overflow for {name}"))?;
                if end > shard.len() {
                    return Err(format!("tensor {name} out of bounds"));
                }
                return Ok(F32Weight::from_quantized(
                    shard[offset..end].to_vec(),
                    qtype,
                    out_dim,
                    in_dim,
                ));
            }
            load_f32_with_dims(name).map(|opt| match opt {
                Some((data, _)) => {
                    F32Weight::from_slice(transpose_f32(&data, in_dim, out_dim), out_dim, in_dim)
                }
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

    pub fn load_from_safetensors(
        weights_path: &Path,
        config: Eagle3Config,
        target_vocab_size: Option<usize>,
    ) -> Result<Self, String> {
        let mapped = load_mapped_safetensors(weights_path)
            .map_err(|e| format!("load safetensors {}: {e}", weights_path.display()))?;
        Self::load_from_mapped_safetensors_shards(std::slice::from_ref(&mapped), config, target_vocab_size)
    }

    pub fn load_eagle3_draft(
        path: &Path,
        target: Eagle3TargetHints,
    ) -> Result<Self, String> {
        if is_eagle3_safetensors_path(path) {
            let (weight_paths, config_path) = resolve_eagle3_safetensors_paths(path)?;
            let config_json: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(&config_path)
                    .map_err(|e| format!("read {}: {e}", config_path.display()))?,
            )
            .map_err(|e| format!("parse {}: {e}", config_path.display()))?;
            let config = Eagle3Config::from_hf_json(&config_json, Some(&target));
            let mut shards = Vec::with_capacity(weight_paths.len());
            for weights_path in &weight_paths {
                shards.push(
                    load_mapped_safetensors(weights_path)
                        .map_err(|e| format!("load safetensors {}: {e}", weights_path.display()))?,
                );
            }
            return Self::load_from_mapped_safetensors_shards(&shards, config, target.target_vocab_size);
        }
        Err(format!(
            "expected EAGLE3 SafeTensors draft at {}, got unsupported format",
            path.display()
        ))
    }

    fn load_from_mapped_safetensors_shards(
        shards: &[MappedSafeTensorsFile],
        config: Eagle3Config,
        target_vocab_size: Option<usize>,
    ) -> Result<Self, String> {
        let mut canonical: HashMap<String, (usize, String)> = HashMap::new();
        for (shard_idx, mapped) in shards.iter().enumerate() {
            for info in mapped.tensors() {
                if let Some(gguf_name) = map_hf_eagle3_tensor_name(&info.name) {
                    canonical.insert(gguf_name, (shard_idx, info.name.clone()));
                }
            }
        }

        let resolve_name = |gguf_name: &str| -> Option<(usize, String)> {
            canonical.get(gguf_name).cloned()
        };

        let mut model = Self::new(config.clone());
        let cfg = model.config.clone();

        let load_f32_with_dims =
            |gguf_name: &str| -> Result<Option<(Vec<f32>, Vec<u64>)>, String> {
                let (shard_idx, hf_name) = match resolve_name(gguf_name) {
                    Some(names) => names,
                    None => return Ok(None),
                };
                let mapped = &shards[shard_idx];
                let info = mapped
                    .tensor_info(&hf_name)
                    .ok_or_else(|| format!("missing tensor info for {hf_name}"))?;
                let element_count: usize = info.shape.iter().product();
                let bytes = mapped
                    .tensor_data(&hf_name)
                    .ok_or_else(|| format!("missing tensor bytes for {hf_name}"))?;
                let mut data = tensor_bytes_to_f32(info.dtype, bytes, element_count)
                    .map_err(|e| format!("dequantize {hf_name}: {e}"))?;
                let shape_u64: Vec<u64> = info.shape.iter().map(|&d| d as u64).collect();
                if info.shape.len() == 2 {
                    let out_dim = info.shape[0];
                    let in_dim = info.shape[1];
                    if gguf_name.ends_with("attn_q.weight") {
                        apply_llama_qk_permute(
                            &mut data,
                            out_dim,
                            in_dim,
                            cfg.num_attention_heads,
                            cfg.num_key_value_heads,
                            false,
                        );
                    } else if gguf_name.ends_with("attn_k.weight") {
                        apply_llama_qk_permute(
                            &mut data,
                            out_dim,
                            in_dim,
                            cfg.num_attention_heads,
                            cfg.num_key_value_heads,
                            true,
                        );
                    }
                }
                Ok(Some((data, shape_u64)))
            };

        let load_proj = |gguf_name: &str| -> Result<F32Weight, String> {
            let Some((data, shape)) = load_f32_with_dims(gguf_name)? else {
                return Ok(F32Weight::from_slice(Vec::new(), 0, 0));
            };
            if shape.len() != 2 {
                return Err(format!("expected 2-D projection for {gguf_name}"));
            }
            let out_dim = shape[0] as usize;
            let in_dim = shape[1] as usize;
            Ok(F32Weight::from_slice(data, out_dim, in_dim))
        };

        model.fc = load_proj("fc.weight")?;

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

        if let Some((shard_idx, hf_name)) = resolve_name("d2t") {
            let mapped = &shards[shard_idx];
            let info = mapped
                .tensor_info(&hf_name)
                .ok_or_else(|| format!("missing tensor info for {hf_name}"))?;
            let element_count: usize = info.shape.iter().product();
            let bytes = mapped
                .tensor_data(&hf_name)
                .ok_or_else(|| format!("missing tensor bytes for {hf_name}"))?;
            let offsets = tensor_bytes_to_i64(info.dtype, bytes, element_count)
                .map_err(|e| format!("decode d2t: {e}"))?;
            model.d2t = offsets
                .into_iter()
                .enumerate()
                .map(|(i, v)| (v + i as i64) as u64)
                .collect();
        }

        if let Some(vocab) = target_vocab_size {
            model.config.vocab_size = vocab;
            if model.d2t.iter().any(|&id| id as usize >= vocab) {
                return Err(format!(
                    "EAGLE3 d2t target ids out of range for target vocab size {vocab}"
                ));
            }
            let mut seen = std::collections::HashSet::new();
            for &id in &model.d2t {
                if !seen.insert(id) {
                    return Err("EAGLE3 d2t contains duplicate target ids".to_string());
                }
            }
        }

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

fn normalize_hf_eagle3_name(name: &str) -> String {
    if let Some(rest) = name.strip_prefix("midlayer.") {
        return format!("model.layers.0.{rest}");
    }
    if let Some(rest) = name.strip_prefix("layers.0.") {
        return format!("model.layers.0.{rest}");
    }
    name.to_string()
}

fn map_hf_eagle3_tensor_name(hf_name: &str) -> Option<String> {
    let name = normalize_hf_eagle3_name(hf_name);
    if name == "fc.weight" || name == "model.fc.weight" {
        return Some("fc.weight".into());
    }
    if name == "d2t" {
        return Some("d2t".into());
    }
    if name == "t2d" || name.contains("fc_norm") {
        return None;
    }
    if name == "model.embed_tokens.weight" || name == "embed_tokens.weight" {
        return Some("token_embd.weight".into());
    }
    if name == "model.norm.weight" || name == "norm.weight" {
        return Some("output_norm.weight".into());
    }
    if name == "lm_head.weight" || name == "model.lm_head.weight" {
        return Some("output.weight".into());
    }
    let layer = crate::conversion::extract_layer_index(&name)?;
    let prefix = format!("model.layers.{layer}.");
    let suffix = name.strip_prefix(&prefix)?;
    let mapped = match suffix {
        "input_layernorm.weight" => format!("blk.{layer}.attn_norm.weight"),
        "hidden_norm.weight" => format!("blk.{layer}.attn_norm_2.weight"),
        "post_attention_layernorm.weight" => format!("blk.{layer}.ffn_norm.weight"),
        "self_attn.q_proj.weight" => format!("blk.{layer}.attn_q.weight"),
        "self_attn.k_proj.weight" => format!("blk.{layer}.attn_k.weight"),
        "self_attn.v_proj.weight" => format!("blk.{layer}.attn_v.weight"),
        "self_attn.o_proj.weight" => format!("blk.{layer}.attn_output.weight"),
        "mlp.gate_proj.weight" => format!("blk.{layer}.ffn_gate.weight"),
        "mlp.up_proj.weight" => format!("blk.{layer}.ffn_up.weight"),
        "mlp.down_proj.weight" => format!("blk.{layer}.ffn_down.weight"),
        _ => return None,
    };
    Some(mapped)
}

fn apply_llama_qk_permute(
    data: &mut [f32],
    out_dim: usize,
    in_dim: usize,
    n_head: usize,
    n_kv_head: usize,
    is_k: bool,
) {
    if out_dim == 0 || in_dim == 0 || n_head == 0 {
        return;
    }
    let n_head_eff = if is_k && n_kv_head != n_head {
        n_kv_head
    } else {
        n_head
    };
    if out_dim % n_head_eff != 0 {
        return;
    }
    let half = out_dim / n_head_eff / 2;
    if half == 0 {
        return;
    }
    let mut temp = vec![0.0_f32; data.len()];
    for h in 0..n_head_eff {
        for a in 0..half {
            for b in 0..2 {
                let src_row = h * 2 * half + b * half + a;
                let dst_row = h * 2 * half + a * 2 + b;
                let src_start = src_row * in_dim;
                let dst_start = dst_row * in_dim;
                temp[dst_start..dst_start + in_dim]
                    .copy_from_slice(&data[src_start..src_start + in_dim]);
            }
        }
    }
    data.copy_from_slice(&temp);
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
    use safetensors::tensor::{Dtype, TensorView};
    use std::collections::HashMap;
    use std::io::Write;

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
            head_dim: None,
            intermediate_size: 14_336,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
        };
        assert_eq!(cfg.encoder_input_width(), 4096 * 3);
    }

    #[test]
    fn eagle3_default_extract_layers_match_llama_cpp() {
        assert_eq!(default_eagle3_extract_layers(78), vec![2, 39, 75]);
    }

    #[test]
    fn eagle3_hf_tensor_name_mapping() {
        assert_eq!(
            map_hf_eagle3_tensor_name("model.fc.weight").as_deref(),
            Some("fc.weight")
        );
        assert_eq!(
            map_hf_eagle3_tensor_name("model.layers.0.self_attn.q_proj.weight").as_deref(),
            Some("blk.0.attn_q.weight")
        );
        assert_eq!(
            map_hf_eagle3_tensor_name("model.layers.0.hidden_norm.weight").as_deref(),
            Some("blk.0.attn_norm_2.weight")
        );
        assert!(map_hf_eagle3_tensor_name("t2d").is_none());
    }

    #[test]
    fn eagle3_from_hf_json_uses_target_layers() {
        let draft = serde_json::json!({
            "hidden_size": 6144,
            "num_hidden_layers": 1,
            "draft_vocab_size": 32000,
            "vocab_size": 154880,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "intermediate_size": 40960,
            "rms_norm_eps": 1e-5,
            "rope_parameters": { "rope_theta": 10000.0 }
        });
        let target = Eagle3TargetHints {
            target_hidden_size: 6144,
            target_layer_count: 78,
            target_vocab_size: Some(154880),
        };
        let cfg = Eagle3Config::from_hf_json(&draft, Some(&target));
        assert_eq!(cfg.extract_layers, vec![2, 39, 75]);
        assert_eq!(cfg.target_hidden_size, 6144);
        assert_eq!(cfg.vocab_size, 154880);
    }

    #[test]
    fn eagle3_d2t_offset_to_absolute() {
        let offsets = vec![100_i64, 200, 50];
        let absolute: Vec<u64> = offsets
            .into_iter()
            .enumerate()
            .map(|(i, v)| (v + i as i64) as u64)
            .collect();
        assert_eq!(absolute, vec![100, 201, 52]);
    }

    #[test]
    fn eagle3_load_minimal_bf16_safetensors() {
        let tmp = std::env::temp_dir().join(format!("eagle3-test-{}.safetensors", std::process::id()));
        let cfg_dir = tmp.with_extension("dir");
        std::fs::create_dir_all(&cfg_dir).expect("tmpdir");
        let weights = cfg_dir.join("model.safetensors");

        let bf16_one = 0x3f80u16.to_le_bytes();
        let fc_bytes: Vec<u8> = (0..4).flat_map(|_| bf16_one).collect();
        let fc = TensorView::new(Dtype::BF16, vec![2, 2], &fc_bytes).unwrap();
        let d2t_bytes = 10i64.to_le_bytes().to_vec();
        let d2t = TensorView::new(Dtype::I64, vec![1], &d2t_bytes).unwrap();
        let mut tensors = HashMap::new();
        tensors.insert("model.fc.weight".to_string(), fc);
        tensors.insert("d2t".to_string(), d2t);
        let st = safetensors::tensor::serialize(&tensors, &None).unwrap();
        let mut file = std::fs::File::create(&weights).unwrap();
        file.write_all(&st).unwrap();

        let config_path = cfg_dir.join("config.json");
        std::fs::write(
            &config_path,
            serde_json::json!({
                "hidden_size": 2,
                "num_hidden_layers": 1,
                "draft_vocab_size": 1,
                "vocab_size": 16,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "intermediate_size": 4,
                "rms_norm_eps": 1e-5
            })
            .to_string(),
        )
        .unwrap();

        let target = Eagle3TargetHints {
            target_hidden_size: 2,
            target_layer_count: 8,
            target_vocab_size: Some(16),
        };
        let model = Eagle3DraftModel::load_eagle3_draft(&cfg_dir, target).expect("load draft");
        assert!(model.fc.is_loaded());
        assert_eq!(model.d2t, vec![10]);
        let _ = std::fs::remove_dir_all(&cfg_dir);
    }
}
