#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use crate::flash_attention::{flash_attention_decode_heads_f16, flash_attention_decode_heads_f32};
use crate::gguf::{GgufQuantizationType, MappedGgufFile};
use crate::kv_cache::{KvCache, KvCacheConfig};
use crate::model::{Logits, Model, ModelError, Session, Token};
use crate::quantization::{dequantize_scalar, quantized_size};
use crate::tensor::{
    DType, GemvJob, apply_geglu_inplace_f32, apply_rope_f32, apply_swiglu_inplace_f32,
    f16_le_to_f32, gemm_quantized_f32, gemv_f32, gemv_quantized_experts_f32,
    gemv_quantized_experts_gate_up_f32, gemv_quantized_f32, gemv_quantized_multi_f32, rms_norm_f32,
};
use memmap2::Mmap;
use std::sync::Arc;

/// Cached `OXIDIZE_TRACE_FWD` gate. The trace checks sit inside per-layer
/// per-token forward loops; an uncached `env::var_os` there is a libc
/// environment scan on every layer of every token.
pub(crate) fn trace_fwd_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("OXIDIZE_TRACE_FWD").is_some())
}

/// Cached `OXIDIZE_TRACE_VALS` gate (see [`trace_fwd_enabled`]).
pub(crate) fn trace_vals_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("OXIDIZE_TRACE_VALS").is_some())
}

/// Detected model architecture from GGUF metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelArchitecture {
    #[default]
    Llama,
    Mistral,
    Mixtral,
    DeepSeek,
    Qwen,
    Gemma,
    Phi,
    Falcon,
    Gpt2,
    GptJ,
    GptNeoX,
    MiniMax,
    /// LiquidAI LFM2 hybrid (short-conv mixing + interleaved GQA attention), dense FFN.
    Lfm2,
    /// LiquidAI LFM2 hybrid with sparse MoE FFN (lfm2moe).
    Lfm2Moe,
}

impl ModelArchitecture {
    /// Detect architecture from GGUF metadata.
    pub fn from_gguf(mapped: &MappedGgufFile) -> Self {
        let parsed = mapped.parsed();
        if let Some(arch) = parsed.architecture() {
            match arch {
                "llama" => Self::Llama,
                "mistral" => Self::Mistral,
                "mixtral" => Self::Mixtral,
                "deepseek" | "deepseek2" | "deepseek_v2" | "deepseek_v3" | "deepseek_moe" => {
                    Self::DeepSeek
                }
                "qwen" | "qwen2" | "qwen2moe" | "qwen3" | "qwen3moe" | "qwen35" | "qwen3_5"
                | "qwen3_5_text" | "qwen35_text" | "qwen3_5_moe" | "qwen3_5_moe_text"
                | "qwen35moe" => Self::Qwen,
                "gemma" | "gemma2" | "gemma3" | "gemma4" => Self::Gemma,
                "phi" | "phi3" => Self::Phi,
                "falcon" => Self::Falcon,
                "gpt2" => Self::Gpt2,
                "gptj" => Self::GptJ,
                "gptneox" => Self::GptNeoX,
                "minimax" | "minimax-m2" | "minimax-text-01" => Self::MiniMax,
                "lfm2" => Self::Lfm2,
                "lfm2moe" => Self::Lfm2Moe,
                _ => Self::Llama,
            }
        } else {
            Self::Llama
        }
    }

    /// Whether this architecture uses Alibi positional encoding (no RoPE).
    pub fn uses_alibi(&self) -> bool {
        matches!(self, Self::Falcon | Self::Gpt2 | Self::GptJ | Self::GptNeoX)
    }

    /// Whether this architecture uses sliding window attention.
    pub fn uses_sliding_window(&self) -> bool {
        matches!(self, Self::Qwen | Self::Mistral)
    }

    /// Whether this architecture uses MoE FFN.
    pub fn uses_moe(&self) -> bool {
        matches!(self, Self::Mixtral | Self::MiniMax | Self::Lfm2Moe)
    }

    /// Whether this architecture uses LFM2 short-convolution token mixing on
    /// non-attention layers (in addition to interleaved GQA attention layers).
    pub fn uses_shortconv(&self) -> bool {
        matches!(self, Self::Lfm2 | Self::Lfm2Moe)
    }

    /// Whether this architecture uses parallel attention + FFN (fused residual).
    pub fn uses_parallel_attn_ffn(&self) -> bool {
        matches!(self, Self::Gemma | Self::Phi)
    }

    /// Whether this architecture uses MLA compressed attention.
    pub fn uses_mla(&self) -> bool {
        matches!(self, Self::DeepSeek)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct InferenceConfig {
    pub vocab_size: usize,
    pub context_size: usize,
    pub layer_count: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub key_value_head_dim: usize,
    pub kv_cache_dtype: DType,
    /// Quantization scheme for I8/I16 KV cache (no effect on F32/F16).
    pub kv_quantization: crate::kv_cache::KvQuantization,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub architecture: ModelArchitecture,
    /// Sliding window size (0 = full attention). Used by Qwen/Mistral.
    pub sliding_window: usize,
    /// Number of MoE experts (0 = dense). Used by Mixtral.
    pub num_experts: usize,
    /// Number of active MoE experts per token. Used by Mixtral.
    pub num_experts_per_tok: usize,
    /// Per-expert FFN intermediate width. Differs from `intermediate_size` in
    /// LFM2MoE (experts 1792 vs dense 7168). 0 = fall back to intermediate_size.
    pub expert_intermediate_size: usize,
    /// Alibi number of heads for slope computation (0 = not used).
    pub alibi_num_heads: usize,
    /// LFM2 short-convolution cache length / kernel width (0 = no shortconv).
    pub shortconv_l_cache: usize,
    /// Number of leading dense FFN blocks before MoE begins (LFM2MoE/DeepSeek).
    pub leading_dense_layers: usize,
    /// MoE router uses sigmoid gating with a per-layer expert bias (LFM2MoE),
    /// instead of softmax. The bias is added for selection only; weights are the
    /// raw sigmoid scores, renormalized over the selected experts.
    pub expert_gating_sigmoid: bool,
    /// Number of head dimensions that receive RoPE rotation (0 = full head_dim).
    /// Models like MiniMax-M2 use partial RoPE: first `rope_dim` dims are rotated,
    /// remaining dims (NoPE) are left unchanged.
    pub rope_dim: usize,
    /// RoPE theta for sliding-window (local) attention layers (0 = use `rope_theta`
    /// for all layers). Gemma 3/4 use 10000 on local layers and 1000000 on global.
    pub rope_theta_swa: f32,
    /// Interleaving period for global attention layers when sliding window is in
    /// use: every `sliding_window_pattern`-th layer is global, the rest are local
    /// SWA (0 = no interleaving). Gemma 2 = 2, Gemma 3/4 = 6.
    pub sliding_window_pattern: usize,
    /// Multiplier applied to token embeddings after lookup (1.0 = none).
    /// Gemma scales by sqrt(hidden_size).
    pub embedding_scale: f32,
    /// Use GeGLU (tanh-GELU) instead of SwiGLU (SiLU) for dense FFN. Gemma uses GeGLU.
    pub gelu_ffn: bool,
    /// Gemma "sandwich" normalization: a post-attention norm is applied to the
    /// attention output and a post-FFN norm to the FFN output, each *before* the
    /// residual add (in addition to the standard pre-attention / pre-FFN norms).
    pub sandwich_norm: bool,
    /// Qwen-style RMSNorm scales by `(1 + weight)` instead of `weight` alone.
    pub rms_norm_weight_plus_one: bool,
    /// Number of appended multi-token-prediction (MTP / nextn) draft layers.
    /// These layers live after the causal backbone in GGUF (`blk.N.nextn.*`) and
    /// are not counted in `layer_count`.
    pub nextn_predict_layers: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            context_size: 4096,
            layer_count: 32,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            key_value_head_dim: 0,
            kv_cache_dtype: DType::F32,
            kv_quantization: Default::default(),
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            architecture: ModelArchitecture::Llama,
            sliding_window: 0,
            num_experts: 0,
            num_experts_per_tok: 0,
            expert_intermediate_size: 0,
            alibi_num_heads: 0,
            shortconv_l_cache: 0,
            leading_dense_layers: 0,
            expert_gating_sigmoid: false,
            rope_dim: 0,
            rope_theta_swa: 0.0,
            sliding_window_pattern: 0,
            embedding_scale: 1.0,
            gelu_ffn: false,
            sandwich_norm: false,
            rms_norm_weight_plus_one: false,
            nextn_predict_layers: 0,
        }
    }
}

impl InferenceConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Effective RoPE dimension: how many elements per head receive rotation.
    /// Defaults to `kv_head_dim()` when `rope_dim` is 0 (full-head RoPE).
    pub fn effective_rope_dim(&self) -> usize {
        if self.rope_dim > 0 {
            self.rope_dim.min(self.kv_head_dim())
        } else {
            self.kv_head_dim()
        }
    }

    pub fn kv_head_dim(&self) -> usize {
        if self.key_value_head_dim > 0 {
            self.key_value_head_dim
        } else {
            self.head_dim()
        }
    }

    /// Whether layer `layer_idx` uses global (full-context) attention rather than
    /// sliding-window attention.
    ///
    /// - No sliding window configured (`sliding_window == 0`): every layer is global.
    /// - Uniform-SWA models (Mistral/Qwen) set `sliding_window > 0` but leave
    ///   `sliding_window_pattern == 0`, meaning *every* layer is local (SWA applies
    ///   to all layers); none are global.
    /// - Gemma interleaving (`sliding_window_pattern > 0`): every `pattern`-th layer
    ///   (1-indexed) is global, the rest are local.
    pub fn layer_is_global(&self, layer_idx: usize) -> bool {
        if self.sliding_window == 0 {
            return true;
        }
        if self.sliding_window_pattern == 0 {
            return false;
        }
        (layer_idx + 1).is_multiple_of(self.sliding_window_pattern)
    }

    /// RoPE theta for `layer_idx`: global layers use `rope_theta`; sliding-window
    /// layers use `rope_theta_swa` when set, otherwise fall back to `rope_theta`.
    pub fn layer_rope_theta(&self, layer_idx: usize) -> f32 {
        if self.rope_theta_swa > 0.0 && !self.layer_is_global(layer_idx) {
            self.rope_theta_swa
        } else {
            self.rope_theta
        }
    }

    /// Effective sliding-window size for `layer_idx` (0 = full attention).
    pub fn layer_sliding_window(&self, layer_idx: usize) -> usize {
        if self.sliding_window > 0 && !self.layer_is_global(layer_idx) {
            self.sliding_window
        } else {
            0
        }
    }

    /// Map `general.architecture` values to the GGUF metadata key prefix.
    fn gguf_metadata_prefix(arch: &str) -> &str {
        match arch {
            "qwen3_5_moe_text" | "qwen3_5_moe" | "qwen35moe" | "qwen3_5" | "qwen3_5_text"
            | "qwen35_text" => "qwen35",
            other => other,
        }
    }

    /// Build an InferenceConfig from a mapped GGUF file by reading metadata under
    /// the actual architecture prefix (e.g. `qwen3.*`, `llama.*`, `gemma3.*`).
    /// Falls back to weight tensor dimensions when metadata is missing.
    pub fn from_gguf(mapped: &MappedGgufFile) -> Self {
        let metadata = &mapped.parsed().metadata;
        let raw_arch = mapped
            .parsed()
            .architecture()
            .unwrap_or("llama")
            .to_string();
        let architecture = ModelArchitecture::from_gguf(mapped);

        let metadata_prefix = Self::gguf_metadata_prefix(&raw_arch);
        // Canonicalize the arch string so downstream behavior matches (RMSNorm
        // (1+w), GDN detection, etc.) see `qwen35` even for `qwen3_5_text`.
        let arch = metadata_prefix.to_string();
        let key = |suffix: &str| format!("{metadata_prefix}.{suffix}");
        let arch_u32 = |suffix: &str| {
            metadata_u32_lookup(metadata, &key(suffix)).or_else(|| {
                if metadata_prefix == arch {
                    None
                } else {
                    metadata_u32_lookup(metadata, &format!("{arch}.{suffix}"))
                }
            })
        };
        let arch_f32 = |suffix: &str| {
            metadata_f32_lookup(metadata, &key(suffix)).or_else(|| {
                if metadata_prefix == arch {
                    None
                } else {
                    metadata_f32_lookup(metadata, &format!("{arch}.{suffix}"))
                }
            })
        };

        let token_embd_dims = first_tensor_dims(mapped, "tok_embeddings.weight")
            .or_else(|| first_tensor_dims(mapped, "token_embd.weight"));

        let hidden_size = arch_u32("embedding_length")
            .or_else(|| {
                token_embd_dims.as_ref().and_then(|d| match d.len() {
                    0 => None,
                    1 => d.first().copied().map(|v| v as u32),
                    _ => d.get(1).copied().map(|v| v as u32),
                })
            })
            .map(|v| v as usize)
            .unwrap_or(4096);

        let vocab_size = arch_u32("vocab_size")
            .or_else(|| metadata_u32_lookup(metadata, "general.vocab_size"))
            .or_else(|| metadata_u32_lookup(metadata, "tokenizer.ggml.tokens.count"))
            .or_else(|| {
                token_embd_dims.as_ref().and_then(|d| match d.len() {
                    0 | 1 => None,
                    // GGUF dim order differs between writers (llama.cpp files
                    // store hidden-first, oxidize-converted files vocab-first);
                    // the vocab axis is always the larger of the two.
                    _ => d.iter().copied().max().map(|v| v as u32),
                })
            })
            .map(|v| v as usize)
            .unwrap_or(32000);

        let context_size = arch_u32("context_length")
            .map(|v| v as usize)
            .unwrap_or(4096);

        // Multi-token-prediction (MTP/nextn) layers are appended after the main
        // stack (e.g. qwen35 blk.64 with nextn.* tensors); they are draft heads,
        // not part of the causal backbone, so exclude them from layer_count.
        let nextn_layers = arch_u32("nextn_predict_layers").unwrap_or(0) as usize;
        let layer_count =
            (arch_u32("block_count").unwrap_or(32) as usize).saturating_sub(nextn_layers);

        let intermediate_size = arch_u32("feed_forward_length")
            .map(|v| v as usize)
            .or_else(|| {
                first_layer_tensor_dims(mapped, "ffn_gate.weight")
                    .or_else(|| first_layer_tensor_dims(mapped, "ffn_up.weight"))
                    .and_then(|d| d.get(1).copied())
                    .map(|v| v as usize)
            })
            .unwrap_or(11008);

        let num_attention_heads = arch_u32("attention.head_count").unwrap_or(32) as usize;

        // GQA: KV heads. Default to num_attention_heads only when key is absent
        // AND we can't infer from attn_k dims.
        // head_count_kv may be a scalar (most archs) or a per-layer array (LFM2,
        // where shortconv layers report 0). Take the largest attention layer.
        let attn_k_out = first_layer_tensor_dims(mapped, "attn_k.weight")
            .or_else(|| {
                mapped
                    .mapped_tensor_infos()
                    .iter()
                    .find(|t| t.name.ends_with(".attn_k.weight"))
                    .map(|t| t.dimensions.clone())
            })
            .and_then(|d| d.get(1).copied());
        let head_dim_guess = hidden_size.checked_div(num_attention_heads).unwrap_or(0);
        let num_key_value_heads = arch_u32("attention.head_count_kv")
            .map(|v| v as usize)
            .or_else(|| {
                metadata_u32_array_max(metadata, &key("attention.head_count_kv"))
                    .map(|v| v as usize)
            })
            .filter(|&v| v > 0)
            // Fall back to inferring KV head count from the attention-layer K
            // projection width and the (uniform) head dim, then to MHA.
            .or_else(|| {
                attn_k_out
                    .zip((head_dim_guess > 0).then_some(head_dim_guess))
                    .and_then(|(w, hd)| (w as usize).checked_div(hd))
                    .filter(|&v| v > 0)
            })
            .unwrap_or(num_attention_heads);

        // Per-head dim for K (and V). Prefer explicit key_length; otherwise
        // infer from attn_k_out / num_kv_heads. Falls back to hidden/n_heads.
        let mut key_value_head_dim = arch_u32("attention.key_length")
            .map(|v| v as usize)
            .or_else(|| {
                attn_k_out.and_then(|width| (width as usize).checked_div(num_key_value_heads))
            })
            .unwrap_or_else(|| hidden_size.checked_div(num_attention_heads).unwrap_or(0));
        if architecture.uses_mla() {
            let mla_k = arch_u32("attention.key_length_mla")
                .map(|v| v as usize)
                .or_else(|| {
                    first_layer_tensor_dims(mapped, "attn_q_b.weight")
                        .and_then(|d| d.get(1).copied())
                        .map(|w| (w as usize) / num_attention_heads.max(1))
                });
            if let Some(k) = mla_k.filter(|&k| k > 0) {
                key_value_head_dim = k;
            }
        }

        let rms_norm_eps = arch_f32("attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
        let mut rope_theta = arch_f32("rope.freq_base").unwrap_or(10000.0);
        if matches!(
            arch.as_str(),
            "qwen3_5_moe_text" | "qwen3_5_moe" | "qwen35moe" | "qwen3_5" | "qwen35"
        ) && rope_theta <= 10000.0
        {
            // HF stores theta under rope_parameters; older GGUF converts may omit rope.freq_base.
            rope_theta = 10_000_000.0;
        }
        let sliding_window = arch_u32("attention.sliding_window")
            .map(|v| v as usize)
            .unwrap_or(0);
        let mut num_experts = arch_u32("expert_count")
            .or_else(|| metadata_u32_lookup(metadata, "expert_count"))
            .map(|v| v as usize)
            .unwrap_or(0);
        if num_experts == 0 {
            num_experts = mapped
                .mapped_tensor_infos()
                .iter()
                .find(|t| t.name == "blk.1.ffn_gate_inp.weight")
                .and_then(|t| t.dimensions.get(1).copied())
                .unwrap_or(0) as usize;
        }
        let mut num_experts_per_tok = arch_u32("expert_used_count")
            .or_else(|| metadata_u32_lookup(metadata, "expert_used_count"))
            .map(|v| v as usize)
            .unwrap_or(0);
        if num_experts_per_tok == 0 && architecture.uses_mla() && num_experts > 0 {
            num_experts_per_tok = 8;
        }
        let mut expert_intermediate_size = arch_u32("expert_feed_forward_length")
            .or_else(|| metadata_u32_lookup(metadata, "expert_feed_forward_length"))
            .map(|v| v as usize)
            .unwrap_or(0);
        if expert_intermediate_size == 0 {
            expert_intermediate_size = first_layer_tensor_dims(mapped, "ffn_gate_shexp.weight")
                .or_else(|| {
                    mapped
                        .mapped_tensor_infos()
                        .iter()
                        .find(|t| t.name.ends_with(".ffn_gate_shexp.weight"))
                        .map(|t| t.dimensions.clone())
                })
                .and_then(|d| d.get(1).copied())
                .map(|v| v as usize)
                .unwrap_or(0);
        }

        // LFM2 short-convolution cache length (kernel width). Present for lfm2/lfm2moe.
        let shortconv_l_cache = arch_u32("shortconv.l_cache")
            .map(|v| v as usize)
            .unwrap_or(0);
        // Leading dense FFN blocks before MoE (lfm2moe, deepseek-style).
        let leading_dense_layers = arch_u32("leading_dense_block_count")
            .map(|v| v as usize)
            .unwrap_or(0);
        // expert_gating_func: 1 = softmax, 2 = sigmoid (lfm2moe uses sigmoid).
        let expert_gating_sigmoid = arch_u32("expert_gating_func")
            .or_else(|| metadata_u32_lookup(metadata, "expert_gating_func"))
            .map(|v| v == 2)
            .unwrap_or(false);

        // Partial RoPE: number of head dimensions that receive rotation.
        // 0 means "use full kv_head_dim" (standard). MiniMax-M2 uses 64 of 128.
        let mut rope_dim = arch_u32("rope.dimension_count")
            .map(|v| v as usize)
            .unwrap_or(0);
        if rope_dim == 0
            && matches!(
                arch.as_str(),
                "qwen3_5_moe_text" | "qwen3_5_moe" | "qwen35moe" | "qwen3_5" | "qwen35"
            )
            && key_value_head_dim > 0
        {
            // Qwen3.5 partial_rotary_factor=0.25 on 256-dim heads.
            rope_dim = key_value_head_dim / 4;
        }

        // ---- Gemma-family specifics ----
        // Gemma 2/3/4 interleave local sliding-window and global attention layers,
        // use dual RoPE theta, scale embeddings by sqrt(hidden), use GeGLU, and
        // apply sandwich normalization. All gated behind the Gemma architecture so
        // other models are unaffected.
        let is_gemma = architecture == ModelArchitecture::Gemma;
        let sliding_window_pattern = if is_gemma {
            // GGUF may record it explicitly; otherwise fall back to the known
            // per-generation default (Gemma 2 = every 2nd, Gemma 3/4 = every 6th).
            arch_u32("attention.sliding_window_pattern")
                .map(|v| v as usize)
                .filter(|&v| v > 0)
                .unwrap_or(match arch.as_str() {
                    "gemma2" => 2,
                    _ => 6,
                })
        } else {
            0
        };
        let rope_theta_swa = if is_gemma {
            arch_f32("rope.freq_base_swa")
                .filter(|&v| v > 0.0)
                .unwrap_or(10000.0)
        } else {
            0.0
        };
        let embedding_scale = if is_gemma {
            (hidden_size as f32).sqrt()
        } else {
            1.0
        };
        let gelu_ffn = is_gemma;
        let sandwich_norm = is_gemma;
        // Only the Qwen3.5/GDN lineage uses the Gemma-style (1 + w) RMSNorm
        // convention. Standard Qwen2/Qwen3/qwen3moe use plain w * x_hat —
        // keying this on the whole Qwen family garbled every official Qwen
        // GGUF in code paths that honor the flag (layer-wise).
        let mut rms_norm_weight_plus_one = matches!(
            arch.as_str(),
            "qwen35" | "qwen35moe" | "qwen3_5_moe" | "qwen3_5_moe_text"
        );
        // Temp override to verify the baked-vs-raw (1+w) hypothesis.
        if let Ok(v) = std::env::var("OXIDIZE_RMS_PLUS_ONE") {
            rms_norm_weight_plus_one = v != "0";
        }

        Self {
            vocab_size,
            context_size,
            layer_count,
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_key_value_heads,
            key_value_head_dim,
            kv_cache_dtype: DType::F32,
            kv_quantization: crate::kv_cache::KvQuantization::default(),
            rms_norm_eps,
            rope_theta,
            architecture,
            sliding_window,
            num_experts,
            num_experts_per_tok,
            expert_intermediate_size,
            alibi_num_heads: 0,
            shortconv_l_cache,
            leading_dense_layers,
            expert_gating_sigmoid,
            rope_dim,
            rope_theta_swa,
            sliding_window_pattern,
            embedding_scale,
            gelu_ffn,
            sandwich_norm,
            rms_norm_weight_plus_one,
            nextn_predict_layers: nextn_layers,
        }
    }
}

fn metadata_u32_lookup(
    metadata: &std::collections::BTreeMap<String, crate::gguf::GgufMetadataValue>,
    key: &str,
) -> Option<u32> {
    use crate::gguf::GgufMetadataValue;
    match metadata.get(key) {
        Some(GgufMetadataValue::Uint8(v)) => Some((*v).into()),
        Some(GgufMetadataValue::Uint16(v)) => Some((*v).into()),
        Some(GgufMetadataValue::Uint32(v)) => Some(*v),
        Some(GgufMetadataValue::Uint64(v)) => (*v).try_into().ok(),
        Some(GgufMetadataValue::Int8(v)) if *v >= 0 => Some((*v as u8).into()),
        Some(GgufMetadataValue::Int16(v)) if *v >= 0 => Some((*v as u16).into()),
        Some(GgufMetadataValue::Int32(v)) if *v >= 0 => (*v).try_into().ok(),
        Some(GgufMetadataValue::Int64(v)) if *v >= 0 => (*v).try_into().ok(),
        _ => None,
    }
}

/// Largest integer value in an array-typed metadata field. Used for LFM2's
/// per-layer `attention.head_count_kv` (0 on shortconv layers, KV count on
/// attention layers) — the max gives the actual attention KV-head count.
fn metadata_u32_array_max(
    metadata: &std::collections::BTreeMap<String, crate::gguf::GgufMetadataValue>,
    key: &str,
) -> Option<u32> {
    use crate::gguf::GgufMetadataValue;
    let arr = match metadata.get(key) {
        Some(GgufMetadataValue::Array(a)) => a,
        _ => return None,
    };
    arr.values
        .iter()
        .filter_map(|v| match v {
            GgufMetadataValue::Uint8(x) => Some((*x).into()),
            GgufMetadataValue::Uint16(x) => Some((*x).into()),
            GgufMetadataValue::Uint32(x) => Some(*x),
            GgufMetadataValue::Uint64(x) => (*x).try_into().ok(),
            GgufMetadataValue::Int8(x) if *x >= 0 => Some((*x as u8).into()),
            GgufMetadataValue::Int16(x) if *x >= 0 => Some((*x as u16).into()),
            GgufMetadataValue::Int32(x) if *x >= 0 => (*x).try_into().ok(),
            GgufMetadataValue::Int64(x) if *x >= 0 => (*x).try_into().ok(),
            _ => None,
        })
        .max()
}

fn metadata_f32_lookup(
    metadata: &std::collections::BTreeMap<String, crate::gguf::GgufMetadataValue>,
    key: &str,
) -> Option<f32> {
    use crate::gguf::GgufMetadataValue;
    match metadata.get(key) {
        Some(GgufMetadataValue::Float32(v)) => Some(*v),
        Some(GgufMetadataValue::Float64(v)) => Some(*v as f32),
        Some(GgufMetadataValue::Int8(v)) => Some(*v as f32),
        Some(GgufMetadataValue::Int16(v)) => Some(*v as f32),
        Some(GgufMetadataValue::Int32(v)) => Some(*v as f32),
        Some(GgufMetadataValue::Int64(v)) => Some(*v as f32),
        Some(GgufMetadataValue::Uint8(v)) => Some(*v as f32),
        Some(GgufMetadataValue::Uint16(v)) => Some(*v as f32),
        Some(GgufMetadataValue::Uint32(v)) => Some(*v as f32),
        Some(GgufMetadataValue::Uint64(v)) => Some(*v as f32),
        _ => None,
    }
}

fn first_tensor_dims(mapped: &MappedGgufFile, name: &str) -> Option<Vec<u64>> {
    mapped
        .mapped_tensor_infos()
        .iter()
        .find(|t| t.name == name)
        .map(|t| t.dimensions.clone())
}

fn first_layer_tensor_dims(mapped: &MappedGgufFile, suffix: &str) -> Option<Vec<u64>> {
    mapped
        .mapped_tensor_infos()
        .iter()
        .find(|t| t.name.starts_with("blk.0.") && t.name.ends_with(suffix))
        .map(|t| t.dimensions.clone())
}

/// Pre-allocated scratch buffers reused across tokens and layers to eliminate
/// per-token `Vec<f32>` allocations in the hot decode path.
#[derive(Debug, Clone, PartialEq)]
pub struct Workspace {
    // Persistent activation across layers.
    pub x: Vec<f32>,
    // Hidden-size scratch (RMSNorm output, attention output, FFN output, etc.)
    pub hidden_a: Vec<f32>,
    pub hidden_b: Vec<f32>,
    // Intermediate-size scratch (gate, up, SwiGLU)
    pub intermediate_a: Vec<f32>,
    pub intermediate_b: Vec<f32>,
    pub intermediate_c: Vec<f32>,
    // Q/K/V projection scratch — sized to max Q/K/V length.
    pub q_full: Vec<f32>,
    pub k_vec: Vec<f32>,
    pub v_vec: Vec<f32>,
    // Attention result scratch.
    pub attn_result: Vec<f32>,
    // Q scratch for flash-attention when Q heads must be truncated.
    pub flash_q: Vec<f32>,
    // Per-head scratch (flash-attn output, RoPE, per-head norm).
    pub head_scratch: Vec<f32>,
    // KV cache copy fallback buffers.
    pub kv_keys_copy: Vec<f32>,
    pub kv_values_copy: Vec<f32>,
    // Final logits buffer.
    pub logits: Vec<f32>,
    // MoE router scratch reused across layers.
    pub moe_router_logits: Vec<f32>,
    pub moe_expert_scores: Vec<(usize, f32)>,
    // Mamba/SSM scratch.
    pub mamba_scratch: Vec<f32>,
    pub conv_out: Vec<f32>,
    // LFM2 short-convolution scratch (reused per layer; avoids 3 allocs/token).
    pub shortconv_bcx: Vec<f32>,
    pub shortconv_bx: Vec<f32>,
    // Batched MoE expert outputs (gate/up/down for top-k experts).
    pub moe_gate_all: Vec<f32>,
    pub moe_up_all: Vec<f32>,
    pub moe_down_all: Vec<f32>,
}

impl Workspace {
    pub fn for_config(config: &InferenceConfig) -> Self {
        let h = config.hidden_size;
        let inter = config.intermediate_size;
        let expert_inter = config.expert_intermediate_size.max(inter);
        let max_kv_len = config.num_key_value_heads * config.kv_head_dim();
        let mla_head_dim = config.kv_head_dim().max(config.head_dim());
        let mla_storage = if config.architecture.uses_mla() {
            config.num_attention_heads.saturating_mul(mla_head_dim)
        } else {
            0
        };
        let max_qkv = (h * 3).max(inter).max(expert_inter).max(mla_storage);
        let kv_scratch = max_kv_len.max(mla_storage);
        let head_dim = config.head_dim().max(config.kv_head_dim()).max(192);
        let kv_copy_size = config.context_size * max_kv_len;
        let n_experts_per_tok = config.num_experts_per_tok.max(1);

        Self {
            x: vec![0.0_f32; h],
            hidden_a: vec![0.0_f32; h],
            hidden_b: vec![0.0_f32; h],
            intermediate_a: vec![0.0_f32; inter.max(expert_inter)],
            intermediate_b: vec![0.0_f32; inter.max(expert_inter)],
            intermediate_c: vec![0.0_f32; inter.max(expert_inter)],
            q_full: vec![0.0_f32; max_qkv],
            k_vec: vec![0.0_f32; kv_scratch],
            v_vec: vec![0.0_f32; kv_scratch],
            attn_result: vec![0.0_f32; max_qkv],
            flash_q: vec![0.0_f32; max_qkv.max(64)],
            head_scratch: vec![0.0_f32; head_dim],
            kv_keys_copy: vec![0.0_f32; kv_copy_size],
            kv_values_copy: vec![0.0_f32; kv_copy_size],
            logits: vec![0.0_f32; config.vocab_size],
            moe_router_logits: vec![0.0_f32; config.num_experts.max(1)],
            moe_expert_scores: vec![(0, 0.0_f32); config.num_experts.max(1)],
            mamba_scratch: vec![0.0_f32; h.max(576)],
            conv_out: vec![0.0_f32; max_qkv],
            shortconv_bcx: vec![0.0_f32; h * 3],
            shortconv_bx: vec![0.0_f32; h],
            moe_gate_all: vec![0.0_f32; n_experts_per_tok * expert_inter],
            moe_up_all: vec![0.0_f32; n_experts_per_tok * expert_inter],
            moe_down_all: vec![0.0_f32; n_experts_per_tok * h],
        }
    }
}

/// Fixed-size ring buffer for LFM2 shortconv / Mamba conv history (O(1) push).
#[derive(Debug, Clone, PartialEq)]
struct ConvHistoryRing {
    slots: Vec<f32>,
    dim: usize,
    capacity: usize,
    head: usize,
    len: usize,
}

impl ConvHistoryRing {
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

#[derive(Debug, Clone)]
pub enum WeightStorage {
    F32(Vec<f32>),
    Quantized(GgufQuantizationType, Vec<u8>),
    /// Zero-copy mmap-backed quantized weights: (qtype, mmap Arc, offset, size).
    /// Keeps weights on disk via page cache; dequantization happens on-the-fly.
    MmapQuantized(GgufQuantizationType, Arc<Mmap>, usize, usize),
}

impl PartialEq for WeightStorage {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (WeightStorage::F32(a), WeightStorage::F32(b)) => a == b,
            (WeightStorage::Quantized(qa, a), WeightStorage::Quantized(qb, b)) => {
                qa == qb && a == b
            }
            (
                WeightStorage::MmapQuantized(qa, _, oa, sa),
                WeightStorage::MmapQuantized(qb, _, ob, sb),
            ) => qa == qb && oa == ob && sa == sb,
            _ => false,
        }
    }
}

impl Default for WeightStorage {
    fn default() -> Self {
        WeightStorage::F32(Vec::new())
    }
}

impl WeightStorage {
    pub fn is_empty(&self) -> bool {
        match self {
            WeightStorage::F32(v) => v.is_empty(),
            WeightStorage::Quantized(_, v) => v.is_empty(),
            WeightStorage::MmapQuantized(_, _, _, size) => *size == 0,
        }
    }

    pub fn output_dim(&self, input_dim: usize) -> usize {
        match self {
            WeightStorage::F32(v) => v.len() / input_dim,
            WeightStorage::Quantized(qtype, v) => {
                let (block_width, block_size) = weight_block_info(*qtype);
                let bytes_per_row = (input_dim / block_width) * block_size;
                if bytes_per_row == 0 {
                    return 0;
                }
                v.len() / bytes_per_row
            }
            WeightStorage::MmapQuantized(qtype, _, _, size) => {
                let (block_width, block_size) = weight_block_info(*qtype);
                let bytes_per_row = (input_dim / block_width) * block_size;
                if bytes_per_row == 0 {
                    return 0;
                }
                size / bytes_per_row
            }
        }
    }
}

fn weight_block_info(qtype: GgufQuantizationType) -> (usize, usize) {
    match qtype {
        GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M => (256, 144),
        GgufQuantizationType::Q6_K => (256, 210),
        GgufQuantizationType::Q8_0 => (32, 34),
        GgufQuantizationType::NVFP4 => (64, 36),
        GgufQuantizationType::IQ1_S => (256, 50),
        GgufQuantizationType::IQ1_M => (256, 56),
        _ => (1, 4), // fallback to f32
    }
}

/// Borrow a quantized weight tensor's raw bytes for the batched expert GEMV.
/// Returns `None` for f32 weights (which use the per-expert fallback path).
fn expert_matrix(weight: &WeightStorage) -> Option<(GgufQuantizationType, &[u8])> {
    match weight {
        WeightStorage::Quantized(qtype, data) => Some((*qtype, data.as_slice())),
        WeightStorage::MmapQuantized(qtype, mmap, offset, size) => {
            Some((*qtype, &mmap[*offset..*offset + *size]))
        }
        WeightStorage::F32(_) => None,
    }
}

fn gemv_expert_weight(
    storage: &WeightStorage,
    expert_idx: usize,
    n_experts: usize,
    rows: usize,
    cols: usize,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), String> {
    let values_per_expert = rows * cols;
    match storage {
        WeightStorage::F32(data) => {
            let start = expert_idx * values_per_expert;
            let end = start + values_per_expert;
            if end > data.len() {
                return Err(format!(
                    "expert {} out of range: need {} values, have {}",
                    expert_idx,
                    n_experts * values_per_expert,
                    data.len()
                ));
            }
            gemv_f32(&data[start..end], rows, cols, input, output).map_err(|e| format!("{:?}", e))
        }
        WeightStorage::Quantized(qtype, data) => {
            let (block_width, block_size) = weight_block_info(*qtype);
            let blocks_per_row = cols / block_width;
            let values_per_expert_blocks = rows * blocks_per_row;
            let expert_size = values_per_expert_blocks * block_size;
            let start = expert_idx * expert_size;
            let end = start + expert_size;
            if end > data.len() {
                return Err(format!(
                    "expert {} out of range for quant {:?}: need {} bytes, have {}",
                    expert_idx,
                    qtype,
                    n_experts * expert_size,
                    data.len()
                ));
            }
            gemv_quantized_f32(*qtype, &data[start..end], rows, cols, input, output)
                .map_err(|e| format!("{:?}", e))
        }
        WeightStorage::MmapQuantized(qtype, mmap, base_offset, total_size) => {
            let (block_width, block_size) = weight_block_info(*qtype);
            let blocks_per_row = cols / block_width;
            let values_per_expert_blocks = rows * blocks_per_row;
            let expert_size = values_per_expert_blocks * block_size;
            let start = *base_offset + expert_idx * expert_size;
            let end = start + expert_size;
            let mmap_total = *base_offset + *total_size;
            if end > mmap_total {
                return Err(format!(
                    "expert {} out of range for mmap quant {:?}: need offset {}, have {}",
                    expert_idx, qtype, end, mmap_total
                ));
            }
            gemv_quantized_f32(*qtype, &mmap[start..end], rows, cols, input, output)
                .map_err(|e| format!("{:?}", e))
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
    // GGUF stores linear weights in natural row-major layout: `rows` (output
    // features) of `cols` (input features) contiguous floats/quantized blocks.
    // For `y = W @ x` we use the fused natural GEMV: `output[j] = sum_i W[j, i] * input[i]`.
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

/// Run several same-input projections (q/k/v, gate/up) as ONE fused parallel
/// region via [`gemv_quantized_multi_f32`]. Entries with `rows == 0` are
/// skipped; F32-stored weights run as sequential [`gemv_weight`] calls after
/// the fused region (rare: quantized models keep only norms in f32).
fn gemv_weight_fused(
    parts: Vec<(&WeightStorage, usize, &mut [f32])>,
    cols: usize,
    input: &[f32],
) -> Result<(), String> {
    let mut jobs: Vec<GemvJob<'_>> = Vec::with_capacity(parts.len());
    let mut serial: Vec<(&WeightStorage, usize, &mut [f32])> = Vec::new();
    for (storage, rows, output) in parts {
        if rows == 0 {
            continue;
        }
        match storage {
            WeightStorage::Quantized(qtype, data) => jobs.push(GemvJob {
                quantization: *qtype,
                matrix: data,
                rows,
                output,
            }),
            WeightStorage::MmapQuantized(qtype, mmap, offset, size) => jobs.push(GemvJob {
                quantization: *qtype,
                matrix: &mmap[*offset..*offset + *size],
                rows,
                output,
            }),
            WeightStorage::F32(_) => serial.push((storage, rows, output)),
        }
    }
    gemv_quantized_multi_f32(&mut jobs, cols, input).map_err(|e| format!("{:?}", e))?;
    for (storage, rows, output) in serial {
        gemv_weight(storage, rows, cols, input, output)?;
    }
    Ok(())
}

/// Add a per-row bias (repeating modulo `bias.len()` when shorter than a row)
/// to every position of a `[batch, row]`-style buffer. Used to apply attention
/// biases across all batch tokens after a batched GEMM.
fn add_repeating_bias(buf: &mut [f32], bias: &[f32]) {
    if bias.is_empty() {
        return;
    }
    let bl = bias.len();
    for (i, x) in buf.iter_mut().enumerate() {
        *x += bias[i % bl];
    }
}

/// Batched GEMM against a model weight. `inputs` is `[batch, cols]` row-major
/// and `outputs` is `[batch, rows]` row-major (same natural GGUF layout as
/// [`gemv_weight`]). For quantized weights this reaches the decode-once fast
/// path that amortizes one dequantization across all batch tokens — the main
/// reason batched prefill is much faster than a `forward_single` loop.
fn gemm_weight(
    storage: &WeightStorage,
    rows: usize,
    cols: usize,
    inputs: &[f32],
    outputs: &mut [f32],
    batch: usize,
) -> Result<(), String> {
    if batch == 0 {
        return Ok(());
    }
    if batch == 1 {
        return gemv_weight(storage, rows, cols, inputs, outputs);
    }
    match storage {
        WeightStorage::F32(data) => {
            // No batched fp32 GEMM in the natural layout; fall back to a per-batch
            // GEMV loop. F32 weights are rare in practice (quantized models only
            // hold the norms as f32, and those don't go through this path).
            for b in 0..batch {
                let in_slice = &inputs[b * cols..(b + 1) * cols];
                let out_slice = &mut outputs[b * rows..(b + 1) * rows];
                gemv_f32(data, rows, cols, in_slice, out_slice).map_err(|e| format!("{:?}", e))?;
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

#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct LayerWeights {
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
    /// Gemma post-feedforward norm (applied to FFN output before the residual add).
    post_ffn_norm: Vec<f32>,
    // Dense FFN weights (used when num_experts == 0)
    ffn_gate: WeightStorage,
    ffn_up: WeightStorage,
    ffn_down: WeightStorage,
    ffn_down_bias: Vec<f32>,
    // MoE expert weights (used when num_experts > 0)
    // Shape: [n_experts, intermediate_size, hidden_size] for gate/up
    //         [n_experts, hidden_size, intermediate_size] for down
    ffn_gate_exps: WeightStorage,
    ffn_up_exps: WeightStorage,
    ffn_down_exps: WeightStorage,
    // MoE router: [hidden_size, n_experts]
    ffn_gate_inp: WeightStorage,
    attn_qkv: WeightStorage,
    // SSM / Mamba tensors
    attn_gate: WeightStorage,
    ssm_a: Vec<f32>,
    ssm_alpha: Vec<f32>,
    ssm_beta: Vec<f32>,
    ssm_conv1d: Vec<f32>,
    ssm_dt_bias: Vec<f32>,
    ssm_norm: Vec<f32>,
    ssm_out: WeightStorage,
    // Per-head norms for standard attention
    attn_q_norm: Vec<f32>,
    attn_k_norm: Vec<f32>,
    // LFM2 short-convolution operator (token mixing on non-attention layers).
    // in_proj: [hidden] -> [3*hidden] producing (B, C, x); conv: depthwise
    // causal conv1d weights laid out [l_cache, hidden]; out_proj: [hidden]->[hidden].
    shortconv_in_proj: WeightStorage,
    shortconv_conv: Vec<f32>,
    shortconv_out_proj: WeightStorage,
    // LFM2MoE per-layer expert routing bias (exp_probs_b), added to sigmoid
    // scores for top-k selection only.
    ffn_exp_probs_b: Vec<f32>,
    // DeepSeek2 MLA compressed attention (Kimi K2.x).
    mla_q_a: WeightStorage,
    mla_q_a_norm: Vec<f32>,
    mla_q_b: WeightStorage,
    mla_kv_a_mqa: WeightStorage,
    mla_kv_a_norm: Vec<f32>,
    mla_k_b: WeightStorage,
    mla_v_b: WeightStorage,
    // DeepSeek MoE shared expert (shexp) branch.
    ffn_gate_shexp: WeightStorage,
    ffn_up_shexp: WeightStorage,
    ffn_down_shexp: WeightStorage,
}

/// Qwen3.5/Qwen3.6-style in-model MTP (`nextn`) draft block.
///
/// GGUF stores one extra decoder block after the target stack (`blk.N.*`) plus
/// the `blk.N.nextn.*` fusion/head tensors. The regular block weights are kept
/// in `layer`; the extra tensors combine a token embedding and the target hidden
/// state, then project the MTP hidden state back through a shared or dedicated
/// output head.
#[derive(Debug, Clone, PartialEq, Default)]
struct MtpWeights {
    layer: LayerWeights,
    eh_proj: WeightStorage,
    enorm: Vec<f32>,
    hnorm: Vec<f32>,
    embed_tokens: WeightStorage,
    shared_head_norm: Vec<f32>,
    shared_head_head: WeightStorage,
}

impl MtpWeights {
    fn is_usable(&self, config: &InferenceConfig) -> bool {
        let h = config.hidden_size;
        !self.eh_proj.is_empty()
            && self.eh_proj.output_dim(h.saturating_mul(2)) == h
            && self.enorm.len() == h
            && self.hnorm.len() == h
            && !self.layer.attn_norm.is_empty()
            && !self.layer.attn_q.is_empty()
            && !self.layer.attn_k.is_empty()
            && !self.layer.attn_v.is_empty()
            && !self.layer.attn_output.is_empty()
            && !self.layer.ffn_gate.is_empty()
            && !self.layer.ffn_up.is_empty()
            && !self.layer.ffn_down.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct InferenceModel {
    config: InferenceConfig,
    tok_embeddings: WeightStorage,
    tok_embeddings_cols: usize,
    norm_weight: Vec<f32>,
    output_weight: WeightStorage,
    layers: Vec<LayerWeights>,
    mtp: Option<MtpWeights>,
    kv_cache: KvCache,
    /// Maps absolute layer index → KV cache layer index for attention layers.
    /// Non-attention (shortconv, Mamba) layers have `None` and never write the KV cache.
    /// Allows KV cache to be sized to only the attention-layer count instead of
    /// all layers (e.g. 6 instead of 24 for LFM2MoE), saving several GB.
    kv_layer_map: Vec<Option<usize>>,
    // Mamba/SSM persistent state
    ssm_states: Vec<Vec<f32>>, // [layer][state_dim]
    ssm_conv_buffers: Vec<ConvHistoryRing>,
    workspace: Workspace,
    /// Final output-normalized hidden row for the most recent target token.
    /// Native MTP consumes this row as its target-hidden input.
    last_output_hidden: Vec<f32>,
}

impl InferenceModel {
    /// Access the model's inference configuration.
    pub fn config(&self) -> &InferenceConfig {
        &self.config
    }
}

/// Decode token `token_idx`'s embedding (length `h`) out of a quantized
/// `vocab × h` matrix stored in GGUF natural layout (one contiguous row per
/// token, each row consisting of `h / block_width` quantized blocks).
pub(crate) fn lookup_quantized_embedding(
    h: usize,
    qtype: GgufQuantizationType,
    data: &[u8],
    token_idx: usize,
    x: &mut [f32],
) {
    let block_size = match qtype {
        GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M => 144,
        GgufQuantizationType::Q6_K => 210,
        GgufQuantizationType::Q8_0 => 34,
        GgufQuantizationType::NVFP4 => 36,
        GgufQuantizationType::IQ1_S => 50,
        GgufQuantizationType::IQ1_M => 56,
        _ => return,
    };
    let block_width = match qtype {
        GgufQuantizationType::Q8_0 => 32,
        GgufQuantizationType::NVFP4 => 64,
        _ => 256,
    };
    let blocks_per_row = h / block_width;
    let row_start = token_idx * blocks_per_row * block_size;
    let row = &data[row_start..row_start + blocks_per_row * block_size];
    match qtype {
        GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M => {
            let _ =
                crate::quantization::dequantize_q4_k_scalar(row, &mut x[..blocks_per_row * 256]);
        }
        GgufQuantizationType::Q6_K => {
            let _ =
                crate::quantization::dequantize_q6_k_scalar(row, &mut x[..blocks_per_row * 256]);
        }
        GgufQuantizationType::Q8_0 => {
            // Q8_0: block = [d_f16 (2 bytes), 32× int8].
            for (block_idx, block) in row.chunks_exact(block_size).enumerate() {
                let d = f16_le_to_f32([block[0], block[1]]);
                let out = &mut x[block_idx * 32..block_idx * 32 + 32];
                for (i, b) in block[2..2 + 32].iter().enumerate() {
                    out[i] = (*b as i8) as f32 * d;
                }
            }
        }
        GgufQuantizationType::IQ1_S => {
            let _ =
                crate::quantization::dequantize_iq1_s_scalar(row, &mut x[..blocks_per_row * 256]);
        }
        GgufQuantizationType::IQ1_M => {
            let _ =
                crate::quantization::dequantize_iq1_m_scalar(row, &mut x[..blocks_per_row * 256]);
        }
        GgufQuantizationType::NVFP4 => {
            let _ =
                crate::quantization::dequantize_nvfp4_scalar(row, &mut x[..blocks_per_row * 64]);
        }
        _ => {}
    }
}

fn lookup_embedding_from_storage(
    storage: &WeightStorage,
    hidden_size: usize,
    vocab_size: usize,
    token: Token,
    out: &mut [f32],
) {
    out.fill(0.0_f32);
    if out.len() != hidden_size || hidden_size == 0 || vocab_size == 0 {
        return;
    }
    let token_idx = (token as usize).min(vocab_size.saturating_sub(1));
    match storage {
        WeightStorage::F32(data) => {
            let start = token_idx.saturating_mul(hidden_size);
            let end = start.saturating_add(hidden_size);
            if end <= data.len() {
                out.copy_from_slice(&data[start..end]);
            }
        }
        WeightStorage::Quantized(qtype, data) => {
            lookup_quantized_embedding(hidden_size, *qtype, data, token_idx, out);
        }
        WeightStorage::MmapQuantized(qtype, mmap, offset, size) => {
            let data = &mmap[*offset..*offset + *size];
            lookup_quantized_embedding(hidden_size, *qtype, data, token_idx, out);
        }
    }
}

impl InferenceModel {
    pub fn load_from_gguf(
        mapped: &MappedGgufFile,
        mut config: InferenceConfig,
        use_mmap: bool,
    ) -> Result<Self, String> {
        // Architecture-aware configuration
        config.architecture = ModelArchitecture::from_gguf(mapped);
        if config.alibi_num_heads == 0 {
            config.alibi_num_heads = config.num_attention_heads;
        }
        let mut tok_embeddings: Option<WeightStorage> = None;
        let mut tok_embeddings_cols: usize = config.hidden_size;
        let mut norm_weight: Option<Vec<f32>> = None;
        let mut output_weight: Option<WeightStorage> = None;
        let mut layers: Vec<LayerWeights> = vec![LayerWeights::default(); config.layer_count];
        let mut mtp: Option<MtpWeights> =
            (config.nextn_predict_layers > 0).then(MtpWeights::default);
        let mmap_arc = if use_mmap { Some(mapped.mmap()) } else { None };

        let tensor_list = mapped.mapped_tensor_infos();
        for tensor in tensor_list.iter() {
            let qtype = GgufQuantizationType::from_ggml_type(tensor.ggml_type);
            let value_count: usize = tensor.dimensions.iter().map(|&d| d as usize).product();
            let qsize = quantized_size(qtype, value_count)
                .map_err(|e| format!("quantized_size: {:?}", e))?;
            let offset = tensor.absolute_offset as usize;
            let qdata = &mapped.bytes()[offset..offset + qsize];

            // Helper to decide whether to keep quantized or dequantize
            let should_keep_quantized = |name: &str| -> bool {
                // Keep weight matrices quantized; dequantize norms and small vectors
                name.ends_with(".weight") && !name.contains("norm") && !name.contains("bias")
            };

            let load_tensor = |name: &str,
                               qtype: GgufQuantizationType,
                               qdata: &[u8],
                               count: usize|
             -> Result<WeightStorage, String> {
                // Keep only formats with implemented on-the-fly GEMV kernels quantized.
                let is_supported_quant_gemv = matches!(
                    qtype,
                    GgufQuantizationType::Q8_0
                        | GgufQuantizationType::Q4_K_S
                        | GgufQuantizationType::Q4_K_M
                        | GgufQuantizationType::Q6_K
                        | GgufQuantizationType::IQ1_S
                        | GgufQuantizationType::IQ1_M
                        | GgufQuantizationType::NVFP4
                );
                if should_keep_quantized(name) && is_supported_quant_gemv {
                    if let Some(ref arc) = mmap_arc {
                        Ok(WeightStorage::MmapQuantized(
                            qtype,
                            arc.clone(),
                            offset,
                            qsize,
                        ))
                    } else {
                        Ok(WeightStorage::Quantized(qtype, qdata.to_vec()))
                    }
                } else {
                    let mut f32_data = vec![0.0_f32; count];
                    dequantize_scalar(qtype, qdata, &mut f32_data)
                        .map_err(|e| format!("dequantize_scalar: {:?}", e))?;
                    Ok(WeightStorage::F32(f32_data))
                }
            };

            let load_bias = |qtype: GgufQuantizationType,
                             qdata: &[u8],
                             count: usize|
             -> Result<Vec<f32>, String> {
                let mut f32_data = vec![0.0_f32; count];
                dequantize_scalar(qtype, qdata, &mut f32_data)
                    .map_err(|e| format!("dequantize_scalar: {:?}", e))?;
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
                    tok_embeddings = Some(load_tensor(&tensor.name, qtype, qdata, value_count)?);
                }
                // LFM2 has no separate output_norm; token_embd_norm is the final norm.
                "norm.weight" | "output_norm.weight" | "token_embd_norm.weight" => {
                    let mut f32_data = vec![0.0_f32; value_count];
                    dequantize_scalar(qtype, qdata, &mut f32_data)
                        .map_err(|e| format!("dequantize_scalar: {:?}", e))?;
                    norm_weight = Some(f32_data);
                }
                "output.weight" => {
                    output_weight = Some(load_tensor(&tensor.name, qtype, qdata, value_count)?);
                }
                name if name.starts_with("blk.") => {
                    let parts: Vec<&str> = name.split('.').collect();
                    if parts.len() < 4 {
                        continue;
                    }
                    let layer_idx: usize = parts[1]
                        .parse()
                        .map_err(|_| format!("bad layer index in tensor name: {}", name))?;
                    if layer_idx >= config.layer_count {
                        if let Some(mtp) = mtp.as_mut()
                            && layer_idx == config.layer_count
                        {
                            if parts.get(2) == Some(&"nextn") {
                                let nextn_name = parts.get(3).copied().unwrap_or("");
                                let nextn_suffix = parts.get(4).copied();
                                match (nextn_name, nextn_suffix) {
                                    ("eh_proj", Some("weight")) => {
                                        mtp.eh_proj = load_tensor(name, qtype, qdata, value_count)?;
                                    }
                                    ("enorm", Some("weight")) | ("enorm", None) => {
                                        mtp.enorm = load_bias(qtype, qdata, value_count)?;
                                    }
                                    ("hnorm", Some("weight")) | ("hnorm", None) => {
                                        mtp.hnorm = load_bias(qtype, qdata, value_count)?;
                                    }
                                    ("embed_tokens", Some("weight")) => {
                                        mtp.embed_tokens =
                                            load_tensor(name, qtype, qdata, value_count)?;
                                    }
                                    ("shared_head_norm", Some("weight"))
                                    | ("shared_head_norm", None) => {
                                        mtp.shared_head_norm =
                                            load_bias(qtype, qdata, value_count)?;
                                    }
                                    ("shared_head_head", Some("weight"))
                                    | ("shared_head", Some("weight")) => {
                                        mtp.shared_head_head =
                                            load_tensor(name, qtype, qdata, value_count)?;
                                    }
                                    _ => {}
                                }
                            } else {
                                let weight_name = parts[2];
                                let suffix = parts.get(3).copied();
                                match (weight_name, suffix) {
                                    ("attn_norm", _) => {
                                        mtp.layer.attn_norm = load_bias(qtype, qdata, value_count)?;
                                    }
                                    ("attn_q", Some("weight")) => {
                                        mtp.layer.attn_q =
                                            load_tensor(name, qtype, qdata, value_count)?;
                                    }
                                    ("attn_q", Some("bias")) => {
                                        mtp.layer.attn_q_bias =
                                            load_bias(qtype, qdata, value_count)?;
                                    }
                                    ("attn_k", Some("weight")) => {
                                        mtp.layer.attn_k =
                                            load_tensor(name, qtype, qdata, value_count)?;
                                    }
                                    ("attn_k", Some("bias")) => {
                                        mtp.layer.attn_k_bias =
                                            load_bias(qtype, qdata, value_count)?;
                                    }
                                    ("attn_v", Some("weight")) => {
                                        mtp.layer.attn_v =
                                            load_tensor(name, qtype, qdata, value_count)?;
                                    }
                                    ("attn_v", Some("bias")) => {
                                        mtp.layer.attn_v_bias =
                                            load_bias(qtype, qdata, value_count)?;
                                    }
                                    ("attn_output", Some("weight")) => {
                                        mtp.layer.attn_output =
                                            load_tensor(name, qtype, qdata, value_count)?;
                                    }
                                    ("attn_output", Some("bias")) => {
                                        mtp.layer.attn_output_bias =
                                            load_bias(qtype, qdata, value_count)?;
                                    }
                                    ("attn_q_norm", _) => {
                                        mtp.layer.attn_q_norm =
                                            load_bias(qtype, qdata, value_count)?;
                                    }
                                    ("attn_k_norm", _) => {
                                        mtp.layer.attn_k_norm =
                                            load_bias(qtype, qdata, value_count)?;
                                    }
                                    ("ffn_norm", _) | ("post_attention_norm", _) => {
                                        mtp.layer.post_attention_norm =
                                            load_bias(qtype, qdata, value_count)?;
                                    }
                                    ("ffn_gate", _) => {
                                        mtp.layer.ffn_gate =
                                            load_tensor(name, qtype, qdata, value_count)?;
                                    }
                                    ("ffn_up", _) => {
                                        mtp.layer.ffn_up =
                                            load_tensor(name, qtype, qdata, value_count)?;
                                    }
                                    ("ffn_down", Some("weight")) => {
                                        mtp.layer.ffn_down =
                                            load_tensor(name, qtype, qdata, value_count)?;
                                    }
                                    ("ffn_down", Some("bias")) => {
                                        mtp.layer.ffn_down_bias =
                                            load_bias(qtype, qdata, value_count)?;
                                    }
                                    _ => {}
                                }
                            }
                        }
                        continue;
                    }
                    let weight_name = parts[2];
                    let suffix = parts.get(3).copied();
                    match (weight_name, suffix) {
                        ("attn_norm", _) => {
                            let mut f32_data = vec![0.0_f32; value_count];
                            dequantize_scalar(qtype, qdata, &mut f32_data)
                                .map_err(|e| format!("dequantize: {:?}", e))?;
                            layers[layer_idx].attn_norm = f32_data;
                        }
                        ("attn_q", Some("weight")) => {
                            layers[layer_idx].attn_q = load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("attn_q", Some("bias")) => {
                            layers[layer_idx].attn_q_bias = load_bias(qtype, qdata, value_count)?
                        }
                        ("attn_k", Some("weight")) => {
                            layers[layer_idx].attn_k = load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("attn_k", Some("bias")) => {
                            layers[layer_idx].attn_k_bias = load_bias(qtype, qdata, value_count)?
                        }
                        ("attn_v", Some("weight")) => {
                            layers[layer_idx].attn_v = load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("attn_v", Some("bias")) => {
                            layers[layer_idx].attn_v_bias = load_bias(qtype, qdata, value_count)?
                        }
                        ("attn_output", Some("weight")) => {
                            layers[layer_idx].attn_output =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("attn_output", Some("bias")) => {
                            layers[layer_idx].attn_output_bias =
                                load_bias(qtype, qdata, value_count)?
                        }
                        ("attn_qkv", _) => {
                            layers[layer_idx].attn_qkv =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("attn_gate", _) => {
                            layers[layer_idx].attn_gate =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("attn_q_norm", _) => {
                            let mut f32_data = vec![0.0_f32; value_count];
                            dequantize_scalar(qtype, qdata, &mut f32_data)
                                .map_err(|e| format!("dequantize: {:?}", e))?;
                            layers[layer_idx].attn_q_norm = f32_data;
                        }
                        ("attn_k_norm", _) => {
                            let mut f32_data = vec![0.0_f32; value_count];
                            dequantize_scalar(qtype, qdata, &mut f32_data)
                                .map_err(|e| format!("dequantize: {:?}", e))?;
                            layers[layer_idx].attn_k_norm = f32_data;
                        }
                        ("ffn_norm", _) => {
                            let mut f32_data = vec![0.0_f32; value_count];
                            dequantize_scalar(qtype, qdata, &mut f32_data)
                                .map_err(|e| format!("dequantize: {:?}", e))?;
                            layers[layer_idx].ffn_norm = f32_data;
                        }
                        ("post_attention_norm", _) => {
                            let mut f32_data = vec![0.0_f32; value_count];
                            dequantize_scalar(qtype, qdata, &mut f32_data)
                                .map_err(|e| format!("dequantize: {:?}", e))?;
                            layers[layer_idx].post_attention_norm = f32_data;
                        }
                        ("post_ffw_norm", _) | ("post_ffn_norm", _) => {
                            let mut f32_data = vec![0.0_f32; value_count];
                            dequantize_scalar(qtype, qdata, &mut f32_data)
                                .map_err(|e| format!("dequantize: {:?}", e))?;
                            layers[layer_idx].post_ffn_norm = f32_data;
                        }
                        ("ffn_gate", _) => {
                            layers[layer_idx].ffn_gate =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("ffn_up", _) => {
                            layers[layer_idx].ffn_up = load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("ffn_down", Some("weight")) => {
                            layers[layer_idx].ffn_down =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("ffn_down", Some("bias")) => {
                            layers[layer_idx].ffn_down_bias = load_bias(qtype, qdata, value_count)?
                        }
                        // MoE expert weights
                        ("ffn_gate_exps", _) => {
                            layers[layer_idx].ffn_gate_exps =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("ffn_up_exps", _) => {
                            layers[layer_idx].ffn_up_exps =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("ffn_down_exps", _) => {
                            layers[layer_idx].ffn_down_exps =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("ffn_gate_inp", _) => {
                            layers[layer_idx].ffn_gate_inp =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("ssm_a", _) => {
                            let mut f32_data = vec![0.0_f32; value_count];
                            dequantize_scalar(qtype, qdata, &mut f32_data)
                                .map_err(|e| format!("dequantize: {:?}", e))?;
                            layers[layer_idx].ssm_a = f32_data;
                        }
                        ("ssm_alpha", _) => {
                            let mut f32_data = vec![0.0_f32; value_count];
                            dequantize_scalar(qtype, qdata, &mut f32_data)
                                .map_err(|e| format!("dequantize: {:?}", e))?;
                            layers[layer_idx].ssm_alpha = f32_data;
                        }
                        ("ssm_beta", _) => {
                            let mut f32_data = vec![0.0_f32; value_count];
                            dequantize_scalar(qtype, qdata, &mut f32_data)
                                .map_err(|e| format!("dequantize: {:?}", e))?;
                            layers[layer_idx].ssm_beta = f32_data;
                        }
                        ("ssm_conv1d", _) => {
                            let mut f32_data = vec![0.0_f32; value_count];
                            dequantize_scalar(qtype, qdata, &mut f32_data)
                                .map_err(|e| format!("dequantize: {:?}", e))?;
                            layers[layer_idx].ssm_conv1d = f32_data;
                        }
                        ("ssm_dt", Some("bias")) => {
                            let mut f32_data = vec![0.0_f32; value_count];
                            dequantize_scalar(qtype, qdata, &mut f32_data)
                                .map_err(|e| format!("dequantize: {:?}", e))?;
                            layers[layer_idx].ssm_dt_bias = f32_data;
                        }
                        ("ssm_norm", _) => {
                            let mut f32_data = vec![0.0_f32; value_count];
                            dequantize_scalar(qtype, qdata, &mut f32_data)
                                .map_err(|e| format!("dequantize: {:?}", e))?;
                            layers[layer_idx].ssm_norm = f32_data;
                        }
                        ("ssm_out", _) => {
                            layers[layer_idx].ssm_out =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        // LFM2 short-convolution operator
                        ("shortconv", Some("in_proj")) => {
                            layers[layer_idx].shortconv_in_proj =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("shortconv", Some("out_proj")) => {
                            layers[layer_idx].shortconv_out_proj =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("shortconv", Some("conv")) => {
                            let mut f32_data = vec![0.0_f32; value_count];
                            dequantize_scalar(qtype, qdata, &mut f32_data)
                                .map_err(|e| format!("dequantize: {:?}", e))?;
                            layers[layer_idx].shortconv_conv = f32_data;
                        }
                        // LFM2MoE per-layer expert routing bias
                        ("exp_probs_b", _) => {
                            layers[layer_idx].ffn_exp_probs_b =
                                load_bias(qtype, qdata, value_count)?
                        }
                        ("attn_q_a", Some("weight")) => {
                            layers[layer_idx].mla_q_a =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("attn_q_a_norm", _) => {
                            let mut f32_data = vec![0.0_f32; value_count];
                            dequantize_scalar(qtype, qdata, &mut f32_data)
                                .map_err(|e| format!("dequantize: {:?}", e))?;
                            layers[layer_idx].mla_q_a_norm = f32_data;
                        }
                        ("attn_q_b", Some("weight")) => {
                            layers[layer_idx].mla_q_b =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("attn_kv_a_mqa", Some("weight")) => {
                            layers[layer_idx].mla_kv_a_mqa =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("attn_kv_a_norm", _) => {
                            let mut f32_data = vec![0.0_f32; value_count];
                            dequantize_scalar(qtype, qdata, &mut f32_data)
                                .map_err(|e| format!("dequantize: {:?}", e))?;
                            layers[layer_idx].mla_kv_a_norm = f32_data;
                        }
                        ("attn_k_b", Some("weight")) => {
                            layers[layer_idx].mla_k_b =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("attn_v_b", Some("weight")) => {
                            layers[layer_idx].mla_v_b =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("ffn_gate_shexp", _) => {
                            layers[layer_idx].ffn_gate_shexp =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("ffn_up_shexp", _) => {
                            layers[layer_idx].ffn_up_shexp =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        ("ffn_down_shexp", _) => {
                            layers[layer_idx].ffn_down_shexp =
                                load_tensor(name, qtype, qdata, value_count)?
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        let tok_embeddings = tok_embeddings.ok_or("missing tok_embeddings.weight")?;
        let norm_weight = norm_weight.ok_or("missing norm.weight")?;
        let output_weight = output_weight.unwrap_or_else(|| tok_embeddings.clone());
        let mtp = mtp.and_then(|weights| {
            if weights.is_usable(&config) {
                Some(weights)
            } else {
                eprintln!(
                    "MTP metadata advertises {} nextn layer(s), but required blk.{}.nextn/decoder tensors were incomplete; disabling native MTP",
                    config.nextn_predict_layers, config.layer_count
                );
                None
            }
        });

        eprintln!(
            "InferenceConfig: vocab={}, context={}, layers={}, mtp_nextn={}, hidden={}, intermediate={}, heads={}, kv_heads={}, kv_head_dim={}, eps={}, theta={}",
            config.vocab_size,
            config.context_size,
            config.layer_count,
            config.nextn_predict_layers,
            config.hidden_size,
            config.intermediate_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.kv_head_dim(),
            config.rms_norm_eps,
            config.rope_theta
        );

        // Build a map from absolute layer index to KV cache layer index.
        // Only attention layers (those with a non-empty attn_q projection) write to
        // the KV cache.  Shortconv and Mamba layers skip the KV cache entirely, so
        // sizing the cache to all `layer_count` layers wastes memory proportional to
        // the number of non-attention layers (e.g. 18 out of 24 for LFM2MoE, saving
        // ~8.8 GB for a 128k-context model).
        let mut kv_layer_map: Vec<Option<usize>> = Vec::with_capacity(layers.len());
        let mut attn_layer_count: usize = 0;
        for layer in layers.iter().take(config.layer_count) {
            let is_attn = !layer.attn_q.is_empty() || !layer.mla_kv_a_mqa.is_empty();
            if is_attn {
                kv_layer_map.push(Some(attn_layer_count));
                attn_layer_count += 1;
            } else {
                kv_layer_map.push(None);
            }
        }

        let kv_head_count = if config.architecture.uses_mla() {
            config.num_attention_heads
        } else {
            config.num_key_value_heads
        };
        let kv_cache_config = KvCacheConfig {
            layer_count: attn_layer_count,
            context_size: config.context_size,
            head_count: kv_head_count,
            head_dim: config.kv_head_dim(),
            dtype: config.kv_cache_dtype,
            quantization: config.kv_quantization,
        };
        let kv_cache = KvCache::new(kv_cache_config).map_err(|e| format!("kv_cache: {:?}", e))?;

        // Initialize Mamba/SSM state
        let mut ssm_states: Vec<Vec<f32>> = Vec::with_capacity(config.layer_count);
        let mut ssm_conv_buffers: Vec<ConvHistoryRing> = Vec::with_capacity(config.layer_count);
        let shortconv_hist = config.shortconv_l_cache.saturating_sub(1).max(3);
        for layer in layers.iter().take(config.layer_count) {
            let state_dim = layer.ssm_a.len().max(1);
            ssm_states.push(vec![0.0_f32; state_dim]);
            let hist_dim = if !layer.shortconv_in_proj.is_empty() {
                config.hidden_size
            } else {
                layer.attn_qkv.output_dim(config.hidden_size).max(1)
            };
            let cap = if !layer.shortconv_in_proj.is_empty() {
                shortconv_hist
            } else {
                3
            };
            ssm_conv_buffers.push(ConvHistoryRing::new(cap, hist_dim));
        }

        let workspace = Workspace::for_config(&config);
        let last_output_hidden = vec![0.0_f32; config.hidden_size];

        Ok(Self {
            config,
            tok_embeddings,
            tok_embeddings_cols,
            norm_weight,
            output_weight,
            layers,
            mtp,
            kv_cache,
            kv_layer_map,
            ssm_states,
            ssm_conv_buffers,
            workspace,
            last_output_hidden,
        })
    }

    pub fn forward_tokens_no_logits(
        &mut self,
        tokens: &[Token],
        session: &mut Session,
    ) -> Result<(), ModelError> {
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
        if tokens.len() > 1 && self.layers_supported_for_batched() {
            self.forward_batched(tokens, start_pos, false)?;
        } else {
            for (i, &token) in tokens.iter().enumerate() {
                self.forward_single(token, start_pos + i, false)?;
            }
        }
        session.record_tokens(tokens.len());
        Ok(())
    }

    /// True when all layers use the standard attention + FFN path that
    /// [`forward_batched`] can handle. Mamba/SSM and MoE layers need per-token
    /// state and aren't supported by the batched path yet.
    fn layers_supported_for_batched(&self) -> bool {
        if self.layers.is_empty() {
            return false;
        }
        for layer in &self.layers {
            let is_mamba = !layer.attn_qkv.is_empty() && layer.attn_q.is_empty();
            if is_mamba {
                return false;
            }
            let is_moe = !layer.ffn_gate_exps.is_empty()
                || !layer.ffn_up_exps.is_empty()
                || !layer.ffn_down_exps.is_empty()
                || !layer.ffn_gate_inp.is_empty();
            if is_moe {
                return false;
            }
            // No standard attention → can't batch the layer (degenerate case).
            if layer.attn_q.is_empty() {
                return false;
            }
        }
        true
    }

    /// Batched prefill: process `tokens` in one shot, sharing each weight
    /// matrix across all batch positions via [`gemm_weight`]. Per-token work
    /// (per-head Q/K norms, RoPE, KV cache writes, attention) stays in a
    /// position loop — only the matmuls are batched, but those dominate.
    ///
    /// Only the final token's logits are computed when `need_logits` is true;
    /// intermediate tokens' logits would be discarded anyway.
    #[allow(clippy::too_many_lines)]
    fn forward_batched(
        &mut self,
        tokens: &[Token],
        start_pos: usize,
        need_logits: bool,
    ) -> Result<Option<Logits>, ModelError> {
        let batch = tokens.len();
        debug_assert!(batch >= 1);
        let cfg = self.config.clone();
        let h = cfg.hidden_size;
        let n = cfg.num_attention_heads;
        let kvh = cfg.num_key_value_heads;

        // 1. Embedding lookup for every batch position into x_batch[batch, h].
        let mut x_batch = vec![0.0_f32; batch * h];
        for (i, &token) in tokens.iter().enumerate() {
            let token_idx = (token as usize).min(cfg.vocab_size.saturating_sub(1));
            let target = &mut x_batch[i * h..(i + 1) * h];
            match &self.tok_embeddings {
                WeightStorage::F32(data) => {
                    let row = &data[token_idx * h..(token_idx + 1) * h];
                    target.copy_from_slice(row);
                }
                WeightStorage::Quantized(qtype, data) => {
                    lookup_quantized_embedding(h, *qtype, data, token_idx, target);
                }
                WeightStorage::MmapQuantized(qtype, mmap, offset, size) => {
                    let data = &mmap[*offset..*offset + *size];
                    lookup_quantized_embedding(h, *qtype, data, token_idx, target);
                }
            }
            if cfg.embedding_scale != 1.0 {
                for v in target.iter_mut() {
                    *v *= cfg.embedding_scale;
                }
            }
        }

        // Scratch buffers reused across layers. Allocated once per call (batched
        // prefill is not in the per-token hot path, so this is fine).
        let layer0 = &self.layers[0];
        let q_len = layer0.attn_q.output_dim(h);
        let kv_len = if !layer0.attn_k.is_empty() {
            layer0.attn_k.output_dim(h)
        } else {
            0
        };
        let aoil0 = if !layer0.attn_output.is_empty() {
            layer0.attn_output.output_dim(h)
        } else {
            0
        };
        let q_len_used0 = if aoil0 > 0 {
            q_len.min(aoil0)
        } else if q_len > h {
            h
        } else {
            q_len
        };
        let q_head_dim = if n > 0 && q_len_used0.is_multiple_of(n) {
            q_len_used0 / n
        } else {
            q_len_used0
        };
        let kv_head_dim = if kvh > 0 && kv_len > 0 && kv_len.is_multiple_of(kvh) {
            kv_len / kvh
        } else if kv_len > 0 {
            kv_len
        } else {
            q_head_dim
        };
        let i_size = cfg.intermediate_size;

        let mut normed_batch = vec![0.0_f32; batch * h];
        let mut q_batch = vec![0.0_f32; batch * q_len];
        let mut k_batch = vec![0.0_f32; batch * kv_len.max(1)];
        let mut v_batch = vec![0.0_f32; batch * kv_len.max(1)];
        let mut attn_result_batch = vec![0.0_f32; batch * q_len_used0];
        let mut attn_proj_batch = vec![0.0_f32; batch * h];
        let mut gate_batch = vec![0.0_f32; batch * i_size];
        let mut up_batch = vec![0.0_f32; batch * i_size];
        let mut ffn_out_batch = vec![0.0_f32; batch * h];
        let mut head_scratch = vec![0.0_f32; q_head_dim.max(kv_head_dim)];
        let mut qk_norm_scratch = vec![0.0_f32; q_len.max(kv_len)];

        for layer_idx in 0..cfg.layer_count {
            let layer = &self.layers[layer_idx];
            // Map to the KV cache layer index (only attention layers are stored).
            let kv_layer_idx = self
                .kv_layer_map
                .get(layer_idx)
                .copied()
                .flatten()
                .unwrap_or(layer_idx);

            let ffn_norm_weight: &[f32] = if cfg.sandwich_norm {
                // Gemma: post_attention_norm is a sandwich norm (applied to the
                // attention output), NOT the pre-FFN norm. Use ffn_norm here.
                &layer.ffn_norm
            } else if !layer.post_attention_norm.is_empty() {
                &layer.post_attention_norm
            } else if !layer.ffn_norm.is_empty() {
                &layer.ffn_norm
            } else {
                &[]
            };

            // Per-layer RoPE theta and sliding-window size (Gemma interleaves
            // local SWA and global attention layers with distinct RoPE bases).
            let layer_rope = cfg.layer_rope_theta(layer_idx);
            let layer_window = cfg.layer_sliding_window(layer_idx);

            // 2. Per-token attn RMSNorm into normed_batch.
            for i in 0..batch {
                rms_norm_f32(
                    &x_batch[i * h..(i + 1) * h],
                    &layer.attn_norm,
                    cfg.rms_norm_eps,
                    &mut normed_batch[i * h..(i + 1) * h],
                )
                .map_err(|e| ModelError::InferenceFailed(format!("rms_norm: {:?}", e)))?;
            }

            // 3. Batched Q/K/V via GEMM — the main win over per-token GEMV.
            gemm_weight(&layer.attn_q, q_len, h, &normed_batch, &mut q_batch, batch)
                .map_err(|e| ModelError::InferenceFailed(format!("attn_q: {:?}", e)))?;
            if !layer.attn_q_bias.is_empty() {
                add_repeating_bias(&mut q_batch, &layer.attn_q_bias);
            }
            if kv_len > 0 {
                gemm_weight(
                    &layer.attn_k,
                    kv_len,
                    h,
                    &normed_batch,
                    &mut k_batch[..batch * kv_len],
                    batch,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("attn_k: {:?}", e)))?;
                if !layer.attn_k_bias.is_empty() {
                    add_repeating_bias(&mut k_batch[..batch * kv_len], &layer.attn_k_bias);
                }
                gemm_weight(
                    &layer.attn_v,
                    kv_len,
                    h,
                    &normed_batch,
                    &mut v_batch[..batch * kv_len],
                    batch,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("attn_v: {:?}", e)))?;
                if !layer.attn_v_bias.is_empty() {
                    add_repeating_bias(&mut v_batch[..batch * kv_len], &layer.attn_v_bias);
                }
            }

            if trace_fwd_enabled() {
                let s = |v: &[f32]| v.iter().map(|x| *x as f64).sum::<f64>();
                for t in 0..batch {
                    eprintln!(
                        "STAGE inf pos={} layer={layer_idx} normed={:.6e} q={:.6e} k={:.6e} v={:.6e}",
                        start_pos + t,
                        s(&normed_batch[t * h..(t + 1) * h]),
                        s(&q_batch[t * q_len..(t + 1) * q_len]),
                        s(&k_batch[t * kv_len..(t + 1) * kv_len]),
                        s(&v_batch[t * kv_len..(t + 1) * kv_len])
                    );
                }
            }
            let q_heads = q_len_used0 / q_head_dim.max(1);
            let kv_heads = kv_len.checked_div(kv_head_dim).unwrap_or(0);

            // 4. Per-token: Q/K norm, RoPE, KV cache writes.
            for i in 0..batch {
                let pos = start_pos + i;
                let q = &mut q_batch[i * q_len..i * q_len + q_len_used0];
                let k = &mut k_batch[i * kv_len..(i + 1) * kv_len];
                let v = &v_batch[i * kv_len..(i + 1) * kv_len];

                if !layer.attn_q_norm.is_empty() && q.len() == layer.attn_q_norm.len() {
                    let normed_q = &mut qk_norm_scratch[..q.len()];
                    rms_norm_f32(q, &layer.attn_q_norm, cfg.rms_norm_eps, normed_q)
                        .map_err(|e| ModelError::InferenceFailed(format!("q_norm: {:?}", e)))?;
                    q.copy_from_slice(normed_q);
                } else if !layer.attn_q_norm.is_empty() && q_head_dim == layer.attn_q_norm.len() {
                    for head in 0..q_heads {
                        let start = head * q_head_dim;
                        let end = start + q_head_dim;
                        if end > q.len() {
                            break;
                        }
                        let normed_head = &mut head_scratch[..q_head_dim];
                        normed_head.fill(0.0_f32);
                        rms_norm_f32(
                            &q[start..end],
                            &layer.attn_q_norm,
                            cfg.rms_norm_eps,
                            normed_head,
                        )
                        .map_err(|e| ModelError::InferenceFailed(format!("q_norm: {:?}", e)))?;
                        q[start..end].copy_from_slice(normed_head);
                    }
                }
                if !layer.attn_k_norm.is_empty() && k.len() == layer.attn_k_norm.len() {
                    let normed_k = &mut qk_norm_scratch[..k.len()];
                    rms_norm_f32(k, &layer.attn_k_norm, cfg.rms_norm_eps, normed_k)
                        .map_err(|e| ModelError::InferenceFailed(format!("k_norm: {:?}", e)))?;
                    k.copy_from_slice(normed_k);
                } else if !layer.attn_k_norm.is_empty() && kv_head_dim == layer.attn_k_norm.len() {
                    for head in 0..kv_heads {
                        let start = head * kv_head_dim;
                        let end = start + kv_head_dim;
                        if end > k.len() {
                            break;
                        }
                        let normed_head = &mut head_scratch[..kv_head_dim];
                        normed_head.fill(0.0_f32);
                        rms_norm_f32(
                            &k[start..end],
                            &layer.attn_k_norm,
                            cfg.rms_norm_eps,
                            normed_head,
                        )
                        .map_err(|e| ModelError::InferenceFailed(format!("k_norm: {:?}", e)))?;
                        k[start..end].copy_from_slice(normed_head);
                    }
                }

                // RoPE Q — only rotate the first `rope_dim` elements per head (partial RoPE).
                let q_rope_len = cfg.effective_rope_dim().min(q_head_dim);
                for head in 0..q_heads {
                    let off = head * q_head_dim;
                    if off + q_head_dim > q.len() {
                        break;
                    }
                    let rotated = &mut head_scratch[..q_rope_len];
                    rotated.fill(0.0_f32);
                    apply_rope_f32(
                        &q[off..off + q_rope_len],
                        pos,
                        q_rope_len,
                        layer_rope,
                        rotated,
                    )
                    .map_err(|e| ModelError::InferenceFailed(format!("rope q: {:?}", e)))?;
                    q[off..off + q_rope_len].copy_from_slice(rotated);
                }
                // RoPE K — partial RoPE: same rope_dim slice.
                let k_rope_len = cfg.effective_rope_dim().min(kv_head_dim);
                for head in 0..kv_heads {
                    let off = head * kv_head_dim;
                    if off + kv_head_dim > k.len() {
                        break;
                    }
                    let rotated = &mut head_scratch[..k_rope_len];
                    rotated.fill(0.0_f32);
                    apply_rope_f32(
                        &k[off..off + k_rope_len],
                        pos,
                        k_rope_len,
                        layer_rope,
                        rotated,
                    )
                    .map_err(|e| ModelError::InferenceFailed(format!("rope k: {:?}", e)))?;
                    k[off..off + k_rope_len].copy_from_slice(rotated);
                }

                self.kv_cache
                    .set(kv_layer_idx, pos, k, v)
                    .map_err(|e| ModelError::InferenceFailed(format!("kv set: {:?}", e)))?;
            }

            // 5. Per-token: attention. Each position attends to its own causal
            // prefix (positions 0..=pos).
            //
            // For F32 KV caches we try to borrow the prefix directly (zero-copy).
            // For quantized KV caches we copy into temporary buffers. Both paths
            // use the same flash attention kernel.
            let mut key_copy_buf: Vec<f32> = Vec::new();
            let mut value_copy_buf: Vec<f32> = Vec::new();

            for i in 0..batch {
                let pos = start_pos + i;
                let seq_len = pos + 1;
                let q = &q_batch[i * q_len..i * q_len + q_len_used0];
                let attn_out_slice = &mut attn_result_batch[i * q_len_used0..(i + 1) * q_len_used0];
                attn_out_slice.fill(0.0_f32);

                let (key_cache, value_cache): (&[f32], &[f32]) = {
                    let key_borrow = self
                        .kv_cache
                        .f32_layer_key_prefix(kv_layer_idx, seq_len)
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("kv borrow keys: {:?}", e))
                        })?;
                    let value_borrow = self
                        .kv_cache
                        .f32_layer_value_prefix(kv_layer_idx, seq_len)
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("kv borrow vals: {:?}", e))
                        })?;

                    if let (Some(keys), Some(values)) = (key_borrow, value_borrow) {
                        (keys, values)
                    } else {
                        // Quantized KV cache: copy into reusable buffers.
                        let needed = seq_len * kv_len;
                        if key_copy_buf.len() < needed {
                            key_copy_buf.resize(needed, 0.0_f32);
                        }
                        if value_copy_buf.len() < needed {
                            value_copy_buf.resize(needed, 0.0_f32);
                        }
                        self.kv_cache
                            .copy_layer_keys(kv_layer_idx, seq_len, &mut key_copy_buf[..needed])
                            .map_err(|e| {
                                ModelError::InferenceFailed(format!("kv copy keys: {:?}", e))
                            })?;
                        self.kv_cache
                            .copy_layer_values(kv_layer_idx, seq_len, &mut value_copy_buf[..needed])
                            .map_err(|e| {
                                ModelError::InferenceFailed(format!("kv copy vals: {:?}", e))
                            })?;
                        (&key_copy_buf[..needed], &value_copy_buf[..needed])
                    }
                };

                // Sliding-window attention: a local layer attends only to the most
                // recent `layer_window` positions. Since RoPE encodes absolute
                // positions, slicing off the oldest key/value rows yields exactly
                // the windowed-causal mask with relative positions preserved.
                let (eff_seq_len, key_cache, value_cache) =
                    if layer_window > 0 && seq_len > layer_window {
                        let skip = (seq_len - layer_window) * kv_len;
                        (layer_window, &key_cache[skip..], &value_cache[skip..])
                    } else {
                        (seq_len, key_cache, value_cache)
                    };

                flash_attention_decode_heads_f32(
                    q,
                    key_cache,
                    value_cache,
                    eff_seq_len,
                    kv_head_dim,
                    kv_len,
                    q_heads,
                    kv_heads,
                    attn_out_slice,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("flash attn: {:?}", e)))?;
            }

            // 6. Batched attn_output projection.
            if !layer.attn_output.is_empty() && aoil0 > 0 {
                gemm_weight(
                    &layer.attn_output,
                    h,
                    aoil0,
                    &attn_result_batch,
                    &mut attn_proj_batch,
                    batch,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("attn_output: {:?}", e)))?;
                if !layer.attn_output_bias.is_empty() {
                    add_repeating_bias(&mut attn_proj_batch, &layer.attn_output_bias);
                }
            } else {
                attn_proj_batch.fill(0.0_f32);
            }

            // 6b. Gemma sandwich norm: normalize the attention output before the
            // residual add (post_attention_norm).
            if cfg.sandwich_norm && !layer.post_attention_norm.is_empty() {
                for i in 0..batch {
                    let range = i * h..(i + 1) * h;
                    rms_norm_f32(
                        &attn_proj_batch[range.clone()],
                        &layer.post_attention_norm,
                        cfg.rms_norm_eps,
                        &mut normed_batch[range.clone()],
                    )
                    .map_err(|e| ModelError::InferenceFailed(format!("post_attn_norm: {:?}", e)))?;
                    attn_proj_batch[range.clone()].copy_from_slice(&normed_batch[range]);
                }
            }

            // 7. Residual add (attn).
            for i in 0..batch * h {
                x_batch[i] += attn_proj_batch[i];
            }

            // 8. FFN: per-token RMSNorm, batched gate+up, SwiGLU, batched down.
            let has_ffn = !layer.ffn_gate.is_empty()
                && !layer.ffn_up.is_empty()
                && !layer.ffn_down.is_empty()
                && !ffn_norm_weight.is_empty();
            if has_ffn {
                for i in 0..batch {
                    rms_norm_f32(
                        &x_batch[i * h..(i + 1) * h],
                        ffn_norm_weight,
                        cfg.rms_norm_eps,
                        &mut normed_batch[i * h..(i + 1) * h],
                    )
                    .map_err(|e| ModelError::InferenceFailed(format!("ffn_norm: {:?}", e)))?;
                }

                gemm_weight(
                    &layer.ffn_gate,
                    i_size,
                    h,
                    &normed_batch,
                    &mut gate_batch,
                    batch,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("ffn_gate: {:?}", e)))?;
                gemm_weight(
                    &layer.ffn_up,
                    i_size,
                    h,
                    &normed_batch,
                    &mut up_batch,
                    batch,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("ffn_up: {:?}", e)))?;

                if cfg.gelu_ffn {
                    apply_geglu_inplace_f32(&mut gate_batch, &up_batch);
                } else {
                    for (g, u) in gate_batch.iter_mut().zip(up_batch.iter()) {
                        let sigmoid = 1.0_f32 / (1.0 + (-*g).exp());
                        *g = *g * sigmoid * *u;
                    }
                }

                gemm_weight(
                    &layer.ffn_down,
                    h,
                    i_size,
                    &gate_batch,
                    &mut ffn_out_batch,
                    batch,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("ffn_down: {:?}", e)))?;
                if !layer.ffn_down_bias.is_empty() {
                    add_repeating_bias(&mut ffn_out_batch, &layer.ffn_down_bias);
                }

                // Gemma sandwich norm: normalize the FFN output before residual.
                if cfg.sandwich_norm && !layer.post_ffn_norm.is_empty() {
                    for i in 0..batch {
                        let range = i * h..(i + 1) * h;
                        rms_norm_f32(
                            &ffn_out_batch[range.clone()],
                            &layer.post_ffn_norm,
                            cfg.rms_norm_eps,
                            &mut normed_batch[range.clone()],
                        )
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("post_ffn_norm: {:?}", e))
                        })?;
                        ffn_out_batch[range.clone()].copy_from_slice(&normed_batch[range]);
                    }
                }

                for i in 0..batch * h {
                    x_batch[i] += ffn_out_batch[i];
                }
            }
            if trace_fwd_enabled() {
                for t in 0..batch {
                    let sum: f64 = x_batch[t * h..(t + 1) * h].iter().map(|v| *v as f64).sum();
                    eprintln!(
                        "TRACE inf pos={} layer={layer_idx} sum={sum:.9e}",
                        start_pos + t
                    );
                }
            }
        }

        if !need_logits {
            return Ok(None);
        }

        // Final norm + lm_head only for the last batch position.
        let last = &x_batch[(batch - 1) * h..batch * h];
        let mut final_normed = vec![0.0_f32; h];
        rms_norm_f32(last, &self.norm_weight, cfg.rms_norm_eps, &mut final_normed)
            .map_err(|e| ModelError::InferenceFailed(format!("final_norm: {:?}", e)))?;
        self.last_output_hidden = final_normed.clone();
        let mut logits = vec![0.0_f32; cfg.vocab_size];
        gemv_weight(
            &self.output_weight,
            cfg.vocab_size,
            h,
            &final_normed,
            &mut logits,
        )
        .map_err(|e| ModelError::InferenceFailed(format!("output: {:?}", e)))?;
        Ok(Some(logits))
    }

    fn forward_single(
        &mut self,
        token: Token,
        pos: usize,
        need_logits: bool,
    ) -> Result<Option<Logits>, ModelError> {
        let token_t0 = crate::tensor::decode_profile_enabled().then(std::time::Instant::now);
        self.embed_token_into_workspace(token);
        let layer_count = self.config.layer_count;
        self.run_layer_range_in_workspace(pos, 0..layer_count)?;
        if !need_logits {
            return Ok(None);
        }
        let logits = self.final_head_from_workspace().map(Some);
        if let Some(t0) = token_t0 {
            crate::tensor::decode_profile_record("token_forward", t0.elapsed().as_nanos() as u64);
        }
        logits
    }

    /// Write `token`'s embedding into `workspace.x[..hidden_size]`. First stage
    /// of pipeline-parallel decode.
    pub fn embed_token_into_workspace(&mut self, token: Token) {
        let h = self.config.hidden_size;
        let x = &mut self.workspace.x[..h];
        x.fill(0.0_f32);
        let token_idx = (token as usize).min(self.config.vocab_size.saturating_sub(1));
        match &self.tok_embeddings {
            WeightStorage::F32(data) => {
                let row = &data[token_idx * h..(token_idx + 1) * h];
                x.copy_from_slice(row);
            }
            WeightStorage::Quantized(qtype, data) => {
                lookup_quantized_embedding(h, *qtype, data, token_idx, x);
            }
            WeightStorage::MmapQuantized(qtype, mmap, offset, size) => {
                let data = &mmap[*offset..*offset + *size];
                lookup_quantized_embedding(h, *qtype, data, token_idx, x);
            }
        }
        let scale = self.config.embedding_scale;
        if scale != 1.0 {
            for v in x.iter_mut() {
                *v *= scale;
            }
        }
    }

    /// Read the current hidden state from `workspace.x[..hidden_size]`.
    pub fn hidden_state(&self) -> &[f32] {
        &self.workspace.x[..self.config.hidden_size]
    }

    /// Hidden size from the loaded config (so pipeline drivers can size
    /// activation buffers without re-parsing GGUF metadata).
    pub fn config_hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    /// Overwrite the current hidden state with `hidden`. Used by pipeline-
    /// parallel stages that receive activations over the ring.
    pub fn set_hidden_state(&mut self, hidden: &[f32]) -> Result<(), ModelError> {
        let h = self.config.hidden_size;
        if hidden.len() != h {
            return Err(ModelError::InferenceFailed(format!(
                "set_hidden_state: expected {} floats, got {}",
                h,
                hidden.len()
            )));
        }
        self.workspace.x[..h].copy_from_slice(hidden);
        Ok(())
    }

    /// RMSNorm on `hidden` using the model's final norm weights (for LoRA / training).
    pub fn apply_final_norm(&self, hidden: &[f32], out: &mut [f32]) -> Result<(), ModelError> {
        let h = self.config.hidden_size;
        if hidden.len() != h || out.len() != h {
            return Err(ModelError::InferenceFailed(format!(
                "apply_final_norm: expected {h} floats, got hidden={} out={}",
                hidden.len(),
                out.len()
            )));
        }
        rms_norm_f32(hidden, &self.norm_weight, self.config.rms_norm_eps, out)
            .map_err(|e| ModelError::InferenceFailed(format!("final_norm: {:?}", e)))
    }

    /// Final norm weights (read-only) for external training loops.
    pub fn final_norm_weight(&self) -> &[f32] {
        &self.norm_weight
    }

    /// Whether this GGUF contains a usable native MTP/nextn draft block.
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// Number of nextn layers advertised by GGUF metadata.
    pub fn nextn_predict_layers(&self) -> usize {
        self.config.nextn_predict_layers
    }

    /// Final output-normalized hidden row for the latest committed target token.
    pub fn last_output_hidden(&self) -> &[f32] {
        &self.last_output_hidden
    }

    /// Project already-normalized hidden states through the output (lm_head) matrix.
    pub fn lm_head_logits_from_normed(
        &self,
        normed: &[f32],
        logits: &mut [f32],
    ) -> Result<(), ModelError> {
        let h = self.config.hidden_size;
        if normed.len() != h {
            return Err(ModelError::InferenceFailed(format!(
                "lm_head_logits_from_normed: expected {h} floats, got {}",
                normed.len()
            )));
        }
        if logits.len() != self.config.vocab_size {
            return Err(ModelError::InferenceFailed(format!(
                "lm_head_logits_from_normed: logits len {} != vocab {}",
                logits.len(),
                self.config.vocab_size
            )));
        }
        logits.fill(0.0_f32);
        gemv_weight(
            &self.output_weight,
            self.config.vocab_size,
            h,
            normed,
            logits,
        )
        .map_err(|e| ModelError::InferenceFailed(format!("output: {:?}", e)))
    }

    /// Apply final RMSNorm + lm_head to the current hidden state in
    /// `workspace.x` and return the logits. Last stage of pipeline-parallel.
    pub fn final_head_from_workspace(&mut self) -> Result<Logits, ModelError> {
        let h = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let rms_norm_eps = self.config.rms_norm_eps;
        let (logits_out, last_hidden) = {
            let ws = &mut self.workspace;
            let x = &ws.x[..h];
            let normed = &mut ws.hidden_a[..h];
            normed.fill(0.0_f32);
            rms_norm_f32(x, &self.norm_weight, rms_norm_eps, normed)
                .map_err(|e| ModelError::InferenceFailed(format!("final_norm: {:?}", e)))?;
            let last_hidden = normed.to_vec();
            let logits = &mut ws.logits[..vocab_size];
            logits.fill(0.0_f32);
            gemv_weight(&self.output_weight, vocab_size, h, normed, logits)
                .map_err(|e| ModelError::InferenceFailed(format!("output: {:?}", e)))?;
            (logits.to_vec(), last_hidden)
        };
        self.last_output_hidden = last_hidden;
        Ok(logits_out)
    }

    /// Generate draft tokens with the native in-GGUF MTP/nextn block.
    ///
    /// `start_token` and `start_hidden` must describe the same committed target
    /// position. The first MTP step predicts the token after `start_token`; each
    /// accepted MTP row then feeds its sampled token and post-head-norm hidden row
    /// back into the next MTP step.
    pub fn draft_mtp_tokens(
        &mut self,
        start_token: Token,
        start_hidden: &[f32],
        max_tokens: usize,
        sampling: crate::sampling::SamplingConfig,
        random: &mut dyn FnMut() -> f32,
    ) -> Result<(Vec<Token>, Vec<Logits>), ModelError> {
        if max_tokens == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        if self.mtp.is_none() {
            return Err(ModelError::InferenceFailed(
                "model does not contain a usable MTP/nextn block".to_string(),
            ));
        }
        let h = self.config.hidden_size;
        if start_hidden.len() != h {
            return Err(ModelError::InferenceFailed(format!(
                "MTP hidden width mismatch: expected {h}, got {}",
                start_hidden.len()
            )));
        }

        let mtp_kv_config = KvCacheConfig {
            layer_count: 1,
            context_size: max_tokens.max(1),
            head_count: self.config.num_key_value_heads,
            head_dim: self.config.kv_head_dim(),
            dtype: DType::F32,
            quantization: crate::kv_cache::KvQuantization::default(),
        };
        let mut mtp_kv = KvCache::new(mtp_kv_config)
            .map_err(|e| ModelError::InferenceFailed(format!("mtp kv_cache: {e:?}")))?;

        let mut draft_tokens = Vec::with_capacity(max_tokens);
        let mut draft_logits = Vec::with_capacity(max_tokens);
        let mut current_token = start_token;
        let mut current_hidden = start_hidden.to_vec();
        for pos in 0..max_tokens {
            let (logits, next_hidden) =
                self.mtp_forward_one(current_token, &current_hidden, pos, &mut mtp_kv)?;
            let token = crate::sampling::sample(&logits, sampling, random())
                .map_err(|e| ModelError::InferenceFailed(format!("MTP sample: {e:?}")))?;
            draft_tokens.push(token);
            draft_logits.push(logits);
            current_token = token;
            current_hidden = next_hidden;
        }

        Ok((draft_tokens, draft_logits))
    }

    fn mtp_forward_one(
        &mut self,
        token: Token,
        previous_hidden: &[f32],
        pos: usize,
        mtp_kv: &mut KvCache,
    ) -> Result<(Logits, Vec<f32>), ModelError> {
        let mtp = self
            .mtp
            .as_ref()
            .ok_or_else(|| ModelError::InferenceFailed("missing MTP/nextn weights".to_string()))?;
        let h = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let rms_norm_eps = self.config.rms_norm_eps;

        let embed_storage = if mtp.embed_tokens.is_empty() {
            &self.tok_embeddings
        } else {
            &mtp.embed_tokens
        };
        let mut token_embedding = vec![0.0_f32; h];
        lookup_embedding_from_storage(embed_storage, h, vocab_size, token, &mut token_embedding);

        let mut embed_normed = vec![0.0_f32; h];
        rms_norm_f32(
            &token_embedding,
            &mtp.enorm,
            rms_norm_eps,
            &mut embed_normed,
        )
        .map_err(|e| ModelError::InferenceFailed(format!("mtp enorm: {e:?}")))?;
        let mut hidden_normed = vec![0.0_f32; h];
        rms_norm_f32(
            previous_hidden,
            &mtp.hnorm,
            rms_norm_eps,
            &mut hidden_normed,
        )
        .map_err(|e| ModelError::InferenceFailed(format!("mtp hnorm: {e:?}")))?;

        let mut concat = vec![0.0_f32; h * 2];
        concat[..h].copy_from_slice(&embed_normed);
        concat[h..].copy_from_slice(&hidden_normed);

        let mut fused = vec![0.0_f32; h];
        gemv_weight(&mtp.eh_proj, h, h * 2, &concat, &mut fused)
            .map_err(|e| ModelError::InferenceFailed(format!("mtp eh_proj: {e}")))?;
        self.workspace.x[..h].copy_from_slice(&fused);

        self.run_mtp_layer_in_workspace(pos, mtp_kv)?;

        let mtp = self
            .mtp
            .as_ref()
            .ok_or_else(|| ModelError::InferenceFailed("missing MTP/nextn weights".to_string()))?;
        let norm_weight = if mtp.shared_head_norm.is_empty() {
            &self.norm_weight
        } else {
            &mtp.shared_head_norm
        };
        let head_weight = if mtp.shared_head_head.is_empty() {
            &self.output_weight
        } else {
            &mtp.shared_head_head
        };

        let x = self.workspace.x[..h].to_vec();
        let mut mtp_hidden = vec![0.0_f32; h];
        rms_norm_f32(&x, norm_weight, rms_norm_eps, &mut mtp_hidden)
            .map_err(|e| ModelError::InferenceFailed(format!("mtp shared_head_norm: {e:?}")))?;
        let mut logits = vec![0.0_f32; vocab_size];
        gemv_weight(head_weight, vocab_size, h, &mtp_hidden, &mut logits)
            .map_err(|e| ModelError::InferenceFailed(format!("mtp shared_head: {e}")))?;
        Ok((logits, mtp_hidden))
    }

    fn run_mtp_layer_in_workspace(
        &mut self,
        pos: usize,
        mtp_kv: &mut KvCache,
    ) -> Result<(), ModelError> {
        let mtp = self
            .mtp
            .as_ref()
            .ok_or_else(|| ModelError::InferenceFailed("missing MTP/nextn weights".to_string()))?;
        let layer = &mtp.layer;
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let n = cfg.num_attention_heads;
        let k = cfg.num_key_value_heads;
        let mut x = self.workspace.x[..h].to_vec();

        let mut normed = vec![0.0_f32; h];
        rms_norm_f32(&x, &layer.attn_norm, cfg.rms_norm_eps, &mut normed)
            .map_err(|e| ModelError::InferenceFailed(format!("mtp attn_norm: {e:?}")))?;

        let qg_len = layer.attn_q.output_dim(h);
        let kv_len = layer.attn_k.output_dim(h);
        let attn_output_input_len = layer.attn_output.output_dim(h);
        if qg_len == 0 || kv_len == 0 || attn_output_input_len == 0 {
            return Err(ModelError::InferenceFailed(format!(
                "invalid MTP attention dims qg={qg_len} kv={kv_len} out_in={attn_output_input_len}"
            )));
        }

        let mut qg = vec![0.0_f32; qg_len];
        let mut k_vec = vec![0.0_f32; kv_len];
        let mut v_vec = vec![0.0_f32; kv_len];
        gemv_weight_fused(
            vec![
                (&layer.attn_q, qg_len, &mut qg[..]),
                (&layer.attn_k, kv_len, &mut k_vec[..]),
                (&layer.attn_v, kv_len, &mut v_vec[..]),
            ],
            h,
            &normed,
        )
        .map_err(|e| ModelError::InferenceFailed(format!("mtp qkv: {e}")))?;
        if !layer.attn_q_bias.is_empty() {
            for (i, q) in qg.iter_mut().enumerate() {
                *q += layer.attn_q_bias[i % layer.attn_q_bias.len()];
            }
        }
        if !layer.attn_k_bias.is_empty() {
            for (i, value) in k_vec.iter_mut().enumerate() {
                *value += layer.attn_k_bias[i % layer.attn_k_bias.len()];
            }
        }
        if !layer.attn_v_bias.is_empty() {
            for (i, value) in v_vec.iter_mut().enumerate() {
                *value += layer.attn_v_bias[i % layer.attn_v_bias.len()];
            }
        }

        let q_len = qg_len.min(attn_output_input_len);
        let gate = (qg_len >= q_len.saturating_mul(2)).then(|| qg[q_len..q_len + q_len].to_vec());
        let mut q = qg[..q_len].to_vec();
        let q_head_dim = if n > 0 && q_len.is_multiple_of(n) {
            q_len / n
        } else {
            q_len
        };
        let q_heads = q_len.checked_div(q_head_dim.max(1)).unwrap_or(1);
        let kv_head_dim = if k > 0 && kv_len.is_multiple_of(k) {
            kv_len / k
        } else {
            kv_len
        };
        let kv_heads = kv_len.checked_div(kv_head_dim.max(1)).unwrap_or(1);

        if !layer.attn_q_norm.is_empty() && q.len() == layer.attn_q_norm.len() {
            let mut normed_q = vec![0.0_f32; q.len()];
            rms_norm_f32(&q, &layer.attn_q_norm, cfg.rms_norm_eps, &mut normed_q)
                .map_err(|e| ModelError::InferenceFailed(format!("mtp q_norm: {e:?}")))?;
            q.copy_from_slice(&normed_q);
        } else if !layer.attn_q_norm.is_empty() && q_head_dim == layer.attn_q_norm.len() {
            let mut normed_head = vec![0.0_f32; q_head_dim];
            for head in 0..q_heads {
                let start = head * q_head_dim;
                let end = start + q_head_dim;
                if end > q.len() {
                    break;
                }
                rms_norm_f32(
                    &q[start..end],
                    &layer.attn_q_norm,
                    cfg.rms_norm_eps,
                    &mut normed_head,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("mtp q_norm: {e:?}")))?;
                q[start..end].copy_from_slice(&normed_head);
            }
        }
        if !layer.attn_k_norm.is_empty() && k_vec.len() == layer.attn_k_norm.len() {
            let mut normed_k = vec![0.0_f32; k_vec.len()];
            rms_norm_f32(&k_vec, &layer.attn_k_norm, cfg.rms_norm_eps, &mut normed_k)
                .map_err(|e| ModelError::InferenceFailed(format!("mtp k_norm: {e:?}")))?;
            k_vec.copy_from_slice(&normed_k);
        } else if !layer.attn_k_norm.is_empty() && kv_head_dim == layer.attn_k_norm.len() {
            let mut normed_head = vec![0.0_f32; kv_head_dim];
            for head in 0..kv_heads {
                let start = head * kv_head_dim;
                let end = start + kv_head_dim;
                if end > k_vec.len() {
                    break;
                }
                rms_norm_f32(
                    &k_vec[start..end],
                    &layer.attn_k_norm,
                    cfg.rms_norm_eps,
                    &mut normed_head,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("mtp k_norm: {e:?}")))?;
                k_vec[start..end].copy_from_slice(&normed_head);
            }
        }

        let q_rope_len = cfg.effective_rope_dim().min(q_head_dim);
        let mut rope_scratch = vec![0.0_f32; q_rope_len.max(kv_head_dim)];
        for head in 0..q_heads {
            let off = head * q_head_dim;
            if off + q_head_dim > q.len() {
                break;
            }
            let rotated = &mut rope_scratch[..q_rope_len];
            apply_rope_f32(
                &q[off..off + q_rope_len],
                pos,
                q_rope_len,
                cfg.rope_theta,
                rotated,
            )
            .map_err(|e| ModelError::InferenceFailed(format!("mtp rope q: {e:?}")))?;
            q[off..off + q_rope_len].copy_from_slice(rotated);
        }
        let k_rope_len = cfg.effective_rope_dim().min(kv_head_dim);
        for head in 0..kv_heads {
            let off = head * kv_head_dim;
            if off + kv_head_dim > k_vec.len() {
                break;
            }
            let rotated = &mut rope_scratch[..k_rope_len];
            apply_rope_f32(
                &k_vec[off..off + k_rope_len],
                pos,
                k_rope_len,
                cfg.rope_theta,
                rotated,
            )
            .map_err(|e| ModelError::InferenceFailed(format!("mtp rope k: {e:?}")))?;
            k_vec[off..off + k_rope_len].copy_from_slice(rotated);
        }

        mtp_kv
            .set(0, pos, &k_vec, &v_vec)
            .map_err(|e| ModelError::InferenceFailed(format!("mtp kv set: {e:?}")))?;
        let seq_len = pos + 1;
        let key_cache = mtp_kv
            .f32_layer_key_prefix(0, seq_len)
            .map_err(|e| ModelError::InferenceFailed(format!("mtp kv keys: {e:?}")))?
            .ok_or_else(|| ModelError::InferenceFailed("MTP KV cache is not f32".to_string()))?;
        let value_cache = mtp_kv
            .f32_layer_value_prefix(0, seq_len)
            .map_err(|e| ModelError::InferenceFailed(format!("mtp kv values: {e:?}")))?
            .ok_or_else(|| ModelError::InferenceFailed("MTP KV cache is not f32".to_string()))?;

        let q_for_flash = if q_head_dim > kv_head_dim {
            let mut truncated = vec![0.0_f32; q_heads * kv_head_dim];
            for head in 0..q_heads {
                let src = head * q_head_dim;
                let dst = head * kv_head_dim;
                truncated[dst..dst + kv_head_dim].copy_from_slice(&q[src..src + kv_head_dim]);
            }
            truncated
        } else {
            q.clone()
        };
        let mut attn_result = vec![0.0_f32; q_for_flash.len()];
        flash_attention_decode_heads_f32(
            &q_for_flash,
            key_cache,
            value_cache,
            seq_len,
            kv_head_dim,
            kv_len,
            q_heads,
            kv_heads,
            &mut attn_result,
        )
        .map_err(|e| ModelError::InferenceFailed(format!("mtp attention: {e:?}")))?;
        if let Some(gate) = gate.as_ref()
            && gate.len() == attn_result.len()
        {
            for (out, gate_value) in attn_result.iter_mut().zip(gate.iter()) {
                let sigmoid = 1.0_f32 / (1.0 + (-*gate_value).exp());
                *out *= sigmoid;
            }
        }

        let attn_input = if attn_result.len() == attn_output_input_len {
            attn_result
        } else {
            let mut padded = vec![0.0_f32; attn_output_input_len];
            let copy = padded.len().min(attn_result.len());
            padded[..copy].copy_from_slice(&attn_result[..copy]);
            padded
        };
        let mut attn_out = vec![0.0_f32; h];
        gemv_weight(
            &layer.attn_output,
            h,
            attn_output_input_len,
            &attn_input,
            &mut attn_out,
        )
        .map_err(|e| ModelError::InferenceFailed(format!("mtp attn_output: {e}")))?;
        if !layer.attn_output_bias.is_empty() {
            for (i, out) in attn_out.iter_mut().enumerate() {
                *out += layer.attn_output_bias[i % layer.attn_output_bias.len()];
            }
        }
        for i in 0..h {
            x[i] += attn_out[i];
        }

        let ffn_norm_weight = if !layer.post_attention_norm.is_empty() {
            &layer.post_attention_norm
        } else {
            &layer.ffn_norm
        };
        if ffn_norm_weight.is_empty() {
            return Err(ModelError::InferenceFailed(
                "MTP block is missing post_attention_norm/ffn_norm".to_string(),
            ));
        }
        let mut ffn_normed = vec![0.0_f32; h];
        rms_norm_f32(&x, ffn_norm_weight, cfg.rms_norm_eps, &mut ffn_normed)
            .map_err(|e| ModelError::InferenceFailed(format!("mtp ffn_norm: {e:?}")))?;
        let mut gate = vec![0.0_f32; cfg.intermediate_size];
        let mut up = vec![0.0_f32; cfg.intermediate_size];
        gemv_weight_fused(
            vec![
                (&layer.ffn_gate, cfg.intermediate_size, &mut gate[..]),
                (&layer.ffn_up, cfg.intermediate_size, &mut up[..]),
            ],
            h,
            &ffn_normed,
        )
        .map_err(|e| ModelError::InferenceFailed(format!("mtp ffn gate/up: {e}")))?;
        if cfg.gelu_ffn {
            apply_geglu_inplace_f32(&mut gate, &up);
        } else {
            apply_swiglu_inplace_f32(&mut gate, &up);
        }
        let mut ffn_out = vec![0.0_f32; h];
        gemv_weight(
            &layer.ffn_down,
            h,
            cfg.intermediate_size,
            &gate,
            &mut ffn_out,
        )
        .map_err(|e| ModelError::InferenceFailed(format!("mtp ffn_down: {e}")))?;
        if !layer.ffn_down_bias.is_empty() {
            for (i, out) in ffn_out.iter_mut().enumerate() {
                *out += layer.ffn_down_bias[i % layer.ffn_down_bias.len()];
            }
        }
        for i in 0..h {
            x[i] += ffn_out[i];
        }

        self.workspace.x[..h].copy_from_slice(&x);
        Ok(())
    }

    /// Run layers `range` against the hidden state currently in
    /// `workspace.x[..hidden_size]`, mutating it in place. `pos` is the
    /// absolute position for KV cache writes / RoPE.
    pub fn run_layer_range_in_workspace(
        &mut self,
        pos: usize,
        range: std::ops::Range<usize>,
    ) -> Result<(), ModelError> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let n = cfg.num_attention_heads;
        let k = cfg.num_key_value_heads;
        let ws = &mut self.workspace;
        // Silence unused warnings for paths that don't reference both names.
        let _ = (n, k);

        for layer_idx in range {
            let layer = &self.layers[layer_idx];

            // Detect LFM2 short-convolution layers (have shortconv.in_proj, no attention).
            let is_shortconv = !layer.shortconv_in_proj.is_empty();
            // Detect Mamba layers (have attn_qkv but no attn_q)
            let is_mamba = !layer.attn_qkv.is_empty() && layer.attn_q.is_empty();

            // Determine which norm weight to use for FFN
            let ffn_norm_weight: &[f32] = if cfg.sandwich_norm {
                // Gemma: post_attention_norm is a sandwich norm (applied to the
                // attention output), NOT the pre-FFN norm. Use ffn_norm here.
                &layer.ffn_norm
            } else if !layer.post_attention_norm.is_empty() {
                &layer.post_attention_norm
            } else if !layer.ffn_norm.is_empty() {
                &layer.ffn_norm
            } else {
                &[]
            };

            // Per-layer RoPE theta and sliding-window size (Gemma local/global mix).
            let layer_rope = cfg.layer_rope_theta(layer_idx);
            let layer_window = cfg.layer_sliding_window(layer_idx);

            if is_shortconv {
                // ---- LFM2 short-convolution token mixing ----
                // operator_norm -> in_proj -> (B,C,x) -> Bx=B*x ->
                // causal depthwise conv1d (kernel = l_cache) -> y=C*conv -> out_proj
                let l_cache = cfg.shortconv_l_cache.max(1);
                let d = layer.shortconv_in_proj.output_dim(h) / 3;

                let normed = &mut ws.hidden_b[..h];
                normed.fill(0.0_f32);
                rms_norm_f32(&ws.x[..h], &layer.attn_norm, cfg.rms_norm_eps, normed)
                    .map_err(|e| ModelError::InferenceFailed(format!("shortconv_norm: {:?}", e)))?;
                let bcx = &mut ws.shortconv_bcx[..3 * d];
                bcx.fill(0.0_f32);
                gemv_weight(&layer.shortconv_in_proj, 3 * d, h, normed, bcx).map_err(|e| {
                    ModelError::InferenceFailed(format!("shortconv_in_proj: {:?}", e))
                })?;

                // Bx = B * x   (B = bcx[0..d], C = bcx[d..2d], x = bcx[2d..3d])
                let bx = &mut ws.shortconv_bx[..d];
                for i in 0..d {
                    bx[i] = bcx[i] * bcx[2 * d + i];
                }

                // Causal depthwise conv1d. Weights laid out [l_cache, d] tap-major;
                // the last tap aligns with the current token (llama.cpp ssm_conv order).
                let conv_out = &mut ws.conv_out[..d];
                let have_conv = layer.shortconv_conv.len() == l_cache * d;
                {
                    let buf = &self.ssm_conv_buffers[layer_idx];
                    if have_conv {
                        // Weights are channel-major: [d, l_cache] with the l_cache
                        // taps contiguous per channel. Last tap = current token.
                        for c in 0..d {
                            let base = c * l_cache;
                            let mut sum = layer.shortconv_conv[base + (l_cache - 1)] * bx[c];
                            for j in 1..l_cache {
                                if let Some(prev) = buf.past_frame(j) {
                                    sum += layer.shortconv_conv[base + (l_cache - 1 - j)] * prev[c];
                                }
                            }
                            conv_out[c] = sum;
                        }
                    } else {
                        conv_out.copy_from_slice(bx);
                    }
                }

                // y = C * conv_out
                for i in 0..d {
                    conv_out[i] *= bcx[d + i];
                }

                // out_proj: [d] -> [h]
                let attn_out = &mut ws.hidden_a[..h];
                attn_out.fill(0.0_f32);
                gemv_weight(&layer.shortconv_out_proj, h, d, conv_out, attn_out).map_err(|e| {
                    ModelError::InferenceFailed(format!("shortconv_out_proj: {:?}", e))
                })?;

                self.ssm_conv_buffers[layer_idx].push(bx);

                for i in 0..h {
                    ws.x[i] += attn_out[i];
                }
            } else if is_mamba {
                // ---- Mamba/SSM layer ----
                let mamba_out = {
                    let normed = &mut ws.hidden_a[..h];
                    normed.fill(0.0_f32);
                    rms_norm_f32(&ws.x[..h], &layer.attn_norm, cfg.rms_norm_eps, normed)
                        .map_err(|e| ModelError::InferenceFailed(format!("mamba_norm: {:?}", e)))?;

                    // Gate branch
                    let gate_len = if !layer.attn_gate.is_empty() {
                        layer.attn_gate.output_dim(h)
                    } else {
                        0
                    };
                    if gate_len > 0 {
                        let gate = &mut ws.intermediate_a[..gate_len];
                        gate.fill(0.0_f32);
                        gemv_weight(&layer.attn_gate, gate_len, h, normed, gate).map_err(|e| {
                            ModelError::InferenceFailed(format!("attn_gate: {:?}", e))
                        })?;
                    }

                    // SSM branch projection: [h] -> [qkv_out_len]
                    let qkv_out_len = layer.attn_qkv.output_dim(h);
                    let x_proj = &mut ws.q_full[..qkv_out_len];
                    x_proj.fill(0.0_f32);
                    gemv_weight(&layer.attn_qkv, qkv_out_len, h, normed, x_proj)
                        .map_err(|e| ModelError::InferenceFailed(format!("attn_qkv: {:?}", e)))?;

                    // Causal conv1d over qkv_out_len channels
                    let conv_kernel = 4_usize;
                    let conv_out = &mut ws.conv_out[..qkv_out_len];
                    conv_out.fill(0.0_f32);
                    if !layer.ssm_conv1d.is_empty()
                        && layer.ssm_conv1d.len() == conv_kernel * qkv_out_len
                    {
                        let buffer = &self.ssm_conv_buffers[layer_idx];
                        for c in 0..qkv_out_len {
                            let mut sum = 0.0_f32;
                            // Tap-major [kernel, channels]; newest input uses the last tap.
                            sum +=
                                layer.ssm_conv1d[(conv_kernel - 1) * qkv_out_len + c] * x_proj[c];
                            for b in 1..conv_kernel {
                                if let Some(prev) = buffer.past_frame(b) {
                                    let weight_idx = (conv_kernel - 1 - b) * qkv_out_len + c;
                                    sum += layer.ssm_conv1d[weight_idx] * prev[c];
                                }
                            }
                            conv_out[c] = sum;
                        }
                    } else {
                        conv_out.copy_from_slice(x_proj);
                    }

                    self.ssm_conv_buffers[layer_idx].push(x_proj);

                    // SiLU activation
                    for val in conv_out.iter_mut() {
                        *val = *val * (1.0_f32 / (1.0_f32 + (-*val).exp()));
                    }

                    // Split into SSM input and gate
                    let half = qkv_out_len / 2;
                    let mut mamba_out = vec![0.0_f32; half];
                    let mut x_ssm = conv_out[..half].to_vec();
                    let z_gate: Vec<f32> = if qkv_out_len > half {
                        conv_out[half..].to_vec()
                    } else {
                        vec![0.0_f32; half]
                    };

                    // Group RMSNorm on x_ssm
                    if !layer.ssm_norm.is_empty() && !x_ssm.is_empty() {
                        let group_size = layer.ssm_norm.len();
                        if x_ssm.len().is_multiple_of(group_size) {
                            let num_groups = x_ssm.len() / group_size;
                            for g in 0..num_groups {
                                let start = g * group_size;
                                let end = start + group_size;
                                let mut normed_group = vec![0.0_f32; group_size];
                                rms_norm_f32(
                                    &x_ssm[start..end],
                                    &layer.ssm_norm,
                                    cfg.rms_norm_eps,
                                    &mut normed_group,
                                )
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!("ssm_norm: {:?}", e))
                                })?;
                                x_ssm[start..end].copy_from_slice(&normed_group);
                            }
                        }
                    }

                    // Selective Scan SSM
                    let state_dim = self.ssm_states[layer_idx].len();
                    if state_dim > 0
                        && !layer.ssm_a.is_empty()
                        && !layer.ssm_alpha.is_empty()
                        && !layer.ssm_beta.is_empty()
                    {
                        // Compute Bx: ssm_beta maps x_ssm -> state
                        // ssm_beta is [x_ssm_len, state_dim], stored row-major
                        let mut bx = vec![0.0_f32; state_dim];
                        let x_ssm_len = x_ssm.len();
                        if layer.ssm_beta.len() == x_ssm_len * state_dim {
                            for (j, &x_value) in x_ssm.iter().enumerate().take(x_ssm_len) {
                                for (i, bx_value) in bx.iter_mut().enumerate().take(state_dim) {
                                    *bx_value += layer.ssm_beta[j * state_dim + i] * x_value;
                                }
                            }
                        }

                        // Update state: h = h * exp(A * delta) + Bx * delta
                        for (i, &bx_value) in bx.iter().enumerate().take(state_dim) {
                            let a = layer.ssm_a[i % layer.ssm_a.len()];
                            let dt = if !layer.ssm_dt_bias.is_empty() {
                                let b = layer.ssm_dt_bias[i % layer.ssm_dt_bias.len()];
                                (1.0_f32 + b.exp()).ln() // softplus
                            } else {
                                0.01_f32
                            };
                            let decay = (a * dt).exp();
                            self.ssm_states[layer_idx][i] =
                                self.ssm_states[layer_idx][i] * decay + bx_value * dt;
                        }

                        // Compute output: y = C * h = ssm_alpha * state
                        // ssm_alpha is [y_len, state_dim]
                        let y_len = layer.ssm_alpha.len() / state_dim;
                        let mut y_ssm = vec![0.0_f32; y_len];
                        if layer.ssm_alpha.len() == y_len * state_dim {
                            for (j, y_value) in y_ssm.iter_mut().enumerate().take(y_len) {
                                for (i, &state_value) in self.ssm_states[layer_idx]
                                    .iter()
                                    .enumerate()
                                    .take(state_dim)
                                {
                                    *y_value += layer.ssm_alpha[j * state_dim + i] * state_value;
                                }
                            }
                        }

                        // Pad or truncate y_ssm to match the Mamba inner width.
                        if y_ssm.len() >= mamba_out.len() {
                            let out_len = mamba_out.len();
                            mamba_out.copy_from_slice(&y_ssm[..out_len]);
                        } else {
                            mamba_out[..y_ssm.len()].copy_from_slice(&y_ssm);
                        }
                    }

                    // Apply gate: y = y * silu(z_gate or gate)
                    let gate_to_use: Vec<f32> = if gate_len > 0 && gate_len == mamba_out.len() {
                        // Use attn_gate if available
                        let silu_gate = &mut ws.intermediate_a[..gate_len];
                        for val in silu_gate.iter_mut() {
                            *val = *val * (1.0_f32 / (1.0_f32 + (-*val).exp()));
                        }
                        silu_gate[..mamba_out.len()].to_vec()
                    } else if z_gate.len() == mamba_out.len() {
                        // Use second half of qkv projection
                        z_gate.clone()
                    } else {
                        vec![]
                    };

                    if gate_to_use.len() == mamba_out.len() {
                        for i in 0..mamba_out.len() {
                            mamba_out[i] *= gate_to_use[i];
                        }
                    }

                    // Final output projection
                    let mut residual = vec![0.0_f32; h];
                    if !layer.ssm_out.is_empty() {
                        let out_len = layer.ssm_out.output_dim(mamba_out.len());
                        if out_len > 0 {
                            let mut projected = vec![0.0_f32; out_len];
                            gemv_weight(
                                &layer.ssm_out,
                                out_len,
                                mamba_out.len(),
                                &mamba_out,
                                &mut projected,
                            )
                            .map_err(|e| {
                                ModelError::InferenceFailed(format!("ssm_out: {:?}", e))
                            })?;
                            let copy_len = h.min(projected.len());
                            residual[..copy_len].copy_from_slice(&projected[..copy_len]);
                        }
                    } else {
                        let copy_len = h.min(mamba_out.len());
                        residual[..copy_len].copy_from_slice(&mamba_out[..copy_len]);
                    }
                    residual
                };

                for i in 0..h {
                    ws.x[i] += mamba_out[i];
                }
            } else if !layer.mla_kv_a_mqa.is_empty() {
                let kv_layer_idx = self
                    .kv_layer_map
                    .get(layer_idx)
                    .copied()
                    .flatten()
                    .unwrap_or(layer_idx);
                ws.hidden_a[..h].fill(0.0_f32);
                {
                    let kv_cache = &mut self.kv_cache;
                    Self::deepseek_mla_layer(kv_cache, layer, cfg, kv_layer_idx, pos, ws)?;
                }
                for i in 0..h {
                    ws.x[i] += ws.hidden_a[i];
                }
            } else if !layer.attn_q.is_empty() {
                // ---- Standard attention ----
                // Look up the KV cache index for this attention layer.  Non-attention
                // layers are skipped above, so this entry is always Some for this path.
                let kv_layer_idx = self
                    .kv_layer_map
                    .get(layer_idx)
                    .copied()
                    .flatten()
                    .unwrap_or(layer_idx);
                let attn_out = &mut ws.hidden_a[..h];
                attn_out.fill(0.0_f32);
                {
                    let normed = &mut ws.hidden_b[..h];
                    normed.fill(0.0_f32);
                    rms_norm_f32(&ws.x[..h], &layer.attn_norm, cfg.rms_norm_eps, normed)
                        .map_err(|e| ModelError::InferenceFailed(format!("rms_norm: {:?}", e)))?;

                    // Compute dynamic dimensions
                    let q_len = layer.attn_q.output_dim(h);
                    let kv_len = if !layer.attn_k.is_empty() {
                        layer.attn_k.output_dim(h)
                    } else {
                        0
                    };
                    let attn_output_input_len = if !layer.attn_output.is_empty() {
                        layer.attn_output.output_dim(h)
                    } else {
                        0
                    };

                    let q_full = &mut ws.q_full[..q_len];
                    q_full.fill(0.0_f32);
                    let k_vec = &mut ws.k_vec[..kv_len];
                    k_vec.fill(0.0_f32);
                    let v_vec = &mut ws.v_vec[..kv_len];
                    v_vec.fill(0.0_f32);

                    // Run Q, K, V projections as ONE fused parallel region —
                    // they share the same normed input and write to
                    // non-overlapping buffers (q_full, k_vec, v_vec).
                    gemv_weight_fused(
                        vec![
                            (&layer.attn_q, q_len, &mut *q_full),
                            (
                                &layer.attn_k,
                                if layer.attn_k.is_empty() { 0 } else { kv_len },
                                &mut *k_vec,
                            ),
                            (
                                &layer.attn_v,
                                if layer.attn_v.is_empty() { 0 } else { kv_len },
                                &mut *v_vec,
                            ),
                        ],
                        h,
                        normed,
                    )
                    .map_err(|e| ModelError::InferenceFailed(format!("attn_qkv: {:?}", e)))?;
                    let glue_t0 =
                        crate::tensor::decode_profile_enabled().then(std::time::Instant::now);

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

                    // In MLA-style attention, attn_q output is split into Q and auxiliary projection.
                    let q_len_actual = if attn_output_input_len > 0 {
                        q_len.min(attn_output_input_len)
                    } else if q_len > h {
                        h
                    } else {
                        q_len
                    };
                    let q = &mut ws.q_full[..q_len_actual];

                    // Compute head dimensions based on ACTUAL Q length after splitting
                    let q_len_used = q.len();
                    let q_head_dim = if n > 0 && q_len_used.is_multiple_of(n) {
                        q_len_used / n
                    } else {
                        q_len_used
                    };
                    let q_heads = q_len_used.checked_div(q_head_dim).unwrap_or(1);

                    let kv_head_dim = if k > 0 && kv_len.is_multiple_of(k) {
                        kv_len / k
                    } else if kv_len > 0 {
                        kv_len
                    } else {
                        q_head_dim
                    };
                    let kv_heads = kv_len.checked_div(kv_head_dim).unwrap_or(1);

                    // Apply per-head Q/K norm if available
                    if !layer.attn_q_norm.is_empty() && q.len() == layer.attn_q_norm.len() {
                        let normed_q = &mut ws.flash_q[..q.len()];
                        rms_norm_f32(q, &layer.attn_q_norm, cfg.rms_norm_eps, normed_q)
                            .map_err(|e| ModelError::InferenceFailed(format!("q_norm: {:?}", e)))?;
                        q.copy_from_slice(normed_q);
                    } else if !layer.attn_q_norm.is_empty() && q_head_dim == layer.attn_q_norm.len()
                    {
                        for head in 0..q_heads {
                            let start = head * q_head_dim;
                            let end = start + q_head_dim;
                            if end > q.len() {
                                break;
                            }
                            let normed_head = &mut ws.head_scratch[..q_head_dim];
                            normed_head.fill(0.0_f32);
                            rms_norm_f32(
                                &q[start..end],
                                &layer.attn_q_norm,
                                cfg.rms_norm_eps,
                                normed_head,
                            )
                            .map_err(|e| ModelError::InferenceFailed(format!("q_norm: {:?}", e)))?;
                            q[start..end].copy_from_slice(normed_head);
                        }
                    }
                    if !layer.attn_k_norm.is_empty() && k_vec.len() == layer.attn_k_norm.len() {
                        let normed_k = &mut ws.attn_result[..k_vec.len()];
                        rms_norm_f32(k_vec, &layer.attn_k_norm, cfg.rms_norm_eps, normed_k)
                            .map_err(|e| ModelError::InferenceFailed(format!("k_norm: {:?}", e)))?;
                        k_vec.copy_from_slice(normed_k);
                    } else if !layer.attn_k_norm.is_empty()
                        && kv_head_dim == layer.attn_k_norm.len()
                    {
                        for head in 0..kv_heads {
                            let start = head * kv_head_dim;
                            let end = start + kv_head_dim;
                            if end > k_vec.len() {
                                break;
                            }
                            let normed_head = &mut ws.head_scratch[..kv_head_dim];
                            normed_head.fill(0.0_f32);
                            rms_norm_f32(
                                &k_vec[start..end],
                                &layer.attn_k_norm,
                                cfg.rms_norm_eps,
                                normed_head,
                            )
                            .map_err(|e| ModelError::InferenceFailed(format!("k_norm: {:?}", e)))?;
                            k_vec[start..end].copy_from_slice(normed_head);
                        }
                    }

                    // apply RoPE to Q (partial RoPE: first rope_dim elements per head)
                    let q_rope_len = cfg.effective_rope_dim().min(q_head_dim);
                    for head in 0..q_heads {
                        let off = head * q_head_dim;
                        if off + q_head_dim > q.len() {
                            break;
                        }
                        let rotated = &mut ws.head_scratch[..q_rope_len];
                        rotated.fill(0.0_f32);
                        apply_rope_f32(
                            &q[off..off + q_rope_len],
                            pos,
                            q_rope_len,
                            layer_rope,
                            rotated,
                        )
                        .map_err(|e| ModelError::InferenceFailed(format!("rope q: {:?}", e)))?;
                        q[off..off + q_rope_len].copy_from_slice(rotated);
                    }

                    // apply RoPE to K (partial RoPE)
                    let k_rope_len = cfg.effective_rope_dim().min(kv_head_dim);
                    for head in 0..kv_heads {
                        let off = head * kv_head_dim;
                        if off + kv_head_dim > k_vec.len() {
                            break;
                        }
                        let rotated = &mut ws.head_scratch[..k_rope_len];
                        rotated.fill(0.0_f32);
                        apply_rope_f32(
                            &k_vec[off..off + k_rope_len],
                            pos,
                            k_rope_len,
                            layer_rope,
                            rotated,
                        )
                        .map_err(|e| ModelError::InferenceFailed(format!("rope k: {:?}", e)))?;
                        k_vec[off..off + k_rope_len].copy_from_slice(rotated);
                    }

                    // store in KV cache
                    self.kv_cache
                        .set(kv_layer_idx, pos, k_vec, v_vec)
                        .map_err(|e| ModelError::InferenceFailed(format!("kv set: {:?}", e)))?;

                    let seq_len = pos + 1;

                    // compute attention using parallel flash attention decode over heads
                    let attn_result = &mut ws.attn_result[..q_len_used];
                    attn_result.fill(0.0_f32);

                    // For MLA-style where q_head_dim > kv_head_dim, truncate Q heads into scratch.
                    // Otherwise pass the Q projection directly and avoid a per-layer allocation.
                    let q_for_flash: &[f32] = if q_head_dim > kv_head_dim {
                        let q_truncated = &mut ws.flash_q[..q_heads * kv_head_dim];
                        for head in 0..q_heads {
                            let src_start = head * q_head_dim;
                            let dst_start = head * kv_head_dim;
                            q_truncated[dst_start..dst_start + kv_head_dim]
                                .copy_from_slice(&q[src_start..src_start + kv_head_dim]);
                        }
                        q_truncated
                    } else {
                        q
                    };

                    // Borrow the KV prefix in its storage dtype when the logical
                    // prefix is still contiguous in storage (F32 directly, F16 as
                    // half bits converted in-kernel); otherwise dequantize-copy
                    // into workspace buffers. Borrowing avoids materializing an
                    // f32 prefix copy per layer per token, and F16 also halves
                    // the attention DRAM reads vs an F32 cache.
                    let f16_keys = self
                        .kv_cache
                        .f16_layer_key_prefix(kv_layer_idx, seq_len)
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("kv borrow f16 keys: {:?}", e))
                        })?;
                    let f16_values = self
                        .kv_cache
                        .f16_layer_value_prefix(kv_layer_idx, seq_len)
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("kv borrow f16 values: {:?}", e))
                        })?;
                    if let (Some(key16), Some(value16)) = (f16_keys, f16_values) {
                        // Sliding-window attention: a local layer attends only to
                        // the most recent `layer_window` positions (see the F32
                        // branch below for why slicing preserves the mask).
                        let (eff_seq_len, key16, value16) =
                            if layer_window > 0 && seq_len > layer_window {
                                let skip = (seq_len - layer_window) * kv_len;
                                (layer_window, &key16[skip..], &value16[skip..])
                            } else {
                                (seq_len, key16, value16)
                            };
                        if let Some(t0) = glue_t0 {
                            crate::tensor::decode_profile_record(
                                "pre_attn_glue",
                                t0.elapsed().as_nanos() as u64,
                            );
                        }
                        let attn_t0 =
                            crate::tensor::decode_profile_enabled().then(std::time::Instant::now);
                        flash_attention_decode_heads_f16(
                            q_for_flash,
                            key16,
                            value16,
                            eff_seq_len,
                            kv_head_dim,
                            kv_len,
                            q_heads,
                            kv_heads,
                            attn_result,
                        )
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!(
                                "flash attention heads (f16): {:?}",
                                e
                            ))
                        })?;
                        if let Some(t0) = attn_t0 {
                            crate::tensor::decode_profile_record(
                                "attention",
                                t0.elapsed().as_nanos() as u64,
                            );
                        }
                    } else {
                        let borrowed_key_cache = self
                            .kv_cache
                            .f32_layer_key_prefix(kv_layer_idx, seq_len)
                            .map_err(|e| {
                                ModelError::InferenceFailed(format!("kv borrow keys: {:?}", e))
                            })?;
                        let borrowed_value_cache = self
                            .kv_cache
                            .f32_layer_value_prefix(kv_layer_idx, seq_len)
                            .map_err(|e| {
                                ModelError::InferenceFailed(format!("kv borrow values: {:?}", e))
                            })?;

                        let key_cache: &[f32];
                        let value_cache: &[f32];
                        if let (Some(keys), Some(values)) =
                            (borrowed_key_cache, borrowed_value_cache)
                        {
                            key_cache = keys;
                            value_cache = values;
                        } else {
                            let key_copy = &mut ws.kv_keys_copy[..seq_len * kv_len];
                            let value_copy = &mut ws.kv_values_copy[..seq_len * kv_len];
                            self.kv_cache
                                .copy_layer_keys(kv_layer_idx, seq_len, key_copy)
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!("kv copy keys: {:?}", e))
                                })?;
                            self.kv_cache
                                .copy_layer_values(kv_layer_idx, seq_len, value_copy)
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!("kv copy values: {:?}", e))
                                })?;
                            key_cache = key_copy;
                            value_cache = value_copy;
                        }

                        // Sliding-window attention: a local layer attends only to the
                        // most recent `layer_window` positions. RoPE encodes absolute
                        // positions, so slicing off the oldest rows yields the
                        // windowed-causal mask with relative positions preserved.
                        let (eff_seq_len, key_cache, value_cache) =
                            if layer_window > 0 && seq_len > layer_window {
                                let skip = (seq_len - layer_window) * kv_len;
                                (layer_window, &key_cache[skip..], &value_cache[skip..])
                            } else {
                                (seq_len, key_cache, value_cache)
                            };
                        if let Some(t0) = glue_t0 {
                            crate::tensor::decode_profile_record(
                                "pre_attn_glue",
                                t0.elapsed().as_nanos() as u64,
                            );
                        }
                        let attn_t0 =
                            crate::tensor::decode_profile_enabled().then(std::time::Instant::now);
                        flash_attention_decode_heads_f32(
                            q_for_flash,
                            key_cache,
                            value_cache,
                            eff_seq_len,
                            kv_head_dim,
                            kv_len,
                            q_heads,
                            kv_heads,
                            attn_result,
                        )
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("flash attention heads: {:?}", e))
                        })?;
                        if let Some(t0) = attn_t0 {
                            crate::tensor::decode_profile_record(
                                "attention",
                                t0.elapsed().as_nanos() as u64,
                            );
                        }
                    }

                    // Reconcile attention result size with attn_output expected input
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
                        gemv_weight(
                            &layer.attn_output,
                            h,
                            attn_output_input_len,
                            attn_input,
                            attn_out,
                        )
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("attn_output: {:?}", e))
                        })?;
                        if !layer.attn_output_bias.is_empty() {
                            for (i, out) in attn_out.iter_mut().enumerate() {
                                *out += layer.attn_output_bias[i % layer.attn_output_bias.len()];
                            }
                        }
                    }
                }

                // Gemma sandwich norm: normalize the attention output before the
                // residual add (post_attention_norm).
                if cfg.sandwich_norm && !layer.post_attention_norm.is_empty() {
                    let normed_attn = &mut ws.hidden_b[..h];
                    rms_norm_f32(
                        attn_out,
                        &layer.post_attention_norm,
                        cfg.rms_norm_eps,
                        normed_attn,
                    )
                    .map_err(|e| ModelError::InferenceFailed(format!("post_attn_norm: {:?}", e)))?;
                    attn_out.copy_from_slice(normed_attn);
                }

                for i in 0..h {
                    ws.x[i] += attn_out[i];
                }
            }

            // ---- FFN (shared between Mamba and standard layers) ----
            let has_dense_ffn = !layer.ffn_gate.is_empty()
                && !layer.ffn_up.is_empty()
                && !layer.ffn_down.is_empty()
                && !ffn_norm_weight.is_empty();
            let has_moe = cfg.num_experts > 0
                && !layer.ffn_gate_exps.is_empty()
                && !layer.ffn_up_exps.is_empty()
                && !layer.ffn_down_exps.is_empty()
                && !layer.ffn_gate_inp.is_empty()
                && !ffn_norm_weight.is_empty();

            if has_dense_ffn || has_moe {
                let ffn_out = &mut ws.hidden_a[..h];
                ffn_out.fill(0.0_f32);
                {
                    let normed = &mut ws.hidden_b[..h];
                    normed.fill(0.0_f32);
                    rms_norm_f32(&ws.x[..h], ffn_norm_weight, cfg.rms_norm_eps, normed)
                        .map_err(|e| ModelError::InferenceFailed(format!("ffn_norm: {:?}", e)))?;

                    if has_moe {
                        let moe_i = if cfg.expert_intermediate_size > 0 {
                            cfg.expert_intermediate_size
                        } else {
                            cfg.intermediate_size
                        };
                        let n_sel = cfg.num_experts_per_tok.max(1).min(cfg.num_experts);
                        let gate_scratch = &mut ws.moe_gate_all[..n_sel * moe_i];
                        let up_scratch = &mut ws.moe_up_all[..n_sel * moe_i];
                        let expert_out = &mut ws.moe_down_all[..n_sel * h];
                        Self::moe_ffn_forward_single(
                            layer,
                            cfg,
                            normed,
                            ffn_out,
                            gate_scratch,
                            up_scratch,
                            expert_out,
                            &mut ws.moe_router_logits[..cfg.num_experts],
                            &mut ws.moe_expert_scores[..cfg.num_experts],
                        )?;
                        if !layer.ffn_gate_shexp.is_empty()
                            && !layer.ffn_up_shexp.is_empty()
                            && !layer.ffn_down_shexp.is_empty()
                        {
                            let shexp_i = if cfg.expert_intermediate_size > 0 {
                                cfg.expert_intermediate_size
                            } else {
                                cfg.intermediate_size
                            };
                            let gate = &mut ws.intermediate_a[..shexp_i];
                            let up = &mut ws.intermediate_b[..shexp_i];
                            let shexp_out = &mut ws.intermediate_c[..h];
                            gate.fill(0.0_f32);
                            up.fill(0.0_f32);
                            shexp_out.fill(0.0_f32);
                            gemv_weight(&layer.ffn_gate_shexp, shexp_i, h, normed, gate).map_err(
                                |e| ModelError::InferenceFailed(format!("shexp gate: {:?}", e)),
                            )?;
                            gemv_weight(&layer.ffn_up_shexp, shexp_i, h, normed, up).map_err(
                                |e| ModelError::InferenceFailed(format!("shexp up: {:?}", e)),
                            )?;
                            apply_swiglu_inplace_f32(gate, up);
                            gemv_weight(&layer.ffn_down_shexp, h, shexp_i, gate, shexp_out)
                                .map_err(|e| {
                                    ModelError::InferenceFailed(format!("shexp down: {:?}", e))
                                })?;
                            for i in 0..h {
                                ffn_out[i] += shexp_out[i];
                            }
                        }
                    } else {
                        let gate = &mut ws.intermediate_a[..cfg.intermediate_size];
                        gate.fill(0.0_f32);
                        let up = &mut ws.intermediate_b[..cfg.intermediate_size];
                        up.fill(0.0_f32);
                        // Gate and up share the normed input; run both as ONE
                        // fused parallel region (two nested regions stole work
                        // from each other and halved streaming throughput).
                        gemv_weight_fused(
                            vec![
                                (&layer.ffn_gate, cfg.intermediate_size, &mut *gate),
                                (&layer.ffn_up, cfg.intermediate_size, &mut *up),
                            ],
                            h,
                            normed,
                        )
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("ffn_gate_up: {:?}", e))
                        })?;

                        // GeGLU for Gemma, otherwise SwiGLU (AVX2 fast path).
                        if cfg.gelu_ffn {
                            apply_geglu_inplace_f32(gate, up);
                        } else {
                            apply_swiglu_inplace_f32(gate, up);
                        }

                        gemv_weight(&layer.ffn_down, h, cfg.intermediate_size, gate, ffn_out)
                            .map_err(|e| {
                                ModelError::InferenceFailed(format!("ffn_down: {:?}", e))
                            })?;
                        if !layer.ffn_down_bias.is_empty() {
                            for (i, out) in ffn_out.iter_mut().enumerate() {
                                *out += layer.ffn_down_bias[i % layer.ffn_down_bias.len()];
                            }
                        }
                    }
                }

                // Gemma sandwich norm: normalize the FFN output before residual.
                if cfg.sandwich_norm && !layer.post_ffn_norm.is_empty() {
                    let normed_ffn = &mut ws.hidden_b[..h];
                    rms_norm_f32(ffn_out, &layer.post_ffn_norm, cfg.rms_norm_eps, normed_ffn)
                        .map_err(|e| {
                            ModelError::InferenceFailed(format!("post_ffn_norm: {:?}", e))
                        })?;
                    ffn_out.copy_from_slice(normed_ffn);
                }

                for i in 0..h {
                    ws.x[i] += ffn_out[i];
                }
            }
            if trace_fwd_enabled() {
                let sum: f64 = ws.x[..h].iter().map(|v| *v as f64).sum();
                eprintln!("TRACE inf pos={pos} layer={layer_idx} sum={sum:.9e}");
            }
        }
        Ok(())
    }

    fn gemv_weight_head(
        storage: &WeightStorage,
        rows: usize,
        cols: usize,
        head: usize,
        n_heads: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), String> {
        if n_heads == 0 {
            return Err("n_heads is zero".to_string());
        }
        match storage {
            WeightStorage::F32(data) => {
                let per_head = data.len() / n_heads;
                let start = head * per_head;
                let end = start + per_head;
                if end > data.len() {
                    return Err("head slice out of range".to_string());
                }
                gemv_f32(&data[start..end], rows, cols, input, output)
                    .map_err(|e| format!("{:?}", e))
            }
            WeightStorage::Quantized(qtype, data) => {
                let (block_width, block_size) = weight_block_info(*qtype);
                let blocks_per_row = cols / block_width;
                let per_head = rows * blocks_per_row * block_size;
                let start = head * per_head;
                let end = start + per_head;
                gemv_quantized_f32(*qtype, &data[start..end], rows, cols, input, output)
                    .map_err(|e| format!("{:?}", e))
            }
            WeightStorage::MmapQuantized(qtype, mmap, offset, size) => {
                let data = &mmap[*offset..*offset + *size];
                let (block_width, block_size) = weight_block_info(*qtype);
                let blocks_per_row = cols / block_width;
                let per_head = rows * blocks_per_row * block_size;
                let start = head * per_head;
                let end = start + per_head;
                gemv_quantized_f32(*qtype, &data[start..end], rows, cols, input, output)
                    .map_err(|e| format!("{:?}", e))
            }
        }
    }

    fn mla_v_b_head(
        storage: &WeightStorage,
        kv_lora: usize,
        v_dim: usize,
        head: usize,
        n_heads: usize,
        kv_cmpr: &[f32],
        out: &mut [f32],
    ) -> Result<(), String> {
        out.fill(0.0);
        if let WeightStorage::F32(data) = storage {
            for v in 0..v_dim {
                let mut sum = 0.0_f32;
                for l in 0..kv_lora {
                    let idx = l * v_dim * n_heads + v * n_heads + head;
                    if idx < data.len() {
                        sum += data[idx] * kv_cmpr[l];
                    }
                }
                out[v] = sum;
            }
            return Ok(());
        }
        let per_head_elems = kv_lora * v_dim;
        let mut w_host = vec![0.0_f32; per_head_elems];
        match storage {
            WeightStorage::Quantized(qtype, data) => {
                let per_head_bytes = data.len() / n_heads.max(1);
                let start = head * per_head_bytes;
                let end = (head + 1) * per_head_bytes;
                if end > data.len() {
                    return Err(format!(
                        "v_b head {head} out of range (bytes {end} > {})",
                        data.len()
                    ));
                }
                let mut deq = vec![0.0_f32; per_head_elems];
                dequantize_scalar(*qtype, &data[start..end], &mut deq)
                    .map_err(|e| format!("dequant v_b: {:?}", e))?;
                w_host.copy_from_slice(&deq);
            }
            WeightStorage::MmapQuantized(qtype, mmap, offset, size) => {
                let data = &mmap[*offset..*offset + *size];
                let per_head_bytes = data.len() / n_heads.max(1);
                let start = head * per_head_bytes;
                let end = (head + 1) * per_head_bytes;
                if end > data.len() {
                    return Err(format!(
                        "v_b head {head} out of range (bytes {end} > {})",
                        data.len()
                    ));
                }
                let mut deq = vec![0.0_f32; per_head_elems];
                dequantize_scalar(*qtype, &data[start..end], &mut deq)
                    .map_err(|e| format!("dequant v_b: {:?}", e))?;
                w_host.copy_from_slice(&deq);
            }
            WeightStorage::F32(_) => {}
        }
        for v in 0..v_dim {
            let mut sum = 0.0_f32;
            for l in 0..kv_lora {
                sum += w_host[l * v_dim + v] * kv_cmpr[l];
            }
            out[v] = sum;
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn deepseek_mla_layer(
        kv_cache: &mut KvCache,
        layer: &LayerWeights,
        cfg: &InferenceConfig,
        kv_layer_idx: usize,
        pos: usize,
        ws: &mut Workspace,
    ) -> Result<(), ModelError> {
        let h = cfg.hidden_size;
        let x = &ws.x[..h];
        let attn_out = &mut ws.hidden_a[..h];
        let n_heads = cfg.num_attention_heads;
        let kv_lora = layer.mla_kv_a_norm.len();
        let q_lora_dim = layer.mla_q_a.output_dim(h);
        let q_len = layer.mla_q_b.output_dim(q_lora_dim);
        let k_head_dim = cfg.kv_head_dim();
        let kv_pe_dim = layer.mla_kv_a_mqa.output_dim(h).saturating_sub(kv_lora);
        let k_nope_dim = layer.mla_k_b.output_dim(kv_lora) / n_heads.max(1);
        let v_head_dim = layer.mla_v_b.output_dim(kv_lora) / n_heads.max(1);
        let q_pe_dim = k_head_dim.saturating_sub(k_nope_dim);

        let normed = &mut ws.hidden_b[..h];
        normed.fill(0.0_f32);
        rms_norm_f32(x, &layer.attn_norm, cfg.rms_norm_eps, normed)
            .map_err(|e| ModelError::InferenceFailed(format!("mla attn_norm: {:?}", e)))?;

        let q_lora = layer.mla_q_a.output_dim(h);
        let c_q = &mut ws.intermediate_a[..q_lora];
        c_q.fill(0.0_f32);
        gemv_weight(&layer.mla_q_a, q_lora, h, normed, c_q)
            .map_err(|e| ModelError::InferenceFailed(format!("mla q_a: {:?}", e)))?;
        if !layer.mla_q_a_norm.is_empty() {
            let normed_q = &mut ws.intermediate_b[..q_lora];
            normed_q.copy_from_slice(c_q);
            rms_norm_f32(normed_q, &layer.mla_q_a_norm, cfg.rms_norm_eps, c_q)
                .map_err(|e| ModelError::InferenceFailed(format!("mla q_a_norm: {:?}", e)))?;
        }

        let q = &mut ws.q_full[..q_len];
        q.fill(0.0_f32);
        gemv_weight(&layer.mla_q_b, q_len, q_lora, c_q, q)
            .map_err(|e| ModelError::InferenceFailed(format!("mla q_b: {:?}", e)))?;

        let kv_out = layer.mla_kv_a_mqa.output_dim(h);
        let kv_pe = &mut ws.intermediate_c[..kv_out];
        kv_pe.fill(0.0_f32);
        gemv_weight(&layer.mla_kv_a_mqa, kv_out, h, normed, kv_pe)
            .map_err(|e| ModelError::InferenceFailed(format!("mla kv_a: {:?}", e)))?;

        let c_kv = &mut ws.mamba_scratch[..kv_lora];
        c_kv.copy_from_slice(&kv_pe[..kv_lora]);
        if !layer.mla_kv_a_norm.is_empty() {
            let c_kv_tmp = &mut ws.intermediate_b[..kv_lora];
            c_kv_tmp.copy_from_slice(c_kv);
            rms_norm_f32(c_kv_tmp, &layer.mla_kv_a_norm, cfg.rms_norm_eps, c_kv)
                .map_err(|e| ModelError::InferenceFailed(format!("mla kv_norm: {:?}", e)))?;
        }

        let k_pe_raw = &kv_pe[kv_lora..kv_lora + kv_pe_dim];
        let k_pe_rope = &mut ws.flash_q[..kv_pe_dim];
        apply_rope_f32(k_pe_raw, pos, kv_pe_dim, cfg.rope_theta, k_pe_rope)
            .map_err(|e| ModelError::InferenceFailed(format!("mla k_pe rope: {:?}", e)))?;

        let total_k = n_heads * k_head_dim;
        let total_v = n_heads * v_head_dim;
        let k_store = &mut ws.k_vec[..total_k];
        let v_store = &mut ws.v_vec[..total_v];
        k_store.fill(0.0_f32);
        v_store.fill(0.0_f32);

        for head in 0..n_heads {
            let k_off = head * k_head_dim;
            Self::gemv_weight_head(
                &layer.mla_k_b,
                k_nope_dim,
                kv_lora,
                head,
                n_heads,
                c_kv,
                &mut k_store[k_off..k_off + k_nope_dim],
            )
            .map_err(|e| ModelError::InferenceFailed(format!("mla k_b h{head}: {e}")))?;
            let rope_off = k_off + k_nope_dim;
            let copy = q_pe_dim
                .min(kv_pe_dim)
                .min(k_head_dim.saturating_sub(k_nope_dim));
            k_store[rope_off..rope_off + copy].copy_from_slice(&k_pe_rope[..copy]);

            let v_off = head * v_head_dim;
            Self::mla_v_b_head(
                &layer.mla_v_b,
                kv_lora,
                v_head_dim,
                head,
                n_heads,
                c_kv,
                &mut v_store[v_off..v_off + v_head_dim],
            )
            .map_err(|e| ModelError::InferenceFailed(format!("mla v_b h{head}: {e}")))?;

            let q_off = head * k_head_dim;
            if q_off + k_head_dim <= q.len() && q_pe_dim > 0 {
                let q_pe = &mut q[q_off + k_nope_dim..q_off + k_head_dim];
                let rotated = &mut ws.head_scratch[..q_pe_dim];
                rotated.fill(0.0_f32);
                apply_rope_f32(q_pe, pos, q_pe_dim, cfg.rope_theta, rotated)
                    .map_err(|e| ModelError::InferenceFailed(format!("mla q_pe: {:?}", e)))?;
                q_pe.copy_from_slice(&rotated[..q_pe.len()]);
            }
        }

        let mut v_padded = vec![0.0_f32; total_k];
        for head in 0..n_heads {
            let v_off = head * v_head_dim;
            let k_off = head * k_head_dim;
            v_padded[k_off..k_off + v_head_dim]
                .copy_from_slice(&v_store[v_off..v_off + v_head_dim]);
        }
        kv_cache
            .set(kv_layer_idx, pos, k_store, &v_padded)
            .map_err(|e| ModelError::InferenceFailed(format!("mla kv set: {:?}", e)))?;

        let seq_len = pos + 1;
        let attn_result = &mut ws.attn_result[..total_k];
        attn_result.fill(0.0_f32);
        let scale = 1.0_f32 / (k_head_dim as f32).sqrt();

        for head in 0..n_heads {
            let q_off = head * k_head_dim;
            let k_off = head * k_head_dim;
            let _v_off = head * v_head_dim;
            let q_h = &q[q_off..q_off + k_head_dim];
            let mut scores = vec![0.0_f32; seq_len];
            for t in 0..seq_len {
                let mut k_t = vec![0.0_f32; total_k];
                kv_cache
                    .get_key(kv_layer_idx, t, &mut k_t)
                    .map_err(|e| ModelError::InferenceFailed(format!("mla get_k: {:?}", e)))?;
                let mut dot = 0.0_f32;
                for i in 0..k_head_dim {
                    dot += q_h[i] * k_t[q_off + i];
                }
                scores[t] = dot * scale;
            }
            let max_s = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum = 0.0_f32;
            for s in &mut scores {
                *s = (*s - max_s).exp();
                sum += *s;
            }
            if sum > 0.0 {
                for s in &mut scores {
                    *s /= sum;
                }
            }
            for i in 0..v_head_dim {
                let mut acc = 0.0_f32;
                for t in 0..seq_len {
                    let mut v_t = vec![0.0_f32; total_k];
                    kv_cache
                        .get_value(kv_layer_idx, t, &mut v_t)
                        .map_err(|e| ModelError::InferenceFailed(format!("mla get_v: {:?}", e)))?;
                    acc += scores[t] * v_t[k_off + i];
                }
                attn_result[q_off + i] = acc;
            }
        }

        let attn_in_len = layer.attn_output.output_dim(h);
        let attn_input = if attn_in_len > 0 && attn_result.len() >= attn_in_len {
            &attn_result[..attn_in_len]
        } else {
            &attn_result[..total_k.min(attn_result.len())]
        };
        gemv_weight(
            &layer.attn_output,
            h,
            attn_input.len(),
            attn_input,
            attn_out,
        )
        .map_err(|e| ModelError::InferenceFailed(format!("mla attn_out: {:?}", e)))?;
        Ok(())
    }

    /// MoE FFN forward for a single token.
    /// 1. Compute router logits: normed @ ffn_gate_inp
    /// 2. Softmax + top-k expert selection
    /// 3. For each selected expert, compute gate/up/down
    /// 4. Weighted sum of expert outputs
    fn moe_ffn_forward_single(
        layer: &LayerWeights,
        cfg: &InferenceConfig,
        normed: &[f32],
        ffn_out: &mut [f32],
        gate_scratch: &mut [f32],
        up_scratch: &mut [f32],
        expert_out: &mut [f32],
        router_logits: &mut [f32],
        expert_scores: &mut [(usize, f32)],
    ) -> Result<(), ModelError> {
        moe_ffn_forward_weights(
            &MoeFfnWeights::from_layer(layer),
            cfg,
            normed,
            ffn_out,
            gate_scratch,
            up_scratch,
            expert_out,
            router_logits,
            expert_scores,
        )
    }
}

/// MoE FFN weight bundle shared by inference and layer-wise runtimes.
pub(crate) struct MoeFfnWeights<'a> {
    pub gate_inp: &'a WeightStorage,
    pub gate_exps: &'a WeightStorage,
    pub up_exps: &'a WeightStorage,
    pub down_exps: &'a WeightStorage,
    pub exp_probs_b: &'a [f32],
}

impl<'a> MoeFfnWeights<'a> {
    pub(crate) fn from_layer(layer: &'a LayerWeights) -> Self {
        Self {
            gate_inp: &layer.ffn_gate_inp,
            gate_exps: &layer.ffn_gate_exps,
            up_exps: &layer.ffn_up_exps,
            down_exps: &layer.ffn_down_exps,
            exp_probs_b: &layer.ffn_exp_probs_b,
        }
    }
}

pub(crate) fn moe_ffn_forward_weights(
    layer: &MoeFfnWeights<'_>,
    cfg: &InferenceConfig,
    normed: &[f32],
    ffn_out: &mut [f32],
    gate_scratch: &mut [f32],
    up_scratch: &mut [f32],
    expert_out: &mut [f32],
    router_logits: &mut [f32],
    expert_scores: &mut [(usize, f32)],
) -> Result<(), ModelError> {
    let h = cfg.hidden_size;
    // Experts may use a narrower intermediate width than the dense FFN
    // (LFM2MoE: 1792 vs 7168). Fall back to intermediate_size otherwise.
    let i_size = if cfg.expert_intermediate_size > 0 {
        cfg.expert_intermediate_size
    } else {
        cfg.intermediate_size
    };
    let n_experts = cfg.num_experts;
    let n_experts_per_tok = cfg.num_experts_per_tok.max(1).min(n_experts);
    let sigmoid_gating = cfg.expert_gating_sigmoid;

    // 1. Router logits: [n_experts]
    router_logits.fill(0.0_f32);
    gemv_weight(layer.gate_inp, n_experts, h, normed, router_logits)
        .map_err(|e| ModelError::InferenceFailed(format!("moe router: {:?}", e)))?;

    // 2. Gating. Softmax (Mixtral) or sigmoid + per-layer expert bias (LFM2MoE).
    // For sigmoid gating the bias is added for top-k *selection* only; the
    // routing weights are the raw sigmoid scores, renormalized over the
    // selected experts. `router_logits` holds the weight, `expert_scores.1`
    // the selection score.
    if sigmoid_gating {
        for logit in router_logits.iter_mut() {
            *logit = 1.0_f32 / (1.0 + (-*logit).exp());
        }
        for (i, &w) in router_logits.iter().enumerate() {
            let bias = layer.exp_probs_b.get(i).copied().unwrap_or(0.0);
            expert_scores[i] = (i, w + bias);
        }
    } else {
        let max_logit = router_logits
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum_exp = 0.0_f32;
        for logit in router_logits.iter_mut() {
            *logit = (*logit - max_logit).exp();
            sum_exp += *logit;
        }
        if sum_exp > 0.0 {
            for logit in router_logits.iter_mut() {
                *logit /= sum_exp;
            }
        }
        for (i, &w) in router_logits.iter().enumerate() {
            expert_scores[i] = (i, w);
        }
    }

    // 3. Top-k expert selection by selection score.
    let compare_score = |a: &(usize, f32), b: &(usize, f32)| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    };
    if n_experts_per_tok < expert_scores.len() {
        let (selected, _, _) =
            expert_scores.select_nth_unstable_by(n_experts_per_tok, compare_score);
        selected.sort_by(compare_score);
    } else {
        expert_scores.sort_by(compare_score);
    }

    // Renormalize routing weights over the selected experts (Qwen/Mixtral norm_topk_prob).
    let weight_norm = {
        let s: f32 = expert_scores
            .iter()
            .take(n_experts_per_tok)
            .map(|&(idx, _)| router_logits[idx])
            .sum();
        if s > 0.0 { s } else { 1.0 }
    };

    // 4. Gather the selected experts and their routing weights.
    let n_sel = n_experts_per_tok;
    let mut selected: Vec<usize> = Vec::with_capacity(n_sel);
    let mut weights: Vec<f32> = Vec::with_capacity(n_sel);
    for &(expert_idx, _sel_score) in expert_scores.iter().take(n_sel) {
        selected.push(expert_idx);
        weights.push(router_logits[expert_idx] / weight_norm);
    }

    // 5. Expert FFN. Prefer the batched path (one parallel region per
    // projection across all selected experts) for quantized experts; this
    // avoids 12 separate rayon dispatches per MoE layer. Fall back to the
    // per-expert path for f32 experts.
    if let (Some((gq, gm)), Some((uq, um)), Some((dq, dm))) = (
        expert_matrix(layer.gate_exps),
        expert_matrix(layer.up_exps),
        expert_matrix(layer.down_exps),
    ) {
        let gate_all = &mut gate_scratch[..n_sel * i_size];
        let up_all = &mut up_scratch[..n_sel * i_size];
        if gq == uq {
            // Fused: gate + up in ONE parallel region (halves the
            // fork/join + steal overhead of the two largest dispatches).
            // The kernel needs gate|up laid out contiguously to dispatch both
            // projections as a single pool region, so we cannot write directly
            // into the two separate scratch buffers. Use a thread-local buffer
            // (decode forward runs on the single submitter thread) rather than
            // a per-layer-per-token heap alloc + two memcpys back into
            // gate_all/up_all — that copy was ~14% of main-thread decode time.
            // The kernel writes every output element, so no zero-fill is
            // needed; the SwiGLU and down-projection read the two halves in
            // place, leaving gate_all/up_all unused on this path.
            thread_local! {
                static GATE_UP: std::cell::RefCell<Vec<f32>> =
                    const { std::cell::RefCell::new(Vec::new()) };
            }
            let _ = (&gate_all, &up_all);
            return GATE_UP.with_borrow_mut(|gate_up| {
                gate_up.resize(2 * n_sel * i_size, 0.0_f32);
                gemv_quantized_experts_gate_up_f32(
                    gq, gm, um, n_experts, &selected, i_size, h, normed, gate_up,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("moe gate+up: {:?}", e)))?;
                let (gate_half, up_half) = gate_up.split_at_mut(n_sel * i_size);
                // SwiGLU into gate_half; it becomes the down-projection input
                // (contiguous [n_sel, i_size], stride i_size per expert).
                for (g, u) in gate_half.iter_mut().zip(up_half.iter()) {
                    let sigmoid = 1.0_f32 / (1.0 + (-*g).exp());
                    *g = *g * sigmoid * *u;
                }
                let down_all = &mut expert_out[..n_sel * h];
                gemv_quantized_experts_f32(
                    dq, dm, n_experts, &selected, h, i_size, gate_half, i_size, down_all,
                )
                .map_err(|e| ModelError::InferenceFailed(format!("moe down: {:?}", e)))?;
                for (slot, &weight) in weights.iter().enumerate() {
                    let d = &down_all[slot * h..(slot + 1) * h];
                    for (out, val) in ffn_out.iter_mut().zip(d.iter()) {
                        *out += weight * val;
                    }
                }
                Ok(())
            });
        }
        // Non-fused path actually consumes gate_all/up_all — zero them here
        // (the fused branch above returns early without touching them, so the
        // previous unconditional fill was wasted decode-hot-path traffic).
        gate_all.fill(0.0_f32);
        up_all.fill(0.0_f32);
        gemv_quantized_experts_f32(gq, gm, n_experts, &selected, i_size, h, normed, 0, gate_all)
            .map_err(|e| ModelError::InferenceFailed(format!("moe gate: {:?}", e)))?;
        gemv_quantized_experts_f32(uq, um, n_experts, &selected, i_size, h, normed, 0, up_all)
            .map_err(|e| ModelError::InferenceFailed(format!("moe up: {:?}", e)))?;
        // SwiGLU into gate_all; it then becomes the down-projection input
        // (one contiguous [n_sel, i_size] buffer, stride i_size per expert).
        for (g, u) in gate_all.iter_mut().zip(up_all.iter()) {
            let sigmoid = 1.0_f32 / (1.0 + (-*g).exp());
            *g = *g * sigmoid * *u;
        }
        let down_all = &mut expert_out[..n_sel * h];
        down_all.fill(0.0_f32);
        gemv_quantized_experts_f32(
            dq, dm, n_experts, &selected, h, i_size, gate_all, i_size, down_all,
        )
        .map_err(|e| ModelError::InferenceFailed(format!("moe down: {:?}", e)))?;
        for (slot, &weight) in weights.iter().enumerate() {
            let d = &down_all[slot * h..(slot + 1) * h];
            for (out, val) in ffn_out.iter_mut().zip(d.iter()) {
                *out += weight * val;
            }
        }
        return Ok(());
    }

    // Fallback: per-expert FFN for f32 expert weights.
    for (slot, &expert_idx) in selected.iter().enumerate() {
        let weight = weights[slot];
        let gate = &mut gate_scratch[..i_size];
        let up = &mut up_scratch[..i_size];
        gate.fill(0.0_f32);
        up.fill(0.0_f32);
        expert_out.fill(0.0_f32);

        gemv_expert_weight(
            layer.gate_exps,
            expert_idx,
            n_experts,
            i_size,
            h,
            normed,
            gate,
        )
        .map_err(|e| ModelError::InferenceFailed(format!("moe gate: {:?}", e)))?;
        gemv_expert_weight(layer.up_exps, expert_idx, n_experts, i_size, h, normed, up)
            .map_err(|e| ModelError::InferenceFailed(format!("moe up: {:?}", e)))?;

        for (g, u) in gate.iter_mut().zip(up.iter()) {
            let sigmoid = 1.0_f32 / (1.0 + (-*g).exp());
            *g = *g * sigmoid * *u;
        }

        gemv_expert_weight(
            layer.down_exps,
            expert_idx,
            n_experts,
            h,
            i_size,
            gate,
            expert_out,
        )
        .map_err(|e| ModelError::InferenceFailed(format!("moe down: {:?}", e)))?;

        for (out, val) in ffn_out.iter_mut().zip(expert_out.iter()) {
            *out += weight * val;
        }
    }

    Ok(())
}

impl Model for InferenceModel {
    fn rewind_to(&mut self, consumed_tokens: usize) -> Result<(), ModelError> {
        let position = if consumed_tokens == 0 {
            0
        } else {
            consumed_tokens.saturating_sub(1)
        };
        self.kv_cache
            .rewind_to(position)
            .map_err(|e| ModelError::InferenceFailed(format!("{e:?}")))
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
        let mut logits = Vec::with_capacity(tokens.len());
        for (i, &token) in tokens.iter().enumerate() {
            let pos = start_pos + i;
            if let Some(step_logits) = self.forward_single(token, pos, true)? {
                logits.push(step_logits);
            }
        }
        session.record_tokens(tokens.len());
        Ok(logits)
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
        let logits = if tokens.len() > 1 && self.layers_supported_for_batched() {
            // Prefill the prompt in one batched pass so every weight matmul is a
            // GEMM (decode-once per weight block) rather than `tokens.len()`
            // separate GEMVs. Intermediate logits are discarded so only the
            // last token's lm_head is computed.
            //
            // Batched prefill is now enabled for all KV cache dtypes (not just
            // F32). The KV cache writes in forward_batched use kv_cache.set()
            // which already handles all quant types correctly.
            self.forward_batched(tokens, start_pos, true)?
                .unwrap_or_default()
        } else {
            let mut logits = Vec::new();
            for (i, &token) in tokens.iter().enumerate() {
                let pos = start_pos + i;
                let need_logits = i + 1 == tokens.len();
                if let Some(final_logits) = self.forward_single(token, pos, need_logits)? {
                    logits = final_logits;
                }
            }
            logits
        };
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GgufFile, GgufMetadataValue, GgufTensorInfo, MappedGgufFile};
    use std::collections::BTreeMap;

    #[test]
    fn qwen35_mtp_metadata_subtracts_nextn_layers() {
        let mapped = MappedGgufFile::from_parsed_for_test(GgufFile {
            version: 3,
            tensor_count: 1,
            metadata: BTreeMap::from([
                (
                    "general.architecture".to_owned(),
                    GgufMetadataValue::String("qwen35".to_owned()),
                ),
                (
                    "qwen35.block_count".to_owned(),
                    GgufMetadataValue::Uint32(65),
                ),
                (
                    "qwen35.nextn_predict_layers".to_owned(),
                    GgufMetadataValue::Uint32(1),
                ),
                (
                    "qwen35.embedding_length".to_owned(),
                    GgufMetadataValue::Uint32(5120),
                ),
                (
                    "qwen35.feed_forward_length".to_owned(),
                    GgufMetadataValue::Uint32(17408),
                ),
                (
                    "qwen35.attention.head_count".to_owned(),
                    GgufMetadataValue::Uint32(24),
                ),
                (
                    "qwen35.attention.head_count_kv".to_owned(),
                    GgufMetadataValue::Uint32(4),
                ),
                (
                    "qwen35.attention.key_length".to_owned(),
                    GgufMetadataValue::Uint32(256),
                ),
            ]),
            tensor_infos: vec![GgufTensorInfo {
                name: "tok_embeddings.weight".to_owned(),
                dimensions: vec![5120, 248320],
                ggml_type: 0,
                relative_offset: 0,
                absolute_offset: 0,
            }],
            alignment: 32,
            data_section_start: 0,
        });

        let cfg = InferenceConfig::from_gguf(&mapped);

        assert_eq!(cfg.architecture, ModelArchitecture::Qwen);
        assert_eq!(cfg.layer_count, 64);
        assert_eq!(cfg.nextn_predict_layers, 1);
        assert_eq!(cfg.hidden_size, 5120);
        assert_eq!(cfg.kv_head_dim(), 256);
        assert_eq!(cfg.rope_dim, 64);
    }

    #[test]
    fn gemma_sliding_window_pattern_selects_global_layers() {
        // Gemma 3/4: every 6th layer (1-indexed) is global, the rest local SWA.
        let cfg = InferenceConfig {
            architecture: ModelArchitecture::Gemma,
            sliding_window: 1024,
            sliding_window_pattern: 6,
            rope_theta: 1_000_000.0,
            rope_theta_swa: 10_000.0,
            ..Default::default()
        };
        // Layers 0..4 are local (SWA), layer 5 (1-indexed 6th) is global.
        for l in 0..5 {
            assert!(!cfg.layer_is_global(l), "layer {l} should be local");
            assert_eq!(cfg.layer_rope_theta(l), 10_000.0);
            assert_eq!(cfg.layer_sliding_window(l), 1024);
        }
        assert!(cfg.layer_is_global(5), "layer 5 should be global");
        assert_eq!(cfg.layer_rope_theta(5), 1_000_000.0);
        assert_eq!(cfg.layer_sliding_window(5), 0);
        assert!(cfg.layer_is_global(11));
    }

    #[test]
    fn non_gemma_layers_are_all_global() {
        let cfg = InferenceConfig {
            sliding_window_pattern: 0,
            sliding_window: 0,
            rope_theta: 10_000.0,
            ..Default::default()
        };
        assert!(cfg.layer_is_global(0));
        assert!(cfg.layer_is_global(7));
        assert_eq!(cfg.layer_rope_theta(3), 10_000.0);
        assert_eq!(cfg.layer_sliding_window(3), 0);
    }

    #[test]
    fn uniform_swa_models_apply_sliding_window_to_every_layer() {
        // Mistral/Qwen: sliding_window set, pattern 0 -> SWA on every layer.
        let cfg = InferenceConfig {
            sliding_window: 4096,
            sliding_window_pattern: 0,
            rope_theta: 1_000_000.0,
            ..Default::default()
        };
        for l in [0usize, 1, 5, 31] {
            assert!(!cfg.layer_is_global(l), "layer {l} should be local SWA");
            assert_eq!(cfg.layer_sliding_window(l), 4096);
        }
    }

    fn tiny_inference_model() -> InferenceModel {
        let config = InferenceConfig {
            vocab_size: 3,
            context_size: 8,
            layer_count: 0,
            hidden_size: 2,
            intermediate_size: 0,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            key_value_head_dim: 2,
            kv_cache_dtype: DType::F32,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            ..Default::default()
        };
        let kv_cache_config = KvCacheConfig {
            layer_count: config.layer_count,
            context_size: config.context_size,
            head_count: config.num_key_value_heads,
            head_dim: config.kv_head_dim(),
            dtype: DType::F32,
            quantization: Default::default(),
        };

        InferenceModel {
            config: config.clone(),
            tok_embeddings: WeightStorage::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            tok_embeddings_cols: 3,
            norm_weight: vec![1.0, 1.0],
            output_weight: WeightStorage::F32(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            layers: Vec::new(),
            mtp: None,
            kv_cache: KvCache::new(kv_cache_config).expect("tiny kv cache should be valid"),
            kv_layer_map: Vec::new(),
            ssm_states: Vec::new(),
            ssm_conv_buffers: Vec::new(),
            workspace: Workspace::for_config(&config),
            last_output_hidden: vec![0.0_f32; config.hidden_size],
        }
    }

    #[test]
    fn batched_prefill_rejects_moe_layers() {
        let mut model = tiny_inference_model();
        let layer = LayerWeights {
            attn_q: WeightStorage::F32(vec![1.0]),
            ffn_gate_exps: WeightStorage::F32(vec![1.0]),
            ..Default::default()
        };
        model.layers.push(layer);

        assert!(!model.layers_supported_for_batched());
    }

    #[test]
    fn workspace_buffer_capacities_cover_model_dimensions() {
        let config = InferenceConfig {
            vocab_size: 100,
            context_size: 16,
            layer_count: 2,
            hidden_size: 32,
            intermediate_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            key_value_head_dim: 16,
            kv_cache_dtype: DType::F32,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            ..Default::default()
        };
        let ws = Workspace::for_config(&config);

        assert_eq!(ws.x.len(), config.hidden_size);
        assert_eq!(ws.hidden_a.len(), config.hidden_size);
        assert_eq!(ws.hidden_b.len(), config.hidden_size);
        assert_eq!(ws.intermediate_a.len(), config.intermediate_size);
        assert_eq!(ws.intermediate_b.len(), config.intermediate_size);
        assert_eq!(ws.intermediate_c.len(), config.intermediate_size);

        let max_qkv = (config.hidden_size * 3).max(config.intermediate_size);
        assert_eq!(ws.q_full.len(), max_qkv);

        let max_kv_len = config.num_key_value_heads * config.kv_head_dim();
        assert_eq!(ws.k_vec.len(), max_kv_len);
        assert_eq!(ws.v_vec.len(), max_kv_len);
        assert_eq!(ws.attn_result.len(), max_qkv);

        let head_dim = config.head_dim().max(config.kv_head_dim()).max(192);
        assert_eq!(ws.head_scratch.len(), head_dim);

        let kv_copy_size = config.context_size * max_kv_len;
        assert_eq!(ws.kv_keys_copy.len(), kv_copy_size);
        assert_eq!(ws.kv_values_copy.len(), kv_copy_size);
        assert_eq!(ws.logits.len(), config.vocab_size);
        assert_eq!(ws.mamba_scratch.len(), config.hidden_size.max(576));
        assert_eq!(ws.conv_out.len(), max_qkv);
    }

    #[test]
    fn forward_single_can_skip_final_logits() {
        let mut model = tiny_inference_model();

        let skipped_logits = model
            .forward_single(0, 0, false)
            .expect("non-final token should run successfully");
        assert_eq!(skipped_logits, None);

        let final_logits = model
            .forward_single(1, 1, true)
            .expect("final token should run successfully")
            .expect("final token should produce logits");
        assert_eq!(final_logits.len(), model.vocab_size());
    }

    #[test]
    fn forward_returns_only_final_token_logits() {
        let mut prefill_model = tiny_inference_model();
        let mut prefill_session = Session::new();
        let prefill_logits = prefill_model
            .forward(&[0, 1], &mut prefill_session)
            .expect("prefill should produce final logits");

        let mut single_model = tiny_inference_model();
        let mut single_session = Session::new();
        let single_logits = single_model
            .forward(&[1], &mut single_session)
            .expect("single final token should produce logits");

        assert_eq!(prefill_logits, single_logits);
        assert_eq!(prefill_session.consumed_tokens(), 2);
        assert_eq!(single_session.consumed_tokens(), 1);
    }

    #[test]
    fn native_mtp_draft_runs_on_tiny_weights() {
        let mut model = tiny_inference_model();
        model.config.nextn_predict_layers = 1;
        model.config.intermediate_size = 2;
        let mut layer = LayerWeights {
            attn_norm: vec![1.0, 1.0],
            attn_q: WeightStorage::F32(vec![0.0; 4 * 2]),
            attn_k: WeightStorage::F32(vec![0.0; 2 * 2]),
            attn_v: WeightStorage::F32(vec![0.0; 2 * 2]),
            attn_output: WeightStorage::F32(vec![0.0; 2 * 2]),
            post_attention_norm: vec![1.0, 1.0],
            ffn_gate: WeightStorage::F32(vec![0.0; 2 * 2]),
            ffn_up: WeightStorage::F32(vec![0.0; 2 * 2]),
            ffn_down: WeightStorage::F32(vec![0.0; 2 * 2]),
            ..LayerWeights::default()
        };
        // Keep the MTP layer full-attention and dense; q output is [q; gate].
        layer.attn_q_bias = vec![0.0; 4];
        model.mtp = Some(MtpWeights {
            layer,
            eh_proj: WeightStorage::F32(vec![0.0; 2 * 4]),
            enorm: vec![1.0, 1.0],
            hnorm: vec![1.0, 1.0],
            shared_head_norm: vec![1.0, 1.0],
            ..MtpWeights::default()
        });

        let mut random = || 0.0_f32;
        let (tokens, logits) = model
            .draft_mtp_tokens(
                0,
                &[0.0, 0.0],
                2,
                crate::sampling::SamplingConfig {
                    temperature: 0.0,
                    top_k: Some(1),
                    ..Default::default()
                },
                &mut random,
            )
            .expect("tiny MTP draft should run");

        assert_eq!(tokens, vec![2, 2]);
        assert_eq!(logits.len(), 2);
        assert!(logits.iter().all(|step| step.len() == model.vocab_size()));
    }

    /// Whole-model forward(0..L) must equal split forward(0..K) + forward(K..L)
    /// on the same hidden state across many sequential positions. Detects bugs
    /// in run_layer_range_in_workspace that only show up with longer prompts.
    #[test]
    fn split_layer_range_matches_full_forward_long_sequence() {
        let mut whole = tiny_inference_model();
        let mut split = tiny_inference_model();
        let l = whole.config.layer_count;
        let k = l / 2;
        let positions = 20_usize; // long enough to exercise KV cache
        for pos in 0..positions {
            let tok = (pos % whole.config.vocab_size) as u32;
            let full = whole
                .forward_single(tok, pos, true)
                .expect("full forward ok")
                .expect("logits");
            // Split path: embed, run head layers, snapshot hidden,
            // set hidden, run tail layers, final head.
            split.embed_token_into_workspace(tok);
            split
                .run_layer_range_in_workspace(pos, 0..k)
                .expect("head layers ok");
            let mid_hidden = split.hidden_state().to_vec();
            split.set_hidden_state(&mid_hidden).expect("set hidden ok");
            split
                .run_layer_range_in_workspace(pos, k..l)
                .expect("tail layers ok");
            let split_logits = split.final_head_from_workspace().expect("final head ok");
            assert_eq!(full.len(), split_logits.len());
            for (i, (a, b)) in full.iter().zip(split_logits.iter()).enumerate() {
                assert!((a - b).abs() < 1e-4, "pos={pos} idx={i} full={a} split={b}");
            }
        }
    }
}
