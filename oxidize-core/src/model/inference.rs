#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

pub(super) use crate::flash_attention::{
    flash_attention_decode_heads_f16, flash_attention_decode_heads_f32,
};
pub(super) use crate::gguf::{GgufQuantizationType, MappedGgufFile};
pub(super) use crate::kv_cache::{KvCache, KvCacheConfig};
pub(super) use crate::model::{Logits, Model, ModelError, Session, Token};
pub(super) use crate::quantization::{dequantize_scalar, quant_block_layout, quantized_size};
pub(super) use crate::tensor::{
    DType, GemvJob, apply_geglu_inplace_f32, apply_rope_f32_yarn, apply_swiglu_inplace_f32,
    f16_le_to_f32, gemm_quantized_f32, gemv_f32, gemv_quantized_experts_f32,
    gemv_quantized_experts_gate_up_f32, gemv_quantized_f32, gemv_quantized_multi_f32, rms_norm_f32,
};
pub(super) use memmap2::Mmap;
pub(super) use std::sync::Arc;

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
    /// Z.ai GLM MoE with DSA sparse attention (GLM-5.x family).
    GlmMoeDsa,
    /// Tencent Hunyuan (hy_v3): GQA + qk_norm attention, sigmoid-routed MoE
    /// with a shared expert and leading dense blocks. Standard attention (no MLA).
    HunyuanMoe,
}

impl ModelArchitecture {
    /// Detect architecture from GGUF metadata.
    pub fn from_gguf(mapped: &MappedGgufFile) -> Self {
        let parsed = mapped.parsed();
        if let Some(arch) = parsed.architecture() {
            let arch = arch.replace('-', "_");
            match arch.as_str() {
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
                "glm" | "glm4" | "glm_moe" | "glm_moe_dsa" | "glm_dsa" | "glmmoe" | "glmmoedsa" => {
                    Self::GlmMoeDsa
                }
                "hunyuan" | "hunyuan_moe" | "hunyuanmoe" | "hy_v3" | "hyv3" | "hunyuan_v3" => {
                    Self::HunyuanMoe
                }
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
        matches!(
            self,
            Self::Mixtral
                | Self::MiniMax
                | Self::Lfm2Moe
                | Self::DeepSeek
                | Self::GlmMoeDsa
                | Self::HunyuanMoe
        )
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
        matches!(self, Self::DeepSeek | Self::GlmMoeDsa)
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
    /// DeepSeek-V3/Kimi routed-expert output scale (HF `routed_scaling_factor`,
    /// llama.cpp `expert_weights_scale`). The routed experts' weighted sum is
    /// multiplied by this before the shared-expert/residual add. 1.0 = none.
    /// Kimi-K2 uses ~2.827; without it the routed branch is far too weak.
    pub expert_weights_scale: f32,
    /// DeepSeek-V3 group-limited routing: number of expert groups (`n_group`).
    /// 0 or 1 = no group routing (plain global top-k). Kimi-K2 = 1.
    pub expert_group_count: usize,
    /// DeepSeek-V3 group-limited routing: groups kept per token (`topk_group`).
    /// Only consulted when `expert_group_count > 1`.
    pub expert_group_used_count: usize,
    /// YaRN rope extension factor (0 = disabled). GGUF `rope.scaling.factor`.
    pub yarn_factor: f32,
    /// Training context length before YaRN extension. GGUF
    /// `rope.scaling.original_context_length`.
    pub yarn_orig_ctx: f32,
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
            expert_weights_scale: 1.0,
            expert_group_count: 0,
            expert_group_used_count: 0,
            yarn_factor: 0.0,
            yarn_orig_ctx: 0.0,
        }
    }
}

impl InferenceConfig {
    #[inline]
    pub fn apply_rope_head(
        &self,
        input: &[f32],
        position: usize,
        head_dim: usize,
        theta: f32,
        output: &mut [f32],
    ) -> Result<(), crate::tensor::RopeError> {
        apply_rope_f32_yarn(
            input,
            position,
            head_dim,
            theta,
            output,
            self.yarn_factor,
            self.yarn_orig_ctx,
        )
    }

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
            // Unsloth GLM-5.2 GGUF uses hyphenated metadata keys (`glm-dsa.*`).
            "glm_dsa" => "glm-dsa",
            other => other,
        }
    }

    /// Canonical architecture string used for downstream behavior (RMSNorm
    /// `(1+w)`, GDN detection, RoPE overrides, etc.). Qwen3.5 variants are
    /// normalized to `qwen35` so the rest of the loader sees one family.
    fn canonical_architecture(arch: &str) -> &str {
        match arch {
            "qwen3_5_moe_text" | "qwen3_5_moe" | "qwen35moe" | "qwen3_5" | "qwen3_5_text"
            | "qwen35_text" => "qwen35",
            "glm_dsa" => "glm-dsa",
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
        let arch = Self::canonical_architecture(&raw_arch).to_string();
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
        let arch_str = |suffix: &str| {
            metadata_str_lookup(metadata, &key(suffix)).or_else(|| {
                if metadata_prefix == arch {
                    None
                } else {
                    metadata_str_lookup(metadata, &format!("{arch}.{suffix}"))
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
        // expert_gating_func: 1 = softmax, 2 = sigmoid (lfm2moe/deepseek2 use sigmoid).
        let expert_gating_sigmoid = arch_u32("expert_gating_func")
            .or_else(|| metadata_u32_lookup(metadata, "expert_gating_func"))
            .map(|v| v == 2)
            .unwrap_or(false);
        // DeepSeek-V3/Kimi routed-expert scaling (`routed_scaling_factor`) and
        // group-limited routing (`n_group` / `topk_group`). Absent for other
        // MoE archs, so they default to 1.0 / no-group and behave unchanged.
        let expert_weights_scale = arch_f32("expert_weights_scale")
            .or_else(|| metadata_f32_lookup(metadata, "expert_weights_scale"))
            .filter(|&v| v > 0.0)
            .unwrap_or(1.0);
        let expert_group_count = arch_u32("expert_group_count")
            .or_else(|| metadata_u32_lookup(metadata, "expert_group_count"))
            .map(|v| v as usize)
            .unwrap_or(0);
        let expert_group_used_count = arch_u32("expert_group_used_count")
            .or_else(|| metadata_u32_lookup(metadata, "expert_group_used_count"))
            .map(|v| v as usize)
            .unwrap_or(0);

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

        let (yarn_factor, yarn_orig_ctx) =
            if arch_str("rope.scaling.type").as_deref() == Some("yarn") {
                let factor = arch_f32("rope.scaling.factor").unwrap_or(0.0);
                let orig = arch_u32("rope.scaling.original_context_length")
                    .map(|v| v as f32)
                    .unwrap_or(0.0);
                (factor, orig)
            } else {
                (0.0, 0.0)
            };

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
            expert_weights_scale,
            expert_group_count,
            expert_group_used_count,
            yarn_factor,
            yarn_orig_ctx,
        }
    }
}

pub(super) fn metadata_u32_lookup(
    metadata: &std::collections::BTreeMap<String, crate::gguf::GgufMetadataValue>,
    key: &str,
) -> Option<u32> {
    metadata.get(key).and_then(|v| v.as_u32())
}

/// Largest integer value in an array-typed metadata field. Used for LFM2's
/// per-layer `attention.head_count_kv` (0 on shortconv layers, KV count on
/// attention layers) — the max gives the actual attention KV-head count.
pub(super) fn metadata_u32_array_max(
    metadata: &std::collections::BTreeMap<String, crate::gguf::GgufMetadataValue>,
    key: &str,
) -> Option<u32> {
    let arr = match metadata.get(key) {
        Some(crate::gguf::GgufMetadataValue::Array(a)) => a,
        _ => return None,
    };
    arr.values.iter().filter_map(|v| v.as_u32()).max()
}

pub(super) fn metadata_f32_lookup(
    metadata: &std::collections::BTreeMap<String, crate::gguf::GgufMetadataValue>,
    key: &str,
) -> Option<f32> {
    metadata.get(key).and_then(|v| v.as_f32())
}

pub(super) fn metadata_str_lookup(
    metadata: &std::collections::BTreeMap<String, crate::gguf::GgufMetadataValue>,
    key: &str,
) -> Option<String> {
    match metadata.get(key) {
        Some(crate::gguf::GgufMetadataValue::String(v)) => Some(v.clone()),
        _ => None,
    }
}

pub(super) fn first_tensor_dims(mapped: &MappedGgufFile, name: &str) -> Option<Vec<u64>> {
    mapped
        .mapped_tensor_infos()
        .iter()
        .find(|t| t.name == name)
        .map(|t| t.dimensions.clone())
}

pub(super) fn first_layer_tensor_dims(mapped: &MappedGgufFile, suffix: &str) -> Option<Vec<u64>> {
    mapped
        .mapped_tensor_infos()
        .iter()
        .find(|t| t.name.starts_with("blk.0.") && t.name.ends_with(suffix))
        .map(|t| t.dimensions.clone())
}

pub(super) fn kv_cache_token_size_for_layers(
    config: &InferenceConfig,
    layers: &[LayerWeights],
) -> usize {
    let widest_loaded_kv = layers
        .iter()
        .take(config.layer_count)
        .filter(|layer| {
            !layer.attn_k.is_empty() || !layer.attn_v.is_empty() || !layer.mla_kv_a_mqa.is_empty()
        })
        .map(|layer| {
            if !layer.mla_kv_a_mqa.is_empty() {
                // DeepSeek-V3 MLA: the KV cache stores the full decompressed
                // per-head K/V (n_heads * k_head_dim), not the compressed
                // latent. k_head_dim = cfg.kv_head_dim() (the MLA key length).
                let kv_lora = layer.mla_kv_a_norm.len();
                let n_heads = config.num_attention_heads.max(1);
                let k_head_dim = config.kv_head_dim();
                let _k_nope_dim = layer.mla_k_b.output_dim(kv_lora) / n_heads;
                let v_head_dim = layer.mla_v_b.output_dim(kv_lora) / n_heads;
                let total_k = n_heads * k_head_dim;
                let total_v = n_heads * v_head_dim;
                total_k.max(total_v)
            } else {
                let key_width = layer.attn_k.output_dim(config.hidden_size);
                let value_width = layer.attn_v.output_dim(config.hidden_size);
                key_width.max(value_width)
            }
        })
        .max()
        .unwrap_or(0);
    if widest_loaded_kv > 0 {
        widest_loaded_kv
    } else {
        config.num_key_value_heads * config.kv_head_dim()
    }
}

pub(super) fn attention_head_dims(
    config: &InferenceConfig,
    layer: &LayerWeights,
    q_len: usize,
    kv_len: usize,
) -> (usize, usize, usize, usize) {
    let q_head_dim = if !layer.attn_q_norm.is_empty()
        && q_len.is_multiple_of(layer.attn_q_norm.len())
    {
        layer.attn_q_norm.len()
    } else if config.num_attention_heads > 0 && q_len.is_multiple_of(config.num_attention_heads) {
        q_len / config.num_attention_heads
    } else {
        q_len
    };
    let q_heads = q_len.checked_div(q_head_dim.max(1)).unwrap_or(1);
    let kv_head_dim = if !layer.attn_k_norm.is_empty()
        && kv_len.is_multiple_of(layer.attn_k_norm.len())
    {
        layer.attn_k_norm.len()
    } else if config.num_key_value_heads > 0 && kv_len.is_multiple_of(config.num_key_value_heads) {
        kv_len / config.num_key_value_heads
    } else if kv_len > 0 {
        kv_len
    } else {
        q_head_dim
    };
    let kv_heads = kv_len.checked_div(kv_head_dim.max(1)).unwrap_or(1);
    (q_head_dim, q_heads, kv_head_dim, kv_heads)
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
pub(super) struct ConvHistoryRing {
    slots: Vec<f32>,
    dim: usize,
    capacity: usize,
    head: usize,
    len: usize,
}

impl ConvHistoryRing {
    pub(super) fn new(capacity: usize, dim: usize) -> Self {
        Self {
            slots: vec![0.0_f32; capacity.saturating_mul(dim)],
            dim,
            capacity: capacity.max(1),
            head: 0,
            len: 0,
        }
    }

    pub(super) fn push(&mut self, frame: &[f32]) {
        if self.dim == 0 || frame.len() != self.dim {
            return;
        }
        let start = self.head * self.dim;
        self.slots[start..start + self.dim].copy_from_slice(frame);
        self.head = (self.head + 1) % self.capacity;
        self.len = (self.len + 1).min(self.capacity);
    }

    pub(super) fn past_frame(&self, steps_back: usize) -> Option<&[f32]> {
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

pub(super) fn weight_block_info(qtype: GgufQuantizationType) -> (usize, usize) {
    quant_block_layout(qtype).unwrap_or((1, 4))
}

/// Borrow a quantized weight tensor's raw bytes for the batched expert GEMV.
/// Returns `None` for f32 weights (which use the per-expert fallback path).
pub(super) fn expert_matrix(weight: &WeightStorage) -> Option<(GgufQuantizationType, &[u8])> {
    match weight {
        WeightStorage::Quantized(qtype, data) => Some((*qtype, data.as_slice())),
        WeightStorage::MmapQuantized(qtype, mmap, offset, size) => {
            Some((*qtype, &mmap[*offset..*offset + *size]))
        }
        WeightStorage::F32(_) => None,
    }
}

pub(super) fn gemv_expert_weight(
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

pub(super) fn gemv_weight(
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
pub(super) fn gemv_weight_fused(
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
pub(super) fn add_repeating_bias(buf: &mut [f32], bias: &[f32]) {
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
pub(super) fn gemm_weight(
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
    pub(super) attn_norm: Vec<f32>,
    pub(super) attn_q: WeightStorage,
    pub(super) attn_q_bias: Vec<f32>,
    pub(super) attn_k: WeightStorage,
    pub(super) attn_k_bias: Vec<f32>,
    pub(super) attn_v: WeightStorage,
    pub(super) attn_v_bias: Vec<f32>,
    pub(super) attn_output: WeightStorage,
    pub(super) attn_output_bias: Vec<f32>,
    pub(super) ffn_norm: Vec<f32>,
    pub(super) post_attention_norm: Vec<f32>,
    /// Gemma post-feedforward norm (applied to FFN output before the residual add).
    pub(super) post_ffn_norm: Vec<f32>,
    // Dense FFN weights (used when num_experts == 0)
    pub(super) ffn_gate: WeightStorage,
    pub(super) ffn_up: WeightStorage,
    pub(super) ffn_down: WeightStorage,
    pub(super) ffn_down_bias: Vec<f32>,
    // MoE expert weights (used when num_experts > 0)
    // Shape: [n_experts, intermediate_size, hidden_size] for gate/up
    //         [n_experts, hidden_size, intermediate_size] for down
    pub(super) ffn_gate_exps: WeightStorage,
    pub(super) ffn_up_exps: WeightStorage,
    pub(super) ffn_down_exps: WeightStorage,
    // MoE router: [hidden_size, n_experts]
    pub(super) ffn_gate_inp: WeightStorage,
    pub(super) attn_qkv: WeightStorage,
    // SSM / Mamba tensors
    pub(super) attn_gate: WeightStorage,
    pub(super) ssm_a: Vec<f32>,
    pub(super) ssm_alpha: Vec<f32>,
    pub(super) ssm_beta: Vec<f32>,
    pub(super) ssm_conv1d: Vec<f32>,
    pub(super) ssm_dt_bias: Vec<f32>,
    pub(super) ssm_norm: Vec<f32>,
    pub(super) ssm_out: WeightStorage,
    // Per-head norms for standard attention
    pub(super) attn_q_norm: Vec<f32>,
    pub(super) attn_k_norm: Vec<f32>,
    // LFM2 short-convolution operator (token mixing on non-attention layers).
    // in_proj: [hidden] -> [3*hidden] producing (B, C, x); conv: depthwise
    // causal conv1d weights laid out [l_cache, hidden]; out_proj: [hidden]->[hidden].
    pub(super) shortconv_in_proj: WeightStorage,
    pub(super) shortconv_conv: Vec<f32>,
    pub(super) shortconv_out_proj: WeightStorage,
    // LFM2MoE per-layer expert routing bias (exp_probs_b), added to sigmoid
    // scores for top-k selection only.
    pub(super) ffn_exp_probs_b: Vec<f32>,
    // DeepSeek2 MLA compressed attention (Kimi K2.x).
    pub(super) mla_q_a: WeightStorage,
    pub(super) mla_q_a_norm: Vec<f32>,
    pub(super) mla_q_b: WeightStorage,
    pub(super) mla_kv_a_mqa: WeightStorage,
    pub(super) mla_kv_a_norm: Vec<f32>,
    pub(super) mla_k_b: WeightStorage,
    pub(super) mla_v_b: WeightStorage,
    // DeepSeek MoE shared expert (shexp) branch.
    pub(super) ffn_gate_shexp: WeightStorage,
    // Optional DeepSeek shared-expert gate. Some DeepSeek-family checkpoints
    // store `mlp.shared_expert_gate.weight`; when present it sigmoid-scales the
    // unconditional shared expert output, but it is not part of routed top-k.
    pub(super) ffn_gate_inp_shexp: WeightStorage,
    pub(super) ffn_up_shexp: WeightStorage,
    pub(super) ffn_down_shexp: WeightStorage,
}

/// Qwen3.5/Qwen3.6-style in-model MTP (`nextn`) draft block.
///
/// GGUF stores one extra decoder block after the target stack (`blk.N.*`) plus
/// the `blk.N.nextn.*` fusion/head tensors. The regular block weights are kept
/// in `layer`; the extra tensors combine a token embedding and the target hidden
/// state, then project the MTP hidden state back through a shared or dedicated
/// output head.
#[derive(Debug, Clone, PartialEq, Default)]
pub(super) struct MtpWeights {
    pub(super) layer: LayerWeights,
    pub(super) eh_proj: WeightStorage,
    pub(super) enorm: Vec<f32>,
    pub(super) hnorm: Vec<f32>,
    pub(super) embed_tokens: WeightStorage,
    pub(super) shared_head_norm: Vec<f32>,
    pub(super) shared_head_head: WeightStorage,
}

impl MtpWeights {
    pub(super) fn is_usable(&self, config: &InferenceConfig) -> bool {
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
    pub(super) config: InferenceConfig,
    pub(super) tok_embeddings: WeightStorage,
    pub(super) tok_embeddings_cols: usize,
    pub(super) norm_weight: Vec<f32>,
    pub(super) output_weight: WeightStorage,
    pub(super) layers: Vec<LayerWeights>,
    pub(super) mtp: Option<MtpWeights>,
    pub(super) kv_cache: KvCache,
    /// Maps absolute layer index → KV cache layer index for attention layers.
    /// Non-attention (shortconv, Mamba) layers have `None` and never write the KV cache.
    /// Allows KV cache to be sized to only the attention-layer count instead of
    /// all layers (e.g. 6 instead of 24 for LFM2MoE), saving several GB.
    pub(super) kv_layer_map: Vec<Option<usize>>,
    // Mamba/SSM persistent state
    pub(super) ssm_states: Vec<Vec<f32>>, // [layer][state_dim]
    pub(super) ssm_conv_buffers: Vec<ConvHistoryRing>,
    pub(super) workspace: Workspace,
    /// Final output-normalized hidden row for the most recent target token.
    /// Native MTP consumes this row as its target-hidden input.
    pub(super) last_output_hidden: Vec<f32>,
    /// Target layer indices whose hidden states are snapshotted for EAGLE3 draft fusion.
    pub(super) eagle3_capture_layers: Vec<usize>,
    pub(super) eagle3_layer_hiddens: Vec<Option<Vec<f32>>>,
    /// Token pending GPU embedding lookup in the next `run_layer_range` call.
    #[cfg(feature = "cuda")]
    pub(super) pending_embed_token: Option<crate::model::Token>,
}

/// Caller-owned per-sequence KV buffer for [`InferenceModel::forward_batch`].
///
/// The model's own `kv_cache` is a single flat timeline and cannot multiplex N
/// concurrent decode sequences, so continuous batching keeps each sequence's KV
/// in its own `SeqKv`. Layout (F32 only) is layer-major then position-major:
/// element for `(kv_layer_idx, pos, channel)` lives at
/// `kv_layer_idx * capacity_tokens * kv_len + pos * kv_len + channel`,
/// where `kv_len = num_key_value_heads * kv_head_dim`. `len` is the number of
/// KV positions already written (== the sequence's next decode position).
#[derive(Debug, Clone)]
pub struct SeqKv {
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub len: usize,
    pub capacity_tokens: usize,
}

impl SeqKv {
    /// Allocate a zeroed KV buffer for one sequence sized for `kv_layer_count`
    /// attention layers, `capacity_tokens` positions, and `kv_len` channels.
    pub fn new(kv_layer_count: usize, capacity_tokens: usize, kv_len: usize) -> Self {
        let elems = kv_layer_count * capacity_tokens * kv_len;
        Self {
            key: vec![0.0_f32; elems],
            value: vec![0.0_f32; elems],
            len: 0,
            capacity_tokens,
        }
    }
}

impl InferenceModel {
    /// Access the model's inference configuration.
    pub fn config(&self) -> &InferenceConfig {
        &self.config
    }

    /// Number of attention layers stored in the KV cache (== the number of
    /// per-layer slots a [`SeqKv`] buffer must reserve for `forward_batch`).
    pub fn kv_layer_count(&self) -> usize {
        self.kv_cache.config().layer_count
    }

    /// Whether continuous-batching decode is enabled (env `OX_BATCHED_DECODE`).
    /// The paged runtime / batched-decode bench consult this before routing N
    /// decode tokens through [`InferenceModel::forward_batch`]; OFF by default so
    /// every existing path stays byte-identical.
    pub fn batched_decode_enabled(&self) -> bool {
        layers::ox_batched_decode_enabled()
    }

    /// KV row width (`num_key_value_heads * kv_head_dim`) used to size a
    /// [`SeqKv`] buffer for `forward_batch`.
    pub fn kv_row_len(&self) -> usize {
        if let Some(layer0) = self.layers.first() {
            if !layer0.attn_k.is_empty() {
                return layer0.attn_k.output_dim(self.config.hidden_size);
            }
        }
        self.kv_cache.config().token_size()
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

pub(super) fn lookup_embedding_from_storage(
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

#[path = "inference/batch_engine.rs"]
mod batch_engine;
pub use batch_engine::{BatchConfig, ContinuousBatchEngine, SeqId, StepOutput};
#[path = "inference/forward.rs"]
mod forward;
#[path = "inference/layers.rs"]
mod layers;
#[path = "inference/load.rs"]
mod load;
#[path = "inference/moe.rs"]
mod moe;
#[path = "inference/mtp.rs"]
mod mtp;

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

    // 2b. DeepSeek-V3 group-limited routing. Experts are partitioned into
    // `expert_group_count` contiguous groups; each group is ranked by the sum
    // of its top-2 selection scores, the top `expert_group_used_count` groups
    // are kept, and all experts outside them are masked (-inf) before the
    // global top-k below. `expert_group_count <= 1` (e.g. Kimi-K2) is a no-op,
    // leaving the existing global top-k path byte-for-byte unchanged.
    if cfg.expert_group_count > 1
        && cfg.expert_group_used_count > 0
        && cfg.expert_group_used_count < cfg.expert_group_count
        && n_experts % cfg.expert_group_count == 0
    {
        let n_group = cfg.expert_group_count;
        let group_size = n_experts / n_group;
        // Reuse a thread-local scratch buffer for the per-group scores instead
        // of allocating a fresh `Vec` every decode step (this routing block
        // runs once per token).
        thread_local! {
            static GROUP_SCORES: std::cell::RefCell<Vec<(usize, f32)>> =
                const { std::cell::RefCell::new(Vec::new()) };
        }
        GROUP_SCORES.with_borrow_mut(|group_scores| {
            group_scores.clear();
            group_scores.extend((0..n_group).map(|g| {
                let grp = &expert_scores[g * group_size..g * group_size + group_size];
                let (mut top1, mut top2) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
                for &(_, s) in grp {
                    if s > top1 {
                        top2 = top1;
                        top1 = s;
                    } else if s > top2 {
                        top2 = s;
                    }
                }
                (g, if top2.is_finite() { top1 + top2 } else { top1 })
            }));
            group_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for &(g, _) in group_scores.iter().skip(cfg.expert_group_used_count) {
                for e in &mut expert_scores[g * group_size..g * group_size + group_size] {
                    e.1 = f32::NEG_INFINITY;
                }
            }
        });
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

    // 4. Gather the selected experts and their routing weights. The routed
    // contribution is scaled by `expert_weights_scale` (DeepSeek-V3/Kimi
    // `routed_scaling_factor`); folding it into the per-expert weight here
    // applies it uniformly across the fused, non-fused, and f32 expert paths
    // below. Defaults to 1.0 for every non-DeepSeek MoE arch.
    let routed_scale = if cfg.expert_weights_scale > 0.0 {
        cfg.expert_weights_scale
    } else {
        1.0
    };
    let n_sel = n_experts_per_tok;
    let mut selected: Vec<usize> = Vec::with_capacity(n_sel);
    let mut weights: Vec<f32> = Vec::with_capacity(n_sel);
    for &(expert_idx, _sel_score) in expert_scores.iter().take(n_sel) {
        selected.push(expert_idx);
        weights.push(routed_scale * router_logits[expert_idx] / weight_norm);
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
            .map_err(|e| ModelError::InferenceFailed(format!("{e:?}")))?;
        // On the CUDA gpu-native attention path the device-resident F16 KV cache
        // is authoritative for the whole run (the host cache above is warmed but
        // not read by the fused decode). It must be rolled back in lock-step for
        // speculative rejection/commit. The host gets the inclusive INDEX
        // (`consumed_tokens - 1`); the device gets the row COUNT
        // (`consumed_tokens`) — gpu_kv_rewind sets kv_seq_len = count, which keeps
        // exactly `consumed_tokens` valid rows, matching the host after rewind.
        // Gated by both cfg(cuda) and the runtime OX_GPU_ATTN flag so non-CUDA
        // builds and non-gpu-attn runs are byte-identical to the host-only path.
        #[cfg(feature = "cuda")]
        if layers::ox_gpu_attn_enabled() {
            let kv_layers = self.kv_cache.config().layer_count;
            for kv_layer in 0..kv_layers {
                crate::cuda::gpu_kv_rewind(kv_layer, consumed_tokens).map_err(|e| {
                    ModelError::InferenceFailed(format!("gpu_kv_rewind l{kv_layer}: {e}"))
                })?;
            }
        }
        Ok(())
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
        let use_batched_verifier = std::env::var("OX_SPEC_VERIFY_BATCHED")
            .ok()
            .is_some_and(|value| !matches!(value.as_str(), "0" | "false" | "off"))
            && tokens.len() > 1
            && self.layers_supported_for_batched();
        if use_batched_verifier {
            let logits = self
                .forward_batched(tokens, start_pos, true)?
                .unwrap_or_default();
            session.record_tokens(tokens.len());
            return Ok(logits);
        }
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
        // Under OX_GPU_ATTN the device F16 KV cache (not the host cache) is
        // authoritative for the whole run. forward_batched only warms the HOST
        // cache, so device prompt rows [0, prompt_len) would stay zero and the
        // fused decode would attend over an all-zero prefix → token-0 divergence +
        // early EOS. Force the per-token path so each prompt position is appended
        // to the device cache via gpu_attn_block_fused_q4k's launch_kv_append_f16
        // (identical RoPE + f16 store to decode). With OX_GPU_ATTN unset this is
        // byte-identical to the previous condition.
        let use_batched = tokens.len() > 1
            && self.layers_supported_for_batched()
            && !layers::ox_gpu_attn_enabled();
        let logits = if use_batched {
            // Prefill the prompt in one batched pass so every weight matmul is a
            // GEMM (decode-once per weight block) rather than `tokens.len()`
            // separate GEMVs. Intermediate logits are discarded so only the
            // last token's lm_head is computed.
            //
            // Batched prefill is now enabled for all KV cache dtypes (not just
            // F32). The KV cache writes in forward_batched use kv_cache.set()
            // which already handles all quant types correctly.
            self.forward_batched(tokens, start_pos, true)?
                .and_then(|mut rows| rows.pop())
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
                mmap_index: 0,
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
    fn deepseek_v3_moe_metadata_is_parsed_for_kimi_style_routing() {
        let mapped = MappedGgufFile::from_parsed_for_test(GgufFile {
            version: 3,
            tensor_count: 3,
            metadata: BTreeMap::from([
                (
                    "general.architecture".to_owned(),
                    GgufMetadataValue::String("deepseek2".to_owned()),
                ),
                (
                    "deepseek2.block_count".to_owned(),
                    GgufMetadataValue::Uint32(61),
                ),
                (
                    "deepseek2.embedding_length".to_owned(),
                    GgufMetadataValue::Uint32(7168),
                ),
                (
                    "deepseek2.feed_forward_length".to_owned(),
                    GgufMetadataValue::Uint32(18432),
                ),
                (
                    "deepseek2.attention.head_count".to_owned(),
                    GgufMetadataValue::Uint32(64),
                ),
                (
                    "deepseek2.attention.head_count_kv".to_owned(),
                    GgufMetadataValue::Uint32(64),
                ),
                (
                    "deepseek2.attention.key_length_mla".to_owned(),
                    GgufMetadataValue::Uint32(128),
                ),
                (
                    "deepseek2.expert_count".to_owned(),
                    GgufMetadataValue::Uint32(384),
                ),
                (
                    "deepseek2.expert_used_count".to_owned(),
                    GgufMetadataValue::Uint32(8),
                ),
                (
                    "deepseek2.expert_feed_forward_length".to_owned(),
                    GgufMetadataValue::Uint32(2048),
                ),
                (
                    "deepseek2.leading_dense_block_count".to_owned(),
                    GgufMetadataValue::Uint32(1),
                ),
                (
                    "deepseek2.expert_gating_func".to_owned(),
                    GgufMetadataValue::Uint32(2),
                ),
                (
                    "deepseek2.expert_weights_scale".to_owned(),
                    GgufMetadataValue::Float32(2.827),
                ),
                (
                    "deepseek2.expert_group_count".to_owned(),
                    GgufMetadataValue::Uint32(1),
                ),
            ]),
            tensor_infos: vec![
                GgufTensorInfo {
                    name: "tok_embeddings.weight".to_owned(),
                    dimensions: vec![7168, 160000],
                    ggml_type: 0,
                    relative_offset: 0,
                    absolute_offset: 0,
                    mmap_index: 0,
                },
                GgufTensorInfo {
                    name: "blk.1.ffn_gate_inp.weight".to_owned(),
                    dimensions: vec![7168, 384],
                    ggml_type: 0,
                    relative_offset: 0,
                    absolute_offset: 0,
                    mmap_index: 0,
                },
                GgufTensorInfo {
                    name: "blk.1.ffn_gate_shexp.weight".to_owned(),
                    dimensions: vec![7168, 2048],
                    ggml_type: 0,
                    relative_offset: 0,
                    absolute_offset: 0,
                    mmap_index: 0,
                },
            ],
            alignment: 32,
            data_section_start: 0,
        });

        let cfg = InferenceConfig::from_gguf(&mapped);

        assert_eq!(cfg.architecture, ModelArchitecture::DeepSeek);
        assert!(cfg.architecture.uses_moe());
        assert!(cfg.architecture.uses_mla());
        assert_eq!(cfg.layer_count, 61);
        assert_eq!(cfg.hidden_size, 7168);
        assert_eq!(cfg.num_experts, 384);
        assert_eq!(cfg.num_experts_per_tok, 8);
        assert_eq!(cfg.expert_intermediate_size, 2048);
        assert_eq!(cfg.leading_dense_layers, 1);
        assert!(cfg.expert_gating_sigmoid);
        assert!((cfg.expert_weights_scale - 2.827).abs() < 1e-6);
        assert_eq!(cfg.expert_group_count, 1);
        assert_eq!(cfg.kv_head_dim(), 128);
    }

    #[test]
    fn glm_dsa_config_from_gguf_metadata() {
        let mapped = MappedGgufFile::from_parsed_for_test(GgufFile {
            version: 3,
            tensor_count: 1,
            metadata: BTreeMap::from([
                (
                    "general.architecture".to_owned(),
                    GgufMetadataValue::String("glm-dsa".to_owned()),
                ),
                (
                    "glm-dsa.block_count".to_owned(),
                    GgufMetadataValue::Uint32(79),
                ),
                (
                    "glm-dsa.nextn_predict_layers".to_owned(),
                    GgufMetadataValue::Uint32(1),
                ),
                (
                    "glm-dsa.embedding_length".to_owned(),
                    GgufMetadataValue::Uint32(6144),
                ),
                (
                    "glm-dsa.feed_forward_length".to_owned(),
                    GgufMetadataValue::Uint32(12288),
                ),
                (
                    "glm-dsa.attention.head_count".to_owned(),
                    GgufMetadataValue::Uint32(64),
                ),
                (
                    "glm-dsa.attention.head_count_kv".to_owned(),
                    GgufMetadataValue::Uint32(1),
                ),
                (
                    "glm-dsa.attention.key_length_mla".to_owned(),
                    GgufMetadataValue::Uint32(256),
                ),
                (
                    "glm-dsa.expert_count".to_owned(),
                    GgufMetadataValue::Uint32(256),
                ),
                (
                    "glm-dsa.expert_used_count".to_owned(),
                    GgufMetadataValue::Uint32(8),
                ),
                (
                    "glm-dsa.expert_feed_forward_length".to_owned(),
                    GgufMetadataValue::Uint32(2048),
                ),
                (
                    "glm-dsa.leading_dense_block_count".to_owned(),
                    GgufMetadataValue::Uint32(3),
                ),
                (
                    "glm-dsa.expert_gating_func".to_owned(),
                    GgufMetadataValue::Uint32(2),
                ),
                (
                    "glm-dsa.expert_weights_scale".to_owned(),
                    GgufMetadataValue::Float32(2.5),
                ),
                (
                    "glm-dsa.rope.freq_base".to_owned(),
                    GgufMetadataValue::Float32(8_000_000.0),
                ),
                (
                    "glm-dsa.vocab_size".to_owned(),
                    GgufMetadataValue::Uint32(154880),
                ),
            ]),
            tensor_infos: vec![GgufTensorInfo {
                name: "tok_embeddings.weight".to_owned(),
                dimensions: vec![6144, 154880],
                ggml_type: 0,
                relative_offset: 0,
                absolute_offset: 0,
                mmap_index: 0,
            }],
            alignment: 32,
            data_section_start: 0,
        });

        let cfg = InferenceConfig::from_gguf(&mapped);

        assert_eq!(cfg.architecture, ModelArchitecture::GlmMoeDsa);
        assert!(cfg.architecture.uses_moe());
        assert!(cfg.architecture.uses_mla());
        assert_eq!(cfg.layer_count, 78);
        assert_eq!(cfg.hidden_size, 6144);
        assert_eq!(cfg.num_experts, 256);
        assert_eq!(cfg.num_experts_per_tok, 8);
        assert_eq!(cfg.leading_dense_layers, 3);
        assert!(cfg.expert_gating_sigmoid);
        assert!((cfg.expert_weights_scale - 2.5).abs() < 1e-6);
        assert_eq!(cfg.kv_head_dim(), 256);
        assert_eq!(cfg.vocab_size, 154880);
        assert!((cfg.rope_theta - 8_000_000.0).abs() < 1.0);
    }

    #[test]
    fn hunyuan_moe_config_from_gguf_metadata() {
        // Tencent Hunyuan (hy_v3): standard GQA + qk_norm attention, sigmoid
        // MoE with a shared expert and one leading dense block. Not MLA.
        let mapped = MappedGgufFile::from_parsed_for_test(GgufFile {
            version: 3,
            tensor_count: 1,
            metadata: BTreeMap::from([
                (
                    "general.architecture".to_owned(),
                    GgufMetadataValue::String("hunyuan-moe".to_owned()),
                ),
                (
                    "hunyuan-moe.block_count".to_owned(),
                    GgufMetadataValue::Uint32(80),
                ),
                (
                    "hunyuan-moe.embedding_length".to_owned(),
                    GgufMetadataValue::Uint32(4096),
                ),
                (
                    "hunyuan-moe.feed_forward_length".to_owned(),
                    GgufMetadataValue::Uint32(13312),
                ),
                (
                    "hunyuan-moe.attention.head_count".to_owned(),
                    GgufMetadataValue::Uint32(64),
                ),
                (
                    "hunyuan-moe.attention.head_count_kv".to_owned(),
                    GgufMetadataValue::Uint32(8),
                ),
                (
                    "hunyuan-moe.attention.key_length".to_owned(),
                    GgufMetadataValue::Uint32(128),
                ),
                (
                    "hunyuan-moe.attention.value_length".to_owned(),
                    GgufMetadataValue::Uint32(128),
                ),
                (
                    "hunyuan-moe.expert_count".to_owned(),
                    GgufMetadataValue::Uint32(192),
                ),
                (
                    "hunyuan-moe.expert_used_count".to_owned(),
                    GgufMetadataValue::Uint32(8),
                ),
                (
                    "hunyuan-moe.expert_feed_forward_length".to_owned(),
                    GgufMetadataValue::Uint32(1536),
                ),
                (
                    "hunyuan-moe.expert_shared_count".to_owned(),
                    GgufMetadataValue::Uint32(1),
                ),
                (
                    "hunyuan-moe.leading_dense_block_count".to_owned(),
                    GgufMetadataValue::Uint32(1),
                ),
                (
                    "hunyuan-moe.expert_gating_func".to_owned(),
                    GgufMetadataValue::Uint32(2),
                ),
                (
                    "hunyuan-moe.expert_weights_scale".to_owned(),
                    GgufMetadataValue::Float32(2.826),
                ),
                (
                    "hunyuan-moe.rope.freq_base".to_owned(),
                    GgufMetadataValue::Float32(11_158_840.0),
                ),
                (
                    "hunyuan-moe.vocab_size".to_owned(),
                    GgufMetadataValue::Uint32(120832),
                ),
            ]),
            tensor_infos: vec![GgufTensorInfo {
                name: "tok_embeddings.weight".to_owned(),
                dimensions: vec![4096, 120832],
                ggml_type: 0,
                relative_offset: 0,
                absolute_offset: 0,
                mmap_index: 0,
            }],
            alignment: 32,
            data_section_start: 0,
        });

        let cfg = InferenceConfig::from_gguf(&mapped);

        assert_eq!(cfg.architecture, ModelArchitecture::HunyuanMoe);
        assert!(cfg.architecture.uses_moe());
        assert!(!cfg.architecture.uses_mla());
        assert_eq!(cfg.layer_count, 80);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_experts, 192);
        assert_eq!(cfg.num_experts_per_tok, 8);
        assert_eq!(cfg.expert_intermediate_size, 1536);
        assert_eq!(cfg.leading_dense_layers, 1);
        assert!(cfg.expert_gating_sigmoid);
        assert!((cfg.expert_weights_scale - 2.826).abs() < 1e-6);
        assert_eq!(cfg.num_attention_heads, 64);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.kv_head_dim(), 128);
        assert_eq!(cfg.vocab_size, 120832);
        assert!((cfg.rope_theta - 11_158_840.0).abs() < 1.0);
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
    fn gemma4_mixed_kv_layers_size_cache_for_widest_projection() {
        let cfg = InferenceConfig {
            architecture: ModelArchitecture::Gemma,
            hidden_size: 3840,
            num_key_value_heads: 8,
            key_value_head_dim: 512,
            ..Default::default()
        };
        let local_layer = LayerWeights {
            attn_k: WeightStorage::F32(vec![0.0; 3840 * 2048]),
            attn_v: WeightStorage::F32(vec![0.0; 3840 * 2048]),
            ..Default::default()
        };
        let global_layer = LayerWeights {
            attn_k: WeightStorage::F32(vec![0.0; 3840 * 512]),
            attn_v: WeightStorage::F32(vec![0.0; 3840 * 512]),
            ..Default::default()
        };

        assert_eq!(
            kv_cache_token_size_for_layers(&cfg, &[local_layer, global_layer]),
            2048
        );
    }

    #[test]
    fn gemma4_global_attention_uses_norm_width_for_kv_head_dim() {
        let cfg = InferenceConfig {
            num_attention_heads: 16,
            num_key_value_heads: 8,
            ..Default::default()
        };
        let layer = LayerWeights {
            attn_q_norm: vec![0.0; 512],
            attn_k_norm: vec![0.0; 512],
            ..Default::default()
        };

        assert_eq!(
            attention_head_dims(&cfg, &layer, 8192, 512),
            (512, 16, 512, 1)
        );
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
            eagle3_capture_layers: Vec::new(),
            eagle3_layer_hiddens: Vec::new(),
            #[cfg(feature = "cuda")]
            pending_embed_token: None,
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
    fn batched_prefill_rejects_mixed_attention_widths() {
        let mut model = tiny_inference_model();
        model.config.hidden_size = 2;
        model.layers = vec![
            LayerWeights {
                attn_q: WeightStorage::F32(vec![0.0; 4]),
                attn_k: WeightStorage::F32(vec![0.0; 4]),
                attn_v: WeightStorage::F32(vec![0.0; 4]),
                ..Default::default()
            },
            LayerWeights {
                attn_q: WeightStorage::F32(vec![0.0; 8]),
                attn_k: WeightStorage::F32(vec![0.0; 2]),
                attn_v: WeightStorage::F32(vec![0.0; 2]),
                ..Default::default()
            },
        ];

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

    /// A tiny model with one real attention layer so the KV cache holds rows
    /// (the default `tiny_inference_model` has `layer_count: 0`). Used to exercise
    /// `rewind_to` against a populated cache for speculative rollback parity.
    fn tiny_inference_model_one_layer() -> InferenceModel {
        let mut model = tiny_inference_model();
        model.config.layer_count = 1;
        model.config.intermediate_size = 2;
        let layer = LayerWeights {
            attn_norm: vec![1.0, 1.0],
            attn_q: WeightStorage::F32(vec![0.05; 2 * 2]),
            attn_k: WeightStorage::F32(vec![0.05; 2 * 2]),
            attn_v: WeightStorage::F32(vec![0.05; 2 * 2]),
            attn_output: WeightStorage::F32(vec![0.05; 2 * 2]),
            post_attention_norm: vec![1.0, 1.0],
            ffn_gate: WeightStorage::F32(vec![0.05; 2 * 2]),
            ffn_up: WeightStorage::F32(vec![0.05; 2 * 2]),
            ffn_down: WeightStorage::F32(vec![0.05; 2 * 2]),
            ..Default::default()
        };
        model.layers = vec![layer];
        let kv_cache_config = KvCacheConfig {
            layer_count: 1,
            context_size: model.config.context_size,
            head_count: model.config.num_key_value_heads,
            head_dim: model.config.kv_head_dim(),
            dtype: DType::F32,
            quantization: Default::default(),
        };
        model.kv_cache = KvCache::new(kv_cache_config).expect("one-layer kv cache should be valid");
        model.kv_layer_map = vec![Some(0)];
        model.workspace = Workspace::for_config(&model.config);
        model
    }

    /// Speculative rollback parity (CPU): `Model::rewind_to` paired with
    /// `Session::rewind_to` must leave the model in a state where re-forwarding the
    /// rewound suffix reproduces the original per-position logits byte-for-byte.
    /// This is the lossless-under-greedy invariant the CUDA device-cache rewind
    /// (`gpu_kv_rewind`) mirrors on the GPU-native path. The device branch in
    /// `rewind_to` is `cfg(cuda)`-gated, so on this CPU build only the host path
    /// runs and it must already hold.
    #[test]
    fn rewind_to_is_lossless_for_reforwarded_suffix() {
        let prompt: [Token; 3] = [0, 1, 2];

        // Reference: forward the whole prompt once, capturing per-position logits.
        let mut reference = tiny_inference_model_one_layer();
        let mut ref_session = Session::new();
        let reference_logits = reference
            .forward_many(&prompt, &mut ref_session)
            .expect("reference forward_many should succeed");
        assert_eq!(reference_logits.len(), prompt.len());
        assert_eq!(ref_session.consumed_tokens(), prompt.len());

        // Subject: forward the prefix, rewind to the end of the first token, then
        // re-forward the suffix. The re-forwarded positions must match the
        // reference exactly.
        let mut subject = tiny_inference_model_one_layer();
        let mut session = Session::new();
        let _ = subject
            .forward_many(&prompt, &mut session)
            .expect("subject prefix forward should succeed");
        assert_eq!(session.consumed_tokens(), prompt.len());

        // Roll both the model KV cache and the session back to "1 token consumed".
        let rewind_to = 1usize;
        subject
            .rewind_to(rewind_to)
            .expect("rewind_to should accept an in-range position");
        session.rewind_to(rewind_to);
        assert_eq!(session.consumed_tokens(), rewind_to);

        // Re-forward positions 1..3 and compare against the reference.
        let suffix = &prompt[rewind_to..];
        let replayed = subject
            .forward_many(suffix, &mut session)
            .expect("suffix replay should succeed");
        assert_eq!(replayed.len(), suffix.len());
        assert_eq!(session.consumed_tokens(), prompt.len());
        for (offset, replay_logits) in replayed.iter().enumerate() {
            assert_eq!(
                replay_logits,
                &reference_logits[rewind_to + offset],
                "replayed logits at position {} diverged from reference",
                rewind_to + offset
            );
        }
    }

    /// `rewind_to(0)` must be accepted on a populated cache and leave the model
    /// equivalent to a freshly constructed one (a fresh forward reproduces the
    /// from-scratch logits). Guards the speculative "commit to empty prefix" edge.
    #[test]
    fn rewind_to_zero_restores_fresh_state() {
        let mut fresh = tiny_inference_model_one_layer();
        let mut fresh_session = Session::new();
        let fresh_logits = fresh
            .forward(&[0, 1], &mut fresh_session)
            .expect("fresh forward should succeed");

        let mut reused = tiny_inference_model_one_layer();
        let mut session = Session::new();
        let _ = reused
            .forward(&[2, 0, 1], &mut session)
            .expect("warm-up forward should succeed");
        reused
            .rewind_to(0)
            .expect("rewind_to(0) should be accepted");
        session.rewind_to(0);
        assert_eq!(session.consumed_tokens(), 0);

        let after_reset = reused
            .forward(&[0, 1], &mut session)
            .expect("post-reset forward should succeed");
        assert_eq!(after_reset, fresh_logits);
        assert_eq!(session.consumed_tokens(), 2);
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
                false,
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

    /// CORE CONTINUOUS-BATCHING INVARIANT: a batch of N sequences run through
    /// `forward_batch` must produce per-sequence logits IDENTICAL (bit-for-bit)
    /// to running each sequence alone. The N sequences here are byte-identical
    /// (same token + same position + same KV history), so every `result[i]` must
    /// equal `result[0]` at every decode step — the amortized GEMM must not mix
    /// rows. Also asserts the absolute value matches the N=1 reference.
    #[test]
    fn forward_batch_per_sequence_logits_match_single_sequence() {
        let prompt: [Token; 3] = [0, 1, 2];
        let steps = 5usize;
        let cap = 32usize;

        // --- N=1 reference: drive forward_batch one sequence, greedy decode. ---
        let mut reference = tiny_inference_model_one_layer();
        let kv_layers = reference.kv_layer_count();
        let kv_len = reference.kv_row_len();
        let mut ref_kv = vec![SeqKv::new(kv_layers, cap, kv_len)];
        let mut ref_pos = 0usize;
        // Seed the prompt (positions 0..prompt.len()).
        let mut ref_last: Token = 0;
        for &tok in &prompt {
            let out = reference
                .forward_batch(&[(tok, ref_pos)], &mut ref_kv, true)
                .expect("reference seed forward_batch");
            ref_last = argmax(&out[0]) as Token;
            ref_pos += 1;
        }
        let mut reference_decode_logits: Vec<Logits> = Vec::with_capacity(steps);
        for _ in 0..steps {
            let out = reference
                .forward_batch(&[(ref_last, ref_pos)], &mut ref_kv, true)
                .expect("reference decode forward_batch");
            ref_last = argmax(&out[0]) as Token;
            reference_decode_logits.push(out.into_iter().next().unwrap());
            ref_pos += 1;
        }

        // --- N>1 batched: 3 byte-identical sequences decoded together. ---
        let n = 3usize;
        let mut model = tiny_inference_model_one_layer();
        let mut kv: Vec<SeqKv> = (0..n).map(|_| SeqKv::new(kv_layers, cap, kv_len)).collect();
        let mut pos = 0usize;
        let mut last = vec![0 as Token; n];
        for &tok in &prompt {
            let rows: Vec<(Token, usize)> = (0..n).map(|_| (tok, pos)).collect();
            let out = model
                .forward_batch(&rows, &mut kv, true)
                .expect("batched seed forward_batch");
            for i in 0..n {
                last[i] = argmax(&out[i]) as Token;
            }
            pos += 1;
        }
        for step in 0..steps {
            let rows: Vec<(Token, usize)> = (0..n).map(|i| (last[i], pos)).collect();
            let out = model
                .forward_batch(&rows, &mut kv, true)
                .expect("batched decode forward_batch");
            assert_eq!(out.len(), n);
            // (a) all rows identical to each other (no cross-row contamination).
            for i in 1..n {
                assert_eq!(
                    out[i], out[0],
                    "step {step}: batched row {i} diverged from row 0"
                );
            }
            // (b) batched row equals the N=1 reference bit-for-bit.
            assert_eq!(
                out[0], reference_decode_logits[step],
                "step {step}: batched logits diverged from single-sequence reference"
            );
            for i in 0..n {
                last[i] = argmax(&out[i]) as Token;
            }
            pos += 1;
        }
    }

    /// DISTINCT sequences in one batch must each match their own standalone run.
    /// Two same-length but DIFFERENT-token sequences are seeded and decoded; each
    /// batched row must equal that sequence run alone at every step, proving the
    /// batched GEMM keeps per-row KV/position isolation (no cross-contamination).
    #[test]
    fn forward_batch_distinct_sequences_match_standalone() {
        let cap = 32usize;
        let steps = 4usize;
        let prompt_a: [Token; 3] = [0, 1, 2];
        let prompt_b: [Token; 3] = [2, 0, 1];

        // Standalone runner: seed `prompt`, then greedily decode `steps`,
        // returning the per-step logits.
        fn run_alone(prompt: &[Token], steps: usize, cap: usize) -> Vec<Logits> {
            let mut m = tiny_inference_model_one_layer();
            let kvl = m.kv_layer_count();
            let kw = m.kv_row_len();
            let mut kv = vec![SeqKv::new(kvl, cap, kw)];
            let mut pos = 0usize;
            let mut last: Token = 0;
            for &t in prompt {
                let out = m.forward_batch(&[(t, pos)], &mut kv, true).expect("seed");
                last = argmax(&out[0]) as Token;
                pos += 1;
            }
            let mut decoded = Vec::with_capacity(steps);
            for _ in 0..steps {
                let mut out = m
                    .forward_batch(&[(last, pos)], &mut kv, true)
                    .expect("decode");
                last = argmax(&out[0]) as Token;
                decoded.push(out.remove(0));
                pos += 1;
            }
            decoded
        }

        let a_alone = run_alone(&prompt_a, steps, cap);
        let b_alone = run_alone(&prompt_b, steps, cap);

        // Batched A+B together.
        let mut m = tiny_inference_model_one_layer();
        let kvl = m.kv_layer_count();
        let kw = m.kv_row_len();
        let mut kv = vec![SeqKv::new(kvl, cap, kw), SeqKv::new(kvl, cap, kw)];
        let mut pos = 0usize;
        let mut last = [0 as Token; 2];
        for p in 0..prompt_a.len() {
            let rows = vec![(prompt_a[p], pos), (prompt_b[p], pos)];
            let out = m.forward_batch(&rows, &mut kv, true).expect("joint seed");
            last[0] = argmax(&out[0]) as Token;
            last[1] = argmax(&out[1]) as Token;
            pos += 1;
        }
        for step in 0..steps {
            let rows = vec![(last[0], pos), (last[1], pos)];
            let out = m.forward_batch(&rows, &mut kv, true).expect("joint decode");
            assert_eq!(
                out[0], a_alone[step],
                "step {step}: batched seq A diverged from standalone A"
            );
            assert_eq!(
                out[1], b_alone[step],
                "step {step}: batched seq B diverged from standalone B"
            );
            last[0] = argmax(&out[0]) as Token;
            last[1] = argmax(&out[1]) as Token;
            pos += 1;
        }
    }

    /// VARIABLE-LENGTH continuous batching: real serving mixes sequences at
    /// different KV depths in ONE batch call (a freshly-admitted request decodes
    /// alongside an older one). This drives `forward_batch` with rows whose
    /// positions differ — `rows=[(tok_a, 4), (tok_b, 2)]` — so each row reads a
    /// genuinely different `kv[i].len`, exercising per-sequence KV prefix reads,
    /// causal extent (`seq_len = kv[i].len + 1`), and per-row RoPE position. Each
    /// batched row must still match its own standalone run bit-for-bit.
    #[test]
    fn forward_batch_variable_length_sequences() {
        let cap = 32usize;
        let steps = 4usize;
        // Sequence A is prefilled with 4 tokens, sequence B with 2 — so on the
        // first joint decode call kv[A].len=4 and kv[B].len=2 (variable length).
        let prompt_a: [Token; 4] = [0, 1, 2, 1];
        let prompt_b: [Token; 2] = [2, 0];

        // Standalone runner: seed `prompt`, then greedily decode `steps`,
        // returning the per-step logits. Mirrors run_alone above but reused here
        // with differing prompt lengths to build the variable-length references.
        fn run_alone(prompt: &[Token], steps: usize, cap: usize) -> Vec<Logits> {
            let mut m = tiny_inference_model_one_layer();
            let kvl = m.kv_layer_count();
            let kw = m.kv_row_len();
            let mut kv = vec![SeqKv::new(kvl, cap, kw)];
            let mut pos = 0usize;
            let mut last: Token = 0;
            for &t in prompt {
                let out = m.forward_batch(&[(t, pos)], &mut kv, true).expect("seed");
                last = argmax(&out[0]) as Token;
                pos += 1;
            }
            let mut decoded = Vec::with_capacity(steps);
            for _ in 0..steps {
                let mut out = m
                    .forward_batch(&[(last, pos)], &mut kv, true)
                    .expect("decode");
                last = argmax(&out[0]) as Token;
                decoded.push(out.remove(0));
                pos += 1;
            }
            decoded
        }

        let a_alone = run_alone(&prompt_a, steps, cap);
        let b_alone = run_alone(&prompt_b, steps, cap);

        // Batched A+B together, where A and B start at DIFFERENT KV depths. Each
        // sequence advances its own `pos` independently; the joint call always
        // passes each row at its own `kv[i].len`.
        let mut m = tiny_inference_model_one_layer();
        let kvl = m.kv_layer_count();
        let kw = m.kv_row_len();
        let mut kv = vec![SeqKv::new(kvl, cap, kw), SeqKv::new(kvl, cap, kw)];
        let mut pos = [0usize; 2];
        let mut last = [0 as Token; 2];

        // Seed both prompts in lockstep up to the SHORTER prompt; then finish
        // seeding A's tail alone so its kv length runs ahead of B's.
        let shared = prompt_a.len().min(prompt_b.len());
        for p in 0..shared {
            let rows = vec![(prompt_a[p], pos[0]), (prompt_b[p], pos[1])];
            let out = m.forward_batch(&rows, &mut kv, true).expect("joint seed");
            last[0] = argmax(&out[0]) as Token;
            last[1] = argmax(&out[1]) as Token;
            pos[0] += 1;
            pos[1] += 1;
        }
        // Seed A's remaining prompt tokens alone (single-row batch over kv[0..1]).
        for &t in &prompt_a[shared..] {
            let out = m
                .forward_batch(&[(t, pos[0])], &mut kv[0..1], true)
                .expect("seq A tail seed");
            last[0] = argmax(&out[0]) as Token;
            pos[0] += 1;
        }
        // Sanity: the two sequences are now at different KV depths.
        assert_eq!(kv[0].len, prompt_a.len());
        assert_eq!(kv[1].len, prompt_b.len());
        assert_ne!(kv[0].len, kv[1].len, "test must exercise variable lengths");

        for step in 0..steps {
            // The joint decode call mixes pos[0] != pos[1] — exactly the case the
            // synchronized-length tests never hit.
            let rows = vec![(last[0], pos[0]), (last[1], pos[1])];
            assert_ne!(
                rows[0].1, rows[1].1,
                "batch rows must have distinct positions"
            );
            let out = m.forward_batch(&rows, &mut kv, true).expect("joint decode");
            assert_eq!(
                out[0], a_alone[step],
                "step {step}: variable-length batched seq A diverged from standalone A"
            );
            assert_eq!(
                out[1], b_alone[step],
                "step {step}: variable-length batched seq B diverged from standalone B"
            );
            last[0] = argmax(&out[0]) as Token;
            last[1] = argmax(&out[1]) as Token;
            pos[0] += 1;
            pos[1] += 1;
        }
    }

    /// Greedy standalone decode reference: feed `prompt`, then emit exactly
    /// `max_new` tokens (the first comes from the prompt's final-position logits,
    /// matching [`ContinuousBatchEngine`]'s prefill semantics). Returns the
    /// generated token sequence.
    fn decode_alone(prompt: &[Token], max_new: usize, cap: usize) -> Vec<Token> {
        let mut m = tiny_inference_model_one_layer();
        let kvl = m.kv_layer_count();
        let kw = m.kv_row_len();
        let mut kv = vec![SeqKv::new(kvl, cap, kw)];
        let mut pos = 0usize;
        let mut first_logits: Logits = Vec::new();
        let last_idx = prompt.len() - 1;
        for (i, &t) in prompt.iter().enumerate() {
            let need = i == last_idx;
            let out = m
                .forward_batch(&[(t, pos)], &mut kv, need)
                .expect("standalone seed");
            if need {
                first_logits = out.into_iter().next().unwrap_or_default();
            }
            pos += 1;
        }
        let mut last = argmax(&first_logits) as Token;
        let mut toks = vec![last];
        while toks.len() < max_new {
            let out = m
                .forward_batch(&[(last, pos)], &mut kv, true)
                .expect("standalone decode");
            last = argmax(&out[0]) as Token;
            toks.push(last);
            pos += 1;
        }
        toks
    }

    /// CONTINUOUS-BATCHING ENGINE EQUIVALENCE: heterogeneous prompts (different
    /// lengths) and different `max_new` budgets, all driven by one
    /// [`ContinuousBatchEngine`] with greedy selection, must each produce the
    /// EXACT token sequence they produce when decoded alone. Sequences finish at
    /// different steps, exercising mid-batch retirement (swap_remove keeps the
    /// active KV slice aligned).
    #[test]
    fn engine_heterogeneous_matches_standalone() {
        let cap = 64usize;
        let prompts: [Vec<Token>; 3] = [vec![0, 1, 2], vec![2, 0], vec![1, 2, 0, 1]];
        let max_new: [usize; 3] = [5, 7, 4];

        let refs: Vec<Vec<Token>> = prompts
            .iter()
            .zip(max_new.iter())
            .map(|(p, &mn)| decode_alone(p, mn, cap))
            .collect();

        let mut model = tiny_inference_model_one_layer();
        let cfg = BatchConfig {
            max_batch: 8,
            default_capacity_tokens: cap,
        };
        let mut engine = ContinuousBatchEngine::new(&model, cfg);
        let ids: Vec<SeqId> = prompts
            .iter()
            .zip(max_new.iter())
            .map(|(p, &mn)| engine.submit(p.clone(), mn, None))
            .collect();

        let mut got: BTreeMap<SeqId, Vec<Token>> = BTreeMap::new();
        let mut steps = 0usize;
        while engine.has_work() {
            let outs = engine
                .step(&mut model, |_id, logits| argmax(logits) as Token)
                .expect("engine step");
            for o in outs {
                got.entry(o.seq_id).or_default().push(o.token);
            }
            steps += 1;
            assert!(steps < 1000, "engine failed to terminate");
        }

        for (i, &id) in ids.iter().enumerate() {
            assert_eq!(
                got.get(&id).map(Vec::as_slice),
                Some(refs[i].as_slice()),
                "engine seq {id} (prompt {:?}) diverged from standalone",
                prompts[i]
            );
            assert_eq!(got[&id].len(), max_new[i], "seq {id} wrong token count");
        }
        assert!(!engine.has_work());
        assert_eq!(engine.active_len(), 0);
    }

    /// MID-FLIGHT ADMISSION: a request submitted AFTER another is already
    /// decoding must still match its standalone run, proving the batched GEMM
    /// keeps per-sequence isolation when batch membership changes over time
    /// (the defining property of continuous batching).
    #[test]
    fn engine_mid_flight_admission_matches_standalone() {
        let cap = 64usize;
        let prompt_a: Vec<Token> = vec![0, 1, 2, 1];
        let prompt_b: Vec<Token> = vec![2, 0];
        let max_a = 8usize;
        let max_b = 6usize;

        let ref_a = decode_alone(&prompt_a, max_a, cap);
        let ref_b = decode_alone(&prompt_b, max_b, cap);

        let mut model = tiny_inference_model_one_layer();
        let mut engine = ContinuousBatchEngine::new(
            &model,
            BatchConfig {
                max_batch: 8,
                default_capacity_tokens: cap,
            },
        );

        let mut got: BTreeMap<SeqId, Vec<Token>> = BTreeMap::new();
        let id_a = engine.submit(prompt_a.clone(), max_a, None);

        // Run A alone for two steps (prefill + one decode) before B joins.
        for _ in 0..2 {
            let outs = engine
                .step(&mut model, |_id, l| argmax(l) as Token)
                .expect("step A-only");
            for o in outs {
                got.entry(o.seq_id).or_default().push(o.token);
            }
        }
        assert_eq!(engine.active_len(), 1, "A should be solo-active before B");

        // B joins mid-flight.
        let id_b = engine.submit(prompt_b.clone(), max_b, None);
        while engine.has_work() {
            let outs = engine
                .step(&mut model, |_id, l| argmax(l) as Token)
                .expect("step A+B");
            for o in outs {
                got.entry(o.seq_id).or_default().push(o.token);
            }
        }

        assert_eq!(
            got.get(&id_a).map(Vec::as_slice),
            Some(ref_a.as_slice()),
            "mid-flight: seq A diverged from standalone"
        );
        assert_eq!(
            got.get(&id_b).map(Vec::as_slice),
            Some(ref_b.as_slice()),
            "mid-flight: seq B diverged from standalone"
        );
    }

    fn argmax(v: &[f32]) -> usize {
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &x) in v.iter().enumerate() {
            if x > best_v {
                best_v = x;
                best = i;
            }
        }
        best
    }
}
