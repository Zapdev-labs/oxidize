//! Architecture training plan.
//!
//! Every [`ModelArchitecture`] the inference engine knows about gets an
//! explicit plan: what the layer-wise trainer actually executes, and which
//! parameters have a correct gradient. The match is exhaustive so a new
//! architecture fails to compile until training states what it can do.

use oxidize_core::inference::{InferenceConfig, ModelArchitecture};

use crate::error::{FinetuneError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForwardFidelity {
    /// The layer-wise forward implements this architecture's blocks.
    Full,
    /// A block the trainer does not run. Training is refused unless the
    /// caller opts in.
    Partial(&'static str),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArchPlan {
    pub architecture: ModelArchitecture,
    pub name: &'static str,
    pub attention: String,
    pub ffn: String,
    pub position: String,
    pub norm: String,
    pub residual: &'static str,
    pub fidelity: ForwardFidelity,
    pub trainable: &'static str,
}

impl ArchPlan {
    pub fn from_config(cfg: &InferenceConfig) -> Self {
        let name = arch_name(cfg.architecture);
        let fidelity = match cfg.architecture {
            ModelArchitecture::Falcon | ModelArchitecture::Gpt2 => ForwardFidelity::Partial(
                "architecture declares ALiBi; the CPU decoder and this trainer apply RoPE",
            ),
            ModelArchitecture::Llama
            | ModelArchitecture::Mistral
            | ModelArchitecture::Mixtral
            | ModelArchitecture::DeepSeek
            | ModelArchitecture::Qwen
            | ModelArchitecture::Gemma
            | ModelArchitecture::Phi
            | ModelArchitecture::GptJ
            | ModelArchitecture::GptNeoX
            | ModelArchitecture::MiniMax
            | ModelArchitecture::Lfm2
            | ModelArchitecture::Lfm2Moe
            | ModelArchitecture::GlmMoeDsa
            | ModelArchitecture::HunyuanMoe => ForwardFidelity::Full,
        };
        Self {
            architecture: cfg.architecture,
            name,
            attention: describe_attention(cfg),
            ffn: describe_ffn(cfg),
            position: describe_position(cfg),
            norm: describe_norm(cfg),
            residual: match cfg.architecture {
                ModelArchitecture::Gemma | ModelArchitecture::Phi => {
                    "sequential pre-norm residual (CPU decode ignores the parallel-attention flag)"
                }
                _ => "sequential pre-norm residual",
            },
            fidelity,
            trainable: "LM-head LoRA. The gradient is exact: logits = W h + scale·B A h, and only A and B are updated. Attention, FFN, router, and expert weights stay frozen because the quantized stack has no autograd.",
        }
    }

    pub fn render(&self) -> String {
        let forward = match self.fidelity {
            ForwardFidelity::Full => "full".to_owned(),
            ForwardFidelity::Partial(reason) => format!("partial — {reason}"),
        };
        format!(
            "architecture: {name}\n\
             attention:    {attention}\n\
             ffn:          {ffn}\n\
             position:     {position}\n\
             norm:         {norm}\n\
             residual:     {residual}\n\
             forward:      {forward}\n\
             trainable:    {trainable}",
            name = self.name,
            attention = self.attention,
            ffn = self.ffn,
            position = self.position,
            norm = self.norm,
            residual = self.residual,
            trainable = self.trainable,
        )
    }

    pub fn ensure_supported(&self, allow_partial: bool) -> Result<()> {
        if let ForwardFidelity::Partial(reason) = self.fidelity
            && !allow_partial
        {
            return Err(FinetuneError::Model(format!(
                "{name} is only partially supported ({reason}). Pass --allow-partial-arch to train the blocks the layer-wise forward does implement.",
                name = self.name
            )));
        }
        Ok(())
    }
}

fn arch_name(arch: ModelArchitecture) -> &'static str {
    match arch {
        ModelArchitecture::Llama => "llama",
        ModelArchitecture::Mistral => "mistral",
        ModelArchitecture::Mixtral => "mixtral",
        ModelArchitecture::DeepSeek => "deepseek",
        ModelArchitecture::Qwen => "qwen",
        ModelArchitecture::Gemma => "gemma",
        ModelArchitecture::Phi => "phi",
        ModelArchitecture::Falcon => "falcon",
        ModelArchitecture::Gpt2 => "gpt2",
        ModelArchitecture::GptJ => "gptj",
        ModelArchitecture::GptNeoX => "gptneox",
        ModelArchitecture::MiniMax => "minimax",
        ModelArchitecture::Lfm2 => "lfm2",
        ModelArchitecture::Lfm2Moe => "lfm2moe",
        ModelArchitecture::GlmMoeDsa => "glm-moe-dsa",
        ModelArchitecture::HunyuanMoe => "hunyuan-moe",
    }
}

fn describe_attention(cfg: &InferenceConfig) -> String {
    let mut parts = Vec::new();
    if cfg.architecture.uses_mla() {
        parts.push("multi-head latent attention".to_owned());
    } else if cfg.architecture.uses_shortconv() {
        parts.push(format!(
            "short-conv hybrid (kernel {}) + GQA {}/{}",
            cfg.shortconv_l_cache.max(1),
            cfg.num_attention_heads,
            cfg.num_key_value_heads
        ));
    } else {
        parts.push(format!(
            "GQA {} query / {} kv heads",
            cfg.num_attention_heads, cfg.num_key_value_heads
        ));
    }
    if cfg.sliding_window > 0 {
        if cfg.sliding_window_pattern > 0 {
            parts.push(format!(
                "sliding window {} (global every {} layers)",
                cfg.sliding_window, cfg.sliding_window_pattern
            ));
        } else {
            parts.push(format!("sliding window {}", cfg.sliding_window));
        }
    }
    parts.join(", ")
}

fn describe_ffn(cfg: &InferenceConfig) -> String {
    if cfg.architecture.uses_moe() || cfg.num_experts > 0 {
        let gate = if cfg.expert_gating_sigmoid {
            ", sigmoid router"
        } else {
            ", softmax router"
        };
        format!(
            "MoE {} experts, top-{}{gate}",
            cfg.num_experts,
            cfg.num_experts_per_tok.max(1)
        )
    } else if cfg.gelu_ffn {
        "GeGLU (tanh approximation)".to_owned()
    } else {
        "SwiGLU".to_owned()
    }
}

fn describe_position(cfg: &InferenceConfig) -> String {
    let mut text = format!("RoPE theta {}", cfg.rope_theta);
    if cfg.rope_dim > 0 {
        text.push_str(&format!(", partial dim {}", cfg.rope_dim));
    }
    if cfg.rope_theta_swa > 0.0 {
        text.push_str(&format!(", local theta {}", cfg.rope_theta_swa));
    }
    if cfg.yarn_factor > 0.0 {
        text.push_str(&format!(", YaRN factor {}", cfg.yarn_factor));
    }
    match cfg.architecture {
        ModelArchitecture::Falcon | ModelArchitecture::Gpt2 => {
            text.push_str("; architecture declares ALiBi, CPU decode still applies RoPE");
        }
        ModelArchitecture::GptJ | ModelArchitecture::GptNeoX => {
            text.push_str("; enum flag is ALiBi, model and CPU decode use RoPE");
        }
        _ => {}
    }
    text
}

fn describe_norm(cfg: &InferenceConfig) -> String {
    let mut text = if cfg.rms_norm_weight_plus_one {
        "RMSNorm (1 + weight)".to_owned()
    } else {
        "RMSNorm".to_owned()
    };
    if cfg.sandwich_norm {
        text.push_str(", sandwich");
    }
    if cfg.embedding_scale != 1.0 {
        text.push_str(&format!(", embedding scale {}", cfg.embedding_scale));
    }
    text
}

/// First input index and number of supervised positions for a prompt+continuation.
///
/// Position `i` predicts token `i + 1`. Continuation tokens occupy
/// `ids[prompt_len..]`, so the first predicting state is the last prompt token.
pub fn continuation_span(prompt_len: usize, seq_len: usize) -> Option<(usize, usize)> {
    if seq_len < 2 || prompt_len >= seq_len {
        return None;
    }
    let start = prompt_len.saturating_sub(1);
    let n = seq_len - 1 - start;
    if n == 0 { None } else { Some((start, n)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan_for(arch: ModelArchitecture) -> ArchPlan {
        let mut cfg = InferenceConfig::default();
        cfg.architecture = arch;
        cfg.num_attention_heads = 8;
        cfg.num_key_value_heads = 2;
        cfg.hidden_size = 32;
        if arch.uses_moe() {
            cfg.num_experts = 4;
            cfg.num_experts_per_tok = 2;
        }
        if arch.uses_sliding_window() {
            cfg.sliding_window = 128;
        }
        if arch == ModelArchitecture::Gemma {
            cfg.gelu_ffn = true;
            cfg.sandwich_norm = true;
            cfg.embedding_scale = 8.0;
            cfg.sliding_window = 64;
            cfg.sliding_window_pattern = 6;
            cfg.rope_theta_swa = 10_000.0;
        }
        ArchPlan::from_config(&cfg)
    }

    #[test]
    fn every_architecture_has_a_plan() {
        let all = [
            ModelArchitecture::Llama,
            ModelArchitecture::Mistral,
            ModelArchitecture::Mixtral,
            ModelArchitecture::DeepSeek,
            ModelArchitecture::Qwen,
            ModelArchitecture::Gemma,
            ModelArchitecture::Phi,
            ModelArchitecture::Falcon,
            ModelArchitecture::Gpt2,
            ModelArchitecture::GptJ,
            ModelArchitecture::GptNeoX,
            ModelArchitecture::MiniMax,
            ModelArchitecture::Lfm2,
            ModelArchitecture::Lfm2Moe,
            ModelArchitecture::GlmMoeDsa,
            ModelArchitecture::HunyuanMoe,
        ];
        for arch in all {
            let plan = plan_for(arch);
            assert_eq!(plan.architecture, arch);
            assert!(!plan.render().is_empty());
            let alibi_on_rope_path =
                matches!(arch, ModelArchitecture::Falcon | ModelArchitecture::Gpt2);
            if alibi_on_rope_path {
                assert!(
                    matches!(plan.fidelity, ForwardFidelity::Partial(_)),
                    "{arch:?}"
                );
                assert!(plan.ensure_supported(false).is_err(), "{arch:?}");
                assert!(plan.ensure_supported(true).is_ok(), "{arch:?}");
            } else {
                assert!(plan.ensure_supported(false).is_ok(), "{arch:?}");
                assert_eq!(plan.fidelity, ForwardFidelity::Full);
            }
        }
    }

    #[test]
    fn deepseek_plan_names_latent_attention() {
        let plan = plan_for(ModelArchitecture::DeepSeek);
        assert!(plan.attention.contains("latent"), "{}", plan.attention);
        assert!(plan.ffn.contains("MoE"), "{}", plan.ffn);
        let glm = plan_for(ModelArchitecture::GlmMoeDsa);
        assert!(glm.attention.contains("latent"), "{}", glm.attention);
    }

    #[test]
    fn gemma_plan_names_its_real_blocks() {
        let plan = plan_for(ModelArchitecture::Gemma);
        assert!(plan.ffn.contains("GeGLU"), "{}", plan.ffn);
        assert!(plan.norm.contains("sandwich"), "{}", plan.norm);
        assert!(plan.residual.contains("sequential"), "{}", plan.residual);
        assert!(
            plan.residual
                .contains("ignores the parallel-attention flag"),
            "{}",
            plan.residual
        );
        assert!(plan.attention.contains("sliding window 64"));
        assert!(plan.position.contains("local theta"));
    }

    #[test]
    fn alibi_flagged_architectures_still_describe_rope() {
        for arch in [
            ModelArchitecture::Falcon,
            ModelArchitecture::Gpt2,
            ModelArchitecture::GptJ,
            ModelArchitecture::GptNeoX,
        ] {
            let plan = plan_for(arch);
            assert!(plan.position.contains("RoPE"), "{}", plan.position);
            assert!(plan.position.contains("ALiBi"), "{}", plan.position);
        }
        assert!(matches!(
            plan_for(ModelArchitecture::GptJ).fidelity,
            ForwardFidelity::Full
        ));
        assert!(matches!(
            plan_for(ModelArchitecture::GptNeoX).fidelity,
            ForwardFidelity::Full
        ));
    }

    #[test]
    fn partial_forward_is_refused_unless_allowed() {
        let mut plan = plan_for(ModelArchitecture::Llama);
        plan.fidelity = ForwardFidelity::Partial("missing block");
        assert!(plan.ensure_supported(false).is_err());
        assert!(plan.ensure_supported(true).is_ok());
    }

    #[test]
    fn continuation_span_scores_only_the_answer() {
        assert_eq!(continuation_span(3, 6), Some((2, 3)));
        assert_eq!(continuation_span(0, 4), Some((0, 3)));
        assert_eq!(continuation_span(4, 4), None);
        assert_eq!(continuation_span(1, 1), None);
    }
}
