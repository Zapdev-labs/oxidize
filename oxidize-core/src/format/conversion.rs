use crate::gguf::GgufQuantizationType;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelArchitecture {
    Llama,
    Mistral,
    Qwen,
    Gemma,
    Phi,
    Unknown(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversionPlan {
    pub architecture: ModelArchitecture,
    pub tensor_name_map: BTreeMap<String, String>,
    pub target_quantization: Option<GgufQuantizationType>,
    pub special_tokens: BTreeMap<String, u32>,
}

pub fn detect_architecture(metadata: &BTreeMap<String, String>) -> ModelArchitecture {
    let arch = metadata
        .get("general.architecture")
        .or_else(|| metadata.get("model_type"))
        .map(|value| value.to_ascii_lowercase());
    match arch.as_deref() {
        Some("llama") => ModelArchitecture::Llama,
        Some("mistral") => ModelArchitecture::Mistral,
        Some("qwen") | Some("qwen2") | Some("qwen2moe") | Some("qwen3") | Some("qwen35") => {
            ModelArchitecture::Qwen
        }
        Some("gemma") => ModelArchitecture::Gemma,
        Some("phi") => ModelArchitecture::Phi,
        Some(other) => ModelArchitecture::Unknown(other.to_string()),
        None => ModelArchitecture::Unknown("missing".to_string()),
    }
}

pub fn map_hf_tensor_name(name: &str) -> String {
    match name {
        "model.embed_tokens.weight" => "tok_embeddings.weight".to_owned(),
        "lm_head.weight" => "output.weight".to_owned(),
        "model.norm.weight" => "norm.weight".to_owned(),
        _ => {
            let Some((layer, suffix)) = name
                .strip_prefix("model.layers.")
                .and_then(|rest| rest.split_once('.'))
            else {
                return name.to_owned();
            };

            if let Some(rest) = suffix.strip_prefix("block_sparse_moe.experts.") {
                let Some((expert, expert_weight)) = rest.split_once('.') else {
                    return name.to_owned();
                };
                let mapped_expert_weight = match expert_weight {
                    "w1.weight" => "ffn_gate",
                    "w2.weight" => "ffn_down",
                    "w3.weight" => "ffn_up",
                    _ => return name.to_owned(),
                };
                return format!("blk.{layer}.{mapped_expert_weight}.{expert}.weight");
            }

            let mapped_suffix = match suffix {
                "input_layernorm.weight" => "attn_norm.weight",
                "post_attention_layernorm.weight" => "ffn_norm.weight",
                "self_attn.q_proj.weight" => "attn_q.weight",
                "self_attn.k_proj.weight" => "attn_k.weight",
                "self_attn.v_proj.weight" => "attn_v.weight",
                "self_attn.o_proj.weight" => "attn_output.weight",
                // Attention QKV/output biases (present in Qwen2 and similar
                // architectures). Dropping these silently breaks attention and
                // yields fluent-but-incoherent output.
                "self_attn.q_proj.bias" => "attn_q.bias",
                "self_attn.k_proj.bias" => "attn_k.bias",
                "self_attn.v_proj.bias" => "attn_v.bias",
                "self_attn.o_proj.bias" => "attn_output.bias",
                "mlp.up_proj.weight" => "ffn_up.weight",
                "mlp.gate_proj.weight" => "ffn_gate.weight",
                "mlp.down_proj.weight" => "ffn_down.weight",
                "mlp.up_proj.bias" => "ffn_up.bias",
                "mlp.gate_proj.bias" => "ffn_gate.bias",
                "mlp.down_proj.bias" => "ffn_down.bias",
                "block_sparse_moe.gate.weight" => "ffn_gate_inp.weight",
                _ => return name.to_owned(),
            };
            format!("blk.{layer}.{mapped_suffix}")
        }
    }
}

/// Normalize a tensor name from GGUF or HF conventions into oxidize's canonical
/// GGUF naming. Returns `None` for tensors that should be skipped (e.g. vision).
pub fn normalize_gguf_tensor_name(name: &str) -> Option<String> {
    match name {
        "tok_embeddings.weight"
        | "token_embd.weight"
        | "output.weight"
        | "norm.weight"
        | "output_norm.weight" => Some(name.to_owned()),
        n if n.starts_with("blk.") => Some(n.to_owned()),
        _ => {
            let mapped = map_hf_tensor_name(name);
            if mapped.starts_with("blk.")
                || matches!(
                    mapped.as_str(),
                    "tok_embeddings.weight"
                        | "token_embd.weight"
                        | "output.weight"
                        | "norm.weight"
                        | "output_norm.weight"
                )
            {
                Some(mapped)
            } else {
                None
            }
        }
    }
}

pub fn build_conversion_plan(
    metadata: &BTreeMap<String, String>,
    tensors: impl IntoIterator<Item = String>,
    target_quantization: Option<GgufQuantizationType>,
) -> ConversionPlan {
    let tensor_name_map = tensors
        .into_iter()
        .map(|name| {
            let mapped = map_hf_tensor_name(&name);
            (name, mapped)
        })
        .collect();
    ConversionPlan {
        architecture: detect_architecture(metadata),
        tensor_name_map,
        target_quantization,
        special_tokens: BTreeMap::new(),
    }
}

pub fn parse_special_token_id(metadata: &BTreeMap<String, String>, key: &str) -> Option<u32> {
    metadata
        .get(key)
        .and_then(|value| value.parse::<u32>().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_qkv_attention_biases() {
        // Qwen2-style attention biases must be preserved with canonical names;
        // dropping them silently breaks attention (fluent-but-incoherent output).
        for (hf, gguf) in [
            ("model.layers.3.self_attn.q_proj.bias", "blk.3.attn_q.bias"),
            ("model.layers.3.self_attn.k_proj.bias", "blk.3.attn_k.bias"),
            ("model.layers.3.self_attn.v_proj.bias", "blk.3.attn_v.bias"),
            (
                "model.layers.3.self_attn.o_proj.bias",
                "blk.3.attn_output.bias",
            ),
        ] {
            assert_eq!(map_hf_tensor_name(hf), gguf, "bias mapping for {hf}");
        }
    }

    #[test]
    fn conversion_detects_arch_and_maps_tensors() {
        let mut metadata = BTreeMap::new();
        metadata.insert("model_type".into(), "qwen2".into());
        let plan = build_conversion_plan(
            &metadata,
            ["model.layers.0.self_attn.q_proj.weight".to_string()],
            Some(GgufQuantizationType::Q4_K_M),
        );
        assert_eq!(plan.architecture, ModelArchitecture::Qwen);
        assert_eq!(
            plan.tensor_name_map["model.layers.0.self_attn.q_proj.weight"],
            "blk.0.attn_q.weight"
        );
    }

    #[test]
    fn conversion_detects_qwen_metadata_variants() {
        let mut metadata = BTreeMap::new();
        metadata.insert("model_type".into(), "qwen35".into());
        assert_eq!(detect_architecture(&metadata), ModelArchitecture::Qwen);

        metadata.insert("model_type".into(), "qwen2moe".into());
        assert_eq!(detect_architecture(&metadata), ModelArchitecture::Qwen);
    }

    #[test]
    fn conversion_maps_hf_tensor_names_to_canonical_names() {
        assert_eq!(
            map_hf_tensor_name("model.embed_tokens.weight"),
            "tok_embeddings.weight"
        );
        assert_eq!(
            map_hf_tensor_name("model.layers.2.mlp.down_proj.weight"),
            "blk.2.ffn_down.weight"
        );
        assert_eq!(
            map_hf_tensor_name("model.layers.3.block_sparse_moe.experts.7.w1.weight"),
            "blk.3.ffn_gate.7.weight"
        );
    }
}
