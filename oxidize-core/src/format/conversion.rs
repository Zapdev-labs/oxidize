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
        Some("qwen") | Some("qwen2") | Some("qwen3") => ModelArchitecture::Qwen,
        Some("gemma") => ModelArchitecture::Gemma,
        Some("phi") => ModelArchitecture::Phi,
        Some(other) => ModelArchitecture::Unknown(other.to_string()),
        None => ModelArchitecture::Unknown("missing".to_string()),
    }
}

pub fn map_hf_tensor_name(name: &str) -> String {
    name.strip_prefix("model.")
        .unwrap_or(name)
        .replace("embed_tokens", "token_embd")
        .replace("self_attn.q_proj", "attn_q")
        .replace("self_attn.k_proj", "attn_k")
        .replace("self_attn.v_proj", "attn_v")
        .replace("self_attn.o_proj", "attn_output")
        .replace("mlp.gate_proj", "ffn_gate")
        .replace("mlp.up_proj", "ffn_up")
        .replace("mlp.down_proj", "ffn_down")
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
            "layers.0.attn_q.weight"
        );
    }
}
