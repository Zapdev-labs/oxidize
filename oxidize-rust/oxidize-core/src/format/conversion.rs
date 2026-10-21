use crate::gguf::GgufQuantizationType;
use safetensors::tensor::Dtype;
use std::collections::BTreeMap;

/// A decoded tensor staged for GGUF output: `(name, dtype, shape, raw bytes)`.
pub(crate) type StagedTensor = (String, Dtype, Vec<usize>, Vec<u8>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelArchitecture {
    Llama,
    Mistral,
    Qwen,
    DeepSeek,
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
        Some("qwen") | Some("qwen2") | Some("qwen2moe") | Some("qwen3") | Some("qwen35")
        | Some("qwen35moe") => ModelArchitecture::Qwen,
        Some("deepseek") | Some("deepseek2") | Some("deepseek_v2") | Some("deepseek_v3")
        | Some("deepseek_moe") => ModelArchitecture::DeepSeek,
        Some("gemma") => ModelArchitecture::Gemma,
        Some("phi") => ModelArchitecture::Phi,
        Some(other) => ModelArchitecture::Unknown(other.to_string()),
        None => ModelArchitecture::Unknown("missing".to_string()),
    }
}

/// Map Qwen3.5/3.6 MTP (multi-token prediction) HF tensor names to oxidize's
/// `nextn` GGUF naming. Returns `None` if the name is not an MTP tensor.
///
/// This handles the nested form `model.layers.{L}.mtp.*` where the MTP module is
/// stored as a sub-module of layer `L`. The flat form `mtp.*` (stored as a top-
/// level module) is handled separately by `rewrite_flat_mtp_names` once the
/// causal backbone layer count is known.
///
/// Mapping for nested form:
/// * `model.layers.{L}.mtp.fc.weight` -> `blk.{L}.nextn.eh_proj.weight`
/// * `model.layers.{L}.mtp.pre_fc_norm_embedding.weight` -> `blk.{L}.nextn.enorm.weight`
/// * `model.layers.{L}.mtp.pre_fc_norm_hidden.weight` -> `blk.{L}.nextn.hnorm.weight`
/// * `model.layers.{L}.mtp.norm.weight` -> `blk.{L}.nextn.shared_head_norm.weight`
/// * `model.layers.{L}.mtp.embed_tokens.weight` -> `blk.{L}.nextn.embed_tokens.weight`
/// * `model.layers.{L}.mtp.lm_head.weight` -> `blk.{L}.nextn.shared_head_head.weight`
/// * `model.layers.{L}.mtp.layers.{N}.*` -> `blk.{L+N}.*`
pub fn map_qwen_mtp_tensor_name(name: &str) -> Option<String> {
    let stripped = name
        .strip_prefix("model.language_model.")
        .or_else(|| name.strip_prefix("model."))
        .unwrap_or(name);

    let rest = stripped.strip_prefix("layers.")?;
    let (layer_str, rest) = rest.split_once('.')?;
    let layer: usize = layer_str.parse().ok()?;
    let rest = rest.strip_prefix("mtp.")?;

    map_qwen_mtp_inner(rest, layer)
}

fn map_qwen_mtp_inner(rest: &str, layer: usize) -> Option<String> {
    // Fusion head tensors live directly under `mtp.*`.
    if let Some((head_name, suffix)) = rest.rsplit_once('.')
        && (suffix == "weight" || suffix == "bias")
    {
        let mapped_head = match head_name {
            "fc" => "nextn.eh_proj",
            "pre_fc_norm_embedding" => "nextn.enorm",
            "pre_fc_norm_hidden" => "nextn.hnorm",
            "norm" => "nextn.shared_head_norm",
            "embed_tokens" => "nextn.embed_tokens",
            "lm_head" => "nextn.shared_head_head",
            _ => "",
        };
        if !mapped_head.is_empty() {
            let mapped_suffix = if suffix == "bias" { ".bias" } else { ".weight" };
            return Some(format!("blk.{layer}.{mapped_head}{mapped_suffix}"));
        }
    }

    // Nested MTP transformer block: `mtp.layers.{N}.(...)` -> `blk.{layer+N}.(...)`.
    let rest = rest.strip_prefix("layers.")?;
    let (mtp_layer_str, rest) = rest.split_once('.')?;
    let mtp_layer: usize = mtp_layer_str.parse().ok()?;
    let mapped_layer = layer + mtp_layer;

    let mapped_suffix = match rest {
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.q_proj.bias" => "attn_q.bias",
        "self_attn.k_proj.bias" => "attn_k.bias",
        "self_attn.v_proj.bias" => "attn_v.bias",
        "self_attn.o_proj.bias" => "attn_output.bias",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        "mlp.gate_proj.bias" => "ffn_gate.bias",
        "mlp.up_proj.bias" => "ffn_up.bias",
        "mlp.down_proj.bias" => "ffn_down.bias",
        _ => return None,
    };
    Some(format!("blk.{mapped_layer}.{mapped_suffix}"))
}

/// Map flat Qwen3.5/3.6 MTP tensor names (`mtp.fc.weight`, `mtp.layers.0.*`)
/// to oxidize's `nextn` GGUF naming using a caller-supplied causal backbone
/// layer count as the MTP base layer.
pub fn map_flat_qwen_mtp_tensor_name(name: &str, base_layer: usize) -> Option<String> {
    let stripped = name
        .strip_prefix("model.language_model.")
        .or_else(|| name.strip_prefix("model."))
        .unwrap_or(name);

    let rest = stripped.strip_prefix("mtp.")?;
    map_qwen_mtp_inner(rest, base_layer)
}
/// HF-prefixed tensors (e.g. `model.language_model.layers.0.linear_attn.in_proj_a.weight`)
/// are converted via [`map_hf_tensor_name`]; already-canonical names pass through.
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
            if mapped.is_empty() {
                None
            } else {
                Some(mapped)
            }
        }
    }
}

/// List normalized tensor suffix keys (`attn_qkv.weight`, etc.) for one layer.
pub fn gguf_layer_tensor_keys(
    tensor_names: impl IntoIterator<Item = String>,
    layer_idx: usize,
) -> Vec<String> {
    let prefix = format!("blk.{layer_idx}.");
    let mut keys: Vec<String> = tensor_names
        .into_iter()
        .filter_map(|raw| normalize_gguf_tensor_name(&raw))
        .filter_map(|canonical| canonical.strip_prefix(&prefix).map(str::to_owned))
        .collect();
    keys.sort();
    keys.dedup();
    keys
}

pub fn map_hf_tensor_name(name: &str) -> String {
    if name.starts_with("model.visual.") {
        return String::new();
    }

    // Qwen3.5/3.6 in-model multi-token-prediction (MTP / nextn) tensors.
    // These live under `model.layers.{L}.mtp.*` and map to oxidize's
    // `blk.{L}.nextn.*` fusion head plus an appended transformer block.
    if let Some(mapped) = map_qwen_mtp_tensor_name(name) {
        return mapped;
    }

    let stripped = name
        .strip_prefix("model.language_model.")
        .or_else(|| name.strip_prefix("model."))
        .unwrap_or(name);

    match stripped {
        "embed_tokens.weight" => "tok_embeddings.weight".to_owned(),
        "norm.weight" => "norm.weight".to_owned(),
        "lm_head.weight" => "output.weight".to_owned(),
        _ => {
            let Some((layer, suffix)) = stripped
                .strip_prefix("layers.")
                .and_then(|rest| rest.split_once('.'))
            else {
                return name.to_owned();
            };
            if layer.parse::<usize>().is_err() {
                return name.to_owned();
            }

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

            if let Some(rest) = suffix.strip_prefix("mlp.experts.")
                && let Some((expert, expert_weight)) = rest.split_once('.')
            {
                let mapped_expert_weight = match expert_weight {
                    "gate_proj.weight" => "ffn_gate",
                    "up_proj.weight" => "ffn_up",
                    "down_proj.weight" => "ffn_down",
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
                "self_attn.q_proj.bias" => "attn_q.bias",
                "self_attn.k_proj.bias" => "attn_k.bias",
                "self_attn.v_proj.bias" => "attn_v.bias",
                "self_attn.o_proj.bias" => "attn_output.bias",
                "self_attn.q_norm.weight" => "attn_q_norm.weight",
                "self_attn.k_norm.weight" => "attn_k_norm.weight",
                "linear_attn.in_proj_qkv.weight" => "attn_qkv.weight",
                "linear_attn.in_proj_z.weight" => "attn_gate.weight",
                "linear_attn.in_proj_b.weight" => "ssm_beta.weight",
                "linear_attn.in_proj_a.weight" => "ssm_alpha.weight",
                "linear_attn.A_log" => "ssm_a.weight",
                "linear_attn.dt_bias" => "ssm_dt.bias",
                "linear_attn.norm.weight" => "ssm_norm.weight",
                "linear_attn.out_proj.weight" => "ssm_out.weight",
                "mlp.up_proj.weight" => "ffn_up.weight",
                "mlp.gate_proj.weight" => "ffn_gate.weight",
                "mlp.down_proj.weight" => "ffn_down.weight",
                "mlp.up_proj.bias" => "ffn_up.bias",
                "mlp.gate_proj.bias" => "ffn_gate.bias",
                "mlp.down_proj.bias" => "ffn_down.bias",
                "mlp.gate.weight" => "ffn_gate_inp.weight",
                "mlp.experts.down_proj" => "ffn_down_exps.weight",
                "mlp.shared_expert.gate_proj.weight" => "ffn_gate_shexp.weight",
                "mlp.shared_expert.up_proj.weight" => "ffn_up_shexp.weight",
                "mlp.shared_expert.down_proj.weight" => "ffn_down_shexp.weight",
                "mlp.shared_expert_gate.weight" => "ffn_gate_inp_shexp.weight",
                "block_sparse_moe.gate.weight" => "ffn_gate_inp.weight",
                // Hunyuan (hy_v3): sigmoid router, per-expert selection bias,
                // and a single always-on shared expert (`shared_mlp`).
                "mlp.router.gate.weight" => "ffn_gate_inp.weight",
                "mlp.expert_bias" => "exp_probs_b.bias",
                "mlp.shared_mlp.gate_proj.weight" => "ffn_gate_shexp.weight",
                "mlp.shared_mlp.up_proj.weight" => "ffn_up_shexp.weight",
                "mlp.shared_mlp.down_proj.weight" => "ffn_down_shexp.weight",
                _ => return name.to_owned(),
            };
            format!("blk.{layer}.{mapped_suffix}")
        }
    }
}

/// Split Qwen3.5-MoE fused `gate_up_proj` [E, 2*I, H] into separate gate/up expert tensors.
pub fn split_fused_gate_up_proj(
    layer: usize,
    dtype: Dtype,
    shape: &[usize],
    raw: &[u8],
) -> Option<Vec<StagedTensor>> {
    if shape.len() != 3 || !shape[1].is_multiple_of(2) {
        return None;
    }
    let experts = shape[0];
    let half = shape[1] / 2;
    let hidden = shape[2];
    let elem_size = dtype_element_size(dtype)?;
    let row_stride = shape[1] * hidden * elem_size;
    let half_stride = half * hidden * elem_size;

    let mut gate_data = Vec::with_capacity(experts * half * hidden * elem_size);
    let mut up_data = Vec::with_capacity(experts * half * hidden * elem_size);
    for e in 0..experts {
        let base = e * row_stride;
        gate_data.extend_from_slice(&raw[base..base + half_stride]);
        up_data.extend_from_slice(&raw[base + half_stride..base + row_stride]);
    }

    Some(vec![
        (
            format!("blk.{layer}.ffn_gate_exps.weight"),
            dtype,
            vec![experts, half, hidden],
            gate_data,
        ),
        (
            format!("blk.{layer}.ffn_up_exps.weight"),
            dtype,
            vec![experts, half, hidden],
            up_data,
        ),
    ])
}

/// Flatten `linear_attn.conv1d.weight` [C, 1, K] into oxidize's [K, C] layout.
pub fn flatten_linear_attn_conv1d(
    layer: usize,
    dtype: Dtype,
    shape: &[usize],
    raw: &[u8],
) -> Option<StagedTensor> {
    if shape.len() != 3 || shape[1] != 1 {
        return None;
    }
    let channels = shape[0];
    let kernel = shape[2];
    let elem_size = dtype_element_size(dtype)?;
    let mut flat = vec![0_u8; channels * kernel * elem_size];
    for k in 0..kernel {
        for c in 0..channels {
            let src = (c * kernel + k) * elem_size;
            let dst = (k * channels + c) * elem_size;
            flat[dst..dst + elem_size].copy_from_slice(&raw[src..src + elem_size]);
        }
    }
    Some((
        format!("blk.{layer}.ssm_conv1d.weight"),
        dtype,
        vec![kernel * channels],
        flat,
    ))
}

fn dtype_element_size(dtype: Dtype) -> Option<usize> {
    match dtype {
        Dtype::F32 => Some(4),
        Dtype::F16 => Some(2),
        Dtype::BF16 => Some(2),
        _ => None,
    }
}

/// Expand HF tensors into GGUF-ready tensors (split fused MoE, skip vision).
///
/// A fused `gate_up_proj` that cannot be split is a hard error: emitting the
/// unsplit tensor would produce a GGUF missing `ffn_gate_exps`/`ffn_up_exps`
/// and break MoE inference (the streaming path already errors here).
pub fn preprocess_hf_tensors_for_gguf(
    tensors: Vec<StagedTensor>,
) -> Result<Vec<StagedTensor>, String> {
    let mut out = Vec::with_capacity(tensors.len() + 64);
    for (name, dtype, shape, raw) in tensors {
        if name.starts_with("model.visual.") {
            continue;
        }
        if name.ends_with(".mlp.experts.gate_up_proj") {
            let layer = extract_layer_index(&name).ok_or_else(|| {
                format!(
                    "fused gate_up_proj tensor {name:?} has no parseable layer index; \
                     cannot split into ffn_gate_exps/ffn_up_exps"
                )
            })?;
            let split = split_fused_gate_up_proj(layer, dtype, &shape, &raw).ok_or_else(|| {
                format!(
                    "failed to split fused gate_up_proj tensor {name:?} (shape {shape:?}); \
                     the GGUF would be missing ffn_gate_exps/ffn_up_exps and MoE \
                     inference would break"
                )
            })?;
            out.extend(split);
            continue;
        }
        if name.ends_with(".linear_attn.conv1d.weight")
            && let Some(layer) = extract_layer_index(&name)
            && let Some(flat) = flatten_linear_attn_conv1d(layer, dtype, &shape, &raw)
        {
            out.push(flat);
            continue;
        }
        out.push((name, dtype, shape, raw));
    }
    Ok(out)
}

pub fn extract_layer_index(name: &str) -> Option<usize> {
    let stripped = name
        .strip_prefix("model.language_model.layers.")
        .or_else(|| name.strip_prefix("model.layers."))?;
    stripped.split('.').next()?.parse().ok()
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
    fn conversion_detects_deepseek_metadata_variants() {
        let mut metadata = BTreeMap::new();
        metadata.insert("model_type".into(), "deepseek_v3".into());
        assert_eq!(detect_architecture(&metadata), ModelArchitecture::DeepSeek);

        metadata.insert("model_type".into(), "deepseek2".into());
        assert_eq!(detect_architecture(&metadata), ModelArchitecture::DeepSeek);
    }

    #[test]
    fn maps_qwen35_mtp_tensors() {
        // Nested form: MTP stored as a sub-module of the last backbone layer.
        assert_eq!(
            map_hf_tensor_name("model.layers.32.mtp.fc.weight"),
            "blk.32.nextn.eh_proj.weight"
        );
        assert_eq!(
            map_hf_tensor_name("model.layers.32.mtp.pre_fc_norm_embedding.weight"),
            "blk.32.nextn.enorm.weight"
        );
        assert_eq!(
            map_hf_tensor_name("model.layers.32.mtp.pre_fc_norm_hidden.weight"),
            "blk.32.nextn.hnorm.weight"
        );
        assert_eq!(
            map_hf_tensor_name("model.layers.32.mtp.norm.weight"),
            "blk.32.nextn.shared_head_norm.weight"
        );
        assert_eq!(
            map_hf_tensor_name("model.layers.32.mtp.layers.0.self_attn.q_proj.weight"),
            "blk.32.attn_q.weight"
        );
        assert_eq!(
            map_hf_tensor_name("model.layers.32.mtp.layers.0.mlp.down_proj.weight"),
            "blk.32.ffn_down.weight"
        );

        // Flat form: MTP saved as a top-level module; needs base layer supplied.
        assert_eq!(
            map_flat_qwen_mtp_tensor_name("mtp.fc.weight", 32),
            Some("blk.32.nextn.eh_proj.weight".to_owned())
        );
        assert_eq!(
            map_flat_qwen_mtp_tensor_name("mtp.pre_fc_norm_embedding.weight", 32),
            Some("blk.32.nextn.enorm.weight".to_owned())
        );
        assert_eq!(
            map_flat_qwen_mtp_tensor_name("mtp.layers.0.self_attn.q_proj.weight", 32),
            Some("blk.32.attn_q.weight".to_owned())
        );
        assert_eq!(
            map_flat_qwen_mtp_tensor_name("mtp.layers.0.mlp.down_proj.weight", 32),
            Some("blk.32.ffn_down.weight".to_owned())
        );
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

    #[test]
    fn conversion_maps_qwen35_moe_language_model_tensors() {
        assert_eq!(
            normalize_gguf_tensor_name(
                "model.language_model.layers.0.linear_attn.in_proj_a.weight"
            ),
            Some("blk.0.ssm_alpha.weight".to_owned())
        );
        assert_eq!(
            map_hf_tensor_name("model.language_model.embed_tokens.weight"),
            "tok_embeddings.weight"
        );
        assert_eq!(
            map_hf_tensor_name("model.language_model.layers.0.linear_attn.in_proj_qkv.weight"),
            "blk.0.attn_qkv.weight"
        );
        assert_eq!(
            map_hf_tensor_name("model.language_model.layers.0.linear_attn.in_proj_a.weight"),
            "blk.0.ssm_alpha.weight"
        );
        assert_eq!(
            map_hf_tensor_name("model.language_model.layers.3.mlp.gate.weight"),
            "blk.3.ffn_gate_inp.weight"
        );
        assert_eq!(
            map_hf_tensor_name("model.language_model.layers.0.mlp.experts.down_proj"),
            "blk.0.ffn_down_exps.weight"
        );
        assert_eq!(
            map_hf_tensor_name("model.visual.blocks.0.attn.qkv.weight"),
            ""
        );
    }

    #[test]
    fn split_fused_gate_up_proj_splits_halves() {
        let shape = [2_usize, 4, 2];
        let raw: Vec<u8> = (0_u8..(2 * 4 * 2 * 4)).collect();
        let split = split_fused_gate_up_proj(1, Dtype::F32, &shape, &raw).expect("split");
        assert_eq!(split.len(), 2);
        assert_eq!(split[0].0, "blk.1.ffn_gate_exps.weight");
        assert_eq!(split[0].2, vec![2, 2, 2]);
        assert_eq!(split[1].0, "blk.1.ffn_up_exps.weight");
    }
}
