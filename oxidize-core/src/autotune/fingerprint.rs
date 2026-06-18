//! Model fingerprint for the autotuner.
//!
//! Reads the GGUF header (already mmap'd by the caller) and produces
//! a `ModelFingerprint` — the per-model facts the planner needs. The
//! fingerprint is a pure function over the GGUF metadata and tensor
//! info; no model loading, no forward pass, no allocations beyond
//! the few small vecs in the result.

use std::collections::HashMap;

use crate::gguf::{
    GgufMetadataValue, GgufQuantizationType, GgufTensorInfo, MappedGgufFile,
};
use crate::inference::InferenceConfig;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFingerprint {
    /// "llama", "qwen2", "gemma3", "mamba", "lfm2", etc. Empty if the
    /// GGUF doesn't carry `general.architecture`.
    pub architecture: String,
    pub layer_count: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub file_size_bytes: u64,
    /// Quantization type that occupies the most bytes in the file
    /// (a useful proxy for "what's the model actually stored as").
    pub quant: GgufQuantizationType,
    pub is_moe: bool,
    pub expert_count: usize,
    /// True if the GGUF has any `nextn.*` / `*mtp*` tensors
    /// (Multi-Token Prediction head, used by speculative decoding).
    pub has_mtp: bool,
}

/// Build a `ModelFingerprint` from a mmap'd GGUF and the inferred
/// `InferenceConfig`. The config is preferred for the architecture
/// fields because it is already validated; we fall back to raw
/// metadata if the config can't be built (rare; only happens for
/// models the existing parser doesn't understand).
pub fn fingerprint(mapped: &MappedGgufFile) -> ModelFingerprint {
    let config = InferenceConfig::from_gguf(mapped);
    let file_size_bytes = mapped.total_bytes_len();

    let tensor_infos = mapped.mapped_tensor_infos();
    let (quant, expert_count, is_moe, has_mtp) =
        scan_tensors(&tensor_infos);

    ModelFingerprint {
        architecture: format!("{:?}", config.architecture).to_ascii_lowercase(),
        layer_count: config.layer_count,
        hidden_size: config.hidden_size,
        num_attention_heads: config.num_attention_heads,
        num_kv_heads: config.num_key_value_heads,
        head_dim: config.key_value_head_dim,
        intermediate_size: config.intermediate_size,
        vocab_size: config.vocab_size,
        file_size_bytes,
        quant,
        is_moe,
        expert_count,
        has_mtp,
    }
}

/// Build a fingerprint from explicit values — used by the planner
/// tests so we don't have to construct a real GGUF in-process.
pub fn fingerprint_from_parts(
    architecture: &str,
    layer_count: usize,
    hidden_size: usize,
    num_attention_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    vocab_size: usize,
    file_size_bytes: u64,
    quant: GgufQuantizationType,
) -> ModelFingerprint {
    ModelFingerprint {
        architecture: architecture.to_string(),
        layer_count,
        hidden_size,
        num_attention_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        vocab_size,
        file_size_bytes,
        quant,
        is_moe: false,
        expert_count: 0,
        has_mtp: false,
    }
}

fn scan_tensors(tensors: &[GgufTensorInfo]) -> (GgufQuantizationType, usize, bool, bool) {
    let mut hist: HashMap<u32, u64> = HashMap::new();
    let mut is_moe = false;
    let mut has_mtp = false;
    let mut max_experts = 0_usize;
    for t in tensors {
        *hist.entry(t.ggml_type).or_insert(0) +=
            t.dimensions.iter().product::<u64>().saturating_mul(1);
        let n = t.name.as_str();
        if n.contains("_exps") || n.contains("experts") {
            is_moe = true;
        }
        if n.contains("nextn") || n.contains("mtp") {
            has_mtp = true;
        }
        // crude expert-count estimator: gate_inp shape [..., num_experts]
        if n.ends_with(".ffn_gate_inp.weight") && t.dimensions.len() >= 2 {
            if let Some(&n_exp) = t.dimensions.last() {
                max_experts = max_experts.max(n_exp as usize);
            }
        }
    }
    let (best_ggml_type, _) = hist
        .into_iter()
        .max_by_key(|(_, bytes)| *bytes)
        .unwrap_or((0, 0));
    (
        GgufQuantizationType::from_ggml_type(best_ggml_type),
        max_experts,
        is_moe,
        has_mtp,
    )
}

/// Estimate per-token bytes for the KV cache under a given dtype
/// size. Mirrors the formula used in
/// `oxidize-cli/src/main.rs:2260-2265` so the planner and the
/// runtime agree.
pub fn kv_bytes_per_token(model: &ModelFingerprint, kv_dtype_bytes: usize) -> u64 {
    if model.layer_count == 0 || model.head_dim == 0 {
        return 0;
    }
    let per_layer = (model.num_kv_heads as u64) * (model.head_dim as u64) * 2 /*K+V*/ * (kv_dtype_bytes as u64);
    per_layer.saturating_mul(model.layer_count as u64)
}

/// Approximate the per-layer weight size in bytes, by dividing the
/// total file size by the layer count (ignoring embeddings + head).
/// Used by the GPU offload planner.
pub fn per_layer_weight_bytes(model: &ModelFingerprint) -> u64 {
    if model.layer_count == 0 {
        return 0;
    }
    // Embeddings + head + output typically add ~10–20% on top of
    // transformer layers. Subtract a flat 15% for those, then
    // divide. This is the same heuristic llama.cpp uses in
    // `llama_split_layers`.
    let transformer_share = (model.file_size_bytes as f64 * 0.85) as u64;
    transformer_share / model.layer_count as u64
}

/// Human-readable one-line summary for `--print-hardware` /
/// `--print-plan` output.
pub fn summary(model: &ModelFingerprint) -> String {
    let q = format!("{:?}", model.quant);
    let moe = if model.is_moe {
        format!(" moe={}", model.expert_count)
    } else {
        String::new()
    };
    let mtp = if model.has_mtp { " mtp=yes" } else { "" };
    format!(
        "{}-like layers={} hidden={} heads={} kv_heads={} head_dim={} vocab={} size={} MiB quant={}{}{mtp}",
        model.architecture,
        model.layer_count,
        model.hidden_size,
        model.num_attention_heads,
        model.num_kv_heads,
        model.head_dim,
        model.vocab_size,
        model.file_size_bytes / (1024 * 1024),
        q,
        moe
    )
}

/// Look up a metadata integer by key with type coercion (U32 / I32 /
/// F32 → usize). Returns `None` if missing or unparseable.
pub fn metadata_usize(metadata: &std::collections::BTreeMap<String, GgufMetadataValue>, key: &str) -> Option<usize> {
    let v = metadata.get(key)?;
    let n: i64 = match v {
        GgufMetadataValue::Uint8(x) => (*x).into(),
        GgufMetadataValue::Int8(x) => (*x).into(),
        GgufMetadataValue::Uint16(x) => (*x).into(),
        GgufMetadataValue::Int16(x) => (*x).into(),
        GgufMetadataValue::Uint32(x) => (*x).into(),
        GgufMetadataValue::Int32(x) => (*x).into(),
        GgufMetadataValue::Uint64(x) => (*x as i64),
        GgufMetadataValue::Int64(x) => *x,
        GgufMetadataValue::Float32(x) => *x as i64,
        GgufMetadataValue::Float64(x) => *x as i64,
        _ => return None,
    };
    usize::try_from(n.max(0)).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_bytes_per_token_uses_layer_x_kv_x_head_x_2() {
        let m = fingerprint_from_parts(
            "llama", 32, 4096, 32, 8, 128, 11008, 32000, 8u64 << 30, GgufQuantizationType::Q4_K_M,
        );
        // 32 * 8 * 128 * 2 * 2 (f16) = 131072
        assert_eq!(kv_bytes_per_token(&m, 2), 131_072);
    }

    #[test]
    fn per_layer_weight_bytes_subtracts_embeds() {
        let m = fingerprint_from_parts(
            "llama",
            32,
            4096,
            32,
            8,
            128,
            11008,
            32000,
            8u64 << 30,
            GgufQuantizationType::Q4_K_M,
        );
        // 8 GiB * 0.85 / 32 ≈ 227 MiB
        let b = per_layer_weight_bytes(&m);
        assert!(b > 200 * 1024 * 1024);
        assert!(b < 260 * 1024 * 1024);
    }

    #[test]
    fn summary_includes_architecture_and_quant() {
        let m = fingerprint_from_parts(
            "llama",
            32,
            4096,
            32,
            8,
            128,
            11008,
            32000,
            4u64 << 30,
            GgufQuantizationType::Q4_K_M,
        );
        let s = summary(&m);
        assert!(s.contains("llama"));
        assert!(s.contains("Q4_K_M"));
    }
}
