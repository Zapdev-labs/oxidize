#![allow(clippy::type_complexity)]

use crate::conversion::{
    extract_layer_index, flatten_linear_attn_conv1d, map_flat_qwen_mtp_tensor_name,
    map_hf_tensor_name, preprocess_hf_tensors_for_gguf, split_fused_gate_up_proj,
};
use crate::gguf::{GgufMetadataArray, GgufMetadataType, GgufMetadataValue, GgufQuantizationType};
use crate::quantization::{quantize_scalar, quantized_size};
use anyhow::{Context, Result, anyhow, bail};
use safetensors::tensor::{Dtype, SafeTensors};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct SafetensorsToGgufConfig {
    pub arch_override: Option<String>,
    pub map_hf_tensor_names: bool,
    pub config_path: Option<PathBuf>,
    pub target_quantization: Option<GgufQuantizationType>,
}

impl Default for SafetensorsToGgufConfig {
    fn default() -> Self {
        Self {
            arch_override: None,
            map_hf_tensor_names: true,
            config_path: None,
            target_quantization: None,
        }
    }
}

#[derive(Debug)]
struct OutputTensor {
    name: String,
    dimensions: Vec<u64>,
    ggml_type: u32,
    data: Vec<u8>,
}

/// Read the causal backbone layer count from a HF config.json, looking in both
/// the root and `text_config` for `num_hidden_layers`.
fn mtp_base_layer_from_config(cfg_path: Option<&Path>) -> Option<usize> {
    let cfg_path = cfg_path?;
    let raw = std::fs::read_to_string(cfg_path).ok()?;
    let json: Value = serde_json::from_str(&raw).ok()?;
    let cfg = json
        .get("text_config")
        .filter(|v| v.is_object())
        .unwrap_or(&json);
    cfg.get("num_hidden_layers")?.as_u64().map(|n| n as usize)
}

/// Rewrite flat Qwen3.5/3.6 MTP tensor names (`mtp.fc.weight`, `mtp.layers.0.*`)
/// to oxidize's `blk.{base}.nextn.*` naming. The base layer is the number of
/// causal backbone layers (e.g. 32 for a 32-layer model), so the MTP block is
/// appended immediately after the main stack.
fn rewrite_flat_mtp_tensor_names(
    tensors: &mut [(String, Dtype, Vec<usize>, Vec<u8>)],
    base_layer: usize,
) {
    for (name, _, _, _) in tensors.iter_mut() {
        if let Some(mapped) = map_flat_qwen_mtp_tensor_name(name, base_layer) {
            *name = mapped;
        }
    }
}

/// Requantize every quantizable tensor in an existing GGUF to `target`.
///
/// Tensors that are already quantized (not F32/F16/BF16) or are 1-D
/// (embeddings/biases) are copied verbatim.  The returned bytes are a
/// valid GGUF v3 file ready to be written to disk.
pub fn quantize_gguf_to_target(input: &[u8], target: GgufQuantizationType) -> Result<Vec<u8>> {
    use crate::gguf::parse_gguf;

    let parsed = parse_gguf(input).map_err(|e| anyhow!("{e:?}"))?;
    let mut metadata = parsed.metadata.clone();

    // Map GgufQuantizationType → ggml_type ID used in file_type metadata.
    let file_type_id: u32 = match target {
        GgufQuantizationType::Q8_0 => 7,
        GgufQuantizationType::Q4_0 => 2,
        GgufQuantizationType::Q4_1 => 3,
        GgufQuantizationType::Q5_0 => 8,
        GgufQuantizationType::Q5_1 => 9,
        _ => u32::MAX,
    };
    if file_type_id != u32::MAX {
        metadata.insert(
            "general.file_type".to_owned(),
            GgufMetadataValue::Uint32(file_type_id),
        );
    }

    let mut tensors: Vec<OutputTensor> = Vec::with_capacity(parsed.tensor_infos.len());
    for info in &parsed.tensor_infos {
        let source = GgufQuantizationType::from_ggml_type(info.ggml_type);
        let value_count: usize = info.dimensions.iter().map(|&d| d as usize).product();

        let input_size = quantized_size(source, value_count).map_err(|e| anyhow!("{e:?}"))?;
        let start = info.absolute_offset as usize;
        let tensor_bytes = &input[start..start + input_size];

        let can_quantize = info.dimensions.len() >= 2
            && matches!(
                source,
                GgufQuantizationType::F32 | GgufQuantizationType::F16 | GgufQuantizationType::BF16
            )
            && quantized_size(target, value_count).is_ok();

        let (ggml_type, data) = if can_quantize {
            let out_size = quantized_size(target, value_count).map_err(|e| anyhow!("{e:?}"))?;
            let mut out = vec![0_u8; out_size];
            quantize_scalar(source, target, tensor_bytes, &mut out)
                .map_err(|e| anyhow!("quantize {}: {e:?}", info.name))?;
            let type_id: u32 = match target {
                GgufQuantizationType::F32 => 0,
                GgufQuantizationType::F16 => 1,
                GgufQuantizationType::Q4_0 => 2,
                GgufQuantizationType::Q4_1 => 3,
                GgufQuantizationType::Q5_0 => 6,
                GgufQuantizationType::Q5_1 => 7,
                GgufQuantizationType::Q8_0 => 8,
                GgufQuantizationType::Q2_K => 10,
                GgufQuantizationType::Q3_K_S => 11,
                GgufQuantizationType::Q3_K_M => 12,
                GgufQuantizationType::Q3_K_L => 13,
                GgufQuantizationType::Q4_K_S => 14,
                GgufQuantizationType::Q4_K_M => 15,
                GgufQuantizationType::Q5_K_S => 16,
                GgufQuantizationType::Q5_K_M => 17,
                GgufQuantizationType::Q6_K => 18,
                other => {
                    bail!("unsupported GGUF target type {other:?}")
                }
            };
            (type_id, out)
        } else {
            (info.ggml_type, tensor_bytes.to_vec())
        };

        tensors.push(OutputTensor {
            name: info.name.clone(),
            dimensions: info.dimensions.clone(),
            ggml_type,
            data,
        });
    }

    write_gguf(parsed.version, &metadata, &tensors, parsed.alignment)
}

/// Convert a single SafeTensors file or a HuggingFace model directory to GGUF v3.
pub fn convert_safetensors_to_gguf(
    input: &Path,
    output: &Path,
    config: &SafetensorsToGgufConfig,
) -> Result<usize> {
    if input.is_dir() && find_weight_index(input)?.is_some() {
        return convert_safetensors_dir_streaming(input, output, config);
    }

    let (tensors, st_meta, config_dir) = load_all_tensors(input)?;
    let mut tensors = preprocess_hf_tensors_for_gguf(tensors).map_err(|e| anyhow!(e))?;
    let arch = resolve_architecture(config, &st_meta, config_dir.as_deref(), input)?;

    let mut metadata = build_base_metadata(&st_meta, &arch, input);
    let auto_config = config_dir.as_ref().map(|d| d.join("config.json"));
    let cfg_path = config.config_path.as_ref().or(auto_config.as_ref());
    if let Some(cfg_path) = cfg_path.filter(|p| p.is_file()) {
        merge_hf_config_metadata(&mut metadata, &arch, cfg_path)?;
    }

    // Qwen3.5/3.6 MTP modules may be saved either as `model.layers.{L}.mtp.*`
    // (handled by `map_hf_tensor_name`) or as flat top-level `mtp.*` tensors.
    // For the flat form we need the backbone layer count to know where to place
    // the appended nextn block, so rewrite the names once the config is loaded.
    if let Some(base_layer) = mtp_base_layer_from_config(cfg_path.map(|p| p.as_path())) {
        rewrite_flat_mtp_tensor_names(&mut tensors, base_layer);
    }

    // Embed tokenizer metadata so the converted GGUF is self-contained. HF
    // models ship the tokenizer separately (tokenizer.json + config), which the
    // GGUF tokenizer loader cannot read directly — without this the model loads
    // but fails with MissingMetadata("tokenizer.ggml.model").
    let tokenizer_dir = config_dir
        .clone()
        .or_else(|| cfg_path.and_then(|p| p.parent().map(Path::to_path_buf)));
    if let Some(dir) = tokenizer_dir
        && let Err(error) = merge_hf_tokenizer_metadata(&mut metadata, &dir)
    {
        eprintln!(
            "warning: failed to embed tokenizer metadata from {}: {error:#}",
            dir.display()
        );
    }

    let output_tensors = build_output_tensors(&tensors, config.map_hf_tensor_names)?;
    let gguf_bytes = write_gguf(3, &metadata, &output_tensors, 32)?;
    // Apply target quantization on the single-file / non-index path too — only
    // the streaming directory path quantized before, so plain file conversions
    // silently emitted an unquantized GGUF.
    let gguf_bytes = match config.target_quantization {
        Some(target) => quantize_gguf_to_target(&gguf_bytes, target)?,
        None => gguf_bytes,
    };
    std::fs::write(output, &gguf_bytes)
        .with_context(|| format!("failed to write {}", output.display()))?;
    Ok(output_tensors.len())
}

fn resolve_architecture(
    config: &SafetensorsToGgufConfig,
    st_meta: &BTreeMap<String, String>,
    config_dir: Option<&Path>,
    input: &Path,
) -> Result<String> {
    if let Some(arch) = &config.arch_override {
        return Ok(arch.clone());
    }
    if let Some(dir) = config_dir {
        let cfg_path = dir.join("config.json");
        if cfg_path.is_file()
            && let Ok(arch) = read_arch_from_hf_config(&cfg_path)
        {
            return Ok(arch);
        }
    }
    Ok(st_meta
        .get("model_type")
        .or_else(|| st_meta.get("architecture"))
        .cloned()
        .unwrap_or_else(|| default_arch_from_path(input)))
}

fn default_arch_from_path(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_owned()
}

fn read_arch_from_hf_config(path: &Path) -> Result<String> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let json: Value = serde_json::from_str(&raw).context("invalid config.json")?;
    let arch = json
        .get("model_type")
        .or_else(|| json.pointer("/text_config/model_type"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_owned())
        .ok_or_else(|| anyhow!("config.json missing model_type"))?;
    Ok(normalize_hf_arch(&arch))
}

fn normalize_hf_arch(model_type: &str) -> String {
    match model_type.to_ascii_lowercase().as_str() {
        "qwen2" | "qwen2_moe" | "qwen2moe" => "qwen2".to_owned(),
        "qwen3" | "qwen3_moe" => "qwen3".to_owned(),
        "qwen3_5" | "qwen35" | "qwen3_5_moe" | "qwen3_5_moe_text" | "qwen35moe" => {
            "qwen35".to_owned()
        }
        "llama" | "mistral" | "gemma" | "phi" | "phi3" | "mixtral" => model_type.to_owned(),
        "hy_v3" | "hyv3" | "hunyuan_v3" | "hunyuan" | "hunyuan_moe" | "hunyuanmoe" => {
            "hunyuan-moe".to_owned()
        }
        other => other.to_owned(),
    }
}

fn load_all_tensors(
    input: &Path,
) -> Result<(
    Vec<(String, Dtype, Vec<usize>, Vec<u8>)>,
    BTreeMap<String, String>,
    Option<PathBuf>,
)> {
    if input.is_file() {
        let (tensors, meta) = load_safetensors_file(input)?;
        return Ok((tensors, meta, None));
    }
    if !input.is_dir() {
        bail!(
            "input path {} is neither a file nor a directory",
            input.display()
        );
    }

    let index_path = find_weight_index(input)?;
    let mut st_meta = BTreeMap::new();
    let mut out: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = Vec::new();

    if let Some(index_path) = index_path {
        let index_raw = std::fs::read_to_string(&index_path)
            .with_context(|| format!("failed to read {}", index_path.display()))?;
        let index: Value = serde_json::from_str(&index_raw).context("invalid weight index JSON")?;
        if let Some(meta) = index.get("metadata").and_then(|v| v.as_object()) {
            for (k, v) in meta {
                if let Some(s) = v.as_str() {
                    st_meta.insert(k.clone(), s.to_owned());
                }
            }
        }
        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| anyhow!("weight index missing weight_map"))?;

        let mut shard_cache: BTreeMap<String, Vec<(String, Dtype, Vec<usize>, Vec<u8>)>> =
            BTreeMap::new();
        for (tensor_name, shard_name) in weight_map {
            let shard_name = shard_name
                .as_str()
                .ok_or_else(|| anyhow!("weight_map entry for {tensor_name} is not a string"))?;
            let shard_path = input.join(shard_name);
            if !shard_cache.contains_key(shard_name) {
                let (tensors, meta) = load_safetensors_file(&shard_path)?;
                st_meta.extend(meta);
                shard_cache.insert(shard_name.to_owned(), tensors);
            }
            let shard_tensors = shard_cache.get(shard_name).unwrap();
            let found = shard_tensors
                .iter()
                .find(|(n, ..)| n == tensor_name)
                .cloned()
                .ok_or_else(|| {
                    anyhow!(
                        "tensor {tensor_name} not found in shard {}",
                        shard_path.display()
                    )
                })?;
            out.push(found);
        }
        return Ok((out, st_meta, Some(input.to_path_buf())));
    }

    let mut paths: Vec<PathBuf> = std::fs::read_dir(input)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .collect();
    paths.sort();
    if paths.is_empty() {
        bail!("no .safetensors files found in {}", input.display());
    }
    for path in paths {
        let (tensors, meta) = load_safetensors_file(&path)?;
        st_meta.extend(meta);
        for tensor in tensors {
            if out.iter().any(|(n, ..)| n == &tensor.0) {
                bail!(
                    "duplicate tensor {} in directory {}",
                    tensor.0,
                    input.display()
                );
            }
            out.push(tensor);
        }
    }
    Ok((out, st_meta, Some(input.to_path_buf())))
}

fn find_weight_index(dir: &Path) -> Result<Option<PathBuf>> {
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".safetensors.index.json"))
        })
        .collect();
    candidates.sort();
    Ok(candidates.into_iter().next())
}

fn load_safetensors_tensor_index(
    path: &Path,
) -> Result<(Vec<(String, Dtype, Vec<usize>)>, BTreeMap<String, String>)> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    // SAFETY: shard files are opened read-only and not modified while mapped.
    let mmap = unsafe { crate::bytes::map_readonly(&file) }
        .with_context(|| format!("failed to mmap {}", path.display()))?;
    let st = SafeTensors::deserialize(&mmap)
        .map_err(|e| anyhow!("failed to parse SafeTensors: {e:?}"))?;
    let meta = read_safetensors_metadata(&mmap)?;
    let mut tensors = Vec::with_capacity(st.len());
    for (name, view) in st.tensors() {
        tensors.push((name.to_owned(), view.dtype(), view.shape().to_vec()));
    }
    Ok((tensors, meta))
}

fn load_safetensors_file(
    path: &Path,
) -> Result<(
    Vec<(String, Dtype, Vec<usize>, Vec<u8>)>,
    BTreeMap<String, String>,
)> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    // SAFETY: read-only mapping; file handle kept alive for the mapping's lifetime.
    // SAFETY: shard files are opened read-only and not modified while mapped.
    let mmap = unsafe { crate::bytes::map_readonly(&file) }
        .with_context(|| format!("failed to mmap {}", path.display()))?;
    let st = SafeTensors::deserialize(&mmap)
        .map_err(|e| anyhow!("failed to parse SafeTensors: {e:?}"))?;
    let meta = read_safetensors_metadata(&mmap)?;
    let mut tensors = Vec::with_capacity(st.len());
    for (name, view) in st.tensors() {
        tensors.push((
            name.to_owned(),
            view.dtype(),
            view.shape().to_vec(),
            view.data().to_vec(),
        ));
    }
    Ok((tensors, meta))
}

fn read_safetensors_metadata(mmap: &[u8]) -> Result<BTreeMap<String, String>> {
    if mmap.len() < 8 {
        return Ok(BTreeMap::new());
    }
    let header_len = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
    if 8 + header_len > mmap.len() {
        bail!("safetensors header length exceeds file size");
    }
    let header_json: Value = serde_json::from_slice(&mmap[8..8 + header_len])
        .context("failed to parse safetensors header JSON")?;
    let Some(meta_obj) = header_json.get("__metadata__").and_then(|v| v.as_object()) else {
        return Ok(BTreeMap::new());
    };
    Ok(meta_obj
        .iter()
        .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_owned())))
        .collect())
}

fn build_base_metadata(
    st_meta: &BTreeMap<String, String>,
    arch: &str,
    input_path: &Path,
) -> BTreeMap<String, GgufMetadataValue> {
    let mut meta: BTreeMap<String, GgufMetadataValue> = BTreeMap::new();
    meta.insert(
        "general.architecture".to_owned(),
        GgufMetadataValue::String(arch.to_owned()),
    );

    if let Some(name) = st_meta
        .get("model_name")
        .or_else(|| st_meta.get("name"))
        .cloned()
        .or_else(|| {
            input_path
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.to_owned())
        })
    {
        meta.insert("general.name".to_owned(), GgufMetadataValue::String(name));
    }

    for (key, value) in st_meta {
        if matches!(
            key.as_str(),
            "model_type" | "architecture" | "model_name" | "name"
        ) {
            continue;
        }
        meta.insert(
            format!("general.safetensors.{key}"),
            GgufMetadataValue::String(value.clone()),
        );
    }
    meta
}

fn merge_hf_config_metadata(
    meta: &mut BTreeMap<String, GgufMetadataValue>,
    arch: &str,
    config_path: &Path,
) -> Result<()> {
    let raw = std::fs::read_to_string(config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let json: Value = serde_json::from_str(&raw).context("invalid config.json")?;
    let cfg = json
        .get("text_config")
        .filter(|v| v.is_object())
        .unwrap_or(&json);

    let prefix = |suffix: &str| format!("{arch}.{suffix}");
    let insert_u32 = |meta: &mut BTreeMap<_, _>, key: &str, field: &str| {
        if let Some(v) = cfg.get(field).and_then(json_u32) {
            meta.insert(key.to_owned(), GgufMetadataValue::Uint32(v));
        }
    };
    let insert_f32 = |meta: &mut BTreeMap<_, _>, key: &str, field: &str| -> bool {
        if let Some(v) = cfg.get(field).and_then(json_f32) {
            meta.insert(key.to_owned(), GgufMetadataValue::Float32(v));
            true
        } else {
            false
        }
    };

    insert_u32(meta, &prefix("embedding_length"), "hidden_size");
    let block_count = cfg.get("num_hidden_layers").and_then(json_u32);
    let nextn_layers = cfg.get("mtp_num_hidden_layers").and_then(json_u32);
    // Qwen3.5/3.6-style in-model multi-token prediction (MTP/nextn) layers are
    // appended after the main transformer stack. Oxidize's loader treats
    // `block_count` as the total number of `blk.*` layers (causal backbone +
    // nextn) and subtracts `nextn_predict_layers` to obtain the backbone count.
    // HF configs store these counts separately, so add them together.
    if let Some(block_count) = block_count {
        let total = if let Some(nextn) = nextn_layers {
            block_count + nextn
        } else {
            block_count
        };
        meta.insert(prefix("block_count"), GgufMetadataValue::Uint32(total));
    }
    if let Some(nextn) = nextn_layers {
        meta.insert(
            prefix("nextn_predict_layers"),
            GgufMetadataValue::Uint32(nextn),
        );
    }
    insert_u32(meta, &prefix("feed_forward_length"), "intermediate_size");
    insert_u32(meta, &prefix("attention.head_count"), "num_attention_heads");
    insert_u32(
        meta,
        &prefix("attention.head_count_kv"),
        "num_key_value_heads",
    );

    // Per-head dimension. Prefer an explicit `head_dim` field; otherwise derive
    // it from hidden_size / num_attention_heads. Writing key_length/value_length
    // lets the engine size the KV cache from metadata instead of inferring it
    // from tensor dimensions (which would otherwise mis-derive GQA head dims).
    let head_dim = cfg.get("head_dim").and_then(json_u32).or_else(|| {
        let hidden = cfg.get("hidden_size").and_then(json_u32)?;
        let heads = cfg.get("num_attention_heads").and_then(json_u32)?;
        (heads > 0).then(|| hidden / heads)
    });
    if let Some(head_dim) = head_dim {
        meta.insert(
            prefix("attention.key_length"),
            GgufMetadataValue::Uint32(head_dim),
        );
        meta.insert(
            prefix("attention.value_length"),
            GgufMetadataValue::Uint32(head_dim),
        );
    }
    insert_u32(meta, &prefix("vocab_size"), "vocab_size");
    insert_u32(meta, &prefix("context_length"), "max_position_embeddings");
    insert_f32(
        meta,
        &prefix("attention.layer_norm_rms_epsilon"),
        "rms_norm_eps",
    );
    if !insert_f32(meta, &prefix("rope.freq_base"), "rope_theta")
        && let Some(rp) = cfg.get("rope_parameters").and_then(|v| v.as_object())
        && let Some(theta) = rp.get("rope_theta").and_then(json_f32)
    {
        meta.insert(
            prefix("rope.freq_base").to_owned(),
            GgufMetadataValue::Float32(theta),
        );
    }
    insert_u32(meta, &prefix("attention.sliding_window"), "sliding_window");
    insert_u32(meta, &prefix("expert_count"), "num_experts");
    insert_u32(meta, &prefix("expert_used_count"), "num_experts_per_tok");
    insert_u32(
        meta,
        &prefix("expert_feed_forward_length"),
        "moe_intermediate_size",
    );
    // Shared (always-on) experts + leading dense blocks (Hunyuan/DeepSeek MoE).
    insert_u32(meta, &prefix("expert_shared_count"), "num_shared_experts");
    insert_u32(
        meta,
        &prefix("leading_dense_block_count"),
        "first_k_dense_replace",
    );
    // Routed-expert output scale (Hunyuan `router_scaling_factor`,
    // DeepSeek `routed_scaling_factor`).
    if !insert_f32(meta, &prefix("expert_weights_scale"), "router_scaling_factor") {
        insert_f32(
            meta,
            &prefix("expert_weights_scale"),
            "routed_scaling_factor",
        );
    }
    // Sigmoid routing (expert_gating_func: 1 = softmax, 2 = sigmoid). Hunyuan
    // sets `moe_router_use_sigmoid`; DeepSeek/LFM2MoE use `scoring_func`.
    let sigmoid_router = cfg
        .get("moe_router_use_sigmoid")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
        || cfg
            .get("scoring_func")
            .and_then(|v| v.as_str())
            .map(|s| s.eq_ignore_ascii_case("sigmoid"))
            .unwrap_or(false);
    if sigmoid_router {
        meta.insert(prefix("expert_gating_func"), GgufMetadataValue::Uint32(2));
    }

    // general.architecture MUST match the metadata key prefix (`arch`),
    // otherwise the loader builds keys like `qwen3_5_text.attention.head_count`
    // that don't exist and silently falls back to defaults. Use the already
    // resolved `arch` rather than re-deriving from a (possibly `_text`-suffixed
    // multimodal) model_type.
    meta.insert(
        "general.architecture".to_owned(),
        GgufMetadataValue::String(arch.to_owned()),
    );
    Ok(())
}

fn json_u32(v: &Value) -> Option<u32> {
    v.as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .or_else(|| v.as_i64().and_then(|n| u32::try_from(n).ok()))
}

fn json_f32(v: &Value) -> Option<f32> {
    v.as_f64().map(|n| n as f32)
}

/// Parse the HuggingFace tokenizer files in `dir` and embed the equivalent
/// `tokenizer.ggml.*` metadata into the GGUF. Supports BPE (gpt2-style, e.g.
/// Qwen/Llama-3) and Unigram (SentencePiece, e.g. Llama/Gemma) tokenizers.
fn merge_hf_tokenizer_metadata(
    meta: &mut BTreeMap<String, GgufMetadataValue>,
    dir: &Path,
) -> Result<()> {
    let tok_path = dir.join("tokenizer.json");
    if !tok_path.is_file() {
        bail!(
            "tokenizer.json not found in {} (HF repo may omit it)",
            dir.display()
        );
    }
    let raw = std::fs::read_to_string(&tok_path)
        .with_context(|| format!("failed to read {}", tok_path.display()))?;
    let tok: Value = serde_json::from_str(&raw).context("invalid tokenizer.json")?;
    let model = tok
        .get("model")
        .ok_or_else(|| anyhow!("tokenizer.json missing model section"))?;
    let model_type = model.get("type").and_then(|v| v.as_str()).unwrap_or("");

    // vocab: token -> id. Determine the highest id so the GGUF token arrays are
    // dense (id == array index), which the loader relies on.
    let vocab = model
        .get("vocab")
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow!("tokenizer.json model.vocab missing or not an object"))?;
    let mut max_id = 0u64;
    for v in vocab.values() {
        if let Some(id) = v.as_u64() {
            max_id = max_id.max(id);
        }
    }

    // added_tokens (special markers like <|im_start|>) may sit above the base
    // vocab and define the real upper bound.
    let added = tok.get("added_tokens").and_then(|v| v.as_array());
    if let Some(added) = added {
        for entry in added {
            if let Some(id) = entry.get("id").and_then(|v| v.as_u64()) {
                max_id = max_id.max(id);
            }
        }
    }

    let len = (max_id + 1) as usize;
    let mut tokens: Vec<String> = vec![String::new(); len];
    // 1 = NORMAL, 3 = CONTROL (matches ggml token_type values).
    let mut token_types: Vec<i32> = vec![1; len];
    let mut scores: Vec<f32> = vec![0.0; len];

    for (token, id_val) in vocab {
        if let Some(id) = id_val.as_u64() {
            tokens[id as usize] = token.clone();
        }
    }
    // Unigram stores [token, score] pairs instead of a flat map.
    if model_type.eq_ignore_ascii_case("unigram")
        && let Some(arr) = model.get("vocab").and_then(|v| v.as_array())
    {
        for (id, pair) in arr.iter().enumerate() {
            if id >= len {
                break;
            }
            if let Some(p) = pair.as_array() {
                if let Some(t) = p.first().and_then(|v| v.as_str()) {
                    tokens[id] = t.to_owned();
                }
                if let Some(s) = p.get(1).and_then(|v| v.as_f64()) {
                    scores[id] = s as f32;
                }
            }
        }
    }
    if let Some(added) = added {
        for entry in added {
            let Some(id) = entry.get("id").and_then(|v| v.as_u64()) else {
                continue;
            };
            let id = id as usize;
            if id >= len {
                continue;
            }
            if let Some(content) = entry.get("content").and_then(|v| v.as_str()) {
                tokens[id] = content.to_owned();
            }
            if entry
                .get("special")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                token_types[id] = 3;
            }
        }
    }

    let ggml_model = match model_type.to_ascii_lowercase().as_str() {
        "bpe" => "gpt2",
        "unigram" => "llama",
        "wordpiece" => "bert",
        other => bail!("unsupported tokenizer.json model.type {other:?}"),
    };
    meta.insert(
        "tokenizer.ggml.model".to_owned(),
        GgufMetadataValue::String(ggml_model.to_owned()),
    );
    meta.insert("tokenizer.ggml.tokens".to_owned(), string_array(tokens));
    meta.insert(
        "tokenizer.ggml.token_type".to_owned(),
        i32_array(token_types),
    );
    if ggml_model == "llama" {
        meta.insert("tokenizer.ggml.scores".to_owned(), f32_array(scores));
    }

    if ggml_model == "gpt2" {
        let merges = parse_merges(model.get("merges"))?;
        meta.insert("tokenizer.ggml.merges".to_owned(), string_array(merges));
    }

    // Special token ids and chat template come from config.json /
    // tokenizer_config.json. tokenizer_config.json wins when both are present.
    apply_special_token_ids(meta, dir);

    Ok(())
}

/// Merges are stored either as `"a b"` strings (older tokenizers) or as
/// `["a", "b"]` pairs (tokenizers >= 0.20). Normalize both to `"a b"`.
fn parse_merges(merges: Option<&Value>) -> Result<Vec<String>> {
    let Some(Value::Array(arr)) = merges else {
        return Ok(Vec::new());
    };
    let mut out = Vec::with_capacity(arr.len());
    for entry in arr {
        match entry {
            Value::String(s) => out.push(s.clone()),
            Value::Array(pair) => {
                let left = pair.first().and_then(|v| v.as_str());
                let right = pair.get(1).and_then(|v| v.as_str());
                if let (Some(l), Some(r)) = (left, right) {
                    out.push(format!("{l} {r}"));
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

fn apply_special_token_ids(meta: &mut BTreeMap<String, GgufMetadataValue>, dir: &Path) {
    let set_id = |meta: &mut BTreeMap<_, _>, key: &str, json: &Value, field: &str| {
        if let Some(id) = json.get(field).and_then(json_u32) {
            meta.insert(key.to_owned(), GgufMetadataValue::Uint32(id));
        }
    };

    for name in ["config.json", "tokenizer_config.json"] {
        let path = dir.join(name);
        let Ok(raw) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(json) = serde_json::from_str::<Value>(&raw) else {
            continue;
        };
        set_id(meta, "tokenizer.ggml.bos_token_id", &json, "bos_token_id");
        set_id(meta, "tokenizer.ggml.eos_token_id", &json, "eos_token_id");
        set_id(
            meta,
            "tokenizer.ggml.padding_token_id",
            &json,
            "pad_token_id",
        );
        set_id(
            meta,
            "tokenizer.ggml.unknown_token_id",
            &json,
            "unk_token_id",
        );
        if let Some(tmpl) = json.get("chat_template").and_then(|v| v.as_str()) {
            meta.insert(
                "tokenizer.chat_template".to_owned(),
                GgufMetadataValue::String(tmpl.to_owned()),
            );
        }
    }
}

fn string_array(values: Vec<String>) -> GgufMetadataValue {
    GgufMetadataValue::Array(GgufMetadataArray {
        element_type: GgufMetadataType::String,
        values: values.into_iter().map(GgufMetadataValue::String).collect(),
    })
}

fn i32_array(values: Vec<i32>) -> GgufMetadataValue {
    GgufMetadataValue::Array(GgufMetadataArray {
        element_type: GgufMetadataType::Int32,
        values: values.into_iter().map(GgufMetadataValue::Int32).collect(),
    })
}

fn f32_array(values: Vec<f32>) -> GgufMetadataValue {
    GgufMetadataValue::Array(GgufMetadataArray {
        element_type: GgufMetadataType::Float32,
        values: values.into_iter().map(GgufMetadataValue::Float32).collect(),
    })
}

fn build_output_tensors(
    tensors: &[(String, Dtype, Vec<usize>, Vec<u8>)],
    map_hf_names: bool,
) -> Result<Vec<OutputTensor>> {
    let mut out: Vec<OutputTensor> = Vec::with_capacity(tensors.len());
    for (name, dtype, shape, raw_data) in tensors {
        let output_name = if name.starts_with("blk.")
            || name == "tok_embeddings.weight"
            || name == "output.weight"
            || name == "norm.weight"
        {
            name.clone()
        } else if map_hf_names {
            map_hf_tensor_name(name)
        } else {
            name.clone()
        };
        if output_name.is_empty() {
            continue;
        }
        let dimensions: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
        let (ggml_type, data) = match dtype {
            Dtype::F32 => (0_u32, raw_data.clone()),
            Dtype::F16 => (1_u32, raw_data.clone()),
            // ggml does not have a distinct unsigned byte tensor type. For
            // byte-packed quantized payloads (for example NF4 stored as U8 in
            // SafeTensors), preserve the bytes losslessly and mark the tensor
            // as ggml I8. Consumers that understand the model-specific
            // quantization metadata can reinterpret the payload as unsigned
            // bytes without changing the file contents.
            Dtype::U8 | Dtype::I8 => (24_u32, raw_data.clone()),
            Dtype::I16 => (25_u32, raw_data.clone()),
            Dtype::I32 => (26_u32, raw_data.clone()),
            Dtype::I64 => (27_u32, raw_data.clone()),
            // GGUF/ggml supports BF16 tensors directly (ggml_type 30). Keep
            // BF16 payloads packed instead of expanding them to F32, which is
            // both lossless and avoids doubling large text-encoder weights.
            Dtype::BF16 => (30_u32, raw_data.clone()),
            other => bail!("unsupported SafeTensors dtype {other:?} in tensor {name}"),
        };
        out.push(OutputTensor {
            name: output_name,
            dimensions,
            ggml_type,
            data,
        });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

#[derive(Debug, Clone, Copy)]
enum StreamTransform {
    Passthrough,
    SplitGateUpGate,
    SplitGateUpUp,
    FlattenConv1d,
}

#[derive(Debug, Clone)]
struct PlannedTensor {
    name: String,
    dimensions: Vec<u64>,
    ggml_type: u32,
    source_name: String,
    source_shard: PathBuf,
    transform: StreamTransform,
}

fn dtype_to_ggml_type(dtype: Dtype) -> Result<u32> {
    Ok(match dtype {
        Dtype::F32 => 0,
        Dtype::F16 => 1,
        Dtype::U8 | Dtype::I8 => 24,
        Dtype::I16 => 25,
        Dtype::I32 => 26,
        Dtype::I64 => 27,
        Dtype::BF16 => 30,
        other => bail!("unsupported SafeTensors dtype {other:?}"),
    })
}

fn tensor_byte_len(ggml_type: u32, dimensions: &[u64]) -> Result<usize> {
    let count: u64 = dimensions.iter().product();
    let count = usize::try_from(count).map_err(|_| anyhow!("tensor element count overflow"))?;
    let elem = match ggml_type {
        0 => 4,
        1 | 30 => 2,
        24 => 1, // I8 / U8
        25 => 2, // I16
        26 => 4,
        27 => 8,
        other => bail!("unsupported ggml tensor type {other}"),
    };
    count
        .checked_mul(elem)
        .ok_or_else(|| anyhow!("tensor byte length overflow"))
}

fn plan_stream_outputs(
    name: &str,
    dtype: Dtype,
    shape: &[usize],
    shard_path: &Path,
    map_hf_names: bool,
    mtp_base_layer: Option<usize>,
) -> Result<Vec<PlannedTensor>> {
    if name.starts_with("model.visual.") {
        return Ok(Vec::new());
    }

    let ggml_type = dtype_to_ggml_type(dtype)?;
    let shard = shard_path.to_path_buf();
    let source_name = name.to_owned();

    if name.ends_with(".mlp.experts.gate_up_proj") {
        let Some(layer) = extract_layer_index(name) else {
            return Ok(Vec::new());
        };
        if shape.len() != 3 || !shape[1].is_multiple_of(2) {
            bail!("invalid gate_up_proj shape for {name}: {shape:?}");
        }
        let experts = shape[0];
        let half = shape[1] / 2;
        let hidden = shape[2];
        return Ok(vec![
            PlannedTensor {
                name: format!("blk.{layer}.ffn_gate_exps.weight"),
                dimensions: vec![experts as u64, half as u64, hidden as u64],
                ggml_type,
                source_name: source_name.clone(),
                source_shard: shard.clone(),
                transform: StreamTransform::SplitGateUpGate,
            },
            PlannedTensor {
                name: format!("blk.{layer}.ffn_up_exps.weight"),
                dimensions: vec![experts as u64, half as u64, hidden as u64],
                ggml_type,
                source_name,
                source_shard: shard,
                transform: StreamTransform::SplitGateUpUp,
            },
        ]);
    }

    if name.ends_with(".linear_attn.conv1d.weight") {
        let Some(layer) = extract_layer_index(name) else {
            return Ok(Vec::new());
        };
        if shape.len() != 3 || shape[1] != 1 {
            bail!("invalid conv1d shape for {name}: {shape:?}");
        }
        let channels = shape[0];
        let kernel = shape[2];
        return Ok(vec![PlannedTensor {
            name: format!("blk.{layer}.ssm_conv1d.weight"),
            dimensions: vec![(kernel * channels) as u64],
            ggml_type,
            source_name,
            source_shard: shard,
            transform: StreamTransform::FlattenConv1d,
        }]);
    }

    let output_name = if name.starts_with("blk.")
        || name == "tok_embeddings.weight"
        || name == "output.weight"
        || name == "norm.weight"
    {
        name.to_owned()
    } else if let Some(base) = mtp_base_layer {
        // Flat Qwen3.5/3.6 MTP tensors (`mtp.fc.weight`, `mtp.layers.0.*`) need
        // the backbone layer count to be placed correctly.
        map_flat_qwen_mtp_tensor_name(name, base)
            .or_else(|| {
                if map_hf_names {
                    Some(map_hf_tensor_name(name))
                } else {
                    None
                }
            })
            .filter(|n| !n.is_empty())
            .unwrap_or_else(|| name.to_owned())
    } else if map_hf_names {
        map_hf_tensor_name(name)
    } else {
        name.to_owned()
    };
    if output_name.is_empty() {
        return Ok(Vec::new());
    }

    Ok(vec![PlannedTensor {
        name: output_name,
        dimensions: shape.iter().map(|&d| d as u64).collect(),
        ggml_type,
        source_name,
        source_shard: shard,
        transform: StreamTransform::Passthrough,
    }])
}

fn read_tensor_from_shard(
    shard_path: &Path,
    tensor_name: &str,
) -> Result<(Dtype, Vec<usize>, Vec<u8>)> {
    let file = File::open(shard_path)
        .with_context(|| format!("failed to open {}", shard_path.display()))?;
    // SAFETY: shard files are opened read-only and not modified while mapped.
    let mmap = unsafe { crate::bytes::map_readonly(&file) }
        .with_context(|| format!("failed to mmap {}", shard_path.display()))?;
    let st = SafeTensors::deserialize(&mmap)
        .map_err(|e| anyhow!("failed to parse SafeTensors: {e:?}"))?;
    let view = st.tensor(tensor_name).map_err(|e| {
        anyhow!(
            "tensor {tensor_name} missing in {}: {e:?}",
            shard_path.display()
        )
    })?;
    Ok((view.dtype(), view.shape().to_vec(), view.data().to_vec()))
}

fn materialize_planned_tensor(plan: &PlannedTensor) -> Result<Vec<u8>> {
    let (dtype, shape, raw) = read_tensor_from_shard(&plan.source_shard, &plan.source_name)?;
    match plan.transform {
        StreamTransform::Passthrough => Ok(raw),
        StreamTransform::SplitGateUpGate | StreamTransform::SplitGateUpUp => {
            let Some(layer) = extract_layer_index(&plan.source_name) else {
                bail!("missing layer index for {}", plan.source_name);
            };
            let split = split_fused_gate_up_proj(layer, dtype, &shape, &raw)
                .ok_or_else(|| anyhow!("failed to split gate_up_proj {}", plan.source_name))?;
            let idx = match plan.transform {
                StreamTransform::SplitGateUpGate => 0,
                StreamTransform::SplitGateUpUp => 1,
                _ => unreachable!(),
            };
            Ok(split[idx].3.clone())
        }
        StreamTransform::FlattenConv1d => {
            let Some(layer) = extract_layer_index(&plan.source_name) else {
                bail!("missing layer index for {}", plan.source_name);
            };
            let (_, _, _, flat) = flatten_linear_attn_conv1d(layer, dtype, &shape, &raw)
                .ok_or_else(|| anyhow!("failed to flatten conv1d {}", plan.source_name))?;
            Ok(flat)
        }
    }
}

fn convert_safetensors_dir_streaming(
    input: &Path,
    output: &Path,
    config: &SafetensorsToGgufConfig,
) -> Result<usize> {
    let index_path = find_weight_index(input)?
        .ok_or_else(|| anyhow!("missing safetensors index in {}", input.display()))?;
    let index_raw = std::fs::read_to_string(&index_path)?;
    let index: Value = serde_json::from_str(&index_raw).context("invalid weight index JSON")?;

    let mut st_meta = BTreeMap::new();
    if let Some(meta) = index.get("metadata").and_then(|v| v.as_object()) {
        for (k, v) in meta {
            if let Some(s) = v.as_str() {
                st_meta.insert(k.clone(), s.to_owned());
            }
        }
    }

    let weight_map = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow!("weight index missing weight_map"))?;

    let mut shard_meta_cache: BTreeMap<String, Vec<(String, Dtype, Vec<usize>)>> = BTreeMap::new();
    let mut planned: Vec<PlannedTensor> = Vec::new();
    let auto_config = input.join("config.json");
    let cfg_path = config.config_path.as_ref().unwrap_or(&auto_config);
    let mtp_base_layer = mtp_base_layer_from_config(Some(cfg_path));

    for (tensor_name, shard_name_val) in weight_map {
        let shard_name = shard_name_val
            .as_str()
            .ok_or_else(|| anyhow!("weight_map entry for {tensor_name} is not a string"))?;
        let shard_path = input.join(shard_name);
        if !shard_meta_cache.contains_key(shard_name) {
            let (tensor_index, meta) = load_safetensors_tensor_index(&shard_path)?;
            st_meta.extend(meta);
            shard_meta_cache.insert(shard_name.to_owned(), tensor_index);
        }
        let shard_tensors = shard_meta_cache.get(shard_name).unwrap();
        let Some((dtype, shape)) = shard_tensors
            .iter()
            .find(|(n, ..)| n == tensor_name)
            .map(|(_, d, s)| (*d, s.clone()))
        else {
            bail!(
                "tensor {tensor_name} not found in shard {}",
                shard_path.display()
            );
        };
        planned.extend(plan_stream_outputs(
            tensor_name,
            dtype,
            &shape,
            &shard_path,
            config.map_hf_tensor_names,
            mtp_base_layer,
        )?);
    }

    planned.sort_by(|a, b| a.name.cmp(&b.name));
    eprintln!(
        "streaming convert: {} HF tensors -> {} GGUF tensors",
        weight_map.len(),
        planned.len()
    );

    let arch = resolve_architecture(config, &st_meta, Some(input), input)?;
    let mut metadata = build_base_metadata(&st_meta, &arch, input);
    if cfg_path.is_file() {
        merge_hf_config_metadata(&mut metadata, &arch, cfg_path)?;
    }
    if let Err(error) = merge_hf_tokenizer_metadata(&mut metadata, input) {
        eprintln!(
            "warning: failed to embed tokenizer metadata from {}: {error:#}",
            input.display()
        );
    }

    if let Some(target) = config.target_quantization
        && let Some(file_type) = gguf_file_type_id(target)
    {
        metadata.insert(
            "general.file_type".to_owned(),
            GgufMetadataValue::Uint32(file_type),
        );
    }

    write_gguf_streaming(
        output,
        3,
        &metadata,
        &planned,
        32,
        config.target_quantization,
    )?;
    Ok(planned.len())
}

fn gguf_file_type_id(target: GgufQuantizationType) -> Option<u32> {
    match target {
        GgufQuantizationType::Q8_0 => Some(7),
        GgufQuantizationType::Q4_0 => Some(2),
        GgufQuantizationType::Q4_1 => Some(3),
        GgufQuantizationType::Q4_K_M => Some(15),
        GgufQuantizationType::Q4_K_S => Some(14),
        GgufQuantizationType::Q6_K => Some(18),
        _ => None,
    }
}

fn ggml_type_id(target: GgufQuantizationType) -> Result<u32> {
    Ok(match target {
        GgufQuantizationType::F32 => 0,
        GgufQuantizationType::F16 => 1,
        GgufQuantizationType::Q4_0 => 2,
        GgufQuantizationType::Q4_1 => 3,
        GgufQuantizationType::Q5_0 => 6,
        GgufQuantizationType::Q5_1 => 7,
        GgufQuantizationType::Q8_0 => 8,
        GgufQuantizationType::Q2_K => 10,
        GgufQuantizationType::Q3_K_S => 11,
        GgufQuantizationType::Q3_K_M => 12,
        GgufQuantizationType::Q3_K_L => 13,
        GgufQuantizationType::Q4_K_S => 14,
        GgufQuantizationType::Q4_K_M => 15,
        GgufQuantizationType::Q5_K_S => 16,
        GgufQuantizationType::Q5_K_M => 17,
        GgufQuantizationType::Q6_K => 18,
        other => bail!("unsupported GGUF target type {other:?}"),
    })
}

fn planned_data_len(plan: &PlannedTensor, target: Option<GgufQuantizationType>) -> Result<usize> {
    let raw = tensor_byte_len(plan.ggml_type, &plan.dimensions)?;
    if plan.dimensions.len() < 2 {
        return Ok(raw);
    }
    let Some(target) = target else {
        return Ok(raw);
    };
    if !matches!(plan.ggml_type, 0 | 1 | 30) {
        return Ok(raw);
    }
    let count: usize = plan
        .dimensions
        .iter()
        .map(|d| usize::try_from(*d).unwrap_or(0))
        .product();
    quantized_size(target, count).map_err(|e| anyhow!("{e:?}"))
}

fn maybe_quantize_tensor_data(
    target: Option<GgufQuantizationType>,
    ggml_type: u32,
    dimensions: &[u64],
    data: Vec<u8>,
) -> Result<(u32, Vec<u8>)> {
    if dimensions.len() < 2 {
        return Ok((ggml_type, data));
    }
    let Some(target) = target else {
        return Ok((ggml_type, data));
    };
    if !matches!(ggml_type, 0 | 1 | 30) {
        return Ok((ggml_type, data));
    }
    let source = GgufQuantizationType::from_ggml_type(ggml_type);
    let count: usize = dimensions
        .iter()
        .map(|d| usize::try_from(*d).unwrap_or(0))
        .product();
    let out_size = quantized_size(target, count).map_err(|e| anyhow!("{e:?}"))?;
    let mut out = vec![0_u8; out_size];
    quantize_scalar(source, target, &data, &mut out).map_err(|e| anyhow!("{e:?}"))?;
    Ok((ggml_type_id(target)?, out))
}

fn write_gguf_streaming(
    path: &Path,
    version: u32,
    metadata: &BTreeMap<String, GgufMetadataValue>,
    planned: &[PlannedTensor],
    alignment: u64,
    target: Option<GgufQuantizationType>,
) -> Result<()> {
    if alignment == 0 || !alignment.is_power_of_two() {
        bail!("invalid GGUF alignment: {alignment}");
    }

    let mut data_lens = Vec::with_capacity(planned.len());
    let mut output_types = Vec::with_capacity(planned.len());
    for plan in planned {
        data_lens.push(planned_data_len(plan, target)?);
        output_types.push(
            if let Some(t) = target
                && plan.dimensions.len() >= 2
                && matches!(plan.ggml_type, 0 | 1 | 30)
            {
                ggml_type_id(t)?
            } else {
                plan.ggml_type
            },
        );
    }

    let mut relative_offsets = Vec::with_capacity(planned.len());
    let mut cursor: u64 = 0;
    for &len in &data_lens {
        cursor = align_up(cursor, alignment)?;
        relative_offsets.push(cursor);
        cursor = cursor
            .checked_add(len as u64)
            .ok_or_else(|| anyhow!("tensor data offset overflow"))?;
    }

    let mut header = Vec::new();
    header.extend_from_slice(b"GGUF");
    header.extend_from_slice(&version.to_le_bytes());
    header.extend_from_slice(&(planned.len() as u64).to_le_bytes());
    header.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (key, value) in metadata {
        write_string(&mut header, key);
        write_metadata_value(&mut header, value)?;
    }
    for (plan, (&rel_offset, &out_type)) in planned
        .iter()
        .zip(relative_offsets.iter().zip(output_types.iter()))
    {
        write_string(&mut header, &plan.name);
        header.extend_from_slice(&(plan.dimensions.len() as u32).to_le_bytes());
        for dim in &plan.dimensions {
            header.extend_from_slice(&dim.to_le_bytes());
        }
        header.extend_from_slice(&out_type.to_le_bytes());
        header.extend_from_slice(&rel_offset.to_le_bytes());
    }
    pad_to(&mut header, alignment)?;
    let data_start = header.len() as u64;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut out = BufWriter::new(file);
    out.write_all(&header)?;

    for (idx, plan) in planned.iter().enumerate() {
        if idx % 25 == 0 {
            eprintln!(
                "writing tensor {}/{}: {}",
                idx + 1,
                planned.len(),
                plan.name
            );
        }
        let file_offset = data_start + relative_offsets[idx];
        out.seek(SeekFrom::Start(file_offset))?;
        let raw = materialize_planned_tensor(plan)?;
        let (_ggml_type, data) =
            maybe_quantize_tensor_data(target, plan.ggml_type, &plan.dimensions, raw)?;
        if data.len() != data_lens[idx] {
            bail!(
                "tensor {} byte length mismatch: expected {}, got {}",
                plan.name,
                data_lens[idx],
                data.len()
            );
        }
        out.write_all(&data)?;
        let aligned_end = align_up(file_offset + data.len() as u64, alignment)? as u64;
        let pad_len = aligned_end.saturating_sub(file_offset + data.len() as u64);
        if pad_len > 0 {
            out.write_all(&vec![0_u8; pad_len as usize])?;
        }
    }
    out.flush()?;
    Ok(())
}

fn write_gguf(
    version: u32,
    metadata: &BTreeMap<String, GgufMetadataValue>,
    tensors: &[OutputTensor],
    alignment: u64,
) -> Result<Vec<u8>> {
    if alignment == 0 || !alignment.is_power_of_two() {
        bail!("invalid GGUF alignment: {alignment}");
    }

    let mut relative_offsets: Vec<u64> = Vec::with_capacity(tensors.len());
    let mut cursor: u64 = 0;
    for tensor in tensors {
        cursor = align_up(cursor, alignment)?;
        relative_offsets.push(cursor);
        cursor = cursor
            .checked_add(tensor.data.len() as u64)
            .ok_or_else(|| anyhow!("tensor data offset overflow"))?;
    }

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

    for (key, value) in metadata {
        write_string(&mut out, key);
        write_metadata_value(&mut out, value)?;
    }

    for (tensor, &rel_offset) in tensors.iter().zip(relative_offsets.iter()) {
        write_string(&mut out, &tensor.name);
        out.extend_from_slice(&(tensor.dimensions.len() as u32).to_le_bytes());
        for dim in &tensor.dimensions {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        out.extend_from_slice(&tensor.ggml_type.to_le_bytes());
        out.extend_from_slice(&rel_offset.to_le_bytes());
    }

    pad_to(&mut out, alignment)?;
    let data_start = out.len() as u64;

    for (tensor, &rel_offset) in tensors.iter().zip(relative_offsets.iter()) {
        let target = (data_start + rel_offset) as usize;
        if out.len() < target {
            out.resize(target, 0);
        }
        out.extend_from_slice(&tensor.data);
        pad_to(&mut out, alignment)?;
    }

    Ok(out)
}

fn write_metadata_value(out: &mut Vec<u8>, value: &GgufMetadataValue) -> Result<()> {
    let vtype = metadata_type(value);
    out.extend_from_slice(&(vtype as u32).to_le_bytes());
    write_metadata_payload(out, value, vtype)
}

fn write_metadata_payload(
    out: &mut Vec<u8>,
    value: &GgufMetadataValue,
    vtype: GgufMetadataType,
) -> Result<()> {
    match (vtype, value) {
        (GgufMetadataType::Uint8, GgufMetadataValue::Uint8(v)) => out.push(*v),
        (GgufMetadataType::Int8, GgufMetadataValue::Int8(v)) => out.push(*v as u8),
        (GgufMetadataType::Uint16, GgufMetadataValue::Uint16(v)) => {
            out.extend_from_slice(&v.to_le_bytes())
        }
        (GgufMetadataType::Int16, GgufMetadataValue::Int16(v)) => {
            out.extend_from_slice(&v.to_le_bytes())
        }
        (GgufMetadataType::Uint32, GgufMetadataValue::Uint32(v)) => {
            out.extend_from_slice(&v.to_le_bytes())
        }
        (GgufMetadataType::Int32, GgufMetadataValue::Int32(v)) => {
            out.extend_from_slice(&v.to_le_bytes())
        }
        (GgufMetadataType::Float32, GgufMetadataValue::Float32(v)) => {
            out.extend_from_slice(&v.to_le_bytes())
        }
        (GgufMetadataType::Bool, GgufMetadataValue::Bool(v)) => out.push(u8::from(*v)),
        (GgufMetadataType::String, GgufMetadataValue::String(v)) => write_string(out, v),
        (GgufMetadataType::Array, GgufMetadataValue::Array(arr)) => write_metadata_array(out, arr)?,
        (GgufMetadataType::Uint64, GgufMetadataValue::Uint64(v)) => {
            out.extend_from_slice(&v.to_le_bytes())
        }
        (GgufMetadataType::Int64, GgufMetadataValue::Int64(v)) => {
            out.extend_from_slice(&v.to_le_bytes())
        }
        (GgufMetadataType::Float64, GgufMetadataValue::Float64(v)) => {
            out.extend_from_slice(&v.to_le_bytes())
        }
        _ => bail!("metadata type mismatch"),
    }
    Ok(())
}

fn write_metadata_array(out: &mut Vec<u8>, arr: &GgufMetadataArray) -> Result<()> {
    out.extend_from_slice(&(arr.element_type as u32).to_le_bytes());
    out.extend_from_slice(&(arr.values.len() as u64).to_le_bytes());
    for value in &arr.values {
        write_metadata_payload(out, value, arr.element_type)?;
    }
    Ok(())
}

fn metadata_type(value: &GgufMetadataValue) -> GgufMetadataType {
    match value {
        GgufMetadataValue::Uint8(_) => GgufMetadataType::Uint8,
        GgufMetadataValue::Int8(_) => GgufMetadataType::Int8,
        GgufMetadataValue::Uint16(_) => GgufMetadataType::Uint16,
        GgufMetadataValue::Int16(_) => GgufMetadataType::Int16,
        GgufMetadataValue::Uint32(_) => GgufMetadataType::Uint32,
        GgufMetadataValue::Int32(_) => GgufMetadataType::Int32,
        GgufMetadataValue::Float32(_) => GgufMetadataType::Float32,
        GgufMetadataValue::Bool(_) => GgufMetadataType::Bool,
        GgufMetadataValue::String(_) => GgufMetadataType::String,
        GgufMetadataValue::Array(_) => GgufMetadataType::Array,
        GgufMetadataValue::Uint64(_) => GgufMetadataType::Uint64,
        GgufMetadataValue::Int64(_) => GgufMetadataType::Int64,
        GgufMetadataValue::Float64(_) => GgufMetadataType::Float64,
    }
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn pad_to(out: &mut Vec<u8>, alignment: u64) -> Result<()> {
    let aligned = align_up(out.len() as u64, alignment)? as usize;
    out.resize(aligned, 0);
    Ok(())
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|v| v & !mask)
        .ok_or_else(|| anyhow!("alignment overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::parse_gguf;
    use safetensors::tensor::{Dtype, TensorView};
    use std::collections::HashMap;
    use std::io::Write;

    fn write_test_safetensors(path: &Path, tensors: &[(&str, Dtype, Vec<usize>, Vec<u8>)]) {
        let views: HashMap<String, TensorView> = tensors
            .iter()
            .map(|(name, dtype, shape, data)| {
                (
                    name.to_string(),
                    TensorView::new(*dtype, shape.clone(), data).unwrap(),
                )
            })
            .collect();
        let bytes = safetensors::tensor::serialize(&views, &None).unwrap();
        let mut f = File::create(path).unwrap();
        f.write_all(&bytes).unwrap();
    }

    fn tmp_path(suffix: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("oxidize-st2gguf-{nanos}{suffix}"))
    }

    #[test]
    fn maps_hf_tensor_names() {
        let input = tmp_path(".safetensors");
        let output = tmp_path(".gguf");
        let data = vec![0u8; 64];
        write_test_safetensors(
            &input,
            &[(
                "model.layers.0.self_attn.q_proj.weight",
                Dtype::F32,
                vec![4, 4],
                data,
            )],
        );
        convert_safetensors_to_gguf(
            &input,
            &output,
            &SafetensorsToGgufConfig {
                arch_override: Some("llama".into()),
                ..Default::default()
            },
        )
        .unwrap();
        let parsed = parse_gguf(&std::fs::read(&output).unwrap()).unwrap();
        assert_eq!(parsed.tensor_infos[0].name, "blk.0.attn_q.weight");
        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
    }

    #[test]
    fn reads_config_json_metadata() {
        let dir = std::env::temp_dir().join(format!(
            "oxidize-st2gguf-cfg-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("model.safetensors");
        let output = tmp_path(".gguf");
        write_test_safetensors(&input, &[("weight", Dtype::F32, vec![2, 2], vec![0u8; 16])]);
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"llama","hidden_size":128,"num_hidden_layers":2,"num_attention_heads":4,"vocab_size":256}"#,
        )
        .unwrap();
        convert_safetensors_to_gguf(&dir, &output, &SafetensorsToGgufConfig::default()).unwrap();
        let parsed = parse_gguf(&std::fs::read(&output).unwrap()).unwrap();
        assert_eq!(
            parsed.metadata.get("llama.embedding_length"),
            Some(&GgufMetadataValue::Uint32(128))
        );
        assert_eq!(
            parsed.metadata.get("llama.block_count"),
            Some(&GgufMetadataValue::Uint32(2))
        );
        let _ = std::fs::remove_dir_all(dir);
        let _ = std::fs::remove_file(output);
    }
}
