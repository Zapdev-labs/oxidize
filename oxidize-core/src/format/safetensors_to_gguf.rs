use crate::conversion::map_hf_tensor_name;
use crate::gguf::{GgufMetadataArray, GgufMetadataType, GgufMetadataValue, GgufQuantizationType};
use crate::quantization::{quantize_scalar, quantized_size};
use anyhow::{Context, Result, anyhow, bail};
use safetensors::tensor::{Dtype, SafeTensors};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct SafetensorsToGgufConfig {
    pub arch_override: Option<String>,
    pub map_hf_tensor_names: bool,
    pub config_path: Option<PathBuf>,
}

impl Default for SafetensorsToGgufConfig {
    fn default() -> Self {
        Self {
            arch_override: None,
            map_hf_tensor_names: true,
            config_path: None,
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

/// Requantize every quantizable tensor in an existing GGUF to `target`.
///
/// Tensors that are already quantized (not F32/F16/BF16) or are 1-D
/// (embeddings/biases) are copied verbatim.  The returned bytes are a
/// valid GGUF v3 file ready to be written to disk.
pub fn quantize_gguf_to_target(
    input: &[u8],
    target: GgufQuantizationType,
) -> Result<Vec<u8>> {
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
                GgufQuantizationType::F32
                    | GgufQuantizationType::F16
                    | GgufQuantizationType::BF16
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
    let (tensors, st_meta, config_dir) = load_all_tensors(input)?;
    let arch = resolve_architecture(config, &st_meta, config_dir.as_deref(), input)?;

    let mut metadata = build_base_metadata(&st_meta, &arch, input);
    let auto_config = config_dir.as_ref().map(|d| d.join("config.json"));
    let cfg_path = config
        .config_path
        .as_ref()
        .or(auto_config.as_ref());
    if let Some(cfg_path) = cfg_path.filter(|p| p.is_file()) {
        merge_hf_config_metadata(&mut metadata, &arch, cfg_path)?;
    }

    // Embed tokenizer metadata so the converted GGUF is self-contained. HF
    // models ship the tokenizer separately (tokenizer.json + config), which the
    // GGUF tokenizer loader cannot read directly — without this the model loads
    // but fails with MissingMetadata("tokenizer.ggml.model").
    let tokenizer_dir = config_dir
        .clone()
        .or_else(|| cfg_path.and_then(|p| p.parent().map(Path::to_path_buf)));
    if let Some(dir) = tokenizer_dir {
        if let Err(error) = merge_hf_tokenizer_metadata(&mut metadata, &dir) {
            eprintln!(
                "warning: failed to embed tokenizer metadata from {}: {error:#}",
                dir.display()
            );
        }
    }

    let output_tensors = build_output_tensors(&tensors, config.map_hf_tensor_names)?;
    let gguf_bytes = write_gguf(3, &metadata, &output_tensors, 32)?;
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
        if cfg_path.is_file() {
            if let Ok(arch) = read_arch_from_hf_config(&cfg_path) {
                return Ok(arch);
            }
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
        "qwen3_5" | "qwen35" => "qwen35".to_owned(),
        "llama" | "mistral" | "gemma" | "phi" | "phi3" | "mixtral" => model_type.to_owned(),
        other => other.to_owned(),
    }
}

fn load_all_tensors(
    input: &Path,
) -> Result<(Vec<(String, Dtype, Vec<usize>, Vec<u8>)>, BTreeMap<String, String>, Option<PathBuf>)> {
    if input.is_file() {
        let (tensors, meta) = load_safetensors_file(input)?;
        return Ok((tensors, meta, None));
    }
    if !input.is_dir() {
        bail!("input path {} is neither a file nor a directory", input.display());
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
            let shard_name = shard_name.as_str().ok_or_else(|| {
                anyhow!("weight_map entry for {tensor_name} is not a string")
            })?;
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
                bail!("duplicate tensor {} in directory {}", tensor.0, input.display());
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

fn load_safetensors_file(
    path: &Path,
) -> Result<(Vec<(String, Dtype, Vec<usize>, Vec<u8>)>, BTreeMap<String, String>)> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    // SAFETY: read-only mapping; file handle kept alive for the mapping's lifetime.
    let mmap = unsafe { memmap2::Mmap::map(&file) }
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
        if matches!(key.as_str(), "model_type" | "architecture" | "model_name" | "name") {
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
    let insert_f32 = |meta: &mut BTreeMap<_, _>, key: &str, field: &str| {
        if let Some(v) = cfg.get(field).and_then(json_f32) {
            meta.insert(key.to_owned(), GgufMetadataValue::Float32(v));
        }
    };

    insert_u32(meta, &prefix("embedding_length"), "hidden_size");
    insert_u32(meta, &prefix("block_count"), "num_hidden_layers");
    insert_u32(meta, &prefix("feed_forward_length"), "intermediate_size");
    insert_u32(meta, &prefix("attention.head_count"), "num_attention_heads");
    insert_u32(meta, &prefix("attention.head_count_kv"), "num_key_value_heads");

    // Per-head dimension. Prefer an explicit `head_dim` field; otherwise derive
    // it from hidden_size / num_attention_heads. Writing key_length/value_length
    // lets the engine size the KV cache from metadata instead of inferring it
    // from tensor dimensions (which would otherwise mis-derive GQA head dims).
    let head_dim = cfg
        .get("head_dim")
        .and_then(json_u32)
        .or_else(|| {
            let hidden = cfg.get("hidden_size").and_then(json_u32)?;
            let heads = cfg.get("num_attention_heads").and_then(json_u32)?;
            (heads > 0).then(|| hidden / heads)
        });
    if let Some(head_dim) = head_dim {
        meta.insert(prefix("attention.key_length"), GgufMetadataValue::Uint32(head_dim));
        meta.insert(
            prefix("attention.value_length"),
            GgufMetadataValue::Uint32(head_dim),
        );
    }
    insert_u32(meta, &prefix("vocab_size"), "vocab_size");
    insert_u32(
        meta,
        &prefix("context_length"),
        "max_position_embeddings",
    );
    insert_f32(
        meta,
        &prefix("attention.layer_norm_rms_epsilon"),
        "rms_norm_eps",
    );
    insert_f32(meta, &prefix("rope.freq_base"), "rope_theta");
    insert_u32(meta, &prefix("attention.sliding_window"), "sliding_window");
    insert_u32(meta, &prefix("expert_count"), "num_experts");
    insert_u32(meta, &prefix("expert_used_count"), "num_experts_per_tok");

    if let Some(model_type) = cfg.get("model_type").and_then(|v| v.as_str()) {
        meta.insert(
            "general.architecture".to_owned(),
            GgufMetadataValue::String(normalize_hf_arch(model_type)),
        );
    }
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
    if model_type.eq_ignore_ascii_case("unigram") {
        if let Some(arr) = model.get("vocab").and_then(|v| v.as_array()) {
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
            if entry.get("special").and_then(|v| v.as_bool()).unwrap_or(false) {
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
    meta.insert(
        "tokenizer.ggml.tokens".to_owned(),
        string_array(tokens),
    );
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
    let mut set_id = |meta: &mut BTreeMap<_, _>, key: &str, json: &Value, field: &str| {
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
        let output_name = if map_hf_names {
            map_hf_tensor_name(name)
        } else {
            name.clone()
        };
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
        (GgufMetadataType::Array, GgufMetadataValue::Array(arr)) => {
            write_metadata_array(out, arr)?
        }
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
