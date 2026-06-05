//! `ModelRuntime` — loaded model, tokenizer, and generation defaults.
//!
//! Wraps the different model implementations (`InferenceModel`, `LayerWiseModel`,
//! `MlxInferenceModel`) behind a single [`Model`] trait object so the rest of the
//! server can be backend-agnostic.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;

use oxidize_core::{
    dflash::{DFlashConfig, DFlashDraftModel},
    gguf::{GgufMetadataValue, MappedGgufFile},
    inference::{InferenceConfig, InferenceModel},
    layer_wise::LayerWiseModel,
    model::{Model, ModelError, Session, Token},
    model_loader::{GgufModelLoader, ModelLoader},
    tokenizer::{
        LoadedTokenizer, load_tokenizer_from_gguf_file, load_tokenizer_from_gguf_metadata,
    },
};

use crate::cli::Args;

// #region agent log
fn agent_debug_log_runtime(
    hypothesis_id: &str,
    location: &str,
    message: &str,
    data: serde_json::Value,
) {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0);
    let payload = serde_json::json!({
        "sessionId": "49b0b9",
        "runId": "initial",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": timestamp
    });
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/home/dih/oxidize/.cursor/debug-49b0b9.log")
    {
        use std::io::Write;
        let _ = writeln!(file, "{payload}");
    }
}
// #endregion

pub struct ModelRuntime {
    pub id: String,
    pub tokenizer: LoadedTokenizer,
    pub chat_template: Option<String>,
    pub model: StdMutex<LoadedModel>,
    pub defaults: GenerationDefaults,
}

#[derive(Debug, Clone, Copy)]
pub struct GenerationDefaults {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub prefill_batch_size: usize,
}

pub enum LoadedModel {
    Inference(Box<InferenceModel>),
    LayerWise(Box<LayerWiseModel>),
    DFlash(Box<DFlashDraftModel>),
    #[cfg(target_os = "macos")]
    Mlx(Box<oxidize_core::mlx_inference::MlxInferenceModel>),
    #[cfg(not(target_os = "macos"))]
    #[allow(dead_code)]
    Mlx(Box<InferenceModel>),
}

impl Model for LoadedModel {
    fn forward(&mut self, tokens: &[Token], session: &mut Session) -> Result<Vec<f32>, ModelError> {
        match self {
            Self::Inference(model) => model.forward(tokens, session),
            Self::LayerWise(model) => model.forward(tokens, session),
            Self::DFlash(model) => model.forward(tokens, session),
            #[cfg(target_os = "macos")]
            Self::Mlx(model) => model.forward(tokens, session),
            #[cfg(not(target_os = "macos"))]
            Self::Mlx(model) => model.forward(tokens, session),
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            Self::Inference(model) => model.vocab_size(),
            Self::LayerWise(model) => model.vocab_size(),
            Self::DFlash(model) => model.vocab_size(),
            #[cfg(target_os = "macos")]
            Self::Mlx(model) => model.vocab_size(),
            #[cfg(not(target_os = "macos"))]
            Self::Mlx(model) => model.vocab_size(),
        }
    }

    fn context_size(&self) -> usize {
        match self {
            Self::Inference(model) => model.context_size(),
            Self::LayerWise(model) => model.context_size(),
            Self::DFlash(model) => model.context_size(),
            #[cfg(target_os = "macos")]
            Self::Mlx(model) => model.context_size(),
            #[cfg(not(target_os = "macos"))]
            Self::Mlx(model) => model.context_size(),
        }
    }

    fn layer_count(&self) -> usize {
        match self {
            Self::Inference(model) => model.layer_count(),
            Self::LayerWise(model) => model.layer_count(),
            Self::DFlash(model) => model.layer_count(),
            #[cfg(target_os = "macos")]
            Self::Mlx(model) => model.layer_count(),
            #[cfg(not(target_os = "macos"))]
            Self::Mlx(model) => model.layer_count(),
        }
    }

    fn rewind_to(&mut self, consumed_tokens: usize) -> Result<(), ModelError> {
        match self {
            Self::Inference(model) => model.rewind_to(consumed_tokens),
            Self::LayerWise(model) => model.rewind_to(consumed_tokens),
            Self::DFlash(model) => model.rewind_to(consumed_tokens),
            #[cfg(target_os = "macos")]
            Self::Mlx(model) => model.rewind_to(consumed_tokens),
            #[cfg(not(target_os = "macos"))]
            Self::Mlx(model) => model.rewind_to(consumed_tokens),
        }
    }
}

pub fn load_model_runtime(args: &Args) -> Result<Option<Arc<ModelRuntime>>, String> {
    let Some(model_path) = args.model.as_ref() else {
        return Ok(None);
    };
    let (effective_backend, warning) = args.backend.to_core_backend().effective();
    if let Some(msg) = warning {
        tracing::warn!("{msg}");
    }
    let loader = GgufModelLoader;
    let mapped = loader
        .load_with_progress(model_path, |progress| {
            tracing::info!(
                stage = progress.stage,
                percent = progress.percent,
                "loading model"
            );
        })
        .map_err(|error| format!("failed to load model: {error:?}"))?;
    optimize_mapped_model_memory(&mapped, args);
    let metadata = &mapped.parsed().metadata;
    let is_dflash = matches!(
        mapped.parsed().architecture(),
        Some("dflash" | "dflash-draft")
    );
    // #region agent log
    let mapped_infos = mapped.mapped_tensor_infos();
    agent_debug_log_runtime(
        "H0_REPRO_PATH,H2_TENSOR_NAMES,H5_OUTPUT_PROJECTION",
        "oxidize-server/src/runtime/model.rs:load_model_runtime",
        "classified GGUF before server model construction",
        serde_json::json!({
            "architecture": mapped.parsed().architecture(),
            "is_dflash": is_dflash,
            "tensor_count": mapped_infos.len(),
            "has_lm_head": mapped_infos.iter().any(|tensor| tensor.name == "lm_head.weight"),
            "has_output": mapped_infos.iter().any(|tensor| tensor.name == "output.weight"),
            "has_embed_tokens": mapped_infos.iter().any(|tensor| tensor.name == "model.embed_tokens.weight"),
            "has_tok_embeddings": mapped_infos.iter().any(|tensor| tensor.name == "tok_embeddings.weight")
        }),
    );
    // #endregion
    if args.ctx_size == Some(0) {
        return Err("invalid --ctx-size: must be greater than 0".into());
    }

    // Try to load tokenizer from the model GGUF first, then fall back to an
    // external tokenizer model (e.g. the target model paired with a draft).
    let tokenizer = load_tokenizer_from_gguf_metadata(metadata)
        .or_else(|_| {
            load_tokenizer_from_gguf_file(args.tokenizer_model.as_deref()).and_then(|opt| {
                opt.ok_or_else(|| {
                    "external tokenizer model did not contain tokenizer metadata".to_string()
                })
            })
        })
        .map_err(|error| format!("failed to load tokenizer: {error:?}"))?;
    let chat_template = metadata
        .get("tokenizer.chat_template")
        .or_else(|| metadata.get("tokenizer.ggml.chat_template"))
        .and_then(|value| match value {
            GgufMetadataValue::String(template) => Some(template.clone()),
            _ => None,
        });

    let model = if is_dflash {
        let has_output_projection = mapped
            .mapped_tensor_infos()
            .iter()
            .any(|tensor| matches!(tensor.name.as_str(), "lm_head.weight" | "output.weight"));
        if !has_output_projection {
            return Err(
                "DFlash draft GGUF cannot be used as a standalone generation model: missing output projection/logits head. Use a full target model for the API server."
                    .to_string(),
            );
        }
        let dflash_config = DFlashConfig::from_gguf(&mapped);
        let dflash = DFlashDraftModel::load_from_gguf(&mapped, dflash_config)
            .map_err(|error| format!("failed to load DFlash model: {error}"))?;
        if !dflash.output.is_loaded() {
            return Err(
                "DFlash draft GGUF cannot be used as a standalone generation model: missing output projection/logits head. Use a full target model for the API server."
                    .to_string(),
            );
        }
        LoadedModel::DFlash(Box::new(dflash))
    } else if args.layer_wise {
        let mut config = inference_config_from_gguf(&mapped, args);
        if args.turboquant_kv {
            config.kv_quantization = oxidize_core::kv_cache::KvQuantization::TurboQuant;
        }
        LoadedModel::LayerWise(Box::new(
            LayerWiseModel::load_from_gguf(&mapped, config, args.layer_cache)
                .map_err(|error| format!("failed to load layer-wise model: {error}"))?,
        ))
    } else if effective_backend == oxidize_core::backend::Backend::Mlx {
        let mut config = inference_config_from_gguf(&mapped, args);
        if args.turboquant_kv {
            config.kv_quantization = oxidize_core::kv_cache::KvQuantization::TurboQuant;
        }
        #[cfg(target_os = "macos")]
        {
            match oxidize_core::mlx_inference::MlxInferenceModel::load_from_gguf(&mapped, config) {
                Ok(m) => {
                    tracing::info!("MLX backend: loaded model into unified memory");
                    LoadedModel::Mlx(Box::new(m))
                }
                Err(error) => {
                    tracing::warn!("MLX initialization failed: {error}; falling back to CPU");
                    LoadedModel::Inference(Box::new(
                        InferenceModel::load_from_gguf(&mapped, config, args.cpu_optimized)
                            .map_err(|error| format!("failed to load model weights: {error}"))?,
                    ))
                }
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            tracing::warn!("MLX backend requested but unavailable on Linux; falling back to CPU");
            LoadedModel::Inference(Box::new(
                InferenceModel::load_from_gguf(&mapped, config, args.cpu_optimized)
                    .map_err(|error| format!("failed to load model weights: {error}"))?,
            ))
        }
    } else {
        let mut config = inference_config_from_gguf(&mapped, args);
        if args.turboquant_kv {
            config.kv_quantization = oxidize_core::kv_cache::KvQuantization::TurboQuant;
        }
        LoadedModel::Inference(Box::new(
            InferenceModel::load_from_gguf(&mapped, config, args.cpu_optimized)
                .map_err(|error| format!("failed to load model weights: {error}"))?,
        ))
    };

    Ok(Some(Arc::new(ModelRuntime {
        id: args.model_id.clone(),
        tokenizer,
        chat_template,
        model: StdMutex::new(model),
        defaults: GenerationDefaults {
            max_tokens: args.max_tokens,
            temperature: args.temperature,
            top_p: args.top_p,
            top_k: args.top_k,
            prefill_batch_size: args.prefill_batch_size,
        },
    })))
}

fn optimize_mapped_model_memory(mapped: &MappedGgufFile, args: &Args) {
    let apply_hints =
        args.cpu_optimized || args.ram_offload || args.mmap_prefetch || args.mmap_hugepages;
    if !apply_hints {
        return;
    }

    if let Err(error) = mapped.advise_random_access() {
        tracing::warn!(%error, "mmap random-access hint failed");
    }
    if (args.cpu_optimized || args.ram_offload || args.mmap_prefetch)
        && let Err(error) = mapped.advise_will_need()
    {
        tracing::warn!(%error, "mmap prefetch hint failed");
    }
    if (args.cpu_optimized || args.mmap_hugepages)
        && let Err(error) = mapped.advise_huge_pages()
    {
        tracing::warn!(%error, "mmap hugepage hint failed");
    }
    if args.ram_offload {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        let (mlocked, checksum, elapsed_ms) = mapped.prefault_pages_locked(threads);
        tracing::info!(
            gib = mapped.bytes().len() as f64 / 1024.0 / 1024.0 / 1024.0,
            elapsed_ms,
            threads,
            mlocked,
            checksum,
            "ram offload prefaulted model pages in parallel"
        );
    }
}

fn inference_config_from_gguf(mapped: &MappedGgufFile, args: &Args) -> InferenceConfig {
    let mut config = InferenceConfig::from_gguf(mapped);
    if let Some(ctx) = args.ctx_size {
        config.context_size = ctx;
    }
    if args.cpu_optimized {
        config.context_size = config.context_size.min(2048);
    }
    config
}

#[allow(dead_code)]
pub fn metadata_u32(metadata: &BTreeMap<String, GgufMetadataValue>, key: &str) -> Option<u32> {
    match metadata.get(key) {
        Some(GgufMetadataValue::Uint8(value)) => Some((*value).into()),
        Some(GgufMetadataValue::Uint16(value)) => Some((*value).into()),
        Some(GgufMetadataValue::Uint32(value)) => Some(*value),
        Some(GgufMetadataValue::Uint64(value)) => (*value).try_into().ok(),
        Some(GgufMetadataValue::Int8(value)) if *value >= 0 => Some((*value as u8).into()),
        Some(GgufMetadataValue::Int16(value)) if *value >= 0 => Some((*value as u16).into()),
        Some(GgufMetadataValue::Int32(value)) if *value >= 0 => (*value).try_into().ok(),
        Some(GgufMetadataValue::Int64(value)) if *value >= 0 => (*value).try_into().ok(),
        _ => None,
    }
}

#[allow(dead_code)]
pub fn metadata_f32(metadata: &BTreeMap<String, GgufMetadataValue>, key: &str) -> Option<f32> {
    match metadata.get(key) {
        Some(GgufMetadataValue::Float32(value)) => Some(*value),
        Some(GgufMetadataValue::Float64(value)) => Some(*value as f32),
        Some(GgufMetadataValue::Int8(value)) => Some(*value as f32),
        Some(GgufMetadataValue::Int16(value)) => Some(*value as f32),
        Some(GgufMetadataValue::Int32(value)) => Some(*value as f32),
        Some(GgufMetadataValue::Int64(value)) => Some(*value as f32),
        Some(GgufMetadataValue::Uint8(value)) => Some(*value as f32),
        Some(GgufMetadataValue::Uint16(value)) => Some(*value as f32),
        Some(GgufMetadataValue::Uint32(value)) => Some(*value as f32),
        Some(GgufMetadataValue::Uint64(value)) => Some(*value as f32),
        _ => None,
    }
}

#[allow(dead_code)]
pub fn tensor_dims(mapped: &MappedGgufFile, name: &str) -> Option<Vec<u64>> {
    mapped
        .mapped_tensor_infos()
        .iter()
        .find(|tensor| tensor.name == name)
        .map(|tensor| tensor.dimensions.clone())
}

#[allow(dead_code)]
pub fn first_layer_tensor_dims(mapped: &MappedGgufFile, suffix: &str) -> Option<Vec<u64>> {
    mapped
        .mapped_tensor_infos()
        .iter()
        .find(|tensor| tensor.name.starts_with("blk.") && tensor.name.ends_with(suffix))
        .map(|tensor| tensor.dimensions.clone())
}
