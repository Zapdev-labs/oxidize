//! `ModelRuntime` — loaded model, tokenizer, and generation defaults.
//!
//! Wraps the different model implementations (`InferenceModel`, `LayerWiseModel`,
//! `MlxInferenceModel`) behind a single [`Model`] trait object so the rest of the
//! server can be backend-agnostic.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::time::Instant;

use oxidize_core::{
    gguf::{GgufMetadataValue, MappedGgufFile},
    inference::{InferenceConfig, InferenceModel},
    layer_wise::LayerWiseModel,
    model::{Model, ModelError, Session, Token},
    model_loader::{GgufModelLoader, ModelLoader},
    tokenizer::{LoadedTokenizer, load_tokenizer_from_gguf_metadata},
};

use crate::cli::Args;

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
    if args.ctx_size == Some(0) {
        return Err("invalid --ctx-size: must be greater than 0".into());
    }
    let mut config = inference_config_from_gguf(&mapped, args.ctx_size);
    if args.turboquant_kv {
        config.kv_quantization = oxidize_core::kv_cache::KvQuantization::TurboQuant;
    }
    let tokenizer = load_tokenizer_from_gguf_metadata(metadata)
        .map_err(|error| format!("failed to load tokenizer: {error:?}"))?;
    let chat_template = metadata
        .get("tokenizer.chat_template")
        .or_else(|| metadata.get("tokenizer.ggml.chat_template"))
        .and_then(|value| match value {
            GgufMetadataValue::String(template) => Some(template.clone()),
            _ => None,
        });
    let model = if args.layer_wise {
        LoadedModel::LayerWise(Box::new(
            LayerWiseModel::load_from_gguf(&mapped, config, args.layer_cache)
                .map_err(|error| format!("failed to load layer-wise model: {error}"))?,
        ))
    } else if effective_backend == oxidize_core::backend::Backend::Mlx {
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
        let started = Instant::now();
        let checksum = mapped.prefault_pages();
        tracing::info!(
            gib = mapped.bytes().len() as f64 / 1024.0 / 1024.0 / 1024.0,
            elapsed_ms = started.elapsed().as_millis(),
            checksum,
            "ram offload prefaulted model pages"
        );
    }
}

fn inference_config_from_gguf(mapped: &MappedGgufFile, ctx_size: Option<usize>) -> InferenceConfig {
    let mut config = InferenceConfig::from_gguf(mapped);
    if let Some(ctx) = ctx_size {
        config.context_size = ctx;
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
