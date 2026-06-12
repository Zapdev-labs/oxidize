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
    pub draft: Option<StdMutex<DFlashDraftModel>>,
    pub draft_tokens: usize,
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

    fn forward_many(
        &mut self,
        tokens: &[Token],
        session: &mut Session,
    ) -> Result<Vec<Vec<f32>>, ModelError> {
        match self {
            Self::Inference(model) => model.forward_many(tokens, session),
            Self::LayerWise(model) => model.forward_many(tokens, session),
            Self::DFlash(model) => model.forward_many(tokens, session),
            #[cfg(target_os = "macos")]
            Self::Mlx(model) => model.forward_many(tokens, session),
            #[cfg(not(target_os = "macos"))]
            Self::Mlx(model) => model.forward_many(tokens, session),
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
        })
        .or_else(|| {
            matches!(
                mapped.parsed().architecture(),
                Some("qwen" | "qwen2" | "qwen2moe" | "qwen35" | "qwen3" | "qwen3_5_moe")
            )
            .then(|| "<|im_start|>".to_owned())
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
        let mut layer_wise = LayerWiseModel::load_from_gguf(&mapped, config, args.layer_cache)
            .map_err(|error| format!("failed to load layer-wise model: {error}"))?;
        layer_wise
            .warm_layer_cache()
            .map_err(|error| format!("failed to warm layer cache: {error}"))?;
        LoadedModel::LayerWise(Box::new(layer_wise))
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

    let target_hidden_size = inference_config_from_gguf(&mapped, args).hidden_size;
    let target_layer_count = match &model {
        LoadedModel::Inference(m) => m.layer_count(),
        LoadedModel::LayerWise(m) => m.layer_count(),
        LoadedModel::DFlash(m) => m.layer_count(),
        #[cfg(target_os = "macos")]
        LoadedModel::Mlx(m) => m.layer_count(),
        #[cfg(not(target_os = "macos"))]
        LoadedModel::Mlx(m) => m.layer_count(),
    };
    let (draft, draft_tokens) = load_speculative_draft(
        args,
        &loader,
        &mapped,
        target_hidden_size,
        target_layer_count,
    )?;

    Ok(Some(Arc::new(ModelRuntime {
        id: args.model_id.clone(),
        tokenizer,
        chat_template,
        model: StdMutex::new(model),
        draft,
        draft_tokens,
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
        let threads = if args.ram_offload_threads > 0 {
            args.ram_offload_threads
        } else {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8)
        };
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
    config.kv_cache_dtype = args.kv_cache_dtype.dtype();
    if args.no_turboquant_kv {
        config.kv_quantization = oxidize_core::kv_cache::KvQuantization::Asymmetric;
    } else if args.turboquant_kv {
        config.kv_quantization = oxidize_core::kv_cache::KvQuantization::TurboQuant;
    }
    if let Some(ctx) = args.ctx_size {
        config.context_size = ctx;
    }
    if args.cpu_optimized {
        config.context_size = config.context_size.min(2048);
    }
    if args.ctx_size.is_none() && !args.cpu_optimized {
        let kv_bytes_per_token = config.layer_count
            * config.num_key_value_heads
            * config.kv_head_dim()
            * 2
            * config.kv_cache_dtype.size_in_bytes();
        let kv_full = (config.context_size as u64).saturating_mul(kv_bytes_per_token as u64);
        #[cfg(target_os = "linux")]
        let available = oxidize_core::gguf::linux_mem_available_bytes().unwrap_or(u64::MAX);
        #[cfg(not(target_os = "linux"))]
        let available = u64::MAX;
        let model_bytes = mapped.bytes().len() as u64;
        let overhead = 8u64 << 30;
        let kv_budget = available
            .saturating_sub(model_bytes)
            .saturating_sub(overhead);
        if kv_full > kv_budget && kv_bytes_per_token > 0 {
            let capped = ((kv_budget / kv_bytes_per_token as u64) as usize / 512).max(1) * 512;
            tracing::info!(
                from = config.context_size,
                to = capped,
                "context capped to fit KV cache in available RAM"
            );
            config.context_size = capped;
        }
    }
    config
}

fn load_speculative_draft(
    args: &Args,
    loader: &GgufModelLoader,
    target_mapped: &MappedGgufFile,
    target_hidden_size: usize,
    target_layer_count: usize,
) -> Result<(Option<StdMutex<DFlashDraftModel>>, usize), String> {
    let Some(draft_path) = args.draft_model.as_deref() else {
        return Ok((None, args.draft_tokens.max(1)));
    };

    let draft_mapped = loader.load(draft_path).map_err(|error| {
        format!(
            "failed to load DFlash draft model {}: {error:?}",
            draft_path.display()
        )
    })?;
    let draft_arch = draft_mapped.parsed().architecture();
    if !matches!(draft_arch, Some("dflash" | "dflash-draft")) {
        return Err(format!(
            "--draft-model must point to a DFlash GGUF, got architecture {draft_arch:?}"
        ));
    }

    let draft_config = DFlashConfig::from_gguf(&draft_mapped);
    let mut draft_model = DFlashDraftModel::load_from_gguf(&draft_mapped, draft_config)
        .map_err(|error| format!("failed to load DFlash draft model: {error}"))?;
    draft_model
        .load_external_io_from_gguf(target_mapped)
        .map_err(|error| format!("failed to borrow draft IO from target GGUF: {error}"))?;

    let incompatible_hidden = draft_model.config.hidden_size != target_hidden_size;
    let incompatible_layers = draft_model
        .config
        .target_layer_ids
        .iter()
        .any(|&layer| layer >= target_layer_count);
    if incompatible_hidden || incompatible_layers {
        return Err(format!(
            "DFlash draft is incompatible with target (draft_hidden={}, target_hidden={}, draft_target_layers={:?}, target_layers={})",
            draft_model.config.hidden_size,
            target_hidden_size,
            draft_model.config.target_layer_ids,
            target_layer_count
        ));
    }

    tracing::info!(
        draft = %draft_path.display(),
        draft_tokens = args.draft_tokens,
        "enabled DFlash speculative decoding for API server"
    );
    Ok((Some(StdMutex::new(draft_model)), args.draft_tokens.max(1)))
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
