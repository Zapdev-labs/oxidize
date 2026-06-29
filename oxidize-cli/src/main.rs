mod backend;
mod help;
mod pipeline;

use backend::Backend;
use clap::{Parser, ValueEnum};
use help::{print_model_list, print_ollama_help, print_run_help, print_serve_help};
use oxidize_core::generation::{
    Eagle3GenerationStream, GenerationConfig, GenerationStream, MtpGenerationStream,
    SpeculativeGenerationConfig, SpeculativeGenerationStream,
};
use oxidize_core::gguf::MappedGgufFile;
use oxidize_core::inference::{InferenceConfig, InferenceModel};
use oxidize_core::lora::{AdapterKind, LoraPlan, plan_lora_application};
use oxidize_core::model::{Model, Session};
use oxidize_core::model_loader::{GgufModelLoader, LoadProgress, ModelLoader};
use oxidize_core::offload::{
    LayerOffloadPlan, MultiGpuConfig, MultiGpuOffloadPlan, ParallelismStrategy, plan_layer_offload,
    plan_multi_gpu_offload,
};
use oxidize_core::safetensors_to_gguf::{SafetensorsToGgufConfig, convert_safetensors_to_gguf};
use oxidize_core::sampling::SamplingConfig;
use oxidize_core::tensor::DType;
use oxidize_core::tokenizer::{
    EncodeOptions, LoadedTokenizer, TiktokenTokenizer, load_tokenizer_from_gguf_metadata,
};
use serde::Deserialize;

use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::io::{self, BufRead, IsTerminal, Write};
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::sync::Arc;
use std::task::Wake;
use std::time::{Duration, Instant};

const PROFILE_CHILD_ENV: &str = "OXIDIZE_PROFILE_CHILD";

// Submodules split out of the original monolithic main.rs (mechanical move).
// `main.rs` is the binary crate root, so its child modules would otherwise
// resolve to `src/<name>.rs`; `#[path]` keeps them under `src/main/`.
#[path = "main/command_rewrite.rs"]
mod command_rewrite;
#[path = "main/conversation.rs"]
mod conversation;
#[path = "main/generation.rs"]
mod generation;
#[path = "main/gpu_cluster.rs"]
mod gpu_cluster;
#[path = "main/inference.rs"]
mod inference;
#[path = "main/model_resolution.rs"]
mod model_resolution;
#[path = "main/rendering.rs"]
mod rendering;
#[path = "main/server.rs"]
mod server;

use command_rewrite::*;
use conversation::*;
use generation::*;
use gpu_cluster::*;
use inference::*;
use model_resolution::*;
use rendering::*;
use server::*;

#[derive(Debug, Parser)]
#[command(name = "oxidize")]
struct Args {
    #[arg(long, default_value = "hello")]
    prompt: String,
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = Backend::Cpu)]
    backend: Backend,
    #[arg(long, default_value_t = 0)]
    n_gpu_layers: usize,
    #[arg(long, default_value_t = 1)]
    gpus: usize,
    #[arg(long, default_value = "pipeline")]
    parallelism: String,
    #[arg(long = "lora")]
    lora_paths: Vec<PathBuf>,
    #[arg(long, default_value_t = false)]
    chat: bool,
    #[arg(long, value_enum)]
    profile: Option<Profiler>,
    #[arg(long)]
    profile_output: Option<PathBuf>,
    #[arg(long, default_value_t = 512)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,
    #[arg(long)]
    top_p: Option<f32>,
    #[arg(long)]
    top_k: Option<usize>,
    #[arg(long, default_value_t = false)]
    layer_wise: bool,
    #[arg(long, default_value_t = 1)]
    layer_cache: usize,
    /// Use TurboQuant block quantization for q4/q8 KV cache (default).
    #[arg(long, default_value_t = false)]
    turboquant: bool,
    /// Use the legacy asymmetric q4/q8 KV cache quantizer instead of TurboQuant.
    #[arg(long, default_value_t = false)]
    no_turboquant: bool,
    #[arg(long, default_value_t = false)]
    cpu_optimized: bool,
    #[arg(long, default_value_t = false)]
    ram_offload: bool,
    /// Number of threads for parallel RAM prefault (0 = auto = logical CPUs).
    #[arg(long, default_value_t = 0)]
    ram_offload_threads: usize,
    #[arg(long, default_value_t = false)]
    mmap_prefetch: bool,
    #[arg(long, default_value_t = false)]
    mmap_hugepages: bool,
    #[arg(long)]
    ctx_size: Option<usize>,
    #[arg(long)]
    threads: Option<usize>,
    #[arg(long, value_enum, default_value_t = KvCacheDType::F32)]
    kv_cache_dtype: KvCacheDType,
    /// Start a distributed mesh node instead of loading a model locally.
    #[arg(long, default_value_t = false)]
    mesh: bool,
    /// Port for libp2p mesh listener (0 = ephemeral). Only used with --mesh.
    #[arg(long, default_value_t = 0)]
    mesh_port: u16,
    /// Run as pipeline head (stage 0): tokenize prompt, run first half of
    /// layers, ship hidden state to --pipe-peer, print tail-sampled tokens.
    #[arg(long, default_value_t = false)]
    pipe_head: bool,
    /// Run as pipeline tail (last stage): listen on --pipe-listen, run second
    /// half of layers + lm_head, send sampled tokens back.
    #[arg(long, default_value_t = false)]
    pipe_tail: bool,
    /// TCP address of the next pipeline stage (head connects here).
    #[arg(long)]
    pipe_peer: Option<String>,
    /// TCP address to listen on for the previous pipeline stage (tail binds).
    #[arg(long)]
    pipe_listen: Option<String>,
    /// Maximum tokens to generate in pipeline mode.
    #[arg(long, default_value_t = 64)]
    pipe_max_tokens: usize,
    #[arg(long, hide = true, default_value_t = false)]
    serve_api: bool,
    /// Skip starting the OpenAI-compatible API/WebSocket server during `oxidize run`.
    #[arg(long, default_value_t = false)]
    no_api: bool,
    #[arg(long, hide = true, default_value_t = false)]
    api_only: bool,
    #[arg(long, hide = true, default_value = "127.0.0.1")]
    api_host: String,
    #[arg(long, hide = true, default_value_t = 8080)]
    api_port: u16,
    /// External GGUF file that contains the tokenizer metadata.
    /// Useful for draft models (e.g. DFlash) that do not embed a tokenizer.
    #[arg(long)]
    tokenizer_model: Option<PathBuf>,
    /// Enable vision/multimodal mode for image understanding.
    #[arg(long, default_value_t = false)]
    vision: bool,
    /// Path to image file for multimodal inference.
    #[arg(long)]
    image: Option<PathBuf>,
    /// Path to DFlash draft model for speculative decoding.
    #[arg(long)]
    draft_model: Option<PathBuf>,
    /// Number of draft tokens per speculative step.
    #[arg(long, default_value_t = 4)]
    draft_tokens: usize,
    /// Force DFlash speculative decoding even when the draft was trained for a different target.
    /// Output remains target-verified, but draft acceptance may be poor.
    #[arg(long, default_value_t = false)]
    force_dflash: bool,
    /// Disable native in-GGUF MTP/nextn speculative decoding when present.
    #[arg(long, default_value_t = false)]
    no_mtp: bool,
    /// Self-speculative QuantSpec decoding (hierarchical I8 TurboQuant MTP draft KV).
    #[arg(long, default_value_t = false)]
    quantspec: bool,
    /// Auto-detect hardware and pick inference knobs (threads, ctx,
    /// KV dtype, n_gpu_layers, layer_wise, mmap, mlock, ISA, pipeline).
    /// On by default for `run`; explicit flags always win.
    #[arg(long, default_value_t = true)]
    auto: bool,
    /// Opt out of auto-tuning (revert to explicit-flag-only behavior).
    #[arg(long, default_value_t = false)]
    no_auto: bool,
    /// Print the resolved autotune plan to stderr before generation
    /// starts. "json" emits machine-readable JSON instead of text.
    #[arg(long, default_value = "auto")]
    print_plan: String,
    /// Internal: set if the user passed `--n-gpu-layers`. Used by
    /// the autotuner to avoid overriding an explicit value.
    #[arg(skip)]
    n_gpu_layers_set: bool,
    /// Internal: set if the user passed `--kv-cache-dtype`.
    #[arg(skip)]
    kv_cache_dtype_set: bool,
}

/// True if `argv` contains `--flag` (exact match) or
/// `--flag=value` (prefix match). Used by the autotuner to detect
/// which non-Option flags the user set on the command line.
fn user_passed_flag(argv: &[String], flag: &str) -> bool {
    argv.iter()
        .any(|a| a == flag || a.starts_with(&format!("{flag}=")))
}

fn greeting(prompt: &str) -> String {
    format!("oxidize-cli: {prompt}")
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum Profiler {
    Perf,
    Samply,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum KvCacheDType {
    F32,
    F16,
    Q8,
    Q4,
}

impl KvCacheDType {
    fn dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
            Self::Q8 => DType::I8,
            Self::Q4 => DType::I16,
        }
    }
}

fn main() {
    let rewritten_args = match rewrite_run_args(std::env::args_os()) {
        Ok(args) => args,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };
    let args = match Args::try_parse_from(rewritten_args) {
        Ok(args) => args,
        Err(error) => error.exit(),
    };

    // Detect which non-Option flags the user explicitly set, so the
    // autotuner can avoid overriding them.
    let n_gpu_layers_set =
        user_passed_flag(&std::env::args().collect::<Vec<_>>(), "--n-gpu-layers");
    let kv_cache_dtype_set =
        user_passed_flag(&std::env::args().collect::<Vec<_>>(), "--kv-cache-dtype");
    let mut args = Args {
        n_gpu_layers_set,
        kv_cache_dtype_set,
        ..args
    };
    let (effective_backend, warning) = args.backend.to_core_backend().effective();
    if let Some(msg) = warning {
        eprintln!("warning: {msg}");
    }
    let backend_label = match effective_backend {
        oxidize_core::backend::Backend::Mlx => "Apple Silicon",
        oxidize_core::backend::Backend::Metal => "Metal GPU",
        oxidize_core::backend::Backend::Cuda => "CUDA GPU",
        oxidize_core::backend::Backend::Rocm => "ROCm GPU",
        oxidize_core::backend::Backend::Cpu => "CPU",
        oxidize_core::backend::Backend::Vulkan => "Vulkan GPU",
        oxidize_core::backend::Backend::IntelArc => "Intel Arc GPU (Vulkan)",
    };
    println!(
        "backend: {} ({})",
        effective_backend.as_str(),
        backend_label
    );
    // Build the global rayon pool with one worker per physical core. Decode
    // GEMV is DRAM-bound, so SMT siblings add contention, not throughput (16
    // logical threads on an 8-core part measures slower than 8). Pin each
    // worker to one CPU in core-first order; otherwise the scheduler migrates
    // workers between cores (and NUMA nodes) mid-token, turning local DRAM
    // streams into remote ones and defeating the prefetcher. Disable pinning
    // with OXIDIZE_NO_PIN=1.
    //
    // The pool can only be built once and must be built before any rayon use.
    // When `--auto` will tune an actual model it can lower the thread count
    // (e.g. for GPU offload), so for that path we defer the build until after
    // the plan is applied — building it here would pin the wrong thread count
    // permanently. Model loading itself does not touch the global pool.
    fn build_rayon_pool(threads: usize) -> Result<(), rayon::ThreadPoolBuildError> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .start_handler(oxidize_core::spinpool::pin_to_slot)
            .build_global()
    }
    fn resolve_threads(args: &Args) -> usize {
        args.threads
            .filter(|t| *t > 0)
            .unwrap_or_else(oxidize_core::spinpool::physical_core_count)
    }
    let defer_pool_for_autotune = args.auto
        && !args.no_auto
        && args.model.is_some()
        && args.threads.filter(|t| *t > 0).is_none()
        && args.profile.is_none()
        && !args.api_only
        && !args.pipe_head
        && !args.pipe_tail
        && !args.mesh;
    if !defer_pool_for_autotune {
        if let Err(error) = build_rayon_pool(resolve_threads(&args)) {
            eprintln!("failed to set rayon thread pool: {error}");
            return;
        }
    }
    if let Some(profiler) = args.profile
        && !is_profiling_child()
    {
        match run_profiled_inference(profiler, args.profile_output.as_ref()) {
            Ok(status) => std::process::exit(status.code().unwrap_or(1)),
            Err(error) => {
                eprintln!("failed to run profiler: {error}");
                std::process::exit(1);
            }
        }
    }
    if args.serve_api && args.api_only {
        if let Err(error) = run_api_server_in_process(&args) {
            eprintln!("server failed: {error}");
            std::process::exit(1);
        }
        return;
    }
    if args.serve_api
        && !args.no_api
        && let Err(error) = spawn_api_server_background(&args)
    {
        eprintln!("failed to start API server: {error}");
    }
    if args.pipe_head {
        let model = match args.model.as_ref() {
            Some(m) => m.clone(),
            None => {
                eprintln!("--pipe-head requires --model PATH");
                std::process::exit(1);
            }
        };
        let peer = match args.pipe_peer.as_ref() {
            Some(p) => p.clone(),
            None => {
                eprintln!("--pipe-head requires --pipe-peer HOST:PORT");
                std::process::exit(1);
            }
        };
        if let Err(e) = pipeline::run_head(&model, &peer, &args.prompt, args.pipe_max_tokens, true)
        {
            eprintln!("pipeline head failed: {e}");
            std::process::exit(1);
        }
        return;
    }
    if args.pipe_tail {
        let model = match args.model.as_ref() {
            Some(m) => m.clone(),
            None => {
                eprintln!("--pipe-tail requires --model PATH");
                std::process::exit(1);
            }
        };
        let listen = args
            .pipe_listen
            .clone()
            .unwrap_or_else(|| "0.0.0.0:42424".to_string());
        if let Err(e) = pipeline::run_tail(&model, &listen, true) {
            eprintln!("pipeline tail failed: {e}");
            std::process::exit(1);
        }
        return;
    }
    if args.mesh && args.chat {
        if let Err(error) = run_mesh_chat_mode(args.mesh_port) {
            eprintln!("mesh chat mode failed: {error}");
            std::process::exit(1);
        }
        return;
    }
    if args.chat && args.model.is_none() {
        let stdin = io::stdin();
        let mut reader = stdin.lock();
        let stdout = io::stdout();
        let mut writer = stdout.lock();
        if let Err(error) = run_chat_mode(&mut reader, &mut writer) {
            eprintln!("chat mode failed: {error}");
        }
        return;
    }
    if args.mesh {
        if let Err(error) = run_mesh_mode(args.mesh_port) {
            eprintln!("mesh mode failed: {error}");
            std::process::exit(1);
        }
        return;
    }
    if let Some(model_path) = args.model.clone() {
        let loader = GgufModelLoader;
        let mapped = match loader.load_with_progress(&model_path, |progress| {
            println!("{}", render_load_progress(progress))
        }) {
            Ok(mapped) => mapped,
            Err(error) => {
                eprintln!("failed to load model: {error}");
                return;
            }
        };
        // Run autotune after the model is mapped (so we can
        // fingerprint it) but before the rest of the pipeline —
        // `apply_plan` mutates `args` to fill in any field the user
        // didn't set explicitly.
        if args.auto && !args.no_auto {
            let inv = oxidize_core::autotune::detect();
            let model = oxidize_core::autotune::fingerprint(&mapped);
            let plan = oxidize_core::autotune::plan(&inv, &model);
            let print = match args.print_plan.as_str() {
                "json" => true,
                "auto" => atty_stdout(),
                "yes" | "true" | "1" => true,
                "no" | "false" | "0" => false,
                other => {
                    eprintln!(
                        "warning: unknown --print-plan value '{}', defaulting to text",
                        other
                    );
                    true
                }
            };
            if print {
                if args.print_plan == "json" {
                    eprintln!(
                        "{}",
                        serde_json::to_string_pretty(&plan_to_json(&plan))
                            .unwrap_or_else(|_| "{}".to_string())
                    );
                } else {
                    eprintln!("\n[oxidize auto-tune plan]\n{}", plan.summary());
                }
            }
            apply_plan_to_args(&mut args, &plan, &inv);
        }
        // Now that autotune has finalized `args.threads`, build the rayon pool
        // if we deferred it above. This is the first point rayon is used.
        if defer_pool_for_autotune && let Err(error) = build_rayon_pool(resolve_threads(&args)) {
            eprintln!("failed to set rayon thread pool: {error}");
            return;
        }
        optimize_mapped_model_memory(&mapped, &args);
        {
            for lora_path in &args.lora_paths {
                match loader.load(lora_path) {
                    Ok(adapter) => match plan_lora_application(
                        &mapped.parsed().tensor_infos,
                        &adapter.parsed().tensor_infos,
                        mapped.parsed().quantization_type(),
                    ) {
                        Ok(plan) => println!("{}", render_lora_plan(&plan)),
                        Err(error) => eprintln!("failed to plan adapter: {error:?}"),
                    },
                    Err(error) => eprintln!("failed to load adapter: {error}"),
                }
            }
            if args.gpus > 1 {
                let Some(strategy) = parse_parallelism(&args.parallelism) else {
                    eprintln!(
                        "invalid --parallelism value: {} (expected: tensor|pipeline)",
                        args.parallelism
                    );
                    return;
                };
                let config = MultiGpuConfig {
                    gpu_count: args.gpus,
                    n_gpu_layers: args.n_gpu_layers,
                    strategy,
                };
                match plan_multi_gpu_offload(&mapped.parsed().tensor_infos, &config) {
                    Ok(plan) => println!("{}", render_multi_gpu_offload_plan(&plan)),
                    Err(error) => {
                        eprintln!("failed to build multi-gpu offload plan: {error:?}")
                    }
                }
            } else {
                let plan = plan_layer_offload(&mapped.parsed().tensor_infos, args.n_gpu_layers);
                println!("{}", render_offload_plan(&plan));
            }

            // Extract model config from GGUF metadata and run generation
            let metadata = &mapped.parsed().metadata;
            let is_dflash = matches!(
                mapped.parsed().architecture(),
                Some("dflash" | "dflash-draft")
            );
            if args.ctx_size == Some(0) {
                eprintln!("invalid --ctx-size: must be greater than 0");
                return;
            }
            if is_dflash && args.draft_model.is_none() && !dflash_gguf_has_io_tensors(&mapped) {
                eprintln!(
                    "DFlash draft GGUF cannot be used as --model for normal generation. Use the full target GGUF with --model and pass this DFlash file via --draft-model, or use a DFlash GGUF that includes lm_head.weight and model.embed_tokens.weight (e.g. *-fullhead.gguf)."
                );
                return;
            }
            let mut config = InferenceConfig::from_gguf(&mapped);
            config.kv_cache_dtype = args.kv_cache_dtype.dtype();
            if args.no_turboquant {
                config.kv_quantization = oxidize_core::kv_cache::KvQuantization::Asymmetric;
            } else if args.turboquant {
                config.kv_quantization = oxidize_core::kv_cache::KvQuantization::TurboQuant;
            }
            if let Some(ctx) = args.ctx_size {
                config.context_size = ctx;
            }
            if args.cpu_optimized {
                config.context_size = config.context_size.min(2048);
            }
            // Auto-cap context to what fits in available RAM.
            // KV cache = layers × ctx × kv_heads × head_dim × 2 (K+V) × dtype_bytes.
            // If the full context would need more than available RAM headroom, shrink it.
            if args.ctx_size.is_none() && !args.cpu_optimized {
                let kv_bytes_per_token = config.layer_count
                        * config.num_key_value_heads
                        * config.kv_head_dim()
                        * 2  // K + V
                        * config.kv_cache_dtype.size_in_bytes();
                let kv_full: u64 =
                    (config.context_size as u64).saturating_mul(kv_bytes_per_token as u64);
                #[cfg(target_os = "linux")]
                let available = oxidize_core::gguf::linux_mem_available_bytes().unwrap_or(u64::MAX);
                #[cfg(not(target_os = "linux"))]
                let available = u64::MAX;
                // Reserve headroom for the model weights (file-backed but needed during
                // inference) plus 8 GiB for OS/workspace/overhead.
                let model_bytes = mapped.bytes().len() as u64;
                let overhead = 8u64 << 30; // 8 GiB
                let kv_budget = available
                    .saturating_sub(model_bytes)
                    .saturating_sub(overhead);
                if kv_full > kv_budget && kv_bytes_per_token > 0 {
                    let capped = (kv_budget / kv_bytes_per_token as u64) as usize;
                    // Round down to nearest power-of-2 multiple of 512.
                    let capped = (capped / 512).max(1) * 512;
                    eprintln!(
                        "context: capped {} → {} tokens (KV cache would need {:.1} GiB, budget {:.1} GiB)",
                        config.context_size,
                        capped,
                        kv_full as f64 / (1 << 30) as f64,
                        kv_budget as f64 / (1 << 30) as f64,
                    );
                    config.context_size = capped;
                }
            }
            // Load tokenizer from GGUF metadata, falling back to an external model.
            // For DFlash smoke runs with borrowed IO, prefer the external
            // tokenizer so sampled ids match the borrowed output head.
            let tokenizer_result = if is_dflash && args.tokenizer_model.is_some() {
                oxidize_core::tokenizer::load_tokenizer_from_gguf_file(
                    args.tokenizer_model.as_deref(),
                )
                .and_then(|opt| {
                    opt.ok_or_else(|| {
                        "external tokenizer model did not contain tokenizer metadata".to_string()
                    })
                })
                .map_err(|_e| {
                    oxidize_core::tokenizer::TokenizerLoadError::MissingMetadata(
                        "tokenizer.ggml.model",
                    )
                })
                .or_else(|_| load_tokenizer_from_gguf_metadata(metadata))
            } else {
                load_tokenizer_from_gguf_metadata(metadata).or_else(|_| {
                    if is_dflash && dflash_gguf_has_io_tensors(&mapped) {
                        Ok(dflash_byte_smoke_tokenizer())
                    } else {
                        oxidize_core::tokenizer::load_tokenizer_from_gguf_file(
                            args.tokenizer_model.as_deref(),
                        )
                        .and_then(|opt| {
                            opt.ok_or_else(|| {
                                "external tokenizer model did not contain tokenizer metadata"
                                    .to_string()
                            })
                        })
                        .map_err(|_e| {
                            oxidize_core::tokenizer::TokenizerLoadError::MissingMetadata(
                                "tokenizer.ggml.model",
                            )
                        })
                    }
                })
            };
            let tokenizer = match tokenizer_result {
                Ok(t) => t,
                Err(error) => {
                    eprintln!("failed to load tokenizer: {error:?}");
                    return;
                }
            };
            let stdout = io::stdout();
            let mut writer = stdout.lock();
            if let Some(draft_model_path) = args.draft_model.as_deref() {
                if is_dflash {
                    eprintln!(
                        "DFlash GGUFs are draft models, not target models. Use --model with the full target GGUF and --draft-model with the DFlash GGUF."
                    );
                    return;
                }

                if oxidize_core::eagle3::is_eagle3_safetensors_path(draft_model_path) {
                    if args.layer_wise {
                        eprintln!(
                            "EAGLE3 speculative decoding requires a dense InferenceModel target (omit --layer-wise)"
                        );
                        return;
                    }
                    let mut target_model =
                        match InferenceModel::load_from_gguf(&mapped, config.clone(), true) {
                            Ok(model) => model,
                            Err(error) => {
                                eprintln!("failed to load target model weights: {error}");
                                return;
                            }
                        };
                    let target_hints = oxidize_core::eagle3::Eagle3TargetHints {
                        target_hidden_size: config.hidden_size,
                        target_layer_count: target_model.layer_count(),
                        target_vocab_size: Some(target_model.vocab_size()),
                    };
                    let mut draft_model =
                        match oxidize_core::eagle3::Eagle3DraftModel::load_eagle3_draft(
                            draft_model_path,
                            target_hints,
                        ) {
                            Ok(model) => model,
                            Err(error) => {
                                eprintln!("failed to load EAGLE3 SafeTensors draft: {error}");
                                return;
                            }
                        };
                    eprintln!(
                        "using EAGLE3 speculative decoding (SafeTensors): target={} draft={} draft_tokens={}",
                        model_path.display(),
                        draft_model_path.display(),
                        args.draft_tokens
                    );
                    if let Err(error) = generate_with_eagle3_draft(
                        &args.prompt,
                        &mut target_model,
                        &mut draft_model,
                        &tokenizer,
                        args.max_tokens,
                        args.temperature,
                        args.top_p,
                        args.top_k,
                        args.draft_tokens,
                        &mut writer,
                    ) {
                        eprintln!("generation failed: {error}");
                    }
                    return;
                }

                let draft_mapped = match loader.load(draft_model_path) {
                    Ok(mapped) => mapped,
                    Err(error) => {
                        eprintln!(
                            "failed to load draft model {}: {error}",
                            draft_model_path.display()
                        );
                        return;
                    }
                };
                let draft_arch = draft_mapped.parsed().architecture();
                if matches!(draft_arch, Some("eagle3" | "eagle")) {
                    if args.layer_wise {
                        eprintln!(
                            "EAGLE3 speculative decoding requires a dense InferenceModel target (omit --layer-wise)"
                        );
                        return;
                    }
                    let mut target_model =
                        match InferenceModel::load_from_gguf(&mapped, config.clone(), true) {
                            Ok(model) => model,
                            Err(error) => {
                                eprintln!("failed to load target model weights: {error}");
                                return;
                            }
                        };
                    let draft_config =
                        oxidize_core::eagle3::Eagle3Config::from_gguf(&draft_mapped);
                    let mut draft_model = match oxidize_core::eagle3::Eagle3DraftModel::load_from_gguf(
                        &draft_mapped,
                        draft_config,
                    ) {
                        Ok(model) => model,
                        Err(error) => {
                            eprintln!("failed to load EAGLE3 draft model: {error}");
                            return;
                        }
                    };
                    eprintln!(
                        "using EAGLE3 speculative decoding: target={} draft={} draft_tokens={}",
                        model_path.display(),
                        draft_model_path.display(),
                        args.draft_tokens
                    );
                    if let Err(error) = generate_with_eagle3_draft(
                        &args.prompt,
                        &mut target_model,
                        &mut draft_model,
                        &tokenizer,
                        args.max_tokens,
                        args.temperature,
                        args.top_p,
                        args.top_k,
                        args.draft_tokens,
                        &mut writer,
                    ) {
                        eprintln!("generation failed: {error}");
                    }
                    return;
                }

                let mut target_model: Box<dyn Model> = if args.layer_wise {
                    match oxidize_core::layer_wise::LayerWiseModel::load_from_gguf(
                        &mapped,
                        config.clone(),
                        args.layer_cache,
                    ) {
                        Ok(mut model) => {
                            if let Err(error) = model.warm_layer_cache() {
                                eprintln!("failed to warm layer cache: {error}");
                                return;
                            }
                            Box::new(model)
                        }
                        Err(error) => {
                            eprintln!("failed to load layer-wise target model: {error}");
                            return;
                        }
                    }
                } else {
                    match InferenceModel::load_from_gguf(&mapped, config.clone(), true) {
                        Ok(model) => Box::new(model),
                        Err(error) => {
                            eprintln!("failed to load target model weights: {error}");
                            return;
                        }
                    }
                };
                let target_hidden_size = config.hidden_size;
                let target_layer_count = target_model.layer_count();

                if !matches!(draft_arch, Some("dflash" | "dflash-draft")) {
                    eprintln!(
                        "--draft-model must point to a DFlash, EAGLE3 GGUF, or EAGLE3 SafeTensors draft, got architecture {:?}",
                        draft_arch
                    );
                    return;
                }
                let draft_config = oxidize_core::dflash::DFlashConfig::from_gguf(&draft_mapped);
                let mut draft_model = match oxidize_core::dflash::DFlashDraftModel::load_from_gguf(
                    &draft_mapped,
                    draft_config,
                ) {
                    Ok(model) => model,
                    Err(error) => {
                        eprintln!("failed to load DFlash draft model: {error}");
                        return;
                    }
                };
                if let Err(error) = draft_model.load_external_io_from_gguf(&mapped) {
                    eprintln!(
                        "failed to borrow draft token embeddings/output from target GGUF: {error}"
                    );
                    return;
                }
                let incompatible_hidden = draft_model.config.hidden_size != target_hidden_size;
                let incompatible_layers = draft_model
                    .config
                    .target_layer_ids
                    .iter()
                    .any(|&layer| layer >= target_layer_count);
                if incompatible_hidden || incompatible_layers {
                    if args.force_dflash {
                        eprintln!(
                            "forcing DFlash with incompatible target (draft_hidden={}, target_hidden={}, draft_target_layers={:?}, target_layers={}); target verification still controls output, but acceptance may be poor",
                            draft_model.config.hidden_size,
                            target_hidden_size,
                            draft_model.config.target_layer_ids,
                            target_layer_count
                        );
                    } else {
                        eprintln!(
                            "DFlash draft is incompatible with target (draft_hidden={}, target_hidden={}, draft_target_layers={:?}, target_layers={}); falling back to target-only generation (pass --force-dflash to test anyway)",
                            draft_model.config.hidden_size,
                            target_hidden_size,
                            draft_model.config.target_layer_ids,
                            target_layer_count
                        );
                        if let Err(error) = generate_with_model(
                            &args.prompt,
                            target_model.as_mut(),
                            &tokenizer,
                            args.max_tokens,
                            args.temperature,
                            args.top_p,
                            args.top_k,
                            &mut writer,
                        ) {
                            eprintln!("generation failed: {error}");
                        }
                        return;
                    }
                }
                if draft_model.vocab_size() != target_model.vocab_size() {
                    eprintln!(
                        "DFlash draft vocab ({}) does not match target vocab ({}) after borrowing target IO",
                        draft_model.vocab_size(),
                        target_model.vocab_size()
                    );
                    return;
                }
                // DFlash speculative decoding is opt-in behind OX_DFLASH_SPECULATIVE.
                // When the flag is unset (default), a loaded draft model is ignored
                // and we fall back to plain target-only generation, preserving the
                // previous non-speculative default behavior exactly.
                if std::env::var("OX_DFLASH_SPECULATIVE").is_err() {
                    eprintln!(
                        "DFlash draft loaded but OX_DFLASH_SPECULATIVE is not set; using target-only generation (set OX_DFLASH_SPECULATIVE=1 to enable speculative decoding)"
                    );
                    if let Err(error) = generate_with_model(
                        &args.prompt,
                        target_model.as_mut(),
                        &tokenizer,
                        args.max_tokens,
                        args.temperature,
                        args.top_p,
                        args.top_k,
                        &mut writer,
                    ) {
                        eprintln!("generation failed: {error}");
                    }
                    return;
                }
                eprintln!(
                    "using DFlash speculative decoding: target={} draft={} draft_tokens={}",
                    model_path.display(),
                    draft_model_path.display(),
                    args.draft_tokens
                );
                if let Err(error) = generate_with_dflash_draft(
                    &args.prompt,
                    target_model.as_mut(),
                    &mut draft_model,
                    &tokenizer,
                    args.max_tokens,
                    args.temperature,
                    args.top_p,
                    args.top_k,
                    args.draft_tokens,
                    &mut writer,
                ) {
                    eprintln!("generation failed: {error}");
                }
                return;
            }

            if !is_dflash
                && !args.layer_wise
                && effective_backend != oxidize_core::backend::Backend::Mlx
            {
                let use_mmap = true;
                let mut concrete_model =
                    match InferenceModel::load_from_gguf(&mapped, config.clone(), use_mmap) {
                        Ok(model) => model,
                        Err(error) => {
                            eprintln!("failed to load model weights: {error}");
                            return;
                        }
                    };
                if args.quantspec && !args.no_mtp && !args.chat {
                    if !concrete_model.has_mtp() {
                        eprintln!(
                            "--quantspec requires native MTP/nextn weights in the target GGUF; falling back to target-only generation"
                        );
                    } else {
                        eprintln!(
                            "using QuantSpec self-speculative decoding (I8 TurboQuant MTP draft KV): target={} draft_tokens={}",
                            model_path.display(),
                            args.draft_tokens
                        );
                        if let Err(error) = generate_with_quantspec(
                            &args.prompt,
                            &mut concrete_model,
                            &tokenizer,
                            args.max_tokens,
                            args.temperature,
                            args.top_p,
                            args.top_k,
                            args.draft_tokens,
                            &mut writer,
                        ) {
                            eprintln!("generation failed: {error}");
                        }
                        return;
                    }
                }
                if concrete_model.has_mtp() && !args.no_mtp && !args.chat && !args.quantspec {
                    eprintln!(
                        "using native MTP/nextn speculative decoding: target={} nextn_layers={} draft_tokens={}",
                        model_path.display(),
                        concrete_model.nextn_predict_layers(),
                        args.draft_tokens
                    );
                    if let Err(error) = generate_with_mtp_model(
                        &args.prompt,
                        &mut concrete_model,
                        &tokenizer,
                        args.max_tokens,
                        args.temperature,
                        args.top_p,
                        args.top_k,
                        args.draft_tokens,
                        &mut writer,
                        false,
                    ) {
                        eprintln!("generation failed: {error}");
                    }
                    return;
                }
                if concrete_model.has_mtp() && args.chat && !args.no_mtp {
                    eprintln!(
                        "native MTP/nextn is available but chat mode currently uses target-only generation"
                    );
                }
                let mut model: Box<dyn Model> = Box::new(concrete_model);
                if args.chat {
                    let stdin = io::stdin();
                    let mut reader = stdin.lock();
                    if let Err(error) = run_model_chat_mode(
                        &mut reader,
                        &mut writer,
                        &mut model,
                        &tokenizer,
                        args.max_tokens,
                        args.temperature,
                        args.top_p,
                        args.top_k,
                    ) {
                        eprintln!("chat mode failed: {error}");
                    }
                    return;
                }

                if let Err(error) = generate_with_model(
                    &args.prompt,
                    &mut model,
                    &tokenizer,
                    args.max_tokens,
                    args.temperature,
                    args.top_p,
                    args.top_k,
                    &mut writer,
                ) {
                    eprintln!("generation failed: {error}");
                }
                return;
            }

            let mut model: Box<dyn Model> = if is_dflash {
                let dflash_config = oxidize_core::dflash::DFlashConfig::from_gguf(&mapped);
                match oxidize_core::dflash::DFlashDraftModel::load_from_gguf(&mapped, dflash_config)
                {
                    Ok(mut m) => {
                        if (!m.output.is_loaded() || !m.tok_embeddings.is_loaded())
                            && let Some(io_model_path) = args.tokenizer_model.as_deref()
                        {
                            match loader.load(io_model_path) {
                                Ok(io_mapped) => {
                                    if let Err(error) = m.load_external_io_from_gguf(&io_mapped) {
                                        eprintln!(
                                            "failed to borrow DFlash IO tensors from {}: {error}",
                                            io_model_path.display()
                                        );
                                        return;
                                    }
                                    eprintln!(
                                        "borrowed DFlash token embeddings/output from {} for smoke-test generation",
                                        io_model_path.display()
                                    );
                                }
                                Err(error) => {
                                    eprintln!(
                                        "failed to load DFlash IO model {}: {error}",
                                        io_model_path.display()
                                    );
                                    return;
                                }
                            }
                        }
                        if !m.output.is_loaded() || !m.tok_embeddings.is_loaded() {
                            eprintln!(
                                "DFlash draft GGUF is still missing token embeddings or lm_head; use *-fullhead.gguf or pass --tokenizer-model with a GGUF that has output.weight and embed_tokens."
                            );
                            return;
                        }
                        eprintln!(
                            "DFlash standalone generation using builtin lm_head/embeddings in {}",
                            model_path.display()
                        );
                        Box::new(m)
                    }
                    Err(error) => {
                        eprintln!("failed to load DFlash model: {error}");
                        return;
                    }
                }
            } else if args.layer_wise {
                match oxidize_core::layer_wise::LayerWiseModel::load_from_gguf(
                    &mapped,
                    config,
                    args.layer_cache,
                ) {
                    Ok(mut m) => {
                        if let Err(error) = m.warm_layer_cache() {
                            eprintln!("failed to warm layer cache: {error}");
                            return;
                        }
                        Box::new(m)
                    }
                    Err(error) => {
                        eprintln!("failed to load layer-wise model: {error}");
                        return;
                    }
                }
            } else if effective_backend == oxidize_core::backend::Backend::Mlx {
                #[cfg(target_os = "macos")]
                {
                    match oxidize_core::mlx_inference::MlxInferenceModel::load_from_gguf(
                        &mapped, config,
                    ) {
                        Ok(m) => {
                            println!("MLX backend: loaded model into unified memory");
                            Box::new(m)
                        }
                        Err(error) => {
                            eprintln!("MLX initialization failed: {error}; falling back to CPU");
                            let use_mmap = true;
                            match InferenceModel::load_from_gguf(&mapped, config, use_mmap) {
                                Ok(m) => Box::new(m),
                                Err(error) => {
                                    eprintln!("failed to load model weights: {error}");
                                    return;
                                }
                            }
                        }
                    }
                }
                #[cfg(not(target_os = "macos"))]
                {
                    eprintln!(
                        "MLX backend requested but unavailable on Linux; falling back to CPU"
                    );
                    let use_mmap = true;
                    match InferenceModel::load_from_gguf(&mapped, config, use_mmap) {
                        Ok(m) => Box::new(m),
                        Err(error) => {
                            eprintln!("failed to load model weights: {error}");
                            return;
                        }
                    }
                }
            } else {
                let use_mmap = true;
                match InferenceModel::load_from_gguf(&mapped, config, use_mmap) {
                    Ok(m) => Box::new(m),
                    Err(error) => {
                        eprintln!("failed to load model weights: {error}");
                        return;
                    }
                }
            };

            if args.chat {
                let stdin = io::stdin();
                let mut reader = stdin.lock();
                if let Err(error) = run_model_chat_mode(
                    &mut reader,
                    &mut writer,
                    &mut model,
                    &tokenizer,
                    args.max_tokens,
                    args.temperature,
                    args.top_p,
                    args.top_k,
                ) {
                    eprintln!("chat mode failed: {error}");
                }
                return;
            }

            if let Err(error) = generate_with_model(
                &args.prompt,
                &mut model,
                &tokenizer,
                args.max_tokens,
                args.temperature,
                args.top_p,
                args.top_k,
                &mut writer,
            ) {
                eprintln!("generation failed: {error}");
            }
        }
        return;
    }
    let stdout = io::stdout();
    let mut writer = stdout.lock();
    if let Err(error) = run_single_shot_mode(&args.prompt, &mut writer) {
        eprintln!("single-shot mode failed: {error}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    #[test]
    fn greeting_uses_prompt() {
        assert_eq!(greeting("test"), "oxidize-cli: test");
    }

    #[test]
    fn renders_offload_plan_summary() {
        let summary = render_offload_plan(&LayerOffloadPlan {
            n_gpu_layers: 2,
            total_layers: 32,
            gpu_tensor_count: 12,
            cpu_tensor_count: 44,
        });
        assert_eq!(
            summary,
            "offload plan: gpu_layers=2/32 gpu_tensors=12 cpu_tensors=44"
        );
    }

    #[test]
    fn parses_parallelism_strategy() {
        assert_eq!(
            parse_parallelism("tensor"),
            Some(ParallelismStrategy::Tensor)
        );
        assert_eq!(
            parse_parallelism("pipeline"),
            Some(ParallelismStrategy::Pipeline)
        );
        assert_eq!(parse_parallelism("invalid"), None);
    }

    #[test]
    fn selects_preferred_gguf_quantization() {
        let files = vec![
            "model-q8_0.gguf".to_owned(),
            "model-q4_k_m.gguf".to_owned(),
            "model-q5_0.gguf".to_owned(),
        ];
        assert_eq!(
            select_default_gguf(&files),
            Some("model-q4_k_m.gguf".to_owned())
        );
    }

    #[test]
    fn cache_safe_name_removes_repo_separator() {
        assert_eq!(
            cache_safe_name("freakyskittle/codeforge-slm-1.5b"),
            "freakyskittle-codeforge-slm-1-5b"
        );
    }

    #[test]
    fn renders_multi_gpu_offload_plan_summary() {
        let summary = render_multi_gpu_offload_plan(&MultiGpuOffloadPlan {
            strategy: ParallelismStrategy::Pipeline,
            total_layers: 32,
            n_gpu_layers: 6,
            total_gpu_tensor_count: 36,
            cpu_tensor_count: 20,
            gpu_assignments: vec![
                oxidize_core::offload::GpuAssignment {
                    gpu_index: 0,
                    layer_count: 3,
                    tensor_count: 18,
                },
                oxidize_core::offload::GpuAssignment {
                    gpu_index: 1,
                    layer_count: 3,
                    tensor_count: 18,
                },
            ],
            pipeline_stages: vec![],
        });
        assert_eq!(
            summary,
            "offload plan: strategy=pipeline gpu_layers=6/32 gpu_tensors=36 cpu_tensors=20 gpu0:layers=3 tensors=18 gpu1:layers=3 tensors=18"
        );
    }

    #[test]
    fn renders_lora_plan_summary() {
        let summary = render_lora_plan(&LoraPlan {
            kind: AdapterKind::Qlora,
            targets: vec![oxidize_core::lora::LoraTarget {
                base_tensor: "blk.0.attn_q.weight".to_owned(),
                lora_a_tensor: "blk.0.attn_q.weight.lora_a.weight".to_owned(),
                lora_b_tensor: "blk.0.attn_q.weight.lora_b.weight".to_owned(),
            }],
            missing_base_tensors: vec!["blk.1.attn_q.weight".to_owned()],
        });
        assert_eq!(
            summary,
            "adapter plan: mode=qlora matched_targets=1 missing_base_targets=1"
        );
    }

    #[test]
    fn chat_mode_replies_and_exits() {
        let mut reader = io::Cursor::new("hello\nexit\n");
        let mut writer = Vec::new();
        run_chat_mode(&mut reader, &mut writer).expect("chat mode should succeed");

        let output = String::from_utf8(writer).expect("valid utf8 output");
        assert!(output.contains("oxidize-cli chat mode. type 'exit' to quit."));
        assert!(output.contains("generation progress: 1/2 tokens"));
        assert!(output.contains("oxidize-cli: hello"));
        assert!(output.contains("> bye"));
    }

    #[test]
    fn chat_mode_ignores_blank_lines() {
        let mut reader = io::Cursor::new("\nworld\nquit\n");
        let mut writer = Vec::new();
        run_chat_mode(&mut reader, &mut writer).expect("chat mode should succeed");

        let output = String::from_utf8(writer).expect("valid utf8 output");
        assert!(output.contains("oxidize-cli: world"));
    }

    #[test]
    fn chat_mode_supports_multiline_prompt_input() {
        let mut reader = io::Cursor::new("hello \\\nworld\nquit\n");
        let mut writer = Vec::new();
        run_chat_mode(&mut reader, &mut writer).expect("chat mode should succeed");

        let output = String::from_utf8(writer).expect("valid utf8 output");
        assert!(output.contains("| "));
        assert!(output.contains("oxidize-cli: hello \nworld"));
    }

    #[test]
    fn chat_mode_renders_and_clears_conversation_history() {
        let mut reader = io::Cursor::new("hello\nworld\n/history\n/clear\n/history\nquit\n");
        let mut writer = Vec::new();
        run_chat_mode(&mut reader, &mut writer).expect("chat mode should succeed");

        let output = String::from_utf8(writer).expect("valid utf8 output");
        assert!(output.contains("1. user: hello\n   assistant: oxidize-cli: hello"));
        assert!(output.contains("2. user: world\n   assistant: oxidize-cli: world"));
        assert!(output.contains("conversation history cleared"));
        assert!(output.contains("no conversation history"));
    }

    #[test]
    fn chat_mode_reuses_cached_prompt_response() {
        let mut reader = io::Cursor::new("hello\nhello\nquit\n");
        let mut writer = Vec::new();
        run_chat_mode(&mut reader, &mut writer).expect("chat mode should succeed");

        let output = String::from_utf8(writer).expect("valid utf8 output");
        assert_eq!(output.matches("generation progress: 1/2 tokens").count(), 1);
        assert_eq!(output.matches("generation progress: 2/2 tokens").count(), 1);
        assert_eq!(output.matches("oxidize-cli: hello").count(), 2);
        assert_eq!(
            output
                .matches("generation stats: tokens=0 speed=0.00 tok/s (cache hit)")
                .count(),
            1
        );
    }

    #[test]
    fn write_generated_response_cached_hits_cache() {
        let mut prompt_cache = PromptCache::default();
        let mut first_writer = Vec::new();
        write_generated_response_cached("hello", &mut prompt_cache, &mut first_writer)
            .expect("first response should succeed");
        let first_output = String::from_utf8(first_writer).expect("valid utf8 output");
        assert!(first_output.contains("generation progress: 1/2 tokens"));
        assert!(first_output.contains("generation progress: 2/2 tokens"));

        let mut second_writer = Vec::new();
        let response =
            write_generated_response_cached("hello", &mut prompt_cache, &mut second_writer)
                .expect("cached response should succeed");
        let second_output = String::from_utf8(second_writer).expect("valid utf8 output");
        assert_eq!(response, "oxidize-cli: hello");
        assert_eq!(
            second_output,
            "oxidize-cli: hello\ngeneration stats: tokens=0 speed=0.00 tok/s (cache hit)\n"
        );
    }

    #[test]
    fn conversation_history_render_uses_empty_state() {
        let history = ConversationHistory::default();
        assert_eq!(history.render(), "no conversation history");
    }

    #[test]
    fn single_shot_mode_writes_one_response() {
        let mut writer = Vec::new();
        write_generated_response_with_clock("hello", &mut writer, {
            let mut ticks = [0u64, 500].into_iter();
            move || {
                Instant::now()
                    .checked_add(Duration::from_millis(ticks.next().expect("clock tick")))
                    .expect("valid instant")
            }
        })
        .expect("single-shot mode should succeed");
        let output = String::from_utf8(writer).expect("valid utf8 output");
        assert_eq!(
            output,
            "generation progress: 1/2 tokens\ngeneration progress: 2/2 tokens\noxidize-cli: hello\ngeneration stats: tokens=2 speed=4.00 tok/s\n"
        );
    }

    #[test]
    fn single_shot_mode_skips_blank_prompt() {
        let mut writer = Vec::new();
        run_single_shot_mode("   ", &mut writer).expect("single-shot mode should succeed");
        assert!(writer.is_empty());
    }

    #[test]
    fn renders_load_progress_with_bytes_when_available() {
        let rendered = render_load_progress(LoadProgress {
            stage: "mapping",
            percent: 35,
            bytes_processed: Some(1024),
            total_bytes: Some(4096),
        });
        assert_eq!(rendered, "load progress: 35% stage=mapping bytes=1024/4096");
    }

    #[test]
    fn renders_load_progress_without_bytes_when_unavailable() {
        let rendered = render_load_progress(LoadProgress {
            stage: "starting",
            percent: 0,
            bytes_processed: None,
            total_bytes: None,
        });
        assert_eq!(rendered, "load progress: 0% stage=starting");
    }

    #[test]
    fn write_generated_response_emits_progress_and_final_response() {
        let mut writer = Vec::new();
        let response = write_generated_response_with_clock("there", &mut writer, {
            let mut ticks = [0u64, 200].into_iter();
            move || {
                Instant::now()
                    .checked_add(Duration::from_millis(ticks.next().expect("clock tick")))
                    .expect("valid instant")
            }
        })
        .expect("generation should succeed");
        let output = String::from_utf8(writer).expect("valid utf8 output");
        assert_eq!(response, "oxidize-cli: there");
        assert_eq!(
            output,
            "generation progress: 1/2 tokens\ngeneration progress: 2/2 tokens\noxidize-cli: there\ngeneration stats: tokens=2 speed=10.00 tok/s\n"
        );
    }

    #[test]
    fn format_generation_stats_reports_tokens_and_speed() {
        let stats = format_generation_stats(12, Duration::from_secs(3));
        assert_eq!(stats, "generation stats: tokens=12 speed=4.00 tok/s");
    }

    #[test]
    fn profiler_command_builds_perf_record_invocation() {
        let output = PathBuf::from("cpu.perf.data");
        let exe = PathBuf::from("target/debug/oxidize-cli");
        let args = vec!["--prompt".to_owned(), "ping".to_owned()];
        let command = profiler_command(Profiler::Perf, Some(&output), &exe, &args);

        assert_eq!(command.get_program().to_string_lossy(), "perf");
        let got = command
            .get_args()
            .map(|arg| arg.to_string_lossy().to_string())
            .collect::<Vec<_>>();
        assert_eq!(
            got,
            vec![
                "record",
                "--call-graph=dwarf",
                "-o",
                "cpu.perf.data",
                "target/debug/oxidize-cli",
                "--prompt",
                "ping"
            ]
        );
    }

    #[test]
    fn profiler_command_builds_samply_record_invocation() {
        let exe = PathBuf::from("target/debug/oxidize-cli");
        let args = vec!["--prompt".to_owned(), "ping".to_owned()];
        let command = profiler_command(Profiler::Samply, None, &exe, &args);

        assert_eq!(command.get_program().to_string_lossy(), "samply");
        let got = command
            .get_args()
            .map(|arg| arg.to_string_lossy().to_string())
            .collect::<Vec<_>>();
        assert_eq!(
            got,
            vec!["record", "target/debug/oxidize-cli", "--prompt", "ping"]
        );
    }

    #[test]
    fn strips_profile_flags_from_passthrough_arguments() {
        let result = filter_passthrough_args(
            [
                "--prompt",
                "ping",
                "--profile",
                "perf",
                "--profile-output",
                "perf.data",
                "--chat",
            ]
            .into_iter()
            .map(str::to_owned),
        );
        assert_eq!(result, vec!["--prompt", "ping", "--chat"]);

        let result_equals = filter_passthrough_args(
            [
                "--prompt=ping",
                "--profile=samply",
                "--profile-output=out.json",
            ]
            .into_iter()
            .map(str::to_owned),
        );
        assert_eq!(result_equals, vec!["--prompt=ping"]);
    }

    #[test]
    fn rewrites_run_command_to_model_flags() {
        let args = rewrite_run_args(
            ["oxidize", "run", "local.gguf", "hello", "--max-tokens", "7"]
                .into_iter()
                .map(OsString::from),
        )
        .expect("run args should rewrite");
        assert!(args.contains(&OsString::from("--model")));
        assert!(args.contains(&OsString::from("local.gguf")));
        assert!(!args.contains(&OsString::from("--serve-api")));
        assert!(args.contains(&OsString::from("--prompt")));
        assert!(args.contains(&OsString::from("hello")));
        assert!(args.contains(&OsString::from("--max-tokens")));
        assert!(args.contains(&OsString::from("7")));
        assert!(args.contains(&OsString::from("--cpu-optimized")));
        assert!(args.contains(&OsString::from("--mmap-prefetch")));
        assert!(args.contains(&OsString::from("--mmap-hugepages")));
        assert!(args.contains(&OsString::from("--kv-cache-dtype")));
        assert!(args.contains(&OsString::from("f16")));
    }

    #[test]
    fn serve_rewrite_accepts_port() {
        let args = rewrite_run_args(
            ["oxidize", "serve", "local.gguf", "--port", "3000"]
                .into_iter()
                .map(OsString::from),
        )
        .expect("serve args should rewrite");
        assert!(args.contains(&OsString::from("--api-port")));
        assert!(args.contains(&OsString::from("3000")));
        assert!(args.contains(&OsString::from("--serve-api")));
        assert!(args.contains(&OsString::from("--api-only")));
    }

    #[test]
    fn run_rewrite_does_not_treat_option_values_as_prompt() {
        let args = rewrite_run_args(
            ["oxidize", "run", "local.gguf", "--max-tokens", "7"]
                .into_iter()
                .map(OsString::from),
        )
        .expect("run args should rewrite");
        assert!(args.contains(&OsString::from("--max-tokens")));
        assert!(!args.contains(&OsString::from("--prompt")));
        assert!(args.contains(&OsString::from("--chat")));
        assert!(!args.contains(&OsString::from("--api-only")));
    }

    #[test]
    fn run_rewrite_without_prompt_opens_chat_tui_and_server() {
        let args = rewrite_run_args(
            ["oxidize", "run", "local.gguf"]
                .into_iter()
                .map(OsString::from),
        )
        .expect("run args should rewrite");
        assert!(args.contains(&OsString::from("--chat")));
        assert!(args.contains(&OsString::from("--serve-api")));
        assert!(!args.contains(&OsString::from("--prompt")));
    }

    #[test]
    fn run_rewrite_with_prompt_skips_background_server() {
        let args = rewrite_run_args(
            ["oxidize", "run", "local.gguf", "hello"]
                .into_iter()
                .map(OsString::from),
        )
        .expect("run args should rewrite");
        assert!(args.contains(&OsString::from("--prompt")));
        assert!(!args.contains(&OsString::from("--api-only")));
        assert!(!args.contains(&OsString::from("--serve-api")));
    }

    #[test]
    fn run_rewrite_no_api_skips_server() {
        let args = rewrite_run_args(
            ["oxidize", "run", "local.gguf", "--no-api"]
                .into_iter()
                .map(OsString::from),
        )
        .expect("run args should rewrite");
        assert!(args.contains(&OsString::from("--no-api")));
        assert!(!args.contains(&OsString::from("--serve-api")));
    }

    #[test]
    fn detects_profiling_child_environment() {
        let key = OsString::from(PROFILE_CHILD_ENV);
        let prior = std::env::var_os(&key);
        // SAFETY: tests mutate process env in a scoped way and restore previous value.
        unsafe { std::env::remove_var(&key) };
        assert!(!is_profiling_child());
        // SAFETY: tests mutate process env in a scoped way and restore previous value.
        unsafe { std::env::set_var(&key, "1") };
        assert!(is_profiling_child());
        match prior {
            Some(value) => {
                // SAFETY: restore previous env value for this process.
                unsafe { std::env::set_var(&key, value) }
            }
            None => {
                // SAFETY: restore previous env absence for this process.
                unsafe { std::env::remove_var(&key) }
            }
        }
    }

    #[test]
    fn readme_includes_quick_start_and_validation_commands() {
        let readme = include_str!("../../README.md");
        for section in [
            "## Quick start",
            "## Performance tuning guide",
            "### Clone and build",
            "### Run tests and lint",
            "## Common usage",
        ] {
            assert!(
                readme.contains(section),
                "README must include section: {section}"
            );
        }
        for command in [
            "make build",
            "make test",
            "make lint",
            "--profile perf",
            "--parallelism pipeline",
            "--parallelism tensor",
        ] {
            assert!(
                readme.contains(command),
                "README must include command: {command}"
            );
        }
    }
}
