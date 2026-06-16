mod pipeline;

use clap::{Parser, ValueEnum};
use oxidize_core::generation::{
    GenerationConfig, GenerationStream, MtpGenerationStream, SpeculativeGenerationConfig,
    SpeculativeGenerationStream,
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

// #region agent log
fn agent_debug_log_cli(hypothesis_id: &str, location: &str, message: &str, data: &str) {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0);
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/home/dih/oxidize/.cursor/debug-49b0b9.log")
    {
        let _ = writeln!(
            file,
            "{{\"sessionId\":\"49b0b9\",\"runId\":\"initial\",\"hypothesisId\":\"{}\",\"location\":\"{}\",\"message\":\"{}\",\"data\":{},\"timestamp\":{}}}",
            hypothesis_id, location, message, data, timestamp
        );
    }
}
// #endregion

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
    /// Disable native in-GGUF MTP/nextn speculative decoding when present.
    #[arg(long, default_value_t = false)]
    no_mtp: bool,
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
    argv.iter().any(|a| a == flag || a.starts_with(&format!("{flag}=")))
}

fn print_run_help() {
    println!(
        "Usage: oxidize run <model> [prompt] [options]\n\n\
         Models can be local .gguf files or Hugging Face GGUF repos.\n\n\
         Examples:\n\
           oxidize run ./models/model.gguf \"hello\"\n\
           oxidize run Qwen/Qwen2.5-0.5B-Instruct-GGUF --file qwen2.5-0.5b-instruct-q4_k_m.gguf --chat\n\
           oxidize run TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \"write a haiku\" --max-tokens 128\n\n\
         Common options: --chat, --prompt, --max-tokens, --temperature, --backend, --threads, --no-api"
    );
}

fn print_serve_help() {
    println!(
        "Usage: oxidize serve [model] [options]\n\n\
         Starts the OpenAI-compatible API server.\n\n\
         Examples:\n\
           oxidize serve ./models/Qwen3-4B-Q4_K_M.gguf\n\
           oxidize serve --host 0.0.0.0 --port 11434\n\
           oxidize serve ./models/model.gguf --temperature 0 --top-k 1\n\n\
         Common options: --host, --port, --model, --max-tokens, --temperature, --top-p, --top-k, --threads"
    );
}

fn print_ollama_help() {
    println!(
        "Usage: oxidize <command> [args]\n\n\
         Commands:\n\
           run <model> [prompt]     Run a model locally\n\
           serve [model]            Start the OpenAI-compatible server\n\
           list                     List local GGUF models in ./models\n\n\
         Examples:\n\
           oxidize run ./models/Qwen3-4B-Q4_K_M.gguf \"hello\"\n\
           oxidize serve ./models/Qwen3-4B-Q4_K_M.gguf\n\
           oxidize list"
    );
}

fn print_model_list() -> io::Result<()> {
    let models_dir = std::env::current_dir()?.join("models");
    let mut rows = Vec::new();
    if models_dir.is_dir() {
        for entry in std::fs::read_dir(&models_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
            {
                let metadata = entry.metadata()?;
                let size_gib = metadata.len() as f64 / 1024.0 / 1024.0 / 1024.0;
                rows.push((path, size_gib));
            }
        }
    }
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    println!("{:<48} {:>9} PATH", "NAME", "SIZE");
    for (path, size_gib) in rows {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("<invalid>");
        println!("{name:<48} {size_gib:>8.2}G {}", path.display());
    }
    Ok(())
}

fn resolve_model_spec(spec: &str, hf_file: Option<&str>) -> io::Result<PathBuf> {
    let path = PathBuf::from(spec);
    if path.exists() || !spec.contains('/') {
        return Ok(path);
    }

    let api = HfApi::new()?;
    resolve_hf_model_spec(&api, spec, hf_file)
}

#[derive(Debug, Clone)]
struct HfApi {
    cache_dir: PathBuf,
    agent: ureq::Agent,
}

#[derive(Debug, Clone)]
struct HfRepo {
    id: String,
    cache_dir: PathBuf,
    agent: ureq::Agent,
}

#[derive(Debug, Deserialize)]
struct HfRepoInfo {
    #[serde(default)]
    siblings: Vec<HfSibling>,
}

#[derive(Debug, Deserialize)]
struct HfSibling {
    rfilename: String,
}

impl HfApi {
    fn new() -> io::Result<Self> {
        Ok(Self {
            cache_dir: oxidize_cache_dir()?.join("hf"),
            agent: ureq::AgentBuilder::new()
                .timeout_connect(Duration::from_secs(30))
                .timeout_read(Duration::from_secs(600))
                .build(),
        })
    }

    fn model(&self, id: String) -> HfRepo {
        HfRepo {
            id,
            cache_dir: self.cache_dir.clone(),
            agent: self.agent.clone(),
        }
    }
}

impl HfRepo {
    fn info(&self) -> io::Result<HfRepoInfo> {
        let url = format!("https://huggingface.co/api/models/{}", self.id);
        self.agent
            .get(&url)
            .call()
            .map_err(|error| io::Error::other(format!("{error}")))?
            .into_json::<HfRepoInfo>()
            .map_err(|error| io::Error::other(format!("{error}")))
    }

    fn get(&self, filename: &str) -> io::Result<PathBuf> {
        let target = self
            .cache_dir
            .join(cache_safe_name(&self.id))
            .join("main")
            .join(filename);
        if target.exists() {
            return Ok(target);
        }
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            self.id, filename
        );
        let mut response = self
            .agent
            .get(&url)
            .call()
            .map_err(|error| io::Error::other(format!("{error}")))?
            .into_reader();
        let partial = target.with_extension("partial");
        let mut file = std::fs::File::create(&partial)?;
        io::copy(&mut response, &mut file)?;
        std::fs::rename(&partial, &target)?;
        Ok(target)
    }
}

fn model_files_for_repo(repo: &HfRepo, spec: &str) -> io::Result<(Vec<String>, Vec<String>)> {
    let info = repo
        .info()
        .map_err(|error| io::Error::other(format!("failed to inspect HF repo {spec}: {error}")))?;
    let mut ggufs = Vec::new();
    let mut safetensors = Vec::new();
    for sibling in info.siblings {
        let name = sibling.rfilename;
        let lower = name.to_ascii_lowercase();
        if lower.ends_with(".gguf") {
            ggufs.push(name);
        } else if lower.ends_with(".safetensors") {
            // Skip LoRA/PEFT adapter shards — they are low-rank deltas, not full
            // weights. Merging them into the base GGUF as standalone tensors
            // corrupts the model (produces real-token gibberish). Only the base
            // model weights should be converted.
            let base = lower.rsplit('/').next().unwrap_or(&lower);
            if base.starts_with("adapter_model") || base.starts_with("adapter.") {
                continue;
            }
            safetensors.push(name);
        }
    }
    ggufs.sort();
    safetensors.sort();
    Ok((ggufs, safetensors))
}

fn select_default_gguf(ggufs: &[String]) -> Option<String> {
    const PREFERRED: &[&str] = &[
        "q4_k_m", "q4_k_s", "q4_0", "q5_k_m", "q5_0", "q3_k_m", "q8_0",
    ];
    for needle in PREFERRED {
        if let Some(name) = ggufs
            .iter()
            .find(|name| name.to_ascii_lowercase().contains(needle))
        {
            return Some(name.clone());
        }
    }
    ggufs.first().cloned()
}

fn oxidize_cache_dir() -> io::Result<PathBuf> {
    if let Some(home) = std::env::var_os("HOME") {
        Ok(PathBuf::from(home).join(".cache").join("oxidize"))
    } else {
        Ok(std::env::temp_dir().join("oxidize"))
    }
}

fn cache_safe_name(spec: &str) -> String {
    spec.chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect()
}

fn copy_hf_file_to_dir(repo: &HfRepo, filename: &str, dir: &Path) -> io::Result<PathBuf> {
    let source = repo.get(filename).map_err(|error| {
        io::Error::other(format!("failed to download hf file {filename}: {error}"))
    })?;
    let target = dir.join(filename);
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if !target.exists() {
        std::fs::copy(&source, &target)?;
    }
    Ok(target)
}

fn convert_hf_safetensors_repo(
    repo: &HfRepo,
    spec: &str,
    safetensors: &[String],
) -> io::Result<PathBuf> {
    let cache_root = oxidize_cache_dir()?
        .join("hf-converted")
        .join(cache_safe_name(spec));
    let source_dir = cache_root.join("source");
    std::fs::create_dir_all(&source_dir)?;
    // The final model is stored as Q8_0 for fast CPU inference (~3x faster than BF16).
    // A BF16 intermediate is kept only as a conversion scratch file.
    let output = cache_root.join("model-q8.gguf");
    if output.exists() {
        eprintln!("using cached converted GGUF {}", output.display());
        return Ok(output);
    }
    // Fall back to the legacy unquantized name if present (created by older versions).
    let legacy = cache_root.join("model.gguf");
    if legacy.exists() {
        eprintln!(
            "found legacy BF16 GGUF {}; requantizing to Q8_0 for faster inference",
            legacy.display()
        );
        return requantize_gguf_to_q8(&legacy, &output);
    }

    eprintln!(
        "hf://{spec}: no .gguf files, downloading {} SafeTensors file(s) for local GGUF conversion",
        safetensors.len()
    );
    for filename in safetensors {
        copy_hf_file_to_dir(repo, filename, &source_dir)?;
    }
    let config_path = match copy_hf_file_to_dir(repo, "config.json", &source_dir) {
        Ok(path) => Some(path),
        Err(error) => {
            eprintln!("hf://{spec}: config.json unavailable during conversion: {error}");
            None
        }
    };
    // The tokenizer ships separately from the weights on HF. Fetch the standard
    // tokenizer files so the converter can embed tokenizer.ggml.* metadata into
    // the GGUF; without them the model loads but has no usable tokenizer.
    for filename in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ] {
        if let Err(error) = copy_hf_file_to_dir(repo, filename, &source_dir) {
            eprintln!("hf://{spec}: {filename} unavailable during conversion: {error}");
        }
    }

    // Always convert from the directory so the converter can resolve the
    // architecture and tokenizer from config.json / tokenizer.json sitting
    // alongside the weights. Passing a single .safetensors file would hide
    // those and fall back to deriving the arch from the filename.
    let input = source_dir.clone();
    let config = SafetensorsToGgufConfig {
        config_path,
        ..SafetensorsToGgufConfig::default()
    };
    let intermediate = cache_root.join("model.gguf");
    eprintln!("converting hf://{spec} SafeTensors to BF16 GGUF");
    convert_safetensors_to_gguf(&input, &intermediate, &config).map_err(|error| {
        io::Error::other(format!(
            "failed to convert hf://{spec} SafeTensors to GGUF: {error}"
        ))
    })?;
    requantize_gguf_to_q8(&intermediate, &output)
}

fn requantize_gguf_to_q8(input: &std::path::Path, output: &std::path::Path) -> io::Result<PathBuf> {
    use oxidize_core::gguf::GgufQuantizationType;
    use oxidize_core::safetensors_to_gguf::quantize_gguf_to_target;
    eprintln!(
        "requantizing {} → Q8_0 (3x faster CPU inference) → {}",
        input.display(),
        output.display()
    );
    let input_bytes = std::fs::read(input)
        .map_err(|e| io::Error::other(format!("failed to read GGUF for requantization: {e}")))?;
    let quantized = quantize_gguf_to_target(&input_bytes, GgufQuantizationType::Q8_0)
        .map_err(|e| io::Error::other(format!("Q8_0 requantization failed: {e}")))?;
    std::fs::write(output, &quantized)
        .map_err(|e| io::Error::other(format!("failed to write Q8_0 GGUF: {e}")))?;
    eprintln!(
        "Q8_0 GGUF written ({:.1} MB) — model ready",
        quantized.len() as f64 / 1_048_576.0
    );
    Ok(output.to_path_buf())
}

fn gguf_repo_candidates(spec: &str) -> Vec<String> {
    let mut candidates = Vec::new();
    if spec.ends_with("-GGUF") || spec.ends_with("-gguf") {
        candidates.push(spec.to_owned());
        return candidates;
    }
    candidates.push(format!("{spec}-GGUF"));
    if let Some((_, model)) = spec.rsplit_once('/') {
        candidates.push(format!("bartowski/{model}-GGUF"));
        candidates.push(format!("unsloth/{model}-GGUF"));
        candidates.push(format!("TheBloke/{model}-GGUF"));
    }
    candidates
}

fn resolve_hf_model_spec(api: &HfApi, spec: &str, hf_file: Option<&str>) -> io::Result<PathBuf> {
    let mut attempted = Vec::new();
    for candidate in std::iter::once(spec.to_owned()).chain(gguf_repo_candidates(spec)) {
        if attempted.contains(&candidate) {
            continue;
        }
        attempted.push(candidate.clone());
        let repo = api.model(candidate.clone());
        let (ggufs, safetensors) = match model_files_for_repo(&repo, &candidate) {
            Ok(files) => files,
            Err(error) if candidate != spec => {
                eprintln!("hf://{candidate}: {error}; trying next GGUF mirror");
                continue;
            }
            Err(error) => return Err(error),
        };
        let filename = if let Some(file) = hf_file {
            file.to_owned()
        } else if let Some(filename) = select_default_gguf(&ggufs) {
            if ggufs.len() > 1 {
                eprintln!(
                    "hf://{candidate}: selected {filename} (use --file <name> to choose another quant)"
                );
            }
            filename
        } else {
            if candidate == spec && !safetensors.is_empty() {
                return convert_hf_safetensors_repo(&repo, &candidate, &safetensors);
            }
            eprintln!("hf://{candidate}: no .gguf files; trying known GGUF mirrors");
            continue;
        };

        eprintln!("downloading hf://{candidate}/{filename}");
        return repo.get(&filename).map_err(|error| {
            io::Error::other(format!(
                "failed to download hf://{candidate}/{filename}: {error}"
            ))
        });
    }

    Err(io::Error::other(format!(
        "could not find a downloadable GGUF for {spec}; tried: {}. Pass an exact GGUF repo with --file <name> if needed.",
        attempted.join(", ")
    )))
}

fn has_flag(args: &[OsString], name: &str) -> bool {
    args.iter()
        .any(|arg| arg == name || arg.to_string_lossy().starts_with(&format!("{name}=")))
}

/// Handle the `oxidize gpu-cluster <subcommand>` family.
///
/// Subcommands:
///   generate [--family b200|a100|rtx-pro-6000] [--nodes N] [--gpus-per-node N]
///            Emit the Kubernetes/Helm manifests from the GPU cluster spec.
///   detect   Probe the local node for NVIDIA GPUs via nvidia-smi.
///   profiles List the known GPU tier profiles.
fn run_gpu_cluster(args: &[String]) -> i32 {
    use oxidize_core::gpu_cluster as gc;

    let sub = args.first().map(String::as_str).unwrap_or("help");
    match sub {
        "profiles" => {
            for p in gc::all_profiles() {
                println!(
                    "{:<14} product={:<26} arch={:<9} mem={}MiB tdp={}W nvlink={} mig={} timeslice={} net={}",
                    p.family.slug(),
                    p.product,
                    p.generation,
                    p.memory_mib,
                    p.tdp_watts,
                    p.nvlink,
                    p.mig_capable,
                    p.time_slice_replicas,
                    p.network_class,
                );
            }
            0
        }
        "detect" => {
            let gpus = gc::detect_gpus();
            if gpus.is_empty() {
                println!("no NVIDIA GPUs detected (nvidia-smi unavailable or no devices)");
                return 0;
            }
            for g in &gpus {
                let fam = g.family.map(|f| f.slug()).unwrap_or("unknown");
                println!(
                    "GPU {}: {} ({}MiB) family={} mig={}",
                    g.index, g.name, g.memory_total_mib, fam, g.mig_enabled
                );
            }
            println!("--- summary ---");
            for (fam, n) in gc::summarize(&gpus) {
                println!("{}: {}", fam.slug(), n);
            }
            0
        }
        "generate" => {
            let mut family: Option<gc::GpuFamily> = None;
            let mut nodes: u32 = 0;
            let mut gpus_per_node: u32 = 0;
            let mut i = 1;
            while i < args.len() {
                match args[i].as_str() {
                    "--family" => {
                        i += 1;
                        match args.get(i).and_then(|v| gc::GpuFamily::from_slug(v)) {
                            Some(f) => family = Some(f),
                            None => {
                                eprintln!("error: --family expects b200|a100|rtx-pro-6000");
                                return 2;
                            }
                        }
                    }
                    "--nodes" => {
                        i += 1;
                        match args.get(i) {
                            Some(v) => match v.parse() {
                                Ok(n) => nodes = n,
                                Err(_) => {
                                    eprintln!(
                                        "error: --nodes expects a positive integer, got '{v}'"
                                    );
                                    return 2;
                                }
                            },
                            None => {
                                eprintln!("error: --nodes requires a value");
                                return 2;
                            }
                        }
                    }
                    "--gpus-per-node" => {
                        i += 1;
                        match args.get(i) {
                            Some(v) => match v.parse() {
                                Ok(n) => gpus_per_node = n,
                                Err(_) => {
                                    eprintln!(
                                        "error: --gpus-per-node expects a positive integer, got '{v}'"
                                    );
                                    return 2;
                                }
                            },
                            None => {
                                eprintln!("error: --gpus-per-node requires a value");
                                return 2;
                            }
                        }
                    }
                    other => {
                        eprintln!("error: unknown flag {other}");
                        return 2;
                    }
                }
                i += 1;
            }

            // Default to the full three-tier cluster from the spec when no
            // single family is selected.
            let specs = match family {
                Some(f) => {
                    let count = if nodes > 0 {
                        nodes
                    } else {
                        default_node_count(f)
                    };
                    let gpn = if gpus_per_node > 0 {
                        gpus_per_node
                    } else {
                        default_gpus_per_node(f)
                    };
                    vec![gc::NodePoolSpec::new(f, count, gpn)]
                }
                None => vec![
                    gc::NodePoolSpec::new(gc::GpuFamily::B200, 8, 8),
                    gc::NodePoolSpec::new(gc::GpuFamily::A100, 16, 8),
                    gc::NodePoolSpec::new(gc::GpuFamily::RtxPro6000, 4, 2),
                ],
            };
            let families: Vec<gc::GpuFamily> = specs.iter().map(|s| s.family).collect();

            print!("{}", gc::node_pools_yaml(&specs));
            println!("---");
            print!("{}", gc::device_plugin_config_yaml(&families));
            for f in &families {
                if let Some(mig) = gc::mig_config_yaml(*f) {
                    println!("---");
                    print!("{mig}");
                }
            }
            println!("---");
            print!("{}", gc::prometheus_rules_yaml());
            for f in &families {
                println!("---");
                print!("{}", gc::helm_values_yaml(*f));
            }
            0
        }
        _ => {
            eprintln!(
                "usage: oxidize gpu-cluster <generate|detect|profiles>\n\
                 \n\
                 generate [--family b200|a100|rtx-pro-6000] [--nodes N] [--gpus-per-node N]\n\
                 detect   probe local NVIDIA GPUs via nvidia-smi\n\
                 profiles list known GPU tier profiles"
            );
            1
        }
    }
}

fn default_node_count(f: oxidize_core::gpu_cluster::GpuFamily) -> u32 {
    use oxidize_core::gpu_cluster::GpuFamily::*;
    match f {
        B200 => 8,
        A100 => 16,
        RtxPro6000 => 4,
    }
}

fn default_gpus_per_node(f: oxidize_core::gpu_cluster::GpuFamily) -> u32 {
    use oxidize_core::gpu_cluster::GpuFamily::*;
    match f {
        B200 | A100 => 8,
        RtxPro6000 => 2,
    }
}

fn rewrite_run_args<I>(input: I) -> io::Result<Vec<OsString>>
where
    I: IntoIterator<Item = OsString>,
{
    let raw = input.into_iter().collect::<Vec<_>>();
    match raw.get(1).and_then(|arg| arg.to_str()) {
        Some("-h" | "--help") => {
            print_ollama_help();
            std::process::exit(0);
        }
        Some("list" | "ls") => {
            print_model_list()?;
            std::process::exit(0);
        }
        Some("serve") => return rewrite_serve_args(raw),
        Some("gpu-cluster") => {
            let rest: Vec<String> = raw
                .iter()
                .skip(2)
                .filter_map(|a| a.to_str().map(str::to_string))
                .collect();
            let code = run_gpu_cluster(&rest);
            std::process::exit(code);
        }
        Some("run") => {}
        _ => {
            return Ok(raw);
        }
    }
    if raw.get(1).and_then(|arg| arg.to_str()) != Some("run") {
        return Ok(raw);
    }
    if raw.len() == 2
        || matches!(
            raw.get(2).and_then(|arg| arg.to_str()),
            Some("-h" | "--help")
        )
    {
        print_run_help();
        std::process::exit(0);
    }

    let program = raw[0].clone();
    let mut model: Option<String> = None;
    let mut hf_file: Option<String> = None;
    let mut prompt: Option<OsString> = None;
    let mut rewritten = vec![program];
    let mut args = raw.into_iter().skip(2).peekable();

    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("--file") | Some("--hf-file") => {
                let Some(file) = args.next() else {
                    return Err(io::Error::other("--file requires a GGUF filename"));
                };
                hf_file = Some(file.to_string_lossy().into_owned());
            }
            Some(value) if value.starts_with("--file=") => {
                hf_file = Some(value["--file=".len()..].to_owned());
            }
            Some(value) if value.starts_with("--hf-file=") => {
                hf_file = Some(value["--hf-file=".len()..].to_owned());
            }
            Some("--api-host") => {
                rewritten.push("--api-host".into());
                let Some(value) = args.next() else {
                    return Err(io::Error::other("--api-host requires a value"));
                };
                rewritten.push(value);
            }
            Some(value) if value.starts_with("--api-host=") => {
                rewritten.push("--api-host".into());
                rewritten.push(value["--api-host=".len()..].into());
            }
            Some("--api-port") => {
                rewritten.push("--api-port".into());
                let Some(value) = args.next() else {
                    return Err(io::Error::other("--api-port requires a value"));
                };
                rewritten.push(value);
            }
            Some(value) if value.starts_with("--api-port=") => {
                rewritten.push("--api-port".into());
                rewritten.push(value["--api-port=".len()..].into());
            }
            Some(value) if !value.starts_with('-') && model.is_none() => {
                model = Some(value.to_owned());
            }
            Some(value) if !value.starts_with('-') && prompt.is_none() => {
                prompt = Some(arg);
            }
            Some("--no-api") => {
                rewritten.push("--no-api".into());
            }
            Some(
                "--prompt"
                | "--model"
                | "--backend"
                | "--n-gpu-layers"
                | "--gpus"
                | "--parallelism"
                | "--lora"
                | "--profile"
                | "--profile-output"
                | "--max-tokens"
                | "--temperature"
                | "--top-p"
                | "--top-k"
                | "--layer-cache"
                | "--ctx-size"
                | "--threads"
                | "--kv-cache-dtype"
                | "--mesh-port"
                | "--pipe-peer"
                | "--pipe-listen"
                | "--pipe-max-tokens"
                | "--tokenizer-model"
                | "--ram-offload-threads",
            ) => {
                rewritten.push(arg);
                let Some(value) = args.next() else {
                    return Err(io::Error::other("option requires a value"));
                };
                rewritten.push(value);
            }
            _ => rewritten.push(arg),
        }
    }

    let Some(model) = model else {
        return Err(io::Error::other(
            "oxidize run requires a model name or local .gguf path",
        ));
    };
    let model_path = resolve_model_spec(&model, hf_file.as_deref())?;
    rewritten.push("--model".into());
    rewritten.push(model_path.into_os_string());
    let one_shot = prompt.is_some();
    if let Some(prompt) = prompt {
        rewritten.push("--prompt".into());
        rewritten.push(prompt);
    } else if !has_flag(&rewritten, "--chat") {
        rewritten.push("--chat".into());
    }

    for flag in ["--cpu-optimized", "--mmap-prefetch", "--mmap-hugepages"] {
        if !has_flag(&rewritten, flag) {
            rewritten.push(flag.into());
        }
    }
    if !has_flag(&rewritten, "--kv-cache-dtype") {
        // f16/f32 are the KV dtypes decode attention can borrow zero-copy
        // (f16 converts in-kernel via F16C); q8 dequantizes the WHOLE K/V
        // prefix into workspace buffers every layer, every token. f16 also
        // halves attention DRAM reads vs f32 as the context grows. Pass
        // --kv-cache-dtype q8 to trade decode speed for memory.
        rewritten.push("--kv-cache-dtype".into());
        rewritten.push("f16".into());
    }
    // One-shot prompt runs exit right after generation, so a background API
    // server would just load the model a second time (concurrently, stealing
    // memory bandwidth from prefill) and die with the process.
    let skip_api = one_shot
        || has_flag(&rewritten, "--no-api")
        || has_flag(&rewritten, "--mesh")
        || has_flag(&rewritten, "--pipe-head")
        || has_flag(&rewritten, "--pipe-tail");
    if !skip_api && !has_flag(&rewritten, "--serve-api") {
        rewritten.push("--serve-api".into());
    }
    Ok(rewritten)
}

fn rewrite_serve_args(raw: Vec<OsString>) -> io::Result<Vec<OsString>> {
    if raw.len() >= 3
        && matches!(
            raw.get(2).and_then(|arg| arg.to_str()),
            Some("-h" | "--help")
        )
    {
        print_serve_help();
        std::process::exit(0);
    }

    let program = raw[0].clone();
    let mut rewritten = vec![program, "--serve-api".into(), "--api-only".into()];
    let mut model: Option<String> = None;
    let mut hf_file: Option<String> = None;
    let mut args = raw.into_iter().skip(2).peekable();

    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("--host") => {
                rewritten.push("--api-host".into());
                let Some(value) = args.next() else {
                    return Err(io::Error::other("--host requires a value"));
                };
                rewritten.push(value);
            }
            Some(value) if value.starts_with("--host=") => {
                rewritten.push("--api-host".into());
                rewritten.push(value["--host=".len()..].into());
            }
            Some("--port") => {
                rewritten.push("--api-port".into());
                let Some(value) = args.next() else {
                    return Err(io::Error::other("--port requires a value"));
                };
                rewritten.push(value);
            }
            Some(value) if value.starts_with("--port=") => {
                rewritten.push("--api-port".into());
                rewritten.push(value["--port=".len()..].into());
            }
            Some("--file") | Some("--hf-file") => {
                let Some(file) = args.next() else {
                    return Err(io::Error::other("--file requires a GGUF filename"));
                };
                hf_file = Some(file.to_string_lossy().into_owned());
            }
            Some(value) if value.starts_with("--file=") => {
                hf_file = Some(value["--file=".len()..].to_owned());
            }
            Some(value) if value.starts_with("--hf-file=") => {
                hf_file = Some(value["--hf-file=".len()..].to_owned());
            }
            Some(value) if !value.starts_with('-') && model.is_none() => {
                model = Some(value.to_owned());
            }
            Some(
                "--model"
                | "--backend"
                | "--max-tokens"
                | "--temperature"
                | "--top-p"
                | "--top-k"
                | "--ctx-size"
                | "--threads"
                | "--kv-cache-dtype"
                | "--tokenizer-model"
                | "--draft-model"
                | "--draft-tokens"
                | "--layer-cache"
                | "--ram-offload-threads",
            ) => {
                rewritten.push(arg);
                let Some(value) = args.next() else {
                    return Err(io::Error::other("option requires a value"));
                };
                rewritten.push(value);
            }
            _ => rewritten.push(arg),
        }
    }

    if let Some(model) = model {
        let model_path = resolve_model_spec(&model, hf_file.as_deref())?;
        rewritten.push("--model".into());
        rewritten.push(model_path.into_os_string());
    }
    if !has_flag(&rewritten, "--kv-cache-dtype") {
        // Match the `run` rewrite: f16 KV is the zero-copy decode path with
        // half the attention reads of f32 (see the comment there).
        rewritten.push("--kv-cache-dtype".into());
        rewritten.push("f16".into());
    }
    if !has_flag(&rewritten, "--cpu-optimized") {
        rewritten.push("--cpu-optimized".into());
    }
    Ok(rewritten)
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

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum Backend {
    Cpu,
    Metal,
    /// macOS only
    Mlx,
    Cuda,
    Vulkan,
    /// Intel Arc GPUs via Vulkan compute
    IntelArc,
}

impl Backend {
    fn to_core_backend(self) -> oxidize_core::backend::Backend {
        match self {
            Backend::Cpu => oxidize_core::backend::Backend::Cpu,
            Backend::Metal => oxidize_core::backend::Backend::Metal,
            Backend::Mlx => oxidize_core::backend::Backend::Mlx,
            Backend::Cuda => oxidize_core::backend::Backend::Cuda,
            Backend::Vulkan => oxidize_core::backend::Backend::Vulkan,
            Backend::IntelArc => oxidize_core::backend::Backend::IntelArc,
        }
    }

    #[allow(dead_code)]
    fn as_arg(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Metal => "metal",
            Backend::Mlx => "mlx",
            Backend::Cuda => "cuda",
            Backend::Vulkan => "vulkan",
            Backend::IntelArc => "intel-arc",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ConversationTurn {
    user: String,
    assistant: String,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct ConversationHistory {
    turns: Vec<ConversationTurn>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct PromptCache {
    responses: HashMap<String, String>,
}

impl PromptCache {
    fn get(&self, prompt: &str) -> Option<&str> {
        self.responses.get(prompt).map(String::as_str)
    }

    fn insert(&mut self, prompt: &str, response: &str) {
        self.responses
            .insert(prompt.to_owned(), response.to_owned());
    }
}

impl ConversationHistory {
    fn add_turn(&mut self, user: &str, assistant: &str) {
        self.turns.push(ConversationTurn {
            user: user.to_owned(),
            assistant: assistant.to_owned(),
        });
    }

    fn clear(&mut self) {
        self.turns.clear();
    }

    fn render(&self) -> String {
        if self.turns.is_empty() {
            return "no conversation history".to_owned();
        }

        self.turns
            .iter()
            .enumerate()
            .map(|(index, turn)| {
                format!(
                    "{}. user: {}\n   assistant: {}",
                    index + 1,
                    turn.user,
                    turn.assistant
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn render_offload_plan(plan: &LayerOffloadPlan) -> String {
    format!(
        "offload plan: gpu_layers={}/{} gpu_tensors={} cpu_tensors={}",
        plan.n_gpu_layers, plan.total_layers, plan.gpu_tensor_count, plan.cpu_tensor_count
    )
}

fn parse_parallelism(value: &str) -> Option<ParallelismStrategy> {
    match value {
        "tensor" => Some(ParallelismStrategy::Tensor),
        "pipeline" => Some(ParallelismStrategy::Pipeline),
        _ => None,
    }
}

fn render_multi_gpu_offload_plan(plan: &MultiGpuOffloadPlan) -> String {
    let strategy = match plan.strategy {
        ParallelismStrategy::Tensor => "tensor",
        ParallelismStrategy::Pipeline => "pipeline",
    };
    let assignments = plan
        .gpu_assignments
        .iter()
        .map(|assignment| {
            format!(
                "gpu{}:layers={} tensors={}",
                assignment.gpu_index, assignment.layer_count, assignment.tensor_count
            )
        })
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        "offload plan: strategy={strategy} gpu_layers={}/{} gpu_tensors={} cpu_tensors={} {assignments}",
        plan.n_gpu_layers, plan.total_layers, plan.total_gpu_tensor_count, plan.cpu_tensor_count
    )
}

fn render_lora_plan(plan: &LoraPlan) -> String {
    let mode = match plan.kind {
        AdapterKind::Lora => "lora",
        AdapterKind::Qlora => "qlora",
    };
    format!(
        "adapter plan: mode={mode} matched_targets={} missing_base_targets={}",
        plan.targets.len(),
        plan.missing_base_tensors.len()
    )
}

fn read_chat_prompt<R: BufRead, W: Write>(
    reader: &mut R,
    writer: &mut W,
) -> io::Result<Option<String>> {
    let mut lines = Vec::new();
    loop {
        let mut input = String::new();
        if reader.read_line(&mut input)? == 0 {
            if lines.is_empty() {
                return Ok(None);
            }
            break;
        }

        let trimmed = input.trim_end_matches(['\r', '\n']);
        let continues = trimmed.ends_with('\\');
        let line = if continues {
            trimmed[..trimmed.len() - 1].to_owned()
        } else {
            trimmed.to_owned()
        };
        lines.push(line);

        if continues {
            write!(writer, "| ")?;
            writer.flush()?;
            continue;
        }
        break;
    }
    Ok(Some(lines.join("\n")))
}

fn run_chat_mode<R: BufRead, W: Write>(reader: &mut R, writer: &mut W) -> io::Result<()> {
    writeln!(writer, "oxidize-cli chat mode. type 'exit' to quit.")?;
    let mut history = ConversationHistory::default();
    let mut prompt_cache = PromptCache::default();
    loop {
        write!(writer, "> ")?;
        writer.flush()?;

        let Some(input) = read_chat_prompt(reader, writer)? else {
            break;
        };

        let prompt = input.trim();
        if prompt.eq_ignore_ascii_case("exit") || prompt.eq_ignore_ascii_case("quit") {
            writeln!(writer, "bye")?;
            break;
        }
        if prompt.is_empty() {
            continue;
        }

        if prompt.eq_ignore_ascii_case("/history") {
            writeln!(writer, "{}", history.render())?;
            continue;
        }
        if prompt.eq_ignore_ascii_case("/clear") {
            history.clear();
            writeln!(writer, "conversation history cleared")?;
            continue;
        }

        let response = write_generated_response_cached(prompt, &mut prompt_cache, writer)?;
        history.add_turn(prompt, &response);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_model_chat_mode<R: BufRead, W: Write, M: Model>(
    reader: &mut R,
    writer: &mut W,
    model: &mut M,
    tokenizer: &LoadedTokenizer,
    max_tokens: usize,
    temperature: f32,
    top_p: Option<f32>,
    top_k: Option<usize>,
) -> io::Result<()> {
    writeln!(
        writer,
        "╭─ oxidize chat ─────────────────────────────────────╮"
    )?;
    writeln!(
        writer,
        "│ API server is running if enabled; type /exit to quit │"
    )?;
    writeln!(
        writer,
        "│ multiline: end a line with \\                         │"
    )?;
    writeln!(
        writer,
        "╰──────────────────────────────────────────────────────╯"
    )?;
    let mut history = ConversationHistory::default();
    loop {
        write!(writer, "\nYou › ")?;
        writer.flush()?;
        let Some(input) = read_chat_prompt(reader, writer)? else {
            break;
        };
        let prompt = input.trim();
        if prompt.eq_ignore_ascii_case("/exit")
            || prompt.eq_ignore_ascii_case("exit")
            || prompt.eq_ignore_ascii_case("quit")
        {
            writeln!(writer, "bye")?;
            break;
        }
        if prompt.eq_ignore_ascii_case("/history") {
            writeln!(writer, "{}", history.render())?;
            continue;
        }
        if prompt.eq_ignore_ascii_case("/clear") {
            history.clear();
            writeln!(writer, "conversation history cleared")?;
            continue;
        }
        if prompt.is_empty() {
            continue;
        }
        writeln!(writer, "\nAssistant ›")?;
        let response = generate_with_model(
            prompt,
            model,
            tokenizer,
            max_tokens,
            temperature,
            top_p,
            top_k,
            writer,
        )?;
        history.add_turn(prompt, &response);
    }
    Ok(())
}

fn run_single_shot_mode<W: Write>(prompt: &str, writer: &mut W) -> io::Result<()> {
    let prompt = prompt.trim();
    if prompt.is_empty() {
        return Ok(());
    }
    write_generated_response(prompt, writer).map(|_| ())
}

fn render_load_progress(progress: LoadProgress) -> String {
    match (progress.bytes_processed, progress.total_bytes) {
        (Some(bytes_processed), Some(total_bytes)) => {
            format!(
                "load progress: {}% stage={} bytes={}/{}",
                progress.percent, progress.stage, bytes_processed, total_bytes
            )
        }
        _ => format!(
            "load progress: {}% stage={}",
            progress.percent, progress.stage
        ),
    }
}

fn write_generated_response<W: Write>(prompt: &str, writer: &mut W) -> io::Result<String> {
    write_generated_response_with_clock(prompt, writer, Instant::now)
}

fn write_generated_response_cached<W: Write>(
    prompt: &str,
    prompt_cache: &mut PromptCache,
    writer: &mut W,
) -> io::Result<String> {
    if let Some(cached_response) = prompt_cache.get(prompt) {
        writeln!(writer, "{cached_response}")?;
        writeln!(
            writer,
            "generation stats: tokens=0 speed=0.00 tok/s (cache hit)"
        )?;
        return Ok(cached_response.to_owned());
    }

    let response = write_generated_response(prompt, writer)?;
    prompt_cache.insert(prompt, &response);
    Ok(response)
}

fn format_generation_stats(tokens: usize, elapsed: Duration) -> String {
    let elapsed_seconds = elapsed.as_secs_f64();
    let speed = if elapsed_seconds > 0.0 {
        tokens as f64 / elapsed_seconds
    } else {
        0.0
    };
    format!(
        "generation stats: tokens={} speed={:.2} tok/s",
        tokens, speed
    )
}

fn suppressed_generation_tokens(tokenizer: &LoadedTokenizer, vocab_size: usize) -> Vec<u32> {
    let special_tokens = tokenizer.special_tokens();
    let mut suppressed = Vec::new();
    let mut seen = HashSet::new();
    for token in [
        special_tokens.unknown,
        special_tokens.bos,
        special_tokens.pad,
        special_tokens.separator,
        special_tokens.cls,
        special_tokens.mask,
    ]
    .into_iter()
    .flatten()
    {
        if seen.insert(token) {
            suppressed.push(token);
        }
    }

    for token in 0..vocab_size {
        let token = token as u32;
        if seen.contains(&token) || special_tokens.eos == Some(token) {
            continue;
        }
        if let Ok(piece) = tokenizer.decode(&[token])
            && (piece.starts_with("[PAD") || (piece.starts_with('<') && piece.ends_with('>')))
            && seen.insert(token)
        {
            suppressed.push(token);
        }
    }
    suppressed
}

fn write_generated_response_with_clock<W: Write, F: FnMut() -> Instant>(
    prompt: &str,
    writer: &mut W,
    mut now: F,
) -> io::Result<String> {
    let started_at = now();
    let response = greeting(prompt);
    let response_tokens = response.split_whitespace().count();
    if response_tokens == 0 {
        writeln!(writer, "{response}")?;
        writeln!(
            writer,
            "{}",
            format_generation_stats(response_tokens, now().saturating_duration_since(started_at))
        )?;
        return Ok(response);
    }
    for generated in 1..=response_tokens {
        writeln!(
            writer,
            "generation progress: {generated}/{response_tokens} tokens"
        )?;
    }
    writeln!(writer, "{response}")?;
    writeln!(
        writer,
        "{}",
        format_generation_stats(response_tokens, now().saturating_duration_since(started_at))
    )?;
    Ok(response)
}

fn dflash_gguf_has_io_tensors(mapped: &MappedGgufFile) -> bool {
    let infos = mapped.mapped_tensor_infos();
    let has_output = infos
        .iter()
        .any(|tensor| tensor.name == "lm_head.weight" || tensor.name == "output.weight");
    let has_embed = infos.iter().any(|tensor| {
        tensor.name == "model.embed_tokens.weight" || tensor.name == "tok_embeddings.weight"
    });
    has_output && has_embed
}

fn dflash_byte_smoke_tokenizer() -> LoadedTokenizer {
    static TOK: std::sync::OnceLock<LoadedTokenizer> = std::sync::OnceLock::new();
    TOK.get_or_init(|| {
        let bytes: &'static [[u8; 1]] = Box::leak(
            (0u8..=255)
                .map(|byte| [byte])
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        let vocab: Vec<&[u8]> = bytes.iter().map(|entry| entry.as_slice()).collect();
        LoadedTokenizer::Tiktoken(TiktokenTokenizer::new(&vocab, &[]))
    })
    .clone()
}

#[allow(clippy::too_many_arguments)]
fn generate_with_model<W: Write, M: Model + ?Sized>(
    prompt: &str,
    model: &mut M,
    tokenizer: &LoadedTokenizer,
    max_tokens: usize,
    temperature: f32,
    top_p: Option<f32>,
    top_k: Option<usize>,
    writer: &mut W,
) -> io::Result<String> {
    use futures_core::Stream;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context, Poll, Waker};

    let started_at = Instant::now();
    let mut session = Session::new();

    // Encode prompt using the model's tokenizer (add BOS for generation)
    let prompt_tokens = tokenizer.encode_with_special_tokens(
        prompt,
        EncodeOptions {
            add_bos: tokenizer.add_bos_default(),
            add_eos: false,
            pad_to: None,
        },
    );

    let eos_token = tokenizer.special_tokens().eos;
    let suppressed_tokens = suppressed_generation_tokens(tokenizer, model.vocab_size());

    let config = GenerationConfig {
        max_new_tokens: max_tokens,
        stop_token: eos_token,
        suppressed_tokens,
        sampling: SamplingConfig {
            temperature,
            top_p,
            top_k,
            ..SamplingConfig::default()
        },
        ..GenerationConfig::default()
    };

    let mut rng = rand::thread_rng();
    let mut stream = GenerationStream::new(model, &mut session, &prompt_tokens, config, || {
        rand::Rng::r#gen::<f32>(&mut rng)
    });

    let waker = Waker::from(Arc::new(NoopWaker));
    let mut cx = Context::from_waker(&waker);
    let mut pinned = Pin::new(&mut stream);

    let mut generated_tokens: Vec<u32> = Vec::new();

    loop {
        match Stream::poll_next(pinned.as_mut(), &mut cx) {
            Poll::Ready(Some(Ok(token))) => {
                generated_tokens.push(token);
            }
            Poll::Ready(Some(Err(e))) => {
                return Err(io::Error::other(format!("generation error: {:?}", e)));
            }
            Poll::Ready(None) => break,
            Poll::Pending => break,
        }
    }

    let response = tokenizer
        .decode_without_special_tokens(&generated_tokens)
        .unwrap_or_default();
    if !response.is_empty() {
        write!(writer, "{response}")?;
    } else if !generated_tokens.is_empty() {
        write!(writer, "[generated token ids: {generated_tokens:?}]")?;
    }
    writer.flush()?;

    let elapsed = started_at.elapsed();
    writeln!(writer)?;
    writeln!(
        writer,
        "{}",
        format_generation_stats(generated_tokens.len(), elapsed)
    )?;

    Ok(response)
}

#[allow(clippy::too_many_arguments)]
fn generate_with_dflash_draft<W: Write, M: Model + ?Sized>(
    prompt: &str,
    target_model: &mut M,
    draft_model: &mut oxidize_core::dflash::DFlashDraftModel,
    tokenizer: &LoadedTokenizer,
    max_tokens: usize,
    temperature: f32,
    top_p: Option<f32>,
    top_k: Option<usize>,
    draft_tokens: usize,
    writer: &mut W,
) -> io::Result<String> {
    use futures_core::Stream;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context, Poll, Waker};

    let started_at = Instant::now();
    let mut session = Session::new();
    let prompt_tokens = tokenizer.encode_with_special_tokens(
        prompt,
        EncodeOptions {
            add_bos: tokenizer.add_bos_default(),
            add_eos: false,
            pad_to: None,
        },
    );
    let eos_token = tokenizer.special_tokens().eos;
    let suppressed_tokens = suppressed_generation_tokens(tokenizer, target_model.vocab_size());
    let generation = GenerationConfig {
        max_new_tokens: max_tokens,
        stop_token: eos_token,
        suppressed_tokens,
        sampling: SamplingConfig {
            temperature,
            top_p,
            top_k,
            ..SamplingConfig::default()
        },
        ..GenerationConfig::default()
    };
    let config = SpeculativeGenerationConfig {
        generation,
        draft_tokens_per_step: draft_tokens.max(1),
    };

    let mut rng = rand::thread_rng();
    let mut stream = SpeculativeGenerationStream::new(
        target_model,
        draft_model,
        &mut session,
        &prompt_tokens,
        config,
        || rand::Rng::r#gen::<f32>(&mut rng),
    );
    let waker = Waker::from(Arc::new(NoopWaker));
    let mut cx = Context::from_waker(&waker);
    let mut pinned = Pin::new(&mut stream);
    let mut generated_tokens: Vec<u32> = Vec::new();

    loop {
        match Stream::poll_next(pinned.as_mut(), &mut cx) {
            Poll::Ready(Some(Ok(token))) => generated_tokens.push(token),
            Poll::Ready(Some(Err(e))) => {
                return Err(io::Error::other(format!("generation error: {:?}", e)));
            }
            Poll::Ready(None) => break,
            Poll::Pending => break,
        }
    }

    let response = tokenizer
        .decode_without_special_tokens(&generated_tokens)
        .unwrap_or_default();
    if !response.is_empty() {
        write!(writer, "{response}")?;
        writer.flush()?;
    }
    let elapsed = started_at.elapsed();
    writeln!(writer)?;
    writeln!(
        writer,
        "{}",
        format_generation_stats(generated_tokens.len(), elapsed)
    )?;
    Ok(response)
}

#[allow(clippy::too_many_arguments)]
fn generate_with_mtp_model<W: Write>(
    prompt: &str,
    target_model: &mut InferenceModel,
    tokenizer: &LoadedTokenizer,
    max_tokens: usize,
    temperature: f32,
    top_p: Option<f32>,
    top_k: Option<usize>,
    draft_tokens: usize,
    writer: &mut W,
) -> io::Result<String> {
    use futures_core::Stream;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context, Poll, Waker};

    let started_at = Instant::now();
    let mut session = Session::new();
    let prompt_tokens = tokenizer.encode_with_special_tokens(
        prompt,
        EncodeOptions {
            add_bos: tokenizer.add_bos_default(),
            add_eos: false,
            pad_to: None,
        },
    );
    let eos_token = tokenizer.special_tokens().eos;
    let suppressed_tokens = suppressed_generation_tokens(tokenizer, target_model.vocab_size());
    let generation = GenerationConfig {
        max_new_tokens: max_tokens,
        stop_token: eos_token,
        suppressed_tokens,
        sampling: SamplingConfig {
            temperature,
            top_p,
            top_k,
            ..SamplingConfig::default()
        },
        ..GenerationConfig::default()
    };
    let config = SpeculativeGenerationConfig {
        generation,
        draft_tokens_per_step: draft_tokens.max(1),
    };

    let mut rng = rand::thread_rng();
    let mut stream =
        MtpGenerationStream::new(target_model, &mut session, &prompt_tokens, config, || {
            rand::Rng::r#gen::<f32>(&mut rng)
        });
    let waker = Waker::from(Arc::new(NoopWaker));
    let mut cx = Context::from_waker(&waker);
    let mut pinned = Pin::new(&mut stream);
    let mut generated_tokens: Vec<u32> = Vec::new();

    loop {
        match Stream::poll_next(pinned.as_mut(), &mut cx) {
            Poll::Ready(Some(Ok(token))) => generated_tokens.push(token),
            Poll::Ready(Some(Err(e))) => {
                return Err(io::Error::other(format!("generation error: {:?}", e)));
            }
            Poll::Ready(None) => break,
            Poll::Pending => break,
        }
    }

    let response = tokenizer
        .decode_without_special_tokens(&generated_tokens)
        .unwrap_or_default();
    if !response.is_empty() {
        write!(writer, "{response}")?;
    } else if !generated_tokens.is_empty() {
        write!(writer, "[generated token ids: {generated_tokens:?}]")?;
    }
    writer.flush()?;
    let elapsed = started_at.elapsed();
    writeln!(writer)?;
    writeln!(
        writer,
        "{}",
        format_generation_stats(generated_tokens.len(), elapsed)
    )?;
    Ok(response)
}

struct NoopWaker;

impl Wake for NoopWaker {
    fn wake(self: Arc<Self>) {}
}

fn is_profiling_child() -> bool {
    std::env::var_os(PROFILE_CHILD_ENV).is_some()
}

fn current_args_without_profile_flags() -> Vec<String> {
    filter_passthrough_args(std::env::args().skip(1))
}

fn filter_passthrough_args<I>(input: I) -> Vec<String>
where
    I: IntoIterator<Item = String>,
{
    let mut filtered = Vec::new();
    let mut args = input.into_iter().peekable();
    while let Some(arg) = args.next() {
        let remove_next = arg == "--profile" || arg == "--profile-output";
        let remove_current =
            remove_next || arg.starts_with("--profile=") || arg.starts_with("--profile-output=");
        if !remove_current {
            filtered.push(arg);
        }
        if remove_next {
            let _ = args.next();
        }
    }
    filtered
}

fn profiler_command(
    profiler: Profiler,
    output: Option<&PathBuf>,
    exe: &PathBuf,
    passthrough_args: &[String],
) -> Command {
    let mut command = match profiler {
        Profiler::Perf => {
            let mut cmd = Command::new("perf");
            cmd.arg("record").arg("--call-graph=dwarf");
            if let Some(path) = output {
                cmd.arg("-o").arg(path);
            }
            cmd
        }
        Profiler::Samply => {
            let mut cmd = Command::new("samply");
            cmd.arg("record");
            if let Some(path) = output {
                cmd.arg("-o").arg(path);
            }
            cmd
        }
    };
    command.env(PROFILE_CHILD_ENV, "1");
    command.arg(exe).args(passthrough_args);
    command
}

fn run_profiled_inference(profiler: Profiler, output: Option<&PathBuf>) -> io::Result<ExitStatus> {
    let exe = std::env::current_exe()?;
    let passthrough_args = current_args_without_profile_flags();
    let mut command = profiler_command(profiler, output, &exe, &passthrough_args);
    command.status()
}

fn run_api_server_blocking(server_args: oxidize_server::Args) -> io::Result<()> {
    let rt = tokio::runtime::Runtime::new()
        .map_err(|error| io::Error::other(format!("tokio runtime: {error}")))?;
    rt.block_on(async move {
        let (effective_backend, warning) = server_args.backend.to_core_backend().effective();
        if let Some(msg) = warning {
            eprintln!("warning: {msg}");
        }
        eprintln!(
            "server: loading model={} backend={} addr={}:{}",
            server_args
                .model
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "<none>".to_string()),
            effective_backend.as_str(),
            server_args.host,
            server_args.port
        );
        let model = oxidize_server::load_model_runtime(&server_args).map_err(|error| {
            io::Error::other(format!("failed to initialize server model: {error}"))
        })?;
        let api_key = std::env::var("OXIDIZE_API_KEY")
            .ok()
            .filter(|value| !value.is_empty());
        let state = oxidize_server::AppState {
            limiter: Arc::new(oxidize_server::RequestLimiter::new(
                oxidize_server::RequestLimitConfig::default(),
            )),
            batcher: Arc::new(oxidize_server::ContinuousBatcher::default()),
            auth: oxidize_server::AuthConfig {
                api_key: api_key.map(Arc::<str>::from),
            },
            model,
            paged: None,
            mesh: None,
            audit: Arc::new(oxidize_server::audit::AuditLogger::new()),
            metrics: Arc::new(
                oxidize_server::metrics::MetricsRegistry::new()
                    .map_err(|error| io::Error::other(format!("metrics registry: {error}")))?,
            ),
        };
        let app = oxidize_server::build_app_with_state(state);
        let listener =
            tokio::net::TcpListener::bind(SocketAddr::new(server_args.host, server_args.port))
                .await
                .map_err(|error| io::Error::other(format!("failed to bind server: {error}")))?;
        eprintln!(
            "server: listening on http://{}:{} (REST /v1/*, WebSocket ws://{}:{}/v1/realtime)",
            server_args.host, server_args.port, server_args.host, server_args.port
        );
        let shutdown_signal = oxidize_server::shutdown::ShutdownSignal::new();
        oxidize_server::shutdown::serve_with_graceful_shutdown(listener, app, shutdown_signal)
            .await;
        Ok(())
    })
}

fn spawn_api_server_background(args: &Args) -> io::Result<()> {
    if args.model.is_none() {
        return Ok(());
    }
    let server_args = server_args_from_cli(args)?;
    let host = server_args.host;
    let port = server_args.port;
    std::thread::Builder::new()
        .name("oxidize-api".into())
        .spawn(move || {
            if let Err(error) = run_api_server_blocking(server_args) {
                eprintln!("api server failed: {error}");
            }
        })?;
    eprintln!(
        "api server: starting in background at http://{}:{} (REST /v1/*, WebSocket /v1/realtime)",
        host, port
    );
    Ok(())
}

fn server_backend_from_cli(backend: Backend) -> oxidize_server::Backend {
    match backend {
        Backend::Cpu => oxidize_server::Backend::Cpu,
        Backend::Metal => oxidize_server::Backend::Metal,
        Backend::Mlx => oxidize_server::Backend::Mlx,
        Backend::Cuda => oxidize_server::Backend::Cuda,
        Backend::Vulkan => oxidize_server::Backend::Vulkan,
        Backend::IntelArc => oxidize_server::Backend::IntelArc,
    }
}

fn server_args_from_cli(args: &Args) -> io::Result<oxidize_server::Args> {
    let host = args
        .api_host
        .parse::<IpAddr>()
        .map_err(|error| io::Error::other(format!("invalid --host/--api-host: {error}")))?;
    let model_id = args
        .model
        .as_ref()
        .and_then(|path| path.file_stem())
        .and_then(|stem| stem.to_str())
        .unwrap_or("oxidize-default")
        .to_string();
    Ok(oxidize_server::Args {
        host,
        port: args.api_port,
        model: args.model.clone(),
        backend: server_backend_from_cli(args.backend),
        batch_mode: oxidize_server::BatchMode::Sequential,
        model_id,
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        ctx_size: args.ctx_size,
        prefill_batch_size: 512,
        cpu_optimized: args.cpu_optimized,
        ram_offload: args.ram_offload,
        mmap_prefetch: args.mmap_prefetch,
        mmap_hugepages: args.mmap_hugepages,
        layer_wise: args.layer_wise,
        layer_cache: args.layer_cache,
        turboquant_kv: args.turboquant,
        no_turboquant_kv: args.no_turboquant,
        mesh: args.mesh,
        mesh_port: args.mesh_port,
        tokenizer_model: args.tokenizer_model.clone(),
        draft_model: args.draft_model.clone(),
        draft_tokens: args.draft_tokens,
        kv_cache_dtype: match args.kv_cache_dtype {
            KvCacheDType::F32 => oxidize_server::KvCacheDType::F32,
            KvCacheDType::F16 => oxidize_server::KvCacheDType::F16,
            KvCacheDType::Q8 => oxidize_server::KvCacheDType::Q8,
            KvCacheDType::Q4 => oxidize_server::KvCacheDType::Q4,
        },
        threads: args.threads.filter(|threads| *threads > 0).unwrap_or(0),
        ram_offload_threads: args.ram_offload_threads,
        auto: args.auto,
        no_auto: args.no_auto,
        print_plan: args.print_plan.clone(),
    })
}

fn run_api_server_in_process(args: &Args) -> io::Result<()> {
    run_api_server_blocking(server_args_from_cli(args)?)
}

fn optimize_mapped_model_memory(mapped: &MappedGgufFile, args: &Args) {
    let apply_hints =
        args.cpu_optimized || args.ram_offload || args.mmap_prefetch || args.mmap_hugepages;
    if !apply_hints {
        return;
    }

    if let Err(error) = mapped.advise_random_access() {
        eprintln!("mmap random-access hint failed: {error}");
    }
    if (args.cpu_optimized || args.ram_offload || args.mmap_prefetch)
        && let Err(error) = mapped.advise_will_need()
    {
        eprintln!("mmap prefetch hint failed: {error}");
    }
    if (args.cpu_optimized || args.mmap_hugepages)
        && let Err(error) = mapped.advise_huge_pages()
    {
        eprintln!("mmap hugepage hint failed: {error}");
    }
    if args.ram_offload {
        let n_threads = if args.ram_offload_threads > 0 {
            args.ram_offload_threads
        } else {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8)
        };
        let gib = mapped.bytes().len() as f64 / (1024.0 * 1024.0 * 1024.0);
        eprintln!(
            "ram offload: locking {gib:.2} GiB with {n_threads} threads (mlock + parallel prefault)"
        );
        let (mlocked, checksum, ms) = mapped.prefault_pages_locked(n_threads);
        let throughput = gib / (ms as f64 / 1000.0);
        let lock_status = if mlocked {
            "mlocked (pages pinned, eviction-proof)"
        } else {
            "MADV_WILLNEED only (model too large to mlock safely, or no CAP_IPC_LOCK)"
        };
        eprintln!(
            "ram offload: done in {ms}ms ({throughput:.1} GB/s) checksum={checksum:#04x} | {lock_status}"
        );
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
    let n_gpu_layers_set = user_passed_flag(&std::env::args().collect::<Vec<_>>(), "--n-gpu-layers");
    let kv_cache_dtype_set = user_passed_flag(&std::env::args().collect::<Vec<_>>(), "--kv-cache-dtype");
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
        oxidize_core::backend::Backend::Cpu => "CPU",
        oxidize_core::backend::Backend::Vulkan => "Vulkan GPU",
        oxidize_core::backend::Backend::IntelArc => "Intel Arc GPU (Vulkan)",
    };
    println!(
        "backend: {} ({})",
        effective_backend.as_str(),
        backend_label
    );
    let threads = if let Some(t) = args.threads.filter(|t| *t > 0) {
        t
    } else {
        // One worker per physical core: decode GEMV is DRAM-bound, so SMT
        // siblings add contention, not throughput (16 logical threads on an
        // 8-core part measures slower than 8).
        oxidize_core::spinpool::physical_core_count()
    };
    // Pin each rayon worker to one CPU in core-first order. Without this the
    // scheduler migrates workers between cores (and NUMA nodes) mid-token,
    // turning local DRAM streams into remote ones and defeating the hardware
    // prefetcher. Disable with OXIDIZE_NO_PIN=1.
    let pool_builder = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .start_handler(oxidize_core::spinpool::pin_to_slot);
    if let Err(error) = pool_builder.build_global() {
        eprintln!("failed to set rayon thread pool: {error}");
        return;
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
                // #region agent log
                let mapped_infos = mapped.mapped_tensor_infos();
                let architecture = mapped.parsed().architecture().unwrap_or("<none>");
                let has_lm_head = mapped_infos
                    .iter()
                    .any(|tensor| tensor.name == "lm_head.weight");
                let has_output = mapped_infos
                    .iter()
                    .any(|tensor| tensor.name == "output.weight");
                let has_embed_tokens = mapped_infos
                    .iter()
                    .any(|tensor| tensor.name == "model.embed_tokens.weight");
                let has_tok_embeddings = mapped_infos
                    .iter()
                    .any(|tensor| tensor.name == "tok_embeddings.weight");
                agent_debug_log_cli(
                    "H0_REPRO_PATH,H2_TENSOR_NAMES,H5_OUTPUT_PROJECTION",
                    "oxidize-cli/src/main.rs:run_model_mode",
                    "classified GGUF before CLI model construction",
                    &format!(
                        "{{\"architecture\":\"{}\",\"is_dflash\":{},\"tensor_count\":{},\"has_lm_head\":{},\"has_output\":{},\"has_embed_tokens\":{},\"has_tok_embeddings\":{}}}",
                        architecture,
                        is_dflash,
                        mapped_infos.len(),
                        has_lm_head,
                        has_output,
                        has_embed_tokens,
                        has_tok_embeddings
                    ),
                );
                // #endregion
                if args.ctx_size == Some(0) {
                    eprintln!("invalid --ctx-size: must be greater than 0");
                    return;
                }
                if is_dflash && args.draft_model.is_none() && !dflash_gguf_has_io_tensors(&mapped) {
                    agent_debug_log_cli(
                        "H5_OUTPUT_PROJECTION",
                        "oxidize-cli/src/main.rs:run_model_mode",
                        "rejecting standalone dflash draft as generation target",
                        "{\"reason\":\"dflash_requires_target_model_context\"}",
                    );
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
                    let available =
                        oxidize_core::gguf::linux_mem_available_bytes().unwrap_or(u64::MAX);
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
                            "external tokenizer model did not contain tokenizer metadata"
                                .to_string()
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

                    let draft_mapped = match loader.load(draft_model_path) {
                        Ok(mapped) => mapped,
                        Err(error) => {
                            eprintln!(
                                "failed to load DFlash draft model {}: {error}",
                                draft_model_path.display()
                            );
                            return;
                        }
                    };
                    let draft_arch = draft_mapped.parsed().architecture();
                    if !matches!(draft_arch, Some("dflash" | "dflash-draft")) {
                        eprintln!(
                            "--draft-model must point to a DFlash GGUF, got architecture {:?}",
                            draft_arch
                        );
                        return;
                    }
                    let draft_config = oxidize_core::dflash::DFlashConfig::from_gguf(&draft_mapped);
                    let mut draft_model =
                        match oxidize_core::dflash::DFlashDraftModel::load_from_gguf(
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
                        eprintln!(
                            "DFlash draft is incompatible with target (draft_hidden={}, target_hidden={}, draft_target_layers={:?}, target_layers={}); falling back to target-only generation",
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
                    if draft_model.vocab_size() != target_model.vocab_size() {
                        eprintln!(
                            "DFlash draft vocab ({}) does not match target vocab ({}) after borrowing target IO",
                            draft_model.vocab_size(),
                            target_model.vocab_size()
                        );
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
                    if concrete_model.has_mtp() && !args.no_mtp && !args.chat {
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
                    match oxidize_core::dflash::DFlashDraftModel::load_from_gguf(
                        &mapped,
                        dflash_config,
                    ) {
                        Ok(mut m) => {
                            if (!m.output.is_loaded() || !m.tok_embeddings.is_loaded())
                                && let Some(io_model_path) = args.tokenizer_model.as_deref()
                            {
                                match loader.load(io_model_path) {
                                    Ok(io_mapped) => {
                                        if let Err(error) = m.load_external_io_from_gguf(&io_mapped)
                                        {
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
                                eprintln!(
                                    "MLX initialization failed: {error}; falling back to CPU"
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

/// Apply the autotune plan to `args`. Only fills in fields the user
/// didn't explicitly set. Designed to be safe to call even when
/// the user has set most flags (those are left untouched).
fn apply_plan_to_args(
    args: &mut Args,
    plan: &oxidize_core::autotune::TuningPlan,
    inv: &oxidize_core::autotune::HardwareInventory,
) {
    let overrides = oxidize_core::autotune::overrides_from_plan(plan);
    // Threads: always fill in if user didn't pass --threads.
    if args.threads.is_none() {
        if let Some(t) = overrides.threads {
            if t > 0 {
                args.threads = Some(t);
            }
        }
    }
    // Ctx size: only if user didn't pass --ctx-size.
    if args.ctx_size.is_none() {
        if let Some(c) = overrides.ctx_size {
            if c > 0 {
                args.ctx_size = Some(c);
            }
        }
    }
    // n_gpu_layers: only if user didn't pass --n-gpu-layers.
    if !args.n_gpu_layers_set {
        if let Some(n) = overrides.n_gpu_layers {
            args.n_gpu_layers = n;
        }
    }
    // kv_cache_dtype: only if user didn't pass --kv-cache-dtype.
    if !args.kv_cache_dtype_set {
        use oxidize_core::tensor::DType;
        let desired = match plan.kv_cache_dtype {
            DType::F16 => KvCacheDType::F16,
            DType::F32 => KvCacheDType::F32,
            DType::I8 => KvCacheDType::Q8,
            DType::I16 => KvCacheDType::Q4,
            _ => KvCacheDType::F16,
        };
        args.kv_cache_dtype = desired;
    }
    // TurboQuant: only if user didn't pass either turboquant flag.
    if !args.turboquant && !args.no_turboquant {
        if let Some(true) = overrides.turboquant {
            args.turboquant = true;
        }
    }
    // layer_cache: only if user kept the default of 1.
    if args.layer_cache == 1 {
        if let Some(c) = overrides.layer_cache {
            if c > 0 && c != 1 {
                args.layer_cache = c;
            }
        }
    }
    // layer_wise: only if user kept the default of false AND the plan
    // recommends it. Documented as best-effort: we can't distinguish
    // `--no-layer-wise` from "user didn't set", so a user who
    // explicitly wants to disable layer_wise should use --no-auto.
    if !args.layer_wise {
        if let Some(true) = overrides.layer_wise {
            args.layer_wise = true;
        }
    }
    // cpu_optimized: never auto-enable (it caps ctx to 2048 and
    // disables the existing auto-cap; it would silently override
    // a lot of user intent). The plan still hints via rationale.
    // ram_offload + mmap hints: best-effort, same caveat.
    if !args.ram_offload {
        if let Some(true) = overrides.ram_offload {
            args.ram_offload = true;
        }
    }
    if !args.mmap_hugepages {
        if let Some(true) = overrides.mmap_hugepages {
            args.mmap_hugepages = true;
        }
    }
    if !args.mmap_prefetch {
        if let Some(true) = overrides.mmap_prefetch {
            args.mmap_prefetch = true;
        }
    }
    eprintln!(
        "[oxidize auto-tune] applied: threads={:?} ctx={:?} n_gpu_layers={} kv={:?} layer_wise={} layer_cache={} turboquant={} (cores={} ram={} GiB gpu={} MiB)",
        args.threads,
        args.ctx_size,
        args.n_gpu_layers,
        args.kv_cache_dtype,
        args.layer_wise,
        args.layer_cache,
        args.turboquant,
        inv.physical_cores,
        inv.total_ram_bytes / (1u64 << 30),
        inv.gpu_vram_bytes / (1024 * 1024),
    );
}

/// JSON-friendly snapshot of a `TuningPlan` for tooling.
fn plan_to_json(plan: &oxidize_core::autotune::TuningPlan) -> serde_json::Value {
    use oxidize_core::autotune::{OxkIsa, OxkTile, PipelineMode, SpeculativeSpec};
    let isa = match plan.oxk_isa {
        OxkIsa::Scalar => "scalar",
        OxkIsa::Avx2 => "avx2",
        OxkIsa::Avx512 => "avx512",
    };
    let tile = match plan.oxk_tile {
        OxkTile::T1 => 1,
        OxkTile::T4 => 4,
        OxkTile::T8 => 8,
        OxkTile::T16 => 16,
    };
    let pipe = match plan.pipeline {
        PipelineMode::Sequential => "sequential",
        PipelineMode::Continuous => "continuous",
        PipelineMode::Paged => "paged",
        PipelineMode::Asymmetric => "asymmetric",
    };
    let spec = match plan.speculative {
        SpeculativeSpec::None => "none",
        SpeculativeSpec::DFlash => "dflash",
        SpeculativeSpec::Mtp => "mtp",
    };
    serde_json::json!({
        "threads": plan.threads,
        "ctx_size": plan.ctx_size,
        "kv_cache_dtype": format!("{:?}", plan.kv_cache_dtype),
        "n_gpu_layers": plan.n_gpu_layers,
        "mmap": plan.mmap,
        "mlock": plan.mlock,
        "mmap_hugepages": plan.mmap_hugepages,
        "mmap_prefetch": plan.mmap_prefetch,
        "numa_replicate_dense": plan.numa_replicate_dense,
        "layer_wise": plan.layer_wise,
        "layer_cache": plan.layer_cache,
        "pipeline": pipe,
        "speculative": spec,
        "decode_tile_tokens": plan.decode_tile_tokens,
        "oxk_isa": isa,
        "oxk_tile": tile,
        "expected_prompt_tps": plan.expected_prompt_tps,
        "expected_decode_tps": plan.expected_decode_tps,
        "rationale": plan.rationale,
    })
}

/// True if stdout is attached to a terminal (best-effort: uses
/// `std::io::IsTerminal` from stdlib).
fn atty_stdout() -> bool {
    std::io::stdout().is_terminal()
}

/// Run the CLI in distributed mesh node mode.
/// Delegates to `oxidize_core::mesh::run_mesh_node` which builds the
/// libp2p swarm, starts mDNS, subscribes to all 6 GossipSub topics, and
/// drives the event loop.
fn run_mesh_mode(mesh_port: u16) -> io::Result<()> {
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| io::Error::other(format!("tokio runtime: {e}")))?;
    rt.block_on(async {
        oxidize_core::mesh::run_mesh_node(mesh_port, None, None, None)
            .await
            .map_err(|e| io::Error::other(format!("mesh node error: {e}")))
    })
}

/// Run the CLI in mesh + chat combined mode.
///
/// Starts a background mesh node, waits for leader election, then opens an
/// interactive REPL.  Each user prompt is broadcast to the mesh master via
/// GossipSub `COMMANDS` and response tokens stream back through the mesh data
/// plane (real or local fallback) and are printed token-by-token.
fn run_mesh_chat_mode(mesh_port: u16) -> io::Result<()> {
    use oxidize_core::mesh::{MeshChatPrompt, MeshChatToken};

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| io::Error::other(format!("tokio runtime: {e}")))?;

    let (prompt_tx, prompt_rx) = tokio::sync::mpsc::unbounded_channel::<MeshChatPrompt>();
    let (token_tx, mut token_rx) = tokio::sync::mpsc::unbounded_channel::<MeshChatToken>();

    // Spawn the mesh node in a background task within the same runtime.
    let mesh_handle = rt.spawn(async move {
        let result =
            oxidize_core::mesh::run_mesh_node(mesh_port, None, Some(prompt_rx), Some(token_tx))
                .await;
        if let Err(ref e) = result {
            eprintln!("mesh node error: {e}");
        }
        result
    });

    // Give the mesh node a moment to start up and discover peers.
    rt.block_on(async {
        tokio::time::sleep(Duration::from_secs(2)).await;
    });

    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let stdout = io::stdout();
    let mut writer = stdout.lock();

    writeln!(writer, "oxidize-cli mesh chat mode. type 'exit' to quit.")?;
    let mut history = ConversationHistory::default();
    let mut prompt_cache = PromptCache::default();
    let mut request_counter: usize = 0;

    loop {
        write!(writer, "> ")?;
        writer.flush()?;

        let Some(input) = read_chat_prompt(&mut reader, &mut writer)? else {
            break;
        };

        let prompt = input.trim();
        if prompt.eq_ignore_ascii_case("exit") || prompt.eq_ignore_ascii_case("quit") {
            writeln!(writer, "bye")?;
            break;
        }
        if prompt.is_empty() {
            continue;
        }
        if prompt.eq_ignore_ascii_case("/history") {
            writeln!(writer, "{}", history.render())?;
            continue;
        }
        if prompt.eq_ignore_ascii_case("/clear") {
            history.clear();
            writeln!(writer, "conversation history cleared")?;
            continue;
        }

        let cached = prompt_cache.get(prompt);
        let response = if let Some(cached) = cached {
            writeln!(writer, "{cached}")?;
            writeln!(
                writer,
                "generation stats: tokens=0 speed=0.00 tok/s (cache hit)"
            )?;
            cached.to_owned()
        } else {
            request_counter += 1;
            let request_id = format!("cli-{}", request_counter);

            // Broadcast the prompt to the mesh via the chat engine.
            let mesh_prompt = MeshChatPrompt {
                request_id: request_id.clone(),
                prompt: prompt.to_string(),
                max_tokens: 8, // short deterministic tokens for the demo
                temperature: 0.0,
                top_p: 0.0,
            };
            if prompt_tx.send(mesh_prompt).is_err() {
                writeln!(writer, "mesh prompt channel closed")?;
                break;
            }

            // Drain streaming tokens from the mesh data plane.
            let mut token_count = 0usize;
            let mut response_parts = Vec::new();
            let start = Instant::now();

            // Give the master a moment to receive the prompt and start generating.
            std::thread::sleep(Duration::from_millis(100));

            loop {
                match token_rx.try_recv() {
                    Ok(token) => {
                        if token.request_id == request_id {
                            write!(writer, "{}", token.token)?;
                            writer.flush()?;
                            token_count += 1;
                            response_parts.push(token.token);
                            if token.is_final {
                                writeln!(writer)?;
                                break;
                            } else {
                                write!(writer, " ")?;
                            }
                            // Tiny artificial pacing so the TUI shows progress.
                            std::thread::sleep(Duration::from_millis(20));
                        }
                    }
                    Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                        // Poll briefly then give up if nothing arrives.
                        std::thread::sleep(Duration::from_millis(50));
                        if start.elapsed() > Duration::from_secs(5) {
                            // Timeout waiting for tokens.
                            break;
                        }
                    }
                    Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                        writeln!(writer, "mesh token channel disconnected")?;
                        break;
                    }
                }
            }

            let elapsed = start.elapsed().as_secs_f64();
            let speed = if elapsed > 0.0 {
                token_count as f64 / elapsed
            } else {
                0.0
            };
            writeln!(
                writer,
                "generation stats: tokens={token_count} speed={speed:.2} tok/s (mesh)"
            )?;
            let response = response_parts.join(" ");
            prompt_cache.insert(prompt, &response);
            response
        };

        history.add_turn(prompt, &response);
    }

    // Abort the mesh node when chat exits.
    mesh_handle.abort();
    let _ = rt.block_on(mesh_handle);
    Ok(())
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
