mod pipeline;

use clap::{Parser, ValueEnum};
use oxidize_core::generation::{GenerationConfig, GenerationStream};
use oxidize_core::gguf::MappedGgufFile;
use oxidize_core::inference::{InferenceConfig, InferenceModel};
use oxidize_core::lora::{AdapterKind, LoraPlan, plan_lora_application};
use oxidize_core::model::{Model, Session};
use oxidize_core::model_loader::{GgufModelLoader, LoadProgress, ModelLoader};
use oxidize_core::offload::{
    LayerOffloadPlan, MultiGpuConfig, MultiGpuOffloadPlan, ParallelismStrategy, plan_layer_offload,
    plan_multi_gpu_offload,
};
use oxidize_core::sampling::SamplingConfig;
use oxidize_core::tensor::DType;
use oxidize_core::tokenizer::{EncodeOptions, LoadedTokenizer, load_tokenizer_from_gguf_metadata};

use std::collections::{HashMap, HashSet};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::process::{Command, ExitStatus};
use std::sync::Arc;
use std::task::Wake;
use std::time::{Duration, Instant};

const PROFILE_CHILD_ENV: &str = "OXIDIZE_PROFILE_CHILD";

#[derive(Debug, Parser)]
#[command(name = "oxidize-cli")]
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
    #[arg(long, default_value_t = false)]
    turboquant: bool,
    #[arg(long, default_value_t = false)]
    cpu_optimized: bool,
    #[arg(long, default_value_t = false)]
    ram_offload: bool,
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
}

impl Backend {
    fn to_core_backend(self) -> oxidize_core::backend::Backend {
        match self {
            Backend::Cpu => oxidize_core::backend::Backend::Cpu,
            Backend::Metal => oxidize_core::backend::Backend::Metal,
            Backend::Mlx => oxidize_core::backend::Backend::Mlx,
            Backend::Cuda => oxidize_core::backend::Backend::Cuda,
            Backend::Vulkan => oxidize_core::backend::Backend::Vulkan,
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

#[allow(clippy::too_many_arguments)]
fn generate_with_model<W: Write, M: Model>(
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
            add_bos: true,
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

    // Decode generated tokens back to text
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
        let started = Instant::now();
        let checksum = mapped.prefault_pages();
        println!(
            "ram offload: prefaulted {:.2} GiB into page cache in {:.2?} (checksum={checksum})",
            mapped.bytes().len() as f64 / 1024.0 / 1024.0 / 1024.0,
            started.elapsed()
        );
    }
}

fn main() {
    let args = Args::parse();
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
    };
    println!(
        "backend: {} ({})",
        effective_backend.as_str(),
        backend_label
    );
    let threads = if let Some(t) = args.threads.filter(|t| *t > 0) {
        t
    } else {
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(8)
    };
    if let Err(error) = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
    {
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
        if let Err(e) =
            pipeline::run_head(&model, &peer, &args.prompt, args.pipe_max_tokens, true)
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
    if args.chat {
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
    if let Some(model_path) = args.model.as_ref() {
        let loader = GgufModelLoader;
        match loader.load_with_progress(model_path, |progress| {
            println!("{}", render_load_progress(progress))
        }) {
            Ok(mapped) => {
                optimize_mapped_model_memory(&mapped, &args);
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
                if mapped.parsed().architecture() == Some("dflash-draft") {
                    eprintln!(
                        "DFlash draft GGUF loaded successfully, but it is not a standalone text model: it has no token embeddings/lm_head. Use `oxidize-bench --model {}` to benchmark draft forward passes, or pair it with a target model for speculative decoding.",
                        model_path.display()
                    );
                    return;
                }
                if args.ctx_size == Some(0) {
                    eprintln!("invalid --ctx-size: must be greater than 0");
                    return;
                }
                let mut config = InferenceConfig::from_gguf(&mapped);
                config.kv_cache_dtype = args.kv_cache_dtype.dtype();
                if args.turboquant {
                    config.kv_quantization =
                        oxidize_core::kv_cache::KvQuantization::TurboQuant;
                }
                if let Some(ctx) = args.ctx_size {
                    config.context_size = ctx;
                }
                if args.cpu_optimized {
                    config.context_size = config.context_size.min(2048);
                }
                // Load tokenizer from GGUF metadata
                let tokenizer = match load_tokenizer_from_gguf_metadata(metadata) {
                    Ok(t) => t,
                    Err(error) => {
                        eprintln!("failed to load tokenizer: {error:?}");
                        return;
                    }
                };

                let stdout = io::stdout();
                let mut writer = stdout.lock();
                let mut model: Box<dyn Model> = if args.layer_wise {
                    match oxidize_core::layer_wise::LayerWiseModel::load_from_gguf(
                        &mapped,
                        config,
                        args.layer_cache,
                    ) {
                        Ok(m) => Box::new(m),
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
                                let use_mmap = args.cpu_optimized;
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
                        let use_mmap = args.cpu_optimized;
                        match InferenceModel::load_from_gguf(&mapped, config, use_mmap) {
                            Ok(m) => Box::new(m),
                            Err(error) => {
                                eprintln!("failed to load model weights: {error}");
                                return;
                            }
                        }
                    }
                } else {
                    let use_mmap = args.cpu_optimized;
                    match InferenceModel::load_from_gguf(&mapped, config, use_mmap) {
                        Ok(m) => Box::new(m),
                        Err(error) => {
                            eprintln!("failed to load model weights: {error}");
                            return;
                        }
                    }
                };

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
            Err(error) => eprintln!("failed to load model: {error}"),
        }
        return;
    }
    let stdout = io::stdout();
    let mut writer = stdout.lock();
    if let Err(error) = run_single_shot_mode(&args.prompt, &mut writer) {
        eprintln!("single-shot mode failed: {error}");
    }
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
