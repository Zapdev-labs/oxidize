use clap::{Parser, ValueEnum};
use oxidize_core::generation::{GenerationConfig, GenerationStream};
use oxidize_core::inference::{InferenceConfig, InferenceModel};
use oxidize_core::lora::{AdapterKind, LoraPlan, plan_lora_application};
use oxidize_core::model::{Model, Session};
use oxidize_core::model_loader::{GgufModelLoader, LoadProgress, ModelLoader};
use oxidize_core::offload::{
    LayerOffloadPlan, MultiGpuConfig, MultiGpuOffloadPlan, ParallelismStrategy, plan_layer_offload,
    plan_multi_gpu_offload,
};
use oxidize_core::sampling::SamplingConfig;
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
}

fn greeting(prompt: &str) -> String {
    format!("oxidize-cli: {prompt}")
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum Profiler {
    Perf,
    Samply,
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

fn main() {
    let args = Args::parse();
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
    if let Some(model_path) = args.model {
        let loader = GgufModelLoader;
        match loader.load_with_progress(&model_path, |progress| {
            println!("{}", render_load_progress(progress))
        }) {
            Ok(mapped) => {
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
                let vocab_size = metadata_u32(metadata, "llama.vocab_size")
                    .or_else(|| metadata_u32(metadata, "qwen35.vocab_size"))
                    .or_else(|| metadata_u32(metadata, "qwen2.vocab_size"))
                    .or_else(|| metadata_u32(metadata, "qwen.vocab_size"))
                    .or_else(|| metadata_u32(metadata, "general.vocab_size"))
                    .or_else(|| metadata_u32(metadata, "tokenizer.ggml.tokens.count"))
                    .or_else(|| {
                        tensor_dims(&mapped, "tok_embeddings.weight")
                            .and_then(|d| d.get(1).copied())
                            .map(|v| v as u32)
                    })
                    .or_else(|| {
                        tensor_dims(&mapped, "token_embd.weight")
                            .and_then(|d| d.get(1).copied())
                            .map(|v| v as u32)
                    })
                    .unwrap_or(32000) as usize;
                let context_size = metadata_u32(metadata, "llama.context_length")
                    .or_else(|| metadata_u32(metadata, "qwen35.context_length"))
                    .or_else(|| metadata_u32(metadata, "qwen2.context_length"))
                    .or_else(|| metadata_u32(metadata, "qwen.context_length"))
                    .or_else(|| metadata_u32(metadata, "gemma4.context_length"))
                    .or_else(|| metadata_u32(metadata, "gemma.context_length"))
                    .or_else(|| metadata_u32(metadata, "llama.embedding_length"))
                    .unwrap_or(4096) as usize;
                let layer_count = metadata_u32(metadata, "llama.block_count")
                    .or_else(|| metadata_u32(metadata, "qwen35.block_count"))
                    .or_else(|| metadata_u32(metadata, "qwen2.block_count"))
                    .or_else(|| metadata_u32(metadata, "qwen.block_count"))
                    .or_else(|| metadata_u32(metadata, "gemma4.block_count"))
                    .or_else(|| metadata_u32(metadata, "gemma.block_count"))
                    .unwrap_or(32) as usize;
                let hidden_size = metadata_u32(metadata, "llama.embedding_length")
                    .or_else(|| metadata_u32(metadata, "qwen35.embedding_length"))
                    .or_else(|| metadata_u32(metadata, "qwen2.embedding_length"))
                    .or_else(|| metadata_u32(metadata, "qwen.embedding_length"))
                    .or_else(|| metadata_u32(metadata, "gemma4.embedding_length"))
                    .or_else(|| metadata_u32(metadata, "gemma.embedding_length"))
                    .or_else(|| {
                        tensor_dims(&mapped, "tok_embeddings.weight")
                            .and_then(|d| d.first().copied())
                            .map(|v| v as u32)
                    })
                    .or_else(|| {
                        tensor_dims(&mapped, "token_embd.weight")
                            .and_then(|d| d.first().copied())
                            .map(|v| v as u32)
                    })
                    .unwrap_or(4096) as usize;
                let intermediate_size = metadata_u32(metadata, "llama.feed_forward_length")
                    .or_else(|| metadata_u32(metadata, "qwen35.feed_forward_length"))
                    .or_else(|| metadata_u32(metadata, "qwen2.feed_forward_length"))
                    .or_else(|| metadata_u32(metadata, "qwen.feed_forward_length"))
                    .or_else(|| metadata_u32(metadata, "gemma4.feed_forward_length"))
                    .or_else(|| metadata_u32(metadata, "gemma.feed_forward_length"))
                    .unwrap_or(11008) as usize;
                let num_attention_heads = metadata_u32(metadata, "llama.attention.head_count")
                    .or_else(|| metadata_u32(metadata, "qwen35.attention.head_count"))
                    .or_else(|| metadata_u32(metadata, "qwen2.attention.head_count"))
                    .or_else(|| metadata_u32(metadata, "qwen.attention.head_count"))
                    .or_else(|| metadata_u32(metadata, "gemma4.attention.head_count"))
                    .or_else(|| metadata_u32(metadata, "gemma.attention.head_count"))
                    .unwrap_or(32) as usize;
                let num_key_value_heads = metadata_u32(metadata, "llama.attention.head_count_kv")
                    .or_else(|| metadata_u32(metadata, "qwen35.attention.head_count_kv"))
                    .or_else(|| metadata_u32(metadata, "qwen2.attention.head_count_kv"))
                    .or_else(|| metadata_u32(metadata, "qwen.attention.head_count_kv"))
                    .or_else(|| metadata_u32(metadata, "gemma4.attention.head_count_kv"))
                    .or_else(|| metadata_u32(metadata, "gemma.attention.head_count_kv"))
                    .unwrap_or(num_attention_heads as u32)
                    as usize;
                let key_value_head_dim = first_layer_tensor_dims(&mapped, "attn_k.weight")
                    .and_then(|d| d.get(1).copied())
                    .and_then(|width| width.checked_div(num_key_value_heads as u64))
                    .and_then(|value| value.try_into().ok())
                    .or_else(|| metadata_u32(metadata, "llama.attention.key_length"))
                    .or_else(|| metadata_u32(metadata, "qwen35.attention.key_length"))
                    .or_else(|| metadata_u32(metadata, "qwen2.attention.key_length"))
                    .or_else(|| metadata_u32(metadata, "qwen.attention.key_length"))
                    .or_else(|| {
                        first_layer_tensor_dims(&mapped, "attn_k.weight")
                            .and_then(|d| d.get(1).copied())
                            .and_then(|width| width.checked_div(num_key_value_heads as u64))
                            .and_then(|value| value.try_into().ok())
                    })
                    .unwrap_or((hidden_size / num_attention_heads) as u32)
                    as usize;
                let rms_norm_eps = metadata_f32(metadata, "llama.attention.layer_norm_rms_epsilon")
                    .or_else(|| metadata_f32(metadata, "qwen35.attention.layer_norm_rms_epsilon"))
                    .or_else(|| metadata_f32(metadata, "qwen2.attention.layer_norm_rms_epsilon"))
                    .or_else(|| metadata_f32(metadata, "qwen.attention.layer_norm_rms_epsilon"))
                    .or_else(|| metadata_f32(metadata, "gemma4.attention.layer_norm_rms_epsilon"))
                    .or_else(|| metadata_f32(metadata, "gemma.attention.layer_norm_rms_epsilon"))
                    .unwrap_or(1e-5);
                let rope_theta = metadata_f32(metadata, "llama.rope.freq_base")
                    .or_else(|| metadata_f32(metadata, "qwen35.rope.freq_base"))
                    .or_else(|| metadata_f32(metadata, "qwen2.rope.freq_base"))
                    .or_else(|| metadata_f32(metadata, "qwen.rope.freq_base"))
                    .or_else(|| metadata_f32(metadata, "gemma4.rope.freq_base"))
                    .or_else(|| metadata_f32(metadata, "gemma.rope.freq_base"))
                    .unwrap_or(10000.0);

                let config = InferenceConfig {
                    vocab_size,
                    context_size,
                    layer_count,
                    hidden_size,
                    intermediate_size,
                    num_attention_heads,
                    num_key_value_heads,
                    key_value_head_dim,
                    rms_norm_eps,
                    rope_theta,
                };
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
                let mut model = match InferenceModel::load_from_gguf(&mapped, config) {
                    Ok(m) => m,
                    Err(error) => {
                        eprintln!("failed to load model weights: {error}");
                        return;
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

fn metadata_u32(
    metadata: &std::collections::BTreeMap<String, oxidize_core::gguf::GgufMetadataValue>,
    key: &str,
) -> Option<u32> {
    use oxidize_core::gguf::GgufMetadataValue;
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

fn tensor_dims(mapped: &oxidize_core::gguf::MappedGgufFile, name: &str) -> Option<Vec<u64>> {
    mapped
        .mapped_tensor_infos()
        .iter()
        .find(|t| t.name == name)
        .map(|t| t.dimensions.clone())
}

fn first_layer_tensor_dims(
    mapped: &oxidize_core::gguf::MappedGgufFile,
    suffix: &str,
) -> Option<Vec<u64>> {
    mapped
        .mapped_tensor_infos()
        .iter()
        .find(|t| t.name.starts_with("blk.") && t.name.ends_with(suffix))
        .map(|t| t.dimensions.clone())
}

fn metadata_f32(
    metadata: &std::collections::BTreeMap<String, oxidize_core::gguf::GgufMetadataValue>,
    key: &str,
) -> Option<f32> {
    use oxidize_core::gguf::GgufMetadataValue;
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
