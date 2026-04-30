use clap::Parser;
use llamas_core::lora::{plan_lora_application, AdapterKind, LoraPlan};
use llamas_core::model_loader::{GgufModelLoader, LoadProgress, ModelLoader};
use llamas_core::offload::{
    plan_layer_offload, plan_multi_gpu_offload, LayerOffloadPlan, MultiGpuConfig,
    MultiGpuOffloadPlan, ParallelismStrategy,
};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "llamas-cli")]
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
}

fn greeting(prompt: &str) -> String {
    format!("llamas-cli: {prompt}")
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

fn run_chat_mode<R: BufRead, W: Write>(reader: &mut R, writer: &mut W) -> io::Result<()> {
    writeln!(writer, "llamas-cli chat mode. type 'exit' to quit.")?;
    let mut history = ConversationHistory::default();
    loop {
        write!(writer, "> ")?;
        writer.flush()?;

        let mut input = String::new();
        if reader.read_line(&mut input)? == 0 {
            break;
        }

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

        let response = write_generated_response(prompt, writer)?;
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
    let response = greeting(prompt);
    let response_tokens = response.split_whitespace().count();
    if response_tokens == 0 {
        writeln!(writer, "{response}")?;
        return Ok(response);
    }
    for generated in 1..=response_tokens {
        writeln!(
            writer,
            "generation progress: {generated}/{response_tokens} tokens"
        )?;
    }
    writeln!(writer, "{response}")?;
    Ok(response)
}

fn main() {
    let args = Args::parse();
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
        match loader.load_with_progress(model_path, |progress| {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greeting_uses_prompt() {
        assert_eq!(greeting("test"), "llamas-cli: test");
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
                llamas_core::offload::GpuAssignment {
                    gpu_index: 0,
                    layer_count: 3,
                    tensor_count: 18,
                },
                llamas_core::offload::GpuAssignment {
                    gpu_index: 1,
                    layer_count: 3,
                    tensor_count: 18,
                },
            ],
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
            targets: vec![llamas_core::lora::LoraTarget {
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
        assert!(output.contains("llamas-cli chat mode. type 'exit' to quit."));
        assert!(output.contains("generation progress: 1/2 tokens"));
        assert!(output.contains("llamas-cli: hello"));
        assert!(output.contains("> bye"));
    }

    #[test]
    fn chat_mode_ignores_blank_lines() {
        let mut reader = io::Cursor::new("\nworld\nquit\n");
        let mut writer = Vec::new();
        run_chat_mode(&mut reader, &mut writer).expect("chat mode should succeed");

        let output = String::from_utf8(writer).expect("valid utf8 output");
        assert!(output.contains("llamas-cli: world"));
    }

    #[test]
    fn chat_mode_renders_and_clears_conversation_history() {
        let mut reader = io::Cursor::new("hello\nworld\n/history\n/clear\n/history\nquit\n");
        let mut writer = Vec::new();
        run_chat_mode(&mut reader, &mut writer).expect("chat mode should succeed");

        let output = String::from_utf8(writer).expect("valid utf8 output");
        assert!(output.contains("1. user: hello\n   assistant: llamas-cli: hello"));
        assert!(output.contains("2. user: world\n   assistant: llamas-cli: world"));
        assert!(output.contains("conversation history cleared"));
        assert!(output.contains("no conversation history"));
    }

    #[test]
    fn conversation_history_render_uses_empty_state() {
        let history = ConversationHistory::default();
        assert_eq!(history.render(), "no conversation history");
    }

    #[test]
    fn single_shot_mode_writes_one_response() {
        let mut writer = Vec::new();
        run_single_shot_mode("hello", &mut writer).expect("single-shot mode should succeed");
        let output = String::from_utf8(writer).expect("valid utf8 output");
        assert_eq!(
            output,
            "generation progress: 1/2 tokens\ngeneration progress: 2/2 tokens\nllamas-cli: hello\n"
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
        assert_eq!(
            rendered,
            "load progress: 35% stage=mapping bytes=1024/4096"
        );
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
        let response =
            write_generated_response("there", &mut writer).expect("generation should succeed");
        let output = String::from_utf8(writer).expect("valid utf8 output");
        assert_eq!(response, "llamas-cli: there");
        assert_eq!(
            output,
            "generation progress: 1/2 tokens\ngeneration progress: 2/2 tokens\nllamas-cli: there\n"
        );
    }
}
