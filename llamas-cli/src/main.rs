use clap::Parser;
use llamas_core::model_loader::{GgufModelLoader, ModelLoader};
use llamas_core::offload::{
    LayerOffloadPlan, MultiGpuConfig, MultiGpuOffloadPlan, ParallelismStrategy, plan_layer_offload,
    plan_multi_gpu_offload,
};
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
}

fn greeting(prompt: &str) -> String {
    format!("llamas-cli: {prompt}")
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

fn main() {
    let args = Args::parse();
    if let Some(model_path) = args.model {
        let loader = GgufModelLoader;
        match loader.load(model_path) {
            Ok(mapped) => {
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
                        Err(error) => eprintln!("failed to build multi-gpu offload plan: {error:?}"),
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
    println!("{}", greeting(&args.prompt));
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
        assert_eq!(parse_parallelism("tensor"), Some(ParallelismStrategy::Tensor));
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
}
