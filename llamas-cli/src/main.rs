use clap::Parser;
use llamas_core::model_loader::{GgufModelLoader, ModelLoader};
use llamas_core::offload::{LayerOffloadPlan, plan_layer_offload};
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

fn main() {
    let args = Args::parse();
    if let Some(model_path) = args.model {
        let loader = GgufModelLoader;
        match loader.load(model_path) {
            Ok(mapped) => {
                let plan = plan_layer_offload(&mapped.parsed().tensor_infos, args.n_gpu_layers);
                println!("{}", render_offload_plan(&plan));
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
}
