use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use oxidize_train::{TrainingConfig, load_csv_dataset, train_classifier};

#[derive(Debug, Parser)]
#[command(name = "oxidize-train")]
struct Args {
    #[arg(long)]
    train_csv: PathBuf,
    #[arg(long)]
    label_column: Option<usize>,
    #[arg(long, default_value_t = 20)]
    epochs: usize,
    #[arg(long, default_value_t = 32)]
    batch_size: usize,
    #[arg(long, default_value_t = 1e-3)]
    learning_rate: f32,
    #[arg(long, default_value_t = 0.01)]
    weight_decay: f32,
    #[arg(long, default_value_t = 128)]
    hidden_size: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let dataset = load_csv_dataset(&args.train_csv, args.label_column)?;
    let (_, report) = train_classifier(
        &dataset,
        TrainingConfig {
            epochs: args.epochs,
            batch_size: args.batch_size,
            learning_rate: args.learning_rate,
            weight_decay: args.weight_decay,
            hidden_size: args.hidden_size,
            seed: args.seed,
        },
    )?;

    println!("oxidize-train: samples={}", report.samples);
    println!("oxidize-train: features={}", report.features);
    println!("oxidize-train: classes={}", report.classes);
    println!("oxidize-train: final_loss={:.6}", report.final_loss);
    println!("oxidize-train: accuracy={:.4}", report.accuracy);
    Ok(())
}
