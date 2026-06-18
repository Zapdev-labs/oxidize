use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Parser, Subcommand, ValueEnum};
use oxidize_train::video::{
    FrameConfig, LabelTask, VideoTrainingConfig, build_prototype, default_model_path,
    ffmpeg_available, load_dataset, save_model, train_video_classifier,
};
use oxidize_train::{TrainingConfig, load_csv_dataset, train_classifier};

#[derive(Debug, Parser)]
#[command(
    name = "oxidize-train",
    about = "Train classifiers on CPU: tabular CSV rows or short-video clips"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Train the MLP classifier on a CSV dataset.
    Csv(CsvArgs),
    /// Train a video classifier on a directory of clips + JSON metadata.
    Video(VideoArgs),
    /// Render a "base" prototype clip by averaging selected creators' frames.
    Prototype(PrototypeArgs),
}

#[derive(Debug, Args)]
struct CsvArgs {
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

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TaskArg {
    /// Predict which creator/account a clip is from.
    Creator,
    /// Predict a view-count bucket (how viral).
    Virality,
    /// Predict a like/view engagement bucket.
    Engagement,
}

impl From<TaskArg> for LabelTask {
    fn from(value: TaskArg) -> Self {
        match value {
            TaskArg::Creator => LabelTask::Creator,
            TaskArg::Virality => LabelTask::Virality,
            TaskArg::Engagement => LabelTask::Engagement,
        }
    }
}

#[derive(Debug, Args)]
struct VideoArgs {
    /// Root directory holding clips and `*_metadata.json` files.
    #[arg(long)]
    data: PathBuf,
    /// Frame cache directory (default: `<data>/.oxidize-frames`).
    #[arg(long)]
    cache: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = TaskArg::Creator)]
    task: TaskArg,
    /// Frames sampled per clip.
    #[arg(long, default_value_t = 8)]
    frames: usize,
    #[arg(long = "frame-size", default_value_t = 64)]
    frame_size: usize,
    #[arg(long = "patch-size", default_value_t = 16)]
    patch_size: usize,
    #[arg(long = "embed-dim", default_value_t = 128)]
    embed_dim: usize,
    #[arg(long = "hidden-size", default_value_t = 256)]
    hidden_size: usize,
    #[arg(long, default_value_t = 30)]
    epochs: usize,
    #[arg(long = "batch-size", default_value_t = 16)]
    batch_size: usize,
    #[arg(long = "learning-rate", default_value_t = 1e-3)]
    learning_rate: f32,
    #[arg(long = "weight-decay", default_value_t = 0.01)]
    weight_decay: f32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Fraction of clips held out for validation.
    #[arg(long = "val-split", default_value_t = 0.15)]
    val_split: f32,
    /// Quantile buckets for virality/engagement tasks.
    #[arg(long, default_value_t = 3)]
    buckets: usize,
    /// Optional cap on number of clips (for quick runs).
    #[arg(long = "max-videos")]
    max_videos: Option<usize>,
    /// Downsample each class to the smallest class size before training.
    #[arg(long)]
    balance: bool,
    /// Where to write the trained model (JSON).
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct PrototypeArgs {
    /// Root directory holding clips and `*_metadata.json` files.
    #[arg(long)]
    data: PathBuf,
    /// Frame cache directory (default: `<data>/.oxidize-frames`).
    #[arg(long)]
    cache: Option<PathBuf>,
    /// Creator(s) to exclude from the average (repeatable).
    #[arg(long)]
    exclude: Vec<String>,
    /// Frames sampled per clip (temporal slots in the base video).
    #[arg(long, default_value_t = 8)]
    frames: usize,
    #[arg(long = "frame-size", default_value_t = 64)]
    frame_size: usize,
    /// Output resolution of the rendered base video (square).
    #[arg(long, default_value_t = 256)]
    upscale: u32,
    /// Output frame rate (smoothed via interpolation).
    #[arg(long, default_value_t = 24)]
    fps: u32,
    /// Where to write the base video (mp4).
    #[arg(long)]
    out: PathBuf,
    /// Optional contact-sheet PNG of the averaged frames.
    #[arg(long)]
    sheet: Option<PathBuf>,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Csv(args) => run_csv(args),
        Command::Video(args) => run_video(args),
        Command::Prototype(args) => run_prototype(args),
    }
}

fn run_csv(args: CsvArgs) -> Result<()> {
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

fn run_video(args: VideoArgs) -> Result<()> {
    if !ffmpeg_available() {
        eprintln!(
            "oxidize-train video: warning — `ffmpeg` not found on PATH; only already-cached frames can be used."
        );
    }

    let config = VideoTrainingConfig {
        data_root: args.data,
        cache_dir: args.cache.unwrap_or_default(),
        task: args.task.into(),
        frames: FrameConfig {
            num_frames: args.frames,
            frame_size: args.frame_size,
            patch_size: args.patch_size,
        },
        embed_dim: args.embed_dim,
        hidden_size: args.hidden_size,
        epochs: args.epochs,
        batch_size: args.batch_size,
        learning_rate: args.learning_rate,
        weight_decay: args.weight_decay,
        seed: args.seed,
        val_split: args.val_split,
        buckets: args.buckets,
        max_videos: args.max_videos,
        balance: args.balance,
    };

    let dataset = load_dataset(&config)?;
    let (model, report) = train_video_classifier(&dataset, &config)?;

    let out = args.out.unwrap_or_else(|| default_model_path(&config));
    save_model(&out, &model, &dataset, &config)?;

    println!();
    println!("oxidize-train video: task            = {}", report.task);
    println!("oxidize-train video: clips           = {}", report.clips);
    println!(
        "oxidize-train video: train/val      = {}/{}",
        report.train_clips, report.val_clips
    );
    println!("oxidize-train video: classes         = {}", report.classes);
    for (name, count) in report.class_names.iter().zip(&report.class_histogram) {
        println!("oxidize-train video:   - {name}: {count}");
    }
    println!(
        "oxidize-train video: final_train_loss= {:.4}",
        report.final_train_loss
    );
    println!(
        "oxidize-train video: train_accuracy  = {:.4}",
        report.train_accuracy
    );
    println!(
        "oxidize-train video: val_accuracy    = {:.4}  (majority baseline {:.4})",
        report.val_accuracy, report.majority_baseline
    );
    println!("oxidize-train video: model           = {}", out.display());
    Ok(())
}

fn run_prototype(args: PrototypeArgs) -> Result<()> {
    let cache = args
        .cache
        .unwrap_or_else(|| args.data.join(".oxidize-frames"));
    let frames = FrameConfig {
        num_frames: args.frames,
        frame_size: args.frame_size,
        patch_size: 16,
    };

    let report = build_prototype(
        &args.data,
        &cache,
        frames,
        &args.exclude,
        &args.out,
        args.sheet.as_deref(),
        args.upscale,
        args.fps,
    )?;

    println!();
    println!("oxidize-train prototype: clips    = {}", report.clips);
    println!("oxidize-train prototype: frames   = {}", report.frames);
    println!(
        "oxidize-train prototype: creators = {}",
        report.creators.join(", ")
    );
    println!(
        "oxidize-train prototype: video    = {}",
        report.out.display()
    );
    if let Some(sheet) = &report.sheet {
        println!("oxidize-train prototype: sheet    = {}", sheet.display());
    }
    Ok(())
}
