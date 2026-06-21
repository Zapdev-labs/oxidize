use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Parser, Subcommand, ValueEnum};
use oxidize_train::video::{
    FrameConfig, GenTrainingConfig, LabelTask, VideoTrainingConfig, default_generator_path,
    default_model_path, ffmpeg_available, generate_video, load_dataset, load_gen_dataset,
    load_generator, save_generator, save_model, train_generator, train_video_classifier,
};
use oxidize_train::{TrainingConfig, load_csv_dataset, train_classifier};

#[derive(Debug, Parser)]
#[command(
    name = "oxidize-train",
    about = "Train classifiers and generative video models on CPU"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Train the MLP classifier on a CSV dataset.
    Csv(CsvArgs),
    /// Train a clip classifier (virality / engagement labels).
    Video(VideoArgs),
    /// Train an autoregressive next-frame video generator.
    GenTrain(GenTrainArgs),
    /// Generate a new video from a trained generator checkpoint.
    Generate(GenerateArgs),
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
    Creator,
    Virality,
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
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    cache: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = TaskArg::Virality)]
    task: TaskArg,
    #[arg(long, default_value_t = 8)]
    frames: usize,
    #[arg(long = "frame-size", default_value_t = 64)]
    frame_size: usize,
    #[arg(long = "patch-size", default_value_t = 16)]
    patch_size: usize,
    #[arg(long = "embed-dim", default_value_t = 64)]
    embed_dim: usize,
    #[arg(long = "hidden-size", default_value_t = 128)]
    hidden_size: usize,
    #[arg(long, default_value_t = 40)]
    epochs: usize,
    #[arg(long = "batch-size", default_value_t = 16)]
    batch_size: usize,
    #[arg(long = "learning-rate", default_value_t = 5e-4)]
    learning_rate: f32,
    #[arg(long = "weight-decay", default_value_t = 0.05)]
    weight_decay: f32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long = "val-split", default_value_t = 0.15)]
    val_split: f32,
    #[arg(long, default_value_t = 3)]
    buckets: usize,
    #[arg(long = "max-videos")]
    max_videos: Option<usize>,
    #[arg(long)]
    balance: bool,
    #[arg(long)]
    exclude: Vec<String>,
    #[arg(long, default_value_t = true)]
    merge_aliases: bool,
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct GenTrainArgs {
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    cache: Option<PathBuf>,
    #[arg(long, default_value_t = 12)]
    frames: usize,
    /// Portrait short edge (9:16). Ignored if `--frame-size` or explicit width/height set.
    #[arg(long = "portrait-width", default_value_t = 72)]
    portrait_width: usize,
    #[arg(long = "frame-width", default_value_t = 0)]
    frame_width: usize,
    #[arg(long = "frame-height", default_value_t = 0)]
    frame_height: usize,
    /// Legacy square edge; 0 = use 9:16 portrait.
    #[arg(long = "frame-size", default_value_t = 0)]
    frame_size: usize,
    #[arg(long = "patch-size", default_value_t = 8)]
    patch_size: usize,
    #[arg(long = "context-frames", default_value_t = 4)]
    context_frames: usize,
    #[arg(long = "hidden-size", default_value_t = 256)]
    hidden_size: usize,
    #[arg(long = "patch-hidden", default_value_t = 128)]
    patch_hidden: usize,
    #[arg(long, default_value_t = 40)]
    epochs: usize,
    #[arg(long = "batch-size", default_value_t = 64)]
    batch_size: usize,
    #[arg(long = "learning-rate", default_value_t = 1e-3)]
    learning_rate: f32,
    #[arg(long = "weight-decay", default_value_t = 0.001)]
    weight_decay: f32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long = "val-split", default_value_t = 0.1)]
    val_split: f32,
    #[arg(long = "max-videos")]
    max_videos: Option<usize>,
    #[arg(long)]
    exclude: Vec<String>,
    #[arg(long, default_value_t = true)]
    merge_aliases: bool,
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct GenerateArgs {
    /// Trained generator checkpoint (JSON).
    #[arg(long)]
    model: PathBuf,
    /// Clip source root (for seed frames).
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    cache: Option<PathBuf>,
    #[arg(long)]
    exclude: Vec<String>,
    #[arg(long, default_value_t = true)]
    merge_aliases: bool,
    /// Number of new frames to synthesize after the seed context.
    #[arg(long = "output-frames", default_value_t = 32)]
    output_frames: usize,
    /// Sampling noise (0 = deterministic).
    #[arg(long, default_value_t = 0.02)]
    temperature: f32,
    #[arg(long, default_value_t = 24)]
    fps: u32,
    /// Upscale output to this height (720×1280 TikTok). 0 = native model resolution.
    #[arg(long = "upscale-height", default_value_t = 1280)]
    upscale_height: u32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long)]
    out: PathBuf,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Csv(args) => run_csv(args),
        Command::Video(args) => run_video(args),
        Command::GenTrain(args) => run_gen_train(args),
        Command::Generate(args) => run_generate(args),
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
        eprintln!("oxidize-train video: warning — ffmpeg not found; only cached frames work.");
    }

    let exclude = default_exclude(args.exclude);
    let config = VideoTrainingConfig {
        data_root: args.data,
        cache_dir: args.cache.unwrap_or_default(),
        task: args.task.into(),
        frames: FrameConfig {
            num_frames: args.frames,
            frame_width: 0,
            frame_height: 0,
            frame_size: args.frame_size,
            patch_size: args.patch_size,
        }
        .resolve_dimensions(),
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
        exclude,
        merge_aliases: args.merge_aliases,
    };

    let dataset = load_dataset(&config)?;
    let (model, report) = train_video_classifier(&dataset, &config)?;
    let out = args.out.unwrap_or_else(|| default_model_path(&config));
    save_model(&out, &model, &dataset, &config)?;

    print_video_report(&report, &out);
    Ok(())
}

fn run_gen_train(args: GenTrainArgs) -> Result<()> {
    if !ffmpeg_available() {
        eprintln!("oxidize-train gen-train: warning — ffmpeg not found.");
    }

    let exclude = default_exclude(args.exclude);
    let frames = resolve_frame_config(
        args.frames,
        args.frame_size,
        args.frame_width,
        args.frame_height,
        args.patch_size,
        args.portrait_width,
    );
    let config = GenTrainingConfig {
        data_root: args.data,
        cache_dir: args.cache.unwrap_or_default(),
        frames,
        context_frames: args.context_frames,
        hidden_size: args.hidden_size,
        patch_hidden: args.patch_hidden,
        epochs: args.epochs,
        batch_size: args.batch_size,
        learning_rate: args.learning_rate,
        weight_decay: args.weight_decay,
        seed: args.seed,
        val_split: args.val_split,
        max_videos: args.max_videos,
        exclude,
        merge_aliases: args.merge_aliases,
    };

    let dataset = load_gen_dataset(&config)?;
    let (model, report) = train_generator(&dataset, &config)?;
    let out = args.out.unwrap_or_else(|| default_generator_path(&config));
    save_generator(&out, &model, &config)?;

    println!();
    println!("oxidize-train gen-train: pairs      = {}", report.pairs);
    println!(
        "oxidize-train gen-train: train/val   = {}/{}",
        report.train_pairs, report.val_pairs
    );
    println!(
        "oxidize-train gen-train: final_mse   = {:.5}",
        report.final_train_loss
    );
    println!(
        "oxidize-train gen-train: val_mse     = {:.5}  (best {:.5})",
        report.val_loss, report.best_val_loss
    );
    println!("oxidize-train gen-train: model       = {}", out.display());
    Ok(())
}

fn run_generate(args: GenerateArgs) -> Result<()> {
    let saved = load_generator(&args.model)?;
    let cache = args
        .cache
        .unwrap_or_else(|| args.data.join(".oxidize-frames"));
    let exclude = default_exclude(args.exclude);

    let report = generate_video(
        &saved.generator,
        &saved,
        &args.data,
        &cache,
        &exclude,
        args.merge_aliases,
        args.output_frames,
        args.temperature,
        args.fps,
        &args.out,
        args.seed,
        args.upscale_height,
    )?;

    println!();
    println!("oxidize-train generate: seed_clip = {}", report.seed_clip);
    println!("oxidize-train generate: frames    = {}", report.frames);
    println!(
        "oxidize-train generate: video     = {}",
        report.out.display()
    );
    Ok(())
}

fn default_exclude(exclude: Vec<String>) -> Vec<String> {
    if exclude.is_empty() {
        vec!["cellow111".into()]
    } else {
        exclude
    }
}

fn resolve_frame_config(
    num_frames: usize,
    frame_size: usize,
    frame_width: usize,
    frame_height: usize,
    patch_size: usize,
    portrait_width: usize,
) -> FrameConfig {
    if frame_size > 0 {
        FrameConfig {
            num_frames,
            frame_width: 0,
            frame_height: 0,
            frame_size,
            patch_size,
        }
        .resolve_dimensions()
    } else if frame_width > 0 && frame_height > 0 {
        FrameConfig {
            num_frames,
            frame_width,
            frame_height,
            frame_size: 0,
            patch_size,
        }
        .resolve_dimensions()
    } else {
        FrameConfig::portrait_9_16(portrait_width, patch_size, num_frames)
    }
}

fn print_video_report(report: &oxidize_train::video::VideoTrainingReport, out: &std::path::Path) {
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
        "oxidize-train video: val_accuracy    = {:.4}  (best {:.4}, majority baseline {:.4})",
        report.val_accuracy, report.best_val_accuracy, report.majority_baseline
    );
    println!("oxidize-train video: model           = {}", out.display());
}
