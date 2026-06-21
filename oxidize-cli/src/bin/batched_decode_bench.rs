//! Continuous-batching decode throughput proving vehicle.
//!
//! Decode is memory-bound: every token streams all model weights from memory
//! regardless of how many sequences are served. Running `--batch B` sequences as
//! ONE set of batched GEMMs (via `InferenceModel::forward_batch`) amortizes those
//! weight reads, so aggregate tok/s should grow with `B` until compute-bound.
//!
//! This bench seeds `B` byte-identical sequences (same prompt), then times `S`
//! batched decode steps and prints aggregate throughput as `"{:.2} tok/s"` so the
//! Modal harness regex (`(\d+\.\d+)\s*tok/s`) matches. It also asserts the core
//! correctness invariant: every per-sequence logit vector is identical to row 0
//! at every step (the batched GEMM must not mix rows).
//!
//! Gate: honors `OX_BATCHED_DECODE`; with the flag unset the bench still runs
//! `forward_batch` (it is the bench's only execution path), but the flag is what
//! a future paged runtime consults before taking the batched decode branch.

use clap::Parser;
use oxidize_core::inference::{InferenceConfig, InferenceModel, SeqKv};
use oxidize_core::model_loader::ModelLoader;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Parser)]
#[command(name = "batched_decode_bench")]
struct Args {
    /// Path to a GGUF model file.
    #[arg(long)]
    model: PathBuf,
    /// Number of concurrent decode sequences (batch width).
    #[arg(long, default_value_t = 1)]
    batch: usize,
    /// Number of decode steps to time.
    #[arg(long, default_value_t = 64)]
    steps: usize,
    /// Seed prompt token ids (comma-separated). Defaults to a short fixed prompt.
    #[arg(long, default_value = "1,2,3,4")]
    prompt: String,
    /// KV capacity (positions) reserved per sequence.
    #[arg(long, default_value_t = 512)]
    capacity: usize,
    /// Cap the model context to bound memory.
    #[arg(long, default_value_t = 8192)]
    max_context: usize,
    /// Verify every batched row matches row 0 bit-for-bit each step.
    #[arg(long, default_value_t = true)]
    verify: bool,
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best as u32
}

fn main() {
    let args = Args::parse();
    let batch = args.batch.max(1);

    let prompt: Vec<u32> = args
        .prompt
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().parse::<u32>().expect("prompt token must be u32"))
        .collect();
    let prompt = if prompt.is_empty() {
        vec![1u32]
    } else {
        prompt
    };

    let batched_flag = std::env::var("OX_BATCHED_DECODE").unwrap_or_default();
    // When OX_GPU_BATCHED=1 (+ CUDA active + an eligible model) the decode rows
    // are driven through the on-device batched forward (`forward_batch_gpu`), with
    // a transparent fall back to the CPU `forward_batch` when it returns `None`.
    let gpu_batched = std::env::var("OX_GPU_BATCHED")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    eprintln!(
        "batched_decode_bench: model={} batch={batch} steps={} prompt_len={} OX_BATCHED_DECODE={:?} OX_GPU_BATCHED={gpu_batched}",
        args.model.display(),
        args.steps,
        prompt.len(),
        batched_flag
    );

    let loader = oxidize_core::model_loader::GgufModelLoader;
    let mapped = loader.load(&args.model).expect("failed to load GGUF");
    let mut config = InferenceConfig::from_gguf(&mapped);
    if config.context_size > args.max_context {
        config.context_size = args.max_context;
    }
    let mut model = InferenceModel::load_from_gguf(&mapped, config, true)
        .expect("failed to load inference model");
    eprintln!(
        "batched_decode_bench: batched_decode_enabled={}",
        model.batched_decode_enabled()
    );

    let kv_layers = model.kv_layer_count();
    let kv_len = model.kv_row_len();
    let cap = args.capacity.max(prompt.len() + args.steps + 1);

    // Per-sequence KV buffers — the model never touches its own kv_cache here.
    let mut kv: Vec<SeqKv> = (0..batch)
        .map(|_| SeqKv::new(kv_layers, cap, kv_len))
        .collect();

    // Seed every sequence with the SAME prompt → byte-identical sequences, so
    // each decoded row must equal row 0 (a direct cross-row correctness check).
    // GPU-then-CPU dispatch (mirrors ContinuousBatchEngine::step). When
    // `gpu_batched` is false, or the device path is ineligible, this is exactly
    // the CPU `forward_batch`.
    let decode = |model: &mut InferenceModel,
                      rows: &[(u32, usize)],
                      kv: &mut [SeqKv]|
     -> Vec<Vec<f32>> {
        if gpu_batched {
            if let Some(l) = model
                .forward_batch_gpu(rows, kv, true)
                .expect("forward_batch_gpu")
            {
                return l;
            }
        }
        model.forward_batch(rows, kv, true).expect("forward_batch")
    };

    let mut last = vec![0u32; batch];
    let mut pos = 0usize;
    for &tok in &prompt {
        let rows: Vec<(u32, usize)> = (0..batch).map(|_| (tok, pos)).collect();
        let out = decode(&mut model, &rows, &mut kv);
        for i in 0..batch {
            last[i] = argmax(&out[i]);
        }
        pos += 1;
    }

    // Timed decode loop: each step runs ALL B sequences as one batched forward.
    let start = Instant::now();
    for step in 0..args.steps {
        let rows: Vec<(u32, usize)> = (0..batch).map(|i| (last[i], pos)).collect();
        let out = decode(&mut model, &rows, &mut kv);
        if args.verify {
            for i in 1..batch {
                assert!(
                    out[i] == out[0],
                    "step {step}: batched row {i} diverged from row 0 (cross-row contamination)"
                );
            }
        }
        for i in 0..batch {
            last[i] = argmax(&out[i]);
        }
        pos += 1;
    }
    let elapsed = start.elapsed().as_secs_f64();

    let total_tokens = batch * args.steps;
    let aggregate_tps = total_tokens as f64 / elapsed;
    let per_seq_tps = args.steps as f64 / elapsed;

    println!(
        "batch={batch} steps={} total_tokens={total_tokens} elapsed={elapsed:.3}s per_seq={per_seq_tps:.2} tok/s",
        args.steps
    );
    // Aggregate throughput in the exact format the Modal regex matches.
    println!("{aggregate_tps:.2} tok/s");
}
