use clap::Parser;
use oxidize_core::dflash::{DFlashConfig, DFlashDecoderLayer, DFlashDraftModel, DFlashAttentionLayer, F32Weight};
use oxidize_core::model::{Model, Session};
use std::time::{Duration, Instant};

#[derive(Debug, Parser)]
#[command(name = "oxidize-bench")]
struct Args {
    #[arg(long, default_value_t = 128)]
    draft_tokens: usize,
    #[arg(long, default_value_t = 5)]
    iterations: usize,
    #[arg(long, default_value_t = false)]
    verbose: bool,
}

fn main() {
    let args = Args::parse();

    println!("=== Oxidize DFlash Benchmark ===\n");

    let config = DFlashConfig::qwen3_6_35b_a3b_dflash();
    println!("Configuration:");
    println!("  hidden_size: {}", config.hidden_size);
    println!("  num_layers: {}", config.num_hidden_layers);
    println!("  vocab_size: {}", config.vocab_size);
    println!("  draft_tokens_per_step: {}", args.draft_tokens);
    println!("  iterations: {}", args.iterations);
    println!();

    let mut draft_model = create_random_draft_model(&config);
    let mut session = Session::new();

    println!("Running draft model forward pass benchmark...\n");

    let mut total_tokens = 0usize;
    let mut total_duration = Duration::ZERO;

    for i in 0..args.iterations {
        let start = Instant::now();
        let mut tokens_generated = 0;

        let noise_token = config.mask_token_id;
        let mut current_token = noise_token;

        for _ in 0..args.draft_tokens {
            let logits = draft_model
                .forward(&[current_token], &mut session)
                .unwrap();

            let next_token = greedy_sample(&logits);
            current_token = next_token;
            tokens_generated += 1;
        }

        let elapsed = start.elapsed();
        total_tokens += tokens_generated;
        total_duration += elapsed;

        if args.verbose {
            let tps = tokens_generated as f64 / elapsed.as_secs_f64();
            println!("  Iteration {}: {} tokens in {:.2?} ({:.2} tok/s)",
                i + 1, tokens_generated, elapsed, tps);
        }

        draft_model.reset_cache();
        session = Session::new();
    }

    let avg_tps = total_tokens as f64 / total_duration.as_secs_f64();
    let avg_latency = total_duration / total_tokens as u32;

    println!("\n=== Results ===");
    println!("Total tokens: {}", total_tokens);
    println!("Total time: {:.2?}", total_duration);
    println!("Throughput: {:.2} tok/s", avg_tps);
    println!("Avg latency/token: {:.2?}", avg_latency);

    println!("\nBenchmark complete.");
}

fn greedy_sample(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}

fn create_random_draft_model(config: &DFlashConfig) -> DFlashDraftModel {
    let vocab_size = config.vocab_size;
    let hidden = config.hidden_size;
    let layers = config.num_hidden_layers;
    let kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim();
    let intermediate = config.intermediate_size;

    let mut token_embeddings = vec![0.0f32; vocab_size * hidden];
    for v in token_embeddings.iter_mut() {
        *v = (rand::random::<f32>() - 0.5) * 0.02;
    }

    let mut fc_weight = vec![0.0f32; hidden * hidden];
    let fc_bias = vec![0.0f32; hidden];
    for v in fc_weight.iter_mut() {
        *v = (rand::random::<f32>() - 0.5) * 0.02;
    }

    let mut layers_vec = Vec::with_capacity(layers);
    for _ in 0..layers {
        // q_proj: hidden -> hidden
        let q_proj = random_weight(hidden, hidden);
        // k_proj/v_proj: hidden -> kv_heads * head_dim
        // (when target_hidden is None, input is just normed hidden of size hidden)
        let k_proj = random_weight(kv_heads * head_dim, hidden);
        let v_proj = random_weight(kv_heads * head_dim, hidden);
        // o_proj: hidden -> hidden
        let o_proj = random_weight(hidden, hidden);
        let q_norm = vec![1.0f32; head_dim];
        let k_norm = vec![1.0f32; head_dim];
        // MLP projections
        let gate_proj = random_weight(intermediate, hidden);
        let up_proj = random_weight(intermediate, hidden);
        let down_proj = random_weight(hidden, intermediate);
        let input_layernorm = vec![1.0f32; hidden];
        let post_attention_layernorm = vec![1.0f32; hidden];

        layers_vec.push(DFlashDecoderLayer {
            input_layernorm,
            attention: DFlashAttentionLayer {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm_weight: q_norm,
                k_norm_weight: k_norm,
            },
            post_attention_layernorm,
            mlp_gate: gate_proj,
            mlp_up: up_proj,
            mlp_down: down_proj,
        });
    }

    let norm = vec![1.0f32; hidden];
    let mut output_weight = vec![0.0f32; vocab_size * hidden];
    for v in output_weight.iter_mut() {
        *v = (rand::random::<f32>() - 0.5) * 0.02;
    }

    DFlashDraftModel {
        config: config.clone(),
        fc: F32Weight::from_slice(fc_weight, hidden, hidden),
        fc_bias,
        hidden_norm: norm.clone(),
        layers: layers_vec,
        norm,
        output: F32Weight::from_slice(output_weight, vocab_size, hidden),
        tok_embeddings: F32Weight::from_slice(token_embeddings, vocab_size, hidden),
        kv_cache: vec![Vec::new(); config.num_hidden_layers],
        position_offset: 0,
    }
}

fn random_weight(rows: usize, cols: usize) -> F32Weight {
    let mut data = vec![0.0f32; rows * cols];
    for v in data.iter_mut() {
        *v = (rand::random::<f32>() - 0.5) * 0.02;
    }
    F32Weight::from_slice(data, rows, cols)
}
