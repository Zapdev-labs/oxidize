use clap::Parser;
use oxidize_core::dflash::{DFlashConfig, DFlashDraftModel};
use oxidize_core::model_loader::ModelLoader;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Debug, Parser)]
#[command(name = "oxidize-bench")]
struct Args {
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long, default_value_t = 128)]
    draft_tokens: usize,
    #[arg(long, default_value_t = 5)]
    iterations: usize,
    #[arg(long, default_value_t = false)]
    verbose: bool,
    #[arg(long, default_value_t = false)]
    random_weights: bool,
}

fn main() {
    let args = Args::parse();

    println!("=== Oxidize DFlash Benchmark ===\n");

    let mut draft_model: DFlashDraftModel;
    let config: DFlashConfig;

    if let Some(model_path) = args.model {
        println!("Loading model from: {}\n", model_path.display());
        let loader = oxidize_core::model_loader::GgufModelLoader;
        let mapped = loader.load(&model_path).expect("Failed to load GGUF");

        // Extract config from metadata
        let metadata = &mapped.parsed().metadata;
        let hidden_size =
            metadata_u32(metadata, "dflash-draft.embedding_length").unwrap_or(5120) as usize;
        let num_layers = metadata_u32(metadata, "dflash-draft.block_count").unwrap_or(5) as usize;
        let num_attention_heads =
            metadata_u32(metadata, "dflash-draft.attention.head_count").unwrap_or(32) as usize;
        let num_key_value_heads =
            metadata_u32(metadata, "dflash-draft.attention.head_count_kv").unwrap_or(8) as usize;
        let intermediate_size =
            metadata_u32(metadata, "dflash-draft.feed_forward_length").unwrap_or(17408) as usize;
        let block_size =
            metadata_u32(metadata, "dflash-draft.dflash.block_size").unwrap_or(16) as usize;
        let mask_token_id =
            metadata_u32(metadata, "dflash-draft.dflash.mask_token_id").unwrap_or(151665);
        let n_target_features = metadata_u32(metadata, "dflash-draft.dflash.n_target_features")
            .unwrap_or(25600) as usize;
        let rope_theta = metadata_f32(metadata, "dflash-draft.rope.freq_base").unwrap_or(1e7);
        let rms_norm_eps =
            metadata_f32(metadata, "dflash-draft.attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
        let context_length =
            metadata_u32(metadata, "dflash-draft.context_length").unwrap_or(262144) as usize;

        config = DFlashConfig {
            hidden_size,
            num_hidden_layers: num_layers,
            num_target_layers: num_layers,
            block_size,
            target_layer_ids: vec![],
            mask_token_id,
            vocab_size: n_target_features,
            num_attention_heads,
            num_key_value_heads,
            intermediate_size,
            rms_norm_eps,
            rope_theta,
        };

        println!("Model config from GGUF:");
        println!("  hidden_size: {}", hidden_size);
        println!("  num_layers: {}", num_layers);
        println!("  num_attention_heads: {}", num_attention_heads);
        println!("  num_key_value_heads: {}", num_key_value_heads);
        println!("  intermediate_size: {}", intermediate_size);
        println!("  block_size: {}", block_size);
        println!("  mask_token_id: {}", mask_token_id);
        println!("  n_target_features (vocab): {}", n_target_features);
        println!("  rope_theta: {}", rope_theta);
        println!("  rms_norm_eps: {}", rms_norm_eps);
        println!("  context_length: {}", context_length);
        println!();

        draft_model = DFlashDraftModel::load_from_gguf(&mapped, config.clone())
            .expect("Failed to load DFlash model from GGUF");
    } else if args.random_weights {
        println!("Using random weights for testing...\n");
        config = DFlashConfig::default();
        draft_model = create_random_draft_model(&config);
    } else {
        eprintln!("Error: Either --model <path> or --random-weights must be specified");
        eprintln!("  Example: oxidize-bench --model models/Qwen3.6-27B-DFlash-Q4_K_M.gguf");
        eprintln!("  Example: oxidize-bench --random-weights");
        std::process::exit(1);
    }

    println!("Running draft model forward pass benchmark...\n");
    println!("  draft_tokens_per_step: {}", args.draft_tokens);
    println!("  iterations: {}", args.iterations);
    println!();

    let mut total_tokens = 0usize;
    let mut total_duration = Duration::ZERO;

    for i in 0..args.iterations {
        let start = Instant::now();
        let mut tokens_generated = 0;

        let noise_token = config.mask_token_id;
        let current_token = noise_token;

        // For draft-only models without lm_head, benchmark forward_token directly.
        for _ in 0..args.draft_tokens {
            let _hidden = draft_model
                .forward_token(current_token, None)
                .expect("forward_token failed");

            tokens_generated += 1;
        }

        let elapsed = start.elapsed();
        total_tokens += tokens_generated;
        total_duration += elapsed;

        if args.verbose {
            let tps = tokens_generated as f64 / elapsed.as_secs_f64();
            println!(
                "  Iteration {}: {} tokens in {:.2?} ({:.2} tok/s)",
                i + 1,
                tokens_generated,
                elapsed,
                tps
            );
        }

        draft_model.reset_cache();
    }

    let avg_tps = total_tokens as f64 / total_duration.as_secs_f64();
    let avg_latency = total_duration / total_tokens as u32;

    println!("\n=== Results ===");
    println!("Total tokens: {}", total_tokens);
    println!("Total time: {:.2?}", total_duration);
    println!("Throughput: {:.2} tok/s", avg_tps);
    println!("Avg latency/token: {:.2?}", avg_latency);
    println!("\nNote: Draft-only model has no lm_head; benchmarked forward_token() only.");

    println!("\nBenchmark complete.");
}

fn metadata_u32(
    metadata: &std::collections::BTreeMap<String, oxidize_core::gguf::GgufMetadataValue>,
    key: &str,
) -> Option<u32> {
    use oxidize_core::gguf::GgufMetadataValue;
    match metadata.get(key)? {
        GgufMetadataValue::Uint32(v) => Some(*v),
        GgufMetadataValue::Int32(v) => Some(*v as u32),
        GgufMetadataValue::Uint64(v) => Some(*v as u32),
        GgufMetadataValue::Int64(v) => Some(*v as u32),
        GgufMetadataValue::Float32(v) => Some(*v as u32),
        _ => None,
    }
}

fn metadata_f32(
    metadata: &std::collections::BTreeMap<String, oxidize_core::gguf::GgufMetadataValue>,
    key: &str,
) -> Option<f32> {
    use oxidize_core::gguf::GgufMetadataValue;
    match metadata.get(key)? {
        GgufMetadataValue::Float32(v) => Some(*v),
        GgufMetadataValue::Float64(v) => Some(*v as f32),
        GgufMetadataValue::Uint32(v) => Some(*v as f32),
        GgufMetadataValue::Int32(v) => Some(*v as f32),
        _ => None,
    }
}

fn create_random_draft_model(config: &DFlashConfig) -> DFlashDraftModel {
    use oxidize_core::dflash::{DFlashAttentionLayer, DFlashDecoderLayer, F32Weight};
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
        let q_proj = random_weight(hidden, hidden);
        let k_proj = random_weight(kv_heads * head_dim, hidden);
        let v_proj = random_weight(kv_heads * head_dim, hidden);
        let o_proj = random_weight(hidden, hidden);
        let q_norm = vec![1.0f32; head_dim];
        let k_norm = vec![1.0f32; head_dim];
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
        target_hidden_cache: Vec::new(),
        position_offset: 0,
    }
}

fn random_weight(rows: usize, cols: usize) -> oxidize_core::dflash::F32Weight {
    use oxidize_core::dflash::F32Weight;
    let mut data = vec![0.0f32; rows * cols];
    for v in data.iter_mut() {
        *v = (rand::random::<f32>() - 0.5) * 0.02;
    }
    F32Weight::from_slice(data, rows, cols)
}
