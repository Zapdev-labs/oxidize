use clap::Parser;
use oxidize_core::dflash::{DFlashConfig, DFlashDraftModel, DFlashKvLayerCache};
use oxidize_core::inference::{InferenceConfig, InferenceModel};
use oxidize_core::layer_wise::LayerWiseModel;
use oxidize_core::model::{Model, Session};
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
    #[arg(long)]
    prompt_tokens: Option<usize>,
    #[arg(long, default_value = "decode")]
    mode: String,
    #[arg(long, default_value = "inference")]
    engine: String,
    #[arg(long, default_value_t = 2)]
    layer_cache_size: usize,
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

    if let Some(model_path) = &args.model {
        println!("Loading model from: {}\n", model_path.display());
        let loader = oxidize_core::model_loader::GgufModelLoader;
        let mapped = loader.load(model_path).expect("Failed to load GGUF");

        // Extract config from metadata
        let metadata = &mapped.parsed().metadata;
        let arch = metadata_string(metadata, "general.architecture");
        let arch_key = |suffix: &str| arch.as_ref().map(|a| format!("{a}.{suffix}"));
        let arch_u32 = |suffix: &str| arch_key(suffix).and_then(|key| metadata_u32(metadata, &key));
        let arch_f32 = |suffix: &str| arch_key(suffix).and_then(|key| metadata_f32(metadata, &key));
        let hidden_size = metadata_u32(metadata, "dflash-draft.hidden_size")
            .or_else(|| metadata_u32(metadata, "dflash-draft.embedding_length"))
            .or_else(|| arch_u32("embedding_length"))
            .unwrap_or(5120) as usize;
        let num_layers = metadata_u32(metadata, "dflash-draft.num_hidden_layers")
            .or_else(|| metadata_u32(metadata, "dflash-draft.block_count"))
            .or_else(|| arch_u32("block_count"))
            .unwrap_or(5) as usize;
        let num_attention_heads = metadata_u32(metadata, "dflash-draft.num_attention_heads")
            .or_else(|| metadata_u32(metadata, "dflash-draft.attention.head_count"))
            .or_else(|| arch_u32("attention.head_count"))
            .unwrap_or(32) as usize;
        let num_key_value_heads = metadata_u32(metadata, "dflash-draft.num_key_value_heads")
            .or_else(|| metadata_u32(metadata, "dflash-draft.attention.head_count_kv"))
            .or_else(|| arch_u32("attention.head_count_kv"))
            .unwrap_or(8) as usize;
        let key_value_head_dim = metadata_u32(metadata, "dflash-draft.attention.key_length")
            .or_else(|| arch_u32("attention.key_length"))
            .unwrap_or((hidden_size / num_attention_heads) as u32)
            as usize;
        let intermediate_size = metadata_u32(metadata, "dflash-draft.intermediate_size")
            .or_else(|| metadata_u32(metadata, "dflash-draft.feed_forward_length"))
            .or_else(|| arch_u32("feed_forward_length"))
            .unwrap_or(17408) as usize;
        let block_size = metadata_u32(metadata, "dflash-draft.block_size")
            .or_else(|| metadata_u32(metadata, "dflash-draft.dflash.block_size"))
            .unwrap_or(16) as usize;
        let mask_token_id = metadata_u32(metadata, "dflash-draft.mask_token_id")
            .or_else(|| metadata_u32(metadata, "dflash-draft.dflash.mask_token_id"))
            .unwrap_or(151665);
        let n_target_features = metadata_u32(metadata, "dflash-draft.vocab_size")
            .or_else(|| metadata_u32(metadata, "dflash-draft.n_target_features"))
            .or_else(|| metadata_u32(metadata, "dflash-draft.dflash.n_target_features"))
            .or_else(|| arch_u32("vocab_size"))
            .or_else(|| token_embedding_vocab(&mapped))
            .unwrap_or(25600) as usize;
        let rope_theta = metadata_f32(metadata, "dflash-draft.rope_theta")
            .or_else(|| metadata_f32(metadata, "dflash-draft.rope.freq_base"))
            .or_else(|| arch_f32("rope.freq_base"))
            .unwrap_or(1e7);
        let rms_norm_eps = metadata_f32(metadata, "dflash-draft.rms_norm_eps")
            .or_else(|| metadata_f32(metadata, "dflash-draft.attention.layer_norm_rms_epsilon"))
            .or_else(|| arch_f32("attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);
        let context_length = metadata_u32(metadata, "dflash-draft.context_length")
            .or_else(|| arch_u32("context_length"))
            .unwrap_or(262144) as usize;

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
        println!("  key_value_head_dim: {}", key_value_head_dim);
        println!("  intermediate_size: {}", intermediate_size);
        println!("  block_size: {}", block_size);
        println!("  mask_token_id: {}", mask_token_id);
        println!("  n_target_features (vocab): {}", n_target_features);
        println!("  rope_theta: {}", rope_theta);
        println!("  rms_norm_eps: {}", rms_norm_eps);
        println!("  context_length: {}", context_length);
        println!();

        if args.engine == "inference" || args.engine == "layerwise" {
            let inference_config =
                inference_config_from_dflash(&config, context_length, key_value_head_dim);
            if args.engine == "inference" {
                let mut model = InferenceModel::load_from_gguf(&mapped, inference_config, true)
                    .expect("Failed to load inference GGUF model");
                run_inference_model_benchmark(&args, &mut model, config.mask_token_id);
                return;
            }
            let mut model: Box<dyn Model> = Box::new(
                LayerWiseModel::load_from_gguf(&mapped, inference_config, args.layer_cache_size)
                    .expect("Failed to load layer-wise GGUF model"),
            );
            run_standard_model_benchmark(&args, model.as_mut(), config.mask_token_id);
            return;
        }

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

    println!("Running draft model benchmark...\n");
    println!("  mode: {}", args.mode);
    println!("  draft_tokens_per_step: {}", args.draft_tokens);
    if let Some(prompt_tokens) = args.prompt_tokens {
        println!("  prompt_tokens: {}", prompt_tokens);
    }
    println!("  iterations: {}", args.iterations);
    println!();

    let mut total_tokens = 0usize;
    let mut total_duration = Duration::ZERO;

    for i in 0..args.iterations {
        draft_model.reserve_cache_tokens(args.draft_tokens);
        let start = Instant::now();
        let tokens_generated = match args.mode.as_str() {
            "decode" => {
                run_decode_iteration(&mut draft_model, config.mask_token_id, args.draft_tokens)
            }
            "prompt" | "pp" => run_prompt_iteration(
                &mut draft_model,
                config.mask_token_id,
                args.prompt_tokens.unwrap_or(args.draft_tokens),
            ),
            other => {
                eprintln!("Error: unsupported --mode '{other}' (expected decode, prompt, or pp)");
                std::process::exit(2);
            }
        };

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
    if args.mode == "decode" {
        println!("\nNote: decode mode benchmarks forward_token() only.");
    } else {
        println!("\nNote: prompt mode benchmarks Model::forward() over the prompt token slice.");
    }

    println!("\nBenchmark complete.");
}

fn run_standard_model_benchmark(args: &Args, model: &mut dyn Model, token: u32) {
    println!("Running standard model benchmark...\n");
    println!("  engine: {}", args.engine);
    println!("  mode: {}", args.mode);
    println!("  draft_tokens_per_step: {}", args.draft_tokens);
    if let Some(prompt_tokens) = args.prompt_tokens {
        println!("  prompt_tokens: {}", prompt_tokens);
    }
    println!("  iterations: {}", args.iterations);
    println!();

    let tokens_per_iteration = if matches!(args.mode.as_str(), "prompt" | "pp") {
        args.prompt_tokens.unwrap_or(args.draft_tokens)
    } else {
        args.draft_tokens
    };
    let tokens = vec![token; tokens_per_iteration];
    let mut total_tokens = 0usize;
    let mut total_duration = Duration::ZERO;

    for i in 0..args.iterations {
        let mut session = Session::new();
        model.rewind_to(0).expect("rewind failed");
        let start = Instant::now();
        match args.mode.as_str() {
            "prompt" | "pp" => {
                let _logits = model.forward(&tokens, &mut session).expect("prompt failed");
            }
            "decode" => {
                for &tok in &tokens {
                    let _logits = model.forward(&[tok], &mut session).expect("decode failed");
                }
            }
            other => {
                eprintln!("Error: unsupported --mode '{other}' (expected decode, prompt, or pp)");
                std::process::exit(2);
            }
        }
        let elapsed = start.elapsed();
        total_tokens += tokens_per_iteration;
        total_duration += elapsed;
        if args.verbose {
            let tps = tokens_per_iteration as f64 / elapsed.as_secs_f64();
            println!(
                "  Iteration {}: {} tokens in {:.2?} ({:.2} tok/s)",
                i + 1,
                tokens_per_iteration,
                elapsed,
                tps
            );
        }
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

fn run_inference_model_benchmark(args: &Args, model: &mut InferenceModel, token: u32) {
    println!("Running inference model benchmark (no final logits)...\n");
    println!("  engine: {}", args.engine);
    println!("  mode: {}", args.mode);
    println!("  draft_tokens_per_step: {}", args.draft_tokens);
    if let Some(prompt_tokens) = args.prompt_tokens {
        println!("  prompt_tokens: {}", prompt_tokens);
    }
    println!("  iterations: {}", args.iterations);
    println!();

    let tokens_per_iteration = if matches!(args.mode.as_str(), "prompt" | "pp") {
        args.prompt_tokens.unwrap_or(args.draft_tokens)
    } else {
        args.draft_tokens
    };
    let tokens = vec![token; tokens_per_iteration];
    let mut total_tokens = 0usize;
    let mut total_duration = Duration::ZERO;

    for i in 0..args.iterations {
        let mut session = Session::new();
        model.rewind_to(0).expect("rewind failed");
        let start = Instant::now();
        match args.mode.as_str() {
            "prompt" | "pp" => model
                .forward_tokens_no_logits(&tokens, &mut session)
                .expect("prompt failed"),
            "decode" => {
                for &tok in &tokens {
                    model
                        .forward_tokens_no_logits(&[tok], &mut session)
                        .expect("decode failed");
                }
            }
            other => {
                eprintln!("Error: unsupported --mode '{other}' (expected decode, prompt, or pp)");
                std::process::exit(2);
            }
        }
        let elapsed = start.elapsed();
        total_tokens += tokens_per_iteration;
        total_duration += elapsed;
        if args.verbose {
            let tps = tokens_per_iteration as f64 / elapsed.as_secs_f64();
            println!(
                "  Iteration {}: {} tokens in {:.2?} ({:.2} tok/s)",
                i + 1,
                tokens_per_iteration,
                elapsed,
                tps
            );
        }
    }

    let avg_tps = total_tokens as f64 / total_duration.as_secs_f64();
    let avg_latency = total_duration / total_tokens as u32;
    println!("\n=== Results ===");
    println!("Total tokens: {}", total_tokens);
    println!("Total time: {:.2?}", total_duration);
    println!("Throughput: {:.2} tok/s", avg_tps);
    println!("Avg latency/token: {:.2?}", avg_latency);
    println!("\nNote: inference benchmark skips final logits to measure token processing.");
    println!("\nBenchmark complete.");
}

fn inference_config_from_dflash(
    config: &DFlashConfig,
    context_length: usize,
    key_value_head_dim: usize,
) -> InferenceConfig {
    InferenceConfig {
        vocab_size: config.vocab_size,
        context_size: context_length,
        layer_count: config.num_hidden_layers,
        hidden_size: config.hidden_size,
        intermediate_size: config.intermediate_size,
        num_attention_heads: config.num_attention_heads,
        num_key_value_heads: config.num_key_value_heads,
        key_value_head_dim,
        kv_cache_dtype: oxidize_core::tensor::DType::F32,
        rms_norm_eps: config.rms_norm_eps,
        rope_theta: config.rope_theta,
        architecture: Default::default(),
        sliding_window: 0,
        num_experts: 0,
        num_experts_per_tok: 0,
        alibi_num_heads: 0,
    }
}

fn run_decode_iteration(
    draft_model: &mut DFlashDraftModel,
    current_token: u32,
    draft_tokens: usize,
) -> usize {
    let mut tokens_generated = 0;
    for _ in 0..draft_tokens {
        let _hidden = draft_model
            .forward_token(current_token, None)
            .expect("forward_token failed");
        draft_model.position_offset += 1;
        tokens_generated += 1;
    }
    tokens_generated
}

fn run_prompt_iteration(
    draft_model: &mut DFlashDraftModel,
    token: u32,
    prompt_tokens: usize,
) -> usize {
    let tokens = vec![token; prompt_tokens];
    let mut session = Session::new();
    let _logits = draft_model
        .forward(&tokens, &mut session)
        .expect("prompt forward failed");
    prompt_tokens
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

fn metadata_string(
    metadata: &std::collections::BTreeMap<String, oxidize_core::gguf::GgufMetadataValue>,
    key: &str,
) -> Option<String> {
    use oxidize_core::gguf::GgufMetadataValue;
    match metadata.get(key)? {
        GgufMetadataValue::String(v) => Some(v.clone()),
        _ => None,
    }
}

fn token_embedding_vocab(mapped: &oxidize_core::gguf::MappedGgufFile) -> Option<u32> {
    mapped
        .mapped_tensor_infos()
        .iter()
        .find(|tensor| tensor.name == "token_embd.weight" || tensor.name == "tok_embeddings.weight")
        .and_then(|tensor| tensor.dimensions.get(1).copied())
        .and_then(|value| value.try_into().ok())
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
        kv_cache: vec![DFlashKvLayerCache::new(); config.num_hidden_layers],
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
