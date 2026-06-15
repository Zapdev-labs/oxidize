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
    #[arg(long)]
    min_throughput: Option<f64>,
    #[arg(long, default_value_t = 8192)]
    max_context: usize,
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

        if args.engine == "inference" || args.engine == "layerwise" {
            let mut inference_config = InferenceConfig::from_gguf(&mapped);
            if inference_config.context_size > args.max_context {
                inference_config.context_size = args.max_context;
            }
            let benchmark_token = 0_u32;
            println!("InferenceConfig from GGUF:");
            println!("  vocab_size: {}", inference_config.vocab_size);
            println!("  context_size: {}", inference_config.context_size);
            println!("  layer_count: {}", inference_config.layer_count);
            println!("  hidden_size: {}", inference_config.hidden_size);
            println!(
                "  intermediate_size: {}",
                inference_config.intermediate_size
            );
            println!(
                "  num_attention_heads: {}",
                inference_config.num_attention_heads
            );
            println!(
                "  num_key_value_heads: {}",
                inference_config.num_key_value_heads
            );
            println!(
                "  key_value_head_dim: {}",
                inference_config.key_value_head_dim
            );
            println!("  rms_norm_eps: {}", inference_config.rms_norm_eps);
            println!("  rope_theta: {}", inference_config.rope_theta);
            println!("  benchmark_token: {}", benchmark_token);
            println!();

            if args.engine == "inference" {
                let mut model = InferenceModel::load_from_gguf(&mapped, inference_config, true)
                    .expect("Failed to load inference GGUF model");
                run_inference_model_benchmark(&args, &mut model, benchmark_token);
                return;
            }

            let mut model: Box<dyn Model> = Box::new(
                LayerWiseModel::load_from_gguf(&mapped, inference_config, args.layer_cache_size)
                    .expect("Failed to load layer-wise GGUF model"),
            );
            run_standard_model_benchmark(&args, model.as_mut(), benchmark_token);
            return;
        }

        // Extract config from metadata
        let metadata = &mapped.parsed().metadata;
        let arch = metadata_string(metadata, "general.architecture");
        let arch_key = |suffix: &str| arch.as_ref().map(|a| format!("{a}.{suffix}"));
        let arch_u32 = |suffix: &str| arch_key(suffix).and_then(|key| metadata_u32(metadata, &key));
        let arch_f32 = |suffix: &str| arch_key(suffix).and_then(|key| metadata_f32(metadata, &key));
        let inferred = infer_dflash_config_from_tensors(&mapped);
        config = DFlashConfig::from_gguf(&mapped);
        let hidden_size = config.hidden_size;
        let num_layers = config.num_hidden_layers;
        let num_attention_heads = config.num_attention_heads;
        let num_key_value_heads = config.num_key_value_heads;
        let key_value_head_dim = metadata_u32(metadata, "dflash-draft.attention.key_length")
            .or_else(|| arch_u32("attention.key_length"))
            .or(inferred.head_dim.map(|v| v as u32))
            .unwrap_or((hidden_size / num_attention_heads) as u32)
            as usize;
        let intermediate_size = config.intermediate_size;
        let block_size = config.block_size;
        let mask_token_id = config.mask_token_id;
        let n_target_features = config.vocab_size;
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

    print_benchmark_results(total_tokens, total_duration, avg_tps, avg_latency, &args);
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
    print_benchmark_results(total_tokens, total_duration, avg_tps, avg_latency, args);
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
    print_benchmark_results(total_tokens, total_duration, avg_tps, avg_latency, args);
    println!("\nNote: inference benchmark skips final logits to measure token processing.");
    println!("\nBenchmark complete.");
}

fn print_benchmark_results(
    total_tokens: usize,
    total_duration: Duration,
    throughput: f64,
    avg_latency: Duration,
    args: &Args,
) {
    println!("\n=== Results ===");
    println!("Total tokens: {}", total_tokens);
    println!("Total time: {:.2?}", total_duration);
    println!("Throughput: {:.2} tok/s", throughput);
    println!("Avg latency/token: {:.2?}", avg_latency);

    if let Some(min_throughput) = args.min_throughput {
        if !min_throughput.is_finite() || min_throughput <= 0.0 {
            eprintln!("Error: --min-throughput must be a positive finite number");
            std::process::exit(2);
        }

        if throughput < min_throughput {
            eprintln!(
                "Benchmark throughput regression: {:.2} tok/s below required {:.2} tok/s",
                throughput, min_throughput
            );
            std::process::exit(1);
        }

        println!(
            "Throughput gate: passed ({:.2} >= {:.2} tok/s)",
            throughput, min_throughput
        );
    }
}

#[derive(Default)]
struct InferredDFlashConfig {
    head_dim: Option<usize>,
}

fn infer_dflash_config_from_tensors(
    mapped: &oxidize_core::gguf::MappedGgufFile,
) -> InferredDFlashConfig {
    let mut out = InferredDFlashConfig::default();
    let tensors = &mapped.parsed().tensor_infos;
    if let Some(t) = tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_q_norm.weight")
        && let Some(&dim) = t.dimensions.first()
    {
        out.head_dim = Some(dim as usize);
    }
    out
}

#[allow(dead_code)]
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
        kv_quantization: Default::default(),
        rms_norm_eps: config.rms_norm_eps,
        rope_theta: config.rope_theta,
        architecture: Default::default(),
        sliding_window: 0,
        num_experts: 0,
        num_experts_per_tok: 0,
        expert_intermediate_size: 0,
        alibi_num_heads: 0,
        shortconv_l_cache: 0,
        leading_dense_layers: 0,
        expert_gating_sigmoid: false,
        rope_dim: 0,
        rope_theta_swa: 0.0,
        sliding_window_pattern: 0,
        embedding_scale: 1.0,
        gelu_ffn: false,
        sandwich_norm: false,
        rms_norm_weight_plus_one: false,
        nextn_predict_layers: 0,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_min_throughput_gate() {
        let args = Args::parse_from([
            "oxidize-bench",
            "--random-weights",
            "--mode",
            "pp",
            "--prompt-tokens",
            "512",
            "--min-throughput",
            "30.1",
        ]);

        assert_eq!(args.min_throughput, Some(30.1));
        assert_eq!(args.mode, "pp");
        assert_eq!(args.prompt_tokens, Some(512));
    }
}
