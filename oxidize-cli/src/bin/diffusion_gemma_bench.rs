//! Block-diffusion DiffusionGemma benchmark on the OXK kernels.
//!
//! Usage: diffusion_gemma_bench <model.gguf> [prompt] [steps]
//! Runs one denoise canvas and reports canvas tok/s plus the per-step mean-entropy trace
//! (which should collapse toward the StableAndConfident stop, mirroring the reference).

use std::env;
use std::path::Path;

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args
        .get(1)
        .expect("Usage: diffusion_gemma_bench <model.gguf> [prompt] [steps]");
    let prompt_text = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "What is the capital of France?".to_string());
    let steps: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(oxidize_core::diffusion_gemma::STEPS);

    eprintln!("loading {path} ...");
    let t_load = std::time::Instant::now();
    let model = oxidize_core::diffusion_gemma::DiffusionGemma::load(path).expect("load failed");
    eprintln!("loaded in {:.1}s", t_load.elapsed().as_secs_f64());

    // tokenize the prompt (fall back to a bare BOS prefix if no tokenizer)
    let tokenizer = oxidize_core::tokenizer::load_tokenizer_from_gguf_file(Some(Path::new(path)))
        .ok()
        .flatten();
    let prompt: Vec<u32> = match &tokenizer {
        Some(tok) => {
            let mut ids = vec![2u32]; // BOS
            ids.extend(tok.encode(&prompt_text));
            ids
        }
        None => vec![2u32],
    };
    eprintln!("prompt tokens: {}", prompt.len());

    let stats = model
        .generate(&prompt, steps, 1234)
        .expect("generation failed");

    println!("=== diffusion-gemma (OXK) ===");
    for (step, ent, acc) in &stats.entropy_trace {
        println!(
            "step {step:3}  mean_entropy={ent:.4}  accepted={acc}/{}",
            stats.canvas_tokens
        );
    }
    if let Some(tok) = &tokenizer {
        if let Ok(text) = tok.decode(&stats.tokens) {
            println!("=== canvas (decoded) ===\n{text}");
        }
    }
    println!("=== perf ===");
    println!(
        "1 block, {} denoising steps, {} canvas tokens in {:.2} s ({:.2} canvas tok/s, {:.3} s/step)",
        stats.steps_run,
        stats.canvas_tokens,
        stats.gen_secs,
        stats.canvas_tok_s,
        stats.gen_secs / stats.steps_run as f64,
    );
}
