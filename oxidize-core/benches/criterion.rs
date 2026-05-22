use std::path::PathBuf;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use oxidize_core::benchmark_suite::{
    benchmark_memory_delta_bytes, benchmark_text_perplexity, loader_vs_llama_cpp_cases,
    perplexity_dataset_cases,
};
use oxidize_core::flash_attention::{flash_attention_decode_f32, flash_attention_prefill_f32};
use oxidize_core::model_loader::{GgufModelLoader, ModelLoader, load_gguf_llama_cpp_baseline};

fn benchmark_loader_against_llama_cpp_baseline(c: &mut Criterion) {
    let loader = GgufModelLoader;
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for case in loader_vs_llama_cpp_cases(&manifest_dir) {
        let mapped_name = format!("loader/mapped_gguf/{}", case.name);
        let baseline_name = format!("loader/llama_cpp_baseline/{}", case.name);
        c.bench_function(&mapped_name, |b| {
            b.iter(|| {
                let model = loader
                    .load(&case.path)
                    .expect("mapped loader should parse benchmark fixture");
                black_box(model.parsed().tensor_count)
            });
        });

        c.bench_function(&baseline_name, |b| {
            b.iter(|| {
                let model = load_gguf_llama_cpp_baseline(&case.path)
                    .expect("baseline loader should parse benchmark fixture");
                black_box(model.parsed().tensor_count)
            });
        });
    }
}

fn benchmark_perplexity_on_standard_datasets(c: &mut Criterion) {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for case in perplexity_dataset_cases(&manifest_dir) {
        let benchmark_name = format!("perplexity/dataset/{}", case.name);
        let text = std::fs::read_to_string(&case.path).unwrap_or_else(|_| {
            "this benchmark uses a fallback sample when the dataset file is not available"
                .to_string()
        });
        c.bench_function(&benchmark_name, |b| {
            b.iter(|| {
                black_box(benchmark_text_perplexity(&text));
            });
        });
    }
}

fn benchmark_loader_memory_usage(c: &mut Criterion) {
    let loader = GgufModelLoader;
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for case in loader_vs_llama_cpp_cases(&manifest_dir) {
        let mapped_name = format!("memory/loader/mapped_gguf/{}", case.name);
        let baseline_name = format!("memory/loader/llama_cpp_baseline/{}", case.name);

        c.bench_function(&mapped_name, |b| {
            b.iter(|| {
                let memory_delta = benchmark_memory_delta_bytes(|| {
                    let model = loader
                        .load(&case.path)
                        .expect("mapped loader should parse benchmark fixture");
                    black_box(model.parsed().tensor_count);
                });
                black_box(memory_delta)
            });
        });

        c.bench_function(&baseline_name, |b| {
            b.iter(|| {
                let memory_delta = benchmark_memory_delta_bytes(|| {
                    let model = load_gguf_llama_cpp_baseline(&case.path)
                        .expect("baseline loader should parse benchmark fixture");
                    black_box(model.parsed().tensor_count);
                });
                black_box(memory_delta)
            });
        });
    }
}

fn benchmark_flash_attention_decode(c: &mut Criterion) {
    let head_dim = 128;
    let kv_heads = 8;
    let kv_len = kv_heads * head_dim;
    for seq_len in [64, 256, 512, 1024, 2048] {
        let query: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let key_layer: Vec<f32> = (0..seq_len * kv_len)
            .map(|i| ((i as f32 * 0.007).cos() * 0.5) - 0.1)
            .collect();
        let value_layer: Vec<f32> = (0..seq_len * kv_len)
            .map(|i| ((i as f32 * 0.013).sin() * 0.4) + 0.05)
            .collect();
        let mut output = vec![0.0_f32; head_dim];

        c.bench_function(&format!("flash_attention/decode/{seq_len}"), |b| {
            b.iter(|| {
                flash_attention_decode_f32(
                    black_box(&query),
                    black_box(&key_layer),
                    black_box(&value_layer),
                    seq_len,
                    head_dim,
                    kv_len,
                    0,
                    &mut output,
                )
                .expect("decode should succeed");
                black_box(&output);
            });
        });
    }
}

fn benchmark_flash_attention_prefill(c: &mut Criterion) {
    let head_dim = 128;
    for (q_seq, kv_seq) in [(64, 64), (128, 128), (256, 256), (512, 512)] {
        let query: Vec<f32> = (0..q_seq * head_dim)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let key: Vec<f32> = (0..kv_seq * head_dim)
            .map(|i| (i as f32 * 0.007).cos())
            .collect();
        let value: Vec<f32> = (0..kv_seq * head_dim)
            .map(|i| (i as f32 * 0.013).sin())
            .collect();
        let mut output = vec![0.0_f32; q_seq * head_dim];

        c.bench_function(&format!("flash_attention/prefill/{q_seq}x{kv_seq}"), |b| {
            b.iter(|| {
                flash_attention_prefill_f32(
                    black_box(&query),
                    black_box(&key),
                    black_box(&value),
                    q_seq,
                    kv_seq,
                    head_dim,
                    &mut output,
                )
                .expect("prefill should succeed");
                black_box(&output);
            });
        });
    }
}

criterion_group!(
    benches,
    benchmark_loader_against_llama_cpp_baseline,
    benchmark_perplexity_on_standard_datasets,
    benchmark_loader_memory_usage,
    benchmark_flash_attention_decode,
    benchmark_flash_attention_prefill,
);
criterion_main!(benches);
