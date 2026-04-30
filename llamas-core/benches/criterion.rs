use std::path::PathBuf;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use llamas_core::benchmark_suite::{
    benchmark_text_perplexity, loader_vs_llama_cpp_cases, perplexity_dataset_cases,
};
use llamas_core::model_loader::{GgufModelLoader, ModelLoader, load_gguf_llama_cpp_baseline};

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

criterion_group!(
    benches,
    benchmark_loader_against_llama_cpp_baseline,
    benchmark_perplexity_on_standard_datasets
);
criterion_main!(benches);
