use std::path::PathBuf;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use llamas_core::benchmark_suite::loader_vs_llama_cpp_cases;
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

criterion_group!(benches, benchmark_loader_against_llama_cpp_baseline);
criterion_main!(benches);
