use std::path::PathBuf;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use llamas_core::model_loader::{GgufModelLoader, ModelLoader, load_gguf_llama_cpp_baseline};

fn benchmark_loader_against_llama_cpp_baseline(c: &mut Criterion) {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("valid-v3.gguf");

    let loader = GgufModelLoader;
    c.bench_function("loader/mapped_gguf", |b| {
        b.iter(|| {
            let model = loader
                .load(&fixture)
                .expect("mapped loader should parse benchmark fixture");
            black_box(model.parsed().tensor_count)
        });
    });

    c.bench_function("loader/llama_cpp_baseline", |b| {
        b.iter(|| {
            let model = load_gguf_llama_cpp_baseline(&fixture)
                .expect("baseline loader should parse benchmark fixture");
            black_box(model.parsed().tensor_count)
        });
    });
}

criterion_group!(benches, benchmark_loader_against_llama_cpp_baseline);
criterion_main!(benches);
