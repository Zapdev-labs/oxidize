use criterion::{Criterion, black_box, criterion_group, criterion_main};
use llamas_core::benchmark_input;

fn benchmark_workspace_health(c: &mut Criterion) {
    c.bench_function("workspace_health", |b| {
        b.iter(|| black_box(benchmark_input()));
    });
}

criterion_group!(benches, benchmark_workspace_health);
criterion_main!(benches);
