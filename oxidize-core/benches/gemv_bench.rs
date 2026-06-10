use std::time::{Duration, Instant};

fn bench_gemv_f32(rows: usize, cols: usize, iters: usize) -> Duration {
    let matrix = vec![1.0_f32; rows * cols];
    let vector = vec![1.0_f32; cols];
    let mut output = vec![0.0_f32; rows];

    // Warmup
    oxidize_core::tensor::gemv_f32(&matrix, rows, cols, &vector, &mut output,
    ).unwrap();

    let start = Instant::now();
    for _ in 0..iters {
        oxidize_core::tensor::gemv_f32(
            &matrix, rows, cols, &vector, &mut output,
        ).unwrap();
    }
    start.elapsed()
}

fn bench_gemv_q8_0(rows: usize, cols: usize, iters: usize) -> Duration {
    use oxidize_core::gguf::GgufQuantizationType;
    use oxidize_core::quantization::{quantize_scalar, quantized_size};

    let matrix = vec![1.0_f32; rows * cols];
    let vector = vec![1.0_f32; cols];
    let mut output = vec![0.0_f32; rows];

    let mut matrix_bytes = Vec::with_capacity(matrix.len() * 4);
    for v in &matrix {
        matrix_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let qsize = quantized_size(GgufQuantizationType::Q8_0, matrix.len()).unwrap();
    let mut quantized = vec![0_u8; qsize];
    quantize_scalar(
        GgufQuantizationType::F32,
        GgufQuantizationType::Q8_0,
        &matrix_bytes,
        &mut quantized,
    ).unwrap();

    // Warmup
    oxidize_core::tensor::gemv_quantized_f32(
        GgufQuantizationType::Q8_0, &quantized, rows, cols, &vector, &mut output,
    ).unwrap();

    let start = Instant::now();
    for _ in 0..iters {
        oxidize_core::tensor::gemv_quantized_f32(
            GgufQuantizationType::Q8_0, &quantized, rows, cols, &vector, &mut output,
        ).unwrap();
    }
    start.elapsed()
}

fn main() {
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("ERROR: This benchmark requires the 'cuda' feature to be enabled.");
        eprintln!("       Run with: cargo run --bench gemv_bench --features cuda");
        std::process::exit(1);
    }

    #[cfg(feature = "cuda")]
    {
        use oxidize_core::cuda::cuda_build_info;
        let info = cuda_build_info();
        if !info.detected_at_build {
            eprintln!("ERROR: CUDA was not detected at build time.");
            eprintln!("       Re-build with CUDA toolkit installed and the 'cuda' feature enabled.");
            std::process::exit(1);
        }
    }

    println!("=== Oxidize CUDA GEMV Benchmark ===\n");

    let configs = vec![
        ("small  (512×512)", 512, 512, 10000),
        ("medium (4096×4096)", 4096, 4096, 2000),
        ("large  (11008×4096)", 11008, 4096, 1000),
    ];

    for (name, rows, cols, iters) in configs {
        println!("{}  –  {} iterations", name, iters);
        let dur_f32 = bench_gemv_f32(rows, cols, iters);
        let tps_f32 = iters as f64 / dur_f32.as_secs_f64();
        let us_per_f32 = dur_f32.as_secs_f64() * 1e6 / iters as f64;
        println!("  f32 GEMV:  {:.2} ops/s  ({:.3} µs/op)", tps_f32, us_per_f32);

        let dur_q8 = bench_gemv_q8_0(rows, cols, iters);
        let tps_q8 = iters as f64 / dur_q8.as_secs_f64();
        let us_per_q8 = dur_q8.as_secs_f64() * 1e6 / iters as f64;
        println!("  q8_0 GEMV: {:.2} ops/s  ({:.3} µs/op)", tps_q8, us_per_q8);
        println!();
    }
}
