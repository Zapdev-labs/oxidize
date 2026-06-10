use std::time::{Duration, Instant};

fn gemv(rows: usize, cols: usize, matrix: &[f32], vector: &[f32], output: &mut [f32]) {
    oxidize_core::tensor::gemv_f32(matrix, rows, cols, vector, output)
        .expect("gemv_f32 should not fail with valid dimensions");
}

fn bench_layer_by_layer(
    vocab: usize,
    h: usize,
    inter: usize,
    layers: usize,
    max_resident: usize,
    iters: usize,
) -> (Duration, usize) {
    // Random weights per layer
    let mut attn_q: Vec<Vec<f32>> = Vec::with_capacity(layers);
    let mut attn_k: Vec<Vec<f32>> = Vec::with_capacity(layers);
    let mut attn_v: Vec<Vec<f32>> = Vec::with_capacity(layers);
    let mut attn_o: Vec<Vec<f32>> = Vec::with_capacity(layers);
    let mut ffn_gate: Vec<Vec<f32>> = Vec::with_capacity(layers);
    let mut ffn_up: Vec<Vec<f32>> = Vec::with_capacity(layers);
    let mut ffn_down: Vec<Vec<f32>> = Vec::with_capacity(layers);

    for _ in 0..layers {
        let mut w = vec![0.0_f32; h * h];
        for v in w.iter_mut() { *v = fastrand::f32() * 0.02; }
        attn_q.push(w);
        let mut w = vec![0.0_f32; h * h];
        for v in w.iter_mut() { *v = fastrand::f32() * 0.02; }
        attn_k.push(w);
        let mut w = vec![0.0_f32; h * h];
        for v in w.iter_mut() { *v = fastrand::f32() * 0.02; }
        attn_v.push(w);
        let mut w = vec![0.0_f32; h * h];
        for v in w.iter_mut() { *v = fastrand::f32() * 0.02; }
        attn_o.push(w);
        let mut w = vec![0.0_f32; inter * h];
        for v in w.iter_mut() { *v = fastrand::f32() * 0.02; }
        ffn_gate.push(w);
        let mut w = vec![0.0_f32; inter * h];
        for v in w.iter_mut() { *v = fastrand::f32() * 0.02; }
        ffn_up.push(w);
        let mut w = vec![0.0_f32; h * inter];
        for v in w.iter_mut() { *v = fastrand::f32() * 0.02; }
        ffn_down.push(w);
    }

    let mut x = vec![0.0_f32; h];
    let mut scratch = vec![0.0_f32; h];
    let mut bufs = LayerGemvBuffers::new(h, inter);

    #[cfg(feature = "cuda")]
    {
        use oxidize_core::cuda::{set_layer_config, preload_layer, CudaLayerConfig};
        set_layer_config(CudaLayerConfig {
            max_resident_layers: max_resident,
            max_vram_bytes: 0,
        }).expect("set_layer_config should succeed");

        // Preload initial layers
        for l in 0..layers.min(max_resident) {
            preload_layer(l, &[
                (&attn_q[l], h, h),
                (&attn_k[l], h, h),
                (&attn_v[l], h, h),
                (&attn_o[l], h, h),
                (&ffn_gate[l], inter, h),
                (&ffn_up[l], inter, h),
                (&ffn_down[l], h, inter),
            ]).expect("preload_layer should succeed");
        }
    }

    // Warmup
    for l in 0..layers {
        #[cfg(feature = "cuda")]
        {
            use oxidize_core::cuda::preload_layer;
            preload_layer(l, &[
                (&attn_q[l], h, h),
                (&attn_k[l], h, h),
                (&attn_v[l], h, h),
                (&attn_o[l], h, h),
                (&ffn_gate[l], inter, h),
                (&ffn_up[l], inter, h),
                (&ffn_down[l], h, inter),
            ]).expect("preload_layer should succeed");
        }
        layer_gemvs(l, h, inter, &attn_q, &attn_k, &attn_v, &attn_o,
                    &ffn_gate, &ffn_up, &ffn_down, &mut x, &mut scratch, &mut bufs);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iters {
        x.fill(0.0);
        for l in 0..layers {
            #[cfg(feature = "cuda")]
            {
                use oxidize_core::cuda::preload_layer;
                preload_layer(l, &[
                    (&attn_q[l], h, h),
                    (&attn_k[l], h, h),
                    (&attn_v[l], h, h),
                    (&attn_o[l], h, h),
                    (&ffn_gate[l], inter, h),
                    (&ffn_up[l], inter, h),
                    (&ffn_down[l], h, inter),
                ]).expect("preload_layer should succeed");
            }
            layer_gemvs(l, h, inter, &attn_q, &attn_k, &attn_v, &attn_o,
                        &ffn_gate, &ffn_up, &ffn_down, &mut x, &mut scratch, &mut bufs);
        }
    }
    let elapsed = start.elapsed();

    #[cfg(feature = "cuda")]
    {
        use oxidize_core::cuda::resident_vram_bytes;
        let bytes = resident_vram_bytes();
        (elapsed, bytes)
    }
    #[cfg(not(feature = "cuda"))]
    {
        (elapsed, 0)
    }
}

struct LayerGemvBuffers {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    ffn_out: Vec<f32>,
}

impl LayerGemvBuffers {
    fn new(h: usize, inter: usize) -> Self {
        Self {
            q: vec![0.0_f32; h],
            k: vec![0.0_f32; h],
            v: vec![0.0_f32; h],
            attn_out: vec![0.0_f32; h],
            gate: vec![0.0_f32; inter],
            up: vec![0.0_f32; inter],
            ffn_out: vec![0.0_f32; h],
        }
    }
}

fn layer_gemvs(
    l: usize,
    h: usize,
    inter: usize,
    attn_q: &[Vec<f32>],
    attn_k: &[Vec<f32>],
    attn_v: &[Vec<f32>],
    attn_o: &[Vec<f32>],
    ffn_gate: &[Vec<f32>],
    ffn_up: &[Vec<f32>],
    ffn_down: &[Vec<f32>],
    x: &mut [f32],
    scratch: &mut [f32],
    bufs: &mut LayerGemvBuffers,
) {
    let LayerGemvBuffers { q, k, v, attn_out, gate, up, ffn_out } = bufs;

    q.fill(0.0);
    k.fill(0.0);
    v.fill(0.0);
    attn_out.fill(0.0);
    gate.fill(0.0);
    up.fill(0.0);
    ffn_out.fill(0.0);

    gemv(h, h, &attn_q[l], x, q);
    gemv(h, h, &attn_k[l], x, k);
    gemv(h, h, &attn_v[l], x, v);

    let head_dim = h;
    let mut qk = 0.0_f32;
    let scale = 1.0 / (head_dim as f32).sqrt();
    for i in 0..h {
        qk += q[i] * k[i] * scale;
    }
    let qk_softmax = 1.0_f32; // single element softmax is identity
    for i in 0..h {
        attn_out[i] = v[i] * qk_softmax;
    }

    gemv(h, h, &attn_o[l], attn_out, scratch);
    for i in 0..h {
        x[i] += scratch[i];
    }

    gemv(inter, h, &ffn_gate[l], x, gate);
    gemv(inter, h, &ffn_up[l], x, up);
    for i in 0..inter {
        gate[i] = gate[i] * (1.0 / (1.0 + (-gate[i]).exp())) * up[i];
    }
    gemv(h, inter, &ffn_down[l], gate, ffn_out);

    for i in 0..h {
        x[i] += ffn_out[i];
    }
}

fn main() {
    println!("=== Oxidize Layer-by-Layer VRAM Benchmark ===\n");

    let h = 4096_usize;
    let inter = 11008_usize;
    let layers = 32_usize;
    let iters = 20_usize;

    println!("Model: {} layers, hidden={}, inter={}", layers, h, inter);
    println!("Iterations per config: {}\n", iters);

    // Calculate approximate weight size per layer
    let bytes_per_layer = (
        4 * h * h +   // 4 attention projections
        2 * inter * h + // gate + up
        1 * h * inter   // down
    ) * std::mem::size_of::<f32>();
    println!("Approx weight bytes per layer: {:.1} MB", bytes_per_layer as f64 / 1e6);
    println!("Total model weights: {:.1} MB\n", (bytes_per_layer * layers) as f64 / 1e6);

    // Benchmark 1: All layers resident (unlimited)
    println!("[Config 1] All {} layers resident", layers);
    let (dur_all, vram_all) = bench_layer_by_layer(32000, h, inter, layers, layers, iters);
    let tps_all = (iters * layers) as f64 / dur_all.as_secs_f64();
    println!("  Throughput: {:.2} layers/s", tps_all);
    println!("  VRAM used:  {:.1} MB\n", vram_all as f64 / 1e6);

    // Benchmark 2: Only 2 layers resident (layer-by-layer)
    println!("[Config 2] Only 2 layers resident (LRU eviction)");
    let (dur_2, vram_2) = bench_layer_by_layer(32000, h, inter, layers, 2, iters);
    let tps_2 = (iters * layers) as f64 / dur_2.as_secs_f64();
    println!("  Throughput: {:.2} layers/s", tps_2);
    println!("  VRAM used:  {:.1} MB\n", vram_2 as f64 / 1e6);

    // Benchmark 3: Only 1 layer resident
    println!("[Config 3] Only 1 layer resident (LRU eviction)");
    let (dur_1, vram_1) = bench_layer_by_layer(32000, h, inter, layers, 1, iters);
    let tps_1 = (iters * layers) as f64 / dur_1.as_secs_f64();
    println!("  Throughput: {:.2} layers/s", tps_1);
    println!("  VRAM used:  {:.1} MB\n", vram_1 as f64 / 1e6);

    println!("=== Summary ===");
    println!("All layers:     {:.2} layers/s,  {:.1} MB VRAM", tps_all, vram_all as f64 / 1e6);
    println!("2-layer cache:  {:.2} layers/s,  {:.1} MB VRAM  ({:.1}% of full speed)",
             tps_2, vram_2 as f64 / 1e6, tps_2 / tps_all * 100.0);
    println!("1-layer cache:  {:.2} layers/s,  {:.1} MB VRAM  ({:.1}% of full speed)",
             tps_1, vram_1 as f64 / 1e6, tps_1 / tps_all * 100.0);
}
