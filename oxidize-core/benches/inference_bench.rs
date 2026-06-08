use std::time::{Duration, Instant};

fn gemv(rows: usize, cols: usize, matrix: &[f32], vector: &[f32], output: &mut [f32]) {
    oxidize_core::tensor::gemv_f32(matrix, rows, cols, vector, output).unwrap();
}

fn rms_norm(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    oxidize_core::tensor::rms_norm_f32(input, weight, eps, output).unwrap();
}

fn softmax(input: &[f32], output: &mut [f32]) {
    oxidize_core::tensor::softmax_f32(input, output).unwrap();
}

fn swiglu(gate: &mut [f32], up: &[f32]) {
    oxidize_core::tensor::apply_swiglu_inplace_f32(gate, up);
}

/// Simulates one transformer layer forward pass.
fn layer_forward(
    x: &mut [f32],
    h: usize,
    inter: usize,
    attn_q_w: &[f32],
    attn_k_w: &[f32],
    attn_v_w: &[f32],
    attn_o_w: &[f32],
    ffn_gate_w: &[f32],
    ffn_up_w: &[f32],
    ffn_down_w: &[f32],
    scratch: &mut [f32],
) {
    // --- Attention ---
    let mut q = vec![0.0_f32; h];
    let mut k = vec![0.0_f32; h];
    let mut v = vec![0.0_f32; h];
    let mut attn_out = vec![0.0_f32; h];

    gemv(h, h, attn_q_w, x, &mut q);
    gemv(h, h, attn_k_w, x, &mut k);
    gemv(h, h, attn_v_w, x, &mut v);

    // Simplified attention: Q @ K^T @ V (single head for bench)
    let head_dim = h;
    let mut qk = vec![0.0_f32; 1];
    let scale = 1.0 / (head_dim as f32).sqrt();
    for i in 0..h {
        qk[0] += q[i] * k[i] * scale;
    }
    let mut qk_out = vec![0.0_f32; 1];
    softmax(&qk, &mut qk_out);
    for i in 0..h {
        attn_out[i] = v[i] * qk_out[0];
    }

    gemv(h, h, attn_o_w, &attn_out, scratch);
    for i in 0..h {
        x[i] += scratch[i];
    }

    // --- FFN ---
    let mut gate = vec![0.0_f32; inter];
    let mut up = vec![0.0_f32; inter];
    let mut ffn_out = vec![0.0_f32; h];

    gemv(inter, h, ffn_gate_w, x, &mut gate);
    gemv(inter, h, ffn_up_w, x, &mut up);
    swiglu(&mut gate, &up);
    gemv(h, inter, ffn_down_w, &gate, &mut ffn_out);

    for i in 0..h {
        x[i] += ffn_out[i];
    }
}

fn bench_model(
    vocab: usize,
    h: usize,
    inter: usize,
    layers: usize,
    iters: usize,
) -> Duration {
    // Random weights
    let mut tok_emb = vec![0.0_f32; vocab * h];
    let norm_w = vec![1.0_f32; h];
    let mut lm_head = vec![0.0_f32; vocab * h];
    let mut attn_q = vec![0.0_f32; layers * h * h];
    let mut attn_k = vec![0.0_f32; layers * h * h];
    let mut attn_v = vec![0.0_f32; layers * h * h];
    let mut attn_o = vec![0.0_f32; layers * h * h];
    let mut ffn_gate = vec![0.0_f32; layers * inter * h];
    let mut ffn_up = vec![0.0_f32; layers * inter * h];
    let mut ffn_down = vec![0.0_f32; layers * h * inter];

    for v in tok_emb.iter_mut() { *v = fastrand::f32() * 0.02; }
    for v in lm_head.iter_mut() { *v = fastrand::f32() * 0.02; }
    for v in attn_q.iter_mut() { *v = fastrand::f32() * 0.02; }
    for v in attn_k.iter_mut() { *v = fastrand::f32() * 0.02; }
    for v in attn_v.iter_mut() { *v = fastrand::f32() * 0.02; }
    for v in attn_o.iter_mut() { *v = fastrand::f32() * 0.02; }
    for v in ffn_gate.iter_mut() { *v = fastrand::f32() * 0.02; }
    for v in ffn_up.iter_mut() { *v = fastrand::f32() * 0.02; }
    for v in ffn_down.iter_mut() { *v = fastrand::f32() * 0.02; }

    let token_id = 0_usize;
    let mut x = vec![0.0_f32; h];
    let mut scratch = vec![0.0_f32; h];

    let mut x_normed = vec![0.0_f32; h];
    let mut logits = vec![0.0_f32; vocab];
    let mut probs = vec![0.0_f32; vocab];

    // Warmup
    x.copy_from_slice(&tok_emb[token_id * h..(token_id + 1) * h]);
    rms_norm(&x, &norm_w, 1e-5, &mut x_normed);
    x.copy_from_slice(&x_normed);
    for l in 0..layers {
        layer_forward(
            &mut x, h, inter,
            &attn_q[l * h * h..(l + 1) * h * h],
            &attn_k[l * h * h..(l + 1) * h * h],
            &attn_v[l * h * h..(l + 1) * h * h],
            &attn_o[l * h * h..(l + 1) * h * h],
            &ffn_gate[l * inter * h..(l + 1) * inter * h],
            &ffn_up[l * inter * h..(l + 1) * inter * h],
            &ffn_down[l * h * inter..(l + 1) * h * inter],
            &mut scratch,
        );
    }
    rms_norm(&x, &norm_w, 1e-5, &mut x_normed);
    gemv(vocab, h, &lm_head, &x_normed, &mut logits);
    softmax(&logits, &mut probs);

    // Benchmark
    let start = Instant::now();
    for _ in 0..iters {
        x.copy_from_slice(&tok_emb[token_id * h..(token_id + 1) * h]);
        rms_norm(&x, &norm_w, 1e-5, &mut x_normed);
        x.copy_from_slice(&x_normed);
        for l in 0..layers {
            layer_forward(
                &mut x, h, inter,
                &attn_q[l * h * h..(l + 1) * h * h],
                &attn_k[l * h * h..(l + 1) * h * h],
                &attn_v[l * h * h..(l + 1) * h * h],
                &attn_o[l * h * h..(l + 1) * h * h],
                &ffn_gate[l * inter * h..(l + 1) * inter * h],
                &ffn_up[l * inter * h..(l + 1) * inter * h],
                &ffn_down[l * h * inter..(l + 1) * h * inter],
                &mut scratch,
            );
        }
        rms_norm(&x, &norm_w, 1e-5, &mut x_normed);
        gemv(vocab, h, &lm_head, &x_normed, &mut logits);
    }
    start.elapsed()
}

fn main() {
    println!("=== Oxidize Normal Inference Benchmark ===\n");

    let models = vec![
        ("TinyLlama-1.1B-ish  (n=22, h=2048, inter=5632)", 32000, 2048, 5632, 22, 20),
        ("Llama-7B-ish        (n=32, h=4096, inter=11008)", 32000, 4096, 11008, 32, 10),
        ("Llama-13B-ish       (n=40, h=5120, inter=13824)", 32000, 5120, 13824, 40, 5),
    ];

    for (name, vocab, h, inter, layers, iters) in models {
        println!("{}  –  {} iters", name, iters);
        let dur = bench_model(vocab, h, inter, layers, iters);
        let tps = iters as f64 / dur.as_secs_f64();
        let ms_per = dur.as_secs_f64() * 1000.0 / iters as f64;
        println!("  Throughput: {:.2} tok/s", tps);
        println!("  Latency:    {:.2} ms/token\n", ms_per);
    }
}
