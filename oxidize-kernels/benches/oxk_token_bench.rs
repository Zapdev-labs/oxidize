//! OXK full decode-token bench — Qwen3-30B-A3B.
//!
//! Times one *real* decode token's worth of GEMVs (every attention + MoE
//! projection across all layers, plus lm_head) so the reported tok/s reflects
//! actual per-token work instead of a single isolated GEMV. tok/s is derived
//! from measured GB/s applied to the known per-token weight-byte volume — no
//! "1 GEMV = 1 token" fiction.
//!
//! Env: OXK_BENCH_LAYERS (default 48), OXK_BENCH_TOKENS (default 5).
//!
//! Keep the plan in sync with oxidize-golang/cmd/bench_oxk/main.go and
//! bench_oxk.py.

use std::hint::black_box;
use std::time::Instant;

use oxidize_kernels::{
    BLOCK_Q4_K_SIZE, BLOCK_Q8_K_BYTES, QK_K, gemv_q4k_range, oxk_avx2_available, oxk_cpu_summary,
    quantize_q8_k_into,
};

// Canonical Qwen3-30B-A3B config.
const HIDDEN: usize = 2048;
const NUM_LAYERS: usize = 48;
const Q_OUT: usize = 4096;
const KV_OUT: usize = 512;
const N_EXPERTS: usize = 8; // active per token
const MOE_INTER: usize = 768;
const ROUTER_OUT: usize = 128;
const VOCAB: usize = 151936;

struct Op {
    rows: usize,
    cols: usize,
    count: usize,
}

fn one_layer_plan() -> Vec<Op> {
    vec![
        Op { rows: Q_OUT, cols: HIDDEN, count: 1 },     // attn.q
        Op { rows: KV_OUT, cols: HIDDEN, count: 1 },    // attn.k
        Op { rows: KV_OUT, cols: HIDDEN, count: 1 },    // attn.v
        Op { rows: HIDDEN, cols: Q_OUT, count: 1 },     // attn.o
        Op { rows: ROUTER_OUT, cols: HIDDEN, count: 1 }, // moe.router
        Op { rows: MOE_INTER, cols: HIDDEN, count: N_EXPERTS }, // moe.gate
        Op { rows: MOE_INTER, cols: HIDDEN, count: N_EXPERTS }, // moe.up
        Op { rows: HIDDEN, cols: MOE_INTER, count: N_EXPERTS }, // moe.down
    ]
}

fn token_plan(n_layers: usize) -> Vec<Op> {
    let mut ops = Vec::new();
    for _ in 0..n_layers {
        ops.extend(one_layer_plan());
    }
    ops.push(Op { rows: VOCAB, cols: HIDDEN, count: 1 }); // lm_head
    ops
}

fn plan_bytes(ops: &[Op]) -> usize {
    ops.iter()
        .map(|o| o.rows * (o.cols / QK_K) * BLOCK_Q4_K_SIZE * o.count)
        .sum()
}

fn plan_flops(ops: &[Op]) -> f64 {
    ops.iter()
        .map(|o| o.rows as f64 * o.cols as f64 * 2.0 * o.count as f64)
        .sum()
}

fn fill_pseudo(bytes: &mut [u8], mut state: u64) {
    for b in bytes.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *b = state as u8;
    }
    // Tame the per-block f16 headers (d, dmin) so accumulators stay finite.
    for block in bytes.chunks_exact_mut(BLOCK_Q4_K_SIZE) {
        for half in 0..2 {
            let raw = u16::from_le_bytes([block[half * 2], block[half * 2 + 1]]);
            let tamed = (raw & 0x83ff) | (0x3000 + ((raw >> 10) & 0x7) * 0x400);
            block[half * 2..half * 2 + 2].copy_from_slice(&tamed.to_le_bytes());
        }
    }
}

fn env_usize(key: &str, def: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(def)
}

fn main() {
    println!("=== Rust OXK full decode-token bench (Qwen3-30B-A3B) ===");
    println!("avx2={} cpu: {}", oxk_avx2_available(), oxk_cpu_summary());

    let n_layers = env_usize("OXK_BENCH_LAYERS", NUM_LAYERS).clamp(1, NUM_LAYERS);
    let tokens = env_usize("OXK_BENCH_TOKENS", 5).max(1);

    let timed_ops = token_plan(n_layers);
    let full_ops = token_plan(NUM_LAYERS);
    let timed_bytes = plan_bytes(&timed_ops);
    let full_bytes = plan_bytes(&full_ops);
    let full_flops = plan_flops(&full_ops);

    // One contiguous weight buffer; each op streams a distinct region so the
    // kernel hits DRAM like a real decode (no cache reuse across ops).
    let mut weights = vec![0_u8; timed_bytes];
    fill_pseudo(&mut weights, 0x5eed);

    // Pre-quantized Q8_K inputs, one per distinct cols width.
    let widths = [HIDDEN, Q_OUT, MOE_INTER];
    let mut q8k_by_cols: Vec<(usize, Vec<u8>)> = Vec::new();
    let mut max_rows = 0;
    for w in widths {
        let blocks = w / QK_K;
        let vector: Vec<f32> = (0..w).map(|i| (i % 255) as f32 / 64.0 - 2.0).collect();
        let mut q8k = vec![0_u8; blocks * BLOCK_Q8_K_BYTES];
        quantize_q8_k_into(&vector, blocks, &mut q8k);
        q8k_by_cols.push((w, q8k));
    }
    for o in &timed_ops {
        max_rows = max_rows.max(o.rows);
    }
    let mut out = vec![0_f32; max_rows];

    let q8k_for = |cols: usize| -> &[u8] {
        &q8k_by_cols.iter().find(|(w, _)| *w == cols).unwrap().1
    };

    let run_token = |weights: &[u8], out: &mut [f32]| -> f32 {
        let mut sink = 0.0_f32;
        let mut cursor = 0;
        for o in &timed_ops {
            let blocks = o.cols / QK_K;
            let row_bytes = blocks * BLOCK_Q4_K_SIZE;
            let q8k = q8k_for(o.cols);
            for _ in 0..o.count {
                let region = &weights[cursor..cursor + o.rows * row_bytes];
                gemv_q4k_range(region, blocks, q8k, &mut out[..o.rows]);
                cursor += o.rows * row_bytes;
                sink += out[0];
            }
        }
        sink
    };

    // Warmup.
    black_box(run_token(&weights, &mut out));

    let start = Instant::now();
    let mut sink = 0.0_f32;
    for _ in 0..tokens {
        sink += run_token(&weights, &mut out);
    }
    let elapsed = start.elapsed().as_secs_f64();
    black_box(sink);

    let gbps = timed_bytes as f64 * tokens as f64 / 1e9 / elapsed;
    let gflops = plan_flops(&timed_ops) * tokens as f64 / 1e9 / elapsed;
    let full_token_sec = full_bytes as f64 / 1e9 / gbps;
    let proj_tok_s = 1.0 / full_token_sec;

    println!("\nTimed: {n_layers} layer(s) + lm_head, {tokens} token-pass(es)  (sink={sink:.3e})");
    println!("Streamed per timed pass: {:.2} MB", timed_bytes as f64 / 1e6);
    println!("Throughput:              {gflops:.2} GFLOP/s");
    println!("Memory bandwidth:        {gbps:.2} GB/s");
    println!(
        "Full token (48L) weight bytes: {:.2} GB, {:.1} GFLOP",
        full_bytes as f64 / 1e9,
        full_flops / 1e9
    );
    println!("Projected full-token decode:   {full_token_sec:.3} s/token  =>  {proj_tok_s:.3} tok/s");
}
