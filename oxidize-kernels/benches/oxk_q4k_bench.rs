//! OXK Q4_K row-dot / GEMV microbench (single-threaded, Gate B input).
//!
//! Reports GB/s of Q4_K weight bytes streamed per kernel variant. Compare
//! against the legacy kernels by running the e2e GEMV bench in oxidize-core
//! with `OXIDIZE_GEMV=legacy|oxk` (same shapes, same thread pool).
//!
//! Env: OXK_BENCH_SECS (default 5, use >=30 for sustained turbo behavior),
//!      OXK_BENCH_DIMS "rows x cols" pairs, e.g. "4096x4096,6144x2048".

use std::hint::black_box;
use std::time::{Duration, Instant};

use oxidize_kernels::{
    gemv_q4k_range, oxk_avx2_available, q4k_q8k_row_dot_scalar, quantize_q8_k_into,
    BLOCK_Q4_K_SIZE, BLOCK_Q8_K_BYTES, QK_K,
};

fn fill_pseudo(bytes: &mut [u8], mut state: u64) {
    for b in bytes {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *b = state as u8;
    }
}

struct Fixture {
    weights: Vec<u8>,
    q8k: Vec<u8>,
    rows: usize,
    blocks_per_row: usize,
}

fn fixture(rows: usize, cols: usize) -> Fixture {
    assert_eq!(cols % QK_K, 0);
    let blocks_per_row = cols / QK_K;
    let mut weights = vec![0_u8; rows * blocks_per_row * BLOCK_Q4_K_SIZE];
    fill_pseudo(&mut weights, 0x5eed);
    // Tame f16 headers so accumulators stay finite.
    for block in weights.chunks_exact_mut(BLOCK_Q4_K_SIZE) {
        for half in 0..2 {
            let raw = u16::from_le_bytes([block[half * 2], block[half * 2 + 1]]);
            let tamed = (raw & 0x83ff) | (0x3000 + ((raw >> 10) & 0x7) * 0x400);
            block[half * 2..half * 2 + 2].copy_from_slice(&tamed.to_le_bytes());
        }
    }
    let vector: Vec<f32> = (0..cols).map(|i| ((i * 37 % 255) as f32 - 127.0) / 64.0).collect();
    let mut q8k = vec![0_u8; blocks_per_row * BLOCK_Q8_K_BYTES];
    quantize_q8_k_into(&vector, blocks_per_row, &mut q8k);
    Fixture { weights, q8k, rows, blocks_per_row }
}

/// Run `body` (one full pass over the matrix) repeatedly for `secs`; return GB/s.
fn time_gbps(fix: &Fixture, secs: f64, mut body: impl FnMut(&Fixture) -> f32) -> f64 {
    // Warmup pass.
    black_box(body(fix));
    let bytes_per_pass = fix.weights.len() as f64;
    let start = Instant::now();
    let mut passes = 0_u64;
    let budget = Duration::from_secs_f64(secs);
    while start.elapsed() < budget {
        black_box(body(fix));
        passes += 1;
    }
    bytes_per_pass * passes as f64 / start.elapsed().as_secs_f64() / 1e9
}

fn main() {
    let secs: f64 = std::env::var("OXK_BENCH_SECS").ok().and_then(|v| v.parse().ok()).unwrap_or(5.0);
    let dims = std::env::var("OXK_BENCH_DIMS").unwrap_or_else(|_| "4096x4096,6144x2048,768x2048".into());
    println!("oxk_q4k_bench: secs/variant={secs} avx2={}", oxk_avx2_available());

    for dim in dims.split(',') {
        let (r, c) = dim.trim().split_once('x').expect("dims as RxC");
        let (rows, cols): (usize, usize) = (r.parse().unwrap(), c.parse().unwrap());
        let fix = fixture(rows, cols);
        let row_bytes = fix.blocks_per_row * BLOCK_Q4_K_SIZE;
        println!("== {rows} rows x {cols} cols ({:.1} MB) ==", fix.weights.len() as f64 / 1e6);

        let scalar = time_gbps(&fix, (secs / 10.0).max(0.5), |f| {
            let mut acc = 0.0;
            for row in f.weights.chunks_exact(row_bytes) {
                acc += q4k_q8k_row_dot_scalar(row, f.blocks_per_row, &f.q8k);
            }
            acc
        });
        println!("  scalar          {scalar:7.3} GB/s");

        if oxk_avx2_available() {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                use oxidize_kernels::{
                    q4k_q8k_row_dot_avx2, q4k_q8k_row_dot_x4_avx2, q4k_q8k_row_dot_x8_avx2,
                };
                let x1 = time_gbps(&fix, secs, |f| {
                    let mut acc = 0.0;
                    for row in f.weights.chunks_exact(row_bytes) {
                        acc += unsafe { q4k_q8k_row_dot_avx2(row, f.blocks_per_row, &f.q8k) };
                    }
                    acc
                });
                println!("  oxk x1          {x1:7.3} GB/s");
                let x4 = time_gbps(&fix, secs, |f| {
                    let mut acc = 0.0;
                    let mut quad = [0.0_f32; 4];
                    let mut r = 0;
                    while r + 4 <= f.rows {
                        unsafe {
                            q4k_q8k_row_dot_x4_avx2(
                                f.weights.as_ptr().add(r * row_bytes),
                                row_bytes,
                                f.blocks_per_row,
                                &f.q8k,
                                &mut quad,
                            )
                        };
                        acc += quad[0];
                        r += 4;
                    }
                    acc
                });
                println!("  oxk x4          {x4:7.3} GB/s");
                let x8 = time_gbps(&fix, secs, |f| {
                    let mut acc = 0.0;
                    let mut octet = [0.0_f32; 8];
                    let mut r = 0;
                    while r + 8 <= f.rows {
                        unsafe {
                            q4k_q8k_row_dot_x8_avx2(
                                f.weights.as_ptr().add(r * row_bytes),
                                row_bytes,
                                f.blocks_per_row,
                                &f.q8k,
                                &mut octet,
                            )
                        };
                        acc += octet[0];
                        r += 8;
                    }
                    acc
                });
                println!("  oxk x8          {x8:7.3} GB/s");
            }
        }

        let mut out = vec![0.0_f32; fix.rows];
        let range = time_gbps(&fix, secs, |f| {
            gemv_q4k_range(&f.weights, f.blocks_per_row, &f.q8k, &mut out);
            out[0]
        });
        println!("  oxk gemv range  {range:7.3} GB/s");
    }
}
