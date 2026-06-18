//! OXK: custom Oxidize CPU kernels for quantized GEMV.
//!
//! Phase 1 scope (see `.cursor/plans/xeon-oxk-kernels.md`): Q4_K × Q8_K row
//! dots (scalar reference + AVX2 ×1/×4/×8) and a contiguous-range GEMV helper.
//! The per-row math is bit-identical to the legacy kernels in
//! `oxidize-core/src/compute/tensor.rs` — same integer op sequence and the
//! same per-block f32 accumulation order — so parity tests assert exact
//! equality. OXK's speed bets over legacy are structural: an ×8 multi-row
//! variant (more independent DRAM streams in flight on AVX2-only decode) and
//! a wider software-prefetch window tuned for Xeon Silver.
//!
//! This crate is self-contained (no deps, no oxidize-core) so it can be
//! benchmarked and tested in isolation; `oxidize-core` consumes it behind the
//! optional `oxk` cargo feature with runtime selection via `OXIDIZE_GEMV`.

pub mod cpu;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod q4k_avx2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod q4k_avx512;
mod q4k_dequant;
mod q4k_scalar;
mod q8k;
pub mod prune;

pub use cpu::{CpuInfo, CpuVendor, OxkTune, cpu_vendor, cpuinfo, oxk_cpu_summary};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use q4k_avx2::{
    q4k_q8k_row_dot_avx2, q4k_q8k_row_dot_x4_avx2, q4k_q8k_row_dot_x8_avx2,
    q4k_q8k_row_dot_x16_avx2,
};
pub use q4k_dequant::dequantize_q4_k_into;
pub use q4k_scalar::q4k_q8k_row_dot_scalar;
pub use q8k::quantize_q8_k_into;
pub use prune::{apply_mask_inplace, magnitude_mask, wanda_mask};

/// Values per super-block (matches GGUF K-quants).
pub const QK_K: usize = 256;
/// Bytes per Q4_K block: f16 d + f16 dmin + 12 scale bytes + 128 nibbles.
pub const BLOCK_Q4_K_SIZE: usize = 144;
/// Bytes per Q8_K block: f32 d + 256 int8 + 16 i16 bsums.
pub const BLOCK_Q8_K_BYTES: usize = 4 + 256 + 32;

/// Whether the AVX2 kernels in this crate can run on the current CPU.
#[inline]
pub fn oxk_avx2_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

/// Whether AVX-512F+BW (non-VNNI) kernels can run.
#[inline]
pub fn oxk_avx512_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

/// Whether AVX-512 VNNI kernels can run.
#[inline]
pub fn oxk_avx512vnni_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        oxk_avx512_available() && std::arch::is_x86_feature_detected!("avx512vnni")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

/// Whether AVX-VNNI (256-bit) kernels can run.
#[inline]
pub fn oxk_avxvnni_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        oxk_avx2_available() && std::arch::is_x86_feature_detected!("avxvnni")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

/// Select the best ISA tile size for the detected CPU + env overrides.
/// Resolved ONCE per process: this runs inside `gemv_q4k_range`, which the
/// pool workers call once per chunk — a per-call `env::var` here showed up
/// at >1% of total decode samples (libc getenv scans the environment).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn select_isa() -> &'static str {
    static ISA: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
    ISA.get_or_init(|| match std::env::var("OXIDIZE_OXK_ISA").as_deref() {
        Ok("scalar") => "scalar",
        Ok("avx2") => "avx2",
        Ok("avx512") => "avx512",
        Ok("avx512vnni") => "avx512vnni",
        Ok("avxvnni") => "avxvnni",
        Ok(other) => {
            eprintln!(
                "OXIDIZE_OXK_ISA={other} unknown (use scalar|avx2|avx512|avx512vnni|avxvnni); using auto"
            );
            "auto"
        }
        Err(_) => "auto",
    })
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn select_isa() -> &'static str {
    "scalar"
}

/// Lead multi-row tile width for the AVX2 range GEMV, resolved once per
/// process. Default 16 (the widest) on every vendor, with
/// `OXIDIZE_OXK_TILE={1,4,8,16}` for per-part retuning; the result is
/// bit-identical regardless of width.
///
/// Counterintuitively the WIDEST tile wins in real decode even though a
/// single-threaded microbench prefers x1 (Xeon Silver 4110: x1 = 4.23 GB/s vs
/// x8 = 3.76). The microbench is L3-resident, so it only sees the wide tile's
/// register pressure; real decode streams each expert matrix cold from DRAM,
/// where the wide tile's 16 independent outstanding loads hide memory latency.
/// Interleaved e2e A/B on Qwen3-30B-A3B (28T) was decisive and monotone:
/// tile16 11.7/10.0 > tile8 7.5/7.0 > tile1 4.8/4.3 tok/s — so narrowing the
/// tile on Intel (the microbench's suggestion) would roughly halve decode.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn max_tile() -> usize {
    static TILE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *TILE.get_or_init(|| {
        if let Ok(Ok(t)) = std::env::var("OXIDIZE_OXK_TILE").map(|v| v.parse::<usize>())
            && matches!(t, 1 | 4 | 8 | 16)
        {
            return t;
        }
        16
    })
}

/// Dot a contiguous range of Q4_K rows against one pre-quantized Q8_K vector.
///
/// `rows` must point at `out.len()` rows of `blocks_per_row` Q4_K blocks laid
/// out back-to-back (`row_bytes = blocks_per_row * BLOCK_Q4_K_SIZE` apart);
/// `q8k` holds `blocks_per_row` Q8_K blocks. Uses the widest available ISA
/// (AVX-512 VNNI → AVX-VNNI → AVX-512 → AVX2 → scalar) with ×8 / ×4 / ×1
/// tiling.
pub fn gemv_q4k_range(rows: &[u8], blocks_per_row: usize, q8k: &[u8], out: &mut [f32]) {
    let row_bytes = blocks_per_row * BLOCK_Q4_K_SIZE;
    debug_assert!(rows.len() >= out.len() * row_bytes);
    debug_assert!(q8k.len() >= blocks_per_row * BLOCK_Q8_K_BYTES);

    let isa = select_isa();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // AVX-512 VNNI (Ice Lake / Sapphire Rapids / Granite Rapids)
        if (isa == "avx512vnni" || isa == "auto") && oxk_avx512vnni_available() {
            let n = out.len();
            let mut r = 0;
            while r + 4 <= n {
                let base = unsafe { rows.as_ptr().add(r * row_bytes) };
                let mut quad = [0.0_f32; 4];
                unsafe {
                    q4k_avx512::q4k_q8k_row_dot_x4_avx512vnni(
                        base,
                        row_bytes,
                        blocks_per_row,
                        q8k,
                        &mut quad,
                    )
                };
                out[r..r + 4].copy_from_slice(&quad);
                r += 4;
            }
            while r < n {
                let row = &rows[r * row_bytes..(r + 1) * row_bytes];
                out[r] =
                    unsafe { q4k_avx512::q4k_q8k_row_dot_avx512vnni(row, blocks_per_row, q8k) };
                r += 1;
            }
            return;
        }

        // AVX-VNNI (Alder Lake+ / Zen 4+)
        if (isa == "avxvnni" || isa == "auto") && oxk_avxvnni_available() {
            let n = out.len();
            let mut r = 0;
            while r + 4 <= n {
                let base = unsafe { rows.as_ptr().add(r * row_bytes) };
                let mut quad = [0.0_f32; 4];
                unsafe {
                    q4k_avx512::q4k_q8k_row_dot_x4_avxvnni(
                        base,
                        row_bytes,
                        blocks_per_row,
                        q8k,
                        &mut quad,
                    )
                };
                out[r..r + 4].copy_from_slice(&quad);
                r += 4;
            }
            while r < n {
                let row = &rows[r * row_bytes..(r + 1) * row_bytes];
                out[r] = unsafe { q4k_avx512::q4k_q8k_row_dot_avxvnni(row, blocks_per_row, q8k) };
                r += 1;
            }
            return;
        }

        // AVX-512F/BW (Skylake-SP / Xeon Silver, etc.)
        if oxk_avx512_available() && (isa == "avx512" || (isa == "auto" && cpuinfo().use_avx512)) {
            let n = out.len();
            let mut r = 0;
            while r + 4 <= n {
                let base = unsafe { rows.as_ptr().add(r * row_bytes) };
                let mut quad = [0.0_f32; 4];
                unsafe {
                    q4k_avx512::q4k_q8k_row_dot_x4_avx512(
                        base,
                        row_bytes,
                        blocks_per_row,
                        q8k,
                        &mut quad,
                    )
                };
                out[r..r + 4].copy_from_slice(&quad);
                r += 4;
            }
            while r < n {
                let row = &rows[r * row_bytes..(r + 1) * row_bytes];
                out[r] = unsafe { q4k_avx512::q4k_q8k_row_dot_avx512(row, blocks_per_row, q8k) };
                r += 1;
            }
            return;
        }

        // AVX2 baseline (Haswell+ and Zen). The lead tile width is
        // vendor-tuned (see `max_tile`): wide multi-row tiles amortize the
        // shared Q8_K load but hold 8 Q8 ymm vectors live across 8-16 row
        // dots, so on register-tight cores (Skylake-SP) x1 is fastest while
        // Zen prefers x16. Each width computes a row bit-identically, so the
        // tile choice never changes the result.
        if (isa == "avx2" || isa == "auto") && oxk_avx2_available() {
            let n = out.len();
            let tile = max_tile();
            let mut r = 0;
            while tile >= 16 && r + 16 <= n {
                let base = unsafe { rows.as_ptr().add(r * row_bytes) };
                let mut hex = [0.0_f32; 16];
                unsafe { q4k_q8k_row_dot_x16_avx2(base, row_bytes, blocks_per_row, q8k, &mut hex) };
                out[r..r + 16].copy_from_slice(&hex);
                r += 16;
            }
            while tile >= 8 && r + 8 <= n {
                let base = unsafe { rows.as_ptr().add(r * row_bytes) };
                let mut octet = [0.0_f32; 8];
                unsafe {
                    q4k_q8k_row_dot_x8_avx2(base, row_bytes, blocks_per_row, q8k, &mut octet)
                };
                out[r..r + 8].copy_from_slice(&octet);
                r += 8;
            }
            while tile >= 4 && r + 4 <= n {
                let base = unsafe { rows.as_ptr().add(r * row_bytes) };
                let mut quad = [0.0_f32; 4];
                unsafe { q4k_q8k_row_dot_x4_avx2(base, row_bytes, blocks_per_row, q8k, &mut quad) };
                out[r..r + 4].copy_from_slice(&quad);
                r += 4;
            }
            while r < n {
                let row = &rows[r * row_bytes..(r + 1) * row_bytes];
                out[r] = unsafe { q4k_q8k_row_dot_avx2(row, blocks_per_row, q8k) };
                r += 1;
            }
            return;
        }
    }

    for (r, out_r) in out.iter_mut().enumerate() {
        let row = &rows[r * row_bytes..(r + 1) * row_bytes];
        *out_r = q4k_q8k_row_dot_scalar(row, blocks_per_row, q8k);
    }
}

/// Decode the (scale, min) pair for sub-group `j` from a Q4_K 12-byte scale
/// field (identical to llama.cpp's `get_scale_min_k4`).
#[inline]
pub(crate) fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0x0f) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

/// f16 (little-endian bytes) → f32, no `half` dependency.
#[inline]
pub(crate) fn f16_le_to_f32(bytes: [u8; 2]) -> f32 {
    let bits = u16::from_le_bytes(bytes);
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x03ff) as u32;
    let f32_bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            // Subnormal: normalize.
            let mut frac_norm = frac;
            let mut e = -14_i32;
            while (frac_norm & 0x0400) == 0 {
                frac_norm <<= 1;
                e -= 1;
            }
            frac_norm &= 0x03ff;
            (sign << 31) | (((e + 127) as u32) << 23) | (frac_norm << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        (sign << 31) | ((exp + 112) << 23) | (frac << 13)
    };
    f32::from_bits(f32_bits)
}

#[inline]
pub(crate) unsafe fn read_q8_k_bsum(bsums: *const u8, index: usize) -> i16 {
    let ptr = unsafe { bsums.add(index * 2) };
    i16::from_le_bytes([unsafe { *ptr }, unsafe { *ptr.add(1) }])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random byte stream (xorshift), no rand dep.
    pub(crate) fn fill_pseudo(bytes: &mut [u8], mut state: u64) {
        for b in bytes {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *b = state as u8;
        }
    }

    pub(crate) fn random_fixture(
        rows: usize,
        blocks_per_row: usize,
        seed: u64,
    ) -> (Vec<u8>, Vec<u8>) {
        let mut weights = vec![0_u8; rows * blocks_per_row * BLOCK_Q4_K_SIZE];
        fill_pseudo(&mut weights, seed);
        // Keep f16 d/dmin fields finite and small: rewrite each block header
        // with exponents well inside the f16 normal range.
        for block in weights.chunks_exact_mut(BLOCK_Q4_K_SIZE) {
            for half in 0..2 {
                let raw = u16::from_le_bytes([block[half * 2], block[half * 2 + 1]]);
                let tamed = (raw & 0x83ff) | (0x3000 + ((raw >> 10) & 0x7) * 0x400);
                block[half * 2..half * 2 + 2].copy_from_slice(&tamed.to_le_bytes());
            }
        }
        let mut vector_bytes = vec![0_u8; blocks_per_row * QK_K];
        fill_pseudo(&mut vector_bytes, seed.wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let vector: Vec<f32> = vector_bytes
            .iter()
            .map(|&b| (b as f32 - 127.5) / 32.0)
            .collect();
        let mut q8k = vec![0_u8; blocks_per_row * BLOCK_Q8_K_BYTES];
        quantize_q8_k_into(&vector, blocks_per_row, &mut q8k);
        (weights, q8k)
    }

    #[test]
    fn avx2_variants_match_scalar_exactly() {
        if !oxk_avx2_available() {
            return;
        }
        for &(rows, bpr, seed) in &[(8usize, 16usize, 1u64), (12, 4, 2), (32, 8, 3)] {
            let (weights, q8k) = random_fixture(rows, bpr, seed);
            let row_bytes = bpr * BLOCK_Q4_K_SIZE;
            let scalar: Vec<f32> = (0..rows)
                .map(|r| {
                    q4k_q8k_row_dot_scalar(&weights[r * row_bytes..(r + 1) * row_bytes], bpr, &q8k)
                })
                .collect();
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                for r in 0..rows {
                    let single = unsafe {
                        q4k_q8k_row_dot_avx2(
                            &weights[r * row_bytes..(r + 1) * row_bytes],
                            bpr,
                            &q8k,
                        )
                    };
                    assert_eq!(single.to_bits(), scalar[r].to_bits(), "x1 row {r}");
                }
                let mut quad = [0.0_f32; 4];
                unsafe {
                    q4k_q8k_row_dot_x4_avx2(weights.as_ptr(), row_bytes, bpr, &q8k, &mut quad)
                };
                for r in 0..4 {
                    assert_eq!(quad[r].to_bits(), scalar[r].to_bits(), "x4 row {r}");
                }
                if rows >= 8 {
                    let mut octet = [0.0_f32; 8];
                    unsafe {
                        q4k_q8k_row_dot_x8_avx2(weights.as_ptr(), row_bytes, bpr, &q8k, &mut octet)
                    };
                    for r in 0..8 {
                        assert_eq!(octet[r].to_bits(), scalar[r].to_bits(), "x8 row {r}");
                    }
                }
                if rows >= 16 {
                    let mut hex = [0.0_f32; 16];
                    unsafe {
                        q4k_q8k_row_dot_x16_avx2(weights.as_ptr(), row_bytes, bpr, &q8k, &mut hex)
                    };
                    for r in 0..16 {
                        assert_eq!(hex[r].to_bits(), scalar[r].to_bits(), "x16 row {r}");
                    }
                }
            }
        }
    }

    #[test]
    fn gemv_range_matches_scalar() {
        // 13 rows exercises the x8 + x4 + x1 tail split.
        let (weights, q8k) = random_fixture(13, 8, 7);
        let row_bytes = 8 * BLOCK_Q4_K_SIZE;
        let mut out = vec![0.0_f32; 13];
        gemv_q4k_range(&weights, 8, &q8k, &mut out);
        for r in 0..13 {
            let want =
                q4k_q8k_row_dot_scalar(&weights[r * row_bytes..(r + 1) * row_bytes], 8, &q8k);
            assert_eq!(out[r].to_bits(), want.to_bits(), "row {r}");
        }
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn avxvnni_matches_scalar_exactly() {
        if !oxk_avxvnni_available() {
            return;
        }
        for &(rows, bpr, seed) in &[(8usize, 16usize, 1u64), (12, 4, 2), (32, 8, 3)] {
            let (weights, q8k) = random_fixture(rows, bpr, seed);
            let row_bytes = bpr * BLOCK_Q4_K_SIZE;
            let scalar: Vec<f32> = (0..rows)
                .map(|r| {
                    q4k_q8k_row_dot_scalar(&weights[r * row_bytes..(r + 1) * row_bytes], bpr, &q8k)
                })
                .collect();
            for r in 0..rows {
                let got = unsafe {
                    q4k_avx512::q4k_q8k_row_dot_avxvnni(
                        &weights[r * row_bytes..(r + 1) * row_bytes],
                        bpr,
                        &q8k,
                    )
                };
                assert_eq!(got.to_bits(), scalar[r].to_bits(), "avxvnni row {r}");
            }
            let mut quad = [0.0_f32; 4];
            unsafe {
                q4k_avx512::q4k_q8k_row_dot_x4_avxvnni(
                    weights.as_ptr(),
                    row_bytes,
                    bpr,
                    &q8k,
                    &mut quad,
                )
            };
            for r in 0..4 {
                assert_eq!(quad[r].to_bits(), scalar[r].to_bits(), "avxvnni x4 row {r}");
            }
        }
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn avx512_matches_scalar_exactly() {
        if !oxk_avx512_available() {
            return;
        }
        for &(rows, bpr, seed) in &[(8usize, 16usize, 1u64), (12, 4, 2), (32, 8, 3)] {
            let (weights, q8k) = random_fixture(rows, bpr, seed);
            let row_bytes = bpr * BLOCK_Q4_K_SIZE;
            let scalar: Vec<f32> = (0..rows)
                .map(|r| {
                    q4k_q8k_row_dot_scalar(&weights[r * row_bytes..(r + 1) * row_bytes], bpr, &q8k)
                })
                .collect();
            for r in 0..rows {
                let got = unsafe {
                    q4k_avx512::q4k_q8k_row_dot_avx512(
                        &weights[r * row_bytes..(r + 1) * row_bytes],
                        bpr,
                        &q8k,
                    )
                };
                assert_eq!(got.to_bits(), scalar[r].to_bits(), "avx512 row {r}");
            }
            let mut quad = [0.0_f32; 4];
            unsafe {
                q4k_avx512::q4k_q8k_row_dot_x4_avx512(
                    weights.as_ptr(),
                    row_bytes,
                    bpr,
                    &q8k,
                    &mut quad,
                )
            };
            for r in 0..4 {
                assert_eq!(quad[r].to_bits(), scalar[r].to_bits(), "avx512 x4 row {r}");
            }
        }
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn avx512vnni_matches_scalar_exactly() {
        if !oxk_avx512vnni_available() {
            return;
        }
        for &(rows, bpr, seed) in &[(8usize, 16usize, 1u64), (12, 4, 2), (32, 8, 3)] {
            let (weights, q8k) = random_fixture(rows, bpr, seed);
            let row_bytes = bpr * BLOCK_Q4_K_SIZE;
            let scalar: Vec<f32> = (0..rows)
                .map(|r| {
                    q4k_q8k_row_dot_scalar(&weights[r * row_bytes..(r + 1) * row_bytes], bpr, &q8k)
                })
                .collect();
            for r in 0..rows {
                let got = unsafe {
                    q4k_avx512::q4k_q8k_row_dot_avx512vnni(
                        &weights[r * row_bytes..(r + 1) * row_bytes],
                        bpr,
                        &q8k,
                    )
                };
                assert_eq!(got.to_bits(), scalar[r].to_bits(), "avx512vnni row {r}");
            }
            let mut quad = [0.0_f32; 4];
            unsafe {
                q4k_avx512::q4k_q8k_row_dot_x4_avx512vnni(
                    weights.as_ptr(),
                    row_bytes,
                    bpr,
                    &q8k,
                    &mut quad,
                )
            };
            for r in 0..4 {
                assert_eq!(
                    quad[r].to_bits(),
                    scalar[r].to_bits(),
                    "avx512vnni x4 row {r}"
                );
            }
        }
    }
}
