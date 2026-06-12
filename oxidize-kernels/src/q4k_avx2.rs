//! AVX2 Q4_K × Q8_K row-dot kernels: ×1, ×4 and ×8 row variants.
//!
//! Math is bit-identical to the scalar reference (and to oxidize-core's
//! legacy `q4_k_q8_k_row_dot_avx2` / `_x4_avx2`): `maddubs` pair sums peak at
//! 3810 so the i16 stage never saturates, the per-block scale `madd` stays in
//! i32 range, and the f32 combine order per block is identical. The multi-row
//! variants share the Q8_K loads and bsum pair-sums across rows and keep one
//! independent accumulator chain per row so the out-of-order core overlaps
//! DRAM latency across row streams; ×8 doubles the streams in flight versus
//! the legacy ×4 ceiling (the OXK bet for AVX2-only Xeons).

#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::cpu::OxkTune;
use crate::{
    BLOCK_Q4_K_SIZE, BLOCK_Q8_K_BYTES, QK_K, f16_le_to_f32, get_scale_min_k4, read_q8_k_bsum,
};

/// Prefetch the weight block `tune.pf_bytes` ahead of `w_ptr` (one Q4_K block
/// spans 144 B ≈ 3 cache lines). Distance and hint come from
/// [`crate::cpu::tune`] (per-vendor default + `OXIDIZE_OXK_PF` /
/// `OXIDIZE_OXK_PF_HINT` overrides); NTA keeps once-per-token weight streams
/// from evicting KV/activations out of L3. The hint branch is perfectly
/// predicted (same arm every call), so the runtime tune costs nothing
/// measurable.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn prefetch_row_ahead(w_ptr: *const u8, tune: OxkTune) {
    if tune.pf_bytes == 0 {
        return;
    }
    let ahead = w_ptr.wrapping_add(tune.pf_bytes).cast::<i8>();
    if tune.pf_nta {
        _mm_prefetch::<{ _MM_HINT_NTA }>(ahead);
        _mm_prefetch::<{ _MM_HINT_NTA }>(ahead.wrapping_add(64));
        _mm_prefetch::<{ _MM_HINT_NTA }>(ahead.wrapping_add(128));
    } else {
        _mm_prefetch::<{ _MM_HINT_T0 }>(ahead);
        _mm_prefetch::<{ _MM_HINT_T0 }>(ahead.wrapping_add(64));
        _mm_prefetch::<{ _MM_HINT_T0 }>(ahead.wrapping_add(128));
    }
}

/// Horizontal sum of 8 packed i32.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum_i32(v: __m256i) -> i32 {
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    let sum128 = _mm_add_epi32(lo, hi);
    let shuf = _mm_shuffle_epi32(sum128, 0b1110);
    let sum64 = _mm_add_epi32(sum128, shuf);
    let shuf2 = _mm_shuffle_epi32(sum64, 0b01);
    let sum32 = _mm_add_epi32(sum64, shuf2);
    _mm_cvtsi128_si32(sum32)
}

/// Process one row's Q4_K block against pre-loaded Q8_K vectors / bsum sums.
/// Returns this block's f32 contribution.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn block_dot_one_row(w_ptr: *const u8, d_q8: f32, q8v: &[__m256i; 8], bs: &[i32; 8]) -> f32 {
    let mask = _mm256_set1_epi8(0x0f);
    let d_w = f16_le_to_f32([*w_ptr, *w_ptr.add(1)]);
    let dmin_w = f16_le_to_f32([*w_ptr.add(2), *w_ptr.add(3)]);
    let scales = std::slice::from_raw_parts(w_ptr.add(4), 12);
    let qs = w_ptr.add(16);

    let mut vec_pos = _mm256_setzero_si256();
    let mut min_acc: i32 = 0;
    for gp in 0..4 {
        let g1 = gp * 2;
        let g2 = g1 + 1;
        let (s1, ms1) = get_scale_min_k4(g1, scales);
        let (s2, ms2) = get_scale_min_k4(g2, scales);
        let packed = _mm256_loadu_si256(qs.add(gp * 32) as *const __m256i);
        let q4_low = _mm256_and_si256(packed, mask);
        let q4_high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
        let p16_low = _mm256_maddubs_epi16(q4_low, q8v[g1]);
        let p16_high = _mm256_maddubs_epi16(q4_high, q8v[g2]);
        // madd(p16, set1_epi16(s)) == s * (p0 + p1) per i32 lane; avoids the
        // slow mullo_epi32. No overflow: |p16| <= 3810, s <= 63.
        let p32_low = _mm256_madd_epi16(p16_low, _mm256_set1_epi16(s1 as i16));
        let p32_high = _mm256_madd_epi16(p16_high, _mm256_set1_epi16(s2 as i16));
        vec_pos = _mm256_add_epi32(vec_pos, _mm256_add_epi32(p32_low, p32_high));
        min_acc += ms1 as i32 * bs[g1];
        min_acc += ms2 as i32 * bs[g2];
    }
    let pos_acc = hsum_i32(vec_pos);
    d_w * d_q8 * pos_acc as f32 - dmin_w * d_q8 * min_acc as f32
}

/// Load the shared per-block Q8_K state: scale, the 8 group vectors and the
/// per-group-pair bsum sums.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn load_q8_block(q8_ptr: *const u8) -> (f32, [__m256i; 8], [i32; 8]) {
    let d_q8 = f32::from_le_bytes([*q8_ptr, *q8_ptr.add(1), *q8_ptr.add(2), *q8_ptr.add(3)]);
    let q8 = q8_ptr.add(4);
    let bsums = q8_ptr.add(4 + QK_K);
    let q8v = [
        _mm256_loadu_si256(q8 as *const __m256i),
        _mm256_loadu_si256(q8.add(32) as *const __m256i),
        _mm256_loadu_si256(q8.add(64) as *const __m256i),
        _mm256_loadu_si256(q8.add(96) as *const __m256i),
        _mm256_loadu_si256(q8.add(128) as *const __m256i),
        _mm256_loadu_si256(q8.add(160) as *const __m256i),
        _mm256_loadu_si256(q8.add(192) as *const __m256i),
        _mm256_loadu_si256(q8.add(224) as *const __m256i),
    ];
    let mut bs = [0_i32; 8];
    for (g, b) in bs.iter_mut().enumerate() {
        *b = read_q8_k_bsum(bsums, g * 2) as i32 + read_q8_k_bsum(bsums, g * 2 + 1) as i32;
    }
    (d_q8, q8v, bs)
}

/// Single-row Q4_K × Q8_K dot.
///
/// # Safety
/// Caller must verify AVX2+FMA; `row` holds `blocks_per_row` Q4_K blocks and
/// `q8k` the matching Q8_K blocks.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn q4k_q8k_row_dot_avx2(row: &[u8], blocks_per_row: usize, q8k: &[u8]) -> f32 {
    let tune = crate::cpu::tune();
    let mut acc = 0.0_f32;
    for block_idx in 0..blocks_per_row {
        let w_ptr = row.as_ptr().add(block_idx * BLOCK_Q4_K_SIZE);
        prefetch_row_ahead(w_ptr, tune);
        let (d_q8, q8v, bs) = load_q8_block(q8k.as_ptr().add(block_idx * BLOCK_Q8_K_BYTES));
        acc += block_dot_one_row(w_ptr, d_q8, &q8v, &bs);
    }
    acc
}

/// Dot 4 consecutive rows (spaced `row_bytes`) against one Q8_K vector.
///
/// # Safety
/// As [`q4k_q8k_row_dot_avx2`]; `rows_base` must point at 4 valid rows.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn q4k_q8k_row_dot_x4_avx2(
    rows_base: *const u8,
    row_bytes: usize,
    blocks_per_row: usize,
    q8k: &[u8],
    out: &mut [f32; 4],
) {
    let tune = crate::cpu::tune();
    let mut acc = [0.0_f32; 4];
    for block_idx in 0..blocks_per_row {
        let (d_q8, q8v, bs) = load_q8_block(q8k.as_ptr().add(block_idx * BLOCK_Q8_K_BYTES));
        for (r, acc_r) in acc.iter_mut().enumerate() {
            let w_ptr = rows_base.add(r * row_bytes + block_idx * BLOCK_Q4_K_SIZE);
            prefetch_row_ahead(w_ptr, tune);
            *acc_r += block_dot_one_row(w_ptr, d_q8, &q8v, &bs);
        }
    }
    *out = acc;
}

/// Dot 8 consecutive rows (spaced `row_bytes`) against one Q8_K vector.
///
/// 8 independent weight streams + accumulator chains per block. On
/// memory-bound AVX2 decode this doubles the outstanding DRAM line fills
/// versus ×4 while still sharing every Q8_K load.
///
/// # Safety
/// As [`q4k_q8k_row_dot_avx2`]; `rows_base` must point at 8 valid rows.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn q4k_q8k_row_dot_x8_avx2(
    rows_base: *const u8,
    row_bytes: usize,
    blocks_per_row: usize,
    q8k: &[u8],
    out: &mut [f32; 8],
) {
    let tune = crate::cpu::tune();
    let mut acc = [0.0_f32; 8];
    for block_idx in 0..blocks_per_row {
        let (d_q8, q8v, bs) = load_q8_block(q8k.as_ptr().add(block_idx * BLOCK_Q8_K_BYTES));
        for (r, acc_r) in acc.iter_mut().enumerate() {
            let w_ptr = rows_base.add(r * row_bytes + block_idx * BLOCK_Q4_K_SIZE);
            prefetch_row_ahead(w_ptr, tune);
            *acc_r += block_dot_one_row(w_ptr, d_q8, &q8v, &bs);
        }
    }
    *out = acc;
}
