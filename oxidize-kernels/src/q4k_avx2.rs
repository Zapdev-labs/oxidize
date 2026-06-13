//! AVX2 Q4_K × Q8_K row-dot kernels: ×1, ×4 and ×8 row variants.
//!
//! Math is bit-identical to the scalar reference.  The performance bet over the
//! legacy kernels is structural: block-level decode (scales, nibble planes) is
//! amortised across the rows in a tile, the accumulators are independent so the
//! out-of-order core overlaps DRAM latency across row streams, and the software
//! prefetcher keeps multiple weight streams well ahead of the ALU.

#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::cpu::OxkTune;
use crate::{
    BLOCK_Q4_K_SIZE, BLOCK_Q8_K_BYTES, QK_K, f16_le_to_f32, get_scale_min_k4, read_q8_k_bsum,
};

/// Decoded Q4_K block state shared by every row in a tile.
#[derive(Clone, Copy)]
struct Q4Block {
    d_w: f32,
    dmin_w: f32,
    /// Per-group scale as i16 broadcast vectors (index = group).
    scale_v: [__m256i; 8],
    /// Per-group min value as i32 (index = group).
    mins: [i32; 8],
    /// Nibble planes for the 4 group-pairs.  `q4_lo[gp]` holds the low nibbles
    /// (group 2*gp) and `q4_hi[gp]` the high nibbles (group 2*gp+1).
    q4_lo: [__m256i; 4],
    q4_hi: [__m256i; 4],
}

/// Prefetch the weight stream for row `r` of a multi-row tile.
/// `w_block` is the current block pointer; `row_bytes` is the distance between
/// the start of consecutive rows.  We prefetch the current block ahead plus,
/// for short rows, the corresponding block in the next tile to help the
/// hardware streamer restart, and for long rows a deeper in-row sweep.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn prefetch_row_stream(
    w_block: *const u8,
    row_bytes: usize,
    blocks_per_row: usize,
    r: usize,
    rows_in_tile: usize,
    tune: OxkTune,
) {
    if tune.pf_bytes == 0 {
        return;
    }
    let ahead = w_block.wrapping_add(tune.pf_bytes).cast::<i8>();
    prefetch3(ahead, tune.pf_nta);

    // Short rows: the hardware prefetcher loses lock when the row ends.  Kick
    // the next tile's stream so it is already moving by the time we get there.
    if blocks_per_row <= 16 {
        let next_tile = w_block.add(r * row_bytes + rows_in_tile * row_bytes);
        let next = next_tile.wrapping_add(tune.pf_bytes).cast::<i8>();
        _mm_prefetch::<{ _MM_HINT_T1 }>(next);
        _mm_prefetch::<{ _MM_HINT_T1 }>(next.wrapping_add(64));
        _mm_prefetch::<{ _MM_HINT_T1 }>(next.wrapping_add(128));
    } else {
        // Long rows: a second, deeper sweep hides latency that the 4-block
        // distance alone cannot cover on contended many-core runs.
        let far = w_block.wrapping_add(16 * BLOCK_Q4_K_SIZE).cast::<i8>();
        _mm_prefetch::<{ _MM_HINT_T1 }>(far);
        _mm_prefetch::<{ _MM_HINT_T1 }>(far.wrapping_add(64));
        _mm_prefetch::<{ _MM_HINT_T1 }>(far.wrapping_add(128));
    }
}

/// Issue three 64-byte-aligned prefetches from `base` using NTA when requested.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn prefetch3(base: *const i8, nta: bool) {
    if nta {
        _mm_prefetch::<{ _MM_HINT_NTA }>(base);
        _mm_prefetch::<{ _MM_HINT_NTA }>(base.wrapping_add(64));
        _mm_prefetch::<{ _MM_HINT_NTA }>(base.wrapping_add(128));
    } else {
        _mm_prefetch::<{ _MM_HINT_T0 }>(base);
        _mm_prefetch::<{ _MM_HINT_T0 }>(base.wrapping_add(64));
        _mm_prefetch::<{ _MM_HINT_T0 }>(base.wrapping_add(128));
    }
}

/// Horizontal sum of 8 packed i32.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn hsum_i32(v: __m256i) -> i32 {
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    let sum128 = _mm_add_epi32(lo, hi);
    let shuf = _mm_shuffle_epi32(sum128, 0b1110);
    let sum64 = _mm_add_epi32(sum128, shuf);
    let shuf2 = _mm_shuffle_epi32(sum64, 0b01);
    let sum32 = _mm_add_epi32(sum64, shuf2);
    _mm_cvtsi128_si32(sum32)
}

/// Decode one Q4_K block into the reusable per-tile form.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn decode_q4_block(w_ptr: *const u8) -> Q4Block {
    let mask = _mm256_set1_epi8(0x0f);
    let d_w = f16_le_to_f32([*w_ptr, *w_ptr.add(1)]);
    let dmin_w = f16_le_to_f32([*w_ptr.add(2), *w_ptr.add(3)]);
    let scales = std::slice::from_raw_parts(w_ptr.add(4), 12);
    let qs = w_ptr.add(16);

    let mut scale_v = [_mm256_setzero_si256(); 8];
    let mut mins = [0_i32; 8];
    let mut q4_lo = [_mm256_setzero_si256(); 4];
    let mut q4_hi = [_mm256_setzero_si256(); 4];

    for gp in 0..4 {
        let g1 = gp * 2;
        let g2 = g1 + 1;
        let (s1, ms1) = get_scale_min_k4(g1, scales);
        let (s2, ms2) = get_scale_min_k4(g2, scales);
        scale_v[g1] = _mm256_set1_epi16(s1 as i16);
        scale_v[g2] = _mm256_set1_epi16(s2 as i16);
        mins[g1] = ms1 as i32;
        mins[g2] = ms2 as i32;

        let packed = _mm256_loadu_si256(qs.add(gp * 32) as *const __m256i);
        q4_lo[gp] = _mm256_and_si256(packed, mask);
        q4_hi[gp] = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
    }

    Q4Block {
        d_w,
        dmin_w,
        scale_v,
        mins,
        q4_lo,
        q4_hi,
    }
}

/// One decoded row dot against pre-loaded Q8_K state.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn row_dot_decoded(b: &Q4Block, d_q8: f32, q8v: &[__m256i; 8], bs: &[i32; 8]) -> f32 {
    let mut vec_pos = _mm256_setzero_si256();
    let mut min_acc: i32 = 0;
    for gp in 0..4 {
        let g1 = gp * 2;
        let g2 = g1 + 1;
        let p16_low = _mm256_maddubs_epi16(b.q4_lo[gp], q8v[g1]);
        let p16_high = _mm256_maddubs_epi16(b.q4_hi[gp], q8v[g2]);
        let p32_low = _mm256_madd_epi16(p16_low, b.scale_v[g1]);
        let p32_high = _mm256_madd_epi16(p16_high, b.scale_v[g2]);
        vec_pos = _mm256_add_epi32(vec_pos, _mm256_add_epi32(p32_low, p32_high));
        min_acc += b.mins[g1] * bs[g1];
        min_acc += b.mins[g2] * bs[g2];
    }
    let pos_acc = hsum_i32(vec_pos);
    b.d_w * d_q8 * pos_acc as f32 - b.dmin_w * d_q8 * min_acc as f32
}

/// Load the shared per-block Q8_K state: scale, the 8 group vectors and the
/// per-group-pair bsum sums.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn load_q8_block(q8_ptr: *const u8) -> (f32, [__m256i; 8], [i32; 8]) {
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
    for (g, bs_g) in bs.iter_mut().enumerate() {
        *bs_g = read_q8_k_bsum(bsums, g * 2) as i32 + read_q8_k_bsum(bsums, g * 2 + 1) as i32;
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
        if tune.pf_bytes != 0 {
            let ahead = w_ptr.wrapping_add(tune.pf_bytes).cast::<i8>();
            prefetch3(ahead, tune.pf_nta);
        }
        let b = decode_q4_block(w_ptr);
        let (d_q8, q8v, bs) = load_q8_block(q8k.as_ptr().add(block_idx * BLOCK_Q8_K_BYTES));
        acc += row_dot_decoded(&b, d_q8, &q8v, &bs);
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
            let w_block = rows_base.add(r * row_bytes + block_idx * BLOCK_Q4_K_SIZE);
            prefetch_row_stream(w_block, row_bytes, blocks_per_row, r, 4, tune);
            let b = decode_q4_block(w_block);
            *acc_r += row_dot_decoded(&b, d_q8, &q8v, &bs);
        }
    }
    *out = acc;
}

/// Dot 8 consecutive rows (spaced `row_bytes`) against one Q8_K vector.
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
            let w_block = rows_base.add(r * row_bytes + block_idx * BLOCK_Q4_K_SIZE);
            prefetch_row_stream(w_block, row_bytes, blocks_per_row, r, 8, tune);
            let b = decode_q4_block(w_block);
            *acc_r += row_dot_decoded(&b, d_q8, &q8v, &bs);
        }
    }
    *out = acc;
}

/// Dot 16 consecutive rows (spaced `row_bytes`) against one Q8_K vector.
///
/// # Safety
/// As [`q4k_q8k_row_dot_avx2`]; `rows_base` must point at 16 valid rows.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn q4k_q8k_row_dot_x16_avx2(
    rows_base: *const u8,
    row_bytes: usize,
    blocks_per_row: usize,
    q8k: &[u8],
    out: &mut [f32; 16],
) {
    let tune = crate::cpu::tune();
    let mut acc = [0.0_f32; 16];
    for block_idx in 0..blocks_per_row {
        let (d_q8, q8v, bs) = load_q8_block(q8k.as_ptr().add(block_idx * BLOCK_Q8_K_BYTES));
        for (r, acc_r) in acc.iter_mut().enumerate() {
            let w_block = rows_base.add(r * row_bytes + block_idx * BLOCK_Q4_K_SIZE);
            prefetch_row_stream(w_block, row_bytes, blocks_per_row, r, 16, tune);
            let b = decode_q4_block(w_block);
            *acc_r += row_dot_decoded(&b, d_q8, &q8v, &bs);
        }
    }
    *out = acc;
}
