//! AVX-512 / VNNI Q4_K × Q8_K row-dot kernels.
//!
//! Three paths live here:
//!   * AVX-512F/BW (non-VNNI) — for Skylake-SP / Xeon Silver and other AVX-512
//!     parts without VNNI.  Uses 512-bit `maddubs`/`madd` to process two groups
//!     per instruction versus one in AVX2.
//!   * AVX-512 VNNI — for Ice Lake / Sapphire Rapids / Granite Rapids.
//!   * AVX-VNNI (256-bit) — for Alder Lake+ client and Zen 4+.
//!
//! All paths stay bit-identical to the scalar reference: integer sums are
//! accumulated in the same group order and the final f32 combine is per-block.

#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::{
    BLOCK_Q4_K_SIZE, BLOCK_Q8_K_BYTES, QK_K, f16_le_to_f32, get_scale_min_k4, read_q8_k_bsum,
};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn load_q8_block_512(q8_block: &[u8]) -> (f32, [__m512i; 4], [i32; 8]) {
    debug_assert!(q8_block.len() >= BLOCK_Q8_K_BYTES);
    let d_q8 = f32::from_le_bytes([q8_block[0], q8_block[1], q8_block[2], q8_block[3]]);
    let q8 = q8_block[4..].as_ptr();
    let bsums = &q8_block[4 + QK_K..];
    let q8v = [
        _mm512_loadu_si512(q8 as *const __m512i),
        _mm512_loadu_si512(q8.add(64) as *const __m512i),
        _mm512_loadu_si512(q8.add(128) as *const __m512i),
        _mm512_loadu_si512(q8.add(192) as *const __m512i),
    ];
    let mut bs = [0_i32; 8];
    for (g, bs_g) in bs.iter_mut().enumerate() {
        *bs_g = read_q8_k_bsum(bsums, g * 2) as i32 + read_q8_k_bsum(bsums, g * 2 + 1) as i32;
    }
    (d_q8, q8v, bs)
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn decode_q4_block_512(w_ptr: *const u8) -> Q4Block512 {
    let mask = _mm256_set1_epi8(0x0f);
    let d_w = f16_le_to_f32([*w_ptr, *w_ptr.add(1)]);
    let dmin_w = f16_le_to_f32([*w_ptr.add(2), *w_ptr.add(3)]);
    let scales = std::slice::from_raw_parts(w_ptr.add(4), 12);
    let qs = w_ptr.add(16);

    let mut q4_512 = [_mm512_setzero_si512(); 4];
    let mut scale_v = [_mm512_setzero_si512(); 4];
    let mut mins = [0_i32; 8];

    for gp in 0..4 {
        let g1 = gp * 2;
        let g2 = g1 + 1;
        let (s1, ms1) = get_scale_min_k4(g1, scales);
        let (s2, ms2) = get_scale_min_k4(g2, scales);
        mins[g1] = ms1 as i32;
        mins[g2] = ms2 as i32;

        let packed = _mm256_loadu_si256(qs.add(gp * 32) as *const __m256i);
        let q4_low = _mm256_and_si256(packed, mask);
        let q4_high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
        q4_512[gp] = _mm512_inserti64x4(_mm512_castsi256_si512(q4_low), q4_high, 1);

        let s_low = _mm256_set1_epi16(s1 as i16);
        let s_high = _mm256_set1_epi16(s2 as i16);
        scale_v[gp] = _mm512_inserti64x4(_mm512_castsi256_si512(s_low), s_high, 1);
    }

    Q4Block512 {
        d_w,
        dmin_w,
        q4_512,
        scale_v,
        mins,
    }
}

#[derive(Clone, Copy)]
struct Q4Block512 {
    d_w: f32,
    dmin_w: f32,
    q4_512: [__m512i; 4],
    scale_v: [__m512i; 4],
    mins: [i32; 8],
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn row_dot_decoded_512(b: &Q4Block512, d_q8: f32, q8v: &[__m512i; 4], bs: &[i32; 8]) -> f32 {
    let mut vec_pos = _mm512_setzero_si512();
    let mut min_acc: i32 = 0;
    for (gp, q8v_gp) in q8v.iter().enumerate() {
        let g1 = gp * 2;
        let g2 = g1 + 1;
        let p16 = _mm512_maddubs_epi16(b.q4_512[gp], *q8v_gp);
        let p32 = _mm512_madd_epi16(p16, b.scale_v[gp]);
        vec_pos = _mm512_add_epi32(vec_pos, p32);
        min_acc += b.mins[g1] * bs[g1];
        min_acc += b.mins[g2] * bs[g2];
    }
    let pos_acc = _mm512_reduce_add_epi32(vec_pos);
    b.d_w * d_q8 * pos_acc as f32 - b.dmin_w * d_q8 * min_acc as f32
}

// ---------------------------------------------------------------------------
// AVX-512F/BW (no VNNI)
// ---------------------------------------------------------------------------

/// Single-row Q4_K × Q8_K dot using AVX-512F/BW.
///
/// # Safety
/// Caller must verify AVX-512F+BW support.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn q4k_q8k_row_dot_avx512(row: &[u8], blocks_per_row: usize, q8k: &[u8]) -> f32 {
    let tune = crate::cpu::tune();
    let mut acc = 0.0_f32;
    for block_idx in 0..blocks_per_row {
        let w_ptr = row.as_ptr().add(block_idx * BLOCK_Q4_K_SIZE);
        if tune.pf_bytes != 0 {
            let ahead = w_ptr.wrapping_add(tune.pf_bytes).cast::<i8>();
            crate::q4k_avx2::prefetch3(ahead, tune.pf_nta);
        }
        let b = decode_q4_block_512(w_ptr);
        let q8_block = &q8k[block_idx * BLOCK_Q8_K_BYTES..][..BLOCK_Q8_K_BYTES];
        let (d_q8, q8v, bs) = load_q8_block_512(q8_block);
        acc += row_dot_decoded_512(&b, d_q8, &q8v, &bs);
    }
    acc
}

/// Dot 4 consecutive rows (spaced `row_bytes`) against one Q8_K vector.
///
/// # Safety
/// As [`q4k_q8k_row_dot_avx512`]; `rows_base` must point at 4 valid rows.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn q4k_q8k_row_dot_x4_avx512(
    rows_base: *const u8,
    row_bytes: usize,
    blocks_per_row: usize,
    q8k: &[u8],
    out: &mut [f32; 4],
) {
    let tune = crate::cpu::tune();
    let mut acc = [0.0_f32; 4];
    for block_idx in 0..blocks_per_row {
        let q8_block = &q8k[block_idx * BLOCK_Q8_K_BYTES..][..BLOCK_Q8_K_BYTES];
        let (d_q8, q8v, bs) = load_q8_block_512(q8_block);
        for (r, acc_r) in acc.iter_mut().enumerate() {
            let w_block = rows_base.add(r * row_bytes + block_idx * BLOCK_Q4_K_SIZE);
            prefetch_row_stream_512(w_block, row_bytes, blocks_per_row, r, 4, tune);
            let b = decode_q4_block_512(w_block);
            *acc_r += row_dot_decoded_512(&b, d_q8, &q8v, &bs);
        }
    }
    *out = acc;
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn prefetch_row_stream_512(
    w_block: *const u8,
    row_bytes: usize,
    blocks_per_row: usize,
    r: usize,
    rows_in_tile: usize,
    tune: crate::cpu::OxkTune,
) {
    if tune.pf_bytes == 0 {
        return;
    }
    let ahead = w_block.wrapping_add(tune.pf_bytes).cast::<i8>();
    crate::q4k_avx2::prefetch3(ahead, tune.pf_nta);
    if blocks_per_row <= 16 {
        let next_tile = w_block.add(rows_in_tile * row_bytes);
        let next = next_tile.wrapping_add(tune.pf_bytes).cast::<i8>();
        _mm_prefetch::<{ _MM_HINT_T1 }>(next);
        _mm_prefetch::<{ _MM_HINT_T1 }>(next.wrapping_add(64));
        _mm_prefetch::<{ _MM_HINT_T1 }>(next.wrapping_add(128));
    } else {
        let far = w_block.wrapping_add(16 * BLOCK_Q4_K_SIZE).cast::<i8>();
        _mm_prefetch::<{ _MM_HINT_T1 }>(far);
        _mm_prefetch::<{ _MM_HINT_T1 }>(far.wrapping_add(64));
        _mm_prefetch::<{ _MM_HINT_T1 }>(far.wrapping_add(128));
    }
    let _ = r;
}

// ---------------------------------------------------------------------------
// AVX-512 VNNI
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Q4BlockVnni512 {
    d_w: f32,
    dmin_w: f32,
    q4_512: [__m512i; 4],
    scale_v: [__m512i; 4],
    mins: [i32; 8],
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn decode_q4_block_vnni512(w_ptr: *const u8) -> Q4BlockVnni512 {
    let mask = _mm256_set1_epi8(0x0f);
    let d_w = f16_le_to_f32([*w_ptr, *w_ptr.add(1)]);
    let dmin_w = f16_le_to_f32([*w_ptr.add(2), *w_ptr.add(3)]);
    let scales = std::slice::from_raw_parts(w_ptr.add(4), 12);
    let qs = w_ptr.add(16);

    let mut q4_512 = [_mm512_setzero_si512(); 4];
    let mut scale_v = [_mm512_setzero_si512(); 4];
    let mut mins = [0_i32; 8];

    for gp in 0..4 {
        let g1 = gp * 2;
        let g2 = g1 + 1;
        let (s1, ms1) = get_scale_min_k4(g1, scales);
        let (s2, ms2) = get_scale_min_k4(g2, scales);
        mins[g1] = ms1 as i32;
        mins[g2] = ms2 as i32;

        let packed = _mm256_loadu_si256(qs.add(gp * 32) as *const __m256i);
        let q4_low = _mm256_and_si256(packed, mask);
        let q4_high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
        q4_512[gp] = _mm512_inserti64x4(_mm512_castsi256_si512(q4_low), q4_high, 1);

        let s_low = _mm256_set1_epi32(s1 as i32);
        let s_high = _mm256_set1_epi32(s2 as i32);
        scale_v[gp] = _mm512_inserti64x4(_mm512_castsi256_si512(s_low), s_high, 1);
    }

    Q4BlockVnni512 {
        d_w,
        dmin_w,
        q4_512,
        scale_v,
        mins,
    }
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn row_dot_decoded_vnni512(
    b: &Q4BlockVnni512,
    d_q8: f32,
    q8v: &[__m512i; 4],
    bs: &[i32; 8],
) -> f32 {
    let mut vec_pos = _mm512_setzero_si512();
    let mut min_acc: i32 = 0;
    for (gp, q8v_gp) in q8v.iter().enumerate() {
        let g1 = gp * 2;
        let g2 = g1 + 1;
        let prod = _mm512_dpbusd_epi32(_mm512_setzero_si512(), b.q4_512[gp], *q8v_gp);
        let scaled = _mm512_mullo_epi32(prod, b.scale_v[gp]);
        vec_pos = _mm512_add_epi32(vec_pos, scaled);
        min_acc += b.mins[g1] * bs[g1];
        min_acc += b.mins[g2] * bs[g2];
    }
    let pos_acc = _mm512_reduce_add_epi32(vec_pos);
    b.d_w * d_q8 * pos_acc as f32 - b.dmin_w * d_q8 * min_acc as f32
}

/// Single-row Q4_K × Q8_K dot using AVX-512 VNNI.
///
/// # Safety
/// Caller must verify AVX-512F+BW+VNNI support.
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub unsafe fn q4k_q8k_row_dot_avx512vnni(row: &[u8], blocks_per_row: usize, q8k: &[u8]) -> f32 {
    let tune = crate::cpu::tune();
    let mut acc = 0.0_f32;
    for block_idx in 0..blocks_per_row {
        let w_ptr = row.as_ptr().add(block_idx * BLOCK_Q4_K_SIZE);
        if tune.pf_bytes != 0 {
            let ahead = w_ptr.wrapping_add(tune.pf_bytes).cast::<i8>();
            crate::q4k_avx2::prefetch3(ahead, tune.pf_nta);
        }
        let b = decode_q4_block_vnni512(w_ptr);
        let q8_block = &q8k[block_idx * BLOCK_Q8_K_BYTES..][..BLOCK_Q8_K_BYTES];
        let (d_q8, q8v, bs) = load_q8_block_512(q8_block);
        acc += row_dot_decoded_vnni512(&b, d_q8, &q8v, &bs);
    }
    acc
}

/// Dot 4 consecutive rows using AVX-512 VNNI.
///
/// # Safety
/// As [`q4k_q8k_row_dot_avx512vnni`].
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub unsafe fn q4k_q8k_row_dot_x4_avx512vnni(
    rows_base: *const u8,
    row_bytes: usize,
    blocks_per_row: usize,
    q8k: &[u8],
    out: &mut [f32; 4],
) {
    let tune = crate::cpu::tune();
    let mut acc = [0.0_f32; 4];
    for block_idx in 0..blocks_per_row {
        let q8_block = &q8k[block_idx * BLOCK_Q8_K_BYTES..][..BLOCK_Q8_K_BYTES];
        let (d_q8, q8v, bs) = load_q8_block_512(q8_block);
        for (r, acc_r) in acc.iter_mut().enumerate() {
            let w_block = rows_base.add(r * row_bytes + block_idx * BLOCK_Q4_K_SIZE);
            prefetch_row_stream_512(w_block, row_bytes, blocks_per_row, r, 4, tune);
            let b = decode_q4_block_vnni512(w_block);
            *acc_r += row_dot_decoded_vnni512(&b, d_q8, &q8v, &bs);
        }
    }
    *out = acc;
}

// ---------------------------------------------------------------------------
// AVX-VNNI (256-bit)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Q4BlockVnni256 {
    d_w: f32,
    dmin_w: f32,
    q4_lo: [__m256i; 4],
    q4_hi: [__m256i; 4],
    scale_v: [__m256i; 8],
    mins: [i32; 8],
}

#[inline]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn decode_q4_block_vnni256(w_ptr: *const u8) -> Q4BlockVnni256 {
    let mask = _mm256_set1_epi8(0x0f);
    let d_w = f16_le_to_f32([*w_ptr, *w_ptr.add(1)]);
    let dmin_w = f16_le_to_f32([*w_ptr.add(2), *w_ptr.add(3)]);
    let scales = std::slice::from_raw_parts(w_ptr.add(4), 12);
    let qs = w_ptr.add(16);

    let mut q4_lo = [_mm256_setzero_si256(); 4];
    let mut q4_hi = [_mm256_setzero_si256(); 4];
    let mut scale_v = [_mm256_setzero_si256(); 8];
    let mut mins = [0_i32; 8];

    for gp in 0..4 {
        let g1 = gp * 2;
        let g2 = g1 + 1;
        let (s1, ms1) = get_scale_min_k4(g1, scales);
        let (s2, ms2) = get_scale_min_k4(g2, scales);
        mins[g1] = ms1 as i32;
        mins[g2] = ms2 as i32;
        scale_v[g1] = _mm256_set1_epi32(s1 as i32);
        scale_v[g2] = _mm256_set1_epi32(s2 as i32);

        let packed = _mm256_loadu_si256(qs.add(gp * 32) as *const __m256i);
        q4_lo[gp] = _mm256_and_si256(packed, mask);
        q4_hi[gp] = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
    }

    Q4BlockVnni256 {
        d_w,
        dmin_w,
        q4_lo,
        q4_hi,
        scale_v,
        mins,
    }
}

#[inline]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn row_dot_decoded_vnni256(
    b: &Q4BlockVnni256,
    d_q8: f32,
    q8v: &[__m256i; 8],
    bs: &[i32; 8],
) -> f32 {
    let mut vec_pos = _mm256_setzero_si256();
    let mut min_acc: i32 = 0;
    for g in 0..8 {
        let plane = if g & 1 == 0 {
            b.q4_lo[g >> 1]
        } else {
            b.q4_hi[g >> 1]
        };
        let prod = _mm256_dpbusd_epi32(_mm256_setzero_si256(), plane, q8v[g]);
        let scaled = _mm256_mullo_epi32(prod, b.scale_v[g]);
        vec_pos = _mm256_add_epi32(vec_pos, scaled);
        min_acc += b.mins[g] * bs[g];
    }
    let pos_acc = crate::q4k_avx2::hsum_i32(vec_pos);
    b.d_w * d_q8 * pos_acc as f32 - b.dmin_w * d_q8 * min_acc as f32
}

/// Single-row Q4_K × Q8_K dot using AVX-VNNI (256-bit).
///
/// # Safety
/// Caller must verify AVX2+AVX-VNNI support.
#[target_feature(enable = "avx2,avxvnni")]
pub unsafe fn q4k_q8k_row_dot_avxvnni(row: &[u8], blocks_per_row: usize, q8k: &[u8]) -> f32 {
    let tune = crate::cpu::tune();
    let mut acc = 0.0_f32;
    for block_idx in 0..blocks_per_row {
        let w_ptr = row.as_ptr().add(block_idx * BLOCK_Q4_K_SIZE);
        if tune.pf_bytes != 0 {
            let ahead = w_ptr.wrapping_add(tune.pf_bytes).cast::<i8>();
            crate::q4k_avx2::prefetch3(ahead, tune.pf_nta);
        }
        let b = decode_q4_block_vnni256(w_ptr);
        let (d_q8, q8v, bs) =
            crate::q4k_avx2::load_q8_block(&q8k[block_idx * BLOCK_Q8_K_BYTES..][..BLOCK_Q8_K_BYTES]);
        acc += row_dot_decoded_vnni256(&b, d_q8, &q8v, &bs);
    }
    acc
}

/// Dot 4 consecutive rows using AVX-VNNI.
///
/// # Safety
/// As [`q4k_q8k_row_dot_avxvnni`].
#[target_feature(enable = "avx2,avxvnni")]
pub unsafe fn q4k_q8k_row_dot_x4_avxvnni(
    rows_base: *const u8,
    row_bytes: usize,
    blocks_per_row: usize,
    q8k: &[u8],
    out: &mut [f32; 4],
) {
    let tune = crate::cpu::tune();
    let mut acc = [0.0_f32; 4];
    for block_idx in 0..blocks_per_row {
        let (d_q8, q8v, bs) =
            crate::q4k_avx2::load_q8_block(&q8k[block_idx * BLOCK_Q8_K_BYTES..][..BLOCK_Q8_K_BYTES]);
        for (r, acc_r) in acc.iter_mut().enumerate() {
            let w_block = rows_base.add(r * row_bytes + block_idx * BLOCK_Q4_K_SIZE);
            crate::q4k_avx2::prefetch_row_stream(w_block, row_bytes, blocks_per_row, r, 4, tune);
            let b = decode_q4_block_vnni256(w_block);
            *acc_r += row_dot_decoded_vnni256(&b, d_q8, &q8v, &bs);
        }
    }
    *out = acc;
}
