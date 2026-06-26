use super::*;

/// Quantize `vector` (length `n_blocks * 256`) into `n_blocks` Q8_K blocks.
pub(crate) fn quantize_vector_q8_k_into(vector: &[f32], n_blocks: usize, out: &mut [u8]) {
    debug_assert_eq!(vector.len(), n_blocks * QK_K);
    debug_assert_eq!(out.len(), n_blocks * BLOCK_Q8_K_BYTES);
    for (b, block_in) in vector.chunks_exact(QK_K).enumerate().take(n_blocks) {
        let block_out = &mut out[b * BLOCK_Q8_K_BYTES..(b + 1) * BLOCK_Q8_K_BYTES];
        quantize_block_q8_k_scalar(block_in, block_out);
    }
}

pub(super) fn quantize_block_q8_k_scalar(block_in: &[f32], block_out: &mut [u8]) {
    debug_assert_eq!(block_in.len(), QK_K);
    debug_assert_eq!(block_out.len(), BLOCK_Q8_K_BYTES);
    let mut amax = 0.0_f32;
    let mut max = 0.0_f32;
    for &v in block_in {
        let av = v.abs();
        if av > amax {
            amax = av;
            max = v;
        }
    }
    if amax == 0.0 {
        // d = 0, all qs = 0, bsums = 0.
        block_out[..4].copy_from_slice(&0.0_f32.to_le_bytes());
        for byte in &mut block_out[4..] {
            *byte = 0;
        }
        return;
    }
    // iscale = -128 / max (sign-preserving to keep symmetry with [-128, 127])
    let iscale = -128.0_f32 / max;
    let d = 1.0_f32 / iscale;
    block_out[..4].copy_from_slice(&d.to_le_bytes());
    let qs_off = 4;
    for (i, &v) in block_in.iter().enumerate() {
        let scaled = iscale * v;
        let q = scaled.round() as i32;
        let q = q.clamp(-128, 127) as i8;
        block_out[qs_off + i] = q as u8;
    }
    // bsums: 16 int16 sums, one per 16-element group.
    let bsums_off = qs_off + QK_K;
    for g in 0..(QK_K / 16) {
        let mut sum: i32 = 0;
        for i in 0..16 {
            sum += (block_out[qs_off + g * 16 + i] as i8) as i32;
        }
        let sum16 = sum.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        let lo = (sum16 as u16) as u8;
        let hi = ((sum16 as u16) >> 8) as u8;
        block_out[bsums_off + g * 2] = lo;
        block_out[bsums_off + g * 2 + 1] = hi;
    }
}

/// Per-row Q4_K × Q8_K dot product using AVX2 integer multiply-adds.
/// Returns the f32 dot product for one output row across all blocks in the row.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(super) unsafe fn q4_k_q8_k_row_dot_avx2(row: &[u8], blocks_per_row: usize, q8k: &[u8]) -> f32 {
    let mask = _mm256_set1_epi8(0x0f);
    let mut acc = 0.0_f32;
    for block_idx in 0..blocks_per_row {
        let w_off = block_idx * BLOCK_Q4_K_SIZE;
        let q8_off = block_idx * BLOCK_Q8_K_BYTES;
        let block = &row[w_off..w_off + BLOCK_Q4_K_SIZE];
        let q8_block = &q8k[q8_off..q8_off + BLOCK_Q8_K_BYTES];

        let ahead = block
            .as_ptr()
            .wrapping_add(4 * BLOCK_Q4_K_SIZE)
            .cast::<i8>();
        _mm_prefetch::<{ _MM_HINT_T0 }>(ahead);
        _mm_prefetch::<{ _MM_HINT_T0 }>(ahead.wrapping_add(64));
        _mm_prefetch::<{ _MM_HINT_T0 }>(ahead.wrapping_add(128));

        let d_w = f16_le_to_f32([block[0], block[1]]);
        let dmin_w = f16_le_to_f32([block[2], block[3]]);
        let d_q8 = f32::from_le_bytes([q8_block[0], q8_block[1], q8_block[2], q8_block[3]]);
        let scales = &block[4..16];
        let qs = block[16..].as_ptr();
        let q8 = q8_block[4..].as_ptr();
        let bsums = &q8_block[4 + QK_K..];

        // Single vector accumulator across all 8 sub-groups → 1 hsum per block.
        let mut vec_pos = _mm256_setzero_si256();
        let mut min_acc: i32 = 0;
        for gp in 0..4 {
            let g1 = gp * 2;
            let g2 = g1 + 1;
            let (s1, ms1) = get_scale_min_k4(g1, scales);
            let (s2, ms2) = get_scale_min_k4(g2, scales);
            let packed = unsafe { _mm256_loadu_si256(qs.add(gp * 32) as *const __m256i) };
            let q4_low = _mm256_and_si256(packed, mask);
            let q4_high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
            let q8_low = unsafe { _mm256_loadu_si256(q8.add(g1 * 32) as *const __m256i) };
            let q8_high = unsafe { _mm256_loadu_si256(q8.add(g2 * 32) as *const __m256i) };
            let p16_low = _mm256_maddubs_epi16(q4_low, q8_low);
            let p16_high = _mm256_maddubs_epi16(q4_high, q8_high);
            // madd(p16, set1_epi16(s)) == s * (p0 + p1) per i32 lane — identical
            // to madd(p16, ones) * s, but avoids the slow mullo_epi32 (10c lat).
            // No overflow: |p16| <= 2*15*127 = 3810, s <= 63 -> 240_030 << i32::MAX.
            let p32_low = _mm256_madd_epi16(p16_low, _mm256_set1_epi16(s1 as i16));
            let p32_high = _mm256_madd_epi16(p16_high, _mm256_set1_epi16(s2 as i16));
            vec_pos = _mm256_add_epi32(vec_pos, _mm256_add_epi32(p32_low, p32_high));

            let bs1 =
                read_q8_k_bsum(bsums, g1 * 2) as i32 + read_q8_k_bsum(bsums, g1 * 2 + 1) as i32;
            let bs2 =
                read_q8_k_bsum(bsums, g2 * 2) as i32 + read_q8_k_bsum(bsums, g2 * 2 + 1) as i32;
            min_acc += ms1 as i32 * bs1;
            min_acc += ms2 as i32 * bs2;
        }
        let pos_acc = unsafe { hsum_i32_avx2(vec_pos) };
        acc += d_w * d_q8 * pos_acc as f32 - dmin_w * d_q8 * min_acc as f32;
    }
    acc
}

/// Dot 4 consecutive weight rows against one shared Q8_K vector.
///
/// Per-row math is bit-identical to [`q4_k_q8_k_row_dot_avx2`] (same op
/// sequence and accumulation order); the win is structural: the Q8_K input
/// vectors and bsum pair-sums are loaded/computed once per block and reused by
/// all 4 rows, and the 4 scalar accumulators form independent dependency
/// chains so the out-of-order core can overlap DRAM (often remote-NUMA)
/// latency across rows instead of stalling on one row's stream.
///
/// # Safety
/// `rows_base` must point to 4 rows of `blocks_per_row` Q4_K blocks spaced
/// `row_bytes` apart; `q8k` must hold `blocks_per_row` Q8_K blocks. Caller
/// must have verified AVX2+FMA support.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(super) unsafe fn q4_k_q8_k_row_dot_x4_avx2(
    rows_base: *const u8,
    row_bytes: usize,
    blocks_per_row: usize,
    q8k: &[u8],
    out: &mut [f32; 4],
) {
    let mask = _mm256_set1_epi8(0x0f);
    let mut acc = [0.0_f32; 4];
    for block_idx in 0..blocks_per_row {
        let q8_block = &q8k[block_idx * BLOCK_Q8_K_BYTES..][..BLOCK_Q8_K_BYTES];
        let d_q8 = f32::from_le_bytes([q8_block[0], q8_block[1], q8_block[2], q8_block[3]]);
        let q8 = q8_block[4..].as_ptr();
        let bsums = &q8_block[4 + QK_K..];

        // Shared across all 4 rows: the 8 q8 sub-group vectors and the
        // per-group-pair bsum sums (these depend only on the input vector).
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

        for (r, acc_r) in acc.iter_mut().enumerate() {
            let w_ptr = rows_base.add(r * row_bytes + block_idx * BLOCK_Q4_K_SIZE);
            // Same prefetch policy as the single-row kernel, per stream.
            let ahead = w_ptr.add(4 * BLOCK_Q4_K_SIZE).cast::<i8>();
            _mm_prefetch::<{ _MM_HINT_T0 }>(ahead);
            _mm_prefetch::<{ _MM_HINT_T0 }>(ahead.add(64));
            _mm_prefetch::<{ _MM_HINT_T0 }>(ahead.add(128));
            // For SHORT rows also sweep the NEXT quad's row r into L2, one
            // quad-time ahead: 10-block rows (1.4KB) restart the hardware
            // prefetcher every 22 cache lines, costing ~10% of DRAM bandwidth
            // on 2560-column matrices. Advancing one block per iteration, the
            // pointer covers the whole next row by quad end. Long rows keep
            // the prefetcher locked on their own — the extra reach only
            // pollutes L2 there.
            if blocks_per_row <= 16 {
                let next_quad = w_ptr.add(4 * row_bytes).cast::<i8>();
                _mm_prefetch::<{ _MM_HINT_T1 }>(next_quad);
                _mm_prefetch::<{ _MM_HINT_T1 }>(next_quad.add(64));
                _mm_prefetch::<{ _MM_HINT_T1 }>(next_quad.add(128));
            } else {
                // Long rows: a second, deeper in-row sweep (T1, 16 blocks =
                // 2.3KB ahead) — the 576B T0 distance alone leaves the stream
                // ~8% under the short-row shapes once those got their sweep.
                let far = w_ptr.add(16 * BLOCK_Q4_K_SIZE).cast::<i8>();
                _mm_prefetch::<{ _MM_HINT_T1 }>(far);
                _mm_prefetch::<{ _MM_HINT_T1 }>(far.add(64));
                _mm_prefetch::<{ _MM_HINT_T1 }>(far.add(128));
            }

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
                let p32_low = _mm256_madd_epi16(p16_low, _mm256_set1_epi16(s1 as i16));
                let p32_high = _mm256_madd_epi16(p16_high, _mm256_set1_epi16(s2 as i16));
                vec_pos = _mm256_add_epi32(vec_pos, _mm256_add_epi32(p32_low, p32_high));
                min_acc += ms1 as i32 * bs[g1];
                min_acc += ms2 as i32 * bs[g2];
            }
            let pos_acc = hsum_i32_avx2(vec_pos);
            *acc_r += d_w * d_q8 * pos_acc as f32 - dmin_w * d_q8 * min_acc as f32;
        }
    }
    *out = acc;
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub(super) unsafe fn q4_k_q8_k_row_dot_x4_avx2(
    _rows_base: *const u8,
    _row_bytes: usize,
    _blocks_per_row: usize,
    _q8k: &[u8],
    _out: &mut [f32; 4],
) {
    unreachable!("x4 kernel is gated on x86 availability at call sites")
}

/// Integer Q6_K x Q8_K row dot (llama.cpp-style). Decodes 6-bit weights to
/// unsigned 0..63, runs `maddubs`/`madd` integer dot products against the
/// pre-quantized Q8_K input, and removes the implicit -32 offset analytically
/// via the Q8_K per-16 bsums: sum((q6u-32)*q8) = maddubs-sum - 32*bsum. This
/// replaces the f32 decode+FMA Q6_K kernel on the GEMV hot paths (~5x fewer
/// ops per byte). No overflow: maddubs pair <= 2*63*127 = 16_002 (i16),
/// madd with |scale| <= 127 -> ~4.1M per lane-pair (i32).
///
/// # Safety
/// Caller must verify AVX2; `row` holds `blocks_per_row` Q6_K blocks and
/// `q8k` the matching Q8_K blocks.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(super) unsafe fn q6_k_q8_k_row_dot_avx2(row: &[u8], blocks_per_row: usize, q8k: &[u8]) -> f32 {
    let mask_low = _mm256_set1_epi8(0x0f);
    let mask_high = _mm256_set1_epi8(0x03);
    let mut acc = 0.0_f32;
    for block_idx in 0..blocks_per_row {
        let w_off = block_idx * BLOCK_Q6_K_SIZE;
        let block = &row[w_off..w_off + BLOCK_Q6_K_SIZE];
        let q8_block = &q8k[block_idx * BLOCK_Q8_K_BYTES..][..BLOCK_Q8_K_BYTES];
        let d_q8 = f32::from_le_bytes([q8_block[0], q8_block[1], q8_block[2], q8_block[3]]);
        let q8 = q8_block[4..].as_ptr();
        let bsums = &q8_block[4 + QK_K..];
        let d = f16_le_to_f32([block[208], block[209]]);
        let ql = block[0..128].as_ptr();
        let qh = block[128..192].as_ptr();
        let sc = &block[192..208];

        let mut vec_pos = _mm256_setzero_si256();
        let mut min_acc: i32 = 0;
        for half in 0..2 {
            let s_base = half * 8;
            let v_base = half * 128;
            let ql_lo = _mm256_loadu_si256(ql.add(half * 64) as *const __m256i);
            let ql_hi = _mm256_loadu_si256(ql.add(half * 64 + 32) as *const __m256i);
            let qh_v = _mm256_loadu_si256(qh.add(half * 32) as *const __m256i);

            // Four 32-value groups per half; mapping mirrors q6_k_row_dot_avx2.
            let q1 = _mm256_or_si256(
                _mm256_and_si256(ql_lo, mask_low),
                _mm256_slli_epi16(_mm256_and_si256(qh_v, mask_high), 4),
            );
            let q2 = _mm256_or_si256(
                _mm256_and_si256(ql_hi, mask_low),
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_v, 2), mask_high), 4),
            );
            let q3 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(ql_lo, 4), mask_low),
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_v, 4), mask_high), 4),
            );
            let q4 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(ql_hi, 4), mask_low),
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_v, 6), mask_high), 4),
            );

            for (g, qv) in [q1, q2, q3, q4].into_iter().enumerate() {
                let sa = sc[s_base + g * 2] as i8 as i16;
                let sb = sc[s_base + g * 2 + 1] as i8 as i16;
                let q8v = _mm256_loadu_si256(q8.add(v_base + g * 32) as *const __m256i);
                let p16 = _mm256_maddubs_epi16(qv, q8v);
                let scale_pair = _mm256_set_m128i(_mm_set1_epi16(sb), _mm_set1_epi16(sa));
                vec_pos = _mm256_add_epi32(vec_pos, _mm256_madd_epi16(p16, scale_pair));
                let g0 = half * 8 + g * 2;
                min_acc += sa as i32 * read_q8_k_bsum(bsums, g0) as i32;
                min_acc += sb as i32 * read_q8_k_bsum(bsums, g0 + 1) as i32;
            }
        }
        let pos = hsum_i32_avx2(vec_pos);
        acc += d * d_q8 * (pos - 32 * min_acc) as f32;
    }
    acc
}

/// 4-row variant of [`q6_k_q8_k_row_dot_avx2`]: shares the Q8_K loads and
/// keeps 4 independent accumulator chains in flight (same structure as
/// [`q4_k_q8_k_row_dot_x4_avx2`]).
///
/// # Safety
/// Same as the single-row kernel; `rows_base` must point at 4 rows spaced
/// `row_bytes` apart.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(super) unsafe fn q6_k_q8_k_row_dot_x4_avx2(
    rows_base: *const u8,
    row_bytes: usize,
    blocks_per_row: usize,
    q8k: &[u8],
    out: &mut [f32; 4],
) {
    let mask_low = _mm256_set1_epi8(0x0f);
    let mask_high = _mm256_set1_epi8(0x03);
    let mut acc = [0.0_f32; 4];
    for block_idx in 0..blocks_per_row {
        let q8_block = &q8k[block_idx * BLOCK_Q8_K_BYTES..][..BLOCK_Q8_K_BYTES];
        let d_q8 = f32::from_le_bytes([q8_block[0], q8_block[1], q8_block[2], q8_block[3]]);
        let q8 = q8_block[4..].as_ptr();
        let bsums = &q8_block[4 + QK_K..];
        let mut bs = [0_i32; 16];
        for (g, b) in bs.iter_mut().enumerate() {
            *b = read_q8_k_bsum(bsums, g) as i32;
        }
        let q8v: [__m256i; 8] = [
            _mm256_loadu_si256(q8 as *const __m256i),
            _mm256_loadu_si256(q8.add(32) as *const __m256i),
            _mm256_loadu_si256(q8.add(64) as *const __m256i),
            _mm256_loadu_si256(q8.add(96) as *const __m256i),
            _mm256_loadu_si256(q8.add(128) as *const __m256i),
            _mm256_loadu_si256(q8.add(160) as *const __m256i),
            _mm256_loadu_si256(q8.add(192) as *const __m256i),
            _mm256_loadu_si256(q8.add(224) as *const __m256i),
        ];

        for (r, acc_r) in acc.iter_mut().enumerate() {
            let w_ptr = rows_base.add(r * row_bytes + block_idx * BLOCK_Q6_K_SIZE);
            let ahead = w_ptr.add(3 * BLOCK_Q6_K_SIZE).cast::<i8>();
            _mm_prefetch::<{ _MM_HINT_T0 }>(ahead);
            _mm_prefetch::<{ _MM_HINT_T0 }>(ahead.add(64));
            _mm_prefetch::<{ _MM_HINT_T0 }>(ahead.add(128));
            // Next-quad sweep for short rows, deeper in-row sweep for long
            // rows; see the Q4_K x4 kernel.
            if blocks_per_row <= 16 {
                let next_quad = w_ptr.add(4 * row_bytes).cast::<i8>();
                _mm_prefetch::<{ _MM_HINT_T1 }>(next_quad);
                _mm_prefetch::<{ _MM_HINT_T1 }>(next_quad.add(64));
                _mm_prefetch::<{ _MM_HINT_T1 }>(next_quad.add(128));
            } else {
                let far = w_ptr.add(16 * BLOCK_Q6_K_SIZE).cast::<i8>();
                _mm_prefetch::<{ _MM_HINT_T1 }>(far);
                _mm_prefetch::<{ _MM_HINT_T1 }>(far.add(64));
                _mm_prefetch::<{ _MM_HINT_T1 }>(far.add(128));
            }

            let d = f16_le_to_f32([*w_ptr.add(208), *w_ptr.add(209)]);
            let ql = w_ptr;
            let qh = w_ptr.add(128);
            let sc = std::slice::from_raw_parts(w_ptr.add(192) as *const i8, 16);

            let mut vec_pos = _mm256_setzero_si256();
            let mut min_acc: i32 = 0;
            for half in 0..2 {
                let s_base = half * 8;
                let _v_base = half * 128;
                let ql_lo = _mm256_loadu_si256(ql.add(half * 64) as *const __m256i);
                let ql_hi = _mm256_loadu_si256(ql.add(half * 64 + 32) as *const __m256i);
                let qh_v = _mm256_loadu_si256(qh.add(half * 32) as *const __m256i);
                let q1 = _mm256_or_si256(
                    _mm256_and_si256(ql_lo, mask_low),
                    _mm256_slli_epi16(_mm256_and_si256(qh_v, mask_high), 4),
                );
                let q2 = _mm256_or_si256(
                    _mm256_and_si256(ql_hi, mask_low),
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_v, 2), mask_high), 4),
                );
                let q3 = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16(ql_lo, 4), mask_low),
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_v, 4), mask_high), 4),
                );
                let q4 = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16(ql_hi, 4), mask_low),
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh_v, 6), mask_high), 4),
                );
                for (g, qv) in [q1, q2, q3, q4].into_iter().enumerate() {
                    let sa = sc[s_base + g * 2] as i16;
                    let sb = sc[s_base + g * 2 + 1] as i16;
                    let p16 = _mm256_maddubs_epi16(qv, q8v[half * 4 + g]);
                    let scale_pair = _mm256_set_m128i(_mm_set1_epi16(sb), _mm_set1_epi16(sa));
                    vec_pos = _mm256_add_epi32(vec_pos, _mm256_madd_epi16(p16, scale_pair));
                    let g0 = half * 8 + g * 2;
                    min_acc += sa as i32 * bs[g0];
                    min_acc += sb as i32 * bs[g0 + 1];
                }
            }
            let pos = hsum_i32_avx2(vec_pos);
            *acc_r += d * d_q8 * (pos - 32 * min_acc) as f32;
        }
    }
    *out = acc;
}

/// AVX-512 VNNI variant of [`q4_k_q8_k_row_dot_avx2`]. Uses `_mm512_dpbusd_epi32`
/// to fuse the `maddubs` + `madd(ones)` int8→int32 reduction into a single
/// instruction, and processes the two scale sub-groups (g1, g2) of each `gp`
/// iteration together in one 512-bit lane group. Integer math is identical to
/// the AVX2 kernel; only the instruction sequence differs, so results match
/// bit-for-bit in the integer domain.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(super) unsafe fn q4_k_q8_k_row_dot_vnni(row: &[u8], blocks_per_row: usize, q8k: &[u8]) -> f32 {
    let mask = _mm256_set1_epi8(0x0f);
    let mut acc = 0.0_f32;
    for block_idx in 0..blocks_per_row {
        let w_off = block_idx * BLOCK_Q4_K_SIZE;
        let block = &row[w_off..w_off + BLOCK_Q4_K_SIZE];
        let q8_block = &q8k[block_idx * BLOCK_Q8_K_BYTES..][..BLOCK_Q8_K_BYTES];

        let d_w = f16_le_to_f32([block[0], block[1]]);
        let dmin_w = f16_le_to_f32([block[2], block[3]]);
        let d_q8 = f32::from_le_bytes([q8_block[0], q8_block[1], q8_block[2], q8_block[3]]);
        let scales = &block[4..16];
        let qs = block[16..].as_ptr();
        let q8 = q8_block[4..].as_ptr();
        let bsums = &q8_block[4 + QK_K..];

        let mut vec_pos = _mm512_setzero_si512();
        let mut min_acc: i32 = 0;
        for gp in 0..4 {
            let g1 = gp * 2;
            let g2 = g1 + 1;
            let (s1, ms1) = get_scale_min_k4(g1, scales);
            let (s2, ms2) = get_scale_min_k4(g2, scales);
            let packed = _mm256_loadu_si256(qs.add(gp * 32) as *const __m256i);
            // q4_low pairs with group g1's q8 (low half), q4_high with g2 (high half).
            let q4_low = _mm256_and_si256(packed, mask);
            let q4_high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
            // g1 and g2 q8 quants are contiguous (g2 = g1 + 1) → one 512-bit load.
            let q8_512 = _mm512_loadu_si512(q8.add(g1 * 32) as *const __m512i);
            let q4_512 = _mm512_inserti64x4(_mm512_castsi256_si512(q4_low), q4_high, 1);
            // dpbusd: unsigned q4 (0..15) × signed q8 → int32, 16 lanes.
            // Lanes 0..8 = group g1 (scale s1), lanes 8..16 = group g2 (scale s2).
            let prod = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q4_512, q8_512);
            let scale_v = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_set1_epi32(s1 as i32)),
                _mm256_set1_epi32(s2 as i32),
                1,
            );
            vec_pos = _mm512_add_epi32(vec_pos, _mm512_mullo_epi32(prod, scale_v));

            let bs1 =
                read_q8_k_bsum(bsums, g1 * 2) as i32 + read_q8_k_bsum(bsums, g1 * 2 + 1) as i32;
            let bs2 =
                read_q8_k_bsum(bsums, g2 * 2) as i32 + read_q8_k_bsum(bsums, g2 * 2 + 1) as i32;
            min_acc += ms1 as i32 * bs1;
            min_acc += ms2 as i32 * bs2;
        }
        let pos_acc = _mm512_reduce_add_epi32(vec_pos);
        acc += d_w * d_q8 * pos_acc as f32 - dmin_w * d_q8 * min_acc as f32;
    }
    acc
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(super) unsafe fn q4_k_q8_k_row_dot_chunk_avx2(
    row: &[u8],
    blocks_per_row: usize,
    q8_panel: &[u8],
    q8_stride: usize,
    token_start: usize,
    token_count: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(out.len(), token_count);
    let mask = _mm256_set1_epi8(0x0f);

    for block_idx in 0..blocks_per_row {
        let w_ptr = row.as_ptr().wrapping_add(block_idx * BLOCK_Q4_K_SIZE);
        let d_w = f16_le_to_f32([unsafe { *w_ptr }, unsafe { *w_ptr.add(1) }]);
        let dmin_w = f16_le_to_f32([unsafe { *w_ptr.add(2) }, unsafe { *w_ptr.add(3) }]);
        let scales = unsafe { std::slice::from_raw_parts(w_ptr.add(4), 12) };
        let qs = unsafe { w_ptr.add(16) };

        // Decode all 8 (scale, min_scale) pairs once per block.
        let mut g_scales = [0_i32; 8];
        let mut g_min_scales = [0_i32; 8];
        for g in 0..8 {
            let (s, ms) = get_scale_min_k4(g, scales);
            g_scales[g] = s as i32;
            g_min_scales[g] = ms as i32;
        }

        // Pre-decode q4 nibbles for all 4 group-pairs once (shared across tokens).
        let mut q4_lo = [_mm256_setzero_si256(); 4];
        let mut q4_hi = [_mm256_setzero_si256(); 4];
        for gp in 0..4 {
            let packed = unsafe { _mm256_loadu_si256(qs.add(gp * 32) as *const __m256i) };
            q4_lo[gp] = _mm256_and_si256(packed, mask);
            q4_hi[gp] = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
        }

        // Broadcast scales as i16 for madd_epi16: madd(p16, set1_epi16(s)) ==
        // s * (p0 + p1) per i32 lane — identical to madd(p16, ones) * s but
        // avoids the slow mullo_epi32. No overflow: |p16| <= 3810, s <= 63.
        let s_v = [
            _mm256_set1_epi16(g_scales[0] as i16),
            _mm256_set1_epi16(g_scales[1] as i16),
            _mm256_set1_epi16(g_scales[2] as i16),
            _mm256_set1_epi16(g_scales[3] as i16),
            _mm256_set1_epi16(g_scales[4] as i16),
            _mm256_set1_epi16(g_scales[5] as i16),
            _mm256_set1_epi16(g_scales[6] as i16),
            _mm256_set1_epi16(g_scales[7] as i16),
        ];

        for (token, out_value) in out.iter_mut().enumerate().take(token_count) {
            let q8_off = (token_start + token) * q8_stride + block_idx * BLOCK_Q8_K_BYTES;
            let q8_block = &q8_panel[q8_off..q8_off + BLOCK_Q8_K_BYTES];
            let d_q8 = f32::from_le_bytes([q8_block[0], q8_block[1], q8_block[2], q8_block[3]]);
            let q8 = q8_block[4..].as_ptr();
            let bsums = &q8_block[4 + QK_K..];

            // Single vector accumulator across all 8 groups → 1 hsum per block.
            let mut vec_pos = _mm256_setzero_si256();
            for gp in 0..4 {
                let g1 = gp * 2;
                let g2 = g1 + 1;
                let q8_low = unsafe { _mm256_loadu_si256(q8.add(g1 * 32) as *const __m256i) };
                let q8_high = unsafe { _mm256_loadu_si256(q8.add(g2 * 32) as *const __m256i) };
                let p16_low = _mm256_maddubs_epi16(q4_lo[gp], q8_low);
                let p16_high = _mm256_maddubs_epi16(q4_hi[gp], q8_high);
                let scaled_low = _mm256_madd_epi16(p16_low, s_v[g1]);
                let scaled_high = _mm256_madd_epi16(p16_high, s_v[g2]);
                vec_pos = _mm256_add_epi32(vec_pos, _mm256_add_epi32(scaled_low, scaled_high));
            }
            let pos = unsafe { hsum_i32_avx2(vec_pos) };

            // Min correction: scalar over 8 groups is cheap, but use the
            // precomputed bsums directly as i16.
            let mut min: i32 = 0;
            for (g, min_scale) in g_min_scales.iter().enumerate() {
                let bs =
                    read_q8_k_bsum(bsums, g * 2) as i32 + read_q8_k_bsum(bsums, g * 2 + 1) as i32;
                min += min_scale * bs;
            }

            let d_scale = d_w * d_q8;
            let dmin_scale = dmin_w * d_q8;
            *out_value += d_scale * pos as f32 - dmin_scale * min as f32;
        }
    }
}

#[inline]
pub(super) fn read_q8_k_bsum(bsums: &[u8], index: usize) -> i16 {
    crate::bytes::read_q8_k_bsum(bsums, index)
}

/// Horizontal sum of 8 packed int32 values.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
pub(super) unsafe fn hsum_i32_avx2(v: __m256i) -> i32 {
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    let sum128 = _mm_add_epi32(lo, hi);
    let shuf = _mm_shuffle_epi32(sum128, 0b1110);
    let sum64 = _mm_add_epi32(sum128, shuf);
    let shuf2 = _mm_shuffle_epi32(sum64, 0b01);
    let sum32 = _mm_add_epi32(sum64, shuf2);
    _mm_cvtsi128_si32(sum32)
}

#[inline]
pub(super) fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0x0f) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

#[inline]
pub(super) fn q4_k_value(block: &[u8], idx: usize) -> f32 {
    let d = f16_le_to_f32([block[0], block[1]]);
    let min = f16_le_to_f32([block[2], block[3]]);
    let scales = &block[4..16];
    let qs = &block[16..144];
    let group = idx / 32;
    let pair = group / 2;
    let (scale, min_scale) = get_scale_min_k4(group, scales);
    let q = if group.is_multiple_of(2) {
        qs[pair * 32 + (idx % 32)] & 0x0f
    } else {
        qs[pair * 32 + (idx % 32)] >> 4
    };
    d * scale as f32 * q as f32 - min * min_scale as f32
}

#[inline]
#[allow(dead_code)]
pub(super) fn q4_k_dot(block: &[u8], vector: &[f32]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { q4_k_dot_avx2(block, vector) };
        }
    }
    q4_k_dot_scalar(block, vector)
}

/// AVX2 + FMA dot product between a dequantized Q4_K block (256 weights) and a
/// 256-element vector slice. Dequantizes nibbles on the fly so the matrix is
/// read at 4-bit density — this is the decode hot path.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(dead_code)]
pub(super) unsafe fn q4_k_dot_avx2(block: &[u8], vector: &[f32]) -> f32 {
    let d = f16_le_to_f32([block[0], block[1]]);
    let min = f16_le_to_f32([block[2], block[3]]);
    let scales = &block[4..16];
    let qs = block.as_ptr().wrapping_add(16);
    let mask = _mm_set1_epi8(0x0f);
    let mut acc = _mm256_setzero_ps();
    for group_pair in 0..4 {
        let group1 = group_pair * 2;
        let group2 = group1 + 1;
        let (scale1, min_scale1) = get_scale_min_k4(group1, scales);
        let (scale2, min_scale2) = get_scale_min_k4(group2, scales);
        let d1 = _mm256_set1_ps(d * scale1 as f32);
        let min1 = _mm256_set1_ps(min * min_scale1 as f32);
        let d2 = _mm256_set1_ps(d * scale2 as f32);
        let min2 = _mm256_set1_ps(min * min_scale2 as f32);
        let q_base = group_pair * 32;
        let v_base = group_pair * 64;
        for l in (0..32).step_by(8) {
            let packed = unsafe { _mm_loadl_epi64(qs.add(q_base + l).cast::<__m128i>()) };
            let low_u8 = _mm_and_si128(packed, mask);
            let high_u8 = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
            let low = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(low_u8));
            let high = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(high_u8));
            // term = d * q - min
            let term1 = _mm256_fmsub_ps(low, d1, min1);
            let term2 = _mm256_fmsub_ps(high, d2, min2);
            let v1 = unsafe { _mm256_loadu_ps(vector.as_ptr().add(v_base + l)) };
            let v2 = unsafe { _mm256_loadu_ps(vector.as_ptr().add(v_base + 32 + l)) };
            acc = _mm256_fmadd_ps(term1, v1, acc);
            acc = _mm256_fmadd_ps(term2, v2, acc);
        }
    }
    // Horizontal sum of the 8 lanes.
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let result = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(result)
}

#[inline]
pub(super) fn q4_k_dot_scalar(block: &[u8], vector: &[f32]) -> f32 {
    let d = f16_le_to_f32([block[0], block[1]]);
    let min = f16_le_to_f32([block[2], block[3]]);
    let scales = &block[4..16];
    let qs = &block[16..144];
    let mut sum = 0.0_f32;
    for group_pair in 0..4 {
        let group1 = group_pair * 2;
        let group2 = group1 + 1;
        let (scale1, min_scale1) = get_scale_min_k4(group1, scales);
        let (scale2, min_scale2) = get_scale_min_k4(group2, scales);
        let d1 = d * scale1 as f32;
        let min1 = min * min_scale1 as f32;
        let d2 = d * scale2 as f32;
        let min2 = min * min_scale2 as f32;
        let q_base = group_pair * 32;
        let v_base = group_pair * 64;
        for l in 0..32 {
            let packed = qs[q_base + l];
            sum += (d1 * (packed & 0x0f) as f32 - min1) * vector[v_base + l];
            sum += (d2 * (packed >> 4) as f32 - min2) * vector[v_base + 32 + l];
        }
    }
    sum
}

#[inline]
pub(super) fn accumulate_q4_k_block(block: &[u8], factor: f32, output: &mut [f32]) {
    let d = f16_le_to_f32([block[0], block[1]]);
    let min = f16_le_to_f32([block[2], block[3]]);
    let scales = &block[4..16];
    let qs = &block[16..144];
    for group_pair in 0..4 {
        let group1 = group_pair * 2;
        let group2 = group1 + 1;
        let (scale1, min_scale1) = get_scale_min_k4(group1, scales);
        let (scale2, min_scale2) = get_scale_min_k4(group2, scales);
        let d1 = d * scale1 as f32;
        let min1 = min * min_scale1 as f32;
        let d2 = d * scale2 as f32;
        let min2 = min * min_scale2 as f32;
        let q_base = group_pair * 32;
        let out_base = group_pair * 64;
        for l in 0..32 {
            let packed = qs[q_base + l];
            output[out_base + l] += (d1 * (packed & 0x0f) as f32 - min1) * factor;
            output[out_base + 32 + l] += (d2 * (packed >> 4) as f32 - min2) * factor;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(dead_code)]
pub(super) unsafe fn accumulate_q4_k_block_avx2(block: *const u8, factor: f32, output: *mut f32) {
    let d = f16_le_to_f32(unsafe { [*block, *block.add(1)] });
    let min = f16_le_to_f32(unsafe { [*block.add(2), *block.add(3)] });
    let scales = unsafe { std::slice::from_raw_parts(block.add(4), 12) };
    let qs = unsafe { block.add(16) };
    let mask = _mm_set1_epi8(0x0f);
    let factor_v = _mm256_set1_ps(factor);

    for group_pair in 0..4 {
        let group1 = group_pair * 2;
        let group2 = group1 + 1;
        let (scale1, min_scale1) = get_scale_min_k4(group1, scales);
        let (scale2, min_scale2) = get_scale_min_k4(group2, scales);
        let d1 = _mm256_set1_ps(d * scale1 as f32);
        let min1 = _mm256_set1_ps(min * min_scale1 as f32);
        let d2 = _mm256_set1_ps(d * scale2 as f32);
        let min2 = _mm256_set1_ps(min * min_scale2 as f32);
        let q_base = group_pair * 32;
        let out_base = group_pair * 64;
        for l in (0..32).step_by(8) {
            let packed = unsafe { _mm_loadl_epi64(qs.add(q_base + l).cast::<__m128i>()) };
            let low_u8 = _mm_and_si128(packed, mask);
            let high_u8 = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
            let low = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(low_u8));
            let high = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(high_u8));
            let vals1 = _mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(low, d1), min1), factor_v);
            let vals2 = _mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(high, d2), min2), factor_v);
            let out1 = unsafe { output.add(out_base + l) };
            let out2 = unsafe { output.add(out_base + 32 + l) };
            let cur1 = unsafe { _mm256_loadu_ps(out1) };
            let cur2 = unsafe { _mm256_loadu_ps(out2) };
            unsafe {
                _mm256_storeu_ps(out1, _mm256_add_ps(cur1, vals1));
                _mm256_storeu_ps(out2, _mm256_add_ps(cur2, vals2));
            }
        }
    }
}

pub(super) fn gemv_q4_k_f32_fused(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    let blocks_per_row = cols / QK_K;
    let expected_matrix_len = rows * blocks_per_row * BLOCK_Q4_K_SIZE;
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
        });
    }
    if vector.len() != cols {
        return Err(GemvError::InvalidVectorLength {
            expected: cols,
            actual: vector.len(),
        });
    }
    if output.len() != rows {
        return Err(GemvError::InvalidOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let use_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let use_avx2 = false;

    let row_bytes = blocks_per_row * BLOCK_Q4_K_SIZE;
    let compute_row = |row_idx: usize| -> f32 {
        let row_start = row_idx * row_bytes;
        let row = &quantized_matrix[row_start..row_start + row_bytes];
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if use_avx2 {
                return unsafe { q4_k_row_dot_avx2(row, blocks_per_row, vector) };
            }
        }
        let _ = use_avx2;
        let mut sum = 0.0_f32;
        for (block_idx, block) in row.chunks_exact(BLOCK_Q4_K_SIZE).enumerate() {
            let v_off = block_idx * QK_K;
            sum += q4_k_dot_scalar(block, &vector[v_off..v_off + QK_K]);
        }
        sum
    };

    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        output
            .par_iter_mut()
            .with_min_len(32)
            .enumerate()
            .for_each(|(row_idx, out)| *out = compute_row(row_idx));
    } else {
        for (row_idx, out) in output.iter_mut().enumerate() {
            *out = compute_row(row_idx);
        }
    }
    Ok(())
}

/// Whole-row Q4_K dot product. Accumulates over all `blocks_per_row` blocks
/// into four independent AVX2 registers (broken-up dependency chain so the
/// CPU can keep 4 FMA ops in flight per cycle on Zen 3/4) and does a single
/// horizontal reduce at the end.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(super) unsafe fn q4_k_row_dot_avx2(row: &[u8], blocks_per_row: usize, vector: &[f32]) -> f32 {
    let mask = _mm_set1_epi8(0x0f);
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    for block_idx in 0..blocks_per_row {
        let block_ptr = row.as_ptr().wrapping_add(block_idx * BLOCK_Q4_K_SIZE);
        let v_ptr = vector.as_ptr().wrapping_add(block_idx * QK_K);

        let d = f16_le_to_f32([unsafe { *block_ptr }, unsafe { *block_ptr.add(1) }]);
        let min = f16_le_to_f32([unsafe { *block_ptr.add(2) }, unsafe { *block_ptr.add(3) }]);
        let scales = unsafe { std::slice::from_raw_parts(block_ptr.add(4), 12) };
        let qs = unsafe { block_ptr.add(16) };

        for group_pair in 0..4 {
            let group1 = group_pair * 2;
            let group2 = group1 + 1;
            let (scale1, min_scale1) = get_scale_min_k4(group1, scales);
            let (scale2, min_scale2) = get_scale_min_k4(group2, scales);
            let d1 = _mm256_set1_ps(d * scale1 as f32);
            let min1 = _mm256_set1_ps(min * min_scale1 as f32);
            let d2 = _mm256_set1_ps(d * scale2 as f32);
            let min2 = _mm256_set1_ps(min * min_scale2 as f32);
            let q_base = group_pair * 32;
            let v_base = group_pair * 64;
            // Unroll inner loop to 4 lanes feeding 4 accumulators.
            let p0 = unsafe { _mm_loadl_epi64(qs.add(q_base).cast::<__m128i>()) };
            let p1 = unsafe { _mm_loadl_epi64(qs.add(q_base + 8).cast::<__m128i>()) };
            let p2 = unsafe { _mm_loadl_epi64(qs.add(q_base + 16).cast::<__m128i>()) };
            let p3 = unsafe { _mm_loadl_epi64(qs.add(q_base + 24).cast::<__m128i>()) };

            let l0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_and_si128(p0, mask)));
            let h0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_and_si128(
                _mm_srli_epi16(p0, 4),
                mask,
            )));
            let l1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_and_si128(p1, mask)));
            let h1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_and_si128(
                _mm_srli_epi16(p1, 4),
                mask,
            )));
            let l2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_and_si128(p2, mask)));
            let h2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_and_si128(
                _mm_srli_epi16(p2, 4),
                mask,
            )));
            let l3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_and_si128(p3, mask)));
            let h3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_and_si128(
                _mm_srli_epi16(p3, 4),
                mask,
            )));

            let v_l0 = unsafe { _mm256_loadu_ps(v_ptr.add(v_base)) };
            let v_l1 = unsafe { _mm256_loadu_ps(v_ptr.add(v_base + 8)) };
            let v_l2 = unsafe { _mm256_loadu_ps(v_ptr.add(v_base + 16)) };
            let v_l3 = unsafe { _mm256_loadu_ps(v_ptr.add(v_base + 24)) };
            let v_h0 = unsafe { _mm256_loadu_ps(v_ptr.add(v_base + 32)) };
            let v_h1 = unsafe { _mm256_loadu_ps(v_ptr.add(v_base + 40)) };
            let v_h2 = unsafe { _mm256_loadu_ps(v_ptr.add(v_base + 48)) };
            let v_h3 = unsafe { _mm256_loadu_ps(v_ptr.add(v_base + 56)) };

            let t_l0 = _mm256_fmsub_ps(l0, d1, min1);
            let t_l1 = _mm256_fmsub_ps(l1, d1, min1);
            let t_l2 = _mm256_fmsub_ps(l2, d1, min1);
            let t_l3 = _mm256_fmsub_ps(l3, d1, min1);
            let t_h0 = _mm256_fmsub_ps(h0, d2, min2);
            let t_h1 = _mm256_fmsub_ps(h1, d2, min2);
            let t_h2 = _mm256_fmsub_ps(h2, d2, min2);
            let t_h3 = _mm256_fmsub_ps(h3, d2, min2);

            acc0 = _mm256_fmadd_ps(t_l0, v_l0, acc0);
            acc1 = _mm256_fmadd_ps(t_l1, v_l1, acc1);
            acc2 = _mm256_fmadd_ps(t_l2, v_l2, acc2);
            acc3 = _mm256_fmadd_ps(t_l3, v_l3, acc3);
            acc0 = _mm256_fmadd_ps(t_h0, v_h0, acc0);
            acc1 = _mm256_fmadd_ps(t_h1, v_h1, acc1);
            acc2 = _mm256_fmadd_ps(t_h2, v_h2, acc2);
            acc3 = _mm256_fmadd_ps(t_h3, v_h3, acc3);
        }
    }

    let acc01 = _mm256_add_ps(acc0, acc1);
    let acc23 = _mm256_add_ps(acc2, acc3);
    let acc = _mm256_add_ps(acc01, acc23);
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

#[inline]
pub(super) fn q2_k_dot_scalar(block: &[u8], vector: &[f32]) -> f32 {
    let scales = &block[0..16];
    let qs = &block[16..80];
    let d = f16_le_to_f32([block[80], block[81]]);
    let min_v = f16_le_to_f32([block[82], block[83]]);
    let mut sum = 0.0_f32;
    let mut is: usize = 0;
    let mut weight_off: usize = 0;
    for outer in 0..2 {
        let qs_off = outer * 32;
        for _ in 0..4 {
            let sc1 = scales[is];
            is += 1;
            let sc2 = scales[is];
            is += 1;
            let shift = ((is / 2 - 1) % 4) * 2;
            let dl1 = d * (sc1 & 0x0F) as f32;
            let ml1 = min_v * (sc1 >> 4) as f32;
            let dl2 = d * (sc2 & 0x0F) as f32;
            let ml2 = min_v * (sc2 >> 4) as f32;
            for l in 0..16 {
                let q = ((qs[qs_off + l] >> shift) & 3) as f32;
                sum += (dl1 * q - ml1) * vector[weight_off + l];
            }
            for l in 0..16 {
                let q = ((qs[qs_off + 16 + l] >> shift) & 3) as f32;
                sum += (dl2 * q - ml2) * vector[weight_off + 16 + l];
            }
            weight_off += 32;
        }
    }
    sum
}

#[inline]
pub(super) fn q2_k_dot(block: &[u8], vector: &[f32]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { q2_k_dot_avx2(block, vector) };
        }
    }
    q2_k_dot_scalar(block, vector)
}

/// AVX2 + FMA dot product between a Q2_K block (256 weights, 2-bit packed)
/// and a 256-element f32 vector. Dequantizes 2-bit quants on the fly, mirroring
/// the Q4_K fast path so the matrix is read at ~2 bits/weight — this is the
/// decode hot path for Q2_K-quantized models.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(super) unsafe fn q2_k_dot_avx2(block: &[u8], vector: &[f32]) -> f32 {
    let scales = &block[0..16];
    let qs_ptr = block.as_ptr().wrapping_add(16);
    let d = f16_le_to_f32([block[80], block[81]]);
    let min_v = f16_le_to_f32([block[82], block[83]]);
    let mask3 = _mm_set1_epi8(3);
    let mut acc = _mm256_setzero_ps();

    // Unroll the inner shift loop so _mm_srli_epi16 takes a const immediate.
    macro_rules! sub_block {
        ($shift:literal, $is_idx:expr, $qs_outer:expr, $v_outer:expr, $weight_idx:expr) => {{
            let sc1 = scales[$is_idx];
            let sc2 = scales[$is_idx + 1];
            let dl1 = _mm256_set1_ps(d * (sc1 & 0x0F) as f32);
            let ml1 = _mm256_set1_ps(min_v * (sc1 >> 4) as f32);
            let dl2 = _mm256_set1_ps(d * (sc2 & 0x0F) as f32);
            let ml2 = _mm256_set1_ps(min_v * (sc2 >> 4) as f32);
            // Lower 16 weights (two lanes of 8).
            for half in 0..2 {
                let lane_ptr = $qs_outer.add(half * 8);
                let q8 = _mm_loadl_epi64(lane_ptr as *const __m128i);
                let shifted = if $shift == 0 {
                    q8
                } else {
                    _mm_srli_epi16(q8, $shift)
                };
                let masked = _mm_and_si128(shifted, mask3);
                let f32x8 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(masked));
                let term = _mm256_fmsub_ps(f32x8, dl1, ml1);
                let v = _mm256_loadu_ps(vector.as_ptr().add($weight_idx + half * 8));
                acc = _mm256_fmadd_ps(term, v, acc);
            }
            // Upper 16 weights (two lanes of 8).
            for half in 0..2 {
                let lane_ptr = $qs_outer.add(16 + half * 8);
                let q8 = _mm_loadl_epi64(lane_ptr as *const __m128i);
                let shifted = if $shift == 0 {
                    q8
                } else {
                    _mm_srli_epi16(q8, $shift)
                };
                let masked = _mm_and_si128(shifted, mask3);
                let f32x8 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(masked));
                let term = _mm256_fmsub_ps(f32x8, dl2, ml2);
                let v = _mm256_loadu_ps(vector.as_ptr().add($weight_idx + 16 + half * 8));
                acc = _mm256_fmadd_ps(term, v, acc);
            }
            let _ = $v_outer; // silence unused (kept for symmetry)
        }};
    }

    // Outer 0: qs[0..32], weights 0..128.
    let qs0 = qs_ptr;
    sub_block!(0, 0, qs0, 0usize, 0usize);
    sub_block!(2, 2, qs0, 0usize, 32usize);
    sub_block!(4, 4, qs0, 0usize, 64usize);
    sub_block!(6, 6, qs0, 0usize, 96usize);
    // Outer 1: qs[32..64], weights 128..256.
    let qs1 = qs_ptr.add(32);
    sub_block!(0, 8, qs1, 0usize, 128usize);
    sub_block!(2, 10, qs1, 0usize, 160usize);
    sub_block!(4, 12, qs1, 0usize, 192usize);
    sub_block!(6, 14, qs1, 0usize, 224usize);

    // Horizontal sum.
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

pub(super) fn gemv_q2_k_f32_fused(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    let blocks_per_row = cols / QK_K;
    let expected_matrix_len = rows * blocks_per_row * BLOCK_Q2_K_SIZE;
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
        });
    }
    if vector.len() != cols {
        return Err(GemvError::InvalidVectorLength {
            expected: cols,
            actual: vector.len(),
        });
    }
    if output.len() != rows {
        return Err(GemvError::InvalidOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }

    let compute_row = |row_idx: usize| {
        let mut sum = 0.0_f32;
        let row_start = row_idx * blocks_per_row * BLOCK_Q2_K_SIZE;
        let row_blocks =
            &quantized_matrix[row_start..row_start + (blocks_per_row * BLOCK_Q2_K_SIZE)];
        for (block_idx, block) in row_blocks.chunks_exact(BLOCK_Q2_K_SIZE).enumerate() {
            let vector_offset = block_idx * QK_K;
            sum += q2_k_dot(block, &vector[vector_offset..vector_offset + QK_K]);
        }
        sum
    };

    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        output
            .par_iter_mut()
            .with_min_len(32)
            .enumerate()
            .for_each(|(row_idx, out)| *out = compute_row(row_idx));
    } else {
        for (row_idx, out) in output.iter_mut().enumerate() {
            *out = compute_row(row_idx);
        }
    }
    Ok(())
}

#[inline]
pub(super) fn q6_k_value(block: &[u8], idx: usize) -> f32 {
    let d = f16_le_to_f32([block[208], block[209]]);
    let ql = &block[0..128];
    let qh = &block[128..192];
    let sc = &block[192..208];
    let half = idx / 128;
    let rem = idx % 128;
    let l = rem % 32;
    let group = rem / 32;
    // Each 128-weight half of the block consumes its own 64-byte ql window and
    // 32-byte qh window; advance into the second half for idx >= 128.
    let ql_base = half * 64;
    let qh_base = half * 32;
    let (q_low, q_high, scale_idx) = match group {
        0 => (ql[ql_base + l] & 0x0f, qh[qh_base + l] & 0x03, l / 16),
        1 => (
            ql[ql_base + l + 32] & 0x0f,
            (qh[qh_base + l] >> 2) & 0x03,
            l / 16 + 2,
        ),
        2 => (
            ql[ql_base + l] >> 4,
            (qh[qh_base + l] >> 4) & 0x03,
            l / 16 + 4,
        ),
        _ => (
            ql[ql_base + l + 32] >> 4,
            (qh[qh_base + l] >> 6) & 0x03,
            l / 16 + 6,
        ),
    };
    let scale = sc[scale_idx + half * 8] as i8 as f32;
    let q = ((q_low as i32) | ((q_high as i32) << 4)) - 32;
    d * scale * q as f32
}

#[inline]
pub(super) fn q6_k_dot(block: &[u8], vector: &[f32]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { q6_k_dot_avx2(block, vector) };
        }
    }
    q6_k_dot_scalar(block, vector)
}

#[inline]
pub(super) fn q6_k_dot_scalar(block: &[u8], vector: &[f32]) -> f32 {
    let d = f16_le_to_f32([block[208], block[209]]);
    let ql = &block[0..128];
    let qh = &block[128..192];
    let sc = &block[192..208];
    let mut sum = 0.0_f32;
    let mut q_ptr = 0;
    for half in 0..2 {
        let scale_base = half * 8;
        let ql_base = half * 64;
        let qh_base = half * 32;
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[ql_base + l] & 0x0f) as i32 | (((qh[qh_base + l] & 3) as i32) << 4)) - 32;
            let q2 = ((ql[ql_base + l + 32] & 0x0f) as i32
                | ((((qh[qh_base + l] >> 2) & 3) as i32) << 4))
                - 32;
            let q3 =
                ((ql[ql_base + l] >> 4) as i32 | ((((qh[qh_base + l] >> 4) & 3) as i32) << 4)) - 32;
            let q4 = ((ql[ql_base + l + 32] >> 4) as i32
                | ((((qh[qh_base + l] >> 6) & 3) as i32) << 4))
                - 32;
            sum += d * sc[scale_base + is] as i8 as f32 * q1 as f32 * vector[q_ptr + l];
            sum += d * sc[scale_base + is + 2] as i8 as f32 * q2 as f32 * vector[q_ptr + 32 + l];
            sum += d * sc[scale_base + is + 4] as i8 as f32 * q3 as f32 * vector[q_ptr + 64 + l];
            sum += d * sc[scale_base + is + 6] as i8 as f32 * q4 as f32 * vector[q_ptr + 96 + l];
        }
        q_ptr += 128;
    }
    sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(super) unsafe fn q6_k_dot_avx2(block: &[u8], vector: &[f32]) -> f32 {
    let d = f16_le_to_f32([block[208], block[209]]);
    let ql = block.as_ptr();
    let qh = block.as_ptr().wrapping_add(128);
    let sc = &block[192..208];
    let mask_low = _mm_set1_epi8(0x0f);
    let mask_high = _mm_set1_epi8(0x03);
    let offset = _mm256_set1_ps(32.0);
    let mut acc = _mm256_setzero_ps();

    for half in 0..2 {
        let scale_base = half * 8;
        let v_base = half * 128;
        let ql_base = half * 64;
        let qh_base = half * 32;
        for l in (0..32).step_by(8) {
            let is = l / 16;
            let ql1 = unsafe { _mm_loadl_epi64(ql.add(ql_base + l).cast::<__m128i>()) };
            let ql2 = unsafe { _mm_loadl_epi64(ql.add(ql_base + l + 32).cast::<__m128i>()) };
            let qh_v = unsafe { _mm_loadl_epi64(qh.add(qh_base + l).cast::<__m128i>()) };

            let low1 = _mm_and_si128(ql1, mask_low);
            let low2 = _mm_and_si128(ql2, mask_low);
            let low3 = _mm_and_si128(_mm_srli_epi16(ql1, 4), mask_low);
            let low4 = _mm_and_si128(_mm_srli_epi16(ql2, 4), mask_low);
            let high1 = _mm_slli_epi16(_mm_and_si128(qh_v, mask_high), 4);
            let high2 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_v, 2), mask_high), 4);
            let high3 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_v, 4), mask_high), 4);
            let high4 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_v, 6), mask_high), 4);

            macro_rules! acc_group {
                ($low:expr, $high:expr, $scale_idx:expr, $vec_off:expr) => {{
                    let q_u8 = _mm_or_si128($low, $high);
                    let q = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q_u8)), offset);
                    let scale = _mm256_set1_ps(d * sc[$scale_idx] as i8 as f32);
                    let w = _mm256_mul_ps(q, scale);
                    let v = unsafe { _mm256_loadu_ps(vector.as_ptr().add($vec_off)) };
                    acc = _mm256_fmadd_ps(w, v, acc);
                }};
            }

            acc_group!(low1, high1, scale_base + is, v_base + l);
            acc_group!(low2, high2, scale_base + is + 2, v_base + 32 + l);
            acc_group!(low3, high3, scale_base + is + 4, v_base + 64 + l);
            acc_group!(low4, high4, scale_base + is + 6, v_base + 96 + l);
        }
    }

    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

#[inline]
pub(super) fn accumulate_q6_k_block(block: &[u8], factor: f32, output: &mut [f32]) {
    let d = f16_le_to_f32([block[208], block[209]]);
    let ql = &block[0..128];
    let qh = &block[128..192];
    let sc = &block[192..208];
    let mut q_ptr = 0;
    for half in 0..2 {
        let scale_base = half * 8;
        let ql_base = half * 64;
        let qh_base = half * 32;
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[ql_base + l] & 0x0f) as i32 | (((qh[qh_base + l] & 3) as i32) << 4)) - 32;
            let q2 = ((ql[ql_base + l + 32] & 0x0f) as i32
                | ((((qh[qh_base + l] >> 2) & 3) as i32) << 4))
                - 32;
            let q3 =
                ((ql[ql_base + l] >> 4) as i32 | ((((qh[qh_base + l] >> 4) & 3) as i32) << 4)) - 32;
            let q4 = ((ql[ql_base + l + 32] >> 4) as i32
                | ((((qh[qh_base + l] >> 6) & 3) as i32) << 4))
                - 32;
            output[q_ptr + l] += d * sc[scale_base + is] as i8 as f32 * q1 as f32 * factor;
            output[q_ptr + 32 + l] += d * sc[scale_base + is + 2] as i8 as f32 * q2 as f32 * factor;
            output[q_ptr + 64 + l] += d * sc[scale_base + is + 4] as i8 as f32 * q3 as f32 * factor;
            output[q_ptr + 96 + l] += d * sc[scale_base + is + 6] as i8 as f32 * q4 as f32 * factor;
        }
        q_ptr += 128;
    }
}

// IQ1_S GEMV: decode-once approach.
// Each block is 50 bytes for 256 values. We dequantize each block to f32
// in a scratch buffer, then do a standard f32 dot product.
// This is slower than native IQ1_S dot products but requires no large
// lookup table and is correct.

#[inline]
#[allow(clippy::needless_range_loop)]
pub(super) fn iq1s_grid_decode(index: u16, out: &mut [i8; 8]) {
    let mut idx = index;
    for i in 0..8 {
        let bits = (idx & 3) as i8;
        out[i] = match bits {
            0 => -1,
            1 => 0,
            _ => 1,
        };
        idx >>= 2;
        if i == 3 {
            idx = index >> 8;
        }
    }
}

#[inline]
pub(super) fn iq1s_dequantize_block(block: &[u8], out: &mut [f32]) {
    assert_eq!(block.len(), BLOCK_IQ1_S_SIZE);
    assert_eq!(out.len(), QK_K);
    let d = f16_le_to_f32([block[0], block[1]]);
    let qs = &block[2..34];
    let qh = &block[34..50];
    let qh_u16: [u16; 16] = std::array::from_fn(|i| u16::from_le_bytes([qh[i * 2], qh[i * 2 + 1]]));
    let mut out_ptr = 0_usize;
    let mut grid_vals = [0_i8; 8];
    for ib in 0..(QK_K / 32) {
        let dl = d * (2.0 * (((qh_u16[ib] >> 12) & 7) as f32) + 1.0);
        let delta = if qh_u16[ib] & 0x8000 != 0 {
            -IQ1S_DELTA
        } else {
            IQ1S_DELTA
        };
        for l in 0..4 {
            let grid_idx = (qs[l + ib * 4] as u16) | (((qh_u16[ib] >> (3 * l)) & 7) << 8);
            iq1s_grid_decode(grid_idx, &mut grid_vals);
            for j in 0..8 {
                out[out_ptr + j] = dl * (grid_vals[j] as f32 + delta);
            }
            out_ptr += 8;
        }
    }
}

#[inline]
pub(super) fn ue4m3_to_f32(byte: u8) -> f32 {
    let exp = (byte >> 3) & 0x0f;
    let mant = byte & 0x07;
    if exp == 0 {
        (mant as f32) * 2.0_f32.powi(-9)
    } else {
        (1.0 + (mant as f32) / 8.0) * 2.0_f32.powi(exp as i32 - 7)
    }
}

#[inline]
#[allow(clippy::needless_range_loop)]
pub(super) fn nvfp4_dequantize_block(block: &[u8], out: &mut [f32]) {
    assert_eq!(block.len(), BLOCK_NVFP4_SIZE);
    assert_eq!(out.len(), QK_NVFP4);
    let scales = &block[..QK_NVFP4 / QK_NVFP4_SUB];
    let qs = &block[QK_NVFP4 / QK_NVFP4_SUB..];
    for sub in 0..(QK_NVFP4 / QK_NVFP4_SUB) {
        let scale = ue4m3_to_f32(scales[sub]);
        let q_base = sub * (QK_NVFP4_SUB / 2);
        let out_base = sub * QK_NVFP4_SUB;
        for j in 0..(QK_NVFP4_SUB / 2) {
            let packed = qs[q_base + j];
            out[out_base + j] = scale * E2M1_DOUBLED_VALUES[(packed & 0x0f) as usize];
            out[out_base + j + QK_NVFP4_SUB / 2] =
                scale * E2M1_DOUBLED_VALUES[(packed >> 4) as usize];
        }
    }
}

pub(super) fn gemv_nvfp4_f32_fused(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    let blocks_per_row = cols / QK_NVFP4;
    let expected_matrix_len = rows * blocks_per_row * BLOCK_NVFP4_SIZE;
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
        });
    }
    if vector.len() != cols {
        return Err(GemvError::InvalidVectorLength {
            expected: cols,
            actual: vector.len(),
        });
    }
    if output.len() != rows {
        return Err(GemvError::InvalidOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }

    let compute_row = |row_idx: usize, scratch: &mut [f32; QK_NVFP4]| {
        let mut sum = 0.0_f32;
        let row_start = row_idx * blocks_per_row * BLOCK_NVFP4_SIZE;
        let row_blocks =
            &quantized_matrix[row_start..row_start + blocks_per_row * BLOCK_NVFP4_SIZE];
        for (block_idx, block) in row_blocks.chunks_exact(BLOCK_NVFP4_SIZE).enumerate() {
            let v_off = block_idx * QK_NVFP4;
            nvfp4_dequantize_block(block, scratch);
            sum += dot_f32_fast(scratch, &vector[v_off..v_off + QK_NVFP4]);
        }
        sum
    };

    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(row_idx, out)| {
                let mut scratch = [0.0_f32; QK_NVFP4];
                *out = compute_row(row_idx, &mut scratch);
            });
    } else {
        let mut scratch = [0.0_f32; QK_NVFP4];
        for (row_idx, out) in output.iter_mut().enumerate() {
            *out = compute_row(row_idx, &mut scratch);
        }
    }
    Ok(())
}

pub(super) fn gemv_iq1_s_f32_fused(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    let blocks_per_row = cols / QK_K;
    let expected_matrix_len = rows * blocks_per_row * BLOCK_IQ1_S_SIZE;
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
        });
    }
    if vector.len() != cols {
        return Err(GemvError::InvalidVectorLength {
            expected: cols,
            actual: vector.len(),
        });
    }
    if output.len() != rows {
        return Err(GemvError::InvalidOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }

    let compute_row = |row_idx: usize, scratch: &mut [f32; QK_K]| {
        let mut sum = 0.0_f32;
        let row_start = row_idx * blocks_per_row * BLOCK_IQ1_S_SIZE;
        let row_blocks =
            &quantized_matrix[row_start..row_start + (blocks_per_row * BLOCK_IQ1_S_SIZE)];
        for (block_idx, block) in row_blocks.chunks_exact(BLOCK_IQ1_S_SIZE).enumerate() {
            let v_off = block_idx * QK_K;
            iq1s_dequantize_block(block, scratch);
            sum += crate::flash_attention::dot_product_f32(scratch, &vector[v_off..v_off + QK_K]);
        }
        sum
    };

    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(row_idx, out)| {
                let mut scratch = [0.0_f32; QK_K];
                *out = compute_row(row_idx, &mut scratch);
            });
    } else {
        let mut scratch = [0.0_f32; QK_K];
        for (row_idx, out) in output.iter_mut().enumerate() {
            *out = compute_row(row_idx, &mut scratch);
        }
    }
    Ok(())
}

// IQ1_M GEMV: similar decode-once approach.

#[inline]
pub(super) fn iq1m_dequantize_block(block: &[u8], out: &mut [f32]) {
    assert_eq!(block.len(), BLOCK_IQ1_M_SIZE);
    assert_eq!(out.len(), QK_K);
    let qs = &block[0..32];
    let qh = &block[32..48];
    let scales = &block[48..56];
    let sc: [u16; 4] =
        std::array::from_fn(|i| u16::from_le_bytes([scales[i * 2], scales[i * 2 + 1]]));
    let scale_u16 =
        (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    let d = f16_bits_to_f32(scale_u16);
    let mut out_ptr = 0_usize;
    let mut grid_vals = [0_i8; 8];
    for ib in 0..(QK_K / 32) {
        let sc_ib = scales[ib / 2];
        let dl1 = d * (2.0 * (((sc_ib >> (6 * (ib % 2))) & 0x7) as f32) + 1.0);
        let dl2 = d * (2.0 * (((sc_ib >> (6 * (ib % 2) + 3)) & 0x7) as f32) + 1.0);
        let idx0 = qs[ib * 4] as u16 | ((qh[ib * 2] as u16) << 8 & 0x700);
        let idx1 = qs[ib * 4 + 1] as u16 | ((qh[ib * 2] as u16) << 4 & 0x700);
        let idx2 = qs[ib * 4 + 2] as u16 | ((qh[ib * 2 + 1] as u16) << 8 & 0x700);
        let idx3 = qs[ib * 4 + 3] as u16 | ((qh[ib * 2 + 1] as u16) << 4 & 0x700);
        let deltas = [
            if qh[ib * 2] & 0x08 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            },
            if qh[ib * 2] & 0x80 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            },
            if qh[ib * 2 + 1] & 0x08 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            },
            if qh[ib * 2 + 1] & 0x80 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            },
        ];
        iq1s_grid_decode(idx0, &mut grid_vals);
        for j in 0..8 {
            out[out_ptr + j] = dl1 * (grid_vals[j] as f32 + deltas[0]);
        }
        out_ptr += 8;
        iq1s_grid_decode(idx1, &mut grid_vals);
        for j in 0..8 {
            out[out_ptr + j] = dl1 * (grid_vals[j] as f32 + deltas[1]);
        }
        out_ptr += 8;
        iq1s_grid_decode(idx2, &mut grid_vals);
        for j in 0..8 {
            out[out_ptr + j] = dl2 * (grid_vals[j] as f32 + deltas[2]);
        }
        out_ptr += 8;
        iq1s_grid_decode(idx3, &mut grid_vals);
        for j in 0..8 {
            out[out_ptr + j] = dl2 * (grid_vals[j] as f32 + deltas[3]);
        }
        out_ptr += 8;
    }
}

pub(super) fn gemv_iq1_m_f32_fused(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    let blocks_per_row = cols / QK_K;
    let expected_matrix_len = rows * blocks_per_row * BLOCK_IQ1_M_SIZE;
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
        });
    }
    if vector.len() != cols {
        return Err(GemvError::InvalidVectorLength {
            expected: cols,
            actual: vector.len(),
        });
    }
    if output.len() != rows {
        return Err(GemvError::InvalidOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }

    let compute_row = |row_idx: usize, scratch: &mut [f32; QK_K]| {
        let mut sum = 0.0_f32;
        let row_start = row_idx * blocks_per_row * BLOCK_IQ1_M_SIZE;
        let row_blocks =
            &quantized_matrix[row_start..row_start + (blocks_per_row * BLOCK_IQ1_M_SIZE)];
        for (block_idx, block) in row_blocks.chunks_exact(BLOCK_IQ1_M_SIZE).enumerate() {
            let v_off = block_idx * QK_K;
            iq1m_dequantize_block(block, scratch);
            sum += crate::flash_attention::dot_product_f32(scratch, &vector[v_off..v_off + QK_K]);
        }
        sum
    };

    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(row_idx, out)| {
                let mut scratch = [0.0_f32; QK_K];
                *out = compute_row(row_idx, &mut scratch);
            });
    } else {
        let mut scratch = [0.0_f32; QK_K];
        for (row_idx, out) in output.iter_mut().enumerate() {
            *out = compute_row(row_idx, &mut scratch);
        }
    }
    Ok(())
}

/// Decode one IQ4_XS block (256 weights) and dot it against 256 vector values.
///
/// Reuses the tested scalar decoder; the non-linear LUT lookup makes a dedicated
/// integer kernel awkward, and token generation is memory-bound anyway, so
/// decode-then-dot is the pragmatic path.
pub(super) fn iq4_xs_dot(block: &[u8], vector: &[f32]) -> f32 {
    let mut decoded = [0.0_f32; QK_K];
    if dequantize_iq4_xs_scalar(block, &mut decoded).is_err() {
        return 0.0;
    }
    let mut sum = 0.0_f32;
    for i in 0..QK_K {
        sum += decoded[i] * vector[i];
    }
    sum
}

/// IQ4_XS GEMV: per row, decode each 136-byte block and dot against the input.
pub(super) fn gemv_iq4_xs_f32(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    let blocks_per_row = cols / QK_K;
    let expected_matrix_len = rows * blocks_per_row * BLOCK_IQ4_XS_SIZE;
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
        });
    }
    if vector.len() != cols {
        return Err(GemvError::InvalidVectorLength {
            expected: cols,
            actual: vector.len(),
        });
    }
    if output.len() != rows {
        return Err(GemvError::InvalidOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }

    let row_bytes = blocks_per_row * BLOCK_IQ4_XS_SIZE;
    let compute_row = |row_idx: usize| -> f32 {
        let row_start = row_idx * row_bytes;
        let row = &quantized_matrix[row_start..row_start + row_bytes];
        let mut sum = 0.0_f32;
        for (block_idx, block) in row.chunks_exact(BLOCK_IQ4_XS_SIZE).enumerate() {
            let v_off = block_idx * QK_K;
            sum += iq4_xs_dot(block, &vector[v_off..v_off + QK_K]);
        }
        sum
    };

    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        output
            .par_iter_mut()
            .with_min_len(32)
            .enumerate()
            .for_each(|(row_idx, out)| *out = compute_row(row_idx));
    } else {
        for (row_idx, out) in output.iter_mut().enumerate() {
            *out = compute_row(row_idx);
        }
    }
    Ok(())
}

pub(super) fn gemv_q6_k_f32_fused(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    let blocks_per_row = cols / QK_K;
    let expected_matrix_len = rows * blocks_per_row * BLOCK_Q6_K_SIZE;
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
        });
    }
    if vector.len() != cols {
        return Err(GemvError::InvalidVectorLength {
            expected: cols,
            actual: vector.len(),
        });
    }
    if output.len() != rows {
        return Err(GemvError::InvalidOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let use_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let use_avx2 = false;

    let row_bytes = blocks_per_row * BLOCK_Q6_K_SIZE;
    let compute_row = |row_idx: usize| -> f32 {
        let row_start = row_idx * row_bytes;
        let row = &quantized_matrix[row_start..row_start + row_bytes];
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if use_avx2 {
                return unsafe { q6_k_row_dot_avx2(row, blocks_per_row, vector) };
            }
        }
        let _ = use_avx2;
        let mut sum = 0.0_f32;
        for (block_idx, block) in row.chunks_exact(BLOCK_Q6_K_SIZE).enumerate() {
            let v_off = block_idx * QK_K;
            sum += q6_k_dot_scalar(block, &vector[v_off..v_off + QK_K]);
        }
        sum
    };

    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        output
            .par_iter_mut()
            .with_min_len(32)
            .enumerate()
            .for_each(|(row_idx, out)| *out = compute_row(row_idx));
    } else {
        for (row_idx, out) in output.iter_mut().enumerate() {
            *out = compute_row(row_idx);
        }
    }
    Ok(())
}

/// Whole-row Q6_K dot product. Same idea as `q4_k_row_dot_avx2`: one final
/// horizontal reduce instead of one per block.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(super) unsafe fn q6_k_row_dot_avx2(row: &[u8], blocks_per_row: usize, vector: &[f32]) -> f32 {
    let mask_low = _mm_set1_epi8(0x0f);
    let mask_high = _mm_set1_epi8(0x03);
    let offset = _mm256_set1_ps(32.0);
    let mut acc = _mm256_setzero_ps();

    for block_idx in 0..blocks_per_row {
        let block_ptr = row.as_ptr().wrapping_add(block_idx * BLOCK_Q6_K_SIZE);
        let v_ptr = vector.as_ptr().wrapping_add(block_idx * QK_K);

        let d = f16_le_to_f32([unsafe { *block_ptr.add(208) }, unsafe {
            *block_ptr.add(209)
        }]);
        let ql = block_ptr;
        let qh = block_ptr.wrapping_add(128);
        let sc = unsafe { std::slice::from_raw_parts(block_ptr.add(192), 16) };

        for half in 0..2 {
            let scale_base = half * 8;
            let v_base = half * 128;
            let ql_base = half * 64;
            let qh_base = half * 32;
            for l in (0..32).step_by(8) {
                let is = l / 16;
                let ql1 = unsafe { _mm_loadl_epi64(ql.add(ql_base + l).cast::<__m128i>()) };
                let ql2 = unsafe { _mm_loadl_epi64(ql.add(ql_base + l + 32).cast::<__m128i>()) };
                let qh_v = unsafe { _mm_loadl_epi64(qh.add(qh_base + l).cast::<__m128i>()) };

                let low1 = _mm_and_si128(ql1, mask_low);
                let low2 = _mm_and_si128(ql2, mask_low);
                let low3 = _mm_and_si128(_mm_srli_epi16(ql1, 4), mask_low);
                let low4 = _mm_and_si128(_mm_srli_epi16(ql2, 4), mask_low);
                let high1 = _mm_slli_epi16(_mm_and_si128(qh_v, mask_high), 4);
                let high2 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_v, 2), mask_high), 4);
                let high3 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_v, 4), mask_high), 4);
                let high4 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_v, 6), mask_high), 4);

                macro_rules! acc_group {
                    ($low:expr, $high:expr, $scale_idx:expr, $vec_off:expr) => {{
                        let q_u8 = _mm_or_si128($low, $high);
                        let q =
                            _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q_u8)), offset);
                        let scale = _mm256_set1_ps(d * sc[$scale_idx] as i8 as f32);
                        let w = _mm256_mul_ps(q, scale);
                        let v = unsafe { _mm256_loadu_ps(v_ptr.add($vec_off)) };
                        acc = _mm256_fmadd_ps(w, v, acc);
                    }};
                }

                acc_group!(low1, high1, scale_base + is, v_base + l);
                acc_group!(low2, high2, scale_base + is + 2, v_base + 32 + l);
                acc_group!(low3, high3, scale_base + is + 4, v_base + 64 + l);
                acc_group!(low4, high4, scale_base + is + 6, v_base + 96 + l);
            }
        }
    }

    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

#[allow(clippy::too_many_arguments, dead_code)]
pub(super) fn gemv_qk_f32_fused(
    quantized_matrix: &[u8],
    rows: usize,
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
    block_size: usize,
    bits: usize,
    zero_point: f32,
) -> Result<(), GemvError> {
    let blocks_per_row = cols / QK_K;
    let expected_matrix_len = rows * blocks_per_row * block_size;
    if quantized_matrix.len() != expected_matrix_len {
        return Err(GemvError::InvalidMatrixLength {
            expected: expected_matrix_len,
            actual: quantized_matrix.len(),
        });
    }
    if vector.len() != cols {
        return Err(GemvError::InvalidVectorLength {
            expected: cols,
            actual: vector.len(),
        });
    }
    if output.len() != rows {
        return Err(GemvError::InvalidOutputLength {
            expected: rows,
            actual: output.len(),
        });
    }

    let compute_row = |row_idx: usize| {
        let mut sum = 0.0_f32;
        let row_start = row_idx * blocks_per_row * block_size;
        let row_blocks = &quantized_matrix[row_start..row_start + (blocks_per_row * block_size)];
        for (block_idx, block) in row_blocks.chunks_exact(block_size).enumerate() {
            let d = f16_le_to_f32([block[0], block[1]]);
            let bitstream = &block[2..];
            let vector_offset = block_idx * QK_K;
            for idx in 0..QK_K {
                let q = extract_bits(bitstream, idx, bits) as f32;
                sum += (q - zero_point) * d * vector[vector_offset + idx];
            }
        }
        sum
    };

    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        output
            .par_iter_mut()
            .with_min_len(32)
            .enumerate()
            .for_each(|(row_idx, out)| *out = compute_row(row_idx));
    } else {
        for (row_idx, out) in output.iter_mut().enumerate() {
            *out = compute_row(row_idx);
        }
    }
    Ok(())
}

pub fn extract_bits(bitstream: &[u8], index: usize, bits: usize) -> u32 {
    let bit_offset = index * bits;
    let byte_index = bit_offset / 8;
    let shift = bit_offset % 8;

    let mut acc = 0_u32;
    for i in 0..4 {
        if let Some(byte) = bitstream.get(byte_index + i) {
            acc |= (*byte as u32) << (8 * i);
        }
    }

    (acc >> shift) & ((1_u32 << bits) - 1)
}

pub(super) fn gemv_q8_0_f32_fused(
    quantized_matrix: &[u8],
    cols: usize,
    vector: &[f32],
    output: &mut [f32],
) -> Result<(), GemvError> {
    let rows = output.len();
    let blocks_per_row = cols / QK8_0;
    let compute_row = |row_idx: usize| {
        let mut sum = 0.0_f32;
        let row_start = row_idx * blocks_per_row * BLOCK_Q8_0_SIZE;
        let row_blocks =
            &quantized_matrix[row_start..row_start + (blocks_per_row * BLOCK_Q8_0_SIZE)];
        for (block_idx, block) in row_blocks.chunks_exact(BLOCK_Q8_0_SIZE).enumerate() {
            let vector_offset = block_idx * QK8_0;
            sum += q8_0_dot(block, &vector[vector_offset..vector_offset + QK8_0]);
        }
        sum
    };

    if rows.saturating_mul(cols) >= PARALLEL_GEMV_MIN_OPS {
        output
            .par_iter_mut()
            .with_min_len(32)
            .enumerate()
            .for_each(|(row_idx, out)| *out = compute_row(row_idx));
    } else {
        for (row_idx, out) in output.iter_mut().enumerate() {
            *out = compute_row(row_idx);
        }
    }
    Ok(())
}

#[inline]
pub(super) fn q8_0_dot(block: &[u8], vector: &[f32]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { q8_0_dot_avx2(block, vector) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { q8_0_dot_neon_aarch64(block, vector) };
        }
    }
    #[cfg(target_arch = "arm")]
    {
        if std::arch::is_arm_feature_detected!("neon") {
            return unsafe { q8_0_dot_neon_arm(block, vector) };
        }
    }
    q8_0_dot_scalar(block, vector)
}

#[inline]
pub(super) fn q8_0_dot_scalar(block: &[u8], vector: &[f32]) -> f32 {
    let scale = f16_le_to_f32([block[0], block[1]]);
    block[2..]
        .iter()
        .zip(vector)
        .map(|(q, v)| (*q as i8) as f32 * scale * *v)
        .sum()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(super) unsafe fn q8_0_dot_avx2(block: &[u8], vector: &[f32]) -> f32 {
    let scale = _mm256_set1_ps(f16_le_to_f32([block[0], block[1]]));
    let mut acc = _mm256_setzero_ps();
    for lane in 0..4 {
        let q8 = unsafe { _mm_loadl_epi64(block.as_ptr().add(2 + lane * 8).cast::<__m128i>()) };
        let q = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
        let v = unsafe { _mm256_loadu_ps(vector.as_ptr().add(lane * 8)) };
        acc = _mm256_fmadd_ps(_mm256_mul_ps(q, scale), v, acc);
    }
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(super) unsafe fn q8_0_dot_neon_aarch64(block: &[u8], vector: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let scale = vdupq_n_f32(f16_le_to_f32([block[0], block[1]]));
    let mut acc = vdupq_n_f32(0.0);
    for lane in 0..4 {
        let q8 = unsafe { vld1_s8(block.as_ptr().add(2 + lane * 8).cast::<i8>()) };
        let q16 = vmovl_s8(q8);
        let q_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16)));
        let q_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16)));
        let v_lo = unsafe { vld1q_f32(vector.as_ptr().add(lane * 8)) };
        let v_hi = unsafe { vld1q_f32(vector.as_ptr().add(lane * 8 + 4)) };
        acc = vfmaq_f32(acc, vmulq_f32(q_lo, scale), v_lo);
        acc = vfmaq_f32(acc, vmulq_f32(q_hi, scale), v_hi);
    }
    vaddvq_f32(acc)
}

#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(super) unsafe fn q8_0_dot_neon_arm(block: &[u8], vector: &[f32]) -> f32 {
    use std::arch::arm::*;

    let scale = vdupq_n_f32(f16_le_to_f32([block[0], block[1]]));
    let mut acc = vdupq_n_f32(0.0);
    for lane in 0..4 {
        let q8 = unsafe { vld1_s8(block.as_ptr().add(2 + lane * 8).cast::<i8>()) };
        let q16 = vmovl_s8(q8);
        let q_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16)));
        let q_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16)));
        let v_lo = unsafe { vld1q_f32(vector.as_ptr().add(lane * 8)) };
        let v_hi = unsafe { vld1q_f32(vector.as_ptr().add(lane * 8 + 4)) };
        acc = vmlaq_f32(acc, vmulq_f32(q_lo, scale), v_lo);
        acc = vmlaq_f32(acc, vmulq_f32(q_hi, scale), v_hi);
    }
    let pair = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
    let pair = vpadd_f32(pair, pair);
    vget_lane_f32(pair, 0)
}

// Transposed GEMV functions for GGUF weight matrices stored as [input_dim, output_dim]
