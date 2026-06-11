use crate::tensor::AttentionError;

const FLASH_BLOCK_SIZE: usize = 64;
// Above this sequence length decode attention fans heads out through
// run_chunks. The spin pool keeps region dispatch in the low microseconds,
// so parallel attention pays off almost immediately (the old threshold of
// 128 left attention single-threaded for the entire early context — ~135us
// of the ~95us-per-layer decode glue at seq 100).
const PARALLEL_FLASH_ATTN_MIN_SEQ_LEN: usize = 16;

/// Compute dot product of two equal-length f32 slices.
/// Uses AVX-512 > AVX2 > NEON > scalar based on target features.
#[inline]
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl") {
            return unsafe { dot_product_f32_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_product_f32_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { dot_product_f32_neon_aarch64(a, b) };
        }
    }

    #[cfg(target_arch = "arm")]
    {
        if std::arch::is_arm_feature_detected!("neon") {
            return unsafe { dot_product_f32_neon_arm(a, b) };
        }
    }

    let mut sum = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        sum += x * y;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl")]
unsafe fn dot_product_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let mut sum = _mm512_setzero_ps();

    let chunks = len / 16;
    for i in 0..chunks {
        let va = unsafe { _mm512_loadu_ps(a.as_ptr().add(i * 16)) };
        let vb = unsafe { _mm512_loadu_ps(b.as_ptr().add(i * 16)) };
        sum = _mm512_fmadd_ps(va, vb, sum);
    }

    let mut total = _mm512_reduce_add_ps(sum);

    for i in (chunks * 16)..len {
        total += unsafe { a.get_unchecked(i) * b.get_unchecked(i) };
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let mut sum = _mm256_setzero_ps();

    let chunks = len / 8;
    for i in 0..chunks {
        let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(i * 8)) };
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr().add(i * 8)) };
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    // Horizontal sum of 8 floats
    let mut result = [0.0_f32; 8];
    unsafe { _mm256_storeu_ps(result.as_mut_ptr(), sum) };
    let mut total = result.iter().sum::<f32>();

    // Tail
    for i in (chunks * 8)..len {
        total += unsafe { a.get_unchecked(i) * b.get_unchecked(i) };
    }

    total
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_product_f32_neon_aarch64(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let len = a.len();
    let mut sum = vdupq_n_f32(0.0);

    let chunks = len / 4;
    for i in 0..chunks {
        let va = unsafe { vld1q_f32(a.as_ptr().add(i * 4)) };
        let vb = unsafe { vld1q_f32(b.as_ptr().add(i * 4)) };
        sum = vfmaq_f32(sum, va, vb);
    }

    let mut total = vaddvq_f32(sum);

    for i in (chunks * 4)..len {
        total += unsafe { a.get_unchecked(i) * b.get_unchecked(i) };
    }

    total
}

#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon")]
unsafe fn dot_product_f32_neon_arm(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::arm::*;

    let len = a.len();
    let mut sum = vdupq_n_f32(0.0);

    let chunks = len / 4;
    for i in 0..chunks {
        let va = unsafe { vld1q_f32(a.as_ptr().add(i * 4)) };
        let vb = unsafe { vld1q_f32(b.as_ptr().add(i * 4)) };
        sum = vmlaq_f32(sum, va, vb);
    }

    let pair = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    let pair = vpadd_f32(pair, pair);
    let mut total = vget_lane_f32(pair, 0);

    for i in (chunks * 4)..len {
        total += unsafe { a.get_unchecked(i) * b.get_unchecked(i) };
    }

    total
}

/// KV element type for the decode kernel: f32 rows pass through (bit-identical
/// to the historical f32-only kernel), u16 rows are IEEE half bits converted
/// on the fly (F16C on x86). Borrowing the cache in its storage dtype halves
/// attention DRAM traffic vs materializing an f32 prefix copy per layer.
pub trait KvElem: Copy + Sync {
    fn dot(query: &[f32], row: &[Self]) -> f32;
    fn axpy(out: &mut [f32], scale: f32, row: &[Self]);
}

impl KvElem for f32 {
    #[inline]
    fn dot(query: &[f32], row: &[f32]) -> f32 {
        dot_product_f32(query, row)
    }

    #[inline]
    fn axpy(out: &mut [f32], scale: f32, row: &[f32]) {
        for (o, v) in out.iter_mut().zip(row.iter()) {
            *o += scale * v;
        }
    }
}

impl KvElem for u16 {
    #[inline]
    fn dot(query: &[f32], row: &[u16]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if f16c_available() {
            // Safety: feature checked above.
            return unsafe { dot_product_f32_f16_avx2(query, row) };
        }
        let mut sum = 0.0_f32;
        for (q, &bits) in query.iter().zip(row.iter()) {
            sum += q * crate::tensor::f16_le_to_f32(bits.to_le_bytes());
        }
        sum
    }

    #[inline]
    fn axpy(out: &mut [f32], scale: f32, row: &[u16]) {
        #[cfg(target_arch = "x86_64")]
        if f16c_available() {
            // Safety: feature checked above.
            unsafe { axpy_f32_f16_avx2(out, scale, row) };
            return;
        }
        for (o, &bits) in out.iter_mut().zip(row.iter()) {
            *o += scale * crate::tensor::f16_le_to_f32(bits.to_le_bytes());
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn f16c_available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        is_x86_feature_detected!("f16c")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("avx2")
    })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_product_f32_f16_avx2(a: &[f32], b: &[u16]) -> f32 {
    use std::arch::x86_64::*;
    let len = a.len().min(b.len());
    let mut sum = _mm256_setzero_ps();
    let chunks = len / 8;
    for i in 0..chunks {
        let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(i * 8)) };
        let vh = unsafe { _mm_loadu_si128(b.as_ptr().add(i * 8) as *const __m128i) };
        let vb = _mm256_cvtph_ps(vh);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    let mut result = [0.0_f32; 8];
    unsafe { _mm256_storeu_ps(result.as_mut_ptr(), sum) };
    let mut total = result.iter().sum::<f32>();
    for i in (chunks * 8)..len {
        total += a[i] * crate::tensor::f16_le_to_f32(b[i].to_le_bytes());
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn axpy_f32_f16_avx2(out: &mut [f32], scale: f32, row: &[u16]) {
    use std::arch::x86_64::*;
    let len = out.len().min(row.len());
    let vs = _mm256_set1_ps(scale);
    let chunks = len / 8;
    for i in 0..chunks {
        let vh = unsafe { _mm_loadu_si128(row.as_ptr().add(i * 8) as *const __m128i) };
        let vv = _mm256_cvtph_ps(vh);
        let vo = unsafe { _mm256_loadu_ps(out.as_ptr().add(i * 8)) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(i * 8), _mm256_fmadd_ps(vs, vv, vo)) };
    }
    for i in (chunks * 8)..len {
        out[i] += scale * crate::tensor::f16_le_to_f32(row[i].to_le_bytes());
    }
}

/// Decode-phase flash attention: single query attends to a full key/value sequence.
///
/// This is optimized for the decode phase (one query vector, many key/value vectors)
/// where memory bandwidth dominates. It tiles over the sequence dimension to keep
/// working sets in cache and avoids allocating temporary vectors.
///
/// # Layout
/// - `query`: `[head_dim]` — the single query vector for this head
/// - `key_layer`: `[seq_len][kv_len]` — full key cache for this layer, row-major
/// - `value_layer`: `[seq_len][kv_len]` — full value cache for this layer, row-major
/// - `kv_head`: which kv head within each row to read (offset `kv_head * head_dim`)
/// - `output`: `[head_dim]` — single output vector
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_decode_f32(
    query: &[f32],
    key_layer: &[f32],
    value_layer: &[f32],
    seq_len: usize,
    head_dim: usize,
    kv_len: usize,
    kv_head: usize,
    output: &mut [f32],
) -> Result<(), AttentionError> {
    flash_attention_decode_impl(
        query,
        key_layer,
        value_layer,
        seq_len,
        head_dim,
        kv_len,
        kv_head,
        output,
    )
}

/// [`flash_attention_decode_f32`] over f16-bit K/V rows (the KV cache's F16
/// storage borrowed directly, no f32 prefix materialization).
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_decode_f16(
    query: &[f32],
    key_layer: &[u16],
    value_layer: &[u16],
    seq_len: usize,
    head_dim: usize,
    kv_len: usize,
    kv_head: usize,
    output: &mut [f32],
) -> Result<(), AttentionError> {
    flash_attention_decode_impl(
        query,
        key_layer,
        value_layer,
        seq_len,
        head_dim,
        kv_len,
        kv_head,
        output,
    )
}

#[allow(clippy::too_many_arguments)]
fn flash_attention_decode_impl<E: KvElem>(
    query: &[f32],
    key_layer: &[E],
    value_layer: &[E],
    seq_len: usize,
    head_dim: usize,
    kv_len: usize,
    kv_head: usize,
    output: &mut [f32],
) -> Result<(), AttentionError> {
    if query.len() != head_dim {
        return Err(AttentionError::InvalidQueryLength {
            expected: head_dim,
            actual: query.len(),
        });
    }
    let expected_kv = seq_len.saturating_mul(kv_len);
    if key_layer.len() != expected_kv {
        return Err(AttentionError::InvalidKeyLength {
            expected: expected_kv,
            actual: key_layer.len(),
        });
    }
    if value_layer.len() != expected_kv {
        return Err(AttentionError::InvalidValueLength {
            expected: expected_kv,
            actual: value_layer.len(),
        });
    }
    if output.len() != head_dim {
        return Err(AttentionError::InvalidOutputLength {
            expected: head_dim,
            actual: output.len(),
        });
    }
    if seq_len == 0 {
        output.fill(0.0);
        return Ok(());
    }
    if head_dim == 0 || !kv_len.is_multiple_of(head_dim) {
        return Err(AttentionError::InvalidKeyLength {
            expected: seq_len.saturating_mul(kv_len),
            actual: key_layer.len(),
        });
    }
    let kv_heads = kv_len / head_dim;
    if kv_head >= kv_heads {
        return Err(AttentionError::InvalidKvHead { kv_head, kv_heads });
    }

    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let kv_offset = kv_head * head_dim;

    // Online softmax with running accumulation.
    // See "Online normalizer calculation for softmax" (Milakov & Gimelshein, 2018)
    let mut running_max = f32::NEG_INFINITY;
    let mut running_sum = 0.0_f32;

    // First pass: process in cache-friendly blocks to find max score
    // and compute unnormalized attention output.
    output.fill(0.0);

    let mut token = 0_usize;
    while token < seq_len {
        let block_end = (token + FLASH_BLOCK_SIZE).min(seq_len);

        // Compute scores for this block
        for t in token..block_end {
            let row_off = t * kv_len + kv_offset;
            let key_row = &key_layer[row_off..row_off + head_dim];

            let mut score = E::dot(query, key_row);
            score *= scale;

            let new_max = running_max.max(score);
            let exp_factor = (running_max - new_max).exp();
            let exp_score = (score - new_max).exp();

            // Rescale accumulated output
            if exp_factor != 1.0 {
                for out in output.iter_mut() {
                    *out *= exp_factor;
                }
            }

            // Add weighted value
            let val_row_off = t * kv_len + kv_offset;
            let value_row = &value_layer[val_row_off..val_row_off + head_dim];
            E::axpy(output, exp_score, value_row);

            running_sum = running_sum * exp_factor + exp_score;
            running_max = new_max;
        }

        token = block_end;
    }

    // Normalize
    if running_sum > 0.0 {
        let inv_sum = 1.0 / running_sum;
        for out in output.iter_mut() {
            *out *= inv_sum;
        }
    }

    Ok(())
}

/// Parallel flash attention decode over multiple heads.
/// Each head is processed independently; heads are parallelized when seq_len
/// exceeds a threshold to amortize rayon overhead.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_decode_heads_f32(
    query_heads: &[f32],
    key_layer: &[f32],
    value_layer: &[f32],
    seq_len: usize,
    head_dim: usize,
    kv_len: usize,
    num_heads: usize,
    kv_heads: usize,
    output_heads: &mut [f32],
) -> Result<(), AttentionError> {
    flash_attention_decode_heads_impl(
        query_heads,
        key_layer,
        value_layer,
        seq_len,
        head_dim,
        kv_len,
        num_heads,
        kv_heads,
        output_heads,
    )
}

/// [`flash_attention_decode_heads_f32`] over f16-bit K/V (borrowed F16 cache).
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_decode_heads_f16(
    query_heads: &[f32],
    key_layer: &[u16],
    value_layer: &[u16],
    seq_len: usize,
    head_dim: usize,
    kv_len: usize,
    num_heads: usize,
    kv_heads: usize,
    output_heads: &mut [f32],
) -> Result<(), AttentionError> {
    flash_attention_decode_heads_impl(
        query_heads,
        key_layer,
        value_layer,
        seq_len,
        head_dim,
        kv_len,
        num_heads,
        kv_heads,
        output_heads,
    )
}

#[allow(clippy::too_many_arguments)]
fn flash_attention_decode_heads_impl<E: KvElem>(
    query_heads: &[f32],
    key_layer: &[E],
    value_layer: &[E],
    seq_len: usize,
    head_dim: usize,
    kv_len: usize,
    num_heads: usize,
    kv_heads: usize,
    output_heads: &mut [f32],
) -> Result<(), AttentionError> {
    let q_len = num_heads * head_dim;
    if query_heads.len() != q_len {
        return Err(AttentionError::InvalidQueryLength {
            expected: q_len,
            actual: query_heads.len(),
        });
    }
    let expected_kv = seq_len.saturating_mul(kv_len);
    if key_layer.len() != expected_kv {
        return Err(AttentionError::InvalidKeyLength {
            expected: expected_kv,
            actual: key_layer.len(),
        });
    }
    if value_layer.len() != expected_kv {
        return Err(AttentionError::InvalidValueLength {
            expected: expected_kv,
            actual: value_layer.len(),
        });
    }
    if output_heads.len() != q_len {
        return Err(AttentionError::InvalidOutputLength {
            expected: q_len,
            actual: output_heads.len(),
        });
    }

    if head_dim == 0 {
        return Err(AttentionError::ZeroHeadDim);
    }
    if kv_heads == 0 || !num_heads.is_multiple_of(kv_heads) {
        return Err(AttentionError::InvalidHeadGrouping {
            num_heads,
            kv_heads,
        });
    }
    let group_size = num_heads / kv_heads;

    // Parallelize over heads when the sequence is long enough to justify overhead.
    let use_parallel = seq_len >= PARALLEL_FLASH_ATTN_MIN_SEQ_LEN && num_heads > 1;

    if use_parallel {
        // Dispatch heads through run_chunks (spin pool when enabled) rather
        // than a raw rayon region: decode interleaves these head regions with
        // the GEMV regions, and mixing two dispatch mechanisms leaves one
        // pool's workers waking (or spinning) against the other's.
        let error: std::sync::Mutex<Option<AttentionError>> = std::sync::Mutex::new(None);
        let out_base = output_heads.as_mut_ptr() as usize;
        crate::spinpool::run_chunks(num_heads, |head| {
            // Safety: each head owns a disjoint output slice; the buffer
            // outlives the region.
            let out_head = unsafe {
                std::slice::from_raw_parts_mut(
                    (out_base as *mut f32).add(head * head_dim),
                    head_dim,
                )
            };
            let kv_head = head / group_size;
            let q_head = &query_heads[head * head_dim..(head + 1) * head_dim];
            if let Err(e) = flash_attention_decode_impl(
                q_head,
                key_layer,
                value_layer,
                seq_len,
                head_dim,
                kv_len,
                kv_head,
                out_head,
            ) && let Ok(mut slot) = error.lock()
            {
                slot.get_or_insert(e);
            }
        });
        if let Some(e) = error.into_inner().unwrap_or(None) {
            return Err(e);
        }
    } else {
        for head in 0..num_heads {
            let kv_head = head / group_size;
            let q_head = &query_heads[head * head_dim..(head + 1) * head_dim];
            let out_head = &mut output_heads[head * head_dim..(head + 1) * head_dim];
            flash_attention_decode_impl(
                q_head,
                key_layer,
                value_layer,
                seq_len,
                head_dim,
                kv_len,
                kv_head,
                out_head,
            )?;
        }
    }

    Ok(())
}

/// Prefill-phase flash attention: many queries attend to many keys/values.
///
/// `query` is `[q_seq_len][head_dim]`, `key`/`value` are `[kv_seq_len][head_dim]`.
/// `output` is `[q_seq_len][head_dim]`.
pub fn flash_attention_prefill_f32(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    q_seq_len: usize,
    kv_seq_len: usize,
    head_dim: usize,
    output: &mut [f32],
) -> Result<(), AttentionError> {
    if query.len() != q_seq_len * head_dim {
        return Err(AttentionError::InvalidQueryLength {
            expected: q_seq_len * head_dim,
            actual: query.len(),
        });
    }
    if key.len() != kv_seq_len * head_dim {
        return Err(AttentionError::InvalidKeyLength {
            expected: kv_seq_len * head_dim,
            actual: key.len(),
        });
    }
    if value.len() != kv_seq_len * head_dim {
        return Err(AttentionError::InvalidValueLength {
            expected: kv_seq_len * head_dim,
            actual: value.len(),
        });
    }
    if output.len() != q_seq_len * head_dim {
        return Err(AttentionError::InvalidOutputLength {
            expected: q_seq_len * head_dim,
            actual: output.len(),
        });
    }

    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    for q_i in 0..q_seq_len {
        let q_off = q_i * head_dim;
        let q_vec = &query[q_off..q_off + head_dim];
        let out_off = q_i * head_dim;
        let out_vec = &mut output[out_off..out_off + head_dim];

        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0_f32;
        out_vec.fill(0.0);

        let mut token = 0_usize;
        while token < kv_seq_len {
            let block_end = (token + FLASH_BLOCK_SIZE).min(kv_seq_len);

            for t in token..block_end {
                let k_off = t * head_dim;
                let key_row = &key[k_off..k_off + head_dim];

                let mut score = dot_product_f32(q_vec, key_row);
                score *= scale;

                let new_max = running_max.max(score);
                let exp_factor = (running_max - new_max).exp();
                let exp_score = (score - new_max).exp();

                if exp_factor != 1.0 {
                    for out in out_vec.iter_mut() {
                        *out *= exp_factor;
                    }
                }

                let v_off = t * head_dim;
                let value_row = &value[v_off..v_off + head_dim];
                for (out, v) in out_vec.iter_mut().zip(value_row.iter()) {
                    *out += exp_score * v;
                }

                running_sum = running_sum * exp_factor + exp_score;
                running_max = new_max;
            }

            token = block_end;
        }

        if running_sum > 0.0 {
            let inv_sum = 1.0 / running_sum;
            for out in out_vec.iter_mut() {
                *out *= inv_sum;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The f16 K/V decode path must match the f32 path within half-precision
    /// rounding (the only difference is each K/V element passing through f16).
    #[test]
    fn decode_heads_f16_matches_f32() {
        let (seq_len, head_dim, num_heads, kv_heads) = (37_usize, 64_usize, 4_usize, 2_usize);
        let kv_len = kv_heads * head_dim;
        let kv: Vec<f32> = (0..seq_len * kv_len)
            .map(|i| ((i as f32) * 0.013).sin() * 0.5)
            .collect();
        let vv: Vec<f32> = (0..seq_len * kv_len)
            .map(|i| ((i as f32) * 0.007).cos() * 0.5)
            .collect();
        let query: Vec<f32> = (0..num_heads * head_dim)
            .map(|i| ((i as f32) * 0.011).sin())
            .collect();
        let k16: Vec<u16> = kv
            .iter()
            .map(|&v| crate::kv_cache::f32_to_f16_bits(v))
            .collect();
        let v16: Vec<u16> = vv
            .iter()
            .map(|&v| crate::kv_cache::f32_to_f16_bits(v))
            .collect();
        // Reference over the f16-rounded values so only kernel differences count.
        let k_r: Vec<f32> = k16
            .iter()
            .map(|&b| crate::tensor::f16_bits_to_f32(b))
            .collect();
        let v_r: Vec<f32> = v16
            .iter()
            .map(|&b| crate::tensor::f16_bits_to_f32(b))
            .collect();

        let mut out_f32 = vec![0.0_f32; num_heads * head_dim];
        flash_attention_decode_heads_f32(
            &query,
            &k_r,
            &v_r,
            seq_len,
            head_dim,
            kv_len,
            num_heads,
            kv_heads,
            &mut out_f32,
        )
        .unwrap();
        let mut out_f16 = vec![0.0_f32; num_heads * head_dim];
        flash_attention_decode_heads_f16(
            &query,
            &k16,
            &v16,
            seq_len,
            head_dim,
            kv_len,
            num_heads,
            kv_heads,
            &mut out_f16,
        )
        .unwrap();
        for (i, (a, b)) in out_f32.iter().zip(&out_f16).enumerate() {
            assert!(
                (a - b).abs() <= 1e-5 + a.abs() * 1e-4,
                "lane {i}: f32 {a} vs f16 {b}"
            );
        }
    }

    fn reference_attention_decode(
        query: &[f32],
        key_layer: &[f32],
        value_layer: &[f32],
        seq_len: usize,
        head_dim: usize,
        kv_len: usize,
        kv_head: usize,
    ) -> Vec<f32> {
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        let kv_offset = kv_head * head_dim;

        let mut scores = vec![0.0_f32; seq_len];
        for (t, score) in scores.iter_mut().enumerate().take(seq_len) {
            let row_off = t * kv_len + kv_offset;
            let key_row = &key_layer[row_off..row_off + head_dim];
            let mut dot = 0.0_f32;
            for (q, k) in query.iter().zip(key_row.iter()) {
                dot += q * k;
            }
            *score = dot * scale;
        }

        let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        for s in scores.iter_mut() {
            *s = (*s - max_score).exp();
        }
        let sum: f32 = scores.iter().sum();
        for s in scores.iter_mut() {
            *s /= sum;
        }

        let mut output = vec![0.0_f32; head_dim];
        for (t, score) in scores.iter().enumerate().take(seq_len) {
            let val_row_off = t * kv_len + kv_offset;
            let value_row = &value_layer[val_row_off..val_row_off + head_dim];
            for (out, v) in output.iter_mut().zip(value_row.iter()) {
                *out += score * v;
            }
        }
        output
    }

    #[test]
    fn flash_decode_rejects_out_of_range_kv_head() {
        let head_dim = 4;
        let kv_len = 8;
        let seq_len = 2;
        let query = vec![0.0_f32; head_dim];
        let key_layer = vec![0.0_f32; seq_len * kv_len];
        let value_layer = vec![0.0_f32; seq_len * kv_len];
        let mut output = vec![0.0_f32; head_dim];

        let err = flash_attention_decode_f32(
            &query,
            &key_layer,
            &value_layer,
            seq_len,
            head_dim,
            kv_len,
            2,
            &mut output,
        )
        .expect_err("kv_head beyond kv_heads should fail");
        assert!(matches!(
            err,
            AttentionError::InvalidKvHead {
                kv_head: 2,
                kv_heads: 2
            }
        ));
    }

    #[test]
    fn flash_decode_heads_rejects_non_divisible_grouping() {
        let head_dim = 4;
        let kv_len = 8;
        let seq_len = 2;
        let query_heads = vec![0.0_f32; 5 * head_dim];
        let key_layer = vec![0.0_f32; seq_len * kv_len];
        let value_layer = vec![0.0_f32; seq_len * kv_len];
        let mut output = vec![0.0_f32; 5 * head_dim];

        let err = flash_attention_decode_heads_f32(
            &query_heads,
            &key_layer,
            &value_layer,
            seq_len,
            head_dim,
            kv_len,
            5,
            2,
            &mut output,
        )
        .expect_err("non-divisible num_heads/kv_heads should fail");
        assert!(matches!(
            err,
            AttentionError::InvalidHeadGrouping {
                num_heads: 5,
                kv_heads: 2
            }
        ));
    }

    #[test]
    fn flash_decode_matches_reference() {
        let head_dim = 4;
        let kv_heads = 2;
        let kv_len = kv_heads * head_dim;
        let seq_len = 5;

        let query = vec![0.3_f32, -0.8, 1.1, 0.2];
        let key_layer: Vec<f32> = (0..seq_len * kv_len)
            .map(|i| ((i as f32 * 0.07).cos() * 1.3) - 0.2)
            .collect();
        let value_layer: Vec<f32> = (0..seq_len * kv_len)
            .map(|i| ((i as f32 * 0.13).sin() * 0.9) + 0.1)
            .collect();

        for kv_head in 0..kv_heads {
            let expected = reference_attention_decode(
                &query,
                &key_layer,
                &value_layer,
                seq_len,
                head_dim,
                kv_len,
                kv_head,
            );
            let mut actual = vec![0.0_f32; head_dim];
            flash_attention_decode_f32(
                &query,
                &key_layer,
                &value_layer,
                seq_len,
                head_dim,
                kv_len,
                kv_head,
                &mut actual,
            )
            .expect("flash decode should succeed");

            for (a, e) in actual.iter().zip(expected.iter()) {
                assert!(
                    (a - e).abs() < 1e-5,
                    "head {} mismatch: {} vs {}",
                    kv_head,
                    a,
                    e
                );
            }
        }
    }

    #[test]
    fn flash_decode_handles_long_sequence() {
        let head_dim = 8;
        let kv_heads = 4;
        let kv_len = kv_heads * head_dim;
        let seq_len = 513; // cross block boundary

        let query: Vec<f32> = (0..head_dim)
            .map(|i| (i as f32 * 0.13).sin() - 0.25)
            .collect();
        let key_layer: Vec<f32> = (0..seq_len * kv_len)
            .map(|i| ((i as f32 * 0.017).cos() * 1.3) - 0.2)
            .collect();
        let value_layer: Vec<f32> = (0..seq_len * kv_len)
            .map(|i| ((i as f32 * 0.031).sin() * 0.9) + 0.1)
            .collect();

        for kv_head in 0..kv_heads {
            let expected = reference_attention_decode(
                &query,
                &key_layer,
                &value_layer,
                seq_len,
                head_dim,
                kv_len,
                kv_head,
            );
            let mut actual = vec![0.0_f32; head_dim];
            flash_attention_decode_f32(
                &query,
                &key_layer,
                &value_layer,
                seq_len,
                head_dim,
                kv_len,
                kv_head,
                &mut actual,
            )
            .expect("flash decode long seq should succeed");

            for (a, e) in actual.iter().zip(expected.iter()) {
                assert!(
                    (a - e).abs() < 1e-4,
                    "head {} mismatch: {} vs {}",
                    kv_head,
                    a,
                    e
                );
            }
        }
    }

    #[test]
    fn flash_decode_heads_matches_single_head_reference() {
        let head_dim = 4;
        let kv_heads = 2;
        let num_heads = 4;
        let kv_len = kv_heads * head_dim;
        let seq_len = 5;

        let query_heads: Vec<f32> = (0..num_heads * head_dim)
            .map(|i| ((i as f32 * 0.07).cos() * 1.3) - 0.2)
            .collect();
        let key_layer: Vec<f32> = (0..seq_len * kv_len)
            .map(|i| ((i as f32 * 0.07).cos() * 1.3) - 0.2)
            .collect();
        let value_layer: Vec<f32> = (0..seq_len * kv_len)
            .map(|i| ((i as f32 * 0.13).sin() * 0.9) + 0.1)
            .collect();

        let mut actual = vec![0.0_f32; num_heads * head_dim];
        flash_attention_decode_heads_f32(
            &query_heads,
            &key_layer,
            &value_layer,
            seq_len,
            head_dim,
            kv_len,
            num_heads,
            kv_heads,
            &mut actual,
        )
        .expect("flash decode heads should succeed");

        for head in 0..num_heads {
            let expected = reference_attention_decode(
                &query_heads[head * head_dim..(head + 1) * head_dim],
                &key_layer,
                &value_layer,
                seq_len,
                head_dim,
                kv_len,
                head / (num_heads / kv_heads),
            );
            for (a, e) in actual[head * head_dim..(head + 1) * head_dim]
                .iter()
                .zip(expected.iter())
            {
                assert!(
                    (a - e).abs() < 1e-5,
                    "head {} mismatch: {} vs {}",
                    head,
                    a,
                    e
                );
            }
        }
    }

    #[test]
    fn flash_prefill_matches_reference() {
        let head_dim = 4;
        let q_seq = 3;
        let kv_seq = 5;

        let query: Vec<f32> = (0..q_seq * head_dim)
            .map(|i| (i as f32 * 0.11).sin())
            .collect();
        let key: Vec<f32> = (0..kv_seq * head_dim)
            .map(|i| (i as f32 * 0.07).cos())
            .collect();
        let value: Vec<f32> = (0..kv_seq * head_dim)
            .map(|i| (i as f32 * 0.13).sin())
            .collect();

        let mut actual = vec![0.0_f32; q_seq * head_dim];
        flash_attention_prefill_f32(&query, &key, &value, q_seq, kv_seq, head_dim, &mut actual)
            .expect("flash prefill should succeed");

        // Compute reference per-query
        for q_i in 0..q_seq {
            let q_off = q_i * head_dim;
            let q_vec = &query[q_off..q_off + head_dim];
            let mut expected = vec![0.0_f32; head_dim];

            let scale = 1.0_f32 / (head_dim as f32).sqrt();
            let mut scores = vec![0.0_f32; kv_seq];
            for (t, score) in scores.iter_mut().enumerate().take(kv_seq) {
                let k_off = t * head_dim;
                let mut dot = 0.0_f32;
                for (q, k) in q_vec.iter().zip(key[k_off..k_off + head_dim].iter()) {
                    dot += q * k;
                }
                *score = dot * scale;
            }
            let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
            }
            let sum: f32 = scores.iter().sum();
            for s in scores.iter_mut() {
                *s /= sum;
            }
            for (t, score) in scores.iter().enumerate().take(kv_seq) {
                let v_off = t * head_dim;
                for (out, v) in expected
                    .iter_mut()
                    .zip(value[v_off..v_off + head_dim].iter())
                {
                    *out += score * v;
                }
            }

            let out_off = q_i * head_dim;
            for (a, e) in actual[out_off..out_off + head_dim]
                .iter()
                .zip(expected.iter())
            {
                assert!((a - e).abs() < 1e-5, "q_i {} mismatch: {} vs {}", q_i, a, e);
            }
        }
    }
}
