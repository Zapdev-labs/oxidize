use crate::tensor::AttentionError;

const FLASH_BLOCK_SIZE: usize = 64;

/// Compute dot product of two equal-length f32 slices.
/// Uses AVX2 when available on x86-64, otherwise falls back to scalar.
#[inline]
fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return unsafe { dot_product_f32_avx2(a, b) };
    }

    let mut sum = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        sum += x * y;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let mut sum = _mm256_setzero_ps();

    let chunks = len / 8;
    for i in 0..chunks {
        let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    // Horizontal sum of 8 floats
    let mut result = [0.0_f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), sum);
    let mut total = result.iter().sum::<f32>();

    // Tail
    for i in (chunks * 8)..len {
        total += a.get_unchecked(i) * b.get_unchecked(i);
    }

    total
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

    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let kv_offset = kv_head.saturating_mul(head_dim);

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

            let mut score = dot_product_f32(query, key_row);
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
            for (out, v) in output.iter_mut().zip(value_row.iter()) {
                *out += exp_score * v;
            }

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
        for t in 0..seq_len {
            let row_off = t * kv_len + kv_offset;
            let key_row = &key_layer[row_off..row_off + head_dim];
            let mut dot = 0.0_f32;
            for (q, k) in query.iter().zip(key_row.iter()) {
                dot += q * k;
            }
            scores[t] = dot * scale;
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
        for t in 0..seq_len {
            let val_row_off = t * kv_len + kv_offset;
            let value_row = &value_layer[val_row_off..val_row_off + head_dim];
            for (out, v) in output.iter_mut().zip(value_row.iter()) {
                *out += scores[t] * v;
            }
        }
        output
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
                &query, &key_layer, &value_layer, seq_len, head_dim, kv_len, kv_head,
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
                assert!((a - e).abs() < 1e-5, "head {} mismatch: {} vs {}", kv_head, a, e);
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
                &query, &key_layer, &value_layer, seq_len, head_dim, kv_len, kv_head,
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
                assert!((a - e).abs() < 1e-4, "head {} mismatch: {} vs {}", kv_head, a, e);
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
            for t in 0..kv_seq {
                let k_off = t * head_dim;
                let mut dot = 0.0_f32;
                for (q, k) in q_vec.iter().zip(key[k_off..k_off + head_dim].iter()) {
                    dot += q * k;
                }
                scores[t] = dot * scale;
            }
            let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
            }
            let sum: f32 = scores.iter().sum();
            for s in scores.iter_mut() {
                *s /= sum;
            }
            for t in 0..kv_seq {
                let v_off = t * head_dim;
                for (out, v) in expected.iter_mut().zip(value[v_off..v_off + head_dim].iter()) {
                    *out += scores[t] * v;
                }
            }

            let out_off = q_i * head_dim;
            for (a, e) in actual[out_off..out_off + head_dim].iter().zip(expected.iter()) {
                assert!((a - e).abs() < 1e-5, "q_i {} mismatch: {} vs {}", q_i, a, e);
            }
        }
    }
}
