use super::*;

pub fn scaled_dot_product_attention_f32(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    seq_len: usize,
    dim: usize,
    output: &mut [f32],
) -> Result<(), AttentionError> {
    if query.len() != dim {
        return Err(AttentionError::InvalidQueryLength {
            expected: dim,
            actual: query.len(),
        });
    }

    let expected_kv_len = seq_len.saturating_mul(dim);
    if key.len() != expected_kv_len {
        return Err(AttentionError::InvalidKeyLength {
            expected: expected_kv_len,
            actual: key.len(),
        });
    }
    if value.len() != expected_kv_len {
        return Err(AttentionError::InvalidValueLength {
            expected: expected_kv_len,
            actual: value.len(),
        });
    }
    if output.len() != dim {
        return Err(AttentionError::InvalidOutputLength {
            expected: dim,
            actual: output.len(),
        });
    }

    output.fill(0.0);
    if seq_len == 0 {
        return Ok(());
    }

    let scale = 1.0_f32 / (dim as f32).sqrt();
    let mut running_max = f32::NEG_INFINITY;
    let mut running_sum = 0.0_f32;
    let mut token_offset = 0_usize;
    let mut block_acc = vec![0.0_f32; dim];
    let mut block_scores = vec![0.0_f32; FLASH_ATTENTION_BLOCK_TOKENS];

    while token_offset < seq_len {
        let block_len = (seq_len - token_offset).min(FLASH_ATTENTION_BLOCK_TOKENS);
        let block_start = token_offset * dim;
        let block_end = block_start + block_len * dim;
        let key_block = &key[block_start..block_end];
        let value_block = &value[block_start..block_end];

        let mut block_max = f32::NEG_INFINITY;
        for (idx, key_row) in key_block.chunks_exact(dim).enumerate() {
            let score = query
                .iter()
                .zip(key_row.iter())
                .map(|(q, k)| q * k)
                .sum::<f32>()
                * scale;
            block_scores[idx] = score;
            block_max = block_max.max(score);
        }

        block_acc.fill(0.0);
        let mut block_sum = 0.0_f32;
        for (idx, value_row) in value_block.chunks_exact(dim).enumerate() {
            let score = block_scores[idx];
            let weight = (score - block_max).exp();
            block_sum += weight;
            for (acc, v) in block_acc.iter_mut().zip(value_row.iter()) {
                *acc += weight * v;
            }
        }

        let merged_max = running_max.max(block_max);
        let running_scale = (running_max - merged_max).exp();
        let block_scale = (block_max - merged_max).exp();
        for (out, acc) in output.iter_mut().zip(block_acc.iter()) {
            *out = *out * running_scale + acc * block_scale;
        }
        running_sum = running_sum * running_scale + block_sum * block_scale;
        running_max = merged_max;
        token_offset += block_len;
    }

    let inv_sum = 1.0_f32 / running_sum;
    for out in output.iter_mut() {
        *out *= inv_sum;
    }

    Ok(())
}

/// Apply rotary position embedding in NeoX/HF style (split-half), matching
/// llama.cpp's `GGML_ROPE_TYPE_NEOX` mode used by Llama, Mistral, Qwen, Gemma,
/// Phi and most other modern decoder-only transformers. Pairs index `i` with
/// index `head_dim/2 + i` rather than `2i` with `2i+1` (which is the GPT-J /
/// "norm" mode and produces garbage on NeoX-trained weights).
pub fn apply_rope_f32(
    input: &[f32],
    position: usize,
    head_dim: usize,
    theta: f32,
    output: &mut [f32],
) -> Result<(), RopeError> {
    if input.len() != head_dim {
        return Err(RopeError::InvalidInputLength {
            expected: head_dim,
            actual: input.len(),
        });
    }
    if output.len() != head_dim {
        return Err(RopeError::InvalidOutputLength {
            expected: head_dim,
            actual: output.len(),
        });
    }
    if !head_dim.is_multiple_of(2) {
        return Err(RopeError::OddHeadDim { head_dim });
    }

    if position == 0 {
        output.copy_from_slice(input);
        return Ok(());
    }

    let position_f = position as f32;
    let half_dim = head_dim / 2;
    let inv_head_dim = 1.0_f32 / head_dim as f32;
    // freq_i = theta^(-(2*i)/head_dim). Computing `powf` for every pair is
    // surprisingly expensive in prompt processing (tokens × heads × layers).
    // Use the geometric recurrence instead: freq_{i+1} = freq_i * base.
    let freq_multiplier = theta.powf(-2.0 * inv_head_dim);
    let mut freq = 1.0_f32;

    for i in 0..half_dim {
        let x0 = input[i];
        let x1 = input[half_dim + i];
        let angle = position_f * freq;
        let cos_angle = angle.cos();
        let sin_angle = angle.sin();

        output[i] = x0 * cos_angle - x1 * sin_angle;
        output[half_dim + i] = x0 * sin_angle + x1 * cos_angle;
        freq *= freq_multiplier;
    }

    Ok(())
}

/// AVX2 SwiGLU kernel. Uses a bit-trick fast-exp approximation for sigmoid
/// (~1% error vs true sigmoid, negligible for inference quality).
///
/// sigmoid(x) ≈ 1/(1+exp(-x)) where exp(-x) ≈ 2^(-x·log₂e) via IEEE 754 cast.
/// Followed by one Newton-Raphson step on the reciprocal for better precision.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(super) unsafe fn swiglu_avx2(gate: &[f32], up: &[f32], output: &mut [f32]) {
    let n = output.len();
    let chunks = n / 8;

    // SAFETY: All AVX2 intrinsics are unsafe; we are inside an `unsafe fn`
    // that is only called after runtime feature detection confirms AVX2+FMA.
    unsafe {
        let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
        let scale23 = _mm256_set1_ps((1u32 << 23) as f32);
        let _bias23 = _mm256_set1_ps((127u32 << 23) as f32);
        let one = _mm256_set1_ps(1.0_f32);
        let two = _mm256_set1_ps(2.0_f32);
        let zero = _mm256_setzero_ps();
        let clamp_hi = _mm256_set1_ps(12.0_f32);
        let clamp_lo = _mm256_set1_ps(-12.0_f32);

        for i in 0..chunks {
            let g = _mm256_loadu_ps(gate[i * 8..].as_ptr());
            let u = _mm256_loadu_ps(up[i * 8..].as_ptr());

            // neg_g clamped to [-12, 12] to keep 2^(neg_g·log2e) in valid i32 range
            let neg_g = _mm256_sub_ps(zero, g);
            let neg_g = _mm256_max_ps(_mm256_min_ps(neg_g, clamp_hi), clamp_lo);

            // exp(-g) ≈ 2^(neg_g·log₂e): multiply, add the float-domain bias 127·2²³,
            // then reinterpret the bits as f32 (standard Schraudolph approximation).
            let exp_arg = _mm256_fmadd_ps(neg_g, log2e, _mm256_set1_ps(127.0_f32));
            let exp_bits = _mm256_cvtps_epi32(_mm256_mul_ps(exp_arg, scale23));
            let exp_neg_g = _mm256_castsi256_ps(exp_bits);

            // sigmoid = 1 / (1 + exp(-g))
            let denom = _mm256_add_ps(one, exp_neg_g);
            // rcp + one Newton-Raphson step: r2 = r·(2 - denom·r)
            let rcp = _mm256_rcp_ps(denom);
            let sigmoid = _mm256_mul_ps(rcp, _mm256_fnmadd_ps(denom, rcp, two));

            // silu(g)·up = g·sigmoid(g)·up
            let result = _mm256_mul_ps(_mm256_mul_ps(g, sigmoid), u);
            _mm256_storeu_ps(output[i * 8..].as_mut_ptr(), result);
        }
    }

    // Scalar tail for remainder elements
    for i in (chunks * 8)..n {
        let g = gate[i];
        let sigmoid = 1.0_f32 / (1.0_f32 + (-g).exp());
        output[i] = g * sigmoid * up[i];
    }
}

/// SwiGLU written back into `gate`: `gate[i] = gate[i] * sigmoid(gate[i]) * up[i]`.
/// Uses an AVX2 inplace fast path when available to avoid scalar `exp()` overhead.
pub fn apply_swiglu_inplace_f32(gate: &mut [f32], up: &[f32]) {
    let n = gate.len().min(up.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: gate_ptr and up_ptr point to non-overlapping slices (distinct
        // allocations).  Within each iteration we load 8 gate values into a
        // register, compute silu, then store — no aliased concurrent access.
        unsafe { swiglu_avx2_inplace(gate.as_mut_ptr(), up.as_ptr(), n) };
        return;
    }
    for i in 0..n {
        let g = gate[i];
        let sigmoid = 1.0_f32 / (1.0_f32 + (-g).exp());
        gate[i] = g * sigmoid * up[i];
    }
}

/// AVX2 SwiGLU inplace: reads gate[i], computes silu(gate[i])*up[i], writes back.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(super) unsafe fn swiglu_avx2_inplace(g_ptr: *mut f32, u_ptr: *const f32, n: usize) {
    let chunks = n / 8;
    // SAFETY: All AVX2 intrinsics are unsafe; we are inside an `unsafe fn`
    // that is only called after runtime feature detection confirms AVX2+FMA.
    unsafe {
        let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
        let scale23 = _mm256_set1_ps((1u32 << 23) as f32);
        let one = _mm256_set1_ps(1.0_f32);
        let two = _mm256_set1_ps(2.0_f32);
        let zero = _mm256_setzero_ps();
        let clamp_hi = _mm256_set1_ps(12.0_f32);
        let clamp_lo = _mm256_set1_ps(-12.0_f32);

        for i in 0..chunks {
            let off = i * 8;
            let g = _mm256_loadu_ps(g_ptr.add(off));
            let u = _mm256_loadu_ps(u_ptr.add(off));

            let neg_g = _mm256_sub_ps(zero, g);
            let neg_g = _mm256_max_ps(_mm256_min_ps(neg_g, clamp_hi), clamp_lo);
            let exp_arg = _mm256_fmadd_ps(neg_g, log2e, _mm256_set1_ps(127.0_f32));
            let exp_bits = _mm256_cvtps_epi32(_mm256_mul_ps(exp_arg, scale23));
            let exp_neg_g = _mm256_castsi256_ps(exp_bits);
            let denom = _mm256_add_ps(one, exp_neg_g);
            let rcp = _mm256_rcp_ps(denom);
            let sigmoid = _mm256_mul_ps(rcp, _mm256_fnmadd_ps(denom, rcp, two));
            let result = _mm256_mul_ps(_mm256_mul_ps(g, sigmoid), u);
            _mm256_storeu_ps(g_ptr.add(off), result);
        }
        for i in (chunks * 8)..n {
            let g = *g_ptr.add(i);
            let sigmoid = 1.0_f32 / (1.0_f32 + (-g).exp());
            *g_ptr.add(i) = g * sigmoid * *u_ptr.add(i);
        }
    }
}

/// GeGLU written back into `gate`: `gate[i] = gelu(gate[i]) * up[i]` using the
/// tanh GELU approximation (matches Gemma's `gelu_pytorch_tanh` activation).
/// Scalar implementation — Gemma FFN width is modest and this is not the
/// dominant cost. Kept separate from SwiGLU so SiLU models are untouched.
pub fn apply_geglu_inplace_f32(gate: &mut [f32], up: &[f32]) {
    const K: f32 = 0.797_884_6_f32; // sqrt(2/pi)
    let n = gate.len().min(up.len());
    for i in 0..n {
        let g = gate[i];
        let gelu = 0.5 * g * (1.0 + (K * (g + 0.044_715 * g * g * g)).tanh());
        gate[i] = gelu * up[i];
    }
}

pub fn apply_swiglu_f32(gate: &[f32], up: &[f32], output: &mut [f32]) -> Result<(), SwiGluError> {
    let expected_len = output.len();
    if gate.len() != expected_len {
        return Err(SwiGluError::InvalidGateLength {
            expected: expected_len,
            actual: gate.len(),
        });
    }
    if up.len() != expected_len {
        return Err(SwiGluError::InvalidUpLength {
            expected: expected_len,
            actual: up.len(),
        });
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { swiglu_avx2(gate, up, output) };
        return Ok(());
    }

    for ((gate_value, up_value), out) in gate.iter().zip(up.iter()).zip(output.iter_mut()) {
        let sigmoid = 1.0_f32 / (1.0 + (-gate_value).exp());
        *out = gate_value * sigmoid * up_value;
    }

    Ok(())
}

pub fn rms_norm_f32(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    output: &mut [f32],
) -> Result<(), RmsNormError> {
    let hidden_dim = output.len();
    if hidden_dim == 0 {
        return Err(RmsNormError::ZeroDimension);
    }
    if input.len() != hidden_dim {
        return Err(RmsNormError::InvalidInputLength {
            expected: hidden_dim,
            actual: input.len(),
        });
    }
    if weight.len() != hidden_dim {
        return Err(RmsNormError::InvalidWeightLength {
            expected: hidden_dim,
            actual: weight.len(),
        });
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { rms_norm_f32_avx2(input, weight, eps, output) };
            return Ok(());
        }
    }

    let sum_sq = input.iter().map(|value| value * value).sum::<f32>();
    let mean_sq = sum_sq / hidden_dim as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();

    for ((value, scale), out) in input.iter().zip(weight.iter()).zip(output.iter_mut()) {
        *out = value * inv_rms * scale;
    }
    Ok(())
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(super) unsafe fn rms_norm_f32_avx2(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    output: &mut [f32],
) {
    let len = output.len();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut i = 0;

    while i + 32 <= len {
        let v0 = _mm256_loadu_ps(input.as_ptr().add(i));
        let v1 = _mm256_loadu_ps(input.as_ptr().add(i + 8));
        let v2 = _mm256_loadu_ps(input.as_ptr().add(i + 16));
        let v3 = _mm256_loadu_ps(input.as_ptr().add(i + 24));
        acc0 = _mm256_fmadd_ps(v0, v0, acc0);
        acc1 = _mm256_fmadd_ps(v1, v1, acc1);
        acc2 = _mm256_fmadd_ps(v2, v2, acc2);
        acc3 = _mm256_fmadd_ps(v3, v3, acc3);
        i += 32;
    }
    while i + 8 <= len {
        let v = _mm256_loadu_ps(input.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(v, v, acc0);
        i += 8;
    }

    let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let mut sum_sq = _mm_cvtss_f32(_mm_add_ss(sums, shuf2));

    while i < len {
        let v = *input.get_unchecked(i);
        sum_sq += v * v;
        i += 1;
    }

    let inv_rms = 1.0_f32 / (sum_sq / len as f32 + eps).sqrt();
    let inv = _mm256_set1_ps(inv_rms);
    let mut j = 0;
    while j + 32 <= len {
        let v0 = _mm256_loadu_ps(input.as_ptr().add(j));
        let w0 = _mm256_loadu_ps(weight.as_ptr().add(j));
        let v1 = _mm256_loadu_ps(input.as_ptr().add(j + 8));
        let w1 = _mm256_loadu_ps(weight.as_ptr().add(j + 8));
        let v2 = _mm256_loadu_ps(input.as_ptr().add(j + 16));
        let w2 = _mm256_loadu_ps(weight.as_ptr().add(j + 16));
        let v3 = _mm256_loadu_ps(input.as_ptr().add(j + 24));
        let w3 = _mm256_loadu_ps(weight.as_ptr().add(j + 24));
        _mm256_storeu_ps(
            output.as_mut_ptr().add(j),
            _mm256_mul_ps(_mm256_mul_ps(v0, inv), w0),
        );
        _mm256_storeu_ps(
            output.as_mut_ptr().add(j + 8),
            _mm256_mul_ps(_mm256_mul_ps(v1, inv), w1),
        );
        _mm256_storeu_ps(
            output.as_mut_ptr().add(j + 16),
            _mm256_mul_ps(_mm256_mul_ps(v2, inv), w2),
        );
        _mm256_storeu_ps(
            output.as_mut_ptr().add(j + 24),
            _mm256_mul_ps(_mm256_mul_ps(v3, inv), w3),
        );
        j += 32;
    }
    while j + 8 <= len {
        let v = _mm256_loadu_ps(input.as_ptr().add(j));
        let w = _mm256_loadu_ps(weight.as_ptr().add(j));
        _mm256_storeu_ps(
            output.as_mut_ptr().add(j),
            _mm256_mul_ps(_mm256_mul_ps(v, inv), w),
        );
        j += 8;
    }
    while j < len {
        *output.get_unchecked_mut(j) = *input.get_unchecked(j) * inv_rms * *weight.get_unchecked(j);
        j += 1;
    }
}

/// Fused RMS-normalization + transposed GEMV for the attention Q projection.
/// Computes RMSNorm of `input`, then performs transposed GEMV using the
/// normalized vector as the GEMV input.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_gemv_f32_transposed(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    matrix: &[f32],
    rows: usize,
    cols: usize,
    output: &mut [f32],
) -> Result<(), RmsNormError> {
    if input.len() != rows {
        return Err(RmsNormError::InvalidInputLength {
            expected: rows,
            actual: input.len(),
        });
    }
    if weight.len() != rows {
        return Err(RmsNormError::InvalidWeightLength {
            expected: rows,
            actual: weight.len(),
        });
    }
    if matrix.len() != rows * cols {
        return Err(RmsNormError::InvalidOutputLength {
            expected: rows * cols,
            actual: matrix.len(),
        });
    }
    if output.len() != cols {
        return Err(RmsNormError::InvalidOutputLength {
            expected: cols,
            actual: output.len(),
        });
    }
    if rows == 0 {
        return Err(RmsNormError::ZeroDimension);
    }

    let sum_sq = input.iter().map(|v| v * v).sum::<f32>();
    let mean_sq = sum_sq / rows as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();

    output.fill(0.0);
    for (i, (value, scale)) in input.iter().zip(weight.iter()).enumerate() {
        let scaled = value * inv_rms * scale;
        let row = &matrix[i * cols..(i + 1) * cols];
        for (j, &mat_val) in row.iter().enumerate() {
            output[j] += mat_val * scaled;
        }
    }
    Ok(())
}

pub fn layer_norm_f32(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    eps: f32,
    output: &mut [f32],
) -> Result<(), LayerNormError> {
    let hidden_dim = output.len();
    if input.len() != hidden_dim {
        return Err(LayerNormError::InvalidInputLength {
            expected: hidden_dim,
            actual: input.len(),
        });
    }
    if weight.len() != hidden_dim {
        return Err(LayerNormError::InvalidWeightLength {
            expected: hidden_dim,
            actual: weight.len(),
        });
    }
    if bias.len() != hidden_dim {
        return Err(LayerNormError::InvalidBiasLength {
            expected: hidden_dim,
            actual: bias.len(),
        });
    }

    let mean = input.iter().sum::<f32>() / hidden_dim as f32;
    let variance = input
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f32>()
        / hidden_dim as f32;
    let inv_std = 1.0 / (variance + eps).sqrt();

    for (((value, scale), shift), out) in input
        .iter()
        .zip(weight.iter())
        .zip(bias.iter())
        .zip(output.iter_mut())
    {
        *out = (value - mean) * inv_std * scale + shift;
    }
    Ok(())
}

pub fn softmax_f32(input: &[f32], output: &mut [f32]) -> Result<(), SoftmaxError> {
    let expected_len = output.len();
    if input.len() != expected_len {
        return Err(SoftmaxError::InvalidInputLength {
            expected: expected_len,
            actual: input.len(),
        });
    }

    let max_input = input.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    let mut sum_exp = 0.0_f64;
    for (x, out) in input.iter().zip(output.iter_mut()) {
        let exp_value = (x - max_input).exp();
        *out = exp_value;
        sum_exp += exp_value as f64;
    }

    let inv_sum = (1.0_f64 / sum_exp) as f32;
    for out in output.iter_mut() {
        *out *= inv_sum;
    }

    Ok(())
}
