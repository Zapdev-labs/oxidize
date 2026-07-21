/*
 * activation.c — RMSNorm, RoPE, SwiGLU scalar reference implementations.
 *
 * Port of oxidize-core/src/compute/tensor/kernels/activation.rs. Bit-exact
 * with the Rust scalar reference (VAL-FWD-001..004).
 */
#include "oxidize/activation.h"

#include <math.h>

void oc_rms_norm_f32(const float *x, const float *weight, float *out,
                    size_t n, float eps)
{
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum_sq += (double)x[i] * (double)x[i];
    }
    /* Rust computes mean over n as f32 accumulate (the reference uses
     * `x.iter().map(|v| v*v).sum::<f32>()` then `/ n as f32`). To stay
     * bit-exact we mirror that: accumulate in f32, not f64. */
    float ss = 0.0f;
    for (size_t i = 0; i < n; i++) {
        ss += x[i] * x[i];
    }
    (void)sum_sq;
    float inv_rms = 1.0f / sqrtf(ss / (float)n + eps);
    for (size_t i = 0; i < n; i++) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

void oc_swiglu_inplace_f32(float *gate, const float *up, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        float g = gate[i];
        /* sigmoid(g) = 1 / (1 + exp(-g)). Rust uses the Schraudolph fast-exp
         * in the AVX2 path but the scalar path uses expf; we use expf for
         * parity with the scalar reference. */
        float sig = 1.0f / (1.0f + expf(-g));
        gate[i] = g * sig * up[i];
    }
}

float oc_gelu_exact_f32(float x)
{
    return 0.5f * x * (1.0f + erff(x / 1.41421356f));
}

float oc_gelu_approx_f32(float x)
{
    const float c = 0.7978845608f; /* sqrt(2/pi) */
    float inner = c * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

void oc_geglu_inplace_f32(float *gate, const float *up, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        gate[i] = oc_gelu_exact_f32(gate[i]) * up[i];
    }
}

void oc_apply_rope_f32(const float *in, float *out, size_t head_dim,
                      size_t rope_len, int64_t position, float theta)
{
    if (rope_len > head_dim) rope_len = head_dim;
    /* Position 0 fast path: no rotation. */
    if (position == 0) {
        if (in != out) {
            for (size_t i = 0; i < head_dim; i++) out[i] = in[i];
        }
        return;
    }
    /* Copy unrotated tail (partial RoPE) first in case in/out alias. */
    for (size_t i = rope_len; i < head_dim; i++) {
        out[i] = in[i];
    }
    size_t half = rope_len / 2;
    /* freq starts at 1.0 (= theta^0) and is multiplied by theta^(-2/head_dim)
     * each step. Pair i uses freq = theta^(-2*i/head_dim). */
    float freq_mul = powf(theta, -2.0f / (float)rope_len);
    float freq = 1.0f;
    /* If in/out alias, we must read both halves before writing. Use a local
     * copy of the low half to avoid clobbering reads of the high half. */
    for (size_t i = 0; i < half; i++) {
        float x0 = in[i];
        float x1 = in[half + i];
        float angle = (float)position * freq;
        float c = cosf(angle);
        float s = sinf(angle);
        out[i]        = x0 * c - x1 * s;
        out[half + i] = x0 * s + x1 * c;
        freq *= freq_mul;
    }
}

void oc_apply_rope_yarn_f32(const float *in, float *out, size_t head_dim,
                             size_t rope_len, int64_t position, float theta,
                             float yarn_factor, uint32_t yarn_orig_ctx)
{
    if (yarn_factor <= 0.0f || yarn_orig_ctx == 0 ||
        (uint64_t)position <= (uint64_t)yarn_orig_ctx) {
        oc_apply_rope_f32(in, out, head_dim, rope_len, position, theta);
        return;
    }
    if (rope_len > head_dim) rope_len = head_dim;
    if (position == 0) {
        if (in != out)
            for (size_t i = 0; i < head_dim; i++) out[i] = in[i];
        return;
    }
    /* Copy unrotated tail. */
    for (size_t i = rope_len; i < head_dim; i++) out[i] = in[i];

    size_t half = rope_len / 2;
    float freq_mul = powf(theta, -2.0f / (float)rope_len);
    float freq = 1.0f;

    /* YaRN: scale position beyond orig_ctx.
     * ramp = smooth interpolation between [orig_ctx * 0.8, orig_ctx * 1.2].
     * Beyond orig_ctx * 1.2, use full YaRN scaling: scale = orig_ctx / position. */
    float pos_f = (float)position;
    float orig_f = (float)yarn_orig_ctx;
    float scale = 1.0f;
    if (pos_f > orig_f * 0.8f) {
        float ramp = (pos_f - orig_f * 0.8f) / (orig_f * 0.4f);
        if (ramp > 1.0f) ramp = 1.0f;
        scale = 1.0f - ramp + ramp * (orig_f / pos_f);
    }

    for (size_t i = 0; i < half; i++) {
        float x0 = in[i];
        float x1 = in[half + i];
        float angle = pos_f * scale * freq;
        float c = cosf(angle);
        float s = sinf(angle);
        out[i]        = x0 * c - x1 * s;
        out[half + i] = x0 * s + x1 * c;
        freq *= freq_mul;
    }
}
