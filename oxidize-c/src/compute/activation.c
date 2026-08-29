/* activation.c — RMSNorm, RoPE, SwiGLU scalar reference implementations. Port of oxidize-core/src/compute/tensor/kernels/activation.rs. Bit-exact with the Rust scalar reference (VAL-FWD-001..004). */
#include "oxidize/activation.h"
#include "oxidize/attn_kernels.h"

#include <math.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

void oc_rms_norm_f32(const float *x, const float *weight, float *out,
                    size_t n, float eps)
{
    float ss = oc_attn_dot_f32(x, x, n);
    float inv_rms = 1.0f / sqrtf(ss / (float)n + eps);
    oc_attn_rms_apply_f32(x, weight, inv_rms, out, n);
}

void oc_swiglu_inplace_f32(float *gate, const float *up, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        float g = gate[i];
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
    if (rope_len == 0) return;
    size_t half = rope_len / 2;
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

/* Interleaved ("NORM") RoPE: rotates the pair (2i, 2i+1) rather than (i, i + rope_len/2). */
void oc_apply_rope_norm_f32(const float *in, float *out, size_t head_dim,
                            size_t rope_len, int64_t position, float theta)
{
    if (rope_len > head_dim) rope_len = head_dim;
    if (position == 0) {
        if (in != out) {
            for (size_t i = 0; i < head_dim; i++) out[i] = in[i];
        }
        return;
    }
    for (size_t i = rope_len; i < head_dim; i++) out[i] = in[i];
    if (rope_len == 0) return;

    /* Pair p = i/2 uses freq = theta^(-2p/rope_len), the same frequency
     * ladder as the split-half form — only the pairing differs. */
    const float freq_mul = powf(theta, -2.0f / (float)rope_len);
    float freq = 1.0f;
    for (size_t i = 0; i + 1 < rope_len; i += 2) {
        const float x0 = in[i];
        const float x1 = in[i + 1];
        const float angle = (float)position * freq;
        const float c = cosf(angle);
        const float s = sinf(angle);
        out[i]     = x0 * c - x1 * s;
        out[i + 1] = x0 * s + x1 * c;
        freq *= freq_mul;
    }
}


void oc_softmax_f32(const float *input, float *output, size_t n)
{
    if (n == 0 || !input || !output) return;

    float max_val = input[0];
    for (size_t i = 1; i < n; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    double sum_exp = 0.0;
    for (size_t i = 0; i < n; i++) {
        float e = expf(input[i] - max_val);
        output[i] = e;
        sum_exp += (double)e;
    }

    float inv_sum = (float)(1.0 / sum_exp);
    for (size_t i = 0; i < n; i++)
        output[i] *= inv_sum;
}


void oc_layer_norm_f32(const float *input, const float *weight, const float *bias,
                        float *output, size_t n, float eps)
{
    if (n == 0 || !input || !weight || !bias || !output) return;

    float mean = 0.0f;
    for (size_t i = 0; i < n; i++) mean += input[i];
    mean /= (float)n;

    float variance = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float centered = input[i] - mean;
        variance += centered * centered;
    }
    variance /= (float)n;

    float inv_std = 1.0f / sqrtf(variance + eps);
    for (size_t i = 0; i < n; i++)
        output[i] = (input[i] - mean) * inv_std * weight[i] + bias[i];
}


void oc_swiglu_f32(const float *gate, const float *up, float *output, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        float g = gate[i];
        float sigmoid = 1.0f / (1.0f + expf(-g));
        output[i] = g * sigmoid * up[i];
    }
}


void oc_scaled_dot_product_attention_f32(const float *query,
                                          const float *key,
                                          const float *value,
                                          size_t seq_len, size_t dim,
                                          float *output)
{
    if (!query || !key || !value || !output || dim == 0) return;

    for (size_t i = 0; i < dim; i++) output[i] = 0.0f;
    if (seq_len == 0) return;

    float scale = 1.0f / sqrtf((float)dim);

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float *acc = output;

    for (size_t t = 0; t < seq_len; t++) {
        const float *k_row = &key[t * dim];
        const float *v_row = &value[t * dim];

        float score = 0.0f;
        for (size_t i = 0; i < dim; i++)
            score += query[i] * k_row[i];
        score *= scale;

        float old_max = running_max;
        if (score > running_max) {
            running_max = score;
            float rescale = expf(old_max - running_max);
            running_sum = running_sum * rescale + 1.0f;
            for (size_t i = 0; i < dim; i++)
                acc[i] = acc[i] * rescale + v_row[i];
        } else {
            float e = expf(score - running_max);
            running_sum += e;
            for (size_t i = 0; i < dim; i++)
                acc[i] += v_row[i] * e;
        }
    }

    float inv_sum = 1.0f / running_sum;
    for (size_t i = 0; i < dim; i++)
        acc[i] *= inv_sum;
}


void oc_rms_norm_f32_qwen(const float *x, const float *weight, float *out,
                           size_t n, float eps, bool weight_plus_one)
{
    if (!x || !weight || !out || n == 0) return;

    float ss = 0.0f;
    for (size_t i = 0; i < n; i++)
        ss += x[i] * x[i];

    float inv_rms = 1.0f / sqrtf(ss / (float)n + eps);
    for (size_t i = 0; i < n; i++) {
        float w = weight_plus_one ? (1.0f + weight[i]) : weight[i];
        out[i] = x[i] * inv_rms * w;
    }
}
void oc_apply_rope_yarn_f32(const float *in, float *out, size_t head_dim,
                             size_t rope_len, int64_t position, float theta,
                             float yarn_factor, uint32_t yarn_orig_ctx)
{
    /* Default amplitude: 1 + 0.1*ln(factor), the standard YaRN mscale. */
    oc_apply_rope_yarn_scaled_f32(in, out, head_dim, rope_len, position, theta,
                                  yarn_factor, yarn_orig_ctx, -1.0f);
}

void oc_apply_rope_yarn_scaled_f32(const float *in, float *out, size_t head_dim,
                                    size_t rope_len, int64_t position,
                                    float theta, float yarn_factor,
                                    uint32_t yarn_orig_ctx, float attn_factor)
{
    bool yarn = yarn_factor > 1.0f && yarn_orig_ctx > 0;
    if (!yarn) {
        oc_apply_rope_f32(in, out, head_dim, rope_len, position, theta);
        return;
    }
    if (rope_len > head_dim) rope_len = head_dim;
    float mscale = (attn_factor >= 0.0f) ? attn_factor
                                         : (1.0f + 0.1f * logf(yarn_factor));
    if (position == 0) {
        if (in != out)
            for (size_t i = 0; i < head_dim; i++) out[i] = in[i];
        for (size_t i = 0; i < rope_len; i++) out[i] *= mscale;
        return;
    }
    /* Copy unrotated tail. */
    for (size_t i = rope_len; i < head_dim; i++) out[i] = in[i];

    size_t half = rope_len / 2;
    float freq_mul = powf(theta, -2.0f / (float)rope_len);
    float freq = 1.0f;

    /* YaRN parameters (matching Rust apply_rope_f32_yarn). */
    float freq_scale = 1.0f / yarn_factor;
    /* Compute correction range.
     * corr_dim(n_dims, orig_ctx, n_rot, base) =
     *   n_dims * ln(orig_ctx / (n_rot * 2*PI)) / (2 * ln(base)) */
    float base_log = 2.0f * logf(theta);
    float ratio_fast = (float)yarn_orig_ctx / (32.0f * 2.0f * 3.14159265f);
    float ratio_slow = (float)yarn_orig_ctx / (1.0f * 2.0f * 3.14159265f);
    float corr_lo = floorf((float)rope_len * logf(ratio_fast) / base_log);
    float corr_hi = ceilf((float)rope_len * logf(ratio_slow) / base_log);
    if (corr_lo < 0.0f) corr_lo = 0.0f;
    if (corr_hi > (float)rope_len - 1.0f) corr_hi = (float)rope_len - 1.0f;
    float denom = corr_hi - corr_lo;

    float pos_f = (float)position;

    for (size_t i = 0; i < half; i++) {
        float x0 = in[i];
        float x1 = in[half + i];
        float theta_extrap = pos_f * freq;
        float theta_interp = theta_extrap * freq_scale;
        float ramp = 1.0f - ((float)i - corr_lo) /
                     (denom > 0.001f ? denom : 0.001f);
        if (ramp < 0.0f) ramp = 0.0f;
        if (ramp > 1.0f) ramp = 1.0f;
        float angle = theta_interp * (1.0f - ramp) + theta_extrap * ramp;
        float c = cosf(angle) * mscale;
        float s = sinf(angle) * mscale;
        out[i]        = x0 * c - x1 * s;
        out[half + i] = x0 * s + x1 * c;
        freq *= freq_mul;
    }
}
