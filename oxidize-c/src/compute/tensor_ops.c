/*
 * tensor_ops.c — High-level tensor operations implementation.
 *
 * All operations use plain C loops with no external dependencies.
 * SIMD-optimized versions can be added later via function pointers
 * or compile-time dispatch (matching the simd.h pattern).
 */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/tensor_ops.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── Element-wise operations ──────────────────────────────────────────── */

void oc_tensor_add_f32(const float *a, const float *b, float *out, size_t n)
{
    for (size_t i = 0; i < n; i++) out[i] = a[i] + b[i];
}

void oc_tensor_mul_f32(const float *a, const float *b, float *out, size_t n)
{
    for (size_t i = 0; i < n; i++) out[i] = a[i] * b[i];
}

void oc_tensor_scale_f32(const float *a, float scale, float *out, size_t n)
{
    for (size_t i = 0; i < n; i++) out[i] = a[i] * scale;
}

void oc_tensor_add_scalar_f32(const float *a, float scalar, float *out, size_t n)
{
    for (size_t i = 0; i < n; i++) out[i] = a[i] + scalar;
}

void oc_tensor_exp_f32(const float *a, float *out, size_t n)
{
    for (size_t i = 0; i < n; i++) out[i] = expf(a[i]);
}

void oc_tensor_tanh_f32(const float *a, float *out, size_t n)
{
    for (size_t i = 0; i < n; i++) out[i] = tanhf(a[i]);
}

void oc_tensor_sigmoid_f32(const float *a, float *out, size_t n)
{
    for (size_t i = 0; i < n; i++)
        out[i] = 1.0f / (1.0f + expf(-a[i]));
}

void oc_tensor_silu_f32(const float *a, float *out, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        float sig = 1.0f / (1.0f + expf(-a[i]));
        out[i] = a[i] * sig;
    }
}

void oc_tensor_gelu_f32(const float *a, float *out, size_t n)
{
    /* tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
    const float c = 0.7978845608f; /* sqrt(2/pi) */
    for (size_t i = 0; i < n; i++) {
        float x = a[i];
        float x3 = x * x * x;
        float inner = c * (x + 0.044715f * x3);
        out[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

void oc_tensor_relu_f32(const float *a, float *out, size_t n)
{
    for (size_t i = 0; i < n; i++)
        out[i] = a[i] > 0.0f ? a[i] : 0.0f;
}

void oc_tensor_iadd_f32(float *a, const float *b, size_t n)
{
    for (size_t i = 0; i < n; i++) a[i] += b[i];
}

void oc_tensor_imul_f32(float *a, const float *b, size_t n)
{
    for (size_t i = 0; i < n; i++) a[i] *= b[i];
}

void oc_tensor_iscale_f32(float *a, float scale, size_t n)
{
    for (size_t i = 0; i < n; i++) a[i] *= scale;
}

/* ─── Reductions ────────────────────────────────────────────────────────── */

float oc_tensor_sum_f32(const float *a, size_t n)
{
    float s = 0.0f;
    for (size_t i = 0; i < n; i++) s += a[i];
    return s;
}

float oc_tensor_max_f32(const float *a, size_t n)
{
    if (n == 0) return 0.0f;
    float m = a[0];
    for (size_t i = 1; i < n; i++)
        if (a[i] > m) m = a[i];
    return m;
}

size_t oc_tensor_argmax_f32(const float *a, size_t n)
{
    if (n == 0) return 0;
    size_t idx = 0;
    float m = a[0];
    for (size_t i = 1; i < n; i++) {
        if (a[i] > m) { m = a[i]; idx = i; }
    }
    return idx;
}

float oc_tensor_l2_norm_f32(const float *a, size_t n)
{
    float s = 0.0f;
    for (size_t i = 0; i < n; i++) s += a[i] * a[i];
    return sqrtf(s);
}

float oc_tensor_mean_f32(const float *a, size_t n)
{
    if (n == 0) return 0.0f;
    return oc_tensor_sum_f32(a, n) / (float)n;
}

float oc_tensor_variance_f32(const float *a, size_t n)
{
    if (n == 0) return 0.0f;
    float mean = oc_tensor_mean_f32(a, n);
    float var = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = a[i] - mean;
        var += d * d;
    }
    return var / (float)n;
}

/* ─── Copy / transpose / concat ─────────────────────────────────────────── */

void oc_tensor_copy_f32(const float *src, float *dst, size_t n)
{
    memcpy(dst, src, n * sizeof(float));
}

void oc_tensor_transpose_f32(const float *src, float *dst,
                              size_t rows, size_t cols)
{
    for (size_t r = 0; r < rows; r++)
        for (size_t c = 0; c < cols; c++)
            dst[c * rows + r] = src[r * cols + c];
}

void oc_tensor_concat_f32(const float *a, const float *b, float *out,
                           size_t m, size_t n_a, size_t n_b)
{
    for (size_t i = 0; i < m; i++) {
        memcpy(out + i * (n_a + n_b), a + i * n_a, n_a * sizeof(float));
        memcpy(out + i * (n_a + n_b) + n_a, b + i * n_b, n_b * sizeof(float));
    }
}

void oc_tensor_repeat_row_f32(const float *row, float *out,
                                size_t m, size_t n)
{
    for (size_t i = 0; i < m; i++)
        memcpy(out + i * n, row, n * sizeof(float));
}

/* ─── Softmax ──────────────────────────────────────────────────────────── */

void oc_tensor_softmax_f32(const float *a, float *out, size_t n)
{
    if (n == 0) return;
    float max_val = oc_tensor_max_f32(a, n);
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        out[i] = expf(a[i] - max_val);
        sum += out[i];
    }
    if (sum == 0.0f) sum = 1.0f;
    float inv = 1.0f / sum;
    for (size_t i = 0; i < n; i++) out[i] *= inv;
}

void oc_tensor_softmax_online_f32(const float *a, float *out, size_t n)
{
    if (n == 0) return;
    /* Online softmax: track max and sum in one pass. */
    float max_val = a[0];
    float sum = 1.0f;
    out[0] = 1.0f;
    for (size_t i = 1; i < n; i++) {
        float e = expf(a[i] - max_val);
        float new_max = a[i] > max_val ? a[i] : max_val;
        float rescale = expf(max_val - new_max);
        sum = sum * rescale + e;
        for (size_t j = 0; j <= i; j++)
            out[j] *= rescale;
        out[i] = e;
        max_val = new_max;
    }
    if (sum == 0.0f) sum = 1.0f;
    float inv = 1.0f / sum;
    for (size_t i = 0; i < n; i++) out[i] *= inv;
}

void oc_tensor_softmax_temp_f32(const float *a, float *out, size_t n, float temp)
{
    if (n == 0) return;
    if (temp <= 0.0f) temp = 1.0f;
    float inv_temp = 1.0f / temp;
    float max_val = oc_tensor_max_f32(a, n) * inv_temp;
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        out[i] = expf(a[i] * inv_temp - max_val);
        sum += out[i];
    }
    if (sum == 0.0f) sum = 1.0f;
    float inv = 1.0f / sum;
    for (size_t i = 0; i < n; i++) out[i] *= inv;
}

void oc_tensor_log_softmax_f32(const float *a, float *out, size_t n)
{
    if (n == 0) return;
    float max_val = oc_tensor_max_f32(a, n);
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++)
        sum += expf(a[i] - max_val);
    float logsum = max_val + logf(sum);
    for (size_t i = 0; i < n; i++)
        out[i] = a[i] - logsum;
}

/* ─── Normalization ────────────────────────────────────────────────────── */

void oc_tensor_layer_norm_f32(const float *a, const float *weight,
                               const float *bias, float *out,
                               size_t n, float eps)
{
    float mean = oc_tensor_mean_f32(a, n);
    float var = oc_tensor_variance_f32(a, n);
    float inv_std = 1.0f / sqrtf(var + eps);
    for (size_t i = 0; i < n; i++) {
        float normed = (a[i] - mean) * inv_std;
        out[i] = normed * weight[i] + (bias ? bias[i] : 0.0f);
    }
}

void oc_tensor_rms_norm_f32(const float *a, const float *weight,
                              float *out, size_t n, float eps)
{
    float sum_sq = 0.0f;
    for (size_t i = 0; i < n; i++) sum_sq += a[i] * a[i];
    float rms = sqrtf(sum_sq / (float)n + eps);
    float inv_rms = 1.0f / rms;
    for (size_t i = 0; i < n; i++)
        out[i] = a[i] * inv_rms * weight[i];
}

/* ─── Rotary position embedding ─────────────────────────────────────────── */

void oc_tensor_rope_neox_f32(float *x, size_t head_dim,
                              uint32_t position, float freq_base)
{
    size_t half = head_dim / 2;
    float inv_freq = 1.0f / freq_base;
    for (size_t i = 0; i < half; i++) {
        float freq = inv_freq * powf(freq_base, (float)i / (float)half);
        float angle = (float)position * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);
        float x0 = x[i];
        float x1 = x[i + half];
        x[i] = x0 * cos_a - x1 * sin_a;
        x[i + half] = x0 * sin_a + x1 * cos_a;
    }
}

void oc_tensor_rope_gptj_f32(float *x, size_t head_dim,
                              uint32_t position, float freq_base)
{
    size_t half = head_dim / 2;
    float inv_freq = 1.0f / freq_base;
    for (size_t i = 0; i < half; i++) {
        float freq = inv_freq * powf(freq_base, (float)i / (float)half);
        float angle = (float)position * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);
        float x0 = x[2 * i];
        float x1 = x[2 * i + 1];
        x[2 * i] = x0 * cos_a - x1 * sin_a;
        x[2 * i + 1] = x0 * sin_a + x1 * cos_a;
    }
}

void oc_tensor_rope_neox_row_f32(float *x, size_t n_head, size_t head_dim,
                                   uint32_t position, float freq_base)
{
    for (size_t h = 0; h < n_head; h++) {
        oc_tensor_rope_neox_f32(x + h * head_dim, head_dim, position, freq_base);
    }
}

/* ─── GEMM ──────────────────────────────────────────────────────────────── */

void oc_tensor_gemm_f32(const float *A, const float *B, float *C,
                         size_t M, size_t K, size_t N)
{
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++)
                sum += A[m * K + k] * B[k * N + n];
            C[m * N + n] = sum;
        }
    }
}

void oc_tensor_gemm_at_f32(const float *A, const float *B, float *C,
                            size_t M, size_t K, size_t N)
{
    /* A is [K×M], so A^T is [M×K]. A^T[m,k] = A[k,m]. */
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++)
                sum += A[k * M + m] * B[k * N + n];
            C[m * N + n] = sum;
        }
    }
}

void oc_tensor_gemm_batch_f32(const float *A, const float *B, float *C,
                                size_t batch, size_t M, size_t K, size_t N)
{
    for (size_t b = 0; b < batch; b++) {
        oc_tensor_gemm_f32(A + b * M * K, B, C + b * M * N, M, K, N);
    }
}

/* ─── Attention helpers ────────────────────────────────────────────────── */

void oc_tensor_attention_head_f32(const float *Q, const float *K,
                                    const float *V, float *out,
                                    size_t seq_len, size_t d_head, float scale)
{
    /* Compute attention scores: scores[i] = Q · K[i] * scale. */
    float *scores = malloc(seq_len * sizeof(float));
    if (!scores) return;

    for (size_t i = 0; i < seq_len; i++) {
        float dot = 0.0f;
        for (size_t d = 0; d < d_head; d++)
            dot += Q[d] * K[i * d_head + d];
        scores[i] = dot * scale;
    }

    /* Softmax. */
    oc_tensor_softmax_f32(scores, scores, seq_len);

    /* Weighted sum of V. */
    for (size_t d = 0; d < d_head; d++) {
        float sum = 0.0f;
        for (size_t i = 0; i < seq_len; i++)
            sum += scores[i] * V[i * d_head + d];
        out[d] = sum;
    }

    free(scores);
}

void oc_tensor_attention_mha_f32(const float *Q, const float *K,
                                   const float *V, float *out,
                                   size_t n_head_q, size_t n_head_kv,
                                   size_t seq_len, size_t d_head)
{
    float scale = 1.0f / sqrtf((float)d_head);
    size_t kv_stride = (n_head_kv > 0) ? n_head_kv : 1;

    for (size_t h = 0; h < n_head_q; h++) {
        size_t kv_h = h * n_head_kv / n_head_q; /* GQA mapping */
        if (kv_h >= n_head_kv) kv_h = n_head_kv - 1;

        const float *Q_h = Q + h * d_head;
        const float *K_h = K + kv_h * seq_len * d_head;
        const float *V_h = V + kv_h * seq_len * d_head;
        float *out_h = out + h * d_head;

        oc_tensor_attention_head_f32(Q_h, K_h, V_h, out_h, seq_len, d_head, scale);
    }
}

/* ─── Utility ───────────────────────────────────────────────────────────── */

void oc_tensor_fill_f32(float *a, float val, size_t n)
{
    for (size_t i = 0; i < n; i++) a[i] = val;
}

void oc_tensor_zero_f32(float *a, size_t n)
{
    memset(a, 0, n * sizeof(float));
}

void oc_tensor_random_f32(float *a, size_t n, uint32_t seed)
{
    /* Simple LCG (Linear Congruential Generator). */
    uint32_t state = seed ? seed : 1;
    for (size_t i = 0; i < n; i++) {
        state = state * 1103515245u + 12345u;
        a[i] = (float)(state >> 8) / (float)(1u << 24);
    }
}

void oc_tensor_print_f32(const float *a, size_t n, const char *name)
{
    fprintf(stderr, "%s: [", name ? name : "tensor");
    size_t limit = n < 8 ? n : 8;
    for (size_t i = 0; i < limit; i++)
        fprintf(stderr, "%.4f%s", a[i], i < limit - 1 ? ", " : "");
    if (n > 8) fprintf(stderr, ", ...");
    fprintf(stderr, "] (n=%zu)\n", n);
}

void oc_tensor_print_2d_f32(const float *a, size_t rows, size_t cols,
                             const char *name)
{
    fprintf(stderr, "%s [%zu×%zu]:\n", name ? name : "tensor", rows, cols);
    size_t row_limit = rows < 4 ? rows : 4;
    size_t col_limit = cols < 8 ? cols : 8;
    for (size_t r = 0; r < row_limit; r++) {
        fprintf(stderr, "  [");
        for (size_t c = 0; c < col_limit; c++)
            fprintf(stderr, "%.4f%s", a[r * cols + c], c < col_limit - 1 ? ", " : "");
        if (cols > 8) fprintf(stderr, ", ...");
        fprintf(stderr, "]\n");
    }
    if (rows > 4) fprintf(stderr,  "  ...\n");
}
