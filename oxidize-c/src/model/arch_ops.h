/*
 * arch_ops.h — small private helpers shared by the per-architecture
 * reference engines (qwen_arch.c / mistral_arch.c / gemma_arch.c /
 * phi_arch.c). These engines are test/reference implementations; the
 * production inference path lives in llama.c / inf_model.c.
 *
 * Each helper is a byte-identical extraction of code that previously
 * appeared in all four files. No behavior change.
 */
#ifndef OXIDIZE_C_MODEL_ARCH_OPS_H
#define OXIDIZE_C_MODEL_ARCH_OPS_H

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "oxidize/error.h"

/* RMSNorm hidden in place: normed[i] = x[i] * rms * w[i]. If w is NULL,
 * copy x unchanged. (n == hidden_dim) */
static inline void oc_arch_rms_norm(const float *x, const float *w,
                                    float *normed, size_t n, float eps)
{
    if (w) {
        float ss = 0.0f;
        for (size_t i = 0; i < n; i++) ss += x[i] * x[i];
        float rms = 1.0f / sqrtf(ss / n + eps);
        for (size_t i = 0; i < n; i++) normed[i] = x[i] * rms * w[i];
    } else {
        memcpy(normed, x, n * sizeof(float));
    }
}

/* Row-major matvec: out[r] = sum_c W[r*n + c] * in[c]. If W is NULL, out is
 * zeroed (calloc'd buffers pass through unchanged in effect). */
static inline void oc_arch_matvec(const float *w, const float *in,
                                  float *out, size_t rows, size_t cols)
{
    if (!w) return;
    for (size_t r = 0; r < rows; r++) {
        float dot = 0.0f;
        for (size_t c = 0; c < cols; c++) dot += w[r * cols + c] * in[c];
        out[r] = dot;
    }
}

/* Apply interleaved RoPE to one head of size head_dim at sequence
 * position pos. */
static inline void oc_arch_rope_head(float *head, size_t head_dim,
                                     size_t pos, float theta)
{
    for (size_t d = 0; d < head_dim; d += 2) {
        float freq = pos / powf(theta, (float)(d / 2) / (float)(head_dim / 2));
        float c = cosf(freq), s = sinf(freq);
        float h0 = head[d], h1 = head[d + 1];
        head[d] = h0 * c - h1 * s;
        head[d + 1] = h0 * s + h1 * c;
    }
}

/* tanh-approximation GELU used by Gemma / Phi FFNs. */
static inline float oc_arch_gelu_tanh(float x)
{
    float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

#endif /* OXIDIZE_C_MODEL_ARCH_OPS_H */
