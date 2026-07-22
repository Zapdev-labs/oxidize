/*
 * flash_attention.c — fused flash attention kernel (CPU scalar).
 */
#include "oxidize/flash_attention.h"

#include <math.h>
#include <string.h>

OcError oc_flash_attention_head(const float *q,
                                 const float *k_cache,
                                 const float *v_cache,
                                 size_t seq_len,
                                 size_t head_dim,
                                 float *out,
                                 float *temp)
{
    if (!q || !k_cache || !v_cache || !out) return OC_ERR_INVALID_ARG;
    (void)temp; /* not needed in the scalar implementation */

    float scale = 1.0f / sqrtf((float)head_dim);

    /* Online softmax: maintain running max and sum. */
    float run_max = -INFINITY;
    float run_sum = 0.0f;
    memset(out, 0, head_dim * sizeof(float));

    for (size_t t = 0; t < seq_len; t++) {
        const float *k_t = k_cache + t * head_dim;
        /* QK dot product. */
        float dot = 0.0f;
        for (size_t i = 0; i < head_dim; i++) {
            dot += q[i] * k_t[i];
        }
        float score = dot * scale;
        float new_max = (score > run_max) ? score : run_max;
        float exp_factor = expf(run_max - new_max);
        float exp_score = expf(score - new_max);
        /* Rescale running output. */
        for (size_t i = 0; i < head_dim; i++) {
            out[i] *= exp_factor;
        }
        /* Add new V contribution. */
        const float *v_t = v_cache + t * head_dim;
        for (size_t i = 0; i < head_dim; i++) {
            out[i] += exp_score * v_t[i];
        }
        run_sum = run_sum * exp_factor + exp_score;
        run_max = new_max;
    }

    /* Normalize. */
    if (run_sum > 0.0f) {
        float inv = 1.0f / run_sum;
        for (size_t i = 0; i < head_dim; i++) {
            out[i] *= inv;
        }
    }
    return OC_OK;
}

OcError oc_flash_attention_multi_head(const float *q,
                                      const float *k_cache,
                                      const float *v_cache,
                                      size_t n_heads,
                                      size_t n_heads_kv,
                                      size_t seq_len,
                                      size_t head_dim,
                                      float *out,
                                      float *temp)
{
    if (!q || !k_cache || !v_cache || !out) return OC_ERR_INVALID_ARG;
    size_t group_size = n_heads / n_heads_kv;
    size_t kv_row_floats = n_heads_kv * head_dim;

    for (size_t h = 0; h < n_heads; h++) {
        size_t kv_head = h / group_size;
        const float *q_h = q + h * head_dim;
        float *out_h = out + h * head_dim;

        /* K/V for this head: stride is kv_row_floats per position. */
        const float *k_h = k_cache + kv_head * head_dim;
        const float *v_h = v_cache + kv_head * head_dim;

        /* We need to compute attention with the correct stride.
         * k_cache layout: [seq_len, n_heads_kv, head_dim]
         * So k_cache[t * kv_row_floats + kv_head * head_dim + i] = K[t][kv_head][i]
         *
         * For the flash attention call, we need a contiguous [seq_len, head_dim]
         * view. Since the data is strided, we compute manually. */
        float scale = 1.0f / sqrtf((float)head_dim);
        float run_max = -INFINITY;
        float run_sum = 0.0f;
        memset(out_h, 0, head_dim * sizeof(float));

        for (size_t t = 0; t < seq_len; t++) {
            const float *k_t = k_h + t * kv_row_floats;
            float dot = 0.0f;
            for (size_t i = 0; i < head_dim; i++) {
                dot += q_h[i] * k_t[i];
            }
            float score = dot * scale;
            float new_max = (score > run_max) ? score : run_max;
            float exp_factor = expf(run_max - new_max);
            float exp_score = expf(score - new_max);
            for (size_t i = 0; i < head_dim; i++) {
                out_h[i] *= exp_factor;
            }
            const float *v_t = v_h + t * kv_row_floats;
            for (size_t i = 0; i < head_dim; i++) {
                out_h[i] += exp_score * v_t[i];
            }
            run_sum = run_sum * exp_factor + exp_score;
            run_max = new_max;
        }

        if (run_sum > 0.0f) {
            float inv = 1.0f / run_sum;
            for (size_t i = 0; i < head_dim; i++) {
                out_h[i] *= inv;
            }
        }
    }
    (void)temp;
    return OC_OK;
}

OcError oc_flash_attention_sliding(const float *q,
                                    const float *k_cache,
                                    const float *v_cache,
                                    size_t seq_len,
                                    size_t head_dim,
                                    size_t window_start,
                                    float *out,
                                    float *temp)
{
    if (!q || !k_cache || !v_cache || !out) return OC_ERR_INVALID_ARG;
    (void)temp;

    size_t start = (window_start < seq_len) ? window_start : seq_len;
    float scale = 1.0f / sqrtf((float)head_dim);
    float run_max = -INFINITY;
    float run_sum = 0.0f;
    memset(out, 0, head_dim * sizeof(float));

    for (size_t t = start; t < seq_len; t++) {
        const float *k_t = k_cache + t * head_dim;
        float dot = 0.0f;
        for (size_t i = 0; i < head_dim; i++) {
            dot += q[i] * k_t[i];
        }
        float score = dot * scale;
        float new_max = (score > run_max) ? score : run_max;
        float exp_factor = expf(run_max - new_max);
        float exp_score = expf(score - new_max);
        for (size_t i = 0; i < head_dim; i++) {
            out[i] *= exp_factor;
        }
        const float *v_t = v_cache + t * head_dim;
        for (size_t i = 0; i < head_dim; i++) {
            out[i] += exp_score * v_t[i];
        }
        run_sum = run_sum * exp_factor + exp_score;
        run_max = new_max;
    }

    if (run_sum > 0.0f) {
        float inv = 1.0f / run_sum;
        for (size_t i = 0; i < head_dim; i++) {
            out[i] *= inv;
        }
    }
    return OC_OK;
}

OcError oc_attention_scores(const float *q,
                             const float *k_cache,
                             size_t seq_len,
                             size_t head_dim,
                             float *scores)
{
    if (!q || !k_cache || !scores) return OC_ERR_INVALID_ARG;
    float scale = 1.0f / sqrtf((float)head_dim);

    for (size_t t = 0; t < seq_len; t++) {
        const float *k_t = k_cache + t * head_dim;
        float dot = 0.0f;
        for (size_t i = 0; i < head_dim; i++) {
            dot += q[i] * k_t[i];
        }
        scores[t] = dot * scale;
    }
    return OC_OK;
}
