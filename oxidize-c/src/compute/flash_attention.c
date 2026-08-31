/*
 * flash_attention.c — fused flash attention kernel (CPU scalar).
 */
#include "oxidize/flash_attention.h"

#include <math.h>
#include <string.h>

/* Shared online-softmax attention core.
 * Attends q over K/V rows [start, seq_len), where row t lives at
 * k + t * kv_stride (same for v). Writes the normalized result to out. */
static void flash_online_softmax(const float *q,
                                 const float *k,
                                 const float *v,
                                 size_t start,
                                 size_t seq_len,
                                 size_t head_dim,
                                 size_t kv_stride,
                                 float *out)
{
    float scale = 1.0f / sqrtf((float)head_dim);
    float run_max = -INFINITY;
    float run_sum = 0.0f;
    memset(out, 0, head_dim * sizeof(float));

    for (size_t t = start; t < seq_len; t++) {
        const float *k_t = k + t * kv_stride;
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
        const float *v_t = v + t * kv_stride;
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
}

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

    flash_online_softmax(q, k_cache, v_cache, 0, seq_len, head_dim,
                         head_dim, out);
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

        /* We need to compute attention with the correct stride. */
        flash_online_softmax(q_h, k_h, v_h, 0, seq_len, head_dim,
                             kv_row_floats, out_h);
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
    flash_online_softmax(q, k_cache, v_cache, start, seq_len, head_dim,
                         head_dim, out);
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


uint16_t oc_f32_to_f16_bits(float value)
{
    uint32_t x;
    memcpy(&x, &value, sizeof(x));
    uint16_t sign = (uint16_t)((x >> 16) & 0x8000);
    int32_t exp = (int32_t)((x >> 23) & 0xFF);
    uint32_t frac = x & 0x007FFFFF;

    if (exp == 0xFF) {
        if (frac == 0) return sign | 0x7C00;
        uint16_t nan = (uint16_t)(frac >> 13);
        return sign | 0x7C00 | nan | (nan == 0 ? 1 : 0);
    }

    int32_t exp16 = exp - 127 + 15;
    if (exp16 >= 0x1F) return sign | 0x7C00;
    if (exp16 <= 0) {
        if (exp16 < -10) return sign;
        uint32_t mant = frac | 0x00800000;
        uint32_t shift = (uint32_t)(14 - exp16);
        uint16_t half_frac = (uint16_t)(mant >> shift);
        if (((mant >> (shift - 1)) & 1) != 0) half_frac++;
        return sign | half_frac;
    }

    uint16_t half_exp = (uint16_t)exp16 << 10;
    uint16_t half_frac = (uint16_t)(frac >> 13);
    if ((frac & 0x00001000) != 0) {
        half_frac++;
        if ((half_frac & 0x0400) != 0) {
            half_frac = 0;
            half_exp += 0x0400;
            if (half_exp >= 0x7C00) return sign | 0x7C00;
        }
    }
    return sign | half_exp | half_frac;
}

float oc_f16_to_f32_bits(uint16_t h)
{
    uint16_t sign = (h >> 15) & 1;
    uint16_t exp = (h >> 10) & 0x1F;
    uint16_t mant = h & 0x3FF;
    float val;
    if (exp == 0) {
        if (mant == 0) val = 0.0f;
        else val = (float)mant * 0.000000059604645f; /* 2^-24 */
    } else if (exp == 0x1F) {
        val = (mant == 0) ? INFINITY : NAN;
    } else {
        val = (1.0f + (float)mant / 1024.0f) *
              ldexpf(1.0f, (int)exp - 15);
    }
    return sign ? -val : val;
}


/* Per-head decode with online-softmax block tiling.
 * key_layer/value_layer layout: [seq_len][kv_len] where kv_len = kv_heads * head_dim.
 * kv_head selects which head within the KV to attend to. */
static void flash_decode_head_f32(const float *q,
                                  const float *key_layer,
                                  const float *value_layer,
                                  size_t seq_len,
                                  size_t head_dim,
                                  size_t kv_len,
                                  size_t kv_head,
                                  float *output)
{
    if (seq_len == 0) {
        memset(output, 0, head_dim * sizeof(float));
        return;
    }

    float scale = 1.0f / sqrtf((float)head_dim);
    size_t kv_offset = kv_head * head_dim;

    float run_max = -INFINITY;
    float run_sum = 0.0f;
    memset(output, 0, head_dim * sizeof(float));

    size_t token = 0;
    while (token < seq_len) {
        size_t block_end = token + OC_FLASH_BLOCK_SIZE;
        if (block_end > seq_len) block_end = seq_len;

        for (size_t t = token; t < block_end; t++) {
            const float *key_row = key_layer + t * kv_len + kv_offset;
            float dot = 0.0f;
            for (size_t i = 0; i < head_dim; i++)
                dot += q[i] * key_row[i];
            float score = dot * scale;

            float new_max = (score > run_max) ? score : run_max;
            float exp_factor = expf(run_max - new_max);
            float exp_score = expf(score - new_max);

            if (exp_factor != 1.0f) {
                for (size_t i = 0; i < head_dim; i++)
                    output[i] *= exp_factor;
            }

            const float *value_row = value_layer + t * kv_len + kv_offset;
            for (size_t i = 0; i < head_dim; i++)
                output[i] += exp_score * value_row[i];

            run_sum = run_sum * exp_factor + exp_score;
            run_max = new_max;
        }
        token = block_end;
    }

    if (run_sum > 0.0f) {
        float inv = 1.0f / run_sum;
        for (size_t i = 0; i < head_dim; i++)
            output[i] *= inv;
    }
}

/* Same as above but K/V are f16. */
static void flash_decode_head_f16(const float *q,
                                  const uint16_t *key_layer,
                                  const uint16_t *value_layer,
                                  size_t seq_len,
                                  size_t head_dim,
                                  size_t kv_len,
                                  size_t kv_head,
                                  float *output)
{
    if (seq_len == 0) {
        memset(output, 0, head_dim * sizeof(float));
        return;
    }

    float scale = 1.0f / sqrtf((float)head_dim);
    size_t kv_offset = kv_head * head_dim;

    float run_max = -INFINITY;
    float run_sum = 0.0f;
    memset(output, 0, head_dim * sizeof(float));

    size_t token = 0;
    while (token < seq_len) {
        size_t block_end = token + OC_FLASH_BLOCK_SIZE;
        if (block_end > seq_len) block_end = seq_len;

        for (size_t t = token; t < block_end; t++) {
            const uint16_t *key_row16 = key_layer + t * kv_len + kv_offset;
            float dot = 0.0f;
            for (size_t i = 0; i < head_dim; i++) {
                float k = oc_f16_to_f32_bits(key_row16[i]);
                dot += q[i] * k;
            }
            float score = dot * scale;

            float new_max = (score > run_max) ? score : run_max;
            float exp_factor = expf(run_max - new_max);
            float exp_score = expf(score - new_max);

            if (exp_factor != 1.0f) {
                for (size_t i = 0; i < head_dim; i++)
                    output[i] *= exp_factor;
            }

            const uint16_t *value_row16 = value_layer + t * kv_len + kv_offset;
            for (size_t i = 0; i < head_dim; i++) {
                float v = oc_f16_to_f32_bits(value_row16[i]);
                output[i] += exp_score * v;
            }

            run_sum = run_sum * exp_factor + exp_score;
            run_max = new_max;
        }
        token = block_end;
    }

    if (run_sum > 0.0f) {
        float inv = 1.0f / run_sum;
        for (size_t i = 0; i < head_dim; i++)
            output[i] *= inv;
    }
}

OcError oc_flash_attention_decode_heads_f32(const float *query_heads,
                                             const float *key_layer,
                                             const float *value_layer,
                                             size_t seq_len,
                                             size_t head_dim,
                                             size_t kv_len,
                                             size_t num_heads,
                                             size_t kv_heads,
                                             float *output_heads)
{
    if (!query_heads || !key_layer || !value_layer || !output_heads)
        return OC_ERR_INVALID_ARG;
    if (head_dim == 0) return OC_ERR_INVALID_ARG;
    if (kv_heads == 0 || (num_heads % kv_heads) != 0)
        return OC_ERR_INVALID_ARG;

    size_t group_size = num_heads / kv_heads;

    for (size_t head = 0; head < num_heads; head++) {
        size_t kv_head = head / group_size;
        const float *q_head = query_heads + head * head_dim;
        float *out_head = output_heads + head * head_dim;
        flash_decode_head_f32(q_head, key_layer, value_layer,
                              seq_len, head_dim, kv_len, kv_head, out_head);
    }
    return OC_OK;
}

OcError oc_flash_attention_decode_heads_f16(const float *query_heads,
                                             const uint16_t *key_layer,
                                             const uint16_t *value_layer,
                                             size_t seq_len,
                                             size_t head_dim,
                                             size_t kv_len,
                                             size_t num_heads,
                                             size_t kv_heads,
                                             float *output_heads)
{
    if (!query_heads || !key_layer || !value_layer || !output_heads)
        return OC_ERR_INVALID_ARG;
    if (head_dim == 0) return OC_ERR_INVALID_ARG;
    if (kv_heads == 0 || (num_heads % kv_heads) != 0)
        return OC_ERR_INVALID_ARG;

    size_t group_size = num_heads / kv_heads;

    for (size_t head = 0; head < num_heads; head++) {
        size_t kv_head = head / group_size;
        const float *q_head = query_heads + head * head_dim;
        float *out_head = output_heads + head * head_dim;
        flash_decode_head_f16(q_head, key_layer, value_layer,
                              seq_len, head_dim, kv_len, kv_head, out_head);
    }
    return OC_OK;
}

OcError oc_flash_attention_prefill_f32(const float *query,
                                        const float *key,
                                        const float *value,
                                        size_t q_seq_len,
                                        size_t kv_seq_len,
                                        size_t head_dim,
                                        float *output)
{
    if (!query || !key || !value || !output)
        return OC_ERR_INVALID_ARG;
    if (head_dim == 0) return OC_ERR_INVALID_ARG;

    float scale = 1.0f / sqrtf((float)head_dim);

    for (size_t q_i = 0; q_i < q_seq_len; q_i++) {
        const float *q_vec = query + q_i * head_dim;
        float *out_vec = output + q_i * head_dim;

        float run_max = -INFINITY;
        float run_sum = 0.0f;
        memset(out_vec, 0, head_dim * sizeof(float));

        size_t token = 0;
        while (token < kv_seq_len) {
            size_t block_end = token + OC_FLASH_BLOCK_SIZE;
            if (block_end > kv_seq_len) block_end = kv_seq_len;

            for (size_t t = token; t < block_end; t++) {
                const float *key_row = key + t * head_dim;
                float dot = 0.0f;
                for (size_t i = 0; i < head_dim; i++)
                    dot += q_vec[i] * key_row[i];
                float score = dot * scale;

                float new_max = (score > run_max) ? score : run_max;
                float exp_factor = expf(run_max - new_max);
                float exp_score = expf(score - new_max);

                if (exp_factor != 1.0f) {
                    for (size_t i = 0; i < head_dim; i++)
                        out_vec[i] *= exp_factor;
                }

                const float *value_row = value + t * head_dim;
                for (size_t i = 0; i < head_dim; i++)
                    out_vec[i] += exp_score * value_row[i];

                run_sum = run_sum * exp_factor + exp_score;
                run_max = new_max;
            }
            token = block_end;
        }

        if (run_sum > 0.0f) {
            float inv = 1.0f / run_sum;
            for (size_t i = 0; i < head_dim; i++)
                out_vec[i] *= inv;
        }
    }
    return OC_OK;
}
