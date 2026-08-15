#include "oxidize/qwen35_delta.h"

#include "oxidize/attn_kernels.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx512f")))
static void delta_head_avx512(float *matrix, const float *q, const float *k,
                              const float *v, float *out, size_t dv, size_t dk,
                              float beta, float decay)
{
    const __m512 vdec = _mm512_set1_ps(decay);
    size_t i = 0;
    for (; i + 4 <= dv; i += 4) {
        float *row0 = matrix + (i + 0) * dk;
        float *row1 = matrix + (i + 1) * dk;
        float *row2 = matrix + (i + 2) * dk;
        float *row3 = matrix + (i + 3) * dk;
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();
        size_t j = 0;
        for (; j + 16 <= dk; j += 16) {
            __m512 kv = _mm512_loadu_ps(k + j);
            __m512 r0 = _mm512_mul_ps(_mm512_loadu_ps(row0 + j), vdec);
            __m512 r1 = _mm512_mul_ps(_mm512_loadu_ps(row1 + j), vdec);
            __m512 r2 = _mm512_mul_ps(_mm512_loadu_ps(row2 + j), vdec);
            __m512 r3 = _mm512_mul_ps(_mm512_loadu_ps(row3 + j), vdec);
            _mm512_storeu_ps(row0 + j, r0);
            _mm512_storeu_ps(row1 + j, r1);
            _mm512_storeu_ps(row2 + j, r2);
            _mm512_storeu_ps(row3 + j, r3);
            acc0 = _mm512_fmadd_ps(r0, kv, acc0);
            acc1 = _mm512_fmadd_ps(r1, kv, acc1);
            acc2 = _mm512_fmadd_ps(r2, kv, acc2);
            acc3 = _mm512_fmadd_ps(r3, kv, acc3);
        }
        float sk0 = _mm512_reduce_add_ps(acc0);
        float sk1 = _mm512_reduce_add_ps(acc1);
        float sk2 = _mm512_reduce_add_ps(acc2);
        float sk3 = _mm512_reduce_add_ps(acc3);
        for (; j < dk; j++) {
            row0[j] *= decay; sk0 += row0[j] * k[j];
            row1[j] *= decay; sk1 += row1[j] * k[j];
            row2[j] *= decay; sk2 += row2[j] * k[j];
            row3[j] *= decay; sk3 += row3[j] * k[j];
        }
        const float d0 = (v[i + 0] - sk0) * beta;
        const float d1 = (v[i + 1] - sk1) * beta;
        const float d2 = (v[i + 2] - sk2) * beta;
        const float d3 = (v[i + 3] - sk3) * beta;
        const __m512 vd0 = _mm512_set1_ps(d0);
        const __m512 vd1 = _mm512_set1_ps(d1);
        const __m512 vd2 = _mm512_set1_ps(d2);
        const __m512 vd3 = _mm512_set1_ps(d3);
        acc0 = acc1 = acc2 = acc3 = _mm512_setzero_ps();
        j = 0;
        for (; j + 16 <= dk; j += 16) {
            __m512 kv = _mm512_loadu_ps(k + j);
            __m512 qv = _mm512_loadu_ps(q + j);
            __m512 r0 = _mm512_fmadd_ps(vd0, kv, _mm512_loadu_ps(row0 + j));
            __m512 r1 = _mm512_fmadd_ps(vd1, kv, _mm512_loadu_ps(row1 + j));
            __m512 r2 = _mm512_fmadd_ps(vd2, kv, _mm512_loadu_ps(row2 + j));
            __m512 r3 = _mm512_fmadd_ps(vd3, kv, _mm512_loadu_ps(row3 + j));
            _mm512_storeu_ps(row0 + j, r0);
            _mm512_storeu_ps(row1 + j, r1);
            _mm512_storeu_ps(row2 + j, r2);
            _mm512_storeu_ps(row3 + j, r3);
            acc0 = _mm512_fmadd_ps(r0, qv, acc0);
            acc1 = _mm512_fmadd_ps(r1, qv, acc1);
            acc2 = _mm512_fmadd_ps(r2, qv, acc2);
            acc3 = _mm512_fmadd_ps(r3, qv, acc3);
        }
        float o0 = _mm512_reduce_add_ps(acc0);
        float o1 = _mm512_reduce_add_ps(acc1);
        float o2 = _mm512_reduce_add_ps(acc2);
        float o3 = _mm512_reduce_add_ps(acc3);
        for (; j < dk; j++) {
            row0[j] += d0 * k[j]; o0 += row0[j] * q[j];
            row1[j] += d1 * k[j]; o1 += row1[j] * q[j];
            row2[j] += d2 * k[j]; o2 += row2[j] * q[j];
            row3[j] += d3 * k[j]; o3 += row3[j] * q[j];
        }
        out[i + 0] = o0;
        out[i + 1] = o1;
        out[i + 2] = o2;
        out[i + 3] = o3;
    }
    for (; i < dv; i++) {
        float *row = matrix + i * dk;
        __m512 acc_k = _mm512_setzero_ps();
        size_t j = 0;
        for (; j + 16 <= dk; j += 16) {
            __m512 r = _mm512_mul_ps(_mm512_loadu_ps(row + j), vdec);
            acc_k = _mm512_fmadd_ps(r, _mm512_loadu_ps(k + j), acc_k);
            _mm512_storeu_ps(row + j, r);
        }
        float state_k = _mm512_reduce_add_ps(acc_k);
        for (; j < dk; j++) {
            row[j] *= decay;
            state_k += row[j] * k[j];
        }
        float delta = (v[i] - state_k) * beta;
        __m512 vd = _mm512_set1_ps(delta);
        __m512 acc_q = _mm512_setzero_ps();
        j = 0;
        for (; j + 16 <= dk; j += 16) {
            __m512 r = _mm512_fmadd_ps(vd, _mm512_loadu_ps(k + j),
                                       _mm512_loadu_ps(row + j));
            _mm512_storeu_ps(row + j, r);
            acc_q = _mm512_fmadd_ps(r, _mm512_loadu_ps(q + j), acc_q);
        }
        float o = _mm512_reduce_add_ps(acc_q);
        for (; j < dk; j++) {
            row[j] += delta * k[j];
            o += row[j] * q[j];
        }
        out[i] = o;
    }
}
#endif

static bool checked_mul(size_t a, size_t b, size_t *out)
{
    if (a != 0 && b > SIZE_MAX / a) return false;
    *out = a * b;
    return true;
}

static bool geometry_sizes(const OcQwen35DeltaGeometry *g,
                           size_t *key_dim, size_t *value_dim,
                           size_t *conv_dim, size_t *conv_state_len,
                           size_t *recurrent_state_len)
{
    size_t twice_key;
    if (g == NULL || g->n_key_heads == 0 || g->n_value_heads == 0 ||
        g->key_head_dim == 0 || g->value_head_dim == 0 ||
        g->conv_kernel < 2 || g->n_value_heads % g->n_key_heads != 0) {
        return false;
    }
    if (!checked_mul(g->n_key_heads, g->key_head_dim, key_dim) ||
        !checked_mul(g->n_value_heads, g->value_head_dim, value_dim) ||
        !checked_mul(*key_dim, 2, &twice_key) ||
        twice_key > SIZE_MAX - *value_dim) {
        return false;
    }
    *conv_dim = twice_key + *value_dim;
    if (!checked_mul(*conv_dim, g->conv_kernel - 1, conv_state_len) ||
        !checked_mul(*value_dim, g->key_head_dim, recurrent_state_len)) {
        return false;
    }
    return true;
}

OcError oc_qwen35_delta_state_init(OcQwen35DeltaState *state,
                                   const OcQwen35DeltaGeometry *geometry,
                                   float *conv_state, size_t conv_state_len,
                                   float *recurrent_state,
                                   size_t recurrent_state_len)
{
    size_t key_dim, value_dim, conv_dim, want_conv, want_recurrent;
    if (state == NULL || !geometry_sizes(geometry, &key_dim, &value_dim,
                                         &conv_dim, &want_conv,
                                         &want_recurrent) ||
        conv_state == NULL || recurrent_state == NULL ||
        conv_state_len < want_conv || recurrent_state_len < want_recurrent) {
        return OC_ERR_INVALID_ARG;
    }
    (void)key_dim;
    (void)value_dim;
    (void)conv_dim;
    state->geometry = *geometry;
    state->conv_state = conv_state;
    state->conv_state_len = want_conv;
    state->recurrent_state = recurrent_state;
    state->recurrent_state_len = want_recurrent;
    oc_qwen35_delta_state_reset(state);
    return OC_OK;
}

void oc_qwen35_delta_state_reset(OcQwen35DeltaState *state)
{
    if (state == NULL) return;
    if (state->conv_state != NULL) {
        memset(state->conv_state, 0, state->conv_state_len * sizeof(float));
    }
    if (state->recurrent_state != NULL) {
        memset(state->recurrent_state, 0,
               state->recurrent_state_len * sizeof(float));
    }
}

void oc_qwen35_delta_state_free(OcQwen35DeltaState *state)
{
    if (state != NULL) memset(state, 0, sizeof(*state));
}

static float sigmoid_stable(float x)
{
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    float e = expf(x);
    return e / (1.0f + e);
}

static float softplus_stable(float x)
{
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

static float silu_stable(float x)
{
    if (x >= 0.0f) return x / (1.0f + expf(-x));
    float e = expf(x);
    return x * e / (1.0f + e);
}

typedef struct {
    const OcQwen35DeltaGeometry *g;
    float *state;
    const float *weight;
    const float *input;
    float *output;
} ConvCtx;

static void conv_channels(size_t begin, size_t end, size_t tid, void *user_data)
{
    (void)tid;
    ConvCtx *ctx = user_data;
    size_t kernel = ctx->g->conv_kernel;
    size_t history = kernel - 1;
    for (size_t channel = begin; channel < end; channel++) {
        float *past = ctx->state + channel * history;
        const float *weight = ctx->weight + channel * kernel;
        float sum = weight[kernel - 1] * ctx->input[channel];
        for (size_t i = 0; i < history; i++) sum += weight[i] * past[i];
        memmove(past, past + 1, (history - 1) * sizeof(float));
        past[history - 1] = ctx->input[channel];
        ctx->output[channel] = silu_stable(sum);
    }
}

typedef struct {
    const OcQwen35DeltaGeometry *g;
    float *conv_output;
    float eps;
} NormalizeCtx;

static void normalize_key_heads(size_t begin, size_t end, size_t tid,
                                void *user_data)
{
    (void)tid;
    NormalizeCtx *ctx = user_data;
    size_t dim = ctx->g->key_head_dim;
    size_t key_total = ctx->g->n_key_heads * dim;
    float q_scale = 1.0f / sqrtf((float)dim);
    for (size_t head = begin; head < end; head++) {
        float *q = ctx->conv_output + head * dim;
        float *k = ctx->conv_output + key_total + head * dim;
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            q_sum += q[i] * q[i];
            k_sum += k[i] * k[i];
        }
        float q_inv = 1.0f / sqrtf(q_sum > ctx->eps ? q_sum : ctx->eps);
        float k_inv = 1.0f / sqrtf(k_sum > ctx->eps ? k_sum : ctx->eps);
        for (size_t i = 0; i < dim; i++) {
            q[i] *= q_inv * q_scale;
            k[i] *= k_inv;
        }
    }
}

typedef struct {
    const OcQwen35DeltaGeometry *g;
    float *recurrent;
    const float *conv_output;
    const OcQwen35DeltaParams *params;
    const OcQwen35DeltaInput *input;
    float *output;
} DeltaCtx;

static void delta_heads(size_t begin, size_t end, size_t tid, void *user_data)
{
    (void)tid;
    DeltaCtx *ctx = user_data;
    size_t dk = ctx->g->key_head_dim;
    size_t dv = ctx->g->value_head_dim;
    size_t key_total = ctx->g->n_key_heads * dk;
    const float *values = ctx->conv_output + 2 * key_total;
    for (size_t head = begin; head < end; head++) {
        size_t value_heads_per_key =
            ctx->g->n_value_heads / ctx->g->n_key_heads;
        size_t key_head = head / value_heads_per_key;
        const float *q = ctx->conv_output + key_head * dk;
        const float *k = ctx->conv_output + key_total + key_head * dk;
        const float *v = values + head * dv;
        float *matrix = ctx->recurrent + head * dv * dk;
        float *out = ctx->output + head * dv;
        float beta = sigmoid_stable(ctx->input->beta[head]);
        float decay = expf(ctx->params->ssm_a[head] *
                           softplus_stable(ctx->input->alpha[head] +
                                           ctx->params->dt_bias[head]));
#if defined(__x86_64__) || defined(__i386__)
        static int has_avx512 = -1;
        if (has_avx512 < 0)
            has_avx512 = __builtin_cpu_supports("avx512f") ? 1 : 0;
        if (dk >= 16 && has_avx512) {
            delta_head_avx512(matrix, q, k, v, out, dv, dk, beta, decay);
            continue;
        }
#endif
        for (size_t i = 0; i < dv; i++) {
            float *row = matrix + i * dk;
            oc_attn_scale_f32(row, decay, dk);
            float state_k = oc_attn_dot_f32(row, k, dk);
            float delta = (v[i] - state_k) * beta;
            oc_attn_axpy_f32(row, k, delta, dk);
            out[i] = oc_attn_dot_f32(row, q, dk);
        }
    }
}

typedef struct {
    const OcQwen35DeltaGeometry *g;
    const OcQwen35DeltaParams *params;
    const float *gate;
    float *output;
} GateCtx;

static void gate_heads(size_t begin, size_t end, size_t tid, void *user_data)
{
    (void)tid;
    GateCtx *ctx = user_data;
    size_t dv = ctx->g->value_head_dim;
    for (size_t head = begin; head < end; head++) {
        float *out = ctx->output + head * dv;
        const float *gate = ctx->gate + head * dv;
        float sum = oc_attn_dot_f32(out, out, dv);
        float inv = 1.0f / sqrtf(sum / (float)dv + ctx->params->norm_eps);
        for (size_t i = 0; i < dv; i++) {
            out[i] = out[i] * inv * ctx->params->norm_weight[i] *
                     silu_stable(gate[i]);
        }
    }
}

OcError oc_qwen35_delta_step(OcQwen35DeltaState *state,
                             const OcQwen35DeltaParams *params,
                             const OcQwen35DeltaInput *input,
                             float *conv_output, size_t conv_output_len,
                             float *output, size_t output_len)
{
    size_t key_dim, value_dim, conv_dim, conv_state_len, recurrent_state_len;
    if (state == NULL || params == NULL || input == NULL ||
        !geometry_sizes(&state->geometry, &key_dim, &value_dim, &conv_dim,
                        &conv_state_len, &recurrent_state_len) ||
        state->conv_state == NULL || state->recurrent_state == NULL ||
        state->conv_state_len < conv_state_len ||
        state->recurrent_state_len < recurrent_state_len ||
        params->conv_weight == NULL || params->ssm_a == NULL ||
        params->dt_bias == NULL || params->norm_weight == NULL ||
        !isfinite(params->norm_eps) || params->norm_eps <= 0.0f ||
        params->conv_weight_len / state->geometry.conv_kernel < conv_dim ||
        params->ssm_a_len < state->geometry.n_value_heads ||
        params->dt_bias_len < state->geometry.n_value_heads ||
        params->norm_weight_len < state->geometry.value_head_dim ||
        input->qkv == NULL || input->gate == NULL || input->beta == NULL ||
        input->alpha == NULL || input->qkv_len < conv_dim ||
        input->gate_len < value_dim ||
        input->beta_len < state->geometry.n_value_heads ||
        input->alpha_len < state->geometry.n_value_heads ||
        conv_output == NULL || conv_output_len < conv_dim ||
        output == NULL || output_len < value_dim) {
        return OC_ERR_INVALID_ARG;
    }
    /* Prefill calls this once per token. Nested pool dispatches (4 per
     * token × ~48 recurrent layers) cost more than the DeltaNet work. */
    ConvCtx conv = {&state->geometry, state->conv_state,
                    params->conv_weight, input->qkv, conv_output};
    conv_channels(0, conv_dim, 0, &conv);

    NormalizeCtx normalize = {&state->geometry, conv_output, params->norm_eps};
    normalize_key_heads(0, state->geometry.n_key_heads, 0, &normalize);

    DeltaCtx delta = {&state->geometry, state->recurrent_state, conv_output,
                      params, input, output};
    delta_heads(0, state->geometry.n_value_heads, 0, &delta);

    GateCtx gate = {&state->geometry, params, input->gate, output};
    gate_heads(0, state->geometry.n_value_heads, 0, &gate);
    return OC_OK;
}
