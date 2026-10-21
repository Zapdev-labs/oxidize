#define _GNU_SOURCE
/*
 * eagle4_hf.c — run a published EAGLE-4 BF16 SafeTensors draft.
 *
 * Weights stay packed. On AVX-512 BF16 hosts the GEMV is vdpbf16ps over
 * the rows, split across the compute pool. The Qwen3.8-27B draft's
 * Q/K/V projections read the concatenated embedding and hidden state.
 */
#include "oxidize/eagle4.h"

#include "oxidize/flash_attention.h"
#include "oxidize/parallel.h"
#include "oxidize/safetensors.h"
#include "oxidize/simd.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__unix__)
#include <sys/mman.h>
#endif

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#define E4_HF_MAX_DIM 262144u

typedef struct E4HfW {
    const uint16_t *data;
    size_t rows;
    size_t cols;
} E4HfW;

typedef struct E4HfModel {
    OcSafetensorsFile st;
    E4HfW embed;
    E4HfW fc;
    E4HfW fc_norm[3];
    E4HfW hidden_norm;
    E4HfW input_ln;
    E4HfW post_ln;
    E4HfW out_norm;
    E4HfW q, k, v, o;
    E4HfW gate, up, down;
    E4HfW lm_down, lm_score, lm_head;
    size_t hidden;
    size_t vocab;
    size_t rank;
    size_t inter;
    size_t n_heads;
    size_t n_kv;
    size_t head_dim;
    size_t window;
    size_t n_extract;
    float eps;
    float rope_theta;
    float *arena;
    uint16_t *x_bf;
    float *g;
    float *emb;
    float *emb_n;
    float *g_n;
    float *cat;
    float *qbuf;
    float *kbuf;
    float *vbuf;
    float *attn;
    float *hidden_buf;
    float *ffn;
    float *gate_buf;
    float *up_buf;
    float *act;
    float *mlp;
    float *proj;
    float *scores;
    float *fused_in;
    float *fc_out;
    float *kv_k;
    float *kv_v;
    size_t kv_seq;
    size_t kv_cap;
    size_t position;
    double us_qkv;
    double us_mlp;
    double us_attn;
    double us_head;
} E4HfModel;

static double now_us(void)
{
    struct timespec ts;
    if (timespec_get(&ts, TIME_UTC) != TIME_UTC) return 0.0;
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

static float bf16_f32(uint16_t h)
{
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static uint16_t f32_bf16(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    uint32_t lsb = (u >> 16) & 1u;
    uint32_t rounding = 0x7fffu + lsb;
    if (u > 0xffff0000u - rounding) return (uint16_t)(u >> 16);
    u += rounding;
    return (uint16_t)(u >> 16);
}

#if defined(__x86_64__)
__attribute__((target("avx512bf16,avx512f")))
static float dot_bf16_avx512(const uint16_t *w, const uint16_t *x, size_t n)
{
    __m512 a0 = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        a0 = _mm512_dpbf16_ps(a0,
            (__m512bh)_mm512_loadu_si512(w + i),
            (__m512bh)_mm512_loadu_si512(x + i));
    }
    float sum = _mm512_reduce_add_ps(a0);
    for (; i < n; i++) sum += bf16_f32(w[i]) * bf16_f32(x[i]);
    return sum;
}

/* Four rows share each x tile so the activation stays in registers and the
 * four weight streams stay in flight. */
__attribute__((target("avx512bf16,avx512f")))
static void gemv_bf16_avx512_range(const uint16_t *w, const uint16_t *x,
                                   size_t cols, float *out, size_t begin, size_t end)
{
    size_t r = begin;
    for (; r + 4 <= end; r += 4) {
        const uint16_t *w0 = w + (r + 0) * cols;
        const uint16_t *w1 = w + (r + 1) * cols;
        const uint16_t *w2 = w + (r + 2) * cols;
        const uint16_t *w3 = w + (r + 3) * cols;
        __m512 a0 = _mm512_setzero_ps();
        __m512 a1 = _mm512_setzero_ps();
        __m512 b0 = _mm512_setzero_ps();
        __m512 b1 = _mm512_setzero_ps();
        __m512 c0 = _mm512_setzero_ps();
        __m512 c1 = _mm512_setzero_ps();
        __m512 d0 = _mm512_setzero_ps();
        __m512 d1 = _mm512_setzero_ps();
        size_t i = 0;
        for (; i + 64 <= cols; i += 64) {
            __m512bh x0 = (__m512bh)_mm512_loadu_si512(x + i);
            __m512bh x1 = (__m512bh)_mm512_loadu_si512(x + i + 32);
            a0 = _mm512_dpbf16_ps(a0, (__m512bh)_mm512_loadu_si512(w0 + i), x0);
            a1 = _mm512_dpbf16_ps(a1, (__m512bh)_mm512_loadu_si512(w0 + i + 32), x1);
            b0 = _mm512_dpbf16_ps(b0, (__m512bh)_mm512_loadu_si512(w1 + i), x0);
            b1 = _mm512_dpbf16_ps(b1, (__m512bh)_mm512_loadu_si512(w1 + i + 32), x1);
            c0 = _mm512_dpbf16_ps(c0, (__m512bh)_mm512_loadu_si512(w2 + i), x0);
            c1 = _mm512_dpbf16_ps(c1, (__m512bh)_mm512_loadu_si512(w2 + i + 32), x1);
            d0 = _mm512_dpbf16_ps(d0, (__m512bh)_mm512_loadu_si512(w3 + i), x0);
            d1 = _mm512_dpbf16_ps(d1, (__m512bh)_mm512_loadu_si512(w3 + i + 32), x1);
        }
        for (; i + 32 <= cols; i += 32) {
            __m512bh xv = (__m512bh)_mm512_loadu_si512(x + i);
            a0 = _mm512_dpbf16_ps(a0, (__m512bh)_mm512_loadu_si512(w0 + i), xv);
            b0 = _mm512_dpbf16_ps(b0, (__m512bh)_mm512_loadu_si512(w1 + i), xv);
            c0 = _mm512_dpbf16_ps(c0, (__m512bh)_mm512_loadu_si512(w2 + i), xv);
            d0 = _mm512_dpbf16_ps(d0, (__m512bh)_mm512_loadu_si512(w3 + i), xv);
        }
        out[r] = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
        out[r + 1] = _mm512_reduce_add_ps(_mm512_add_ps(b0, b1));
        out[r + 2] = _mm512_reduce_add_ps(_mm512_add_ps(c0, c1));
        out[r + 3] = _mm512_reduce_add_ps(_mm512_add_ps(d0, d1));
        if (i < cols) {
            for (size_t k = 0; k < 4; k++) {
                const uint16_t *row = w + (r + k) * cols;
                float tail = 0.0f;
                for (size_t j = i; j < cols; j++)
                    tail += bf16_f32(row[j]) * bf16_f32(x[j]);
                out[r + k] += tail;
            }
        }
    }
    for (; r < end; r++)
        out[r] = dot_bf16_avx512(w + r * cols, x, cols);
}
#endif

static float dot_f32_bf16_row(const uint16_t *w, const float *x, size_t n)
{
    float r = 0.0f;
    if (oc_simd_try_dflash2_dot_bf16(w, x, n, &r)) return r;
    size_t i;
    for (i = 0; i < n; i++) r += bf16_f32(w[i]) * x[i];
    return r;
}

static int g_avx512_bf16 = -1;

static int have_avx512_bf16(void)
{
#if defined(__x86_64__)
    if (g_avx512_bf16 < 0) {
        __builtin_cpu_init();
        g_avx512_bf16 = __builtin_cpu_supports("avx512bf16") ? 1 : 0;
    }
    return g_avx512_bf16;
#else
    return 0;
#endif
}

typedef struct E4Gemv {
    const uint16_t *w;
    const uint16_t *xbf;
    const float *xf;
    size_t cols;
    float *out;
    int bf16_dp;
} E4Gemv;

static void gemv_slice(size_t begin, size_t end, size_t tid, void *ud)
{
    E4Gemv *j = (E4Gemv *)ud;
    size_t r;
    (void)tid;
    if (j->bf16_dp) {
#if defined(__x86_64__)
        gemv_bf16_avx512_range(j->w, j->xbf, j->cols, j->out, begin, end);
        return;
#else
        (void)begin;
        (void)end;
#endif
    }
    for (r = begin; r < end; r++)
        j->out[r] = dot_f32_bf16_row(j->w + r * j->cols, j->xf, j->cols);
}

static void f32_to_bf16_n(const float *x, uint16_t *dst, size_t n)
{
    size_t i;
    for (i = 0; i < n; i++) dst[i] = f32_bf16(x[i]);
}

static void gemv_bf16(E4HfModel *m, const E4HfW *w, const float *x, float *out)
{
    E4Gemv job;
    if (!w->data || w->rows == 0 || w->cols == 0) {
        memset(out, 0, w->rows * sizeof(float));
        return;
    }
    job.w = w->data;
    job.cols = w->cols;
    job.out = out;
    job.xf = x;
    job.bf16_dp = have_avx512_bf16();
    job.xbf = NULL;
    if (job.bf16_dp) {
        f32_to_bf16_n(x, m->x_bf, w->cols);
        job.xbf = m->x_bf;
    }
    if (w->rows >= 512) {
        oc_parallel_for(w->rows, gemv_slice, &job);
        return;
    }
    gemv_slice(0, w->rows, 0, &job);
}

static void rms_bf16(const float *x, const uint16_t *w, float *out, size_t n, float eps)
{
    float ss = 0.0f;
    float inv;
    size_t i;
    for (i = 0; i < n; i++) ss += x[i] * x[i];
    inv = 1.0f / sqrtf(ss / (float)n + eps);
    if (!w) {
        for (i = 0; i < n; i++) out[i] = x[i] * inv;
        return;
    }
    for (i = 0; i < n; i++) out[i] = x[i] * inv * bf16_f32(w[i]);
}

static void rope_pair(float *q, size_t n_heads, size_t hd, size_t pos, float theta)
{
    size_t h, d;
    if (hd < 2) return;
    for (h = 0; h < n_heads; h++) {
        float *qh = q + h * hd;
        for (d = 0; d + 1 < hd; d += 2) {
            float freq = (float)pos / powf(theta, (float)(d / 2) / (float)(hd / 2));
            float c = cosf(freq);
            float s = sinf(freq);
            float a = qh[d];
            float b = qh[d + 1];
            qh[d] = a * c - b * s;
            qh[d + 1] = a * s + b * c;
        }
    }
}

static int bind_tensor(const OcSafetensorsFile *st, const char *name, E4HfW *w, int required)
{
    const OcSafetensorsTensor *t = NULL;
    const void *data = NULL;
    if (oc_safetensors_get_tensor(st, name, &t) != OC_OK) return required ? 0 : 1;
    if (strcmp(t->dtype, "BF16") != 0) return 0;
    if (oc_safetensors_get_tensor_data(st, t, &data) != OC_OK) return 0;
    if (((uintptr_t)data & 1u) != 0) return 0;
    w->data = (const uint16_t *)data;
    if (t->n_dims == 1) {
        w->rows = (size_t)t->shape[0];
        w->cols = 1;
    } else if (t->n_dims == 2) {
        w->rows = (size_t)t->shape[0];
        w->cols = (size_t)t->shape[1];
    } else {
        return 0;
    }
    return 1;
}

static void hf_free(E4HfModel *m)
{
    if (!m) return;
    oc_safetensors_close(&m->st);
    free(m->arena);
    free(m->x_bf);
    free(m->kv_k);
    free(m->kv_v);
    memset(m, 0, sizeof(*m));
}

static OcError hf_open(E4HfModel *m, const char *path)
{
    size_t h, max_cols, kv_len, cap, total;
    float *cursor;
    OcError err;
    memset(m, 0, sizeof(*m));
    err = oc_safetensors_open(path, &m->st);
    if (err != OC_OK) return err;
#if defined(__unix__)
    if (m->st.mmapped && m->st.raw_data && m->st.file_size > m->st.data_start) {
        uint8_t *base = (uint8_t *)m->st.raw_data - m->st.data_start;
        madvise(base, (size_t)m->st.file_size, MADV_HUGEPAGE);
    }
#endif
    if (!bind_tensor(&m->st, "embed_tokens.weight", &m->embed, 1) ||
        !bind_tensor(&m->st, "fc.weight", &m->fc, 1) ||
        !bind_tensor(&m->st, "layers.0.self_attn.q_proj.weight", &m->q, 1) ||
        !bind_tensor(&m->st, "layers.0.self_attn.k_proj.weight", &m->k, 1) ||
        !bind_tensor(&m->st, "layers.0.self_attn.v_proj.weight", &m->v, 1) ||
        !bind_tensor(&m->st, "layers.0.self_attn.o_proj.weight", &m->o, 1) ||
        !bind_tensor(&m->st, "layers.0.mlp.gate_proj.weight", &m->gate, 1) ||
        !bind_tensor(&m->st, "layers.0.mlp.up_proj.weight", &m->up, 1) ||
        !bind_tensor(&m->st, "layers.0.mlp.down_proj.weight", &m->down, 1) ||
        !bind_tensor(&m->st, "efficient_lm_head_down_proj.weight", &m->lm_down, 1) ||
        !bind_tensor(&m->st, "efficient_lm_head.weight", &m->lm_score, 1) ||
        !bind_tensor(&m->st, "lm_head.weight", &m->lm_head, 1)) {
        hf_free(m);
        return OC_ERR_FORMAT;
    }
    bind_tensor(&m->st, "fc_norm.0.weight", &m->fc_norm[0], 0);
    bind_tensor(&m->st, "fc_norm.1.weight", &m->fc_norm[1], 0);
    bind_tensor(&m->st, "fc_norm.2.weight", &m->fc_norm[2], 0);
    bind_tensor(&m->st, "layers.0.hidden_norm.weight", &m->hidden_norm, 0);
    bind_tensor(&m->st, "layers.0.input_layernorm.weight", &m->input_ln, 0);
    bind_tensor(&m->st, "layers.0.post_attention_layernorm.weight", &m->post_ln, 0);
    bind_tensor(&m->st, "norm.weight", &m->out_norm, 0);

    h = m->embed.cols;
    if (h == 0 || m->fc.rows != h || m->q.cols != h * 2u || m->q.rows == 0) {
        hf_free(m);
        return OC_ERR_FORMAT;
    }
    m->hidden = h;
    m->vocab = m->embed.rows;
    m->rank = m->lm_down.rows;
    m->inter = m->gate.rows;
    m->head_dim = 128;
    if (h % m->head_dim != 0) m->head_dim = h;
    m->n_heads = m->q.rows / m->head_dim;
    m->n_kv = m->k.rows / m->head_dim;
    if (m->n_heads == 0 || m->n_kv == 0 || (m->n_heads % m->n_kv) != 0) {
        hf_free(m);
        return OC_ERR_FORMAT;
    }
    m->window = 4096;
    m->n_extract = m->fc.cols / h;
    if (m->n_extract == 0 || m->n_extract > 3 || m->fc.cols != m->n_extract * h) {
        hf_free(m);
        return OC_ERR_FORMAT;
    }
    m->eps = 1e-6f;
    m->rope_theta = 10000000.0f;

    max_cols = m->fc.cols;
    if (m->q.cols > max_cols) max_cols = m->q.cols;
    if (m->down.cols > max_cols) max_cols = m->down.cols;
    m->x_bf = (uint16_t *)malloc(max_cols * sizeof(uint16_t));
    kv_len = m->n_kv * m->head_dim;
    cap = m->window + 8u;
    m->kv_cap = cap;
    m->kv_k = (float *)calloc(cap * kv_len, sizeof(float));
    m->kv_v = (float *)calloc(cap * kv_len, sizeof(float));
    total = h * 16u + m->q.rows + m->k.rows + m->v.rows + m->inter * 3u +
            m->rank + m->vocab + m->fc.cols;
    m->arena = (float *)calloc(total ? total : 1u, sizeof(float));
    if (!m->x_bf || !m->kv_k || !m->kv_v || !m->arena) {
        hf_free(m);
        return OC_ERR_OOM;
    }
    cursor = m->arena;
#define TAKE(dst, n) do { (dst) = cursor; cursor += (n); } while (0)
    TAKE(m->g, h);
    TAKE(m->emb, h);
    TAKE(m->emb_n, h);
    TAKE(m->g_n, h);
    TAKE(m->cat, h * 2u);
    TAKE(m->qbuf, m->q.rows);
    TAKE(m->kbuf, m->k.rows);
    TAKE(m->vbuf, m->v.rows);
    TAKE(m->attn, m->q.rows);
    TAKE(m->hidden_buf, h);
    TAKE(m->ffn, h);
    TAKE(m->gate_buf, m->inter);
    TAKE(m->up_buf, m->inter);
    TAKE(m->act, m->inter);
    TAKE(m->mlp, h);
    TAKE(m->proj, m->rank);
    TAKE(m->scores, m->vocab);
    TAKE(m->fused_in, m->fc.cols);
    TAKE(m->fc_out, h);
#undef TAKE
    return OC_OK;
}

static void encode(E4HfModel *m, const float *features)
{
    size_t h = m->hidden;
    size_t i;
    for (i = 0; i < m->n_extract; i++) {
        rms_bf16(features + i * h,
                 m->fc_norm[i].data, m->fused_in + i * h, h, m->eps);
    }
    gemv_bf16(m, &m->fc, m->fused_in, m->fc_out);
    memcpy(m->g, m->fc_out, h * sizeof(float));
}

static uint32_t top1(const float *s, size_t n)
{
    size_t i, best = 0;
    for (i = 1; i < n; i++)
        if (s[i] > s[best]) best = i;
    return (uint32_t)best;
}

static void forward_token(E4HfModel *m, uint32_t token)
{
    size_t h = m->hidden;
    size_t hd = m->head_dim;
    size_t kv_len = m->n_kv * hd;
    size_t seq, start, nwin;
    double t0;
    const uint16_t *erow = m->embed.data + (size_t)token * h;
    size_t i;

    for (i = 0; i < h; i++) m->emb[i] = bf16_f32(erow[i]);
    rms_bf16(m->emb, m->input_ln.data, m->emb_n, h, m->eps);
    rms_bf16(m->g, m->hidden_norm.data, m->g_n, h, m->eps);
    memcpy(m->cat, m->emb_n, h * sizeof(float));
    memcpy(m->cat + h, m->g_n, h * sizeof(float));

    t0 = now_us();
    gemv_bf16(m, &m->q, m->cat, m->qbuf);
    gemv_bf16(m, &m->k, m->cat, m->kbuf);
    gemv_bf16(m, &m->v, m->cat, m->vbuf);
    m->us_qkv += now_us() - t0;

    rope_pair(m->qbuf, m->n_heads, hd, m->position, m->rope_theta);
    rope_pair(m->kbuf, m->n_kv, hd, m->position, m->rope_theta);
    if (m->kv_seq < m->kv_cap) {
        memcpy(m->kv_k + m->kv_seq * kv_len, m->kbuf, kv_len * sizeof(float));
        memcpy(m->kv_v + m->kv_seq * kv_len, m->vbuf, kv_len * sizeof(float));
        m->kv_seq++;
    }
    seq = m->kv_seq;
    start = 0;
    if (seq > m->window) start = seq - m->window;
    nwin = seq - start;
    t0 = now_us();
    oc_flash_attention_decode_heads_f32(
        m->qbuf, m->kv_k + start * kv_len, m->kv_v + start * kv_len,
        nwin, hd, kv_len, m->n_heads, m->n_kv, m->attn);
    gemv_bf16(m, &m->o, m->attn, m->hidden_buf);
    m->us_attn += now_us() - t0;
    for (i = 0; i < h; i++) m->hidden_buf[i] += m->g[i];

    t0 = now_us();
    rms_bf16(m->hidden_buf, m->post_ln.data, m->ffn, h, m->eps);
    gemv_bf16(m, &m->gate, m->ffn, m->gate_buf);
    gemv_bf16(m, &m->up, m->ffn, m->up_buf);
    for (i = 0; i < m->inter; i++) {
        float z = m->gate_buf[i];
        m->act[i] = (z / (1.0f + expf(-z))) * m->up_buf[i];
    }
    gemv_bf16(m, &m->down, m->act, m->mlp);
    m->us_mlp += now_us() - t0;
    for (i = 0; i < h; i++) m->g[i] = m->hidden_buf[i] + m->mlp[i];

    t0 = now_us();
    rms_bf16(m->g, m->out_norm.data, m->hidden_buf, h, m->eps);
    gemv_bf16(m, &m->lm_down, m->hidden_buf, m->proj);
    gemv_bf16(m, &m->lm_score, m->proj, m->scores);
    m->us_head += now_us() - t0;
    m->position++;
}

OcError oc_eagle4_hf_bench(const char *path, int warmup, int iters, OcEagle4HfBench *out)
{
    E4HfModel model;
    float *features = NULL;
    OcError err;
    int i;
    double t0, t1, fault0;
    size_t params = 0;
    if (!path || !out) return OC_ERR_INVALID_ARG;
    if (warmup < 1) warmup = 1;
    if (iters < 1) iters = 1;
    memset(out, 0, sizeof(*out));
    err = hf_open(&model, path);
    if (err != OC_OK) return err;
    features = (float *)malloc(model.fc.cols * sizeof(float));
    if (!features) {
        hf_free(&model);
        return OC_ERR_OOM;
    }
    for (i = 0; i < (int)model.fc.cols; i++)
        features[i] = ((i * 17) % 100) / 500.0f - 0.1f;

    fault0 = now_us();
    encode(&model, features);
    forward_token(&model, 1);
    out->fault_us = now_us() - fault0;
    model.kv_seq = 0;
    model.position = 0;
    model.us_qkv = model.us_mlp = model.us_attn = model.us_head = 0;

    t0 = now_us();
    encode(&model, features);
    t1 = now_us();
    out->encode_us = t1 - t0;

    for (i = 0; i < warmup; i++) forward_token(&model, (uint32_t)(i + 1));
    model.us_qkv = model.us_mlp = model.us_attn = model.us_head = 0;
    t0 = now_us();
    for (i = 0; i < iters; i++)
        forward_token(&model, (uint32_t)((i * 17 + 3) % 4096));
    t1 = now_us();
    out->token_us = (t1 - t0) / (double)iters;
    out->tok_per_s = out->token_us > 0.0 ? 1e6 / out->token_us : 0.0;
    out->qkv_us = model.us_qkv / (double)iters;
    out->mlp_us = model.us_mlp / (double)iters;
    out->attn_us = model.us_attn / (double)iters;
    out->head_us = model.us_head / (double)iters;
    out->hidden = model.hidden;
    out->vocab = model.vocab;
    out->top_token = top1(model.scores, model.vocab);
    out->threads = (int)oc_parallel_n_threads();
    params = model.embed.rows * model.embed.cols;
    params += model.fc.rows * model.fc.cols;
    params += model.q.rows * model.q.cols;
    params += model.k.rows * model.k.cols;
    params += model.v.rows * model.v.cols;
    params += model.o.rows * model.o.cols;
    params += model.gate.rows * model.gate.cols;
    params += model.up.rows * model.up.cols;
    params += model.down.rows * model.down.cols;
    params += model.lm_down.rows * model.lm_down.cols;
    params += model.lm_score.rows * model.lm_score.cols;
    params += model.lm_head.rows * model.lm_head.cols;
    out->params = params;
    free(features);
    hf_free(&model);
    return OC_OK;
}
