/*
 * eagle4.c — EAGLE-4 drafter, harness, SafeTensors training, GGUF export.
 */
#include "oxidize/eagle4.h"
#include "oxidize/util/checked.h"
#include "arch_ops.h"

#include "oxidize/flash_attention.h"
#include "oxidize/gguf.h"
#include "oxidize/gguf_writer.h"
#include "oxidize/parallel.h"
#include "oxidize/safetensors.h"
#include "oxidize/simd.h"
#include "oxidize/util/clock.h"

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define E4_OWNED 96
#define E4_CFG_N 28
#define E4_SEL_MAX 512
#define E4_MAX_DIM (1u << 24)

typedef struct E4Scored {
    float score;
    uint32_t id;
} E4Scored;

typedef struct E4Kv {
    float *keys;
    float *values;
    size_t seq_len;
    size_t capacity;
    size_t kv_len;
    size_t window; /* rows kept after a trim; 0 = unbounded */
    size_t base;   /* rows trimmed off the front so far */
    int hold;      /* nonzero: never trim (a snapshot may roll back) */
} E4Kv;

typedef struct E4State {
    size_t seq[OC_EAGLE4_MAX_LAYERS];
    size_t position;
    uint32_t snap;
} E4State;

typedef struct E4Impl {
    float *arena;
    float *fused_in;
    float *fc_out;
    float *g;
    float *g_before;
    float *emb;
    float *emb_n;
    float *g_n;
    float *cat;
    float *x;
    float *q;
    float *k;
    float *v;
    float *attn;
    float *pre_mlp;
    float *hidden;
    float *ffn;
    float *gate;
    float *up;
    float *act;
    float *mlp;
    float *proj;
    float *scores;
    float *logits;
    float *dh;
    float *snaps;
    /* Per-snapshot KV row each tree node appended (k then v per layer),
     * its absolute position, and the parent snapshot (-1 = root). Tree
     * siblings share KV offsets, so restoring a node rewrites its path. */
    float *snap_kv;
    size_t snap_pos[OC_EAGLE4_MAX_NODES + 2u][OC_EAGLE4_MAX_LAYERS];
    int32_t snap_parent[OC_EAGLE4_MAX_NODES + 2u];
    size_t n_snaps;
    size_t snap_used;
    size_t feat_width;
    size_t q_size;
    size_t kv_size;
    size_t snap_layers; /* layer stride of snap_kv */
    E4Scored *heap;
    E4Kv kv[OC_EAGLE4_MAX_LAYERS];
    void *owned[E4_OWNED];
    size_t n_owned;
    uint32_t top_ids[OC_EAGLE4_TOP_KEEP];
    float top_probs[OC_EAGLE4_TOP_KEEP];
    uint32_t n_top;
} E4Impl;

typedef struct E4Ready {
    int32_t tree_idx;
    float path;
    E4State state;
    uint32_t tok[OC_EAGLE4_TOP_KEEP];
    float pr[OC_EAGLE4_TOP_KEEP];
    uint32_t n_tok;
} E4Ready;

static E4Impl *impl_of(OcEagle4Draft *m)
{
    return m ? (E4Impl *)m->impl : NULL;
}

static const E4Impl *impl_of_c(const OcEagle4Draft *m)
{
    return m ? (const E4Impl *)m->impl : NULL;
}

static void *e4_own(OcEagle4Draft *m, size_t bytes)
{
    E4Impl *im = impl_of(m);
    void *p;
    if (!im || im->n_owned >= E4_OWNED) return NULL;
    p = calloc(1, bytes ? bytes : 1);
    if (!p) return NULL;
    im->owned[im->n_owned++] = p;
    return p;
}

static float dot_f(const float *a, const float *b, size_t n)
{
    float r = 0.0f;
    size_t i;
    if (n && oc_simd_try_dflash2_dot_f32(a, b, n, &r)) return r;
    for (i = 0; i < n; i++) r += a[i] * b[i];
    return r;
}

typedef struct E4Gemv {
    const float *w;
    size_t cols;
    const float *x;
    float *out;
} E4Gemv;

static void gemv_slice(size_t begin, size_t end, size_t tid, void *ud)
{
    E4Gemv *j = (E4Gemv *)ud;
    size_t r;
    (void)tid;
    for (r = begin; r < end; r++)
        j->out[r] = dot_f(j->w + r * j->cols, j->x, j->cols);
}

static void gemv(const float *w, size_t rows, size_t cols, const float *x, float *out)
{
    if (!out || rows == 0) return;
    if (!w || !x || cols == 0) {
        memset(out, 0, rows * sizeof(float));
        return;
    }
    /* Thread dispatch swamps a rank-64 vocabulary scan. Parallelize only
     * the full-width head, which is large enough to hide the pool. */
    if (rows * cols >= (1u << 22)) {
        E4Gemv j;
        j.w = w;
        j.cols = cols;
        j.x = x;
        j.out = out;
        oc_parallel_for(rows, gemv_slice, &j);
        return;
    }
    if (oc_simd_try_dflash2_gemv_rows_f32(w, rows, cols, x, out)) return;
    {
        size_t r;
        for (r = 0; r < rows; r++) out[r] = dot_f(w + r * cols, x, cols);
    }
}

static void gemv_add_t(const float *w, size_t rows, size_t cols,
                       const float *dh, float *dx)
{
    size_t r, c;
    for (r = 0; r < rows; r++) {
        float s = dh[r];
        const float *row = w + r * cols;
        for (c = 0; c < cols; c++) dx[c] += row[c] * s;
    }
}

static void sgd_outer(float *w, size_t rows, size_t cols, const float *dh,
                      const float *x, float lr)
{
    size_t r, c;
    for (r = 0; r < rows; r++) {
        float s = lr * dh[r];
        float *row = w + r * cols;
        for (c = 0; c < cols; c++) row[c] -= s * x[c];
    }
}

static void rms_forward(const float *x, const float *w, float *out, size_t n, float eps)
{
    float ss = 0.0f;
    float inv;
    size_t i;
    if (!w) {
        memcpy(out, x, n * sizeof(float));
        return;
    }
    for (i = 0; i < n; i++) ss += x[i] * x[i];
    inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
}

static float silu_d(float x)
{
    float sg = 1.0f / (1.0f + expf(-x));
    return sg * (1.0f + x * (1.0f - sg));
}

static int scored_worse(E4Scored a, E4Scored b)
{
    if (a.score < b.score) return 1;
    if (a.score > b.score) return 0;
    return a.id > b.id;
}

static int scored_better(E4Scored a, E4Scored b)
{
    if (a.score > b.score) return 1;
    if (a.score < b.score) return 0;
    return a.id < b.id;
}

static void sift_down(E4Scored *h, uint32_t n, uint32_t i)
{
    for (;;) {
        uint32_t l = i * 2u + 1u;
        uint32_t r = l + 1u;
        uint32_t w = i;
        if (l < n && scored_worse(h[l], h[w])) w = l;
        if (r < n && scored_worse(h[r], h[w])) w = r;
        if (w == i) break;
        {
            E4Scored t = h[i];
            h[i] = h[w];
            h[w] = t;
        }
        i = w;
    }
}

static void heap_add(E4Scored *h, uint32_t *n, uint32_t k, float score, uint32_t id)
{
    E4Scored s;
    s.score = score;
    s.id = id;
    if (*n < k) {
        uint32_t i = *n;
        h[(*n)++] = s;
        while (i > 0) {
            uint32_t p = (i - 1u) / 2u;
            if (!scored_worse(h[i], h[p])) break;
            {
                E4Scored t = h[i];
                h[i] = h[p];
                h[p] = t;
            }
            i = p;
        }
        return;
    }
    if (!scored_worse(h[0], s)) return;
    h[0] = s;
    sift_down(h, *n, 0);
}

static void sort_best(E4Scored *a, uint32_t n)
{
    uint32_t i;
    for (i = 1; i < n; i++) {
        E4Scored k = a[i];
        uint32_t j = i;
        while (j > 0 && scored_better(k, a[j - 1u])) {
            a[j] = a[j - 1u];
            j--;
        }
        a[j] = k;
    }
}

static void rope_inplace(float *q, size_t n_heads, size_t hd, size_t pos, float theta)
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

static OcError kv_append(E4Kv *c, const float *k, const float *v)
{
    size_t bytes;
    if (!c->hold && c->window > 0 && c->seq_len >= c->window + 16u) {
        /* Attention only reads the last `window` rows; drop the rest. */
        size_t drop = c->seq_len - c->window;
        memmove(c->keys, c->keys + drop * c->kv_len, c->window * c->kv_len * sizeof(float));
        memmove(c->values, c->values + drop * c->kv_len,
                c->window * c->kv_len * sizeof(float));
        c->seq_len = c->window;
        c->base += drop;
    }
    if (c->seq_len >= c->capacity) {
        size_t ncap = c->capacity ? c->capacity * 2u : 32u;
        float *nk;
        float *nv;
        if (ncap <= c->seq_len) ncap = c->seq_len + 1u;
        bytes = ncap * c->kv_len * sizeof(float);
        nk = (float *)realloc(c->keys, bytes);
        if (!nk) return OC_ERR_OOM;
        c->keys = nk;
        nv = (float *)realloc(c->values, bytes);
        if (!nv) return OC_ERR_OOM;
        c->values = nv;
        c->capacity = ncap;
    }
    memcpy(c->keys + c->seq_len * c->kv_len, k, c->kv_len * sizeof(float));
    memcpy(c->values + c->seq_len * c->kv_len, v, c->kv_len * sizeof(float));
    c->seq_len++;
    return OC_OK;
}

static void capture_state(OcEagle4Draft *m, E4State *s)
{
    E4Impl *im = impl_of(m);
    size_t h = m->config.hidden_size;
    size_t i;
    s->position = m->position;
    for (i = 0; i < m->config.num_hidden_layers && i < OC_EAGLE4_MAX_LAYERS; i++)
        s->seq[i] = im->kv[i].base + im->kv[i].seq_len;
    if (im->snap_used >= im->n_snaps) s->snap = (uint32_t)im->n_snaps - 1u;
    else s->snap = (uint32_t)im->snap_used++;
    memcpy(im->snaps + (size_t)s->snap * h, im->g, h * sizeof(float));
    im->snap_parent[s->snap] = -1;
}

/* Record the KV row the last forward appended, as a child of `parent`. */
static void capture_node(OcEagle4Draft *m, E4State *s, const E4State *parent)
{
    E4Impl *im = impl_of(m);
    size_t kvl = im->kv_size;
    size_t i;
    capture_state(m, s);
    if (!im->snap_kv || s->snap == parent->snap) return;
    im->snap_parent[s->snap] = (int32_t)parent->snap;
    for (i = 0; i < m->config.num_hidden_layers && i < OC_EAGLE4_MAX_LAYERS; i++) {
        E4Kv *c = &im->kv[i];
        float *dst = im->snap_kv + ((size_t)s->snap * im->snap_layers + i) * 2u * kvl;
        size_t row = c->seq_len - 1u;
        im->snap_pos[s->snap][i] = c->base + row;
        memcpy(dst, c->keys + row * kvl, kvl * sizeof(float));
        memcpy(dst + kvl, c->values + row * kvl, kvl * sizeof(float));
    }
}

static void set_hold(E4Impl *im, int on)
{
    size_t i;
    for (i = 0; i < OC_EAGLE4_MAX_LAYERS; i++) im->kv[i].hold = on;
}

static void restore_state(OcEagle4Draft *m, const E4State *s)
{
    E4Impl *im = impl_of(m);
    size_t h = m->config.hidden_size;
    size_t i;
    m->position = s->position;
    for (i = 0; i < m->config.num_hidden_layers && i < OC_EAGLE4_MAX_LAYERS; i++)
        im->kv[i].seq_len = s->seq[i] >= im->kv[i].base ? s->seq[i] - im->kv[i].base : 0;
    if (im->snap_kv) {
        int32_t sn = im->snap_parent[s->snap];
        sn = (sn < 0) ? -1 : (int32_t)s->snap;
        while (sn >= 0) {
            size_t kvl = im->kv_size;
            for (i = 0; i < m->config.num_hidden_layers && i < OC_EAGLE4_MAX_LAYERS; i++) {
                E4Kv *c = &im->kv[i];
                const float *src =
                    im->snap_kv + ((size_t)sn * im->snap_layers + i) * 2u * kvl;
                size_t pos = im->snap_pos[sn][i];
                if (pos < c->base || pos - c->base >= c->seq_len) continue;
                memcpy(c->keys + (pos - c->base) * kvl, src, kvl * sizeof(float));
                memcpy(c->values + (pos - c->base) * kvl, src + kvl, kvl * sizeof(float));
            }
            sn = im->snap_parent[sn];
        }
    }
    memcpy(im->g, im->snaps + (size_t)s->snap * h, h * sizeof(float));
    m->g_embeddings = im->g;
    m->kv_seq0 = im->kv[0].seq_len;
}

static uint32_t map_id(const OcEagle4Draft *m, uint32_t draft_id)
{
    if (m->d2t && (size_t)draft_id < m->n_d2t) return m->d2t[draft_id];
    return draft_id;
}

static void softmax_keep(E4Impl *im)
{
    float mx;
    float sum = 0.0f;
    uint32_t i;
    if (im->n_top == 0) return;
    mx = im->top_probs[0];
    for (i = 1; i < im->n_top; i++)
        if (im->top_probs[i] > mx) mx = im->top_probs[i];
    for (i = 0; i < im->n_top; i++) {
        im->top_probs[i] = expf(im->top_probs[i] - mx);
        sum += im->top_probs[i];
    }
    if (sum <= 0.0f) sum = 1.0f;
    for (i = 0; i < im->n_top; i++) im->top_probs[i] /= sum;
}

static void remember_top(E4Impl *im, const E4Scored *ranked, uint32_t n)
{
    uint32_t i;
    im->n_top = 0;
    for (i = 0; i < n && im->n_top < OC_EAGLE4_TOP_KEEP; i++) {
        uint32_t j;
        int dup = 0;
        for (j = 0; j < im->n_top; j++) {
            if (im->top_ids[j] == ranked[i].id) {
                dup = 1;
                break;
            }
        }
        if (dup) continue;
        im->top_ids[im->n_top] = ranked[i].id;
        im->top_probs[im->n_top] = ranked[i].score;
        im->n_top++;
    }
    softmax_keep(im);
}

static void fill_head(OcEagle4Draft *m, const float *hidden)
{
    E4Impl *im = impl_of(m);
    size_t h = m->config.hidden_size;
    size_t dv = m->config.draft_vocab_size;
    size_t V = m->config.vocab_size;
    size_t i;
    for (i = 0; i < V; i++) im->logits[i] = -HUGE_VALF;

    if (m->full_lm_head || !m->lm_score.data || !m->lm_down.data) {
        if (m->lm_head.data && m->lm_head.rows == dv && m->lm_head.cols == h) {
            gemv(m->lm_head.data, dv, h, hidden, im->scores);
            m->stats.exact_rows += dv;
        } else {
            memset(im->scores, 0, dv * sizeof(float));
        }
        for (i = 0; i < dv; i++) {
            uint32_t dst = map_id(m, (uint32_t)i);
            if ((size_t)dst < V) im->logits[dst] = im->scores[i];
        }
    } else {
        uint32_t k = (uint32_t)m->config.candidate_k;
        uint32_t nheap = 0;
        if (k > (uint32_t)dv) k = (uint32_t)dv;
        if (m->lm_down.rows == m->config.rank && m->lm_down.cols == h)
            gemv(m->lm_down.data, m->config.rank, h, hidden, im->proj);
        else
            memset(im->proj, 0, m->config.rank * sizeof(float));
        if (m->lm_score.rows == dv && m->lm_score.cols == m->config.rank)
            gemv(m->lm_score.data, dv, m->config.rank, im->proj, im->scores);
        else
            memset(im->scores, 0, dv * sizeof(float));
        m->stats.score_rows += dv;
        for (i = 0; i < dv; i++)
            heap_add(im->heap, &nheap, k, im->scores[i], (uint32_t)i);
        sort_best(im->heap, nheap);
        for (i = 0; i < nheap; i++) {
            uint32_t id = im->heap[i].id;
            float exact;
            uint32_t dst;
            if (m->lm_head.data && m->lm_head.rows == dv && m->lm_head.cols == h &&
                (size_t)id < dv)
                exact = dot_f(m->lm_head.data + (size_t)id * h, hidden, h);
            else
                exact = im->scores[id];
            dst = map_id(m, id);
            im->heap[i].score = exact;
            im->heap[i].id = dst;
            if ((size_t)dst < V) im->logits[dst] = exact;
        }
        m->stats.exact_rows += nheap;
        sort_best(im->heap, nheap);
        remember_top(im, im->heap, nheap);
        return;
    }

    {
        uint32_t k = (uint32_t)m->config.candidate_k;
        uint32_t nheap = 0;
        if (k > (uint32_t)V) k = (uint32_t)V;
        for (i = 0; i < V; i++)
            heap_add(im->heap, &nheap, k, im->logits[i], (uint32_t)i);
        sort_best(im->heap, nheap);
        remember_top(im, im->heap, nheap);
    }
}

static OcError layer_forward(OcEagle4Draft *m, size_t li, const float *x)
{
    E4Impl *im = impl_of(m);
    OcEagle4Config *cfg = &m->config;
    OcEagle4Layer *layer = &m->layers[li];
    size_t h = cfg->hidden_size;
    size_t hd = cfg->head_dim;
    size_t n_heads = cfg->num_attention_heads;
    size_t n_kv = cfg->num_key_value_heads;
    const float *xn;
    size_t start;
    size_t nwin;
    size_t seq;
    OcError err;

    if (layer->attn_norm)
        rms_forward(x, layer->attn_norm, im->ffn, h, cfg->rms_norm_eps);
    xn = layer->attn_norm ? im->ffn : x;

    if (layer->q_proj.data)
        gemv(layer->q_proj.data, im->q_size, h, xn, im->q);
    else
        memset(im->q, 0, im->q_size * sizeof(float));
    if (layer->k_proj.data)
        gemv(layer->k_proj.data, im->kv_size, h, xn, im->k);
    else
        memset(im->k, 0, im->kv_size * sizeof(float));
    if (layer->v_proj.data)
        gemv(layer->v_proj.data, im->kv_size, h, xn, im->v);
    else
        memset(im->v, 0, im->kv_size * sizeof(float));

    if (cfg->use_rope) {
        rope_inplace(im->q, n_heads, hd, m->position, cfg->rope_theta);
        rope_inplace(im->k, n_kv, hd, m->position, cfg->rope_theta);
    }
    err = kv_append(&im->kv[li], im->k, im->v);
    if (err != OC_OK) return err;

    seq = im->kv[li].seq_len;
    start = 0;
    if (cfg->sliding_window > 0 && seq > cfg->sliding_window)
        start = seq - cfg->sliding_window;
    nwin = seq - start;
    m->stats.attn_keys += nwin;
    if (nwin == 0) {
        memset(im->attn, 0, im->q_size * sizeof(float));
    } else {
        err = oc_flash_attention_decode_heads_f32(
            im->q,
            im->kv[li].keys + start * im->kv[li].kv_len,
            im->kv[li].values + start * im->kv[li].kv_len,
            nwin, hd, im->kv[li].kv_len, n_heads, n_kv, im->attn);
        if (err != OC_OK) return err;
    }

    if (layer->o_proj.data && layer->o_proj.rows == h)
        gemv(layer->o_proj.data, h, im->q_size, im->attn, im->hidden);
    else if (im->q_size == h)
        memcpy(im->hidden, im->attn, h * sizeof(float));
    else
        memset(im->hidden, 0, h * sizeof(float));
    {
        size_t i;
        for (i = 0; i < h; i++) im->hidden[i] += x[i];
    }
    memcpy(im->pre_mlp, im->hidden, h * sizeof(float));

    if (layer->ffn_norm)
        rms_forward(im->pre_mlp, layer->ffn_norm, im->ffn, h, cfg->rms_norm_eps);
    else
        memcpy(im->ffn, im->pre_mlp, h * sizeof(float));

    if (cfg->n_experts > 1 && layer->router) {
        float rscore[OC_EAGLE4_MAX_EXPERTS];
        uint32_t picked[OC_EAGLE4_MAX_EXPERTS];
        float weight[OC_EAGLE4_MAX_EXPERTS];
        size_t ne = cfg->n_experts;
        size_t top = cfg->top_k_experts ? cfg->top_k_experts : 1;
        size_t e, t, i;
        float mx, sum;
        if (ne > OC_EAGLE4_MAX_EXPERTS) ne = OC_EAGLE4_MAX_EXPERTS;
        if (top > ne) top = ne;
        gemv(layer->router, ne, h, im->ffn, rscore);
        for (t = 0; t < top; t++) {
            size_t best = 0;
            for (e = 1; e < ne; e++)
                if (rscore[e] > rscore[best]) best = e;
            picked[t] = (uint32_t)best;
            weight[t] = rscore[best];
            rscore[best] = -HUGE_VALF;
        }
        mx = weight[0];
        for (t = 1; t < top; t++)
            if (weight[t] > mx) mx = weight[t];
        sum = 0.0f;
        for (t = 0; t < top; t++) {
            weight[t] = expf(weight[t] - mx);
            sum += weight[t];
        }
        if (sum <= 0.0f) sum = 1.0f;
        memset(im->mlp, 0, h * sizeof(float));
        for (t = 0; t < top; t++) {
            OcEagle4Weight *g = &layer->expert_gate[picked[t]];
            OcEagle4Weight *u = &layer->expert_up[picked[t]];
            OcEagle4Weight *d = &layer->expert_down[picked[t]];
            float wgt = weight[t] / sum;
            size_t inter = cfg->intermediate_size;
            if (g->data) gemv(g->data, inter, h, im->ffn, im->gate);
            else memset(im->gate, 0, inter * sizeof(float));
            if (u->data) gemv(u->data, inter, h, im->ffn, im->up);
            else memset(im->up, 0, inter * sizeof(float));
            for (i = 0; i < inter; i++) im->act[i] = oc_arch_silu(im->gate[i]) * im->up[i];
            if (d->data) gemv(d->data, h, inter, im->act, im->hidden);
            else memset(im->hidden, 0, h * sizeof(float));
            for (i = 0; i < h; i++) im->mlp[i] += wgt * im->hidden[i];
        }
    } else {
        size_t inter = cfg->intermediate_size;
        size_t i;
        if (layer->mlp_gate.data)
            gemv(layer->mlp_gate.data, inter, h, im->ffn, im->gate);
        else
            memset(im->gate, 0, inter * sizeof(float));
        if (layer->mlp_up.data)
            gemv(layer->mlp_up.data, inter, h, im->ffn, im->up);
        else
            memset(im->up, 0, inter * sizeof(float));
        if (layer->mlp_gate.data || layer->mlp_up.data) {
            for (i = 0; i < inter; i++) im->act[i] = oc_arch_silu(im->gate[i]) * im->up[i];
            if (layer->mlp_down.data)
                gemv(layer->mlp_down.data, h, inter, im->act, im->mlp);
            else
                memset(im->mlp, 0, h * sizeof(float));
        } else {
            memset(im->mlp, 0, h * sizeof(float));
        }
    }
    {
        size_t i;
        for (i = 0; i < h; i++) im->hidden[i] = im->pre_mlp[i] + im->mlp[i];
    }
    return OC_OK;
}

void oc_eagle4_config_init(OcEagle4Config *cfg)
{
    size_t i;
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->hidden_size = 64;
    cfg->target_hidden_size = 64;
    cfg->n_extract_layers = 3;
    cfg->extract_layers[0] = 2;
    cfg->extract_layers[1] = 16;
    cfg->extract_layers[2] = 29;
    for (i = 3; i < OC_EAGLE4_MAX_EXTRACT; i++) cfg->extract_layers[i] = 0;
    cfg->num_attention_heads = 4;
    cfg->num_key_value_heads = 4;
    cfg->head_dim = 16;
    cfg->intermediate_size = 128;
    cfg->vocab_size = 1024;
    cfg->draft_vocab_size = 1024;
    cfg->rank = 32;
    cfg->candidate_k = 32;
    cfg->sliding_window = 64;
    cfg->num_hidden_layers = 1;
    cfg->n_experts = 0;
    cfg->top_k_experts = 1;
    cfg->rms_norm_eps = 1e-5f;
    cfg->rope_theta = 10000.0f;
    cfg->use_rope = false;
    cfg->norm_before_residual = false;
}

void oc_eagle4_config_set_released(OcEagle4Config *cfg)
{
    if (!cfg) return;
    cfg->sliding_window = OC_EAGLE4_RELEASED_WINDOW;
    cfg->rank = OC_EAGLE4_RELEASED_RANK;
    cfg->candidate_k = OC_EAGLE4_RELEASED_CANDIDATES;
}

void oc_eagle4_harness_init(OcEagle4Harness *h)
{
    if (!h) return;
    memset(h, 0, sizeof(*h));
    h->draft_step_us = 80.0f;
    h->verify_per_row_us = 2.0f;
    h->min_gain_per_us = 0.0004f;
    h->max_depth = OC_EAGLE4_RELEASED_DEPTH;
    h->max_branch = OC_EAGLE4_RELEASED_BRANCH;
    h->verify_cap = OC_EAGLE4_RELEASED_VERIFY;
    h->tree_batch_max = 4;
    h->chain_width_batch = 64;
    h->narrow_batch = 384;
}

OcError oc_eagle4_model_init(OcEagle4Draft *m, const OcEagle4Config *cfg)
{
    OcEagle4Config c;
    E4Impl *im;
    size_t h, hd, layers, cap, kv_len, feat, qsz, kvsz, inter, V, dv, R, snaps, total, cursor_n;
    float *cursor;
    size_t i;

    if (!m || !cfg) return OC_ERR_INVALID_ARG;
    c = *cfg;
    if (c.hidden_size == 0 || c.vocab_size == 0 || c.num_attention_heads == 0 ||
        c.num_key_value_heads == 0 || c.intermediate_size == 0)
        return OC_ERR_INVALID_ARG;
    if (c.hidden_size > E4_MAX_DIM || c.vocab_size > E4_MAX_DIM ||
        c.intermediate_size > E4_MAX_DIM || c.num_attention_heads > E4_MAX_DIM ||
        c.num_key_value_heads > E4_MAX_DIM || c.head_dim > E4_MAX_DIM ||
        c.draft_vocab_size > E4_MAX_DIM || c.target_hidden_size > E4_MAX_DIM ||
        c.rank > E4_MAX_DIM || c.sliding_window > E4_MAX_DIM ||
        c.candidate_k > E4_MAX_DIM)
        return OC_ERR_INVALID_ARG;
    if (c.head_dim == 0) c.head_dim = c.hidden_size / c.num_attention_heads;
    if (c.head_dim == 0 || (c.num_attention_heads % c.num_key_value_heads) != 0)
        return OC_ERR_INVALID_ARG;
    if (c.num_hidden_layers == 0) c.num_hidden_layers = 1;
    if (c.num_hidden_layers > OC_EAGLE4_MAX_LAYERS) return OC_ERR_INVALID_ARG;
    if (c.n_extract_layers == 0 || c.n_extract_layers > OC_EAGLE4_MAX_EXTRACT)
        return OC_ERR_INVALID_ARG;
    if (c.n_experts > OC_EAGLE4_MAX_EXPERTS) return OC_ERR_INVALID_ARG;
    if (c.draft_vocab_size == 0) c.draft_vocab_size = c.vocab_size;
    if (c.target_hidden_size == 0) c.target_hidden_size = c.hidden_size;
    if (c.rank == 0) c.rank = 1;
    if (c.candidate_k == 0) c.candidate_k = 1;
    if (c.candidate_k > c.draft_vocab_size) c.candidate_k = c.draft_vocab_size;
    if (c.sliding_window == 0) c.sliding_window = 1;
    if (c.n_experts > 1 && c.top_k_experts == 0) c.top_k_experts = 1;
    if (c.rms_norm_eps <= 0.0f) c.rms_norm_eps = 1e-5f;

    memset(m, 0, sizeof(*m));
    m->config = c;
    im = (E4Impl *)calloc(1, sizeof(*im));
    if (!im) return OC_ERR_OOM;
    m->impl = im;

    h = c.hidden_size;
    hd = c.head_dim;
    layers = c.num_hidden_layers;
    feat = c.n_extract_layers * c.target_hidden_size;
    qsz = c.num_attention_heads * hd;
    kvsz = c.num_key_value_heads * hd;
    inter = c.intermediate_size;
    V = c.vocab_size;
    dv = c.draft_vocab_size;
    R = c.rank;
    snaps = (size_t)OC_EAGLE4_MAX_NODES + 2u;
    im->feat_width = feat;
    im->q_size = qsz;
    im->kv_size = kvsz;
    im->n_snaps = snaps;

    {
        /* Each term is a product of dims capped at E4_MAX_DIM (2^24). */
        size_t terms[][2] = {
            { c.n_extract_layers, c.target_hidden_size }, { h, 18u },
            { c.num_attention_heads, hd }, { c.num_attention_heads, hd },
            { c.num_attention_heads, hd }, { c.num_key_value_heads, hd },
            { c.num_key_value_heads, hd }, { inter, 3u }, { R, 1u }, { dv, 1u },
            { V, 1u }, { snaps, h },
        };
        total = 0;
        for (i = 0; i < sizeof(terms) / sizeof(terms[0]); i++) {
            size_t t;
            if (!oc_size_mul(terms[i][0], terms[i][1], &t) || total > SIZE_MAX - t) {
                oc_eagle4_model_free(m);
                return OC_ERR_INVALID_ARG;
            }
            total += t;
        }
        if (total > SIZE_MAX / sizeof(float)) {
            oc_eagle4_model_free(m);
            return OC_ERR_INVALID_ARG;
        }
    }
    im->arena = (float *)calloc(total ? total : 1, sizeof(float));
    im->heap = (E4Scored *)calloc(c.candidate_k ? c.candidate_k : 1, sizeof(E4Scored));
    if (!im->arena || !im->heap) {
        oc_eagle4_model_free(m);
        return OC_ERR_OOM;
    }
    cursor = im->arena;
    cursor_n = total;
#define TAKE(dst, n) do { \
        size_t _n = (n); \
        if (cursor_n < _n) { oc_eagle4_model_free(m); return OC_ERR_INTERNAL; } \
        (dst) = cursor; \
        cursor += _n; \
        cursor_n -= _n; \
    } while (0)
    TAKE(im->fused_in, feat);
    TAKE(im->fc_out, h);
    TAKE(im->g, h);
    TAKE(im->g_before, h);
    TAKE(im->emb, h);
    TAKE(im->emb_n, h);
    TAKE(im->g_n, h);
    TAKE(im->cat, h * 2u);
    TAKE(im->x, h);
    TAKE(im->q, qsz);
    TAKE(im->k, kvsz);
    TAKE(im->v, kvsz);
    TAKE(im->attn, qsz);
    TAKE(im->pre_mlp, h);
    TAKE(im->hidden, h);
    TAKE(im->ffn, h);
    TAKE(im->gate, inter);
    TAKE(im->up, inter);
    TAKE(im->act, inter);
    TAKE(im->mlp, h);
    TAKE(im->proj, R);
    TAKE(im->scores, dv);
    TAKE(im->logits, V);
    TAKE(im->dh, h);
    TAKE(im->snaps, snaps * h);
#undef TAKE

    kv_len = kvsz;
    cap = c.sliding_window + 16u;
    if (cap < 32u) cap = 32u;
    {
        size_t cells;
        size_t kv_total, snap_total;
        /* KV ring (keys+values per layer) plus the per-node snapshot copy;
         * reject anything past a 16 GiB budget before allocating. */
        const uint64_t cap_floats = (UINT64_C(16) << 30) / sizeof(float);
        if (!oc_size_mul(cap, kv_len, &cells) || cells > SIZE_MAX / sizeof(float) ||
            !oc_size_mul(cells, layers * 2u, &kv_total) ||
            !oc_size_mul(snaps * layers * 2u, kv_len, &snap_total) ||
            (uint64_t)kv_total > cap_floats ||
            (uint64_t)snap_total > cap_floats - (uint64_t)kv_total) {
            oc_eagle4_model_free(m);
            return OC_ERR_INVALID_ARG;
        }
    }
    im->snap_layers = layers;
    for (i = 0; i < layers; i++) {
        im->kv[i].kv_len = kv_len;
        im->kv[i].window = c.sliding_window;
        im->kv[i].capacity = cap;
        im->kv[i].keys = (float *)calloc(cap * kv_len, sizeof(float));
        im->kv[i].values = (float *)calloc(cap * kv_len, sizeof(float));
        if (!im->kv[i].keys || !im->kv[i].values) {
            oc_eagle4_model_free(m);
            return OC_ERR_OOM;
        }
    }
    {
        size_t n;
        if (!oc_size_mul(snaps * layers * 2u, kvsz, &n) ||
            n > SIZE_MAX / sizeof(float)) {
            oc_eagle4_model_free(m);
            return OC_ERR_INVALID_ARG;
        }
        im->snap_kv = (float *)calloc(n ? n : 1, sizeof(float));
        if (!im->snap_kv) {
            oc_eagle4_model_free(m);
            return OC_ERR_OOM;
        }
    }
    m->g_embeddings = im->g;
    m->loaded = true;
    return OC_OK;
}

void oc_eagle4_model_free(OcEagle4Draft *m)
{
    E4Impl *im;
    size_t i;
    if (!m) return;
    im = impl_of(m);
    if (im) {
        for (i = 0; i < im->n_owned; i++) free(im->owned[i]);
        for (i = 0; i < OC_EAGLE4_MAX_LAYERS; i++) {
            free(im->kv[i].keys);
            free(im->kv[i].values);
        }
        free(im->arena);
        free(im->snap_kv);
        free(im->heap);
        free(im);
    }
    memset(m, 0, sizeof(*m));
}

OcError oc_eagle4_alloc_f32(OcEagle4Draft *m, float **dst, size_t n)
{
    float *p;
    if (!m || !m->impl || !dst || n == 0) return OC_ERR_INVALID_ARG;
    if (n > SIZE_MAX / sizeof(float)) return OC_ERR_INVALID_ARG;
    p = (float *)e4_own(m, n * sizeof(float));
    if (!p) return OC_ERR_OOM;
    *dst = p;
    return OC_OK;
}

OcError oc_eagle4_alloc_weight(OcEagle4Draft *m, OcEagle4Weight *w, size_t rows, size_t cols)
{
    float *p;
    if (!m || !m->impl || !w || rows == 0 || cols == 0) return OC_ERR_INVALID_ARG;
    if (rows > SIZE_MAX / cols || rows * cols > SIZE_MAX / sizeof(float))
        return OC_ERR_INVALID_ARG;
    p = (float *)e4_own(m, rows * cols * sizeof(float));
    if (!p) return OC_ERR_OOM;
    w->data = p;
    w->rows = rows;
    w->cols = cols;
    return OC_OK;
}

void oc_eagle4_reset_cache(OcEagle4Draft *m)
{
    E4Impl *im;
    size_t i;
    if (!m) return;
    im = impl_of(m);
    if (!im) return;
    for (i = 0; i < m->config.num_hidden_layers && i < OC_EAGLE4_MAX_LAYERS; i++)
    {
        im->kv[i].seq_len = 0;
        im->kv[i].base = 0;
    }
    m->position = 0;
    m->kv_seq0 = 0;
    if (im->g && m->config.hidden_size)
        memset(im->g, 0, m->config.hidden_size * sizeof(float));
}

void oc_eagle4_reset_stats(OcEagle4Draft *m)
{
    if (!m) return;
    memset(&m->stats, 0, sizeof(m->stats));
}

size_t oc_eagle4_kv_len(const OcEagle4Draft *m)
{
    const E4Impl *im = impl_of_c(m);
    if (!im) return 0;
    return im->kv[0].seq_len;
}

OcError oc_eagle4_encode_features(OcEagle4Draft *m, const float *features, size_t n_features)
{
    E4Impl *im;
    size_t h, th, n, i;
    if (!m || !m->loaded || !features) return OC_ERR_INVALID_ARG;
    im = impl_of(m);
    h = m->config.hidden_size;
    th = m->config.target_hidden_size;
    n = m->config.n_extract_layers;
    if (n_features != n * th) return OC_ERR_INVALID_ARG;
    for (i = 0; i < n; i++) {
        const float *src = features + i * th;
        float *dst = im->fused_in + i * th;
        rms_forward(src, m->fc_norm[i], dst, th, m->config.rms_norm_eps);
    }
    if (m->fc.data && m->fc.rows == h && m->fc.cols == n * th)
        gemv(m->fc.data, h, n * th, im->fused_in, im->fc_out);
    else {
        size_t copy_n = n * th < h ? n * th : h;
        memset(im->fc_out, 0, h * sizeof(float));
        memcpy(im->fc_out, im->fused_in, copy_n * sizeof(float));
    }
    rms_forward(im->fc_out, m->fc_post_norm, im->g, h, m->config.rms_norm_eps);
    m->g_embeddings = im->g;
    return OC_OK;
}

OcError oc_eagle4_forward(OcEagle4Draft *m, uint32_t token, float *out_logits, size_t logits_len)
{
    E4Impl *im;
    size_t h;
    size_t li;
    OcError err;
    if (!m || !m->loaded) return OC_ERR_INVALID_ARG;
    im = impl_of(m);
    h = m->config.hidden_size;
    if ((size_t)token >= m->config.vocab_size) return OC_ERR_INVALID_ARG;

    memcpy(im->g_before, im->g, h * sizeof(float));
    if (m->tok_embeddings.data && m->tok_embeddings.cols == h &&
        (size_t)token < m->tok_embeddings.rows) {
        memcpy(im->emb, m->tok_embeddings.data + (size_t)token * h, h * sizeof(float));
    } else {
        memset(im->emb, 0, h * sizeof(float));
    }
    rms_forward(im->emb, m->embed_norm, im->emb_n, h, m->config.rms_norm_eps);
    rms_forward(im->g_before, m->input_norm, im->g_n, h, m->config.rms_norm_eps);
    memcpy(im->cat, im->emb_n, h * sizeof(float));
    memcpy(im->cat + h, im->g_n, h * sizeof(float));
    if (m->eh_proj.data && m->eh_proj.rows == h && m->eh_proj.cols == h * 2u)
        gemv(m->eh_proj.data, h, h * 2u, im->cat, im->x);
    else
        memcpy(im->x, im->emb_n, h * sizeof(float));

    for (li = 0; li < m->config.num_hidden_layers; li++) {
        /* layer_forward overwrites im->hidden before adding the residual,
         * so later layers need their input in a separate buffer. */
        if (li > 0) memcpy(im->x, im->hidden, h * sizeof(float));
        err = layer_forward(m, li, im->x);
        if (err != OC_OK) return err;
    }
    memcpy(im->g, im->hidden, h * sizeof(float));
    m->g_embeddings = im->g;
    m->position++;
    m->kv_seq0 = im->kv[0].seq_len;

    if (m->output_norm) {
        rms_forward(im->g, m->output_norm, im->hidden, h, m->config.rms_norm_eps);
        fill_head(m, im->hidden);
    } else {
        fill_head(m, im->g);
    }
    if (out_logits && logits_len) {
        size_t n = logits_len < m->config.vocab_size ? logits_len : m->config.vocab_size;
        memcpy(out_logits, im->logits, n * sizeof(float));
    }
    return OC_OK;
}

void oc_eagle4_policy_for_batch(const OcEagle4Harness *h, uint32_t batch,
                                size_t context_len, OcEagle4Policy *out)
{
    OcEagle4Harness local;
    if (!out) return;
    if (!h) {
        oc_eagle4_harness_init(&local);
        h = &local;
    }
    if (batch == 0) batch = 1;
    out->mode = OC_EAGLE4_MODE_TREE;
    out->max_depth = h->max_depth ? h->max_depth : OC_EAGLE4_RELEASED_DEPTH;
    out->max_branch = h->max_branch ? h->max_branch : OC_EAGLE4_RELEASED_BRANCH;
    out->verify_cap = h->verify_cap ? h->verify_cap : OC_EAGLE4_RELEASED_VERIFY;
    if (batch > h->tree_batch_max) {
        out->mode = OC_EAGLE4_MODE_CHAIN;
        out->max_branch = 1;
        if (out->verify_cap > 8) out->verify_cap = 8;
    }
    if (batch >= h->chain_width_batch) {
        out->mode = OC_EAGLE4_MODE_CHAIN;
        out->max_branch = 1;
        out->verify_cap = 4;
    }
    if (batch >= h->narrow_batch) {
        out->mode = OC_EAGLE4_MODE_CHAIN;
        out->max_branch = 1;
        out->verify_cap = 3;
    }
    if (context_len >= 32768u && batch > h->tree_batch_max &&
        batch < h->chain_width_batch) {
        out->mode = OC_EAGLE4_MODE_CHAIN;
        out->max_branch = 1;
        out->verify_cap = 8;
    }
}

static float level_gain(const OcEagle4Cand *c, const uint32_t *sel, uint32_t n)
{
    float gain = 0.0f;
    uint32_t i;
    for (i = 0; i < n; i++) {
        int32_t parent = c[sel[i]].parent;
        uint32_t j;
        int seen = 0;
        float sum = 0.0f;
        float parent_path = 0.0f;
        for (j = 0; j < i; j++) {
            if (c[sel[j]].parent == parent) {
                seen = 1;
                break;
            }
        }
        if (seen) continue;
        for (j = 0; j < n; j++) {
            float cond;
            if (c[sel[j]].parent != parent) continue;
            cond = c[sel[j]].cond_prob;
            if (cond < 0.0f) cond = 0.0f;
            sum += cond;
            if (cond > 1e-8f) parent_path = c[sel[j]].path_prob / cond;
        }
        if (sum > 1.0f) sum = 1.0f;
        gain += parent_path * sum;
    }
    return gain;
}

uint32_t oc_eagle4_select_level(const OcEagle4Harness *h, const OcEagle4Policy *policy,
                                uint32_t batch, uint32_t depth, uint32_t already,
                                const OcEagle4Cand *cands, uint32_t n_cands,
                                uint32_t *chosen, uint32_t max_chosen)
{
    OcEagle4Harness local;
    uint32_t idx[E4_SEL_MAX];
    uint32_t n, i, best, n_el, remaining, n_sel;
    int32_t best_parent;
    if (!policy || !cands || !chosen || max_chosen == 0 || n_cands == 0) return 0;
    if (!h) {
        oc_eagle4_harness_init(&local);
        h = &local;
    }
    if (batch == 0) batch = 1;
    if (depth == 0) depth = 1;
    if (policy->verify_cap <= already) return 0;
    remaining = policy->verify_cap - already;
    if (remaining > max_chosen) remaining = max_chosen;
    n = n_cands > E4_SEL_MAX ? E4_SEL_MAX : n_cands;
    best = 0;
    for (i = 1; i < n; i++) {
        if (cands[i].path_prob > cands[best].path_prob ||
            (cands[i].path_prob == cands[best].path_prob && i < best))
            best = i;
    }
    best_parent = cands[best].parent;
    n_el = 0;
    for (i = 0; i < n; i++) {
        if (policy->max_branch <= 1 || depth >= 3) {
            if (cands[i].parent != best_parent) continue;
        }
        idx[n_el++] = i;
    }
    for (i = 1; i < n_el; i++) {
        uint32_t k = idx[i];
        uint32_t j = i;
        while (j > 0 && cands[k].path_prob > cands[idx[j - 1u]].path_prob) {
            idx[j] = idx[j - 1u];
            j--;
        }
        idx[j] = k;
    }
    n_sel = 0;
    for (i = 0; i < n_el && n_sel < remaining; i++) {
        int32_t parent = cands[idx[i]].parent;
        uint32_t limit;
        uint32_t count = 0;
        uint32_t t;
        if (cands[idx[i]].cond_prob <= 0.0f && cands[idx[i]].path_prob <= 0.0f)
            continue;
        if (policy->max_branch <= 1 || depth >= 3) limit = 1;
        else if (depth <= 1) limit = policy->max_branch;
        else limit = (parent == best_parent) ? 2u : 1u;
        if (limit == 0) limit = 1;
        for (t = 0; t < n_sel; t++)
            if (cands[chosen[t]].parent == parent) count++;
        if (count >= limit) continue;
        chosen[n_sel++] = idx[i];
    }
    while (n_sel > 0) {
        float gain = level_gain(cands, chosen, n_sel);
        float cost = h->draft_step_us * (float)n_sel +
                     h->verify_per_row_us * (float)batch * (float)n_sel;
        if (cost < 1e-8f) cost = 1e-8f;
        if (gain / cost >= h->min_gain_per_us) break;
        if (already == 0 && n_sel == 1) break;
        n_sel--;
    }
    return n_sel;
}

static void copy_top_into(const E4Impl *im, E4Ready *r, float path)
{
    uint32_t i;
    r->path = path;
    r->n_tok = im->n_top;
    if (r->n_tok > OC_EAGLE4_TOP_KEEP) r->n_tok = OC_EAGLE4_TOP_KEEP;
    for (i = 0; i < r->n_tok; i++) {
        r->tok[i] = im->top_ids[i];
        r->pr[i] = im->top_probs[i];
    }
}

OcError oc_eagle4_propose(OcEagle4Draft *m, const OcEagle4Harness *h,
                          uint32_t prefix_token, uint32_t batch, size_t context_len,
                          OcEagle4Tree *out)
{
    OcEagle4Harness local;
    OcEagle4Policy policy;
    E4Impl *im;
    E4State prefix;
    E4State pre;
    E4Ready cur[OC_EAGLE4_MAX_NODES];
    uint32_t ncur = 0;
    uint32_t depth;
    OcError err;
    if (!m || !m->loaded || !out) return OC_ERR_INVALID_ARG;
    if (!h) {
        oc_eagle4_harness_init(&local);
        h = &local;
    }
    im = impl_of(m);
    oc_eagle4_policy_for_batch(h, batch, context_len, &policy);
    memset(out, 0, sizeof(*out));
    im->snap_used = 0;
    capture_state(m, &pre);
    err = oc_eagle4_forward(m, prefix_token, NULL, 0);
    if (err != OC_OK) {
        /* A later layer may fail after earlier layers appended a row. */
        restore_state(m, &pre);
        return err;
    }
    set_hold(im, 1);
    capture_state(m, &prefix);
    cur[0].tree_idx = -1;
    cur[0].state = prefix;
    copy_top_into(im, &cur[0], 1.0f);
    ncur = 1;

    for (depth = 1; depth <= policy.max_depth && ncur &&
         out->n_nodes < policy.verify_cap; depth++) {
        OcEagle4Cand cands[OC_EAGLE4_MAX_NODES * OC_EAGLE4_TOP_KEEP];
        uint32_t ready_i[OC_EAGLE4_MAX_NODES * OC_EAGLE4_TOP_KEEP];
        uint32_t tok[OC_EAGLE4_MAX_NODES * OC_EAGLE4_TOP_KEEP];
        uint32_t chosen[OC_EAGLE4_MAX_NODES];
        uint32_t nc = 0;
        uint32_t nsel;
        uint32_t s;
        E4Ready next[OC_EAGLE4_MAX_NODES];
        uint32_t nnext = 0;
        uint32_t r;
        for (r = 0; r < ncur; r++) {
            uint32_t j;
            uint32_t keep = cur[r].n_tok;
            if (policy.max_branch > 0 && keep > policy.max_branch && depth <= 2)
                keep = policy.max_branch;
            if (policy.max_branch <= 1) keep = cur[r].n_tok ? 1u : 0u;
            for (j = 0; j < keep; j++) {
                if (nc >= OC_EAGLE4_MAX_NODES * OC_EAGLE4_TOP_KEEP) break;
                cands[nc].parent = cur[r].tree_idx;
                cands[nc].cond_prob = cur[r].pr[j];
                cands[nc].path_prob = cur[r].path * cur[r].pr[j];
                ready_i[nc] = r;
                tok[nc] = cur[r].tok[j];
                nc++;
            }
        }
        if (nc == 0) break;
        nsel = oc_eagle4_select_level(h, &policy, batch, depth, out->n_nodes,
                                      cands, nc, chosen, OC_EAGLE4_MAX_NODES);
        if (nsel == 0) break;
        for (s = 0; s < nsel && out->n_nodes < policy.verify_cap; s++) {
            uint32_t ci = chosen[s];
            E4Ready *parent = &cur[ready_i[ci]];
            OcEagle4Node *node;
            if (out->n_nodes >= OC_EAGLE4_MAX_NODES) break;
            node = &out->nodes[out->n_nodes];
            node->token = tok[ci];
            node->parent = parent->tree_idx;
            node->cond_prob = cands[ci].cond_prob;
            node->path_prob = cands[ci].path_prob;
            node->depth = depth;
            restore_state(m, &parent->state);
            err = oc_eagle4_forward(m, node->token, NULL, 0);
            if (err != OC_OK) {
                restore_state(m, &prefix);
                set_hold(im, 0);
                return err;
            }
            if (nnext < OC_EAGLE4_MAX_NODES) {
                next[nnext].tree_idx = (int32_t)out->n_nodes;
                capture_node(m, &next[nnext].state, &parent->state);
                copy_top_into(im, &next[nnext], node->path_prob);
                nnext++;
            }
            out->n_nodes++;
        }
        ncur = nnext;
        if (ncur) memcpy(cur, next, ncur * sizeof(cur[0]));
    }
    restore_state(m, &prefix);
    set_hold(im, 0);
    return OC_OK;
}

OcError oc_eagle4_commit(OcEagle4Draft *m, const uint32_t *tokens, size_t n)
{
    size_t i;
    if (!m || !m->loaded) return OC_ERR_INVALID_ARG;
    if (n && !tokens) return OC_ERR_INVALID_ARG;
    for (i = 0; i < n; i++) {
        OcError err = oc_eagle4_forward(m, tokens[i], NULL, 0);
        if (err != OC_OK) return err;
    }
    return OC_OK;
}

void oc_eagle4_tree_mask(const OcEagle4Tree *tree, uint8_t *mask)
{
    uint32_t i, n;
    if (!tree || !mask) return;
    n = tree->n_nodes;
    memset(mask, 0, (size_t)n * (size_t)n);
    for (i = 0; i < n; i++) {
        int32_t cur = (int32_t)i;
        uint32_t guard = 0;
        while (cur >= 0 && (uint32_t)cur < n && guard++ < n) {
            mask[(size_t)i * n + (uint32_t)cur] = 1;
            cur = tree->nodes[cur].parent;
        }
    }
}

uint32_t oc_eagle4_accept_greedy(const OcEagle4Tree *tree, uint32_t first_target,
                                 const uint32_t *next_target, uint32_t *out,
                                 uint32_t max_out)
{
    int32_t parent = -1;
    uint32_t expect;
    uint32_t n_out = 0;
    if (!tree || !out || max_out == 0) return 0;
    expect = first_target;
    for (;;) {
        uint32_t i;
        int32_t found = -1;
        if (n_out >= max_out) break;
        for (i = 0; i < tree->n_nodes; i++) {
            if (tree->nodes[i].parent == parent && tree->nodes[i].token == expect) {
                found = (int32_t)i;
                break;
            }
        }
        if (found < 0) break;
        out[n_out++] = expect;
        if (!next_target) break;
        expect = next_target[found];
        parent = found;
    }
    return n_out;
}

static float rng_unit(uint32_t *rng)
{
    uint32_t x;
    if (!rng) return 0.0f;
    *rng = *rng * 1664525u + 1013904223u;
    x = *rng >> 8;
    return (float)x * (1.0f / 16777216.0f);
}

static uint32_t sample_dist(const float *p, uint32_t n, uint32_t *rng)
{
    float sum = 0.0f;
    float u, acc = 0.0f;
    uint32_t i;
    if (!p || n == 0) return 0;
    for (i = 0; i < n; i++) if (p[i] > 0.0f) sum += p[i];
    if (sum <= 0.0f) return 0;
    u = rng_unit(rng) * sum;
    for (i = 0; i < n; i++) {
        if (p[i] <= 0.0f) continue; /* u may be 0: never pick a zero-mass entry */
        acc += p[i];
        if (acc >= u) return i;
    }
    for (i = n; i-- > 0;)
        if (p[i] > 0.0f) return i;
    return n - 1u;
}

uint32_t oc_eagle4_speculative_accept(const OcEagle4Step *steps, uint32_t n_steps,
                                      uint32_t *rng, uint32_t *out, uint32_t max_out)
{
    uint32_t i, n_out = 0;
    if (!steps || !out || max_out == 0) return 0;
    for (i = 0; i < n_steps; i++) {
        float ratio;
        float draw;
        if (n_out >= max_out) return n_out;
        if (steps[i].q <= 0.0f) ratio = (steps[i].p > 0.0f) ? 1.0f : 0.0f;
        else {
            ratio = steps[i].p / steps[i].q;
            if (ratio > 1.0f) ratio = 1.0f;
        }
        draw = rng_unit(rng);
        if (draw < ratio) {
            out[n_out++] = steps[i].draft_token;
            continue;
        }
        if (steps[i].target_dist && steps[i].draft_dist && steps[i].vocab &&
            n_out < max_out) {
            float *resid = (float *)malloc((size_t)steps[i].vocab * sizeof(float));
            uint32_t t;
            if (!resid) return n_out;
            for (t = 0; t < steps[i].vocab; t++) {
                float d = steps[i].target_dist[t] - steps[i].draft_dist[t];
                resid[t] = d > 0.0f ? d : 0.0f;
            }
            out[n_out++] = sample_dist(resid, steps[i].vocab, rng);
            free(resid);
        }
        return n_out;
    }
    if (n_steps > 0 && steps[n_steps - 1u].target_dist && n_out < max_out) {
        out[n_out++] = sample_dist(steps[n_steps - 1u].target_dist,
                                   steps[n_steps - 1u].vocab, rng);
    }
    return n_out;
}

bool oc_eagle4_path_is_gguf(const char *path)
{
    size_t n;
    const char *e;
    if (!path) return false;
    n = strlen(path);
    if (n < 5) return false;
    e = path + n - 5;
    return e[0] == '.' && tolower((unsigned char)e[1]) == 'g' &&
           tolower((unsigned char)e[2]) == 'g' &&
           tolower((unsigned char)e[3]) == 'u' &&
           tolower((unsigned char)e[4]) == 'f';
}

static void cfg_store(const OcEagle4Draft *m, float *c)
{
    size_t i;
    memset(c, 0, E4_CFG_N * sizeof(float));
    c[0] = (float)m->config.hidden_size;
    c[1] = (float)m->config.target_hidden_size;
    c[2] = (float)m->config.n_extract_layers;
    c[3] = (float)m->config.num_attention_heads;
    c[4] = (float)m->config.num_key_value_heads;
    c[5] = (float)m->config.head_dim;
    c[6] = (float)m->config.intermediate_size;
    c[7] = (float)m->config.vocab_size;
    c[8] = (float)m->config.draft_vocab_size;
    c[9] = (float)m->config.rank;
    c[10] = (float)m->config.candidate_k;
    c[11] = (float)m->config.sliding_window;
    c[12] = m->config.rms_norm_eps;
    c[13] = m->config.rope_theta;
    c[14] = m->config.use_rope ? 1.0f : 0.0f;
    c[15] = (float)m->config.num_hidden_layers;
    c[16] = (float)m->config.n_experts;
    c[17] = (float)m->config.top_k_experts;
    c[18] = m->config.norm_before_residual ? 1.0f : 0.0f;
    c[19] = (float)m->n_d2t;
    for (i = 0; i < OC_EAGLE4_MAX_EXTRACT; i++)
        c[20 + i] = (float)m->config.extract_layers[i];
}

static int f_size_ok(float v)
{
    return isfinite(v) && v >= 0.0f && v <= (float)E4_MAX_DIM;
}

static size_t f_size(float v)
{
    if (!(v > 0.0f) || !f_size_ok(v)) return 0;
    return (size_t)(v + 0.5f);
}

static int cfg_load(OcEagle4Config *cfg, size_t *n_d2t, const float *c, size_t n)
{
    size_t i;
    if (n < E4_CFG_N) return 0;
    for (i = 0; i < E4_CFG_N; i++) {
        if (i == 12 || i == 13 || i == 14 || i == 18) {
            if (!isfinite(c[i])) return 0;
        } else if (!f_size_ok(c[i])) {
            return 0;
        }
    }
    memset(cfg, 0, sizeof(*cfg));
    cfg->hidden_size = f_size(c[0]);
    cfg->target_hidden_size = f_size(c[1]);
    cfg->n_extract_layers = f_size(c[2]);
    cfg->num_attention_heads = f_size(c[3]);
    cfg->num_key_value_heads = f_size(c[4]);
    cfg->head_dim = f_size(c[5]);
    cfg->intermediate_size = f_size(c[6]);
    cfg->vocab_size = f_size(c[7]);
    cfg->draft_vocab_size = f_size(c[8]);
    cfg->rank = f_size(c[9]);
    cfg->candidate_k = f_size(c[10]);
    cfg->sliding_window = f_size(c[11]);
    cfg->rms_norm_eps = c[12];
    cfg->rope_theta = c[13];
    cfg->use_rope = c[14] > 0.5f;
    cfg->num_hidden_layers = f_size(c[15]);
    cfg->n_experts = f_size(c[16]);
    cfg->top_k_experts = f_size(c[17]);
    cfg->norm_before_residual = c[18] > 0.5f;
    *n_d2t = f_size(c[19]);
    for (i = 0; i < OC_EAGLE4_MAX_EXTRACT; i++)
        cfg->extract_layers[i] = f_size(c[20 + i]);
    return cfg->hidden_size > 0 && cfg->num_key_value_heads > 0 &&
           cfg->num_attention_heads > 0;
}

typedef struct E4Blob {
    char name[80];
    const void *data;
    size_t nbytes;
    size_t rows;
    size_t cols;
    int u32;
    int vec;
} E4Blob;

static int blob_add(E4Blob *b, size_t *n, size_t cap, const char *name,
                    const void *data, size_t rows, size_t cols, int u32, int vec)
{
    size_t bytes;
    if (!data || rows == 0) return 1;
    if (*n >= cap) return 0;
    bytes = rows * (vec ? 1 : cols) * (u32 ? sizeof(uint32_t) : sizeof(float));
    if (!vec && cols == 0) return 1;
    snprintf(b[*n].name, sizeof(b[*n].name), "%s", name);
    b[*n].data = data;
    b[*n].nbytes = bytes;
    b[*n].rows = rows;
    b[*n].cols = vec ? 1 : cols;
    b[*n].u32 = u32;
    b[*n].vec = vec;
    (*n)++;
    return 1;
}

static int collect_blobs(const OcEagle4Draft *m, E4Blob *b, size_t *n, size_t cap)
{
    size_t L, e;
    char name[80];
    if (m->fc.data && !blob_add(b, n, cap, "eagle4.fc", m->fc.data, m->fc.rows, m->fc.cols, 0, 0))
        return 0;
    for (e = 0; e < m->config.n_extract_layers; e++) {
        snprintf(name, sizeof(name), "eagle4.fc_norm.%u", (unsigned)e);
        if (m->fc_norm[e] && !blob_add(b, n, cap, name, m->fc_norm[e], m->config.target_hidden_size, 1, 0, 1))
            return 0;
    }
    if (m->fc_post_norm && !blob_add(b, n, cap, "eagle4.fc_post_norm", m->fc_post_norm, m->config.hidden_size, 1, 0, 1))
        return 0;
    if (m->eh_proj.data && !blob_add(b, n, cap, "eagle4.eh_proj", m->eh_proj.data, m->eh_proj.rows, m->eh_proj.cols, 0, 0))
        return 0;
    if (m->tok_embeddings.data && !blob_add(b, n, cap, "eagle4.embed", m->tok_embeddings.data, m->tok_embeddings.rows, m->tok_embeddings.cols, 0, 0))
        return 0;
    if (m->embed_norm && !blob_add(b, n, cap, "eagle4.embed_norm", m->embed_norm, m->config.hidden_size, 1, 0, 1))
        return 0;
    if (m->input_norm && !blob_add(b, n, cap, "eagle4.input_norm", m->input_norm, m->config.hidden_size, 1, 0, 1))
        return 0;
    if (m->output_norm && !blob_add(b, n, cap, "eagle4.output_norm", m->output_norm, m->config.hidden_size, 1, 0, 1))
        return 0;
    if (m->lm_head.data && !blob_add(b, n, cap, "eagle4.lm_head", m->lm_head.data, m->lm_head.rows, m->lm_head.cols, 0, 0))
        return 0;
    if (m->lm_down.data && !blob_add(b, n, cap, "eagle4.lm_down", m->lm_down.data, m->lm_down.rows, m->lm_down.cols, 0, 0))
        return 0;
    if (m->lm_score.data && !blob_add(b, n, cap, "eagle4.lm_score", m->lm_score.data, m->lm_score.rows, m->lm_score.cols, 0, 0))
        return 0;
    if (m->d2t && m->n_d2t && !blob_add(b, n, cap, "eagle4.d2t", m->d2t, m->n_d2t, 1, 1, 1))
        return 0;
    for (L = 0; L < m->config.num_hidden_layers; L++) {
        const OcEagle4Layer *layer = &m->layers[L];
        snprintf(name, sizeof(name), "eagle4.layer.%u.q", (unsigned)L);
        if (layer->q_proj.data && !blob_add(b, n, cap, name, layer->q_proj.data, layer->q_proj.rows, layer->q_proj.cols, 0, 0)) return 0;
        snprintf(name, sizeof(name), "eagle4.layer.%u.k", (unsigned)L);
        if (layer->k_proj.data && !blob_add(b, n, cap, name, layer->k_proj.data, layer->k_proj.rows, layer->k_proj.cols, 0, 0)) return 0;
        snprintf(name, sizeof(name), "eagle4.layer.%u.v", (unsigned)L);
        if (layer->v_proj.data && !blob_add(b, n, cap, name, layer->v_proj.data, layer->v_proj.rows, layer->v_proj.cols, 0, 0)) return 0;
        snprintf(name, sizeof(name), "eagle4.layer.%u.o", (unsigned)L);
        if (layer->o_proj.data && !blob_add(b, n, cap, name, layer->o_proj.data, layer->o_proj.rows, layer->o_proj.cols, 0, 0)) return 0;
        snprintf(name, sizeof(name), "eagle4.layer.%u.attn_norm", (unsigned)L);
        if (layer->attn_norm && !blob_add(b, n, cap, name, layer->attn_norm, m->config.hidden_size, 1, 0, 1)) return 0;
        snprintf(name, sizeof(name), "eagle4.layer.%u.ffn_norm", (unsigned)L);
        if (layer->ffn_norm && !blob_add(b, n, cap, name, layer->ffn_norm, m->config.hidden_size, 1, 0, 1)) return 0;
        snprintf(name, sizeof(name), "eagle4.layer.%u.gate", (unsigned)L);
        if (layer->mlp_gate.data && !blob_add(b, n, cap, name, layer->mlp_gate.data, layer->mlp_gate.rows, layer->mlp_gate.cols, 0, 0)) return 0;
        snprintf(name, sizeof(name), "eagle4.layer.%u.up", (unsigned)L);
        if (layer->mlp_up.data && !blob_add(b, n, cap, name, layer->mlp_up.data, layer->mlp_up.rows, layer->mlp_up.cols, 0, 0)) return 0;
        snprintf(name, sizeof(name), "eagle4.layer.%u.down", (unsigned)L);
        if (layer->mlp_down.data && !blob_add(b, n, cap, name, layer->mlp_down.data, layer->mlp_down.rows, layer->mlp_down.cols, 0, 0)) return 0;
        snprintf(name, sizeof(name), "eagle4.layer.%u.router", (unsigned)L);
        if (layer->router && m->config.n_experts && !blob_add(b, n, cap, name, layer->router, m->config.n_experts * m->config.hidden_size, 1, 0, 1)) return 0;
        for (e = 0; e < m->config.n_experts && e < OC_EAGLE4_MAX_EXPERTS; e++) {
            snprintf(name, sizeof(name), "eagle4.layer.%u.expert.%u.gate", (unsigned)L, (unsigned)e);
            if (layer->expert_gate[e].data && !blob_add(b, n, cap, name, layer->expert_gate[e].data, layer->expert_gate[e].rows, layer->expert_gate[e].cols, 0, 0)) return 0;
            snprintf(name, sizeof(name), "eagle4.layer.%u.expert.%u.up", (unsigned)L, (unsigned)e);
            if (layer->expert_up[e].data && !blob_add(b, n, cap, name, layer->expert_up[e].data, layer->expert_up[e].rows, layer->expert_up[e].cols, 0, 0)) return 0;
            snprintf(name, sizeof(name), "eagle4.layer.%u.expert.%u.down", (unsigned)L, (unsigned)e);
            if (layer->expert_down[e].data && !blob_add(b, n, cap, name, layer->expert_down[e].data, layer->expert_down[e].rows, layer->expert_down[e].cols, 0, 0)) return 0;
        }
    }
    return 1;
}

/* Write to `path`, or to `ext` when non-NULL (path may then be NULL). An
 * external stream is flushed but not closed; the caller owns it. */
static OcError write_safetensors(const char *path, FILE *ext, const E4Blob *blobs,
                                 size_t nblobs, const float *cfg, int with_cfg)
{
    FILE *fp;
    char *json;
    size_t json_cap = 256u * (nblobs + 2u) + 64u;
    size_t used = 0;
    size_t i;
    size_t data_bytes = 0;
    uint64_t offsets_b[96];
    uint64_t offsets_e[96];
    uint8_t *raw;
    size_t cursor = 0;
    size_t n = nblobs;
    if (n > 96) return OC_ERR_INVALID_ARG;
    if (!ext && (!path || oc_eagle4_path_is_gguf(path))) return OC_ERR_INVALID_ARG;
    json = (char *)malloc(json_cap);
    if (!json) return OC_ERR_OOM;
    used = 0;
    json[used++] = '{';
    if (with_cfg) {
        int wr = snprintf(json + used, json_cap - used,
                          "\"eagle4.config\":{\"dtype\":\"F32\",\"shape\":[%d],\"data_offsets\":[0,0]}",
                          E4_CFG_N);
        if (wr < 0 || (size_t)wr >= json_cap - used) {
            free(json);
            return OC_ERR_OOM;
        }
        used += (size_t)wr;
    }
    for (i = 0; i < n; i++) {
        size_t pad = (8u - (data_bytes & 7u)) & 7u;
        data_bytes += pad;
        offsets_b[i] = (uint64_t)data_bytes;
        data_bytes += blobs[i].nbytes;
        offsets_e[i] = (uint64_t)data_bytes;
    }
    /* Rebuild JSON with real offsets. */
    used = 0;
    json[used++] = '{';
    if (with_cfg) {
        size_t cfg_bytes = (size_t)E4_CFG_N * sizeof(float);
        size_t pad;
        uint64_t b, e;
        int wr;
        /* config is placed after the blobs; recompute below once total known. */
        (void)cfg_bytes;
        (void)pad;
        (void)b;
        (void)e;
        (void)wr;
    }
    free(json);

    {
        size_t cfg_off_b = 0, cfg_off_e = 0;
        if (with_cfg) {
            size_t pad = (8u - (data_bytes & 7u)) & 7u;
            data_bytes += pad;
            cfg_off_b = data_bytes;
            data_bytes += (size_t)E4_CFG_N * sizeof(float);
            cfg_off_e = data_bytes;
        }
        json_cap = 192u * (n + 2u) + 128u;
        json = (char *)malloc(json_cap);
        if (!json) return OC_ERR_OOM;
        used = 0;
        json[used++] = '{';
        if (with_cfg) {
            int wr = snprintf(json + used, json_cap - used,
                              "\"eagle4.config\":{\"dtype\":\"F32\",\"shape\":[%d],\"data_offsets\":[%zu,%zu]}",
                              E4_CFG_N, cfg_off_b, cfg_off_e);
            if (wr < 0 || (size_t)wr >= json_cap - used) {
                free(json);
                return OC_ERR_OOM;
            }
            used += (size_t)wr;
        }
        for (i = 0; i < n; i++) {
            int wr;
            const char *dtype = blobs[i].u32 ? "U32" : "F32";
            if (used + 1 >= json_cap) {
                free(json);
                return OC_ERR_OOM;
            }
            if (with_cfg || i > 0) json[used++] = ',';
            if (blobs[i].vec || blobs[i].cols == 1) {
                wr = snprintf(json + used, json_cap - used,
                              "\"%s\":{\"dtype\":\"%s\",\"shape\":[%zu],\"data_offsets\":[%llu,%llu]}",
                              blobs[i].name, dtype, blobs[i].rows,
                              (unsigned long long)offsets_b[i],
                              (unsigned long long)offsets_e[i]);
            } else {
                wr = snprintf(json + used, json_cap - used,
                              "\"%s\":{\"dtype\":\"%s\",\"shape\":[%zu,%zu],\"data_offsets\":[%llu,%llu]}",
                              blobs[i].name, dtype, blobs[i].rows, blobs[i].cols,
                              (unsigned long long)offsets_b[i],
                              (unsigned long long)offsets_e[i]);
            }
            if (wr < 0 || (size_t)wr >= json_cap - used) {
                free(json);
                return OC_ERR_OOM;
            }
            used += (size_t)wr;
        }
        if (used + 2 >= json_cap) {
            free(json);
            return OC_ERR_OOM;
        }
        json[used++] = '}';
        while ((used & 7u) != 0) {
            if (used + 1 >= json_cap) {
                free(json);
                return OC_ERR_OOM;
            }
            json[used++] = ' ';
        }
        json[used] = '\0';

        raw = (uint8_t *)calloc(data_bytes ? data_bytes : 1, 1);
        if (!raw) {
            free(json);
            return OC_ERR_OOM;
        }
        cursor = 0;
        for (i = 0; i < n; i++) {
            size_t pad = (size_t)offsets_b[i] - cursor;
            cursor += pad;
            memcpy(raw + cursor, blobs[i].data, blobs[i].nbytes);
            cursor += blobs[i].nbytes;
        }
        if (with_cfg) memcpy(raw + cfg_off_b, cfg, (size_t)E4_CFG_N * sizeof(float));

        fp = ext ? ext : fopen(path, "wb");
        if (!fp) {
            free(json);
            free(raw);
            return OC_ERR_IO;
        }
        {
            uint64_t len = (uint64_t)used;
            uint8_t le[8];
            int k;
            for (k = 0; k < 8; k++) le[k] = (uint8_t)((len >> (8 * k)) & 0xff);
            if (fwrite(le, 1, 8, fp) != 8 || fwrite(json, 1, used, fp) != used ||
                (data_bytes && fwrite(raw, 1, data_bytes, fp) != data_bytes)) {
                if (!ext) fclose(fp);
                free(json);
                free(raw);
                return OC_ERR_IO;
            }
        }
        {
            int cerr = ext ? fflush(fp) : fclose(fp);
            free(json);
            free(raw);
            if (cerr != 0) return OC_ERR_IO;
        }
    }
    return OC_OK;
}

OcError oc_eagle4_save_safetensors(const OcEagle4Draft *m, const char *path)
{
    E4Blob blobs[96];
    size_t n = 0;
    float cfg[E4_CFG_N];
    if (!m || !m->loaded || !path) return OC_ERR_INVALID_ARG;
    if (oc_eagle4_path_is_gguf(path)) return OC_ERR_INVALID_ARG;
    cfg_store(m, cfg);
    if (!collect_blobs(m, blobs, &n, 96)) return OC_ERR_OOM;
    return write_safetensors(path, NULL, blobs, n, cfg, 1);
}

OcError oc_eagle4_save_safetensors_fp(const OcEagle4Draft *m, FILE *fp)
{
    E4Blob blobs[96];
    size_t n = 0;
    float cfg[E4_CFG_N];
    if (!m || !m->loaded || !fp) return OC_ERR_INVALID_ARG;
    cfg_store(m, cfg);
    if (!collect_blobs(m, blobs, &n, 96)) return OC_ERR_OOM;
    return write_safetensors(NULL, fp, blobs, n, cfg, 1);
}

/* Expected length of a named norm/router vector. */
static size_t e4_vec_len(const OcEagle4Draft *m, const char *name)
{
    if (strncmp(name, "eagle4.fc_norm.", 15) == 0) return m->config.target_hidden_size;
    if (strstr(name, ".router") != NULL)
        return m->config.n_experts * m->config.hidden_size;
    return m->config.hidden_size;
}

static int e4_w_ok(const OcEagle4Weight *w, size_t rows, size_t cols)
{
    if (!w->data) return 1;
    return (rows == 0 || w->rows == rows) && w->cols == cols;
}

/* Validate every bound weight against the config once at load time. */
static int e4_shapes_ok(const OcEagle4Draft *m)
{
    const OcEagle4Config *c = &m->config;
    size_t h = c->hidden_size;
    size_t qsz = c->num_attention_heads * c->head_dim;
    size_t kvsz = c->num_key_value_heads * c->head_dim;
    size_t inter = c->intermediate_size;
    size_t L, e;
    if (!e4_w_ok(&m->fc, h, c->n_extract_layers * c->target_hidden_size)) return 0;
    if (!e4_w_ok(&m->eh_proj, h, h * 2u)) return 0;
    if (!e4_w_ok(&m->tok_embeddings, 0, h)) return 0;
    if (!e4_w_ok(&m->lm_head, 0, h)) return 0;
    if (!e4_w_ok(&m->lm_down, c->rank, h)) return 0;
    if (!e4_w_ok(&m->lm_score, c->draft_vocab_size, c->rank)) return 0;
    for (L = 0; L < c->num_hidden_layers && L < OC_EAGLE4_MAX_LAYERS; L++) {
        const OcEagle4Layer *ly = &m->layers[L];
        if (!e4_w_ok(&ly->q_proj, qsz, h) || !e4_w_ok(&ly->k_proj, kvsz, h) ||
            !e4_w_ok(&ly->v_proj, kvsz, h) || !e4_w_ok(&ly->o_proj, h, qsz) ||
            !e4_w_ok(&ly->mlp_gate, inter, h) || !e4_w_ok(&ly->mlp_up, inter, h) ||
            !e4_w_ok(&ly->mlp_down, h, inter))
            return 0;
        for (e = 0; e < OC_EAGLE4_MAX_EXPERTS; e++) {
            if (!e4_w_ok(&ly->expert_gate[e], inter, h) ||
                !e4_w_ok(&ly->expert_up[e], inter, h) ||
                !e4_w_ok(&ly->expert_down[e], h, inter))
                return 0;
        }
    }
    return 1;
}

static float bf16_to_f32(uint16_t b)
{
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static OcError copy_tensor_f32(OcEagle4Draft *m, float **vec, OcEagle4Weight *w,
                               const OcSafetensorsTensor *t, const void *data,
                               size_t rows, size_t cols)
{
    const char *dtype = t->dtype;
    size_t n = rows * cols;
    size_t i;
    size_t esz;
    float *dst;
    if (n == 0) return OC_ERR_FORMAT;
    esz = strcmp(dtype, "F32") == 0 ? sizeof(float) : sizeof(uint16_t);
    if (!oc_st_tensor_bytes_ok(t, esz)) return OC_ERR_FORMAT;
    dst = (float *)e4_own(m, n * sizeof(float));
    if (!dst) return OC_ERR_OOM;
    if (strcmp(dtype, "F32") == 0) {
        memcpy(dst, data, n * sizeof(float));
    } else if (strcmp(dtype, "BF16") == 0) {
        const uint16_t *s = (const uint16_t *)data;
        for (i = 0; i < n; i++) dst[i] = bf16_to_f32(s[i]);
    } else if (strcmp(dtype, "F16") == 0) {
        const uint16_t *s = (const uint16_t *)data;
        for (i = 0; i < n; i++) dst[i] = oc_f16_to_f32_bits(s[i]);
    } else {
        return OC_ERR_FORMAT;
    }
    if (vec) *vec = dst;
    if (w) {
        w->data = dst;
        w->rows = rows;
        w->cols = cols;
    }
    return OC_OK;
}

static int tensor_shape(const OcSafetensorsTensor *t, size_t *rows, size_t *cols, int *vec)
{
    if (t->n_dims == 1) {
        *rows = (size_t)t->shape[0];
        *cols = 1;
        *vec = 1;
        return 1;
    }
    if (t->n_dims == 2) {
        *rows = (size_t)t->shape[0];
        *cols = (size_t)t->shape[1];
        *vec = 0;
        return 1;
    }
    return 0;
}

OcError oc_eagle4_load_safetensors(OcEagle4Draft *m, const char *path)
{
    OcSafetensorsFile st;
    const OcSafetensorsTensor *cfg_t = NULL;
    const void *cfg_data = NULL;
    OcEagle4Config cfg;
    size_t n_d2t = 0;
    size_t i;
    OcError err;
    if (!m || !path) return OC_ERR_INVALID_ARG;
    if (oc_eagle4_path_is_gguf(path)) return OC_ERR_INVALID_ARG;
    memset(&st, 0, sizeof(st));
    err = oc_safetensors_open(path, &st);
    if (err != OC_OK) return err;
    err = oc_safetensors_get_tensor(&st, "eagle4.config", &cfg_t);
    if (err != OC_OK) {
        oc_safetensors_close(&st);
        return err;
    }
    err = oc_safetensors_get_tensor_data(&st, cfg_t, &cfg_data);
    if (err != OC_OK || strcmp(cfg_t->dtype, "F32") != 0 || cfg_t->n_dims != 1 ||
        !oc_st_tensor_bytes_ok(cfg_t, sizeof(float))) {
        oc_safetensors_close(&st);
        return OC_ERR_FORMAT;
    }
    if (!cfg_load(&cfg, &n_d2t, (const float *)cfg_data, (size_t)cfg_t->shape[0])) {
        oc_safetensors_close(&st);
        return OC_ERR_FORMAT;
    }
    if (m->impl) oc_eagle4_model_free(m);
    err = oc_eagle4_model_init(m, &cfg);
    if (err != OC_OK) {
        oc_safetensors_close(&st);
        return err;
    }
    for (i = 0; i < st.n_tensors; i++) {
        const OcSafetensorsTensor *t = &st.tensors[i];
        const void *data = NULL;
        size_t rows = 0, cols = 0;
        int vec = 0;
        OcEagle4Weight *w = NULL;
        float **vecp = NULL;
        if (strcmp(t->name, "eagle4.config") == 0) continue;
        if (oc_safetensors_get_tensor_data(&st, t, &data) != OC_OK) {
            oc_eagle4_model_free(m);
            oc_safetensors_close(&st);
            return OC_ERR_FORMAT;
        }
        if (!tensor_shape(t, &rows, &cols, &vec)) continue;
        if (strcmp(t->name, "eagle4.d2t") == 0) {
            size_t n = rows;
            uint32_t *dst;
            size_t k;
            if ((strcmp(t->dtype, "U32") != 0 && strcmp(t->dtype, "I32") != 0) ||
                t->n_dims != 1 || !oc_st_tensor_bytes_ok(t, sizeof(uint32_t))) {
                oc_eagle4_model_free(m);
                oc_safetensors_close(&st);
                return OC_ERR_FORMAT;
            }
            dst = (uint32_t *)e4_own(m, n * sizeof(uint32_t));
            if (!dst) {
                oc_eagle4_model_free(m);
                oc_safetensors_close(&st);
                return OC_ERR_OOM;
            }
            memcpy(dst, data, n * sizeof(uint32_t));
            m->d2t = dst;
            m->n_d2t = n;
            (void)k;
            continue;
        }
        if (strcmp(t->name, "eagle4.fc") == 0) w = &m->fc;
        else if (strcmp(t->name, "eagle4.eh_proj") == 0) w = &m->eh_proj;
        else if (strcmp(t->name, "eagle4.embed") == 0) w = &m->tok_embeddings;
        else if (strcmp(t->name, "eagle4.lm_head") == 0) w = &m->lm_head;
        else if (strcmp(t->name, "eagle4.lm_down") == 0) w = &m->lm_down;
        else if (strcmp(t->name, "eagle4.lm_score") == 0) w = &m->lm_score;
        else if (strcmp(t->name, "eagle4.fc_post_norm") == 0) vecp = &m->fc_post_norm;
        else if (strcmp(t->name, "eagle4.embed_norm") == 0) vecp = &m->embed_norm;
        else if (strcmp(t->name, "eagle4.input_norm") == 0) vecp = &m->input_norm;
        else if (strcmp(t->name, "eagle4.output_norm") == 0) vecp = &m->output_norm;
        else if (strncmp(t->name, "eagle4.fc_norm.", 15) == 0) {
            unsigned idx = 0;
            if (sscanf(t->name + 15, "%u", &idx) == 1 && idx < OC_EAGLE4_MAX_EXTRACT)
                vecp = &m->fc_norm[idx];
        } else if (strncmp(t->name, "eagle4.layer.", 13) == 0) {
            unsigned L = 0, E = 0;
            int pos = -1, epos = -1;
            const char *sfx;
            if (sscanf(t->name, "eagle4.layer.%u.%n", &L, &pos) == 1 && pos > 0 &&
                L < OC_EAGLE4_MAX_LAYERS) {
                OcEagle4Layer *ly = &m->layers[L];
                sfx = t->name + pos;
                if (strcmp(sfx, "q") == 0) w = &ly->q_proj;
                else if (strcmp(sfx, "k") == 0) w = &ly->k_proj;
                else if (strcmp(sfx, "v") == 0) w = &ly->v_proj;
                else if (strcmp(sfx, "o") == 0) w = &ly->o_proj;
                else if (strcmp(sfx, "attn_norm") == 0) vecp = &ly->attn_norm;
                else if (strcmp(sfx, "ffn_norm") == 0) vecp = &ly->ffn_norm;
                else if (strcmp(sfx, "gate") == 0) w = &ly->mlp_gate;
                else if (strcmp(sfx, "up") == 0) w = &ly->mlp_up;
                else if (strcmp(sfx, "down") == 0) w = &ly->mlp_down;
                else if (strcmp(sfx, "router") == 0) vecp = &ly->router;
                else if (sscanf(sfx, "expert.%u.%n", &E, &epos) == 1 && epos > 0 &&
                         E < OC_EAGLE4_MAX_EXPERTS) {
                    sfx += epos;
                    if (strcmp(sfx, "gate") == 0) w = &ly->expert_gate[E];
                    else if (strcmp(sfx, "up") == 0) w = &ly->expert_up[E];
                    else if (strcmp(sfx, "down") == 0) w = &ly->expert_down[E];
                }
            }
        }
        if (!w && !vecp) continue;
        if (vecp && (!vec || rows != e4_vec_len(m, t->name))) {
            oc_eagle4_model_free(m);
            oc_safetensors_close(&st);
            return OC_ERR_FORMAT;
        }
        err = copy_tensor_f32(m, vecp, w, t, data, rows, cols);
        if (err != OC_OK) {
            oc_eagle4_model_free(m);
            oc_safetensors_close(&st);
            return err;
        }
    }
    oc_safetensors_close(&st);
    if (!e4_shapes_ok(m)) {
        oc_eagle4_model_free(m);
        return OC_ERR_FORMAT;
    }
    m->loaded = true;
    return OC_OK;
}

OcError oc_eagle4_write_examples(const char *path, const float *features, size_t n,
                                 size_t width, const uint32_t *tokens, const uint32_t *targets)
{
    E4Blob blobs[3];
    size_t nb = 0;
    if (!path || !features || !tokens || !targets || n == 0 || width == 0)
        return OC_ERR_INVALID_ARG;
    if (oc_eagle4_path_is_gguf(path)) return OC_ERR_INVALID_ARG;
    if (!blob_add(blobs, &nb, 3, "features", features, n, width, 0, 0)) return OC_ERR_OOM;
    if (!blob_add(blobs, &nb, 3, "tokens", tokens, n, 1, 1, 1)) return OC_ERR_OOM;
    if (!blob_add(blobs, &nb, 3, "targets", targets, n, 1, 1, 1)) return OC_ERR_OOM;
    return write_safetensors(path, NULL, blobs, nb, NULL, 0);
}

OcError oc_eagle4_export_gguf(const OcEagle4Draft *m, const char *path)
{
    OcGgufWriter w;
    E4Blob blobs[96];
    size_t n = 0;
    size_t i;
    float cfg[E4_CFG_N];
    OcError err;
    if (!m || !m->loaded || !path) return OC_ERR_INVALID_ARG;
    if (!oc_eagle4_path_is_gguf(path)) return OC_ERR_INVALID_ARG;
    cfg_store(m, cfg);
    if (!collect_blobs(m, blobs, &n, 96)) return OC_ERR_OOM;
    err = oc_gguf_writer_init(path, "eagle4", &w);
    if (err != OC_OK) return err;
    err = oc_gguf_writer_add_string(&w, "eagle4.format", "safetensors");
    if (err == OC_OK)
        err = oc_gguf_writer_add_string(&w, "eagle4.note",
                                        "export only; do not train from this file");
    if (err == OC_OK) {
        uint64_t dims[1];
        dims[0] = E4_CFG_N;
        err = oc_gguf_writer_add_tensor(&w, "eagle4.config", 1, dims, 0, cfg,
                                        (uint64_t)E4_CFG_N * sizeof(float));
    }
    for (i = 0; err == OC_OK && i < n; i++) {
        uint64_t dims[2];
        uint32_t nd = 1;
        dims[0] = blobs[i].cols;
        dims[1] = blobs[i].rows;
        if (!blobs[i].vec && blobs[i].cols != 1) nd = 2;
        else dims[0] = blobs[i].rows;
        if (blobs[i].u32) {
            /* Integer ids up to the vocab fit in f32. Export is not a trainer. */
            size_t count = blobs[i].rows;
            float *tmp = (float *)malloc(count * sizeof(float));
            size_t k;
            const uint32_t *src;
            if (!tmp) {
                oc_gguf_writer_free(&w);
                return OC_ERR_OOM;
            }
            src = (const uint32_t *)blobs[i].data;
            for (k = 0; k < count; k++) tmp[k] = (float)src[k];
            dims[0] = count;
            err = oc_gguf_writer_add_tensor(&w, blobs[i].name, 1, dims, 0, tmp,
                                            (uint64_t)count * sizeof(float));
            free(tmp);
        } else {
            err = oc_gguf_writer_add_tensor(&w, blobs[i].name, nd, dims, 0,
                                            blobs[i].data, blobs[i].nbytes);
        }
    }
    if (err == OC_OK) err = oc_gguf_writer_finalize(&w);
    oc_gguf_writer_free(&w);
    return err;
}

static void ce_grad(const float *logits, size_t n, uint32_t target, float *dlogit,
                    float *loss, uint32_t *pred)
{
    size_t i;
    size_t best = 0;
    float mx = logits[0];
    float sum = 0.0f;
    for (i = 1; i < n; i++)
        if (logits[i] > mx) mx = logits[i];
    for (i = 0; i < n; i++) {
        float e = expf(logits[i] - mx);
        dlogit[i] = e;
        sum += e;
        if (logits[i] > logits[best]) best = i;
    }
    if (sum <= 0.0f) sum = 1.0f;
    if (loss) *loss = -((logits[target] - mx) - logf(sum));
    if (pred) *pred = (uint32_t)best;
    for (i = 0; i < n; i++) {
        float p = dlogit[i] / sum;
        dlogit[i] = p - ((i == (size_t)target) ? 1.0f : 0.0f);
    }
}

static void apply_lowrank(OcEagle4Draft *m, const float *h, uint32_t target,
                          float *dh, float lr)
{
    E4Impl *im = impl_of(m);
    size_t H = m->config.hidden_size;
    size_t dv = m->config.draft_vocab_size;
    size_t R = m->config.rank;
    float *ds;
    float *dproj;
    if (!m->lm_down.data || !m->lm_score.data) return;
    if (m->lm_down.rows != R || m->lm_down.cols != H) return;
    if (m->lm_score.rows != dv || m->lm_score.cols != R) return;
    if ((size_t)target >= dv) return;
    ds = (float *)malloc(dv * sizeof(float));
    dproj = (float *)calloc(R, sizeof(float));
    if (!ds || !dproj) {
        free(ds);
        free(dproj);
        return;
    }
    gemv(m->lm_down.data, R, H, h, im->proj);
    gemv(m->lm_score.data, dv, R, im->proj, im->scores);
    ce_grad(im->scores, dv, target, ds, NULL, NULL);
    sgd_outer(m->lm_score.data, dv, R, ds, im->proj, lr);
    gemv_add_t(m->lm_score.data, dv, R, ds, dproj);
    sgd_outer(m->lm_down.data, R, H, dproj, h, lr);
    gemv_add_t(m->lm_down.data, R, H, dproj, dh);
    free(ds);
    free(dproj);
}

static void trunk_update(OcEagle4Draft *m, const float *dh, float lr)
{
    E4Impl *im = impl_of(m);
    size_t H = m->config.hidden_size;
    size_t inter = m->config.intermediate_size;
    OcEagle4Layer *layer;
    float *d_pre;
    float *d_act;
    float *d_gate;
    float *d_up;
    float *d_ffn;
    size_t i;
    if (m->config.num_hidden_layers != 1) return;
    if (m->config.n_experts > 1) return;
    layer = &m->layers[0];
    d_pre = (float *)malloc(H * sizeof(float));
    d_act = (float *)calloc(inter, sizeof(float));
    d_gate = (float *)calloc(inter, sizeof(float));
    d_up = (float *)calloc(inter, sizeof(float));
    d_ffn = (float *)calloc(H, sizeof(float));
    if (!d_pre || !d_act || !d_gate || !d_up || !d_ffn) {
        free(d_pre);
        free(d_act);
        free(d_gate);
        free(d_up);
        free(d_ffn);
        return;
    }
    memcpy(d_pre, dh, H * sizeof(float));
    if (layer->mlp_gate.data || layer->mlp_up.data || layer->mlp_down.data) {
        if (layer->mlp_down.data && layer->mlp_down.rows == H && layer->mlp_down.cols == inter) {
            sgd_outer(layer->mlp_down.data, H, inter, dh, im->act, lr);
            gemv_add_t(layer->mlp_down.data, H, inter, dh, d_act);
        }
        for (i = 0; i < inter; i++) {
            d_up[i] = d_act[i] * oc_arch_silu(im->gate[i]);
            d_gate[i] = d_act[i] * im->up[i] * silu_d(im->gate[i]);
        }
        if (layer->mlp_up.data && layer->mlp_up.rows == inter && layer->mlp_up.cols == H) {
            sgd_outer(layer->mlp_up.data, inter, H, d_up, im->ffn, lr);
            gemv_add_t(layer->mlp_up.data, inter, H, d_up, d_ffn);
        }
        if (layer->mlp_gate.data && layer->mlp_gate.rows == inter && layer->mlp_gate.cols == H) {
            sgd_outer(layer->mlp_gate.data, inter, H, d_gate, im->ffn, lr);
            gemv_add_t(layer->mlp_gate.data, inter, H, d_gate, d_ffn);
        }
        if (layer->ffn_norm) {
            float *dx = (float *)calloc(H, sizeof(float));
            if (dx) {
                size_t n = H;
                float ss = 0.0f, inv, dot = 0.0f;
                for (i = 0; i < n; i++) ss += im->pre_mlp[i] * im->pre_mlp[i];
                inv = 1.0f / sqrtf(ss / (float)n + m->config.rms_norm_eps);
                for (i = 0; i < n; i++)
                    dot += d_ffn[i] * layer->ffn_norm[i] * im->pre_mlp[i];
                for (i = 0; i < n; i++) {
                    float corr = inv * inv * inv / (float)n * dot;
                    dx[i] = d_ffn[i] * layer->ffn_norm[i] * inv - im->pre_mlp[i] * corr;
                    d_pre[i] += dx[i];
                    layer->ffn_norm[i] -= lr * d_ffn[i] * im->pre_mlp[i] * inv;
                }
                free(dx);
            }
        } else {
            for (i = 0; i < H; i++) d_pre[i] += d_ffn[i];
        }
    }
    if (m->eh_proj.data && m->eh_proj.rows == H && m->eh_proj.cols == H * 2u)
        sgd_outer(m->eh_proj.data, H, H * 2u, d_pre, im->cat, lr);
    free(d_pre);
    free(d_act);
    free(d_gate);
    free(d_up);
    free(d_ffn);
}

static const float *head_hidden(const OcEagle4Draft *m)
{
    const E4Impl *im = impl_of_c(m);
    if (m->output_norm) return im->hidden;
    return im->g;
}

OcError oc_eagle4_eval(OcEagle4Draft *m, const float *features, size_t n_features,
                       uint32_t token, uint32_t target, float *out_loss, uint32_t *out_pred)
{
    E4State st;
    E4Impl *im;
    int saved;
    OcError err;
    float loss = 0.0f;
    uint32_t pred = 0;
    float *dlogit;
    if (!m || !m->loaded) return OC_ERR_INVALID_ARG;
    if ((size_t)target >= m->config.vocab_size) return OC_ERR_INVALID_ARG;
    im = impl_of(m);
    im->snap_used = 0;
    capture_state(m, &st);
    if (features) {
        err = oc_eagle4_encode_features(m, features, n_features);
        if (err != OC_OK) {
            restore_state(m, &st);
            return err;
        }
    }
    saved = m->full_lm_head;
    m->full_lm_head = true;
    err = oc_eagle4_forward(m, token, NULL, 0);
    m->full_lm_head = saved;
    if (err != OC_OK) {
        restore_state(m, &st);
        return err;
    }
    dlogit = (float *)malloc(m->config.vocab_size * sizeof(float));
    if (!dlogit) {
        restore_state(m, &st);
        return OC_ERR_OOM;
    }
    ce_grad(im->logits, m->config.vocab_size, target, dlogit, &loss, &pred);
    free(dlogit);
    restore_state(m, &st);
    if (out_loss) *out_loss = loss;
    if (out_pred) *out_pred = pred;
    return OC_OK;
}

OcError oc_eagle4_train_step(OcEagle4Draft *m, const float *features, size_t n_features,
                             uint32_t token, uint32_t target, float lr,
                             OcEagle4TrainResult *out)
{
    E4State st;
    E4Impl *im;
    int saved;
    OcError err;
    float loss = 0.0f;
    uint32_t pred = 0;
    float *dlogit;
    float *dh;
    const float *h;
    if (!m || !m->loaded) return OC_ERR_INVALID_ARG;
    if ((size_t)target >= m->config.vocab_size) return OC_ERR_INVALID_ARG;
    if (!m->lm_head.data) return OC_ERR_INVALID_ARG;
    /* No target->draft inverse map yet: a d2t draft would train the wrong row. */
    if (m->d2t) return OC_ERR_INVALID_ARG;
    im = impl_of(m);
    im->snap_used = 0;
    capture_state(m, &st);
    if (features) {
        err = oc_eagle4_encode_features(m, features, n_features);
        if (err != OC_OK) {
            restore_state(m, &st);
            return err;
        }
    }
    saved = m->full_lm_head;
    m->full_lm_head = true;
    err = oc_eagle4_forward(m, token, NULL, 0);
    m->full_lm_head = saved;
    if (err != OC_OK) {
        restore_state(m, &st);
        return err;
    }
    dlogit = (float *)malloc(m->config.vocab_size * sizeof(float));
    dh = (float *)calloc(m->config.hidden_size, sizeof(float));
    if (!dlogit || !dh) {
        free(dlogit);
        free(dh);
        restore_state(m, &st);
        return OC_ERR_OOM;
    }
    ce_grad(im->logits, m->config.vocab_size, target, dlogit, &loss, &pred);
    h = head_hidden(m);
    {
        size_t rows = m->lm_head.rows < m->config.vocab_size ? m->lm_head.rows : m->config.vocab_size;
        if (!m->d2t && m->lm_head.cols == m->config.hidden_size) {
            sgd_outer(m->lm_head.data, rows, m->config.hidden_size, dlogit, h, lr);
            gemv_add_t(m->lm_head.data, rows, m->config.hidden_size, dlogit, dh);
        }
    }
    apply_lowrank(m, h, target, dh, lr);
    if (!m->output_norm) trunk_update(m, dh, lr);
    free(dlogit);
    free(dh);
    restore_state(m, &st);
    if (out) {
        out->loss = loss;
        out->pred = pred;
        out->target = target;
        out->steps = 1;
    }
    return OC_OK;
}

/* Returns 0 on success; -1 if an I32 id is negative, or an F32 id is
 * non-finite, non-integral, or out of uint32 range (casting such a float is
 * undefined behaviour). Data offsets are byte offsets, so read via memcpy. */
static int example_at(const void *data, const char *dtype, size_t i, uint32_t *out)
{
    const uint8_t *b = (const uint8_t *)data + i * 4u;
    if (strcmp(dtype, "U32") == 0) {
        memcpy(out, b, sizeof(*out));
        return 0;
    }
    if (strcmp(dtype, "I32") == 0) {
        int32_t v;
        memcpy(&v, b, sizeof(v));
        if (v < 0) return -1;
        *out = (uint32_t)v;
        return 0;
    }
    {
        float v;
        memcpy(&v, b, sizeof(v));
        if (!isfinite(v) || v < 0.0f || v >= 4294967296.0f || floorf(v) != v) return -1;
        *out = (uint32_t)v;
        return 0;
    }
}

/* example_at reads every element as 4 bytes. */
static int e4_example_dtype_ok(const char *dtype)
{
    return strcmp(dtype, "U32") == 0 || strcmp(dtype, "I32") == 0 ||
           strcmp(dtype, "F32") == 0;
}

OcError oc_eagle4_train_safetensors(OcEagle4Draft *m, const char *examples_path,
                                    float lr, uint32_t epochs, OcEagle4TrainResult *out)
{
    OcSafetensorsFile st;
    const OcSafetensorsTensor *ft = NULL;
    const OcSafetensorsTensor *tt = NULL;
    const OcSafetensorsTensor *yt = NULL;
    const void *fd = NULL;
    const void *td = NULL;
    const void *yd = NULL;
    size_t n, width, row, ep;
    OcError err;
    OcEagle4TrainResult last;
    if (!m || !m->loaded || !examples_path) return OC_ERR_INVALID_ARG;
    if (oc_eagle4_path_is_gguf(examples_path)) return OC_ERR_INVALID_ARG;
    memset(&st, 0, sizeof(st));
    err = oc_safetensors_open(examples_path, &st);
    if (err != OC_OK) return err;
    err = oc_safetensors_get_tensor(&st, "features", &ft);
    if (err == OC_OK) err = oc_safetensors_get_tensor(&st, "tokens", &tt);
    if (err == OC_OK) err = oc_safetensors_get_tensor(&st, "targets", &yt);
    if (err == OC_OK) err = oc_safetensors_get_tensor_data(&st, ft, &fd);
    if (err == OC_OK) err = oc_safetensors_get_tensor_data(&st, tt, &td);
    if (err == OC_OK) err = oc_safetensors_get_tensor_data(&st, yt, &yd);
    if (err != OC_OK) {
        oc_safetensors_close(&st);
        return err == OC_ERR_TENSOR ? OC_ERR_FORMAT : err;
    }
    if (strcmp(ft->dtype, "F32") != 0 || ft->n_dims != 2) {
        oc_safetensors_close(&st);
        return OC_ERR_FORMAT;
    }
    n = (size_t)ft->shape[0];
    width = (size_t)ft->shape[1];
    if (tt->n_dims != 1 || yt->n_dims != 1 || (size_t)tt->shape[0] != n ||
        (size_t)yt->shape[0] != n) {
        oc_safetensors_close(&st);
        return OC_ERR_FORMAT;
    }
    if (!e4_example_dtype_ok(tt->dtype) || !e4_example_dtype_ok(yt->dtype) ||
        !oc_st_tensor_bytes_ok(ft, sizeof(float)) || !oc_st_tensor_bytes_ok(tt, 4u) ||
        !oc_st_tensor_bytes_ok(yt, 4u)) {
        oc_safetensors_close(&st);
        return OC_ERR_FORMAT;
    }
    if (epochs == 0) epochs = 1;
    /* Validate every row before the first step, so a bad row cannot leave
     * the draft partially trained. */
    for (row = 0; row < n; row++) {
        uint32_t token, target;
        if (example_at(td, tt->dtype, row, &token) != 0 ||
            example_at(yd, yt->dtype, row, &target) != 0 ||
            (size_t)token >= m->config.vocab_size ||
            (size_t)target >= m->config.vocab_size) {
            oc_safetensors_close(&st);
            return OC_ERR_FORMAT;
        }
    }
    memset(&last, 0, sizeof(last));
    for (ep = 0; ep < epochs; ep++) {
        for (row = 0; row < n; row++) {
            const float *feat = (const float *)fd + row * width;
            uint32_t token = 0, target = 0;
            (void)example_at(td, tt->dtype, row, &token);
            (void)example_at(yd, yt->dtype, row, &target);
            err = oc_eagle4_train_step(m, feat, width, token, target, lr, &last);
            if (err != OC_OK) {
                oc_safetensors_close(&st);
                return err;
            }
            last.steps = (uint32_t)((ep * n) + row + 1u);
        }
    }
    oc_safetensors_close(&st);
    if (out) *out = last;
    return OC_OK;
}

static void fill_unit(float *p, size_t n, uint32_t *seed)
{
    size_t i;
    for (i = 0; i < n; i++) {
        *seed = *seed * 1664525u + 1013904223u;
        p[i] = (float)((*seed >> 8) & 1023) / 1023.0f - 0.5f;
    }
}

OcError oc_eagle4_bench(int iters, OcEagle4BenchReport *out)
{
    OcEagle4Config cfg;
    OcEagle4Draft m;
    OcEagle4Harness h;
    OcEagle4Policy p1, p64, p384;
    OcEagle4Cand cands[4];
    uint32_t chosen[4];
    uint32_t seed = 7;
    int i;
    double t0, t1;
    size_t H = 1024, V = 16384, R = 64, hd = 64, heads = 4, seq = 2048, window = 64;
    float *q = NULL;
    float *k = NULL;
    float *v = NULL;
    float *ao = NULL;
    if (!out) return OC_ERR_INVALID_ARG;
    if (iters < 1) iters = 1;
    memset(out, 0, sizeof(*out));
    oc_eagle4_config_init(&cfg);
    cfg.hidden_size = H;
    cfg.target_hidden_size = H;
    cfg.n_extract_layers = 1;
    cfg.num_attention_heads = heads;
    cfg.num_key_value_heads = heads;
    cfg.head_dim = hd;
    cfg.intermediate_size = H;
    cfg.vocab_size = V;
    cfg.draft_vocab_size = V;
    cfg.rank = R;
    cfg.candidate_k = R;
    cfg.sliding_window = window;
    if (oc_eagle4_model_init(&m, &cfg) != OC_OK) return OC_ERR_OOM;
    if (oc_eagle4_alloc_weight(&m, &m.lm_head, V, H) != OC_OK ||
        oc_eagle4_alloc_weight(&m, &m.lm_down, R, H) != OC_OK ||
        oc_eagle4_alloc_weight(&m, &m.lm_score, V, R) != OC_OK) {
        oc_eagle4_model_free(&m);
        return OC_ERR_OOM;
    }
    fill_unit(m.lm_head.data, V * H, &seed);
    fill_unit(m.lm_down.data, R * H, &seed);
    fill_unit(m.lm_score.data, V * R, &seed);
    fill_unit(impl_of(&m)->g, H, &seed);

    m.full_lm_head = true;
    fill_head(&m, impl_of(&m)->g);
    t0 = oc_clock_wall_sec();
    for (i = 0; i < iters; i++) fill_head(&m, impl_of(&m)->g);
    t1 = oc_clock_wall_sec();
    out->head_full_us = (t1 - t0) * 1e6 / (double)iters;

    m.full_lm_head = false;
    oc_eagle4_reset_stats(&m);
    fill_head(&m, impl_of(&m)->g);
    t0 = oc_clock_wall_sec();
    for (i = 0; i < iters; i++) fill_head(&m, impl_of(&m)->g);
    t1 = oc_clock_wall_sec();
    out->head_lowrank_us = (t1 - t0) * 1e6 / (double)iters;
    if (out->head_lowrank_us > 0.0)
        out->head_speedup = out->head_full_us / out->head_lowrank_us;

    q = (float *)calloc(heads * hd, sizeof(float));
    k = (float *)calloc(seq * heads * hd, sizeof(float));
    v = (float *)calloc(seq * heads * hd, sizeof(float));
    ao = (float *)calloc(heads * hd, sizeof(float));
    if (!q || !k || !v || !ao) {
        free(q);
        free(k);
        free(v);
        free(ao);
        oc_eagle4_model_free(&m);
        return OC_ERR_OOM;
    }
    fill_unit(q, heads * hd, &seed);
    fill_unit(k, seq * heads * hd, &seed);
    fill_unit(v, seq * heads * hd, &seed);
    t0 = oc_clock_wall_sec();
    for (i = 0; i < iters; i++) {
        oc_flash_attention_decode_heads_f32(q, k, v, seq, hd, heads * hd, heads, heads, ao);
    }
    t1 = oc_clock_wall_sec();
    out->attn_full_us = (t1 - t0) * 1e6 / (double)iters;
    t0 = oc_clock_wall_sec();
    for (i = 0; i < iters; i++) {
        size_t start = seq - window;
        oc_flash_attention_decode_heads_f32(
            q, k + start * heads * hd, v + start * heads * hd, window, hd,
            heads * hd, heads, heads, ao);
    }
    t1 = oc_clock_wall_sec();
    out->attn_window_us = (t1 - t0) * 1e6 / (double)iters;
    if (out->attn_window_us > 0.0)
        out->attn_speedup = out->attn_full_us / out->attn_window_us;
    free(q);
    free(k);
    free(v);
    free(ao);

    oc_eagle4_harness_init(&h);
    oc_eagle4_policy_for_batch(&h, 1, 128, &p1);
    oc_eagle4_policy_for_batch(&h, 64, 128, &p64);
    oc_eagle4_policy_for_batch(&h, 384, 128, &p384);
    out->batch1_cap = p1.verify_cap;
    out->batch1_branch = p1.max_branch;
    out->batch64_cap = p64.verify_cap;
    out->batch384_cap = p384.verify_cap;
    for (i = 0; i < 4; i++) {
        cands[i].parent = -1;
        cands[i].cond_prob = 0.4f / (float)(i + 1);
        cands[i].path_prob = cands[i].cond_prob;
    }
    h.min_gain_per_us = 0.0f;
    out->batch1_selected = oc_eagle4_select_level(&h, &p1, 1, 1, 0, cands, 4, chosen, 4);
    oc_eagle4_model_free(&m);
    return OC_OK;
}
