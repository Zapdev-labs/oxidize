/*
 * dflash2.c — DFlash 2 block-diffusion draft model implementation.
 *
 * Port of z-lab/dflash dflash/model.py (DFlash2DraftModel) for the
 * dependency-free C11 port. See dflash2.h for the architecture contract.
 *
 * Reference semantics (dflash/model.py):
 *   - The block forward processes `verify_size` noise rows (embeddings of
 *     [committed, mask, ..., mask]) in ONE parallel pass.
 *   - Attention: k_ctx/v_ctx from the fused target context rows, k_noise/
 *     v_noise from the (conv'd) block hidden; both k streams share one
 *     per-head k_norm; RoPE over true positions; bidirectional sliding
 *     window (is_causal=false, window 2048).
 *   - Grouped dynamic conv is block-local: x[i-t] zero-pads at the start
 *     of THIS forward's token span, not the absolute sequence.
 *   - Selector: top-k over target-lm_head logits, then a greedy (or
 *     temperature-sampled) path via
 *     edge(p->c) = <A[p] o proj(h), B[c]> + unary[c].
 */
#include "oxidize/dflash2.h"
#include "oxidize/parallel.h"
#include "oxidize/safetensors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/* ─── Small helpers ─────────────────────────────────────────────────── */

static void rms_norm_row(const float *x, const float *w, float *out,
                         size_t n, float eps)
{
    float ss = 0.0f;
    for (size_t i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / (float)n + eps);
    for (size_t i = 0; i < n; i++) out[i] = x[i] * r * w[i];
}

static float silu(float x) { return x / (1.0f + expf(-x)); }

/* ─── GEMV (parallel over rows for large matrices) ──────────────────── */

typedef struct OcDFlash2GemvCtx {
    const float *w;
    size_t cols;
    const float *x;
    float *out;
} OcDFlash2GemvCtx;

static void dflash2_gemv_par_fn(size_t begin, size_t end, size_t tid, void *ud)
{
    (void)tid;
    OcDFlash2GemvCtx *c = (OcDFlash2GemvCtx *)ud;
    for (size_t r = begin; r < end; r++) {
        const float *wr = c->w + r * c->cols;
        float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
        size_t ci = 0;
        for (; ci + 4 <= c->cols; ci += 4) {
            acc0 += wr[ci + 0] * c->x[ci + 0];
            acc1 += wr[ci + 1] * c->x[ci + 1];
            acc2 += wr[ci + 2] * c->x[ci + 2];
            acc3 += wr[ci + 3] * c->x[ci + 3];
        }
        for (; ci < c->cols; ci++) acc0 += wr[ci] * c->x[ci];
        c->out[r] = (acc0 + acc1) + (acc2 + acc3);
    }
}

void oc_dflash2_gemv(const float *w, size_t rows, size_t cols,
                     const float *x, float *out)
{
    /* Serial for small matrices; parallel once the work is worth a region. */
    if (rows * cols < (size_t)32768) {
        for (size_t r = 0; r < rows; r++) {
            const float *wr = w + r * cols;
            float acc = 0.0f;
            for (size_t c = 0; c < cols; c++) acc += wr[c] * x[c];
            out[r] = acc;
        }
        return;
    }
    OcDFlash2GemvCtx c = { w, cols, x, out };
    oc_parallel_for(rows, dflash2_gemv_par_fn, &c);
}

/* Convenience wrapper used internally. */
static void gemv(const float *w, size_t rows, size_t cols,
                 const float *x, float *out)
{
    oc_dflash2_gemv(w, rows, cols, x, out);
}

/* GEMM: out[i, :] = W @ x[i, :] for each of `n` input rows.
 * w: [rows, cols]; x: [n, cols]; out: [n, rows]. Weight-stationary: each
 * worker owns a slice of W rows and streams each weight element exactly
 * once per region; the n input rows all reuse the streamed weight row. */
typedef struct OcDFlash2GemmCtx {
    const float *w;
    size_t rows;
    size_t cols;
    const float *x;
    float *out;
    size_t n;
} OcDFlash2GemmCtx;

static void dflash2_gemm_par_fn(size_t begin, size_t end, size_t tid, void *ud)
{
    (void)tid;
    OcDFlash2GemmCtx *c = (OcDFlash2GemmCtx *)ud;
    /* W-row outer loop: weight streamed once per region, x rows stay in
     * cache (x is at most block*4096*4 = 128 KB for the block sizes here). */
    for (size_t r = begin; r < end; r++) {
        const float *wr = c->w + r * c->cols;
        for (size_t i = 0; i < c->n; i++) {
            const float *xi = c->x + i * c->cols;
            float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
            size_t ci = 0;
            for (; ci + 4 <= c->cols; ci += 4) {
                acc0 += wr[ci + 0] * xi[ci + 0];
                acc1 += wr[ci + 1] * xi[ci + 1];
                acc2 += wr[ci + 2] * xi[ci + 2];
                acc3 += wr[ci + 3] * xi[ci + 3];
            }
            for (; ci < c->cols; ci++) acc0 += wr[ci] * xi[ci];
            c->out[i * c->rows + r] = (acc0 + acc1) + (acc2 + acc3);
        }
    }
}

static void gemm(const float *w, size_t rows, size_t cols,
                 const float *x, size_t n, float *out)
{
    if (n == 0) return;
    if (n == 1) {
        gemv(w, rows, cols, x, out);
        return;
    }
    OcDFlash2GemmCtx c = { w, rows, cols, x, out, n };
    oc_parallel_for(rows, dflash2_gemm_par_fn, &c);
}

/* ─── Grouped dynamic causal conv ───────────────────────────────────── */

/*
 * out[i, c] = sum_t (base[t, c] + dyn[i, t, g(c)]) * x[i - t, c]
 * dyn is [len, kernel * groups] with dyn[i, t*groups + g] the dynamic
 * increment for tap t, group g. x[i - t] zero-pads when i < t.
 */
void oc_dflash2_grouped_conv(const float *x, const float *dyn,
                             const float *base,
                             size_t len, size_t hidden,
                             size_t kernel, size_t group_size,
                             float *out)
{
    const size_t groups = hidden / group_size;
    memset(out, 0, len * hidden * sizeof(float));
    for (size_t i = 0; i < len; i++) {
        const float *drow = dyn + i * (kernel * groups);
        float *orow = out + i * hidden;
        for (size_t t = 0; t < kernel && t <= i; t++) {
            const float *xr = x + (i - t) * hidden;
            const float *kbase = base + t * hidden;
            for (size_t g = 0; g < groups; g++) {
                float dyn_t_g = drow[t * groups + g];
                const float *xb = xr + g * group_size;
                const float *kb = kbase + g * group_size;
                float *ob = orow + g * group_size;
                for (size_t c = 0; c < group_size; c++)
                    ob[c] += (kb[c] + dyn_t_g) * xb[c];
            }
        }
    }
}

/* ─── Selector scores ──────────────────────────────────────────────── */

void oc_dflash2_selector_scores(const float *proj_h, const float *A_p,
                                const float *B_k, const float *unary,
                                size_t k, float *scores)
{
    for (size_t ki = 0; ki < k; ki++) {
        const float *b = B_k + ki * OC_DFLASH2_RANK;
        float acc = 0.0f;
        for (size_t r = 0; r < OC_DFLASH2_RANK; r++)
            acc += (A_p[r] * proj_h[r]) * b[r];
        scores[ki] = acc + unary[ki];
    }
}

/* ─── Config ───────────────────────────────────────────────────────── */

void oc_dflash2_config_init(OcDFlash2Config *cfg)
{
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->hidden_size = 4096;
    cfg->intermediate_size = 12288;
    cfg->num_hidden_layers = 5;
    cfg->num_attention_heads = 32;
    cfg->num_key_value_heads = 8;
    cfg->head_dim = 128;
    cfg->vocab_size = 154880;
    cfg->num_target_layers = 45;
    static const size_t ids[5] = { 5, 14, 24, 33, 42 };
    cfg->n_target_layer_ids = 5;
    for (size_t i = 0; i < 5; i++) cfg->target_layer_ids[i] = ids[i];
    cfg->block_size = 8;
    cfg->conv_kernel_size = 2;
    cfg->conv_group_size = 16;
    cfg->mask_token_id = 154856;
    cfg->selector_rank = 256;
    cfg->selector_top_k = 16;
    cfg->sliding_window = 2048;
    cfg->rope_theta = 10000.0f;
    cfg->rms_norm_eps = 1e-5f;
}

/* ─── KV ring ──────────────────────────────────────────────────────── */

OcError oc_dflash2_kvring_init(OcDFlash2KvRing *ring, size_t capacity,
                               size_t n_kv_heads, size_t head_dim)
{
    if (!ring || capacity == 0 || n_kv_heads == 0 || head_dim == 0)
        return OC_ERR_INVALID_ARG;
    memset(ring, 0, sizeof(*ring));
    ring->capacity = capacity;
    ring->n_kv_heads = n_kv_heads;
    ring->head_dim = head_dim;
    size_t vec = n_kv_heads * head_dim;
    ring->k = malloc(capacity * vec * sizeof(float));
    ring->v = malloc(capacity * vec * sizeof(float));
    ring->pos = malloc(capacity * sizeof(int64_t));
    if (!ring->k || !ring->v || !ring->pos) {
        oc_dflash2_kvring_free(ring);
        return OC_ERR_OOM;
    }
    return OC_OK;
}

void oc_dflash2_kvring_free(OcDFlash2KvRing *ring)
{
    if (!ring) return;
    free(ring->k);
    free(ring->v);
    free(ring->pos);
    memset(ring, 0, sizeof(*ring));
}

void oc_dflash2_kvring_clear(OcDFlash2KvRing *ring)
{
    if (!ring) return;
    ring->len = 0;
    ring->total = 0;
}

OcError oc_dflash2_kvring_append(OcDFlash2KvRing *ring,
                                 const float *k, const float *v,
                                 int64_t pos0, size_t n)
{
    if (!ring || !k || !v) return OC_ERR_INVALID_ARG;
    if (n > ring->capacity) return OC_ERR_INVALID_ARG;
    const size_t vec = ring->n_kv_heads * ring->head_dim;
    for (size_t i = 0; i < n; i++) {
        size_t slot = (ring->total + i) % ring->capacity;
        memcpy(ring->k + slot * vec, k + i * vec, vec * sizeof(float));
        memcpy(ring->v + slot * vec, v + i * vec, vec * sizeof(float));
        ring->pos[slot] = pos0 + (int64_t)i;
    }
    ring->total += n;
    ring->len = ring->total < ring->capacity ? ring->total : ring->capacity;
    return OC_OK;
}

void oc_dflash2_kvring_trim(OcDFlash2KvRing *ring, int64_t pos_keep_exclusive)
{
    if (!ring) return;
    /* Drop the newest entries with pos >= pos_keep_exclusive. Writes are
     * in increasing position order, so scan write indices from the end. */
    while (ring->total > 0) {
        size_t w = ring->total - 1;
        size_t slot = w % ring->capacity;
        if (ring->pos[slot] >= pos_keep_exclusive)
            ring->total--;
        else
            break;
    }
    ring->len = ring->total < ring->capacity ? ring->total : ring->capacity;
}

/* ─── Weight loading ────────────────────────────────────────────────── */

static float *load_tensor_f32(const OcSafetensorsFile *f,
                              const char *name, size_t expect_elems)
{
    const OcSafetensorsTensor *t = NULL;
    for (size_t i = 0; i < f->n_tensors; i++) {
        if (strcmp(f->tensors[i].name, name) == 0) {
            t = &f->tensors[i];
            break;
        }
    }
    if (!t) return NULL;
    size_t elems = 1;
    for (uint32_t d = 0; d < t->n_dims; d++) elems *= (size_t)t->shape[d];
    if (elems != expect_elems) return NULL;
    float *dst = malloc(expect_elems * sizeof(float));
    if (!dst) return NULL;
    const uint8_t *raw = (const uint8_t *)f->raw_data + t->data_offset;
    if (strcmp(t->dtype, "F32") == 0) {
        memcpy(dst, raw, expect_elems * 4);
    } else if (strcmp(t->dtype, "BF16") == 0) {
        for (size_t i = 0; i < expect_elems; i++) {
            uint16_t h;
            memcpy(&h, raw + i * 2, 2);
            uint32_t bits = (uint32_t)h << 16;
            memcpy(&dst[i], &bits, 4);
        }
    } else {
        free(dst);
        return NULL;
    }
    return dst;
}

/* Load a [rows, cols] weight; returns OC_ERR_FORMAT when missing/mismatched. */
static OcError load_w(const OcSafetensorsFile *f, const char *name,
                      OcDFlash2Weight *w, size_t rows, size_t cols)
{
    w->generate = NULL;
    w->gen_user = NULL;
    w->data = load_tensor_f32(f, name, rows * cols);
    w->rows = rows;
    w->cols = cols;
    if (!w->data) fprintf(stderr, "dflash2: missing/bad tensor %s\n", name);
    return w->data ? OC_OK : OC_ERR_FORMAT;
}

static OcError load_vec(const OcSafetensorsFile *f, const char *name,
                        float **v, size_t n)
{
    *v = load_tensor_f32(f, name, n);
    if (!*v) fprintf(stderr, "dflash2: missing/bad tensor %s\n", name);
    return *v ? OC_OK : OC_ERR_FORMAT;
}

OcError oc_dflash2_model_load(OcDFlash2Model *m, const char *st_path,
                              const char *config_json_path)
{
    if (!m || !st_path) return OC_ERR_INVALID_ARG;
    memset(m, 0, sizeof(*m));
    oc_dflash2_config_init(&m->cfg);
    /* config.json values for this checkpoint match the defaults; the file
     * is still opened (when provided) to sanity-check the vocab size. */
    if (config_json_path) {
        FILE *fh = fopen(config_json_path, "rb");
        if (fh) {
            char buf[512];
            size_t got = fread(buf, 1, sizeof(buf) - 1, fh);
            buf[got] = '\0';
            fclose(fh);
            long vocab = -1;
            const char *v = strstr(buf, "\"vocab_size\"");
            if (v) vocab = strtol(v + 12, NULL, 10);
            if (vocab > 0 && (size_t)vocab != m->cfg.vocab_size)
                return OC_ERR_FORMAT;
        }
    }

    OcSafetensorsFile f;
    OcError e = oc_safetensors_open(st_path, &f);
    if (e != OC_OK) return e;

    const size_t H = m->cfg.hidden_size;
    const size_t n_layers = m->cfg.num_hidden_layers;
    const size_t n_target_w = m->cfg.n_target_layer_ids * H;

    e = load_w(&f, "fc.weight", &m->fc, H, n_target_w);
    if (e == OC_OK) e = load_vec(&f, "hidden_norm.weight", &m->hidden_norm, H);
    if (e == OC_OK) e = load_vec(&f, "norm.weight", &m->norm, H);
    if (e == OC_OK)
        e = load_w(&f, "candidate_selector.predecessor_codebook",
                   &m->selector.predecessor_codebook,
                   m->cfg.vocab_size, m->cfg.selector_rank);
    if (e == OC_OK)
        e = load_w(&f, "candidate_selector.successor_codebook",
                   &m->selector.successor_codebook,
                   m->cfg.vocab_size, m->cfg.selector_rank);
    if (e == OC_OK)
        e = load_w(&f, "candidate_selector.hidden_projection.weight",
                   &m->selector.hidden_projection,
                   m->cfg.selector_rank, H);

    if (e == OC_OK) {
        m->layers = calloc(n_layers, sizeof(OcDFlash2Layer));
        if (!m->layers) e = OC_ERR_OOM;
    }
    if (e == OC_OK) m->n_layers = n_layers;

    const size_t kvh = m->cfg.num_key_value_heads;
    const size_t hd = m->cfg.head_dim;
    const size_t q_rows = m->cfg.num_attention_heads * hd;
    const size_t kv_rows = kvh * hd;
    const size_t inter = m->cfg.intermediate_size;
    const size_t ksz = m->cfg.conv_kernel_size;
    const size_t groups = H / m->cfg.conv_group_size;

    for (size_t li = 0; e == OC_OK && li < n_layers; li++) {
        char name[160];
        OcDFlash2Layer *L = &m->layers[li];

        snprintf(name, sizeof(name), "layers.%zu.input_layernorm.weight", li);
        e = load_vec(&f, name, &L->input_layernorm, H);
        if (e != OC_OK) break;
        snprintf(name, sizeof(name), "layers.%zu.post_attention_layernorm.weight", li);
        e = load_vec(&f, name, &L->post_attention_layernorm, H);
        if (e != OC_OK) break;

        snprintf(name, sizeof(name), "layers.%zu.self_attn.q_proj.weight", li);
        e = load_w(&f, name, &L->attn.q_proj, q_rows, H);
        if (e != OC_OK) break;
        snprintf(name, sizeof(name), "layers.%zu.self_attn.k_proj.weight", li);
        e = load_w(&f, name, &L->attn.k_proj, kv_rows, H);
        if (e != OC_OK) break;
        snprintf(name, sizeof(name), "layers.%zu.self_attn.v_proj.weight", li);
        e = load_w(&f, name, &L->attn.v_proj, kv_rows, H);
        if (e != OC_OK) break;
        snprintf(name, sizeof(name), "layers.%zu.self_attn.o_proj.weight", li);
        e = load_w(&f, name, &L->attn.o_proj, H, q_rows);
        if (e != OC_OK) break;
        snprintf(name, sizeof(name), "layers.%zu.self_attn.q_norm.weight", li);
        e = load_vec(&f, name, &L->attn.q_norm, hd);
        if (e != OC_OK) break;
        snprintf(name, sizeof(name), "layers.%zu.self_attn.k_norm.weight", li);
        e = load_vec(&f, name, &L->attn.k_norm, hd);
        if (e != OC_OK) break;

        snprintf(name, sizeof(name), "layers.%zu.attention_conv.base_kernel", li);
        e = load_w(&f, name, &L->attn_conv.base_kernel, 2 * ksz, H);
        if (e != OC_OK) break;
        snprintf(name, sizeof(name),
                 "layers.%zu.attention_conv.kernel_projection.weight", li);
        e = load_w(&f, name, &L->attn_conv.kernel_proj, 2 * ksz * groups, H);
        if (e != OC_OK) break;

        snprintf(name, sizeof(name), "layers.%zu.mlp.gate_proj.weight", li);
        e = load_w(&f, name, &L->mlp_gate, inter, H);
        if (e != OC_OK) break;
        snprintf(name, sizeof(name), "layers.%zu.mlp.up_proj.weight", li);
        e = load_w(&f, name, &L->mlp_up, inter, H);
        if (e != OC_OK) break;
        snprintf(name, sizeof(name), "layers.%zu.mlp.down_proj.weight", li);
        e = load_w(&f, name, &L->mlp_down, H, inter);
        if (e != OC_OK) break;

        snprintf(name, sizeof(name), "layers.%zu.mlp_conv.base_kernel", li);
        e = load_w(&f, name, &L->mlp_conv.base_kernel, 2 * ksz, H);
        if (e != OC_OK) break;
        snprintf(name, sizeof(name),
                 "layers.%zu.mlp_conv.kernel_projection.weight", li);
        e = load_w(&f, name, &L->mlp_conv.kernel_proj, 2 * ksz * groups, H);
        if (e != OC_OK) break;
    }

    if (e == OC_OK) {
        m->kv = calloc(n_layers, sizeof(OcDFlash2KvRing));
        if (!m->kv) e = OC_ERR_OOM;
    }
    for (size_t li = 0; e == OC_OK && li < n_layers; li++) {
        size_t cap = m->cfg.sliding_window + m->cfg.block_size;
        e = oc_dflash2_kvring_init(&m->kv[li], cap, kvh, hd);
    }
    if (e == OC_OK) {
        m->target_ctx = malloc(m->cfg.block_size * H * sizeof(float));
        if (!m->target_ctx) e = OC_ERR_OOM;
    }

    oc_safetensors_close(&f);
    if (e != OC_OK) {
        oc_dflash2_model_free(m);
        return e;
    }
    m->loaded = true;
    return OC_OK;
}

void oc_dflash2_model_free(OcDFlash2Model *m)
{
    if (!m) return;
    free(m->fc.data);
    free(m->hidden_norm);
    free(m->norm);
    free(m->selector.predecessor_codebook.data);
    free(m->selector.successor_codebook.data);
    free(m->selector.hidden_projection.data);
    for (size_t li = 0; li < m->n_layers; li++) {
        OcDFlash2Layer *L = &m->layers[li];
        free(L->input_layernorm);
        free(L->post_attention_layernorm);
        free(L->attn.q_proj.data);
        free(L->attn.k_proj.data);
        free(L->attn.v_proj.data);
        free(L->attn.o_proj.data);
        free(L->attn.q_norm);
        free(L->attn.k_norm);
        free(L->attn_conv.base_kernel.data);
        free(L->attn_conv.kernel_proj.data);
        free(L->mlp_gate.data);
        free(L->mlp_up.data);
        free(L->mlp_down.data);
        free(L->mlp_conv.base_kernel.data);
        free(L->mlp_conv.kernel_proj.data);
    }
    free(m->layers);
    if (m->kv) {
        for (size_t li = 0; li < m->n_layers; li++)
            oc_dflash2_kvring_free(&m->kv[li]);
        free(m->kv);
    }
    free(m->target_ctx);
    memset(m, 0, sizeof(*m));
}

void oc_dflash2_reset(OcDFlash2Model *m)
{
    if (!m) return;
    for (size_t li = 0; li < m->n_layers; li++)
        oc_dflash2_kvring_clear(&m->kv[li]);
    m->next_noise_pos = 0;
    m->target_ctx_len = 0;
}
const float *oc_dflash2_last_hidden(const OcDFlash2Model *m, size_t *rows)
{
    if (rows) *rows = m->target_ctx_len;
    return m->target_ctx;
}

/* ─── Context fusion ────────────────────────────────────────────────── */

OcError oc_dflash2_set_context(OcDFlash2Model *m,
                               const float *target_context,
                               size_t n_ctx_rows)
{
    if (!m || !m->loaded || !target_context || n_ctx_rows == 0)
        return OC_ERR_INVALID_ARG;
    if (n_ctx_rows > m->cfg.block_size) return OC_ERR_INVALID_ARG;
    const size_t H = m->cfg.hidden_size;
    const size_t W = m->cfg.n_target_layer_ids * H;

    /* ctx = hidden_norm(fc @ concat(target hiddens)), per row. */
    gemm(m->fc.data, H, W, target_context, n_ctx_rows, m->target_ctx);
    for (size_t r = 0; r < n_ctx_rows; r++)
        rms_norm_row(m->target_ctx + r * H, m->hidden_norm,
                     m->target_ctx + r * H, H, m->cfg.rms_norm_eps);
    m->target_ctx_len = n_ctx_rows;
    /* Ctx rows occupy positions [next_noise_pos - n_ctx, next_noise_pos)
     * in the next propose call (reference:
     * position_ids[:, start - produced : start + verify]); they do not
     * advance the noise start position. */
    return OC_OK;
}

/* ─── Propose (one block forward + selector) ─────────────────────────── */

/* Parallel top-k scan over vocab rows for one draft-hidden row. */
typedef struct OcDFlash2TopKCtx {
    const OcDFlash2Weight *lm;
    const float *x;          /* draft hidden row [H] */
    size_t vocab;
    size_t top_k;
    uint32_t *tidx;          /* [n_threads * top_k] per-thread k-best ids */
    float *tval;             /* [n_threads * top_k] per-thread k-best vals */
    size_t H;
} OcDFlash2TopKCtx;

static void dflash2_topk_par_fn(size_t begin, size_t end, size_t tid,
                                void *ud)
{
    OcDFlash2TopKCtx *c = (OcDFlash2TopKCtx *)ud;
    const size_t top_k = c->top_k;
    const size_t H = c->H;
    uint32_t *idx = c->tidx + tid * top_k;
    float *val = c->tval + tid * top_k;
    float *gen_buf = oc_parallel_scratch(tid, H * sizeof(float));
    if (!gen_buf && !c->lm->data) return;

    for (size_t v = begin; v < end; v++) {
        const float *wr;
        if (c->lm->data) {
            wr = c->lm->data + v * H;
        } else {
            c->lm->generate(v, H, gen_buf, c->lm->gen_user);
            wr = gen_buf;
        }
        const float *xv = c->x;
        float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
        size_t ci = 0;
        for (; ci + 4 <= H; ci += 4) {
            acc0 += wr[ci + 0] * xv[ci + 0];
            acc1 += wr[ci + 1] * xv[ci + 1];
            acc2 += wr[ci + 2] * xv[ci + 2];
            acc3 += wr[ci + 3] * xv[ci + 3];
        }
        for (; ci < H; ci++) acc0 += wr[ci] * xv[ci];
        float v_ = (acc0 + acc1) + (acc2 + acc3);
        if (v_ > val[top_k - 1]) {
            size_t k = top_k - 1;
            val[k] = v_;
            idx[k] = (uint32_t)v;
            while (k > 0 && val[k] > val[k - 1]) {
                float tv = val[k]; val[k] = val[k - 1]; val[k - 1] = tv;
                uint32_t ti = idx[k]; idx[k] = idx[k - 1]; idx[k - 1] = ti;
                k--;
            }
        }
    }
}

typedef struct OcDFlash2ProposeCtx {
    /* Inputs. */
    OcDFlash2Model *m;
    const float *noise_emb;    /* [block, H] */
    const OcDFlash2Weight *lm_head;
    float temperature;
    /* Scratch shared across layer loop. */
    float *hidden;             /* [block, H] current stream */
    float *normed;             /* [block, H] */
    float *conv_dyn;           /* [block, 2*ksz*groups] per-row kernel proj */
    float *conv_dyn_post;      /* saved dynamic kernels (tap 1) [block, ksz*groups] */
    float *conv_scratch;       /* [block, H] */
    float *conv_out;           /* [block, H] */
    float *q;                  /* [block, n_heads*hd] */
    float *k_all;              /* [n_ctx+block, n_kv*hd] */
    float *v_all;              /* [n_ctx+block, n_kv*hd] */
    float *attn_out;           /* [block, n_heads*hd] */
    float *mlp_gu;              /* [block, inter] gate*up */
    float *mlp_out;            /* [block, H] */
    float *logits;             /* [block-1, vocab_scratch] */
} OcDFlash2ProposeCtx;

/*
 * Bidirectional sliding-window attention for the block queries against the
 * KV ring. q: [q_len, n_heads*hd] (normed + RoPE'd); out:
 * [q_len, n_heads*hd]. Mask: |qpos - kpos| < window. Softmax is computed
 * per (query row, query head): each of the n_q_heads heads owns its own
 * max/denominator over the visible ring entries of its kv group.
 */
static void attn_ring(const OcDFlash2KvRing *ring, const float *q,
                      int64_t pos_q0, size_t q_len, size_t n_q_heads,
                      size_t head_dim, float scale, size_t window,
                      float *out)
{
    const size_t n_kv = ring->n_kv_heads;
    const size_t gq = n_q_heads / n_kv;
    const size_t vec = n_kv * head_dim;
    memset(out, 0, q_len * n_q_heads * head_dim * sizeof(float));

    for (size_t qi = 0; qi < q_len; qi++) {
        int64_t qpos = pos_q0 + (int64_t)qi;
        for (size_t h = 0; h < n_q_heads; h++) {
            const size_t hg = h / gq;             /* kv group */
            const float *qh =
                q + (qi * n_q_heads + h) * head_dim;
            float *oh = out + (qi * n_q_heads + h) * head_dim;
            /* pass 1: max score for this query head. */
            float m = -INFINITY;
            for (size_t s = 0; s < ring->len; s++) {
                int64_t dq = qpos - ring->pos[s];
                if (dq >= (int64_t)window || -dq >= (int64_t)window) continue;
                const float *kh = ring->k + s * vec + hg * head_dim;
                float dot = 0.0f;
                for (size_t d = 0; d < head_dim; d++)
                    dot += qh[d] * kh[d];
                float sc = dot * scale;
                if (sc > m) m = sc;
            }
            if (m == -INFINITY) continue;
            /* pass 2: softmax-weighted sum. */
            float denom = 0.0f;
            for (size_t s = 0; s < ring->len; s++) {
                int64_t dq = qpos - ring->pos[s];
                if (dq >= (int64_t)window || -dq >= (int64_t)window) continue;
                const float *kh = ring->k + s * vec + hg * head_dim;
                const float *vh = ring->v + s * vec + hg * head_dim;
                float dot = 0.0f;
                for (size_t d = 0; d < head_dim; d++)
                    dot += qh[d] * kh[d];
                float w = expf(dot * scale - m);
                denom += w;
                for (size_t d = 0; d < head_dim; d++)
                    oh[d] += w * vh[d];
            }
            if (denom > 0.0f) {
                float inv = 1.0f / denom;
                for (size_t d = 0; d < head_dim; d++) oh[d] *= inv;
            }
        }
    }
}

OcError oc_dflash2_propose(OcDFlash2Model *m,
                           const uint32_t *anchor_ids, size_t n_anchor,
                           const float *noise_emb, size_t block,
                           const uint32_t *block_ids,
                           const OcDFlash2Weight *lm_head,
                           float temperature,
                           uint32_t *out_tokens,
                           uint32_t *out_top_k,
                           float *out_top_k_probs)
{
    if (!m || !m->loaded) return OC_ERR_INVALID_ARG;
    if (!anchor_ids || n_anchor == 0) return OC_ERR_INVALID_ARG;
    if (!noise_emb || block == 0 || block > OC_DFLASH2_MAX_BLOCK)
        return OC_ERR_INVALID_ARG;
    if (!lm_head || (!lm_head->data && !lm_head->generate))
        return OC_ERR_INVALID_ARG;
    if (!out_tokens) return OC_ERR_INVALID_ARG;

    const size_t H = m->cfg.hidden_size;
    const size_t hd = m->cfg.head_dim;
    const size_t n_heads = m->cfg.num_attention_heads;
    const size_t n_kv = m->cfg.num_key_value_heads;
    const size_t ksz = m->cfg.conv_kernel_size;
    const size_t gs = m->cfg.conv_group_size;
    const size_t groups = H / gs;
    const size_t inter = m->cfg.intermediate_size;
    const size_t eps = m->cfg.rms_norm_eps;
    const size_t n_ctx = m->target_ctx_len;
    const size_t top_k = m->cfg.selector_top_k;
    const size_t rank = m->cfg.selector_rank;
    const size_t vocab = lm_head->rows;

    /* Absolute positions: context rows [start - n_ctx, start), noise rows
     * [start, start + block) where start = next_noise_pos. */
    const int64_t start = m->next_noise_pos;
    const int64_t ctx_pos0 = start - (int64_t)n_ctx;

    /* Scratch allocations. */
    float *hidden = malloc(block * H * sizeof(float));
    float *normed = malloc(block * H * sizeof(float));
    float *conv_dyn = malloc(block * 2 * ksz * groups * sizeof(float));
    float *conv_dyn_post = malloc(block * ksz * groups * sizeof(float));
    float *conv_scratch = malloc(block * H * sizeof(float));
    float *conv_out = malloc(block * H * sizeof(float));
    float *q = malloc(block * n_heads * hd * sizeof(float));
    float *k_all = malloc((n_ctx + block) * n_kv * hd * sizeof(float));
    float *v_all = malloc((n_ctx + block) * n_kv * hd * sizeof(float));
    float *k_noise = malloc(block * n_kv * hd * sizeof(float));
    float *v_noise = malloc(block * n_kv * hd * sizeof(float));
    float *attn_out = malloc(block * n_heads * hd * sizeof(float));
    float *attn_proj = malloc(block * H * sizeof(float));
    float *mlp_gu = malloc(block * inter * sizeof(float));
    float *mlp_up_out = malloc(block * inter * sizeof(float));
    float *mlp_out = malloc(block * H * sizeof(float));
    float *x = malloc(block * H * sizeof(float));
    if (!hidden || !normed || !conv_dyn || !conv_dyn_post || !conv_scratch ||
        !conv_out || !q || !k_all || !v_all || !k_noise || !v_noise ||
        !attn_out || !attn_proj || !mlp_gu || !mlp_out || !x) {
        free(hidden); free(normed); free(conv_dyn); free(conv_dyn_post);
        free(conv_scratch); free(conv_out); free(q); free(k_all); free(v_all);
        free(k_noise); free(v_noise); free(attn_out); free(attn_proj);
        free(mlp_gu); free(mlp_up_out); free(mlp_out); free(x);
        return OC_ERR_OOM;
    }

    memcpy(hidden, noise_emb, block * H * sizeof(float));

    for (size_t li = 0; li < m->n_layers; li++) {
        OcDFlash2Layer *L = &m->layers[li];
        OcDFlash2KvRing *ring = &m->kv[li];

        /* ── Attention block ──────────────────────────────────── */
        memcpy(x, hidden, block * H * sizeof(float));
        for (size_t i = 0; i < block; i++)
            rms_norm_row(x + i * H, L->input_layernorm, normed + i * H, H, eps);

        /* Conv prepare: dynamic kernels + pre-conv. */
        gemm(L->attn_conv.kernel_proj.data, 2 * ksz * groups, H,
             normed, block, conv_dyn);
        /* dyn view: [block, 2][ksz][groups]; tap-0 kernels at
         * conv_dyn[i, 0*ksz*groups + t*groups + g], tap-1 at +ksz*groups. */
        {
            /* Build tap-0 dyn array in conv_scratch layout
             * [block, ksz*groups]. */
            for (size_t i = 0; i < block; i++) {
                memcpy(conv_scratch + i * ksz * groups,
                       conv_dyn + i * 2 * ksz * groups,
                       ksz * groups * sizeof(float));
                memcpy(conv_dyn_post + i * ksz * groups,
                       conv_dyn + i * 2 * ksz * groups + ksz * groups,
                       ksz * groups * sizeof(float));
            }
            oc_dflash2_grouped_conv(normed, conv_scratch,
                                    L->attn_conv.base_kernel.data,
                                    block, H, ksz, gs, conv_out);
        }

        /* Q from conv'd hidden; K/V from BOTH context and conv'd hidden. */
        gemm(L->attn.q_proj.data, n_heads * hd, H, conv_out, block, q);
        gemm(L->attn.k_proj.data, n_kv * hd, H, m->target_ctx, n_ctx, k_all);
        gemm(L->attn.v_proj.data, n_kv * hd, H, m->target_ctx, n_ctx, v_all);
        gemm(L->attn.k_proj.data, n_kv * hd, H, conv_out, block, k_noise);
        gemm(L->attn.v_proj.data, n_kv * hd, H, conv_out, block, v_noise);

        memcpy(k_all + n_ctx * n_kv * hd, k_noise, block * n_kv * hd * sizeof(float));
        memcpy(v_all + n_ctx * n_kv * hd, v_noise, block * n_kv * hd * sizeof(float));
        const size_t n_k = n_ctx + block;

        /* Per-head RMSNorm: q [block, n_heads, hd]; k_all [n_k, n_kv, hd]. */
        for (size_t i = 0; i < block; i++)
            for (size_t h = 0; h < n_heads; h++) {
                float *qh = q + (i * n_heads + h) * hd;
                float ss = 0.0f;
                for (size_t d = 0; d < hd; d++) ss += qh[d] * qh[d];
                float r = 1.0f / sqrtf(ss / (float)hd + eps);
                for (size_t d = 0; d < hd; d++) qh[d] = qh[d] * r * L->attn.q_norm[d];
            }
        for (size_t i = 0; i < n_k; i++)
            for (size_t h = 0; h < n_kv; h++) {
                float *kh = k_all + (i * n_kv + h) * hd;
                float ss = 0.0f;
                for (size_t d = 0; d < hd; d++) ss += kh[d] * kh[d];
                float r = 1.0f / sqrtf(ss / (float)hd + eps);
                for (size_t d = 0; d < hd; d++) kh[d] = kh[d] * r * L->attn.k_norm[d];
            }

        /* RoPE: k over [ctx_pos0, start + block), q over
         * [start, start + block). theta 10k, head_dim 128. */
        {
            const size_t half = hd / 2;
            float freq[half];
            for (size_t d = 0; d < half; d++)
                freq[d] = powf(m->cfg.rope_theta, -((float)d / (float)half));
            for (size_t i = 0; i < n_k; i++) {
                int64_t pos = ctx_pos0 + (int64_t)i;
                for (size_t h = 0; h < n_kv; h++) {
                    float *kh = k_all + (i * n_kv + h) * hd;
                    for (size_t d = 0; d < half; d++) {
                        float ang = (float)pos * freq[d];
                        float c = cosf(ang), s = sinf(ang);
                        float k0 = kh[d], k1 = kh[d + half];
                        kh[d] = k0 * c - k1 * s;
                        kh[d + half] = k0 * s + k1 * c;
                    }
                }
            }
            for (size_t i = 0; i < block; i++) {
                int64_t pos = start + (int64_t)i;
                for (size_t h = 0; h < n_heads; h++) {
                    float *qh = q + (i * n_heads + h) * hd;
                    for (size_t d = 0; d < half; d++) {
                        float ang = (float)pos * freq[d];
                        float c = cosf(ang), s = sinf(ang);
                        float q0 = qh[d], q1 = qh[d + half];
                        qh[d] = q0 * c - q1 * s;
                        qh[d + half] = q0 * s + q1 * c;
                    }
                }
            }
        }

        /* Append K/V (already RoPE'd) to the ring, then attend. */
        oc_dflash2_kvring_append(ring, k_all, v_all, ctx_pos0, n_k);

        attn_ring(ring, q, start, block, n_heads, hd,
                  1.0f / sqrtf((float)hd), m->cfg.sliding_window, attn_out);

        /* o_proj + conv finish + residual. */
        gemm(L->attn.o_proj.data, H, n_heads * hd, attn_out, block, attn_proj);
        oc_dflash2_grouped_conv(attn_proj, conv_dyn_post,
                                L->attn_conv.base_kernel.data + ksz * H,
                                block, H, ksz, gs, conv_out);
        for (size_t i = 0; i < block * H; i++) hidden[i] += conv_out[i];

        /* ── MLP block ────────────────────────────────────────── */
        memcpy(x, hidden, block * H * sizeof(float));
        for (size_t i = 0; i < block; i++)
            rms_norm_row(x + i * H, L->post_attention_layernorm,
                         normed + i * H, H, eps);

        gemm(L->mlp_conv.kernel_proj.data, 2 * ksz * groups, H,
             normed, block, conv_dyn);
        for (size_t i = 0; i < block; i++) {
            memcpy(conv_scratch + i * ksz * groups,
                   conv_dyn + i * 2 * ksz * groups,
                   ksz * groups * sizeof(float));
            memcpy(conv_dyn_post + i * ksz * groups,
                   conv_dyn + i * 2 * ksz * groups + ksz * groups,
                   ksz * groups * sizeof(float));
        }
        oc_dflash2_grouped_conv(normed, conv_scratch,
                                L->mlp_conv.base_kernel.data,
                                block, H, ksz, gs, conv_out);

        gemm(L->mlp_gate.data, inter, H, conv_out, block, mlp_gu);
        gemm(L->mlp_up.data, inter, H, conv_out, block, mlp_up_out);
        for (size_t i = 0; i < block * inter; i++)
            mlp_gu[i] = silu(mlp_gu[i]) * mlp_up_out[i];
        gemm(L->mlp_down.data, H, inter, mlp_gu, block, mlp_out);
        oc_dflash2_grouped_conv(mlp_out, conv_dyn_post,
                                L->mlp_conv.base_kernel.data + ksz * H,
                                block, H, ksz, gs, conv_out);
        for (size_t i = 0; i < block * H; i++) hidden[i] += conv_out[i];
    }

    /* Final norm on all block rows. */
    for (size_t i = 0; i < block; i++)
        rms_norm_row(hidden + i * H, m->norm, normed + i * H, H, eps);

    /* Draft hidden = rows 1..block-1 (skip the anchor row). */
    const size_t n_draft = block - 1;
    const float *draft_hidden = normed + H;

    /* ── Selector ─────────────────────────────────────────────── */
    /* Logits via the target lm_head for each draft row (parallel over
     * rows for large lm_heads). */
    float *unary = malloc(n_draft * top_k * sizeof(float));
    uint32_t *cand = malloc(n_draft * top_k * sizeof(uint32_t));
    float *proj_h = malloc(n_draft * rank * sizeof(float));
    float *scores = malloc(top_k * sizeof(float));
    float *path_probs = out_top_k_probs;
    if (!unary || !cand || !proj_h || !scores) {
        free(unary); free(cand); free(proj_h); free(scores);
        free(hidden); free(normed); free(conv_dyn); free(conv_dyn_post);
        free(conv_scratch); free(conv_out); free(q); free(k_all); free(v_all);
        free(k_noise); free(v_noise); free(attn_out); free(attn_proj);
        free(mlp_gu); free(mlp_up_out); free(mlp_out); free(x);
        return OC_ERR_OOM;
    }

    /* logits rows computed one at a time (vocab-sized GEMV each). */
    const size_t vocab_eff = vocab < m->cfg.vocab_size ? vocab : m->cfg.vocab_size;
    int logit_err = 0;
    for (size_t p = 0; p < n_draft && !logit_err; p++) {
        if (lm_head->data && vocab_eff <= (size_t)1 << 16) {
            float *lrow = malloc(vocab_eff * sizeof(float));
            if (!lrow) { logit_err = 1; break; }
            gemv(lm_head->data, vocab_eff, H, draft_hidden + p * H, lrow);
            for (size_t k = 0; k < top_k; k++) {
                size_t best = 0;
                for (size_t v = 1; v < vocab_eff; v++)
                    if (lrow[v] > lrow[best]) best = v;
                cand[p * top_k + k] = (uint32_t)best;
                unary[p * top_k + k] = lrow[best];
                lrow[best] = -INFINITY;
            }
            free(lrow);
            continue;
        }
        /* Large-vocab path: parallel streaming top-k over vocab slices.
         * Each worker keeps a local k-best in (tidx, tval); the caller
         * merges. Real (data) and generated (generate) lm_heads exercise
         * identical per-row GEMV FLOPs. */
        {
            const size_t max_t = oc_parallel_n_threads();
            uint32_t *tidx = calloc(max_t * top_k, sizeof(uint32_t));
            float *tval = malloc(max_t * top_k * sizeof(float));
            if (!tidx || !tval) {
                free(tidx); free(tval);
                logit_err = 1;
                break;
            }
            for (size_t i = 0; i < max_t * top_k; i++) tval[i] = -INFINITY;

            OcDFlash2TopKCtx c;
            c.lm = lm_head;
            c.x = draft_hidden + p * H;
            c.vocab = vocab_eff;
            c.top_k = top_k;
            c.tidx = tidx;
            c.tval = tval;
            c.H = H;
            oc_parallel_for(vocab_eff, dflash2_topk_par_fn, &c);

            /* Merge per-thread k-best lists into the final top-k. */
            size_t best_idx[top_k];
            float best_val[top_k];
            for (size_t k = 0; k < top_k; k++) {
                best_val[k] = -INFINITY;
                best_idx[k] = 0;
            }
            for (size_t t = 0; t < max_t; t++) {
                for (size_t k = 0; k < top_k; k++) {
                    float v = tval[t * top_k + k];
                    if (v == -INFINITY) continue;
                    if (v > best_val[top_k - 1]) {
                        size_t j = top_k - 1;
                        best_val[j] = v;
                        best_idx[j] = tidx[t * top_k + k];
                        while (j > 0 && best_val[j] > best_val[j - 1]) {
                            float tv = best_val[j]; best_val[j] = best_val[j-1];
                            best_val[j-1] = tv;
                            size_t ti = best_idx[j]; best_idx[j] = best_idx[j-1];
                            best_idx[j-1] = ti;
                            j--;
                        }
                    }
                }
            }
            for (size_t k = 0; k < top_k; k++) {
                cand[p * top_k + k] = (uint32_t)best_idx[k];
                unary[p * top_k + k] = best_val[k];
            }
            free(tidx);
            free(tval);
        }
    }
    if (logit_err) {
        free(unary); free(cand); free(proj_h); free(scores);
        free(hidden); free(normed); free(conv_dyn); free(conv_dyn_post);
        free(conv_scratch); free(conv_out); free(q); free(k_all); free(v_all);
        free(k_noise); free(v_noise); free(attn_out); free(attn_proj);
        free(mlp_gu); free(mlp_up_out); free(mlp_out); free(x);
        return OC_ERR_OOM;
    }

    /* hidden_projection for each draft row. */
    gemm(m->selector.hidden_projection.data, rank, H,
         draft_hidden, n_draft, proj_h);

    /* Greedy (or temperature) path trace. */
    uint32_t predecessor = anchor_ids[n_anchor - 1];
    for (size_t p = 0; p < n_draft; p++) {
        const float *A_p = m->selector.predecessor_codebook.data +
                           (size_t)predecessor * rank;
        const float *B_k = m->selector.successor_codebook.data;
        float *un = unary + p * top_k;
        /* B rows for the candidates of this position. */
        float *Brows = malloc(top_k * rank * sizeof(float));
        if (!Brows) {
            free(unary); free(cand); free(proj_h); free(scores);
            free(hidden); free(normed); free(conv_dyn); free(conv_dyn_post);
            free(conv_scratch); free(conv_out); free(q); free(k_all);
            free(v_all); free(k_noise); free(v_noise); free(attn_out);
            free(attn_proj); free(mlp_gu); free(mlp_up_out); free(mlp_out); free(x);
            return OC_ERR_OOM;
        }
        for (size_t k = 0; k < top_k; k++)
            memcpy(Brows + k * rank,
                   B_k + (size_t)cand[p * top_k + k] * rank,
                   rank * sizeof(float));
        oc_dflash2_selector_scores(proj_h + p * rank, A_p, Brows, un,
                                   top_k, scores);

        uint32_t choice = 0;
        if (temperature > 0.0f) {
            float mx = scores[0];
            for (size_t k = 1; k < top_k; k++)
                if (scores[k] > mx) mx = scores[k];
            float denom = 0.0f;
            for (size_t k = 0; k < top_k; k++) {
                scores[k] = expf((scores[k] - mx) / temperature);
                denom += scores[k];
            }
            float z = (float)rand() / (float)RAND_MAX * denom;
            float run = 0.0f;
            for (size_t k = 0; k < top_k; k++) {
                run += scores[k];
                if (z <= run) { choice = (uint32_t)k; break; }
            }
            if (out_top_k_probs) {
                for (size_t k = 0; k < top_k; k++)
                    path_probs[p * top_k + k] = scores[k] / denom;
            }
        } else {
            for (size_t k = 1; k < top_k; k++)
                if (scores[k] > scores[choice]) choice = (uint32_t)k;
            if (out_top_k_probs) {
                float mx = scores[choice];
                float denom = 0.0f;
                for (size_t k = 0; k < top_k; k++)
                    denom += expf(scores[k] - mx);
                for (size_t k = 0; k < top_k; k++)
                    path_probs[p * top_k + k] =
                        expf(scores[k] - mx) / denom;
            }
        }

        predecessor = cand[p * top_k + choice];
        out_tokens[p] = predecessor;
        if (out_top_k)
            memcpy(out_top_k + p * top_k, cand + p * top_k,
                   top_k * sizeof(uint32_t));
        free(Brows);
    }

    /* Match the reference cache contract: after each draft forward the
     * cache is cropped to `start` entries (dflash_generate's
     * _crop_to(past_key_values_draft, start)). All noise rows — including
     * the anchor row at `start` — are discarded; committed positions
     * re-enter the ring as target-derived ctx rows on the next step's
     * set_context + append. The caller advances next_noise_pos by the
     * number of accepted tokens. */
    for (size_t li = 0; li < m->n_layers; li++)
        oc_dflash2_kvring_trim(&m->kv[li], start);

    free(unary); free(cand); free(proj_h); free(scores);
    free(hidden); free(normed); free(conv_dyn); free(conv_dyn_post);
    free(conv_scratch); free(conv_out); free(q); free(k_all); free(v_all);
    free(k_noise); free(v_noise); free(attn_out); free(attn_proj);
    free(mlp_gu); free(mlp_up_out); free(mlp_out); free(x);
    (void)block_ids;
    return OC_OK;
}

/* ─── Debug / validation entry points ────────────────────────────────── */

/*
 * Validation backbone-only forward: same math as propose's forward stage
 * but with KV ring management bypassed (ring entries are written, and the
 * ring is restored to its prior state after), and hidden rows returned.
 */
OcError oc_dflash2_forward_debug(OcDFlash2Model *m,
                                 const float *noise_emb, size_t block,
                                 float *out_hidden)
{
    if (!m || !m->loaded || !noise_emb || !out_hidden)
        return OC_ERR_INVALID_ARG;
    if (block == 0 || block > OC_DFLASH2_MAX_BLOCK) return OC_ERR_INVALID_ARG;


    /* Save ring state so validation is side-effect free. */
    const size_t n_layers = m->n_layers;
    size_t saved_total[n_layers];
    for (size_t li = 0; li < n_layers; li++)
        saved_total[li] = m->kv[li].total;

    const size_t H = m->cfg.hidden_size;
    const size_t hd = m->cfg.head_dim;
    const size_t n_heads = m->cfg.num_attention_heads;
    const size_t n_kv = m->cfg.num_key_value_heads;
    const size_t ksz = m->cfg.conv_kernel_size;
    const size_t gs = m->cfg.conv_group_size;
    const size_t groups = H / gs;
    const size_t inter = m->cfg.intermediate_size;
    const float eps = m->cfg.rms_norm_eps;
    const size_t n_ctx = m->target_ctx_len;
    const int64_t start = m->next_noise_pos;
    const int64_t ctx_pos0 = start - (int64_t)n_ctx;

    float *hidden = malloc(block * H * sizeof(float));
    float *normed = malloc(block * H * sizeof(float));
    float *conv_dyn = malloc(block * 2 * ksz * groups * sizeof(float));
    float *conv_dyn_post = malloc(block * ksz * groups * sizeof(float));
    float *conv_scratch = malloc(block * H * sizeof(float));
    float *conv_out = malloc(block * H * sizeof(float));
    float *q = malloc(block * n_heads * hd * sizeof(float));
    float *k_all = malloc((n_ctx + block) * n_kv * hd * sizeof(float));
    float *v_all = malloc((n_ctx + block) * n_kv * hd * sizeof(float));
    float *k_noise = malloc(block * n_kv * hd * sizeof(float));
    float *v_noise = malloc(block * n_kv * hd * sizeof(float));
    float *attn_out = malloc(block * n_heads * hd * sizeof(float));
    float *attn_proj = malloc(block * H * sizeof(float));
    float *mlp_gu = malloc(block * inter * sizeof(float));
    float *mlp_up_out = malloc(block * inter * sizeof(float));
    float *mlp_out = malloc(block * H * sizeof(float));
    float *x = malloc(block * H * sizeof(float));
    if (!hidden || !normed || !conv_dyn || !conv_dyn_post || !conv_scratch ||
        !conv_out || !q || !k_all || !v_all || !k_noise || !v_noise ||
        !attn_out || !attn_proj || !mlp_gu || !mlp_out || !x) {
        free(hidden); free(normed); free(conv_dyn); free(conv_dyn_post);
        free(conv_scratch); free(conv_out); free(q); free(k_all); free(v_all);
        free(k_noise); free(v_noise); free(attn_out); free(attn_proj);
        free(mlp_gu); free(mlp_up_out); free(mlp_out); free(x);
        return OC_ERR_OOM;
    }

    memcpy(hidden, noise_emb, block * H * sizeof(float));

    for (size_t li = 0; li < m->n_layers; li++) {
        
        OcDFlash2Layer *L = &m->layers[li];
        OcDFlash2KvRing *ring = &m->kv[li];

        memcpy(x, hidden, block * H * sizeof(float));
        for (size_t i = 0; i < block; i++)
            rms_norm_row(x + i * H, L->input_layernorm, normed + i * H, H, eps);

        gemm(L->attn_conv.kernel_proj.data, 2 * ksz * groups, H,
             normed, block, conv_dyn);
        for (size_t i = 0; i < block; i++) {
            memcpy(conv_scratch + i * ksz * groups,
                   conv_dyn + i * 2 * ksz * groups,
                   ksz * groups * sizeof(float));
            memcpy(conv_dyn_post + i * ksz * groups,
                   conv_dyn + i * 2 * ksz * groups + ksz * groups,
                   ksz * groups * sizeof(float));
        }
        oc_dflash2_grouped_conv(normed, conv_scratch,
                                L->attn_conv.base_kernel.data,
                                block, H, ksz, gs, conv_out);


        gemm(L->attn.q_proj.data, n_heads * hd, H, conv_out, block, q);
        gemm(L->attn.k_proj.data, n_kv * hd, H, m->target_ctx, n_ctx, k_all);
        gemm(L->attn.v_proj.data, n_kv * hd, H, m->target_ctx, n_ctx, v_all);
        gemm(L->attn.k_proj.data, n_kv * hd, H, conv_out, block, k_noise);
        gemm(L->attn.v_proj.data, n_kv * hd, H, conv_out, block, v_noise);

        memcpy(k_all + n_ctx * n_kv * hd, k_noise,
               block * n_kv * hd * sizeof(float));
        memcpy(v_all + n_ctx * n_kv * hd, v_noise,
               block * n_kv * hd * sizeof(float));
        const size_t n_k = n_ctx + block;

        for (size_t i = 0; i < block; i++)
            for (size_t h = 0; h < n_heads; h++) {
                float *qh = q + (i * n_heads + h) * hd;
                float ss = 0.0f;
                for (size_t d = 0; d < hd; d++) ss += qh[d] * qh[d];
                float r = 1.0f / sqrtf(ss / (float)hd + eps);
                for (size_t d = 0; d < hd; d++)
                    qh[d] = qh[d] * r * L->attn.q_norm[d];
            }
        for (size_t i = 0; i < n_k; i++)
            for (size_t h = 0; h < n_kv; h++) {
                float *kh = k_all + (i * n_kv + h) * hd;
                float ss = 0.0f;
                for (size_t d = 0; d < hd; d++) ss += kh[d] * kh[d];
                float r = 1.0f / sqrtf(ss / (float)hd + eps);
                for (size_t d = 0; d < hd; d++)
                    kh[d] = kh[d] * r * L->attn.k_norm[d];
            }

        {
            const size_t half = hd / 2;
            float freq[half];
            for (size_t d = 0; d < half; d++)
                freq[d] = powf(m->cfg.rope_theta, -((float)d / (float)half));
            for (size_t i = 0; i < n_k; i++) {
                int64_t pos = ctx_pos0 + (int64_t)i;
                for (size_t h = 0; h < n_kv; h++) {
                    float *kh = k_all + (i * n_kv + h) * hd;
                    for (size_t d = 0; d < half; d++) {
                        float ang = (float)pos * freq[d];
                        float c = cosf(ang), s = sinf(ang);
                        float k0 = kh[d], k1 = kh[d + half];
                        kh[d] = k0 * c - k1 * s;
                        kh[d + half] = k0 * s + k1 * c;
                    }
                }
            }
            for (size_t i = 0; i < block; i++) {
                int64_t pos = start + (int64_t)i;
                for (size_t h = 0; h < n_heads; h++) {
                    float *qh = q + (i * n_heads + h) * hd;
                    for (size_t d = 0; d < half; d++) {
                        float ang = (float)pos * freq[d];
                        float c = cosf(ang), s = sinf(ang);
                        float q0 = qh[d], q1 = qh[d + half];
                        qh[d] = q0 * c - q1 * s;
                        qh[d + half] = q0 * s + q1 * c;
                    }
                }
            }
        }

        oc_dflash2_kvring_append(ring, k_all, v_all, ctx_pos0, n_k);
        attn_ring(ring, q, start, block, n_heads, hd,
                  1.0f / sqrtf((float)hd), m->cfg.sliding_window, attn_out);

        gemm(L->attn.o_proj.data, H, n_heads * hd, attn_out, block, attn_proj);
        oc_dflash2_grouped_conv(attn_proj, conv_dyn_post,
                                L->attn_conv.base_kernel.data + ksz * H,
                                block, H, ksz, gs, conv_out);
        for (size_t i = 0; i < block * H; i++) hidden[i] += conv_out[i];

        /* ── MLP block ────────────────────────────────────────── */
        memcpy(x, hidden, block * H * sizeof(float));
        for (size_t i = 0; i < block; i++)
            rms_norm_row(x + i * H, L->post_attention_layernorm,
                         normed + i * H, H, eps);

        gemm(L->mlp_conv.kernel_proj.data, 2 * ksz * groups, H,
             normed, block, conv_dyn);
        for (size_t i = 0; i < block; i++) {
            memcpy(conv_scratch + i * ksz * groups,
                   conv_dyn + i * 2 * ksz * groups,
                   ksz * groups * sizeof(float));
            memcpy(conv_dyn_post + i * ksz * groups,
                   conv_dyn + i * 2 * ksz * groups + ksz * groups,
                   ksz * groups * sizeof(float));
        }
        oc_dflash2_grouped_conv(normed, conv_scratch,
                                L->mlp_conv.base_kernel.data,
                                block, H, ksz, gs, conv_out);

        gemm(L->mlp_gate.data, inter, H, conv_out, block, mlp_gu);
        gemm(L->mlp_up.data, inter, H, conv_out, block, mlp_up_out);
        for (size_t i = 0; i < block * inter; i++)
            mlp_gu[i] = silu(mlp_gu[i]) * mlp_up_out[i];
        gemm(L->mlp_down.data, H, inter, mlp_gu, block, mlp_out);
        oc_dflash2_grouped_conv(mlp_out, conv_dyn_post,
                                L->mlp_conv.base_kernel.data + ksz * H,
                                block, H, ksz, gs, conv_out);
        for (size_t i = 0; i < block * H; i++) hidden[i] += conv_out[i];

    }

    for (size_t i = 0; i < block; i++)
        rms_norm_row(hidden + i * H, m->norm, out_hidden + i * H, H, eps);

    /* Restore ring state. */
    for (size_t li = 0; li < n_layers; li++) {
        m->kv[li].total = saved_total[li];
        m->kv[li].len = saved_total[li] < m->kv[li].capacity
                            ? saved_total[li] : m->kv[li].capacity;
    }

    free(hidden); free(normed); free(conv_dyn); free(conv_dyn_post);
    free(conv_scratch); free(conv_out); free(q); free(k_all); free(v_all);
    free(k_noise); free(v_noise); free(attn_out); free(attn_proj);
    free(mlp_gu); free(mlp_up_out); free(mlp_out); free(x);
    return OC_OK;
}

/*
 * Validation selector-only: mirror CandidateSelector.select with
 * temperature 0 on precomputed draft hidden rows. out_tokens is [n_rows].
 */
OcError oc_dflash2_selector_debug(OcDFlash2Model *m,
                                  const float *draft_hidden, size_t n_rows,
                                  const uint32_t *anchor_ids, size_t n_anchor,
                                  const OcDFlash2Weight *lm_head,
                                  uint32_t *out_tokens,
                                  uint32_t *out_cand)
{
    if (!m || !m->loaded || !draft_hidden || !anchor_ids || n_anchor == 0 ||
        !lm_head || (!lm_head->data && !lm_head->generate) || !out_tokens)
        return OC_ERR_INVALID_ARG;

    const size_t H = m->cfg.hidden_size;
    const size_t top_k = m->cfg.selector_top_k;
    const size_t rank = m->cfg.selector_rank;
    const size_t vocab = lm_head->rows;

    float *unary = malloc(n_rows * top_k * sizeof(float));
    uint32_t *cand = malloc(n_rows * top_k * sizeof(uint32_t));
    float *proj_h = malloc(n_rows * rank * sizeof(float));
    float *scores = malloc(top_k * sizeof(float));
    if (!unary || !cand || !proj_h || !scores) {
        free(unary); free(cand); free(proj_h); free(scores);
        return OC_ERR_OOM;
    }

    for (size_t p = 0; p < n_rows; p++) {
        /* full logits row then top-k selection */
        float *lrow = malloc(vocab * sizeof(float));
        if (!lrow) {
            free(unary); free(cand); free(proj_h); free(scores);
            return OC_ERR_OOM;
        }
        gemv(lm_head->data, vocab, H, draft_hidden + p * H, lrow);
        for (size_t k = 0; k < top_k; k++) {
            size_t best = 0;
            for (size_t v = 1; v < vocab; v++)
                if (lrow[v] > lrow[best]) best = v;
            cand[p * top_k + k] = (uint32_t)best;
            unary[p * top_k + k] = lrow[best];
            lrow[best] = -INFINITY;
        }
        free(lrow);
    }

    gemm(m->selector.hidden_projection.data, rank, H,
         draft_hidden, n_rows, proj_h);

    uint32_t predecessor = anchor_ids[n_anchor - 1];
    for (size_t p = 0; p < n_rows; p++) {
        const float *A_p = m->selector.predecessor_codebook.data +
                           (size_t)predecessor * rank;
        const float *B_base = m->selector.successor_codebook.data;
        float *un = unary + p * top_k;
        for (size_t k = 0; k < top_k; k++) {
            const float *B_row =
                B_base + (size_t)cand[p * top_k + k] * rank;
            float acc = 0.0f;
            for (size_t r = 0; r < rank; r++)
                acc += (A_p[r] * proj_h[p * rank + r]) * B_row[r];
            scores[k] = acc + un[k];
        }
        uint32_t choice = 0;
        for (size_t k = 1; k < top_k; k++)
            if (scores[k] > scores[choice]) choice = (uint32_t)k;
        predecessor = cand[p * top_k + choice];
        out_tokens[p] = predecessor;
        if (out_cand)
            memcpy(out_cand + p * top_k, cand + p * top_k,
                   top_k * sizeof(uint32_t));
    }

    free(unary); free(cand); free(proj_h); free(scores);
    return OC_OK;
}
