/*
 * inkling.c — Inkling-2 loop transformer (inkling2_loop) loader and forward.
 *
 * One forward call runs up to OC_INK_MAX_BATCH tokens through all 48 layer
 * visits. Projections are batched across the call's tokens (one pass over
 * each weight), the MoE groups tokens by expert, and the per-visit
 * short-convolution inputs of the call are kept so a speculative verify can
 * roll back to any accepted prefix.
 */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/inkling.h"

#include "inkling_gpu.h"
#include "oxidize/log.h"
#include "oxidize/nvfp4.h"
#include "oxidize/parallel.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GT_F32   0u
#define GT_BF16  30u
#define GT_NVFP4 40u

/* ─── profiler (OC_INK_PROF=1) ─────────────────────────────────────────── */

#include <time.h>

enum { PF_MIX, PF_CTRL, PF_NORM, PF_PROJ, PF_CONV, PF_ATTN, PF_WO, PF_ROUTER, PF_GATEUP, PF_DOWN,
       PF_COMBINE, PF_LMHEAD, PF__N };
static const char *const PF_NAMES[PF__N] = { "depth_mix", "loop_ctrl", "norm/clip", "qkv_proj", "conv",
                                             "attention", "attn_out", "router", "moe_gate_up",
                                             "moe_down", "moe_combine", "lm_head" };
static double g_prof[PF__N];
static uint64_t g_prof_tokens, g_prof_calls;
static int g_prof_on = -1;

static double prof_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static bool prof_on(void)
{
    if (g_prof_on < 0) g_prof_on = getenv("OC_INK_PROF") != NULL;
    return g_prof_on == 1;
}

#define PROF(id, stmt) do {                              \
        const double t0_ = prof_on() ? prof_now() : 0.0; \
        stmt;                                            \
        if (prof_on()) g_prof[id] += prof_now() - t0_;   \
    } while (0)

static void prof_report(void)
{
    if (!prof_on() || g_prof_calls == 0) return;
    double tot = 0.0;
    for (int i = 0; i < PF__N; i++) tot += g_prof[i];
    fprintf(stderr, "[inkling-prof] %llu calls, %llu tokens, %.3f s profiled\n",
            (unsigned long long)g_prof_calls, (unsigned long long)g_prof_tokens, tot);
    for (int i = 0; i < PF__N; i++)
        fprintf(stderr, "[inkling-prof]   %-12s %8.3f s  %5.1f%%  %8.3f ms/call\n", PF_NAMES[i], g_prof[i],
                tot > 0 ? 100.0 * g_prof[i] / tot : 0.0, 1e3 * g_prof[i] / (double)g_prof_calls);
    memset(g_prof, 0, sizeof g_prof);
    g_prof_tokens = g_prof_calls = 0;
}

#include "inkling_simd.h"

/* ─── linear layers ─────────────────────────────────────────────────────── */

#define LIN_CHUNK 16
#define MAX_SEGS  8

enum { POST_NONE = 0, POST_SOFTCAP = 1 };

/* One output block of a fused projection region: y[t] = W[mat] x[t]. */
typedef struct {
    const OcInkMat   *W;
    size_t            mat;
    const float      *x;          /* f32 input (BF16 / F32 weights) */
    size_t            x_stride;
    const OcNvfp4Act *acts;       /* quantized input (NVFP4 weights) */
    float            *y;
    size_t            y_stride;
    int               post;
    float             post_arg;
} Seg;

typedef struct {
    Seg    seg[MAX_SEGS];
    size_t first[MAX_SEGS + 1];   /* first work item of each segment */
    size_t n_seg, n;
} LinJob;

static void lin_rows(size_t b, size_t e, size_t tid, void *u)
{
    (void)tid;
    const LinJob *j = u;
    size_t si = 0;
    for (size_t it = b; it < e; it++) {
        while (it >= j->first[si + 1]) si++;
        const Seg *g = &j->seg[si];
        const OcInkMat *W = g->W;
        const uint8_t *base = W->data + g->mat * W->mat_bytes;
        size_t r0 = (it - j->first[si]) * LIN_CHUNK, r1 = r0 + LIN_CHUNK;
        if (r1 > W->rows) r1 = W->rows;
        if (W->gtype == GT_NVFP4) {
            float gs = W->scale ? W->scale[g->mat] : 1.0f;
            oc_nvfp4_gemm_rows(base, W->cols, r0, r1, gs, g->acts, j->n, g->y, g->y_stride);
        } else {
            for (size_t r = r0; r < r1; r++) {
                const uint8_t *row = base + r * W->row_bytes;
                for (size_t t = 0; t < j->n; t++) {
                    const float *x = g->x + t * g->x_stride;
                    g->y[t * g->y_stride + r] = W->gtype == GT_BF16 ? ink_dot_bf16((const uint16_t *)row, x, W->cols)
                                                                    : ink_dot((const float *)row, x, W->cols);
                }
            }
        }
        if (g->post == POST_SOFTCAP)
            for (size_t t = 0; t < j->n; t++) ink_softcap(&g->y[t * g->y_stride + r0], r1 - r0, g->post_arg);
    }
}

static void lin_add(LinJob *j, const OcInkMat *W, size_t mat, const float *x, size_t x_stride,
                    const OcNvfp4Act *acts, float *y, size_t y_stride)
{
    Seg *g = &j->seg[j->n_seg++];
    *g = (Seg){ W, mat, x, x_stride, acts, y, y_stride, POST_NONE, 0.0f };
}

static void lin_run(LinJob *j)
{
    j->first[0] = 0;
    for (size_t i = 0; i < j->n_seg; i++)
        j->first[i + 1] = j->first[i] + (j->seg[i].W->rows + LIN_CHUNK - 1) / LIN_CHUNK;
    oc_parallel_for(j->first[j->n_seg], lin_rows, j);
}

/* Activation quantization buffers for up to n vectors of up to cols values. */
typedef struct {
    OcNvfp4Act *acts;
    int8_t     *q;
    float      *d;
    size_t      cap_vec, cap_cols;
} ActBuf;

static OcError actbuf_init(ActBuf *a, size_t n_vec, size_t cols)
{
    a->acts = calloc(n_vec, sizeof *a->acts);
    a->q = aligned_alloc(64, n_vec * cols);
    a->d = calloc(n_vec * (cols / 64), sizeof(float));
    a->cap_vec = n_vec;
    a->cap_cols = cols;
    return a->acts && a->q && a->d ? OC_OK : OC_ERR_OOM;
}

static void actbuf_free(ActBuf *a)
{
    free(a->acts);
    free(a->q);
    free(a->d);
}

static void quant_acts(ActBuf *a, const float *x, size_t x_stride, size_t n, size_t cols)
{
    for (size_t t = 0; t < n; t++) {
        a->acts[t].q = a->q + t * a->cap_cols;
        a->acts[t].d = a->d + t * (a->cap_cols / 64);
        oc_nvfp4_quantize_act(x + t * x_stride, cols, a->acts[t]);
    }
}

/* y[t] = W[mat] x[t] for t < n. NVFP4 weights quantize x into ab first. */
static void linear(const OcInkMat *W, size_t mat, const float *x, size_t x_stride, size_t n,
                   float *y, size_t y_stride, ActBuf *ab)
{
    if (W->gtype == GT_NVFP4) quant_acts(ab, x, x_stride, n, W->cols);
    LinJob j = { .n = n };
    lin_add(&j, W, mat, x, x_stride, ab->acts, y, y_stride);
    lin_run(&j);
}

/* ─── loading ───────────────────────────────────────────────────────────── */

typedef struct {
    OcInklingModel *m;
    OcError         err;
} Ld;

static const OcGgufFile *meta(const OcInklingModel *m) { return &m->map.unified; }

static uint32_t mu32(Ld *L, const char *key, bool required)
{
    uint32_t v = 0;
    char k[128];
    snprintf(k, sizeof k, "inkling2_loop.%s", key);
    if (!oc_gguf_metadata_get_u32(meta(L->m), k, &v) && required) {
        oc_log(OC_LOG_ERROR, "inkling: missing metadata %s", k);
        L->err = OC_ERR_FORMAT;
    }
    return v;
}

static float mf32(Ld *L, const char *key)
{
    float v = 0.0f;
    char k[128];
    snprintf(k, sizeof k, "inkling2_loop.%s", key);
    if (!oc_gguf_metadata_get_f32(meta(L->m), k, &v)) {
        oc_log(OC_LOG_ERROR, "inkling: missing metadata %s", k);
        L->err = OC_ERR_FORMAT;
    }
    return v;
}

static bool load_mat(const OcGgufMmappedFile *map, const char *name, OcInkMat *out)
{
    const OcGgufTensorInfo *ti = oc_gguf_map_tensor_get(map, name);
    if (!ti) return false;
    memset(out, 0, sizeof *out);
    out->gtype = ti->ggml_type;
    out->data = oc_gguf_map_tensor_data(map, ti);
    out->cols = (size_t)ti->dims[0];
    out->rows = ti->n_dims > 1 ? (size_t)ti->dims[1] : 1;
    out->n_mat = ti->n_dims > 2 ? (size_t)ti->dims[2] : 1;
    out->row_bytes = out->gtype == GT_NVFP4 ? oc_nvfp4_row_bytes(out->cols)
                   : out->gtype == GT_BF16 ? out->cols * 2 : out->cols * 4;
    out->mat_bytes = out->row_bytes * out->rows;
    if (out->gtype == GT_NVFP4) {
        char sn[160];
        size_t len = strlen(name);
        snprintf(sn, sizeof sn, "%.*s.scale", (int)(len > 7 ? len - 7 : len), name);
        const OcGgufTensorInfo *si = oc_gguf_map_tensor_get(map, sn);
        out->scale = si ? (const float *)oc_gguf_map_tensor_data(map, si) : NULL;
    }
    return out->gtype == GT_NVFP4 || out->gtype == GT_BF16 || out->gtype == GT_F32;
}

OcError oc_ink_load_mat(const OcGgufMmappedFile *map, const char *name, OcInkMat *out, bool required);
OcError oc_ink_load_mat(const OcGgufMmappedFile *map, const char *name, OcInkMat *out, bool required)
{
    if (load_mat(map, name, out)) return OC_OK;
    memset(out, 0, sizeof *out);
    if (required) oc_log(OC_LOG_ERROR, "inkling: missing or unsupported tensor %s", name);
    return required ? OC_ERR_FORMAT : OC_OK;
}

static const float *f32_tensor(Ld *L, const char *name, bool required)
{
    const OcGgufTensorInfo *ti = oc_gguf_map_tensor_get(&L->m->map, name);
    if (!ti || ti->ggml_type != GT_F32) {
        if (required) {
            oc_log(OC_LOG_ERROR, "inkling: missing F32 tensor %s", name);
            L->err = OC_ERR_FORMAT;
        }
        return NULL;
    }
    return (const float *)oc_gguf_map_tensor_data(&L->m->map, ti);
}

static void mat(Ld *L, const char *name, OcInkMat *out, bool required)
{
    OcError e = oc_ink_load_mat(&L->m->map, name, out, required);
    if (e != OC_OK) L->err = e;
}

bool oc_inkling_gguf_is_inkling(const char *path)
{
    OcGgufMmappedFile map;
    if (oc_gguf_map_open(path, &map) != OC_OK) return false;
    const char *arch = NULL;
    size_t len = 0;
    bool ok = oc_gguf_metadata_get_str(&map.unified, "general.architecture", &arch, &len) &&
              len == 13 && strncmp(arch, "inkling2_loop", 13) == 0;
    oc_gguf_map_free(&map);
    return ok;
}

OcError oc_inkling_load(const char *path, OcInklingModel *m)
{
    memset(m, 0, sizeof *m);
    OcError e = oc_gguf_map_open(path, &m->map);
    if (e != OC_OK) return e;
    Ld L = { m, OC_OK };

    m->n_embd = mu32(&L, "embedding_length", true);
    m->n_layer = mu32(&L, "block_count", true);
    m->n_vocab = mu32(&L, "vocab_size", true);
    m->n_head = mu32(&L, "attention.head_count", true);
    m->n_head_kv = mu32(&L, "attention.head_count_kv", true);
    m->head_dim = mu32(&L, "attention.key_length", true);
    m->n_swa = mu32(&L, "attention.sliding_window", true);
    m->n_rel = mu32(&L, "attention.relative_length", true);
    m->rel_extent = mu32(&L, "attention.relative_extent", true);
    m->log_floor = mu32(&L, "attention.log_scale_floor", true);
    m->n_expert = mu32(&L, "expert_count", true);
    m->n_expert_used = mu32(&L, "expert_used_count", true);
    m->n_ff_exp = mu32(&L, "expert_feed_forward_length", true);
    m->res_block = mu32(&L, "attn_res.block_size", true);
    m->n_conv = mu32(&L, "shortconv.l_cache", true);
    m->core_start = mu32(&L, "loop.start", true);
    m->core_end = mu32(&L, "loop.end", true);
    m->n_passes = mu32(&L, "loop.count", true);
    m->gate_rank = mu32(&L, "loop.gate_rank", true);
    m->n_ctx_train = mu32(&L, "context_length", false);
    m->eps = mf32(&L, "attention.layer_norm_rms_epsilon");
    m->qk_clip = mf32(&L, "attention.qk_clip");
    m->rel_clip = mf32(&L, "attention.relative_clip");
    m->log_alpha = mf32(&L, "attention.log_scale_alpha");
    m->situ_beta = mf32(&L, "activation.situ_beta");
    m->situ_lbeta = mf32(&L, "activation.situ_linear_beta");
    m->route_scale = mf32(&L, "expert_weights_scale");
    m->softcap = mf32(&L, "final_logit_softcapping");
    if (L.err != OC_OK) goto fail;
    if (m->n_layer > OC_INK_MAX_LAYERS || m->n_conv != 4 || m->head_dim % 64 || m->n_embd % 64 ||
        m->n_head % m->n_head_kv || m->n_expert_used > 16 || m->core_end > m->n_layer ||
        m->core_start >= m->core_end || m->n_passes < 1 || m->n_rel > 64) {
        e = OC_ERR_MODEL;
        goto fail;
    }

    const OcGgufMetadataValue *pat = oc_gguf_metadata_get(meta(m), "inkling2_loop.attention.sliding_window_pattern");
    if (!pat || pat->v.arr.len != m->n_layer) { e = OC_ERR_FORMAT; goto fail; }
    for (uint32_t i = 0; i < m->n_layer; i++) m->layers[i].swa = pat->v.arr.values[i].v.b;

    /* visit schedule */
    uint32_t v = 0;
    for (uint32_t il = 0; il < m->core_start; il++) { m->visit_layer[v] = (uint8_t)il; m->visit_pass[v++] = 0; }
    for (uint32_t p = 0; p < m->n_passes; p++)
        for (uint32_t il = m->core_start; il < m->core_end; il++) { m->visit_layer[v] = (uint8_t)il; m->visit_pass[v++] = (uint8_t)p; }
    for (uint32_t il = m->core_end; il < m->n_layer; il++) { m->visit_layer[v] = (uint8_t)il; m->visit_pass[v++] = 0; }
    m->n_visits = v;

    const OcGgufTensorInfo *te = oc_gguf_map_tensor_get(&m->map, "token_embd.weight");
    if (!te || te->ggml_type != GT_BF16 || te->dims[0] != m->n_embd || te->dims[1] < m->n_vocab) {
        e = OC_ERR_FORMAT;
        goto fail;
    }
    m->tok_embd = (const uint16_t *)oc_gguf_map_tensor_data(&m->map, te);
    m->tok_norm = f32_tensor(&L, "token_embd_norm.weight", true);
    m->output_norm = f32_tensor(&L, "output_norm.weight", true);
    m->output_res_query = f32_tensor(&L, "output_res_query.weight", true);
    mat(&L, "output.weight", &m->output, true);

    char n[128];
    for (uint32_t i = 0; i < m->n_layer && L.err == OC_OK; i++) {
        OcInkLayer *ly = &m->layers[i];
#define N(s) (snprintf(n, sizeof n, "blk.%u." s, i), n)
        mat(&L, N("attn_q.weight"), &ly->wq, true);
        mat(&L, N("attn_k.weight"), &ly->wk, true);
        mat(&L, N("attn_v.weight"), &ly->wv, true);
        mat(&L, N("attn_output.weight"), &ly->wo, true);
        mat(&L, N("attn_gate.weight"), &ly->wgate, true);
        mat(&L, N("attn_rel.weight"), &ly->wr, true);
        mat(&L, N("attn_sink_proj.weight"), &ly->sink, true);
        ly->q_clip = f32_tensor(&L, N("attn_q_clip.weight"), true);
        ly->rel_clip = f32_tensor(&L, N("attn_rel_clip.weight"), true);
        ly->rel_bias = f32_tensor(&L, N("attn_rel_bias.weight"), true);
        ly->k_conv = f32_tensor(&L, N("attn_k_conv.weight"), true);
        ly->v_conv = f32_tensor(&L, N("attn_v_conv.weight"), true);
        ly->attn_post_conv = f32_tensor(&L, N("attn_post_conv.weight"), true);
        ly->ffn_post_conv = f32_tensor(&L, N("ffn_post_conv.weight"), true);
        if (ly->swa) {
            ly->attn_norm = f32_tensor(&L, N("attn_norm.weight"), true);
        } else {
            ly->q_inp_norm = f32_tensor(&L, N("attn_q_inp_norm.weight"), true);
            ly->k_inp_norm = f32_tensor(&L, N("attn_k_inp_norm.weight"), true);
            ly->v_inp_norm = f32_tensor(&L, N("attn_v_inp_norm.weight"), true);
            mat(&L, N("attn_k_conv_proj.weight"), &ly->k_conv_proj, true);
            mat(&L, N("attn_v_conv_proj.weight"), &ly->v_conv_proj, true);
        }
        if (i > 0) {
            const OcGgufTensorInfo *ti = oc_gguf_map_tensor_get(&m->map, N("attn_res_query.weight"));
            ly->attn_res_query = f32_tensor(&L, n, true);
            ly->n_attn_res_query = ti && ti->n_dims > 1 ? (uint32_t)ti->dims[1] : 1;
        }
        ly->ffn_res_query = f32_tensor(&L, N("ffn_res_query.weight"), true);
        ly->ffn_norm = f32_tensor(&L, N("ffn_norm.weight"), true);
        mat(&L, N("ffn_gate_inp.weight"), &ly->router, true);
        ly->router_bias = f32_tensor(&L, N("ffn_exp_probs_b.bias"), false);
        mat(&L, N("ffn_gate_up_exps.weight"), &ly->gate_up, true);
        mat(&L, N("ffn_down_exps.weight"), &ly->down, true);
        if (i >= m->core_start && i < m->core_end && m->n_passes > 1) {
            ly->msg_query = f32_tensor(&L, N("loop_msg_query.weight"), true);
            static const char *ctl[OC_INK_N_CTRL] = { "q", "k", "v", "ffn" };
            for (int c = 0; c < OC_INK_N_CTRL; c++) {
                char nn[128];
                snprintf(nn, sizeof nn, "blk.%u.loop_%s_ctrl_down.weight", i, ctl[c]);
                mat(&L, nn, &ly->ctrl_down[c], true);
                snprintf(nn, sizeof nn, "blk.%u.loop_%s_ctrl_gain.weight", i, ctl[c]);
                mat(&L, nn, &ly->ctrl_gain[c], true);
                snprintf(nn, sizeof nn, "blk.%u.loop_%s_ctrl_forget.weight", i, ctl[c]);
                mat(&L, nn, &ly->ctrl_forget[c], true);
            }
        }
#undef N
        if (L.err == OC_OK) {
            const size_t KVr = (size_t)m->n_head_kv * m->head_dim;
            const float *src[4] = { ly->k_conv, ly->v_conv, ly->attn_post_conv, ly->ffn_post_conv };
            const size_t ch[4] = { KVr, KVr, m->n_embd, m->n_embd };
            for (int c = 0; c < 4 && L.err == OC_OK; c++) {
                ly->conv_t[c] = malloc(4 * ch[c] * sizeof(float));
                if (!ly->conv_t[c]) { L.err = OC_ERR_OOM; break; }
                for (size_t k = 0; k < 4; k++)
                    for (size_t x = 0; x < ch[c]; x++) ly->conv_t[c][k * ch[c] + x] = src[c][x * 4 + k];
            }
            ly->rel_bias_t = malloc((size_t)m->n_rel * m->rel_extent * sizeof(float));
            if (!ly->rel_bias_t) L.err = OC_ERR_OOM;
            else
                for (size_t d = 0; d < m->rel_extent; d++)
                    for (size_t r = 0; r < m->n_rel; r++)
                        ly->rel_bias_t[r * m->rel_extent + d] = ly->rel_bias[d * m->n_rel + r];
        }
        if (L.err == OC_OK && (ly->gate_up.rows != 2 * m->n_ff_exp || ly->gate_up.cols != m->n_embd ||
                               ly->down.cols != m->n_ff_exp || ly->gate_up.n_mat != m->n_expert ||
                               ly->gate_up.gtype != GT_NVFP4 || ly->down.gtype != GT_NVFP4)) {
            oc_log(OC_LOG_ERROR, "inkling: layer %u expert tensors have unexpected shape or type", i);
            L.err = OC_ERR_FORMAT;
        }
    }
    if (L.err != OC_OK) { e = L.err; goto fail; }

    const char *tmpl = NULL;
    size_t tlen = 0;
    if (oc_gguf_metadata_get_str(meta(m), "tokenizer.chat_template", &tmpl, &tlen)) {
        m->chat_template = malloc(tlen + 1);
        if (m->chat_template) { memcpy(m->chat_template, tmpl, tlen); m->chat_template[tlen] = 0; }
    }
    if (oc_tokenizer_load_from_gguf(meta(m), &m->tok) == OC_OK) m->have_tok = true;
    else oc_log(OC_LOG_WARN, "inkling: tokenizer failed to load; token ids only");
    return OC_OK;

fail:
    for (uint32_t i = 0; i < OC_INK_MAX_LAYERS; i++) {
        for (int c = 0; c < 4; c++) free(m->layers[i].conv_t[c]);
        free(m->layers[i].rel_bias_t);
    }
    oc_gguf_map_free(&m->map);
    return e != OC_OK ? e : OC_ERR_FORMAT;
}

void oc_inkling_free(OcInklingModel *m)
{
    if (!m) return;
    ink_gpu_model_free(m);
    for (uint32_t i = 0; i < OC_INK_MAX_LAYERS; i++) {
        for (int c = 0; c < 4; c++) free(m->layers[i].conv_t[c]);
        free(m->layers[i].rel_bias_t);
    }
    if (m->have_tok) oc_tokenizer_free(&m->tok);
    free(m->chat_template);
    oc_gguf_map_free(&m->map);
    memset(m, 0, sizeof *m);
}

/* ─── session ───────────────────────────────────────────────────────────── */

#define N_POOL 16

struct OcInklingSession {
    const OcInklingModel *m;
    uint32_t n_ctx, n_past;
    uint32_t last_pos0, last_n, keep;        /* most recent forward call     */
    size_t   n_ch;                           /* conv channels per visit      */
    float   *k[OC_INK_MAX_VISITS], *v[OC_INK_MAX_VISITS];
    uint32_t cap[OC_INK_MAX_VISITS];
    float   *hist[OC_INK_MAX_VISITS];        /* [3 + B][n_ch] conv inputs    */

    /* per-call activations, [B][...] */
    float *cur, *inp_sa, *in3, *q, *kx, *vx, *r, *g, *sink, *kdyn, *vdyn, *o, *a, *xm;
    float *router, *gu, *act, *dn, *mixed, *msg, *tmp, *prelude, *bank_out, *gain, *forget;
    int32_t *it_ex, *it_first, *it_cnt, *it_row0;
    float *pool[N_POOL], *pool_rstd[N_POOL], *gl;
    ActBuf ab, ab_pairs, abq[3];

    uint32_t n_taps;
    int32_t  tap_visits[OC_INK_MAX_TAPS];
    float   *taps;                           /* [max_batch][n_taps][n_embd]  */

    uint32_t max_batch;
    InkGpu  *gpu;                            /* forward on the GPU when set  */
};

static size_t kv_row(const OcInklingModel *m) { return (size_t)m->n_head_kv * m->head_dim; }

OcError oc_inkling_session_new(const OcInklingModel *m, uint32_t n_ctx,
                               const int32_t *tap_visits, uint32_t n_taps,
                               OcInklingSession **out)
{
    if (!m || !out || n_taps > OC_INK_MAX_TAPS || n_ctx == 0) return OC_ERR_INVALID_ARG;
    OcInklingSession *s = calloc(1, sizeof *s);
    if (!s) return OC_ERR_OOM;
    s->m = m;
    s->n_ctx = n_ctx;
    s->max_batch = OC_INK_MAX_BATCH;
    s->n_taps = n_taps;
    for (uint32_t i = 0; i < n_taps; i++) s->tap_visits[i] = tap_visits[i];
    const size_t B = OC_INK_MAX_BATCH, E = m->n_embd, KV = kv_row(m);
    const size_t QD = (size_t)m->n_head * m->head_dim, P = B * m->n_expert_used;
    s->n_ch = 2 * KV + 2 * E;
    OcError e = OC_OK;

    for (uint32_t vi = 0; vi < m->n_visits; vi++) {
        const OcInkLayer *ly = &m->layers[m->visit_layer[vi]];
        uint32_t cap = ly->swa ? m->n_swa + B : n_ctx;
        if (cap > n_ctx) cap = n_ctx;
        s->cap[vi] = cap;
        s->k[vi] = malloc((size_t)cap * KV * sizeof(float));
        s->v[vi] = malloc((size_t)cap * KV * sizeof(float));
        s->hist[vi] = calloc((3 + B) * s->n_ch, sizeof(float));
        if (!s->k[vi] || !s->v[vi] || !s->hist[vi]) e = OC_ERR_OOM;
    }
#define AL(p, n) do { (p) = calloc((n), sizeof(float)); if (!(p)) e = OC_ERR_OOM; } while (0)
    AL(s->cur, B * E); AL(s->inp_sa, B * E); AL(s->in3, 3 * B * E); AL(s->q, B * QD);
    AL(s->kx, B * KV); AL(s->vx, B * KV); AL(s->r, B * m->n_head * m->n_rel); AL(s->g, B * QD);
    AL(s->sink, B * m->n_head); AL(s->kdyn, B * m->n_head_kv * m->n_conv); AL(s->vdyn, B * m->n_head_kv * m->n_conv); AL(s->o, B * QD);
    AL(s->a, B * E); AL(s->xm, B * E); AL(s->router, B * m->n_expert);
    AL(s->gu, P * 2 * m->n_ff_exp); AL(s->act, P * m->n_ff_exp); AL(s->dn, P * E);
    AL(s->mixed, 3 * B * E); AL(s->msg, OC_INK_N_CTRL * B * E); AL(s->tmp, 3 * B * 2 * E);
    AL(s->prelude, B * E); AL(s->bank_out, B * E); AL(s->gain, B * E); AL(s->forget, B * E);
    s->it_ex = calloc(256 * 64, sizeof(int32_t)); s->it_first = calloc(256 * 64, sizeof(int32_t));
    s->it_cnt = calloc(256 * 64, sizeof(int32_t)); s->it_row0 = calloc(256 * 64, sizeof(int32_t));
    if (!s->it_ex || !s->it_first || !s->it_cnt || !s->it_row0) e = OC_ERR_OOM;
    for (int i = 0; i < N_POOL; i++) { AL(s->pool[i], B * E); AL(s->pool_rstd[i], B); }
    AL(s->gl, 3 * B * m->gate_rank);
    AL(s->taps, B * (n_taps ? n_taps : 1) * E);
#undef AL
    if (e == OC_OK) e = actbuf_init(&s->ab, B, 4 * E > QD ? 4 * E : QD);
    if (e == OC_OK) e = actbuf_init(&s->ab_pairs, P, m->n_ff_exp > E ? m->n_ff_exp : E);
    for (int i = 0; i < 3 && e == OC_OK; i++) e = actbuf_init(&s->abq[i], B, E);
    if (e != OC_OK) {
        oc_inkling_session_free(s);
        return e;
    }
    *out = s;
    return OC_OK;
}

void oc_inkling_session_free(OcInklingSession *s)
{
    if (!s) return;
    prof_report();
    ink_gpu_free(s->gpu);
    for (int i = 0; i < OC_INK_MAX_VISITS; i++) { free(s->k[i]); free(s->v[i]); free(s->hist[i]); }
    float *bufs[] = { s->cur, s->inp_sa, s->in3, s->q, s->kx, s->vx, s->r, s->g, s->sink, s->kdyn,
                      s->vdyn, s->o, s->a, s->xm, s->router, s->gu, s->act, s->dn, s->mixed, s->msg,
                      s->tmp, s->prelude, s->bank_out, s->taps, s->gain, s->forget };
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++) free(bufs[i]);
    for (int i = 0; i < N_POOL; i++) { free(s->pool[i]); free(s->pool_rstd[i]); }
    free(s->gl);
    free(s->it_ex); free(s->it_first); free(s->it_cnt); free(s->it_row0);
    actbuf_free(&s->ab);
    actbuf_free(&s->ab_pairs);
    for (int i = 0; i < 3; i++) actbuf_free(&s->abq[i]);
    free(s);
}

void oc_inkling_session_reset(OcInklingSession *s)
{
    s->n_past = 0;
    s->last_n = 0;
    s->keep = 0;
    for (uint32_t vi = 0; vi < s->m->n_visits; vi++) memset(s->hist[vi], 0, (3 + OC_INK_MAX_BATCH) * s->n_ch * sizeof(float));
}

uint32_t oc_inkling_session_pos(const OcInklingSession *s) { return s->n_past; }

uint32_t oc_inkling_session_max_batch(const OcInklingSession *s) { return s->max_batch; }

InkGpu *ink_session_gpu(const OcInklingSession *s) { return s->gpu; }

bool oc_inkling_gpu_compiled(void) { return ink_gpu_compiled(); }

OcError oc_inkling_gpu_load(OcInklingModel *m) { return ink_gpu_model_load(m); }

OcError oc_inkling_session_use_gpu(OcInklingSession *s)
{
    if (s->gpu) return OC_OK;
    if (s->n_past) return OC_ERR_INVALID_ARG;
    OcError e = ink_gpu_new(s->m, s->n_ctx, s->n_taps, &s->gpu);
    if (e != OC_OK) return e;
    float *taps = calloc((size_t)INK_GPU_MAX_BATCH * (s->n_taps ? s->n_taps : 1) * s->m->n_embd, sizeof(float));
    if (!taps) {
        ink_gpu_free(s->gpu);
        s->gpu = NULL;
        return OC_ERR_OOM;
    }
    free(s->taps);
    s->taps = taps;
    s->max_batch = INK_GPU_MAX_BATCH;
    const char *mb = getenv("OC_INK_GPU_BATCH");
    if (mb && atoi(mb) > 0 && atoi(mb) <= INK_GPU_MAX_BATCH) s->max_batch = (uint32_t)atoi(mb);
    return OC_OK;
}

const OcInklingModel *oc_inkling_session_model(const OcInklingSession *s) { return s->m; }

const float *oc_inkling_taps(const OcInklingSession *s, size_t i)
{
    return s->taps + i * s->n_taps * s->m->n_embd;
}

/* Restore the conv history so rows 0..2 hold the inputs before the next
 * token: the last three inputs of the kept prefix of the previous call. */
static void commit_conv(OcInklingSession *s)
{
    if (s->keep == 0) return;
    for (uint32_t vi = 0; vi < s->m->n_visits; vi++)
        memmove(s->hist[vi], s->hist[vi] + (size_t)s->keep * s->n_ch, 3 * s->n_ch * sizeof(float));
    s->keep = 0;
}

OcError oc_inkling_rollback(OcInklingSession *s, size_t n_keep)
{
    if (n_keep > s->last_n || s->keep != s->last_n) return OC_ERR_INVALID_ARG;
    s->keep = (uint32_t)n_keep;
    s->n_past = s->last_pos0 + (uint32_t)n_keep;
    s->last_n = (uint32_t)n_keep;
    return OC_OK;
}

/* ─── layer pieces ──────────────────────────────────────────────────────── */

/* A depth-attention source: n rows of n_embd, with each row's 1/rms. */
typedef struct {
    float *blocks[N_POOL];
    float *rstd[N_POOL];
    size_t n_blocks;
    int    pool_next;
} Blocks;

typedef struct {
    const OcInklingModel *m;
    float *const         *blk;
    float *const         *blk_rstd;
    size_t                n_blk;
    const float          *cur;
    const float          *query;
    size_t                nq, n;
    float                *out;          /* [nq][n][E] */
} MixJob;

/* Depth attention: mix rms-normed [blocks..., cur] with nq learned queries.
 * rms_norm(x) = x * rstd(x), so the blocks are never materialized normed. */
static void mix_tokens(size_t b, size_t e, size_t tid, void *u)
{
    (void)tid;
    const MixJob *j = u;
    const size_t E = j->m->n_embd, S = j->n_blk + 1;
    const float inv = 1.0f / sqrtf((float)E);
    for (size_t t = b; t < e; t++) {
        const float *src[N_POOL + 1];
        float rs[N_POOL + 1];
        for (size_t k = 0; k < j->n_blk; k++) {
            src[k] = j->blk[k] + t * E;
            rs[k] = j->blk_rstd[k][t];
        }
        src[S - 1] = j->cur + t * E;
        rs[S - 1] = ink_rstd(src[S - 1], E, 1e-6f);
        for (size_t qi = 0; qi < j->nq; qi++) {
            float sc[N_POOL + 1], mx = -INFINITY, sum = 0.0f;
            for (size_t k = 0; k < S; k++) {
                sc[k] = ink_dot(j->query + qi * E, src[k], E) * rs[k] * inv;
                if (sc[k] > mx) mx = sc[k];
            }
            for (size_t k = 0; k < S; k++) { sc[k] = expf(sc[k] - mx); sum += sc[k]; }
            float *o = j->out + (qi * j->n + t) * E;
            memset(o, 0, E * sizeof(float));
            for (size_t k = 0; k < S; k++) ink_axpy(o, sc[k] / sum * rs[k], src[k], E);
        }
    }
}

static void depth_mix(const OcInklingModel *m, float *const *blk, float *const *blk_rstd, size_t n_blk,
                      const float *cur, size_t n, const float *query, size_t nq, float *out)
{
    MixJob j = { m, blk, blk_rstd, n_blk, cur, query, nq, n, out };
    oc_parallel_for(n, mix_tokens, &j);
}

typedef struct {
    const OcInkLayer *ly;
    const int        *ctrl;
    float           **cur;
    const float     **msg;
    float            *cat;        /* [n_ctrl][n][2E] */
    const float      *gl;         /* [n_ctrl][n][R]  */
    size_t            n_ctrl, n, E, R;
} CtrlJob;

static void ctrl_norms(size_t b, size_t e, size_t tid, void *u)
{
    (void)tid;
    const CtrlJob *j = u;
    for (size_t it = b; it < e; it++) {
        const size_t k = it / j->n, t = it % j->n;
        float *ck = j->cat + (k * j->n + t) * 2 * j->E;
        ink_rms_norm(j->cur[k] + t * j->E, NULL, j->E, 1e-6f, ck);
        ink_rms_norm(j->msg[k] + t * j->E, NULL, j->E, 1e-6f, ck + j->E);
    }
}

static void ctrl_rows(size_t b, size_t e, size_t tid, void *u)
{
    (void)tid;
    const CtrlJob *j = u;
    const size_t per = (j->E + LIN_CHUNK - 1) / LIN_CHUNK;
    for (size_t it = b; it < e; it++) {
        const size_t k = it / per, c = (size_t)j->ctrl[k];
        size_t r0 = (it % per) * LIN_CHUNK, r1 = r0 + LIN_CHUNK;
        if (r1 > j->E) r1 = j->E;
        const OcInkMat *G = &j->ly->ctrl_gain[c], *F = &j->ly->ctrl_forget[c];
        for (size_t t = 0; t < j->n; t++) {
            const float *g = j->gl + (k * j->n + t) * j->R;
            float gn[LIN_CHUNK], fg[LIN_CHUNK];
            for (size_t r = r0; r < r1; r++) {
                gn[r - r0] = ink_dot_bf16((const uint16_t *)(G->data + r * G->row_bytes), g, j->R);
                fg[r - r0] = ink_dot_bf16((const uint16_t *)(F->data + r * F->row_bytes), g, j->R);
            }
            ink_ctrl_mix(j->cur[k] + t * j->E + r0, gn, fg, j->msg[k] + t * j->E + r0, r1 - r0);
        }
    }
}

/* cur_k <- (1 - forget) * cur_k + gain * msg_k for each listed controller,
 * with tanh gain/forget from a rank-R projection of [rms(cur), rms(msg)]. */
static void loop_control(OcInklingSession *s, const OcInkLayer *ly, const int *ctrl, size_t n_ctrl,
                         float **cur, const float **msg, size_t n)
{
    const OcInklingModel *m = s->m;
    const size_t E = m->n_embd, R = m->gate_rank;
    CtrlJob cj = { ly, ctrl, cur, msg, s->tmp, s->gl, n_ctrl, n, E, R };
    oc_parallel_for(n_ctrl * n, ctrl_norms, &cj);
    LinJob j = { .n = n };
    for (size_t k = 0; k < n_ctrl; k++)
        lin_add(&j, &ly->ctrl_down[ctrl[k]], 0, s->tmp + k * n * 2 * E, 2 * E, NULL, s->gl + k * n * R, R);
    lin_run(&j);
    oc_parallel_for(n_ctrl * ((E + LIN_CHUNK - 1) / LIN_CHUNK), ctrl_rows, &cj);
}

/* ─── short convolutions ────────────────────────────────────────────────── */

typedef struct {
    OcInklingSession *s;
    uint32_t          vi;
    size_t            off, C;
    float            *x;
    const float      *wT;          /* [n_conv][C], oldest tap first */
    const float      *dyn;         /* [n][n_heads][n_conv] or NULL  */
    size_t            n_heads;
} ConvJob;

#ifdef INK_X86
__attribute__((target("avx2,fma")))
static void conv_row_avx2(float *xo, const float *const *h, const float *wT, size_t C, const float *tap,
                          size_t hs, float scale)
{
    const __m256 vs = _mm256_set1_ps(scale);
    for (size_t c = 0; c < C; c += 8) {
        const __m256 h0 = _mm256_loadu_ps(h[0] + c), h1 = _mm256_loadu_ps(h[1] + c);
        const __m256 h2 = _mm256_loadu_ps(h[2] + c), h3 = _mm256_loadu_ps(h[3] + c);
        __m256 acc = _mm256_mul_ps(_mm256_loadu_ps(wT + c), h0);
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(wT + C + c), h1, acc);
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(wT + 2 * C + c), h2, acc);
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(wT + 3 * C + c), h3, acc);
        if (tap) {
            const float *tp = tap + (c / hs) * 4;
            acc = _mm256_fmadd_ps(_mm256_set1_ps(tp[0]), h3, acc);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(tp[1]), h2, acc);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(tp[2]), h1, acc);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(tp[3]), h0, acc);
        }
        _mm256_storeu_ps(xo + c, _mm256_mul_ps(_mm256_add_ps(h3, acc), vs));
    }
}
#endif

static void conv_tokens(size_t b, size_t e, size_t tid, void *u)
{
    (void)tid;
    const ConvJob *j = u;
    const size_t NC = j->s->n_ch, C = j->C;
    const float *H = j->s->hist[j->vi];
    const float scale = 1.0f / sqrtf((float)((j->dyn ? 2 : 1) * 4 + 1));
    const size_t hs = j->dyn ? C / j->n_heads : C;
    for (size_t t = b; t < e; t++) {
        const float *h[4];
        for (int k = 0; k < 4; k++) h[k] = H + (t + (size_t)k) * NC + j->off;
        const float *tap = j->dyn ? j->dyn + t * j->n_heads * 4 : NULL;
        float *xo = j->x + t * C;
#ifdef INK_X86
        if (ink_avx2() && C % 8 == 0 && hs % 8 == 0) {
            conv_row_avx2(xo, h, j->wT, C, tap, hs, scale);
            continue;
        }
#endif
        for (size_t c = 0; c < C; c++) {
            float acc = 0.0f;
            for (int k = 0; k < 4; k++) acc += j->wT[(size_t)k * C + c] * h[k][c];
            if (tap) for (int l = 0; l < 4; l++) acc += tap[(c / hs) * 4 + (size_t)l] * h[3 - l][c];
            xo[c] = (h[3][c] + acc) * scale;
        }
    }
}

/* Causal short convolution with residual over channels [off, off + C) of the
 * visit's history; dyn adds per-head input-dependent taps. */
static void convolve(OcInklingSession *s, uint32_t vi, size_t off, size_t C, float *x, size_t n,
                     const float *wT, const float *dyn, size_t n_heads)
{
    float *H = s->hist[vi];
    const size_t NC = s->n_ch;
    for (size_t t = 0; t < n; t++) memcpy(H + (3 + t) * NC + off, x + t * C, C * sizeof(float));
    ConvJob j = { s, vi, off, C, x, wT, dyn, n_heads };
    oc_parallel_for(n, conv_tokens, &j);
}

/* ─── attention ─────────────────────────────────────────────────────────── */

typedef struct {
    OcInklingSession *s;
    const OcInkLayer *ly;
    uint32_t          vi;
    size_t            n;
    uint32_t          pos0;
    float           **in;
    bool              shared;
} AttnJob;

/* Input norms of the attention inputs, per token. */
static void in_norm_tokens(size_t b, size_t e, size_t tid, void *u)
{
    (void)tid;
    const AttnJob *j = u;
    const OcInklingModel *m = j->s->m;
    const OcInkLayer *ly = j->ly;
    const size_t E = m->n_embd;
    for (size_t t = b; t < e; t++) {
        if (ly->swa) {
            for (int k = 0; k < (j->shared ? 1 : 3); k++)
                ink_rms_norm(j->in[k] + t * E, ly->attn_norm, E, m->eps, j->in[k] + t * E);
        } else {
            ink_rms_norm(j->in[0] + t * E, ly->q_inp_norm, E, m->eps, j->in[0] + t * E);
            ink_rms_norm(j->in[1] + t * E, ly->k_inp_norm, E, m->eps, j->in[1] + t * E);
            ink_rms_norm(j->in[2] + t * E, ly->v_inp_norm, E, m->eps, j->in[2] + t * E);
        }
    }
}

static void l2_clip(float *x, const float *scale, size_t n, float thr, float mul)
{
    if (scale) for (size_t i = 0; i < n; i++) x[i] *= scale[i];
    const float norm = sqrtf(ink_sumsq(x, n)) + 1e-6f;
    float f = thr / norm;
    if (f > 1.0f) f = 1.0f;
    f *= mul;
    if (f != 1.0f) ink_scale(x, NULL, f, n, x);
}

/* Q/R clip and log scaling, K clip, V norm, then the KV cache write. */
static void qk_tokens(size_t b, size_t e, size_t tid, void *u)
{
    (void)tid;
    const AttnJob *j = u;
    OcInklingSession *s = j->s;
    const OcInklingModel *m = s->m;
    const OcInkLayer *ly = j->ly;
    const size_t HD = m->head_dim, KV = kv_row(m), QD = (size_t)m->n_head * HD, NR = m->n_rel;
    for (size_t t = b; t < e; t++) {
        const uint32_t p = j->pos0 + (uint32_t)t;
        float tau = 1.0f;
        if (!ly->swa) {
            const float x = ((float)p + 1.0f) / (float)m->log_floor;
            tau = 1.0f + m->log_alpha * logf(x > 1.0f ? x : 1.0f);
        }
        for (size_t h = 0; h < m->n_head; h++) {
            l2_clip(s->q + t * QD + h * HD, ly->q_clip + h * HD, HD, m->qk_clip, tau);
            l2_clip(s->r + (t * m->n_head + h) * NR, ly->rel_clip + h * NR, NR, m->rel_clip, tau);
        }
        for (size_t g = 0; g < m->n_head_kv; g++) {
            l2_clip(s->kx + t * KV + g * HD, NULL, HD, m->qk_clip, 1.0f);
            float *vv = s->vx + t * KV + g * HD;
            ink_rms_norm(vv, NULL, HD, m->eps, vv);
        }
        const uint32_t slot = ly->swa ? p % s->cap[j->vi] : p;
        memcpy(s->k[j->vi] + (size_t)slot * KV, s->kx + t * KV, KV * sizeof(float));
        memcpy(s->v[j->vi] + (size_t)slot * KV, s->vx + t * KV, KV * sizeof(float));
    }
}

/* One item per (token, KV head): the query heads of the group share each K
 * and V row. Scores carry the learned relative-position bias; the per-head
 * sink logit joins the softmax denominator. */
static void attn_items(size_t b, size_t e, size_t tid, void *u)
{
    const AttnJob *j = u;
    OcInklingSession *s = j->s;
    const OcInklingModel *m = s->m;
    const OcInkLayer *ly = j->ly;
    const size_t HD = m->head_dim, KV = kv_row(m), QD = (size_t)m->n_head * HD, NR = m->n_rel;
    const size_t NKV = m->n_head_kv, grp = m->n_head / NKV, EXT = m->rel_extent;
    const uint32_t cap = s->cap[j->vi];
    float *scr = oc_parallel_scratch(tid, grp * (EXT + (size_t)s->n_ctx + HD) * sizeof(float));
    if (!scr) return;
    float *bias = scr, *sc = scr + grp * EXT, *acc = sc + grp * (size_t)s->n_ctx;
    const float kscale = 1.0f / sqrtf((float)HD);
    const float *K = s->k[j->vi], *V = s->v[j->vi];
    for (size_t it = b; it < e; it++) {
        const size_t t = it / NKV, g = it % NKV;
        const uint32_t p = j->pos0 + (uint32_t)t;
        const uint32_t lo = ly->swa && p + 1 > m->n_swa ? p + 1 - m->n_swa : 0;
        const size_t nk = p - lo + 1, nb = nk < EXT ? nk : EXT;
        for (size_t hi = 0; hi < grp; hi++) {
            const float *rr = s->r + (t * m->n_head + g * grp + hi) * NR;
            float *bh = bias + hi * EXT;
            memset(bh, 0, nb * sizeof(float));
            for (size_t r = 0; r < NR; r++) ink_axpy(bh, rr[r], ly->rel_bias_t + r * EXT, nb);
        }
        for (size_t jk = 0; jk < nk; jk++) {
            const uint32_t pk = lo + (uint32_t)jk, d = p - pk;
            const float *kr = K + (size_t)(ly->swa ? pk % cap : pk) * KV + g * HD;
            for (size_t hi = 0; hi < grp; hi++) {
                float v = ink_dot(s->q + t * QD + (g * grp + hi) * HD, kr, HD) * kscale;
                if (d < EXT) v += bias[hi * EXT + d];
                sc[hi * s->n_ctx + jk] = v;
            }
        }
        float den[16];
        for (size_t hi = 0; hi < grp; hi++) {
            float *sh = sc + hi * s->n_ctx;
            const float sink = s->sink[t * m->n_head + g * grp + hi];
            float mx = sink;
            for (size_t jk = 0; jk < nk; jk++) if (sh[jk] > mx) mx = sh[jk];
            den[hi] = expf(sink - mx) + ink_exp_sub_sum(sh, nk, mx);
            memset(acc + hi * HD, 0, HD * sizeof(float));
        }
        for (size_t jk = 0; jk < nk; jk++) {
            const uint32_t pk = lo + (uint32_t)jk;
            const float *vr = V + (size_t)(ly->swa ? pk % cap : pk) * KV + g * HD;
            for (size_t hi = 0; hi < grp; hi++) ink_axpy(acc + hi * HD, sc[hi * s->n_ctx + jk], vr, HD);
        }
        for (size_t hi = 0; hi < grp; hi++) {
            const size_t h = g * grp + hi;
            ink_gate(acc + hi * HD, s->g + t * QD + h * HD, 1.0f / den[hi], HD, s->o + t * QD + h * HD);
        }
    }
}

/* ─── MoE ───────────────────────────────────────────────────────────────── */

typedef struct {
    OcInklingSession *s;
    const OcInkMat   *W;
    const int32_t    *pair_expert;    /* [n_items] expert of each work item */
    const int32_t    *item_first;     /* first pair index of each expert    */
    const int32_t    *item_count;
    const int32_t    *item_row0;
    const OcNvfp4Act *acts;
    float            *out;
    size_t            out_stride;
    /* gate_up only: rows r and ff + r form one item, and the situ activation
     * of those rows is written to act */
    size_t            ff;
    float            *act;
    float             beta, lbeta;
} MoeJob;

#define MOE_ROWS 64

static void moe_items(size_t b, size_t e, size_t tid, void *u)
{
    (void)tid;
    const MoeJob *j = u;
    for (size_t it = b; it < e; it++) {
        const int32_t ex = j->pair_expert[it], first = j->item_first[it], cnt = j->item_count[it];
        const size_t r0 = (size_t)j->item_row0[it];
        size_t r1 = r0 + MOE_ROWS;
        if (r1 > j->W->rows) r1 = j->W->rows;
        const uint8_t *w = j->W->data + (size_t)ex * j->W->mat_bytes;
        const float gs = j->W->scale ? j->W->scale[ex] : 1.0f;
        float *out = j->out + (size_t)first * j->out_stride;
        if (!j->act) {
            oc_nvfp4_gemm_rows(w, j->W->cols, r0, r1, gs, j->acts + first, (size_t)cnt, out, j->out_stride);
            continue;
        }
        if (r1 > j->ff) r1 = j->ff;
        oc_nvfp4_gemm_rows(w, j->W->cols, r0, r1, gs, j->acts + first, (size_t)cnt, out, j->out_stride);
        oc_nvfp4_gemm_rows(w, j->W->cols, j->ff + r0, j->ff + r1, gs, j->acts + first, (size_t)cnt, out, j->out_stride);
        for (int32_t p = 0; p < cnt; p++) {
            const float *gu = out + (size_t)p * j->out_stride;
            ink_situ(gu + r0, gu + j->ff + r0, j->act + (size_t)(first + p) * j->ff + r0, r1 - r0, j->beta, j->lbeta);
        }
    }
}

typedef struct {
    const float *dn;
    float       *out;
    const int32_t (*pair_of)[16];
    const float (*wsel)[16];
    size_t       K, E;
} CombJob;

static void combine_tokens(size_t b, size_t e, size_t tid, void *u)
{
    (void)tid;
    const CombJob *j = u;
    for (size_t t = b; t < e; t++) {
        float *o = j->out + t * j->E;
        memset(o, 0, j->E * sizeof(float));
        for (size_t k = 0; k < j->K; k++) ink_axpy(o, j->wsel[t][k], j->dn + (size_t)j->pair_of[t][k] * j->E, j->E);
    }
}

static void moe(OcInklingSession *s, const OcInkLayer *ly, const float *x, size_t n, float *out)
{
    const OcInklingModel *m = s->m;
    const size_t E = m->n_embd, NE = m->n_expert, K = m->n_expert_used, FF = m->n_ff_exp;
    quant_acts(&s->ab, x, E, n, E);
    {
        LinJob rj = { .n = n };
        lin_add(&rj, &ly->router, 0, x, E, s->ab.acts, s->router, NE);
        PROF(PF_ROUTER, lin_run(&rj));
    }

    int32_t sel[OC_INK_MAX_BATCH][16];
    float   wsel[OC_INK_MAX_BATCH][16];
    for (size_t t = 0; t < n; t++) {
        float probs[256], score[256];
        for (size_t i = 0; i < NE; i++) {
            probs[i] = ink_sigmoidf(s->router[t * NE + i]);
            score[i] = probs[i] + (ly->router_bias ? ly->router_bias[i] : 0.0f);
        }
        float sum = 0.0f;
        for (size_t k = 0; k < K; k++) {
            size_t best = 0;
            float bv = -INFINITY;
            for (size_t i = 0; i < NE; i++) if (score[i] > bv) { bv = score[i]; best = i; }
            score[best] = -INFINITY;
            sel[t][k] = (int32_t)best;
            wsel[t][k] = probs[best];
            sum += probs[best];
        }
        if (sum < 1e-20f) sum = 1e-20f;
        for (size_t k = 0; k < K; k++) wsel[t][k] = wsel[t][k] / sum * m->route_scale;
    }

    /* pairs grouped by expert: pair p = (token, slot), contiguous per expert */
    int32_t cnt[256] = { 0 }, first[256], pt[OC_INK_MAX_BATCH * 16], pk[OC_INK_MAX_BATCH * 16];
    for (size_t t = 0; t < n; t++) for (size_t k = 0; k < K; k++) cnt[sel[t][k]]++;
    int32_t acc = 0;
    for (size_t i = 0; i < NE; i++) { first[i] = acc; acc += cnt[i]; }
    int32_t fill[256];
    memcpy(fill, first, sizeof fill);
    for (size_t t = 0; t < n; t++)
        for (size_t k = 0; k < K; k++) {
            int32_t p = fill[sel[t][k]]++;
            pt[p] = (int32_t)t;
            pk[p] = (int32_t)k;
        }
    const size_t n_pairs = (size_t)acc;

    /* gate_up: each pair's activation is its token's x (router and experts share it) */
    OcNvfp4Act pair_acts[OC_INK_MAX_BATCH * 16];
    for (size_t p = 0; p < n_pairs; p++) pair_acts[p] = s->ab.acts[pt[p]];

    int32_t *it_ex = s->it_ex, *it_first = s->it_first, *it_cnt = s->it_cnt, *it_row0 = s->it_row0;
    size_t n_items = 0;
    for (size_t i = 0; i < NE; i++) {
        if (!cnt[i]) continue;
        for (size_t r = 0; r < FF; r += MOE_ROWS) {
            it_ex[n_items] = (int32_t)i; it_first[n_items] = first[i]; it_cnt[n_items] = cnt[i];
            it_row0[n_items++] = (int32_t)r;
        }
    }
    MoeJob j1 = { s, &ly->gate_up, it_ex, it_first, it_cnt, it_row0, pair_acts, s->gu, 2 * FF,
                  FF, s->act, m->situ_beta, m->situ_lbeta };
    PROF(PF_GATEUP, oc_parallel_for(n_items, moe_items, &j1));
    quant_acts(&s->ab_pairs, s->act, FF, n_pairs, FF);

    n_items = 0;
    for (size_t i = 0; i < NE; i++) {
        if (!cnt[i]) continue;
        for (size_t r = 0; r < ly->down.rows; r += MOE_ROWS) {
            it_ex[n_items] = (int32_t)i; it_first[n_items] = first[i]; it_cnt[n_items] = cnt[i];
            it_row0[n_items++] = (int32_t)r;
        }
    }
    MoeJob j2 = { s, &ly->down, it_ex, it_first, it_cnt, it_row0, s->ab_pairs.acts, s->dn, E, 0, NULL, 0, 0 };
    PROF(PF_DOWN, oc_parallel_for(n_items, moe_items, &j2));

    /* combine in slot order per token for a deterministic sum */
    int32_t pair_of[OC_INK_MAX_BATCH][16];
    for (size_t p = 0; p < n_pairs; p++) pair_of[pt[p]][pk[p]] = (int32_t)p;
    CombJob cj = { s->dn, out, (const int32_t (*)[16])pair_of, (const float (*)[16])wsel, K, E };
    PROF(PF_COMBINE, oc_parallel_for(n, combine_tokens, &cj));
}

/* ─── one layer visit ───────────────────────────────────────────────────── */

static void push_block(OcInklingSession *s, Blocks *bl, const float *x, size_t n)
{
    const size_t E = s->m->n_embd;
    const int i = bl->pool_next++;
    memcpy(s->pool[i], x, n * E * sizeof(float));
    for (size_t t = 0; t < n; t++) s->pool_rstd[i][t] = ink_rstd(x + t * E, E, 1e-6f);
    bl->blocks[bl->n_blocks] = s->pool[i];
    bl->rstd[bl->n_blocks++] = s->pool_rstd[i];
}

static void visit(OcInklingSession *s, uint32_t vi, uint32_t il, uint32_t ip, size_t n, uint32_t pos0,
                  Blocks *bl, const float *msgs)
{
    const OcInklingModel *m = s->m;
    const OcInkLayer *ly = &m->layers[il];
    const size_t E = m->n_embd, KV = kv_row(m), HD = m->head_dim, QD = (size_t)m->n_head * HD;
    const size_t NH = m->n_head, NKV = m->n_head_kv, NR = m->n_rel, BE = OC_INK_MAX_BATCH * E;
    float *cur = s->cur;
    const bool shared_kv_in = ly->swa && !msgs;

    memcpy(s->inp_sa, cur, n * E * sizeof(float));

    /* attention inputs */
    float *in[3] = { s->in3, s->in3 + BE, s->in3 + 2 * BE };
    const int n_in = shared_kv_in ? 1 : 3;
    if (ly->attn_res_query) {
        PROF(PF_MIX, depth_mix(m, bl->blocks, bl->rstd, bl->n_blocks, cur, n, ly->attn_res_query,
                               ly->n_attn_res_query, s->mixed));
        for (int k = 0; k < n_in; k++)
            memcpy(in[k], s->mixed + (size_t)(ly->swa ? 0 : k) * n * E, n * E * sizeof(float));
    } else {
        for (int k = 0; k < n_in; k++) memcpy(in[k], cur, n * E * sizeof(float));
    }
    if (msgs) {
        static const int ctrl_qkv[3] = { 0, 1, 2 };
        const float *mg[3] = { msgs, msgs + n * E, msgs + 2 * n * E };
        PROF(PF_CTRL, loop_control(s, ly, ctrl_qkv, 3, in, mg, n));
    }

    AttnJob aj = { s, ly, vi, n, pos0, in, shared_kv_in };
    PROF(PF_NORM, oc_parallel_for(n, in_norm_tokens, &aj));

    /* every projection of the attention inputs in one worker region */
    quant_acts(&s->abq[0], in[0], E, n, E);
    const OcNvfp4Act *aq = s->abq[0].acts, *ak = aq, *av = aq;
    if (!shared_kv_in) {
        quant_acts(&s->abq[1], in[1], E, n, E);
        quant_acts(&s->abq[2], in[2], E, n, E);
        ak = s->abq[1].acts;
        av = s->abq[2].acts;
    }
    const float *kin = shared_kv_in ? in[0] : in[1], *vin = shared_kv_in ? in[0] : in[2];
    LinJob pj = { .n = n };
    lin_add(&pj, &ly->wq, 0, in[0], E, aq, s->q, QD);
    lin_add(&pj, &ly->wr, 0, in[0], E, aq, s->r, NH * NR);
    lin_add(&pj, &ly->wgate, 0, in[0], E, aq, s->g, QD);
    lin_add(&pj, &ly->sink, 0, in[0], E, aq, s->sink, NH);
    lin_add(&pj, &ly->wk, 0, kin, E, ak, s->kx, KV);
    lin_add(&pj, &ly->wv, 0, vin, E, av, s->vx, KV);
    const float *kdyn = NULL, *vdyn = NULL;
    if (!ly->swa) {
        const size_t w = NKV * m->n_conv;
        lin_add(&pj, &ly->k_conv_proj, 0, kin, E, ak, s->kdyn, w);
        lin_add(&pj, &ly->v_conv_proj, 0, vin, E, av, s->vdyn, w);
        kdyn = s->kdyn;
        vdyn = s->vdyn;
    }
    PROF(PF_PROJ, lin_run(&pj));
    PROF(PF_CONV, convolve(s, vi, 0, KV, s->kx, n, ly->conv_t[0], kdyn, NKV));
    PROF(PF_CONV, convolve(s, vi, KV, KV, s->vx, n, ly->conv_t[1], vdyn, NKV));

    PROF(PF_NORM, oc_parallel_for(n, qk_tokens, &aj));
    PROF(PF_ATTN, oc_parallel_for(n * NKV, attn_items, &aj));

    PROF(PF_WO, linear(&ly->wo, 0, s->o, QD, n, s->a, E, &s->ab));
    PROF(PF_CONV, convolve(s, vi, 2 * KV, E, s->a, n, ly->conv_t[2], NULL, 0));

    if (ip % m->res_block == 0) {
        push_block(s, bl, s->inp_sa, n);
        memcpy(cur, s->a, n * E * sizeof(float));
    } else {
        for (size_t i = 0; i < n * E; i++) cur[i] = s->inp_sa[i] + s->a[i];
    }

    /* FFN: cur is the ffn residual input */
    PROF(PF_MIX, depth_mix(m, bl->blocks, bl->rstd, bl->n_blocks, cur, n, ly->ffn_res_query, 1, s->xm));
    float *x = s->xm;
    if (msgs) {
        static const int ctrl_ffn = 3;
        const float *mg = msgs + 3 * n * E;
        PROF(PF_CTRL, loop_control(s, ly, &ctrl_ffn, 1, &x, &mg, n));
    }
    for (size_t t = 0; t < n; t++) ink_rms_norm(x + t * E, ly->ffn_norm, E, m->eps, x + t * E);
    moe(s, ly, x, n, s->a);
    PROF(PF_CONV, convolve(s, vi, 2 * KV + E, E, s->a, n, ly->conv_t[3], NULL, 0));
    for (size_t i = 0; i < n * E; i++) cur[i] += s->a[i];
}

/* ─── forward ───────────────────────────────────────────────────────────── */

static void save_tap(OcInklingSession *s, int32_t visit, const float *x, size_t n)
{
    const size_t E = s->m->n_embd;
    for (uint32_t k = 0; k < s->n_taps; k++) {
        if (s->tap_visits[k] != visit) continue;
        for (size_t t = 0; t < n; t++) memcpy(s->taps + (t * s->n_taps + k) * E, x + t * E, E * sizeof(float));
    }
}

OcError oc_inkling_forward(OcInklingSession *s, const uint32_t *tokens, size_t n, float *logits, bool all_logits)
{
    const OcInklingModel *m = s->m;
    if (n == 0 || n > s->max_batch) return OC_ERR_INVALID_ARG;
    if (s->n_past + n > s->n_ctx) return OC_ERR_INVALID_ARG;
    for (size_t t = 0; t < n; t++) if (tokens[t] >= m->n_vocab) return OC_ERR_INVALID_ARG;
    if (s->gpu) {
        const double t0 = prof_on() ? prof_now() : 0.0;
        OcError e = ink_gpu_forward(s->gpu, tokens, n, s->n_past, s->keep, logits, all_logits, s->tap_visits,
                                    s->n_taps, s->taps);
        if (e != OC_OK) return e;
        if (prof_on()) { g_prof[PF_ATTN] += prof_now() - t0; g_prof_calls++; g_prof_tokens += n; }
        s->last_pos0 = s->n_past;
        s->last_n = (uint32_t)n;
        s->keep = (uint32_t)n;
        s->n_past += (uint32_t)n;
        return OC_OK;
    }
    commit_conv(s);
    const size_t E = m->n_embd;
    const uint32_t pos0 = s->n_past;

    for (size_t t = 0; t < n; t++) {
        const uint16_t *row = m->tok_embd + (size_t)tokens[t] * E;
        float *x = s->cur + t * E;
        for (size_t i = 0; i < E; i++) x[i] = ink_bf16f(row[i]);
        ink_rms_norm(x, m->tok_norm, E, m->eps, x);
    }

    Blocks bl = { { 0 }, { 0 }, 0, 0 };
    uint32_t vi = 0;
    for (uint32_t il = 0; il < m->core_start; il++, vi++) {
        save_tap(s, (int32_t)vi, s->cur, n);
        visit(s, vi, il, il, n, pos0, &bl, NULL);
    }
    memcpy(s->prelude, s->cur, n * E * sizeof(float));
    const size_t prelude_n = bl.n_blocks;
    float *bank[N_POOL], *bank_rstd[N_POOL];
    size_t n_bank = 0;
    for (uint32_t p = 0; p < m->n_passes; p++) {
        if (p > 0) {
            n_bank = 0;
            for (size_t k = prelude_n + 1; k < bl.n_blocks; k++) {
                bank[n_bank] = bl.blocks[k];
                bank_rstd[n_bank++] = bl.rstd[k];
            }
            memcpy(s->bank_out, s->cur, n * E * sizeof(float));
            memcpy(s->cur, s->prelude, n * E * sizeof(float));
            bl.n_blocks = prelude_n;
        }
        for (uint32_t il = m->core_start; il < m->core_end; il++, vi++) {
            save_tap(s, (int32_t)vi, s->cur, n);
            if (p == 0) {
                visit(s, vi, il, il, n, pos0, &bl, NULL);
                continue;
            }
            /* messages: depth attention over [bank..., bank_out] with 4 queries */
            PROF(PF_MIX, depth_mix(m, bank, bank_rstd, n_bank, s->bank_out, n, m->layers[il].msg_query,
                                   OC_INK_N_CTRL, s->msg));
            visit(s, vi, il, il, n, pos0, &bl, s->msg);
        }
    }
    for (uint32_t il = m->core_end; il < m->n_layer; il++, vi++) {
        save_tap(s, (int32_t)vi, s->cur, n);
        visit(s, vi, il, il, n, pos0, &bl, NULL);
    }

    depth_mix(m, bl.blocks, bl.rstd, bl.n_blocks, s->cur, n, m->output_res_query, 1, s->xm);
    for (size_t t = 0; t < n; t++) ink_rms_norm(s->xm + t * E, m->output_norm, E, m->eps, s->xm + t * E);
    save_tap(s, (int32_t)m->n_visits, s->xm, n);

    if (logits) {
        const size_t t0 = all_logits ? 0 : n - 1, nl = all_logits ? n : 1;
        if (m->output.gtype == GT_NVFP4) quant_acts(&s->ab, s->xm + t0 * E, E, nl, E);
        LinJob oj = { .n = nl };
        lin_add(&oj, &m->output, 0, s->xm + t0 * E, E, s->ab.acts, logits, m->n_vocab);
        if (m->softcap > 0.0f) {
            oj.seg[0].post = POST_SOFTCAP;
            oj.seg[0].post_arg = m->softcap;
        }
        PROF(PF_LMHEAD, lin_run(&oj));
    }

    if (prof_on()) { g_prof_calls++; g_prof_tokens += n; }
    s->last_pos0 = pos0;
    s->last_n = (uint32_t)n;
    s->keep = (uint32_t)n;
    s->n_past = pos0 + (uint32_t)n;
    return OC_OK;
}

OcError oc_inkling_prefill(OcInklingSession *s, const uint32_t *tokens, size_t n, float *logits)
{
    for (size_t i = 0; i < n; i += s->max_batch) {
        size_t c = n - i < s->max_batch ? n - i : s->max_batch;
        OcError e = oc_inkling_forward(s, tokens + i, c, i + c == n ? logits : NULL, false);
        if (e != OC_OK) return e;
    }
    return OC_OK;
}
