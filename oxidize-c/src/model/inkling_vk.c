/*
 * inkling_vk.c — Inkling-2 forward on a Vulkan GPU (OC_VULKAN builds).
 *
 * Mirrors oc_inkling_forward step for step in one command buffer per call.
 * Weights stay resident on the GPU (NVFP4 matrices in the planar layout of
 * oc_nvfp4_planar), as do the activations, KV caches and conv histories.
 * The token embedding runs on the CPU; logits and drafter taps come back
 * through host-visible memory.
 */
#define _POSIX_C_SOURCE 200809L
#include "inkling_gpu.h"

#ifdef OC_VULKAN

#include "oxidize/log.h"
#include "oxidize/nvfp4.h"
#include "oxidize/vk.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GT_F32   0u
#define GT_BF16  30u
#define GT_NVFP4 40u
#define N_POOL   16
#define ATTN_ROWS 64  /* query rows (token x head) per attn.comp workgroup */
#define ATTN_DTQ 4    /* query tokens per attn_dec.comp tile */
#define ATTN_SHARE 256  /* key positions per attn_dec.comp share */
#define ATTN_DEC 8    /* calls of up to this many tokens use attn_dec.comp */
#define POST_NONE    0u
#define POST_SITU    1u
#define POST_SOFTCAP 2u
#define POST_SITU_Q  3u

static const uint32_t SPV_QUANT[] =
#include "../backends/vk_shaders/quant_act.spv.h"
    ;
static const uint32_t SPV_MM[] =
#include "../backends/vk_shaders/nvfp4_mm.spv.h"
    ;
static const uint32_t SPV_MV[] =
#include "../backends/vk_shaders/nvfp4_mv.spv.h"
    ;
static const uint32_t SPV_NORM[] =
#include "../backends/vk_shaders/rmsnorm.spv.h"
    ;
static const uint32_t SPV_EW[] =
#include "../backends/vk_shaders/ew.spv.h"
    ;
static const uint32_t SPV_MIX[] =
#include "../backends/vk_shaders/depth_mix.spv.h"
    ;
static const uint32_t SPV_CTRL[] =
#include "../backends/vk_shaders/ctrl_mix.spv.h"
    ;
static const uint32_t SPV_LIN[] =
#include "../backends/vk_shaders/linear_f.spv.h"
    ;
static const uint32_t SPV_LINT[] =
#include "../backends/vk_shaders/linear_t.spv.h"
    ;
static const uint32_t SPV_CONV[] =
#include "../backends/vk_shaders/conv.spv.h"
    ;
static const uint32_t SPV_COMMIT[] =
#include "../backends/vk_shaders/conv_commit.spv.h"
    ;
static const uint32_t SPV_QK[] =
#include "../backends/vk_shaders/qk_post.spv.h"
    ;
static const uint32_t SPV_ATTN[] =
#include "../backends/vk_shaders/attn.spv.h"
    ;
static const uint32_t SPV_TOPK[] =
#include "../backends/vk_shaders/topk.spv.h"
    ;
static const uint32_t SPV_GROUP[] =
#include "../backends/vk_shaders/group.spv.h"
    ;
static const uint32_t SPV_NORMQ[] =
#include "../backends/vk_shaders/normq.spv.h"
    ;
static const uint32_t SPV_ROUTE[] =
#include "../backends/vk_shaders/route_small.spv.h"
    ;
static const uint32_t SPV_CDOWN[] =
#include "../backends/vk_shaders/ctrl_down.spv.h"
    ;
static const uint32_t SPV_MIXS[] =
#include "../backends/vk_shaders/depth_mix_split.spv.h"
    ;
static const uint32_t SPV_ATTND[] =
#include "../backends/vk_shaders/attn_dec.spv.h"
    ;
static const uint32_t SPV_ACOMB[] =
#include "../backends/vk_shaders/attn_comb.spv.h"
    ;
static const uint32_t SPV_WALKJ[] =
#include "../backends/vk_shaders/walk_joint.spv.h"
    ;
static const uint32_t SPV_WALKG[] =
#include "../backends/vk_shaders/walk_gate.spv.h"
    ;
static const uint32_t SPV_WALKP[] =
#include "../backends/vk_shaders/walk_pick.spv.h"
    ;
static const uint32_t SPV_DSROPE[] =
#include "../backends/vk_shaders/ds_rope.spv.h"
    ;
static const uint32_t SPV_DSATTN[] =
#include "../backends/vk_shaders/ds_attn.spv.h"
    ;

/* push blocks; each matches its shader's layout */
typedef struct { uint64_t x, q, d, rows; uint32_t n, cols, x_stride, q_stride, d_stride, pad; } QuantPush;
typedef struct {
    uint64_t w, gscale, aq, ad, items, cidx, y, mat_bytes;
    uint32_t rows, cols, n, y_stride, aq_stride, ad_stride, post, ff;
    float    beta, lbeta, cap, pad;
    uint64_t yd;
} MmPush;
typedef struct { uint64_t x, y, w, rs; uint32_t len, xs, ys, pad; float eps, pad2; } NormPush;
typedef struct { uint64_t y, a, b; uint32_t rows, cols, ys, as, bs, mode; } EwPush;
typedef struct { uint64_t args, cur, query, out; uint32_t n_blk, nq, n, E, ostride, orep; } MixPush;
typedef struct { uint64_t x, m, gl, G, F; uint32_t n, E, R, pad; } CtrlPush;
typedef struct { uint64_t w, x, y; uint32_t rows, cols, xs, ys, bf16; float cap; } LinPush;
typedef struct { uint64_t w, x, y; uint32_t rows, cols, n, xs, ys, bf16; float cap, pad; } LintPush;
typedef struct {
    uint64_t hist, x, y, wT, dyn, res, dn, wsel, pair_of;
    uint32_t n, C, NC, off, n_heads, hs, K, hmax;
    float    scale, pad;
} ConvPush;
typedef struct { uint64_t hist; uint32_t n_visits, NC, vstride, keep; } CommitPush;
typedef struct {
    uint64_t q, r, kx, vx, qclip, rclip, Kc, Vc;
    uint32_t n, pos0, NH, NKV, NR, swa, cap, log_floor;
    float    qk_clip, rel_clip, log_alpha, eps;
} QkPush;
typedef struct {
    uint64_t q, r, g, sink, K, V, relT, o, part;
    uint32_t n, pos0, NH, NKV, NR, EXT, cap, swa, n_swa, nsplit, s0;
    float    kscale;
} AttnPush;
typedef struct { uint64_t part, sink, g, o, q, d; uint32_t NH, NKV, nsplit, QHM; } AcombPush;
typedef struct { uint64_t state, w1, xn, frnn, tok, joint; uint32_t R, E, i, anchor; } WalkJointPush;
typedef struct { uint64_t jo, b, state, q, d; uint32_t R, pad; } WalkGatePush;
typedef struct {
    uint64_t base, bias, b2, part, tok, ptop;
    uint32_t V, i, nwg, phase;
    float    cap, pad;
} WalkPickPush;
typedef struct { uint64_t logits, bias, sel, wsel; uint32_t NE, K; float rscale, pad; } TopkPush;
typedef struct {
    uint64_t sel, pair_of, pair_tok, items, items_s, ind;
    uint32_t n, NE, K, gx_gu, gx_dn, gx_gu_s, gx_dn_s, pad;
} GroupPush;
typedef struct { uint64_t x, y, w, q, d; uint32_t len, xs, ys, qs; float eps; uint32_t nmix; float mixw[4]; } NormqPush;
typedef struct {
    uint64_t logits, bias, wsel, pair_of, pair_tok, items, items_s, ind;
    uint32_t n, NE, K, gx_gu, gx_dn, gx_gu_s, gx_dn_s;
    float    rscale;
} RoutePush;
typedef struct { uint64_t w, x, m, y; uint32_t E, R, bf16, pad; float eps, pad2; } CdownPush;
typedef struct { uint64_t args, cur, query, out, part; uint32_t n_blk, nq, n, E, nsplit, phase, ostride, orep; } MixsPush;
typedef struct { uint64_t x, w, y, inv; uint32_t xs, ys, pos0, ocap, orow0, tmin; float eps, pad; } RopePush;
typedef struct {
    uint64_t q, K, V, kb, vb, o;
    uint32_t n_ctx, lo, cap, swa, nb, NH, NKV, pad;
    float    ks, pad2;
} DsAttnPush;

/* ─── weights ───────────────────────────────────────────────────────────── */

typedef struct {
    uint64_t a, scale;
    uint32_t gtype, rows, cols, n_mat;
    uint64_t mat_bytes;
} GMat;

typedef struct {
    GMat wq, wk, wv, wo, wgate, wr, sink, kcp, vcp, router, gate_up, down;
    GMat cdown[OC_INK_N_CTRL], cgain[OC_INK_N_CTRL], cforget[OC_INK_N_CTRL];
    uint64_t attn_norm, q_inp_norm, k_inp_norm, v_inp_norm, q_clip, rel_clip, rel_bias_t, conv_t[4];
    uint64_t attn_res_query, ffn_res_query, ffn_norm, router_bias, msg_query;
} GLayer;

#define MAX_CHUNK 64

typedef struct {
    OcVk     *vk;
    OcVkBuf   chunk[MAX_CHUNK];
    size_t    cap[MAX_CHUNK], used[MAX_CHUNK];
    uint32_t  n_chunk;
    size_t    chunk_cap;
    int       pass;          /* 0: size the chunks, 1: place and copy */
    bool      fail;
    GLayer    L[OC_INK_MAX_LAYERS];
    GMat      output;
    uint64_t  output_norm, output_res_query;
    OcVkPipe *quant, *mm, *mv[9], *mv_situ[9], *norm, *ew, *mix, *ctrl, *lin, *lint, *conv, *commit, *qk, *attn, *topk, *group, *normq, *route, *cdown, *mixs, *dsrope, *dsattn, *attnd, *acomb, *walkj, *walkg, *walkp;
} GModel;

/* First-fit placement with fixed chunk capacity, identical in both passes. */
static uint8_t *place(GModel *g, size_t bytes, uint64_t *addr)
{
    *addr = 0;
    bytes = (bytes + 255) & ~(size_t)255;
    uint32_t c = 0;
    while (c < g->n_chunk && g->used[c] + bytes > g->chunk_cap) c++;
    if (c == g->n_chunk) {
        if (g->pass == 1 || c == MAX_CHUNK || bytes > g->chunk_cap) { g->fail = true; return NULL; }
        g->n_chunk++;
        g->used[c] = 0;
    }
    const size_t off = g->used[c];
    g->used[c] += bytes;
    if (g->pass == 0) return NULL;
    if (off + bytes > g->chunk[c].size) { g->fail = true; return NULL; }
    *addr = g->chunk[c].addr + off;
    return (uint8_t *)g->chunk[c].map + off;
}

static void up_mat(GModel *g, const OcInkMat *W, GMat *o)
{
    memset(o, 0, sizeof *o);
    if (!W->data) return;
    o->gtype = W->gtype;
    o->rows = (uint32_t)W->rows;
    o->cols = (uint32_t)W->cols;
    o->n_mat = (uint32_t)W->n_mat;
    o->mat_bytes = W->mat_bytes;
    uint8_t *dst = place(g, W->mat_bytes * W->n_mat, &o->a);
    if (dst) {
        for (size_t i = 0; i < W->n_mat; i++) {
            if (W->gtype == GT_NVFP4) oc_nvfp4_planar(W->data + i * W->mat_bytes, W->rows, W->cols, dst + i * W->mat_bytes);
            else memcpy(dst + i * W->mat_bytes, W->data + i * W->mat_bytes, W->mat_bytes);
        }
    }
    if (W->scale) {
        uint8_t *s = place(g, W->n_mat * sizeof(float), &o->scale);
        if (s) memcpy(s, W->scale, W->n_mat * sizeof(float));
    }
}

/* BF16 [rows][cols] uploaded transposed to [cols][rows]. */
static void up_bf16_t(GModel *g, const OcInkMat *W, GMat *o)
{
    memset(o, 0, sizeof *o);
    if (!W->data) return;
    o->gtype = W->gtype;
    o->rows = (uint32_t)W->cols;
    o->cols = (uint32_t)W->rows;
    o->n_mat = 1;
    o->mat_bytes = W->rows * W->cols * 2;
    uint16_t *dst = (uint16_t *)place(g, o->mat_bytes, &o->a);
    if (!dst) return;
    const uint16_t *src = (const uint16_t *)W->data;
    for (size_t r = 0; r < W->rows; r++)
        for (size_t c = 0; c < W->cols; c++) dst[c * W->rows + r] = src[r * W->cols + c];
}

static uint64_t up_raw(GModel *g, const void *x, size_t bytes)
{
    if (!x) return 0;
    uint64_t a;
    uint8_t *dst = place(g, bytes, &a);
    if (dst) memcpy(dst, x, bytes);
    return a;
}

static uint64_t up_f32(GModel *g, const float *x, size_t n) { return up_raw(g, x, n * sizeof(float)); }

static void upload(GModel *g, const OcInklingModel *m)
{
    const size_t E = m->n_embd, HD = m->head_dim, NH = m->n_head, NR = m->n_rel, KV = (size_t)m->n_head_kv * HD;
    for (uint32_t il = 0; il < m->n_layer; il++) {
        const OcInkLayer *ly = &m->layers[il];
        GLayer *L = &g->L[il];
        up_mat(g, &ly->gate_up, &L->gate_up);
        up_mat(g, &ly->down, &L->down);
        up_mat(g, &ly->wq, &L->wq);
        up_mat(g, &ly->wk, &L->wk);
        up_mat(g, &ly->wv, &L->wv);
        up_mat(g, &ly->wo, &L->wo);
        up_mat(g, &ly->wgate, &L->wgate);
        up_mat(g, &ly->wr, &L->wr);
        up_mat(g, &ly->sink, &L->sink);
        up_mat(g, &ly->k_conv_proj, &L->kcp);
        up_mat(g, &ly->v_conv_proj, &L->vcp);
        up_mat(g, &ly->router, &L->router);
        for (int c = 0; c < OC_INK_N_CTRL; c++) {
            up_mat(g, &ly->ctrl_down[c], &L->cdown[c]);
            up_bf16_t(g, &ly->ctrl_gain[c], &L->cgain[c]);
            up_bf16_t(g, &ly->ctrl_forget[c], &L->cforget[c]);
        }
        L->attn_norm = up_f32(g, ly->attn_norm, E);
        L->q_inp_norm = up_f32(g, ly->q_inp_norm, E);
        L->k_inp_norm = up_f32(g, ly->k_inp_norm, E);
        L->v_inp_norm = up_f32(g, ly->v_inp_norm, E);
        L->q_clip = up_f32(g, ly->q_clip, NH * HD);
        L->rel_clip = up_f32(g, ly->rel_clip, NH * NR);
        L->rel_bias_t = up_f32(g, ly->rel_bias_t, NR * m->rel_extent);
        for (int c = 0; c < 4; c++) L->conv_t[c] = up_f32(g, ly->conv_t[c], 4 * (c < 2 ? KV : E));
        L->attn_res_query = up_f32(g, ly->attn_res_query, (size_t)ly->n_attn_res_query * E);
        L->ffn_res_query = up_f32(g, ly->ffn_res_query, E);
        L->ffn_norm = up_f32(g, ly->ffn_norm, E);
        L->router_bias = up_f32(g, ly->router_bias, m->n_expert);
        L->msg_query = up_f32(g, ly->msg_query, (size_t)OC_INK_N_CTRL * E);
    }
    up_mat(g, &m->output, &g->output);
    g->output_norm = up_f32(g, m->output_norm, E);
    g->output_res_query = up_f32(g, m->output_res_query, E);
}

/* An integrated GPU allocates from the host's RAM: refuse an allocation
 * that would leave less than 1 GB of MemAvailable, rather than have the
 * kernel kill other processes to make room. */
static bool ram_ok(const OcVk *vk, size_t bytes, const char *what)
{
    if (!oc_vk_info(vk)->integrated) return true;
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return true;
    char line[128];
    unsigned long long kb = 0;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) break;
    fclose(f);
    const double avail = (double)kb * 1024.0, need = (double)bytes + (double)(1u << 30);
    if (kb == 0 || need <= avail) return true;
    oc_log(OC_LOG_ERROR, "inkling-gpu: %s needs %.2f GB but only %.2f GB of RAM is available (1 GB is kept free)", what,
           (double)bytes / 1e9, avail / 1e9);
    return false;
}

/* After a sizing pass of the uploads: allocate the chunks, then rewind for
 * the copy pass. */
static OcError gmodel_alloc(GModel *g, const char *what)
{
    if (g->fail) return OC_ERR_OOM;
    size_t total = 0;
    for (uint32_t c = 0; c < g->n_chunk; c++) total += g->used[c];
    if (!ram_ok(g->vk, total, what)) return OC_ERR_OOM;
    for (uint32_t c = 0; c < g->n_chunk; c++) {
        g->cap[c] = g->used[c];
        OcVkBuf *b = &g->chunk[c];
        if (oc_vk_buf_new(g->vk, g->cap[c], OC_VK_MEM_DEVICE, b) != OC_OK || !b->map) {
            if (b->impl) oc_vk_buf_free(g->vk, b);
            if (oc_vk_buf_new(g->vk, g->cap[c], OC_VK_MEM_STREAM, b) != OC_OK || !b->map) {
                oc_log(OC_LOG_ERROR, "inkling-gpu: cannot allocate weight chunk %u (%zu MB)", c, g->cap[c] >> 20);
                return OC_ERR_OOM;
            }
        }
    }
    memset(g->used, 0, sizeof g->used);
    g->pass = 1;
    return OC_OK;
}

static bool supported(const OcInklingModel *m)
{
    if (m->head_dim != 128 || m->n_head / m->n_head_kv > 4 || m->n_rel > 16 || m->n_expert > 256 ||
        m->n_expert_used > 16 || m->n_conv != 4 || m->n_embd % 64 || m->n_ff_exp % 64) return false;
    for (uint32_t il = 0; il < m->n_layer; il++) {
        const OcInkLayer *ly = &m->layers[il];
        const OcInkMat *mats[] = { &ly->wq, &ly->wk, &ly->wv, &ly->wo, &ly->wgate, &ly->wr };
        for (size_t i = 0; i < sizeof mats / sizeof mats[0]; i++)
            if (mats[i]->gtype != GT_NVFP4 || mats[i]->cols > 4096) return false;
        if (ly->down.cols > 4096) return false;
        if (ly->msg_query)
            for (int c = 0; c < OC_INK_N_CTRL; c++)
                if (ly->ctrl_gain[c].gtype != GT_BF16 || ly->ctrl_forget[c].gtype != GT_BF16 || ly->ctrl_down[c].gtype == GT_NVFP4)
                    return false;
        if (ly->router.gtype == GT_NVFP4 || ly->sink.gtype == GT_NVFP4) return false;
    }
    return m->output.gtype == GT_NVFP4 || m->output.gtype == GT_BF16 || m->output.gtype == GT_F32;
}

static void gmodel_free(GModel *g)
{
    if (!g) return;
    OcVkPipe *pp[] = { g->quant, g->mm, g->norm, g->ew, g->mix, g->ctrl, g->lin, g->lint, g->conv, g->commit, g->qk, g->attn,
                       g->topk, g->group, g->normq, g->route, g->cdown, g->mixs, g->dsrope, g->dsattn, g->attnd, g->acomb, g->walkj, g->walkg, g->walkp };
    for (size_t i = 0; i < sizeof pp / sizeof pp[0]; i++) oc_vk_pipe_free(g->vk, pp[i]);
    for (int i = 1; i <= 8; i++) { oc_vk_pipe_free(g->vk, g->mv[i]); oc_vk_pipe_free(g->vk, g->mv_situ[i]); }
    for (uint32_t i = 0; i < MAX_CHUNK; i++) if (g->chunk[i].impl) oc_vk_buf_free(g->vk, &g->chunk[i]);
    oc_vk_free(g->vk);
    free(g);
}

bool ink_gpu_compiled(void) { return true; }

OcError ink_gpu_model_load(OcInklingModel *m)
{
    if (m->gpu) return OC_OK;
    if (!supported(m)) {
        oc_log(OC_LOG_ERROR, "inkling-gpu: model layout not supported by the GPU path");
        return OC_ERR_MODEL;
    }
    GModel *g = calloc(1, sizeof *g);
    if (!g) return OC_ERR_OOM;
    OcError e = oc_vk_init(&g->vk);
    if (e != OC_OK) { free(g); return e; }
    const OcVkInfo *in = oc_vk_info(g->vk);
    g->chunk_cap = in->max_alloc < ((size_t)2 << 30) ? (size_t)(in->max_alloc & ~(uint64_t)4095) : (size_t)2 << 30;

    upload(g, m);                                  /* pass 0: sizes */
    if ((e = gmodel_alloc(g, "the model's weights")) != OC_OK) goto fail;
    upload(g, m);
    if (g->fail) { e = OC_ERR_OOM; goto fail; }

#define PIPE(dst, spv, push) do { if (e == OC_OK) e = oc_vk_pipe_new(g->vk, spv, sizeof spv, sizeof(push), NULL, 0, &(dst)); } while (0)
    PIPE(g->quant, SPV_QUANT, QuantPush);
    PIPE(g->mm, SPV_MM, MmPush);
    PIPE(g->norm, SPV_NORM, NormPush);
    PIPE(g->ew, SPV_EW, EwPush);
    PIPE(g->mix, SPV_MIX, MixPush);
    PIPE(g->ctrl, SPV_CTRL, CtrlPush);
    PIPE(g->lin, SPV_LIN, LinPush);
    PIPE(g->lint, SPV_LINT, LintPush);
    PIPE(g->conv, SPV_CONV, ConvPush);
    PIPE(g->commit, SPV_COMMIT, CommitPush);
    PIPE(g->qk, SPV_QK, QkPush);
    PIPE(g->attn, SPV_ATTN, AttnPush);
    PIPE(g->topk, SPV_TOPK, TopkPush);
    PIPE(g->group, SPV_GROUP, GroupPush);
    PIPE(g->normq, SPV_NORMQ, NormqPush);
    PIPE(g->route, SPV_ROUTE, RoutePush);
    PIPE(g->cdown, SPV_CDOWN, CdownPush);
    PIPE(g->mixs, SPV_MIXS, MixsPush);
    PIPE(g->dsrope, SPV_DSROPE, RopePush);
    PIPE(g->dsattn, SPV_DSATTN, DsAttnPush);
    PIPE(g->attnd, SPV_ATTND, AttnPush);
    PIPE(g->acomb, SPV_ACOMB, AcombPush);
    PIPE(g->walkj, SPV_WALKJ, WalkJointPush);
    PIPE(g->walkg, SPV_WALKG, WalkGatePush);
    PIPE(g->walkp, SPV_WALKP, WalkPickPush);
#undef PIPE
    for (uint32_t nv = 1; nv <= 8 && e == OC_OK; nv++) {
        const uint32_t s0[2] = { nv, 0 }, s1[2] = { nv, 1 };
        e = oc_vk_pipe_new(g->vk, SPV_MV, sizeof SPV_MV, sizeof(MmPush), s0, 2, &g->mv[nv]);
        if (e == OC_OK) e = oc_vk_pipe_new(g->vk, SPV_MV, sizeof SPV_MV, sizeof(MmPush), s1, 2, &g->mv_situ[nv]);
    }
    if (e != OC_OK) goto fail;
    size_t total = 0;
    for (uint32_t c = 0; c < g->n_chunk; c++) total += g->cap[c];
    oc_log(OC_LOG_INFO, "inkling-gpu: %s, %.2f GB of weights in %u buffers", in->name, (double)total / 1e9, g->n_chunk);
    m->gpu = g;
    return OC_OK;
fail:
    gmodel_free(g);
    return e;
}

void ink_gpu_model_free(OcInklingModel *m)
{
    gmodel_free(m->gpu);
    m->gpu = NULL;
}

/* ─── session ───────────────────────────────────────────────────────────── */

/* GPU profile (OC_INK_PROF=1): timestamps after barrier-separated phases. */
enum { GP_MIX, GP_CTRL, GP_NORM, GP_PROJ, GP_CONV, GP_QK, GP_ATTN, GP_WO, GP_RES, GP_ROUTE, GP_GATEUP, GP_DOWN,
       GP_HEAD, GP_MISC, GP__N };
static const char *const GP_NAMES[GP__N] = { "depth_mix", "loop_ctrl", "norm+quant", "qkv_proj", "conv", "qk_post",
                                             "attention", "attn_out", "residual", "route", "moe_gate_up",
                                             "moe_down", "lm_head", "misc" };
#define MAX_MARKS 4096

/* Conv history rows kept per visit beyond the 3 previous inputs: calls of
 * up to HMAX tokens keep every input and can be rolled back; larger calls
 * keep only their last three inputs. */
#define HMAX 32

/* Calls of up to MIX_SPLIT tokens run depth attention split over channels
 * (depth_mix_split.comp, 256 channels per workgroup). */
#define MIX_SPLIT 16

struct InkGpu {
    int      prof;
    uint32_t last_n;
    bool     last_big;
    uint32_t n_marks;
    uint8_t  mark_cat[MAX_MARKS];
    double   prof_ns[GP__N];
    uint64_t prof_calls, prof_tokens;
    const OcInklingModel *m;
    GModel  *g;
    OcVk    *vk;
    uint32_t n_ctx, B, NC, vstride, n_taps, n_pool;
    OcVkBuf  dev, hist, host, ind, kv[OC_INK_MAX_VISITS];
    OcVkBuf  saved[OC_INK_SAVE_SLOTS];       /* ink_gpu_save: rows 0..2 per visit */
    OcVkBuf  saved_swa[OC_INK_SAVE_SLOTS];   /* [swa visits][K, V][n_swa][KV]     */
    uint64_t kc[OC_INK_MAX_VISITS], vc[OC_INK_MAX_VISITS];
    uint32_t cap[OC_INK_MAX_VISITS];
    uint64_t cur, inp_sa, in3, q, kx, vx, kxc, vxc, ac, r, gt, sink, kdyn, vdyn, o, a, xm, msg, gl;
    uint64_t prelude, bank_out, mixp, mixp2, apart, args_dev;
    uint64_t pool[N_POOL], pool_rs[N_POOL];
    uint64_t router, sel, wsel, pair_of, pair_tok, items, items_s, act, dn;
    uint64_t aq[3], ad[3], aqo, ado, aqp, adp;
    float   *h_x, *h_logits, *h_taps;
    uint64_t a_x, a_logits, a_taps, a_args;
    uint8_t *h_args;
    size_t   args_used, args_cap;
};

typedef struct {
    uint64_t blk[N_POOL], rs[N_POOL];
    uint32_t n_blk;
    int      pool_next;
} GBlocks;

static uint64_t take(uint64_t base, size_t *off, size_t bytes)
{
    const uint64_t a = base + *off;
    *off += (bytes + 255) & ~(size_t)255;
    return a;
}

static void prof_report(InkGpu *s)
{
    if (!s->prof || !s->prof_calls) return;
    double tot = 0.0;
    for (int i = 0; i < GP__N; i++) tot += s->prof_ns[i];
    fprintf(stderr, "[inkling-gpu-prof] %llu calls, %llu tokens, %.3f s of GPU time\n", (unsigned long long)s->prof_calls,
            (unsigned long long)s->prof_tokens, tot * 1e-9);
    for (int i = 0; i < GP__N; i++)
        fprintf(stderr, "[inkling-gpu-prof]   %-12s %8.3f s  %5.1f%%  %8.3f ms/call\n", GP_NAMES[i], s->prof_ns[i] * 1e-9,
                tot > 0 ? 100.0 * s->prof_ns[i] / tot : 0.0, s->prof_ns[i] * 1e-6 / (double)s->prof_calls);
}

void ink_gpu_free(InkGpu *s)
{
    if (!s) return;
    prof_report(s);
    OcVk *vk = s->vk;
    oc_vk_buf_free(vk, &s->dev);
    oc_vk_buf_free(vk, &s->hist);
    for (int i = 0; i < OC_INK_SAVE_SLOTS; i++) { oc_vk_buf_free(vk, &s->saved[i]); oc_vk_buf_free(vk, &s->saved_swa[i]); }
    oc_vk_buf_free(vk, &s->host);
    oc_vk_buf_free(vk, &s->ind);
    for (int i = 0; i < OC_INK_MAX_VISITS; i++) oc_vk_buf_free(vk, &s->kv[i]);
    free(s);
}

OcError ink_gpu_new(const OcInklingModel *m, uint32_t n_ctx, uint32_t n_taps, InkGpu **out)
{
    *out = NULL;
    if (!m->gpu) return OC_ERR_INVALID_ARG;
    InkGpu *s = calloc(1, sizeof *s);
    if (!s) return OC_ERR_OOM;
    s->m = m;
    s->g = m->gpu;
    s->vk = s->g->vk;
    s->prof = getenv("OC_INK_PROF") != NULL && oc_vk_info(s->vk)->timestamp_ns > 0.0;
    s->n_ctx = n_ctx;
    s->B = INK_GPU_MAX_BATCH;
    s->n_taps = n_taps;
    for (uint32_t vi = 0; vi < m->n_visits; vi++) if (m->visit_layer[vi] % m->res_block == 0) s->n_pool++;
    if (s->n_pool > N_POOL) { free(s); return OC_ERR_MODEL; }
    const size_t B = s->B, E = m->n_embd, HD = m->head_dim, KV = (size_t)m->n_head_kv * HD;
    const size_t QD = (size_t)m->n_head * HD, NE = m->n_expert, K = m->n_expert_used, FF = m->n_ff_exp, P = B * K;
    const size_t R = m->gate_rank ? m->gate_rank : 1, F = sizeof(float);
    s->NC = (uint32_t)(2 * KV + 2 * E);
    s->vstride = (uint32_t)((3 + HMAX) * s->NC);
    OcError e = OC_OK;

    for (uint32_t vi = 0; vi < m->n_visits && e == OC_OK; vi++) {
        const OcInkLayer *ly = &m->layers[m->visit_layer[vi]];
        uint32_t cap = ly->swa ? m->n_swa + (uint32_t)B : n_ctx;
        if (cap > n_ctx) cap = n_ctx;
        s->cap[vi] = cap;
        e = oc_vk_buf_new(s->vk, 2 * (size_t)cap * KV * F, OC_VK_MEM_DEVICE, &s->kv[vi]);
        s->kc[vi] = s->kv[vi].addr;
        s->vc[vi] = s->kv[vi].addr + (size_t)cap * KV * F;
    }
    if (e == OC_OK) e = oc_vk_buf_new(s->vk, (size_t)m->n_visits * s->vstride * F, OC_VK_MEM_DEVICE, &s->hist);
    if (e == OC_OK) e = oc_vk_buf_new(s->vk, 64, OC_VK_MEM_DEVICE, &s->ind);

    /* device activations: one buffer, sized by a dry run of the layout */
    for (int pass = 0; pass < 2 && e == OC_OK; pass++) {
        size_t off = 0;
        const uint64_t base = pass ? s->dev.addr : 0;
#define T(dst, n) (dst) = take(base, &off, (n))
        T(s->cur, B * E * F); T(s->inp_sa, B * E * F); T(s->in3, 3 * B * E * F); T(s->q, B * QD * F);
        T(s->kx, B * KV * F); T(s->vx, B * KV * F); T(s->kxc, B * KV * F); T(s->vxc, B * KV * F);
        T(s->ac, B * E * F); T(s->r, B * m->n_head * m->n_rel * F); T(s->gt, B * QD * F);
        T(s->sink, B * m->n_head * F); T(s->kdyn, B * m->n_head_kv * 4 * F); T(s->vdyn, B * m->n_head_kv * 4 * F);
        T(s->o, B * QD * F); T(s->a, B * E * F); T(s->xm, B * E * F);
        T(s->msg, OC_INK_N_CTRL * B * E * F); T(s->gl, 3 * B * R * F);
        T(s->prelude, B * E * F); T(s->bank_out, B * E * F); T(s->mixp, (size_t)MIX_SPLIT * 16 * 72 * F);
        T(s->mixp2, (size_t)MIX_SPLIT * 16 * 72 * F);
        T(s->apart, (size_t)(ATTN_DEC / ATTN_DTQ) * m->n_head_kv * (n_ctx / ATTN_SHARE + 2) * ATTN_DTQ * 4 * 130 * F); T(s->args_dev, 256 * 1024);
        for (uint32_t i = 0; i < s->n_pool; i++) { T(s->pool[i], B * E * F); T(s->pool_rs[i], B * F); }
        T(s->router, B * NE * F); T(s->sel, B * K * 4); T(s->wsel, B * K * F); T(s->pair_of, B * K * 4); T(s->pair_tok, P * 4);
        T(s->items, (NE + P) * 16); T(s->items_s, (NE + P) * 16); T(s->act, P * FF * F); T(s->dn, P * E * F);
        for (int i = 0; i < 3; i++) { T(s->aq[i], B * E); T(s->ad[i], B * E / 64 * F); }
        T(s->aqo, B * (QD > E ? QD : E)); T(s->ado, B * (QD > E ? QD : E) / 64 * F);
        T(s->aqp, P * FF); T(s->adp, P * FF / 64 * F);
#undef T
        if (pass == 0) e = ram_ok(s->vk, off, "the session's activations") ? oc_vk_buf_new(s->vk, off, OC_VK_MEM_DEVICE, &s->dev)
                                                                          : OC_ERR_OOM;
    }

    s->args_cap = 256 * 1024;
    const size_t hx = B * E * F, hl = (size_t)INK_GPU_MAX_LOGITS * m->n_vocab * F;
    const size_t ht = B * (n_taps ? n_taps : 1) * E * F;
    if (e == OC_OK) e = oc_vk_buf_new(s->vk, hx + hl + ht + s->args_cap, OC_VK_MEM_HOST, &s->host);
    if (e == OC_OK && !s->host.map) e = OC_ERR_BACKEND;
    if (e != OC_OK) { ink_gpu_free(s); return e; }
    uint8_t *hm = s->host.map;
    s->h_x = (float *)hm;                 s->a_x = s->host.addr;
    s->h_logits = (float *)(hm + hx);     s->a_logits = s->host.addr + hx;
    s->h_taps = (float *)(hm + hx + hl);  s->a_taps = s->host.addr + hx + hl;
    s->h_args = hm + hx + hl + ht;        s->a_args = s->host.addr + hx + hl + ht;
    if (s->dev.map) {
        /* depth-mix arguments in device memory when the CPU can write it:
         * every mix workgroup reads them before its source rows */
        s->h_args = (uint8_t *)s->dev.map + (s->args_dev - s->dev.addr);
        s->a_args = s->args_dev;
    }
    *out = s;
    return OC_OK;
}

/* ─── recording helpers ─────────────────────────────────────────────────── */

#define BAR() oc_vk_barrier(s->vk)

/* OC_INK_GPU_DEBUG=1: submit, then report max |x| and non-finite count of
 * nf floats of device memory at addr (the activation buffer is host
 * mapped on UMA GPUs). */
static int g_dbg = -1;
static void chk(InkGpu *s, const char *what, uint32_t vi, uint64_t addr, size_t nf)
{
    if (g_dbg < 0) g_dbg = getenv("OC_INK_GPU_DEBUG") != NULL;
    if (!g_dbg || !s->dev.map) return;
    if (oc_vk_submit(s->vk) != OC_OK) { fprintf(stderr, "[gpu-dbg] submit failed at %s %u\n", what, vi); exit(1); }
    const float *x = (const float *)((const uint8_t *)s->dev.map + (addr - s->dev.addr));
    double mx = 0.0;
    size_t bad = 0;
    for (size_t i = 0; i < nf; i++) {
        const float f = x[i];
        if (!isfinite(f)) bad++;
        else if (fabs(f) > mx) mx = fabs(f);
    }
    fprintf(stderr, "[gpu-dbg] visit %2u %-10s max %.4g  nonfinite %zu/%zu\n", vi, what, mx, bad, nf);
    oc_vk_begin(s->vk);
}

static void mark(InkGpu *s, int cat)
{
    if (!s->prof || s->n_marks >= MAX_MARKS) return;
    s->mark_cat[s->n_marks] = (uint8_t)cat;
    oc_vk_stamp(s->vk, s->n_marks++);
}

/* Submit what is recorded so far and wait (accumulating the profile). */
static OcError submit(InkGpu *s)
{
    OcError e = oc_vk_submit(s->vk);
    if (e != OC_OK) return e;
    if (s->prof && s->n_marks > 1)
        for (uint32_t i = 1; i < s->n_marks; i++)
            s->prof_ns[s->mark_cat[i]] += oc_vk_stamp_ns(s->vk, i) - oc_vk_stamp_ns(s->vk, i - 1);
    s->n_marks = 0;
    return OC_OK;
}

/* Large calls are split into one submission per layer visit: the kernel
 * driver kills GPU jobs that run for more than a couple of seconds. */
static OcError split(InkGpu *s, uint32_t n)
{
    if (n <= 32) return OC_OK;
    OcError e = submit(s);
    if (e != OC_OK) return e;
    e = oc_vk_begin(s->vk);
    mark(s, GP_MISC);
    return e;
}
#define F4(n) ((uint64_t)(n) * 4u)

static uint32_t cdiv(size_t a, size_t b) { return (uint32_t)((a + b - 1) / b); }

/* Rows per nvfp4_mv workgroup without SITU: two per lane cluster for one column. */
static uint32_t mv_rows(uint32_t nv) { return nv == 1 ? 64 : 32; }

static void d_norm(InkGpu *s, uint64_t x, uint64_t y, uint64_t w, uint64_t rs, uint32_t rows, uint32_t len,
                   uint32_t xs, uint32_t ys, float eps)
{
    const NormPush p = { x, y, w, rs, len, xs, ys, 0, eps, 0 };
    oc_vk_dispatch(s->vk, s->g->norm, &p, rows, 1, 1);
}

static void d_ew(InkGpu *s, uint64_t y, uint64_t a, uint64_t b, uint32_t rows, uint32_t cols, uint32_t ys,
                 uint32_t as, uint32_t bs, uint32_t mode)
{
    const EwPush p = { y, a, b, rows, cols, ys, as, bs, mode };
    oc_vk_dispatch(s->vk, s->g->ew, &p, cdiv((size_t)rows * cols, 256), 1, 1);
}

static void d_copy(InkGpu *s, uint64_t y, uint64_t a, uint32_t rows, uint32_t cols)
{
    d_ew(s, y, a, 0, rows, cols, cols, cols, 0, 0);
}

static void d_quant(InkGpu *s, uint64_t x, uint32_t n, uint32_t cols, uint32_t xs, uint64_t q, uint64_t d)
{
    const QuantPush p = { x, q, d, 0, n, cols, xs, cols, cols / 64, 0 };
    oc_vk_dispatch(s->vk, s->g->quant, &p, cdiv((size_t)n * cols / 64, 64), 1, 1);
}

/* Row RMS norm of x [n][len] (weight w) into y (may alias x), quantized in
 * the same pass into q / d with the layout of d_quant. */
static void d_normq(InkGpu *s, uint64_t x, uint64_t y, uint64_t w, uint32_t n, uint32_t len, uint64_t q, uint64_t d,
                    float eps)
{
    const NormqPush p = { x, y, w, q, d, len, len, len, len, eps, 0, { 0 } };
    oc_vk_dispatch(s->vk, s->g->normq, &p, n, 1, 1);
}

/* d_normq of the mix sum_k mixw[k] * x[t][k][..] of nmix <= 4 vectors per row. */
static void d_normq_mix(InkGpu *s, uint64_t x, uint64_t y, uint64_t w, uint32_t n, uint32_t len, uint64_t q,
                        uint64_t d, float eps, uint32_t nmix, const float *mixw)
{
    NormqPush p = { x, y, w, q, d, len, nmix * len, len, len, eps, nmix, { 0 } };
    for (uint32_t k = 0; k < nmix; k++) p.mixw[k] = mixw[k];
    oc_vk_dispatch(s->vk, s->g->normq, &p, n, 1, 1);
}

/* y[t][r] = W x[t]: NVFP4 through the quantized activations aq/ad, F32 and
 * BF16 from the f32 input x. */
static void d_linear(InkGpu *s, const GMat *W, uint64_t x, uint32_t xs, uint64_t aq, uint64_t ad, uint32_t n,
                     uint64_t y, uint32_t ys, float cap)
{
    if (W->gtype == GT_NVFP4) {
        const MmPush p = { .w = W->a, .gscale = W->scale, .aq = aq, .ad = ad, .y = y, .mat_bytes = W->mat_bytes,
                           .rows = W->rows, .cols = W->cols, .n = n, .y_stride = ys, .aq_stride = W->cols,
                           .ad_stride = W->cols / 64, .post = cap > 0.0f ? POST_SOFTCAP : POST_NONE, .cap = cap };
        if (n <= 8) oc_vk_dispatch(s->vk, s->g->mv[n], &p, cdiv(W->rows, mv_rows(n)), 1, 1);
        else oc_vk_dispatch(s->vk, s->g->mm, &p, cdiv(W->rows, 64), cdiv(n, 128), 1);
        return;
    }
    if (n > 8) {
        const LintPush p = { W->a, x, y, W->rows, W->cols, n, xs, ys, W->gtype == GT_BF16, cap, 0 };
        oc_vk_dispatch(s->vk, s->g->lint, &p, cdiv(W->rows, 64), cdiv(n, 64), 1);
        return;
    }
    const size_t esz = W->gtype == GT_BF16 ? 2 : 4;
    for (uint32_t r0 = 0; r0 < W->rows; r0 += 65535) {
        const uint32_t rows = W->rows - r0 < 65535 ? W->rows - r0 : 65535;
        const LinPush p = { W->a + (uint64_t)r0 * W->cols * esz, x, y + F4(r0), rows, W->cols, xs, ys,
                            W->gtype == GT_BF16, cap };
        oc_vk_dispatch(s->vk, s->g->lin, &p, rows, n, 1);
    }
}

/* One depth-attention call. The split kernel (few tokens) runs in two
 * phases with a barrier between them, so independent mixes are staged with
 * mix_prep and run phase by phase to share that barrier; the one-workgroup
 * kernel runs whole in phase 0. Output qi of token t goes to
 * out + (qi + r) * ostride + t * E for r < orep. */
typedef struct {
    bool     ok, split;
    MixsPush ps;
    MixPush  pm;
} MixCall;

static MixCall mix_prep(InkGpu *s, const uint64_t *blk, const uint64_t *rs, uint32_t n_blk, uint64_t cur, uint32_t n,
                        uint64_t query, uint32_t nq, uint64_t out, uint32_t ostride, uint32_t orep, uint64_t part)
{
    MixCall c = { 0 };
    if (s->args_used + 256 > s->args_cap) return c;
    uint64_t *args = (uint64_t *)(s->h_args + s->args_used);
    const uint64_t aa = s->a_args + s->args_used;
    s->args_used += 256;
    for (uint32_t k = 0; k < n_blk; k++) { args[k] = blk[k]; args[16 + k] = rs[k]; }
    const uint32_t E = s->m->n_embd, ns = cdiv(E, 256);
    c.ok = true;
    c.split = n <= MIX_SPLIT && E % ns == 0 && ns <= 16;
    c.ps = (MixsPush){ aa, cur, query, out, part, n_blk, nq, n, E, ns, 0, ostride, orep };
    c.pm = (MixPush){ aa, cur, query, out, n_blk, nq, n, E, ostride, orep };
    return c;
}

static void mix_run(InkGpu *s, MixCall *c, uint32_t phase)
{
    if (!c->ok) return;
    if (c->split) {
        c->ps.phase = phase;
        oc_vk_dispatch(s->vk, s->g->mixs, &c->ps, c->ps.n, c->ps.nsplit, 1);
    } else if (phase == 0) {
        oc_vk_dispatch(s->vk, s->g->mix, &c->pm, c->pm.n, 1, 1);
    }
}

static void d_mix(InkGpu *s, const uint64_t *blk, const uint64_t *rs, uint32_t n_blk, uint64_t cur, uint32_t n,
                  uint64_t query, uint32_t nq, uint64_t out)
{
    MixCall c = mix_prep(s, blk, rs, n_blk, cur, n, query, nq, out, n * s->m->n_embd, 1, s->mixp);
    mix_run(s, &c, 0);
    if (c.split) {
        BAR();
        mix_run(s, &c, 1);
    }
}

/* cur[k] <- (1 - tanh forget) * cur[k] + tanh gain * msg[k] for the
 * controllers ctrl[0..nc): independent, so each stage runs them together. */
static void d_ctrl(InkGpu *s, const GLayer *L, const int *ctrl, int nc, const uint64_t *cur, const uint64_t *msg,
                   uint32_t n)
{
    const uint32_t E = s->m->n_embd, R = s->m->gate_rank;
    const uint64_t gs = F4((uint64_t)s->B * R);
    for (int k = 0; k < nc; k++) {
        const GMat *W = &L->cdown[ctrl[k]];
        const CdownPush p = { W->a, cur[k], msg[k], s->gl + k * gs, E, R, W->gtype == GT_BF16, 0, 1e-6f, 0 };
        oc_vk_dispatch(s->vk, s->g->cdown, &p, R, n, 1);
    }
    BAR();
    for (int k = 0; k < nc; k++) {
        const CtrlPush p = { cur[k], msg[k], s->gl + k * gs, L->cgain[ctrl[k]].a, L->cforget[ctrl[k]].a, n, E, R, 0 };
        oc_vk_dispatch(s->vk, s->g->ctrl, &p, cdiv(E / 2, 64), cdiv(n, 8), 1);
    }
    BAR();
}

static uint64_t hist_at(const InkGpu *s, uint32_t vi, uint32_t row, uint32_t off)
{
    return s->hist.addr + F4((uint64_t)vi * s->vstride + (uint64_t)row * s->NC + off);
}

/* Short convolution of this call's inputs into y, and the history update:
 * every input for calls of up to HMAX tokens (rows 3 + t), else the last
 * three (rows 3..5). The inputs are x, or with comb the MoE combine of the
 * last d_moe's expert outputs. */
static void d_conv(InkGpu *s, uint32_t vi, uint32_t off, uint32_t C, uint64_t x, uint64_t y, uint64_t res,
                   uint32_t n, uint64_t wT, uint64_t dyn, uint32_t n_heads, bool comb)
{
    const ConvPush p = { hist_at(s, vi, 0, 0), x, y, wT, dyn, res, comb ? s->dn : 0, s->wsel, s->pair_of, n, C, s->NC,
                         off, n_heads, dyn ? C / n_heads : C, s->m->n_expert_used, HMAX,
                         1.0f / sqrtf((float)((dyn ? 2 : 1) * 4 + 1)), 0 };
    oc_vk_dispatch(s->vk, s->g->conv, &p, cdiv((size_t)n * C, 256), 1, 1);
}

/* Routing, gate_up and down of the MoE; the combine runs in the following
 * d_conv (comb). */
static void d_moe(InkGpu *s, const GLayer *L, uint64_t x, uint32_t n)
{
    const OcInklingModel *m = s->m;
    const uint32_t E = m->n_embd, NE = m->n_expert, K = m->n_expert_used, FF = m->n_ff_exp;
    d_linear(s, &L->router, x, E, 0, 0, n, s->router, NE, 0.0f);
    BAR();
    /* experts with few pairs run on the matvec kernel (small item list),
     * the rest on the tiled matmul (big list); see moe_group.glsl */
    const uint32_t nv = n < 8 ? n : 8;
    if (n <= 16) {
        const RoutePush rp = { s->router, L->router_bias, s->wsel, s->pair_of, s->pair_tok, s->items, s->items_s,
                               s->ind.addr, n, NE, K, cdiv(FF, 32), cdiv(E, 64), cdiv(FF, 64), cdiv(E, mv_rows(nv)),
                               m->route_scale };
        oc_vk_dispatch(s->vk, s->g->route, &rp, 1, 1, 1);
    } else {
        const TopkPush tp = { s->router, L->router_bias, s->sel, s->wsel, NE, K, m->route_scale, 0 };
        oc_vk_dispatch(s->vk, s->g->topk, &tp, n, 1, 1);
        BAR();
        const GroupPush gp = { s->sel, s->pair_of, s->pair_tok, s->items, s->items_s, s->ind.addr, n, NE, K,
                               cdiv(FF, 32), cdiv(E, 64), cdiv(FF, 64), cdiv(E, mv_rows(nv)), 0 };
        oc_vk_dispatch(s->vk, s->g->group, &gp, 1, 1, 1);
    }
    BAR();
    mark(s, GP_ROUTE);
    MmPush gu = { .w = L->gate_up.a, .gscale = L->gate_up.scale, .aq = s->aq[0], .ad = s->ad[0],
                  .items = s->items, .cidx = s->pair_tok, .y = s->act, .mat_bytes = L->gate_up.mat_bytes,
                  .rows = 2 * FF, .cols = E, .y_stride = FF, .aq_stride = E, .ad_stride = E / 64,
                  .post = POST_SITU, .ff = FF, .beta = m->situ_beta, .lbeta = m->situ_lbeta };
    if (n > 8) oc_vk_dispatch_indirect(s->vk, s->g->mm, &gu, &s->ind, 0);
    gu.items = s->items_s;
    if (n <= 8) {
        /* every item on the matvec, which quantizes its own output */
        gu.post = POST_SITU_Q;
        gu.y = s->aqp;
        gu.yd = s->adp;
    }
    oc_vk_dispatch_indirect(s->vk, s->g->mv_situ[nv], &gu, &s->ind, 24);
    BAR();
    mark(s, GP_GATEUP);
    if (n > 8) {
        d_quant(s, s->act, n * K, FF, FF, s->aqp, s->adp);
        BAR();
    }
    MmPush dn = { .w = L->down.a, .gscale = L->down.scale, .aq = s->aqp, .ad = s->adp, .items = s->items,
                  .y = s->dn, .mat_bytes = L->down.mat_bytes, .rows = E, .cols = FF, .y_stride = E,
                  .aq_stride = FF, .ad_stride = FF / 64 };
    if (n > 8) oc_vk_dispatch_indirect(s->vk, s->g->mm, &dn, &s->ind, 12);
    dn.items = s->items_s;
    oc_vk_dispatch_indirect(s->vk, s->g->mv[nv], &dn, &s->ind, 36);
    BAR();
    mark(s, GP_DOWN);
}

static void push_block(InkGpu *s, GBlocks *bl, uint64_t x, uint32_t n)
{
    const uint32_t E = s->m->n_embd;
    const int i = bl->pool_next++;
    d_copy(s, s->pool[i], x, n, E);
    d_norm(s, x, 0, 0, s->pool_rs[i], n, E, E, E, 1e-6f);
    bl->blk[bl->n_blk] = s->pool[i];
    bl->rs[bl->n_blk++] = s->pool_rs[i];
}

/* One layer visit. On loop passes after the first, msgs holds the loop
 * messages that msgmix (staged by the caller) computes alongside the
 * visit's first depth mix. */
static void visit(InkGpu *s, uint32_t vi, uint32_t il, uint32_t n, uint32_t pos0, GBlocks *bl, uint64_t msgs,
                  MixCall *msgmix)
{
    const OcInklingModel *m = s->m;
    const OcInkLayer *ly = &m->layers[il];
    const GLayer *L = &s->g->L[il];
    const uint32_t E = m->n_embd, HD = m->head_dim, NH = m->n_head, NKV = m->n_head_kv;
    const uint32_t KV = NKV * HD, QD = NH * HD, NR = m->n_rel;
    const bool shared_kv_in = ly->swa && !msgs;
    const int n_in = shared_kv_in ? 1 : 3;
    const uint64_t BE = F4((uint64_t)s->B * E), nE = F4((uint64_t)n * E);
    const uint64_t in[3] = { s->in3, s->in3 + BE, s->in3 + 2 * BE };

    d_copy(s, s->inp_sa, s->cur, n, E);
    /* the attention inputs straight from the depth mix: one per query, or
     * the one query's mix in each of a sliding-window layer's inputs */
    MixCall am = { 0 };
    if (ly->attn_res_query)
        am = mix_prep(s, bl->blk, bl->rs, bl->n_blk, s->cur, n, L->attn_res_query,
                      ly->swa ? 1 : ly->n_attn_res_query, in[0], s->B * E, ly->swa ? (uint32_t)n_in : 1, s->mixp);
    else
        for (int k = 0; k < n_in; k++) d_copy(s, in[k], s->cur, n, E);
    if (msgmix) mix_run(s, msgmix, 0);
    mix_run(s, &am, 0);
    if ((msgmix && msgmix->split) || am.split) {
        BAR();
        if (msgmix) mix_run(s, msgmix, 1);
        mix_run(s, &am, 1);
    }
    BAR();
    mark(s, GP_MIX);
    chk(s, "attn_in", vi, in[0], (size_t)n * E);
    if (msgs) {
        static const int qkv[3] = { 0, 1, 2 };
        const uint64_t mg[3] = { msgs, msgs + nE, msgs + 2 * nE };
        d_ctrl(s, L, qkv, 3, in, mg, n);
        mark(s, GP_CTRL);
        chk(s, "ctrl", vi, in[0], (size_t)n * E);
    }

    const uint64_t in_w[3] = { ly->swa ? L->attn_norm : L->q_inp_norm, ly->swa ? L->attn_norm : L->k_inp_norm,
                               ly->swa ? L->attn_norm : L->v_inp_norm };
    for (int k = 0; k < n_in; k++) d_normq(s, in[k], in[k], in_w[k], n, E, s->aq[k], s->ad[k], m->eps);
    BAR();
    mark(s, GP_NORM);
    const int ki = shared_kv_in ? 0 : 1, vi_ = shared_kv_in ? 0 : 2;
    d_linear(s, &L->wq, in[0], E, s->aq[0], s->ad[0], n, s->q, QD, 0.0f);
    d_linear(s, &L->wr, in[0], E, s->aq[0], s->ad[0], n, s->r, NH * NR, 0.0f);
    d_linear(s, &L->wgate, in[0], E, s->aq[0], s->ad[0], n, s->gt, QD, 0.0f);
    d_linear(s, &L->sink, in[0], E, s->aq[0], s->ad[0], n, s->sink, NH, 0.0f);
    d_linear(s, &L->wk, in[ki], E, s->aq[ki], s->ad[ki], n, s->kx, KV, 0.0f);
    d_linear(s, &L->wv, in[vi_], E, s->aq[vi_], s->ad[vi_], n, s->vx, KV, 0.0f);
    if (!ly->swa) {
        d_linear(s, &L->kcp, in[ki], E, s->aq[ki], s->ad[ki], n, s->kdyn, NKV * m->n_conv, 0.0f);
        d_linear(s, &L->vcp, in[vi_], E, s->aq[vi_], s->ad[vi_], n, s->vdyn, NKV * m->n_conv, 0.0f);
    }
    BAR();
    mark(s, GP_PROJ);
    chk(s, "q", vi, s->q, (size_t)n * QD);
    chk(s, "k", vi, s->kx, (size_t)n * KV);
    d_conv(s, vi, 0, KV, s->kx, s->kxc, 0, n, L->conv_t[0], ly->swa ? 0 : s->kdyn, NKV, false);
    d_conv(s, vi, KV, KV, s->vx, s->vxc, 0, n, L->conv_t[1], ly->swa ? 0 : s->vdyn, NKV, false);
    BAR();
    mark(s, GP_CONV);
    const QkPush qp = { s->q, s->r, s->kxc, s->vxc, L->q_clip, L->rel_clip, s->kc[vi], s->vc[vi], n, pos0, NH, NKV,
                        NR, ly->swa, s->cap[vi], m->log_floor, m->qk_clip, m->rel_clip, m->log_alpha, m->eps };
    oc_vk_dispatch(s->vk, s->g->qk, &qp, n, NH + NKV, 1);
    BAR();
    mark(s, GP_QK);
    AttnPush ap = { s->q, s->r, s->gt, s->sink, s->kc[vi], s->vc[vi], L->rel_bias_t, s->o, s->apart, n, pos0, NH, NKV,
                    NR, m->rel_extent, s->cap[vi], ly->swa, m->n_swa, 0, 0, 1.0f / sqrtf((float)HD) };
    if (n <= ATTN_DEC) {
        /* keys in fixed 256-position shares, merged in order with the sink:
         * a row's result does not depend on the other rows of the call, so a
         * verify call reproduces plain decoding exactly */
        const uint32_t klo = ly->swa && pos0 + 1 > m->n_swa ? pos0 + 1 - m->n_swa : 0;
        ap.s0 = klo / ATTN_SHARE;
        ap.nsplit = (pos0 + n - 1) / ATTN_SHARE - ap.s0 + 1;
        oc_vk_dispatch(s->vk, s->g->attnd, &ap, cdiv(n, ATTN_DTQ), NKV, ap.nsplit);
        BAR();
        /* the merge also quantizes the output for wo */
        const AcombPush cp = { s->apart, s->sink, s->gt, s->o, s->aqo, s->ado, NH, NKV, ap.nsplit, ATTN_DTQ * 4 };
        oc_vk_dispatch(s->vk, s->g->acomb, &cp, n * (NH / NKV), NKV, 1);
        BAR();
    } else {
        oc_vk_dispatch(s->vk, s->g->attn, &ap, cdiv(n, ATTN_ROWS / (NH / NKV)), NKV, 1);
        BAR();
        d_quant(s, s->o, n, QD, QD, s->aqo, s->ado);
        BAR();
    }
    mark(s, GP_ATTN);
    chk(s, "attn", vi, s->o, (size_t)n * QD);
    d_linear(s, &L->wo, s->o, QD, s->aqo, s->ado, n, s->a, E, 0.0f);
    BAR();
    mark(s, GP_WO);
    chk(s, "wo", vi, s->a, (size_t)n * E);
    /* attention post-conv straight into cur, with the residual unless this
     * visit starts a depth-attention block */
    if (il % m->res_block == 0) {
        push_block(s, bl, s->inp_sa, n);
        d_conv(s, vi, 2 * KV, E, s->a, s->cur, 0, n, L->conv_t[2], 0, 0, false);
    } else {
        d_conv(s, vi, 2 * KV, E, s->a, s->cur, s->inp_sa, n, L->conv_t[2], 0, 0, false);
    }
    BAR();
    mark(s, GP_RES);

    d_mix(s, bl->blk, bl->rs, bl->n_blk, s->cur, n, L->ffn_res_query, 1, s->xm);
    BAR();
    mark(s, GP_MIX);
    if (msgs) {
        static const int ffn = 3;
        const uint64_t mg = msgs + 3 * nE;
        d_ctrl(s, L, &ffn, 1, &s->xm, &mg, n);
        mark(s, GP_CTRL);
    }
    d_normq(s, s->xm, s->xm, L->ffn_norm, n, E, s->aq[0], s->ad[0], m->eps);
    BAR();
    mark(s, GP_NORM);
    chk(s, "ffn_in", vi, s->xm, (size_t)n * E);
    d_moe(s, L, s->xm, n);
    d_conv(s, vi, 2 * KV + E, E, 0, s->cur, s->cur, n, L->conv_t[3], 0, 0, true);
    BAR();
    mark(s, GP_RES);
}

/* The sliding-window K/V rows before position pos, which later positions
 * overwrite in the ring: copied into buf ([swa visits][K, V][n_swa][KV]) on
 * save, back into the cache on restore. */
static void d_swa_window(InkGpu *s, uint64_t buf, uint32_t pos, bool save)
{
    const OcInklingModel *m = s->m;
    const uint32_t KV = m->n_head_kv * m->head_dim, W = m->n_swa, n = pos < W ? pos : W;
    for (uint32_t vi = 0; vi < m->n_visits; vi++) {
        if (!m->layers[m->visit_layer[vi]].swa) continue;
        const uint32_t cap = s->cap[vi], r0 = (pos - n) % cap, n1 = n < cap - r0 ? n : cap - r0;
        for (int h = 0; h < 2; h++, buf += F4((uint64_t)W * KV))
            for (int k = 0; k < 2; k++) {
                const uint32_t rows = k ? n - n1 : n1;
                const uint64_t c = (h ? s->vc[vi] : s->kc[vi]) + (k ? 0 : F4((uint64_t)r0 * KV));
                const uint64_t b = buf + (k ? F4((uint64_t)n1 * KV) : 0);
                if (rows) d_copy(s, save ? b : c, save ? c : b, rows, KV);
            }
    }
}

OcError ink_gpu_save(InkGpu *s, uint32_t keep, uint32_t slot, uint32_t pos)
{
    const OcInklingModel *m = s->m;
    const uint32_t nv = m->n_visits, row3 = 3 * s->NC;
    if (keep && s->last_big) {
        if (keep != s->last_n) return OC_ERR_INVALID_ARG;
        keep = 3;
    }
    uint64_t n_swa_visits = 0;
    for (uint32_t vi = 0; vi < nv; vi++) n_swa_visits += m->layers[m->visit_layer[vi]].swa;
    const size_t swa_bytes = F4(n_swa_visits * 2 * m->n_swa * m->n_head_kv * m->head_dim);
    OcError e = OC_OK;
    OcVkBuf *sv = &s->saved[slot], *sw = &s->saved_swa[slot];
    if (!sv->impl) e = oc_vk_buf_new(s->vk, F4((uint64_t)nv * row3), OC_VK_MEM_DEVICE, sv);
    if (e == OC_OK && swa_bytes && !sw->impl)
        e = ram_ok(s->vk, swa_bytes, "a prompt-cache slot") ? oc_vk_buf_new(s->vk, swa_bytes, OC_VK_MEM_DEVICE, sw) : OC_ERR_OOM;
    if (e == OC_OK) e = oc_vk_begin(s->vk);
    if (e != OC_OK) return e;
    if (keep) {
        const CommitPush cp = { s->hist.addr, nv, s->NC, s->vstride, keep };
        oc_vk_dispatch(s->vk, s->g->commit, &cp, cdiv((size_t)nv * s->NC, 256), 1, 1);
        BAR();
    }
    d_ew(s, sv->addr, s->hist.addr, 0, nv, row3, row3, s->vstride, 0, 0);
    if (swa_bytes) d_swa_window(s, sw->addr, pos, true);
    return oc_vk_submit(s->vk);
}

OcError ink_gpu_restore(InkGpu *s, uint32_t slot, uint32_t pos)
{
    const OcVkBuf *sv = &s->saved[slot], *sw = &s->saved_swa[slot];
    if (!sv->impl) return OC_ERR_INVALID_ARG;
    const uint32_t nv = s->m->n_visits, row3 = 3 * s->NC;
    const OcError e = oc_vk_begin(s->vk);
    if (e != OC_OK) return e;
    d_ew(s, s->hist.addr, sv->addr, 0, nv, row3, s->vstride, row3, 0, 0);
    if (sw->impl) d_swa_window(s, sw->addr, pos, false);
    return oc_vk_submit(s->vk);
}

static void d_tap(InkGpu *s, int32_t visit_id, uint64_t x, uint32_t n, const int32_t *tv, uint32_t nt)
{
    const uint32_t E = s->m->n_embd;
    for (uint32_t k = 0; k < nt; k++)
        if (tv[k] == visit_id) d_ew(s, s->a_taps + F4((uint64_t)k * E), x, 0, n, E, nt * E, E, 0, 0);
}

OcError ink_gpu_forward(InkGpu *s, const uint32_t *tokens, size_t n_, uint32_t pos0, uint32_t keep,
                        float *logits, bool all_logits, const int32_t *tv, uint32_t nt, float *taps)
{
    const OcInklingModel *m = s->m;
    const uint32_t n = (uint32_t)n_, E = m->n_embd;
    if (n == 0 || n > s->B || nt > s->n_taps || (all_logits && n > INK_GPU_MAX_LOGITS)) return OC_ERR_INVALID_ARG;
    if (pos0 + n > s->n_ctx) return OC_ERR_INVALID_ARG;
    if (keep && s->last_big) {
        if (keep != s->last_n) return OC_ERR_INVALID_ARG;   /* a large call cannot be rolled back */
        keep = 3;
    }

    for (uint32_t t = 0; t < n; t++) {
        const uint16_t *row = m->tok_embd + (size_t)tokens[t] * E;
        float *x = s->h_x + (size_t)t * E;
        double ss = 0.0;
        for (uint32_t i = 0; i < E; i++) {
            const uint32_t b = (uint32_t)row[i] << 16;
            float f;
            memcpy(&f, &b, 4);
            x[i] = f;
            ss += (double)f * f;
        }
        const float r = 1.0f / sqrtf((float)ss / (float)E + m->eps);
        for (uint32_t i = 0; i < E; i++) x[i] = x[i] * r * m->tok_norm[i];
    }

    OcError e = oc_vk_begin(s->vk);
    if (e != OC_OK) return e;
    s->args_used = 0;
    s->n_marks = 0;
    mark(s, GP_MISC);
    if (keep) {
        const CommitPush cp = { s->hist.addr, m->n_visits, s->NC, s->vstride, keep };
        oc_vk_dispatch(s->vk, s->g->commit, &cp, cdiv((size_t)m->n_visits * s->NC, 256), 1, 1);
    } else if (pos0 == 0) {
        for (uint32_t vi = 0; vi < m->n_visits; vi++)
            oc_vk_fill(s->vk, &s->hist, F4((uint64_t)vi * s->vstride), F4(3 * s->NC), 0);
    }
    d_copy(s, s->cur, s->a_x, n, E);
    BAR();

    GBlocks bl = { { 0 }, { 0 }, 0, 0 };
    uint32_t vi = 0;
    for (uint32_t il = 0; il < m->core_start; il++, vi++) {
        d_tap(s, (int32_t)vi, s->cur, n, tv, nt);
        visit(s, vi, il, n, pos0, &bl, 0, NULL);
        if ((e = split(s, n)) != OC_OK) return e;
    }
    d_copy(s, s->prelude, s->cur, n, E);
    BAR();
    const uint32_t prelude_n = bl.n_blk;
    uint64_t bank[N_POOL], bank_rs[N_POOL];
    uint32_t n_bank = 0;
    for (uint32_t pass = 0; pass < m->n_passes; pass++) {
        if (pass > 0) {
            n_bank = 0;
            for (uint32_t k = prelude_n + 1; k < bl.n_blk; k++) { bank[n_bank] = bl.blk[k]; bank_rs[n_bank++] = bl.rs[k]; }
            d_copy(s, s->bank_out, s->cur, n, E);
            BAR();
            d_copy(s, s->cur, s->prelude, n, E);
            BAR();
            bl.n_blk = prelude_n;
        }
        for (uint32_t il = m->core_start; il < m->core_end; il++, vi++) {
            d_tap(s, (int32_t)vi, s->cur, n, tv, nt);
            if (pass == 0) {
                visit(s, vi, il, n, pos0, &bl, 0, NULL);
            } else {
                MixCall mm = mix_prep(s, bank, bank_rs, n_bank, s->bank_out, n, s->g->L[il].msg_query, OC_INK_N_CTRL,
                                      s->msg, n * E, 1, s->mixp2);
                visit(s, vi, il, n, pos0, &bl, s->msg, &mm);
            }
            if ((e = split(s, n)) != OC_OK) return e;
        }
    }
    for (uint32_t il = m->core_end; il < m->n_layer; il++, vi++) {
        d_tap(s, (int32_t)vi, s->cur, n, tv, nt);
        visit(s, vi, il, n, pos0, &bl, 0, NULL);
        if ((e = split(s, n)) != OC_OK) return e;
    }

    d_mix(s, bl.blk, bl.rs, bl.n_blk, s->cur, n, s->g->output_res_query, 1, s->xm);
    BAR();
    d_normq(s, s->xm, s->xm, s->g->output_norm, n, E, s->aqo, s->ado, m->eps);
    BAR();
    d_tap(s, (int32_t)m->n_visits, s->xm, n, tv, nt);
    mark(s, GP_MISC);
    const uint32_t t0 = all_logits ? 0 : n - 1, nl = all_logits ? n : 1;
    if (logits) {
        const uint64_t x = s->xm + F4((uint64_t)t0 * E);
        d_linear(s, &s->g->output, x, E, s->aqo + (uint64_t)t0 * E, s->ado + F4((uint64_t)t0 * E / 64), nl, s->a_logits,
                 m->n_vocab, m->softcap);
        BAR();
        mark(s, GP_HEAD);
    }
    if (s->args_used + 256 > s->args_cap) return OC_ERR_INTERNAL;
    e = submit(s);
    if (e != OC_OK) return e;
    s->prof_calls++;
    s->prof_tokens += n;
    s->last_n = n;
    s->last_big = n > HMAX;
    if (logits) memcpy(logits, s->h_logits, (size_t)nl * m->n_vocab * sizeof(float));
    if (taps && nt) memcpy(taps, s->h_taps, (size_t)n * nt * E * sizeof(float));
    return OC_OK;
}

/* ─── DSpark drafter on the GPU ─────────────────────────────────────────── */

typedef struct {
    GMat     wq, wk, wv, wo, wkc, wvc, gate, up, down;
    uint64_t attn_norm, q_norm, k_norm, k_ctx_norm, ctx_norm, ffn_norm;
    float    ctx_mix[4];
    bool     swa;
} GDLayer;

#define DS_NB 8   /* largest drafter block on the GPU (matvec columns) */

struct InkGpuDraft {
    InkGpu  *s;
    const OcInklingDraft *d;
    GModel  *w;                     /* drafter weights; w->vk is the target's */
    GDLayer  L[4];
    GMat     rnnw, w2m;             /* Markov head: RNN projection, folded bias table */
    uint64_t output_norm, rope_inv, rnn_b, w2_b, w1;
    uint32_t n_ctx, cap[4];
    OcVkBuf  kv[4], dev, host;
    uint64_t kc[4], vc[4];
    uint64_t h, xn, q, kb, vb, o, a, gt, up, aq, ad;
    uint64_t base, st, joint, jo, mq, md, bias, part;   /* Markov walk */
    float   *h_in, *h_frnn, *h_ptop;
    uint32_t *h_tok;
    uint64_t a_in, a_frnn, a_tok, a_ptop;
};

static void up_draft(GModel *g, const OcInklingDraft *d, InkGpuDraft *gd)
{
    const size_t E = d->n_embd, HD = d->head_dim;
    for (uint32_t l = 0; l < d->n_layer; l++) {
        const OcInkDraftLayer *ly = &d->layers[l];
        GDLayer *L = &gd->L[l];
        up_mat(g, &ly->wq, &L->wq);
        up_mat(g, &ly->wk, &L->wk);
        up_mat(g, &ly->wv, &L->wv);
        up_mat(g, &ly->wo, &L->wo);
        up_mat(g, &ly->wk_ctx, &L->wkc);
        up_mat(g, &ly->wv_ctx, &L->wvc);
        up_mat(g, &ly->ffn_gate, &L->gate);
        up_mat(g, &ly->ffn_up, &L->up);
        up_mat(g, &ly->ffn_down, &L->down);
        L->attn_norm = up_f32(g, ly->attn_norm, E);
        L->q_norm = up_f32(g, ly->q_norm, HD);
        L->k_norm = up_f32(g, ly->k_norm, HD);
        L->k_ctx_norm = up_f32(g, ly->k_ctx_norm, HD);
        L->ctx_norm = up_f32(g, ly->ctx_norm, E);
        L->ffn_norm = up_f32(g, ly->ffn_norm, E);
        for (uint32_t k = 0; k < d->n_targets; k++) L->ctx_mix[k] = ly->ctx_mix[k];
        L->swa = ly->swa;
    }
    gd->output_norm = up_f32(g, d->output_norm, E);
    const size_t R = d->rank, V = d->n_vocab;
    const OcInkMat rp = { GT_F32, (const uint8_t *)d->rnn_proj, 3 * R, 2 * R + 2 * E, 1, (2 * R + 2 * E) * sizeof(float),
                          3 * R * (2 * R + 2 * E) * sizeof(float), NULL };
    const OcInkMat wm = { GT_NVFP4, d->w2, V, R, 1, oc_nvfp4_row_bytes(R), V * oc_nvfp4_row_bytes(R), &d->w2_scale };
    up_mat(g, &rp, &gd->rnnw);
    up_mat(g, &wm, &gd->w2m);
    gd->rnn_b = up_f32(g, d->rnn_proj_b, 3 * R);
    gd->w2_b = up_f32(g, d->w2_b, V);
    gd->w1 = up_raw(g, d->markov_w1, V * R * sizeof(uint16_t));
    float inv[64];
    for (size_t i = 0; i < HD / 2; i++) inv[i] = powf(d->rope_base, -2.0f * (float)i / (float)HD);
    gd->rope_inv = up_f32(g, inv, HD / 2);
}

void ink_gpu_draft_free(InkGpuDraft *gd)
{
    if (!gd) return;
    OcVk *vk = gd->s->vk;
    for (int l = 0; l < 4; l++) oc_vk_buf_free(vk, &gd->kv[l]);
    oc_vk_buf_free(vk, &gd->dev);
    oc_vk_buf_free(vk, &gd->host);
    if (gd->w)
        for (uint32_t i = 0; i < MAX_CHUNK; i++) if (gd->w->chunk[i].impl) oc_vk_buf_free(vk, &gd->w->chunk[i]);
    free(gd->w);
    free(gd);
}

OcError ink_gpu_draft_new(InkGpu *s, const OcInklingDraft *d, uint32_t n_ctx, InkGpuDraft **out)
{
    *out = NULL;
    if (d->head_dim != 128 || d->n_targets != s->n_taps || d->n_targets > 4 || d->block_size > DS_NB ||
        d->n_layer > 4 || d->n_embd != s->m->n_embd || d->n_embd > 4608 || d->n_head % d->n_head_kv ||
        d->rank != 256 || n_ctx > s->n_ctx)
        return OC_ERR_MODEL;
    InkGpuDraft *gd = calloc(1, sizeof *gd);
    if (!gd) return OC_ERR_OOM;
    gd->s = s;
    gd->d = d;
    gd->n_ctx = n_ctx;
    gd->w = calloc(1, sizeof *gd->w);
    if (!gd->w) { free(gd); return OC_ERR_OOM; }
    gd->w->vk = s->vk;
    gd->w->chunk_cap = s->g->chunk_cap;
    up_draft(gd->w, d, gd);
    OcError e = gmodel_alloc(gd->w, "the drafter's weights");
    if (e == OC_OK) {
        up_draft(gd->w, d, gd);
        if (gd->w->fail) e = OC_ERR_OOM;
    }
    const size_t E = d->n_embd, KV = (size_t)d->n_head_kv * 128, QD = (size_t)d->n_head * 128, FF = d->n_ff;
    const size_t B = s->B, NB = DS_NB, V = d->n_vocab, R = d->rank, F = sizeof(float);
    for (uint32_t l = 0; l < d->n_layer && e == OC_OK; l++) {
        uint32_t cap = d->layers[l].swa ? d->n_swa + 64 : n_ctx;
        if (cap > n_ctx) cap = n_ctx;
        gd->cap[l] = cap;
        e = oc_vk_buf_new(s->vk, 2 * (size_t)cap * KV * F, OC_VK_MEM_DEVICE, &gd->kv[l]);
        gd->kc[l] = gd->kv[l].addr;
        gd->vc[l] = gd->kv[l].addr + (size_t)cap * KV * F;
    }
    const size_t aqb = B * (E > QD ? E : QD) > NB * FF ? B * (E > QD ? E : QD) : NB * FF;
    for (int pass = 0; pass < 2 && e == OC_OK; pass++) {
        size_t off = 0;
        const uint64_t base = pass ? gd->dev.addr : 0;
#define T(dst, n) (dst) = take(base, &off, (n))
        T(gd->h, NB * E * F); T(gd->xn, B * E * F); T(gd->q, NB * QD * F); T(gd->kb, B * KV * F);
        T(gd->vb, B * KV * F); T(gd->o, NB * QD * F); T(gd->a, NB * E * F); T(gd->gt, NB * FF * F);
        T(gd->up, NB * FF * F); T(gd->aq, aqb); T(gd->ad, aqb / 64 * F);
        T(gd->base, (NB - 1) * V * F); T(gd->st, R * F); T(gd->joint, (2 * R + 2 * E) * F); T(gd->jo, 3 * R * F);
        T(gd->mq, R); T(gd->md, R / 64 * F); T(gd->bias, V * F); T(gd->part, (V + 1023) / 1024 * 4 * F);
#undef T
        if (pass == 0) e = oc_vk_buf_new(s->vk, off, OC_VK_MEM_DEVICE, &gd->dev);
    }
    const size_t hi = NB * E * F, hf = E * F;
    if (e == OC_OK) e = oc_vk_buf_new(s->vk, hi + hf + 2 * NB * 4, OC_VK_MEM_HOST, &gd->host);
    if (e == OC_OK && !gd->host.map) e = OC_ERR_BACKEND;
    if (e != OC_OK) { ink_gpu_draft_free(gd); return e; }
    uint8_t *hm = gd->host.map;
    gd->h_in = (float *)hm;                          gd->a_in = gd->host.addr;
    gd->h_frnn = (float *)(hm + hi);                 gd->a_frnn = gd->host.addr + hi;
    gd->h_tok = (uint32_t *)(hm + hi + hf);          gd->a_tok = gd->host.addr + hi + hf;
    gd->h_ptop = (float *)(hm + hi + hf + NB * 4);   gd->a_ptop = gd->host.addr + hi + hf + NB * 4;
    *out = gd;
    return OC_OK;
}

/* Per-head norm (w != 0) and rotary embedding (inv != 0) of n rows of heads
 * 128-wide heads into rows (orow0 + t) % ocap of y. */
static void d_rope(InkGpu *s, uint64_t x, uint32_t xs, uint64_t w, uint64_t inv, uint64_t y, uint32_t ys, uint32_t pos0,
                   uint32_t ocap, uint32_t orow0, uint32_t tmin, uint32_t n, uint32_t heads, float eps)
{
    const RopePush p = { x, w, y, inv, xs, ys, pos0, ocap, orow0, tmin, eps, 0 };
    oc_vk_dispatch(s->vk, s->g->dsrope, &p, n, heads, 1);
}

OcError ink_gpu_draft_inject(InkGpuDraft *gd, uint32_t n, uint32_t pos0)
{
    InkGpu *s = gd->s;
    const OcInklingDraft *d = gd->d;
    if (n == 0) return OC_OK;
    if (n > s->B || pos0 + n > gd->n_ctx) return OC_ERR_INVALID_ARG;
    const uint32_t E = d->n_embd, KV = d->n_head_kv * 128;
    OcError e = oc_vk_begin(s->vk);
    if (e != OC_OK) return e;
    for (uint32_t l = 0; l < d->n_layer; l++) {
        const GDLayer *L = &gd->L[l];
        d_normq_mix(s, s->a_taps, gd->xn, L->ctx_norm, n, E, gd->aq, gd->ad, d->eps, d->n_targets, L->ctx_mix);
        BAR();
        d_linear(s, &L->wkc, gd->xn, E, gd->aq, gd->ad, n, gd->kb, KV, 0.0f);
        d_linear(s, &L->wvc, gd->xn, E, gd->aq, gd->ad, n, gd->vb, KV, 0.0f);
        BAR();
        const uint32_t cap = gd->cap[l], tmin = n > cap ? n - cap : 0;
        d_rope(s, gd->kb, KV, L->k_ctx_norm, gd->rope_inv, gd->kc[l], KV, pos0, cap, pos0, tmin, n, d->n_head_kv, d->eps);
        d_rope(s, gd->vb, KV, 0, 0, gd->vc[l], KV, pos0, cap, pos0, tmin, n, d->n_head_kv, 0.0f);
        BAR();
    }
    return oc_vk_submit(s->vk);
}

OcError ink_gpu_draft_forward(InkGpuDraft *gd, const float *h, uint32_t nb, uint32_t pos0, uint32_t anchor,
                              const float *frnn, uint32_t *tok, float *ptop)
{
    InkGpu *s = gd->s;
    const OcInklingDraft *d = gd->d;
    const uint32_t E = d->n_embd, NH = d->n_head, NKV = d->n_head_kv, QD = NH * 128, KV = NKV * 128, FF = d->n_ff;
    const uint32_t V = d->n_vocab, R = d->rank;
    if (nb < 2 || nb > DS_NB || pos0 > gd->n_ctx) return OC_ERR_INVALID_ARG;
    memcpy(gd->h_in, h, (size_t)nb * E * sizeof(float));
    memcpy(gd->h_frnn, frnn, (size_t)E * sizeof(float));
    OcError e = oc_vk_begin(s->vk);
    if (e != OC_OK) return e;
    d_copy(s, gd->h, gd->a_in, nb, E);
    BAR();
    for (uint32_t l = 0; l < d->n_layer; l++) {
        const GDLayer *L = &gd->L[l];
        d_normq(s, gd->h, gd->xn, L->attn_norm, nb, E, gd->aq, gd->ad, d->eps);
        BAR();
        d_linear(s, &L->wq, gd->xn, E, gd->aq, gd->ad, nb, gd->q, QD, 0.0f);
        d_linear(s, &L->wk, gd->xn, E, gd->aq, gd->ad, nb, gd->kb, KV, 0.0f);
        d_linear(s, &L->wv, gd->xn, E, gd->aq, gd->ad, nb, gd->vb, KV, 0.0f);
        BAR();
        d_rope(s, gd->q, QD, L->q_norm, gd->rope_inv, gd->q, QD, pos0, 0xFFFFFFFFu, 0, 0, nb, NH, d->eps);
        d_rope(s, gd->kb, KV, L->k_norm, gd->rope_inv, gd->kb, KV, pos0, 0xFFFFFFFFu, 0, 0, nb, NKV, d->eps);
        BAR();
        const uint32_t lo = L->swa && pos0 > d->n_swa ? pos0 - d->n_swa : 0;
        const DsAttnPush ap = { gd->q, gd->kc[l], gd->vc[l], gd->kb, gd->vb, gd->o, pos0 - lo, lo, gd->cap[l], L->swa, nb,
                                NH, NKV, 0, 1.0f / sqrtf(128.0f), 0 };
        oc_vk_dispatch(s->vk, s->g->dsattn, &ap, nb, NH, 1);
        BAR();
        d_quant(s, gd->o, nb, QD, QD, gd->aq, gd->ad);
        BAR();
        d_linear(s, &L->wo, gd->o, QD, gd->aq, gd->ad, nb, gd->a, E, 0.0f);
        BAR();
        d_ew(s, gd->h, gd->h, gd->a, nb, E, E, E, E, 1);
        BAR();
        d_normq(s, gd->h, gd->xn, L->ffn_norm, nb, E, gd->aq, gd->ad, d->eps);
        BAR();
        d_linear(s, &L->gate, gd->xn, E, gd->aq, gd->ad, nb, gd->gt, FF, 0.0f);
        d_linear(s, &L->up, gd->xn, E, gd->aq, gd->ad, nb, gd->up, FF, 0.0f);
        BAR();
        d_ew(s, gd->gt, gd->gt, gd->up, nb, FF, FF, FF, FF, 2);
        BAR();
        d_quant(s, gd->gt, nb, FF, FF, gd->aq, gd->ad);
        BAR();
        d_linear(s, &L->down, gd->gt, FF, gd->aq, gd->ad, nb, gd->a, E, 0.0f);
        BAR();
        d_ew(s, gd->h, gd->h, gd->a, nb, E, E, E, E, 1);
        BAR();
    }
    d_normq(s, gd->h, gd->xn, gd->output_norm, nb, E, gd->aq, gd->ad, d->eps);
    BAR();
    d_linear(s, &s->g->output, gd->xn + F4(E), E, gd->aq + E, gd->ad + F4(E / 64), nb - 1, gd->base, V, 0.0f);
    oc_vk_fill(s->vk, &gd->dev, gd->st - gd->dev.addr, (size_t)R * sizeof(float), 0);
    BAR();

    /* Markov walk: per slot the RNN over [state, w1[prev], xn_i, f_rnn], the
     * folded bias table on its Markov vector, and the greedy pick of
     * base + bias, which is prev of the next slot */
    const uint32_t nwg = cdiv(V, 1024), nj = 2 * R + 2 * E;
    for (uint32_t i = 1; i < nb; i++) {
        const WalkJointPush jp = { gd->st, gd->w1, gd->xn, gd->a_frnn, gd->a_tok, gd->joint, R, E, i, anchor };
        oc_vk_dispatch(s->vk, s->g->walkj, &jp, cdiv(nj, 256), 1, 1);
        BAR();
        d_linear(s, &gd->rnnw, gd->joint, nj, 0, 0, 1, gd->jo, 3 * R, 0.0f);
        BAR();
        const WalkGatePush gp = { gd->jo, gd->rnn_b, gd->st, gd->mq, gd->md, R, 0 };
        oc_vk_dispatch(s->vk, s->g->walkg, &gp, 1, 1, 1);
        BAR();
        d_linear(s, &gd->w2m, 0, R, gd->mq, gd->md, 1, gd->bias, V, 0.0f);
        BAR();
        WalkPickPush pp = { gd->base, gd->bias, gd->w2_b, gd->part, gd->a_tok, gd->a_ptop, V, i, nwg, 0, s->m->softcap, 0 };
        oc_vk_dispatch(s->vk, s->g->walkp, &pp, nwg, 1, 1);
        BAR();
        pp.phase = 1;
        oc_vk_dispatch(s->vk, s->g->walkp, &pp, 1, 1, 1);
        BAR();
    }
    if ((e = oc_vk_submit(s->vk)) != OC_OK) return e;
    memcpy(tok, gd->h_tok, (size_t)(nb - 1) * sizeof(uint32_t));
    memcpy(ptop, gd->h_ptop, (size_t)(nb - 1) * sizeof(float));
    return OC_OK;
}

#else /* !OC_VULKAN */

bool ink_gpu_compiled(void) { return false; }
OcError ink_gpu_model_load(OcInklingModel *m) { (void)m; return OC_ERR_BACKEND; }
void ink_gpu_model_free(OcInklingModel *m) { (void)m; }
OcError ink_gpu_new(const OcInklingModel *m, uint32_t n_ctx, uint32_t n_taps, InkGpu **out)
{
    (void)m; (void)n_ctx; (void)n_taps;
    *out = NULL;
    return OC_ERR_BACKEND;
}
void ink_gpu_free(InkGpu *g) { (void)g; }
OcError ink_gpu_forward(InkGpu *g, const uint32_t *tokens, size_t n, uint32_t pos0, uint32_t keep,
                        float *logits, bool all_logits, const int32_t *tap_visits, uint32_t n_taps, float *taps)
{
    (void)g; (void)tokens; (void)n; (void)pos0; (void)keep; (void)logits; (void)all_logits;
    (void)tap_visits; (void)n_taps; (void)taps;
    return OC_ERR_BACKEND;
}

OcError ink_gpu_save(InkGpu *g, uint32_t keep, uint32_t slot, uint32_t pos)
{
    (void)g; (void)keep; (void)slot; (void)pos;
    return OC_ERR_BACKEND;
}
OcError ink_gpu_restore(InkGpu *g, uint32_t slot, uint32_t pos)
{
    (void)g; (void)slot; (void)pos;
    return OC_ERR_BACKEND;
}

void ink_gpu_draft_free(InkGpuDraft *gd) { (void)gd; }
OcError ink_gpu_draft_new(InkGpu *s, const OcInklingDraft *d, uint32_t n_ctx, InkGpuDraft **out)
{
    (void)s; (void)d; (void)n_ctx;
    *out = NULL;
    return OC_ERR_BACKEND;
}
OcError ink_gpu_draft_inject(InkGpuDraft *gd, uint32_t n, uint32_t pos0)
{
    (void)gd; (void)n; (void)pos0;
    return OC_ERR_BACKEND;
}
OcError ink_gpu_draft_forward(InkGpuDraft *gd, const float *h, uint32_t nb, uint32_t pos0, uint32_t anchor,
                              const float *frnn, uint32_t *tok, float *ptop)
{
    (void)gd; (void)h; (void)nb; (void)pos0; (void)anchor; (void)frnn; (void)tok; (void)ptop;
    return OC_ERR_BACKEND;
}

#endif
