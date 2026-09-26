/* test_k2_arch.c — K2-Horizon ("k2-horizon") forward pass.
 *
 * Builds a tiny random K2-Horizon GGUF with oc_gguf_writer — one leading
 * dense block and two MoE + MoVA blocks, two RMSNorm groups — and checks the
 * oxidize-c forward against a deliberately naive reference of the spec
 * written out below (report-k2-math.md / llama.cpp-k2 k2-horizon.cpp):
 *
 *   xn   = GroupRMS(x, attn_norm)               (2 groups, eps 1e-6)
 *   Q,K  = Wq xn, Wk xn;  G = Wgate xn
 *   V    = Wv xn                                (dense block)
 *        = sum_e w_e SiLU(Wv[e] xn)             (MoVA: sigmoid router,
 *                                                top-k on p+bias, weights
 *                                                from unbiased p, normalized
 *                                                with a 6.1035e-5 clamp,
 *                                                times 2.5)
 *   NEOX RoPE over head_dim at base 1e7, causal softmax(QK^T/sqrt(d)) V
 *   out *= softplus_{beta=ln2}(G);  x += Wo out
 *   FFN on GroupRMS(x, ffn_norm): dense SwiGLU, or sigmoid MoE (same routing
 *   rule, top-k of 6) + ungated shared expert
 *   logits = Wout GroupRMS(x, output_norm)
 *
 * The reference runs in double. Decode, batched prefill, and the reference
 * must agree within 1e-4.
 */
#include <criterion/criterion.h>

#include "oxidize/gguf_writer.h"
#include "oxidize/k2_arch.h"
#include "oxidize/kv_rq.h"
#include "oxidize/llama.h"
#include "oxidize/matvec.h"
#include "oxidize/parallel.h"
#include "oxidize/quant.h"
#include "oxidize/sampling.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── Geometry ──────────────────────────────────────────────────────────── */
#define K2_LAYERS   3u
#define K2_DENSE    1u      /* leading dense blocks                        */
#define K2_EMBD     32u
#define K2_GROUPS   2u
#define K2_HEADS    4u
#define K2_KV_HEADS 2u
#define K2_HD       64u     /* != n_embd / n_head, as in the real model;
                               * 64 so the RQ KV cache supports it       */
#define K2_FF       48u
#define K2_NEXP     6u
#define K2_NUSED    2u
#define K2_EFF      8u
#define K2_SHFF     8u
#define K2_NVEXP    5u
#define K2_NVUSED   3u
#define K2_VOCAB    23u
#define K2_CTX      64u
#define K2_THETA    1.0e7f
#define K2_EPS      1e-6f
#define K2_SCALE    2.5f

#define K2_QROWS  (K2_HEADS * K2_HD)
#define K2_KVROWS (K2_KV_HEADS * K2_HD)

#define FIXTURE(name) "/tmp/oxidize-c-k2-" name ".gguf"

/* ─── Deterministic random weights ─────────────────────────────────────── */
static uint64_t g_rng;
static float rnd(void)
{
    g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((g_rng >> 40) & 0xFFFFFF) / (float)0x1000000 * 2.0f - 1.0f;
}

/* A tensor kept for the reference: f32 values exactly as the model sees
 * them (dequantized when the file stores it quantized). */
typedef struct {
    char name[64];
    float *v;
    size_t n;
} RefTensor;

typedef struct {
    RefTensor t[128];
    size_t n;
} RefModel;

static const float *ref_get(const RefModel *m, const char *name)
{
    for (size_t i = 0; i < m->n; i++)
        if (strcmp(m->t[i].name, name) == 0) return m->t[i].v;
    cr_assert_fail("reference tensor %s missing", name);
    return NULL;
}

static const float *ref_getl(const RefModel *m, unsigned l, const char *suf)
{
    char nm[64];
    snprintf(nm, sizeof nm, "blk.%u.%s", l, suf);
    return ref_get(m, nm);
}

static void ref_free(RefModel *m)
{
    for (size_t i = 0; i < m->n; i++) free(m->t[i].v);
    m->n = 0;
}

/* f32 -> f16 bits (round to nearest even), enough for Q8_0 scales. */
static uint16_t f32_to_f16(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) return (uint16_t)sign;          /* flush tiny to zero */
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    uint32_t h = sign | ((uint32_t)exp << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;
    return (uint16_t)h;
}

/* Add a tensor with ggml dims (d0 = cols, d1 = rows, d2 = experts).
 * `amp` scales the random values, `base` offsets them (norm weights ~1).
 * `q8` stores it as Q8_0 (d0 must be a multiple of 32). */
static void add_tensor(OcGgufWriter *w, RefModel *rm, const char *name,
                       uint64_t d0, uint64_t d1, uint64_t d2, float amp,
                       float base, bool q8)
{
    uint64_t dims[3] = { d0, d1 ? d1 : 1, d2 ? d2 : 1 };
    const uint32_t n_dims = d2 ? 3u : (d1 ? 2u : 1u);
    const size_t n = (size_t)dims[0] * dims[1] * dims[2];
    float *vals = malloc(n * sizeof(float));
    cr_assert_not_null(vals);
    for (size_t i = 0; i < n; i++) vals[i] = base + amp * rnd();

    OcError e;
    if (q8) {
        cr_assert_eq(d0 % 32u, 0u, "Q8_0 needs cols %% 32 == 0 (%s)", name);
        const size_t n_blocks = n / 32u;
        uint8_t *buf = malloc(n_blocks * 34u);
        cr_assert_not_null(buf);
        for (size_t b = 0; b < n_blocks; b++) {
            const float *src = vals + b * 32u;
            float amax = 0.0f;
            for (int i = 0; i < 32; i++)
                if (fabsf(src[i]) > amax) amax = fabsf(src[i]);
            const float d = amax / 127.0f;
            const uint16_t dh = f32_to_f16(d);
            memcpy(buf + b * 34u, &dh, 2);
            int8_t *qs = (int8_t *)(buf + b * 34u + 2u);
            for (int i = 0; i < 32; i++)
                qs[i] = (int8_t)(d > 0.0f ? lrintf(src[i] / d) : 0);
        }
        e = oc_gguf_writer_add_tensor(w, name, n_dims, dims, 8 /* Q8_0 */,
                                      buf, (uint64_t)(n_blocks * 34u));
        /* Reference sees exactly what the model sees. */
        const size_t row_bytes = (size_t)d0 / 32u * 34u;
        const size_t n_rows = n / (size_t)d0;
        for (size_t r = 0; r < n_rows; r++)
            oc_quant_dequant_row(OC_QUANT_Q8_0, buf + r * row_bytes, row_bytes,
                                 vals + r * d0, (size_t)d0);
        free(buf);
    } else {
        e = oc_gguf_writer_add_tensor(w, name, n_dims, dims, 0 /* F32 */,
                                      vals, (uint64_t)(n * sizeof(float)));
    }
    cr_assert_eq(e, OC_OK, "add_tensor(%s) failed: %d", name, (int)e);
    cr_assert_lt(rm->n, sizeof rm->t / sizeof rm->t[0]);
    RefTensor *t = &rm->t[rm->n++];
    snprintf(t->name, sizeof t->name, "%s", name);
    t->v = vals;
    t->n = n;
}

static void add_layer_tensor(OcGgufWriter *w, RefModel *rm, unsigned l,
                             const char *suf, uint64_t d0, uint64_t d1,
                             uint64_t d2, float amp, float base, bool q8)
{
    char nm[64];
    snprintf(nm, sizeof nm, "blk.%u.%s", l, suf);
    add_tensor(w, rm, nm, d0, d1, d2, amp, base, q8);
}

/* Write the fixture. `quant` stores the attention projections and the value
 * experts as Q8_0 to exercise the fused multi-matrix path. */
static void add_mtp_tensors(OcGgufWriter *w, RefModel *rm, bool quant);

/* mtp: 0 = base only, 1 = merged file (block_count = base + 1 with the
 * nextn head as blk.K2_LAYERS). */
static void build_k2_gguf_mtp(const char *path, RefModel *rm, uint64_t seed,
                              bool quant, int mtp);

static void build_k2_gguf(const char *path, RefModel *rm, uint64_t seed,
                          bool quant)
{
    build_k2_gguf_mtp(path, rm, seed, quant, 0);
}

static void write_k2_meta(OcGgufWriter *w, uint32_t block_count);

static void build_k2_gguf_mtp(const char *path, RefModel *rm, uint64_t seed,
                              bool quant, int mtp)
{
    g_rng = seed;
    memset(rm, 0, sizeof(*rm));
    OcGgufWriter w;
    cr_assert_eq(oc_gguf_writer_init(path, "k2-horizon", &w), OC_OK);
    write_k2_meta(&w, K2_LAYERS + (mtp ? 1u : 0u));
    if (mtp) {
        oc_gguf_writer_add_uint32(&w, "k2-horizon.nextn_predict_layers", 1);
        oc_gguf_writer_add_uint32(&w, "k2-horizon.nextn.base_block_count",
                                  K2_LAYERS);
    }

    add_tensor(&w, rm, "token_embd.weight", K2_EMBD, K2_VOCAB, 0, 1.0f, 0, false);
    add_tensor(&w, rm, "output_norm.weight", K2_EMBD, 0, 0, 0.3f, 1.0f, false);
    add_tensor(&w, rm, "output.weight", K2_EMBD, K2_VOCAB, 0, 0.4f, 0, false);

    for (unsigned l = 0; l < K2_LAYERS; l++) {
        const bool moe = l >= K2_DENSE;
        add_layer_tensor(&w, rm, l, "attn_norm.weight", K2_EMBD, 0, 0, 0.3f, 1.0f, false);
        add_layer_tensor(&w, rm, l, "attn_q.weight", K2_EMBD, K2_QROWS, 0, 0.35f, 0, quant);
        add_layer_tensor(&w, rm, l, "attn_k.weight", K2_EMBD, K2_KVROWS, 0, 0.35f, 0, quant);
        add_layer_tensor(&w, rm, l, "attn_gate.weight", K2_EMBD, K2_QROWS, 0, 0.5f, 0, quant);
        add_layer_tensor(&w, rm, l, "attn_output.weight", K2_QROWS, K2_EMBD, 0, 0.2f, 0, quant);
        if (!moe) {
            add_layer_tensor(&w, rm, l, "attn_v.weight", K2_EMBD, K2_KVROWS, 0, 0.35f, 0, quant);
        } else {
            add_layer_tensor(&w, rm, l, "attn_v_gate.weight", K2_EMBD, K2_NVEXP, 0, 0.4f, 0, false);
            add_layer_tensor(&w, rm, l, "attn_v_gate.bias", K2_NVEXP, 0, 0, 0.15f, 0, false);
            add_layer_tensor(&w, rm, l, "attn_v_exps.weight", K2_EMBD, K2_KVROWS, K2_NVEXP, 0.35f, 0, quant);
        }
        add_layer_tensor(&w, rm, l, "ffn_norm.weight", K2_EMBD, 0, 0, 0.3f, 1.0f, false);
        if (!moe) {
            add_layer_tensor(&w, rm, l, "ffn_gate.weight", K2_EMBD, K2_FF, 0, 0.3f, 0, false);
            add_layer_tensor(&w, rm, l, "ffn_up.weight", K2_EMBD, K2_FF, 0, 0.3f, 0, false);
            add_layer_tensor(&w, rm, l, "ffn_down.weight", K2_FF, K2_EMBD, 0, 0.2f, 0, false);
        } else {
            add_layer_tensor(&w, rm, l, "ffn_gate_inp.weight", K2_EMBD, K2_NEXP, 0, 0.4f, 0, false);
            add_layer_tensor(&w, rm, l, "exp_probs_b.bias", K2_NEXP, 0, 0, 0.15f, 0, false);
            add_layer_tensor(&w, rm, l, "ffn_gate_exps.weight", K2_EMBD, K2_EFF, K2_NEXP, 0.3f, 0, false);
            add_layer_tensor(&w, rm, l, "ffn_up_exps.weight", K2_EMBD, K2_EFF, K2_NEXP, 0.3f, 0, false);
            add_layer_tensor(&w, rm, l, "ffn_down_exps.weight", K2_EFF, K2_EMBD, K2_NEXP, 0.3f, 0, false);
            add_layer_tensor(&w, rm, l, "ffn_gate_shexp.weight", K2_EMBD, K2_SHFF, 0, 0.3f, 0, false);
            add_layer_tensor(&w, rm, l, "ffn_up_shexp.weight", K2_EMBD, K2_SHFF, 0, 0.3f, 0, false);
            add_layer_tensor(&w, rm, l, "ffn_down_shexp.weight", K2_SHFF, K2_EMBD, 0, 0.3f, 0, false);
        }
    }
    if (mtp) add_mtp_tensors(&w, rm, quant);
    cr_assert_eq(oc_gguf_writer_finalize(&w), OC_OK, "finalize failed");
    oc_gguf_writer_free(&w);
}

static void write_k2_meta(OcGgufWriter *w, uint32_t block_count)
{
    oc_gguf_writer_add_uint32(w, "k2-horizon.block_count", block_count);
    oc_gguf_writer_add_uint32(w, "k2-horizon.context_length", K2_CTX);
    oc_gguf_writer_add_uint32(w, "k2-horizon.embedding_length", K2_EMBD);
    oc_gguf_writer_add_uint32(w, "k2-horizon.feed_forward_length", K2_FF);
    oc_gguf_writer_add_uint32(w, "k2-horizon.attention.head_count", K2_HEADS);
    oc_gguf_writer_add_uint32(w, "k2-horizon.attention.head_count_kv",
                              K2_KV_HEADS);
    oc_gguf_writer_add_uint32(w, "k2-horizon.attention.key_length", K2_HD);
    oc_gguf_writer_add_uint32(w, "k2-horizon.attention.value_length", K2_HD);
    oc_gguf_writer_add_uint32(w, "k2-horizon.rope.dimension_count", K2_HD);
    oc_gguf_writer_add_float32(w, "k2-horizon.rope.freq_base", K2_THETA);
    oc_gguf_writer_add_float32(
        w, "k2-horizon.attention.layer_norm_rms_epsilon", K2_EPS);
    oc_gguf_writer_add_uint32(w, "k2-horizon.attention.group_norm_groups",
                              K2_GROUPS);
    oc_gguf_writer_add_uint32(w, "k2-horizon.expert_count", K2_NEXP);
    oc_gguf_writer_add_uint32(w, "k2-horizon.expert_used_count", K2_NUSED);
    oc_gguf_writer_add_uint32(w, "k2-horizon.expert_feed_forward_length",
                              K2_EFF);
    oc_gguf_writer_add_uint32(w, "k2-horizon.expert_shared_count", 1);
    oc_gguf_writer_add_uint32(
        w, "k2-horizon.expert_shared_feed_forward_length", K2_SHFF);
    oc_gguf_writer_add_uint32(w, "k2-horizon.leading_dense_block_count",
                              K2_DENSE);
    oc_gguf_writer_add_float32(w, "k2-horizon.expert_weights_scale",
                               K2_SCALE);
    oc_gguf_writer_add_uint32(w, "k2-horizon.expert_gating_func", 2);
    oc_gguf_writer_add_uint32(w, "k2-horizon.attention.value_expert_count",
                              K2_NVEXP);
    oc_gguf_writer_add_uint32(
        w, "k2-horizon.attention.value_expert_used_count", K2_NVUSED);

}

/* The nextn head: one dense K2 block plus eh_proj / enorm / hnorm /
 * shared_head_norm, as blk.K2_LAYERS. */
static void add_mtp_tensors(OcGgufWriter *w, RefModel *rm, bool quant)
{
    const unsigned l = K2_LAYERS;
    add_layer_tensor(w, rm, l, "nextn.eh_proj.weight", 2u * K2_EMBD, K2_EMBD, 0, 0.3f, 0, quant);
    add_layer_tensor(w, rm, l, "nextn.enorm.weight", K2_EMBD, 0, 0, 0.3f, 1.0f, false);
    add_layer_tensor(w, rm, l, "nextn.hnorm.weight", K2_EMBD, 0, 0, 0.3f, 1.0f, false);
    add_layer_tensor(w, rm, l, "nextn.shared_head_norm.weight", K2_EMBD, 0, 0, 0.3f, 1.0f, false);
    add_layer_tensor(w, rm, l, "attn_norm.weight", K2_EMBD, 0, 0, 0.3f, 1.0f, false);
    add_layer_tensor(w, rm, l, "attn_q.weight", K2_EMBD, K2_QROWS, 0, 0.35f, 0, quant);
    add_layer_tensor(w, rm, l, "attn_k.weight", K2_EMBD, K2_KVROWS, 0, 0.35f, 0, quant);
    add_layer_tensor(w, rm, l, "attn_v.weight", K2_EMBD, K2_KVROWS, 0, 0.35f, 0, quant);
    add_layer_tensor(w, rm, l, "attn_gate.weight", K2_EMBD, K2_QROWS, 0, 0.5f, 0, quant);
    add_layer_tensor(w, rm, l, "attn_output.weight", K2_QROWS, K2_EMBD, 0, 0.2f, 0, quant);
    add_layer_tensor(w, rm, l, "ffn_norm.weight", K2_EMBD, 0, 0, 0.3f, 1.0f, false);
    add_layer_tensor(w, rm, l, "ffn_gate.weight", K2_EMBD, K2_FF, 0, 0.3f, 0, false);
    add_layer_tensor(w, rm, l, "ffn_up.weight", K2_EMBD, K2_FF, 0, 0.3f, 0, false);
    add_layer_tensor(w, rm, l, "ffn_down.weight", K2_FF, K2_EMBD, 0, 0.2f, 0, false);
}

/* Sidecar: base metadata with block_count = base + 1, nextn keys, and only
 * the head's tensors. `base_count` lets a test lie about the base. */
static void build_k2_mtp_sidecar(const char *path, RefModel *rm, uint64_t seed,
                                 uint32_t base_count)
{
    g_rng = seed;
    OcGgufWriter w;
    cr_assert_eq(oc_gguf_writer_init(path, "k2-horizon", &w), OC_OK);
    write_k2_meta(&w, base_count + 1u);
    oc_gguf_writer_add_uint32(&w, "k2-horizon.nextn_predict_layers", 1);
    oc_gguf_writer_add_uint32(&w, "k2-horizon.nextn.base_block_count", base_count);
    add_mtp_tensors(&w, rm, false);
    cr_assert_eq(oc_gguf_writer_finalize(&w), OC_OK, "finalize failed");
    oc_gguf_writer_free(&w);
}

/* ─── Naive reference (double) ─────────────────────────────────────────── */

/* out[r] = sum_c W[r][c] * in[c]; W row-major [rows][cols]. */
static void ref_mv(const float *W, size_t rows, size_t cols, const double *in,
                   double *out)
{
    for (size_t r = 0; r < rows; r++) {
        double acc = 0.0;
        for (size_t c = 0; c < cols; c++) acc += (double)W[r * cols + c] * in[c];
        out[r] = acc;
    }
}

static void ref_gnorm(const double *x, const float *w, double *out)
{
    const size_t g = K2_EMBD / K2_GROUPS;
    for (size_t gi = 0; gi < K2_GROUPS; gi++) {
        double ss = 0.0;
        for (size_t i = 0; i < g; i++) ss += x[gi * g + i] * x[gi * g + i];
        const double inv = 1.0 / sqrt(ss / (double)g + (double)K2_EPS);
        for (size_t i = 0; i < g; i++)
            out[gi * g + i] = x[gi * g + i] * inv * (double)w[gi * g + i];
    }
}

static double ref_silu(double v) { return v / (1.0 + exp(-v)); }
static double ref_sigmoid(double v) { return 1.0 / (1.0 + exp(-v)); }

/* Sigmoid router: top-k on p + bias, applied weights from unbiased p,
 * normalized (clamped) and scaled. Stable: ties keep the lower index. */
static void ref_route(const double *logits, const float *bias, unsigned n,
                      unsigned k, unsigned *sel, double *wts)
{
    double p[16], score[16];
    bool used[16] = { false };
    for (unsigned i = 0; i < n; i++) {
        p[i] = ref_sigmoid(logits[i]);
        score[i] = p[i] + (bias ? (double)bias[i] : 0.0);
    }
    double sum = 0.0;
    for (unsigned s = 0; s < k; s++) {
        int best = -1;
        for (unsigned i = 0; i < n; i++)
            if (!used[i] && (best < 0 || score[i] > score[best])) best = (int)i;
        used[best] = true;
        sel[s] = (unsigned)best;
        wts[s] = p[best];
        sum += p[best];
    }
    if (sum < 6.103515625e-5) sum = 6.103515625e-5;
    for (unsigned s = 0; s < k; s++) wts[s] = wts[s] / sum * K2_SCALE;
}

static void ref_rope(double *x, int64_t pos)
{
    const unsigned half = K2_HD / 2;
    for (unsigned j = 0; j < half; j++) {
        const double freq = pow((double)K2_THETA, -2.0 * j / (double)K2_HD);
        const double a = (double)pos * freq;
        const double x0 = x[j], x1 = x[half + j];
        x[j] = x0 * cos(a) - x1 * sin(a);
        x[half + j] = x0 * sin(a) + x1 * cos(a);
    }
}

/* Run the reference over tokens[0..T) and write every position's logits. */
static void ref_forward(const RefModel *rm, const uint32_t *tokens, size_t T,
                        double *logits /* [T][vocab] */)
{
    double *kc = calloc((size_t)K2_LAYERS * T * K2_KVROWS, sizeof(double));
    double *vc = calloc((size_t)K2_LAYERS * T * K2_KVROWS, sizeof(double));
    cr_assert(kc && vc);
    const float *emb = ref_get(rm, "token_embd.weight");
    for (size_t t = 0; t < T; t++) {
        double x[K2_EMBD], xn[K2_EMBD];
        for (unsigned i = 0; i < K2_EMBD; i++)
            x[i] = emb[(size_t)tokens[t] * K2_EMBD + i];
        for (unsigned l = 0; l < K2_LAYERS; l++) {
            const bool moe = l >= K2_DENSE;
            ref_gnorm(x, ref_getl(rm, l, "attn_norm.weight"), xn);
            double q[K2_QROWS], k[K2_KVROWS], v[K2_KVROWS], g[K2_QROWS];
            ref_mv(ref_getl(rm, l, "attn_q.weight"), K2_QROWS, K2_EMBD, xn, q);
            ref_mv(ref_getl(rm, l, "attn_k.weight"), K2_KVROWS, K2_EMBD, xn, k);
            ref_mv(ref_getl(rm, l, "attn_gate.weight"), K2_QROWS, K2_EMBD, xn, g);
            if (!moe) {
                ref_mv(ref_getl(rm, l, "attn_v.weight"), K2_KVROWS, K2_EMBD, xn, v);
            } else {
                double rl[K2_NVEXP], wts[K2_NVUSED], eo[K2_KVROWS];
                unsigned sel[K2_NVUSED];
                ref_mv(ref_getl(rm, l, "attn_v_gate.weight"), K2_NVEXP, K2_EMBD,
                       xn, rl);
                ref_route(rl, ref_getl(rm, l, "attn_v_gate.bias"), K2_NVEXP,
                          K2_NVUSED, sel, wts);
                const float *ve = ref_getl(rm, l, "attn_v_exps.weight");
                for (unsigned i = 0; i < K2_KVROWS; i++) v[i] = 0.0;
                for (unsigned s = 0; s < K2_NVUSED; s++) {
                    ref_mv(ve + (size_t)sel[s] * K2_KVROWS * K2_EMBD, K2_KVROWS,
                           K2_EMBD, xn, eo);
                    for (unsigned i = 0; i < K2_KVROWS; i++)
                        v[i] += wts[s] * ref_silu(eo[i]);
                }
            }
            for (unsigned h = 0; h < K2_HEADS; h++) ref_rope(q + h * K2_HD, (int64_t)t);
            for (unsigned h = 0; h < K2_KV_HEADS; h++) ref_rope(k + h * K2_HD, (int64_t)t);
            double *kl = kc + ((size_t)l * T + t) * K2_KVROWS;
            double *vl = vc + ((size_t)l * T + t) * K2_KVROWS;
            memcpy(kl, k, sizeof k);
            memcpy(vl, v, sizeof v);

            double att[K2_QROWS];
            const unsigned grp = K2_HEADS / K2_KV_HEADS;
            for (unsigned h = 0; h < K2_HEADS; h++) {
                const unsigned kvh = h / grp;
                double sc[K2_CTX], mx = -1e300, den = 0.0;
                for (size_t p = 0; p <= t; p++) {
                    const double *kp = kc + ((size_t)l * T + p) * K2_KVROWS + kvh * K2_HD;
                    double d = 0.0;
                    for (unsigned i = 0; i < K2_HD; i++) d += q[h * K2_HD + i] * kp[i];
                    sc[p] = d / sqrt((double)K2_HD);
                    if (sc[p] > mx) mx = sc[p];
                }
                for (size_t p = 0; p <= t; p++) { sc[p] = exp(sc[p] - mx); den += sc[p]; }
                for (unsigned i = 0; i < K2_HD; i++) {
                    double acc = 0.0;
                    for (size_t p = 0; p <= t; p++)
                        acc += sc[p] * vc[((size_t)l * T + p) * K2_KVROWS + kvh * K2_HD + i];
                    att[h * K2_HD + i] = acc / den;
                }
            }
            for (unsigned i = 0; i < K2_QROWS; i++)
                att[i] *= log1p(exp(log(2.0) * g[i])) / log(2.0);
            double o[K2_EMBD];
            ref_mv(ref_getl(rm, l, "attn_output.weight"), K2_EMBD, K2_QROWS, att, o);
            for (unsigned i = 0; i < K2_EMBD; i++) x[i] += o[i];

            ref_gnorm(x, ref_getl(rm, l, "ffn_norm.weight"), xn);
            double f[K2_EMBD] = { 0 };
            if (!moe) {
                double a[K2_FF], b[K2_FF];
                ref_mv(ref_getl(rm, l, "ffn_gate.weight"), K2_FF, K2_EMBD, xn, a);
                ref_mv(ref_getl(rm, l, "ffn_up.weight"), K2_FF, K2_EMBD, xn, b);
                for (unsigned i = 0; i < K2_FF; i++) a[i] = ref_silu(a[i]) * b[i];
                ref_mv(ref_getl(rm, l, "ffn_down.weight"), K2_EMBD, K2_FF, a, f);
            } else {
                double rl[K2_NEXP], wts[K2_NUSED], a[K2_EFF], b[K2_EFF], d[K2_EMBD];
                unsigned sel[K2_NUSED];
                ref_mv(ref_getl(rm, l, "ffn_gate_inp.weight"), K2_NEXP, K2_EMBD, xn, rl);
                ref_route(rl, ref_getl(rm, l, "exp_probs_b.bias"), K2_NEXP,
                          K2_NUSED, sel, wts);
                const float *ge = ref_getl(rm, l, "ffn_gate_exps.weight");
                const float *ue = ref_getl(rm, l, "ffn_up_exps.weight");
                const float *de = ref_getl(rm, l, "ffn_down_exps.weight");
                for (unsigned s = 0; s < K2_NUSED; s++) {
                    const size_t e = sel[s];
                    ref_mv(ge + e * K2_EFF * K2_EMBD, K2_EFF, K2_EMBD, xn, a);
                    ref_mv(ue + e * K2_EFF * K2_EMBD, K2_EFF, K2_EMBD, xn, b);
                    for (unsigned i = 0; i < K2_EFF; i++) a[i] = ref_silu(a[i]) * b[i];
                    ref_mv(de + e * K2_EMBD * K2_EFF, K2_EMBD, K2_EFF, a, d);
                    for (unsigned i = 0; i < K2_EMBD; i++) f[i] += wts[s] * d[i];
                }
                double sa[K2_SHFF], sb[K2_SHFF];
                ref_mv(ref_getl(rm, l, "ffn_gate_shexp.weight"), K2_SHFF, K2_EMBD, xn, sa);
                ref_mv(ref_getl(rm, l, "ffn_up_shexp.weight"), K2_SHFF, K2_EMBD, xn, sb);
                for (unsigned i = 0; i < K2_SHFF; i++) sa[i] = ref_silu(sa[i]) * sb[i];
                ref_mv(ref_getl(rm, l, "ffn_down_shexp.weight"), K2_EMBD, K2_SHFF, sa, d);
                for (unsigned i = 0; i < K2_EMBD; i++) f[i] += d[i];
            }
            for (unsigned i = 0; i < K2_EMBD; i++) x[i] += f[i];
        }
        ref_gnorm(x, ref_get(rm, "output_norm.weight"), xn);
        ref_mv(ref_get(rm, "output.weight"), K2_VOCAB, K2_EMBD, xn,
               logits + t * K2_VOCAB);
    }
    free(kc);
    free(vc);
}

static const uint32_t k_tokens[] = { 3, 17, 5, 22, 0, 9, 11, 3, 14, 7 };
#define K2_T (sizeof k_tokens / sizeof k_tokens[0])

static double max_abs_diff_f(const float *a, const double *b, size_t n)
{
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = fabs((double)a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static double max_abs_diff_ff(const float *a, const float *b, size_t n)
{
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = fabs((double)a[i] - (double)b[i]);
        if (d > m) m = d;
    }
    return m;
}

/* ─── Unit pieces ──────────────────────────────────────────────────────── */

Test(k2_arch, arch_match)
{
    cr_assert(oc_k2_arch_match("k2-horizon"));
    cr_assert(oc_k2_arch_match("k2_horizon"));
    cr_assert_not(oc_k2_arch_match("k2"));
    cr_assert_not(oc_k2_arch_match("llama"));
    cr_assert_not(oc_k2_arch_match(NULL));
}

Test(k2_arch, softplus_ln2_matches_log2_form)
{
    cr_assert_float_eq(oc_k2_softplus_ln2(0.0f), 1.0f, 1e-6f, "gate(0) = 1");
    const float gs[] = { -30.0f, -5.0f, -1.0f, 0.5f, 3.0f, 12.0f, 28.0f, 40.0f };
    for (size_t i = 0; i < sizeof gs / sizeof gs[0]; i++) {
        const double want = log2(1.0 + pow(2.0, (double)gs[i]));
        cr_assert_float_eq((double)oc_k2_softplus_ln2(gs[i]), want,
                           1e-5 * (1.0 + fabs(want)), "g=%f", (double)gs[i]);
    }
    float out[3] = { 2.0f, 2.0f, 2.0f };
    const float gate[3] = { 0.0f, 1.0f, -1.0f };
    oc_k2_softplus_gate_apply(out, gate, 3);
    cr_assert_float_eq(out[0], 2.0f, 1e-6f);
    cr_assert_float_eq(out[1], 2.0f * log2f(3.0f), 1e-5f);
    cr_assert_float_eq(out[2], 2.0f * log2f(1.5f), 1e-5f);
}

Test(k2_arch, grouped_rms_norm_normalizes_each_group)
{
    float x[8] = { 1, 2, 3, 4, 100, 200, 300, 400 };
    float w[8] = { 1, 1, 1, 1, 2, 2, 2, 2 };
    float out[8];
    oc_k2_grouped_rms_norm(x, w, out, 8, 2, 1e-6f);
    /* Each half has rms sqrt(7.5)*scale; the second half is a 100x copy of
     * the first, so after normalization it equals the first times w=2. */
    for (int i = 0; i < 4; i++) {
        const double want = x[i] / sqrt(7.5 + 1e-6);
        cr_assert_float_eq(out[i], want, 1e-5);
        cr_assert_float_eq(out[4 + i], 2.0 * want, 1e-5);
    }
    /* groups = 1 is a plain RMSNorm over all 8. */
    oc_k2_grouped_rms_norm(x, w, out, 8, 1, 1e-6f);
    double ss = 0.0;
    for (int i = 0; i < 8; i++) ss += (double)x[i] * x[i];
    cr_assert_float_eq(out[0], 1.0 / sqrt(ss / 8.0 + 1e-6), 1e-5);
}

Test(k2_arch, route_bias_selects_but_does_not_weight)
{
    /* logits chosen so sigmoid(p) ordering is 0 > 1 > 2 > 3, then a bias
     * promotes expert 3 into the top 2. */
    float probs[4] = { 2.0f, 1.0f, 0.0f, -1.0f };
    const float bias[4] = { 0.0f, 0.0f, 0.0f, 0.9f };
    uint32_t sel[2], idx[4];
    float w[2];
    oc_k2_route(probs, bias, 4, 2, true, 2.5f, sel, w, idx);
    const double p0 = 1.0 / (1.0 + exp(-2.0));
    const double p3 = 1.0 / (1.0 + exp(1.0));
    /* biased scores: 0.88, 0.73, 0.5, 0.27+0.9=1.17 → expert 3 first. */
    cr_assert_eq(sel[0], 3u);
    cr_assert_eq(sel[1], 0u);
    cr_assert_float_eq(w[0], 2.5 * p3 / (p0 + p3), 1e-6);
    cr_assert_float_eq(w[1], 2.5 * p0 / (p0 + p3), 1e-6);
    cr_assert_float_eq(probs[0], p0, 1e-7, "probs hold sigmoid on return");
}

Test(k2_arch, rope_table_matches_double_reference_at_long_positions)
{
    float c[8], s[8];
    const int64_t pos = 262143;
    oc_k2_rope_table(pos, 1.0e7f, 16, c, s);
    for (unsigned j = 0; j < 8; j++) {
        const double a = (double)pos * pow(1.0e7, -2.0 * j / 16.0);
        cr_assert_float_eq(c[j], cos(a), 1e-6, "cos j=%u", j);
        cr_assert_float_eq(s[j], sin(a), 1e-6, "sin j=%u", j);
    }
}

/* ─── Model-level ──────────────────────────────────────────────────────── */

Test(k2_arch, config_and_bindings)
{
    RefModel rm;
    build_k2_gguf(FIXTURE("config"), &rm, 11, false);
    cr_assert(oc_k2_gguf_path_is_k2(FIXTURE("config")));
    OcLlamaModel m;
    cr_assert_eq(oc_llama_load(FIXTURE("config"), &m), OC_OK);
    const OcLlamaConfig *c = &m.cfg;
    cr_assert(c->is_k2);
    cr_assert_eq(c->norm_groups, K2_GROUPS);
    cr_assert_eq(c->value_expert_count, K2_NVEXP);
    cr_assert_eq(c->value_expert_used, K2_NVUSED);
    cr_assert(c->attn_out_gate);
    cr_assert_eq(c->attn_gate_kind, OC_ATTN_GATE_SOFTPLUS_LN2);
    cr_assert(c->expert_gating_sigmoid);
    cr_assert(c->expert_weights_norm);
    cr_assert_float_eq(c->expert_weights_scale, K2_SCALE, 1e-6f);
    cr_assert_eq(c->leading_dense_block_count, K2_DENSE);
    cr_assert_eq(c->head_dim, K2_HD);
    cr_assert_eq(c->rope_dim, K2_HD);
    cr_assert(m.load_flags & OC_LLAMA_LOAD_NORMAL_ADVICE,
              "K2 maps with MADV_NORMAL, not SEQUENTIAL+WILLNEED");

    cr_assert_not_null(m.layers[0].attn_v.data);
    cr_assert_null(m.layers[0].attn_v_exps.data);
    for (unsigned l = K2_DENSE; l < K2_LAYERS; l++) {
        const OcLlamaLayer *L = &m.layers[l];
        cr_assert_null(L->attn_v.data);
        cr_assert_not_null(L->attn_v_exps.data);
        cr_assert_eq(L->attn_v_exps.rows, K2_KVROWS);
        cr_assert_eq(L->attn_v_exps.cols, K2_EMBD);
        cr_assert_eq(L->attn_v_gate.rows, K2_NVEXP);
        cr_assert_not_null(L->attn_v_gate_b);
        cr_assert_float_eq(L->attn_v_gate_b[1],
                           ref_getl(&rm, l, "attn_v_gate.bias")[1], 0.0f);
        cr_assert_not_null(L->ffn_gate_inp.data);
        cr_assert_not_null(L->exp_probs_b);
    }
    oc_llama_free(&m);
    ref_free(&rm);
    remove(FIXTURE("config"));
}

Test(k2_arch, non_k2_file_is_not_peeked_as_k2)
{
    OcGgufWriter w;
    cr_assert_eq(oc_gguf_writer_init(FIXTURE("notk2"), "llama", &w), OC_OK);
    oc_gguf_writer_add_uint32(&w, "llama.block_count", 1);
    cr_assert_eq(oc_gguf_writer_finalize(&w), OC_OK);
    oc_gguf_writer_free(&w);
    cr_assert_not(oc_k2_gguf_path_is_k2(FIXTURE("notk2")));
    cr_assert_not(oc_k2_gguf_path_is_k2("/nonexistent/k2.gguf"));
    remove(FIXTURE("notk2"));
}

Test(k2_arch, decode_matches_naive_reference)
{
    RefModel rm;
    build_k2_gguf(FIXTURE("decode"), &rm, 42, false);
    OcLlamaModel m;
    cr_assert_eq(oc_llama_load(FIXTURE("decode"), &m), OC_OK);
    OcLlamaSession s;
    cr_assert_eq(oc_llama_session_init_kv(&m, &s, OC_KV_F32), OC_OK);

    double *ref = calloc(K2_T * K2_VOCAB, sizeof(double));
    ref_forward(&rm, k_tokens, K2_T, ref);
    float logits[K2_VOCAB];
    for (size_t t = 0; t < K2_T; t++) {
        cr_assert_eq(oc_llama_forward(&s, k_tokens[t], logits), OC_OK);
        const double d = max_abs_diff_f(logits, ref + t * K2_VOCAB, K2_VOCAB);
        cr_assert_lt(d, 1e-4, "decode pos %zu: max |diff| %.3g", t, d);
    }
    free(ref);
    oc_llama_session_free(&s);
    oc_llama_free(&m);
    ref_free(&rm);
    remove(FIXTURE("decode"));
}

Test(k2_arch, prefill_matches_decode_and_reference)
{
    RefModel rm;
    build_k2_gguf(FIXTURE("prefill"), &rm, 7, false);
    OcLlamaModel m;
    cr_assert_eq(oc_llama_load(FIXTURE("prefill"), &m), OC_OK);

    double *ref = calloc(K2_T * K2_VOCAB, sizeof(double));
    ref_forward(&rm, k_tokens, K2_T, ref);

    /* Batched prefill over the first T-1 tokens in chunks of 4 (so a chunk
     * boundary and a partial chunk are both covered), then one decode step
     * that must attend over the prefilled cache. */
    OcLlamaSession s;
    cr_assert_eq(oc_llama_session_init_kv(&m, &s, OC_KV_F32), OC_OK);
    float pl[K2_VOCAB], dl[K2_VOCAB];
    cr_assert_eq(oc_llama_prefill(&s, k_tokens, K2_T - 1, 4, pl), OC_OK);
    double d = max_abs_diff_f(pl, ref + (K2_T - 2) * K2_VOCAB, K2_VOCAB);
    cr_assert_lt(d, 1e-4, "prefill last logits vs ref: %.3g", d);
    cr_assert_eq(oc_llama_forward(&s, k_tokens[K2_T - 1], dl), OC_OK);
    d = max_abs_diff_f(dl, ref + (K2_T - 1) * K2_VOCAB, K2_VOCAB);
    cr_assert_lt(d, 1e-4, "decode after prefill vs ref: %.3g", d);

    /* Same prompt through pure decode. */
    OcLlamaSession s2;
    cr_assert_eq(oc_llama_session_init_kv(&m, &s2, OC_KV_F32), OC_OK);
    float tl[K2_VOCAB];
    for (size_t t = 0; t + 1 < K2_T; t++)
        cr_assert_eq(oc_llama_forward(&s2, k_tokens[t], tl), OC_OK);
    d = max_abs_diff_ff(pl, tl, K2_VOCAB);
    cr_assert_lt(d, 1e-4, "prefill vs decode: %.3g", d);

    oc_llama_session_free(&s2);
    oc_llama_session_free(&s);
    free(ref);
    oc_llama_free(&m);
    ref_free(&rm);
    remove(FIXTURE("prefill"));
}

/* Q8_0 attention and value-expert weights: with the fused integer path
 * disabled the numbers are the dequant reference's; with it enabled the
 * selected value experts run as one fused parallel region, and decode and
 * batched prefill must still agree. */
Test(k2_arch, quantized_value_experts_match_reference_and_prefill)
{
    oc_parallel_set_threads(4);
    RefModel rm;
    build_k2_gguf(FIXTURE("q8"), &rm, 99, true);
    OcLlamaModel m;
    cr_assert_eq(oc_llama_load(FIXTURE("q8"), &m), OC_OK);
    cr_assert_eq(m.layers[1].attn_v_exps.qtype, OC_QUANT_Q8_0);

    double *ref = calloc(K2_T * K2_VOCAB, sizeof(double));
    ref_forward(&rm, k_tokens, K2_T, ref);

    oc_matvec_set_fused(false);
    OcLlamaSession s;
    cr_assert_eq(oc_llama_session_init_kv(&m, &s, OC_KV_F32), OC_OK);
    float logits[K2_VOCAB];
    for (size_t t = 0; t < K2_T; t++) {
        cr_assert_eq(oc_llama_forward(&s, k_tokens[t], logits), OC_OK);
        const double d = max_abs_diff_f(logits, ref + t * K2_VOCAB, K2_VOCAB);
        cr_assert_lt(d, 1e-4, "q8 dequant decode pos %zu: %.3g", t, d);
    }
    oc_llama_session_free(&s);

    oc_matvec_set_fused(true);
    OcLlamaSession a, b;
    cr_assert_eq(oc_llama_session_init_kv(&m, &a, OC_KV_F32), OC_OK);
    cr_assert_eq(oc_llama_session_init_kv(&m, &b, OC_KV_F32), OC_OK);
    float dl[K2_VOCAB], pl[K2_VOCAB];
    for (size_t t = 0; t < K2_T; t++)
        cr_assert_eq(oc_llama_forward(&a, k_tokens[t], dl), OC_OK);
    cr_assert_eq(oc_llama_prefill(&b, k_tokens, K2_T, 0, pl), OC_OK);
    const double d = max_abs_diff_ff(dl, pl, K2_VOCAB);
    cr_assert_lt(d, 1e-4, "fused decode vs fused prefill: %.3g", d);
    /* The int8 activation path stays close to the exact reference. */
    const double dr = max_abs_diff_f(dl, ref + (K2_T - 1) * K2_VOCAB, K2_VOCAB);
    cr_assert_lt(dr, 5e-2, "fused decode vs reference: %.3g", dr);

    oc_llama_session_free(&a);
    oc_llama_session_free(&b);
    free(ref);
    oc_llama_free(&m);
    ref_free(&rm);
    remove(FIXTURE("q8"));
}

/* ─── MTP (nextn) head ─────────────────────────────────────────────────── */

/* Double reference of the head over n rows at positions pos0..pos0+n-1
 * that attend only to each other (causal), i.e. pos0 == 1 on a fresh
 * cache:  x = eh_proj [GRMS(E t, enorm) ; GRMS(h, hnorm)], one dense K2
 * block, s = GRMS(x, shared_head_norm). */
static void ref_mtp_rows(const RefModel *rm, const uint32_t *toks,
                         const double *h, size_t n, int64_t pos0,
                         double *s_out, double *k_out /* [n][KVROWS] */)
{
    const unsigned l = K2_LAYERS;
    const float *emb = ref_get(rm, "token_embd.weight");
    double *kc = calloc(n * K2_KVROWS, sizeof(double));
    double *vc = calloc(n * K2_KVROWS, sizeof(double));
    cr_assert(kc && vc);
    for (size_t t = 0; t < n; t++) {
        double e[K2_EMBD], cat[2 * K2_EMBD], x[K2_EMBD], xn[K2_EMBD];
        for (unsigned i = 0; i < K2_EMBD; i++)
            e[i] = emb[(size_t)toks[t] * K2_EMBD + i];
        ref_gnorm(e, ref_getl(rm, l, "nextn.enorm.weight"), cat);
        ref_gnorm(h + t * K2_EMBD, ref_getl(rm, l, "nextn.hnorm.weight"),
                  cat + K2_EMBD);
        ref_mv(ref_getl(rm, l, "nextn.eh_proj.weight"), K2_EMBD, 2 * K2_EMBD,
               cat, x);
        ref_gnorm(x, ref_getl(rm, l, "attn_norm.weight"), xn);
        double q[K2_QROWS], k[K2_KVROWS], v[K2_KVROWS], g[K2_QROWS];
        ref_mv(ref_getl(rm, l, "attn_q.weight"), K2_QROWS, K2_EMBD, xn, q);
        ref_mv(ref_getl(rm, l, "attn_k.weight"), K2_KVROWS, K2_EMBD, xn, k);
        ref_mv(ref_getl(rm, l, "attn_v.weight"), K2_KVROWS, K2_EMBD, xn, v);
        ref_mv(ref_getl(rm, l, "attn_gate.weight"), K2_QROWS, K2_EMBD, xn, g);
        for (unsigned hh = 0; hh < K2_HEADS; hh++)
            ref_rope(q + hh * K2_HD, pos0 + (int64_t)t);
        for (unsigned hh = 0; hh < K2_KV_HEADS; hh++)
            ref_rope(k + hh * K2_HD, pos0 + (int64_t)t);
        memcpy(kc + t * K2_KVROWS, k, sizeof k);
        memcpy(vc + t * K2_KVROWS, v, sizeof v);
        if (k_out) memcpy(k_out + t * K2_KVROWS, k, sizeof k);
        double att[K2_QROWS];
        const unsigned grp = K2_HEADS / K2_KV_HEADS;
        for (unsigned hh = 0; hh < K2_HEADS; hh++) {
            const unsigned kvh = hh / grp;
            double sc[K2_CTX], mx = -1e300, den = 0.0;
            for (size_t p = 0; p <= t; p++) {
                double d = 0.0;
                for (unsigned i = 0; i < K2_HD; i++)
                    d += q[hh * K2_HD + i] * kc[p * K2_KVROWS + kvh * K2_HD + i];
                sc[p] = d / sqrt((double)K2_HD);
                if (sc[p] > mx) mx = sc[p];
            }
            for (size_t p = 0; p <= t; p++) { sc[p] = exp(sc[p] - mx); den += sc[p]; }
            for (unsigned i = 0; i < K2_HD; i++) {
                double acc = 0.0;
                for (size_t p = 0; p <= t; p++)
                    acc += sc[p] * vc[p * K2_KVROWS + kvh * K2_HD + i];
                att[hh * K2_HD + i] = acc / den;
            }
        }
        for (unsigned i = 0; i < K2_QROWS; i++)
            att[i] *= log1p(exp(log(2.0) * g[i])) / log(2.0);
        double o[K2_EMBD];
        ref_mv(ref_getl(rm, l, "attn_output.weight"), K2_EMBD, K2_QROWS, att, o);
        for (unsigned i = 0; i < K2_EMBD; i++) x[i] += o[i];
        ref_gnorm(x, ref_getl(rm, l, "ffn_norm.weight"), xn);
        double a[K2_FF], b[K2_FF], f[K2_EMBD];
        ref_mv(ref_getl(rm, l, "ffn_gate.weight"), K2_FF, K2_EMBD, xn, a);
        ref_mv(ref_getl(rm, l, "ffn_up.weight"), K2_FF, K2_EMBD, xn, b);
        for (unsigned i = 0; i < K2_FF; i++) a[i] = ref_silu(a[i]) * b[i];
        ref_mv(ref_getl(rm, l, "ffn_down.weight"), K2_EMBD, K2_FF, a, f);
        for (unsigned i = 0; i < K2_EMBD; i++) x[i] += f[i];
        ref_gnorm(x, ref_getl(rm, l, "nextn.shared_head_norm.weight"),
                  s_out + t * K2_EMBD);
    }
    free(kc);
    free(vc);
}

Test(k2_mtp, accept_prefix)
{
    const uint32_t d[4] = { 5, 7, 9, 11 };
    const uint32_t g1[5] = { 5, 7, 9, 11, 2 };
    const uint32_t g2[5] = { 5, 8, 9, 11, 2 };
    const uint32_t g3[5] = { 4, 7, 9, 11, 2 };
    cr_assert_eq(oc_mtp_accept_prefix(d, g1, 4), 4u);
    cr_assert_eq(oc_mtp_accept_prefix(d, g1, 2), 2u);
    cr_assert_eq(oc_mtp_accept_prefix(d, g2, 4), 1u, "stops at first miss");
    cr_assert_eq(oc_mtp_accept_prefix(d, g3, 4), 0u);
    cr_assert_eq(oc_mtp_accept_prefix(d, g1, 0), 0u);
    cr_assert_eq(oc_mtp_accept_prefix(NULL, g1, 3), 0u);
}

Test(k2_mtp, merged_file_binds_head_outside_main_stack)
{
    RefModel rm;
    build_k2_gguf_mtp(FIXTURE("mtp_merged"), &rm, 5, false, 1);
    OcLlamaModel m;
    cr_assert_eq(oc_llama_load(FIXTURE("mtp_merged"), &m), OC_OK);
    cr_assert_eq(m.cfg.n_layer, K2_LAYERS, "blk.N is the head, not a layer");
    cr_assert_eq(m.cfg.nextn_predict_layers, 1u);
    cr_assert(oc_llama_mtp_present(&m));
    cr_assert_eq(m.mtp.eh_proj.cols, 2u * K2_EMBD);
    cr_assert_eq(m.mtp.eh_proj.rows, K2_EMBD);
    cr_assert_not_null(m.mtp.layer.attn_v.data, "head V is plain attn_v");
    cr_assert_not_null(m.mtp.layer.attn_gate.data);
    cr_assert_not_null(m.mtp.shared_head_norm);
    cr_assert_eq(m.mtp.layer.head_dim, K2_HD);
    /* The old Qwen3.5 helpers must refuse the K2 head. */
    OcLlamaSession s;
    cr_assert_eq(oc_llama_session_init_kv(&m, &s, OC_KV_F32), OC_OK);
    uint32_t out[4], n = 0;
    cr_assert_eq(oc_llama_mtp_draft_tokens(&s, 2, out, NULL, &n), OC_ERR_MODEL);
    oc_llama_session_free(&s);
    oc_llama_free(&m);
    ref_free(&rm);
    remove(FIXTURE("mtp_merged"));
}

Test(k2_mtp, sidecar_overlay_binds_and_validates)
{
    RefModel rm, rs, rbad;
    build_k2_gguf(FIXTURE("mtp_base"), &rm, 5, false);
    memset(&rs, 0, sizeof rs);
    memset(&rbad, 0, sizeof rbad);
    build_k2_mtp_sidecar(FIXTURE("mtp_side"), &rs, 77, K2_LAYERS);
    build_k2_mtp_sidecar(FIXTURE("mtp_side_bad"), &rbad, 77, K2_LAYERS - 1);

    OcLlamaModel m;
    cr_assert_eq(oc_llama_load(FIXTURE("mtp_base"), &m), OC_OK);
    cr_assert_not(oc_llama_mtp_present(&m));
    cr_assert_neq(oc_llama_load_mtp_sidecar(&m, FIXTURE("mtp_side_bad")), OC_OK,
                  "base_block_count mismatch must be refused");
    cr_assert_not(oc_llama_mtp_present(&m));
    cr_assert_neq(oc_llama_load_mtp_sidecar(&m, "/nonexistent/mtp.gguf"), OC_OK);
    cr_assert_eq(oc_llama_load_mtp_sidecar(&m, FIXTURE("mtp_side")), OC_OK);
    cr_assert(oc_llama_mtp_present(&m));
    cr_assert_eq(m.cfg.n_layer, K2_LAYERS);
    cr_assert_neq(oc_llama_load_mtp_sidecar(&m, FIXTURE("mtp_side")), OC_OK,
                  "a second head is refused");
    /* The head's KV layer is reserved by sessions created afterwards. */
    OcLlamaSession s;
    cr_assert_eq(oc_llama_session_init_kv(&m, &s, OC_KV_F32), OC_OK);
    cr_assert_eq(oc_llama_mtp_enable(&s, true), OC_OK);
    oc_llama_session_free(&s);
    oc_llama_free(&m);

    /* Loading a sidecar while a session is alive is refused. */
    cr_assert_eq(oc_llama_load(FIXTURE("mtp_base"), &m), OC_OK);
    cr_assert_eq(oc_llama_session_init_kv(&m, &s, OC_KV_F32), OC_OK);
    cr_assert_eq(oc_llama_load_mtp_sidecar(&m, FIXTURE("mtp_side")),
                 OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_llama_mtp_enable(&s, true), OC_ERR_MODEL, "no head");
    oc_llama_session_free(&s);
    oc_llama_free(&m);
    ref_free(&rm);
    ref_free(&rs);
    ref_free(&rbad);
    remove(FIXTURE("mtp_base"));
    remove(FIXTURE("mtp_side"));
    remove(FIXTURE("mtp_side_bad"));
}

Test(k2_mtp, head_rows_match_reference)
{
    RefModel rm;
    build_k2_gguf_mtp(FIXTURE("mtp_ref"), &rm, 21, false, 1);
    OcLlamaModel m;
    cr_assert_eq(oc_llama_load(FIXTURE("mtp_ref"), &m), OC_OK);
    OcLlamaSession s;
    cr_assert_eq(oc_llama_session_init_kv(&m, &s, OC_KV_F32), OC_OK);
    cr_assert_eq(oc_llama_mtp_enable(&s, true), OC_OK);

    enum { N = 7 };   /* > the step buffer's 5 rows: exercises splitting */
    const uint32_t toks[N] = { 4, 19, 0, 7, 7, 13, 22 };
    double hd[N * K2_EMBD], sref[N * K2_EMBD], kref[N * K2_KVROWS];
    float hf[N * K2_EMBD], sout[N * K2_EMBD];
    g_rng = 1234;
    for (size_t i = 0; i < N * K2_EMBD; i++) {
        hf[i] = 1.5f * rnd();
        hd[i] = hf[i];
    }
    ref_mtp_rows(&rm, toks, hd, N, 1, sref, kref);
    cr_assert_eq(oc_llama_mtp_forward_rows(&s, toks, hf, N, 1, sout), OC_OK);
    double d = max_abs_diff_f(sout, sref, N * K2_EMBD);
    cr_assert_lt(d, 1e-4, "head rows vs reference: %.3g", d);
    float kr[K2_KVROWS], vr[K2_KVROWS];
    for (size_t t = 0; t < N; t++) {
        cr_assert_eq(oc_llama_mtp_kv_row(&s, (int64_t)t + 1, kr, vr), OC_OK);
        d = max_abs_diff_f(kr, kref + t * K2_KVROWS, K2_KVROWS);
        cr_assert_lt(d, 1e-4, "head K at pos %zu: %.3g", t + 1, d);
    }
    cr_assert_eq(oc_llama_mtp_forward_rows(&s, toks, hf, 1, 0, sout),
                 OC_ERR_INVALID_ARG, "no MTP entry at position 0");
    oc_llama_session_free(&s);
    oc_llama_free(&m);
    ref_free(&rm);
    remove(FIXTURE("mtp_ref"));
}

/* Prefill (chunks of 4, so entries straddle chunk boundaries) must write
 * head slot p from (E[t_p], h_{p-1}) with h the post-output_norm state. */
Test(k2_mtp, prefill_fills_head_kv_with_shifted_pairs)
{
    RefModel rm;
    build_k2_gguf_mtp(FIXTURE("mtp_pf"), &rm, 8, false, 1);
    OcLlamaModel m;
    cr_assert_eq(oc_llama_load(FIXTURE("mtp_pf"), &m), OC_OK);

    OcLlamaSession a, b, c;
    cr_assert_eq(oc_llama_session_init_kv(&m, &a, OC_KV_F32), OC_OK);
    cr_assert_eq(oc_llama_session_init_kv(&m, &b, OC_KV_F32), OC_OK);
    cr_assert_eq(oc_llama_session_init_kv(&m, &c, OC_KV_F32), OC_OK);
    cr_assert_eq(oc_llama_mtp_enable(&a, true), OC_OK);
    cr_assert_eq(oc_llama_mtp_enable(&c, true), OC_OK);
    float lg[K2_VOCAB];
    cr_assert_eq(oc_llama_prefill(&a, k_tokens, K2_T, 4, lg), OC_OK);

    float h[K2_T * K2_EMBD];
    for (size_t t = 0; t < K2_T; t++) {
        cr_assert_eq(oc_llama_forward(&b, k_tokens[t], lg), OC_OK);
        oc_k2_grouped_rms_norm(b.last_hidden, m.final_norm, h + t * K2_EMBD,
                               K2_EMBD, K2_GROUPS, K2_EPS);
    }
    float sc[K2_T * K2_EMBD];
    cr_assert_eq(oc_llama_mtp_forward_rows(&c, k_tokens + 1, h, K2_T - 1, 1, sc),
                 OC_OK);
    float ka[K2_KVROWS], va[K2_KVROWS], kc[K2_KVROWS], vc[K2_KVROWS];
    for (int64_t p = 1; p < (int64_t)K2_T; p++) {
        cr_assert_eq(oc_llama_mtp_kv_row(&a, p, ka, va), OC_OK);
        cr_assert_eq(oc_llama_mtp_kv_row(&c, p, kc, vc), OC_OK);
        cr_assert_lt(max_abs_diff_ff(ka, kc, K2_KVROWS), 1e-4, "K at %lld", (long long)p);
        cr_assert_lt(max_abs_diff_ff(va, vc, K2_KVROWS), 1e-4, "V at %lld", (long long)p);
    }
    cr_assert_lt(max_abs_diff_ff(a.mtp_hidden, h + (K2_T - 1) * K2_EMBD, K2_EMBD),
                 1e-4, "carried h_{pos-1}");
    oc_llama_session_free(&a);
    oc_llama_session_free(&b);
    oc_llama_session_free(&c);
    oc_llama_free(&m);
    ref_free(&rm);
    remove(FIXTURE("mtp_pf"));
}

#define GEN_N 24u

static void plain_greedy(OcLlamaModel *m, uint32_t *out)
{
    OcLlamaSession s;
    cr_assert_eq(oc_llama_session_init_kv(m, &s, OC_KV_F32), OC_OK);
    float lg[K2_VOCAB];
    cr_assert_eq(oc_llama_prefill(&s, k_tokens, K2_T, 0, lg), OC_OK);
    for (size_t i = 0; i < GEN_N; i++) {
        out[i] = oc_argmax(lg, K2_VOCAB);
        cr_assert_eq(oc_llama_forward(&s, out[i], lg), OC_OK);
    }
    oc_llama_session_free(&s);
}

typedef struct {
    const uint32_t *truth;  /* plain greedy continuation                 */
    int64_t base;           /* position of truth[0]                      */
    int64_t wrong_at;       /* position whose draft is forced wrong (-1) */
} Oracle;

static uint32_t oracle_hook(void *ud, int64_t pos, uint32_t proposed)
{
    const Oracle *o = (const Oracle *)ud;
    (void)proposed;
    const int64_t i = pos - o->base;
    if (i < 0 || i >= (int64_t)GEN_N) return 0;
    if (pos == o->wrong_at) return (o->truth[i] + 1u) % K2_VOCAB;
    return o->truth[i];
}

/* Run MTP greedy with k drafts per step; optionally an oracle hook and a
 * KV cache type (NULL: f32). With keep != NULL the session is handed back
 * for inspection instead of freed. */
static void mtp_greedy_kv(OcLlamaModel *m, uint32_t k, const Oracle *hook,
                          uint32_t *out, OcMtpStats *st,
                          const OcKvOptions *kvo, OcLlamaSession *keep)
{
    OcLlamaSession s;
    if (kvo != NULL) {
        cr_assert_eq(oc_llama_session_init_kv_opts(m, &s, kvo), OC_OK);
        cr_assert_eq(s.kv_type, kvo->type);
    } else {
        cr_assert_eq(oc_llama_session_init_kv(m, &s, OC_KV_F32), OC_OK);
    }
    cr_assert_eq(oc_llama_mtp_enable(&s, true), OC_OK);
    if (hook) oc_llama_mtp_test_set_draft_hook(&s, oracle_hook, (void *)hook);
    float lg[K2_VOCAB];
    cr_assert_eq(oc_llama_prefill(&s, k_tokens, K2_T, 0, lg), OC_OK);
    memset(st, 0, sizeof *st);
    size_t n = 0;
    while (n < GEN_N) {
        size_t got = 0;
        const int64_t p0 = s.pos;
        cr_assert_eq(oc_llama_mtp_step(&s, k, lg, out + n, GEN_N - n, &got, st),
                     OC_OK);
        cr_assert_gt(got, 0u);
        cr_assert_leq(got, (size_t)k + 1);
        cr_assert_eq(s.pos, p0 + (int64_t)got, "rollback to P+1+accepted");
        n += got;
    }
    cr_assert_eq(n, GEN_N);
    if (keep != NULL) *keep = s;
    else oc_llama_session_free(&s);
}

static void mtp_greedy(OcLlamaModel *m, uint32_t k, const Oracle *hook,
                       uint32_t *out, OcMtpStats *st)
{
    mtp_greedy_kv(m, k, hook, out, st, NULL, NULL);
}

Test(k2_mtp, greedy_output_equals_plain_decode)
{
    RefModel rm;
    build_k2_gguf_mtp(FIXTURE("mtp_gen"), &rm, 31, false, 1);
    OcLlamaModel m;
    cr_assert_eq(oc_llama_load(FIXTURE("mtp_gen"), &m), OC_OK);
    uint32_t plain[GEN_N], got[GEN_N];
    plain_greedy(&m, plain);
    OcMtpStats st;
    for (uint32_t k = 0; k <= 3; k++) {
        mtp_greedy(&m, k, NULL, got, &st);
        for (size_t i = 0; i < GEN_N; i++)
            cr_assert_eq(got[i], plain[i], "k=%u token %zu: %u vs %u", k, i,
                         got[i], plain[i]);
        cr_assert_eq(st.emitted, GEN_N);
    }
    oc_llama_free(&m);
    ref_free(&rm);
    remove(FIXTURE("mtp_gen"));
}

/* Oracle drafts force full acceptance, then a rejection in the middle of a
 * chain; the rolled-back caches (main and head) must still reproduce plain
 * greedy exactly, and the counters must see it. */
Test(k2_mtp, oracle_acceptance_and_rollback)
{
    RefModel rm;
    build_k2_gguf_mtp(FIXTURE("mtp_orc"), &rm, 31, false, 1);
    OcLlamaModel m;
    cr_assert_eq(oc_llama_load(FIXTURE("mtp_orc"), &m), OC_OK);
    uint32_t plain[GEN_N], got[GEN_N];
    plain_greedy(&m, plain);

    Oracle o = { plain, (int64_t)K2_T, -1 };
    OcMtpStats st;
    mtp_greedy(&m, 2, &o, got, &st);
    for (size_t i = 0; i < GEN_N; i++) cr_assert_eq(got[i], plain[i]);
    cr_assert_eq(st.steps, GEN_N / 3u, "every step emits 1 + 2 accepted");
    cr_assert_eq(st.accepted, st.drafted);

    o.wrong_at = (int64_t)K2_T + 5;   /* reject once, mid-chain */
    mtp_greedy(&m, 3, &o, got, &st);
    for (size_t i = 0; i < GEN_N; i++) cr_assert_eq(got[i], plain[i], "tok %zu", i);
    cr_assert_lt(st.accepted, st.drafted);
    cr_assert_gt(st.accepted, 0u);
    oc_llama_free(&m);
    ref_free(&rm);
    remove(FIXTURE("mtp_orc"));
}

Test(k2_mtp, step_refuses_desynced_session)
{
    RefModel rm;
    build_k2_gguf_mtp(FIXTURE("mtp_sync"), &rm, 3, false, 1);
    OcLlamaModel m;
    cr_assert_eq(oc_llama_load(FIXTURE("mtp_sync"), &m), OC_OK);
    OcLlamaSession s;
    cr_assert_eq(oc_llama_session_init_kv(&m, &s, OC_KV_F32), OC_OK);
    float lg[K2_VOCAB];
    uint32_t out[4];
    size_t n = 0;
    cr_assert_eq(oc_llama_mtp_step(&s, 1, lg, out, 4, &n, NULL), OC_ERR_MODEL,
                 "not enabled");
    cr_assert_eq(oc_llama_mtp_enable(&s, true), OC_OK);
    cr_assert_eq(oc_llama_prefill(&s, k_tokens, 4, 0, lg), OC_OK);
    cr_assert_eq(oc_llama_mtp_step(&s, 1, lg, out, 4, &n, NULL), OC_OK);
    /* A plain forward skips the head's catch-up: refuse, don't draft junk. */
    cr_assert_eq(oc_llama_forward(&s, out[0], lg), OC_OK);
    cr_assert_eq(oc_llama_mtp_step(&s, 1, lg, out, 4, &n, NULL),
                 OC_ERR_INVALID_ARG);
    oc_llama_session_rewind(&s, 2);
    cr_assert_eq(oc_llama_mtp_step(&s, 1, lg, out, 4, &n, NULL),
                 OC_ERR_INVALID_ARG);
    /* A reset starts over cleanly. */
    oc_llama_session_reset(&s);
    cr_assert_eq(oc_llama_prefill(&s, k_tokens, 4, 0, lg), OC_OK);
    cr_assert_eq(oc_llama_mtp_step(&s, 2, lg, out, 4, &n, NULL), OC_OK);
    oc_llama_session_free(&s);
    oc_llama_free(&m);
    ref_free(&rm);
    remove(FIXTURE("mtp_sync"));
}

/* MTP on the RotorQuant cache with a tiny exact ring (window 8 = centering
 * page): verify rows overwrite ring slots of older positions and complete
 * pages, and rejected rows must be undone. Every k (and forced rejections)
 * gives the k=0 output (verify rows attend like single decode steps: the
 * ring headroom keeps every row's exact window), and afterwards each ring
 * slot holds the newest accepted position with the k=0 run's values. */
Test(k2_mtp, rq_kv_rollback_matches_k0)
{
    RefModel rm;
    build_k2_gguf_mtp(FIXTURE("mtp_rq"), &rm, 31, false, 1);
    OcLlamaModel m;
    cr_assert_eq(oc_llama_load(FIXTURE("mtp_rq"), &m), OC_OK);
    OcKvOptions kvo = { .type = OC_KV_RQ, .rq_k_bits = 4, .rq_v_bits = 4,
                        .rq_sinks = 2, .rq_window = 8 };
    uint32_t ref[GEN_N], got[GEN_N];
    OcMtpStats st;
    OcLlamaSession s0;
    mtp_greedy_kv(&m, 0, NULL, ref, &st, &kvo, &s0);
    cr_assert_eq(s0.kv_type, OC_KV_RQ);
    Oracle o = { ref, (int64_t)K2_T, (int64_t)K2_T + 6 };
    for (int run = 0; run < 4; run++) {
        const uint32_t k = run < 3 ? (uint32_t)run + 1 : 3;
        OcLlamaSession s;
        mtp_greedy_kv(&m, k, run == 3 ? &o : NULL, got, &st, &kvo, &s);
        for (size_t i = 0; i < GEN_N; i++)
            cr_assert_eq(got[i], ref[i], "run %d token %zu", run, i);
        if (run == 3) {
            cr_assert_lt(st.accepted, st.drafted, "a rejection happened");
            cr_assert_gt(st.accepted, 0u);
        }
        cr_assert_eq(s.pos, s0.pos);
        const OcKvRqCache *a = s0.kv_rq, *b = s.kv_rq;
        for (size_t l = 0; l < K2_LAYERS; l++) {
            for (int64_t t = 0; t < s.pos; t++) {
                const int64_t sa = oc_kvrq_slot(a, l, t), sb = oc_kvrq_slot(b, l, t);
                /* ring = window 8 + OC_KV_RQ_RING_EXTRA 8 */
                const bool newest = t < 2 || t >= s.pos - 16;
                cr_assert_eq(sb >= 0, newest, "run %d layer %zu pos %lld exact",
                             run, l, (long long)t);
                cr_assert_eq(sa, sb);
                if (sb < 0) continue;
                for (size_t kind = 0; kind < 2; kind++)
                    for (size_t h = 0; h < K2_KV_HEADS; h++)
                        for (size_t i = 0; i < K2_HD; i++) {
                            const float va = oc_kvrq_xq(a, l, kind, h)[sa * K2_HD + i] *
                                             oc_kvrq_xs(a, l, kind, h)[sa];
                            const float vb = oc_kvrq_xq(b, l, kind, h)[sb * K2_HD + i] *
                                             oc_kvrq_xs(b, l, kind, h)[sb];
                            /* batched vs single-row matmuls: at most one
                             * int8 step apart */
                            cr_assert(fabsf(va - vb) <=
                                      1e-5f + 1.01f * oc_kvrq_xs(a, l, kind, h)[sa],
                                      "run %d layer %zu pos %lld: %g vs %g", run, l,
                                      (long long)t, va, vb);
                        }
            }
        }
        /* Page means (centering) of every complete page, main layers and
         * the head's layer K2_LAYERS: chained draft rows and rejected rows
         * must not leave their values in a fixed mean. */
        for (size_t l = 0; l <= K2_LAYERS; l++) {
            for (size_t pg = 0; (int64_t)((pg + 1) * b->page) <= s.pos; pg++) {
                cr_assert_eq(a->mu_fixed[l * a->n_pages + pg],
                             b->mu_fixed[l * b->n_pages + pg],
                             "run %d layer %zu page %zu fixed", run, l, pg);
                if (!a->mu_fixed[l * a->n_pages + pg]) continue;
                for (size_t kind = 0; kind < 2; kind++)
                    for (size_t h = 0; h < K2_KV_HEADS; h++) {
                        const float *ma = oc_kvrq_mu(a, l, pg, kind, h);
                        const float *mb = oc_kvrq_mu(b, l, pg, kind, h);
                        double num = 0, den = 0;
                        for (size_t i = 0; i < K2_HD; i++) {
                            num += ((double)ma[i] - mb[i]) * ((double)ma[i] - mb[i]);
                            den += (double)ma[i] * ma[i];
                        }
                        cr_assert(num <= 1e-4 * den + 1e-12,
                                  "run %d layer %zu page %zu kind %zu head %zu: "
                                  "mean rel err %g", run, l, pg, kind, h,
                                  sqrt(num / (den + 1e-30)));
                    }
            }
        }
        oc_llama_session_free(&s);
    }
    oc_llama_session_free(&s0);
    oc_llama_free(&m);
    ref_free(&rm);
    remove(FIXTURE("mtp_rq"));
}
