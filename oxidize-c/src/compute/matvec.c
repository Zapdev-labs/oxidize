/*
 * matvec.c — quantized/f32 matrix-vector products (scalar reference).
 *
 * Port of oxidize-core/src/compute/tensor/kernels/gemv.rs
 * (`gemv_f32`, `gemv_dequant_scalar_fallback`). The dequant step reuses
 * the SIMD-accelerated `oc_quant_dequant_row`, so this path inherits the
 * AVX2/AVX-512 speedups on capable hosts.
 */
#include "oxidize/matvec.h"
#include "oxidize/flash_attention.h"
#include "oxidize/oxk.h"
#include "oxidize/parallel.h"
#include "oxidize/quant.h"

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

/* ─── Fused integer GEMV ─────────────────────────────────────────────────
 *
 * The old path dequantized each weight row to f32 and did an f32 dot. For
 * Q4_K that expands 144 bytes into 1 KB per 256 weights, so a 4096-wide row
 * writes and re-reads 16 KB of scratch to consume 2.25 KB of weights — the
 * kernel ends up bound on scratch traffic rather than on the weights.
 *
 * The OXK kernels instead quantize the ACTIVATION to Q8 once per matvec and
 * take integer dot products straight against the packed weights, which is
 * what llama.cpp does and roughly what the measured 25x per-core gap was.
 * Those kernels already existed here (including the AVX2/AVX-512/NEON
 * variants) but nothing outside a synthetic benchmark ever called them.
 *
 * This changes results: the activation carries int8 quantization error it did
 * not before. That is the standard trade — Q8 activations are ~lossless in
 * practice and it is how every fast CPU inference engine works — but it is a
 * real numerical change, so oc_matvec_set_fused() can turn it off, and the
 * dequant path stays as the reference the parity tests compare against.
 */

static bool g_fused_enabled = true;
static bool g_fused_env_checked = false;

void oc_matvec_set_fused(bool enabled)
{
    g_fused_enabled = enabled;
    g_fused_env_checked = true;   /* explicit call wins over the environment */
}

bool oc_matvec_fused_enabled(void)
{
    /* OC_NO_FUSED=1 forces the dequant reference path. This exists so the
     * fused and reference paths can be compared in one binary — an A/B on a
     * real model is the only way to tell a kernel that is exact on synthetic
     * dot products from one that is wrong on real weights. */
    if (!g_fused_env_checked) {
        const char *e = getenv("OC_NO_FUSED");
        if (e != NULL && e[0] != '\0' && e[0] != '0') g_fused_enabled = false;
        g_fused_env_checked = true;
    }
    return g_fused_enabled;
}

/* Which Q8 flavour a weight type pairs with. */
typedef enum { ACT_NONE = 0, ACT_Q8_0, ACT_Q8_K } ActKind;

/* Every OXK kernel here is verified against the GGUF layout by
 * test_oxk_gguf_layout.c, which dots the dequantized weights with the same
 * dequantized activation the kernel consumes — that removes int8 error from
 * the comparison, so any difference is a real disagreement.
 *
 * Q5_K is absent only because oc_quant_pack_row cannot produce it, so there
 * is nothing to verify it against yet; its kernel shares the scale decoding
 * that Q4_K needed fixed, so it is likely fine but stays off until checked. */
static ActKind fused_act_kind(OcGgufQuantizationType qtype, size_t cols)
{
    switch (qtype) {
    case OC_QUANT_Q4_0:
    case OC_QUANT_Q4_1:
    case OC_QUANT_Q8_0:
        return (cols % OC_OXK_QK8_0 == 0) ? ACT_Q8_0 : ACT_NONE;
    case OC_QUANT_Q4_K_S:
    case OC_QUANT_Q4_K_M:
    case OC_QUANT_Q6_K:
        return (cols % OC_OXK_QK_K == 0) ? ACT_Q8_K : ACT_NONE;
    default:
        return ACT_NONE;
    }
}

/* Quantize an f32 activation to Q8_0: per 32 values, [f16 d][32 int8]. */
static void quantize_act_q8_0(const float *x, size_t n, uint8_t *out)
{
    const size_t nb = n / OC_OXK_QK8_0;
    for (size_t b = 0; b < nb; b++) {
        const float *src = x + b * OC_OXK_QK8_0;
        float amax = 0.0f;
        for (size_t i = 0; i < OC_OXK_QK8_0; i++) {
            const float a = fabsf(src[i]);
            if (a > amax) amax = a;
        }
        const float d  = amax / 127.0f;
        const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
        uint8_t *dst = out + b * OC_OXK_BLOCK_Q8_0_SIZE;
        /* Stored little-endian to match oc_oxk_f16_le_to_f32 on the read side. */
        const uint16_t dh = oc_f32_to_f16_bits(d);
        dst[0] = (uint8_t)(dh & 0xFF);
        dst[1] = (uint8_t)(dh >> 8);
        int8_t *q = (int8_t *)(dst + 2);
        for (size_t i = 0; i < OC_OXK_QK8_0; i++) {
            int v = (int)lrintf(src[i] * id);
            if (v >  127) v =  127;
            if (v < -128) v = -128;
            q[i] = (int8_t)v;
        }
    }
}

/* Quantize an f32 activation to Q8_K: per 256 values,
 * [f32 d][256 int8][16 int16 bsums], bsums[i] summing q[i*16 .. i*16+15].
 * The K-quant kernels fold the block minimum out using those partial sums,
 * so they must match the stored q exactly or the correction term is wrong. */
static void quantize_act_q8_k(const float *x, size_t n, uint8_t *out)
{
    const size_t nb = n / OC_OXK_QK_K;
    for (size_t b = 0; b < nb; b++) {
        const float *src = x + b * OC_OXK_QK_K;
        float amax = 0.0f;
        for (size_t i = 0; i < OC_OXK_QK_K; i++) {
            const float a = fabsf(src[i]);
            if (a > amax) amax = a;
        }
        const float d  = amax / 127.0f;
        const float id = (d != 0.0f) ? 1.0f / d : 0.0f;

        uint8_t *dst = out + b * OC_OXK_BLOCK_Q8_K_SIZE;
        memcpy(dst, &d, 4);
        int8_t *q = (int8_t *)(dst + 4);
        for (size_t i = 0; i < OC_OXK_QK_K; i++) {
            int v = (int)lrintf(src[i] * id);
            if (v >  127) v =  127;
            if (v < -128) v = -128;
            q[i] = (int8_t)v;
        }
        uint8_t *bsums = dst + 4 + OC_OXK_QK_K;
        for (size_t g = 0; g < OC_OXK_QK_K / 16u; g++) {
            int32_t s = 0;
            for (size_t i = 0; i < 16; i++) s += q[g * 16 + i];
            int16_t s16 = (int16_t)s;   /* |s| <= 16*127 = 2032, always fits */
            memcpy(bsums + g * 2, &s16, 2);
        }
    }
}

static float fused_row_dot(OcGgufQuantizationType qtype, const uint8_t *row,
                           size_t blocks, const uint8_t *act)
{
    switch (qtype) {
    case OC_QUANT_Q4_0:   return oc_oxk_dot_q4_0_q8_0(row, blocks, act);
    case OC_QUANT_Q4_1:   return oc_oxk_dot_q4_1_q8_0(row, blocks, act);
    case OC_QUANT_Q8_0:   return oc_oxk_dot_q8_0_q8_0(row, blocks, act);
    case OC_QUANT_Q4_K_S:
    case OC_QUANT_Q4_K_M: return oc_oxk_dot_q4_k_q8_k(row, blocks, act);
    case OC_QUANT_Q5_K_S:
    case OC_QUANT_Q5_K_M: return oc_oxk_dot_q5_k_q8_k(row, blocks, act);
    case OC_QUANT_Q6_K:   return oc_oxk_dot_q6_k_q8_k(row, blocks, act);
    default:              return 0.0f;
    }
}

/* These matvecs are the bulk of a forward pass, and each output row is an
 * independent dot product, so the row loop is split across the worker pool.
 *
 * Splitting by row does NOT change the result: every output[j] is still
 * accumulated by one thread in ascending i, exactly as the serial loop did.
 * The output is therefore bit-identical at any thread count, which is what
 * lets the parity tests stay exact. Reductions (a shared accumulator, or
 * splitting one row across threads) would not have that property and are
 * deliberately avoided. */

typedef struct {
    const float *data;
    size_t       cols;
    const float *input;
    float       *output;
} F32Job;

static void matvec_f32_slice(size_t begin, size_t end, size_t tid, void *ud)
{
    (void)tid;
    const F32Job *j = (const F32Job *)ud;
    for (size_t r = begin; r < end; r++) {
        const float *row = j->data + r * j->cols;
        float acc = 0.0f;
        for (size_t i = 0; i < j->cols; i++) {
            acc += row[i] * j->input[i];
        }
        j->output[r] = acc;
    }
}

void oc_matvec_f32(const float *data, size_t rows, size_t cols,
                   const float *input, float *output)
{
    F32Job job = { data, cols, input, output };
    oc_parallel_for(rows, matvec_f32_slice, &job);
}

typedef struct {
    OcGgufQuantizationType qtype;
    const uint8_t *data;
    size_t         cols;
    size_t         row_bytes;
    const float   *input;
    float         *output;
    float         *temp;      /* caller's buffer; used by tid 0 only */
    /* Fused path only: the activation quantized once for the whole matvec,
     * shared read-only by every thread. NULL selects the dequant path. */
    const uint8_t *act;
    size_t         blocks;
} QuantJob;

static void matvec_fused_slice(size_t begin, size_t end, size_t tid, void *ud)
{
    (void)tid;
    const QuantJob *j = (const QuantJob *)ud;
    for (size_t r = begin; r < end; r++) {
        j->output[r] = fused_row_dot(j->qtype, j->data + r * j->row_bytes,
                                     j->blocks, j->act);
    }
}

static void matvec_quant_slice(size_t begin, size_t end, size_t tid, void *ud)
{
    const QuantJob *j = (const QuantJob *)ud;

    /* The caller supplies one dequantization buffer, which cannot be shared
     * once rows are split. Thread 0 keeps using it (so the single-threaded
     * path allocates nothing); the others take per-thread scratch. */
    float *temp = j->temp;
    if (tid != 0) {
        temp = (float *)oc_parallel_scratch(tid, j->cols * sizeof(float));
        if (temp == NULL) return;   /* see the OOM note in the caller */
    }

    for (size_t r = begin; r < end; r++) {
        const uint8_t *row = j->data + r * j->row_bytes;
        /* Dequantize this weight row into `temp` (SIMD on capable hosts). */
        oc_quant_dequant_row(j->qtype, row, j->row_bytes, temp, j->cols);
        float acc = 0.0f;
        for (size_t i = 0; i < j->cols; i++) {
            acc += temp[i] * j->input[i];
        }
        j->output[r] = acc;
    }
}

void oc_matvec_quantized(OcGgufQuantizationType qtype, const uint8_t *data,
                         size_t rows, size_t cols, size_t row_bytes,
                         const float *input, float *output, float *temp)
{
    /* Fused integer path, when the weight type has an OXK kernel and the row
     * divides evenly into blocks. The activation is quantized once here and
     * then shared by every row and every thread. */
    ActKind act_kind = oc_matvec_fused_enabled()
                     ? fused_act_kind(qtype, cols) : ACT_NONE;
    if (act_kind != ACT_NONE) {
        const size_t blocks = (act_kind == ACT_Q8_K)
                            ? cols / OC_OXK_QK_K : cols / OC_OXK_QK8_0;
        const size_t act_bytes = blocks * ((act_kind == ACT_Q8_K)
                            ? OC_OXK_BLOCK_Q8_K_SIZE : OC_OXK_BLOCK_Q8_0_SIZE);
        /* Only trust the fused kernel when the weight stride is exactly what
         * it assumes; a padded or interleaved row would be read wrong. */
        const size_t expect = blocks * ((qtype == OC_QUANT_Q4_0) ? OC_OXK_BLOCK_Q4_0_SIZE
                            : (qtype == OC_QUANT_Q4_1) ? OC_OXK_BLOCK_Q4_1_SIZE
                            : (qtype == OC_QUANT_Q8_0) ? OC_OXK_BLOCK_Q8_0_SIZE
                            : (qtype == OC_QUANT_Q5_K_S ||
                               qtype == OC_QUANT_Q5_K_M) ? OC_OXK_BLOCK_Q5_K_SIZE
                            : (qtype == OC_QUANT_Q6_K) ? OC_OXK_BLOCK_Q6_K_SIZE
                                                       : OC_OXK_BLOCK_Q4_K_SIZE);
        if (row_bytes == expect) {
            /* Thread 0's scratch holds the activation; it is written before
             * the region opens and only read inside it. */
            uint8_t *act = (uint8_t *)oc_parallel_scratch(0, act_bytes);
            if (act != NULL) {
                if (act_kind == ACT_Q8_K) quantize_act_q8_k(input, cols, act);
                else                      quantize_act_q8_0(input, cols, act);
                QuantJob job = { qtype, data, cols, row_bytes, input, output,
                                 temp, act, blocks };
                oc_parallel_for(rows, matvec_fused_slice, &job);
                return;
            }
        }
    }

    /* A scratch allocation failure inside a slice would silently leave that
     * slice's outputs untouched, so pre-zero and pre-reserve: after this,
     * every worker's buffer is already large enough and the slice cannot
     * fail. The function returns void and has no way to report OOM. */
    const size_t nt = oc_parallel_n_threads();
    if (nt > 1 && rows >= 8) {
        for (size_t t = 1; t < nt; t++) {
            if (oc_parallel_scratch(t, cols * sizeof(float)) == NULL) {
                /* Fall back to the serial path rather than produce a
                 * partially-written output vector. */
                QuantJob job = { qtype, data, cols, row_bytes, input,
                                 output, temp, NULL, 0 };
                matvec_quant_slice(0, rows, 0, &job);
                return;
            }
        }
    }
    QuantJob job = { qtype, data, cols, row_bytes, input, output, temp,
                     NULL, 0 };
    oc_parallel_for(rows, matvec_quant_slice, &job);
}

/* ─── Batched matvec ─────────────────────────────────────────────────────
 *
 * Prefill reads the same weight matrix once per prompt token. At 18 GB of
 * weights and ~1.5 GB touched per token on a 30B MoE, a 500-token prompt
 * moves ~750 GB through DRAM — which is why per-token prefill measured 2.9
 * tok/s against llama.cpp's 129. Dotting each weight row against a TILE of
 * activations while it is still in L1 cuts that by the tile width.
 *
 * Why tile rather than loop over all n_vec inside the row loop: the row is
 * ~1 KB but n_vec activations are ~2.3 KB each, so an untiled inner loop
 * streams 1.2 MB of activations per row and simply swaps which operand is
 * DRAM-bound. Tiling to ACT_TILE_BYTES keeps the activation tile in L2 and
 * leaves the weights as the only streamed operand.
 */

/* Target footprint for one activation tile. Sized to sit comfortably inside
 * a typical 1 MB private L2 alongside the streaming weight rows. */
#define OC_ACT_TILE_BYTES (256u * 1024u)

static size_t act_block_bytes(ActKind kind)
{
    return (kind == ACT_Q8_K) ? OC_OXK_BLOCK_Q8_K_SIZE : OC_OXK_BLOCK_Q8_0_SIZE;
}

static size_t act_blocks_for(ActKind kind, size_t cols)
{
    return (kind == ACT_Q8_K) ? cols / OC_OXK_QK_K : cols / OC_OXK_QK8_0;
}

/* The stride check from oc_matvec_quantized(): only trust a fused kernel when
 * the weight row stride is exactly the packed size it assumes. */
static bool fused_stride_ok(OcGgufQuantizationType qtype, size_t blocks,
                            size_t row_bytes)
{
    const size_t expect = blocks * ((qtype == OC_QUANT_Q4_0) ? OC_OXK_BLOCK_Q4_0_SIZE
                        : (qtype == OC_QUANT_Q4_1) ? OC_OXK_BLOCK_Q4_1_SIZE
                        : (qtype == OC_QUANT_Q8_0) ? OC_OXK_BLOCK_Q8_0_SIZE
                        : (qtype == OC_QUANT_Q5_K_S ||
                           qtype == OC_QUANT_Q5_K_M) ? OC_OXK_BLOCK_Q5_K_SIZE
                        : (qtype == OC_QUANT_Q6_K) ? OC_OXK_BLOCK_Q6_K_SIZE
                                                   : OC_OXK_BLOCK_Q4_K_SIZE);
    return row_bytes == expect;
}

/* Activations per tile, clamped so the tile always fits `budget` bytes. */
static size_t act_tile_for(size_t abytes, size_t n_vec, size_t budget)
{
    if (abytes == 0) return 1;
    size_t tile = budget / abytes;
    if (tile == 0) tile = 1;      /* one activation always has to fit */
    if (tile > n_vec) tile = n_vec;
    return tile;
}

size_t oc_matvec_batch_scratch_bytes(size_t max_cols)
{
    /* Largest single-activation encoding at this width, over both Q8
     * flavours. Computed from the block sizes directly rather than from
     * fused_act_kind(), so the bound does not depend on `max_cols` itself
     * being block-aligned. */
    const size_t q8k = (max_cols / OC_OXK_QK_K) * OC_OXK_BLOCK_Q8_K_SIZE;
    const size_t q80 = (max_cols / OC_OXK_QK8_0) * OC_OXK_BLOCK_Q8_0_SIZE;
    const size_t abytes_max = q8k > q80 ? q8k : q80;
    /* tile*abytes <= OC_ACT_TILE_BYTES whenever one activation fits the
     * budget, and exactly abytes when it does not (tile clamps to 1). */
    return abytes_max > OC_ACT_TILE_BYTES ? abytes_max : OC_ACT_TILE_BYTES;
}

typedef struct {
    OcGgufQuantizationType qtype;
    const uint8_t *data;
    size_t         cols;
    size_t         row_bytes;
    size_t         blocks;
    /* Fused path: `tile` activations packed back-to-back, `act_stride` bytes
     * apart. NULL selects the dequant path, which reads `inputs` directly. */
    const uint8_t *acts;
    size_t         act_stride;
    const float   *inputs;      /* dequant path only */
    size_t         in_stride;
    float         *outputs;
    size_t         out_stride;
    size_t         tile;        /* activations handled by this region */
    float         *temp;        /* tid 0 dequant buffer (dequant path) */
} BatchJob;

static void matvec_batch_fused_slice(size_t begin, size_t end, size_t tid,
                                     void *ud)
{
    (void)tid;
    const BatchJob *j = (const BatchJob *)ud;
    for (size_t r = begin; r < end; r++) {
        const uint8_t *row = j->data + r * j->row_bytes;
        /* Row loaded once, reused across the whole activation tile. */
        for (size_t v = 0; v < j->tile; v++) {
            j->outputs[v * j->out_stride + r] =
                fused_row_dot(j->qtype, row, j->blocks,
                              j->acts + v * j->act_stride);
        }
    }
}

/* Q4_K with the row decode hoisted out of the activation loop.
 *
 * The generic slice above still pays the full unpack — 256 nibbles and 8
 * six-bit scale/min pairs per block — once per (row, activation) pair, so a
 * 112-wide tile decodes every row 112 times. Decoding once per row and
 * dotting the prepared form against the tile removes all but 1/tile of that
 * work. Bit-exact with the packed kernel; see oc_oxk_dot_q4_k_prepped(). */
static void matvec_batch_q4k_slice(size_t begin, size_t end, size_t tid,
                                   void *ud)
{
    const BatchJob *j = (const BatchJob *)ud;
    void *prep = oc_parallel_scratch(tid, oc_oxk_q4_k_prep_bytes(j->blocks));
    if (prep == NULL) {
        /* Pre-reserved by the caller, so this should not happen; fall back
         * rather than leave the slice's outputs unwritten. */
        matvec_batch_fused_slice(begin, end, tid, ud);
        return;
    }
    for (size_t r = begin; r < end; r++) {
        oc_oxk_q4_k_prep_row(j->data + r * j->row_bytes, j->blocks, prep);
        for (size_t v = 0; v < j->tile; v++) {
            j->outputs[v * j->out_stride + r] =
                oc_oxk_dot_q4_k_prepped(prep, j->blocks,
                                        j->acts + v * j->act_stride);
        }
    }
}

static void matvec_batch_dequant_slice(size_t begin, size_t end, size_t tid,
                                       void *ud)
{
    const BatchJob *j = (const BatchJob *)ud;
    float *temp = j->temp;
    if (tid != 0) {
        temp = (float *)oc_parallel_scratch(tid, j->cols * sizeof(float));
        if (temp == NULL) return;   /* pre-reserved by the caller */
    }
    for (size_t r = begin; r < end; r++) {
        /* Dequantize the row once, then dot it against every activation —
         * the same amortization the fused path gets, for types with no
         * integer kernel. */
        oc_quant_dequant_row(j->qtype, j->data + r * j->row_bytes,
                             j->row_bytes, temp, j->cols);
        for (size_t v = 0; v < j->tile; v++) {
            const float *in = j->inputs + v * j->in_stride;
            float acc = 0.0f;
            for (size_t i = 0; i < j->cols; i++) acc += temp[i] * in[i];
            j->outputs[v * j->out_stride + r] = acc;
        }
    }
}

void oc_matvec_quantized_batch(OcGgufQuantizationType qtype,
                               const uint8_t *data, size_t rows, size_t cols,
                               size_t row_bytes,
                               const float *inputs, size_t in_stride,
                               float *outputs, size_t out_stride,
                               size_t n_vec, float *temp,
                               uint8_t *act_scratch, size_t act_bytes)
{
    if (n_vec == 0 || rows == 0) return;

    ActKind act_kind = oc_matvec_fused_enabled()
                     ? fused_act_kind(qtype, cols) : ACT_NONE;
    const size_t blocks = (act_kind != ACT_NONE)
                        ? act_blocks_for(act_kind, cols) : 0;
    const size_t abytes = (act_kind != ACT_NONE)
                        ? blocks * act_block_bytes(act_kind) : 0;
    /* A buffer too small for even one activation cannot use the fused path. */
    const bool use_fused = (act_kind != ACT_NONE) && act_scratch != NULL &&
                           abytes > 0 && act_bytes >= abytes &&
                           fused_stride_ok(qtype, blocks, row_bytes);

    if (use_fused) {
        const size_t budget = act_bytes < OC_ACT_TILE_BYTES
                            ? act_bytes : OC_ACT_TILE_BYTES;
        const size_t tile = act_tile_for(abytes, n_vec, budget);

        /* Reserve every worker's prep buffer up front. Doing it inside the
         * region would let an allocation failure silently skip a slice's
         * outputs, and the function cannot report OOM. */
        bool prep_q4k = (qtype == OC_QUANT_Q4_K_S || qtype == OC_QUANT_Q4_K_M);
        if (prep_q4k) {
            const size_t need = oc_oxk_q4_k_prep_bytes(blocks);
            const size_t nthreads = oc_parallel_n_threads();
            for (size_t t = 0; t < nthreads; t++) {
                if (oc_parallel_scratch(t, need) == NULL) { prep_q4k = false; break; }
            }
        }

        for (size_t base = 0; base < n_vec; base += tile) {
            const size_t m = (n_vec - base < tile) ? (n_vec - base) : tile;
            /* Quantize this tile of activations once; every row and every
             * thread then reads them read-only. */
            for (size_t v = 0; v < m; v++) {
                const float *in = inputs + (base + v) * in_stride;
                uint8_t *dst = act_scratch + v * abytes;
                if (act_kind == ACT_Q8_K) quantize_act_q8_k(in, cols, dst);
                else                      quantize_act_q8_0(in, cols, dst);
            }
            BatchJob job = { qtype, data, cols, row_bytes, blocks,
                             act_scratch, abytes, NULL, in_stride,
                             outputs + base * out_stride, out_stride, m,
                             temp };
            /* Hoisting the Q4_K row decode only pays once there is more than
             * one activation to amortize it over; at m == 1 it is pure
             * overhead. */
            if (prep_q4k && m > 1) {
                oc_parallel_for(rows, matvec_batch_q4k_slice, &job);
            } else {
                oc_parallel_for(rows, matvec_batch_fused_slice, &job);
            }
        }
        return;
    }

    /* Dequant reference path. Pre-reserve worker scratch for the same reason
     * oc_matvec_quantized() does: a failure inside a slice would silently
     * leave that slice's outputs unwritten, and this function cannot report
     * OOM. Falling back to serial is correct, just slower. */
    const size_t nt = oc_parallel_n_threads();
    bool serial = false;
    if (nt > 1 && rows >= 8) {
        for (size_t t = 1; t < nt; t++) {
            if (oc_parallel_scratch(t, cols * sizeof(float)) == NULL) {
                serial = true;
                break;
            }
        }
    }
    /* One region over all n_vec: the row is dequantized once per call, so
     * there is no tiling benefit to be had here — `temp` is the only
     * per-row working set and it is already small. */
    BatchJob job = { qtype, data, cols, row_bytes, 0, NULL, 0,
                     inputs, in_stride, outputs, out_stride, n_vec, temp };
    if (serial) matvec_batch_dequant_slice(0, rows, 0, &job);
    else        oc_parallel_for(rows, matvec_batch_dequant_slice, &job);
}

typedef struct {
    const float *data;
    size_t       cols;
    const float *inputs;
    size_t       in_stride;
    float       *outputs;
    size_t       out_stride;
    size_t       n_vec;
} F32BatchJob;

static void matvec_f32_batch_slice(size_t begin, size_t end, size_t tid,
                                   void *ud)
{
    (void)tid;
    const F32BatchJob *j = (const F32BatchJob *)ud;
    for (size_t r = begin; r < end; r++) {
        const float *row = j->data + r * j->cols;
        for (size_t v = 0; v < j->n_vec; v++) {
            const float *in = j->inputs + v * j->in_stride;
            float acc = 0.0f;
            for (size_t i = 0; i < j->cols; i++) acc += row[i] * in[i];
            j->outputs[v * j->out_stride + r] = acc;
        }
    }
}

void oc_matvec_f32_batch(const float *data, size_t rows, size_t cols,
                         const float *inputs, size_t in_stride,
                         float *outputs, size_t out_stride, size_t n_vec)
{
    if (n_vec == 0 || rows == 0) return;
    F32BatchJob job = { data, cols, inputs, in_stride, outputs, out_stride,
                        n_vec };
    oc_parallel_for(rows, matvec_f32_batch_slice, &job);
}

void oc_matvec_quantized_fused(OcGgufQuantizationType qtype,
                               const uint8_t *const *datas, const size_t *rows,
                               size_t cols, const size_t *row_bytes,
                               size_t n_outs, const float *input,
                               float *const *outs, float *temp)
{
    for (size_t k = 0; k < n_outs; k++) {
        oc_matvec_quantized(qtype, datas[k], rows[k], cols, row_bytes[k],
                            input, outs[k], temp);
    }
}
