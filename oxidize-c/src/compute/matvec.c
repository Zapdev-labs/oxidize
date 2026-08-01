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

void oc_matvec_set_fused(bool enabled) { g_fused_enabled = enabled; }
bool oc_matvec_fused_enabled(void)     { return g_fused_enabled; }

/* Which Q8 flavour a weight type pairs with. */
typedef enum { ACT_NONE = 0, ACT_Q8_0, ACT_Q8_K } ActKind;

/* Only the OXK kernels that have been verified against the real GGUF weight
 * layout are used here.
 *
 * Q4_0, Q4_1 and Q4_K are deliberately NOT in this list: their OXK kernels
 * unpack nibbles in a different order than the GGUF layout the loader and
 * oc_quant_dequant_row use. Q4_0's kernel pairs the low/high nibble of byte i
 * with activation elements 2i and 2i+1, where GGUF puts them at i and i+16.
 * Feeding real weights through them produces uncorrelated output (measured
 * relative error 0.47 for Q4_0, 2.20 for Q4_K against the dequant reference
 * with activation quantization error removed).
 *
 * That is a pre-existing bug in those kernels, not in this dispatch: the OXK
 * parity tests only ever compared the SIMD variants against the OXK scalar
 * reference, so every variant shares the same wrong convention and the
 * disagreement with GGUF never showed up. test_oxk_gguf_layout.c now checks
 * each kernel against the dequant path and documents which ones fail.
 *
 * Q5_K has no verification either way here because oc_quant_pack_row cannot
 * produce Q5_K, so it stays off until it can be checked. */
static ActKind fused_act_kind(OcGgufQuantizationType qtype, size_t cols)
{
    switch (qtype) {
    case OC_QUANT_Q8_0:
        return (cols % OC_OXK_QK8_0 == 0) ? ACT_Q8_0 : ACT_NONE;
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
    ActKind act_kind = g_fused_enabled ? fused_act_kind(qtype, cols) : ACT_NONE;
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
