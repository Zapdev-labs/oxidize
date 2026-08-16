/*
 * matvec.c — quantized/f32 matrix-vector products (scalar reference).
 *
 * Port of oxidize-core/src/compute/tensor/kernels/gemv.rs
 * (`gemv_f32`, `gemv_dequant_scalar_fallback`). The dequant step reuses
 * the SIMD-accelerated `oc_quant_dequant_row`, so this path inherits the
 * AVX2/AVX-512 speedups on capable hosts.
 */
#include "oxidize/matvec.h"
#include "oxidize/attn_kernels.h"
#include "oxidize/flash_attention.h"
#include "oxidize/oxk.h"
#include "oxidize/parallel.h"
#include "oxidize/quant.h"

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include <math.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static bool size_mul(size_t a, size_t b, size_t *out)
{
    if (a != 0 && b > SIZE_MAX / a) return false;
    *out = a * b;
    return true;
}

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

static atomic_bool g_fused_enabled = true;
static atomic_bool g_fused_env_checked = false;
static _Thread_local OcMatvecFusedTestStats g_fused_test_stats;

void oc_matvec_fused_test_reset(void)
{
    g_fused_test_stats = (OcMatvecFusedTestStats){0};
}

OcMatvecFusedTestStats oc_matvec_fused_test_stats(void)
{
    return g_fused_test_stats;
}

void oc_matvec_set_fused(bool enabled)
{
    atomic_store(&g_fused_enabled, enabled);
    atomic_store(&g_fused_env_checked, true);
}

bool oc_matvec_fused_enabled(void)
{
    /* OC_NO_FUSED=1 forces the dequant reference path. This exists so the
     * fused and reference paths can be compared in one binary — an A/B on a
     * real model is the only way to tell a kernel that is exact on synthetic
     * dot products from one that is wrong on real weights. */
    if (!atomic_load(&g_fused_env_checked)) {
        const char *e = getenv("OC_NO_FUSED");
        if (e != NULL && e[0] != '\0' && e[0] != '0')
            atomic_store(&g_fused_enabled, false);
        atomic_store(&g_fused_env_checked, true);
    }
    return atomic_load(&g_fused_enabled);
}

/* Which Q8 flavour a weight type pairs with. */
typedef enum { ACT_NONE = 0, ACT_Q8_0, ACT_Q8_K } ActKind;

/* Every OXK kernel here is verified against the GGUF layout by
 * test_oxk_gguf_layout.c, which dots the dequantized weights with the same
 * dequantized activation the kernel consumes — that removes int8 error from
 * the comparison, so any difference is a real disagreement.
 *
 * Q5_K shares the scale decoding Q4_K needed fixed; it is verified against
 * the dequant reference on a real model (OC_NO_FUSED=1 A/B) rather than by
 * test_oxk_gguf_layout, because oc_quant_pack_row cannot produce Q5_K. */
static ActKind fused_act_kind(OcGgufQuantizationType qtype, size_t cols)
{
    switch (qtype) {
    case OC_QUANT_Q4_0:
    case OC_QUANT_Q4_1:
    case OC_QUANT_Q8_0:
        return (cols % OC_OXK_QK8_0 == 0) ? ACT_Q8_0 : ACT_NONE;
    case OC_QUANT_Q2_K:
    case OC_QUANT_Q3_K_S:
    case OC_QUANT_Q3_K_M:
    case OC_QUANT_Q3_K_L:
    case OC_QUANT_Q4_K_S:
    case OC_QUANT_Q4_K_M:
    case OC_QUANT_Q5_K_S:
    case OC_QUANT_Q5_K_M:
    case OC_QUANT_Q6_K:
    case OC_QUANT_IQ1_XXXS:
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
#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx512f,avx512bw,avx512dq,avx512vl")))
static void quantize_act_q8_k_avx512(const float *x, size_t n, uint8_t *out)
{
    const size_t nb = n / OC_OXK_QK_K;
    const __m512 sign = _mm512_set1_ps(-0.0f);
    for (size_t b = 0; b < nb; b++) {
        const float *src = x + b * OC_OXK_QK_K;
        __m512 vmax = _mm512_setzero_ps();
        for (size_t i = 0; i < OC_OXK_QK_K; i += 16) {
            __m512 v = _mm512_loadu_ps(src + i);
            vmax = _mm512_max_ps(vmax, _mm512_andnot_ps(sign, v));
        }
        const float amax = _mm512_reduce_max_ps(vmax);
        const float d = amax / 127.0f;
        const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
        uint8_t *dst = out + b * OC_OXK_BLOCK_Q8_K_SIZE;
        memcpy(dst, &d, 4);
        int8_t *q = (int8_t *)(dst + 4);
        const __m512 vid = _mm512_set1_ps(id);
        for (size_t i = 0; i < OC_OXK_QK_K; i += 16) {
            __m512 v = _mm512_mul_ps(_mm512_loadu_ps(src + i), vid);
            __m512i qi = _mm512_cvtps_epi32(v);
            qi = _mm512_max_epi32(_mm512_set1_epi32(-128),
                                  _mm512_min_epi32(_mm512_set1_epi32(127), qi));
            _mm_storeu_si128((__m128i *)(q + i), _mm512_cvtepi32_epi8(qi));
        }
        uint8_t *bsums = dst + 4 + OC_OXK_QK_K;
        for (size_t g = 0; g < OC_OXK_QK_K / 16u; g++) {
            __m128i qq = _mm_loadu_si128((const __m128i *)(q + g * 16));
            __m256i w = _mm256_cvtepi8_epi16(qq);
            __m256i sum2 = _mm256_madd_epi16(w, _mm256_set1_epi16(1));
            __m128i lo = _mm256_castsi256_si128(sum2);
            __m128i hi = _mm256_extracti128_si256(sum2, 1);
            lo = _mm_add_epi32(lo, hi);
            lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0x4e));
            lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0xb1));
            int16_t s16 = (int16_t)_mm_cvtsi128_si32(lo);
            memcpy(bsums + g * 2, &s16, 2);
        }
    }
}
#endif

static void quantize_act_q8_k(const float *x, size_t n, uint8_t *out)
{
    const size_t nb = n / OC_OXK_QK_K;
#if defined(__x86_64__) || defined(__i386__)
    if (n >= OC_OXK_QK_K && (n % OC_OXK_QK_K) == 0 &&
        __builtin_cpu_supports("avx512f") &&
        __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512dq") &&
        __builtin_cpu_supports("avx512vl")) {
        quantize_act_q8_k_avx512(x, n, out);
        return;
    }
#endif
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
            int16_t s16 = (int16_t)s;
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
    case OC_QUANT_Q2_K:   return oc_oxk_dot_q2_k_q8_k(row, blocks, act);
    case OC_QUANT_Q3_K_S:
    case OC_QUANT_Q3_K_M:
    case OC_QUANT_Q3_K_L: return oc_oxk_dot_q3_k_q8_k(row, blocks, act);
    case OC_QUANT_IQ1_XXXS: return oc_quant_dot_iq1_xxxs_q8_k(row, blocks, act);
    default:              return 0.0f;
    }
}

static float (*fused_dot_fn(OcGgufQuantizationType qtype))(const uint8_t *,
                                                           size_t,
                                                           const uint8_t *)
{
    switch (qtype) {
    case OC_QUANT_Q4_0:   return oc_oxk_dot_q4_0_q8_0;
    case OC_QUANT_Q4_1:   return oc_oxk_dot_q4_1_q8_0;
    case OC_QUANT_Q8_0:   return oc_oxk_dot_q8_0_q8_0;
    case OC_QUANT_Q4_K_S:
    case OC_QUANT_Q4_K_M: return oc_oxk_dot_q4_k_q8_k;
    case OC_QUANT_Q5_K_S:
    case OC_QUANT_Q5_K_M: return oc_oxk_dot_q5_k_q8_k;
    case OC_QUANT_Q6_K:   return oc_oxk_dot_q6_k_q8_k;
    case OC_QUANT_Q2_K:   return oc_oxk_dot_q2_k_q8_k;
    case OC_QUANT_Q3_K_S:
    case OC_QUANT_Q3_K_M:
    case OC_QUANT_Q3_K_L: return oc_oxk_dot_q3_k_q8_k;
    default:              return NULL;
    }
}

static bool fused_stride_ok(OcGgufQuantizationType qtype, size_t blocks,
                            size_t row_bytes);
static size_t act_block_bytes(ActKind kind);
static size_t act_blocks_for(ActKind kind, size_t cols);

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
        j->output[r] = oc_attn_dot_f32(row, j->input, j->cols);
    }
}

void oc_matvec_f32(const float *data, size_t rows, size_t cols,
                   const float *input, float *output)
{
    F32Job job = { data, cols, input, output };
    oc_parallel_for(rows, matvec_f32_slice, &job);
}

#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx512f,avx512bw")))
static float bf16_row_dot_avx512(const uint16_t *w, const float *x, size_t n)
{
    __m512 acc = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i h = _mm256_loadu_si256((const __m256i *)(w + i));
        __m512i w32 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16);
        acc = _mm512_fmadd_ps(_mm512_castsi512_ps(w32),
                              _mm512_loadu_ps(x + i), acc);
    }
    float sum = _mm512_reduce_add_ps(acc);
    for (; i < n; i++) {
        uint32_t bits = (uint32_t)w[i] << 16;
        float f;
        memcpy(&f, &bits, 4);
        sum += f * x[i];
    }
    return sum;
}

__attribute__((target("avx2,fma")))
static float bf16_row_dot_avx2(const uint16_t *w, const float *x, size_t n)
{
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i *)(w + i));
        __m256i w32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
        acc = _mm256_fmadd_ps(_mm256_castsi256_ps(w32),
                              _mm256_loadu_ps(x + i), acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehdup_ps(s));
    s = _mm_add_ss(s, _mm_movehl_ps(s, s));
    float sum = _mm_cvtss_f32(s);
    for (; i < n; i++) {
        uint32_t bits = (uint32_t)w[i] << 16;
        float f;
        memcpy(&f, &bits, 4);
        sum += f * x[i];
    }
    return sum;
}
#endif

static float bf16_row_dot(const uint16_t *w, const float *x, size_t n)
{
#if defined(__x86_64__) || defined(__i386__)
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw"))
        return bf16_row_dot_avx512(w, x, n);
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
        return bf16_row_dot_avx2(w, x, n);
#endif
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        uint32_t bits = (uint32_t)w[i] << 16;
        float f;
        memcpy(&f, &bits, 4);
        sum += f * x[i];
    }
    return sum;
}

typedef struct {
    const uint16_t *data;
    size_t          cols;
    const float    *input;
    float          *output;
} Bf16Job;

static void matvec_bf16_slice(size_t begin, size_t end, size_t tid, void *ud)
{
    (void)tid;
    const Bf16Job *j = (const Bf16Job *)ud;
    for (size_t r = begin; r < end; r++)
        j->output[r] = bf16_row_dot(j->data + r * j->cols, j->input, j->cols);
}

static void oc_matvec_bf16(const uint16_t *data, size_t rows, size_t cols,
                           const float *input, float *output)
{
    Bf16Job job = { data, cols, input, output };
    oc_parallel_for(rows, matvec_bf16_slice, &job);
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
    float        (*row_dot)(const uint8_t *row, size_t blocks, const uint8_t *act);
    /* Prepared-row path: types whose only integer kernel works off a decoded
     * row (Q2_K / Q3_K). The row is decoded into per-thread scratch and
     * dotted once — decode has a single activation, so there is nothing to
     * amortize a wider kernel over, but this still beats dequantizing to f32
     * and doing the dot in floating point. */
    size_t         prep_bytes;
    void         (*prep_fn)(const uint8_t *row, size_t blocks, void *scratch);
    float        (*prep_dot)(const void *prep, size_t blocks,
                             const uint8_t *act);
} QuantJob;

typedef struct {
    OcGgufQuantizationType qtype;
    const uint8_t *const *datas;
    const size_t *rows;
    const size_t *row_bytes;
    float *const *outs;
    size_t n_outs;
    size_t blocks;
    const uint8_t *act;
} FusedMultiJob;

typedef struct {
    OcGgufQuantizationType qtype;
    const uint8_t *const *datas;
    const size_t *rows;
    const size_t *row_bytes;
    float *const *outs;
    const size_t *act_offsets;
    size_t n_outs;
    size_t blocks;
    const uint8_t *acts;
} FusedMultiInputJob;

static void matvec_fused_slice(size_t begin, size_t end, size_t tid, void *ud)
{
    (void)tid;
    const QuantJob *j = (const QuantJob *)ud;
    for (size_t r = begin; r < end; r++) {
#if defined(__x86_64__) || defined(__i386__)
        if (r + 2 < end) {
            _mm_prefetch((const char *)(j->data + (r + 2) * j->row_bytes),
                         _MM_HINT_T0);
        }
#endif
        j->output[r] = j->row_dot
            ? j->row_dot(j->data + r * j->row_bytes, j->blocks, j->act)
            : fused_row_dot(j->qtype, j->data + r * j->row_bytes,
                            j->blocks, j->act);
    }
}

static void matvec_prep_slice(size_t begin, size_t end, size_t tid, void *ud)
{
    const QuantJob *j = (const QuantJob *)ud;
    void *prep = oc_parallel_scratch(tid, j->prep_bytes);
    if (prep == NULL) {
        /* Pre-reserved by the caller; fall back rather than leave this
         * slice's outputs unwritten. */
        matvec_fused_slice(begin, end, tid, ud);
        return;
    }
    for (size_t r = begin; r < end; r++) {
        j->prep_fn(j->data + r * j->row_bytes, j->blocks, prep);
        j->output[r] = j->prep_dot(prep, j->blocks, j->act);
    }
}

/* The prepared-row kernels for a single activation, or all-NULL for types
 * whose packed kernel is already the best single-activation form. */
static bool prep_kernels_for(OcGgufQuantizationType qtype, size_t blocks,
                             size_t *prep_bytes,
                             void (**prep_fn)(const uint8_t *, size_t, void *),
                             float (**prep_dot)(const void *, size_t,
                                                const uint8_t *))
{
    switch (qtype) {
    case OC_QUANT_Q2_K:
        *prep_bytes = oc_oxk_q2_k_prep_bytes(blocks);
        *prep_fn    = oc_oxk_q2_k_prep_row;
        *prep_dot   = oc_oxk_dot_q2_k_prepped_1;
        return true;
    case OC_QUANT_Q3_K_S:
    case OC_QUANT_Q3_K_M:
    case OC_QUANT_Q3_K_L:
        *prep_bytes = oc_oxk_q6_k_prep_bytes(blocks);
        *prep_fn    = oc_oxk_q3_k_prep_row;
        *prep_dot   = oc_oxk_dot_q3_k_prepped_1;
        return true;
    default:
        return false;
    }
}

static void matvec_fused_multi_slice(size_t begin, size_t end, size_t tid,
                                     void *ud)
{
    (void)tid;
    const FusedMultiJob *j = (const FusedMultiJob *)ud;
    size_t part = 0;
    size_t row = begin;

    while (part < j->n_outs && row >= j->rows[part]) {
        row -= j->rows[part];
        part++;
    }
    for (size_t global = begin; global < end; ) {
        const size_t part_end = j->rows[part];
        while (row < part_end && global < end) {
            j->outs[part][row] = fused_row_dot(
                j->qtype, j->datas[part] + row * j->row_bytes[part],
                j->blocks, j->act);
            row++;
            global++;
        }
        part++;
        row = 0;
    }
}

static void matvec_fused_multi_input_slice(size_t begin, size_t end,
                                           size_t tid, void *ud)
{
    (void)tid;
    const FusedMultiInputJob *j = (const FusedMultiInputJob *)ud;
    size_t part = 0;
    size_t row = begin;

    while (part < j->n_outs && row >= j->rows[part]) {
        row -= j->rows[part];
        part++;
    }
    for (size_t global = begin; global < end; ) {
        const size_t part_end = j->rows[part];
        while (row < part_end && global < end) {
            j->outs[part][row] = fused_row_dot(
                j->qtype, j->datas[part] + row * j->row_bytes[part],
                j->blocks, j->acts + j->act_offsets[part]);
            row++;
            global++;
        }
        part++;
        row = 0;
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
        size_t bytes;
        if (!size_mul(j->cols, sizeof(float), &bytes)) return;
        temp = (float *)oc_parallel_scratch(tid, bytes);
        if (temp == NULL) return;   /* see the OOM note in the caller */
    }

    for (size_t r = begin; r < end; r++) {
        const uint8_t *row = j->data + r * j->row_bytes;
        /* Dequantize this weight row into `temp` (SIMD on capable hosts). */
        oc_quant_dequant_row(j->qtype, row, j->row_bytes, temp, j->cols);
        j->output[r] = oc_attn_dot_f32(temp, j->input, j->cols);
    }
}

void oc_matvec_quantized(OcGgufQuantizationType qtype, const uint8_t *data,
                         size_t rows, size_t cols, size_t row_bytes,
                         const float *input, float *output, float *temp)
{
    if (qtype == OC_QUANT_BF16 && row_bytes == cols * 2u) {
        oc_matvec_bf16((const uint16_t *)data, rows, cols, input, output);
        return;
    }
    size_t temp_bytes;
    if (!size_mul(cols, sizeof(float), &temp_bytes)) return;
    /* Fused integer path, when the weight type has an OXK kernel and the row
     * divides evenly into blocks. The activation is quantized once here and
     * then shared by every row and every thread. */
    ActKind act_kind = oc_matvec_fused_enabled()
                     ? fused_act_kind(qtype, cols) : ACT_NONE;
    if (act_kind != ACT_NONE) {
        const size_t blocks = (act_kind == ACT_Q8_K)
                            ? cols / OC_OXK_QK_K : cols / OC_OXK_QK8_0;
        size_t act_bytes;
        if (!size_mul(blocks, (act_kind == ACT_Q8_K)
                            ? OC_OXK_BLOCK_Q8_K_SIZE : OC_OXK_BLOCK_Q8_0_SIZE,
                      &act_bytes))
            act_kind = ACT_NONE;
        /* Only trust the fused kernel when the weight stride is exactly what
         * it assumes; a padded or interleaved row would be read wrong. */
        if (act_kind != ACT_NONE &&
            fused_stride_ok(qtype, blocks, row_bytes)) {
            /* The activation lives in the caller's dequant buffer, not in
             * thread 0's scratch: the prepared-row path below hands every
             * worker — thread 0 included — its scratch for the decoded row,
             * and there is only one scratch per thread, so sharing it would
             * let the row decode overwrite the activation mid-region.
             * `temp` is cols floats, always at least the ~1.14 bytes per
             * column a Q8_K/Q8_0 encoding needs. */
            uint8_t *act = (uint8_t *)temp;
            if (act != NULL && temp_bytes >= act_bytes) {
                if (act_kind == ACT_Q8_K) quantize_act_q8_k(input, cols, act);
                else                      quantize_act_q8_0(input, cols, act);
                QuantJob job = { qtype, data, cols, row_bytes, input, output,
                                 temp, act, blocks, fused_dot_fn(qtype),
                                 0, NULL, NULL };
                /* Reserve every worker's decode buffer before opening the
                 * region — an allocation failure inside it could not be
                 * reported and would drop a slice's outputs. */
                if (prep_kernels_for(qtype, blocks, &job.prep_bytes,
                                     &job.prep_fn, &job.prep_dot)) {
                    const size_t nthreads = oc_parallel_n_threads();
                    for (size_t t = 0; t < nthreads; t++) {
                        if (oc_parallel_scratch(t, job.prep_bytes) == NULL) {
                            job.prep_fn = NULL;
                            break;
                        }
                    }
                }
                if (job.prep_fn != NULL)
                    oc_parallel_for(rows, matvec_prep_slice, &job);
                else
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
            if (oc_parallel_scratch(t, temp_bytes) == NULL) {
                /* Fall back to the serial path rather than produce a
                 * partially-written output vector. */
                QuantJob job = { qtype, data, cols, row_bytes, input,
                                 output, temp, NULL, 0, NULL, 0, NULL, NULL };
                matvec_quant_slice(0, rows, 0, &job);
                return;
            }
        }
    }
    QuantJob job = { qtype, data, cols, row_bytes, input, output, temp,
                     NULL, 0, NULL, 0, NULL, NULL };
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
#define OC_ACT_TILE_BYTES (oc_act_tile_bytes())

/* Overridable with OXC_ACT_TILE_KB so the tile can be retuned per machine
 * without a rebuild: a wider tile sweeps the weights fewer times (the win
 * grows with chunk size) at the cost of spilling the tile out of L2. */
static size_t oc_act_tile_bytes(void)
{
    static size_t cached = 0;
    if (cached != 0) return cached;
    /* 4 MB measured on 2x Xeon Gold 5220R / 32 threads, Qwen3.6-27B prefill of
     * 515 tokens: 256 KB 4.09 tok/s, 1 MB 4.56, 4 MB 4.82, 16 MB 4.90. Past
     * 4 MB the extra weight-sweep savings stop paying for the L2 spill. */
    size_t bytes = 4u * 1024u * 1024u;
    const char *env = getenv("OXC_ACT_TILE_KB");
    if (env != NULL) {
        char *end = NULL;
        unsigned long kb = strtoul(env, &end, 10);
        if (end != env && kb > 0 && kb <= (1u << 20)) bytes = (size_t)kb * 1024u;
    }
    cached = bytes;
    return cached;
}

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
    size_t expect;
    const size_t block_bytes = ((qtype == OC_QUANT_Q4_0) ? OC_OXK_BLOCK_Q4_0_SIZE
                        : (qtype == OC_QUANT_Q4_1) ? OC_OXK_BLOCK_Q4_1_SIZE
                        : (qtype == OC_QUANT_Q8_0) ? OC_OXK_BLOCK_Q8_0_SIZE
                        : (qtype == OC_QUANT_Q5_K_S ||
                           qtype == OC_QUANT_Q5_K_M) ? OC_OXK_BLOCK_Q5_K_SIZE
                        : (qtype == OC_QUANT_Q6_K) ? OC_OXK_BLOCK_Q6_K_SIZE
                        : (qtype == OC_QUANT_Q2_K) ? OC_OXK_BLOCK_Q2_K_SIZE
                        : (qtype == OC_QUANT_Q3_K_S ||
                           qtype == OC_QUANT_Q3_K_M ||
                           qtype == OC_QUANT_Q3_K_L) ? OC_OXK_BLOCK_Q3_K_SIZE
                        : (qtype == OC_QUANT_IQ1_XXXS) ? OC_BLOCK_IQ1_XXXS_SIZE
                                                   : OC_OXK_BLOCK_Q4_K_SIZE);
    return size_mul(blocks, block_bytes, &expect) && row_bytes == expect;
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
    size_t q8k, q80;
    if (!size_mul(max_cols / OC_OXK_QK_K, OC_OXK_BLOCK_Q8_K_SIZE, &q8k) ||
        !size_mul(max_cols / OC_OXK_QK8_0, OC_OXK_BLOCK_Q8_0_SIZE, &q80))
        return SIZE_MAX;
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
    /* Prep path (K-quants with a prepared-row kernel): decode each weight
     * row once into per-thread scratch, then dot the prepared form against
     * the whole activation tile. */
    size_t         prep_bytes;
    void         (*prep_fn)(const uint8_t *row, size_t blocks, void *scratch);
    void         (*multi_fn)(const void *prep, size_t blocks,
                             const uint8_t *acts, size_t act_stride,
                             size_t n_act, float *out);
    void         (*packed_multi)(const uint8_t *row, size_t blocks,
                                 const uint8_t *acts, size_t act_stride,
                                 size_t n_act, float *out);
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

static void matvec_batch_packed_slice(size_t begin, size_t end, size_t tid,
                                      void *ud)
{
    (void)tid;
    const BatchJob *j = (const BatchJob *)ud;
    float res[128];
    for (size_t r = begin; r < end; r++) {
        for (size_t v0 = 0; v0 < j->tile; v0 += 128) {
            const size_t m = (j->tile - v0 < 128) ? (j->tile - v0) : 128;
            j->packed_multi(j->data + r * j->row_bytes, j->blocks,
                            j->acts + v0 * j->act_stride, j->act_stride,
                            m, res);
            for (size_t v = 0; v < m; v++)
                j->outputs[(v0 + v) * j->out_stride + r] = res[v];
        }
    }
}

/* K-quants with the row decode hoisted out of the activation loop.
 *
 * The generic slice above still pays the full unpack — 256 codes and the
 * scale decode per block — once per (row, activation) pair, so a 112-wide
 * tile decodes every row 112 times. Decoding once per row and dotting the
 * prepared form against the tile removes all but 1/tile of that work.
 * Bit-exact with the packed kernels; see oc_oxk_dot_q4_k_prepped() /
 * oc_oxk_dot_q6_k_prepped(). */
static void matvec_batch_prep_slice(size_t begin, size_t end, size_t tid,
                                    void *ud)
{
    const BatchJob *j = (const BatchJob *)ud;
    void *prep = oc_parallel_scratch(tid, j->prep_bytes);
    if (prep == NULL) {
        /* Pre-reserved by the caller, so this should not happen; fall back
         * rather than leave the slice's outputs unwritten. */
        matvec_batch_fused_slice(begin, end, tid, ud);
        return;
    }
    for (size_t r = begin; r < end; r++) {
        j->prep_fn(j->data + r * j->row_bytes, j->blocks, prep);
        /* The multi-activation kernel writes contiguous results; the output
         * layout is activation-major, so gather into a small buffer and
         * scatter. 128 comfortably covers the L2-budget tile widths. */
        float res[128];
        for (size_t v0 = 0; v0 < j->tile; v0 += 128) {
            const size_t m = (j->tile - v0 < 128) ? (j->tile - v0) : 128;
            j->multi_fn(prep, j->blocks, j->acts + v0 * j->act_stride,
                        j->act_stride, m, res);
            for (size_t v = 0; v < m; v++) {
                j->outputs[(v0 + v) * j->out_stride + r] = res[v];
            }
        }
    }
}

static void matvec_batch_dequant_slice(size_t begin, size_t end, size_t tid,
                                       void *ud)
{
    const BatchJob *j = (const BatchJob *)ud;
    float *temp = j->temp;
    if (tid != 0) {
        size_t bytes;
        if (!size_mul(j->cols, sizeof(float), &bytes)) return;
        temp = (float *)oc_parallel_scratch(tid, bytes);
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
            j->outputs[v * j->out_stride + r] =
                oc_attn_dot_f32(temp, in, j->cols);
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
    size_t temp_bytes;
    if (!size_mul(cols, sizeof(float), &temp_bytes)) return;

    ActKind act_kind = oc_matvec_fused_enabled()
                     ? fused_act_kind(qtype, cols) : ACT_NONE;
    const size_t blocks = (act_kind != ACT_NONE)
                        ? act_blocks_for(act_kind, cols) : 0;
    size_t abytes = 0;
    if (act_kind != ACT_NONE &&
        !size_mul(blocks, act_block_bytes(act_kind), &abytes))
        act_kind = ACT_NONE;
    /* A buffer too small for even one activation cannot use the fused path. */
    const bool use_fused = (act_kind != ACT_NONE) && act_scratch != NULL &&
                           abytes > 0 && act_bytes >= abytes &&
                           fused_stride_ok(qtype, blocks, row_bytes);

    if (use_fused) {
        const size_t budget = act_bytes < OC_ACT_TILE_BYTES
                            ? act_bytes : OC_ACT_TILE_BYTES;
        const size_t tile = act_tile_for(abytes, n_vec, budget);

        /* Prepared-row kernels per weight type. Q5_K shares the Q4_K prep
         * layout (codes 0..31 fit the same unsigned bytes), so it reuses
         * the Q4_K multi kernel outright. */
        size_t prep_bytes = 0;
        void (*prep_fn)(const uint8_t *, size_t, void *) = NULL;
        void (*multi_fn)(const void *, size_t, const uint8_t *, size_t,
                         size_t, float *) = NULL;
        void (*packed_multi)(const uint8_t *, size_t, const uint8_t *, size_t,
                             size_t, float *) = NULL;
        switch (qtype) {
        case OC_QUANT_Q4_K_S:
        case OC_QUANT_Q4_K_M:
            prep_bytes = oc_oxk_q4_k_prep_bytes(blocks);
            prep_fn    = oc_oxk_q4_k_prep_row;
            multi_fn   = oc_oxk_dot_q4_k_prepped_multi;
            break;
        case OC_QUANT_Q5_K_S:
        case OC_QUANT_Q5_K_M:
            prep_bytes = oc_oxk_q4_k_prep_bytes(blocks);
            prep_fn    = oc_oxk_q5_k_prep_row;
            multi_fn   = oc_oxk_dot_q4_k_prepped_multi;
            break;
        case OC_QUANT_Q6_K:
            prep_bytes = oc_oxk_q6_k_prep_bytes(blocks);
            prep_fn    = oc_oxk_q6_k_prep_row;
            multi_fn   = oc_oxk_dot_q6_k_prepped_multi;
            break;
        /* Q3_K decodes into the Q6_K prepared layout, so it borrows that
         * layout's scratch size. */
        case OC_QUANT_Q3_K_S:
        case OC_QUANT_Q3_K_M:
        case OC_QUANT_Q3_K_L:
            prep_bytes = oc_oxk_q6_k_prep_bytes(blocks);
            prep_fn    = oc_oxk_q3_k_prep_row;
            multi_fn   = oc_oxk_dot_q3_k_prepped_multi;
            break;
        case OC_QUANT_Q2_K:
            prep_bytes = oc_oxk_q2_k_prep_bytes(blocks);
            prep_fn    = oc_oxk_q2_k_prep_row;
            multi_fn   = oc_oxk_dot_q2_k_prepped_multi;
            break;
        default:
            break;
        }

        /* Reserve every worker's prep buffer up front. Doing it inside the
         * region would let an allocation failure silently skip a slice's
         * outputs, and the function cannot report OOM. */
        if (prep_fn != NULL) {
            const size_t nthreads = oc_parallel_n_threads();
            for (size_t t = 0; t < nthreads; t++) {
                if (oc_parallel_scratch(t, prep_bytes) == NULL) {
                    prep_fn = NULL;
                    break;
                }
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
                             temp, prep_bytes, prep_fn, multi_fn, packed_multi };
            if (prep_fn != NULL && m > 1) {
                oc_parallel_for(rows, matvec_batch_prep_slice, &job);
            } else if (packed_multi != NULL && m > 1) {
                oc_parallel_for(rows, matvec_batch_packed_slice, &job);
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
            if (oc_parallel_scratch(t, temp_bytes) == NULL) {
                serial = true;
                break;
            }
        }
    }
    /* One region over all n_vec: the row is dequantized once per call, so
     * there is no tiling benefit to be had here — `temp` is the only
     * per-row working set and it is already small. */
    BatchJob job = { qtype, data, cols, row_bytes, 0, NULL, 0,
                     inputs, in_stride, outputs, out_stride, n_vec, temp,
                     0, NULL, NULL, NULL };
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
            float acc = oc_attn_dot_f32(row, in, j->cols);
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

void oc_matvec_quantized_fused(const OcGgufQuantizationType *qtypes,
                               const uint8_t *const *datas, const size_t *rows,
                               size_t cols, const size_t *row_bytes,
                               size_t n_outs, const float *input,
                               float *const *outs, float *temp)
{
    if (n_outs == 0) return;

    const OcGgufQuantizationType qtype = qtypes[0];
    const ActKind act_kind = oc_matvec_fused_enabled()
                           ? fused_act_kind(qtype, cols) : ACT_NONE;
    const size_t blocks = (act_kind != ACT_NONE)
                        ? act_blocks_for(act_kind, cols) : 0;
    bool compatible = act_kind != ACT_NONE;
    size_t total_rows = 0;
    for (size_t k = 0; k < n_outs; k++) {
        if (qtypes[k] != qtype || !fused_stride_ok(qtype, blocks, row_bytes[k])) {
            compatible = false;
            break;
        }
        if (rows[k] > SIZE_MAX - total_rows) {
            compatible = false;
            break;
        }
        total_rows += rows[k];
    }

    if (compatible && total_rows != 0) {
        size_t act_bytes;
        if (!size_mul(blocks, act_block_bytes(act_kind), &act_bytes))
            compatible = false;
        if (!compatible) goto fused_fallback;
        uint8_t *act = (uint8_t *)oc_parallel_scratch(0, act_bytes);
        if (act != NULL) {
            if (act_kind == ACT_Q8_K) quantize_act_q8_k(input, cols, act);
            else                      quantize_act_q8_0(input, cols, act);
            FusedMultiJob job = { qtype, datas, rows, row_bytes, outs, n_outs,
                                  blocks, act };
            g_fused_test_stats.activation_quantizations++;
            g_fused_test_stats.parallel_dispatches++;
            oc_parallel_for(total_rows, matvec_fused_multi_slice, &job);
            return;
        }
    }

fused_fallback:
    for (size_t k = 0; k < n_outs; k++) {
        oc_matvec_quantized(qtypes[k], datas[k], rows[k], cols, row_bytes[k],
                            input, outs[k], temp);
        const ActKind fallback_kind = oc_matvec_fused_enabled()
                                    ? fused_act_kind(qtypes[k], cols) : ACT_NONE;
        const size_t fallback_blocks = (fallback_kind != ACT_NONE)
                                     ? act_blocks_for(fallback_kind, cols) : 0;
        if (fallback_kind != ACT_NONE &&
            fused_stride_ok(qtypes[k], fallback_blocks, row_bytes[k])) {
            g_fused_test_stats.activation_quantizations++;
        }
        g_fused_test_stats.parallel_dispatches++;
        g_fused_test_stats.fallback_calls++;
    }
}


void oc_matvec_quantized_multi_input(
    const OcGgufQuantizationType *qtypes,
    const uint8_t *const *datas, const size_t *rows, size_t cols,
    const size_t *row_bytes, size_t n_outs,
    const float *const *inputs, float *const *outs, float *temp)
{
    if (n_outs == 0) return;

    const OcGgufQuantizationType qtype = qtypes[0];
    const ActKind act_kind = oc_matvec_fused_enabled()
                           ? fused_act_kind(qtype, cols) : ACT_NONE;
    const size_t blocks = (act_kind != ACT_NONE)
                        ? act_blocks_for(act_kind, cols) : 0;
    bool compatible = act_kind != ACT_NONE;
    size_t total_rows = 0;
    for (size_t k = 0; k < n_outs; k++) {
        if (qtypes[k] != qtype || !fused_stride_ok(qtype, blocks, row_bytes[k])) {
            compatible = false;
            break;
        }
        if (rows[k] > SIZE_MAX - total_rows) {
            compatible = false;
            break;
        }
        total_rows += rows[k];
    }

    if (compatible && total_rows != 0) {
        size_t one_act_bytes;
        if (size_mul(blocks, act_block_bytes(act_kind), &one_act_bytes) &&
            one_act_bytes != 0 && n_outs <= SIZE_MAX / one_act_bytes) {
            const size_t act_bytes = n_outs * one_act_bytes;
            uint8_t *acts = (uint8_t *)oc_parallel_scratch(0, act_bytes);
            if (acts != NULL) {
                size_t act_offsets[n_outs];
                for (size_t k = 0; k < n_outs; k++) {
                    act_offsets[k] = k * one_act_bytes;
                    if (act_kind == ACT_Q8_K)
                        quantize_act_q8_k(inputs[k], cols, acts + act_offsets[k]);
                    else
                        quantize_act_q8_0(inputs[k], cols, acts + act_offsets[k]);
                }
                FusedMultiInputJob job = {
                    qtype, datas, rows, row_bytes, outs, act_offsets,
                    n_outs, blocks, acts,
                };
                g_fused_test_stats.activation_quantizations += n_outs;
                g_fused_test_stats.parallel_dispatches++;
                oc_parallel_for(total_rows, matvec_fused_multi_input_slice,
                                &job);
                return;
            }
        }
    }

    for (size_t k = 0; k < n_outs; k++) {
        oc_matvec_quantized(qtypes[k], datas[k], rows[k], cols, row_bytes[k],
                            inputs[k], outs[k], temp);
        const ActKind fallback_kind = oc_matvec_fused_enabled()
                                    ? fused_act_kind(qtypes[k], cols) : ACT_NONE;
        const size_t fallback_blocks = (fallback_kind != ACT_NONE)
                                     ? act_blocks_for(fallback_kind, cols) : 0;
        if (fallback_kind != ACT_NONE &&
            fused_stride_ok(qtypes[k], fallback_blocks, row_bytes[k])) {
            g_fused_test_stats.activation_quantizations++;
        }
        g_fused_test_stats.parallel_dispatches++;
        g_fused_test_stats.fallback_calls++;
    }
}
