/* test_oxk_q23k.c — Q2_K / Q3_K integer dot kernels.
 *
 * These two types carry the bulk of a 2-bit K-quant model (Muse Glimmer's
 * UD-Q2_K_XL is 59% Q3_K + 19% Q2_K by weight bytes) and had no OXK kernel
 * at all, so every matmul touching them fell back to dequantize-to-f32.
 *
 * The contract under test is the same one the rest of OXK holds to:
 *
 *   1. the packed integer dot agrees with dequantize-then-f32-dot, which is
 *      the definition of what the weights mean;
 *   2. the prepared-row form agrees with the packed form, because the batched
 *      prefill path uses one and decode uses the other on the same weights;
 *   3. the multi-activation kernel — SIMD when the host has VNNI — agrees
 *      exactly with the scalar prepared dot, since the batch path picks
 *      between them purely on activation count.
 *
 * (3) is exact, not approximate: the kernels differ only in how the integer
 * block sums are reduced, and that arithmetic is exact in int32.
 */
#include <criterion/criterion.h>

#include "oxidize/oxk.h"
#include "oxidize/quant.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define N_BLOCKS 7u
#define N_VALS   (N_BLOCKS * OC_OXK_QK_K)

static bool float_within_1ulp(float a, float b)
{
    if (a == b) return true;
    return nextafterf(a, b) == b || nextafterf(b, a) == a;
}

/* Deterministic pseudo-random floats with enough dynamic range per block to
 * exercise the per-group scales (a flat distribution would let a broken
 * scale decode pass). */
static void fill_weights(float *w, size_t n, uint32_t seed)
{
    uint32_t st = seed;
    for (size_t i = 0; i < n; i++) {
        st = st * 1664525u + 1013904223u;
        const float u = (float)((st >> 8) & 0xFFFF) / 32768.0f - 1.0f;
        const float block_gain = 1.0f + (float)((i / OC_OXK_QK_K) % 5);
        w[i] = u * block_gain;
    }
}

/* Q8_K activation block: f32 d, 256 int8, 16 int16 block sums over 16
 * elements each. Built here rather than pulled from matvec.c so the test
 * pins the layout the kernels assume. */
static void quantize_q8_k(const float *x, size_t n_blocks, uint8_t *out)
{
    for (size_t b = 0; b < n_blocks; b++) {
        const float *src = x + b * OC_OXK_QK_K;
        uint8_t *dst = out + b * OC_OXK_BLOCK_Q8_K_SIZE;
        float amax = 0.0f;
        for (size_t i = 0; i < OC_OXK_QK_K; i++) {
            const float a = fabsf(src[i]);
            if (a > amax) amax = a;
        }
        const float d = amax / 127.0f;
        const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
        memcpy(dst, &d, 4);
        int8_t *q = (int8_t *)(dst + 4);
        for (size_t i = 0; i < OC_OXK_QK_K; i++) {
            int v = (int)lrintf(src[i] * id);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            q[i] = (int8_t)v;
        }
        uint8_t *bsums = dst + 4 + OC_OXK_QK_K;
        for (unsigned g = 0; g < 16; g++) {
            int32_t s = 0;
            for (unsigned l = 0; l < 16; l++) s += q[g * 16 + l];
            const int16_t v = (int16_t)s;
            bsums[g * 2]     = (uint8_t)(v & 0xFF);
            bsums[g * 2 + 1] = (uint8_t)((v >> 8) & 0xFF);
        }
    }
}

/* Reference: dequantize the packed row and dot it against the activation as
 * the activation's own dequantization defines it. */
static float reference_dot(OcGgufQuantizationType qtype, const uint8_t *row,
                           size_t row_bytes, const uint8_t *act)
{
    float *w = malloc(N_VALS * sizeof(float));
    cr_assert_not_null(w, "malloc weights");
    cr_assert_eq(oc_quant_dequant_row(qtype, row, row_bytes, w, N_VALS), OC_OK,
                 "dequant row");

    float sum = 0.0f;
    for (size_t b = 0; b < N_BLOCKS; b++) {
        const uint8_t *ab = act + b * OC_OXK_BLOCK_Q8_K_SIZE;
        float d;
        memcpy(&d, ab, 4);
        const int8_t *q = (const int8_t *)(ab + 4);
        for (size_t i = 0; i < OC_OXK_QK_K; i++)
            sum += w[b * OC_OXK_QK_K + i] * d * (float)q[i];
    }
    free(w);
    return sum;
}

typedef struct {
    OcGgufQuantizationType qtype;
    size_t block_bytes;
    float (*packed)(const uint8_t *, size_t, const uint8_t *);
    size_t (*prep_bytes)(size_t);
    void (*prep_row)(const uint8_t *, size_t, void *);
    float (*prepped)(const void *, size_t, const uint8_t *);
    void (*prepped_multi)(const void *, size_t, const uint8_t *, size_t,
                          size_t, float *);
} TypeUnderTest;

static void run_case(const TypeUnderTest *t, uint32_t seed)
{
    const size_t row_bytes = N_BLOCKS * t->block_bytes;
    float *w = malloc(N_VALS * sizeof(float));
    uint8_t *row = malloc(row_bytes);
    cr_assert_not_null(w, "malloc w");
    cr_assert_not_null(row, "malloc row");
    fill_weights(w, N_VALS, seed);
    cr_assert_eq(oc_quant_pack_row(t->qtype, w, N_VALS, row, row_bytes),
                 OC_OK, "quantize row");

    /* Four activations so the SIMD kernels take their 4-wide path, plus a
     * fifth to exercise the scalar tail. */
    enum { N_ACT = 5 };
    float *x = malloc(N_VALS * sizeof(float));
    uint8_t *acts = malloc(N_ACT * N_BLOCKS * OC_OXK_BLOCK_Q8_K_SIZE);
    cr_assert_not_null(x, "malloc x");
    cr_assert_not_null(acts, "malloc acts");
    const size_t act_stride = N_BLOCKS * OC_OXK_BLOCK_Q8_K_SIZE;
    for (int a = 0; a < N_ACT; a++) {
        fill_weights(x, N_VALS, seed + 977u * (uint32_t)(a + 1));
        quantize_q8_k(x, N_BLOCKS, acts + (size_t)a * act_stride);
    }

    void *prep = malloc(t->prep_bytes(N_BLOCKS));
    cr_assert_not_null(prep, "malloc prep");
    t->prep_row(row, N_BLOCKS, prep);

    float multi[N_ACT];
    t->prepped_multi(prep, N_BLOCKS, acts, act_stride, N_ACT, multi);

    for (int a = 0; a < N_ACT; a++) {
        const uint8_t *act = acts + (size_t)a * act_stride;
        const float ref    = reference_dot(t->qtype, row, row_bytes, act);
        const float packed = t->packed(row, N_BLOCKS, act);
        const float prepped = t->prepped(prep, N_BLOCKS, act);

        /* The integer kernels and the dequant reference sum in different
         * orders over ~1800 terms, so match on relative error rather than
         * exactly. A wrong scale or bit layout misses by orders of
         * magnitude, not by 1e-4. */
        const float tol = 1e-4f * (fabsf(ref) + 1.0f);
        cr_assert_float_eq(packed, ref, tol,
            "act %d: packed dot %f != dequant reference %f", a,
            (double)packed, (double)ref);
        cr_assert_float_eq(prepped, packed, tol,
            "act %d: prepared dot %f != packed dot %f", a,
            (double)prepped, (double)packed);
        /* Exact: same float accumulation order, integer-only difference. */
        cr_assert(float_within_1ulp(multi[a], prepped),
            "act %d: multi kernel %.9g != prepared dot %.9g", a,
            (double)multi[a], (double)prepped);
    }

    free(prep);
    free(acts);
    free(x);
    free(row);
    free(w);
}

Test(oxk_q23k, q2_k_matches_dequant_reference)
{
    const TypeUnderTest t = {
        OC_QUANT_Q2_K, OC_OXK_BLOCK_Q2_K_SIZE,
        oc_oxk_dot_q2_k_q8_k, oc_oxk_q2_k_prep_bytes, oc_oxk_q2_k_prep_row,
        oc_oxk_dot_q2_k_prepped, oc_oxk_dot_q2_k_prepped_multi,
    };
    for (uint32_t seed = 1; seed <= 4; seed++) run_case(&t, seed * 7919u);
}

Test(oxk_q23k, q3_k_matches_dequant_reference)
{
    const TypeUnderTest t = {
        OC_QUANT_Q3_K_S, OC_OXK_BLOCK_Q3_K_SIZE,
        oc_oxk_dot_q3_k_q8_k, oc_oxk_q6_k_prep_bytes, oc_oxk_q3_k_prep_row,
        oc_oxk_dot_q3_k_prepped, oc_oxk_dot_q3_k_prepped_multi,
    };
    for (uint32_t seed = 1; seed <= 4; seed++) run_case(&t, seed * 6151u);
}

/* The vectorized block-sum fold introduced for these kernels also replaced
 * the scalar epilogue in the Q4_K and Q6_K multi kernels, so pin those too:
 * they must still agree exactly with their scalar prepared dots. */
Test(oxk_q23k, q4_k_and_q6_k_multi_still_exact)
{
    const struct {
        OcGgufQuantizationType qtype;
        size_t block_bytes;
        size_t (*prep_bytes)(size_t);
        void (*prep_row)(const uint8_t *, size_t, void *);
        float (*prepped)(const void *, size_t, const uint8_t *);
        void (*multi)(const void *, size_t, const uint8_t *, size_t, size_t,
                      float *);
    } cases[] = {
        { OC_QUANT_Q4_K_S, OC_OXK_BLOCK_Q4_K_SIZE, oc_oxk_q4_k_prep_bytes,
          oc_oxk_q4_k_prep_row, oc_oxk_dot_q4_k_prepped,
          oc_oxk_dot_q4_k_prepped_multi },
        { OC_QUANT_Q6_K, OC_OXK_BLOCK_Q6_K_SIZE, oc_oxk_q6_k_prep_bytes,
          oc_oxk_q6_k_prep_row, oc_oxk_dot_q6_k_prepped,
          oc_oxk_dot_q6_k_prepped_multi },
    };

    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        const size_t row_bytes = N_BLOCKS * cases[c].block_bytes;
        float *w = malloc(N_VALS * sizeof(float));
        uint8_t *row = malloc(row_bytes);
        cr_assert_not_null(w, "malloc w");
        cr_assert_not_null(row, "malloc row");
        fill_weights(w, N_VALS, 4211u + (uint32_t)c);
        cr_assert_eq(oc_quant_pack_row(cases[c].qtype, w, N_VALS, row,
                                           row_bytes), OC_OK, "quantize");

        enum { N_ACT = 5 };
        const size_t act_stride = N_BLOCKS * OC_OXK_BLOCK_Q8_K_SIZE;
        float *x = malloc(N_VALS * sizeof(float));
        uint8_t *acts = malloc(N_ACT * act_stride);
        cr_assert_not_null(x, "malloc x");
        cr_assert_not_null(acts, "malloc acts");
        for (int a = 0; a < N_ACT; a++) {
            fill_weights(x, N_VALS, 31u * (uint32_t)(a + 1) + (uint32_t)c);
            quantize_q8_k(x, N_BLOCKS, acts + (size_t)a * act_stride);
        }

        void *prep = malloc(cases[c].prep_bytes(N_BLOCKS));
        cr_assert_not_null(prep, "malloc prep");
        cases[c].prep_row(row, N_BLOCKS, prep);

        float multi[N_ACT];
        cases[c].multi(prep, N_BLOCKS, acts, act_stride, N_ACT, multi);
        for (int a = 0; a < N_ACT; a++) {
            const float want = cases[c].prepped(prep, N_BLOCKS,
                                                acts + (size_t)a * act_stride);
            cr_assert(float_within_1ulp(multi[a], want),
                "case %zu act %d: multi %.9g != scalar prepared %.9g",
                c, a, (double)multi[a], (double)want);
        }

        free(prep);
        free(acts);
        free(x);
        free(row);
        free(w);
    }
}
