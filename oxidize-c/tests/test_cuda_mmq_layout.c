/* test_cuda_mmq_layout.c — host mirror of the device matvec block layouts.
 *
 * The kernels in src/backends/cuda_mmq.cu can only be built by nvcc and only
 * run on a GPU, so their index math is not otherwise reachable from the test
 * suite. That math — which super-block a group lands in, where its scale bits
 * are split, which nibble half maps to which element of x — is exactly the
 * part that silently produces plausible-looking garbage when it is wrong.
 *
 * These tests re-implement the per-group dot products here, byte for byte
 * against the same layout the device code walks, and check them against
 * oc_quant_dequant_row_scalar + a plain dot product. A divergence means the
 * .cu kernel is wrong in the same way, because the two are written from the
 * same layout description.
 *
 * Keep the MIRROR_* functions structurally identical to their mq_*_group_dot
 * counterparts. If one changes, change both.
 */
#include <criterion/criterion.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "oxidize/quant.h"

/* Mirrors MQ_DEV_IQ4_XS_SIZE / MQ_BLOCK_IQ4_XS_SIZE (no device padding). */
#define MIRROR_IQ4_XS_BLOCK 136u
#define MIRROR_QK_K         256u

/* Mirrors cuda_mmq.cu::MQ_KVALUES_IQ4NL, itself a copy of
 * src/compute/quant_tables.h::KVALUES_IQ4NL. */
static const int8_t MIRROR_KVALUES_IQ4NL[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113
};

static float mirror_f16(uint8_t b0, uint8_t b1)
{
    uint32_t bits = (uint32_t)b0 | ((uint32_t)b1 << 8);
    uint32_t sign = (bits >> 15) & 1u;
    uint32_t exp  = (bits >> 10) & 0x1Fu;
    uint32_t frac = bits & 0x03FFu;
    uint32_t f;

    if (exp == 0u) {
        if (frac == 0u) {
            f = sign << 31;
        } else {
            uint32_t fn = frac;
            int32_t e = -14;
            while ((fn & 0x0400u) == 0u) { fn <<= 1; e -= 1; }
            fn &= 0x03FFu;
            f = (sign << 31) | (((uint32_t)(e + 127)) << 23) | (fn << 13);
        }
    } else if (exp == 0x1Fu) {
        f = (sign << 31) | 0x7F800000u | (frac << 13);
    } else {
        int32_t e = (int32_t)exp - 15 + 127;
        f = (sign << 31) | (((uint32_t)e) << 23) | (frac << 13);
    }
    float out;
    memcpy(&out, &f, sizeof(out));
    return out;
}

/* Mirror of mq_iq4_xs_group_dot: contract weight group `g` of `row` against
 * the 32 floats at `x`. */
static float mirror_iq4_xs_group_dot(const uint8_t *row, uint32_t g,
                                     const float *x)
{
    const uint8_t *blk = row + (size_t)(g >> 3) * MIRROR_IQ4_XS_BLOCK;
    const uint32_t ib = g & 7u;

    const float d = mirror_f16(blk[0], blk[1]);
    const uint32_t scales_h = (uint32_t)blk[2] | ((uint32_t)blk[3] << 8);
    const uint8_t *scales_l = blk + 4;

    const int32_t ls_l =
        (int32_t)((scales_l[ib >> 1] >> (4u * (ib & 1u))) & 0x0Fu);
    const int32_t ls_h = (int32_t)((scales_h >> (2u * ib)) & 3u) << 4;
    const float dl = d * (float)((ls_l | ls_h) - 32);

    const uint8_t *qs = blk + 8 + (size_t)ib * 16u;
    float acc = 0.0f;
    for (uint32_t j = 0u; j < 16u; j++) {
        const uint8_t byte = qs[j];
        acc += (float)MIRROR_KVALUES_IQ4NL[byte & 0x0Fu] * x[j];
        acc += (float)MIRROR_KVALUES_IQ4NL[byte >> 4]    * x[j + 16u];
    }
    return dl * acc;
}

/* Mirror of mq_iq4_xs_expand. */
static void mirror_iq4_xs_expand(const uint8_t *blk, float *out)
{
    const float d = mirror_f16(blk[0], blk[1]);
    const uint32_t scales_h = (uint32_t)blk[2] | ((uint32_t)blk[3] << 8);
    const uint8_t *scales_l = blk + 4;
    const uint8_t *qs = blk + 8;

    for (uint32_t ib = 0u; ib < 8u; ib++) {
        const int32_t ls_l =
            (int32_t)((scales_l[ib >> 1] >> (4u * (ib & 1u))) & 0x0Fu);
        const int32_t ls_h = (int32_t)((scales_h >> (2u * ib)) & 3u) << 4;
        const float dl = d * (float)((ls_l | ls_h) - 32);
        const uint8_t *q = qs + (size_t)ib * 16u;
        float *o = out + (size_t)ib * 32u;
        for (uint32_t j = 0u; j < 16u; j++) {
            o[j]       = dl * (float)MIRROR_KVALUES_IQ4NL[q[j] & 0x0Fu];
            o[j + 16u] = dl * (float)MIRROR_KVALUES_IQ4NL[q[j] >> 4];
        }
    }
}

/* Deterministic value generator — no rand() so failures reproduce exactly. */
static float gen(uint32_t i, uint32_t salt)
{
    uint32_t h = i * 2654435761u + salt * 40503u;
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    /* [-2, 2) */
    return ((float)(h & 0xFFFFu) / 16384.0f) - 2.0f;
}

/* Round-trip a row of `cols` values through IQ4_XS. Caller frees *packed. */
static void pack_row(size_t cols, uint32_t salt, uint8_t **packed,
                     size_t *packed_len, float **ref)
{
    float *src = malloc(cols * sizeof(float));
    cr_assert_not_null(src);
    for (size_t i = 0; i < cols; i++) src[i] = gen((uint32_t)i, salt);

    *packed_len = oc_quantized_size(OC_QUANT_IQ4_XS, cols);
    cr_assert_eq(*packed_len, (cols / MIRROR_QK_K) * MIRROR_IQ4_XS_BLOCK,
                 "IQ4_XS packed size should be 136 bytes per 256 values");
    *packed = malloc(*packed_len);
    cr_assert_not_null(*packed);
    cr_assert_eq(oc_quant_pack_row(OC_QUANT_IQ4_XS, src, cols, *packed,
                                   *packed_len), OC_OK);

    *ref = malloc(cols * sizeof(float));
    cr_assert_not_null(*ref);
    cr_assert_eq(oc_quant_dequant_row_scalar(OC_QUANT_IQ4_XS, *packed,
                                             *packed_len, *ref, cols), OC_OK);
    free(src);
}

/* The expansion must reproduce the canonical dequant byte for byte — this is
 * the get_row path (embedding lookup), where any drift is a wrong embedding. */
Test(cuda_mmq_layout, iq4_xs_expand_matches_dequant)
{
    const size_t cols = MIRROR_QK_K * 6;   /* 6 super-blocks */
    uint8_t *packed = NULL;
    float *ref = NULL;
    size_t packed_len = 0;
    pack_row(cols, 7u, &packed, &packed_len, &ref);

    float *got = malloc(cols * sizeof(float));
    cr_assert_not_null(got);
    for (size_t b = 0; b < cols / MIRROR_QK_K; b++) {
        mirror_iq4_xs_expand(packed + b * MIRROR_IQ4_XS_BLOCK,
                             got + b * MIRROR_QK_K);
    }

    for (size_t i = 0; i < cols; i++) {
        cr_assert_float_eq(got[i], ref[i], 1e-6f,
                           "expand mismatch at %zu: got %g want %g",
                           i, (double)got[i], (double)ref[i]);
    }
    free(packed); free(ref); free(got);
}

/* The group dot must agree with dequantize-then-dot. This is the check that
 * catches a mis-split scale or a swapped nibble half: those still produce
 * finite, reasonable-magnitude numbers, so only a reference comparison finds
 * them. */
Test(cuda_mmq_layout, iq4_xs_group_dot_matches_reference)
{
    const size_t cols = MIRROR_QK_K * 5;
    uint8_t *packed = NULL;
    float *ref = NULL;
    size_t packed_len = 0;
    pack_row(cols, 11u, &packed, &packed_len, &ref);

    float *x = malloc(cols * sizeof(float));
    cr_assert_not_null(x);
    for (size_t i = 0; i < cols; i++) x[i] = gen((uint32_t)i, 99u);

    const uint32_t n_groups = (uint32_t)(cols / 32u);
    for (uint32_t g = 0; g < n_groups; g++) {
        const float *xg = x + (size_t)g * 32u;
        float want = 0.0f;
        for (uint32_t j = 0; j < 32u; j++)
            want += ref[(size_t)g * 32u + j] * xg[j];

        const float got = mirror_iq4_xs_group_dot(packed, g, xg);
        /* Tolerance is relative: group sums run to ~1e3 with these inputs. */
        const float tol = 1e-4f * (fabsf(want) + 1.0f);
        cr_assert_float_eq(got, want, tol,
                           "group %u dot mismatch: got %g want %g",
                           g, (double)got, (double)want);
    }
    free(packed); free(ref); free(x);
}

/* Full-row dot, i.e. what one output element of the matvec computes. Sums
 * every group the way the warp reduction does. */
Test(cuda_mmq_layout, iq4_xs_row_dot_matches_reference)
{
    /* 5376 is Gemma 4's hidden size and a real row length in the model. */
    const size_t cols = 5376;
    cr_assert_eq(cols % MIRROR_QK_K, 0,
                 "5376 must be a whole number of IQ4_XS super-blocks");
    uint8_t *packed = NULL;
    float *ref = NULL;
    size_t packed_len = 0;
    pack_row(cols, 23u, &packed, &packed_len, &ref);

    float *x = malloc(cols * sizeof(float));
    cr_assert_not_null(x);
    for (size_t i = 0; i < cols; i++) x[i] = gen((uint32_t)i, 5u);

    float want = 0.0f;
    for (size_t i = 0; i < cols; i++) want += ref[i] * x[i];

    float got = 0.0f;
    for (uint32_t g = 0; g < cols / 32u; g++)
        got += mirror_iq4_xs_group_dot(packed, g, x + (size_t)g * 32u);

    cr_assert_float_eq(got, want, 1e-3f * (fabsf(want) + 1.0f),
                       "row dot mismatch: got %g want %g",
                       (double)got, (double)want);
    free(packed); free(ref); free(x);
}

/* Every IQ4_XS row length in Gemma 4 31B must be a whole number of
 * super-blocks, or oc_cuda_mmq_supported() rejects the tensor and it silently
 * falls back to the f32 upload path — which for this model does not fit in
 * VRAM. Guard the shapes the loader will actually see. */
Test(cuda_mmq_layout, gemma4_row_lengths_are_block_aligned)
{
    /* Row length = ne0 for each IQ4_XS tensor in gemma-4-31B-it-IQ4_XS.gguf:
     * attn_q/attn_k/attn_v/ffn_gate/ffn_up take n_embd; attn_output takes
     * n_head*head_dim (8192 sliding, 16384 global); ffn_down takes n_ff. */
    const size_t row_lengths[] = { 5376, 21504, 8192, 16384, 4096, 2048 };
    for (size_t i = 0; i < sizeof(row_lengths) / sizeof(row_lengths[0]); i++) {
        cr_assert_eq(row_lengths[i] % MIRROR_QK_K, 0,
                     "row length %zu is not a multiple of 256; the IQ4_XS "
                     "device kernel would reject it", row_lengths[i]);
    }
}

/* ─── Q5_K mirror ────────────────────────────────────────────────────────
 *
 * oc_quant_pack_row cannot produce Q5_K, but any random byte pattern is a
 * structurally valid block (every field is a plain bitfield), so random
 * packed blocks + oc_quant_dequant_row_scalar make a reference. Keep this
 * structurally identical to cuda_mmq.cu::mq_q5k_group_dot. */
#define MIRROR_Q5_K_BLOCK 176u

static void mirror_scale_min_k4(uint32_t j, const uint8_t *scales,
                                uint8_t *out_sc, uint8_t *out_m)
{
    if (j < 4u) {
        *out_sc = (uint8_t)(scales[j] & 63u);
        *out_m  = (uint8_t)(scales[j + 4] & 63u);
    } else {
        *out_sc = (uint8_t)((scales[j + 4] & 0x0Fu) | ((scales[j - 4] >> 6) << 4));
        *out_m  = (uint8_t)(((scales[j + 4] >> 4) & 0x0Fu) | ((scales[j] >> 6) << 4));
    }
}

static float mirror_q5k_group_dot(const uint8_t *row, uint32_t g,
                                  const float *x)
{
    const uint8_t *blk = row + (size_t)(g >> 3) * MIRROR_Q5_K_BLOCK;
    const uint32_t gw = g & 7u;

    const float d  = mirror_f16(blk[0], blk[1]);
    const float mn = mirror_f16(blk[2], blk[3]);
    uint8_t sc, m;
    mirror_scale_min_k4(gw, blk + 4, &sc, &m);

    const uint8_t *qh = blk + 16;
    const uint8_t *qs = blk + 48 + (size_t)(gw >> 1) * 32u;
    const int hi = (gw & 1u) != 0u;

    float qx = 0.0f, sx = 0.0f;
    for (uint32_t l = 0u; l < 32u; l++) {
        const uint32_t byte = qs[l];
        const uint32_t nib = hi ? (byte >> 4) : (byte & 0x0Fu);
        const uint32_t q = nib | (((uint32_t)(qh[l] >> gw) & 1u) << 4);
        qx += x[l] * (float)q;
        sx += x[l];
    }
    return d * (float)sc * qx - mn * (float)m * sx;
}

static void q5k_random_row(size_t cols, uint32_t seed, uint8_t **packed,
                           size_t *packed_len, float **ref)
{
    const size_t blocks = cols / MIRROR_QK_K;
    *packed_len = blocks * MIRROR_Q5_K_BLOCK;
    *packed = malloc(*packed_len);
    *ref = malloc(cols * sizeof(float));
    cr_assert_not_null(*packed);
    cr_assert_not_null(*ref);
    uint32_t s = seed * 2654435761u + 1u;
    for (size_t i = 0; i < *packed_len; i++) {
        s = s * 1664525u + 1013904223u;
        (*packed)[i] = (uint8_t)(s >> 24);
    }
    /* Clamp d/dmin f16 exponents so the reference dot stays finite. */
    for (size_t b = 0; b < blocks; b++) {
        uint8_t *blk = *packed + b * MIRROR_Q5_K_BLOCK;
        blk[1] = (uint8_t)(blk[1] & 0x3B); /* zero sign+top exponent bits */
        blk[3] = (uint8_t)(blk[3] & 0x3B);
    }
    cr_assert_eq(oc_quant_dequant_row_scalar(OC_QUANT_Q5_K_M, *packed,
                                             *packed_len, *ref, cols), OC_OK);
}

Test(cuda_mmq_layout, q5k_group_dot_matches_reference)
{
    const size_t cols = MIRROR_QK_K * 5;
    uint8_t *packed = NULL;
    float *ref = NULL;
    size_t packed_len = 0;
    q5k_random_row(cols, 31u, &packed, &packed_len, &ref);

    float *x = malloc(cols * sizeof(float));
    cr_assert_not_null(x);
    for (size_t i = 0; i < cols; i++) x[i] = gen((uint32_t)i, 47u);

    const uint32_t n_groups = (uint32_t)(cols / 32u);
    for (uint32_t g = 0; g < n_groups; g++) {
        const float *xg = x + (size_t)g * 32u;
        float want = 0.0f;
        for (uint32_t j = 0; j < 32u; j++)
            want += ref[(size_t)g * 32u + j] * xg[j];

        const float got = mirror_q5k_group_dot(packed, g, xg);
        const float tol = 1e-4f * (fabsf(want) + 1.0f);
        cr_assert_float_eq(got, want, tol,
                           "group %u dot mismatch: got %g want %g",
                           g, (double)got, (double)want);
    }
    free(packed); free(ref); free(x);
}

#define MIRROR_Q4_0_BLOCK 18u

static float mirror_q4_0_group_dot(const uint8_t *row, uint32_t g,
                                   const float *x)
{
    const uint8_t *blk = row + (size_t)g * MIRROR_Q4_0_BLOCK;
    const float d = mirror_f16(blk[0], blk[1]);
    const uint8_t *qs = blk + 2;
    float acc = 0.0f;
    for (uint32_t i = 0; i < 16u; i++) {
        const uint8_t p = qs[i];
        acc += (float)((int32_t)(p & 0x0Fu) - 8) * x[i];
        acc += (float)((int32_t)(p >> 4) - 8) * x[i + 16u];
    }
    return acc * d;
}

Test(cuda_mmq_layout, q4_0_group_dot_matches_reference)
{
    const size_t cols = 32u * 8u;
    float *src = malloc(cols * sizeof(float));
    cr_assert_not_null(src);
    for (size_t i = 0; i < cols; i++) src[i] = gen((uint32_t)i, 13u);

    size_t packed_len = oc_quantized_size(OC_QUANT_Q4_0, cols);
    uint8_t *packed = malloc(packed_len);
    float *ref = malloc(cols * sizeof(float));
    cr_assert_not_null(packed);
    cr_assert_not_null(ref);
    cr_assert_eq(oc_quant_pack_row(OC_QUANT_Q4_0, src, cols, packed, packed_len),
                 OC_OK);
    cr_assert_eq(oc_quant_dequant_row_scalar(OC_QUANT_Q4_0, packed, packed_len,
                                             ref, cols), OC_OK);

    float *x = malloc(cols * sizeof(float));
    cr_assert_not_null(x);
    for (size_t i = 0; i < cols; i++) x[i] = gen((uint32_t)i, 17u);

    for (uint32_t g = 0; g < cols / 32u; g++) {
        float want = 0.0f;
        for (uint32_t j = 0; j < 32u; j++)
            want += ref[(size_t)g * 32u + j] * x[(size_t)g * 32u + j];
        const float got = mirror_q4_0_group_dot(packed, g, x + (size_t)g * 32u);
        const float tol = 1e-4f * (fabsf(want) + 1.0f);
        cr_assert_float_eq(got, want, tol,
                           "Q4_0 group %u: got %g want %g",
                           g, (double)got, (double)want);
    }
    free(src); free(packed); free(ref); free(x);
}

Test(cuda_mmq_layout, al_family_aliases_standard_block_sizes)
{
    cr_assert_eq(oc_quantized_size(OC_QUANT_AL5, 256),
                 oc_quantized_size(OC_QUANT_Q4_0, 256));
    cr_assert_eq(oc_quantized_size(OC_QUANT_AL8, 256),
                 oc_quantized_size(OC_QUANT_Q8_0, 256));
    cr_assert_eq(oc_quantized_size(OC_QUANT_AL6, 256),
                 oc_quantized_size(OC_QUANT_Q5_0, 256));
    cr_assert_eq(oc_quantized_size(OC_QUANT_AL5_XS, 32), 14u);
}

#define MIRROR_AL5_XS_BLOCK 14u

static float mirror_al5_xs_group_dot(const uint8_t *row, uint32_t g,
                                     const float *x)
{
    const uint8_t *blk = row + (size_t)g * MIRROR_AL5_XS_BLOCK;
    const float d = mirror_f16(blk[0], blk[1]);
    const uint8_t *packed = blk + 2;
    float acc = 0.0f;
    uint32_t bitpos = 0u;
    for (uint32_t i = 0; i < 32u; i++) {
        uint32_t v = 0u;
        for (int b = 0; b < 3; b++) {
            uint32_t byte_idx = bitpos / 8u;
            uint32_t bit_idx = bitpos % 8u;
            if ((packed[byte_idx] >> bit_idx) & 1u) v |= (1u << b);
            bitpos += 1u;
        }
        acc += (float)((int32_t)v - 4) * x[i];
    }
    return acc * d;
}

Test(cuda_mmq_layout, al5_xs_group_dot_matches_reference)
{
    const size_t cols = 32u * 6u;
    float *src = malloc(cols * sizeof(float));
    cr_assert_not_null(src);
    for (size_t i = 0; i < cols; i++) src[i] = gen((uint32_t)i, 41u);

    size_t packed_len = oc_quantized_size(OC_QUANT_AL5_XS, cols);
    uint8_t *packed = malloc(packed_len);
    float *ref = malloc(cols * sizeof(float));
    cr_assert_not_null(packed);
    cr_assert_not_null(ref);
    cr_assert_eq(oc_quant_pack_row(OC_QUANT_AL5_XS, src, cols, packed,
                                   packed_len), OC_OK);
    cr_assert_eq(oc_quant_dequant_row_scalar(OC_QUANT_AL5_XS, packed, packed_len,
                                             ref, cols), OC_OK);

    float *x = malloc(cols * sizeof(float));
    cr_assert_not_null(x);
    for (size_t i = 0; i < cols; i++) x[i] = gen((uint32_t)i, 3u);

    for (uint32_t g = 0; g < cols / 32u; g++) {
        float want = 0.0f;
        for (uint32_t j = 0; j < 32u; j++)
            want += ref[(size_t)g * 32u + j] * x[(size_t)g * 32u + j];
        const float got = mirror_al5_xs_group_dot(packed, g,
                                                  x + (size_t)g * 32u);
        const float tol = 1e-4f * (fabsf(want) + 1.0f);
        cr_assert_float_eq(got, want, tol,
                           "AL5_XS group %u: got %g want %g",
                           g, (double)got, (double)want);
    }
    free(src); free(packed); free(ref); free(x);
}
