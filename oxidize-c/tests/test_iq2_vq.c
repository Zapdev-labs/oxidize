#include <criterion/criterion.h>

#include "oxidize/iq2_vq.h"
#include "oxidize/matvec.h"
#include "oxidize/oxk.h"
#include "oxidize/quant.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

Test(iq2_vq, block_layout_and_ggml_id)
{
    const OcQuantBlockLayout layout = oc_quant_block_size(OC_QUANT_IQ2_VQ);
    cr_assert_eq(layout.elements_per_block, 128u);
    cr_assert_eq(layout.bytes_per_block, 38u);
    cr_assert_eq(oc_quantized_size(OC_QUANT_IQ2_VQ, 128u), 38u);
    cr_assert_eq(oc_quantized_size(OC_QUANT_IQ2_VQ, 127u), 0u);
    cr_assert(oc_quant_is_runtime_supported(OC_QUANT_IQ2_VQ));
    cr_assert_eq(oc_quant_type_from_ggml_id(70), OC_QUANT_IQ2_VQ);
    cr_assert_eq(oc_quant_type_to_ggml_id(OC_QUANT_IQ2_VQ), 70u);
    cr_assert_str_eq(oc_quant_type_name(OC_QUANT_IQ2_VQ), "IQ2_VQ");
}

Test(iq2_vq, codebook_fingerprint)
{
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < OC_IQ2_VQ_CODEBOOK_VALUES; i++) {
        hash ^= (uint8_t)oc_iq2_vq_codebook[i];
        hash *= UINT64_C(1099511628211);
    }
    cr_assert_eq(hash, UINT64_C(0xac93d9e72a53c9e5));
}

Test(iq2_vq, dequant_all_books_and_indexes)
{
    uint8_t blocks[OC_IQ2_VQ_CODEBOOK_COUNT * 8u * OC_IQ2_VQ_BLOCK_BYTES];
    float values[OC_IQ2_VQ_CODEBOOK_COUNT * 8u * OC_IQ2_VQ_BLOCK_ELEMENTS];
    for (size_t book = 0; book < OC_IQ2_VQ_CODEBOOK_COUNT; book++) {
        for (size_t b = 0; b < 8; b++) {
            uint8_t *block = blocks + (book * 8u + b) * OC_IQ2_VQ_BLOCK_BYTES;
            block[0] = 0x00;
            block[1] = 0x3c;
            for (size_t sub = 0; sub < 4; sub++)
                block[2 + sub] = (uint8_t)((book << 6) | 33u);
            for (size_t g = 0; g < 32; g++)
                block[6 + g] = (uint8_t)(b * 32u + g);
        }
    }
    const size_t n_bytes = sizeof(blocks);
    cr_assert_eq(oc_quant_dequant_row(OC_QUANT_IQ2_VQ, blocks, n_bytes,
                                      values, sizeof(values) / sizeof(values[0])),
                 OC_OK);
    for (size_t book = 0; book < OC_IQ2_VQ_CODEBOOK_COUNT; book++) {
        for (size_t b = 0; b < 8; b++) {
            const float *decoded = values + (book * 8u + b) * 128u;
            for (size_t i = 0; i < 32; i++) {
                const size_t code = b * 32u + i;
                for (size_t j = 0; j < 4; j++)
                    cr_assert_eq(decoded[i * 4u + j],
                        (float)oc_iq2_vq_codebook[book * 1024u + code * 4u + j]);
            }
        }
    }
}

Test(iq2_vq, rejects_bad_lengths_and_has_no_encoder)
{
    uint8_t block[OC_IQ2_VQ_BLOCK_BYTES] = {0};
    float values[128];
    float input[128] = {0};
    uint8_t encoded[OC_IQ2_VQ_BLOCK_BYTES];
    cr_assert_eq(oc_quant_dequant_row(OC_QUANT_IQ2_VQ, block, 37, values, 128),
                 OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_quant_dequant_row(OC_QUANT_IQ2_VQ, block, 38, values, 127),
                 OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_quant_pack_row(OC_QUANT_IQ2_VQ, input, 128, encoded,
                                   sizeof(encoded)), OC_ERR_QUANT);
}

Test(iq2_vq, q8_0_dot_matches_independent_reference)
{
    uint8_t row[OC_IQ2_VQ_BLOCK_BYTES] = {0};
    uint8_t act[4u * OC_OXK_BLOCK_Q8_0_SIZE] = {0};
    row[0] = 0x00;
    row[1] = 0x3c;
    for (size_t sub = 0; sub < 4; sub++) {
        row[2 + sub] = (uint8_t)((sub << 6) | (25u + sub));
        for (size_t g = 0; g < 8; g++) row[6 + sub * 8u + g] = (uint8_t)(g * 31u + sub * 17u);
        uint8_t *ab = act + sub * OC_OXK_BLOCK_Q8_0_SIZE;
        ab[0] = 0x00;
        ab[1] = 0x38;
        for (size_t i = 0; i < 32; i++) ((int8_t *)(ab + 2))[i] = (int8_t)((int)i * 7 - 101);
    }
    float expected = 0.0f;
    for (size_t sub = 0; sub < 4; sub++) {
        const uint8_t scale = row[2 + sub];
        const size_t book = (size_t)(scale >> 6) * 1024u;
        const uint8_t *ab = act + sub * OC_OXK_BLOCK_Q8_0_SIZE;
        int32_t sumi = 0;
        for (size_t g = 0; g < 8; g++) {
            for (size_t j = 0; j < 4; j++) {
                const int8_t w = oc_iq2_vq_codebook[
                    book + (size_t)row[6 + sub * 8u + g] * 4u + j];
                const int8_t a = ((const int8_t *)(ab + 2))[g * 4u + j];
                sumi += (int32_t)w * (int32_t)a;
            }
        }
        const float d = oc_oxk_f16_le_to_f32(row);
        const float ad = oc_oxk_f16_le_to_f32(ab);
        expected += d * (float)((int)(scale & 0x3fu) - 32) * ad * (float)sumi;
    }
    const float actual = oc_oxk_dot_iq2_vq_q8_0_scalar(row, 4, act);
    cr_assert(memcmp(&actual, &expected, sizeof(actual)) == 0,
              "dot mismatch: %.9g vs %.9g", actual, expected);
}

Test(iq2_vq, fused_matvec_uses_q8_0_dot)
{
    const size_t cols = 128, rows = 2;
    const size_t row_bytes = OC_IQ2_VQ_BLOCK_BYTES;
    uint8_t weights[2u * OC_IQ2_VQ_BLOCK_BYTES] = {0};
    uint8_t act[4u * OC_OXK_BLOCK_Q8_0_SIZE] = {0};
    float input[cols];
    float output[rows], temp[cols];
    for (size_t r = 0; r < rows; r++) {
        uint8_t *row = weights + r * row_bytes;
        row[0] = 0x00;
        row[1] = 0x3c;
        for (size_t sub = 0; sub < 4; sub++) {
            row[2 + sub] = (uint8_t)((sub << 6) | 33u);
            for (size_t g = 0; g < 8; g++)
                row[6 + sub * 8u + g] = (uint8_t)(r * 29u + sub * 61u + g * 7u);
        }
    }
    for (size_t i = 0; i < cols; i++) input[i] = sinf((float)i * 0.17f);
    for (size_t b = 0; b < 4; b++) {
        const float *x = input + b * 32u;
        float amax = 0.0f;
        for (size_t i = 0; i < 32; i++) if (fabsf(x[i]) > amax) amax = fabsf(x[i]);
        const float d = amax / 127.0f;
        cr_assert_eq(oc_quant_pack_row(OC_QUANT_F16, &d, 1, act + b * 34u, 2), OC_OK);
        const float id = d == 0.0f ? 0.0f : 1.0f / d;
        int8_t *q = (int8_t *)(act + b * 34u + 2);
        for (size_t i = 0; i < 32; i++) q[i] = (int8_t)lrintf(x[i] * id);
    }
    oc_matvec_set_fused(true);
    oc_matvec_quantized(OC_QUANT_IQ2_VQ, weights, rows, cols, row_bytes,
                        input, output, temp);
    for (size_t r = 0; r < rows; r++) {
        const float expected = oc_oxk_dot_iq2_vq_q8_0_scalar(
            weights + r * row_bytes, 4, act);
        cr_assert(memcmp(&output[r], &expected, sizeof(expected)) == 0,
                  "row %zu: %.9g vs %.9g", r, output[r], expected);
    }
}

static void make_iq2_vq_row(uint8_t *row, size_t cols, size_t row_index)
{
    static const uint8_t sub_scales[4] = {0u, 32u, 63u, 1u};
    const size_t n_blocks = cols / 128u;
    for (size_t b = 0; b < n_blocks; b++) {
        uint8_t *block = row + b * OC_IQ2_VQ_BLOCK_BYTES;
        block[0] = 0x00;
        block[1] = 0x3c;
        for (size_t sub = 0; sub < 4; sub++) {
            const size_t book = (b + sub + row_index) % OC_IQ2_VQ_CODEBOOK_COUNT;
            block[2 + sub] = (uint8_t)((book << 6) | sub_scales[(b + sub) % 4]);
            for (size_t g = 0; g < 8; g++)
                block[6 + sub * 8u + g] =
                    (uint8_t)(b * 73u + sub * 31u + g * 19u + row_index * 11u);
        }
    }
}

static void make_known_q8_0(uint8_t *act, size_t cols)
{
    const size_t n_blocks = cols / 32u;
    for (size_t b = 0; b < n_blocks; b++) {
        uint8_t *block = act + b * OC_OXK_BLOCK_Q8_0_SIZE;
        block[0] = 0x00;
        block[1] = 0x38;
        int8_t *q = (int8_t *)(block + 2);
        for (size_t i = 0; i < 32; i++)
            q[i] = (int8_t)((int)((b * 29u + i * 13u) % 255u) - 127);
    }
}

static void assert_iq2_vq_dot_matches_dequant(const uint8_t *row,
                                               size_t cols,
                                               const uint8_t *act)
{
    const size_t row_bytes = cols / 128u * OC_IQ2_VQ_BLOCK_BYTES;
    const size_t act_bytes = cols / 32u * OC_OXK_BLOCK_Q8_0_SIZE;
    float *weights = malloc(cols * sizeof(float));
    float *activations = malloc(cols * sizeof(float));
    cr_assert(weights && activations);
    cr_assert_eq(oc_quant_dequant_row(OC_QUANT_IQ2_VQ, row, row_bytes,
                                      weights, cols), OC_OK);
    cr_assert_eq(oc_quant_dequant_row(OC_QUANT_Q8_0, act, act_bytes,
                                      activations, cols), OC_OK);
    double expected = 0.0;
    double magnitude = 0.0;
    for (size_t i = 0; i < cols; i++) {
        const double product = (double)weights[i] * activations[i];
        expected += product;
        magnitude += fabs(product);
    }
    const size_t activation_blocks = cols / 32u;
    cr_assert_eq(activation_blocks % 4u, 0u);
    const float actual = oc_oxk_dot_iq2_vq_q8_0_scalar(row, activation_blocks, act);
    cr_assert(fabs((double)actual - expected) <= magnitude * 1e-5 + 1e-4,
              "cols=%zu: %.9g vs %.12g (magnitude %.9g)", cols,
              actual, expected, magnitude);
    free(weights);
    free(activations);
}

Test(iq2_vq, multiblock_dot_640_and_1024_matches_q8_0_dequant)
{
    static const size_t widths[] = {640u, 1024u};
    for (size_t w = 0; w < sizeof(widths) / sizeof(widths[0]); w++) {
        const size_t cols = widths[w];
        const size_t weight_bytes = cols / 128u * OC_IQ2_VQ_BLOCK_BYTES;
        const size_t act_bytes = cols / 32u * OC_OXK_BLOCK_Q8_0_SIZE;
        uint8_t *row = malloc(weight_bytes);
        uint8_t *act = malloc(act_bytes);
        cr_assert(row && act);
        make_iq2_vq_row(row, cols, w);
        make_known_q8_0(act, cols);
        assert_iq2_vq_dot_matches_dequant(row, cols, act);
        free(row);
        free(act);
    }
}

static void store_le16(uint8_t *dst, uint16_t value)
{
    dst[0] = (uint8_t)value;
    dst[1] = (uint8_t)(value >> 8);
}

static uint32_t rng_next(uint32_t *state)
{
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

static uint16_t finite_half(uint32_t *state)
{
    static const uint16_t values[] = {
        0x0000u, 0x0001u, 0x03ffu, 0x0400u, 0x2800u, 0x3555u,
        0x3c00u, 0x4200u, 0x7bffu, 0x8001u, 0xbc00u, 0xfbffu,
    };
    return values[rng_next(state) % (sizeof(values) / sizeof(values[0]))];
}

static void make_iq2_vq_parity_case(uint8_t *row, uint8_t *act,
                                    size_t cols, uint32_t *state)
{
    static const uint8_t edge_scales[4] = {0u, 32u, 63u, 255u};
    for (size_t b = 0; b < cols / 128u; b++) {
        uint8_t *block = row + b * OC_IQ2_VQ_BLOCK_BYTES;
        store_le16(block, finite_half(state));
        for (size_t sub = 0; sub < 4; sub++) {
            block[2 + sub] = b == 0 ? edge_scales[sub]
                                      : (uint8_t)rng_next(state);
            for (size_t g = 0; g < 8; g++)
                block[6 + sub * 8u + g] = (uint8_t)rng_next(state);
        }
    }
    for (size_t b = 0; b < cols / 32u; b++) {
        uint8_t *block = act + b * OC_OXK_BLOCK_Q8_0_SIZE;
        store_le16(block, finite_half(state));
        int8_t *q = (int8_t *)(block + 2);
        for (size_t i = 0; i < 32; i++) q[i] = (int8_t)rng_next(state);
        q[0] = INT8_MIN;
        q[1] = INT8_MAX;
    }
}

Test(iq2_vq, avx2_dispatch_bit_exact_random_rows)
{
    const OcOxkCaps *caps = oc_oxk_caps();
    if ((caps->level != OC_OXK_AVX2 && caps->level != OC_OXK_AVX512) ||
        !caps->has_fma || !caps->has_f16c)
        cr_skip("host has no AVX2, FMA, and F16C");

    static const size_t widths[] = {640u, 1024u};
    uint32_t state = 0xa8f7c3d1u;
    for (size_t w = 0; w < sizeof(widths) / sizeof(widths[0]); w++) {
        const size_t cols = widths[w];
        const size_t row_bytes = cols / 128u * OC_IQ2_VQ_BLOCK_BYTES;
        const size_t act_bytes = cols / 32u * OC_OXK_BLOCK_Q8_0_SIZE;
        const size_t activation_blocks = cols / 32u;
        uint8_t *row = malloc(row_bytes);
        uint8_t *act = malloc(act_bytes);
        cr_assert(row && act);
        cr_assert_eq(activation_blocks % 4u, 0u);
        for (size_t n = 0; n < 1000u; n++) {
            make_iq2_vq_parity_case(row, act, cols, &state);
            const float scalar = oc_oxk_dot_iq2_vq_q8_0_scalar(
                row, activation_blocks, act);
            const float avx2 = oc_oxk_dot_iq2_vq_q8_0_avx2(
                row, activation_blocks, act);
            const float dispatched = oc_oxk_dot_iq2_vq_q8_0(
                row, activation_blocks, act);
            cr_assert(memcmp(&scalar, &avx2, sizeof(scalar)) == 0,
                      "width=%zu row=%zu scalar=%.9g avx2=%.9g",
                      cols, n, scalar, avx2);
            cr_assert(memcmp(&scalar, &dispatched, sizeof(scalar)) == 0,
                      "width=%zu row=%zu scalar=%.9g dispatched=%.9g",
                      cols, n, scalar, dispatched);
        }
        free(row);
        free(act);
    }
}

static void make_q8_0_for_f32(const float *input, size_t cols, uint8_t *act)
{
    for (size_t b = 0; b < cols / 32u; b++) {
        const float *x = input + b * 32u;
        float amax = 0.0f;
        for (size_t i = 0; i < 32; i++)
            if (fabsf(x[i]) > amax) amax = fabsf(x[i]);
        const float d = amax / 127.0f;
        uint8_t *block = act + b * OC_OXK_BLOCK_Q8_0_SIZE;
        cr_assert_eq(oc_quant_pack_row(OC_QUANT_F16, &d, 1, block, 2), OC_OK);
        const float id = d == 0.0f ? 0.0f : 1.0f / d;
        int8_t *q = (int8_t *)(block + 2);
        for (size_t i = 0; i < 32; i++) q[i] = (int8_t)lrintf(x[i] * id);
    }
}

Test(iq2_vq, multiblock_matvec_640_and_1024_matches_dequant)
{
    static const size_t widths[] = {640u, 1024u};
    const size_t rows = 2;
    oc_matvec_set_fused(true);
    for (size_t w = 0; w < sizeof(widths) / sizeof(widths[0]); w++) {
        const size_t cols = widths[w];
        const size_t row_bytes = cols / 128u * OC_IQ2_VQ_BLOCK_BYTES;
        const size_t act_bytes = cols / 32u * OC_OXK_BLOCK_Q8_0_SIZE;
        uint8_t *weights = malloc(rows * row_bytes);
        uint8_t *act = malloc(act_bytes);
        float *input = malloc(cols * sizeof(float));
        float *temp = malloc(cols * sizeof(float));
        float output[rows];
        float *decoded_weights = malloc(cols * sizeof(float));
        float *decoded_act = malloc(cols * sizeof(float));
        cr_assert(weights && act && input && temp && decoded_weights && decoded_act);
        for (size_t r = 0; r < rows; r++)
            make_iq2_vq_row(weights + r * row_bytes, cols, r + w);
        for (size_t i = 0; i < cols; i++)
            input[i] = sinf((float)i * 0.173f) * (1.0f + (float)(i % 7u) * 0.1f);
        make_q8_0_for_f32(input, cols, act);
        oc_matvec_quantized(OC_QUANT_IQ2_VQ, weights, rows, cols, row_bytes,
                            input, output, temp);
        cr_assert_eq(oc_quant_dequant_row(OC_QUANT_Q8_0, act, act_bytes,
                                          decoded_act, cols), OC_OK);
        for (size_t r = 0; r < rows; r++) {
            cr_assert_eq(oc_quant_dequant_row(OC_QUANT_IQ2_VQ,
                weights + r * row_bytes, row_bytes, decoded_weights, cols), OC_OK);
            double expected = 0.0;
            double magnitude = 0.0;
            for (size_t i = 0; i < cols; i++) {
                const double product = (double)decoded_weights[i] * decoded_act[i];
                expected += product;
                magnitude += fabs(product);
            }
            cr_assert(fabs((double)output[r] - expected) <= magnitude * 1e-5 + 1e-4,
                      "cols=%zu row=%zu: %.9g vs %.12g (magnitude %.9g)",
                      cols, r, output[r], expected, magnitude);
        }
        free(weights);
        free(act);
        free(input);
        free(temp);
        free(decoded_weights);
        free(decoded_act);
    }
}

Test(iq2_vq, batched_prefill_preserves_padded_strides_and_dot_results)
{
    enum { ROWS = 2, VECS = 3, OUT_STRIDE = 5 };
    const size_t widths[] = {640, 1024};
    oc_matvec_set_fused(true);
    for (size_t w = 0; w < 2; w++) {
        const size_t cols = widths[w], in_stride = cols+8;
        const size_t row_bytes = cols/128*OC_IQ2_VQ_BLOCK_BYTES;
        uint8_t *weights = malloc(ROWS*row_bytes);
        uint8_t *act = malloc(cols/32*OC_OXK_BLOCK_Q8_0_SIZE);
        float *input = malloc(VECS*in_stride*sizeof(float));
        float *temp = malloc(cols*sizeof(float));
        const size_t scratch_bytes = oc_matvec_batch_scratch_bytes(cols);
        uint8_t *scratch = malloc(scratch_bytes);
        float output[VECS*OUT_STRIDE];
        cr_assert(weights && act && input && temp && scratch);
        for (size_t r = 0; r < ROWS; r++)
            make_iq2_vq_row(weights+r*row_bytes, cols, r);
        for (size_t v = 0; v < VECS; v++)
            for (size_t i = 0; i < in_stride; i++)
                input[v*in_stride+i] = i < cols
                    ? sinf((float)(i+v*17)*0.13f) : 1234.0f;
        for (size_t i = 0; i < VECS*OUT_STRIDE; i++) output[i] = -1234;
        oc_matvec_quantized_batch(OC_QUANT_IQ2_VQ, weights, ROWS, cols,
            row_bytes, input, in_stride, output, OUT_STRIDE, VECS,
            temp, scratch, scratch_bytes);
        for (size_t v = 0; v < VECS; v++) {
            make_q8_0_for_f32(input+v*in_stride, cols, act);
            for (size_t r = 0; r < ROWS; r++) {
                const float expected = oc_oxk_dot_iq2_vq_q8_0_scalar(
                    weights+r*row_bytes, cols/32, act);
                cr_assert_eq(memcmp(output+v*OUT_STRIDE+r, &expected,
                                     sizeof(expected)), 0);
            }
            for (size_t r = ROWS; r < OUT_STRIDE; r++)
                cr_assert_eq(output[v*OUT_STRIDE+r], -1234);
        }
        free(weights);
        free(act);
        free(input);
        free(temp);
        free(scratch);
    }
}
