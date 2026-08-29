#include "framework.h"

#include "oxidize/matvec.h"
#include "oxidize/parallel.h"
#include "oxidize/quant.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static float fused_frand(uint32_t *state)
{
    *state = *state * 1664525u + 1013904223u;
    return (float)((int32_t)(*state >> 8) % 2001 - 1000) / 1000.0f;
}

typedef struct {
    OcGgufQuantizationType qtype;
    size_t rows;
    size_t row_bytes;
    uint8_t *data;
    float *out;
    float *reference;
} FusedPart;

static void fused_part_init(FusedPart *part, OcGgufQuantizationType qtype,
                            size_t rows, size_t cols, uint32_t *seed)
{
    part->qtype = qtype;
    part->rows = rows;
    part->row_bytes = oc_quantized_size(qtype, cols);
    part->data = rows ? malloc(rows * part->row_bytes) : NULL;
    part->out = rows ? calloc(rows, sizeof(*part->out)) : NULL;
    part->reference = rows ? calloc(rows, sizeof(*part->reference)) : NULL;
    cr_assert_gt(part->row_bytes, 0);
    cr_assert(rows == 0 || (part->data && part->out && part->reference));

    float *src = malloc(cols * sizeof(*src));
    cr_assert_not_null(src);
    for (size_t r = 0; r < rows; r++) {
        for (size_t i = 0; i < cols; i++) src[i] = fused_frand(seed);
        cr_assert_eq(oc_quant_pack_row(qtype, src, cols,
                                       part->data + r * part->row_bytes,
                                       part->row_bytes), OC_OK);
    }
    free(src);
}

static void fused_part_free(FusedPart *part)
{
    free(part->data);
    free(part->out);
    free(part->reference);
}

static void run_fused_case(const OcGgufQuantizationType *qtypes,
                           size_t n_parts, const size_t *row_counts,
                           size_t cols, size_t threads,
                           OcMatvecFusedTestStats expected)
{
    FusedPart parts[4] = {{0}};
    const uint8_t *datas[4];
    size_t rows[4], strides[4];
    float *outs[4];
    float *input = malloc(cols * sizeof(*input));
    float *temp = malloc(cols * sizeof(*temp));
    cr_assert_not_null(input);
    cr_assert_not_null(temp);

    uint32_t seed = 0xC0FFEEu;
    for (size_t i = 0; i < cols; i++) input[i] = fused_frand(&seed);
    for (size_t k = 0; k < n_parts; k++) {
        fused_part_init(&parts[k], qtypes[k], row_counts[k], cols, &seed);
        datas[k] = parts[k].data;
        rows[k] = parts[k].rows;
        strides[k] = parts[k].row_bytes;
        outs[k] = parts[k].out;
        oc_matvec_quantized(qtypes[k], datas[k], rows[k], cols, strides[k],
                            input, parts[k].reference, temp);
    }

    cr_assert_eq(oc_parallel_set_threads(threads), OC_OK);
    oc_matvec_fused_test_reset();
    oc_matvec_quantized_fused(qtypes, datas, rows, cols, strides, n_parts,
                               input, outs, temp);
    oc_parallel_set_threads(1);

    for (size_t k = 0; k < n_parts; k++) {
        if (rows[k] != 0) {
            cr_assert_arr_eq(parts[k].out, parts[k].reference,
                             rows[k] * sizeof(*parts[k].out),
                             "part %zu differs from separate matvec", k);
        }
    }
    const OcMatvecFusedTestStats actual = oc_matvec_fused_test_stats();
    cr_assert_eq(actual.activation_quantizations,
                 expected.activation_quantizations);
    cr_assert_eq(actual.parallel_dispatches, expected.parallel_dispatches);
    cr_assert_eq(actual.fallback_calls, expected.fallback_calls);

    for (size_t k = 0; k < n_parts; k++) fused_part_free(&parts[k]);
    free(input);
    free(temp);
}

Test(matvec, fused_multi_projection_q4k)
{
    const OcGgufQuantizationType qtypes[] = {
        OC_QUANT_Q4_K_M, OC_QUANT_Q4_K_M, OC_QUANT_Q4_K_M,
    };
    const size_t rows[] = {31, 0, 47};
    const OcMatvecFusedTestStats expected = {1, 1, 0};
    run_fused_case(qtypes, 3, rows, 256, 1, expected);
    run_fused_case(qtypes, 3, rows, 256, 16, expected);
}

Test(matvec, fused_multi_projection_q5k_q6k)
{
    const OcGgufQuantizationType q5[] = { OC_QUANT_Q5_K_M, OC_QUANT_Q5_K_M };
    const OcGgufQuantizationType q6[] = { OC_QUANT_Q6_K, OC_QUANT_Q6_K };
    const size_t rows[] = {19, 43};
    const OcMatvecFusedTestStats expected = {1, 1, 0};
    run_fused_case(q5, 2, rows, 256, 1, expected);
    run_fused_case(q5, 2, rows, 256, 16, expected);
    run_fused_case(q6, 2, rows, 256, 1, expected);
    run_fused_case(q6, 2, rows, 256, 16, expected);
}

Test(matvec, fused_multi_projection_mixed_type_fallback)
{
    const OcGgufQuantizationType qtypes[] = { OC_QUANT_Q4_K_M, OC_QUANT_Q5_K_M };
    const size_t rows[] = {23, 37};
    const OcMatvecFusedTestStats expected = {2, 2, 2};
    run_fused_case(qtypes, 2, rows, 256, 1, expected);
    run_fused_case(qtypes, 2, rows, 256, 16, expected);
}

Test(matvec, fused_iq1_xxxs)
{
    uint8_t row[38] = {0};
    float input[256];
    float output = 0.0f;
    float temp[256];
    const uint8_t *datas[] = {row};
    const size_t rows[] = {1};
    const size_t strides[] = {sizeof(row)};
    const OcGgufQuantizationType qtypes[] = {OC_QUANT_IQ1_XXXS};
    float *outputs[] = {&output};

    row[1] = 0x3c;
    for (size_t i = 0; i < 256; i++) input[i] = 1.0f;
    oc_matvec_fused_test_reset();
    oc_matvec_quantized_fused(qtypes, datas, rows, 256, strides, 1,
                              input, outputs, temp);
    cr_assert_float_eq(output, -224.0f, 0.0f);
    const OcMatvecFusedTestStats stats = oc_matvec_fused_test_stats();
    cr_assert_eq(stats.activation_quantizations, 1);
    cr_assert_eq(stats.parallel_dispatches, 1);
    cr_assert_eq(stats.fallback_calls, 0);
}
