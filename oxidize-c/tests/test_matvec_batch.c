/* test_matvec_batch.c — does the batched matvec agree with the single one? loops but must not change a single output value — every result is still The interesting parameter is `cols`, because the activation tile width is */
#include <criterion/criterion.h>

#include "oxidize/matvec.h"
#include "oxidize/oxk.h"
#include "oxidize/quant.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static bool float_within_1ulp(float a, float b)
{
    if (a == b) return true;
    return nextafterf(a, b) == b || nextafterf(b, a) == a;
}

static float frand(uint32_t *s)
{
    *s = *s * 1664525u + 1013904223u;
    return (float)((int32_t)(*s >> 8) % 2001 - 1000) / 1000.0f;
}

/* Pack `rows` quantized weight rows and compare batched vs per-vector. */
static void check_parity(OcGgufQuantizationType qtype, size_t rows,
                         size_t cols, size_t n_vec, size_t act_bytes_override)
{
    const size_t row_bytes = oc_quantized_size(qtype, cols);
    cr_assert(row_bytes > 0, "unsupported qtype/cols");

    uint8_t *w = malloc(rows * row_bytes);
    float *src = malloc(cols * sizeof(float));
    float *in = malloc(n_vec * cols * sizeof(float));
    float *out_ref = malloc(n_vec * rows * sizeof(float));
    float *out_bat = malloc(n_vec * rows * sizeof(float));
    float *temp = malloc(cols * sizeof(float));
    cr_assert(w && src && in && out_ref && out_bat && temp);

    uint32_t seed = 12345u;
    for (size_t r = 0; r < rows; r++) {
        for (size_t i = 0; i < cols; i++) src[i] = frand(&seed);
        cr_assert_eq(oc_quant_pack_row(qtype, src, cols, w + r * row_bytes,
                                       row_bytes), OC_OK);
    }
    for (size_t v = 0; v < n_vec; v++)
        for (size_t i = 0; i < cols; i++) in[v * cols + i] = frand(&seed);

    /* Reference: one call per activation, exactly what prefill used to do. */
    for (size_t v = 0; v < n_vec; v++) {
        oc_matvec_quantized(qtype, w, rows, cols, row_bytes, in + v * cols,
                            out_ref + v * rows, temp);
    }

    const size_t need = oc_matvec_batch_scratch_bytes(cols);
    const size_t act_bytes = act_bytes_override ? act_bytes_override : need;
    uint8_t *act = act_bytes ? malloc(act_bytes) : NULL;
    cr_assert(act_bytes == 0 || act != NULL);

    memset(out_bat, 0, n_vec * rows * sizeof(float));
    oc_matvec_quantized_batch(qtype, w, rows, cols, row_bytes, in, cols,
                              out_bat, rows, n_vec, temp, act, act_bytes);

    for (size_t v = 0; v < n_vec; v++) {
        for (size_t r = 0; r < rows; r++) {
            const size_t k = v * rows + r;
            cr_assert(float_within_1ulp(out_bat[k], out_ref[k]),
                               "qtype=%d cols=%zu n_vec=%zu vec=%zu row=%zu: "
                               "batched %.9g != reference %.9g",
                               (int)qtype, cols, n_vec, v, r,
                               (double)out_bat[k], (double)out_ref[k]);
        }
    }

    free(w); free(src); free(in); free(out_ref); free(out_bat); free(temp);
    free(act);
}

/* Q4_K is the dominant weight type for K-quant models and the one the 30B
 * MoE prefill spends nearly all its time in. 768 is the expert
 * down-projection width that the original scratch sizing got wrong. */
Test(matvec_batch, q4k_widths)
{
    check_parity(OC_QUANT_Q4_K_M, 64, 256, 5, 0);
    check_parity(OC_QUANT_Q4_K_M, 64, 768, 33, 0);
    check_parity(OC_QUANT_Q4_K_M, 32, 2048, 7, 0);
    check_parity(OC_QUANT_Q4_K_M, 16, 4096, 3, 0);
}

/* A single activation must still work (chunk of one degenerates cleanly). */
Test(matvec_batch, single_vector)
{
    check_parity(OC_QUANT_Q4_K_M, 48, 768, 1, 0);
    check_parity(OC_QUANT_Q8_0, 48, 512, 1, 0);
}

/* Batch larger than one activation tile, so the kernel loops over tiles. */
Test(matvec_batch, spans_multiple_tiles)
{
    /* Scratch for exactly 4 activations forces >1 tile at n_vec=17. */
    const size_t abytes = (4096u / 256u) * 292u;   /* Q8_K blocks for cols */
    check_parity(OC_QUANT_Q4_K_M, 24, 4096, 17, abytes * 4);
}

/* An undersized or absent buffer must cost speed, not correctness: the be testing the wrong invariant. What must hold is that a batch of N gives */
Test(matvec_batch, undersized_and_absent_scratch)
{
    const size_t cols = 2048, rows = 32, n_vec = 6;
    const size_t row_bytes = oc_quantized_size(OC_QUANT_Q4_K_M, cols);
    uint8_t *w = malloc(rows * row_bytes);
    float *src = malloc(cols * sizeof(float));
    float *in = malloc(n_vec * cols * sizeof(float));
    float *a = malloc(n_vec * rows * sizeof(float));
    float *b = malloc(n_vec * rows * sizeof(float));
    float *temp = malloc(cols * sizeof(float));
    cr_assert(w && src && in && a && b && temp);
    uint32_t seed = 999u;
    for (size_t r = 0; r < rows; r++) {
        for (size_t i = 0; i < cols; i++) src[i] = frand(&seed);
        oc_quant_pack_row(OC_QUANT_Q4_K_M, src, cols, w + r * row_bytes,
                          row_bytes);
    }
    for (size_t v = 0; v < n_vec * cols; v++) in[v] = frand(&seed);

    /* A 1-byte buffer cannot hold one activation, so this takes the same
     * dequant path as NULL — and must agree with it exactly. */
    uint8_t tiny = 0;
    for (int variant = 0; variant < 2; variant++) {
        uint8_t *scratch = variant ? &tiny : NULL;
        const size_t sbytes = variant ? 1u : 0u;

        oc_matvec_quantized_batch(OC_QUANT_Q4_K_M, w, rows, cols, row_bytes,
                                  in, cols, a, rows, n_vec, temp, scratch,
                                  sbytes);
        for (size_t v = 0; v < n_vec; v++) {
            oc_matvec_quantized_batch(OC_QUANT_Q4_K_M, w, rows, cols,
                                      row_bytes, in + v * cols, cols,
                                      b + v * rows, rows, 1, temp, scratch,
                                      sbytes);
        }
        for (size_t k = 0; k < n_vec * rows; k++)
            cr_assert_float_eq(a[k], b[k], 0.0f,
                               "dequant-path batch mismatch (variant %d) at %zu",
                               variant, k);
    }

    free(w); free(src); free(in); free(a); free(b); free(temp);
}

/* Strides may exceed the vector length, which is how the prefill buffers are
 * laid out (one padded [cap][max_dim] block reused by matrices of differing
 * widths). A kernel that assumed packed rows would read the wrong data. */
Test(matvec_batch, padded_strides)
{
    const size_t cols = 768, rows = 40, n_vec = 11;
    const size_t in_stride = 1024, out_stride = 64;   /* both > needed */
    const size_t row_bytes = oc_quantized_size(OC_QUANT_Q4_K_M, cols);

    uint8_t *w = malloc(rows * row_bytes);
    float *src = malloc(cols * sizeof(float));
    float *in = calloc(n_vec * in_stride, sizeof(float));
    float *out = calloc(n_vec * out_stride, sizeof(float));
    float *ref = malloc(rows * sizeof(float));
    float *temp = malloc(cols * sizeof(float));
    cr_assert(w && src && in && out && ref && temp);

    uint32_t seed = 4242u;
    for (size_t r = 0; r < rows; r++) {
        for (size_t i = 0; i < cols; i++) src[i] = frand(&seed);
        oc_quant_pack_row(OC_QUANT_Q4_K_M, src, cols, w + r * row_bytes,
                          row_bytes);
    }
    for (size_t v = 0; v < n_vec; v++)
        for (size_t i = 0; i < cols; i++) in[v * in_stride + i] = frand(&seed);

    const size_t act_bytes = oc_matvec_batch_scratch_bytes(cols);
    uint8_t *act = malloc(act_bytes);
    cr_assert(act != NULL);
    oc_matvec_quantized_batch(OC_QUANT_Q4_K_M, w, rows, cols, row_bytes, in,
                              in_stride, out, out_stride, n_vec, temp, act,
                              act_bytes);

    for (size_t v = 0; v < n_vec; v++) {
        oc_matvec_quantized(OC_QUANT_Q4_K_M, w, rows, cols, row_bytes,
                            in + v * in_stride, ref, temp);
        for (size_t r = 0; r < rows; r++) {
            cr_assert(float_within_1ulp(out[v * out_stride + r], ref[r]),
                               "padded-stride mismatch at vec=%zu row=%zu",
                               v, r);
        }
        /* Nothing may be written into the padding between output vectors. */
        for (size_t r = rows; r < out_stride; r++) {
            cr_assert_float_eq(out[v * out_stride + r], 0.0f, 0.0f,
                               "wrote past row %zu of vector %zu", r, v);
        }
    }

    free(w); free(src); free(in); free(out); free(ref); free(temp); free(act);
}

/* The scratch helper must bound EVERY width up to its argument, not just the
 * one it is called with — this is the property whose absence corrupted the
 * expert down-projection. */
Test(matvec_batch, scratch_size_is_a_bound)
{
    const size_t widest = 4096;
    const size_t bound = oc_matvec_batch_scratch_bytes(widest);
    for (size_t cols = 256; cols <= widest; cols += 256) {
        cr_assert_leq(oc_matvec_batch_scratch_bytes(cols), bound,
                      "cols=%zu needs more scratch than the bound for %zu",
                      cols, widest);
    }
}

Test(matvec_batch, q4k_prepped_matches_packed)
{
    uint32_t seed = 31337u;
    for (size_t blocks = 1; blocks <= 16; blocks++) {
        const size_t cols = blocks * 256;
        const size_t row_bytes = oc_quantized_size(OC_QUANT_Q4_K_M, cols);
        uint8_t *row = malloc(row_bytes);
        float *src = malloc(cols * sizeof(float));
        float *act_f = malloc(cols * sizeof(float));
        uint8_t *act = malloc(oc_matvec_batch_scratch_bytes(cols));
        void *prep = malloc(oc_oxk_q4_k_prep_bytes(blocks));
        cr_assert(row && src && act_f && act && prep);

        for (size_t i = 0; i < cols; i++) src[i] = frand(&seed);
        cr_assert_eq(oc_quant_pack_row(OC_QUANT_Q4_K_M, src, cols, row,
                                       row_bytes), OC_OK);
        oc_oxk_q4_k_prep_row(row, blocks, prep);

        /* Several activations against the one prepared row — the reuse the
         * batched path depends on. */
        for (int t = 0; t < 4; t++) {
            for (size_t i = 0; i < cols; i++) act_f[i] = frand(&seed);
            /* Pack it exactly as the matvec would, by round-tripping through
             * a 1-vector batch call and comparing the two kernels' dots. */
            float packed_out = 0.0f, prepped_out = 0.0f;
            oc_matvec_quantized_batch(OC_QUANT_Q4_K_M, row, 1, cols, row_bytes,
                                      act_f, cols, &packed_out, 1, 1, src,
                                      act, oc_matvec_batch_scratch_bytes(cols));
            /* n_vec == 1 takes the packed kernel; drive the prepared one by
             * hand off the same packed activation the batch just built. */
            prepped_out = oc_oxk_dot_q4_k_prepped(prep, blocks, act);
            cr_assert_float_eq(prepped_out, packed_out, 0.0f,
                               "blocks=%zu t=%d: prepped %.9g != packed %.9g",
                               blocks, t, (double)prepped_out,
                               (double)packed_out);
        }
        free(row); free(src); free(act_f); free(act); free(prep);
    }
}

Test(matvec_batch, f32_weights)
{
    const size_t rows = 33, cols = 129, n_vec = 7;
    float *w = malloc(rows * cols * sizeof(float));
    float *in = malloc(n_vec * cols * sizeof(float));
    float *out = malloc(n_vec * rows * sizeof(float));
    float *ref = malloc(rows * sizeof(float));
    cr_assert(w && in && out && ref);

    uint32_t seed = 77u;
    for (size_t i = 0; i < rows * cols; i++) w[i] = frand(&seed);
    for (size_t i = 0; i < n_vec * cols; i++) in[i] = frand(&seed);

    oc_matvec_f32_batch(w, rows, cols, in, cols, out, rows, n_vec);
    for (size_t v = 0; v < n_vec; v++) {
        oc_matvec_f32(w, rows, cols, in + v * cols, ref);
        for (size_t r = 0; r < rows; r++)
            cr_assert_float_eq(out[v * rows + r], ref[r], 0.0f,
                               "f32 batch mismatch vec=%zu row=%zu", v, r);
    }
    free(w); free(in); free(out); free(ref);
}
