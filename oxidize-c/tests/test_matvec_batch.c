/*
 * test_matvec_batch.c — does the batched matvec agree with the single one?
 *
 * oc_matvec_quantized_batch() is the primitive prompt prefill runs on: it
 * dots each weight row against a tile of activations while the row is in
 * cache, instead of re-reading the matrix once per token. That reshapes the
 * loops but must not change a single output value — every result is still
 * one row dotted with one activation, accumulated in the same order.
 *
 * The interesting parameter is `cols`, because the activation tile width is
 * derived from it by integer division. A first cut of the sizing helper
 * sampled a few widths and took the max, which is NOT a bound: a NARROWER
 * matrix can need marginally more scratch than a wide one, and the expert
 * down-projection (cols=768, narrower than the 2048/4096 that were sampled)
 * overran the buffer and corrupted the forward pass. Hence the deliberately
 * awkward widths below, and the undersized-scratch case.
 */
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

    /* Types whose fused path accumulates in f32 lanes (facc) or whose
     * packed-multi kernel reassociates differ from the sequential dot by a
     * few ulp; other types must still agree to within 1 ulp. */
    const bool reassoc =
        qtype == OC_QUANT_Q4_0 || qtype == OC_QUANT_AL5 ||
        qtype == OC_QUANT_Q8_0 || qtype == OC_QUANT_AL8;
    for (size_t v = 0; v < n_vec; v++) {
        for (size_t r = 0; r < rows; r++) {
            const size_t k = v * rows + r;
            /* abs+rel for the reassociated types: near-zero dots from
             * cancellation make pure relative tolerance meaningless. */
            const double tol = 1e-4 * (1.0 + fabs((double)out_ref[k]));
            cr_assert(reassoc ? fabs((double)out_bat[k] -
                                        (double)out_ref[k]) < tol
                              : float_within_1ulp(out_bat[k], out_ref[k]),
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

/* An undersized or absent buffer must cost speed, not correctness: the
 * kernel clamps its tile to fit and falls back to the dequant path if even
 * one activation does not.
 *
 * Note this compares batch-against-batch, not against oc_matvec_quantized():
 * the dequant path and the fused integer path legitimately disagree (the
 * fused one carries int8 activation error), so a cross-path comparison would
 * be testing the wrong invariant. What must hold is that a batch of N gives
 * the same answers as N batches of one, on whichever path is taken. */
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

/* The prepared-row Q4_K kernel is what the batched path actually runs. It
 * hoists the nibble unpack and scale decode out of the activation loop, so
 * it must agree with the packed kernel to the bit — the integer sums are the
 * same values in a different (associative) order and the per-block float
 * accumulation order is unchanged, so "close" would mean a real bug. */
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

/* The classic-quant packed-multi kernels (Q4_0 / Q8_0 weights against a
 * tile of Q8_0 activations) amortize the row decode across an
 * eight-activation group and accumulate in eight f32 lanes — the integer
 * dots are exact but the float association differs from the sequential
 * single-dot order, so the comparison is a small relative tolerance, not
 * the 1-ulp check the bit-exact paths get. Covers the eight-wide tail
 * (n_act = 13 leaves a partial group). */
static void check_packed_multi(OcGgufQuantizationType qtype, size_t cols,
                               size_t n_act)
{
    const size_t row_bytes = oc_quantized_size(qtype, cols);
    const size_t act_rb   = oc_quantized_size(OC_QUANT_Q8_0, cols);
    const size_t blocks   = cols / OC_OXK_QK8_0;
    uint8_t *row = malloc(row_bytes);
    uint8_t *acts = malloc(n_act * act_rb);
    float *src = malloc(cols * sizeof(float));
    float *ref = malloc(n_act * sizeof(float));
    float *got = malloc(n_act * sizeof(float));
    cr_assert(row && acts && src && ref && got);

    uint32_t seed = 555u + (uint32_t)cols;
    for (size_t i = 0; i < cols; i++) src[i] = frand(&seed);
    cr_assert_eq(oc_quant_pack_row(qtype, src, cols, row, row_bytes), OC_OK);
    for (size_t v = 0; v < n_act; v++) {
        for (size_t i = 0; i < cols; i++) src[i] = frand(&seed);
        cr_assert_eq(oc_quant_pack_row(OC_QUANT_Q8_0, src, cols,
                                       acts + v * act_rb, act_rb), OC_OK);
    }
    float (*dot)(const uint8_t *, size_t, const uint8_t *) =
        (qtype == OC_QUANT_Q4_0) ? oc_oxk_dot_q4_0_q8_0
                                 : oc_oxk_dot_q8_0_q8_0;
    for (size_t v = 0; v < n_act; v++)
        ref[v] = dot(row, blocks, acts + v * act_rb);

    memset(got, 0, n_act * sizeof(float));
    if (qtype == OC_QUANT_Q4_0)
        oc_oxk_dot_q4_0_q8_0_packed_multi(row, blocks, acts, act_rb,
                                          n_act, got);
    else
        oc_oxk_dot_q8_0_q8_0_packed_multi(row, blocks, acts, act_rb,
                                          n_act, got);

    for (size_t v = 0; v < n_act; v++) {
        const double d = fabs((double)got[v] - (double)ref[v]);
        const double rel = d / (fabs((double)ref[v]) + 1e-30);
        cr_assert(rel < 1e-4 || d < 1e-4,
                  "qtype=%d cols=%zu act=%zu: multi %.9g != single %.9g",
                  (int)qtype, cols, v, (double)got[v], (double)ref[v]);
    }
    free(row); free(acts); free(src); free(ref); free(got);
}

Test(matvec_batch, classic_packed_multi)
{
    for (size_t cols = 32; cols <= 2048; cols *= 4) {
        check_packed_multi(OC_QUANT_Q4_0, cols, 13);
        check_packed_multi(OC_QUANT_Q8_0, cols, 13);
    }
    check_packed_multi(OC_QUANT_Q4_0, 896, 129);
    check_packed_multi(OC_QUANT_Q8_0, 896, 129);
}

/* The quad path decodes four Q4_0 rows once into scratch and dots the
 * tile against a 4x4 register-blocked kernel: same exact integer dots and
 * the same f32 reassociation tolerance as packed_multi. Covers row tails
 * (n_rows mod 4) and activation tails (n_act mod 4). */
static void check_quad_multi(size_t cols, size_t rows, size_t n_act)
{
    const size_t row_bytes = oc_quantized_size(OC_QUANT_Q4_0, cols);
    const size_t act_rb   = oc_quantized_size(OC_QUANT_Q8_0, cols);
    const size_t blocks   = cols / OC_OXK_QK8_0;
    uint8_t *w = malloc(rows * row_bytes);
    uint8_t *acts = malloc(n_act * act_rb);
    float *src = malloc(cols * sizeof(float));
    void *prep = malloc(oc_oxk_q4_0_quad_prep_bytes(blocks));
    float *res = malloc(((n_act + 3) / 4 * 4) * 4 * sizeof(float));
    cr_assert(w && acts && src && prep && res);

    uint32_t seed = 777u + (uint32_t)(cols * 131u + rows * 17u + n_act);
    for (size_t r = 0; r < rows; r++) {
        for (size_t i = 0; i < cols; i++) src[i] = frand(&seed);
        cr_assert_eq(oc_quant_pack_row(OC_QUANT_Q4_0, src, cols,
                                       w + r * row_bytes, row_bytes), OC_OK);
    }
    for (size_t v = 0; v < n_act; v++) {
        for (size_t i = 0; i < cols; i++) src[i] = frand(&seed);
        cr_assert_eq(oc_quant_pack_row(OC_QUANT_Q8_0, src, cols,
                                       acts + v * act_rb, act_rb), OC_OK);
    }

    /* Second act buffer with the extended layout the batch path uses:
     * [Q8_0 blocks][pad4][f32 dq tail] — exercises the dq_off read. */
    const size_t dq_off  = (act_rb + 3u) & ~(size_t)3u;
    const size_t act_rb2 = dq_off + blocks * sizeof(float);
    uint8_t *acts2 = malloc(n_act * act_rb2);
    cr_assert(acts2);
    for (size_t v = 0; v < n_act; v++) {
        memcpy(acts2 + v * act_rb2, acts + v * act_rb, act_rb);
        float *dq = (float *)(acts2 + v * act_rb2 + dq_off);
        for (size_t b = 0; b < blocks; b++) {
            /* Keep the true decoded scale in the tail but overwrite the
             * header with f16 64.0 (0x5000): a kernel that ignores dq_off
             * reads the bogus header and fails loudly instead of silently
             * matching the reference. */
            uint8_t *hdr = acts2 + v * act_rb2 + b * OC_OXK_BLOCK_Q8_0_SIZE;
            hdr[0] = 0x00;
            hdr[1] = 0x50;
            dq[b] = oc_oxk_f16_le_to_f32(
                acts + v * act_rb + b * OC_OXK_BLOCK_Q8_0_SIZE);
        }
    }

    for (size_t r0 = 0; r0 < rows; r0 += 4) {
        const size_t nr = (rows - r0 < 4) ? rows - r0 : 4;
        oc_oxk_q4_0_quad_prep(w + r0 * row_bytes, row_bytes, nr, blocks,
                              prep);
        oc_oxk_dot_q4_0_q8_0_quad_multi(prep, blocks, acts, act_rb, 0,
                                        n_act, res);
        for (size_t v = 0; v < n_act; v++) {
            for (size_t r = 0; r < nr; r++) {
                const float ref = oc_oxk_dot_q4_0_q8_0(
                    w + (r0 + r) * row_bytes, blocks, acts + v * act_rb);
                const float got = res[v * 4 + r];
                const double d = fabs((double)got - (double)ref);
                cr_assert(d < 1e-4 * (1.0 + fabs((double)ref)),
                          "cols=%zu rows=%zu act=%zu: quad v%zu r%zu "
                          "%.9g != %.9g",
                          cols, rows, n_act, v, r0 + r,
                          (double)got, (double)ref);
            }
        }
        /* Same again through the f32 dq tail. */
        oc_oxk_dot_q4_0_q8_0_quad_multi(prep, blocks, acts2, act_rb2,
                                        dq_off, n_act, res);
        for (size_t v = 0; v < n_act; v++) {
            for (size_t r = 0; r < nr; r++) {
                const float ref = oc_oxk_dot_q4_0_q8_0(
                    w + (r0 + r) * row_bytes, blocks, acts + v * act_rb);
                const float got = res[v * 4 + r];
                const double d = fabs((double)got - (double)ref);
                cr_assert(d < 1e-4 * (1.0 + fabs((double)ref)),
                          "cols=%zu rows=%zu act=%zu: quad-dqf v%zu r%zu "
                          "%.9g != %.9g",
                          cols, rows, n_act, v, r0 + r,
                          (double)got, (double)ref);
            }
        }
    }
    free(w); free(acts); free(acts2); free(src); free(prep); free(res);
}

Test(matvec_batch, q4_0_quad_multi)
{
    for (size_t cols = 32; cols <= 2048; cols *= 4) {
        check_quad_multi(cols, 9, 13);   /* row tail 1, act tail 1 */
        check_quad_multi(cols, 7, 4);    /* row tail 3, exact act group */
        check_quad_multi(cols, 8, 7);    /* exact quad, act tail 3 */
    }
    check_quad_multi(896, 33, 129);
}

/* The qs form takes per-block activation sums as a side channel so the
 * fused matvec can hoist the sad chain out of the row loop; it must be
 * bit-identical to oc_oxk_dot_q4_0_q8_0. */
Test(matvec_batch, q4_0_qs_matches_standard)
{
    for (size_t cols = 32; cols <= 2048; cols *= 4) {
        const size_t blocks = cols / OC_OXK_QK8_0;
        const size_t row_bytes = oc_quantized_size(OC_QUANT_Q4_0, cols);
        uint8_t *w = malloc(row_bytes);
        float *src = malloc(cols * sizeof(float));
        float *act_f = malloc(cols * sizeof(float));
        uint8_t *act = malloc(blocks * OC_OXK_BLOCK_Q8_0_SIZE);
        int32_t *qsums = malloc(blocks * sizeof(int32_t));
        cr_assert(w && src && act_f && act && qsums);

        uint32_t seed = (uint32_t)(900u + cols);
        for (size_t i = 0; i < cols; i++) src[i] = frand(&seed);
        cr_assert_eq(oc_quant_pack_row(OC_QUANT_Q4_0, src, cols, w,
                                       row_bytes), OC_OK);
        for (int iter = 0; iter < 32; iter++) {
            for (size_t i = 0; i < cols; i++) act_f[i] = frand(&seed);
            cr_assert_eq(oc_quant_pack_row(OC_QUANT_Q8_0, act_f, cols, act,
                                           blocks * OC_OXK_BLOCK_Q8_0_SIZE),
                         OC_OK);
            for (size_t b = 0; b < blocks; b++) {
                const int8_t *q = (const int8_t *)(act +
                    b * OC_OXK_BLOCK_Q8_0_SIZE + 2);
                int32_t s = 0;
                for (size_t i = 0; i < OC_OXK_QK8_0; i++) s += q[i];
                qsums[b] = s;
            }
            const float ref = oc_oxk_dot_q4_0_q8_0(w, blocks, act);
            const float got = oc_oxk_dot_q4_0_q8_0_qs(w, blocks, act, qsums);
            cr_assert_eq(ref, got,
                         "cols=%zu iter=%d: qs %.9g != standard %.9g",
                         cols, iter, (double)got, (double)ref);
        }
        free(w); free(src); free(act_f); free(act); free(qsums);
    }
}

/* f32 weights take a separate kernel; it must match oc_matvec_f32. */
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
