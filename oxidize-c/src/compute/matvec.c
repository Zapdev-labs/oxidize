/*
 * matvec.c — quantized/f32 matrix-vector products (scalar reference).
 *
 * Port of oxidize-core/src/compute/tensor/kernels/gemv.rs
 * (`gemv_f32`, `gemv_dequant_scalar_fallback`). The dequant step reuses
 * the SIMD-accelerated `oc_quant_dequant_row`, so this path inherits the
 * AVX2/AVX-512 speedups on capable hosts.
 */
#include "oxidize/matvec.h"
#include "oxidize/parallel.h"
#include "oxidize/quant.h"

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
} QuantJob;

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
                                 output, temp };
                matvec_quant_slice(0, rows, 0, &job);
                return;
            }
        }
    }
    QuantJob job = { qtype, data, cols, row_bytes, input, output, temp };
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
