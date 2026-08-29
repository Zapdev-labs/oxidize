/* matvec.h — quantized/f32 matrix-vector products (weight × activation). */
#ifndef OXIDIZE_MATVEC_H
#define OXIDIZE_MATVEC_H

#include <stddef.h>
#include <stdint.h>

#include <stdbool.h>

#include "oxidize/error.h"
#include "oxidize/quant.h"

#ifdef __cplusplus
extern "C" {
#endif

/* f32 weight × f32 input → f32 output. `data` is row-major [rows][cols].
 * `input` length == cols, `output` length == rows. */
void oc_matvec_f32(const float *data, size_t rows, size_t cols,
                   const float *input, float *output);

/* Quantized weight × f32 input → f32 output. Dequantizes one weight row at */
void oc_matvec_quantized(OcGgufQuantizationType qtype, const uint8_t *data,
                         size_t rows, size_t cols, size_t row_bytes,
                         const float *input, float *output, float *temp);

/* Fused multi-projection variant: computes several matvecs that share the SAME input vector in one call (so the input is read once from cache). */
void oc_matvec_quantized_fused(const OcGgufQuantizationType *qtypes,
                               const uint8_t *const *datas, const size_t *rows,
                               size_t cols, const size_t *row_bytes,
                               size_t n_outs, const float *input,
                               float *const *outs, float *temp);

/* Multi-input counterpart used by grouped MoE down projections. Each matrix */
void oc_matvec_quantized_multi_input(
    const OcGgufQuantizationType *qtypes,
    const uint8_t *const *datas, const size_t *rows, size_t cols,
    const size_t *row_bytes, size_t n_outs,
    const float *const *inputs, float *const *outs, float *temp);

/* Test instrumentation for oc_matvec_quantized_fused(). These counters are
 * process-global and intended only for focused dispatch tests; reset them
 * immediately before the call under test. */
typedef struct {
    size_t activation_quantizations;
    size_t parallel_dispatches;
    size_t fallback_calls;
} OcMatvecFusedTestStats;

void oc_matvec_fused_test_reset(void);
OcMatvecFusedTestStats oc_matvec_fused_test_stats(void);

void oc_matvec_quantized_batch(OcGgufQuantizationType qtype,
                               const uint8_t *data, size_t rows, size_t cols,
                               size_t row_bytes,
                               const float *inputs, size_t in_stride,
                               float *outputs, size_t out_stride,
                               size_t n_vec, float *temp,
                               uint8_t *act_scratch, size_t act_bytes);

/* Worst-case `act_scratch` size for ANY oc_matvec_quantized_batch() call whose */
size_t oc_matvec_batch_scratch_bytes(size_t max_cols);

void oc_matvec_f32_batch(const float *data, size_t rows, size_t cols,
                         const float *inputs, size_t in_stride,
                         float *outputs, size_t out_stride, size_t n_vec);

/* Enable or disable the fused integer GEMV path (default: enabled). */
void oc_matvec_set_fused(bool enabled);
bool oc_matvec_fused_enabled(void);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_MATVEC_H */
