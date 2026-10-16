#ifndef OXIDIZE_EXL3_H
#define OXIDIZE_EXL3_H
#include "oxidize/error.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Packed little-endian bytes, input-tile-major then output-tile-major.
 * Scales/bias are owned by the overlay; this view owns no storage.
 *
 * Two load-time relabeling hooks keep the packed bytes untouched:
 *   - in_block_perm/out_block_perm relabel 128-dim input/output blocks
 *     (block b reads physical block perm[b]) — the qwen35-family
 *     grouped->tiled V-head reorder is exactly such a permutation;
 *   - packed_kt_stride/packed_nt_base carve a sub-matrix out of a larger
 *     packed tensor (e.g. the qwen4exp indexer's fused q|k projection):
 *     tile (kt, nt) reads packed + (kt*kt_stride + nt + nt_base)*tile_bytes,
 *     with kt_stride defaulting to output_dim/16. svh/bias pointers must
 *     be offset to match. */
typedef struct OcExl3Matrix {
    size_t input_dim, output_dim;
    const uint8_t *packed;
    size_t packed_size;
    unsigned bits_twice, codebook;
    const float *suh, *svh, *bias;
    const uint32_t *in_block_perm;   /* input_dim/128 entries, or NULL  */
    const uint32_t *out_block_perm;  /* output_dim/128 entries, or NULL */
    size_t packed_kt_stride;         /* 0 = output_dim/16               */
    size_t packed_nt_base;           /* 0 = no offset                   */
} OcExl3Matrix;
float oc_exl3_codebook_value(unsigned codebook, uint16_t idx);
OcError oc_exl3_validate(const OcExl3Matrix *m);
OcError oc_exl3_decode_tile(const OcExl3Matrix *m, size_t input_tile,
                          size_t output_tile, float out[256]);
/* Strides count floats. The process-global parallel pool must not be called
 * concurrently or recursively, as required by oc_parallel_for. */
OcError oc_exl3_matmul(const OcExl3Matrix *m, const float *in,
                      size_t in_stride, float *out, size_t out_stride, size_t batch);
OcError oc_exl3_matvec(const OcExl3Matrix *m, const float *in, float *out);
/* Effective output_dim × input_dim matrix (trellis decode plus the su/sv
 * Hadamard scales), row-major — the map oc_exl3_matmul computes. */
OcError oc_exl3_dequant_f32(const OcExl3Matrix *m, float *out);

/* ── Per-expert MoE runner (native qwen4exp: one trellis matrix per expert
 * projection, not a stacked view). The caller phases the work; each call
 * runs ONE oc_parallel_for over (expert, output-block) jobs, so the pool is
 * never entered recursively.
 *
 * experts[e].gate/up map input (n_embd) -> intermediate; down maps
 * intermediate -> (n_embd). Decode variant: every expert consumes the same
 * single input row. Batch variant: expert e consumes input rows
 * rows[off[e]..off[e+1]) (a token-index gather) and writes output rows
 * [off[e], off[e+1]) of the caller's buffers. gate_out/up_out hold one
 * intermediate row per (expert, token) pair, in the same packed order the
 * caller passed (rows[off[e]+j] pairs with expert row e slot j), so the
 * activation pass between the two calls can run on the packed layout. */
typedef struct OcExl3Expert {
    const OcExl3Matrix *gate, *up, *down;
} OcExl3Expert;

OcError oc_exl3_experts_gate_up(const OcExl3Expert *experts, uint32_t n,
                                const float *in, size_t in_stride,
                                const uint32_t *rows, const uint32_t *off,
                                float *gate_out, float *up_out,
                                size_t out_stride);
OcError oc_exl3_experts_down(const OcExl3Expert *experts, uint32_t n,
                             const float *act, size_t act_stride,
                             const uint32_t *rows, const uint32_t *off,
                             float *out, size_t out_stride);
#ifdef __cplusplus
}
#endif
#endif
