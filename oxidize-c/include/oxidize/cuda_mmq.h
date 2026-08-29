/* cuda_mmq.h — device-resident quantized matvec ("MMQ") entry points. tolerance rather than bit-exactly. */
#ifndef OXIDIZE_CUDA_MMQ_H
#define OXIDIZE_CUDA_MMQ_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* True when `qtype` (an OcGgufQuantizationType) can stay packed on the device */
bool oc_cuda_mmq_supported(uint32_t qtype, size_t cols);

/* Device row stride in bytes for a packed row of `cols` elements, or 0 when */
size_t oc_cuda_mmq_row_bytes(uint32_t qtype, size_t cols);

/* On-disk block size, device block size, and blocks per row. Lets the caller
 * validate the source stride and choose a flat vs. strided (padded) copy.
 * Returns false for an unsupported (qtype, cols) pair. */
bool oc_cuda_mmq_block_layout(uint32_t qtype, size_t cols,
                              size_t *src_block, size_t *dev_block,
                              size_t *n_blocks);

/* out[row] = dot(dequant(W[row, :]), x[:]) for row in [0, rows).
 * `d_weights` is the packed device buffer; rows are `row_bytes` apart.
 * Returns false on an unsupported type or a launch error. */
bool oc_cuda_mmq_matvec(uint32_t qtype, const void *d_weights,
                        const float *d_x, float *d_out,
                        size_t rows, size_t cols, void *stream);

/* MoE variant of the above for a stacked expert tensor: expert `e` occupies */
bool oc_cuda_mmq_matvec_expert(uint32_t qtype, const void *d_weights,
                               const float *d_x, float *d_out,
                               size_t rows, size_t cols,
                               const uint32_t *d_expert_sel,
                               uint32_t slot, void *stream);

/* Dequantize a single packed row (`token`) into `d_out` (cols floats).
 * Used for the embedding lookup so the token-embedding table — often the
 * largest single tensor in the model — can stay quantized on the device. */
bool oc_cuda_mmq_get_row(uint32_t qtype, const void *d_weights,
                         uint32_t token, float *d_out,
                         size_t cols, void *stream);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OXIDIZE_CUDA_MMQ_H */
