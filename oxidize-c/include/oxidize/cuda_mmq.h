/*
 * cuda_mmq.h — device-resident quantized matvec ("MMQ") entry points.
 *
 * The original CUDA path in cuda.cu dequantized every weight to f32 on the
 * host and uploaded f32 to the device, which costs 8x VRAM for a Q4_K model
 * and 4x the memory traffic per token on a bandwidth-bound GEMV. The kernels
 * behind this header consume the packed GGUF blocks directly, so weights stay
 * quantized in device memory and are dequantized in registers at use time.
 *
 * Supported types cover everything a Q4_K_M file contains — Q4_K (144B/256),
 * Q6_K (210B/256), Q8_0 (34B/32) — plus IQ4_XS (136B/256), which is what the
 * imatrix-quantized Gemma 4 files use for 410 of their 833 tensors. Anything
 * else falls back to the host dequantize-to-f32 upload path in cuda.cu, which
 * for a 31B model means ~124 GB of f32 weights and does not fit on any single
 * card — so a missing kernel here is a hard load failure, not a slow path.
 *
 * All wrappers are asynchronous: they launch on `stream` and never
 * synchronize, so the per-token layer loop stays a single pipelined
 * submission. Callers check errors with cudaGetLastError()/one sync per token
 * exactly as the f32 path does.
 *
 * Block layouts mirror src/compute/quantization.c bit for bit; see
 * ck_q4k_block_dot / ck_q6k_block_dot for the per-block math. Reductions run
 * in a different order than the scalar CPU reference, so results agree to f32
 * tolerance rather than bit-exactly.
 */
#ifndef OXIDIZE_CUDA_MMQ_H
#define OXIDIZE_CUDA_MMQ_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* True when `qtype` (an OcGgufQuantizationType) can stay packed on the device
 * for a row of `cols` elements. Rejects types without a device kernel and
 * shapes whose row length is not a whole number of blocks — notably K-quants
 * need cols % 256 == 0, which e.g. n_embd = 896 does not satisfy. */
bool oc_cuda_mmq_supported(uint32_t qtype, size_t cols);

/* Device row stride in bytes for a packed row of `cols` elements, or 0 when
 * the (qtype, cols) pair is unsupported. This may exceed the on-disk stride:
 * Q6_K blocks are padded from 210 to 224 bytes so that 16-byte vector loads
 * are legal. Use oc_cuda_mmq_block_layout() to drive the upload copy. */
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

/* Batched form for prefill: `n_vec` activation vectors of `x_stride` floats
 * in `d_x`, `n_vec` output vectors of `out_stride` floats in `d_out` (both
 * activation-major, matching the CPU batch path). One warp dots each weight
 * row against a tile of activations, so packed weights are fetched from DRAM
 * once per tile instead of once per token. */
bool oc_cuda_mmq_matmul(uint32_t qtype, const void *d_weights,
                        const float *d_x, float *d_out,
                        size_t rows, size_t cols, size_t n_vec,
                        size_t x_stride, size_t out_stride, void *stream);

/* MoE variant of the above for a stacked expert tensor: expert `e` occupies
 * rows [e*rows, (e+1)*rows). The expert index is read on the device from
 * `d_expert_sel[slot]`, so router output never round-trips to the host and the
 * per-token work stays a single asynchronous submission. */
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
