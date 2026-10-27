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
 * Q6_K (210B/256), Q8_0 (34B/32) — plus IQ4_XS (136B/256), Q4_0/Q5_0, and the
 * AL-family aliases (AL5=Q4_0, AL6=Q5_0, AL8=Q8_0, AL5_XS 3-bit). Anything
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

/* Same as oc_cuda_mmq_get_row, but the row index is read on the device from
 * *d_token at kernel execution time. Used by the fast path where the next
 * token is produced by an on-device argmax and never reaches the host. */
bool oc_cuda_mmq_get_row_dev(uint32_t qtype, const void *d_weights,
                             const uint32_t *d_token, float *d_out,
                             size_t cols, void *stream);

/* Batched embedding lookup: out[t * out_stride ..] = expand(row d_ids[t]).
 * grid.y selects the token so one launch covers a whole prompt chunk. */
bool oc_cuda_mmq_get_row_batch(uint32_t qtype, const void *d_weights,
                               const uint32_t *d_ids, float *d_out,
                               size_t cols, size_t out_stride,
                               uint32_t n_tok, void *stream);

/* Batched matvec: out[t*out_stride + row] = dot(dequant(W[row]), x[t]).
 * One warp per row; the warp sweeps the row's packed bytes once per token
 * but from L1/L2-resident lines, so DRAM weight traffic is paid ~once per
 * launch instead of once per token — this is what makes prompt prefill
 * memory-bound rather than launch-bound. x strides are in floats. */
bool oc_cuda_mmq_matvec_batch(uint32_t qtype, const void *d_weights,
                              const float *d_x, float *d_out,
                              uint32_t n_tok, size_t x_stride,
                              size_t out_stride,
                              size_t rows, size_t cols, void *stream);

/* ─── dp4a path: int8-quantized activations ──────────────────────────────
 *
 * The f32 matvecs spend ~6 instructions per element on nibble unpack and
 * int→float conversion, which makes them issue-bound well below peak DRAM
 * bandwidth. The xq variants quantize each 32-element activation group to
 * int8 once (scale + exact group sums kept alongside), then dot the packed
 * weights with __dp4a — ~7x fewer instructions per element. Supported for
 * Q4_K/Q5_K/Q6_K/Q8_0/Q4_0/Q5_0 (+aliases); check oc_cuda_mmq_xq_supported
 * and fall back to the f32 entry points otherwise. Results differ from the
 * f32 path only by the activation's int8 rounding.
 *
 * Workspace layout (all indexed [tok * n_groups + group], xq packed x32):
 *   d_xq    int8   [n_tok * n_groups * 32]
 *   d_xsc   float2 [n_tok * n_groups]      (x = scale, y = exact fp32 sum)
 *   d_xqsum int32  [n_tok * n_groups * 2]  (per-16-element int8 sums)
 *   d_xh    f16    [n_tok * n_groups * 32] (xq*scale, for the mma path)
 */

/* True when the qtype has a dp4a kernel and cols % 32 == 0. */
bool oc_cuda_mmq_xq_supported(uint32_t qtype, size_t cols);

/* Quantize x (n_tok rows of x_stride floats each) into the workspace. */
bool oc_cuda_mmq_quantize_x(const float *d_x, int8_t *d_xq, float *d_xsc,
                            int32_t *d_xqsum, void *d_xh,
                            uint32_t n_tok, size_t cols,
                            size_t x_stride, void *stream);

/* out[t*out_stride + row] = dot(dequant(W[row]), unquantize(xq[t])). Each
 * warp handles a 16-token tile of the row; blockIdx.y walks further tiles,
 * so the weight row is fetched once per 16 tokens (from L1/L2 for later
 * tiles) and prompt chunks amortize DRAM traffic like matvec_batch. */
bool oc_cuda_mmq_matvec_xq(uint32_t qtype, const void *d_weights,
                           const int8_t *d_xq, const float *d_xsc,
                           const int32_t *d_xqsum, const void *d_xh,
                           float *d_out, uint32_t n_tok,
                           size_t out_stride, size_t rows,
                           size_t cols, void *stream);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OXIDIZE_CUDA_MMQ_H */
