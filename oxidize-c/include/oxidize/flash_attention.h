/*
 * flash_attention.h — fused QKV flash attention kernel (CPU scalar).
 *
 * Implements the flash attention algorithm (Milakov & Gimelshein 2018)
 * with online softmax for computing attention without materializing the
 * full QK^T matrix. This reduces memory from O(n^2) to O(n) for the
 * attention computation.
 *
 * This is the scalar C implementation; SIMD and CUDA variants are future
 * enhancements.
 */
#ifndef OXIDIZE_FLASH_ATTENTION_H
#define OXIDIZE_FLASH_ATTENTION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Online-softmax flash attention for a single Q head against cached K/V.
 *
 * Computes: out = softmax(Q @ K^T / sqrt(d)) @ V
 *
 * Uses the online softmax algorithm to avoid materializing the full
 * QK^T matrix: processes one query position at a time, accumulating
 * the softmax denominator and weighted values incrementally.
 *
 * Parameters:
 *   q       - query vector [head_dim]
 *   k_cache - KV cache for this head, shape [seq_len, head_dim]
 *   v_cache - same shape
 *   seq_len - number of cached K/V positions to attend to
 *   head_dim - dimension of each head
 *   out     - output vector [head_dim] (written in-place, zeroed first)
 *   temp    - scratch buffer [head_dim] (for intermediate computations)
 *
 * Returns OC_OK or OC_ERR_INVALID_ARG.
 */
OcError oc_flash_attention_head(const float *q,
                                 const float *k_cache,
                                 const float *v_cache,
                                 size_t seq_len,
                                 size_t head_dim,
                                 float *out,
                                 float *temp);

/* Flash attention for a batch of queries (multi-head).
 *
 * Parameters:
 *   q       - queries [n_heads, head_dim]
 *   k_cache - KV cache [seq_len, n_heads_kv, head_dim] (GQA: n_heads_kv <= n_heads)
 *   v_cache - same shape
 *   n_heads - number of query heads
 *   n_heads_kv - number of KV heads (for GQA, n_heads / n_heads_kv = group_size)
 *   seq_len - number of cached positions
 *   head_dim - dimension per head
 *   out     - output [n_heads, head_dim]
 *   temp    - scratch buffer [head_dim]
 */
OcError oc_flash_attention_multi_head(const float *q,
                                      const float *k_cache,
                                      const float *v_cache,
                                      size_t n_heads,
                                      size_t n_heads_kv,
                                      size_t seq_len,
                                      size_t head_dim,
                                      float *out,
                                      float *temp);

/* Flash attention with sliding window (for Gemma2/Phi).
 * Only attends to positions [start, seq_len). */
OcError oc_flash_attention_sliding(const float *q,
                                    const float *k_cache,
                                    const float *v_cache,
                                    size_t seq_len,
                                    size_t head_dim,
                                    size_t window_start,
                                    float *out,
                                    float *temp);

/* Compute attention scores (Q @ K^T) without softmax, for debugging.
 * Writes [seq_len] scores into `scores`. */
OcError oc_attention_scores(const float *q,
                             const float *k_cache,
                             size_t seq_len,
                             size_t head_dim,
                             float *scores);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_FLASH_ATTENTION_H */
