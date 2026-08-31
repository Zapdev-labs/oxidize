/* flash_attention.h — fused QKV flash attention kernel (CPU scalar). */
#ifndef OXIDIZE_FLASH_ATTENTION_H
#define OXIDIZE_FLASH_ATTENTION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Online-softmax flash attention for a single Q head against cached K/V. */
OcError oc_flash_attention_head(const float *q,
                                 const float *k_cache,
                                 const float *v_cache,
                                 size_t seq_len,
                                 size_t head_dim,
                                 float *out,
                                 float *temp);

/* Flash attention for a batch of queries (multi-head). */
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


#define OC_FLASH_BLOCK_SIZE 64
#define OC_PARALLEL_FLASH_MIN_SEQ 16

uint16_t oc_f32_to_f16_bits(float value);

/* f16-to-f32 bit conversion. */
float oc_f16_to_f32_bits(uint16_t h);

/* Parallel multi-head flash attention decode over f32 KV cache. */
OcError oc_flash_attention_decode_heads_f32(const float *query_heads,
                                             const float *key_layer,
                                             const float *value_layer,
                                             size_t seq_len,
                                             size_t head_dim,
                                             size_t kv_len,
                                             size_t num_heads,
                                             size_t kv_heads,
                                             float *output_heads);

/* Same as above but K/V are in f16 (stored as uint16_t bits). */
OcError oc_flash_attention_decode_heads_f16(const float *query_heads,
                                             const uint16_t *key_layer,
                                             const uint16_t *value_layer,
                                             size_t seq_len,
                                             size_t head_dim,
                                             size_t kv_len,
                                             size_t num_heads,
                                             size_t kv_heads,
                                             float *output_heads);

/* Prefill-phase flash attention: many queries attend to many keys/values. */
OcError oc_flash_attention_prefill_f32(const float *query,
                                        const float *key,
                                        const float *value,
                                        size_t q_seq_len,
                                        size_t kv_seq_len,
                                        size_t head_dim,
                                        float *output);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_FLASH_ATTENTION_H */
