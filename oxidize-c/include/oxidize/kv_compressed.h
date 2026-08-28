/*
 * kv_compressed.h — unified compressed KV cache (RotorQuant default, Helix
 * alternate). Callers always pass pre-RoPE keys/queries plus positions; the
 * facade applies RoPE where the scheme needs it.
 */
#ifndef OXIDIZE_KV_COMPRESSED_H
#define OXIDIZE_KV_COMPRESSED_H

#include <stddef.h>

#include "oxidize/error.h"
#include "oxidize/helix_cache.h"
#include "oxidize/rotorquant_cache.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OC_KV_SCHEME_ROTOR = 0,
    OC_KV_SCHEME_HELIX = 1,
} OcKvScheme;

#define OC_KV_SCHEME_DEFAULT OC_KV_SCHEME_ROTOR

/* Interleaved (2i, 2i+1) matches Helix polar pairs and the C++ PR.
 * Split-halves (i, i+half) matches oc_apply_rope_f32 / Qwen-family GGUFs. */
typedef enum {
    OC_KV_ROPE_INTERLEAVED  = 0,
    OC_KV_ROPE_SPLIT_HALVES = 1,
} OcKvRopeLayout;

typedef struct OcCompressedKvCache {
    OcKvScheme      scheme;
    OcKvRopeLayout  rope_layout;
    size_t          head_dim;
    size_t          page_size;
    float           rope_theta;
    size_t          next_page_id;
    OcHelixCache    helix;
    int             has_helix;
    OcRotorQuantCache rotor;
    int             has_rotor;
} OcCompressedKvCache;

OcError oc_compressed_kv_init(OcCompressedKvCache *cache, size_t head_dim,
                              OcKvScheme scheme, size_t page_size,
                              float rope_theta);
void oc_compressed_kv_free(OcCompressedKvCache *cache);

void oc_compressed_kv_set_rope_layout(OcCompressedKvCache *cache,
                                      OcKvRopeLayout layout);

OcKvScheme oc_compressed_kv_scheme(const OcCompressedKvCache *cache);

OcError oc_compressed_kv_store_page(OcCompressedKvCache *cache,
                                    size_t layer, size_t kv_head,
                                    const float *pre_rope_keys,
                                    const float *values,
                                    const size_t *positions,
                                    size_t n_tokens);

OcError oc_compressed_kv_attention(OcCompressedKvCache *cache,
                                   size_t layer, size_t kv_head,
                                   const float *query_pre_rope, size_t query_n,
                                   size_t query_position, float *out);

const OcHelixCache *oc_compressed_kv_helix(const OcCompressedKvCache *cache);
const OcRotorQuantCache *oc_compressed_kv_rotor(const OcCompressedKvCache *cache);

float oc_compressed_kv_compression_ratio(const OcCompressedKvCache *cache);

OcError oc_compressed_kv_rewind(OcCompressedKvCache *cache, size_t n_keep);
void oc_compressed_kv_clear(OcCompressedKvCache *cache);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_KV_COMPRESSED_H */
