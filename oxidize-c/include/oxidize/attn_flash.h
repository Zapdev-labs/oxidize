/*
 * attn_flash.h — flash-decoding / blocked-prefill attention over one KV head
 * of any cache format (f32, int8 or RotorQuant).
 *
 * Decode: the caller splits the history [0, pos] into ranges (one per task,
 * across kv heads x sequence chunks), each range produces a partial
 * (max, sum, acc) for the G query heads that share the kv head, and
 * oc_attn_flash_merge() folds the partials. Every K row is read once and
 * scored against all G query heads; softmax is online, per 64-position
 * block, with an AVX2 exp.
 *
 * Prefill: oc_attn_flash_prefill() takes R query rows (tokens x G heads of
 * one kv head, ascending position) and streams the cache in 64-position
 * tiles through register-blocked f32 microkernels (S = Q K^T, O += P V).
 *
 * Queries must be pre-scaled by the softmax scale; for OC_KVV_RQ they must
 * also be rotated (oc_kvrq_rotate) and the outputs come back in the rotated
 * domain — un-rotate once per head.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "oxidize/kv_rq.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_FLASH_TILE 64u

typedef enum {
    OC_KVV_F32 = 0,
    OC_KVV_Q8  = 1,
    OC_KVV_RQ  = 2,
} OcKvViewKind;

typedef struct {
    OcKvViewKind kind;
    size_t d;
    /* f32: row t of this head at kf + t*fs */
    const float *kf, *vf;
    size_t fs;
    /* q8: codes at kq + t*qs, scale at ksc[t*ss] */
    const int8_t *kq, *vq;
    size_t qs;
    const float *ksc, *vsc;
    size_t ss;
    /* RQ */
    const OcKvRqCache *rq;
    size_t layer, head;
    int64_t hi_written;   /* newest position stored (ring reference)   */
} OcKvView;

/* Scratch floats oc_attn_flash_decode_range() needs for G heads. */
size_t oc_attn_flash_decode_scratch(size_t G);

/* Partial attention of G heads over positions [t0, t1). m/l/acc are
 * initialised here (m = -inf, l = 0, acc = 0). */
void oc_attn_flash_decode_range(const OcKvView *v, const float *q, size_t G,
                                int64_t t0, int64_t t1, float *m, float *l,
                                float *acc, float *scratch);

/* Merge n_parts partials (part p: m[p*G + g], l[p*G + g], acc[(p*G+g)*d])
 * into out[g*d] = normalized attention output. */
void oc_attn_flash_merge(size_t G, size_t d, size_t n_parts, const float *m,
                         const float *l, const float *acc, float *out);

/* Scratch floats oc_attn_flash_prefill() needs for R rows of width d. */
size_t oc_attn_flash_prefill_scratch(size_t R, size_t d);

/* R query rows (row r at q + r*d, position row_pos[r], ascending) attend to
 * cache positions [max(0, row_pos[r] - sw + 1), row_pos[r]] (sw = 0: from
 * 0). Writes normalized outputs to out + r*d. */
void oc_attn_flash_prefill(const OcKvView *v, const float *q, size_t R,
                           const int64_t *row_pos, int64_t sw, float *out,
                           float *scratch);

/* Vectorized exp of n floats in place (AVX2 when available). Values below
 * -87 flush to exactly 0. Exposed for tests. */
void oc_attn_flash_exp(float *x, size_t n);

#ifdef __cplusplus
}
#endif
