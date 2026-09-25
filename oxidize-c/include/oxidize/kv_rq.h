/*
 * kv_rq.h — RotorQuant KV cache (OC_KV_RQ) codec, cache and fused kernels.
 *
 * One cached head vector x (d = head_dim, a power of two <= 256) is stored as
 *
 *     r  = R x            R orthogonal: randomized Hadamard (default) or the
 *                          4-D quaternion "iso" rotation of rotorquant.c
 *     u  = r / ||r||      unit vector, coordinates ~ N(0, 1/d)
 *     i_j = nearest Lloyd-Max centroid of u_j (2, 3 or 4 bits)
 *     s  = ||x|| / ||c[i]||   so the reconstruction s*c[i] keeps x's length
 *
 * and attention never leaves the rotated domain: q is rotated once per head
 * (q.k == Rq.Rk), V is accumulated as sum_t w_t s_t c[i_t] in the rotated
 * domain, and the head output is un-rotated once.
 *
 * Block layout (block_bytes = 2 + d*bits/8), built so eight consecutive
 * coordinates unpack with one 8-byte load + zero-extend + shift/mask and one
 * _mm256_permutevar8x32_ps centroid lookup:
 *
 *   [0..1]   f16 scale s
 *   bits 2 : plane[d/4]      byte j holds coord j + (d/4)*k in bits 2k..2k+1
 *   bits 4 : plane[d/2]      byte j holds coord j (low nibble), j + d/2 (high)
 *   bits 3 : plane2[d/4]     low two bits, laid out as for bits 2
 *            plane1[d/8]     byte j bit k holds the high bit of coord j+(d/8)*k
 *
 * The cache (OcKvRqCache) keeps one lazily-committed mmap(MAP_NORESERVE)
 * region per layer, laid out [K|V][kv_head][pos][block], so memory grows with
 * the filled context rather than n_ctx and decode reads each head's history
 * sequentially. Optional "exact" slots keep attention sinks (the first
 * n_sink positions) and a ring of the most recent `window` positions as
 * rotated int8 (per-vector f32 scale); the RQ block is written for every
 * position regardless, so a position that leaves the window is still there.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/rotorquant.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_KVRQ_DIM_MAX 256u

typedef enum {
    OC_KVRQ_ROT_HADAMARD = 0,   /* random signs + normalized Walsh-Hadamard */
    OC_KVRQ_ROT_ISO      = 1,   /* rotorquant.c 4-D quaternion blocks      */
    OC_KVRQ_ROT_NONE     = 2,   /* identity (tests / ablation only)        */
} OcKvRqRotKind;

typedef struct {
    size_t        d;
    OcKvRqRotKind kind;
    float         sign[OC_KVRQ_DIM_MAX];
    OcRotorQuant  iso;          /* only for OC_KVRQ_ROT_ISO */
} OcKvRqRot;

OcError oc_kvrq_rot_init(OcKvRqRot *r, size_t d, OcKvRqRotKind kind,
                         uint64_t seed);
void oc_kvrq_rot_free(OcKvRqRot *r);
/* out = R in / out = R^T in. `out` may alias `in`. */
void oc_kvrq_rotate(const OcKvRqRot *r, const float *in, float *out);
void oc_kvrq_unrotate(const OcKvRqRot *r, const float *in, float *out);

typedef struct {
    size_t   d;
    unsigned bits;              /* 2, 3 or 4 */
    unsigned n_levels;
    size_t   block_bytes;       /* 2 + d*bits/8 */
    float    centroids[16];     /* ascending, unit-vector coordinates */
    float    bounds[16];        /* n_levels-1 decision thresholds     */
} OcKvRqCodec;

OcError oc_kvrq_codec_init(OcKvRqCodec *c, size_t d, unsigned bits);
/* Encode/decode one vector that is ALREADY in the rotated domain. */
void oc_kvrq_encode(const OcKvRqCodec *c, const float *xr, uint8_t *blk);
void oc_kvrq_decode(const OcKvRqCodec *c, const uint8_t *blk, float *xr);

/* ── Fused kernels (AVX2 with scalar fallback) ─────────────────────────
 * `blocks` are n consecutive blocks (block_bytes apart). q/acc hold G rows
 * of d floats, back to back. */

/* scores[g*ss + t] = <q_g, dec(block_t)>   t in [0,n), g in [0,G) */
void oc_kvrq_score(const OcKvRqCodec *c, const uint8_t *blocks, size_t n,
                   const float *q, size_t G, float *scores, size_t ss);
/* acc_g += sum_t w[g*ws + t] * dec(block_t) */
void oc_kvrq_accum(const OcKvRqCodec *c, const uint8_t *blocks, size_t n,
                   const float *w, size_t ws, size_t G, float *acc);
/* rows[t*d ..] = dec(block_t) */
void oc_kvrq_decode_rows(const OcKvRqCodec *c, const uint8_t *blocks,
                         size_t n, float *rows);

/* Same trio for int8 rows: row t is codes + t*cs (d int8) with scale
 * scales[t*sst]. Shared by the dense Q8 cache and the RQ exact slots. */
void oc_kvq8_score(const int8_t *codes, size_t cs, const float *scales,
                   size_t sst, size_t d, size_t n, const float *q, size_t G,
                   float *scores, size_t ss);
void oc_kvq8_accum(const int8_t *codes, size_t cs, const float *scales,
                   size_t sst, size_t d, size_t n, const float *w, size_t ws,
                   size_t G, float *acc);
void oc_kvq8_decode_rows(const int8_t *codes, size_t cs, const float *scales,
                         size_t sst, size_t d, size_t n, float *rows);
/* And for f32 rows (row t at x + t*xs). */
void oc_kvf32_score(const float *x, size_t xs, size_t d, size_t n,
                    const float *q, size_t G, float *scores, size_t ss);
void oc_kvf32_accum(const float *x, size_t xs, size_t d, size_t n,
                    const float *w, size_t ws, size_t G, float *acc);

/* Symmetric int8 row encode (scale = amax/127), used by the exact slots. */
void oc_kvq8_encode_row(const float *x, size_t d, int8_t *codes, float *scale);

/* ── Cache ───────────────────────────────────────────────────────────── */

typedef struct {
    unsigned      k_bits, v_bits;    /* 2..4 each                       */
    uint32_t      n_sink;            /* exact attention-sink positions  */
    uint32_t      window;            /* exact recent-position ring      */
    OcKvRqRotKind rot;
    uint64_t      seed;
} OcKvRqParams;

typedef struct OcKvRqCache {
    size_t        n_layers, n_kv, d, n_ctx;
    OcKvRqParams  p;
    OcKvRqRot     rot;
    OcKvRqCodec   kc, vc;
    size_t        layer_bytes;       /* bytes of one layer's region     */
    uint8_t     **layer;             /* [n_layers] mmap regions         */
    size_t        n_slots;           /* n_sink + window                 */
    int8_t       *xq;                /* [layer][2][head][slot][d]       */
    float        *xs;                /* [layer][2][head][slot]          */
    int64_t      *tag;               /* [layer][slot] position or -1    */
    size_t        xq_bytes, xs_bytes, tag_bytes;
} OcKvRqCache;

OcError oc_kvrq_cache_init(OcKvRqCache *c, size_t n_layers, size_t n_kv,
                           size_t d, size_t n_ctx, const OcKvRqParams *p);
void oc_kvrq_cache_free(OcKvRqCache *c);
/* Forget everything (tags only; pages stay committed). */
void oc_kvrq_cache_clear(OcKvRqCache *c);
/* Rewind: positions >= pos are no longer valid. */
void oc_kvrq_cache_rewind(OcKvRqCache *c, int64_t pos);
/* Bytes one position costs across all layers/heads (RQ blocks only). */
size_t oc_kvrq_bytes_per_token(const OcKvRqCache *c);

/* Store K and V of one position for every kv head. k/v are [n_kv][d] in the
 * ORIGINAL (unrotated) domain; `scratch` must hold 2*d floats. */
void oc_kvrq_store(OcKvRqCache *c, size_t layer, int64_t pos, const float *k,
                   const float *v, float *scratch);

static inline const uint8_t *oc_kvrq_kblocks(const OcKvRqCache *c,
                                             size_t layer, size_t head)
{
    return c->layer[layer] + head * c->n_ctx * c->kc.block_bytes;
}
static inline const uint8_t *oc_kvrq_vblocks(const OcKvRqCache *c,
                                             size_t layer, size_t head)
{
    return c->layer[layer] + c->n_kv * c->n_ctx * c->kc.block_bytes +
           head * c->n_ctx * c->vc.block_bytes;
}
/* Exact-slot rows for (layer, kind 0=K/1=V, head): slot s at +s*d. */
static inline const int8_t *oc_kvrq_xq(const OcKvRqCache *c, size_t layer,
                                       size_t kind, size_t head)
{
    return c->xq + (((layer * 2u + kind) * c->n_kv + head) * c->n_slots) * c->d;
}
static inline const float *oc_kvrq_xs(const OcKvRqCache *c, size_t layer,
                                      size_t kind, size_t head)
{
    return c->xs + ((layer * 2u + kind) * c->n_kv + head) * c->n_slots;
}
/* Slot index for position t, or -1 when t has no exact slot. */
static inline int64_t oc_kvrq_slot(const OcKvRqCache *c, size_t layer,
                                   int64_t t)
{
    if (t < 0) return -1;
    int64_t s;
    if ((uint64_t)t < c->p.n_sink) s = t;
    else if (c->p.window == 0) return -1;
    else s = (int64_t)c->p.n_sink + t % (int64_t)c->p.window;
    return c->tag[layer * c->n_slots + (size_t)s] == t ? s : -1;
}

#ifdef __cplusplus
}
#endif
