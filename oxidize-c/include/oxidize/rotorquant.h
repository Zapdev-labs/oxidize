/*
 * rotorquant.h — RotorQuant KV-cache compression (PlanarQuant / IsoQuant).
 *
 * Rotation-based vector quantization for KV cache entries of dimension d
 * (d = head_dim, typically 64/128). Port of the Python reference in
 * turboquant/planarquant.py and turboquant/isoquant.py.
 *
 * Why rotations: after a random rotation, a unit vector's coordinates are
 * near-isotropic (~N(0, 1/d)), so a *single* fixed Lloyd-Max scalar codebook
 * is MSE-optimal for every coordinate. No per-block scales, no outlier
 * handling — the rotation does the whitening, the norm carries the scale.
 *
 *   encode: n = ||x||; u = x/n; pad; block-rotate; nearest-centroid per coord
 *   decode: v = centroids[idx]; INVERSE block-rotate; multiply by n
 *
 * Variants:
 *   OC_RQ_PLANAR — 2D Givens blocks, ceil(d/2) groups, 1 angle each.
 *   OC_RQ_ISO    — 4D quaternion blocks, ceil(d/4) groups, full SO(4)
 *                  sandwich T(v) = q_L v conj(q_R); inverse conj(q_L) v q_R.
 *
 * ── CRITICAL ────────────────────────────────────────────────────────────
 * Decode MUST apply the inverse of the encode rotation. Getting this
 * backwards (applying the forward rotation twice) is silent: the output is
 * still a plausible unit-norm vector, but it is wrong. In the reference this
 * regressed WikiText-2 perplexity from 7.05 to 15,369. This API therefore
 * never exposes a "rotate" step to the caller on the encode/decode path —
 * oc_rotorquant_encode()/decode() own the direction. The low-level
 * oc_rotorquant_rotate()/unrotate() pair exists only for tests and fused
 * kernels; they are documented as a matched pair and nothing else.
 *
 * ── Storage layout (one vector) ─────────────────────────────────────────
 *   bytes 0..3   : float32 ||x||, little-endian native
 *   bytes 4..    : d_padded indices of `bits` bits each, tightly packed
 *
 * Indices are packed into a little-endian bit stream: index i occupies bit
 * positions [i*bits, i*bits + bits) counting from bit 0 of byte 0, LSB
 * first. A 3-bit index therefore costs exactly 3 bits — e.g. indices 0..7
 * for d_padded=8, bits=3 occupy 3 bytes, not 8. Trailing bits of the final
 * byte are zero.
 *
 *   bytes_per_vector = 4 + ceil(d_padded * bits / 8)
 *
 * ── Size / compression, d = 128 (d_padded = 128 for both variants) ──────
 *   bits  indices  total   vs f32 (512 B)  vs f16 (256 B)
 *     2     32 B    36 B       14.22x          7.11x
 *     3     48 B    52 B        9.85x          4.92x
 *     4     64 B    68 B        7.53x          3.76x
 *
 * For d = 64 (d_padded = 64): 20/28/36 B → 12.80x/9.14x/7.11x vs f32.
 * The 4-byte norm is the whole overhead; it amortizes away as d grows.
 * Use oc_rotorquant_compression_ratio() to compute this at runtime.
 *
 * The rotations are fixed and seeded (never learned), so an OcRotorQuant
 * built with the same (variant, d, bits, seed) decodes any buffer another
 * instance encoded — including across processes.
 */
#ifndef OXIDIZE_ROTORQUANT_H
#define OXIDIZE_ROTORQUANT_H

#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Constants ──────────────────────────────────────────────────────── */

#define OC_RQ_MIN_BITS   2u
#define OC_RQ_MAX_BITS   4u
#define OC_RQ_MAX_LEVELS 16u  /* 2^OC_RQ_MAX_BITS */
#define OC_RQ_DEFAULT_SEED 42u

/* ─── Types ─────────────────────────────────────────────────────────── */

typedef enum {
    OC_RQ_PLANAR = 0, /* 2D Givens blocks  */
    OC_RQ_ISO    = 1, /* 4D quaternion blocks */
} OcRotorQuantVariant;

/* Quantizer state. Immutable after init; encode/decode are const and
 * therefore safe to share across threads. */
typedef struct OcRotorQuant {
    OcRotorQuantVariant variant;
    size_t   d;                 /* logical vector dimension             */
    unsigned bits;              /* 2..4                                 */
    unsigned n_levels;          /* 1u << bits                           */
    size_t   block;             /* 2 (planar) or 4 (iso)                */
    size_t   n_groups;          /* ceil(d / block)                      */
    size_t   d_padded;          /* n_groups * block                     */
    size_t   bytes_per_vector;  /* 4 + ceil(d_padded*bits/8)            */
    uint64_t seed;
    float    centroids[OC_RQ_MAX_LEVELS];
    /* Rotation parameters, one entry per group.
     * planar: 2 floats  (cos, sin)
     * iso:    8 floats  (q_L[4], q_R[4]), both unit quaternions */
    float   *rot;
} OcRotorQuant;

/* ─── Lifecycle ─────────────────────────────────────────────────────── */

/* Build a quantizer. Solves the Lloyd-Max codebook for (d, bits) and derives
 * the fixed rotations from `seed`. Returns OC_ERR_INVALID_ARG for d == 0 or
 * bits outside [OC_RQ_MIN_BITS, OC_RQ_MAX_BITS], OC_ERR_OOM on allocation
 * failure. Free with oc_rotorquant_free(). */
OcError oc_rotorquant_init(OcRotorQuant *rq, OcRotorQuantVariant variant,
                           size_t d, unsigned bits, uint64_t seed);

/* Release rotation storage and zero the struct. Safe on NULL. */
void oc_rotorquant_free(OcRotorQuant *rq);

/* ─── Encode / decode ───────────────────────────────────────────────── */

/* Bytes one encoded vector occupies. 0 if `rq` is NULL. */
size_t oc_rotorquant_bytes_per_vector(const OcRotorQuant *rq);

/* Compression ratio against an uncompressed cache storing `src_elem_bytes`
 * per element (4 for f32, 2 for f16). Returns 0.0f on bad input. */
float oc_rotorquant_compression_ratio(const OcRotorQuant *rq,
                                      size_t src_elem_bytes);

/* Encode one d-element vector into `out` (bytes_per_vector bytes). */
OcError oc_rotorquant_encode(const OcRotorQuant *rq, const float *x,
                             uint8_t *out);

/* Decode one vector previously written by oc_rotorquant_encode() into `x_out`
 * (d elements). Applies the inverse rotation — see the CRITICAL note above. */
OcError oc_rotorquant_decode(const OcRotorQuant *rq, const uint8_t *in,
                             float *x_out);

/* Batched forms. `x` is n_vectors*d floats; `out`/`in` is
 * n_vectors*bytes_per_vector bytes, vectors stored back to back. */
OcError oc_rotorquant_encode_many(const OcRotorQuant *rq, const float *x,
                                  size_t n_vectors, uint8_t *out);
OcError oc_rotorquant_decode_many(const OcRotorQuant *rq, const uint8_t *in,
                                  size_t n_vectors, float *x_out);

/* ─── Low-level (tests, fused kernels) ──────────────────────────────── */

/* Matched forward/inverse block rotation over a d_padded-element buffer,
 * in place. unrotate(rotate(v)) == v. Encode uses rotate, decode uses
 * unrotate; do not mix them up (see CRITICAL note). */
void oc_rotorquant_rotate(const OcRotorQuant *rq, float *v);
void oc_rotorquant_unrotate(const OcRotorQuant *rq, float *v);

/* Solve the Lloyd-Max fixed point for the coordinate distribution of a
 * randomly rotated unit vector in R^d, approximated by N(0, 1/d). Writes
 * (1u << bits) ascending centroids to `centroids_out`. This is what
 * oc_rotorquant_init() calls; exposed so tests can verify the solver. */
OcError oc_rotorquant_lloyd_max(size_t d, unsigned bits, float *centroids_out);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_ROTORQUANT_H */
