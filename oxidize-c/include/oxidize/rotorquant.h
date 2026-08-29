#ifndef OXIDIZE_ROTORQUANT_H
#define OXIDIZE_ROTORQUANT_H

#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif


#define OC_RQ_MIN_BITS   2u
#define OC_RQ_MAX_BITS   4u
#define OC_RQ_MAX_LEVELS 16u  /* 2^OC_RQ_MAX_BITS */
#define OC_RQ_DEFAULT_SEED 42u


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


/* Build a quantizer. Solves the Lloyd-Max codebook for (d, bits) and derives */
OcError oc_rotorquant_init(OcRotorQuant *rq, OcRotorQuantVariant variant,
                           size_t d, unsigned bits, uint64_t seed);

/* Release rotation storage and zero the struct. Safe on NULL. */
void oc_rotorquant_free(OcRotorQuant *rq);


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


/* Matched forward/inverse block rotation over a d_padded-element buffer,
 * in place. unrotate(rotate(v)) == v. Encode uses rotate, decode uses
 * unrotate; do not mix them up (see CRITICAL note). */
void oc_rotorquant_rotate(const OcRotorQuant *rq, float *v);
void oc_rotorquant_unrotate(const OcRotorQuant *rq, float *v);

/* Solve the Lloyd-Max fixed point for the coordinate distribution of a randomly rotated unit vector in R^d, approximated by N(0, 1/d). */
OcError oc_rotorquant_lloyd_max(size_t d, unsigned bits, float *centroids_out);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_ROTORQUANT_H */
