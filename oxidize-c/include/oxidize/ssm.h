#ifndef OXIDIZE_SSM_H
#define OXIDIZE_SSM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_SSM_MAX_CHECKPOINTS 2
#define OC_SSM_MAX_DIM         8192
#define OC_SSM_MAX_CONV_HISTORY 512

/* Convolution history ring buffer (per-layer). */
typedef struct {
    float   *slots;       /* [capacity * dim] circular buffer */
    size_t   dim;
    size_t   capacity;
    size_t   head;
    size_t   len;
} OcSsmConvRing;

/* SSM state checkpoint for speculative rollback. */
typedef struct {
    size_t   pos;               /* position in sequence */
    float   *states;            /* [n_layers * state_dim] */
    OcSsmConvRing *conv_rings;  /* [n_layers] convolution rings */
    size_t   n_layers;
} OcSsmCheckpoint;

/* SSM engine managing recurrent state across layers. */
typedef struct {
    float   *ssm_states;        /* [n_layers * state_dim] */
    size_t   n_layers;
    size_t   state_dim;
    size_t   ssm_pos;           /* current sequence position */
    OcSsmConvRing *conv_buffers; /* [n_layers] convolution history */
    OcSsmCheckpoint checkpoints[OC_SSM_MAX_CHECKPOINTS];
    size_t   n_checkpoints;
} OcSsmEngine;

/* ConvRing API */
OcError oc_ssm_conv_ring_init(OcSsmConvRing *ring, size_t capacity, size_t dim);
void oc_ssm_conv_ring_free(OcSsmConvRing *ring);
OcError oc_ssm_conv_ring_push(OcSsmConvRing *ring, const float *frame, size_t frame_len);
OcError oc_ssm_conv_ring_past(const OcSsmConvRing *ring, size_t steps_back, const float **out, size_t *out_len);
double oc_ssm_conv_ring_checksum(const OcSsmConvRing *ring);
size_t oc_ssm_conv_ring_len(const OcSsmConvRing *ring);

/* Engine API */
OcError oc_ssm_engine_init(OcSsmEngine *engine, size_t n_layers, size_t state_dim,
                           size_t conv_capacity, size_t conv_dim);
void oc_ssm_engine_free(OcSsmEngine *engine);

/* Push a checkpoint at position pos (keeps at most OC_SSM_MAX_CHECKPOINTS). */
OcError oc_ssm_push_checkpoint(OcSsmEngine *engine, size_t pos);

/* Rollback to checkpoint at position pos. Returns OC_ERR_INVALID_ARG if not found. */
OcError oc_ssm_rollback(OcSsmEngine *engine, size_t pos);

/* Advance the SSM position by n steps. */
OcError oc_ssm_advance(OcSsmEngine *engine, size_t n);

/* Get the current SSM position. */
size_t oc_ssm_position(const OcSsmEngine *engine);

/* Get the number of checkpoints stored. */
size_t oc_ssm_n_checkpoints(const OcSsmEngine *engine);

/* Clear all checkpoints. */
void oc_ssm_clear_checkpoints(OcSsmEngine *engine);

/* Reset engine state (clears states, conv history, checkpoints, pos). */
void oc_ssm_reset(OcSsmEngine *engine);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_SSM_H */
