#define _POSIX_C_SOURCE 200809L
#include "oxidize/ssm.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ─── ConvRing ─────────────────────────────────────────────────────────── */

OcError oc_ssm_conv_ring_init(OcSsmConvRing *ring, size_t capacity, size_t dim)
{
    if (!ring || dim == 0) return OC_ERR_INVALID_ARG;
    if (capacity == 0) capacity = 1;
    ring->dim = dim;
    ring->capacity = capacity;
    ring->head = 0;
    ring->len = 0;
    ring->slots = calloc(capacity * dim, sizeof(float));
    if (!ring->slots) return OC_ERR_OOM;
    return OC_OK;
}

void oc_ssm_conv_ring_free(OcSsmConvRing *ring)
{
    if (!ring) return;
    free(ring->slots);
    memset(ring, 0, sizeof(*ring));
}

OcError oc_ssm_conv_ring_push(OcSsmConvRing *ring, const float *frame, size_t frame_len)
{
    if (!ring || !frame) return OC_ERR_INVALID_ARG;
    if (frame_len != ring->dim) return OC_ERR_INVALID_ARG;
    size_t start = ring->head * ring->dim;
    memcpy(&ring->slots[start], frame, ring->dim * sizeof(float));
    ring->head = (ring->head + 1) % ring->capacity;
    if (ring->len < ring->capacity) ring->len++;
    return OC_OK;
}

OcError oc_ssm_conv_ring_past(const OcSsmConvRing *ring, size_t steps_back, const float **out, size_t *out_len)
{
    if (!ring || !out || !out_len) return OC_ERR_INVALID_ARG;
    if (steps_back == 0 || steps_back > ring->len) return OC_ERR_INVALID_ARG;
    size_t idx = (ring->head + ring->capacity - steps_back) % ring->capacity;
    size_t start = idx * ring->dim;
    *out = &ring->slots[start];
    *out_len = ring->dim;
    return OC_OK;
}

double oc_ssm_conv_ring_checksum(const OcSsmConvRing *ring)
{
    if (!ring || !ring->slots) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < ring->capacity * ring->dim; i++)
        sum += (double)ring->slots[i];
    sum += (double)ring->head * 1e-3;
    sum += (double)ring->len * 1e-6;
    return sum;
}

size_t oc_ssm_conv_ring_len(const OcSsmConvRing *ring)
{
    return ring ? ring->len : 0;
}

/* ─── Engine ──────────────────────────────────────────────────────────── */

OcError oc_ssm_engine_init(OcSsmEngine *engine, size_t n_layers, size_t state_dim,
                           size_t conv_capacity, size_t conv_dim)
{
    if (!engine || n_layers == 0 || state_dim == 0) return OC_ERR_INVALID_ARG;
    if (state_dim > OC_SSM_MAX_DIM) return OC_ERR_INVALID_ARG;
    if (conv_capacity > OC_SSM_MAX_CONV_HISTORY) return OC_ERR_INVALID_ARG;

    memset(engine, 0, sizeof(*engine));
    engine->n_layers = n_layers;
    engine->state_dim = state_dim;
    engine->ssm_pos = 0;
    engine->n_checkpoints = 0;

    engine->ssm_states = calloc(n_layers * state_dim, sizeof(float));
    if (!engine->ssm_states) return OC_ERR_OOM;

    engine->conv_buffers = calloc(n_layers, sizeof(OcSsmConvRing));
    if (!engine->conv_buffers) {
        free(engine->ssm_states);
        engine->ssm_states = NULL;
        return OC_ERR_OOM;
    }

    for (size_t i = 0; i < n_layers; i++) {
        OcError e = oc_ssm_conv_ring_init(&engine->conv_buffers[i], conv_capacity, conv_dim);
        if (e != OC_OK) {
            for (size_t j = 0; j < i; j++)
                oc_ssm_conv_ring_free(&engine->conv_buffers[j]);
            free(engine->conv_buffers);
            engine->conv_buffers = NULL;
            free(engine->ssm_states);
            engine->ssm_states = NULL;
            return e;
        }
    }
    return OC_OK;
}

void oc_ssm_engine_free(OcSsmEngine *engine)
{
    if (!engine) return;
    for (size_t i = 0; i < engine->n_checkpoints; i++) {
        free(engine->checkpoints[i].states);
        if (engine->checkpoints[i].conv_rings) {
            for (size_t j = 0; j < engine->checkpoints[i].n_layers; j++)
                oc_ssm_conv_ring_free(&engine->checkpoints[i].conv_rings[j]);
            free(engine->checkpoints[i].conv_rings);
        }
    }
    if (engine->conv_buffers) {
        for (size_t i = 0; i < engine->n_layers; i++)
            oc_ssm_conv_ring_free(&engine->conv_buffers[i]);
        free(engine->conv_buffers);
    }
    free(engine->ssm_states);
    memset(engine, 0, sizeof(*engine));
}

static OcError clone_state(OcSsmEngine *engine, OcSsmCheckpoint *cp)
{
    cp->pos = engine->ssm_pos;
    cp->n_layers = engine->n_layers;
    cp->states = malloc(engine->n_layers * engine->state_dim * sizeof(float));
    if (!cp->states) return OC_ERR_OOM;
    memcpy(cp->states, engine->ssm_states,
           engine->n_layers * engine->state_dim * sizeof(float));

    cp->conv_rings = calloc(engine->n_layers, sizeof(OcSsmConvRing));
    if (!cp->conv_rings) {
        free(cp->states);
        cp->states = NULL;
        return OC_ERR_OOM;
    }
    for (size_t i = 0; i < engine->n_layers; i++) {
        OcError e = oc_ssm_conv_ring_init(&cp->conv_rings[i],
                                          engine->conv_buffers[i].capacity,
                                          engine->conv_buffers[i].dim);
        if (e != OC_OK) {
            for (size_t j = 0; j < i; j++)
                oc_ssm_conv_ring_free(&cp->conv_rings[j]);
            free(cp->conv_rings);
            cp->conv_rings = NULL;
            free(cp->states);
            cp->states = NULL;
            return e;
        }
        /* Copy slots. */
        memcpy(cp->conv_rings[i].slots, engine->conv_buffers[i].slots,
               engine->conv_buffers[i].capacity * engine->conv_buffers[i].dim * sizeof(float));
        cp->conv_rings[i].head = engine->conv_buffers[i].head;
        cp->conv_rings[i].len = engine->conv_buffers[i].len;
    }
    return OC_OK;
}

OcError oc_ssm_push_checkpoint(OcSsmEngine *engine, size_t pos)
{
    if (!engine) return OC_ERR_INVALID_ARG;

    /* Remove existing checkpoint at same pos. */
    for (size_t i = 0; i < engine->n_checkpoints; i++) {
        if (engine->checkpoints[i].pos == pos) {
            free(engine->checkpoints[i].states);
            if (engine->checkpoints[i].conv_rings) {
                for (size_t j = 0; j < engine->checkpoints[i].n_layers; j++)
                    oc_ssm_conv_ring_free(&engine->checkpoints[i].conv_rings[j]);
                free(engine->checkpoints[i].conv_rings);
            }
            /* Shift remaining. */
            for (size_t j = i; j < engine->n_checkpoints - 1; j++)
                engine->checkpoints[j] = engine->checkpoints[j + 1];
            engine->n_checkpoints--;
            break;
        }
    }

    /* Evict oldest if at max. */
    if (engine->n_checkpoints >= OC_SSM_MAX_CHECKPOINTS) {
        size_t evict = 0;
        free(engine->checkpoints[evict].states);
        if (engine->checkpoints[evict].conv_rings) {
            for (size_t j = 0; j < engine->checkpoints[evict].n_layers; j++)
                oc_ssm_conv_ring_free(&engine->checkpoints[evict].conv_rings[j]);
            free(engine->checkpoints[evict].conv_rings);
        }
        for (size_t j = 0; j < engine->n_checkpoints - 1; j++)
            engine->checkpoints[j] = engine->checkpoints[j + 1];
        engine->n_checkpoints--;
    }

    OcError e = clone_state(engine, &engine->checkpoints[engine->n_checkpoints]);
    if (e != OC_OK) return e;
    engine->n_checkpoints++;
    return OC_OK;
}

OcError oc_ssm_rollback(OcSsmEngine *engine, size_t pos)
{
    if (!engine) return OC_ERR_INVALID_ARG;
    for (size_t i = 0; i < engine->n_checkpoints; i++) {
        if (engine->checkpoints[i].pos == pos) {
            memcpy(engine->ssm_states, engine->checkpoints[i].states,
                   engine->n_layers * engine->state_dim * sizeof(float));
            engine->ssm_pos = pos;
            for (size_t j = 0; j < engine->n_layers; j++) {
                memcpy(engine->conv_buffers[j].slots,
                       engine->checkpoints[i].conv_rings[j].slots,
                       engine->conv_buffers[j].capacity * engine->conv_buffers[j].dim * sizeof(float));
                engine->conv_buffers[j].head = engine->checkpoints[i].conv_rings[j].head;
                engine->conv_buffers[j].len = engine->checkpoints[i].conv_rings[j].len;
            }
            return OC_OK;
        }
    }
    return OC_ERR_INVALID_ARG;
}

OcError oc_ssm_advance(OcSsmEngine *engine, size_t n)
{
    if (!engine) return OC_ERR_INVALID_ARG;
    engine->ssm_pos += n;
    return OC_OK;
}

size_t oc_ssm_position(const OcSsmEngine *engine)
{
    return engine ? engine->ssm_pos : 0;
}

size_t oc_ssm_n_checkpoints(const OcSsmEngine *engine)
{
    return engine ? engine->n_checkpoints : 0;
}

void oc_ssm_clear_checkpoints(OcSsmEngine *engine)
{
    if (!engine) return;
    for (size_t i = 0; i < engine->n_checkpoints; i++) {
        free(engine->checkpoints[i].states);
        if (engine->checkpoints[i].conv_rings) {
            for (size_t j = 0; j < engine->checkpoints[i].n_layers; j++)
                oc_ssm_conv_ring_free(&engine->checkpoints[i].conv_rings[j]);
            free(engine->checkpoints[i].conv_rings);
        }
    }
    engine->n_checkpoints = 0;
}

void oc_ssm_reset(OcSsmEngine *engine)
{
    if (!engine) return;
    if (engine->ssm_states)
        memset(engine->ssm_states, 0, engine->n_layers * engine->state_dim * sizeof(float));
    for (size_t i = 0; i < engine->n_layers; i++) {
        if (engine->conv_buffers[i].slots)
            memset(engine->conv_buffers[i].slots, 0,
                   engine->conv_buffers[i].capacity * engine->conv_buffers[i].dim * sizeof(float));
        engine->conv_buffers[i].head = 0;
        engine->conv_buffers[i].len = 0;
    }
    engine->ssm_pos = 0;
    oc_ssm_clear_checkpoints(engine);
}
