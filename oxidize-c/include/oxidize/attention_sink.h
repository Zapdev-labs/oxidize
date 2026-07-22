/*
 * attention_sink.h — StreamingLLM attention sinks.
 *
 * Keeps the first few K and V vectors fixed (sinks) and rotates the rest
 * using a sliding window, enabling efficient long-context generation.
 */
#ifndef OXIDIZE_ATTENTION_SINK_H
#define OXIDIZE_ATTENTION_SINK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Constants ──────────────────────────────────────────────────────── */

#define OC_SINK_DEFAULT_SINK_SIZE 4
#define OC_SINK_DEFAULT_WINDOW_SIZE 4096
#define OC_SINK_MAX_SINK_SIZE 64

/* ─── Types ─────────────────────────────────────────────────────────── */

typedef struct {
    uint32_t sink_size;       /* number of initial tokens to keep (default 4) */
    uint32_t window_size;     /* sliding window size (default 4096) */
} OcAttentionSinkConfig;

typedef struct {
    OcAttentionSinkConfig config;
    uint32_t head_dim;
    uint32_t n_heads;

    /* Sink K/V cache: sink_size entries. */
    float *sink_k;            /* [sink_size * n_heads * head_dim] */
    float *sink_v;            /* same size */

    /* Window K/V cache: ring buffer of window_size entries. */
    float *window_k;         /* [window_size * n_heads * head_dim] */
    float *window_v;         /* same size */
    uint32_t window_start;    /* index of oldest entry in window */
    uint32_t window_count;   /* number of valid entries in window */
    uint32_t sink_count;      /* number of valid sink entries written */

    /* Total sequence position (monotonically increasing). */
    uint32_t pos;

    bool initialized;
} OcAttentionSink;

/* ─── API ────────────────────────────────────────────────────────────── */

/* Initialize with config. Allocates K/V caches. */
OcError oc_attn_sink_init(OcAttentionSink *sink,
                          const OcAttentionSinkConfig *config,
                          uint32_t head_dim, uint32_t n_heads);

/* Append K/V vectors for a token at `seq_pos`.
 * If seq_pos < sink_size, stores in sink cache.
 * Otherwise stores in window cache (evicting oldest if full).
 * Returns the slot position (index into the combined sink+window cache). */
OcError oc_attn_sink_append(OcAttentionSink *sink,
                            const float *key, const float *value,
                            uint32_t seq_pos, uint32_t *out_slot);

/* Get K/V at a given slot position.
 * Slot 0..sink_size-1 are sink entries.
 * Slot sink_size.. are window entries. */
OcError oc_attn_sink_get(const OcAttentionSink *sink, uint32_t slot,
                         const float **out_key, const float **out_value);

/* Get the number of currently cached entries (sink + window). */
uint32_t oc_attn_sink_size(const OcAttentionSink *sink);

/* Evict the oldest non-sink entry from the window cache. */
OcError oc_attn_sink_evict_oldest(OcAttentionSink *sink);

/* Reset the sink (clear window but keep sinks). */
OcError oc_attn_sink_reset_window(OcAttentionSink *sink);

/* Free resources. */
void oc_attn_sink_free(OcAttentionSink *sink);

/* Initialize config with defaults. */
OcError oc_attn_sink_config_init(OcAttentionSinkConfig *cfg);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_ATTENTION_SINK_H */
