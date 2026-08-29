/*
 * attention_sink.c — StreamingLLM attention sinks implementation.
 */
#include "oxidize/attention_sink.h"

#include <stdlib.h>
#include <string.h>


OcError oc_attn_sink_config_init(OcAttentionSinkConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    cfg->sink_size = OC_SINK_DEFAULT_SINK_SIZE;
    cfg->window_size = OC_SINK_DEFAULT_WINDOW_SIZE;
    return OC_OK;
}

OcError oc_attn_sink_init(OcAttentionSink *sink,
                          const OcAttentionSinkConfig *config,
                          uint32_t head_dim, uint32_t n_heads)
{
    if (!sink || !config || head_dim == 0 || n_heads == 0)
        return OC_ERR_INVALID_ARG;
    if (config->sink_size > OC_SINK_MAX_SINK_SIZE)
        return OC_ERR_INVALID_ARG;

    memset(sink, 0, sizeof(*sink));
    sink->config = *config;
    sink->head_dim = head_dim;
    sink->n_heads = n_heads;

    size_t entry_size = (size_t)head_dim * n_heads * sizeof(float);
    size_t sink_total = entry_size * config->sink_size;
    size_t window_total = entry_size * config->window_size;

    sink->sink_k = malloc(sink_total);
    sink->sink_v = malloc(sink_total);
    sink->window_k = malloc(window_total);
    sink->window_v = malloc(window_total);

    if (!sink->sink_k || !sink->sink_v || !sink->window_k || !sink->window_v) {
        oc_attn_sink_free(sink);
        return OC_ERR_OOM;
    }

    memset(sink->sink_k, 0, sink_total);
    memset(sink->sink_v, 0, sink_total);
    memset(sink->window_k, 0, window_total);
    memset(sink->window_v, 0, window_total);

    sink->window_start = 0;
    sink->window_count = 0;
    sink->pos = 0;
    sink->initialized = true;
    return OC_OK;
}

OcError oc_attn_sink_append(OcAttentionSink *sink,
                            const float *key, const float *value,
                            uint32_t seq_pos, uint32_t *out_slot)
{
    if (!sink || !sink->initialized || !key || !value)
        return OC_ERR_INVALID_ARG;

    size_t entry_size = (size_t)sink->head_dim * sink->n_heads * sizeof(float);

    if (seq_pos < sink->config.sink_size) {
        /* Store in sink cache. */
        size_t offset = seq_pos * entry_size;
        memcpy((char *)sink->sink_k + offset, key, entry_size);
        memcpy((char *)sink->sink_v + offset, value, entry_size);
        if (out_slot) *out_slot = seq_pos;
        if (seq_pos + 1 > sink->sink_count)
            sink->sink_count = seq_pos + 1;
    } else {
        /* Store in window cache (ring buffer). */
        uint32_t idx;
        if (sink->window_count < sink->config.window_size) {
            idx = (sink->window_start + sink->window_count) %
                  sink->config.window_size;
            sink->window_count++;
        } else {
            /* Evict oldest. */
            idx = sink->window_start;
            sink->window_start = (sink->window_start + 1) %
                                 sink->config.window_size;
        }

        size_t offset = (size_t)idx * entry_size;
        memcpy((char *)sink->window_k + offset, key, entry_size);
        memcpy((char *)sink->window_v + offset, value, entry_size);
        if (out_slot) *out_slot = sink->config.sink_size + idx;
    }

    sink->pos = seq_pos + 1;
    return OC_OK;
}

OcError oc_attn_sink_get(const OcAttentionSink *sink, uint32_t slot,
                         const float **out_key, const float **out_value)
{
    if (!sink || !sink->initialized || !out_key || !out_value)
        return OC_ERR_INVALID_ARG;

    size_t entry_size = (size_t)sink->head_dim * sink->n_heads * sizeof(float);

    if (slot < sink->sink_count) {
        *out_key = (const float *)((const char *)sink->sink_k +
                                    slot * entry_size);
        *out_value = (const float *)((const char *)sink->sink_v +
                                      slot * entry_size);
        return OC_OK;
    }

    uint32_t window_slot = slot - sink->config.sink_size;
    if (window_slot >= sink->window_count)
        return OC_ERR_INVALID_ARG;

    /* Map slot to ring buffer index. */
    uint32_t idx = (sink->window_start + window_slot) %
                    sink->config.window_size;
    *out_key = (const float *)((const char *)sink->window_k +
                                (size_t)idx * entry_size);
    *out_value = (const float *)((const char *)sink->window_v +
                                  (size_t)idx * entry_size);
    return OC_OK;
}

uint32_t oc_attn_sink_size(const OcAttentionSink *sink)
{
    if (!sink || !sink->initialized) return 0;
    return sink->sink_count + sink->window_count;
}

OcError oc_attn_sink_evict_oldest(OcAttentionSink *sink)
{
    if (!sink || !sink->initialized) return OC_ERR_INVALID_ARG;
    if (sink->window_count == 0) return OC_OK;

    sink->window_start = (sink->window_start + 1) %
                         sink->config.window_size;
    sink->window_count--;
    return OC_OK;
}

OcError oc_attn_sink_reset_window(OcAttentionSink *sink)
{
    if (!sink || !sink->initialized) return OC_ERR_INVALID_ARG;
    sink->window_start = 0;
    sink->window_count = 0;
    /* Don't clear the actual memory; it'll be overwritten on append. */
    return OC_OK;
}

void oc_attn_sink_free(OcAttentionSink *sink)
{
    if (!sink) return;
    free(sink->sink_k);
    free(sink->sink_v);
    free(sink->window_k);
    free(sink->window_v);
    memset(sink, 0, sizeof(*sink));
}
