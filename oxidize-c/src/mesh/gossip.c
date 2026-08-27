/*
 * gossip.c — gossip protocol for cluster node discovery and health checking.
 *
 * Transport-agnostic membership layer. Each node owns a list of known peers
 * (including itself) and refreshes timestamps via oc_gossip_tick. Peers
 * whose last_seen is older than config.timeout_ms are flagged unhealthy.
 * Peer lists are exchanged with oc_gossip_serialize / deserialize / merge.
 */
#include "oxidize/gossip.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── Helpers ───────────────────────────────────────────────────────── */

static void gossip_node_init(OcGossipNode *node, uint64_t id,
                             const char *addr, uint16_t port,
                             uint64_t last_seen)
{
    memset(node, 0, sizeof(*node));
    node->id        = id;
    node->port      = port;
    node->last_seen = last_seen;
    node->healthy   = true;
    if (addr) {
        snprintf(node->addr, sizeof(node->addr), "%s", addr);
    }
}

/* Find index of node by id, or -1 if absent. */
static int32_t gossip_find(const OcGossipState *state, uint64_t id)
{
    for (uint32_t i = 0; i < state->n_nodes; i++) {
        if (state->nodes[i].id == id) return (int32_t)i;
    }
    return -1;
}

/* ─── Config helpers ────────────────────────────────────────────────── */

OcError oc_gossip_config_init(OcGossipConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    cfg->interval_ms = OC_GOSSIP_DEFAULT_INTERVAL_MS;
    cfg->timeout_ms  = OC_GOSSIP_DEFAULT_TIMEOUT_MS;
    cfg->max_nodes   = OC_GOSSIP_DEFAULT_MAX_NODES;
    cfg->fanout      = OC_GOSSIP_DEFAULT_FANOUT;
    return OC_OK;
}

/* ─── Lifecycle ─────────────────────────────────────────────────────── */

OcError oc_gossip_init(const OcGossipConfig *config, uint64_t self_id,
                       OcGossipState **out)
{
    if (!out) return OC_ERR_INVALID_ARG;
    *out = NULL;

    OcGossipConfig cfg;
    if (config) {
        cfg = *config;
    } else {
        oc_gossip_config_init(&cfg);
    }
    if (cfg.max_nodes == 0) return OC_ERR_INVALID_ARG;
    if (cfg.fanout == 0) cfg.fanout = 1;

    OcGossipState *s = malloc(sizeof(*s));
    if (!s) return OC_ERR_OOM;
    memset(s, 0, sizeof(*s));
    s->config  = cfg;
    s->self_id = self_id;
    s->nodes   = malloc((size_t)cfg.max_nodes * sizeof(OcGossipNode));
    if (!s->nodes) { free(s); return OC_ERR_OOM; }
    s->n_nodes = 0;
    s->last_gossip_time = 0;

    /* Self entry: addr empty, port 0, healthy. */
    gossip_node_init(&s->nodes[0], self_id, NULL, 0, 0);
    s->n_nodes = 1;

    *out = s;
    return OC_OK;
}

void oc_gossip_free(OcGossipState *state)
{
    if (!state) return;
    free(state->nodes);
    memset(state, 0, sizeof(*state));
    free(state);
}

/* ─── Peer management ───────────────────────────────────────────────── */

OcError oc_gossip_add_node(OcGossipState *state, uint64_t id,
                           const char *addr, uint16_t port)
{
    if (!state || !addr) return OC_ERR_INVALID_ARG;

    int32_t idx = gossip_find(state, id);
    if (idx >= 0) {
        /* Refresh existing. */
        OcGossipNode *n = &state->nodes[idx];
        snprintf(n->addr, sizeof(n->addr), "%s", addr);
        n->port    = port;
        n->healthy = true;
        /* last_seen preserved unless caller updates via tick/merge. */
        return OC_OK;
    }
    if (state->n_nodes >= state->config.max_nodes) return OC_ERR_OOM;

    gossip_node_init(&state->nodes[state->n_nodes], id, addr, port,
                     state->last_gossip_time);
    state->n_nodes++;
    return OC_OK;
}

OcError oc_gossip_remove_node(OcGossipState *state, uint64_t id)
{
    if (!state) return OC_ERR_INVALID_ARG;
    if (id == state->self_id) return OC_ERR_INVALID_ARG;

    int32_t idx = gossip_find(state, id);
    if (idx < 0) return OC_OK; /* already absent */

    /* Shift down. */
    for (uint32_t i = (uint32_t)idx; i + 1 < state->n_nodes; i++) {
        state->nodes[i] = state->nodes[i + 1];
    }
    state->n_nodes--;
    return OC_OK;
}

/* ─── Periodic update ───────────────────────────────────────────────── */

OcError oc_gossip_tick(OcGossipState *state, uint64_t current_time_ms)
{
    if (!state) return OC_ERR_INVALID_ARG;
    state->last_gossip_time = current_time_ms;

    /* Self stays healthy; refresh its last_seen. */
    int32_t self_idx = gossip_find(state, state->self_id);
    if (self_idx >= 0) {
        state->nodes[self_idx].last_seen = current_time_ms;
        state->nodes[self_idx].healthy    = true;
    }

    /* Mark unhealthy any node older than timeout. */
    for (uint32_t i = 0; i < state->n_nodes; i++) {
        OcGossipNode *n = &state->nodes[i];
        if (n->id == state->self_id) continue;
        if (current_time_ms > n->last_seen &&
            (current_time_ms - n->last_seen) > state->config.timeout_ms) {
            n->healthy = false;
        }
    }
    return OC_OK;
}

/* ─── Queries ───────────────────────────────────────────────────────── */

OcError oc_gossip_get_nodes(const OcGossipState *state,
                            OcGossipNode *out_nodes, uint32_t max,
                            uint32_t *out_count)
{
    if (!state || !out_count) return OC_ERR_INVALID_ARG;
    uint32_t n = state->n_nodes;
    if (!out_nodes) {
        *out_count = n;
        return OC_OK;
    }
    uint32_t to_copy = n < max ? n : max;
    for (uint32_t i = 0; i < to_copy; i++) {
        out_nodes[i] = state->nodes[i];
    }
    *out_count = to_copy;
    return OC_OK;
}

uint32_t oc_gossip_get_healthy_count(const OcGossipState *state)
{
    if (!state) return 0;
    uint32_t count = 0;
    for (uint32_t i = 0; i < state->n_nodes; i++) {
        if (state->nodes[i].healthy) count++;
    }
    return count;
}

/* ─── Peer exchange ────────────────────────────────────────────────── */

OcError oc_gossip_merge(OcGossipState *state, uint64_t current_time_ms,
                        const OcGossipNode *remote_nodes, uint32_t count)
{
    if (!state || (!remote_nodes && count > 0)) return OC_ERR_INVALID_ARG;

    for (uint32_t i = 0; i < count; i++) {
        const OcGossipNode *r = &remote_nodes[i];
        if (r->id == state->self_id) continue; /* never merge self */

        int32_t idx = gossip_find(state, r->id);
        if (idx >= 0) {
            /* Update fields; take the more-recent last_seen. */
            OcGossipNode *n = &state->nodes[idx];
            snprintf(n->addr, sizeof(n->addr), "%s", r->addr);
            n->port = r->port;
            if (r->last_seen > n->last_seen) {
                n->last_seen = r->last_seen;
            }
            /* A peer reporting itself healthy from a fresh timestamp
             * implies it is reachable — clear the unhealthy flag. */
            if (r->last_seen > 0 && r->healthy) {
                n->healthy = true;
            }
            /* Copy metadata through. */
            snprintf(n->metadata, sizeof(n->metadata), "%s", r->metadata);
            continue;
        }

        /* New node — append if room. */
        if (state->n_nodes >= state->config.max_nodes) continue;
        OcGossipNode *n = &state->nodes[state->n_nodes];
        *n = *r;
        if (n->last_seen == 0) n->last_seen = current_time_ms;
        n->healthy = true;
        state->n_nodes++;
    }
    return OC_OK;
}

/* ─── Serialization ─────────────────────────────────────────────────── */

/* On-the-wire format: u32 count, then count packed OcGossipNode records.
 * We pack by writing fields individually (no padding) so the format is
 * stable across architectures. */

static void put_u32(uint8_t *buf, size_t *off, uint32_t v)
{
    buf[*off]     = (uint8_t)(v & 0xFFu);
    buf[*off + 1] = (uint8_t)((v >> 8) & 0xFFu);
    buf[*off + 2] = (uint8_t)((v >> 16) & 0xFFu);
    buf[*off + 3] = (uint8_t)((v >> 24) & 0xFFu);
    *off += 4;
}

static uint32_t get_u32(const uint8_t *buf, size_t *off)
{
    uint32_t v = (uint32_t)buf[*off]
               | ((uint32_t)buf[*off + 1] << 8)
               | ((uint32_t)buf[*off + 2] << 16)
               | ((uint32_t)buf[*off + 3] << 24);
    *off += 4;
    return v;
}

static void put_u64(uint8_t *buf, size_t *off, uint64_t v)
{
    for (int i = 0; i < 8; i++) {
        buf[*off + i] = (uint8_t)((v >> (8 * i)) & 0xFFu);
    }
    *off += 8;
}

static uint64_t get_u64(const uint8_t *buf, size_t *off)
{
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) {
        v |= (uint64_t)buf[*off + i] << (8 * i);
    }
    *off += 8;
    return v;
}

static size_t gossip_node_packed_size(void)
{
    return sizeof(uint64_t)                        /* id           */
         + OC_GOSSIP_MAX_ADDR_LEN                  /* addr         */
         + sizeof(uint16_t)                        /* port         */
         + sizeof(uint64_t)                        /* last_seen    */
         + sizeof(uint8_t)                         /* healthy      */
         + OC_GOSSIP_MAX_METADATA;                 /* metadata     */
}

static void pack_node(uint8_t *buf, size_t *off, const OcGossipNode *n)
{
    put_u64(buf, off, n->id);
    memcpy(buf + *off, n->addr, OC_GOSSIP_MAX_ADDR_LEN);
    *off += OC_GOSSIP_MAX_ADDR_LEN;
    buf[*off] = (uint8_t)(n->port & 0xFFu);
    buf[*off + 1] = (uint8_t)((n->port >> 8) & 0xFFu);
    *off += 2;
    put_u64(buf, off, n->last_seen);
    buf[*off] = n->healthy ? 1u : 0u;
    *off += 1;
    memcpy(buf + *off, n->metadata, OC_GOSSIP_MAX_METADATA);
    *off += OC_GOSSIP_MAX_METADATA;
}

static void unpack_node(const uint8_t *buf, size_t *off, OcGossipNode *n)
{
    memset(n, 0, sizeof(*n));
    n->id = get_u64(buf, off);
    memcpy(n->addr, buf + *off, OC_GOSSIP_MAX_ADDR_LEN);
    n->addr[OC_GOSSIP_MAX_ADDR_LEN - 1] = '\0';
    *off += OC_GOSSIP_MAX_ADDR_LEN;
    n->port = (uint16_t)((uint16_t)buf[*off]
                        | ((uint16_t)buf[*off + 1] << 8));
    *off += 2;
    n->last_seen = get_u64(buf, off);
    n->healthy = buf[*off] != 0;
    *off += 1;
    memcpy(n->metadata, buf + *off, OC_GOSSIP_MAX_METADATA);
    n->metadata[OC_GOSSIP_MAX_METADATA - 1] = '\0';
    *off += OC_GOSSIP_MAX_METADATA;
}

OcError oc_gossip_serialize(const OcGossipState *state, uint8_t *out_buf,
                            size_t cap, size_t *out_len)
{
    if (!state || !out_len) return OC_ERR_INVALID_ARG;
    if (!out_buf && cap > 0) return OC_ERR_INVALID_ARG;

    size_t needed = 4u + (size_t)state->n_nodes * gossip_node_packed_size();
    if (cap < needed) {
        *out_len = needed;
        return OC_ERR_INVALID_ARG;
    }

    size_t off = 0;
    put_u32(out_buf, &off, state->n_nodes);
    for (uint32_t i = 0; i < state->n_nodes; i++) {
        pack_node(out_buf, &off, &state->nodes[i]);
    }
    *out_len = off;
    return OC_OK;
}

OcError oc_gossip_deserialize(OcGossipState *state, uint64_t current_time_ms,
                              const uint8_t *buf, size_t len)
{
    if (!state || !buf) return OC_ERR_INVALID_ARG;
    if (len < 4u) return OC_ERR_INVALID_ARG;

    size_t off = 0;
    uint32_t count = get_u32(buf, &off);
    if (count > state->config.max_nodes) return OC_ERR_INVALID_ARG;

    size_t per = gossip_node_packed_size();
    if (len < 4u + (size_t)count * per) return OC_ERR_INVALID_ARG;

    /* Decode into a local scratch buffer, then merge. */
    OcGossipNode *tmp = NULL;
    if (count > 0) {
        tmp = malloc((size_t)count * sizeof(OcGossipNode));
        if (!tmp) return OC_ERR_OOM;
        for (uint32_t i = 0; i < count; i++) {
            unpack_node(buf, &off, &tmp[i]);
        }
    }
    OcError e = oc_gossip_merge(state, current_time_ms, tmp, count);
    free(tmp);
    return e;
}
