#include "oxidize/fault_tolerance.h"

#include <stdlib.h>
#include <string.h>


/* Find index of node by id, or -1 if absent. */
static int32_t ft_find(const OcFtManager *mgr, uint64_t node_id)
{
    for (uint32_t i = 0; i < mgr->n_nodes; i++) {
        if (mgr->nodes[i].node_id == node_id) return (int32_t)i;
    }
    return -1;
}

static void ft_node_init(OcFtNodeState *node, uint64_t node_id)
{
    memset(node, 0, sizeof(*node));
    node->node_id           = node_id;
    node->last_heartbeat_ms = 0;
    node->missed_count      = 0;
    node->status            = OC_FT_ALIVE;
}


OcError oc_ft_config_init(OcFtConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    cfg->heartbeat_interval_ms = OC_FT_DEFAULT_INTERVAL_MS;
    cfg->heartbeat_timeout_ms  = OC_FT_DEFAULT_TIMEOUT_MS;
    cfg->max_missed            = OC_FT_DEFAULT_MAX_MISSED;
    cfg->recovery_timeout_ms   = OC_FT_DEFAULT_RECOVERY_MS;
    return OC_OK;
}


OcError oc_ft_init(const OcFtConfig *config, uint64_t self_id,
                   OcFtManager **out)
{
    if (!out) return OC_ERR_INVALID_ARG;
    *out = NULL;

    OcFtConfig cfg;
    if (config) {
        cfg = *config;
    } else {
        oc_ft_config_init(&cfg);
    }
    if (cfg.heartbeat_interval_ms == 0) return OC_ERR_INVALID_ARG;
    if (cfg.heartbeat_timeout_ms  == 0) return OC_ERR_INVALID_ARG;
    if (cfg.max_missed            == 0) cfg.max_missed = 1;

    OcFtManager *m = malloc(sizeof(*m));
    if (!m) return OC_ERR_OOM;
    memset(m, 0, sizeof(*m));
    m->config = cfg;
    m->self_id = self_id;
    m->n_nodes = 0;
    *out = m;
    return OC_OK;
}

void oc_ft_free(OcFtManager *mgr)
{
    if (!mgr) return;
    memset(mgr, 0, sizeof(*mgr));
    free(mgr);
}


OcError oc_ft_add_node(OcFtManager *mgr, uint64_t node_id)
{
    if (!mgr) return OC_ERR_INVALID_ARG;
    if (ft_find(mgr, node_id) >= 0) return OC_OK;  /* idempotent */
    if (mgr->n_nodes >= OC_FT_MAX_NODES) return OC_ERR_OOM;

    ft_node_init(&mgr->nodes[mgr->n_nodes], node_id);
    mgr->n_nodes++;
    return OC_OK;
}

OcError oc_ft_remove_node(OcFtManager *mgr, uint64_t node_id)
{
    if (!mgr) return OC_ERR_INVALID_ARG;
    if (node_id == mgr->self_id) return OC_ERR_INVALID_ARG;

    int32_t idx = ft_find(mgr, node_id);
    if (idx < 0) return OC_ERR_MODEL;

    /* Shift remaining nodes down. */
    for (uint32_t i = (uint32_t)idx; i + 1 < mgr->n_nodes; i++) {
        mgr->nodes[i] = mgr->nodes[i + 1];
    }
    mgr->n_nodes--;
    return OC_OK;
}


OcError oc_ft_heartbeat(OcFtManager *mgr, uint64_t node_id,
                        uint64_t current_ms)
{
    if (!mgr) return OC_ERR_INVALID_ARG;
    int32_t idx = ft_find(mgr, node_id);
    if (idx < 0) return OC_ERR_MODEL;

    OcFtNodeState *n = &mgr->nodes[idx];
    n->last_heartbeat_ms = current_ms;
    n->missed_count      = 0;
    n->status            = OC_FT_ALIVE;
    return OC_OK;
}

OcError oc_ft_tick(OcFtManager *mgr, uint64_t current_ms)
{
    if (!mgr) return OC_ERR_INVALID_ARG;

    for (uint32_t i = 0; i < mgr->n_nodes; i++) {
        OcFtNodeState *n = &mgr->nodes[i];
        if (n->status == OC_FT_DEAD) continue;
        if (n->status == OC_FT_RECOVERING) continue;

        /* Skip self (self never times out via tick). */
        if (n->node_id == mgr->self_id) continue;

        uint64_t elapsed;
        if (current_ms >= n->last_heartbeat_ms) {
            elapsed = current_ms - n->last_heartbeat_ms;
        } else {
            /* Clock skew: treat as no elapsed time. */
            elapsed = 0;
        }

        if (elapsed > mgr->config.heartbeat_timeout_ms) {
            n->missed_count++;
            if (n->missed_count >= mgr->config.max_missed) {
                n->status = OC_FT_DEAD;
            } else {
                n->status = OC_FT_SUSPECT;
            }
        }
    }
    return OC_OK;
}


OcError oc_ft_get_node_state(const OcFtManager *mgr, uint64_t node_id,
                             OcFtNodeState *out_state)
{
    if (!mgr || !out_state) return OC_ERR_INVALID_ARG;
    int32_t idx = ft_find(mgr, node_id);
    if (idx < 0) return OC_ERR_MODEL;
    *out_state = mgr->nodes[idx];
    return OC_OK;
}

uint32_t oc_ft_get_alive_count(const OcFtManager *mgr)
{
    if (!mgr) return 0;
    uint32_t count = 0;
    for (uint32_t i = 0; i < mgr->n_nodes; i++) {
        if (mgr->nodes[i].status == OC_FT_ALIVE) count++;
    }
    return count;
}

OcError oc_ft_get_dead_nodes(const OcFtManager *mgr, uint64_t *out_ids,
                             uint32_t max, uint32_t *out_count)
{
    if (!mgr || !out_count) return OC_ERR_INVALID_ARG;
    uint32_t count = 0;
    for (uint32_t i = 0; i < mgr->n_nodes; i++) {
        if (mgr->nodes[i].status != OC_FT_DEAD) continue;
        if (out_ids && count < max) {
            out_ids[count] = mgr->nodes[i].node_id;
        }
        count++;
    }
    *out_count = count;
    return OC_OK;
}


OcError oc_ft_recover(OcFtManager *mgr, uint64_t node_id)
{
    if (!mgr) return OC_ERR_INVALID_ARG;
    int32_t idx = ft_find(mgr, node_id);
    if (idx < 0) return OC_ERR_MODEL;
    mgr->nodes[idx].status = OC_FT_RECOVERING;
    return OC_OK;
}

const char *oc_ft_status_name(OcFtStatus status)
{
    switch (status) {
    case OC_FT_ALIVE:      return "ALIVE";
    case OC_FT_SUSPECT:    return "SUSPECT";
    case OC_FT_DEAD:       return "DEAD";
    case OC_FT_RECOVERING: return "RECOVERING";
    default:               return "UNKNOWN";
    }
}
