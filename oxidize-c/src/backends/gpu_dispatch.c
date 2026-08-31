/* gpu_dispatch.c — GPU task dispatch for inference workloads. */
#include "oxidize/gpu_dispatch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


OcError oc_gpu_dispatch_config_init(OcGpuDispatchConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    cfg->n_devices        = 0;
    cfg->max_queue_size   = 256;
    cfg->batch_timeout_ms = 10;
    return OC_OK;
}

const char *oc_gpu_task_type_name(OcGpuTaskType type)
{
    switch (type) {
    case OC_GPU_TASK_MATMUL:   return "MATMUL";
    case OC_GPU_TASK_ATTENTION: return "ATTENTION";
    case OC_GPU_TASK_SAMPLING: return "SAMPLING";
    case OC_GPU_TASK_OTHER:    return "OTHER";
    default:                   return "UNKNOWN";
    }
}


OcError oc_gpu_dispatch_init(const OcGpuDispatchConfig *config,
                             OcGpuDispatch **out)
{
    if (!out) return OC_ERR_INVALID_ARG;
    *out = NULL;

    OcGpuDispatchConfig cfg;
    if (config) {
        cfg = *config;
    } else {
        oc_gpu_dispatch_config_init(&cfg);
    }
    if (cfg.max_queue_size == 0) cfg.max_queue_size = 256;

    OcGpuDispatch *d = malloc(sizeof(*d));
    if (!d) return OC_ERR_OOM;
    memset(d, 0, sizeof(*d));
    d->config    = cfg;
    d->n_devices = 0;
    d->queue_cap = cfg.max_queue_size;
    if (d->queue_cap > OC_GPU_MAX_QUEUE) d->queue_cap = OC_GPU_MAX_QUEUE;
    d->n_pending = 0;

    d->task_queue = malloc((size_t)d->queue_cap * sizeof(OcGpuTask));
    if (!d->task_queue) {
        free(d);
        return OC_ERR_OOM;
    }

    OcError e = oc_gpu_dispatch_detect(d);
    if (e != OC_OK) {
        oc_gpu_dispatch_free(d);
        return e;
    }

    *out = d;
    return OC_OK;
}

OcError oc_gpu_dispatch_detect(OcGpuDispatch *dispatch)
{
    if (!dispatch) return OC_ERR_INVALID_ARG;
    /* In the dependency-free C11 port there is no CUDA runtime to query.
     * Gracefully report zero devices; callers may still submit tasks to
     * the queue. The CUDA build (OC_CUDA) would populate devices here. */
    dispatch->n_devices = 0;
    return OC_OK;
}

void oc_gpu_dispatch_free(OcGpuDispatch *dispatch)
{
    if (!dispatch) return;
    if (dispatch->task_queue) {
        free(dispatch->task_queue);
        dispatch->task_queue = NULL;
    }
    memset(dispatch, 0, sizeof(*dispatch));
    free(dispatch);
}


OcError oc_gpu_dispatch_submit(OcGpuDispatch *dispatch, OcGpuTask task)
{
    if (!dispatch) return OC_ERR_INVALID_ARG;
    if (!dispatch->task_queue) return OC_ERR_INTERNAL;
    if (dispatch->n_pending >= dispatch->queue_cap) return OC_ERR_OOM;

    /* Insert in priority order: find first position with strictly lower
     * priority, then shift the rest right (stable within same priority). */
    uint32_t insert_idx = dispatch->n_pending;
    for (uint32_t i = 0; i < dispatch->n_pending; i++) {
        if (dispatch->task_queue[i].priority < task.priority) {
            insert_idx = i;
            break;
        }
    }

    /* Shift [insert_idx, n_pending) right by one. */
    for (uint32_t i = dispatch->n_pending; i > insert_idx; i--) {
        dispatch->task_queue[i] = dispatch->task_queue[i - 1];
    }

    dispatch->task_queue[insert_idx] = task;
    dispatch->n_pending++;
    return OC_OK;
}

uint32_t oc_gpu_dispatch_n_pending(const OcGpuDispatch *dispatch)
{
    return dispatch ? dispatch->n_pending : 0;
}


OcError oc_gpu_dispatch_get_device(const OcGpuDispatch *dispatch,
                                   uint32_t device_id,
                                   OcGpuDevice *out_device)
{
    if (!dispatch || !out_device) return OC_ERR_INVALID_ARG;
    for (uint32_t i = 0; i < dispatch->n_devices; i++) {
        if (dispatch->devices[i].device_id == device_id) {
            *out_device = dispatch->devices[i];
            return OC_OK;
        }
    }
    return OC_ERR_MODEL;
}

uint32_t oc_gpu_dispatch_n_devices(const OcGpuDispatch *dispatch)
{
    return dispatch ? dispatch->n_devices : 0;
}
