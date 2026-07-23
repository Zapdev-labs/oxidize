/*
 * gpu_dispatch.h — GPU task dispatch for inference workloads.
 *
 * Maintains a simple priority work queue of GPU tasks and a registry of
 * detected GPU devices. Since this is the dependency-free C11 port, the
 * device detector returns 0 devices when CUDA is unavailable — the queue
 * still accepts submissions so callers can test the API surface.
 */
#ifndef OXIDIZE_GPU_DISPATCH_H
#define OXIDIZE_GPU_DISPATCH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Constants ──────────────────────────────────────────────────────── */

#define OC_GPU_MAX_DEVICES   16u
#define OC_GPU_MAX_QUEUE     1024u
#define OC_GPU_NAME_LEN      64u

/* ─── Types ─────────────────────────────────────────────────────────── */

typedef enum {
    OC_GPU_TASK_MATMUL    = 0,
    OC_GPU_TASK_ATTENTION  = 1,
    OC_GPU_TASK_SAMPLING   = 2,
    OC_GPU_TASK_OTHER      = 3,
} OcGpuTaskType;

typedef struct OcGpuDispatchConfig {
    uint32_t n_devices;          /* default 0 = auto-detect                */
    uint32_t max_queue_size;     /* default 256                            */
    uint32_t batch_timeout_ms;   /* default 10                            */
} OcGpuDispatchConfig;

typedef struct OcGpuTask {
    uint64_t       id;
    OcGpuTaskType  type;
    uint32_t       priority;     /* higher = more urgent                   */
    void          *data_ptr;
    uint64_t       data_size;
} OcGpuTask;

typedef struct OcGpuDevice {
    uint32_t device_id;
    char     name[OC_GPU_NAME_LEN];
    uint64_t total_memory;       /* bytes                                 */
    uint64_t free_memory;        /* bytes                                 */
    uint32_t compute_capability; /* e.g. 80 for sm_80                     */
    bool     in_use;
} OcGpuDevice;

typedef struct OcGpuDispatch {
    OcGpuDispatchConfig config;
    OcGpuDevice         devices[OC_GPU_MAX_DEVICES];
    uint32_t            n_devices;
    OcGpuTask          *task_queue;   /* heap-allocated ring buffer        */
    uint32_t            queue_cap;
    uint32_t            n_pending;
} OcGpuDispatch;

/* ─── Config helpers ────────────────────────────────────────────────── */

/* Initialize config with defaults. */
OcError oc_gpu_dispatch_config_init(OcGpuDispatchConfig *cfg);

/* Human-readable task type name (e.g. "MATMUL"). Never NULL. */
const char *oc_gpu_task_type_name(OcGpuTaskType type);

/* ─── Lifecycle ─────────────────────────────────────────────────────── */

/* Allocate a dispatch manager. `config` may be NULL (defaults used).
 * Calls oc_gpu_dispatch_detect internally. Free with oc_gpu_dispatch_free. */
OcError oc_gpu_dispatch_init(const OcGpuDispatchConfig *config,
                             OcGpuDispatch **out);

/* Detect available GPU devices. In the dependency-free C11 port, this
 * returns OC_OK with n_devices=0 when CUDA is not compiled in. Free with
 * oc_gpu_dispatch_free. */
OcError oc_gpu_dispatch_detect(OcGpuDispatch *dispatch);

/* Free all owned storage and reset state. Safe on NULL / already-freed. */
void oc_gpu_dispatch_free(OcGpuDispatch *dispatch);

/* ─── Task queue ────────────────────────────────────────────────────── */

/* Submit a task to the work queue. Tasks are inserted in priority order
 * (higher priority first; stable within the same priority by insertion
 * order). Returns OC_ERR_OOM if the queue is full. */
OcError oc_gpu_dispatch_submit(OcGpuDispatch *dispatch, OcGpuTask task);

/* Number of pending tasks in the queue. */
uint32_t oc_gpu_dispatch_n_pending(const OcGpuDispatch *dispatch);

/* ─── Device queries ────────────────────────────────────────────────── */

/* Copy the device info for `device_id` into `out_device`. Returns
 * OC_ERR_MODEL if the device is not present. */
OcError oc_gpu_dispatch_get_device(const OcGpuDispatch *dispatch,
                                   uint32_t device_id,
                                   OcGpuDevice *out_device);

/* Number of detected devices. */
uint32_t oc_gpu_dispatch_n_devices(const OcGpuDispatch *dispatch);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_GPU_DISPATCH_H */
