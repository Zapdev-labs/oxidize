/* mesh_progress.c — Distributed inference task progress tracking. */
#define _POSIX_C_SOURCE 200809L

#include "oxidize/mesh_progress.h"

#include <math.h>
#include <string.h>

/* Helpers.                                                            */

static void copy_str(char *dst, size_t cap, const char *src)
{
    if (!dst || cap == 0) return;
    if (!src) {
        dst[0] = '\0';
        return;
    }
    size_t n = strlen(src);
    if (n >= cap) n = cap - 1;
    memcpy(dst, src, n);
    dst[n] = '\0';
}

static bool state_valid(OcProgressState s)
{
    return (size_t)s < OC_PROGRESS__COUNT;
}

static float clamp01(float v)
{
    if (isnan(v)) return 0.0f;
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

static OcMeshProgress *find_entry(OcMeshProgressTracker *tracker, const char *task_id)
{
    if (!tracker || !task_id) return NULL;
    for (size_t i = 0; i < tracker->count; i++) {
        if (strcmp(tracker->entries[i].task_id, task_id) == 0)
            return &tracker->entries[i];
    }
    return NULL;
}

/* Public API.                                                         */

OcError oc_mesh_progress_init(OcMeshProgressTracker *tracker)
{
    if (!tracker) return OC_ERR_INVALID_ARG;
    memset(tracker, 0, sizeof(*tracker));
    return OC_OK;
}

OcError oc_mesh_progress_add(OcMeshProgressTracker *tracker,
                             const char *node_id, const char *task_id)
{
    if (!tracker || !node_id || !task_id) return OC_ERR_INVALID_ARG;
    if (tracker->count >= OC_MESH_PROGRESS_MAX_ENTRIES) return OC_ERR_OOM;
    if (find_entry(tracker, task_id) != NULL) return OC_ERR_INVALID_ARG;

    OcMeshProgress *e = &tracker->entries[tracker->count];
    memset(e, 0, sizeof(*e));
    copy_str(e->node_id, sizeof(e->node_id), node_id);
    copy_str(e->task_id, sizeof(e->task_id), task_id);
    e->state = OC_PROGRESS_PENDING;
    e->progress = 0.0f;
    e->message[0] = '\0';
    e->started_at = tracker->clock;
    e->updated_at = tracker->clock;
    tracker->count++;
    tracker->clock++;
    return OC_OK;
}

OcError oc_mesh_progress_update(OcMeshProgressTracker *tracker,
                                const char *task_id, OcProgressState state,
                                float progress, const char *message)
{
    if (!tracker || !task_id) return OC_ERR_INVALID_ARG;
    if (!state_valid(state)) return OC_ERR_INVALID_ARG;
    OcMeshProgress *e = find_entry(tracker, task_id);
    if (!e) return OC_ERR_MODEL;

    e->state = state;
    e->progress = clamp01(progress);
    if (message) copy_str(e->message, sizeof(e->message), message);
    e->updated_at = tracker->clock;
    tracker->clock++;
    return OC_OK;
}

const OcMeshProgress *oc_mesh_progress_get(const OcMeshProgressTracker *tracker,
                                          const char *task_id)
{
    if (!tracker || !task_id) return NULL;
    for (size_t i = 0; i < tracker->count; i++) {
        if (strcmp(tracker->entries[i].task_id, task_id) == 0)
            return &tracker->entries[i];
    }
    return NULL;
}

float oc_mesh_progress_overall(const OcMeshProgressTracker *tracker)
{
    if (!tracker || tracker->count == 0) return 0.0f;
    float sum = 0.0f;
    size_t n = 0;
    for (size_t i = 0; i < tracker->count; i++) {
        const OcMeshProgress *e = &tracker->entries[i];
        if (e->state == OC_PROGRESS_FAILED || e->state == OC_PROGRESS_CANCELLED)
            continue;
        sum += e->progress;
        n++;
    }
    if (n == 0) return 0.0f;
    return sum / (float)n;
}

const char *oc_mesh_progress_state_name(OcProgressState state)
{
    switch (state) {
    case OC_PROGRESS_PENDING:   return "pending";
    case OC_PROGRESS_RUNNING:    return "running";
    case OC_PROGRESS_DONE:      return "done";
    case OC_PROGRESS_FAILED:    return "failed";
    case OC_PROGRESS_CANCELLED: return "cancelled";
    default:                     return "unknown";
    }
}

size_t oc_mesh_progress_count_by_state(const OcMeshProgressTracker *tracker,
                                       OcProgressState state)
{
    if (!tracker) return 0;
    size_t n = 0;
    for (size_t i = 0; i < tracker->count; i++) {
        if (tracker->entries[i].state == state) n++;
    }
    return n;
}

OcError oc_mesh_progress_cancel_all(OcMeshProgressTracker *tracker)
{
    if (!tracker) return OC_ERR_INVALID_ARG;
    for (size_t i = 0; i < tracker->count; i++) {
        OcMeshProgress *e = &tracker->entries[i];
        if (e->state == OC_PROGRESS_PENDING || e->state == OC_PROGRESS_RUNNING) {
            e->state = OC_PROGRESS_CANCELLED;
            e->updated_at = tracker->clock;
        }
    }
    tracker->clock++;
    return OC_OK;
}
