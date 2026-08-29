#ifndef OXIDIZE_MESH_PROGRESS_H
#define OXIDIZE_MESH_PROGRESS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_MESH_PROGRESS_MAX_ENTRIES 128
#define OC_MESH_PROGRESS_NODE_LEN    64
#define OC_MESH_PROGRESS_TASK_LEN    128
#define OC_MESH_PROGRESS_MSG_LEN     256

/* Lifecycle state of a tracked task. */
typedef enum {
    OC_PROGRESS_PENDING = 0,
    OC_PROGRESS_RUNNING = 1,
    OC_PROGRESS_DONE = 2,
    OC_PROGRESS_FAILED = 3,
    OC_PROGRESS_CANCELLED = 4,
    OC_PROGRESS__COUNT, /* sentinel; not a valid state */
} OcProgressState;

/* Progress entry for a single (node, task) pair. */
typedef struct {
    char node_id[OC_MESH_PROGRESS_NODE_LEN];
    char task_id[OC_MESH_PROGRESS_TASK_LEN];
    OcProgressState state;
    float progress;          /* clamped to [0.0, 1.0] */
    char message[OC_MESH_PROGRESS_MSG_LEN];
    uint64_t started_at;     /* set on add() */
    uint64_t updated_at;     /* set on add() and update() */
} OcMeshProgress;

/* A tracker holds up to OC_MESH_PROGRESS_MAX_ENTRIES entries. */
typedef struct {
    OcMeshProgress entries[OC_MESH_PROGRESS_MAX_ENTRIES];
    size_t count;
    uint64_t clock; /* internal monotonic clock for started_at/updated_at */
} OcMeshProgressTracker;

/* Initialize a tracker. Clears all entries and resets the clock. */
OcError oc_mesh_progress_init(OcMeshProgressTracker *tracker);

/* Add a new task entry in OC_PROGRESS_PENDING state with progress 0.0.
 * Returns OC_ERR_INVALID_ARG if a task with the same task_id already
 * exists. Returns OC_ERR_OOM when the tracker is full. */
OcError oc_mesh_progress_add(OcMeshProgressTracker *tracker,
                             const char *node_id, const char *task_id);

/* Update the state/progress/message of an existing task. `message` may be
 * NULL (preserves the previous message). `progress` is clamped to [0, 1].
 * Returns OC_ERR_MODEL if the task_id is not found. */
OcError oc_mesh_progress_update(OcMeshProgressTracker *tracker,
                                const char *task_id, OcProgressState state,
                                float progress, const char *message);

/* Look up an entry by task_id. Returns NULL if not found. The pointer is
 * valid as long as the tracker is not mutated. */
const OcMeshProgress *oc_mesh_progress_get(const OcMeshProgressTracker *tracker,
                                          const char *task_id);

/* Overall progress across all entries: mean of the per-entry progress
 * values. Returns 0.0 for an empty tracker. Tasks in OC_PROGRESS_FAILED or
 * OC_PROGRESS_CANCELLED states are excluded from the mean. */
float oc_mesh_progress_overall(const OcMeshProgressTracker *tracker);

/* Human-readable state name ("pending", "running", etc.). Returns
 * "unknown" for invalid states. Never returns NULL. */
const char *oc_mesh_progress_state_name(OcProgressState state);

/* Count entries currently in `state`. */
size_t oc_mesh_progress_count_by_state(const OcMeshProgressTracker *tracker,
                                       OcProgressState state);

/* Move all RUNNING/PENDING tasks to OC_PROGRESS_CANCELLED. Does not affect
 * DONE/FAILED/CANCELLED tasks. */
OcError oc_mesh_progress_cancel_all(OcMeshProgressTracker *tracker);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_MESH_PROGRESS_H */
