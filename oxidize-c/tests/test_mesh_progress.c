/* test_mesh_progress.c — Distributed task progress tracking tests. */
#include "framework.h"
#include "oxidize/mesh_progress.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

/* ----------------------------------------------------------------- */
/* init.                                                              */
/* ----------------------------------------------------------------- */

Test(mesh_progress, init)
{
    OcMeshProgressTracker t;
    cr_assert_eq(oc_mesh_progress_init(&t), OC_OK);
    cr_assert_eq(t.count, 0u);
    cr_assert_eq(t.clock, 0u);
}

OC_TEST_NULL_SAFE(mesh_progress, init_null,
        cr_assert_neq(oc_mesh_progress_init(NULL), OC_OK);)

/* ----------------------------------------------------------------- */
/* add.                                                               */
/* ----------------------------------------------------------------- */

Test(mesh_progress, add_basic)
{
    OcMeshProgressTracker t;
    oc_mesh_progress_init(&t);
    cr_assert_eq(oc_mesh_progress_add(&t, "node-1", "task-1"), OC_OK);
    cr_assert_eq(t.count, 1u);
    cr_assert_str_eq(t.entries[0].node_id, "node-1");
    cr_assert_str_eq(t.entries[0].task_id, "task-1");
    cr_assert_eq(t.entries[0].state, OC_PROGRESS_PENDING);
    cr_assert_float_eq(t.entries[0].progress, 0.0f, 1e-6f);
    cr_assert_eq(t.entries[0].started_at, 0u);
    cr_assert_eq(t.entries[0].updated_at, 0u);
}

Test(mesh_progress, add_null_args)
{
    OcMeshProgressTracker t;
    oc_mesh_progress_init(&t);
    cr_assert_neq(oc_mesh_progress_add(NULL, "n", "t"), OC_OK);
    cr_assert_neq(oc_mesh_progress_add(&t, NULL, "t"), OC_OK);
    cr_assert_neq(oc_mesh_progress_add(&t, "n", NULL), OC_OK);
}

Test(mesh_progress, add_duplicate_task)
{
    OcMeshProgressTracker t;
    oc_mesh_progress_init(&t);
    cr_assert_eq(oc_mesh_progress_add(&t, "n1", "task-1"), OC_OK);
    cr_assert_neq(oc_mesh_progress_add(&t, "n2", "task-1"), OC_OK);
    cr_assert_eq(t.count, 1u);
}

/* ----------------------------------------------------------------- */
/* update.                                                            */
/* ----------------------------------------------------------------- */

Test(mesh_progress, update_progress)
{
    OcMeshProgressTracker t;
    oc_mesh_progress_init(&t);
    oc_mesh_progress_add(&t, "n1", "task-1");
    cr_assert_eq(oc_mesh_progress_update(&t, "task-1", OC_PROGRESS_RUNNING,
                                         0.5f, "halfway"), OC_OK);
    const OcMeshProgress *e = oc_mesh_progress_get(&t, "task-1");
    cr_assert_not_null(e);
    cr_assert_eq(e->state, OC_PROGRESS_RUNNING);
    cr_assert_float_eq(e->progress, 0.5f, 1e-6f);
    cr_assert_str_eq(e->message, "halfway");
    cr_assert_gt(e->updated_at, e->started_at);
}

Test(mesh_progress, update_null_message_keeps)
{
    OcMeshProgressTracker t;
    oc_mesh_progress_init(&t);
    oc_mesh_progress_add(&t, "n1", "task-1");
    oc_mesh_progress_update(&t, "task-1", OC_PROGRESS_RUNNING, 0.1f, "first");
    oc_mesh_progress_update(&t, "task-1", OC_PROGRESS_RUNNING, 0.2f, NULL);
    const OcMeshProgress *e = oc_mesh_progress_get(&t, "task-1");
    cr_assert_str_eq(e->message, "first");
}

Test(mesh_progress, update_not_found)
{
    OcMeshProgressTracker t;
    oc_mesh_progress_init(&t);
    cr_assert_eq(oc_mesh_progress_update(&t, "nope", OC_PROGRESS_RUNNING,
                                         0.5f, "msg"), OC_ERR_MODEL);
}

Test(mesh_progress, update_null_args)
{
    OcMeshProgressTracker t;
    oc_mesh_progress_init(&t);
    cr_assert_neq(oc_mesh_progress_update(NULL, "t", OC_PROGRESS_RUNNING, 0.5f, "m"), OC_OK);
    cr_assert_neq(oc_mesh_progress_update(&t, NULL, OC_PROGRESS_RUNNING, 0.5f, "m"), OC_OK);
    cr_assert_neq(oc_mesh_progress_update(&t, "t", OC_PROGRESS__COUNT, 0.5f, "m"), OC_OK);
}

/* ----------------------------------------------------------------- */
/* progress clamping.                                                 */
/* ----------------------------------------------------------------- */

Test(mesh_progress, progress_clamp_high)
{
    OcMeshProgressTracker t;
    oc_mesh_progress_init(&t);
    oc_mesh_progress_add(&t, "n1", "task-1");
    oc_mesh_progress_update(&t, "task-1", OC_PROGRESS_RUNNING, 5.0f, "over");
    const OcMeshProgress *e = oc_mesh_progress_get(&t, "task-1");
    cr_assert_float_eq(e->progress, 1.0f, 1e-6f);
}

Test(mesh_progress, progress_clamp_low)
{
    OcMeshProgressTracker t;
    oc_mesh_progress_init(&t);
    oc_mesh_progress_add(&t, "n1", "task-1");
    oc_mesh_progress_update(&t, "task-1", OC_PROGRESS_RUNNING, -1.0f, "under");
    const OcMeshProgress *e = oc_mesh_progress_get(&t, "task-1");
    cr_assert_float_eq(e->progress, 0.0f, 1e-6f);
}

Test(mesh_progress, progress_clamp_nan)
{
    OcMeshProgressTracker t;
    oc_mesh_progress_init(&t);
    oc_mesh_progress_add(&t, "n1", "task-1");
    oc_mesh_progress_update(&t, "task-1", OC_PROGRESS_RUNNING, NAN, "nan");
    const OcMeshProgress *e = oc_mesh_progress_get(&t, "task-1");
    cr_assert_float_eq(e->progress, 0.0f, 1e-6f);
}

/* ----------------------------------------------------------------- */
/* get.                                                               */
/* ----------------------------------------------------------------- */

Test(mesh_progress, get_not_found)
{
    OcMeshProgressTracker t;
    oc_mesh_progress_init(&t);
    oc_mesh_progress_add(&t, "n1", "task-1");
    cr_assert_null(oc_mesh_progress_get(&t, "missing"));
}

Test(mesh_progress, get_null)
{
    OcMeshProgressTracker t;
    oc_mesh_progress_init(&t);
    cr_assert_null(oc_mesh_progress_get(NULL, "x"));
    cr_assert_null(oc_mesh_progress_get(&t, NULL));
}

/* ----------------------------------------------------------------- */
/* overall progress.                                                  */
/* ----------------------------------------------------------------- */

Test(mesh_progress, overall_empty)
{
    OcMeshProgressTracker t;
    oc_mesh_progress_init(&t);
    cr_assert_float_eq(oc_mesh_progress_overall(&t), 0.0f, 1e-6f);
}

Test(mesh_progress, overall_calculation)
{
    OcMeshProgressTracker t;
    oc_mesh_progress_init(&t);
    oc_mesh_progress_add(&t, "n1", "t1");
    oc_mesh_progress_add(&t, "n2", "t2");
    oc_mesh_progress_add(&t, "n3", "t3");
    oc_mesh_progress_update(&t, "t1", OC_PROGRESS_RUNNING, 0.5f, NULL);
    oc_mesh_progress_update(&t, "t2", OC_PROGRESS_DONE, 1.0f, NULL);
    oc_mesh_progress_update(&t, "t3", OC_PROGRESS_RUNNING, 0.0f, NULL);
    /* mean of 0.5, 1.0, 0.0 = 0.5 */
    cr_assert_float_eq(oc_mesh_progress_overall(&t), 0.5f, 1e-6f);
}

Test(mesh_progress, overall_excludes_failed_cancelled)
{
    OcMeshProgressTracker t;
    oc_mesh_progress_init(&t);
    oc_mesh_progress_add(&t, "n1", "t1");
    oc_mesh_progress_add(&t, "n2", "t2");
    oc_mesh_progress_update(&t, "t1", OC_PROGRESS_RUNNING, 0.5f, NULL);
    oc_mesh_progress_update(&t, "t2", OC_PROGRESS_FAILED, 1.0f, NULL);
    /* only t1 counts: 0.5 */
    cr_assert_float_eq(oc_mesh_progress_overall(&t), 0.5f, 1e-6f);
}

OC_TEST_NULL_SAFE(mesh_progress, overall_null,
        cr_assert_float_eq(oc_mesh_progress_overall(NULL), 0.0f, 1e-6f);)

/* ----------------------------------------------------------------- */
/* state_name + count_by_state.                                       */
/* ----------------------------------------------------------------- */

Test(mesh_progress, state_name)
{
    cr_assert_str_eq(oc_mesh_progress_state_name(OC_PROGRESS_PENDING), "pending");
    cr_assert_str_eq(oc_mesh_progress_state_name(OC_PROGRESS_RUNNING), "running");
    cr_assert_str_eq(oc_mesh_progress_state_name(OC_PROGRESS_DONE), "done");
    cr_assert_str_eq(oc_mesh_progress_state_name(OC_PROGRESS_FAILED), "failed");
    cr_assert_str_eq(oc_mesh_progress_state_name(OC_PROGRESS_CANCELLED), "cancelled");
    cr_assert_str_eq(oc_mesh_progress_state_name(OC_PROGRESS__COUNT), "unknown");
}

Test(mesh_progress, count_by_state)
{
    OcMeshProgressTracker t;
    oc_mesh_progress_init(&t);
    oc_mesh_progress_add(&t, "n1", "t1");
    oc_mesh_progress_add(&t, "n2", "t2");
    oc_mesh_progress_add(&t, "n3", "t3");
    cr_assert_eq(oc_mesh_progress_count_by_state(&t, OC_PROGRESS_PENDING), 3u);
    oc_mesh_progress_update(&t, "t1", OC_PROGRESS_RUNNING, 0.5f, NULL);
    oc_mesh_progress_update(&t, "t2", OC_PROGRESS_DONE, 1.0f, NULL);
    cr_assert_eq(oc_mesh_progress_count_by_state(&t, OC_PROGRESS_PENDING), 1u);
    cr_assert_eq(oc_mesh_progress_count_by_state(&t, OC_PROGRESS_RUNNING), 1u);
    cr_assert_eq(oc_mesh_progress_count_by_state(&t, OC_PROGRESS_DONE), 1u);
}

Test(mesh_progress, count_by_state_null)
{
    OcMeshProgressTracker t;
    oc_mesh_progress_init(&t);
    cr_assert_eq(oc_mesh_progress_count_by_state(NULL, OC_PROGRESS_PENDING), 0u);
}

/* ----------------------------------------------------------------- */
/* cancel_all.                                                        */
/* ----------------------------------------------------------------- */

Test(mesh_progress, cancel_all)
{
    OcMeshProgressTracker t;
    oc_mesh_progress_init(&t);
    oc_mesh_progress_add(&t, "n1", "t1");
    oc_mesh_progress_add(&t, "n2", "t2");
    oc_mesh_progress_add(&t, "n3", "t3");
    oc_mesh_progress_update(&t, "t1", OC_PROGRESS_RUNNING, 0.5f, NULL);
    oc_mesh_progress_update(&t, "t2", OC_PROGRESS_DONE, 1.0f, NULL);
    oc_mesh_progress_update(&t, "t3", OC_PROGRESS_PENDING, 0.0f, NULL);

    cr_assert_eq(oc_mesh_progress_cancel_all(&t), OC_OK);
    /* t1 and t3 should be cancelled; t2 stays done. */
    cr_assert_eq(oc_mesh_progress_get(&t, "t1")->state, OC_PROGRESS_CANCELLED);
    cr_assert_eq(oc_mesh_progress_get(&t, "t2")->state, OC_PROGRESS_DONE);
    cr_assert_eq(oc_mesh_progress_get(&t, "t3")->state, OC_PROGRESS_CANCELLED);
    cr_assert_eq(oc_mesh_progress_count_by_state(&t, OC_PROGRESS_CANCELLED), 2u);
}

OC_TEST_NULL_SAFE(mesh_progress, cancel_all_null,
        cr_assert_neq(oc_mesh_progress_cancel_all(NULL), OC_OK);)

/* ----------------------------------------------------------------- */
/* Overflow: max 128 entries.                                         */
/* ----------------------------------------------------------------- */

Test(mesh_progress, overflow_max_entries)
{
    OcMeshProgressTracker t;
    oc_mesh_progress_init(&t);
    char task[32];
    for (size_t i = 0; i < OC_MESH_PROGRESS_MAX_ENTRIES; i++) {
        snprintf(task, sizeof(task), "task-%zu", i);
        cr_assert_eq(oc_mesh_progress_add(&t, "n", task), OC_OK);
    }
    cr_assert_eq(t.count, OC_MESH_PROGRESS_MAX_ENTRIES);
    cr_assert_eq(oc_mesh_progress_add(&t, "n", "extra"), OC_ERR_OOM);
    cr_assert_eq(t.count, OC_MESH_PROGRESS_MAX_ENTRIES);
}
