/* test_progress.c — Progress tracking tests. */
#include "framework.h"
#include "oxidize/progress.h"
#include <string.h>

Test(prog, init)
{
    OcProgress prog;
    cr_assert_eq(oc_progress_init(&prog), OC_OK);
    cr_assert_eq(prog.n_stages, 0);
    cr_assert(!prog.running);
    oc_progress_free(&prog);
}

Test(prog, init_null)
{
    cr_assert_neq(oc_progress_init(NULL), OC_OK);
}

Test(prog, add_stage)
{
    OcProgress prog;
    oc_progress_init(&prog);
    cr_assert_eq(oc_progress_add_stage(&prog, "loading", 100), OC_OK);
    cr_assert_eq(prog.n_stages, 1);
    cr_assert(prog.running);
    cr_assert_str_eq(prog.stages[0].name, "loading");
    oc_progress_free(&prog);
}

Test(prog, add_stage_null)
{
    cr_assert_neq(oc_progress_add_stage(NULL, "x", 0), OC_OK);
}

Test(prog, update)
{
    OcProgress prog;
    oc_progress_init(&prog);
    oc_progress_add_stage(&prog, "loading", 100);
    cr_assert_eq(oc_progress_update(&prog, 50), OC_OK);
    cr_assert_eq(prog.stages[0].completed, 50);
    oc_progress_free(&prog);
}

Test(prog, update_complete)
{
    OcProgress prog;
    oc_progress_init(&prog);
    oc_progress_add_stage(&prog, "loading", 100);
    oc_progress_update(&prog, 100);
    cr_assert(prog.stages[0].done);
    oc_progress_free(&prog);
}

Test(prog, advance)
{
    OcProgress prog;
    oc_progress_init(&prog);
    oc_progress_add_stage(&prog, "stage1", 10);
    oc_progress_add_stage(&prog, "stage2", 10);
    cr_assert_eq(oc_progress_advance(&prog), OC_OK);
    cr_assert_eq(prog.current_stage, 1);
    cr_assert(prog.stages[0].done);
    oc_progress_free(&prog);
}

Test(prog, advance_last)
{
    OcProgress prog;
    oc_progress_init(&prog);
    oc_progress_add_stage(&prog, "stage1", 10);
    cr_assert_eq(oc_progress_advance(&prog), OC_OK);
    cr_assert(!prog.running);
    cr_assert_eq(prog.current_stage, 1);
    oc_progress_free(&prog);
}

Test(prog, complete)
{
    OcProgress prog;
    oc_progress_init(&prog);
    oc_progress_add_stage(&prog, "s1", 10);
    oc_progress_add_stage(&prog, "s2", 10);
    cr_assert_eq(oc_progress_complete(&prog), OC_OK);
    cr_assert(!prog.running);
    cr_assert(prog.stages[0].done);
    cr_assert(prog.stages[1].done);
    cr_assert(oc_progress_is_done(&prog));
    oc_progress_free(&prog);
}

Test(prog, fail)
{
    OcProgress prog;
    oc_progress_init(&prog);
    oc_progress_add_stage(&prog, "s1", 10);
    cr_assert_eq(oc_progress_fail(&prog), OC_OK);
    cr_assert(!prog.running);
    cr_assert(prog.stages[0].failed);
    oc_progress_free(&prog);
}

Test(prog, cancel)
{
    OcProgress prog;
    oc_progress_init(&prog);
    oc_progress_add_stage(&prog, "s1", 10);
    cr_assert_eq(oc_progress_cancel(&prog), OC_OK);
    cr_assert(prog.cancelled);
    cr_assert(!prog.running);
    oc_progress_free(&prog);
}

Test(prog, get_current)
{
    OcProgress prog;
    oc_progress_init(&prog);
    oc_progress_add_stage(&prog, "s1", 10);
    const OcProgressStage *s;
    cr_assert_eq(oc_progress_get_current(&prog, &s), OC_OK);
    cr_assert_str_eq(s->name, "s1");
    oc_progress_free(&prog);
}

Test(prog, percent)
{
    OcProgress prog;
    oc_progress_init(&prog);
    oc_progress_add_stage(&prog, "s1", 100);
    oc_progress_update(&prog, 50);
    float pct = oc_progress_percent(&prog);
    cr_assert_float_eq(pct, 50.0f, 0.01f);
    oc_progress_free(&prog);
}

Test(prog, percent_multi)
{
    OcProgress prog;
    oc_progress_init(&prog);
    oc_progress_add_stage(&prog, "s1", 100);
    oc_progress_add_stage(&prog, "s2", 100);
    oc_progress_update(&prog, 50);
    float pct = oc_progress_percent(&prog);
    cr_assert_float_eq(pct, 25.0f, 0.01f);
    oc_progress_free(&prog);
}

Test(prog, elapsed_ms)
{
    OcProgress prog;
    oc_progress_init(&prog);
    oc_progress_add_stage(&prog, "s1", 10);
    uint64_t elapsed = oc_progress_elapsed_ms(&prog);
    cr_assert(elapsed >= 0);
    oc_progress_free(&prog);
}

Test(prog, is_done)
{
    OcProgress prog;
    oc_progress_init(&prog);
    cr_assert(!oc_progress_is_done(&prog));
    oc_progress_add_stage(&prog, "s1", 10);
    cr_assert(!oc_progress_is_done(&prog));
    oc_progress_complete(&prog);
    cr_assert(oc_progress_is_done(&prog));
    oc_progress_free(&prog);
}

Test(prog, is_running)
{
    OcProgress prog;
    oc_progress_init(&prog);
    cr_assert(!oc_progress_is_running(&prog));
    oc_progress_add_stage(&prog, "s1", 10);
    cr_assert(oc_progress_is_running(&prog));
    oc_progress_free(&prog);
}

Test(prog, stage_name)
{
    OcProgress prog;
    oc_progress_init(&prog);
    oc_progress_add_stage(&prog, "loading", 10);
    cr_assert_str_eq(oc_progress_stage_name(&prog), "loading");
    oc_progress_free(&prog);
}

Test(prog, free_null)
{
    oc_progress_free(NULL);
}
