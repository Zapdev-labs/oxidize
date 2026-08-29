/* test_attn_dump.c — attention dumper tests. */
#include "framework.h"
#include "oxidize/attn_dump.h"
#include <stdlib.h>

Test(attn_dump, init_disabled_by_default)
{
    OcAttnDumper d;
    oc_attn_dump_init(&d, NULL);
    /* Should be disabled unless env vars are set. */
    cr_assert(!d.enabled || getenv("OXIDIZE_TRACE_VALS") || getenv("OXIDIZE_TRACE_FWD"));
}

Test(attn_dump, init_with_dir)
{
    OcAttnDumper d;
    oc_attn_dump_init(&d, "/tmp/test_trace");
    if (d.enabled) {
        cr_assert_str_eq(d.output_dir, "/tmp/test_trace");
    }
}

Test(attn_dump, set_context)
{
    OcAttnDumper d;
    oc_attn_dump_init(&d, NULL);
    oc_attn_dump_set_context(&d, 42, 7);
    /* If disabled, step/layer remain 0. */
    if (d.enabled) {
        cr_assert_eq(d.step, 42);
        cr_assert_eq(d.layer, 7);
    }
}
