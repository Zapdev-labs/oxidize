/* test_profiler.c — profiler tests. */
#include <criterion/criterion.h>
#include "oxidize/profiler.h"
#include <string.h>

Test(prof, init)
{
    OcProfiler p;
    oc_profiler_init(&p);
    cr_assert(p.enabled);
    cr_assert_eq(p.tokens_profiled, 0);
}

Test(prof, record)
{
    OcProfiler p;
    oc_profiler_init(&p);
    oc_profiler_record(&p, OC_PROF_ATTENTION, 1000);
    oc_profiler_record(&p, OC_PROF_ATTENTION, 2000);
    cr_assert_eq(p.entries[OC_PROF_ATTENTION].total_ns, 3000);
    cr_assert_eq(p.entries[OC_PROF_ATTENTION].count, 2);
    cr_assert_eq(p.entries[OC_PROF_ATTENTION].min_ns, 1000);
    cr_assert_eq(p.entries[OC_PROF_ATTENTION].max_ns, 2000);
}

Test(prof, total)
{
    OcProfiler p;
    oc_profiler_init(&p);
    oc_profiler_record(&p, OC_PROF_EMBEDDING, 500);
    oc_profiler_record(&p, OC_PROF_ATTENTION, 1500);
    cr_assert_eq(p.entries[OC_PROF_TOTAL].total_ns, 2000);
    cr_assert_eq(p.entries[OC_PROF_TOTAL].count, 2);
}

Test(prof, avg)
{
    OcProfiler p;
    oc_profiler_init(&p);
    oc_profiler_record(&p, OC_PROF_MATVEC, 100);
    oc_profiler_record(&p, OC_PROF_MATVEC, 300);
    cr_assert_float_eq(oc_profiler_avg_ns(&p, OC_PROF_MATVEC), 200.0, 1e-6);
}

Test(prof, pct)
{
    OcProfiler p;
    oc_profiler_init(&p);
    oc_profiler_record(&p, OC_PROF_ATTENTION, 800);
    oc_profiler_record(&p, OC_PROF_MLP_GATE, 200);
    cr_assert_float_eq(oc_profiler_pct(&p, OC_PROF_ATTENTION), 80.0, 1e-3);
    cr_assert_float_eq(oc_profiler_pct(&p, OC_PROF_MLP_GATE), 20.0, 1e-3);
}

Test(prof, disable)
{
    OcProfiler p;
    oc_profiler_init(&p);
    oc_profiler_enable(&p, false);
    oc_profiler_record(&p, OC_PROF_ATTENTION, 1000);
    cr_assert_eq(p.entries[OC_PROF_ATTENTION].count, 0);
}

Test(prof, reset)
{
    OcProfiler p;
    oc_profiler_init(&p);
    oc_profiler_record(&p, OC_PROF_ATTENTION, 1000);
    oc_profiler_reset(&p);
    cr_assert_eq(p.entries[OC_PROF_ATTENTION].count, 0);
}

Test(prof, event_name)
{
    cr_assert_str_eq(oc_prof_event_name(OC_PROF_ATTENTION), "attention");
    cr_assert_str_eq(oc_prof_event_name(OC_PROF_RMSNORM), "rmsnorm");
    cr_assert_str_eq(oc_prof_event_name(OC_PROF_TOTAL), "total");
}

Test(prof, format)
{
    OcProfiler p;
    oc_profiler_init(&p);
    oc_profiler_record(&p, OC_PROF_ATTENTION, 1000000);
    char buf[4096];
    size_t n = oc_profiler_format(&p, buf, sizeof(buf));
    cr_assert(n > 0);
    cr_assert(strstr(buf, "Inference") != NULL);
    cr_assert(strstr(buf, "attention") != NULL);
}

Test(prof, format_json)
{
    OcProfiler p;
    oc_profiler_init(&p);
    oc_profiler_record(&p, OC_PROF_MATVEC, 5000);
    char buf[4096];
    size_t n = oc_profiler_format_json(&p, buf, sizeof(buf));
    cr_assert(n > 0);
    cr_assert(strstr(buf, "events") != NULL);
    cr_assert(strstr(buf, "matvec") != NULL);
}

Test(prof, scope)
{
    OcProfiler p;
    oc_profiler_init(&p);
    OcProfileScope s;
    oc_prof_scope_begin(&s, &p, OC_PROF_SAMPLING);
    /* Simulate some work. */
    volatile int x = 0;
    for (int i = 0; i < 1000; i++) x += i;
    (void)x;
    oc_prof_scope_end(&s);
    cr_assert(p.entries[OC_PROF_SAMPLING].count == 1);
    cr_assert(p.entries[OC_PROF_SAMPLING].total_ns > 0);
}

Test(prof, now_ns)
{
    uint64_t t1 = oc_prof_now_ns();
    uint64_t t2 = oc_prof_now_ns();
    cr_assert(t2 >= t1);
}

Test(prof, null_handling)
{
    oc_profiler_init(NULL);
    oc_profiler_enable(NULL, true);
    oc_profiler_record(NULL, OC_PROF_ATTENTION, 1000);
    cr_assert_eq(oc_profiler_total_ns(NULL, OC_PROF_ATTENTION), 0);
    cr_assert_float_eq(oc_profiler_avg_ns(NULL, OC_PROF_ATTENTION), 0.0, 1e-6);
}
