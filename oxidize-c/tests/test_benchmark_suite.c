/* test_benchmark_suite.c — Benchmark suite tests. */
#include "framework.h"
#include "oxidize/benchmark_suite.h"
#include <string.h>

Test(bsuite, init)
{
    OcBenchSuite suite;
    cr_assert_eq(oc_bench_suite_init(&suite), OC_OK);
    cr_assert_eq(suite.n_results, 0);
    cr_assert(!suite.verbose);
    oc_bench_suite_free(&suite);
}

OC_TEST_NULL_SAFE(bsuite, init_null,
        cr_assert_neq(oc_bench_suite_init(NULL), OC_OK);)

Test(bsuite, run_single)
{
    OcBenchSuite suite;
    oc_bench_suite_init(&suite);
    cr_assert_eq(oc_bench_suite_run(&suite, OC_BENCH_MATVEC_F32, 10), OC_OK);
    cr_assert_eq(suite.n_results, 1);
    cr_assert_gt(suite.results[0].elapsed_sec, 0.0);
    cr_assert_eq(suite.results[0].n_items, 10);
    oc_bench_suite_free(&suite);
}

Test(bsuite, run_default_iterations)
{
    OcBenchSuite suite;
    oc_bench_suite_init(&suite);
    cr_assert_eq(oc_bench_suite_run(&suite, OC_BENCH_QUANTIZE, 0), OC_OK);
    cr_assert_eq(suite.results[0].n_iterations, 100);
    oc_bench_suite_free(&suite);
}

Test(bsuite, run_all)
{
    OcBenchSuite suite;
    oc_bench_suite_init(&suite);
    cr_assert_eq(oc_bench_suite_run_all(&suite), OC_OK);
    cr_assert_eq(suite.n_results, 9);
    oc_bench_suite_free(&suite);
}

Test(bsuite, add_result)
{
    OcBenchSuite suite;
    oc_bench_suite_init(&suite);
    OcBenchResult r = {0};
    strcpy(r.name, "test");
    r.elapsed_sec = 0.5;
    r.tokens_per_sec = 100.0;
    cr_assert_eq(oc_bench_suite_add_result(&suite, &r), OC_OK);
    cr_assert_eq(suite.n_results, 1);
    cr_assert_str_eq(suite.results[0].name, "test");
    oc_bench_suite_free(&suite);
}

OC_TEST_NULL_SAFE(bsuite, add_result_null,
        cr_assert_neq(oc_bench_suite_add_result(NULL, NULL), OC_OK);)

Test(bsuite, get_result)
{
    OcBenchSuite suite;
    oc_bench_suite_init(&suite);
    oc_bench_suite_run(&suite, OC_BENCH_MATVEC_F32, 5);
    const OcBenchResult *r = oc_bench_suite_get_result(&suite, 0);
    cr_assert_not_null(r);
    cr_assert_str_eq(r->name, "matvec_f32");
    oc_bench_suite_free(&suite);
}

Test(bsuite, get_result_oob)
{
    OcBenchSuite suite;
    oc_bench_suite_init(&suite);
    cr_assert_null(oc_bench_suite_get_result(&suite, 0));
    oc_bench_suite_free(&suite);
}

Test(bsuite, n_results)
{
    OcBenchSuite suite;
    oc_bench_suite_init(&suite);
    cr_assert_eq(oc_bench_suite_n_results(&suite), 0);
    oc_bench_suite_run(&suite, OC_BENCH_GENERATE, 5);
    cr_assert_eq(oc_bench_suite_n_results(&suite), 1);
    oc_bench_suite_free(&suite);
}

OC_TEST_NULL_SAFE(bsuite, n_results_null,
        cr_assert_eq(oc_bench_suite_n_results(NULL), 0);)

Test(bsuite, type_name)
{
    cr_assert_str_eq(oc_bench_type_name(OC_BENCH_MATVEC_F32), "matvec_f32");
    cr_assert_str_eq(oc_bench_type_name(OC_BENCH_QUANTIZE), "quantize");
    cr_assert_str_eq(oc_bench_type_name(OC_BENCH_GENERATE), "generate");
    cr_assert_str_eq(oc_bench_type_name(OC_BENCH_SAMPLING), "sampling");
}

Test(bsuite, report)
{
    OcBenchSuite suite;
    oc_bench_suite_init(&suite);
    oc_bench_suite_run(&suite, OC_BENCH_MATVEC_F32, 5);
    char out[4096];
    cr_assert_eq(oc_bench_suite_report(&suite, out, sizeof(out)), OC_OK);
    cr_assert(strstr(out, "matvec_f32") != NULL);
    cr_assert(strstr(out, "Benchmark Suite Report") != NULL);
    oc_bench_suite_free(&suite);
}

OC_TEST_NULL_SAFE(bsuite, report_null,
        cr_assert_neq(oc_bench_suite_report(NULL, NULL, 0), OC_OK);)

Test(bsuite, time_now)
{
    double t1 = oc_bench_time_now();
    cr_assert_gt(t1, 0.0);
    double t2 = oc_bench_time_now();
    cr_assert_geq(t2, t1);
}

Test(bsuite, elapsed)
{
    double start = oc_bench_time_now();
    double e = oc_bench_elapsed(start);
    cr_assert_geq(e, 0.0);
}

OC_TEST_NULL_SAFE(bsuite, free_null,
        oc_bench_suite_free(NULL);)

OC_TEST_NULL_SAFE(bsuite, run_null,
        cr_assert_neq(oc_bench_suite_run(NULL, OC_BENCH_MATVEC_F32, 10), OC_OK);)
