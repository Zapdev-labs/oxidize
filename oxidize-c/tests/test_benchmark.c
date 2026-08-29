/* test_benchmark.c — benchmark utility tests. */
#include "framework.h"
#include "oxidize/benchmark.h"
#include <string.h>

Test(bench, format_json)
{
    OcBenchmarkResult r = {0};
    r.prefill_tok_per_sec = 1000.0;
    r.decode_tok_per_sec = 50.0;
    r.decode_latency_ms = 20.0;
    r.decode_tokens = 100;
    r.n_repeats = 3;
    char buf[2048];
    size_t n = oc_benchmark_format(&r, buf, sizeof(buf));
    cr_assert(n > 0, "should produce JSON");
    cr_assert(strstr(buf, "prefill_tok_per_sec") != NULL);
    cr_assert(strstr(buf, "1000.00") != NULL);
    cr_assert(strstr(buf, "decode_tok_per_sec") != NULL);
}

Test(bench, format_table)
{
    OcBenchmarkResult r = {0};
    r.prefill_tok_per_sec = 500.0;
    r.decode_tok_per_sec = 30.0;
    r.decode_latency_ms = 33.3;
    r.decode_tokens = 50;
    r.n_repeats = 1;
    char buf[2048];
    size_t n = oc_benchmark_format_table(&r, buf, sizeof(buf));
    cr_assert(n > 0, "should produce table");
    cr_assert(strstr(buf, "Benchmark") != NULL);
    cr_assert(strstr(buf, "Prefill") != NULL);
    cr_assert(strstr(buf, "Decode") != NULL);
}

Test(bench, matvec_benchmark)
{
    double tps = oc_benchmark_matvec(64, 128, 10);
    cr_assert(tps > 0.0, "should produce positive throughput");
}

Test(bench, oxk_benchmark)
{
    OcOxkBenchResult r;
    OcError e = oc_benchmark_oxk(64, 128, 10, &r);
    cr_assert_eq(e, OC_OK);
    cr_assert(r.q4_0_tok_per_sec > 0.0);
    cr_assert(r.q8_0_tok_per_sec > 0.0);
}

Test(bench, null_args)
{
    cr_assert_neq(oc_benchmark_run(NULL, NULL, NULL), OC_OK);
    cr_assert_neq(oc_benchmark_scaling(NULL, NULL), OC_OK);
}
