/*
 * benchmark_suite.c — Benchmark suite implementation.
 */
#include "oxidize/benchmark_suite.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

static void copy_str(char *dst, size_t cap, const char *src)
{
    if (!dst || cap == 0 || !src) { if (dst && cap > 0) dst[0] = '\0'; return; }
    size_t n = strlen(src);
    if (n >= cap) n = cap - 1;
    memcpy(dst, src, n);
    dst[n] = '\0';
}

OcError oc_bench_suite_init(OcBenchSuite *suite)
{
    if (!suite) return OC_ERR_INVALID_ARG;
    memset(suite, 0, sizeof(*suite));
    suite->verbose = false;
    return OC_OK;
}

double oc_bench_time_now(void)
{
    struct timespec ts;
    if (timespec_get(&ts, TIME_UTC) != TIME_UTC) return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

double oc_bench_elapsed(double start)
{
    return oc_bench_time_now() - start;
}

OcError oc_bench_suite_add_result(OcBenchSuite *suite,
                                 const OcBenchResult *result)
{
    if (!suite || !result) return OC_ERR_INVALID_ARG;
    if (suite->n_results >= OC_BENCH_MAX_RESULTS) return OC_ERR_OOM;
    suite->results[suite->n_results] = *result;
    suite->n_results++;
    return OC_OK;
}

OcError oc_bench_suite_run(OcBenchSuite *suite, OcBenchType type,
                          size_t n_iterations)
{
    if (!suite) return OC_ERR_INVALID_ARG;
    if (n_iterations == 0) n_iterations = 100;

    OcBenchResult result;
    memset(&result, 0, sizeof(result));
    result.type = type;
    copy_str(result.name, sizeof(result.name), oc_bench_type_name(type));
    result.n_iterations = n_iterations;

    /* Stub benchmark: simulate work with a small allocation + memset. */
    size_t item_size = 1024;
    void *buf = malloc(item_size);
    if (!buf) return OC_ERR_OOM;

    double start = oc_bench_time_now();
    for (size_t i = 0; i < n_iterations; i++) {
        memset(buf, (int)(i & 0xFF), item_size);
    }
    result.elapsed_sec = oc_bench_elapsed(start);
    result.n_items = n_iterations;
    result.data_size_bytes = n_iterations * item_size;
    if (result.elapsed_sec > 0) {
        result.throughput_gbps = (double)result.data_size_bytes /
                                 (result.elapsed_sec * 1e9);
        result.tokens_per_sec = (double)n_iterations / result.elapsed_sec;
    }

    free(buf);
    return oc_bench_suite_add_result(suite, &result);
}

OcError oc_bench_suite_run_all(OcBenchSuite *suite)
{
    if (!suite) return OC_ERR_INVALID_ARG;
    static const OcBenchType all_types[] = {
        OC_BENCH_MATVEC_F32, OC_BENCH_MATVEC_Q8_0, OC_BENCH_MATVEC_Q4_0,
        OC_BENCH_MATVEC_Q4_K, OC_BENCH_QUANTIZE, OC_BENCH_TOKENIZE,
        OC_BENCH_GENERATE, OC_BENCH_PREFILL, OC_BENCH_SAMPLING,
    };
    for (size_t i = 0; i < sizeof(all_types) / sizeof(all_types[0]); i++) {
        OcError e = oc_bench_suite_run(suite, all_types[i], 100);
        if (e != OC_OK) return e;
    }
    return OC_OK;
}

OcError oc_bench_suite_report(const OcBenchSuite *suite, char *out, size_t out_size)
{
    if (!suite || !out || out_size == 0) return OC_ERR_INVALID_ARG;
    size_t pos = 0;
    int written;

    written = snprintf(out + pos, out_size - pos,
        "=== Benchmark Suite Report ===\n"
        "%-20s %-12s %-12s %-12s %-12s\n",
        "Name", "Time(s)", "Tok/s", "Items", "GB/s");
    if (written < 0 || (size_t)written >= out_size - pos) return OC_ERR_OOM;
    pos += written;

    for (uint32_t i = 0; i < suite->n_results && pos < out_size; i++) {
        const OcBenchResult *r = &suite->results[i];
        written = snprintf(out + pos, out_size - pos,
            "%-20s %-12.6f %-12.2f %-12zu %-12.2f\n",
            r->name, r->elapsed_sec, r->tokens_per_sec,
            r->n_items, r->throughput_gbps);
        if (written < 0 || (size_t)written >= out_size - pos) return OC_ERR_OOM;
        pos += written;
    }
    return OC_OK;
}

const OcBenchResult *oc_bench_suite_get_result(const OcBenchSuite *suite, uint32_t idx)
{
    if (!suite || idx >= suite->n_results) return NULL;
    return &suite->results[idx];
}

uint32_t oc_bench_suite_n_results(const OcBenchSuite *suite)
{
    return suite ? suite->n_results : 0;
}

const char *oc_bench_type_name(OcBenchType type)
{
    switch (type) {
    case OC_BENCH_MATVEC_F32:  return "matvec_f32";
    case OC_BENCH_MATVEC_Q8_0: return "matvec_q8_0";
    case OC_BENCH_MATVEC_Q4_0: return "matvec_q4_0";
    case OC_BENCH_MATVEC_Q4_K: return "matvec_q4_k";
    case OC_BENCH_QUANTIZE:    return "quantize";
    case OC_BENCH_TOKENIZE:    return "tokenize";
    case OC_BENCH_GENERATE:    return "generate";
    case OC_BENCH_PREFILL:     return "prefill";
    case OC_BENCH_SAMPLING:    return "sampling";
    default: return "unknown";
    }
}

void oc_bench_suite_free(OcBenchSuite *suite)
{
    if (!suite) return;
    memset(suite, 0, sizeof(*suite));
}
