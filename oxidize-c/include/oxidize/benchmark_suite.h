/* benchmark_suite.h — Comprehensive benchmark suite. */
#ifndef OXIDIZE_BENCHMARK_SUITE_H
#define OXIDIZE_BENCHMARK_SUITE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_BENCH_MAX_RESULTS 32
#define OC_BENCH_MAX_NAME 64

typedef enum {
    OC_BENCH_MATVEC_F32 = 0,
    OC_BENCH_MATVEC_Q8_0 = 1,
    OC_BENCH_MATVEC_Q4_0 = 2,
    OC_BENCH_MATVEC_Q4_K = 3,
    OC_BENCH_QUANTIZE = 4,
    OC_BENCH_TOKENIZE = 5,
    OC_BENCH_GENERATE = 6,
    OC_BENCH_PREFILL = 7,
    OC_BENCH_SAMPLING = 8,
} OcBenchType;

typedef struct {
    char name[OC_BENCH_MAX_NAME];
    OcBenchType type;
    double elapsed_sec;
    double tokens_per_sec;
    size_t n_iterations;
    size_t n_items;
    size_t data_size_bytes;
    double throughput_gbps;
} OcBenchResult;

typedef struct {
    OcBenchResult results[OC_BENCH_MAX_RESULTS];
    uint32_t n_results;
    bool verbose;
} OcBenchSuite;

OcError oc_bench_suite_init(OcBenchSuite *suite);
OcError oc_bench_suite_run(OcBenchSuite *suite, OcBenchType type,
                          size_t n_iterations);
OcError oc_bench_suite_run_all(OcBenchSuite *suite);
OcError oc_bench_suite_add_result(OcBenchSuite *suite,
                                 const OcBenchResult *result);
OcError oc_bench_suite_report(const OcBenchSuite *suite, char *out, size_t out_size);
const OcBenchResult *oc_bench_suite_get_result(const OcBenchSuite *suite, uint32_t idx);
uint32_t oc_bench_suite_n_results(const OcBenchSuite *suite);
const char *oc_bench_type_name(OcBenchType type);
void oc_bench_suite_free(OcBenchSuite *suite);

/* Timing helpers. */
double oc_bench_time_now(void);
double oc_bench_elapsed(double start);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_BENCHMARK_SUITE_H */
