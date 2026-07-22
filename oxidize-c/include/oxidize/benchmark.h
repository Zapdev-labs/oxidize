/*
 * benchmark.h — Inference benchmarking utility.
 *
 * Port of oxidize-core/src/util/benchmark.rs. Provides structured
 * benchmarking for forward passes (prefill + decode), measuring:
 *   - Tokens per second (prefill and decode separately)
 *   - Wall-clock latency per token
 *   - Memory usage before/after
 *   - Throughput vs. context length scaling
 *
 * Results are formatted as JSON for integration with CI gating.
 */
#ifndef OXIDIZE_BENCHMARK_H
#define OXIDIZE_BENCHMARK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/llama.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Benchmark configuration ──────────────────────────────────────────── */

typedef struct OcBenchmarkConfig {
    uint32_t n_warmup;          /* warmup tokens (not measured)            */
    uint32_t n_tokens;          /* measured tokens                          */
    uint32_t batch_size;        /* prefill batch size (1 for pure decode)   */
    uint32_t n_repeats;         /* repeat the whole benchmark N times        */
    bool     verbose;           /* print progress to stderr                  */
    bool     track_memory;      /* measure RSS before/after                   */
    uint32_t prompt_length;     /* prefill prompt length (0 = decode-only)    */
} OcBenchmarkConfig;

#define OC_BENCHMARK_DEFAULT ((OcBenchmarkConfig){ \
    5, 50, 1, 3, true, true, 32 })

/* ─── Benchmark results ────────────────────────────────────────────────── */

typedef struct OcBenchmarkResult {
    /* Prefill (batch) metrics. */
    double   prefill_tok_per_sec;     /* tokens/sec during prefill       */
    double   prefill_latency_ms;      /* total prefill time in ms        */
    uint32_t prefill_tokens;           /* tokens processed in prefill     */

    /* Decode metrics. */
    double   decode_tok_per_sec;       /* tokens/sec during decode       */
    double   decode_latency_ms;        /* avg ms per decode token         */
    double   decode_latency_stddev;   /* stddev of per-token latency      */
    uint32_t decode_tokens;           /* total decode tokens measured     */

    /* Memory. */
    uint64_t mem_before_rss;          /* RSS before benchmark (bytes)    */
    uint64_t mem_after_rss;           /* RSS after benchmark (bytes)     */
    uint64_t mem_peak_rss;            /* Peak RSS observed (bytes)       */

    /* Throughput scaling (tok/s at different ctx lengths). */
    double   throughput_at_128;       /* tok/s at ctx=128                 */
    double   throughput_at_256;
    double   throughput_at_512;
    double   throughput_at_1024;

    /* Repeat statistics. */
    uint32_t n_repeats;
    double   decode_tok_per_sec_min;
    double   decode_tok_per_sec_max;
    double   decode_tok_per_sec_mean;
} OcBenchmarkResult;

/* Run a single benchmark on a loaded model. */
OcError oc_benchmark_run(OcLlamaModel *model, const OcBenchmarkConfig *cfg,
                          OcBenchmarkResult *out);

/* Run a throughput scaling benchmark: measure tok/s at multiple context
 * lengths. Fills the throughput_at_* fields. */
OcError oc_benchmark_scaling(OcLlamaModel *model,
                              OcBenchmarkResult *out);

/* Format benchmark results as JSON. Returns bytes written or 0 on overflow. */
size_t oc_benchmark_format(const OcBenchmarkResult *r, char *buf, size_t cap);

/* Format as a human-readable table. */
size_t oc_benchmark_format_table(const OcBenchmarkResult *r, char *buf, size_t cap);

/* Print results to stderr. */
void oc_benchmark_print(const OcBenchmarkResult *r);

/* ─── Micro-benchmarks ────────────────────────────────────────────────── */

/* Benchmark a specific quantized matvec kernel. Returns tok/s. */
double oc_benchmark_matvec(uint32_t n_rows, uint32_t n_cols,
                           uint32_t n_iters);

/* Benchmark the OXK kernels. Returns tok/s for each quant type. */
typedef struct OcOxkBenchResult {
    double q4_0_tok_per_sec;
    double q4_k_tok_per_sec;
    double q8_0_tok_per_sec;
    double q5_k_tok_per_sec;
    double q6_k_tok_per_sec;
} OcOxkBenchResult;

OcError oc_benchmark_oxk(uint32_t n_rows, uint32_t n_cols,
                         uint32_t n_iters, OcOxkBenchResult *out);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_BENCHMARK_H */
