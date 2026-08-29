/* benchmark.c — Inference benchmarking implementation. */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/benchmark.h"

#include "oxidize/mem_util.h"
#include "oxidize/oxk.h"
#include "oxidize/sampling.h"
#include "oxidize/tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}


OcError oc_benchmark_run(OcLlamaModel *model, const OcBenchmarkConfig *cfg,
                          OcBenchmarkResult *out)
{
    if (!model || !cfg || !out) return OC_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));

    OcLlamaSession sess;
    OcError e = oc_llama_session_init(model, &sess);
    if (e != OC_OK) return e;

    uint32_t n_warmup = cfg->n_warmup ? cfg->n_warmup : 5;
    uint32_t n_meas   = cfg->n_tokens ? cfg->n_tokens : 50;
    uint32_t n_reps   = cfg->n_repeats ? cfg->n_repeats : 1;
    uint32_t batch_size = cfg->batch_size ? cfg->batch_size : 1;

    out->n_repeats = n_reps;

    /* Memory before. */
    if (cfg->track_memory) {
        OcMemUsage mu;
        if (oc_mem_usage_get(&mu) == OC_OK) {
            out->mem_before_rss = mu.rss;
        }
    }

    double *tok_latencies = malloc(n_meas * sizeof(double));
    if (!tok_latencies) { oc_llama_session_free(&sess); return OC_ERR_OOM; }

    double min_tps = 1e18, max_tps = 0.0, sum_tps = 0.0;
    double sum_decode_ms = 0.0;

    for (uint32_t rep = 0; rep < n_reps; rep++) {
        oc_llama_session_reset(&sess);

        /* Prefill phase (skipped entirely when prompt_length == 0 so that a decode-only configuration stays decode-only). */
        uint32_t prompt_len = cfg->prompt_length;
        double prefill_ms = 0.0;
        if (prompt_len > 0) {
            double prefill_start = now_ms();
            for (uint32_t b = 0; b < batch_size; b++) {
                oc_llama_session_reset(&sess);
                for (uint32_t i = 0; i < prompt_len; i++) {
                    uint32_t tok = (i == 0) ? 1 : (i % model->cfg.vocab_size);
                    e = oc_llama_forward(&sess, tok, NULL);
                    if (e != OC_OK) goto fail;
                }
            }
            prefill_ms = (now_ms() - prefill_start) / (double)batch_size;
        }

        if (prompt_len > 0 &&
            (rep == 0 || prefill_ms < out->prefill_latency_ms)) {
            out->prefill_latency_ms = prefill_ms;
            out->prefill_tokens = prompt_len;
            out->prefill_tok_per_sec = (prefill_ms > 0)
                ? (double)prompt_len / (prefill_ms / 1000.0)
                : 0.0;
        }

        /* Warmup decode. */
        for (uint32_t i = 0; i < n_warmup; i++) {
            e = oc_llama_forward(&sess, 1, sess.logits);
            if (e != OC_OK) goto fail;
        }

        /* Measured decode. */
        double decode_start = now_ms();
        for (uint32_t i = 0; i < n_meas; i++) {
            double t0 = now_ms();
            e = oc_llama_forward(&sess, 1, sess.logits);
            if (e != OC_OK) goto fail;
            tok_latencies[i] = now_ms() - t0;
        }
        double decode_end = now_ms();
        double decode_ms = decode_end - decode_start;
        sum_decode_ms += decode_ms;

        double tps = (decode_ms > 0)
            ? (double)n_meas / (decode_ms / 1000.0)
            : 0.0;

        if (tps < min_tps) min_tps = tps;
        if (tps > max_tps) max_tps = tps;
        sum_tps += tps;

        if (cfg->verbose) {
            fprintf(stderr, "  rep %u: prefill %.1f tok/s, decode %.1f tok/s\n",
                    rep, out->prefill_tok_per_sec, tps);
        }
    }

    /* Compute statistics. */
    out->decode_tokens = n_meas;
    out->decode_tok_per_sec = sum_tps / (double)n_reps;
    out->decode_tok_per_sec_min = min_tps;
    out->decode_tok_per_sec_max = max_tps;
    out->decode_tok_per_sec_mean = out->decode_tok_per_sec;
    /* Average wall-clock milliseconds per decoded token. */
    out->decode_latency_ms = (n_meas > 0 && n_reps > 0)
        ? sum_decode_ms / (double)((uint64_t)n_meas * n_reps)
        : 0.0;

    /* Latency stddev. */
    if (n_meas > 1) {
        double mean = 0.0;
        for (uint32_t i = 0; i < n_meas; i++) mean += tok_latencies[i];
        mean /= (double)n_meas;
        double var = 0.0;
        for (uint32_t i = 0; i < n_meas; i++) {
            double d = tok_latencies[i] - mean;
            var += d * d;
        }
        out->decode_latency_stddev = sqrt(var / (double)(n_meas - 1));
    }

    free(tok_latencies);

    /* Memory after. */
    if (cfg->track_memory) {
        OcMemUsage mu;
        if (oc_mem_usage_get(&mu) == OC_OK) {
            out->mem_after_rss = mu.rss;
            out->mem_peak_rss = mu.rss > out->mem_before_rss ? mu.rss : out->mem_before_rss;
        }
    }

    oc_llama_session_free(&sess);
    return OC_OK;

fail:
    free(tok_latencies);
    oc_llama_session_free(&sess);
    return e;
}

OcError oc_benchmark_scaling(OcLlamaModel *model, OcBenchmarkResult *out)
{
    if (!model || !out) return OC_ERR_INVALID_ARG;

    struct { uint32_t ctx; double *out_field; } tests[] = {
        { 128,  &out->throughput_at_128 },
        { 256,  &out->throughput_at_256 },
        { 512,  &out->throughput_at_512 },
        { 1024, &out->throughput_at_1024 },
    };

    for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); i++) {
        /* Skip contexts the model cannot hold (prefill + measured decode);
         * their fields stay 0 (= not measured). */
        if (tests[i].ctx + 20u + 3u > model->cfg.n_ctx) continue;

        OcBenchmarkConfig cfg = OC_BENCHMARK_DEFAULT;
        cfg.n_warmup = 3;
        cfg.n_tokens = 20;
        cfg.n_repeats = 1;
        cfg.prompt_length = tests[i].ctx;
        cfg.verbose = false;
        cfg.track_memory = false;

        OcBenchmarkResult r;
        OcError e = oc_benchmark_run(model, &cfg, &r);
        if (e != OC_OK) return e;
        *tests[i].out_field = r.decode_tok_per_sec;
    }
    return OC_OK;
}


size_t oc_benchmark_format(const OcBenchmarkResult *r, char *buf, size_t cap)
{
    int n = snprintf(buf, cap,
        "{"
        "\"prefill_tok_per_sec\":%.2f,"
        "\"prefill_latency_ms\":%.2f,"
        "\"prefill_tokens\":%u,"
        "\"decode_tok_per_sec\":%.2f,"
        "\"decode_latency_ms\":%.3f,"
        "\"decode_latency_stddev\":%.3f,"
        "\"decode_tokens\":%u,"
        "\"mem_before_rss\":%llu,"
        "\"mem_after_rss\":%llu,"
        "\"mem_peak_rss\":%llu,"
        "\"throughput_at_128\":%.2f,"
        "\"throughput_at_256\":%.2f,"
        "\"throughput_at_512\":%.2f,"
        "\"throughput_at_1024\":%.2f,"
        "\"n_repeats\":%u,"
        "\"decode_tok_per_sec_min\":%.2f,"
        "\"decode_tok_per_sec_max\":%.2f,"
        "\"decode_tok_per_sec_mean\":%.2f"
        "}",
        r->prefill_tok_per_sec, r->prefill_latency_ms, r->prefill_tokens,
        r->decode_tok_per_sec, r->decode_latency_ms, r->decode_latency_stddev,
        r->decode_tokens,
        (unsigned long long)r->mem_before_rss,
        (unsigned long long)r->mem_after_rss,
        (unsigned long long)r->mem_peak_rss,
        r->throughput_at_128, r->throughput_at_256,
        r->throughput_at_512, r->throughput_at_1024,
        r->n_repeats, r->decode_tok_per_sec_min,
        r->decode_tok_per_sec_max, r->decode_tok_per_sec_mean
    );
    if (n < 0 || (size_t)n >= cap) return 0;
    return (size_t)n;
}

size_t oc_benchmark_format_table(const OcBenchmarkResult *r, char *buf, size_t cap)
{
    int n = snprintf(buf, cap,
        "┌────────────────────────────────────────────┐\n"
        "│ Benchmark Results                          │\n"
        "├────────────────────────────────────────────┤\n"
        "│ Prefill:  %7.1f tok/s  (%6.1f ms, %4u tok)│\n"
        "│ Decode:   %7.1f tok/s  (%6.3f ms/tok)     │\n"
        "│ Latency:  mean=%.3f ms  σ=%.3f ms          │\n"
        "│ Memory:   %llu → %llu MB (peak %llu MB)    │\n"
        "│ Repeats:  %u (min=%.1f max=%.1f mean=%.1f) │\n"
        "│ Throughput scaling:                        │\n"
        "│   ctx=128:  %7.1f tok/s                    │\n"
        "│   ctx=256:  %7.1f tok/s                    │\n"
        "│   ctx=512:  %7.1f tok/s                    │\n"
        "│   ctx=1024: %7.1f tok/s                    │\n"
        "└────────────────────────────────────────────┘\n",
        r->prefill_tok_per_sec, r->prefill_latency_ms, r->prefill_tokens,
        r->decode_tok_per_sec, r->decode_latency_ms,
        r->decode_latency_ms, r->decode_latency_stddev,
        (unsigned long long)(r->mem_before_rss >> 20),
        (unsigned long long)(r->mem_after_rss >> 20),
        (unsigned long long)(r->mem_peak_rss >> 20),
        r->n_repeats, r->decode_tok_per_sec_min,
        r->decode_tok_per_sec_max, r->decode_tok_per_sec_mean,
        r->throughput_at_128, r->throughput_at_256,
        r->throughput_at_512, r->throughput_at_1024
    );
    if (n < 0 || (size_t)n >= cap) return 0;
    return (size_t)n;
}

void oc_benchmark_print(const OcBenchmarkResult *r)
{
    char buf[2048];
    if (oc_benchmark_format_table(r, buf, sizeof(buf)) > 0) {
        fprintf(stderr, "%s", buf);
    }
}


double oc_benchmark_matvec(uint32_t n_rows, uint32_t n_cols, uint32_t n_iters)
{
    float *weights = malloc((size_t)n_rows * n_cols * sizeof(float));
    float *input = malloc(n_cols * sizeof(float));
    float *output = malloc(n_rows * sizeof(float));
    if (!weights || !input || !output) {
        free(weights); free(input); free(output);
        return 0.0;
    }

    /* Fill with random data. */
    for (size_t i = 0; i < (size_t)n_rows * n_cols; i++)
        weights[i] = (float)(i % 7) * 0.1f;
    for (size_t i = 0; i < n_cols; i++)
        input[i] = (float)(i % 5) * 0.1f;

    double start = now_ms();
    for (uint32_t iter = 0; iter < n_iters; iter++) {
        for (uint32_t r = 0; r < n_rows; r++) {
            float sum = 0.0f;
            for (uint32_t c = 0; c < n_cols; c++)
                sum += weights[r * n_cols + c] * input[c];
            output[r] = sum;
        }
    }
    double elapsed = now_ms() - start;

    free(weights); free(input); free(output);

    if (elapsed <= 0.0) return 0.0;
    return (double)n_rows * n_iters / (elapsed / 1000.0);
}

/* Time one OXK quantized matvec entry point over `n_iters` iterations. */
static double bench_oxk_matvec(void (*matvec)(const uint8_t *, size_t, size_t,
                                              const float *, float *),
                               uint32_t n_rows, uint32_t n_cols,
                               uint32_t n_iters, uint32_t elems_per_block,
                               uint32_t block_bytes)
{
    size_t blocks = (n_cols + elems_per_block - 1) / elems_per_block;
    size_t row_bytes = blocks * block_bytes;
    size_t x_len = blocks * elems_per_block;

    uint8_t *w = calloc((size_t)n_rows, row_bytes);
    float *x = calloc(x_len, sizeof(float));
    float *y = calloc(n_rows, sizeof(float));
    if (!w || !x || !y) { free(w); free(x); free(y); return 0.0; }
    for (size_t i = 0; i < x_len; i++) x[i] = (float)(i % 5) * 0.1f;

    double start = now_ms();
    for (uint32_t it = 0; it < n_iters; it++)
        matvec(w, n_rows, row_bytes, x, y);
    double elapsed = now_ms() - start;

    free(w); free(x); free(y);
    if (elapsed <= 0.0) return 0.0;
    return (double)n_rows * n_iters / (elapsed / 1000.0);
}

OcError oc_benchmark_oxk(uint32_t n_rows, uint32_t n_cols,
                         uint32_t n_iters, OcOxkBenchResult *out)
{
    if (!out || n_rows == 0 || n_cols == 0 || n_iters == 0)
        return OC_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));

    oc_oxk_init();

    /* Time the real dispatched OXK matvec kernels. Q5_K/Q6_K have no OXK
     * matvec entry point; their fields stay 0 (unavailable) rather than
     * being estimated. */
    out->q4_0_tok_per_sec = bench_oxk_matvec(oc_oxk_matvec_q4_0_f32,
                                             n_rows, n_cols, n_iters,
                                             OC_OXK_QK4_0,
                                             OC_OXK_BLOCK_Q4_0_SIZE);
    out->q4_k_tok_per_sec = bench_oxk_matvec(oc_oxk_matvec_q4_k_f32,
                                             n_rows, n_cols, n_iters,
                                             OC_OXK_QK_K,
                                             OC_OXK_BLOCK_Q4_K_SIZE);
    out->q8_0_tok_per_sec = bench_oxk_matvec(oc_oxk_matvec_q8_0_f32,
                                             n_rows, n_cols, n_iters,
                                             OC_OXK_QK8_0,
                                             OC_OXK_BLOCK_Q8_0_SIZE);

    return OC_OK;
}
