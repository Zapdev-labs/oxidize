/* benchmark_suite.c — Benchmark suite implementation. */
#include "oxidize/benchmark_suite.h"
#include "oxidize/quant.h"
#include "oxidize/activation.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

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


static OcError bench_matvec_f32(size_t n_iterations, OcBenchResult *result)
{
    /* Real F32 GEMV: [rows, cols] @ [cols] → [rows]. */
    const size_t rows = 512;
    const size_t cols = 512;
    float *weights = malloc(rows * cols * sizeof(float));
    float *input = malloc(cols * sizeof(float));
    float *output = malloc(rows * sizeof(float));
    if (!weights || !input || !output) {
        free(weights); free(input); free(output);
        return OC_ERR_OOM;
    }
    /* Fill with deterministic data. */
    for (size_t i = 0; i < rows * cols; i++) weights[i] = (float)(i % 7) * 0.1f;
    for (size_t i = 0; i < cols; i++) input[i] = (float)(i % 5) * 0.2f;

    double start = oc_bench_time_now();
    for (size_t iter = 0; iter < n_iterations; iter++) {
        for (size_t r = 0; r < rows; r++) {
            float dot = 0.0f;
            const float *wrow = weights + r * cols;
            for (size_t c = 0; c < cols; c++)
                dot += wrow[c] * input[c];
            output[r] = dot;
        }
    }
    result->elapsed_sec = oc_bench_elapsed(start);
    result->n_items = n_iterations;
    result->data_size_bytes = n_iterations * (rows * cols + cols + rows) * sizeof(float);
    free(weights); free(input); free(output);
    return OC_OK;
}

static OcError bench_matvec_q8_0(size_t n_iterations, OcBenchResult *result)
{
    /* Q8_0 dequant + GEMV. */
    const size_t rows = 512;
    const size_t block_size = 32;
    const size_t cols = 512;
    const size_t n_blocks = cols / block_size;
    const size_t block_bytes = 2 + block_size; /* f16 d + int8 qs */
    float *input = malloc(cols * sizeof(float));
    float *output = malloc(rows * sizeof(float));
    uint8_t *weights = malloc(rows * n_blocks * block_bytes);
    if (!input || !output || !weights) {
        free(input); free(output); free(weights);
        return OC_ERR_OOM;
    }
    for (size_t i = 0; i < rows * n_blocks * block_bytes; i++)
        weights[i] = (uint8_t)(i * 7 + 3);
    for (size_t i = 0; i < cols; i++) input[i] = (float)(i % 5) * 0.2f;

    double start = oc_bench_time_now();
    for (size_t iter = 0; iter < n_iterations; iter++) {
        for (size_t r = 0; r < rows; r++) {
            float dot = 0.0f;
            for (size_t b = 0; b < n_blocks; b++) {
                const uint8_t *blk = weights + (r * n_blocks + b) * block_bytes;
                /* Dequantize f16 scale. */
                uint16_t d_bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
                uint32_t f32_bits = ((uint32_t)(d_bits & 0x8000) << 16) |
                                    (((uint32_t)(d_bits & 0x7C00) + 0x3800) << 13) |
                                    ((uint32_t)(d_bits & 0x03FF) << 13);
                float d;
                memcpy(&d, &f32_bits, sizeof(float));
                for (size_t j = 0; j < block_size; j++) {
                    float q = (float)((int8_t)blk[2 + j]);
                    dot += d * q * input[b * block_size + j];
                }
            }
            output[r] = dot;
        }
    }
    result->elapsed_sec = oc_bench_elapsed(start);
    result->n_items = n_iterations;
    result->data_size_bytes = n_iterations * rows * cols;
    free(input); free(output); free(weights);
    return OC_OK;
}

static OcError bench_matvec_q4_0(size_t n_iterations, OcBenchResult *result)
{
    /* Q4_0 dequant + GEMV. */
    const size_t rows = 512;
    const size_t block_size = 32;
    const size_t cols = 512;
    const size_t n_blocks = cols / block_size;
    const size_t block_bytes = 2 + block_size / 2; /* f16 d + 4-bit qs */
    float *input = malloc(cols * sizeof(float));
    float *output = malloc(rows * sizeof(float));
    uint8_t *weights = malloc(rows * n_blocks * block_bytes);
    if (!input || !output || !weights) {
        free(input); free(output); free(weights);
        return OC_ERR_OOM;
    }
    for (size_t i = 0; i < rows * n_blocks * block_bytes; i++)
        weights[i] = (uint8_t)(i * 7 + 3);
    /* Set scales to 1.0 to avoid NaN. */
    for (size_t i = 0; i < rows * n_blocks; i++) {
        weights[i * block_bytes] = 0x00;
        weights[i * block_bytes + 1] = 0x3C;
    }
    for (size_t i = 0; i < cols; i++) input[i] = (float)(i % 5) * 0.2f;

    double start = oc_bench_time_now();
    for (size_t iter = 0; iter < n_iterations; iter++) {
        for (size_t r = 0; r < rows; r++) {
            float dot = 0.0f;
            for (size_t b = 0; b < n_blocks; b++) {
                const uint8_t *blk = weights + (r * n_blocks + b) * block_bytes;
                uint16_t d_bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
                uint32_t f32_bits = ((uint32_t)(d_bits & 0x8000) << 16) |
                                    (((uint32_t)(d_bits & 0x7C00) + 0x3800) << 13) |
                                    ((uint32_t)(d_bits & 0x03FF) << 13);
                float d;
                memcpy(&d, &f32_bits, sizeof(float));
                for (size_t j = 0; j < block_size / 2; j++) {
                    uint8_t lo = blk[2 + j] & 0x0F;
                    uint8_t hi = (blk[2 + j] >> 4) & 0x0F;
                    float q_lo = d * ((float)lo - 8.0f);
                    float q_hi = d * ((float)hi - 8.0f);
                    dot += q_lo * input[b * block_size + j * 2];
                    dot += q_hi * input[b * block_size + j * 2 + 1];
                }
            }
            output[r] = dot;
        }
    }
    result->elapsed_sec = oc_bench_elapsed(start);
    result->n_items = n_iterations;
    result->data_size_bytes = n_iterations * rows * cols / 2;
    free(input); free(output); free(weights);
    return OC_OK;
}

static OcError bench_matvec_q4_k(size_t n_iterations, OcBenchResult *result)
{
    /* Q4_K GEMV: use oc_quant_dequant_row for a block then dot product. */
    const size_t block_elems = 256;
    const size_t block_bytes = 144; /* OC_BLOCK_Q4_K_SIZE */
    const size_t n_blocks = 4;
    const size_t cols = block_elems * n_blocks;
    const size_t rows = 64;
    float *input = malloc(cols * sizeof(float));
    float *output = malloc(rows * sizeof(float));
    uint8_t *weights = calloc(rows * n_blocks, block_bytes);
    float *dequant = malloc(block_elems * sizeof(float));
    if (!input || !output || !weights || !dequant) {
        free(input); free(output); free(weights); free(dequant);
        return OC_ERR_OOM;
    }
    for (size_t i = 0; i < cols; i++) input[i] = (float)(i % 5) * 0.01f;

    double start = oc_bench_time_now();
    for (size_t iter = 0; iter < n_iterations; iter++) {
        for (size_t r = 0; r < rows; r++) {
            float dot = 0.0f;
            for (size_t b = 0; b < n_blocks; b++) {
                const uint8_t *blk = weights + (r * n_blocks + b) * block_bytes;
                oc_quant_dequant_row(OC_QUANT_Q4_K_M, blk, block_bytes,
                                     dequant, block_elems);
                for (size_t j = 0; j < block_elems; j++)
                    dot += dequant[j] * input[b * block_elems + j];
            }
            output[r] = dot;
        }
    }
    result->elapsed_sec = oc_bench_elapsed(start);
    result->n_items = n_iterations;
    result->data_size_bytes = n_iterations * rows * cols / 2;
    free(input); free(output); free(weights); free(dequant);
    return OC_OK;
}

static OcError bench_quantize(size_t n_iterations, OcBenchResult *result)
{
    /* Real F32→Q8_0 quantization. */
    const size_t block_size = 32;
    const size_t n_blocks = 16;
    const size_t n_elems = block_size * n_blocks;
    const size_t dst_size = n_blocks * (2 + block_size);
    float *src = malloc(n_elems * sizeof(float));
    uint8_t *dst = malloc(dst_size);
    if (!src || !dst) { free(src); free(dst); return OC_ERR_OOM; }
    for (size_t i = 0; i < n_elems; i++) src[i] = (float)(i % 10) * 0.1f;

    double start = oc_bench_time_now();
    for (size_t iter = 0; iter < n_iterations; iter++)
        oc_quant_pack_row(OC_QUANT_Q8_0, src, n_elems, dst, dst_size);
    result->elapsed_sec = oc_bench_elapsed(start);
    result->n_items = n_iterations;
    result->data_size_bytes = n_iterations * n_elems * sizeof(float);
    free(src); free(dst);
    return OC_OK;
}

static OcError bench_tokenize(size_t n_iterations, OcBenchResult *result)
{
    /* Simulate tokenization throughput: BPE-like splitting. */
    const char *text = "The quick brown fox jumps over the lazy dog. "
                       "Lorem ipsum dolor sit amet, consectetur adipiscing elit.";
    size_t text_len = strlen(text);
    double start = oc_bench_time_now();
    size_t total_tokens = 0;
    for (size_t iter = 0; iter < n_iterations; iter++) {
        size_t pos = 0;
        while (pos < text_len) {
            /* Simple word boundary detection. */
            while (pos < text_len && (text[pos] == ' ' || text[pos] == '\t')) pos++;
            if (pos >= text_len) break;
            while (pos < text_len && text[pos] != ' ' && text[pos] != '\t' &&
                   text[pos] != '.' && text[pos] != ',') pos++;
            if (pos < text_len) pos++; /* consume delimiter */
            total_tokens++;
        }
    }
    result->elapsed_sec = oc_bench_elapsed(start);
    result->n_items = total_tokens;
    result->data_size_bytes = n_iterations * text_len;
    free(NULL);
    return OC_OK;
}

static OcError bench_generate(size_t n_iterations, OcBenchResult *result)
{
    /* Simulate token generation: RMSNorm + GEMV + argmax per token. */
    const size_t hidden = 512;
    const size_t vocab = 1000;
    float *hidden_state = malloc(hidden * sizeof(float));
    float *norm_w = malloc(hidden * sizeof(float));
    float *lm_head = malloc(vocab * hidden * sizeof(float));
    float *logits = malloc(vocab * sizeof(float));
    if (!hidden_state || !norm_w || !lm_head || !logits) {
        free(hidden_state); free(norm_w); free(lm_head); free(logits);
        return OC_ERR_OOM;
    }
    for (size_t i = 0; i < hidden; i++) {
        hidden_state[i] = (float)(i % 7) * 0.01f;
        norm_w[i] = 1.0f;
    }
    for (size_t i = 0; i < vocab * hidden; i++) lm_head[i] = (float)(i % 5) * 0.001f;

    double start = oc_bench_time_now();
    for (size_t iter = 0; iter < n_iterations; iter++) {
        /* RMSNorm. */
        float ss = 0.0f;
        for (size_t i = 0; i < hidden; i++) ss += hidden_state[i] * hidden_state[i];
        float rms = 1.0f / sqrtf(ss / hidden + 1e-6f);
        /* GEMV: logits = lm_head @ normed_hidden. */
        uint32_t best_token = 0;
        float best_logit = -INFINITY;
        for (size_t r = 0; r < vocab; r++) {
            float dot = 0.0f;
            const float *wrow = lm_head + r * hidden;
            for (size_t c = 0; c < hidden; c++)
                dot += wrow[c] * hidden_state[c] * rms * norm_w[c];
            logits[r] = dot;
            if (dot > best_logit) { best_logit = dot; best_token = (uint32_t)r; }
        }
        /* Update hidden state with token embedding. */
        for (size_t i = 0; i < hidden; i++)
            hidden_state[i] = hidden_state[i] * 0.99f + 0.01f * (float)best_token;
    }
    result->elapsed_sec = oc_bench_elapsed(start);
    result->n_items = n_iterations;
    result->data_size_bytes = n_iterations * (vocab * hidden + hidden) * sizeof(float);
    free(hidden_state); free(norm_w); free(lm_head); free(logits);
    return OC_OK;
}

static OcError bench_prefill(size_t n_iterations, OcBenchResult *result)
{
    /* Simulate prefill: batched GEMV over sequence length. */
    const size_t hidden = 512;
    const size_t seq_len = 128;
    float *input = malloc(seq_len * hidden * sizeof(float));
    float *weights = malloc(hidden * hidden * sizeof(float));
    float *output = malloc(seq_len * hidden * sizeof(float));
    if (!input || !weights || !output) {
        free(input); free(weights); free(output);
        return OC_ERR_OOM;
    }
    for (size_t i = 0; i < seq_len * hidden; i++) input[i] = (float)(i % 7) * 0.01f;
    for (size_t i = 0; i < hidden * hidden; i++) weights[i] = (float)(i % 5) * 0.001f;

    double start = oc_bench_time_now();
    for (size_t iter = 0; iter < n_iterations; iter++) {
        for (size_t s = 0; s < seq_len; s++) {
            const float *in = input + s * hidden;
            float *out = output + s * hidden;
            for (size_t r = 0; r < hidden; r++) {
                float dot = 0.0f;
                const float *wrow = weights + r * hidden;
                for (size_t c = 0; c < hidden; c++) dot += wrow[c] * in[c];
                out[r] = dot;
            }
        }
    }
    result->elapsed_sec = oc_bench_elapsed(start);
    result->n_items = n_iterations * seq_len;
    result->data_size_bytes = n_iterations * seq_len * hidden * sizeof(float) * 2;
    free(input); free(weights); free(output);
    return OC_OK;
}

static OcError bench_sampling(size_t n_iterations, OcBenchResult *result)
{
    /* Real softmax + argmax sampling. */
    const size_t vocab = 32000;
    float *logits = malloc(vocab * sizeof(float));
    float *probs = malloc(vocab * sizeof(float));
    if (!logits || !probs) { free(logits); free(probs); return OC_ERR_OOM; }
    for (size_t i = 0; i < vocab; i++) logits[i] = (float)(i % 100) * 0.01f - 0.5f;

    double start = oc_bench_time_now();
    for (size_t iter = 0; iter < n_iterations; iter++) {
        /* Softmax. */
        float max_logit = -INFINITY;
        for (size_t i = 0; i < vocab; i++)
            if (logits[i] > max_logit) max_logit = logits[i];
        float sum = 0.0f;
        for (size_t i = 0; i < vocab; i++) {
            probs[i] = expf(logits[i] - max_logit);
            sum += probs[i];
        }
        float inv = 1.0f / sum;
        /* Argmax (greedy). */
        uint32_t best = 0;
        float best_p = 0.0f;
        for (size_t i = 0; i < vocab; i++) {
            probs[i] *= inv;
            if (probs[i] > best_p) { best_p = probs[i]; best = (uint32_t)i; }
        }
        /* Touch best to prevent optimizer from removing. */
        logits[0] += (float)best * 1e-30f;
    }
    result->elapsed_sec = oc_bench_elapsed(start);
    result->n_items = n_iterations;
    result->data_size_bytes = n_iterations * vocab * sizeof(float) * 2;
    free(logits); free(probs);
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

    OcError e = OC_OK;
    switch (type) {
    case OC_BENCH_MATVEC_F32:  e = bench_matvec_f32(n_iterations, &result); break;
    case OC_BENCH_MATVEC_Q8_0: e = bench_matvec_q8_0(n_iterations, &result); break;
    case OC_BENCH_MATVEC_Q4_0: e = bench_matvec_q4_0(n_iterations, &result); break;
    case OC_BENCH_MATVEC_Q4_K: e = bench_matvec_q4_k(n_iterations, &result); break;
    case OC_BENCH_QUANTIZE:    e = bench_quantize(n_iterations, &result); break;
    case OC_BENCH_TOKENIZE:    e = bench_tokenize(n_iterations, &result); break;
    case OC_BENCH_GENERATE:     e = bench_generate(n_iterations, &result); break;
    case OC_BENCH_PREFILL:     e = bench_prefill(n_iterations, &result); break;
    case OC_BENCH_SAMPLING:    e = bench_sampling(n_iterations, &result); break;
    default:
        return OC_ERR_INVALID_ARG;
    }
    if (e != OC_OK) return e;

    if (result.elapsed_sec > 0) {
        result.throughput_gbps = (double)result.data_size_bytes /
                                 (result.elapsed_sec * 1e9);
        result.tokens_per_sec = (double)result.n_items / result.elapsed_sec;
    }

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
