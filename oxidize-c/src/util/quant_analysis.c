/*
 * quant_analysis.c — Quantization quality analysis implementation.
 */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/quant_analysis.h"

#include "oxidize/quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── Helpers ──────────────────────────────────────────────────────────── */

static int cmp_double(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

static double percentile(double *sorted, size_t n, double p)
{
    if (n == 0) return 0.0;
    size_t idx = (size_t)((double)(n - 1) * p);
    return sorted[idx];
}

/* ─── Quality metrics ──────────────────────────────────────────────────── */

OcError oc_quant_analyze(const float *f32_data, const uint8_t *quant_data,
                          OcGgufQuantizationType qtype, size_t n,
                          OcQuantMetrics *out)
{
    if (!f32_data || !quant_data || !out || n == 0)
        return OC_ERR_INVALID_ARG;

    memset(out, 0, sizeof(*out));
    out->n_elements = n;

    /* Dequantize the quantized data back to f32. */
    float *dequant = malloc(n * sizeof(float));
    if (!dequant) return OC_ERR_OOM;

    OcError e = oc_quant_dequant_row(qtype, quant_data, oc_quantized_size(qtype, n), dequant, n);
    if (e != OC_OK) { free(dequant); return e; }

    /* Compute metrics. */
    double sum_sq_err = 0.0;
    double sum_abs_err = 0.0;
    double max_abs = 0.0;
    double sum_orig_sq = 0.0;
    double sum_dequant_sq = 0.0;
    double sum_dot = 0.0;
    double sum_orig = 0.0;
    double sum_dequant = 0.0;

    for (size_t i = 0; i < n; i++) {
        double diff = (double)f32_data[i] - (double)dequant[i];
        double abs_diff = fabs(diff);
        sum_sq_err += diff * diff;
        sum_abs_err += abs_diff;
        if (abs_diff > max_abs) max_abs = abs_diff;
        sum_orig_sq += (double)f32_data[i] * f32_data[i];
        sum_dequant_sq += (double)dequant[i] * dequant[i];
        sum_dot += (double)f32_data[i] * dequant[i];
        sum_orig += f32_data[i];
        sum_dequant += dequant[i];
    }

    out->mse = sum_sq_err / (double)n;
    out->rmse = sqrt(out->mse);
    out->max_abs_error = max_abs;
    out->mean_abs_error = sum_abs_err / (double)n;

    /* Cosine similarity. */
    double norm_orig = sqrt(sum_orig_sq);
    double norm_dequant = sqrt(sum_dequant_sq);
    if (norm_orig > 0 && norm_dequant > 0)
        out->cos_sim = sum_dot / (norm_orig * norm_dequant);
    else
        out->cos_sim = 0.0;

    /* Signal-to-noise ratio (dB). */
    if (out->mse > 0.0)
        out->signal_noise_ratio = 10.0 * log10(sum_orig_sq / (double)n / out->mse);
    else
        out->signal_noise_ratio = 999.0;

    /* Relative error. */
    double mean_abs_orig = sum_abs_err / (double)n;
    if (mean_abs_orig > 0)
        out->relative_error = out->mean_abs_error / mean_abs_orig;
    else
        out->relative_error = 0.0;

    /* Size comparison. */
    out->original_bytes = n * sizeof(float);
    OcQuantBlockLayout bl = oc_quant_block_size(qtype);
    size_t block_size = bl.elements_per_block;
    size_t n_blocks = (n + block_size - 1) / block_size;
    out->quantized_bytes = oc_quantized_size(qtype, n);
    if (out->quantized_bytes > 0)
        out->compression_ratio = (double)out->original_bytes / (double)out->quantized_bytes;
    else
        out->compression_ratio = 1.0;

    free(dequant);
    return OC_OK;
}

/* ─── Error distribution ──────────────────────────────────────────────── */

OcError oc_quant_error_distribution(const float *f32_data,
                                     const uint8_t *quant_data,
                                     OcGgufQuantizationType qtype,
                                     size_t n, OcQuantErrorDist *out)
{
    if (!f32_data || !quant_data || !out || n == 0)
        return OC_ERR_INVALID_ARG;

    memset(out, 0, sizeof(*out));
    out->n_elements = n;

    /* Dequantize. */
    float *dequant = malloc(n * sizeof(float));
    if (!dequant) return OC_ERR_OOM;
    OcError e = oc_quant_dequant_row(qtype, quant_data, oc_quantized_size(qtype, n), dequant, n);
    if (e != OC_OK) { free(dequant); return e; }

    /* Compute absolute errors. */
    double *errors = malloc(n * sizeof(double));
    if (!errors) { free(dequant); return OC_ERR_OOM; }

    double sum_err = 0.0;
    double max_err = 0.0;
    for (size_t i = 0; i < n; i++) {
        errors[i] = fabs((double)f32_data[i] - (double)dequant[i]);
        sum_err += errors[i];
        if (errors[i] > max_err) max_err = errors[i];
    }

    /* Sort for percentile computation. */
    qsort(errors, n, sizeof(double), cmp_double);

    out->mean_error = sum_err / (double)n;
    out->max_error = max_err;
    out->p50_error = percentile(errors, n, 0.50);
    out->p90_error = percentile(errors, n, 0.90);
    out->p99_error = percentile(errors, n, 0.99);
    out->p999_error = percentile(errors, n, 0.999);

    /* Count outliers (> 3 * stddev). */
    double variance = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = errors[i] - out->mean_error;
        variance += d * d;
    }
    double stddev = sqrt(variance / (double)n);
    double threshold = 3.0 * stddev;
    size_t outliers = 0;
    for (size_t i = 0; i < n; i++) {
        if (errors[i] > threshold) outliers++;
    }
    out->n_outliers = outliers;

    free(errors);
    free(dequant);
    return OC_OK;
}

/* ─── Formatting ──────────────────────────────────────────────────────── */

size_t oc_quant_metrics_format(const OcQuantMetrics *m, char *buf, size_t cap)
{
    int n = snprintf(buf, cap,
        "{\"mse\":%.6e,\"rmse\":%.6e,\"max_abs_error\":%.6e,"
        "\"mean_abs_error\":%.6e,\"cos_sim\":%.6f,"
        "\"snr_db\":%.2f,\"relative_error\":%.6f,"
        "\"n_elements\":%zu,\"original_bytes\":%zu,"
        "\"quantized_bytes\":%zu,\"compression_ratio\":%.2f}",
        m->mse, m->rmse, m->max_abs_error, m->mean_abs_error,
        m->cos_sim, m->signal_noise_ratio, m->relative_error,
        m->n_elements, m->original_bytes, m->quantized_bytes,
        m->compression_ratio);
    if (n < 0 || (size_t)n >= cap) return 0;
    return (size_t)n;
}

size_t oc_quant_metrics_table(const OcQuantMetrics *m, char *buf, size_t cap)
{
    int n = snprintf(buf, cap,
        "┌────────────────────────────────────────┐\n"
        "│ Quantization Quality Analysis         │\n"
        "├────────────────────────────────────────┤\n"
        "│ MSE:             %12.6e          │\n"
        "│ RMSE:            %12.6e          │\n"
        "│ Max Abs Error:   %12.6e          │\n"
        "│ Mean Abs Error:  %12.6e          │\n"
        "│ Cosine Sim:      %12.6f          │\n"
        "│ SNR (dB):        %12.2f          │\n"
        "│ Relative Error:  %12.6f          │\n"
        "│ Elements:        %12zu          │\n"
        "│ Original:        %12zu bytes    │\n"
        "│ Quantized:       %12zu bytes    │\n"
        "│ Compression:     %12.2fx         │\n"
        "└────────────────────────────────────────┘\n",
        m->mse, m->rmse, m->max_abs_error, m->mean_abs_error,
        m->cos_sim, m->signal_noise_ratio, m->relative_error,
        m->n_elements, m->original_bytes, m->quantized_bytes,
        m->compression_ratio);
    if (n < 0 || (size_t)n >= cap) return 0;
    return (size_t)n;
}

/* ─── Quant type info ──────────────────────────────────────────────────── */

const char *oc_quant_analysis_type_name(OcGgufQuantizationType t)
{
    switch (t) {
    case OC_QUANT_F32:  return "F32";
    case OC_QUANT_F16:  return "F16";
    case OC_QUANT_Q4_0: return "Q4_0";
    case OC_QUANT_Q4_1: return "Q4_1";
    case OC_QUANT_Q5_0: return "Q5_0";
    case OC_QUANT_Q5_1: return "Q5_1";
    case OC_QUANT_Q8_0: return "Q8_0";
    case OC_QUANT_Q2_K: return "Q2_K";
    case OC_QUANT_Q3_K_M: return "Q3_K_M";
    case OC_QUANT_Q4_K_M: return "Q4_K_M";
    case OC_QUANT_Q5_K_M: return "Q5_K_M";
    case OC_QUANT_Q6_K: return "Q6_K";
    default: return "UNKNOWN";
    }
}

float oc_quant_bits_per_element(OcGgufQuantizationType t)
{
    switch (t) {
    case OC_QUANT_F32:  return 32.0f;
    case OC_QUANT_F16:  return 16.0f;
    case OC_QUANT_Q4_0: return 4.5f;   /* 4 bits + f16 scale per 32 */
    case OC_QUANT_Q4_1: return 5.0f;   /* 4 bits + f16 d + f16 m */
    case OC_QUANT_Q5_0: return 5.5f;
    case OC_QUANT_Q5_1: return 6.0f;
    case OC_QUANT_Q8_0: return 8.5f;
    case OC_QUANT_Q2_K: return 2.5625f;
    case OC_QUANT_Q3_K_M: return 3.4375f;
    case OC_QUANT_Q4_K_M: return 4.5f;
    case OC_QUANT_Q5_K_M: return 5.5f;
    case OC_QUANT_Q6_K: return 6.5625f;
    default: return 32.0f;
    }
}

double oc_quant_estimated_ppl_delta(OcGgufQuantizationType t)
{
    /* Estimated perplexity increase over F16 (from llama.cpp benchmarks). */
    switch (t) {
    case OC_QUANT_F32:  return 0.0;
    case OC_QUANT_F16:  return 0.0;
    case OC_QUANT_Q8_0: return 0.0;
    case OC_QUANT_Q6_K: return 0.02;
    case OC_QUANT_Q5_K_M: return 0.05;
    case OC_QUANT_Q5_1: return 0.1;
    case OC_QUANT_Q5_0: return 0.12;
    case OC_QUANT_Q4_K_M: return 0.15;
    case OC_QUANT_Q4_1: return 0.3;
    case OC_QUANT_Q4_0: return 0.35;
    case OC_QUANT_Q3_K_M: return 0.5;
    case OC_QUANT_Q2_K: return 1.0;
    default: return 0.0;
    }
}

/* ─── Recommender ─────────────────────────────────────────────────────── */

OcError oc_quant_recommend(uint64_t model_params, uint64_t available_ram,
                             OcQuantGoal goal,
                             OcQuantRecommendation *out)
{
    if (!out) return OC_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));

    /* Estimate model size for each quant type. */
    double f16_size_gb = (double)model_params * 2.0 / 1e9;
    double q4_k_size_gb = (double)model_params * 4.5 / 8.0 / 1e9;
    double q5_k_size_gb = (double)model_params * 5.5 / 8.0 / 1e9;
    double q6_k_size_gb = (double)model_params * 6.5625 / 8.0 / 1e9;
    double q8_0_size_gb = (double)model_params * 8.5 / 8.0 / 1e9;
    double q3_k_size_gb = (double)model_params * 3.4375 / 8.0 / 1e9;
    double q2_k_size_gb = (double)model_params * 2.5625 / 8.0 / 1e9;
    double ram_gb = (double)available_ram / 1e9;

    /* KV cache overhead: ~0.5-2GB depending on context. */
    double kv_overhead = f16_size_gb * 0.1; /* ~10% of model size */

    switch (goal) {
    case OC_QUANT_GOAL_SPEED:
        if (ram_gb >= q8_0_size_gb + kv_overhead) {
            out->recommended = OC_QUANT_Q8_0;
            out->alternative = OC_QUANT_Q6_K;
            out->estimated_size_gb = q8_0_size_gb;
            snprintf(out->rationale, sizeof(out->rationale),
                     "Q8_0 selected for maximum speed: negligible quality loss, "
                     "fastest dequantization, fits in available RAM");
        } else if (ram_gb >= q4_k_size_gb + kv_overhead) {
            out->recommended = OC_QUANT_Q4_K_M;
            out->alternative = OC_QUANT_Q5_K_M;
            out->estimated_size_gb = q4_k_size_gb;
            snprintf(out->rationale, sizeof(out->rationale),
                     "Q4_K selected: good speed/quality tradeoff, fits in RAM");
        } else {
            out->recommended = OC_QUANT_Q3_K_M;
            out->alternative = OC_QUANT_Q2_K;
            out->estimated_size_gb = q3_k_size_gb;
            snprintf(out->rationale, sizeof(out->rationale),
                     "Q3_K selected: fits in limited RAM, higher quality loss");
        }
        out->estimated_tok_per_sec = 100.0 / out->estimated_size_gb;
        break;

    case OC_QUANT_GOAL_QUALITY:
        if (ram_gb >= f16_size_gb + kv_overhead) {
            out->recommended = OC_QUANT_F16;
            out->alternative = OC_QUANT_Q8_0;
            out->estimated_size_gb = f16_size_gb;
            snprintf(out->rationale, sizeof(out->rationale),
                     "F16 selected for maximum quality: no quantization loss");
        } else if (ram_gb >= q6_k_size_gb + kv_overhead) {
            out->recommended = OC_QUANT_Q6_K;
            out->alternative = OC_QUANT_Q5_K_M;
            out->estimated_size_gb = q6_k_size_gb;
            snprintf(out->rationale, sizeof(out->rationale),
                     "Q6_K selected: near-lossless quality, fits in RAM");
        } else if (ram_gb >= q5_k_size_gb + kv_overhead) {
            out->recommended = OC_QUANT_Q5_K_M;
            out->alternative = OC_QUANT_Q4_K_M;
            out->estimated_size_gb = q5_k_size_gb;
            snprintf(out->rationale, sizeof(out->rationale),
                     "Q5_K selected: very good quality, reasonable size");
        } else {
            out->recommended = OC_QUANT_Q4_K_M;
            out->alternative = OC_QUANT_Q3_K_M;
            out->estimated_size_gb = q4_k_size_gb;
            snprintf(out->rationale, sizeof(out->rationale),
                     "Q4_K selected: best quality within available RAM");
        }
        out->estimated_tok_per_sec = 50.0 / out->estimated_size_gb;
        break;

    case OC_QUANT_GOAL_BALANCED:
        if (ram_gb >= q5_k_size_gb + kv_overhead) {
            out->recommended = OC_QUANT_Q5_K_M;
            out->alternative = OC_QUANT_Q4_K_M;
            out->estimated_size_gb = q5_k_size_gb;
            snprintf(out->rationale, sizeof(out->rationale),
                     "Q5_K selected: balanced speed/quality, fits comfortably in RAM");
        } else if (ram_gb >= q4_k_size_gb + kv_overhead) {
            out->recommended = OC_QUANT_Q4_K_M;
            out->alternative = OC_QUANT_Q3_K_M;
            out->estimated_size_gb = q4_k_size_gb;
            snprintf(out->rationale, sizeof(out->rationale),
                     "Q4_K selected: good balance, standard choice for most users");
        } else {
            out->recommended = OC_QUANT_Q3_K_M;
            out->alternative = OC_QUANT_Q2_K;
            out->estimated_size_gb = q3_k_size_gb;
            snprintf(out->rationale, sizeof(out->rationale),
                     "Q3_K selected: fits in limited RAM, acceptable quality");
        }
        out->estimated_tok_per_sec = 70.0 / out->estimated_size_gb;
        break;

    case OC_QUANT_GOAL_MEMORY:
        if (ram_gb >= q2_k_size_gb + kv_overhead) {
            out->recommended = OC_QUANT_Q2_K;
            out->alternative = OC_QUANT_Q3_K_M;
            out->estimated_size_gb = q2_k_size_gb;
            snprintf(out->rationale, sizeof(out->rationale),
                     "Q2_K selected: minimum memory usage, noticeable quality loss");
        } else {
            out->recommended = OC_QUANT_Q2_K;
            out->alternative = OC_QUANT_Q2_K;
            out->estimated_size_gb = q2_k_size_gb;
            snprintf(out->rationale, sizeof(out->rationale),
                     "Q2_K: model may not fit in available RAM even at lowest quant");
        }
        out->estimated_tok_per_sec = 80.0 / out->estimated_size_gb;
        break;
    }

    out->estimated_ppl_delta = oc_quant_estimated_ppl_delta(out->recommended);
    return OC_OK;
}

size_t oc_quant_recommend_format(const OcQuantRecommendation *r, char *buf, size_t cap)
{
    int n = snprintf(buf, cap,
        "{\"recommended\":\"%s\",\"alternative\":\"%s\","
        "\"estimated_ppl_delta\":%.3f,\"estimated_size_gb\":%.2f,"
        "\"estimated_tok_per_sec\":%.1f,\"rationale\":\"%s\"}",
        oc_quant_analysis_type_name(r->recommended),
        oc_quant_analysis_type_name(r->alternative),
        r->estimated_ppl_delta, r->estimated_size_gb,
        r->estimated_tok_per_sec, r->rationale);
    if (n < 0 || (size_t)n >= cap) return 0;
    return (size_t)n;
}

/* ─── Comparison table ─────────────────────────────────────────────────── */

size_t oc_quant_comparison_table(uint64_t model_params, char *buf, size_t cap)
{
    int n = snprintf(buf, cap,
        "┌──────────┬──────────┬──────────┬───────────────┐\n"
        "│ Type     │ Bits/El  │ Size (GB) │ PPL Delta     │\n"
        "├──────────┼──────────┼──────────┼───────────────┤\n");

    if (n < 0 || (size_t)n >= cap) return 0;
    size_t off = (size_t)n;

    OcGgufQuantizationType types[] = {
        OC_QUANT_F32, OC_QUANT_F16,
        OC_QUANT_Q8_0, OC_QUANT_Q6_K, OC_QUANT_Q5_K_M,
        OC_QUANT_Q4_K_M, OC_QUANT_Q3_K_M, OC_QUANT_Q2_K,
    };

    for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); i++) {
        float bpe = oc_quant_bits_per_element(types[i]);
        double size_gb = (double)model_params * bpe / 8.0 / 1e9;
        double ppl = oc_quant_estimated_ppl_delta(types[i]);
        int w = snprintf(buf + off, cap - off,
            "│ %-8s │ %8.2f │ %8.2f │ %13.3f │\n",
            oc_quant_analysis_type_name(types[i]), bpe, size_gb, ppl);
        if (w < 0 || (size_t)w >= cap - off) return 0;
        off += (size_t)w;
    }

    int w2 = snprintf(buf + off, cap - off,
        "└──────────┴──────────┴──────────┴───────────────┘\n");
    if (w2 < 0 || (size_t)w2 >= cap - off) return 0;
    off += (size_t)w2;

    return off;
}
