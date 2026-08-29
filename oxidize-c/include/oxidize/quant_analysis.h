/* quant_analysis.h — Quantization quality analysis and comparison. */
#ifndef OXIDIZE_QUANT_ANALYSIS_H
#define OXIDIZE_QUANT_ANALYSIS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/quant.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef struct OcQuantMetrics {
    double mse;             /* mean squared error (f32 vs dequantized)     */
    double rmse;            /* root mean squared error                      */
    double max_abs_error;   /* maximum absolute error                       */
    double mean_abs_error;  /* mean absolute error                          */
    double cos_sim;          /* cosine similarity (1.0 = identical)         */
    double signal_noise_ratio; /* SNR in dB                                  */
    double relative_error;   /* mean relative error                         */
    size_t n_elements;       /* number of elements compared                 */
    size_t original_bytes;   /* size of f32 weights                         */
    size_t quantized_bytes;  /* size of quantized weights                    */
    double compression_ratio; /* original / quantized                        */
} OcQuantMetrics;

/* Compare f32 weights to dequantized `quant_data`, which must be packed for `qtype` (size from oc_quantized_size(qtype, n)). Returns OC_ERR_INVALID_ARG on NULL args or n==0. */
OcError oc_quant_analyze(const float *f32_data, const uint8_t *quant_data,
                          OcGgufQuantizationType qtype, size_t n,
                          OcQuantMetrics *out);

/* Format metrics as JSON. */
size_t oc_quant_metrics_format(const OcQuantMetrics *m, char *buf, size_t cap);

/* Format as human-readable table. */
size_t oc_quant_metrics_table(const OcQuantMetrics *m, char *buf, size_t cap);


typedef struct OcQuantErrorDist {
    double p50_error;    /* median absolute error                         */
    double p90_error;    /* 90th percentile absolute error                */
    double p99_error;    /* 99th percentile absolute error                */
    double p999_error;   /* 99.9th percentile absolute error              */
    double max_error;     /* maximum absolute error                        */
    double mean_error;    /* mean absolute error                           */
    size_t n_outliers;    /* elements with error > 3*stddev                */
    size_t n_elements;
} OcQuantErrorDist;

/* Compute error distribution (requires sorting, O(n log n)). */
OcError oc_quant_error_distribution(const float *f32_data,
                                     const uint8_t *quant_data,
                                     OcGgufQuantizationType qtype,
                                     size_t n, OcQuantErrorDist *out);


typedef struct OcLayerQuantReport {
    char     layer_name[128];
    char     tensor_name[128];
    OcGgufQuantizationType qtype;
    OcQuantMetrics metrics;
    OcQuantErrorDist dist;
} OcLayerQuantReport;

typedef struct OcModelQuantReport {
    OcLayerQuantReport *layers;
    size_t n_layers;
    double avg_mse;
    double avg_cos_sim;
    double worst_cos_sim;
    char   worst_layer[128];
    double total_original_bytes;
    double total_quantized_bytes;
    double overall_compression;
} OcModelQuantReport;


typedef enum {
    OC_QUANT_GOAL_SPEED     = 0, /* prioritize inference speed              */
    OC_QUANT_GOAL_QUALITY   = 1, /* prioritize output quality               */
    OC_QUANT_GOAL_BALANCED  = 2, /* balance speed and quality               */
    OC_QUANT_GOAL_MEMORY    = 3, /* minimize memory usage                     */
} OcQuantGoal;

typedef struct OcQuantRecommendation {
    OcGgufQuantizationType recommended;
    OcGgufQuantizationType alternative;
    double estimated_ppl_delta;  /* expected perplexity increase            */
    double estimated_size_gb;   /* estimated model size in GB                */
    double estimated_tok_per_sec; /* estimated decode speed                  */
    char   rationale[256];      /* human-readable explanation                */
} OcQuantRecommendation;

/* Recommend a quantization type based on model size, available RAM, and goal. */
OcError oc_quant_recommend(uint64_t model_params, uint64_t available_ram,
                             OcQuantGoal goal,
                             OcQuantRecommendation *out);

/* Format recommendation as JSON. */
size_t oc_quant_recommend_format(const OcQuantRecommendation *r,
                                  char *buf, size_t cap);


/* Generate a comparison table of all quant types for a given model size. */
size_t oc_quant_comparison_table(uint64_t model_params, char *buf, size_t cap);

/* Get the name of a quantization type. */
const char *oc_quant_analysis_type_name(OcGgufQuantizationType t);

/* Get the bits per element for a quant type. */
float oc_quant_bits_per_element(OcGgufQuantizationType t);

/* Get the estimated perplexity delta for a quant type. */
double oc_quant_estimated_ppl_delta(OcGgufQuantizationType t);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_QUANT_ANALYSIS_H */
