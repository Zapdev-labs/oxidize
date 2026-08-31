/* diffusion_gemma.h — Gemma diffusion model support. */
#ifndef OXIDIZE_DIFFUSION_GEMMA_H
#define OXIDIZE_DIFFUSION_GEMMA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OC_DIFF_GEMMA_DDIM = 0,
    OC_DIFF_GEMMA_DPM2M = 1,
    OC_DIFF_GEMMA_EULER = 2,
    OC_DIFF_GEMMA_FLOW_MATCH = 3,
} OcDiffGemmaSampler;

typedef struct {
    uint32_t n_layers;
    uint32_t hidden_dim;
    uint32_t vocab_size;
    uint32_t n_diffusion_steps;
    float    sigma_min;
    float    sigma_max;
    float    rho;
    OcDiffGemmaSampler sampler;
    bool     use_classifier_free_guidance;
    float    guidance_scale;
} OcDiffGemmaConfig;

typedef struct {
    OcDiffGemmaConfig config;
    float *embedding;    /* [vocab_size, hidden_dim] */
    float *output;       /* [vocab_size, hidden_dim] */
    /* Per-layer weight pointers (allocated, zeroed). */
    float *attn_norm;    /* [n_layers * hidden_dim] */
    float *wq;           /* [n_layers * hidden_dim * hidden_dim] */
    float *wk;           /* [n_layers * hidden_dim * hidden_dim] */
    float *wv;           /* [n_layers * hidden_dim * hidden_dim] */
    float *wo;           /* [n_layers * hidden_dim * hidden_dim] */
    float *ffn_norm;     /* [n_layers * hidden_dim] */
    float *w_gate;       /* [n_layers * hidden_dim * hidden_dim] */
    float *w_up;         /* [n_layers * hidden_dim * hidden_dim] */
    float *w_down;       /* [n_layers * hidden_dim * hidden_dim] */
    float *final_norm;   /* [hidden_dim] */
    bool initialized;
} OcDiffGemmaModel;

OcError oc_diff_gemma_config_init(OcDiffGemmaConfig *cfg);
OcError oc_diff_gemma_model_init(OcDiffGemmaModel *model, const OcDiffGemmaConfig *cfg);
OcError oc_diff_gemma_forward(OcDiffGemmaModel *model, uint32_t token, float sigma, float *logits);
OcError oc_diff_gemma_sample(OcDiffGemmaModel *model, float *logits, size_t n, uint32_t step);
OcError oc_diff_gemma_denoise(OcDiffGemmaModel *model, float *tokens, size_t n,
                             float sigma_from, float sigma_to);
const char *oc_diff_gemma_sampler_name(OcDiffGemmaSampler s);
void oc_diff_gemma_free(OcDiffGemmaModel *model);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_DIFFUSION_GEMMA_H */
