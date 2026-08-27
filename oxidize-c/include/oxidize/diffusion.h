/*
 * diffusion.h — Diffusion model support for text-to-image generation.
 *
 * Provides a simple diffusion scheduler and denoising loop for
 * Gemma-based diffusion models. Port from oxidize-core/src/model/diffusion_gemma.rs.
 */
#ifndef OXIDIZE_DIFFUSION_H
#define OXIDIZE_DIFFUSION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_DIFFUSION_MAX_STEPS 1000
#define OC_DIFFUSION_MAX_LATENT_DIM 4096

typedef enum {
    OC_DIFF_DDPM     = 0,
    OC_DIFF_DDIM     = 1,
    OC_DIFF_DPM_2M   = 2,
    OC_DIFF_EULER_A  = 3,
    OC_DIFF_FLOW_MATCH = 4,
} OcDiffSchedulerType;

typedef struct {
    OcDiffSchedulerType type;
    uint32_t num_train_timesteps;
    uint32_t num_inference_steps;
    float beta_start;
    float beta_end;
    float beta_schedule;  /* 0=linear, 1=cosine, 2=scaled_linear */
    float guidance_scale;
    uint32_t seed;
} OcDiffConfig;

typedef struct {
    OcDiffConfig config;
    float *alphas_cumprod;
    uint32_t n_alphas;
    float *sigmas;
    uint32_t n_sigmas;
    uint32_t current_step;
    bool initialized;
} OcDiffScheduler;

typedef struct {
    float *latents;
    uint32_t latent_dim;
    uint32_t batch_size;
    float current_sigma;
} OcDiffState;

OcError oc_diff_config_init(OcDiffConfig *cfg);
OcError oc_diff_scheduler_init(OcDiffScheduler *sched, const OcDiffConfig *cfg);
OcError oc_diff_scheduler_set_timesteps(OcDiffScheduler *sched, uint32_t steps);
OcError oc_diff_scheduler_step(OcDiffScheduler *sched, OcDiffState *state,
                              const float *model_output);
float oc_diff_scheduler_get_sigma(const OcDiffScheduler *sched, uint32_t step);
uint32_t oc_diff_scheduler_n_steps(const OcDiffScheduler *sched);
uint32_t oc_diff_scheduler_current_step(const OcDiffScheduler *sched);
const char *oc_diff_scheduler_name(OcDiffSchedulerType type);
OcError oc_diff_state_init(OcDiffState *state, uint32_t latent_dim, uint32_t batch_size, uint32_t seed);
OcError oc_diff_state_add_noise(OcDiffState *state, float sigma);
OcError oc_diff_state_free(OcDiffState *state);
void oc_diff_scheduler_free(OcDiffScheduler *sched);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_DIFFUSION_H */
