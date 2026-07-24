/*
 * diffusion.c — Diffusion model scheduler implementation.
 */
#include "oxidize/diffusion.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static float lcg_next(uint32_t *state)
{
    *state = *state * 1103515245u + 12345u;
    return (float)((*state >> 8) & 0xFFFFFF) / (float)0x1000000;
}

static float gaussian_rand(uint32_t *state)
{
    float u1 = lcg_next(state);
    float u2 = lcg_next(state);
    if (u1 < 1e-6f) u1 = 1e-6f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

OcError oc_diff_config_init(OcDiffConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    memset(cfg, 0, sizeof(*cfg));
    cfg->type = OC_DIFF_DDIM;
    cfg->num_train_timesteps = 1000;
    cfg->num_inference_steps = 50;
    cfg->beta_start = 0.0001f;
    cfg->beta_end = 0.02f;
    cfg->beta_schedule = 0; /* linear */
    cfg->guidance_scale = 7.5f;
    cfg->seed = 42;
    return OC_OK;
}

OcError oc_diff_scheduler_init(OcDiffScheduler *sched, const OcDiffConfig *cfg)
{
    if (!sched) return OC_ERR_INVALID_ARG;
    memset(sched, 0, sizeof(*sched));
    if (cfg) sched->config = *cfg;
    else oc_diff_config_init(&sched->config);

    uint32_t n = sched->config.num_train_timesteps;
    sched->alphas_cumprod = malloc(n * sizeof(float));
    if (!sched->alphas_cumprod) return OC_ERR_OOM;
    sched->n_alphas = n;

    /* Compute alphas_cumprod. */
    float prev = 1.0f;
    for (uint32_t i = 0; i < n; i++) {
        float beta;
        if (sched->config.beta_schedule == 1) {
            /* cosine. */
            float s = 0.008f;
            float t = (float)i / (float)n;
            float f = cosf((t + s) / (1.0f + s) * 3.14159265f * 0.5f);
            beta = fminf(1.0f - f * f, 0.999f);
        } else if (sched->config.beta_schedule == 2) {
            /* scaled_linear. */
            beta = sched->config.beta_start +
                   (sched->config.beta_end - sched->config.beta_start) *
                   ((float)i / (float)n) * ((float)i / (float)n);
        } else {
            /* linear. */
            beta = sched->config.beta_start +
                   (sched->config.beta_end - sched->config.beta_start) *
                   ((float)i / (float)n);
        }
        float alpha = 1.0f - beta;
        prev = prev * alpha;
        sched->alphas_cumprod[i] = prev;
    }

    sched->sigmas = NULL;
    sched->n_sigmas = 0;
    sched->current_step = 0;
    sched->initialized = true;
    return OC_OK;
}

OcError oc_diff_scheduler_set_timesteps(OcDiffScheduler *sched, uint32_t steps)
{
    if (!sched || !sched->initialized) return OC_ERR_INVALID_ARG;
    if (steps == 0 || steps > OC_DIFFUSION_MAX_STEPS) return OC_ERR_INVALID_ARG;

    free(sched->sigmas);
    sched->sigmas = malloc(steps * sizeof(float));
    if (!sched->sigmas) return OC_ERR_OOM;
    sched->n_sigmas = steps;
    sched->current_step = 0;

    /* Compute sigma schedule. */
    for (uint32_t i = 0; i < steps; i++) {
        uint32_t train_step = (uint32_t)((float)(steps - 1 - i) / (float)(steps - 1) *
                              (float)(sched->n_alphas - 1));
        if (train_step >= sched->n_alphas) train_step = sched->n_alphas - 1;
        float alpha = sched->alphas_cumprod[train_step];
        sched->sigmas[i] = sqrtf((1.0f - alpha) / alpha);
    }
    return OC_OK;
}

OcError oc_diff_scheduler_step(OcDiffScheduler *sched, OcDiffState *state,
                              const float *model_output)
{
    if (!sched || !state || !model_output) return OC_ERR_INVALID_ARG;
    if (sched->current_step >= sched->n_sigmas) return OC_ERR_INVALID_ARG;

    float sigma = sched->sigmas[sched->current_step];
    float next_sigma = (sched->current_step + 1 < sched->n_sigmas)
                        ? sched->sigmas[sched->current_step + 1] : 0.0f;

    /* DDIM step: x_{t-1} = (x_t - sigma * model_output) / sqrt(1 + sigma^2) * sqrt(1 + next_sigma^2) + next_sigma * noise */
    float scale = sqrtf(1.0f + next_sigma * next_sigma) / sqrtf(1.0f + sigma * sigma);

    for (size_t i = 0; i < (size_t)state->latent_dim * state->batch_size; i++) {
        float x = state->latents[i];
        float m = model_output[i];
        state->latents[i] = (x - sigma * m) * scale;
        /* Add noise for next step (except final). */
        if (next_sigma > 0.0f) {
            uint32_t seed = sched->config.seed + sched->current_step * 1000 + i;
            float noise = gaussian_rand(&seed);
            state->latents[i] += next_sigma * noise;
        }
    }
    state->current_sigma = next_sigma;
    sched->current_step++;
    return OC_OK;
}

float oc_diff_scheduler_get_sigma(const OcDiffScheduler *sched, uint32_t step)
{
    if (!sched || step >= sched->n_sigmas) return 0.0f;
    return sched->sigmas[step];
}

uint32_t oc_diff_scheduler_n_steps(const OcDiffScheduler *sched)
{
    return sched ? sched->n_sigmas : 0;
}

uint32_t oc_diff_scheduler_current_step(const OcDiffScheduler *sched)
{
    return sched ? sched->current_step : 0;
}

const char *oc_diff_scheduler_name(OcDiffSchedulerType type)
{
    switch (type) {
    case OC_DIFF_DDPM:       return "ddpm";
    case OC_DIFF_DDIM:      return "ddim";
    case OC_DIFF_DPM_2M:    return "dpm_2m";
    case OC_DIFF_EULER_A:   return "euler_a";
    case OC_DIFF_FLOW_MATCH: return "flow_match";
    default:                 return "unknown";
    }
}

OcError oc_diff_state_init(OcDiffState *state, uint32_t latent_dim,
                          uint32_t batch_size, uint32_t seed)
{
    if (!state || latent_dim == 0) return OC_ERR_INVALID_ARG;
    memset(state, 0, sizeof(*state));
    state->latent_dim = latent_dim;
    state->batch_size = batch_size > 0 ? batch_size : 1;
    size_t n_latents = (size_t)state->latent_dim * (size_t)state->batch_size;
    state->latents = calloc(n_latents, sizeof(float));
    if (!state->latents) return OC_ERR_OOM;
    state->current_sigma = 0.0f;

    /* Initialize with random noise. */
    uint32_t rng = seed;
    for (size_t i = 0; i < n_latents; i++)
        state->latents[i] = gaussian_rand(&rng);

    return OC_OK;
}

OcError oc_diff_state_add_noise(OcDiffState *state, float sigma)
{
    if (!state || !state->latents) return OC_ERR_INVALID_ARG;
    uint32_t rng = 12345;
    for (uint32_t i = 0; i < state->latent_dim * state->batch_size; i++)
        state->latents[i] += sigma * gaussian_rand(&rng);
    state->current_sigma = sigma;
    return OC_OK;
}

OcError oc_diff_state_free(OcDiffState *state)
{
    if (!state) return OC_ERR_INVALID_ARG;
    free(state->latents);
    memset(state, 0, sizeof(*state));
    return OC_OK;
}

void oc_diff_scheduler_free(OcDiffScheduler *sched)
{
    if (!sched) return;
    free(sched->alphas_cumprod);
    free(sched->sigmas);
    memset(sched, 0, sizeof(*sched));
}
