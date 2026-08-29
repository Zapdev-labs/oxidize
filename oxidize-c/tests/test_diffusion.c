/* test_diffusion.c — Diffusion scheduler tests. */
#include "framework.h"
#include "oxidize/diffusion.h"
#include <math.h>
#include <string.h>

Test(diff, config_init)
{
    OcDiffConfig cfg;
    cr_assert_eq(oc_diff_config_init(&cfg), OC_OK);
    cr_assert_eq(cfg.type, OC_DIFF_DDIM);
    cr_assert_eq(cfg.num_train_timesteps, 1000);
    cr_assert_eq(cfg.num_inference_steps, 50);
    cr_assert_float_eq(cfg.beta_start, 0.0001f, 1e-6f);
    cr_assert_float_eq(cfg.beta_end, 0.02f, 1e-6f);
    cr_assert_float_eq(cfg.guidance_scale, 7.5f, 1e-6f);
}

OC_TEST_NULL_SAFE(diff, config_init_null,
        cr_assert_neq(oc_diff_config_init(NULL), OC_OK);)

Test(diff, scheduler_init)
{
    OcDiffConfig cfg;
    oc_diff_config_init(&cfg);
    OcDiffScheduler sched;
    cr_assert_eq(oc_diff_scheduler_init(&sched, &cfg), OC_OK);
    cr_assert(sched.initialized);
    cr_assert_eq(sched.n_alphas, 1000);
    cr_assert(sched.alphas_cumprod[0] < 1.0f);
    cr_assert(sched.alphas_cumprod[999] < sched.alphas_cumprod[0]);
    oc_diff_scheduler_free(&sched);
}

OC_TEST_NULL_SAFE(diff, scheduler_init_null,
        cr_assert_neq(oc_diff_scheduler_init(NULL, NULL), OC_OK);)

Test(diff, scheduler_init_default_config)
{
    OcDiffScheduler sched;
    cr_assert_eq(oc_diff_scheduler_init(&sched, NULL), OC_OK);
    cr_assert(sched.initialized);
    oc_diff_scheduler_free(&sched);
}

Test(diff, set_timesteps)
{
    OcDiffScheduler sched;
    oc_diff_scheduler_init(&sched, NULL);
    cr_assert_eq(oc_diff_scheduler_set_timesteps(&sched, 20), OC_OK);
    cr_assert_eq(sched.n_sigmas, 20);
    cr_assert(sched.sigmas[0] > sched.sigmas[19]);
    oc_diff_scheduler_free(&sched);
}

Test(diff, set_timesteps_invalid)
{
    OcDiffScheduler sched;
    oc_diff_scheduler_init(&sched, NULL);
    cr_assert_neq(oc_diff_scheduler_set_timesteps(&sched, 0), OC_OK);
    oc_diff_scheduler_free(&sched);
}

Test(diff, scheduler_name)
{
    cr_assert_str_eq(oc_diff_scheduler_name(OC_DIFF_DDPM), "ddpm");
    cr_assert_str_eq(oc_diff_scheduler_name(OC_DIFF_DDIM), "ddim");
    cr_assert_str_eq(oc_diff_scheduler_name(OC_DIFF_DPM_2M), "dpm_2m");
    cr_assert_str_eq(oc_diff_scheduler_name(OC_DIFF_EULER_A), "euler_a");
    cr_assert_str_eq(oc_diff_scheduler_name(OC_DIFF_FLOW_MATCH), "flow_match");
}

Test(diff, state_init)
{
    OcDiffState state;
    cr_assert_eq(oc_diff_state_init(&state, 64, 1, 42), OC_OK);
    cr_assert_eq(state.latent_dim, 64);
    cr_assert_eq(state.batch_size, 1);
    bool nonzero = false;
    for (uint32_t i = 0; i < 64; i++) {
        if (state.latents[i] != 0.0f) { nonzero = true; break; }
    }
    cr_assert(nonzero, "latents should have random values");
    oc_diff_state_free(&state);
}

OC_TEST_NULL_SAFE(diff, state_init_null,
        cr_assert_neq(oc_diff_state_init(NULL, 64, 1, 0), OC_OK);)

Test(diff, state_add_noise)
{
    OcDiffState state;
    oc_diff_state_init(&state, 64, 1, 42);
    float original = state.latents[0];
    oc_diff_state_add_noise(&state, 1.0f);
    cr_assert_float_neq(state.latents[0], original, 0.01f);
    cr_assert_float_eq(state.current_sigma, 1.0f, 1e-6f);
    oc_diff_state_free(&state);
}

Test(diff, scheduler_step)
{
    OcDiffScheduler sched;
    oc_diff_scheduler_init(&sched, NULL);
    oc_diff_scheduler_set_timesteps(&sched, 5);
    OcDiffState state;
    oc_diff_state_init(&state, 64, 1, 42);
    float model_output[64] = {0};
    cr_assert_eq(oc_diff_scheduler_step(&sched, &state, model_output), OC_OK);
    cr_assert_eq(sched.current_step, 1);
    oc_diff_state_free(&state);
    oc_diff_scheduler_free(&sched);
}

Test(diff, scheduler_get_sigma)
{
    OcDiffScheduler sched;
    oc_diff_scheduler_init(&sched, NULL);
    oc_diff_scheduler_set_timesteps(&sched, 10);
    float s = oc_diff_scheduler_get_sigma(&sched, 0);
    cr_assert(s > 0.0f);
    cr_assert_eq(oc_diff_scheduler_get_sigma(&sched, 999), 0.0f);
    oc_diff_scheduler_free(&sched);
}

Test(diff, scheduler_n_steps)
{
    OcDiffScheduler sched;
    oc_diff_scheduler_init(&sched, NULL);
    cr_assert_eq(oc_diff_scheduler_n_steps(&sched), 0);
    oc_diff_scheduler_set_timesteps(&sched, 25);
    cr_assert_eq(oc_diff_scheduler_n_steps(&sched), 25);
    oc_diff_scheduler_free(&sched);
}

Test(diff, scheduler_current_step)
{
    OcDiffScheduler sched;
    oc_diff_scheduler_init(&sched, NULL);
    cr_assert_eq(oc_diff_scheduler_current_step(&sched), 0);
    oc_diff_scheduler_free(&sched);
}

Test(diff, scheduler_full_loop)
{
    OcDiffScheduler sched;
    oc_diff_scheduler_init(&sched, NULL);
    oc_diff_scheduler_set_timesteps(&sched, 10);
    OcDiffState state;
    oc_diff_state_init(&state, 32, 1, 42);
    float model_output[32] = {0};
    for (uint32_t i = 0; i < 10; i++) {
        cr_assert_eq(oc_diff_scheduler_step(&sched, &state, model_output), OC_OK);
    }
    cr_assert_eq(sched.current_step, 10);
    oc_diff_state_free(&state);
    oc_diff_scheduler_free(&sched);
}

Test(diff, cosine_schedule)
{
    OcDiffConfig cfg;
    oc_diff_config_init(&cfg);
    cfg.beta_schedule = 1; /* cosine */
    OcDiffScheduler sched;
    oc_diff_scheduler_init(&sched, &cfg);
    cr_assert(sched.alphas_cumprod[0] > 0.0f);
    oc_diff_scheduler_free(&sched);
}

Test(diff, state_free)
{
    OcDiffState state;
    oc_diff_state_init(&state, 32, 1, 0);
    cr_assert_eq(oc_diff_state_free(&state), OC_OK);
    cr_assert_eq(state.latent_dim, 0);
}
