/*
 * test_sampling.c — sampler tests.
 *
 * Asserts:
 *   - greedy = argmax (VAL-SAMP-001)
 *   - temperature=0 falls back to greedy (VAL-SAMP-002)
 *   - repeat-penalty reduces the probability of recent tokens (VAL-SAMP-003)
 *   - top-k restricts sampling to the K highest logits (VAL-SAMP-004)
 *   - top-p restricts to the nucleus (VAL-SAMP-005)
 *   - deterministic given seed (VAL-SAMP-006)
 *
 * Stochastic assertions are statistical: with a fixed seed the result is
 * reproducible; we verify the sampled token is within the allowed candidate
 * set rather than asserting an exact token (which would be brittle to RNG
 * ordering changes).
 */
#include <criterion/criterion.h>

#include "oxidize/sampling.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

Test(sampling, greedy_returns_argmax)
{
    float logits[] = {0.1f, 3.0f, -1.0f, 2.0f};
    OcSamplerConfig cfg = OC_SAMPLER_DEFAULT;
    uint32_t t = oc_sample(logits, 4, &cfg, NULL, 0);
    cr_assert_eq(t, 1u, "argmax of [0.1,3,-1,2] is index 1");
}

Test(sampling, temperature_zero_is_greedy)
{
    float logits[] = {1.0f, 5.0f, 2.0f};
    OcSamplerConfig cfg = { OC_SAMPLER_TEMPERATURE, 0.0f, 0u, 1.0f, 1.0f, 0ull };
    uint32_t t = oc_sample(logits, 3, &cfg, NULL, 0);
    cr_assert_eq(t, 1u, "T=0 must fall back to greedy/argmax");
}

Test(sampling, null_config_is_greedy)
{
    float logits[] = {-5.0f, 10.0f, 0.0f};
    cr_assert_eq(oc_sample(logits, 3, NULL, NULL, 0), 1u);
}

Test(sampling, argmax_handles_ties_deterministically)
{
    /* Two equal maxima — argmax must pick the FIRST (lowest index). */
    float logits[] = {5.0f, 5.0f, 5.0f};
    cr_assert_eq(oc_argmax(logits, 3), 0u);
}

Test(sampling, repeat_penalty_lowers_recent_token_prob)
{
    /* Token 1 has the highest logit, but it's the recent token. With a
     * penalty > 1, token 1's logit is divided, so under greedy sampling the
     * argmax may flip to token 2. Use a strong penalty. */
    float logits[] = {0.0f, 4.0f, 3.9f};
    uint32_t recent[] = {1};
    OcSamplerConfig cfg = OC_SAMPLER_DEFAULT;
    /* No penalty: argmax = 1. */
    cr_assert_eq(oc_sample(logits, 3, &cfg, NULL, 0), 1u);
    /* With penalty 10: logit[1] = 0.4 < logit[2] = 3.9 → argmax flips to 2. */
    cfg.repeat_penalty = 10.0f;
    uint32_t t = oc_sample(logits, 3, &cfg, recent, 1);
    cr_assert_eq(t, 2u, "penalty must lower token 1 below token 2");
}

Test(sampling, repeat_penalty_noop_when_one)
{
    float logits[] = {1.0f, 2.0f, 3.0f};
    float copy[3];
    memcpy(copy, logits, sizeof(logits));
    oc_apply_repeat_penalty(logits, 3, NULL, 0, 1.0f);
    cr_assert_arr_eq(logits, copy, sizeof(logits), "penalty=1 is identity");
}

Test(sampling, top_k_restricts_to_k_candidates)
{
    /* logits: token 0 = 100 (dominant), others tiny. top_k=2 keeps {0,1}.
     * With seed fixed, the sample must be in {0,1}. */
    float logits[] = {100.0f, 1.0f, 0.5f, 0.1f, 0.0f};
    OcSamplerConfig cfg = { OC_SAMPLER_TOP_K, 1.0f, 2u, 1.0f, 1.0f, 42ull };
    /* Run many seeds; every result must be 0 or 1 (the top-2). */
    for (uint64_t s = 1; s < 50; s++) {
        cfg.seed = s;
        uint32_t t = oc_sample(logits, 5, &cfg, NULL, 0);
        cr_assert(t == 0u || t == 1u, "top_k=2 sampled token %u (seed %llu)",
                  t, (unsigned long long)s);
    }
}

Test(sampling, top_k_one_is_argmax)
{
    float logits[] = {0.0f, 9.0f, 1.0f};
    OcSamplerConfig cfg = { OC_SAMPLER_TOP_K, 1.0f, 1u, 1.0f, 1.0f, 7ull };
    cr_assert_eq(oc_sample(logits, 3, &cfg, NULL, 0), 1u);
}

Test(sampling, top_p_nucleus_dominant_token)
{
    /* One token with overwhelming probability → nucleus is just {0}. */
    float logits[] = {100.0f, 0.0f, 0.0f, 0.0f};
    OcSamplerConfig cfg = { OC_SAMPLER_TOP_P, 1.0f, 0u, 0.9f, 1.0f, 3ull };
    for (uint64_t s = 1; s < 20; s++) {
        cfg.seed = s;
        cr_assert_eq(oc_sample(logits, 4, &cfg, NULL, 0), 0u,
                     "nucleus must be {0} for dominant token");
    }
}

Test(sampling, deterministic_given_seed)
{
    float logits[] = {1.0f, 2.0f, 3.0f, 2.5f, 1.5f};
    OcSamplerConfig cfg = { OC_SAMPLER_TEMPERATURE, 0.8f, 0u, 1.0f, 1.0f, 12345ull };
    uint32_t a = oc_sample(logits, 5, &cfg, NULL, 0);
    uint32_t b = oc_sample(logits, 5, &cfg, NULL, 0);
    cr_assert_eq(a, b, "same seed must yield same token");
}

Test(sampling, temperature_scales_logits_before_softmax)
{
    /* Very low temperature → near-greedy (argmax wins). */
    float logits[] = {1.0f, 2.0f, 1.5f};
    OcSamplerConfig cfg = { OC_SAMPLER_TEMPERATURE, 0.01f, 0u, 1.0f, 1.0f, 99ull };
    uint32_t t = oc_sample(logits, 3, &cfg, NULL, 0);
    cr_assert_eq(t, 1u, "low T must concentrate on argmax");
}

Test(sampling, out_of_range_recent_token_ignored)
{
    /* recent token id >= vocab_size must not crash. */
    float logits[] = {0.0f, 1.0f, 2.0f};
    uint32_t recent[] = {999};
    OcSamplerConfig cfg = OC_SAMPLER_DEFAULT;
    cfg.repeat_penalty = 5.0f;
    cr_assert_eq(oc_sample(logits, 3, &cfg, recent, 1), 2u);
}
