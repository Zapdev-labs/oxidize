/* test_advanced_sampling.c — advanced sampling tests. */
#include "framework.h"
#include "oxidize/advanced_sampling.h"
#include <math.h>
#include <string.h>

Test(asamp, mirostat_v1)
{
    float logits[] = {1.0f, 2.0f, 3.0f, 0.5f, -1.0f};
    float mu = 10.0f;
    uint32_t state = 0;
    uint32_t token = oc_sample_mirostat_v1(logits, 5, &mu, 5.0f, 0.1f, &state);
    cr_assert(token < 5, "token should be in range");
    cr_assert(mu != 10.0f, "mu should be updated");
    cr_assert_neq(state, 0, "RNG state should be advanced");
}

Test(asamp, mirostat_v1_uses_caller_rng_state)
{
    const float logits[] = {0.0f, 0.1f, 0.2f, 0.3f};
    float mu_a = 8.0f;
    float mu_b = 8.0f;
    uint32_t state_a = 0x12345678u;
    uint32_t state_b = 0x12345678u;

    uint32_t token_a = oc_sample_mirostat_v1(logits, 4, &mu_a, 5.0f,
                                              0.1f, &state_a);
    uint32_t token_b = oc_sample_mirostat_v1(logits, 4, &mu_b, 5.0f,
                                              0.1f, &state_b);

    cr_assert_eq(state_a, 0x12345678u * 1103515245u + 12345u,
                 "caller RNG state must advance exactly once");
    cr_assert_eq(state_a, state_b);
    cr_assert_eq(token_a, token_b,
                 "identical caller RNG states must reproduce the draw");
    cr_assert_float_eq(mu_a, mu_b, 1e-6f);
}

OC_TEST_NULL_SAFE(asamp, mirostat_v1_null,
        cr_assert_eq(oc_sample_mirostat_v1(NULL, 5, NULL, 5.0f, 0.1f, NULL), 0);)

Test(asamp, mirostat_v2)
{
    float logits[] = {1.0f, 2.0f, 3.0f, 0.5f, -1.0f};
    float mu = 10.0f;
    uint32_t state = 0;
    uint32_t token = oc_sample_mirostat_v2(logits, 5, &mu, 5.0f, 0.1f, &state);
    cr_assert(token < 5);
    cr_assert(mu != 10.0f);
}

Test(asamp, tfs_basic)
{
    float logits[] = {1.0f, 2.0f, 3.0f, 0.5f, -1.0f};
    uint32_t token = oc_sample_tfs(logits, 5, 0.95f, 1.0f);
    cr_assert(token < 5);
}

OC_TEST_NULL_SAFE(asamp, tfs_null,
        cr_assert_eq(oc_sample_tfs(NULL, 5, 0.95f, 1.0f), 0);)

Test(asamp, typical_basic)
{
    float logits[] = {1.0f, 2.0f, 3.0f, 0.5f, -1.0f};
    uint32_t token = oc_sample_typical(logits, 5, 0.95f, 1.0f);
    cr_assert(token < 5);
}

Test(asamp, top_a_basic)
{
    float logits[] = {1.0f, 2.0f, 3.0f, 0.5f, -1.0f};
    uint32_t token = oc_sample_top_a(logits, 5, 1.0f, 1.0f);
    cr_assert(token < 5);
}

Test(asamp, eta_cutoff)
{
    float logits[] = {1.0f, 2.0f, 3.0f, 0.5f, -1.0f};
    uint32_t token = oc_sample_eta_cutoff(logits, 5, 0.001f, 1.0f);
    cr_assert(token < 5);
}

Test(asamp, penalties)
{
    float logits[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    uint32_t recent[] = {2, 2, 4, 4, 4};
    oc_apply_penalties(logits, 5, recent, 5, 0.5f, 0.1f);
    /* Token 4 was seen 3 times, should be penalized. */
    cr_assert(logits[4] < 5.0f, "token 4 should be penalized");
    /* Token 0 was never seen, should be unchanged. */
    cr_assert_float_eq(logits[0], 1.0f, 1e-6f);
}

Test(asamp, penalties_null)
{
    float logits[] = {1.0f, 2.0f};
    oc_apply_penalties(logits, 2, NULL, 0, 0.5f, 0.1f);
    /* Should be a no-op. */
    cr_assert_float_eq(logits[0], 1.0f, 1e-6f);
}

Test(asamp, beam_search_init)
{
    OcBeamSearchState st;
    OcError e = oc_beam_search_init(&st, 4, 20, 1.0f, 0, 1);
    cr_assert_eq(e, OC_OK);
    cr_assert_eq(st.beam_width, 4);
    cr_assert_eq(st.beams[0].n_tokens, 1);
    cr_assert_eq(st.beams[0].tokens[0], 1);
    oc_beam_search_free(&st);
}

OC_TEST_NULL_SAFE(asamp, beam_search_null,
        cr_assert_neq(oc_beam_search_init(NULL, 4, 20, 1.0f, 0, 1), OC_OK);)

OC_TEST_NULL_SAFE(asamp, beam_search_done_empty,
        cr_assert(oc_beam_search_done(NULL));)

Test(asamp, beam_search_best)
{
    OcBeamSearchState st;
    OcError e = oc_beam_search_init(&st, 2, 10, 1.0f, 0, 1);
    cr_assert_eq(e, OC_OK);
    const OcBeam *best = oc_beam_search_best(&st);
    cr_assert_not_null(best);
    cr_assert_eq(best->tokens[0], 1);
    oc_beam_search_free(&st);
}

OC_TEST_NULL_SAFE(asamp, contrastive_null,
        cr_assert_eq(oc_sample_contrastive(NULL, 5, NULL, NULL, 0, 0, 0.5f, 0.5f), 0);)

Test(asamp, contrastive_basic)
{
    float logits[] = {1.0f, 2.0f, 3.0f, 0.5f, -1.0f};
    float keys[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float current[] = {0.5f, 0.5f};
    uint32_t token = oc_sample_contrastive(logits, 5, keys, current, 2, 2, 0.5f, 0.5f);
    cr_assert(token < 5);
}

Test(asamp, chain_init_free)
{
    OcSamplerChain chain;
    oc_sampler_chain_init(&chain);
    cr_assert_eq(chain.n_steps, 0);
    oc_sampler_chain_free(&chain);
    cr_assert_eq(chain.n_steps, 0);
}

Test(asamp, chain_add)
{
    OcSamplerChain chain;
    oc_sampler_chain_init(&chain);
    cr_assert_eq(oc_sampler_chain_add(&chain, OC_SAMPLER_STEP_TEMPERATURE, 0.8f, 0), OC_OK);
    cr_assert_eq(oc_sampler_chain_add(&chain, OC_SAMPLER_STEP_TOP_K, 40, 0), OC_OK);
    cr_assert_eq(chain.n_steps, 2);
    oc_sampler_chain_free(&chain);
}

Test(asamp, chain_sample)
{
    OcSamplerChain chain;
    oc_sampler_chain_init(&chain);
    oc_sampler_chain_add(&chain, OC_SAMPLER_STEP_TEMPERATURE, 0.5f, 0);

    float logits[] = {1.0f, 5.0f, 2.0f, 0.5f, -1.0f};
    uint32_t token = oc_sampler_chain_sample(&chain, logits, 5, NULL, 0, NULL);
    cr_assert(token < 5);
    /* With temperature 0.5, the highest logit (index 1) should likely win. */
    oc_sampler_chain_free(&chain);
}

Test(asamp, chain_applies_transforms_after_terminal_step)
{
    OcSamplerChain chain;
    oc_sampler_chain_init(&chain);
    cr_assert_eq(oc_sampler_chain_add(&chain, OC_SAMPLER_STEP_TFS,
                                      0.95f, 1.0f), OC_OK);
    cr_assert_eq(oc_sampler_chain_add(&chain, OC_SAMPLER_STEP_TEMPERATURE,
                                      0.5f, 0.0f), OC_OK);

    float logits[] = {1.0f, 4.0f, -2.0f};
    (void)oc_sampler_chain_sample(&chain, logits, 3, NULL, 0, NULL);

    cr_assert_float_eq(logits[0], 2.0f, 1e-6f,
                       "transform after terminal step was skipped");
    cr_assert_float_eq(logits[1], 8.0f, 1e-6f,
                       "transform after terminal step was skipped");
    cr_assert_float_eq(logits[2], -4.0f, 1e-6f,
                       "transform after terminal step was skipped");
    oc_sampler_chain_free(&chain);
}

Test(asamp, chain_mirostat_v1_advances_chain_rng)
{
    OcSamplerChain chain;
    oc_sampler_chain_init(&chain);
    cr_assert_eq(oc_sampler_chain_add(&chain, OC_SAMPLER_STEP_MIROSTAT_V1,
                                      5.0f, 0.1f), OC_OK);
    float logits[] = {0.0f, 0.1f, 0.2f, 0.3f};
    float mu = 8.0f;
    uint32_t before = chain.rng_state;
    (void)oc_sampler_chain_sample(&chain, logits, 4, NULL, 0, &mu);
    cr_assert_neq(chain.rng_state, before);
    oc_sampler_chain_free(&chain);
}

OC_TEST_NULL_SAFE(asamp, chain_add_null,
        cr_assert_neq(oc_sampler_chain_add(NULL, OC_SAMPLER_STEP_TEMPERATURE, 0.8f, 0), OC_OK);)
