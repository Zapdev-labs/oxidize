/* test_beam_search.c — Beam search tests. */
#include "framework.h"
#include "oxidize/sampling.h"
#include "oxidize/error.h"
#include <stdlib.h>
#include <math.h>

Test(beam, softmax_probs_basic)
{
    float logits[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float probs[4];
    OcError e = oc_softmax_probs(logits, 4, 1.0f, probs);
    cr_assert_eq(e, OC_OK);
    /* Uniform: each = 0.25. */
    for (int i = 0; i < 4; i++)
        cr_assert_float_eq(probs[i], 0.25f, 0.001f);
}

Test(beam, softmax_probs_temperature)
{
    float logits[] = {1.0f, 0.0f};
    float probs[2];
    oc_softmax_probs(logits, 2, 1.0f, probs);
    /* T=1: softmax([1,0]) = [e/(e+1), 1/(e+1)] ~= [0.731, 0.269]. */
    cr_assert_gt(probs[0], probs[1]);
    cr_assert_float_eq(probs[0], 0.731f, 0.01f);
    cr_assert_float_eq(probs[1], 0.269f, 0.01f);
}

Test(beam, softmax_probs_null)
{
    cr_assert_neq(oc_softmax_probs(NULL, 4, 1.0f, NULL), OC_OK);
    float p[4];
    cr_assert_neq(oc_softmax_probs(NULL, 4, 1.0f, p), OC_OK);
}

Test(beam, basic_search)
{
    /* 2 steps, 4 vocab, beam_width=2. */
    float step0[] = {0.1f, 0.5f, 0.2f, 0.1f}; /* token 1 has highest prob */
    float step1[] = {0.3f, 0.1f, 0.5f, 0.1f}; /* token 2 has highest prob */

    float *logits[] = {step0, step1};
    OcBeamSearchResult result;
    OcError e = oc_beam_search(logits, 2, 4, 2, 0xFFFFFFFFu, &result);
    cr_assert_eq(e, OC_OK);
    cr_assert_eq(result.n_tokens, 2);
    /* Best beam should be [1, 2] (highest prob at each step). */
    cr_assert_eq(result.tokens[0], 1);
    cr_assert_eq(result.tokens[1], 2);
    oc_beam_search_result_free(&result);
}

Test(beam, beam_width_1)
{
    /* beam_width=1 = greedy. */
    float step0[] = {0.1f, 0.9f, 0.1f, 0.1f};
    float step1[] = {0.8f, 0.1f, 0.1f, 0.1f};

    float *logits[] = {step0, step1};
    OcBeamSearchResult result;
    OcError e = oc_beam_search(logits, 2, 4, 1, 0xFFFFFFFFu, &result);
    cr_assert_eq(e, OC_OK);
    cr_assert_eq(result.n_tokens, 2);
    cr_assert_eq(result.tokens[0], 1);
    cr_assert_eq(result.tokens[1], 0);
    oc_beam_search_result_free(&result);
}

Test(beam, eos_terminates)
{
    float step0[] = {0.1f, 0.9f, 0.1f, 0.1f}; /* token 1 */
    float step1[] = {0.1f, 0.1f, 0.1f, 0.9f}; /* token 3 = EOS */

    float *logits[] = {step0, step1};
    OcBeamSearchResult result;
    OcError e = oc_beam_search(logits, 2, 4, 2, 3u, &result);
    cr_assert_eq(e, OC_OK);
    /* Best beam should end with EOS (token 3). */
    cr_assert_eq(result.tokens[result.n_tokens - 1], 3);
    oc_beam_search_result_free(&result);
}

Test(beam, null_safety)
{
    float *logits[] = {NULL};
    OcBeamSearchResult result;
    cr_assert_neq(oc_beam_search(NULL, 1, 4, 2, 0xFFFFFFFFu, &result), OC_OK);
    cr_assert_neq(oc_beam_search(logits, 1, 4, 0, 0xFFFFFFFFu, &result), OC_OK);
    cr_assert_neq(oc_beam_search(logits, 1, 4, 2, 0xFFFFFFFFu, NULL), OC_OK);
    cr_assert_neq(oc_beam_search(logits, 0, 4, 2, 0xFFFFFFFFu, &result), OC_OK);
}

Test(beam, result_free)
{
    OcBeamSearchResult result = {malloc(4 * sizeof(uint32_t)), 4, 1.5f};
    oc_beam_search_result_free(&result);
    cr_assert_null(result.tokens);
    cr_assert_eq(result.n_tokens, 0);
}

/* ─── Speculative decode probability helpers ───────────────────────────── */

Test(beam, residual_probs_basic)
{
    float target[] = {0.5f, 0.3f, 0.2f, 0.0f};
    float draft[]  = {0.2f, 0.5f, 0.2f, 0.1f};
    float out[4];
    oc_residual_probs(target, draft, out, 4);
    /* residual = max(0, t-d) = [0.3, 0, 0, 0], sum=0.3, normalized = [1, 0, 0, 0]. */
    cr_assert_float_eq(out[0], 1.0f, 0.01f);
    cr_assert_float_eq(out[1], 0.0f, 0.01f);
    cr_assert_float_eq(out[2], 0.0f, 0.01f);
    cr_assert_float_eq(out[3], 0.0f, 0.01f);
}

Test(beam, residual_probs_all_zero)
{
    float target[] = {0.25f, 0.25f, 0.25f, 0.25f};
    float draft[]  = {0.25f, 0.25f, 0.25f, 0.25f};
    float out[4];
    oc_residual_probs(target, draft, out, 4);
    /* All diffs = 0, sum = 0, no normalization. */
    for (int i = 0; i < 4; i++)
        cr_assert_float_eq(out[i], 0.0f, 0.001f);
}

Test(beam, sample_probabilities_basic)
{
    float probs[] = {0.0f, 0.5f, 0.5f, 0.0f};
    /* random=0.0 -> first non-zero cumulative -> index 0 (cumulative 0 <= 0). */
    /* Actually target = 0 * 1.0 = 0, cumulative at 0 = 0, so 0 <= 0 is true. */
    size_t idx = oc_sample_probabilities(probs, 4, 0.0f);
    cr_assert_eq(idx, 0);

    /* random=0.5 -> target = 0.5, cumulative: 0, 0.5 -> 0.5 <= 0.5 -> index 1. */
    idx = oc_sample_probabilities(probs, 4, 0.5f);
    cr_assert_eq(idx, 1);

    /* random=0.99 -> target = 0.99, cumulative: 0, 0.5, 1.0 -> 0.99 <= 1.0 -> index 2. */
    idx = oc_sample_probabilities(probs, 4, 0.99f);
    cr_assert_eq(idx, 2);
}

Test(beam, sample_probabilities_argmax_fallback)
{
    float probs[] = {0.0f, 0.0f, 0.0f, 0.0f};
    /* All zeros -> argmax returns first (index 0). */
    size_t idx = oc_sample_probabilities(probs, 4, 0.5f);
    cr_assert_eq(idx, 0);
}

Test(beam, sample_probabilities_weighted)
{
    float probs[] = {0.1f, 0.9f, 0.0f, 0.0f};
    /* random=0.5 -> target = 0.5 * 1.0 = 0.5, cumulative: 0.1, 1.0 -> 0.5 <= 1.0 -> index 1. */
    size_t idx = oc_sample_probabilities(probs, 4, 0.5f);
    cr_assert_eq(idx, 1);
}

/* ─── Repetition penalties ─────────────────────────────────────────────── */

Test(beam, rep_penalty_basic)
{
    float logits[] = {1.0f, 2.0f, 3.0f, 4.0f};
    uint32_t recent[] = {1, 1, 3};
    OcRepetitionPenaltyConfig cfg = {
        .frequency_penalty = 0.1f,
        .presence_penalty = 0.5f,
        .newline_token_id = 0xFFFFFFFFu,
        .newline_penalty = 0.0f,
    };
    oc_apply_repetition_penalties(logits, 4, recent, 3, &cfg);
    /* token 1: freq=2, penalty = 2*0.1 + 0.5 = 0.7 -> 2.0 - 0.7 = 1.3. */
    cr_assert_float_eq(logits[1], 1.3f, 0.001f);
    /* token 3: freq=1, penalty = 0.1 + 0.5 = 0.6 -> 4.0 - 0.6 = 3.4. */
    cr_assert_float_eq(logits[3], 3.4f, 0.001f);
    /* token 0 and 2: no change. */
    cr_assert_float_eq(logits[0], 1.0f, 0.001f);
    cr_assert_float_eq(logits[2], 3.0f, 0.001f);
}

Test(beam, rep_penalty_noop)
{
    float logits[] = {1.0f, 2.0f};
    uint32_t recent[] = {0};
    OcRepetitionPenaltyConfig cfg = {0.0f, 0.0f, 0xFFFFFFFFu, 0.0f};
    oc_apply_repetition_penalties(logits, 2, recent, 1, &cfg);
    /* No penalties configured -> no change. */
    cr_assert_float_eq(logits[0], 1.0f, 0.001f);
    cr_assert_float_eq(logits[1], 2.0f, 0.001f);
}

Test(beam, rep_penalty_newline)
{
    float logits[] = {1.0f, 2.0f, 3.0f};
    uint32_t recent[] = {0};
    OcRepetitionPenaltyConfig cfg = {
        .frequency_penalty = 0.0f,
        .presence_penalty = 0.0f,
        .newline_token_id = 1,
        .newline_penalty = 0.5f,
    };
    oc_apply_repetition_penalties(logits, 3, recent, 1, &cfg);
    /* Only newline penalty applied: token 1 -= 0.5. */
    cr_assert_float_eq(logits[0], 1.0f, 0.001f);
    cr_assert_float_eq(logits[1], 1.5f, 0.001f);
    cr_assert_float_eq(logits[2], 3.0f, 0.001f);
}
