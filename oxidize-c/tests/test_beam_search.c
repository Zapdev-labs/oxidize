/* test_beam_search.c — Beam search tests. */
#include <criterion/criterion.h>
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
