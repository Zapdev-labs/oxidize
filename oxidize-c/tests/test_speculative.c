/*
 * test_speculative.c — speculative decoding component tests.
 *
 * Tests the verification kernel (greedy + stochastic) with synthetic logits,
 * and the config/stats defaults. Full end-to-end generation requires two
 * loaded GGUF models sharing a vocabulary.
 */
#include "framework.h"

#include "oxidize/speculative.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

/* ─── Verification kernel: greedy ──────────────────────────────────────────
 * K=2, vocab=4. Draft proposes [1, 2]. Target argmax at positions 0, 1
 * matches → both accepted + bonus from position 2. */
Test(speculative, greedy_all_accepted)
{
    uint32_t draft_tokens[] = {1, 2};
    /* draft logits (not used in greedy acceptance, only target argmax matters) */
    float dl0[] = {0.0f, 10.0f, 0.0f, 0.0f};
    float dl1[] = {0.0f, 0.0f, 10.0f, 0.0f};
    float *draft_l[] = {dl0, dl1};
    /* target logits: pos 0 argmax=1, pos 1 argmax=2, pos 2 (bonus) argmax=3 */
    float tl0[] = {0.0f, 10.0f, 0.0f, 0.0f};
    float tl1[] = {0.0f, 0.0f, 10.0f, 0.0f};
    float tl2[] = {0.0f, 0.0f, 0.0f, 10.0f};
    float *target_l[] = {tl0, tl1, tl2};

    OcSpeculativeConfig cfg = OC_SPECULATIVE_DEFAULT;
    cfg.greedy = true;
    uint64_t seed = 42;
    OcSpeculativeResult res;
    OcError e = oc_speculative_decode(draft_tokens, draft_l, target_l,
                                      2, 4, &cfg, &seed, &res);
    cr_assert_eq(e, OC_OK);
    cr_assert_eq(res.count, 3, "2 accepted + 1 bonus = 3 tokens");
    cr_assert_eq(res.accepted, 2, "both draft tokens accepted");
    cr_assert_eq(res.tokens[0], 1, "token 0 = draft 1");
    cr_assert_eq(res.tokens[1], 2, "token 1 = draft 2");
    cr_assert_eq(res.tokens[2], 3, "token 2 = bonus argmax=3");
    cr_assert_eq(res.used_residual, false, "no rejection");
}

/* ─── Verification kernel: greedy rejection at step 0 ────────────────────
 * Draft proposes [1, 2]. Target argmax at position 0 is 0, not 1 → reject,
 * emit target argmax (0), stop. Only 1 token emitted. */
Test(speculative, greedy_reject_step0)
{
    uint32_t draft_tokens[] = {1, 2};
    float dl0[] = {0.0f, 10.0f, 0.0f, 0.0f};
    float dl1[] = {0.0f, 0.0f, 10.0f, 0.0f};
    float *draft_l[] = {dl0, dl1};
    /* target logits: pos 0 argmax=0 (≠ draft 1) → reject */
    float tl0[] = {10.0f, 0.0f, 0.0f, 0.0f};
    float tl1[] = {0.0f, 0.0f, 10.0f, 0.0f};
    float tl2[] = {0.0f, 0.0f, 0.0f, 10.0f};
    float *target_l[] = {tl0, tl1, tl2};

    OcSpeculativeConfig cfg = OC_SPECULATIVE_DEFAULT;
    cfg.greedy = true;
    uint64_t seed = 42;
    OcSpeculativeResult res;
    oc_speculative_decode(draft_tokens, draft_l, target_l,
                           2, 4, &cfg, &seed, &res);
    cr_assert_eq(res.count, 1, "only 1 token (rejected at step 0)");
    cr_assert_eq(res.accepted, 0, "no draft tokens accepted");
    cr_assert_eq(res.tokens[0], 0, "emitted target argmax=0");
    cr_assert_eq(res.used_residual, true, "rejection → residual");
}

/* ─── Verification kernel: greedy reject at step 1 ────────────────────────
 * Draft [1, 2]. Target accepts step 0 (argmax=1), rejects step 1 (argmax=3
 * ≠ 2). Emit [1, 3], stop. */
Test(speculative, greedy_reject_step1)
{
    uint32_t draft_tokens[] = {1, 2};
    float dl0[] = {0.0f, 10.0f, 0.0f, 0.0f};
    float dl1[] = {0.0f, 0.0f, 10.0f, 0.0f};
    float *draft_l[] = {dl0, dl1};
    float tl0[] = {0.0f, 10.0f, 0.0f, 0.0f};  /* argmax=1 = draft[0] ✓ */
    float tl1[] = {0.0f, 0.0f, 0.0f, 10.0f};  /* argmax=3 ≠ draft[1]=2 ✗ */
    float tl2[] = {0.0f, 0.0f, 0.0f, 10.0f};
    float *target_l[] = {tl0, tl1, tl2};

    OcSpeculativeConfig cfg = OC_SPECULATIVE_DEFAULT;
    cfg.greedy = true;
    uint64_t seed = 42;
    OcSpeculativeResult res;
    oc_speculative_decode(draft_tokens, draft_l, target_l,
                           2, 4, &cfg, &seed, &res);
    cr_assert_eq(res.count, 2, "1 accepted + 1 residual = 2 tokens");
    cr_assert_eq(res.accepted, 1, "step 0 accepted, step 1 rejected");
    cr_assert_eq(res.tokens[0], 1, "accepted token = 1");
    cr_assert_eq(res.tokens[1], 3, "residual = target argmax = 3");
    cr_assert_eq(res.used_residual, true);
}

/* ─── Stochastic: deterministic acceptance when p >> q ───────────────────
 * Draft proposes token 0 with very low probability, target has it high.
 * p/q > 1 → accept_prob = 1.0 → always accepted. */
Test(speculative, stochastic_accept_high_ratio)
{
    uint32_t draft_tokens[] = {0};
    /* draft gives token 0 very low prob */
    float dl0[] = {-100.0f, 10.0f, 0.0f, 0.0f};
    float *draft_l[] = {dl0};
    /* target gives token 0 high prob */
    float tl0[] = {10.0f, 0.0f, 0.0f, 0.0f};
    float tl1[] = {0.0f, 10.0f, 0.0f, 0.0f};
    float *target_l[] = {tl0, tl1};

    OcSpeculativeConfig cfg = OC_SPECULATIVE_DEFAULT;
    cfg.greedy = false;
    cfg.temperature = 1.0f;
    uint64_t seed = 12345;
    OcSpeculativeResult res;
    oc_speculative_decode(draft_tokens, draft_l, target_l,
                           1, 4, &cfg, &seed, &res);
    cr_assert_eq(res.accepted, 1, "should accept when p/q >= 1");
    cr_assert_eq(res.tokens[0], 0, "accepted token = 0");
    cr_assert_eq(res.count, 2, "1 accepted + 1 bonus");
    cr_assert_eq(res.used_residual, false);
}

/* ─── Config defaults ──────────────────────────────────────────────────── */
Test(speculative, config_defaults)
{
    OcSpeculativeConfig cfg = OC_SPECULATIVE_DEFAULT;
    cr_assert_eq(cfg.draft_tokens_per_step, 4, "default K=4");
    cr_assert_eq(cfg.min_acceptance_rate, 0.3f, "default min accept=0.3");
    cr_assert_eq(cfg.greedy, true, "default greedy");
    cr_assert_eq(cfg.stop_token, 0xFFFFFFFFu, "default no stop");
}

/* ─── Stats ────────────────────────────────────────────────────────────── */
Test(speculative, stats_init)
{
    OcSpeculativeStats stats;
    memset(&stats, 0, sizeof(stats));
    cr_assert_eq(stats.total_draft_tokens, 0);
    cr_assert_eq(stats.accepted_draft_tokens, 0);
    cr_assert_eq(stats.target_forward_passes, 0);
    cr_assert_eq(stats.emitted_tokens, 0);
}

/* ─── Invalid args ─────────────────────────────────────────────────────── */
Test(speculative, decode_null_args)
{
    OcSpeculativeResult res;
    uint64_t seed = 42;
    OcSpeculativeConfig cfg = OC_SPECULATIVE_DEFAULT;
    cr_assert_eq(oc_speculative_decode(NULL, NULL, NULL, 0, 0, &cfg, &seed, &res),
                 OC_ERR_INVALID_ARG);
}

Test(speculative, decode_k_zero)
{
    uint64_t seed = 42;
    OcSpeculativeConfig cfg = OC_SPECULATIVE_DEFAULT;
    OcSpeculativeResult res;
    float *dl[] = {NULL};
    float *tl[] = {NULL, NULL};
    cr_assert_eq(oc_speculative_decode((const uint32_t[]){0}, dl, tl,
                                        0, 4, &cfg, &seed, &res),
                 OC_ERR_INVALID_ARG);
}

Test(speculative, generate_rejects_zero_capacity)
{
    OcLlamaModel target = {0}, draft = {0};
    OcLlamaSession target_sess = { .model = &target };
    OcLlamaSession draft_sess = { .model = &draft };
    OcSpeculativeConfig cfg = OC_SPECULATIVE_DEFAULT;
    uint32_t prompt = 1, output = 0;
    size_t output_len = 99;
    cr_assert_eq(oc_speculative_generate(&target, &target_sess,
        &draft, &draft_sess, &prompt, 1, &cfg, &output, &output_len, 0, NULL),
        OC_ERR_INVALID_ARG);
}

Test(speculative, generate_rejects_qwen35_sessions)
{
    OcLlamaModel target = {0}, draft = {0};
    target.cfg.vocab_size = draft.cfg.vocab_size = 1;
    target.cfg.is_qwen35 = true;
    OcLlamaSession target_sess = { .model = &target };
    OcLlamaSession draft_sess = { .model = &draft };
    OcSpeculativeConfig cfg = OC_SPECULATIVE_DEFAULT;
    uint32_t prompt = 0, output = 0;
    size_t output_len = 0;
    cr_assert_eq(oc_speculative_generate(&target, &target_sess,
        &draft, &draft_sess, &prompt, 1, &cfg, &output, &output_len, 1, NULL),
        OC_ERR_INVALID_ARG);
    cr_assert_eq(output_len, 0);
}

Test(speculative, generate_propagates_prefill_failure)
{
    OcLlamaModel target = {0}, draft = {0};
    target.cfg.vocab_size = draft.cfg.vocab_size = 1;
    OcLlamaSession target_sess = { .model = &target };
    OcLlamaSession draft_sess = { .model = &draft };
    OcSpeculativeConfig cfg = OC_SPECULATIVE_DEFAULT;
    uint32_t prompt = 0, output = 0;
    size_t output_len = 0;
    cr_assert_eq(oc_speculative_generate(&target, &target_sess,
        &draft, &draft_sess, &prompt, 1, &cfg, &output, &output_len, 1, NULL),
        OC_ERR_INVALID_ARG);
    cr_assert_eq(output_len, 0);
}
