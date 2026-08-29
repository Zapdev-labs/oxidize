/* test_validation.c — cross-validation and quality-assessment tests. */
#include <criterion/criterion.h>
#include <math.h>
#include <string.h>
#include "oxidize/validation.h"


static void add_simple_sample(OcValidationState *s, uint32_t expected,
                              uint32_t predicted, float logprob, float weight)
{
    OcValidationSample sample;
    memset(&sample, 0, sizeof(sample));
    sample.input_tokens[0] = expected;  /* minimal input */
    sample.n_input         = 1;
    sample.expected_token  = expected;
    sample.predicted_token = predicted;
    sample.logprob         = logprob;
    sample.weight          = weight;
    cr_assert_eq(oc_validation_add_sample(s, &sample), OC_OK);
}


Test(validation, config_init_defaults)
{
    OcValidationConfig cfg;
    cr_assert_eq(oc_validation_config_init(&cfg), OC_OK);
    cr_assert_eq(cfg.n_folds,     OC_VALIDATION_DEFAULT_N_FOLDS);
    cr_assert_eq(cfg.max_samples, OC_VALIDATION_DEFAULT_MAX_SAMPLES);
    cr_assert_eq(cfg.seed,        OC_VALIDATION_DEFAULT_SEED);
}

Test(validation, config_init_null)
{
    cr_assert_eq(oc_validation_config_init(NULL), OC_ERR_INVALID_ARG);
}


Test(validation, init_default)
{
    OcValidationState *s = NULL;
    cr_assert_eq(oc_validation_init(NULL, &s), OC_OK);
    cr_assert_not_null(s);
    cr_assert_eq(s->n_samples, 0u);
    oc_validation_free(s);
}

Test(validation, init_with_config)
{
    OcValidationConfig cfg;
    oc_validation_config_init(&cfg);
    cfg.n_folds     = 3;
    cfg.max_samples = 50;
    cfg.seed        = 7;
    OcValidationState *s = NULL;
    cr_assert_eq(oc_validation_init(&cfg, &s), OC_OK);
    cr_assert_eq(s->config.n_folds, 3u);
    cr_assert_eq(s->config.max_samples, 50u);
    cr_assert_eq(s->config.seed, 7u);
    oc_validation_free(s);
}

Test(validation, init_zero_folds_normalized)
{
    OcValidationConfig cfg;
    oc_validation_config_init(&cfg);
    cfg.n_folds = 0;
    OcValidationState *s = NULL;
    cr_assert_eq(oc_validation_init(&cfg, &s), OC_OK);
    cr_assert_eq(s->config.n_folds, 1u);
    oc_validation_free(s);
}

Test(validation, init_too_many_folds_capped)
{
    OcValidationConfig cfg;
    oc_validation_config_init(&cfg);
    cfg.n_folds = 99;
    OcValidationState *s = NULL;
    cr_assert_eq(oc_validation_init(&cfg, &s), OC_OK);
    cr_assert_eq(s->config.n_folds, OC_VALIDATION_MAX_FOLDS);
    oc_validation_free(s);
}

Test(validation, free_null_is_safe)
{
    oc_validation_free(NULL);
    cr_assert(true);
}


Test(validation, add_sample)
{
    OcValidationState *s = NULL;
    cr_assert_eq(oc_validation_init(NULL, &s), OC_OK);
    add_simple_sample(s, 5, 5, -0.5f, 1.0f);
    cr_assert_eq(s->n_samples, 1u);
    oc_validation_free(s);
}

Test(validation, add_sample_null_is_error)
{
    OcValidationState *s = NULL;
    cr_assert_eq(oc_validation_init(NULL, &s), OC_OK);
    cr_assert_eq(oc_validation_add_sample(s, NULL), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_validation_add_sample(NULL, NULL), OC_ERR_INVALID_ARG);
    oc_validation_free(s);
}

Test(validation, add_sample_too_many_input_tokens)
{
    OcValidationState *s = NULL;
    cr_assert_eq(oc_validation_init(NULL, &s), OC_OK);
    OcValidationSample sample;
    memset(&sample, 0, sizeof(sample));
    sample.n_input = OC_VALIDATION_MAX_INPUT_TOKENS + 1u;
    cr_assert_eq(oc_validation_add_sample(s, &sample), OC_ERR_INVALID_ARG);
    oc_validation_free(s);
}

Test(validation, add_sample_overflow)
{
    OcValidationConfig cfg;
    oc_validation_config_init(&cfg);
    cfg.max_samples = 2;
    OcValidationState *s = NULL;
    cr_assert_eq(oc_validation_init(&cfg, &s), OC_OK);
    add_simple_sample(s, 1, 1, -0.1f, 1.0f);
    add_simple_sample(s, 2, 2, -0.1f, 1.0f);
    /* Third should fail. */
    OcValidationSample sample;
    memset(&sample, 0, sizeof(sample));
    sample.n_input        = 1;
    sample.expected_token = 3;
    sample.predicted_token = 3;
    sample.logprob        = -0.1f;
    sample.weight         = 1.0f;
    cr_assert_eq(oc_validation_add_sample(s, &sample), OC_ERR_OOM);
    cr_assert_eq(s->n_samples, 2u);
    oc_validation_free(s);
}

Test(validation, clear_samples)
{
    OcValidationState *s = NULL;
    cr_assert_eq(oc_validation_init(NULL, &s), OC_OK);
    add_simple_sample(s, 1, 1, -0.1f, 1.0f);
    add_simple_sample(s, 2, 2, -0.1f, 1.0f);
    cr_assert_eq(s->n_samples, 2u);
    cr_assert_eq(oc_validation_clear(s), OC_OK);
    cr_assert_eq(s->n_samples, 0u);
    /* Config preserved. */
    cr_assert_eq(s->config.n_folds, OC_VALIDATION_DEFAULT_N_FOLDS);
    oc_validation_free(s);
}

Test(validation, clear_empty_state)
{
    OcValidationState *s = NULL;
    cr_assert_eq(oc_validation_init(NULL, &s), OC_OK);
    cr_assert_eq(oc_validation_clear(s), OC_OK);
    cr_assert_eq(s->n_samples, 0u);
    oc_validation_free(s);
}


Test(validation, single_all_correct)
{
    OcValidationState *s = NULL;
    cr_assert_eq(oc_validation_init(NULL, &s), OC_OK);
    for (uint32_t i = 0; i < 10; i++) {
        add_simple_sample(s, i, i, -0.1f, 1.0f);
    }
    OcValidationResult r;
    cr_assert_eq(oc_validation_single(s, &r), OC_OK);
    cr_assert_eq(r.n_samples, 10u);
    cr_assert_float_eq(r.accuracy, 1.0f, 1e-6f);
    cr_assert_eq(r.n_folds, 1u);
    oc_validation_free(s);
}

Test(validation, single_half_correct)
{
    OcValidationState *s = NULL;
    cr_assert_eq(oc_validation_init(NULL, &s), OC_OK);
    for (uint32_t i = 0; i < 10; i++) {
        /* Even i: correct, odd i: wrong (predicted differs by 100). */
        add_simple_sample(s, i, (i % 2 == 0) ? i : i + 100, -0.5f, 1.0f);
    }
    OcValidationResult r;
    cr_assert_eq(oc_validation_single(s, &r), OC_OK);
    cr_assert_float_eq(r.accuracy, 0.5f, 1e-6f);
    oc_validation_free(s);
}

Test(validation, single_loss_matches_neg_logprob)
{
    OcValidationState *s = NULL;
    cr_assert_eq(oc_validation_init(NULL, &s), OC_OK);
    add_simple_sample(s, 1, 1, -0.5f, 1.0f);
    add_simple_sample(s, 2, 2, -1.5f, 1.0f);
    OcValidationResult r;
    cr_assert_eq(oc_validation_single(s, &r), OC_OK);
    /* Mean loss = (0.5 + 1.5) / 2 = 1.0. */
    cr_assert_float_eq(r.loss, 1.0f, 1e-6f);
    oc_validation_free(s);
}

Test(validation, single_weighted_loss)
{
    OcValidationState *s = NULL;
    cr_assert_eq(oc_validation_init(NULL, &s), OC_OK);
    add_simple_sample(s, 1, 1, -0.0f, 3.0f);  /* loss 0, weight 3 */
    add_simple_sample(s, 2, 2, -4.0f, 1.0f);  /* loss 4, weight 1 */
    OcValidationResult r;
    cr_assert_eq(oc_validation_single(s, &r), OC_OK);
    /* Weighted: (0*3 + 4*1) / (3 + 1) = 1.0. */
    cr_assert_float_eq(r.loss, 1.0f, 1e-6f);
    oc_validation_free(s);
}

Test(validation, single_empty_state)
{
    OcValidationState *s = NULL;
    cr_assert_eq(oc_validation_init(NULL, &s), OC_OK);
    OcValidationResult r;
    cr_assert_eq(oc_validation_single(s, &r), OC_OK);
    cr_assert_eq(r.n_samples, 0u);
    cr_assert_float_eq(r.accuracy, 0.0f, 1e-6f);
    cr_assert_float_eq(r.loss, 0.0f, 1e-6f);
    oc_validation_free(s);
}

Test(validation, single_null_args)
{
    OcValidationResult r;
    cr_assert_eq(oc_validation_single(NULL, &r), OC_ERR_INVALID_ARG);
    OcValidationState *s = NULL;
    cr_assert_eq(oc_validation_init(NULL, &s), OC_OK);
    cr_assert_eq(oc_validation_single(s, NULL), OC_ERR_INVALID_ARG);
    oc_validation_free(s);
}


Test(validation, k_fold_deterministic_with_seed)
{
    OcValidationConfig cfg;
    oc_validation_config_init(&cfg);
    cfg.n_folds = 3;
    cfg.seed    = 42;
    OcValidationState *s1 = NULL, *s2 = NULL;
    cr_assert_eq(oc_validation_init(&cfg, &s1), OC_OK);
    cr_assert_eq(oc_validation_init(&cfg, &s2), OC_OK);
    for (uint32_t i = 0; i < 30; i++) {
        add_simple_sample(s1, i, i, -0.2f, 1.0f);
        add_simple_sample(s2, i, i, -0.2f, 1.0f);
    }
    OcValidationResult r1, r2;
    cr_assert_eq(oc_validation_k_fold(s1, &r1), OC_OK);
    cr_assert_eq(oc_validation_k_fold(s2, &r2), OC_OK);
    cr_assert_eq(r1.n_folds, r2.n_folds);
    for (uint32_t i = 0; i < r1.n_folds; i++) {
        cr_assert_float_eq(r1.per_fold_accuracy[i], r2.per_fold_accuracy[i], 1e-6f);
    }
    oc_validation_free(s1);
    oc_validation_free(s2);
}

Test(validation, k_fold_all_correct)
{
    OcValidationConfig cfg;
    oc_validation_config_init(&cfg);
    cfg.n_folds = 5;
    OcValidationState *s = NULL;
    cr_assert_eq(oc_validation_init(&cfg, &s), OC_OK);
    for (uint32_t i = 0; i < 25; i++) {
        add_simple_sample(s, i, i, -0.1f, 1.0f);
    }
    OcValidationResult r;
    cr_assert_eq(oc_validation_k_fold(s, &r), OC_OK);
    cr_assert_eq(r.n_folds, 5u);
    cr_assert_eq(r.n_samples, 25u);
    cr_assert_float_eq(r.accuracy, 1.0f, 1e-6f);
    /* All folds should be 1.0 since every sample is correct. */
    for (uint32_t i = 0; i < r.n_folds; i++) {
        cr_assert_float_eq(r.per_fold_accuracy[i], 1.0f, 1e-6f);
    }
    oc_validation_free(s);
}

Test(validation, k_fold_fewer_samples_than_folds)
{
    OcValidationConfig cfg;
    oc_validation_config_init(&cfg);
    cfg.n_folds = 5;
    OcValidationState *s = NULL;
    cr_assert_eq(oc_validation_init(&cfg, &s), OC_OK);
    add_simple_sample(s, 1, 1, -0.1f, 1.0f);
    add_simple_sample(s, 2, 2, -0.1f, 1.0f);
    OcValidationResult r;
    cr_assert_eq(oc_validation_k_fold(s, &r), OC_OK);
    cr_assert_eq(r.n_folds, 2u);
    oc_validation_free(s);
}

Test(validation, k_fold_empty_state)
{
    OcValidationState *s = NULL;
    cr_assert_eq(oc_validation_init(NULL, &s), OC_OK);
    OcValidationResult r;
    cr_assert_eq(oc_validation_k_fold(s, &r), OC_OK);
    cr_assert_eq(r.n_samples, 0u);
    cr_assert_eq(r.n_folds, OC_VALIDATION_DEFAULT_N_FOLDS);
    oc_validation_free(s);
}

Test(validation, k_fold_null_args)
{
    OcValidationResult r;
    cr_assert_eq(oc_validation_k_fold(NULL, &r), OC_ERR_INVALID_ARG);
}


Test(validation, confusion_matrix_diagonal)
{
    OcValidationState *s = NULL;
    cr_assert_eq(oc_validation_init(NULL, &s), OC_OK);
    /* Two samples, both correct, different classes. */
    add_simple_sample(s, 0, 0, -0.1f, 1.0f);
    add_simple_sample(s, 1, 1, -0.1f, 1.0f);
    uint32_t mat[4];
    cr_assert_eq(oc_validation_confusion_matrix(s, 2, mat), OC_OK);
    cr_assert_eq(mat[0], 1u); /* [0,0] */
    cr_assert_eq(mat[3], 1u); /* [1,1] */
    cr_assert_eq(mat[1], 0u); /* [0,1] */
    cr_assert_eq(mat[2], 0u); /* [1,0] */
    oc_validation_free(s);
}

Test(validation, confusion_matrix_offdiagonal)
{
    OcValidationState *s = NULL;
    cr_assert_eq(oc_validation_init(NULL, &s), OC_OK);
    add_simple_sample(s, 0, 1, -0.1f, 1.0f); /* expected 0, predicted 1 */
    uint32_t mat[4];
    cr_assert_eq(oc_validation_confusion_matrix(s, 2, mat), OC_OK);
    cr_assert_eq(mat[1], 1u); /* [0,1] */
    oc_validation_free(s);
}

Test(validation, confusion_matrix_token_out_of_range)
{
    OcValidationState *s = NULL;
    cr_assert_eq(oc_validation_init(NULL, &s), OC_OK);
    add_simple_sample(s, 5, 0, -0.1f, 1.0f); /* expected 5 > n_classes */
    uint32_t mat[4];
    cr_assert_eq(oc_validation_confusion_matrix(s, 2, mat), OC_ERR_INVALID_ARG);
    oc_validation_free(s);
}

Test(validation, confusion_matrix_zero_classes)
{
    OcValidationState *s = NULL;
    cr_assert_eq(oc_validation_init(NULL, &s), OC_OK);
    uint32_t mat[1];
    cr_assert_eq(oc_validation_confusion_matrix(s, 0, mat), OC_ERR_INVALID_ARG);
    oc_validation_free(s);
}


Test(validation, perplexity_uniform)
{
    OcValidationState *s = NULL;
    cr_assert_eq(oc_validation_init(NULL, &s), OC_OK);
    add_simple_sample(s, 1, 1, -log(10.0), 1.0f);
    add_simple_sample(s, 2, 2, -log(10.0), 1.0f);
    double ppl = 0.0;
    cr_assert_eq(oc_validation_perplexity(s, &ppl), OC_OK);
    cr_assert_float_eq(ppl, 10.0, 1e-3);
    oc_validation_free(s);
}

Test(validation, perplexity_zero_loss)
{
    OcValidationState *s = NULL;
    cr_assert_eq(oc_validation_init(NULL, &s), OC_OK);
    add_simple_sample(s, 1, 1, 0.0f, 1.0f);
    double ppl = 0.0;
    cr_assert_eq(oc_validation_perplexity(s, &ppl), OC_OK);
    cr_assert_float_eq(ppl, 1.0, 1e-6);
    oc_validation_free(s);
}

Test(validation, perplexity_empty_state_is_one)
{
    OcValidationState *s = NULL;
    cr_assert_eq(oc_validation_init(NULL, &s), OC_OK);
    double ppl = 0.0;
    cr_assert_eq(oc_validation_perplexity(s, &ppl), OC_OK);
    cr_assert_float_eq(ppl, 1.0, 1e-6);
    oc_validation_free(s);
}

Test(validation, perplexity_null_args)
{
    double ppl = 0.0;
    cr_assert_eq(oc_validation_perplexity(NULL, &ppl), OC_ERR_INVALID_ARG);
    OcValidationState *s = NULL;
    cr_assert_eq(oc_validation_init(NULL, &s), OC_OK);
    cr_assert_eq(oc_validation_perplexity(s, NULL), OC_ERR_INVALID_ARG);
    oc_validation_free(s);
}
