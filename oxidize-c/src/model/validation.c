/*
 * validation.c — cross-validation and model quality assessment utilities.
 *
 * Stores validation samples, then computes k-fold / single-pass accuracy,
 * cross-entropy loss, confusion matrices, and perplexity. Folds are
 * assigned deterministically via an LCG seeded from OcValidationConfig.seed
 * so results are reproducible across runs.
 */
#include "oxidize/validation.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ─── State ─────────────────────────────────────────────────────────── */

/* ─── Helpers ───────────────────────────────────────────────────────── */

/* Deterministic LCG for reproducible fold assignment. */
static uint32_t lcg_next(uint32_t *state)
{
    *state = (*state * 1103515245u + 12345u) & 0x7FFFFFFFu;
    return *state;
}

static OcError ensure_cap(OcValidationState *s, uint32_t needed)
{
    if (s->cap_samples >= needed) return OC_OK;
    uint32_t new_cap = s->cap_samples > 0 ? s->cap_samples : 16u;
    while (new_cap < needed) new_cap *= 2u;
    if (new_cap > s->config.max_samples) new_cap = s->config.max_samples;
    if (new_cap < needed) return OC_ERR_OOM;

    OcValidationSample *p = realloc(s->samples,
                                    (size_t)new_cap * sizeof(*p));
    if (!p) return OC_ERR_OOM;
    s->samples     = p;
    s->cap_samples = new_cap;
    return OC_OK;
}

/* ─── Config helpers ────────────────────────────────────────────────── */

OcError oc_validation_config_init(OcValidationConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    cfg->n_folds     = OC_VALIDATION_DEFAULT_N_FOLDS;
    cfg->max_samples = OC_VALIDATION_DEFAULT_MAX_SAMPLES;
    cfg->seed        = OC_VALIDATION_DEFAULT_SEED;
    return OC_OK;
}

/* ─── Lifecycle ─────────────────────────────────────────────────────── */

OcError oc_validation_init(const OcValidationConfig *config,
                           OcValidationState **out)
{
    if (!out) return OC_ERR_INVALID_ARG;
    *out = NULL;

    OcValidationConfig cfg;
    if (config) {
        cfg = *config;
    } else {
        oc_validation_config_init(&cfg);
    }
    if (cfg.n_folds == 0) cfg.n_folds = 1;
    if (cfg.n_folds > OC_VALIDATION_MAX_FOLDS) cfg.n_folds = OC_VALIDATION_MAX_FOLDS;
    if (cfg.max_samples == 0) cfg.max_samples = OC_VALIDATION_DEFAULT_MAX_SAMPLES;

    OcValidationState *s = malloc(sizeof(*s));
    if (!s) return OC_ERR_OOM;
    memset(s, 0, sizeof(*s));
    s->config      = cfg;
    s->samples     = NULL;
    s->n_samples   = 0;
    s->cap_samples = 0;

    /* Pre-grow to a small initial capacity. */
    uint32_t init_cap = cfg.max_samples < 16u ? cfg.max_samples : 16u;
    if (init_cap > 0) {
        s->samples = malloc((size_t)init_cap * sizeof(*s->samples));
        if (!s->samples) { free(s); return OC_ERR_OOM; }
        s->cap_samples = init_cap;
    }

    *out = s;
    return OC_OK;
}

void oc_validation_free(OcValidationState *state)
{
    if (!state) return;
    free(state->samples);
    memset(state, 0, sizeof(*state));
    free(state);
}

/* ─── Sample management ────────────────────────────────────────────── */

OcError oc_validation_add_sample(OcValidationState *state,
                                 const OcValidationSample *sample)
{
    if (!state || !sample) return OC_ERR_INVALID_ARG;
    if (sample->n_input > OC_VALIDATION_MAX_INPUT_TOKENS)
        return OC_ERR_INVALID_ARG;
    if (state->n_samples >= state->config.max_samples)
        return OC_ERR_OOM;

    OcError e = ensure_cap(state, state->n_samples + 1u);
    if (e != OC_OK) return e;

    state->samples[state->n_samples] = *sample;
    state->n_samples++;
    return OC_OK;
}

OcError oc_validation_clear(OcValidationState *state)
{
    if (!state) return OC_ERR_INVALID_ARG;
    state->n_samples = 0;
    return OC_OK;
}

/* ─── Metric primitives ────────────────────────────────────────────── */

/* Accuracy over a subset: correct = predicted == expected. */
static float subset_accuracy(const OcValidationSample *s, uint32_t n)
{
    if (n == 0) return 0.0f;
    uint32_t correct = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (s[i].predicted_token == s[i].expected_token) correct++;
    }
    return (float)correct / (float)n;
}

/* Mean cross-entropy loss over a subset. Loss per sample =
 * -logprob (clamped so it is non-negative). Weighted by sample.weight,
 * then divided by total weight. */
static float subset_loss(const OcValidationSample *s, uint32_t n)
{
    if (n == 0) return 0.0f;
    double total_w = 0.0;
    double sum     = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        float lp = s[i].logprob;
        if (lp > 0.0f) lp = 0.0f;        /* logprob <= 0 by definition      */
        float w = s[i].weight > 0.0f ? s[i].weight : 1.0f;
        sum     += (double)(-lp) * (double)w;
        total_w += (double)w;
    }
    if (total_w <= 0.0) return 0.0f;
    return (float)(sum / total_w);
}

/* ─── Metrics ───────────────────────────────────────────────────────── */

OcError oc_validation_k_fold(const OcValidationState *state,
                              OcValidationResult *out_result)
{
    if (!state || !out_result) return OC_ERR_INVALID_ARG;
    memset(out_result, 0, sizeof(*out_result));

    uint32_t n = state->n_samples;
    uint32_t k = state->config.n_folds;
    if (k == 0) k = 1;
    if (k > OC_VALIDATION_MAX_FOLDS) k = OC_VALIDATION_MAX_FOLDS;
    if (k > n && n > 0) k = n;          /* at most one fold per sample     */
    if (k == 0) k = 1;
    out_result->n_folds = k;

    if (n == 0) return OC_OK;

    /* Assign each sample to a fold via LCG-shuffled round-robin. We pick
     * a permutation of [0,n) by LCG and assign fold = i % k. */
    uint32_t *perm = malloc((size_t)n * sizeof(uint32_t));
    if (!perm) return OC_ERR_OOM;
    for (uint32_t i = 0; i < n; i++) perm[i] = i;
    uint32_t rng = state->config.seed;
    for (uint32_t i = n; i > 1; i--) {
        uint32_t j = lcg_next(&rng) % i;
        uint32_t t = perm[i - 1];
        perm[i - 1] = perm[j];
        perm[j]     = t;
    }

    /* For each fold, gather held-out samples and compute accuracy/loss. */
    OcValidationSample *holdout = malloc((size_t)n * sizeof(*holdout));
    if (!holdout) { free(perm); return OC_ERR_OOM; }

    double weighted_acc = 0.0;
    double weighted_loss = 0.0;
    uint32_t total_held = 0;
    for (uint32_t f = 0; f < k; f++) {
        uint32_t hc = 0;
        for (uint32_t i = 0; i < n; i++) {
            if (perm[i] % k == f) {
                holdout[hc++] = state->samples[i];
            }
        }
        float acc  = subset_accuracy(holdout, hc);
        float loss = subset_loss(holdout, hc);
        out_result->per_fold_accuracy[f] = acc;
        weighted_acc  += (double)acc * (double)hc;
        weighted_loss += (double)loss * (double)hc;
        total_held    += hc;
    }
    free(holdout);
    free(perm);

    if (total_held > 0) {
        out_result->accuracy  = (float)(weighted_acc / (double)total_held);
        out_result->loss      = (float)(weighted_loss / (double)total_held);
    }
    out_result->n_samples = n;
    return OC_OK;
}

OcError oc_validation_single(const OcValidationState *state,
                             OcValidationResult *out_result)
{
    if (!state || !out_result) return OC_ERR_INVALID_ARG;
    memset(out_result, 0, sizeof(*out_result));

    uint32_t n = state->n_samples;
    out_result->n_samples = n;
    out_result->n_folds    = 1;
    out_result->accuracy   = subset_accuracy(state->samples, n);
    out_result->loss       = subset_loss(state->samples, n);
    if (n > 0) out_result->per_fold_accuracy[0] = out_result->accuracy;
    return OC_OK;
}

OcError oc_validation_confusion_matrix(const OcValidationState *state,
                                       uint32_t n_classes,
                                       uint32_t *out_matrix)
{
    if (!state || !out_matrix) return OC_ERR_INVALID_ARG;
    if (n_classes == 0) return OC_ERR_INVALID_ARG;
    memset(out_matrix, 0, (size_t)n_classes * (size_t)n_classes * sizeof(uint32_t));

    for (uint32_t i = 0; i < state->n_samples; i++) {
        uint32_t exp = state->samples[i].expected_token;
        uint32_t prd = state->samples[i].predicted_token;
        if (exp >= n_classes || prd >= n_classes) return OC_ERR_INVALID_ARG;
        out_matrix[exp * n_classes + prd]++;
    }
    return OC_OK;
}

OcError oc_validation_perplexity(const OcValidationState *state,
                                double *out_perplexity)
{
    if (!state || !out_perplexity) return OC_ERR_INVALID_ARG;
    *out_perplexity = 0.0;
    uint32_t n = state->n_samples;
    if (n == 0) {
        *out_perplexity = 1.0; /* convention: empty corpus has perplexity 1 */
        return OC_OK;
    }
    double total_w = 0.0;
    double sum_neg_lp = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        float lp = state->samples[i].logprob;
        if (lp > 0.0f) lp = 0.0f;
        float w = state->samples[i].weight > 0.0f ? state->samples[i].weight : 1.0f;
        sum_neg_lp += (double)(-lp) * (double)w;
        total_w    += (double)w;
    }
    if (total_w <= 0.0) {
        *out_perplexity = 1.0;
        return OC_OK;
    }
    double mean_loss = sum_neg_lp / total_w;
    *out_perplexity = exp(mean_loss);
    return OC_OK;
}
