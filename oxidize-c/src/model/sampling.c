/* sampling.c — token sampling from logits. */
#include "oxidize/sampling.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>


static uint64_t rng_next(uint64_t *state)
{
    uint64_t x = *state;
    if (x == 0) x = 0x9e3779b97f4a7c15ULL;   /* avoid all-zero state      */
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

/* Uniform double in [0, 1). */
static double rng_uniform(uint64_t *state)
{
    /* Use the top 53 bits for a full-precision double. */
    return (double)(rng_next(state) >> 11) / (double)(1ULL << 53);
}


uint32_t oc_argmax(const float *logits, size_t vocab_size)
{
    if (vocab_size == 0) return 0;
    size_t best = 0;
    float bestv = logits[0];
    for (size_t i = 1; i < vocab_size; i++) {
        if (logits[i] > bestv) { bestv = logits[i]; best = i; }
    }
    return (uint32_t)best;
}


void oc_apply_repeat_penalty(float *logits, size_t vocab_size,
                             const uint32_t *recent, size_t n_recent,
                             float penalty)
{
    if (recent == NULL || n_recent == 0 || penalty == 1.0f) return;
    for (size_t i = 0; i < n_recent; i++) {
        uint32_t t = recent[i];
        if (t >= vocab_size) continue;
        if (penalty <= 0.0f) continue;
        /* Rust applies: if logit > 0, divide; else multiply. We mirror that
         * to keep sign behavior consistent (penalize both polarities). */
        if (logits[t] > 0.0f) {
            logits[t] /= penalty;
        } else {
            logits[t] *= penalty;
        }
    }
}


static void softmax_inplace(float *logits, size_t n)
{
    if (n == 0) return;
    float mx = logits[0];
    for (size_t i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        logits[i] = expf(logits[i] - mx);
        sum += (double)logits[i];
    }
    if (sum <= 0.0) return;
    float inv = (float)(1.0 / sum);
    for (size_t i = 0; i < n; i++) logits[i] *= inv;
}

static size_t top_k_select(const float *logits, size_t n, size_t k,
                           size_t *out_idx)
{
    if (k == 0 || k > n) k = n;
    /* Copy indices, then selection-sort the top-k. */
    for (size_t i = 0; i < n; i++) out_idx[i] = i;
    for (size_t i = 0; i < k; i++) {
        size_t best = i;
        for (size_t j = i + 1; j < n; j++) {
            if (logits[out_idx[j]] > logits[out_idx[best]]) best = j;
        }
        size_t tmp = out_idx[i]; out_idx[i] = out_idx[best]; out_idx[best] = tmp;
    }
    return k;
}

typedef struct RankedLogit {
    float value;
    size_t index;
} RankedLogit;

static int compare_ranked_logit_desc(const void *left, const void *right)
{
    const RankedLogit *a = left;
    const RankedLogit *b = right;
    bool a_nan = isnan(a->value);
    bool b_nan = isnan(b->value);
    if (a_nan != b_nan) return a_nan ? 1 : -1;
    if (a->value < b->value) return 1;
    if (a->value > b->value) return -1;
    return (a->index > b->index) - (a->index < b->index);
}

static size_t top_p_select(const float *logits, size_t n, float p,
                           size_t *out_idx, float *out_probs)
{
    if (p >= 1.0f) {
        for (size_t i = 0; i < n; i++) out_idx[i] = i;
        return n;
    }
    RankedLogit *ranked = malloc(n * sizeof(*ranked));
    if (!ranked) return 0;
    for (size_t i = 0; i < n; i++) ranked[i] = (RankedLogit){ logits[i], i };
    qsort(ranked, n, sizeof(*ranked), compare_ranked_logit_desc);
    for (size_t i = 0; i < n; i++) out_idx[i] = ranked[i].index;
    free(ranked);
    /* Softmax over the sorted logits (max is out_idx[0]). */
    float mx = logits[out_idx[0]];
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        out_probs[i] = expf(logits[out_idx[i]] - mx);
        sum += (double)out_probs[i];
    }
    if (sum <= 0.0) return 1;
    float inv = (float)(1.0 / sum);
    /* Accumulate until cum >= p. */
    double cum = 0.0;
    size_t keep = n;
    for (size_t i = 0; i < n; i++) {
        out_probs[i] *= inv;
        cum += (double)out_probs[i];
        if (cum >= (double)p) { keep = i + 1; break; }
    }
    if (keep == 0) keep = 1;
    return keep;
}


static uint32_t sample_categorical(const size_t *idx, const float *probs,
                                   size_t n, uint64_t *rng)
{
    double u = rng_uniform(rng);
    double cum = 0.0;
    for (size_t i = 0; i < n; i++) {
        cum += (double)probs[i];
        if (u < cum) return (uint32_t)idx[i];
    }
    return (uint32_t)idx[n - 1];
}


uint32_t oc_mirostat_v2_sample(const float *logits, size_t vocab_size,
                               const OcSamplerConfig *cfg,
                               float *mu, uint64_t *rng_state)
{
    if (vocab_size == 0) return UINT32_MAX;
    if (cfg == NULL) {
        /* No config → degenerate to argmax; leave mu untouched. */
        return oc_argmax(logits, vocab_size);
    }

    float tau = cfg->tau;
    if (!(tau > 0.0f)) tau = 5.0f;             /* sane default if unset     */

    float eta = cfg->eta;
    if (!(eta >= 0.0f)) eta = 0.1f;            /* sane default if unset     */
    if (cfg->learning_rate > 0.0f) eta = cfg->learning_rate;

    float m = (mu != NULL) ? *mu : 10.0f;
    if (!(m > 0.0f)) m = 2.0f * tau;           /* init if caller passed 0    */

    /* Step 1: softmax over the full vocabulary (max-subtraction, stable).
     * We need the per-token probability of the sampled token, so compute the
     * full normalized distribution. */
    float mx = logits[0];
    for (size_t i = 1; i < vocab_size; i++) {
        if (logits[i] > mx) mx = logits[i];
    }
    /* Guard against a degenerate all-(-inf) vocabulary. */
    if (!isfinite(mx)) {
        if (mu != NULL) *mu = m;
        return oc_argmax(logits, vocab_size);
    }

    float *probs = malloc(vocab_size * sizeof(float));
    if (probs == NULL) {
        /* OOM → fall back to argmax; do not corrupt mu. */
        if (mu != NULL) *mu = m;
        return oc_argmax(logits, vocab_size);
    }

    double sum = 0.0;
    for (size_t i = 0; i < vocab_size; i++) {
        probs[i] = expf(logits[i] - mx);
        sum += (double)probs[i];
    }
    if (!(sum > 0.0)) {
        free(probs);
        if (mu != NULL) *mu = m;
        return oc_argmax(logits, vocab_size);
    }
    float inv = (float)(1.0 / sum);
    for (size_t i = 0; i < vocab_size; i++) probs[i] *= inv;

    size_t *idx = malloc(vocab_size * sizeof(*idx));
    float *selected_probs = malloc(vocab_size * sizeof(*selected_probs));
    if (idx == NULL || selected_probs == NULL) {
        free(idx);
        free(selected_probs);
        free(probs);
        if (mu != NULL) *mu = m;
        return oc_argmax(logits, vocab_size);
    }
    size_t keep = 0;
    double kept_sum = 0.0;
    for (size_t i = 0; i < vocab_size; i++) {
        float surprise = probs[i] > 0.0f ? -log2f(probs[i]) : INFINITY;
        if (surprise <= m) {
            idx[keep] = i;
            selected_probs[keep] = probs[i];
            kept_sum += selected_probs[keep];
            keep++;
        }
    }
    if (keep == 0) {
        idx[0] = oc_argmax(logits, vocab_size);
        selected_probs[0] = 1.0f;
        keep = 1;
        kept_sum = 1.0;
    }
    float kept_inv = (float)(1.0 / kept_sum);
    for (size_t i = 0; i < keep; i++) selected_probs[i] *= kept_inv;

    uint64_t rng = (rng_state != NULL) ? *rng_state : cfg->seed;
    if (rng == 0) rng = 0xdeadbeefcafef00dULL;
    uint32_t chosen = sample_categorical(idx, selected_probs, keep, &rng);
    if (rng_state != NULL) *rng_state = rng;

    /* Step 3: observed surprise of the sampled token. */
    float p_chosen = expf(logits[chosen] - mx) * inv;
    if (!(p_chosen > 0.0f)) p_chosen = 1e-38f;   /* avoid -inf on degenerate */
    float surprise = -log2f(p_chosen);
    if (!isfinite(surprise)) surprise = 0.0f;

    free(idx);
    free(selected_probs);
    free(probs);

    /* Step 4 & 5: update + clamp the running surprise estimate. */
    m = m - eta * (surprise - tau);
    if (m < 0.0f) m = 0.0f;
    if (m > 2.0f * tau) m = 2.0f * tau;
    if (mu != NULL) *mu = m;

    /* Step 6: return the sampled token. */
    return (uint32_t)chosen;
}

uint32_t oc_sample(const float *logits_in, size_t vocab_size,
                   OcSamplerConfig *cfg,
                   const uint32_t *recent_tokens, size_t n_recent)
{
    if (vocab_size == 0) return UINT32_MAX;
    if (cfg == NULL) return oc_argmax(logits_in, vocab_size);

    /* Temperature=0 → greedy (penalty still applies; handled below). */
    bool is_greedy = (cfg->type == OC_SAMPLER_GREEDY ||
                      cfg->temperature <= 0.0f);

    /* If greedy AND no penalty: skip the working copy (fast path). */
    if (is_greedy && (cfg->repeat_penalty == 1.0f || n_recent == 0)) {
        return oc_argmax(logits_in, vocab_size);
    }

    /* Working copy: apply repeat-penalty (for all sampler types, including
     * greedy — mirrors the Rust reference which penalizes before sampling). */
    float *logits = malloc(vocab_size * sizeof(float));
    size_t *idx = malloc(vocab_size * sizeof(size_t));
    if (logits == NULL || idx == NULL) {
        free(logits); free(idx);
        return oc_argmax(logits_in, vocab_size);   /* degrade to greedy on OOM */
    }
    memcpy(logits, logits_in, vocab_size * sizeof(float));

    oc_apply_repeat_penalty(logits, vocab_size, recent_tokens, n_recent,
                            cfg->repeat_penalty);

    if (is_greedy) {
        uint32_t r = oc_argmax(logits, vocab_size);
        free(logits); free(idx);
        return r;
    }

    /* Temperature scaling. */
    float t = cfg->temperature;
    for (size_t i = 0; i < vocab_size; i++) logits[i] /= t;

    uint32_t result;
    uint64_t rng = cfg->seed;
    if (rng == 0) rng = 0xdeadbeefcafef00dULL;

    if (cfg->type == OC_SAMPLER_TEMPERATURE) {
        softmax_inplace(logits, vocab_size);
        for (size_t i = 0; i < vocab_size; i++) idx[i] = i;
        result = sample_categorical(idx, logits, vocab_size, &rng);
    } else if (cfg->type == OC_SAMPLER_TOP_K) {
        size_t k = top_k_select(logits, vocab_size, cfg->top_k, idx);
        /* Softmax over just the top-k candidates. */
        float mx = logits[idx[0]];
        double sum = 0.0;
        float *probs = malloc(k * sizeof(float));
        if (probs != NULL) {
            for (size_t i = 0; i < k; i++) {
                probs[i] = expf(logits[idx[i]] - mx);
                sum += (double)probs[i];
            }
            if (sum > 0.0) {
                float inv = (float)(1.0 / sum);
                for (size_t i = 0; i < k; i++) probs[i] *= inv;
            }
            result = sample_categorical(idx, probs, k, &rng);
            free(probs);
        } else {
            result = (uint32_t)idx[0];
        }
    } else if (cfg->type == OC_SAMPLER_TOP_P) {
        float *probs = malloc(vocab_size * sizeof(float));
        if (probs != NULL) {
            size_t keep = top_p_select(logits, vocab_size, cfg->top_p,
                                       idx, probs);
            result = keep > 0 ? sample_categorical(idx, probs, keep, &rng)
                              : oc_argmax(logits, vocab_size);
            free(probs);
        } else {
            result = oc_argmax(logits, vocab_size);
        }
    } else if (cfg->type == OC_SAMPLER_MIROSTAT_V2) {
        /* Mirostat v2 is inherently stateful across calls (mu is the running surprise estimate). oc_sample is single-shot and takes a const config, so we run one step using cfg->mu as the starting estimate and discard the update. For stateful multi-step decoding, call oc_mirostat_v2_sample directly and persist mu between calls. */
        if (!(cfg->mu > 0.0f)) cfg->mu = 2.0f * cfg->tau;
        result = oc_mirostat_v2_sample(logits, vocab_size, cfg, &cfg->mu, &rng);
    } else if (cfg->type == OC_SAMPLER_MIN_P) {
        float *probs = malloc(vocab_size * sizeof(float));
        if (probs != NULL) {
            softmax_inplace(logits, vocab_size);
            float max_p = 0.0f;
            for (size_t i = 0; i < vocab_size; i++) {
                if (logits[i] > max_p) max_p = logits[i];
            }
            float threshold = max_p * cfg->min_p;
            size_t keep = 0;
            for (size_t i = 0; i < vocab_size; i++) {
                if (logits[i] >= threshold) {
                    idx[keep] = i;
                    probs[keep] = logits[i];
                    keep++;
                }
            }
            if (keep == 0) {
                result = oc_argmax(logits, vocab_size);
            } else {
                double sum = 0.0;
                for (size_t i = 0; i < keep; i++) sum += (double)probs[i];
                if (sum > 0.0) {
                    float inv = (float)(1.0 / sum);
                    for (size_t i = 0; i < keep; i++) probs[i] *= inv;
                }
                result = sample_categorical(idx, probs, keep, &rng);
            }
            free(probs);
        } else {
            result = oc_argmax(logits, vocab_size);
        }
    } else if (cfg->type == OC_SAMPLER_TYPICAL_P) {
        /* Locally typical sampling: compute the negative entropy of each token's information content, then filter by the typical_p cumulative threshold. Tokens with "typical" surprise (close to the entropy) are preferred. */
        if (cfg->typical_p <= 0.0f) {
            softmax_inplace(logits, vocab_size);
            for (size_t i = 0; i < vocab_size; i++) idx[i] = i;
            result = sample_categorical(idx, logits, vocab_size, &rng);
            free(logits);
            free(idx);
            return result;
        }
        float *probs = malloc(vocab_size * sizeof(float));
        if (probs != NULL) {
            softmax_inplace(logits, vocab_size);
            memcpy(probs, logits, vocab_size * sizeof(float));
            /* Compute entropy H = -sum(p * log2(p)). */
            double entropy = 0.0;
            for (size_t i = 0; i < vocab_size; i++) {
                if (probs[i] > 0.0f) {
                    entropy -= (double)probs[i] * log2((double)probs[i]);
                }
            }
            /* Score = |(-log2(p)) - H| (surprise deviation from entropy). */
            for (size_t i = 0; i < vocab_size; i++) {
                float surprise = probs[i] > 0.0f ? -log2f(probs[i]) : 999.0f;
                logits[i] = fabsf(surprise - (float)entropy);
            }
            /* Sort ascending by score (lowest deviation = most typical). */
            /* Simple insertion sort for small vocab; use argsort for large. */
            for (size_t i = 0; i < vocab_size; i++) idx[i] = i;
            /* Partial sort: find the cumulative probability threshold. */
            /* Bubble sort (vocab is small enough in practice). */
            for (size_t i = 0; i < vocab_size - 1; i++) {
                for (size_t j = i + 1; j < vocab_size; j++) {
                    if (logits[idx[j]] < logits[idx[i]]) {
                        uint32_t tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
                    }
                }
            }
            /* Accumulate until typical_p threshold. */
            double cum = 0.0;
            size_t keep = 0;
            for (size_t i = 0; i < vocab_size && cum < (double)cfg->typical_p; i++) {
                probs[keep] = probs[idx[i]];
                idx[keep] = idx[i];
                cum += (double)probs[keep];
                keep++;
            }
            if (keep == 0) {
                result = oc_argmax(logits, vocab_size);
            } else {
                /* Renormalize and sample. */
                double sum = 0.0;
                for (size_t i = 0; i < keep; i++) sum += (double)probs[i];
                if (sum > 0.0) {
                    float inv = (float)(1.0 / sum);
                    for (size_t i = 0; i < keep; i++) probs[i] *= inv;
                }
                result = sample_categorical(idx, probs, keep, &rng);
            }
            free(probs);
        } else {
            result = oc_argmax(logits, vocab_size);
        }
    } else if (cfg->type == OC_SAMPLER_TAIL_FREE) {
        /* Tail-free sampling: compute |d2(p)| (second derivative of sorted
         * probabilities), then filter by the z-threshold. */
        float *probs = malloc(vocab_size * sizeof(float));
        if (probs != NULL) {
            softmax_inplace(logits, vocab_size);
            memcpy(probs, logits, vocab_size * sizeof(float));
            /* Sort by descending probability. */
            for (size_t i = 0; i < vocab_size; i++) idx[i] = i;
            for (size_t i = 0; i < vocab_size - 1; i++) {
                for (size_t j = i + 1; j < vocab_size; j++) {
                    if (probs[idx[j]] > probs[idx[i]]) {
                        uint32_t tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
                    }
                }
            }
            /* First derivative: d1[i] = p[i] - p[i+1]. */
            /* Second derivative: d2[i] = |d1[i] - d1[i+1]|. */
            if (vocab_size < 3) {
                result = oc_argmax(logits, vocab_size);
            } else {
                /* Compute d2 and normalize. */
                float *d2 = malloc((vocab_size - 2) * sizeof(float));
                if (d2 != NULL) {
                    for (size_t i = 0; i < vocab_size - 2; i++) {
                        float d1_a = probs[idx[i]] - probs[idx[i+1]];
                        float d1_b = probs[idx[i+1]] - probs[idx[i+2]];
                        d2[i] = fabsf(d1_a - d1_b);
                    }
                    /* Normalize d2 to sum=1. */
                    float d2_sum = 0.0f;
                    for (size_t i = 0; i < vocab_size - 2; i++) d2_sum += d2[i];
                    if (d2_sum > 0.0f) {
                        for (size_t i = 0; i < vocab_size - 2; i++) d2[i] /= d2_sum;
                    }
                    /* Accumulate until z-threshold. */
                    float cum = 0.0f;
                    size_t keep = 0;
                    for (size_t i = 0; i < vocab_size - 2 && cum < cfg->tail_free_z; i++) {
                        cum += d2[i];
                        keep++;
                    }
                    keep++; /* Keep at least one token. */
                    if (keep > vocab_size) keep = vocab_size;
                    /* Renormalize and sample from the kept set. */
                    double sum = 0.0;
                    for (size_t i = 0; i < keep; i++) sum += (double)probs[idx[i]];
                    if (sum > 0.0) {
                        float inv = (float)(1.0 / sum);
                        for (size_t i = 0; i < keep; i++) probs[i] = logits[idx[i]] * inv;
                    }
                    result = sample_categorical(idx, probs, keep, &rng);
                    free(d2);
                } else {
                    result = oc_argmax(logits, vocab_size);
                }
            }
            free(probs);
        } else {
            result = oc_argmax(logits, vocab_size);
        }
    } else {
        result = oc_argmax(logits, vocab_size);
    }

    free(logits);
    free(idx);
    return result;
}


OcError oc_softmax_probs(const float *logits, size_t vocab_size,
                          float temperature, float *out)
{
    if (!logits || !out || vocab_size == 0)
        return OC_ERR_INVALID_ARG;

    if (temperature <= 0.0f)
        temperature = 1.0f;

    /* Find max for numerical stability. */
    float max_val = logits[0];
    for (size_t i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val)
            max_val = logits[i];
    }

    /* Compute exp((x - max) / T) and sum. */
    double sum = 0.0;
    for (size_t i = 0; i < vocab_size; i++) {
        float v = expf((logits[i] - max_val) / temperature);
        out[i] = v;
        sum += v;
    }

    /* Normalize. */
    if (sum > 0.0) {
        float inv_sum = (float)(1.0 / sum);
        for (size_t i = 0; i < vocab_size; i++)
            out[i] *= inv_sum;
    }

    return OC_OK;
}


void oc_residual_probs(const float *target_probs, const float *draft_probs,
                        float *out, size_t vocab_size)
{
    if (!target_probs || !draft_probs || !out || vocab_size == 0) return;

    float sum = 0.0f;
    for (size_t i = 0; i < vocab_size; i++) {
        float diff = target_probs[i] - draft_probs[i];
        out[i] = diff > 0.0f ? diff : 0.0f;
        sum += out[i];
    }

    if (sum > 0.0f && isfinite(sum)) {
        float inv_sum = 1.0f / sum;
        for (size_t i = 0; i < vocab_size; i++)
            out[i] *= inv_sum;
    }
}

size_t oc_sample_probabilities(const float *probs, size_t vocab_size, float random)
{
    if (!probs || vocab_size == 0) return 0;
    if (!isfinite(random) || random < 0.0f || random >= 1.0f) return 0;

    float sum = 0.0f;
    for (size_t i = 0; i < vocab_size; i++)
        sum += probs[i];

    if (sum <= 0.0f || !isfinite(sum)) {
        /* Fall back to argmax. */
        size_t best = 0;
        float best_v = probs[0];
        for (size_t i = 1; i < vocab_size; i++) {
            if (probs[i] > best_v) {
                best_v = probs[i];
                best = i;
            }
        }
        return best;
    }

    float target = random * sum;
    float cumulative = 0.0f;
    for (size_t i = 0; i < vocab_size; i++) {
        cumulative += probs[i];
        if (target <= cumulative)
            return i;
    }
    return vocab_size - 1;
}


void oc_apply_repetition_penalties(float *logits, size_t vocab_size,
                                    const uint32_t *recent, size_t n_recent,
                                    const OcRepetitionPenaltyConfig *cfg)
{
    if (!logits || !cfg || vocab_size == 0) return;
    if (cfg->frequency_penalty == 0.0f && cfg->presence_penalty == 0.0f &&
        cfg->newline_token_id == 0xFFFFFFFFu) return;
    if (!recent || n_recent == 0) return;

    /* Count frequencies. */
    uint32_t *freqs = calloc(vocab_size, sizeof(uint32_t));
    if (!freqs) return;

    for (size_t i = 0; i < n_recent; i++) {
        size_t idx = recent[i];
        if (idx < vocab_size)
            freqs[idx]++;
    }

    for (size_t i = 0; i < vocab_size; i++) {
        if (freqs[i] == 0) continue;
        logits[i] -= cfg->frequency_penalty * (float)freqs[i];
        logits[i] -= cfg->presence_penalty;
    }

    /* Newline penalty. */
    if (cfg->newline_token_id != 0xFFFFFFFFu) {
        size_t nl = cfg->newline_token_id;
        if (nl < vocab_size)
            logits[nl] -= cfg->newline_penalty;
    }

    free(freqs);
}


typedef struct {
    uint32_t *tokens;
    size_t    n_tokens;
    float     score;
    bool      finished;
} OcBeam;

static void beam_free(OcBeam *b)
{
    if (b->tokens) free(b->tokens);
    b->tokens = NULL;
    b->n_tokens = 0;
}

static int compare_beams(const void *a, const void *b)
{
    const OcBeam *ba = (const OcBeam *)a;
    const OcBeam *bb = (const OcBeam *)b;
    if (ba->score > bb->score) return -1;
    if (ba->score < bb->score) return 1;
    return 0;
}

OcError oc_beam_search(float * const *logits_per_step, size_t n_steps,
                        size_t vocab_size, size_t beam_width,
                        uint32_t eos_token, OcBeamSearchResult *out)
{
    OcError e = OC_OK;
    if (!logits_per_step || !out) return OC_ERR_INVALID_ARG;
    if (beam_width == 0) return OC_ERR_INVALID_ARG;
    if (n_steps == 0 || vocab_size == 0) return OC_ERR_INVALID_ARG;

    /* Check for empty logits. */
    for (size_t i = 0; i < n_steps; i++) {
        if (!logits_per_step[i]) return OC_ERR_INVALID_ARG;
    }

    /* Allocate working beams. */
    OcBeam *beams = calloc(beam_width, sizeof(OcBeam));
    if (!beams) return OC_ERR_OOM;
    beams[0].tokens = NULL;
    beams[0].n_tokens = 0;
    beams[0].score = 0.0f;
    beams[0].finished = false;

    size_t n_active_beams = 1;
    float *probs = malloc(vocab_size * sizeof(float));
    if (!probs) { free(beams); return OC_ERR_OOM; }

    /* Maximum candidates: n_active_beams * vocab_size. */
    size_t max_candidates = beam_width * vocab_size;
    OcBeam *candidates = malloc(max_candidates * sizeof(OcBeam));
    if (!candidates) { free(probs); free(beams); return OC_ERR_OOM; }

    for (size_t step = 0; step < n_steps; step++) {
        OcError se = oc_softmax_probs(logits_per_step[step], vocab_size, 1.0f, probs);
        if (se != OC_OK) { e = se; goto cleanup; }

        size_t n_candidates = 0;

        for (size_t bi = 0; bi < n_active_beams; bi++) {
            OcBeam *beam = &beams[bi];
            if (beam->finished) {
                if (n_candidates < max_candidates) {
                    candidates[n_candidates].tokens = malloc((beam->n_tokens + 1) * sizeof(uint32_t));
                    if (candidates[n_candidates].tokens) {
                        if (beam->n_tokens > 0) {
                            memcpy(candidates[n_candidates].tokens, beam->tokens,
                                   beam->n_tokens * sizeof(uint32_t));
                        }
                        candidates[n_candidates].n_tokens = beam->n_tokens;
                        candidates[n_candidates].score = beam->score;
                        candidates[n_candidates].finished = true;
                        n_candidates++;
                    }
                }
                continue;
            }

            for (size_t ti = 0; ti < vocab_size; ti++) {
                float prob = probs[ti];
                if (prob <= 0.0f || !isfinite(prob)) continue;

                if (n_candidates >= max_candidates) break;

                size_t new_len = beam->n_tokens + 1;
                candidates[n_candidates].tokens = malloc(new_len * sizeof(uint32_t));
                if (!candidates[n_candidates].tokens) continue;

                if (beam->n_tokens > 0) {
                    memcpy(candidates[n_candidates].tokens, beam->tokens,
                           beam->n_tokens * sizeof(uint32_t));
                }
                candidates[n_candidates].tokens[beam->n_tokens] = (uint32_t)ti;
                candidates[n_candidates].n_tokens = new_len;
                candidates[n_candidates].score = beam->score + logf(prob);
                candidates[n_candidates].finished = (eos_token != 0xFFFFFFFFu && (uint32_t)ti == eos_token);
                n_candidates++;
            }
        }

        if (n_candidates == 0) {
            e = OC_ERR_INVALID_ARG;
            goto cleanup;
        }

        /* Sort by score descending. */
        qsort(candidates, n_candidates, sizeof(OcBeam), compare_beams);

        /* Free old beams. */
        for (size_t bi = 0; bi < n_active_beams; bi++)
            beam_free(&beams[bi]);

        /* Take top beam_width beams. */
        size_t take = n_candidates < beam_width ? n_candidates : beam_width;
        for (size_t bi = 0; bi < take; bi++) {
            beams[bi] = candidates[bi];
        }
        n_active_beams = take;

        /* The top `take` candidates were moved into `beams` above and are now owned there. Everything past `take` lost the cut and still owns a tokens allocation — free it before the memset below erases the pointer, or it leaks once per round. */
        for (size_t bi = take; bi < n_candidates; bi++)
            beam_free(&candidates[bi]);

        /* Clear candidates pointers (moved to beams). */
        memset(candidates, 0, max_candidates * sizeof(OcBeam));

        /* Check if all finished. */
        bool all_finished = true;
        for (size_t bi = 0; bi < n_active_beams; bi++) {
            if (!beams[bi].finished) { all_finished = false; break; }
        }
        if (all_finished) break;
    }

    /* Find best beam. */
    size_t best_idx = 0;
    float best_score = beams[0].score;
    for (size_t bi = 1; bi < n_active_beams; bi++) {
        if (beams[bi].score > best_score) {
            best_score = beams[bi].score;
            best_idx = bi;
        }
    }

    out->tokens = beams[best_idx].tokens;
    out->n_tokens = beams[best_idx].n_tokens;
    out->score = beams[best_idx].score;

    /* Mark as moved so cleanup doesn't free it. */
    beams[best_idx].tokens = NULL;
    beams[best_idx].n_tokens = 0;

    e = OC_OK;

cleanup:
    for (size_t bi = 0; bi < n_active_beams; bi++)
        beam_free(&beams[bi]);
    for (size_t ci = 0; ci < max_candidates; ci++)
        beam_free(&candidates[ci]);
    free(candidates);
    free(probs);
    free(beams);
    return e;
}

void oc_beam_search_result_free(OcBeamSearchResult *result)
{
    if (!result) return;
    if (result->tokens) free(result->tokens);
    result->tokens = NULL;
    result->n_tokens = 0;
    result->score = 0.0f;
}
