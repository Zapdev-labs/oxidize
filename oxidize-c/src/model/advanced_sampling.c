/*
 * advanced_sampling.c — Advanced sampling strategies implementation.
 */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/advanced_sampling.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>


typedef struct { float logit; size_t idx; } LogitIndex;

static int cmp_logit_desc(const void *a, const void *b)
{
    float la = ((const LogitIndex *)a)->logit;
    float lb = ((const LogitIndex *)b)->logit;
    if (la > lb) return -1;
    if (la < lb) return 1;
    return 0;
}

static LogitIndex *sort_logits(const float *logits, size_t vocab_size)
{
    LogitIndex *arr = malloc(vocab_size * sizeof(LogitIndex));
    if (!arr) return NULL;
    for (size_t i = 0; i < vocab_size; i++) {
        arr[i].logit = logits[i];
        arr[i].idx = i;
    }
    qsort(arr, vocab_size, sizeof(LogitIndex), cmp_logit_desc);
    return arr;
}

static float compute_entropy(const float *probs, size_t n)
{
    float h = 0.0f;
    for (size_t i = 0; i < n; i++) {
        if (probs[i] > 0.0f)
            h -= probs[i] * logf(probs[i]);
    }
    return h;
}

static uint32_t sample_from_probs(const float *probs, size_t n, float u)
{
    /* Inverse CDF sampling. u is a uniform [0,1) random value. */
    float cum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        cum += probs[i];
        if (u <= cum) return (uint32_t)i;
    }
    return (uint32_t)(n - 1);
}

static float simple_rng(void)
{
    static uint32_t state = 12345;
    state = state * 1103515245u + 12345u;
    return (float)(state >> 8) / (float)(1u << 24);
}

/* Draw from a caller-owned RNG state when provided (reproducible and
 * race-free); fall back to the global RNG otherwise. */
static float rng_from_state(uint32_t *state)
{
    if (!state) return simple_rng();
    *state = *state * 1103515245u + 12345u;
    return (float)(*state >> 8) / (float)(1u << 24);
}


uint32_t oc_sample_mirostat_v1(const float *logits, size_t vocab_size,
                                float *mu, float tau, float eta,
                                uint32_t *state)
{
    if (!logits || vocab_size == 0 || !mu) return 0;

    /* Sort logits, compute softmax for top candidates. */
    LogitIndex *sorted = sort_logits(logits, vocab_size);
    if (!sorted) return 0;

    /* Compute k based on mu. */
    float n = (float)vocab_size;
    float k = powf(2.0f * *mu, 2.0f) / n;
    if (k < 1.0f) k = 1.0f;
    if (k > (float)vocab_size) k = (float)vocab_size;
    size_t top_k = (size_t)k;

    /* Compute softmax over top-k. */
    float max_logit = sorted[0].logit;
    float sum = 0.0f;
    float *probs = malloc(top_k * sizeof(float));
    if (!probs) { free(sorted); return 0; }

    for (size_t i = 0; i < top_k; i++) {
        probs[i] = expf(sorted[i].logit - max_logit);
        sum += probs[i];
    }
    if (sum == 0.0f) sum = 1.0f;
    for (size_t i = 0; i < top_k; i++) probs[i] /= sum;

    /* Sample, driving the draw from the caller's RNG state. */
    float u = rng_from_state(state);
    size_t selected = sample_from_probs(probs, top_k, u);
    uint32_t token = (uint32_t)sorted[selected].idx;

    /* Compute observed surprise and update mu. */
    float observed_surprise = -logf(probs[selected] + 1e-10f);
    float error = observed_surprise - tau;
    *mu = *mu - eta * error;

    free(probs);
    free(sorted);
    return token;
}


uint32_t oc_sample_mirostat_v2(const float *logits, size_t vocab_size,
                                float *mu, float tau, float eta,
                                uint32_t *state)
{
    if (!logits || vocab_size == 0 || !mu) return 0;

    /* Compute full softmax. */
    float max_logit = logits[0];
    for (size_t i = 1; i < vocab_size; i++)
        if (logits[i] > max_logit) max_logit = logits[i];

    float *probs = calloc(vocab_size, sizeof(float));
    if (!probs) return 0;

    float sum = 0.0f;
    for (size_t i = 0; i < vocab_size; i++) {
        probs[i] = expf(logits[i] - max_logit);
        sum += probs[i];
    }
    if (sum == 0.0f) sum = 1.0f;
    for (size_t i = 0; i < vocab_size; i++) probs[i] /= sum;

    /* Truncate: keep tokens whose individual surprise -log(p) <= mu,
     * i.e. p >= exp(-mu). Always keep at least the top token. */
    float min_p = expf(-*mu);
    size_t top_k = 0;

    LogitIndex *sorted = sort_logits(logits, vocab_size);
    if (!sorted) { free(probs); return 0; }

    for (size_t i = 0; i < vocab_size; i++) {
        if (probs[sorted[i].idx] < min_p && top_k > 0) break;
        top_k = i + 1;
    }

    /* Normalize and sample. */
    float *top_probs = malloc(top_k * sizeof(float));
    if (!top_probs) { free(sorted); free(probs); return 0; }

    float top_sum = 0.0f;
    for (size_t i = 0; i < top_k; i++) {
        top_probs[i] = probs[sorted[i].idx];
        top_sum += top_probs[i];
    }
    if (top_sum == 0.0f) top_sum = 1.0f;
    for (size_t i = 0; i < top_k; i++) top_probs[i] /= top_sum;

    float u = rng_from_state(state);
    size_t selected = sample_from_probs(top_probs, top_k, u);
    uint32_t token = (uint32_t)sorted[selected].idx;

    /* Update mu using the surprise from the pre-truncation distribution. */
    float observed_surprise = -logf(probs[sorted[selected].idx] + 1e-10f);
    float error = observed_surprise - tau;
    *mu = *mu - eta * error;

    free(top_probs);
    free(sorted);
    free(probs);
    return token;
}


uint32_t oc_sample_tfs(const float *logits, size_t vocab_size,
                       float z, float temperature)
{
    if (!logits || vocab_size == 0) return 0;
    if (z <= 0.0f || z >= 1.0f) z = 0.95f;
    if (temperature <= 0.0f) temperature = 1.0f;

    LogitIndex *sorted = sort_logits(logits, vocab_size);
    if (!sorted) return 0;

    /* Apply temperature and compute softmax. */
    float max_logit = sorted[0].logit / temperature;
    float *probs = calloc(vocab_size, sizeof(float));
    if (!probs) { free(sorted); return 0; }

    float sum = 0.0f;
    for (size_t i = 0; i < vocab_size; i++) {
        probs[i] = expf(sorted[i].logit / temperature - max_logit);
        sum += probs[i];
    }
    if (sum == 0.0f) sum = 1.0f;
    for (size_t i = 0; i < vocab_size; i++) probs[i] /= sum;

    /* Compute first derivative (differences between consecutive probs). */
    /* Compute second derivative (differences of first derivative). */
    /* Find the cutoff point where |second_derivative| < z * max. */
    if (vocab_size < 3) {
        uint32_t token = (uint32_t)sorted[0].idx;
        free(probs); free(sorted);
        return token;
    }

    float max_second_deriv = 0.0f;
    for (size_t i = 1; i + 1 < vocab_size; i++) {
        float d1_prev = probs[i] - probs[i - 1];
        float d1_curr = probs[i + 1] - probs[i];
        float d2 = fabsf(d1_curr - d1_prev);
        if (d2 > max_second_deriv) max_second_deriv = d2;
    }

    float threshold = z * max_second_deriv;
    size_t cutoff = vocab_size;
    for (size_t i = 1; i + 1 < vocab_size; i++) {
        float d1_prev = probs[i] - probs[i - 1];
        float d1_curr = probs[i + 1] - probs[i];
        float d2 = fabsf(d1_curr - d1_prev);
        if (d2 < threshold) {
            cutoff = i;
            break;
        }
    }

    /* Normalize and sample from top-cutoff. */
    float top_sum = 0.0f;
    for (size_t i = 0; i < cutoff; i++)
        top_sum += probs[i];
    if (top_sum == 0.0f) top_sum = 1.0f;

    float u = simple_rng() * top_sum;
    float cum = 0.0f;
    for (size_t i = 0; i < cutoff; i++) {
        cum += probs[i];
        if (u <= cum) {
            uint32_t token = (uint32_t)sorted[i].idx;
            free(probs); free(sorted);
            return token;
        }
    }

    uint32_t token = (uint32_t)sorted[0].idx;
    free(probs); free(sorted);
    return token;
}


uint32_t oc_sample_typical(const float *logits, size_t vocab_size,
                            float p, float temperature)
{
    if (!logits || vocab_size == 0) return 0;
    if (p <= 0.0f || p >= 1.0f) p = 0.95f;

    /* Apply temperature. */
    float *adjusted = malloc(vocab_size * sizeof(float));
    if (!adjusted) return 0;

    float inv_t = 1.0f / (temperature > 0.0f ? temperature : 1.0f);
    float max_logit = logits[0] * inv_t;
    for (size_t i = 1; i < vocab_size; i++)
        if (logits[i] * inv_t > max_logit) max_logit = logits[i] * inv_t;

    float sum = 0.0f;
    for (size_t i = 0; i < vocab_size; i++) {
        adjusted[i] = expf(logits[i] * inv_t - max_logit);
        sum += adjusted[i];
    }
    if (sum == 0.0f) sum = 1.0f;
    for (size_t i = 0; i < vocab_size; i++) adjusted[i] /= sum;

    /* Compute entropy. */
    float entropy = compute_entropy(adjusted, vocab_size);

    /* Compute |H - (-log p_i)| for each token. */
    LogitIndex *sorted = malloc(vocab_size * sizeof(LogitIndex));
    if (!sorted) { free(adjusted); return 0; }

    for (size_t i = 0; i < vocab_size; i++) {
        float neg_log_p = -logf(adjusted[i] + 1e-10f);
        sorted[i].logit = fabsf(entropy - neg_log_p); /* sort ascending */
        sorted[i].idx = i;
    }

    /* Sort ascending by |H - neg_log_p|. */
    qsort(sorted, vocab_size, sizeof(LogitIndex), cmp_logit_desc);
    /* Reverse: we want ascending, but cmp sorts descending. Re-interpret. */
    /* Actually we want tokens closest to the entropy first. Let's sort
     * by the "typicality score" ascending. Create a proper sort. */

    /* Re-sort ascending. */
    for (size_t i = 0; i < vocab_size / 2; i++) {
        LogitIndex tmp = sorted[i];
        sorted[i] = sorted[vocab_size - 1 - i];
        sorted[vocab_size - 1 - i] = tmp;
    }

    /* Accumulate until p. */
    float cum = 0.0f;
    size_t cutoff = 0;
    for (size_t i = 0; i < vocab_size; i++) {
        cum += adjusted[sorted[i].idx];
        cutoff = i + 1;
        if (cum >= p) break;
    }

    /* Sample from the typical set. */
    float u = simple_rng() * cum;
    cum = 0.0f;
    for (size_t i = 0; i < cutoff; i++) {
        cum += adjusted[sorted[i].idx];
        if (u <= cum) {
            uint32_t token = (uint32_t)sorted[i].idx;
            free(adjusted); free(sorted);
            return token;
        }
    }

    uint32_t token = (uint32_t)sorted[0].idx;
    free(adjusted); free(sorted);
    return token;
}


uint32_t oc_sample_top_a(const float *logits, size_t vocab_size,
                          float a, float temperature)
{
    if (!logits || vocab_size == 0) return 0;
    if (a <= 0.0f) a = 1.0f;
    if (temperature <= 0.0f) temperature = 1.0f;

    LogitIndex *sorted = sort_logits(logits, vocab_size);
    if (!sorted) return 0;

    float inv_t = 1.0f / temperature;
    float max_logit = sorted[0].logit * inv_t;
    float sum = 0.0f;
    float *probs = calloc(vocab_size, sizeof(float));
    if (!probs) { free(sorted); return 0; }

    for (size_t i = 0; i < vocab_size; i++) {
        probs[i] = expf(sorted[i].logit * inv_t - max_logit);
        sum += probs[i];
    }
    if (sum == 0.0f) sum = 1.0f;
    for (size_t i = 0; i < vocab_size; i++) probs[i] /= sum;

    /* Threshold = max_prob * a^2. */
    float threshold = probs[0] * a * a;

    size_t cutoff = 0;
    float top_sum = 0.0f;
    for (size_t i = 0; i < vocab_size; i++) {
        if (probs[i] < threshold) break;
        cutoff = i + 1;
        top_sum += probs[i];
    }

    if (cutoff == 0) cutoff = 1;
    if (top_sum == 0.0f) top_sum = 1.0f;

    float u = simple_rng() * top_sum;
    float cum = 0.0f;
    for (size_t i = 0; i < cutoff; i++) {
        cum += probs[i];
        if (u <= cum) {
            uint32_t token = (uint32_t)sorted[i].idx;
            free(probs); free(sorted);
            return token;
        }
    }

    uint32_t token = (uint32_t)sorted[0].idx;
    free(probs); free(sorted);
    return token;
}


uint32_t oc_sample_eta_cutoff(const float *logits, size_t vocab_size,
                                float epsilon, float temperature)
{
    if (!logits || vocab_size == 0) return 0;
    if (epsilon <= 0.0f || epsilon >= 1.0f) epsilon = 1e-4f;
    if (temperature <= 0.0f) temperature = 1.0f;

    /* Compute min probability threshold. */
    float eta = fminf(epsilon, sqrtf(epsilon / (float)vocab_size));

    LogitIndex *sorted = sort_logits(logits, vocab_size);
    if (!sorted) return 0;

    float inv_t = 1.0f / temperature;
    float max_logit = sorted[0].logit * inv_t;
    float sum = 0.0f;
    float *probs = calloc(vocab_size, sizeof(float));
    if (!probs) { free(sorted); return 0; }

    for (size_t i = 0; i < vocab_size; i++) {
        probs[i] = expf(sorted[i].logit * inv_t - max_logit);
        sum += probs[i];
    }
    if (sum == 0.0f) sum = 1.0f;
    for (size_t i = 0; i < vocab_size; i++) probs[i] /= sum;

    /* Keep tokens with prob >= eta. */
    size_t cutoff = 0;
    float top_sum = 0.0f;
    for (size_t i = 0; i < vocab_size; i++) {
        if (probs[i] < eta) break;
        cutoff = i + 1;
        top_sum += probs[i];
    }

    if (cutoff == 0) cutoff = 1;
    if (top_sum == 0.0f) top_sum = 1.0f;

    float u = simple_rng() * top_sum;
    float cum = 0.0f;
    for (size_t i = 0; i < cutoff; i++) {
        cum += probs[i];
        if (u <= cum) {
            uint32_t token = (uint32_t)sorted[i].idx;
            free(probs); free(sorted);
            return token;
        }
    }

    uint32_t token = (uint32_t)sorted[0].idx;
    free(probs); free(sorted);
    return token;
}


void oc_apply_penalties(float *logits, size_t vocab_size,
                        const uint32_t *recent_tokens, size_t n_recent,
                        float frequency_penalty, float presence_penalty)
{
    if (!logits || !recent_tokens || n_recent == 0) return;

    /* Count token frequencies. */
    uint32_t *counts = calloc(vocab_size, sizeof(uint32_t));
    if (!counts) return;

    for (size_t i = 0; i < n_recent; i++) {
        if (recent_tokens[i] < vocab_size)
            counts[recent_tokens[i]]++;
    }

    /* Apply penalties. */
    for (size_t i = 0; i < vocab_size; i++) {
        if (counts[i] > 0) {
            logits[i] -= frequency_penalty * (float)counts[i];
            logits[i] -= presence_penalty;
        }
    }

    free(counts);
}


OcError oc_beam_search_init(OcBeamSearchState *st, size_t beam_width,
                             size_t max_length, float length_penalty,
                             uint32_t eos_token, uint32_t first_token)
{
    if (!st || beam_width == 0 || max_length == 0) return OC_ERR_INVALID_ARG;
    st->beams = calloc(beam_width, sizeof(OcBeam));
    if (!st->beams) return OC_ERR_OOM;
    st->beam_width = beam_width;
    st->max_length = max_length;
    st->length_penalty = length_penalty;
    st->eos_token = eos_token;
    st->n_finished = 0;

    /* Initialize first beam with the first token. */
    st->beams[0].tokens = malloc((max_length + 1) * sizeof(uint32_t));
    if (!st->beams[0].tokens) { free(st->beams); st->beams = NULL; return OC_ERR_OOM; }
    st->beams[0].tokens[0] = first_token;
    st->beams[0].n_tokens = 1;
    st->beams[0].log_prob = 0.0f;
    st->beams[0].finished = false;

    /* Other beams start empty (log_prob = -inf). */
    for (size_t i = 1; i < beam_width; i++) {
        st->beams[i].tokens = malloc((max_length + 1) * sizeof(uint32_t));
        if (!st->beams[i].tokens) {
            for (size_t j = 0; j < i; j++) free(st->beams[j].tokens);
            free(st->beams); st->beams = NULL;
            return OC_ERR_OOM;
        }
        st->beams[i].n_tokens = 0;
        st->beams[i].log_prob = -INFINITY;
        st->beams[i].finished = false;
    }

    return OC_OK;
}

OcError oc_beam_search_step(OcBeamSearchState *st,
                             const float *logits_per_beam,
                             size_t vocab_size)
{
    if (!st || !logits_per_beam || vocab_size == 0)
        return OC_ERR_INVALID_ARG;

    /* For each active beam, compute top beam_width candidates. */
    typedef struct { float score; size_t beam_idx; uint32_t token; } Candidate;
    Candidate *candidates = malloc(st->beam_width * st->beam_width * sizeof(Candidate));
    if (!candidates) return OC_ERR_OOM;
    size_t n_cands = 0;

    for (size_t b = 0; b < st->beam_width; b++) {
        if (st->beams[b].finished) continue;
        const float *logits = logits_per_beam + b * vocab_size;

        /* Softmax to get log probs. */
        float max_logit = logits[0];
        for (size_t i = 1; i < vocab_size; i++)
            if (logits[i] > max_logit) max_logit = logits[i];
        float sum = 0.0f;
        for (size_t i = 0; i < vocab_size; i++)
            sum += expf(logits[i] - max_logit);
        if (sum == 0) sum = 1.0f;

        /* Get top beam_width tokens. */
        LogitIndex *sorted = sort_logits(logits, vocab_size);
        if (!sorted) continue;

        for (size_t k = 0; k < st->beam_width && k < vocab_size; k++) {
            float log_prob = sorted[k].logit - max_logit - logf(sum);
            candidates[n_cands].score = st->beams[b].log_prob + log_prob;
            candidates[n_cands].beam_idx = b;
            candidates[n_cands].token = (uint32_t)sorted[k].idx;
            n_cands++;
        }
        free(sorted);
    }

    /* Sort candidates by score descending. */
    for (size_t i = 0; i < n_cands; i++) {
        for (size_t j = i + 1; j < n_cands; j++) {
            if (candidates[j].score > candidates[i].score) {
                Candidate tmp = candidates[i];
                candidates[i] = candidates[j];
                candidates[j] = tmp;
            }
        }
    }

    /* Select top beam_width. */
    OcBeam *new_beams = calloc(st->beam_width, sizeof(OcBeam));
    if (!new_beams) { free(candidates); return OC_ERR_OOM; }

    size_t selected = 0;

    /* Carry finished hypotheses forward so they are not lost. */
    for (size_t b = 0; b < st->beam_width && selected < st->beam_width; b++) {
        if (!st->beams[b].finished) continue;
        new_beams[selected].tokens = malloc((st->max_length + 1) * sizeof(uint32_t));
        if (!new_beams[selected].tokens) continue;
        memcpy(new_beams[selected].tokens, st->beams[b].tokens,
               st->beams[b].n_tokens * sizeof(uint32_t));
        new_beams[selected].n_tokens = st->beams[b].n_tokens;
        new_beams[selected].log_prob = st->beams[b].log_prob;
        new_beams[selected].finished = true;
        selected++;
    }

    for (size_t i = 0; i < n_cands && selected < st->beam_width; i++) {
        size_t src = candidates[i].beam_idx;
        uint32_t token = candidates[i].token;

        new_beams[selected].tokens = malloc((st->max_length + 1) * sizeof(uint32_t));
        if (!new_beams[selected].tokens) continue;

        /* Copy source beam tokens. */
        memcpy(new_beams[selected].tokens, st->beams[src].tokens,
               st->beams[src].n_tokens * sizeof(uint32_t));
        new_beams[selected].n_tokens = st->beams[src].n_tokens;
        new_beams[selected].tokens[new_beams[selected].n_tokens++] = token;
        new_beams[selected].log_prob = candidates[i].score;
        new_beams[selected].finished = (token == st->eos_token);
        selected++;
    }

    /* If no candidates, keep old beams. */
    if (selected == 0) {
        free(new_beams);
        free(candidates);
        return OC_OK;
    }

    /* Fill remaining beams with -inf. */
    for (size_t i = selected; i < st->beam_width; i++) {
        new_beams[i].tokens = malloc((st->max_length + 1) * sizeof(uint32_t));
        new_beams[i].n_tokens = 0;
        new_beams[i].log_prob = -INFINITY;
        new_beams[i].finished = false;
    }

    /* Swap. */
    for (size_t i = 0; i < st->beam_width; i++)
        free(st->beams[i].tokens);
    free(st->beams);
    st->beams = new_beams;

    /* Recompute finished count from the retained beams. */
    st->n_finished = 0;
    for (size_t i = 0; i < st->beam_width; i++)
        if (st->beams[i].finished) st->n_finished++;

    free(candidates);
    return OC_OK;
}

bool oc_beam_search_done(const OcBeamSearchState *st)
{
    if (!st) return true;
    if (st->n_finished >= st->beam_width) return true;
    for (size_t i = 0; i < st->beam_width; i++)
        if (!st->beams[i].finished && st->beams[i].n_tokens < st->max_length)
            return false;
    return true;
}

const OcBeam *oc_beam_search_best(const OcBeamSearchState *st)
{
    if (!st || !st->beams) return NULL;
    const OcBeam *best = &st->beams[0];
    for (size_t i = 1; i < st->beam_width; i++) {
        if (st->beams[i].n_tokens > 0) {
            float score = st->beams[i].log_prob / powf((float)st->beams[i].n_tokens, st->length_penalty);
            float best_score = best->log_prob / powf((float)best->n_tokens, st->length_penalty);
            if (score > best_score) best = &st->beams[i];
        }
    }
    return best;
}

void oc_beam_search_free(OcBeamSearchState *st)
{
    if (!st) return;
    if (!st->beams) { st->n_finished = 0; return; } /* idempotent free */
    for (size_t i = 0; i < st->beam_width; i++)
        free(st->beams[i].tokens);
    free(st->beams);
    st->beams = NULL;
    st->n_finished = 0;
}


uint32_t oc_sample_contrastive(const float *logits, size_t vocab_size,
                                const float *past_keys,
                                const float *current_key,
                                size_t seq_len, size_t hidden_dim,
                                float alpha, float beta)
{
    if (!logits || vocab_size == 0) return 0;
    /* Contrastive search: for each top-k candidate, compute similarity
     * to past keys and penalize high-similarity tokens.
     * alpha = degeneration penalty, beta = contrastive strength. */

    LogitIndex *sorted = sort_logits(logits, vocab_size);
    if (!sorted) return 0;

    /* Get top candidates. */
    size_t top_k = (size_t)(alpha * (float)vocab_size);
    if (top_k < 1) top_k = 1;
    if (top_k > vocab_size) top_k = vocab_size;
    if (top_k > 10) top_k = 10; /* limit for performance */

    /* Softmax probabilities over the top-k candidates. */
    float max_logit = sorted[0].logit;
    float sum = 0.0f;
    float cand_probs[10];
    for (size_t i = 0; i < top_k; i++) {
        cand_probs[i] = expf(sorted[i].logit - max_logit);
        sum += cand_probs[i];
    }
    if (sum == 0.0f) sum = 1.0f;
    for (size_t i = 0; i < top_k; i++) cand_probs[i] /= sum;

    /* Degeneration penalty: max cosine similarity between the current */
    float max_sim = 0.0f;
    if (past_keys && current_key) {
        for (size_t t = 0; t < seq_len; t++) {
            float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
            for (size_t d = 0; d < hidden_dim; d++) {
                dot += current_key[d] * past_keys[t * hidden_dim + d];
                norm_a += current_key[d] * current_key[d];
                norm_b += past_keys[t * hidden_dim + d] * past_keys[t * hidden_dim + d];
            }
            float sim = dot / (sqrtf(norm_a) * sqrtf(norm_b) + 1e-10f);
            if (sim > max_sim) max_sim = sim;
        }
    }

    float best_score = -INFINITY;
    uint32_t best_token = (uint32_t)sorted[0].idx;

    for (size_t i = 0; i < top_k; i++) {
        /* Score = prob * beta - max_sim * (1 - beta). */
        float score = cand_probs[i] * beta - max_sim * (1.0f - beta);
        if (score > best_score) {
            best_score = score;
            best_token = (uint32_t)sorted[i].idx;
        }
    }

    free(sorted);
    return best_token;
}


void oc_sampler_chain_init(OcSamplerChain *chain)
{
    if (!chain) return;
    chain->steps = NULL;
    chain->n_steps = 0;
    chain->capacity = 0;
    chain->rng_state = 12345u;
}

OcError oc_sampler_chain_add(OcSamplerChain *chain,
                              OcSamplerStepType type,
                              float param1, float param2)
{
    if (!chain) return OC_ERR_INVALID_ARG;
    if (chain->n_steps >= chain->capacity) {
        size_t new_cap = chain->capacity ? chain->capacity * 2 : 8;
        OcSamplerStep *p = realloc(chain->steps, new_cap * sizeof(OcSamplerStep));
        if (!p) return OC_ERR_OOM;
        chain->steps = p;
        chain->capacity = new_cap;
    }
    chain->steps[chain->n_steps].type = type;
    chain->steps[chain->n_steps].param1 = param1;
    chain->steps[chain->n_steps].param2 = param2;
    chain->n_steps++;
    return OC_OK;
}

/* Mask all but the k highest logits to -inf. */
static void filter_top_k(float *logits, size_t vocab_size, size_t k)
{
    if (k == 0 || k >= vocab_size) return;
    LogitIndex *sorted = sort_logits(logits, vocab_size);
    if (!sorted) return;
    for (size_t i = k; i < vocab_size; i++)
        logits[sorted[i].idx] = -INFINITY;
    free(sorted);
}

/* Mask tokens outside the smallest set with cumulative probability >= p. */
static void filter_top_p(float *logits, size_t vocab_size, float p)
{
    if (p <= 0.0f || p >= 1.0f) return;
    LogitIndex *sorted = sort_logits(logits, vocab_size);
    if (!sorted) return;

    float max_logit = sorted[0].logit;
    float sum = 0.0f;
    for (size_t i = 0; i < vocab_size; i++)
        sum += expf(sorted[i].logit - max_logit);
    if (sum == 0.0f) sum = 1.0f;

    float cum = 0.0f;
    size_t keep = vocab_size;
    for (size_t i = 0; i < vocab_size; i++) {
        cum += expf(sorted[i].logit - max_logit) / sum;
        if (cum >= p) { keep = i + 1; break; }
    }
    for (size_t i = keep; i < vocab_size; i++)
        logits[sorted[i].idx] = -INFINITY;
    free(sorted);
}

uint32_t oc_sampler_chain_sample(OcSamplerChain *chain,
                                  float *logits, size_t vocab_size,
                                  const uint32_t *recent_tokens,
                                  size_t n_recent,
                                  float *mirostat_mu)
{
    if (!chain || !logits || vocab_size == 0) return 0;

    const OcSamplerStep *terminal = NULL;
    bool greedy = false;

    for (size_t i = 0; i < chain->n_steps; i++) {
        OcSamplerStep *step = &chain->steps[i];
        switch (step->type) {
        case OC_SAMPLER_STEP_TEMPERATURE:
            if (step->param1 > 0.0f) {
                for (size_t j = 0; j < vocab_size; j++)
                    logits[j] /= step->param1;
            } else {
                greedy = true; /* temperature <= 0 means greedy */
            }
            break;
        case OC_SAMPLER_STEP_PENALTIES:
            oc_apply_penalties(logits, vocab_size,
                               recent_tokens, n_recent,
                               step->param1, step->param2);
            break;
        case OC_SAMPLER_STEP_TOP_K:
            filter_top_k(logits, vocab_size, (size_t)step->param1);
            break;
        case OC_SAMPLER_STEP_TOP_P:
            filter_top_p(logits, vocab_size, step->param1);
            break;
        default:
            /* Terminal sampling step. */
            if (!terminal) terminal = step;
            break;
        }
    }

    if (terminal && !greedy) {
        switch (terminal->type) {
        case OC_SAMPLER_STEP_MIROSTAT_V1:
            if (mirostat_mu)
                return oc_sample_mirostat_v1(logits, vocab_size,
                                             mirostat_mu, terminal->param1,
                                             terminal->param2, &chain->rng_state);
            break;
        case OC_SAMPLER_STEP_MIROSTAT_V2:
            if (mirostat_mu)
                return oc_sample_mirostat_v2(logits, vocab_size,
                                             mirostat_mu, terminal->param1,
                                             terminal->param2, &chain->rng_state);
            break;
        case OC_SAMPLER_STEP_TFS:
            return oc_sample_tfs(logits, vocab_size, terminal->param1, terminal->param2);
        case OC_SAMPLER_STEP_TYPICAL:
            return oc_sample_typical(logits, vocab_size, terminal->param1, terminal->param2);
        case OC_SAMPLER_STEP_TOP_A:
            return oc_sample_top_a(logits, vocab_size, terminal->param1, terminal->param2);
        case OC_SAMPLER_STEP_ETA:
            return oc_sample_eta_cutoff(logits, vocab_size, terminal->param1, terminal->param2);
        default:
            break;
        }
    }

    /* Default: argmax. */
    uint32_t best = 0;
    float max_logit = logits[0];
    for (size_t i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            best = (uint32_t)i;
        }
    }
    return best;
}

void oc_sampler_chain_free(OcSamplerChain *chain)
{
    if (!chain) return;
    free(chain->steps);
    chain->steps = NULL;
    chain->n_steps = 0;
    chain->capacity = 0;
}
