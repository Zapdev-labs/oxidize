#include "sampler.h"

#include <math.h>
#include <stdlib.h>

static uint64_t xs64(uint64_t* s) {
  uint64_t x = *s;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  *s = x;
  return x * 0x2545F4914F6CDD1Dull;
}

typedef struct {
  float p;
  int32_t id;
} Cand;

static int cand_desc(const void* a, const void* b) {
  float pa = ((const Cand*)a)->p, pb = ((const Cand*)b)->p;
  return pa < pb ? 1 : pa > pb ? -1 : 0;
}

void sampler_penalize(const SamplerConfig* s, float* logits, size_t n,
                      const int32_t* recent, size_t n_recent) {
  /* repeat penalty: llama.cpp divide/multiply, applied once per occurrence */
  if (s->repeat_penalty > 1.0f) {
    for (size_t i = 0; i < n_recent; ++i) {
      int32_t id = recent[i];
      if (id < 0 || (size_t)id >= n) continue;
      float l = logits[id];
      logits[id] = l > 0.0f ? l / s->repeat_penalty : l * s->repeat_penalty;
    }
  }
  /* OpenAI-style: frequency scales with count, presence flat once per token.
   * ponytail: O(n_recent^2) first-occurrence scan for presence; the recent
   * window is small (a few dozen ids), so a histogram over the 262k vocab
   * would cost more than it saves. */
  if (s->frequency_penalty != 0.0f || s->presence_penalty != 0.0f) {
    for (size_t i = 0; i < n_recent; ++i) {
      int32_t id = recent[i];
      if (id < 0 || (size_t)id >= n) continue;
      logits[id] -= s->frequency_penalty;
      int seen = 0;
      for (size_t j = 0; j < i; ++j)
        if (recent[j] == id) { seen = 1; break; }
      if (!seen) logits[id] -= s->presence_penalty;
    }
  }
}

void sampler_free(SamplerConfig* s) {
  free(s->scratch);
  s->scratch = NULL;
  s->scratch_cap = 0;
}

static Cand* ensure_scratch(SamplerConfig* s, size_t need) {
  if (s->scratch_cap < need) {
    Cand* p = realloc(s->scratch, need * sizeof(Cand));
    if (!p) return NULL;
    s->scratch = p;
    s->scratch_cap = need;
  }
  return (Cand*)s->scratch;
}

/* Min-heap (keyed on .p) sift-down over h[0..k). */
static void sift_down(Cand* h, size_t k, size_t i) {
  for (;;) {
    size_t l = 2 * i + 1, r = l + 1, m = i;
    if (l < k && h[l].p < h[m].p) m = l;
    if (r < k && h[r].p < h[m].p) m = r;
    if (m == i) return;
    Cand t = h[i];
    h[i] = h[m];
    h[m] = t;
    i = m;
  }
}

int32_t sample_token(SamplerConfig* s, const float* logits, size_t n) {
  if (n == 0) return -1;
  /* Greedy / temperature 0: plain argmax, no allocation. */
  if (s->temperature <= 0.0f) {
    size_t best = 0;
    for (size_t i = 1; i < n; ++i)
      if (logits[i] > logits[best]) best = i;
    return (int32_t)best;
  }

  float mx = logits[0];
  for (size_t i = 1; i < n; ++i)
    if (logits[i] > mx) mx = logits[i];

  size_t topk = (s->top_k > 0 && (size_t)s->top_k < n) ? (size_t)s->top_k : 0;

  Cand* c;
  size_t keep;
  double sum = 0.0;
  if (topk) {
    /* Partial top-k selection: one pass builds the full softmax denominator
     * (index order, matching the full-sort path bit-for-bit) while a size-k
     * min-heap keeps the k largest unnormalised probs. O(n log k), no full
     * sort of the vocabulary. */
    c = ensure_scratch(s, topk);
    if (!c) return -1;
    size_t h = 0; /* heap fill */
    for (size_t i = 0; i < n; ++i) {
      float up = expf((logits[i] - mx) / s->temperature);
      sum += up;
      if (h < topk) {
        c[h].p = up;
        c[h].id = (int32_t)i;
        if (++h == topk)
          for (size_t j = topk / 2; j-- > 0;) sift_down(c, topk, j);
      } else if (up > c[0].p) {
        c[0].p = up;
        c[0].id = (int32_t)i;
        sift_down(c, topk, 0);
      }
    }
    keep = h; /* == topk (topk < n) */
    for (size_t i = 0; i < keep; ++i) c[i].p = (float)((double)c[i].p / sum);
  } else {
    /* No top-k bound: normalise all n and sort. Determinism-preserving
     * fallback (min_p/top_p still pick by descending order); the scratch
     * buffer means no per-token malloc. */
    c = ensure_scratch(s, n);
    if (!c) return -1;
    for (size_t i = 0; i < n; ++i) {
      c[i].id = (int32_t)i;
      c[i].p = expf((logits[i] - mx) / s->temperature);
      sum += c[i].p;
    }
    for (size_t i = 0; i < n; ++i) c[i].p = (float)(c[i].p / sum);
    keep = n;
  }

  qsort(c, keep, sizeof(Cand), cand_desc);

  if (s->min_p > 0.0f && s->min_p <= 1.0f) {
    float thresh = c[0].p * s->min_p;
    size_t k = 1;
    while (k < keep && c[k].p >= thresh) ++k;
    keep = k;
  }
  if (s->top_p > 0.0f && s->top_p < 1.0f) {
    double cum = 0.0;
    size_t k = 0;
    while (k < keep) {
      cum += c[k].p;
      ++k;
      if (cum >= s->top_p) break;
    }
    keep = k > 0 ? k : 1;
  }

  double ksum = 0.0;
  for (size_t i = 0; i < keep; ++i) ksum += c[i].p;
  double r = (double)(xs64(&s->rng) >> 11) / 9007199254740992.0 * ksum;
  double cum = 0.0;
  int32_t pick = c[keep - 1].id;
  for (size_t i = 0; i < keep; ++i) {
    cum += c[i].p;
    if (r <= cum) {
      pick = c[i].id;
      break;
    }
  }
  return pick;
}
