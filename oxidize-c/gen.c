/* Generation loop with optional MTP speculative decoding.
 *
 * Speculative round (mirrors oxidize-core generation.rs::run_mtp_step):
 *   1. draft k tokens with the 1-layer MTP head (cheap, greedy)
 *   2. verify all k in ONE batched main forward (weights read once)
 *   3. accept the longest prefix the main model agrees with; on full accept
 *      also take the bonus token from the last verify logits
 *   4. recurrent SSM state is snapshotted before the verify and restored on
 *      partial accept, then the accepted tokens are re-run as one small batch
 * KV rollback is implicit (positions are overwritten). */
#include "oc.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gen.h"

static uint32_t argmax_(const float *v, size_t n) {
  size_t best = 0;
  for (size_t i = 1; i < n; ++i)
    if (v[i] > v[best]) best = i;
  return (uint32_t)best;
}

static uint64_t rng_state = 0x9e3779b97f4a7c15ull;
void oc_gen_seed(uint64_t seed) { rng_state = seed ? seed : 0x9e3779b97f4a7c15ull; }
static float rng_unit(void) {
  uint64_t x = rng_state;
  x ^= x << 13; x ^= x >> 7; x ^= x << 17;
  rng_state = x;
  return (float)((x >> 11) * (1.0 / 9007199254740992.0));
}

static int cmp_desc(const void *a, const void *b) {
  float d = ((const float *)b)[1] - ((const float *)a)[1];
  return d > 0 ? 1 : d < 0 ? -1 : 0;
}

static uint32_t sample_pick(const float *logits, size_t n, float temp,
                            size_t top_k, float top_p) {
  if (temp <= 0.0f) return argmax_(logits, n);
  float inv_t = 1.0f / temp;
  float (*pairs)[2] = malloc(n * sizeof(*pairs));
  for (size_t i = 0; i < n; ++i) {
    pairs[i][0] = (float)i;
    pairs[i][1] = logits[i] * inv_t;
  }
  qsort(pairs, n, sizeof(*pairs), cmp_desc);
  size_t keep = top_k > 0 && top_k < n ? top_k : n;
  float mx = pairs[0][1];
  double sum = 0;
  for (size_t i = 0; i < keep; ++i) {
    pairs[i][1] = expf(pairs[i][1] - mx);
    sum += pairs[i][1];
  }
  if (top_p < 1.0f) {
    double acc = 0;
    for (size_t i = 0; i < keep; ++i) {
      acc += pairs[i][1] / sum;
      if (acc >= top_p) { keep = i + 1; break; }
    }
    sum = 0;
    for (size_t i = 0; i < keep; ++i) sum += pairs[i][1];
  }
  float r = rng_unit() * (float)sum;
  double acc = 0;
  size_t pick = keep - 1;
  for (size_t i = 0; i < keep; ++i) {
    acc += pairs[i][1];
    if (acc >= r) { pick = i; break; }
  }
  uint32_t id = (uint32_t)pairs[pick][0];
  free(pairs);
  return id;
}

/* prompt-lookup drafting: if the trailing 3-gram of history occurred before,
 * propose the tokens that followed it. Free to produce; verify rejects misses. */
static size_t ngram_draft(const uint32_t *hist, size_t n_hist, size_t k,
                          uint32_t *out) {
  const size_t N = 3;
  if (n_hist < N + 1 || k == 0) return 0;
  const uint32_t *tail = hist + n_hist - N;
  for (size_t start = n_hist - N; start-- > 0;) {
    if (memcmp(hist + start, tail, N * sizeof(uint32_t)) == 0) {
      size_t src = start + N;
      if (src >= n_hist - N) continue;
      size_t avail = (n_hist - N) - src; /* don't copy the tail itself */
      size_t nd = avail < k ? avail : k;
      if (nd == 0) return 0;
      memcpy(out, hist + src, nd * sizeof(uint32_t));
      return nd;
    }
  }
  return 0;
}

size_t oc_generate(oc_gen *g, const uint32_t *prompt, size_t n_prompt,
                   size_t max_new, uint32_t *out) {
  if (!g || !g->m || !prompt || n_prompt == 0 || !out) return 0;
  oc_model *m = g->m;
  const size_t vocab = m->cfg.vocab_size;
  size_t k = g->draft_k;
  if (g->spec_mode == OC_SPEC_MTP && !m->mtp) k = 0;
  if (g->spec_mode == OC_SPEC_OFF) k = 0;
  /* Spec decode needs host SSM-state snapshot/rollback + the MTP CPU path, both
   * CPU-only. The resident GPU forward keeps state on-device, so disable spec
   * there (the GPU is fast enough not to need it). */
  if (m->gpu_active) k = 0;
  if (k > 63) k = 63;

  oc_reset_state(m);
  float *pending = malloc(vocab * sizeof(float));
  float *verify = k ? malloc((k + 1) * vocab * sizeof(float)) : NULL;
  uint32_t *draft = k ? malloc((k + 1) * sizeof(uint32_t)) : NULL;
  size_t sb = k ? oc_state_bytes(m) : 0;
  uint8_t *snap = sb ? malloc(sb) : NULL;

  /* prefill in chunks */
  size_t pos = 0;
  const size_t chunk = 64;
  for (size_t off = 0; off < n_prompt; off += chunk) {
    size_t nb = n_prompt - off < chunk ? n_prompt - off : chunk;
    oc_forward(m, prompt + off, nb, pos, off + nb == n_prompt ? pending : NULL);
    pos += nb;
  }
  uint32_t last = prompt[n_prompt - 1];

  /* token history for n-gram lookup drafting */
  uint32_t *hist = malloc((n_prompt + max_new + 8) * sizeof(uint32_t));
  memcpy(hist, prompt, n_prompt * sizeof(uint32_t));
  size_t n_hist = n_prompt;

  size_t emitted = 0;
  g->drafted = g->accepted = 0;
  while (emitted < max_new) {
    size_t nd = 0;
    if (k > 0 && pos + k + 1 < m->kv_ctx) {
      nd = g->spec_mode == OC_SPEC_MTP
               ? oc_mtp_draft(m, last, pos - 1, k, draft)
               : ngram_draft(hist, n_hist, k, draft);
    }
    if (nd > 0) {
      /* ---- speculative round ---- */
      if (sb) oc_state_save(m, snap);
      size_t verify_start = pos;
      memcpy(verify, pending, vocab * sizeof(float)); /* logits for draft[0] */
      oc_forward_all(m, draft, nd, pos, verify + vocab);
      pos += nd;

      uint32_t tokens[64];
      size_t total = 0;
      bool full = true;
      for (size_t i = 0; i < nd; ++i) {
        uint32_t want = sample_pick(verify + i * vocab, vocab, g->temperature,
                                    g->top_k, g->top_p);
        if (want == draft[i]) {
          tokens[total++] = want;
        } else {
          tokens[total++] = want;
          full = false;
          break;
        }
      }
      if (full && total < 64)
        tokens[total++] = sample_pick(verify + nd * vocab, vocab, g->temperature,
                                      g->top_k, g->top_p);
      g->drafted += nd;
      g->accepted += full ? nd : total - 1;

      if (full) {
        /* state already consumed all nd drafts; forward only the bonus token */
        oc_forward(m, &tokens[total - 1], 1, pos, pending);
        pos += 1;
      } else {
        /* roll back and re-run the committed tokens as one batch */
        if (sb) oc_state_load(m, snap);
        pos = verify_start;
        oc_forward(m, tokens, total, pos, pending);
        pos += total;
      }

      for (size_t i = 0; i < total && emitted < max_new; ++i) {
        uint32_t t = tokens[i];
        if (g->tok && oc_is_eog(g->tok, t)) goto done;
        out[emitted++] = t;
        hist[n_hist++] = t;
        last = t;
        if (g->on_token) g->on_token(t, g->ud);
      }
      /* speculation not paying for itself -> fall back to plain decode */
      if (g->drafted >= 8 * g->draft_k &&
          g->accepted * 3 < g->drafted) /* < ~33% acceptance */
        k = 0;
    } else {
      /* ---- plain decode ---- */
      uint32_t t = sample_pick(pending, vocab, g->temperature, g->top_k, g->top_p);
      if (g->tok && oc_is_eog(g->tok, t)) break;
      out[emitted++] = t;
      hist[n_hist++] = t;
      last = t;
      if (g->on_token) g->on_token(t, g->ud);
      if (emitted >= max_new || pos + 1 >= m->kv_ctx) break;
      oc_forward(m, &t, 1, pos, pending);
      pos += 1;
    }
  }
done:
  free(hist);
  free(pending); free(verify); free(draft); free(snap);
  return emitted;
}
