#ifndef OC_GEN_H
#define OC_GEN_H
#include "oc.h"

enum { OC_SPEC_NGRAM = 0, OC_SPEC_MTP = 1, OC_SPEC_OFF = 2 };

typedef struct {
  oc_model *m;
  oc_tokenizer *tok;          /* for EOG detection (may be NULL) */
  float temperature;          /* <=0 = greedy */
  size_t top_k;
  float top_p;
  float min_p;
  float frequency_penalty;
  float presence_penalty;
  size_t penalty_last_n;
  size_t draft_k;             /* speculative draft length (0 = off) */
  int spec_mode;              /* OC_SPEC_NGRAM (default) | OC_SPEC_MTP | OC_SPEC_OFF */
  void (*on_token)(uint32_t id, void *ud);  /* streaming callback (may be NULL) */
  void *ud;
  size_t drafted, accepted;   /* stats out */
} oc_gen;

void oc_gen_seed(uint64_t seed);
uint32_t oc_sample_token(const oc_gen *g, const float *logits, size_t n,
                         const uint32_t *history, size_t history_count);
/* Prefill prompt (fresh sequence) then generate up to max_new tokens into out.
 * Returns count. Stops on EOG. */
size_t oc_generate(oc_gen *g, const uint32_t *prompt, size_t n_prompt,
                   size_t max_new, uint32_t *out);
#endif
