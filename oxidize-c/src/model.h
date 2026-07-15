/* Architecture dispatch: detect the family from general.architecture and load
 * the matching implementation (gemma4 / qwen36 / generic dense llama). The CLI
 * calls model_load() and then drives the model through the family-agnostic
 * forward hooks below, so main.c never branches on architecture itself.
 *
 * Arch string -> family:
 *   gemma*                        -> gemma4 (its loader gates non-"gemma4")
 *   qwen35 / qwen3.5              -> qwen36 (hybrid gated-DeltaNet)
 *   deepseek2 / deepseek(_v2/_v3) -> deepseek (MLA + group-routed MoE)
 *   llama/mistral/qwen2/qwen3/    -> llama  (generic dense transformer)
 *     yi/phi3
 *   anything else                -> clean "unsupported architecture" error */
#ifndef OC_MODEL_H
#define OC_MODEL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "gguf.h"

typedef enum { MODEL_GEMMA4, MODEL_QWEN36, MODEL_LLAMA, MODEL_DEEPSEEK } ModelFamily;

/* Family-agnostic forward hooks (one token / one batch). The void* is the
 * concrete model owned inside Model.handle. */
typedef float* (*ModelForwardFn)(void* model, int32_t token, size_t pos, bool need_logits);
typedef float* (*ModelForwardBatchFn)(void* model, const int32_t* toks, size_t n,
                                       size_t pos0, bool need_logits);

typedef struct {
  ModelFamily family;
  void* handle; /* heap-owned concrete model (Gemma4Model/Qwen36Model/LlamaModel/DeepseekModel) */
  ModelForwardFn forward;
  ModelForwardBatchFn forward_batch;
  size_t vocab, ctx;
  int gemma_stops;  /* apply gemma4's hardcoded stop ids (1, 106) */
  GgufFile* g;      /* the owned GGUF (lives inside *handle); NUMA + tokenizer */
} Model;

/* Detects the architecture and loads it. Takes ownership of *g on success (0);
 * on failure writes err (including a clear "unsupported architecture: X" for an
 * unrouted arch) and leaves *g for the caller to close. max_ctx caps the KV
 * cache (0 = model context); kv_quant requests the rotated int4 KV cache
 * (gemma4 only, ignored elsewhere). */
int model_load(Model* m, GgufFile* g, size_t max_ctx, bool kv_quant, char* err,
               size_t errlen);
void model_free(Model* m);

#endif
