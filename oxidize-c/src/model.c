#include "model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model_deepseek.h"
#include "model_gemma4.h"
#include "model_llama.h"
#include "model_qwen36.h"

/* void*-typed trampolines so the families keep their concrete signatures. */
static float* fwd_gemma4(void* m, int32_t t, size_t p, bool lg) {
  return gemma4_forward((Gemma4Model*)m, t, p, lg);
}
static float* fwdb_gemma4(void* m, const int32_t* t, size_t n, size_t p, bool lg) {
  return gemma4_forward_batch((Gemma4Model*)m, t, n, p, lg);
}
static float* fwd_qwen36(void* m, int32_t t, size_t p, bool lg) {
  return qwen36_forward((Qwen36Model*)m, t, p, lg);
}
static float* fwdb_qwen36(void* m, const int32_t* t, size_t n, size_t p, bool lg) {
  return qwen36_forward_batch((Qwen36Model*)m, t, n, p, lg);
}
static float* fwd_llama(void* m, int32_t t, size_t p, bool lg) {
  return llama_forward((LlamaModel*)m, t, p, lg);
}
static float* fwdb_llama(void* m, const int32_t* t, size_t n, size_t p, bool lg) {
  return llama_forward_batch((LlamaModel*)m, t, n, p, lg);
}
static float* fwd_deepseek(void* m, int32_t t, size_t p, bool lg) {
  return deepseek_forward((DeepseekModel*)m, t, p, lg);
}
static float* fwdb_deepseek(void* m, const int32_t* t, size_t n, size_t p, bool lg) {
  return deepseek_forward_batch((DeepseekModel*)m, t, n, p, lg);
}

/* Dense (and standard-softmax-gated MoE) decoder-only arch strings that map to
 * the generic llama path. Detection is by exact arch string; the llama loader
 * then validates the tensor set and fails loudly if the layout is not the
 * standard one (e.g. phi3's fused QKV), and detects MoE by tensor presence
 * (ffn_gate_inp + expert stacks). Mixtral is arch "llama" in ggml, so it needs
 * no entry here. qwen2moe/qwen3moe/olmoe use the softmax top-k router the FFN
 * implements. gpt-oss is deliberately absent: it needs attention sinks (a per-
 * head bias added to the softmax denominator) the generic path does not
 * implement, so routing it would produce silently-wrong output rather than a
 * clean error. deepseek2 has its own MLA family below. */
static const char* const LLAMA_ARCHS[] = {
    "llama", "mistral", "qwen2",    "qwen3",    "yi",   "phi3",
    "qwen2moe", "qwen3moe", "qwen3_moe", "olmoe", NULL};

/* DeepSeek-V2/V3 (Multi-head Latent Attention + group-routed MoE) -> the MLA
 * family in model_deepseek.c. All spellings that appear in the wild map here;
 * the loader validates the MLA tensor set and fails loudly on anything else. */
static const char* const DEEPSEEK_ARCHS[] = {
    "deepseek2", "deepseek", "deepseek_v2", "deepseek_v3", "deepseek-v2",
    "deepseek-v3", NULL};

static bool arch_in(const char* a, const char* const* list) {
  for (size_t i = 0; list[i]; ++i)
    if (strcmp(a, list[i]) == 0) return true;
  return false;
}

int model_load(Model* m, GgufFile* g, size_t max_ctx, bool kv_quant, char* err,
               size_t errlen) {
  memset(m, 0, sizeof(*m));
  char* arch = gguf_architecture(g);
  if (!arch) {
    if (err && errlen) snprintf(err, errlen, "model has no general.architecture");
    return -1;
  }

  ModelFamily fam;
  if (strncmp(arch, "gemma", 5) == 0)
    fam = MODEL_GEMMA4;
  else if (strcmp(arch, "qwen35") == 0 || strcmp(arch, "qwen3.5") == 0)
    fam = MODEL_QWEN36;
  else if (arch_in(arch, DEEPSEEK_ARCHS))
    fam = MODEL_DEEPSEEK;
  else if (arch_in(arch, LLAMA_ARCHS))
    fam = MODEL_LLAMA;
  else {
    if (err && errlen) snprintf(err, errlen, "unsupported architecture: %s", arch);
    free(arch);
    return -1;
  }
  free(arch);

  size_t sz = fam == MODEL_GEMMA4    ? sizeof(Gemma4Model)
              : fam == MODEL_QWEN36  ? sizeof(Qwen36Model)
              : fam == MODEL_DEEPSEEK ? sizeof(DeepseekModel)
                                     : sizeof(LlamaModel);
  void* handle = malloc(sz);
  if (!handle) {
    if (err && errlen) snprintf(err, errlen, "model allocation failed");
    return -1;
  }

  int rc;
  switch (fam) {
    case MODEL_GEMMA4:
      rc = gemma4_load((Gemma4Model*)handle, g, max_ctx, kv_quant, err, errlen);
      if (rc == 0) {
        Gemma4Model* gm = handle;
        m->forward = fwd_gemma4;
        m->forward_batch = fwdb_gemma4;
        m->vocab = gm->vocab;
        m->ctx = gm->ctx;
        m->gemma_stops = 1;
        m->g = &gm->g;
      }
      break;
    case MODEL_QWEN36:
      rc = qwen36_load((Qwen36Model*)handle, g, max_ctx, err, errlen);
      if (rc == 0) {
        Qwen36Model* qm = handle;
        m->forward = fwd_qwen36;
        m->forward_batch = fwdb_qwen36;
        m->vocab = qm->vocab;
        m->ctx = qm->ctx;
        m->g = &qm->g;
      }
      break;
    case MODEL_DEEPSEEK:
      rc = deepseek_load((DeepseekModel*)handle, g, max_ctx, err, errlen);
      if (rc == 0) {
        DeepseekModel* dm = handle;
        m->forward = fwd_deepseek;
        m->forward_batch = fwdb_deepseek;
        m->vocab = dm->vocab;
        m->ctx = dm->ctx;
        m->g = &dm->g;
      }
      break;
    default: /* MODEL_LLAMA */
      rc = llama_load((LlamaModel*)handle, g, max_ctx, err, errlen);
      if (rc == 0) {
        LlamaModel* lm = handle;
        m->forward = fwd_llama;
        m->forward_batch = fwdb_llama;
        m->vocab = lm->vocab;
        m->ctx = lm->ctx;
        m->g = &lm->g;
      }
      break;
  }
  if (rc != 0) {
    free(handle);
    return -1;
  }
  m->family = fam;
  m->handle = handle;
  return 0;
}

void model_free(Model* m) {
  if (!m->handle) return;
  switch (m->family) {
    case MODEL_GEMMA4: gemma4_free((Gemma4Model*)m->handle); break;
    case MODEL_QWEN36: qwen36_free((Qwen36Model*)m->handle); break;
    case MODEL_LLAMA: llama_free((LlamaModel*)m->handle); break;
    case MODEL_DEEPSEEK: deepseek_free((DeepseekModel*)m->handle); break;
  }
  free(m->handle);
  memset(m, 0, sizeof(*m));
}
