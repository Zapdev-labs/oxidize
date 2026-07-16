/* oxidize-c stable C ABI — implemented over the existing public runtime
 * (gguf.c / model.c dispatch / tokenizer.c / sampler.c / tensor.c pool). It
 * carries its own generate loop mirroring main.c's; main.c is owned elsewhere
 * and deliberately not shared. */
#include "oxidize.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gguf.h"
#include "model.h"
#include "quant.h"    /* oc_isa_active_name */
#include "sampler.h"
#include "tensor.h"   /* oc_pool_init/free, oc_numa_replicate */
#include "tokenizer.h"

#define OX_RECENT_CAP 64 /* repeat-penalty window (matches main.c) */
#define OX_DEFAULT_SEED 0x9E3779B97F4A7C15ull

/* The thread pool + NUMA replicas are a process-wide singleton (machine state,
 * not model state — see tensor.h). Init lazily on the first open; tear down when
 * the last model closes so a library caller leaks no worker threads.
 * ponytail: single global count, no lock — model open/close is documented
 * single-threaded; add a lock here if that ever stops holding. */
static int g_open_models = 0;

struct OxModel {
  Model model;     /* arch-dispatched weights + forward hooks (owns the GGUF) */
  Tokenizer tok;
  char* arch;      /* malloc'd; OxMetadata.arch points at it */
  size_t n_tensors, n_kv;
  uint64_t default_seed;
  pthread_mutex_t fwd_mu; /* serializes forwards (shared scratch); KV is per-session */
  /* Chat template: detected once at open. Marker strings are literals (no free).
   * has_template == 0 => prompts are passed raw. */
  int has_template;
  const char* turn_open;
  const char* turn_close;
  const char* asst_role;
  const char* gen_suffix;
  int32_t eot_tok; /* id of turn_close, or -1 */
};

struct OxSession {
  OxModel* m;             /* borrowed; must outlive the session */
  SamplerConfig sampler;  /* owns a lazily-grown scratch buffer */
  ModelKv* kv;            /* per-session KV; NULL => shared model primary */
  size_t pos;             /* next KV position (persists across turns) */
  int32_t recent[OX_RECENT_CAP];
  size_t n_recent;
};

/* ---- generation helpers (ported from main.c's statics) -------------------- */

static void push_recent(OxSession* s, int32_t id) {
  if (s->n_recent < OX_RECENT_CAP) {
    s->recent[s->n_recent++] = id;
  } else {
    memmove(s->recent, s->recent + 1, (OX_RECENT_CAP - 1) * sizeof(int32_t));
    s->recent[OX_RECENT_CAP - 1] = id;
  }
}

static int is_stop(const OxSession* s, int32_t next) {
  const Tokenizer* tok = &s->m->tok;
  return next < 0 || (s->m->model.gemma_stops && (next == 1 || next == 106)) ||
         next == (int32_t)tok->eos_id ||
         (tok->eot_id >= 0 && next == (int32_t)tok->eot_id) ||
         (s->m->eot_tok >= 0 && next == s->m->eot_tok);
}

/* ---- model open / close --------------------------------------------------- */

/* Detect the chat template's turn markers exactly as main.c does. */
static void detect_template(OxModel* m) {
  m->has_template = 0;
  m->turn_open = "<start_of_turn>";
  m->turn_close = "<end_of_turn>";
  m->asst_role = "model";
  m->gen_suffix = "";
  m->eot_tok = -1;
  char* tmpl = gguf_get_str(&m->model.g[0], "tokenizer.ggml.chat_template");
  if (!tmpl) return;
  m->has_template = 1;
  if (strstr(tmpl, "<|turn>")) {
    m->turn_open = "<|turn>";
    m->turn_close = "<turn|>";
    m->gen_suffix = "<|channel>thought\n<channel|>";
  } else if (strstr(tmpl, "<|im_start|>")) {
    m->turn_open = "<|im_start|>";
    m->turn_close = "<|im_end|>";
    m->asst_role = "assistant";
    m->gen_suffix = "<think>\n\n</think>\n\n";
  }
  free(tmpl);
  m->eot_tok = tokenizer_piece_id(&m->tok, m->turn_close, strlen(m->turn_close));
}

int ox_model_open(OxModel** out, const char* path, const OxModelOptions* opts,
                  char* err, size_t errlen) {
  if (out) *out = NULL;
  if (!out || !path) {
    if (err && errlen) snprintf(err, errlen, "ox_model_open: NULL argument");
    return -1;
  }
  if (opts && opts->struct_size != sizeof(OxModelOptions)) {
    if (err && errlen)
      snprintf(err, errlen,
               "ox_model_open: OxModelOptions struct_size %zu != %zu (ABI mismatch)",
               opts->struct_size, sizeof(OxModelOptions));
    return -1;
  }
  size_t ctx = opts ? opts->ctx : 0;
  int threads = opts ? opts->threads : 0;
  int kv_quant = opts ? opts->kv_quant : 0;
  uint64_t seed = opts && opts->seed ? opts->seed : OX_DEFAULT_SEED;

  GgufFile g;
  if (gguf_open(&g, path, err, errlen) != 0) return -1; /* gguf_open filled err */

  oc_pool_init(threads);

  OxModel* m = calloc(1, sizeof(*m));
  if (!m) {
    if (err && errlen) snprintf(err, errlen, "ox_model_open: out of memory");
    gguf_close(&g);
    goto fail_pool;
  }
  if (model_load(&m->model, &g, ctx, kv_quant != 0, err, errlen) != 0) {
    gguf_close(&g); /* model_load leaves g for us on failure */
    free(m);
    goto fail_pool;
  }
  /* model_load took ownership of the GGUF; it now lives inside the model. */
  GgufFile* owned = m->model.g;
  oc_numa_replicate(owned->map, owned->size);

  if (tokenizer_init(&m->tok, owned) != 0) {
    if (err && errlen)
      snprintf(err, errlen, "ox_model_open: tokenizer init failed (missing "
                            "tokenizer.ggml.tokens?)");
    model_free(&m->model);
    free(m);
    goto fail_pool;
  }

  m->arch = gguf_architecture(owned); /* may be NULL; metadata reports "" then */
  m->n_tensors = owned->n_tensors;
  m->n_kv = owned->n_kv;
  m->default_seed = seed;
  detect_template(m);
  if (pthread_mutex_init(&m->fwd_mu, NULL) != 0) {
    if (err && errlen) snprintf(err, errlen, "ox_model_open: mutex init failed");
    tokenizer_free(&m->tok);
    model_free(&m->model);
    free(m->arch);
    free(m);
    goto fail_pool;
  }

  g_open_models++;
  *out = m;
  return 0;

fail_pool:
  if (g_open_models == 0) oc_pool_free();
  return -1;
}

void ox_model_close(OxModel* m) {
  if (!m) return;
  tokenizer_free(&m->tok);
  model_free(&m->model);
  pthread_mutex_destroy(&m->fwd_mu);
  free(m->arch);
  free(m);
  if (--g_open_models == 0) oc_pool_free();
}

int ox_metadata(const OxModel* m, OxMetadata* out) {
  if (!m || !out || out->struct_size != sizeof(OxMetadata)) return -1;
  out->arch = m->arch ? m->arch : "";
  out->isa = oc_isa_active_name();
  out->vocab = m->model.vocab;
  out->ctx = m->model.ctx;
  out->n_tensors = m->n_tensors;
  out->n_kv = m->n_kv;
  return 0;
}

/* ---- sessions ------------------------------------------------------------- */

OxSession* ox_session_new(OxModel* m) {
  if (!m) return NULL;
  OxSession* s = calloc(1, sizeof(*s));
  if (!s) return NULL;
  s->m = m;
  s->kv = model_kv_new(&m->model); /* NULL ok: unsupported family or OOM */
  /* greedy by default, mirroring the CLI's zero-initialised sampler. */
  s->sampler.temperature = 0.0f;
  s->sampler.top_p = 1.0f;
  s->sampler.rng = m->default_seed | 1ull;
  return s;
}

void ox_session_free(OxSession* s) {
  if (!s) return;
  model_kv_free(s->kv);
  sampler_free(&s->sampler);
  free(s);
}

void ox_session_reset(OxSession* s) {
  if (!s) return;
  s->pos = 0;
  s->n_recent = 0;
  memset(s->recent, 0, sizeof(s->recent));
  if (s->kv) model_kv_clear(s->kv);
}

void ox_session_set_temperature(OxSession* s, float v) { if (s) s->sampler.temperature = v; }
void ox_session_set_top_k(OxSession* s, int v) { if (s) s->sampler.top_k = v; }
void ox_session_set_top_p(OxSession* s, float v) { if (s) s->sampler.top_p = v; }
void ox_session_set_min_p(OxSession* s, float v) { if (s) s->sampler.min_p = v; }
void ox_session_set_repeat_penalty(OxSession* s, float v) { if (s) s->sampler.repeat_penalty = v; }
void ox_session_set_frequency_penalty(OxSession* s, float v) { if (s) s->sampler.frequency_penalty = v; }
void ox_session_set_presence_penalty(OxSession* s, float v) { if (s) s->sampler.presence_penalty = v; }
void ox_session_set_seed(OxSession* s, uint64_t seed) { if (s) s->sampler.rng = seed | 1ull; }

/* ---- generation ----------------------------------------------------------- */

/* Wrap the user prompt in the model's chat template (if any), keyed on whether
 * this is the first turn (pos == 0) or a continuation. Returns a malloc'd
 * string, or NULL on OOM. Mirrors main.c (system prompt / --raw not exposed
 * here — the raw path is simply "no template"). */
static char* build_prompt(const OxSession* s, const char* user) {
  const OxModel* m = s->m;
  size_t flen = strlen(user) + 320;
  char* full = malloc(flen);
  if (!full) return NULL;
  if (!m->has_template) {
    snprintf(full, flen, "%s", user);
  } else if (s->pos == 0) {
    snprintf(full, flen, "%suser\n%s%s\n%s%s\n%s", m->turn_open, user,
             m->turn_close, m->turn_open, m->asst_role, m->gen_suffix);
  } else {
    snprintf(full, flen, "%s\n%suser\n%s%s\n%s%s\n%s", m->turn_close,
             m->turn_open, user, m->turn_close, m->turn_open, m->asst_role,
             m->gen_suffix);
  }
  return full;
}

int ox_generate(OxSession* s, const char* prompt, int max_tokens, OxTokenCb cb,
                void* user, char* err, size_t errlen) {
  if (!s || !prompt) {
    if (err && errlen) snprintf(err, errlen, "ox_generate: NULL argument");
    return -1;
  }
  OxModel* m = s->m;
  const size_t vocab = m->model.vocab, ctx = m->model.ctx;

  char* full = build_prompt(s, prompt);
  if (!full) {
    if (err && errlen) snprintf(err, errlen, "ox_generate: out of memory");
    return -1;
  }
  size_t n_ids = 0;
  int32_t* ids = tokenizer_encode(&m->tok, full, s->pos == 0, &n_ids);
  free(full);
  if (!ids || n_ids == 0) {
    free(ids);
    if (err && errlen) snprintf(err, errlen, "ox_generate: prompt tokenization failed");
    return -1;
  }
  if (s->pos + n_ids >= ctx) {
    free(ids);
    if (err && errlen)
      snprintf(err, errlen, "ox_generate: context full (%zu + %zu tokens > ctx %zu)",
               s->pos, n_ids, ctx);
    return -1;
  }

  pthread_mutex_lock(&m->fwd_mu);
  if (s->kv) model_kv_install(&m->model, s->kv);

  /* Prefill the whole prompt in one batched pass (weights read once per batch,
   * not once per token), then decode. */
  float* logits = m->model.forward_batch(m->model.handle, ids, n_ids, s->pos, true);
  for (size_t i = 0; i < n_ids; ++i) push_recent(s, ids[i]);
  s->pos += n_ids;
  free(ids);
  if (!logits) {
    if (s->kv) model_kv_release(&m->model, s->kv);
    pthread_mutex_unlock(&m->fwd_mu);
    if (err && errlen) snprintf(err, errlen, "ox_generate: forward failed");
    return -1;
  }

  char buf[512];
  for (int produced = 0; produced < max_tokens && s->pos < ctx; ++produced) {
    sampler_penalize(&s->sampler, logits, vocab, s->recent, s->n_recent);
    int32_t next = sample_token(&s->sampler, logits, vocab);
    if (is_stop(s, next)) break;
    size_t w = tokenizer_decode_token(&m->tok, next, buf, sizeof(buf));
    if (w > 0 && cb && cb(buf, w, user) != 0) break; /* caller-requested stop */
    push_recent(s, next);
    logits = m->model.forward(m->model.handle, next, s->pos++, true);
    if (!logits) break; /* clean end of stream */
  }

  if (s->kv) model_kv_release(&m->model, s->kv);
  pthread_mutex_unlock(&m->fwd_mu);
  return 0;
}

/* ---- version / isa -------------------------------------------------------- */

const char* ox_version(void) { return "oxidize-c 0.1.0"; }
const char* ox_isa(void) { return oc_isa_active_name(); }
