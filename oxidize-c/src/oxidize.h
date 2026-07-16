/* oxidize-c stable embeddable C ABI.
 *
 * The runtime (GGUF loader, tokenizer, quantized kernels, generation loop) is
 * otherwise reachable only through the CLI (main.c). This header is the seam a
 * Go server or a Python `ctypes`/`cffi` binding drives it through: opaque
 * handles, no internal headers leaked, and a `struct_size`-versioned options
 * struct so the ABI can grow without breaking already-compiled callers.
 *
 * Two handles:
 *   OxModel   — one loaded GGUF (weights + tokenizer). Shared, read-only for
 *               inference metadata; heavy to create. Open once.
 *   OxSession — one conversation: its own sampler settings, KV-cache position,
 *               and repeat-penalty window. Cheap; make one per conversation.
 *
 * Thread-safety:
 *   - An OxModel's read-only accessors (ox_metadata, ox_version, ox_isa) are
 *     safe from any thread.
 *   - Each OxSession is single-threaded: never call ox_generate on one session
 *     from two threads at once.
 *   - Sessions that share a model may interleave multi-turn conversations:
 *     each session owns its own KV cache (all families). Forwards still
 *     serialize on a per-model mutex because scratch buffers (x/q/k/...) are
 *     shared. Concurrent ox_generate calls on different sessions of one model
 *     are safe; they queue on that mutex rather than corrupting KV.
 *
 * Errors: functions that can fail take (char* err, size_t errlen) and return
 * 0 on success / -1 on failure, writing a NUL-terminated message into err.
 */
#ifndef OXIDIZE_H
#define OXIDIZE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OxModel OxModel;
typedef struct OxSession OxSession;

/* Model-open options. struct_size MUST be set to sizeof(OxModelOptions) by the
 * caller: it is the first field so a newer library can tell which trailing
 * fields an older caller actually provided. Pass NULL for all defaults. */
typedef struct {
  size_t struct_size; /* = sizeof(OxModelOptions) */
  size_t ctx;         /* max context / KV-cache positions; 0 = model default */
  int threads;        /* worker threads; <= 0 = one per online CPU */
  uint64_t seed;      /* default RNG seed for new sessions; 0 = built-in */
  int kv_quant;       /* != 0 => rotated int4 KV cache (gemma4 only, else ignored) */
} OxModelOptions;

/* --inspect-style summary of a loaded model. struct_size versions it exactly
 * like OxModelOptions. The `arch` / `isa` pointers stay valid until the model
 * is closed; do not free them. */
typedef struct {
  size_t struct_size; /* = sizeof(OxMetadata) */
  const char* arch;   /* general.architecture, e.g. "gemma4" / "llama" */
  const char* isa;    /* active kernel ISA, e.g. "avx512+vnni" / "avx2" / "scalar" */
  size_t vocab;       /* logits per step */
  size_t ctx;         /* KV-cache capacity actually allocated */
  size_t n_tensors;   /* tensors in the GGUF */
  size_t n_kv;        /* metadata key/value pairs in the GGUF */
} OxMetadata;

/* Streaming sink for one decoded piece (a UTF-8 fragment, not NUL-terminated).
 * Return != 0 to stop generation early (a clean stop, not an error). `user` is
 * the opaque pointer passed to ox_generate. */
typedef int (*OxTokenCb)(const char* piece, size_t len, void* user);

/* Open a model from a GGUF file (single file or the shard-0 path of a split).
 * `opts` may be NULL (defaults); if non-NULL its struct_size must match. On
 * success writes *out and returns 0; on failure returns -1 with a message in
 * err and *out left NULL. */
int ox_model_open(OxModel** out, const char* path, const OxModelOptions* opts,
                  char* err, size_t errlen);

/* Release a model. Frees every session first is the caller's job — sessions
 * hold a borrowed pointer to the model. Safe on NULL. */
void ox_model_close(OxModel* m);

/* Fill *out with model metadata. Set out->struct_size before calling. Returns
 * 0, or -1 on a struct_size mismatch / NULL argument. */
int ox_metadata(const OxModel* m, OxMetadata* out);

/* New conversation against a model. Inherits the model's default seed and
 * greedy sampling; tune it with the setters below. NULL on allocation failure. */
OxSession* ox_session_new(OxModel* m);
void ox_session_free(OxSession* s); /* safe on NULL */

/* Reset conversation state for reuse / chat editing: pos=0, clear recent[], and
 * if the session owns a KV cache, zero its buffers and kv_len. Safe on NULL. */
void ox_session_reset(OxSession* s);

/* Sampler configuration (per session, applied to every subsequent ox_generate).
 * Mirrors the CLI flags: temperature <= 0 is greedy argmax; top_k <= 0, top_p
 * >= 1, min_p <= 0, and the penalties == (repeat<=1 / freq,pres==0) each
 * disable their stage. */
void ox_session_set_temperature(OxSession* s, float temperature);
void ox_session_set_top_k(OxSession* s, int top_k);
void ox_session_set_top_p(OxSession* s, float top_p);
void ox_session_set_min_p(OxSession* s, float min_p);
void ox_session_set_repeat_penalty(OxSession* s, float repeat_penalty);
void ox_session_set_frequency_penalty(OxSession* s, float frequency_penalty);
void ox_session_set_presence_penalty(OxSession* s, float presence_penalty);
void ox_session_set_seed(OxSession* s, uint64_t seed);

/* Generate up to max_tokens tokens continuing this session. The prompt is
 * wrapped in the model's chat template when it declares one (raw otherwise) and
 * appended after any already-generated context, so repeated calls on one
 * session form a multi-turn conversation. Each decoded piece is handed to `cb`
 * (may be NULL to run silently). Returns 0 on success (including a callback- or
 * EOS-requested early stop) or -1 with a message in err. */
int ox_generate(OxSession* s, const char* prompt, int max_tokens, OxTokenCb cb,
                void* user, char* err, size_t errlen);

/* Library version string, e.g. "oxidize-c 0.1.0". Static storage. */
const char* ox_version(void);
/* Active kernel ISA string, e.g. "avx2". Static storage. Meaningful only after
 * the runtime has bound its dispatch table (any time; bound before main). */
const char* ox_isa(void);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_H */
