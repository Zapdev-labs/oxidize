/*
 * wasm.h — WebAssembly bridge for oxidize-c.
 *
 * Provides a C ABI over the oxidize-c forward pass designed to run inside a
 * WebAssembly environment (WASI or emscripten). This is the C port of
 * oxidize-core/src/util/web_worker.rs: it mirrors the same Config + State +
 * Message trinity, exposes a small surface for init/free, model load,
 * generation with a per-token callback, and stats reporting, and emits the
 * TypeScript interface contract as a string literal (matching the Rust
 * `WASM_WORKER_TYPESCRIPT_BINDINGS` `#[wasm_bindgen(typescript_custom_section)]`
 * pattern).
 *
 * Conventions follow the rest of the oxidize-c port:
 *   - public types use `OcWasm` PascalCase, functions `oc_wasm_snake_case`,
 *   - every public function returns `OcError` (OC_OK on success),
 *   - the bridge is reentrant and thread-safe under WASM single-threaded
 *     semantics; if `enable_threads` is set the caller is responsible for
 *     serializing calls per bridge instance.
 *
 * Port notes:
 *   - The Rust web_worker uses serde JSON over a `postMessage` channel. The
 *     C port exposes direct C function calls; an optional message queue is
 *     kept for callers that prefer the async-style enqueue API.
 *   - The Rust worker embeds a 60+ line TypeScript interface string; this
 *     port mirrors that contract exactly via oc_wasm_bridge_format_interface().
 */
#ifndef OXIDIZE_WASM_H
#define OXIDIZE_WASM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Constants ──────────────────────────────────────────────────────────── */

/* Default cap on the in-bridge message queue (must be a power of two). */
#define OC_WASM_MSG_QUEUE_CAP   16

/* Default max heap memory the bridge will attempt to reserve for weights
 * and KV cache before falling back to embedded/empty weights. Mirrors the
 * default browser WASM memory limit of 2 GiB. */
#define OC_WASM_DEFAULT_MAX_MEMORY (2ULL * 1024ULL * 1024ULL * 1024ULL)

/* Default cap on generated tokens per generate call when the caller passes
 * 0 for `max_tokens`. Matches the Rust WorkerModelConfig::default context. */
#define OC_WASM_DEFAULT_MAX_TOKENS 4096

/* ─── Message types ─────────────────────────────────────────────────────── */

/* Mirrors the Rust `WorkerInferenceRequest` / `WorkerModelCacheAction` union
 * of message kinds. A single OcWasmMessage tag discriminates the union; the
 * queue holds one variant at a time per slot. */
typedef enum {
    OC_WASM_MSG_NONE     = 0,  /* empty slot                                    */
    OC_WASM_MSG_LOAD     = 1,  /* load model from path or embedded bytes       */
    OC_WASM_MSG_GENERATE = 2,  /* run forward + sampling for a prompt           */
    OC_WASM_MSG_CANCEL   = 3,  /* cancel the in-flight generation                */
    OC_WASM_MSG_STATUS   = 4,  /* request a status snapshot                      */
} OcWasmMessageType;

/* A single queued message. `payload` is a NUL-terminated C string owned by
 * the caller until the bridge drains the queue. For LOAD, payload is the
 * model path; for GENERATE, payload is the prompt; for CANCEL/STATUS it is
 * ignored (may be NULL). */
typedef struct OcWasmMessage {
    OcWasmMessageType type;
    const char       *payload;       /* not owned by the queue                 */
    uint32_t          max_tokens;    /* GENERATE only                           */
    uint32_t          token_id;      /* sequence id assigned by enqueue         */
} OcWasmMessage;

/* ─── Config ─────────────────────────────────────────────────────────────── */

/* WasmBridgeConfig mirrors WorkerModelConfig + the runtime knobs the Rust
 * port derives from compile-time features (SIMD, threads, wasm-bindgen).
 * Here they are runtime booleans so the host can toggle them per bridge. */
typedef struct OcWasmBridgeConfig {
    uint64_t   max_memory;        /* max heap bytes the bridge may use         */
    bool       enable_simd;       /* request SIMD kernels where available       */
    bool       enable_threads;    /* request threaded forward where available  */
    const char *model_path;       /* optional path/URL to load on init          */
    uint32_t   max_tokens;        /* default max_new_tokens per generate        */
    float      temperature;       /* sampling temperature (0 = greedy)          */
    uint32_t   queue_cap;         /* message queue capacity (power of 2; 0 →
                                     OC_WASM_MSG_QUEUE_CAP)                     */
} OcWasmBridgeConfig;

/* Fill `*cfg` with the default config. Used by callers that want to tweak
 * only a few fields. Never fails; safe on NULL (returns without writing). */
void oc_wasm_bridge_config_default(OcWasmBridgeConfig *cfg);

/* ─── Stats ─────────────────────────────────────────────────────────────── */

/* OcWasmStats mirrors WorkerInferenceResponse + cache_stats() in Rust. */
typedef struct OcWasmStats {
    uint64_t tokens_generated;    /* lifetime tokens generated across calls    */
    uint64_t tokens_consumed;     /* lifetime prompt tokens consumed          */
    double   tokens_per_sec;      /* last generation throughput                */
    double   last_latency_ms;     /* last generate() wall-clock latency         */
    uint64_t memory_used;         /* heap bytes currently in use by the bridge */
    uint64_t memory_peak;         /* peak heap bytes observed                  */
    uint32_t messages_queued;    /* current queue depth                        */
    uint32_t messages_processed; /* lifetime messages drained                  */
    bool     model_loaded;       /* a model has been loaded successfully       */
} OcWasmStats;

/* ─── Per-token callback ─────────────────────────────────────────────────── */

/* Called for each generated token during oc_wasm_bridge_generate(). `token`
 * is the sampled token id; `index` is the 0-based position within the
 * current generation; `userdata` is passed through from the caller. The
 * callback may return false to stop generation early (mirrors how the Rust
 * streaming worker breaks on `Poll::Ready(None)`). */
typedef bool (*OcWasmTokenCallback)(uint32_t token, uint32_t index,
                                    void *userdata);

/* ─── Host hooks ────────────────────────────────────────────────────────── */

/* A host hook table is invoked by the bridge to perform the actual model
 * load and forward pass. In a real WASM deployment the host (JS) registers
 * these via an emscripten addFunction table; when no hooks are installed a
 * built-in stub generator produces deterministic synthetic tokens (tests
 * only). */
typedef struct OcWasmHostHooks {
    /* Load model bytes into the host. Returns true on success. */
    bool (*load_model_bytes)(const uint8_t *data, size_t len,
                             const char *path, void *userdata);
    /* Sample the next token given the prompt + generated-so-far tokens.
     * Returns a token id (0 signals end-of-stream / host failure). */
    uint32_t (*sample_token)(const char *prompt,
                             const uint32_t *generated, size_t n_generated,
                             uint32_t max_tokens, float temperature,
                             void *userdata);
    /* Release any host-side state. May be NULL. */
    void (*release)(void *userdata);
} OcWasmHostHooks;

/* ─── Bridge handle ─────────────────────────────────────────────────────── */

/* Opaque bridge state. The full struct is defined in src/util/wasm_bridge.c;
 * callers always hold a pointer. */
typedef struct OcWasmBridge OcWasmBridge;

/* ─── Lifecycle ─────────────────────────────────────────────────────────── */

/* Allocate and initialize a new bridge with the given config. If `cfg` is
 * NULL, defaults are used (oc_wasm_bridge_config_default). Returns NULL on
 * OOM. The returned pointer must be freed with oc_wasm_bridge_free(). */
OcWasmBridge *oc_wasm_bridge_init(const OcWasmBridgeConfig *cfg);

/* Release all resources owned by the bridge (queue, weights, KV cache).
 * Safe on NULL. */
void oc_wasm_bridge_free(OcWasmBridge *br);

/* ─── Model loading ─────────────────────────────────────────────────────── */

/* Load a model from `path` (a host path or URL). The bridge records stats
 * and transitions to `model_loaded == true` on success. Returns OC_OK,
 * OC_ERR_INVALID_ARG if `br`/`path` is NULL, OC_ERR_IO on read failure, or
 * OC_ERR_MODEL on parse failure. Safe to call multiple times (each load
 * replaces the previous model). */
OcError oc_wasm_bridge_load_model(OcWasmBridge *br, const char *path);

/* Load a model from in-memory `data` (`len` bytes). This mirrors the Rust
 * `cache_downloaded_model` path: the host fetches bytes off-thread and
 * hands them to the bridge. Returns OC_OK or an error code. */
OcError oc_wasm_bridge_load_model_bytes(OcWasmBridge *br,
                                        const uint8_t *data, size_t len);

/* ─── Generation ────────────────────────────────────────────────────────── */

/* Generate text from `prompt`. For each produced token, `on_token` (if not
 * NULL) is invoked with the token id, its 0-based index, and `userdata`.
 * Generation stops when `max_tokens` is reached, the callback returns false,
 * the bridge is cancelled (oc_wasm_bridge_cancel), or the model emits EOS.
 *
 * If `br` is NULL or `prompt` is NULL, returns OC_ERR_INVALID_ARG without
 * invoking the callback. If no model is loaded, returns OC_ERR_MODEL.
 *
 * The response (the full generated text, NUL-terminated) is written into
 * `out_buf` (up to `out_cap-1` chars + NUL). If `out_buf` is NULL, no text
 * is assembled (useful for pure token-stream consumers). Returns the number
 * of bytes written excluding NUL, or 0 on error / overflow. On overflow the
 * buffer holds the NUL-terminated partial text, but the 0 return tells the
 * caller the generation did not fit and must not be treated as complete. */
size_t oc_wasm_bridge_generate(OcWasmBridge *br,
                               const char *prompt,
                               uint32_t max_tokens,
                               OcWasmTokenCallback on_token,
                               void *userdata,
                               char *out_buf, size_t out_cap);

/* Request cancellation of any in-flight generation. The cancel flag is
 * atomic, so this is the one bridge call that is safe to make concurrently
 * (e.g. from another thread when `enable_threads` is set) while
 * oc_wasm_bridge_generate() runs; the generate loop observes the flag on
 * its next token step and stops. All other bridge calls must still be
 * serialized per instance. Returns OC_OK or OC_ERR_INVALID_ARG. */
OcError oc_wasm_bridge_cancel(OcWasmBridge *br);

/* ─── Stats ─────────────────────────────────────────────────────────────── */

/* Copy the current stats snapshot into `*out`. Returns OC_OK or
 * OC_ERR_INVALID_ARG if `br`/`out` is NULL. */
OcError oc_wasm_bridge_get_stats(OcWasmBridge *br, OcWasmStats *out);

/* ─── Host hook installation ────────────────────────────────────────────── */

/* Install a host hook table so model loading and token sampling run the
 * host's real forward-pass implementation instead of the built-in stub.
 * The previous hooks' release() is called first. The bridge stores a copy
 * of `*hooks`; `userdata` is passed to every hook. Returns OC_OK or
 * OC_ERR_INVALID_ARG. */
OcError oc_wasm_bridge_install_hooks(OcWasmBridge *br,
                                     const OcWasmHostHooks *hooks,
                                     void *userdata);

/* ─── Message queue (async-style API) ───────────────────────────────────── */

/* Enqueue a message onto the bridge's internal queue. The caller retains
 * ownership of `msg.payload`. On enqueue, `msg.token_id` is filled with a
 * monotonically increasing sequence number. Returns OC_OK,
 * OC_ERR_INVALID_ARG, or OC_ERR_OOM if the queue is full. */
OcError oc_wasm_bridge_enqueue(OcWasmBridge *br, OcWasmMessage *msg);

/* Drain one message from the queue and act on it. LOAD → load model,
 * GENERATE → run generation (without a callback; response is dropped),
 * CANCEL → cancel, STATUS → no-op (use oc_wasm_bridge_get_stats). Returns
 * OC_OK, OC_ERR_INVALID_ARG, or OC_ERR_FORMAT if the queue is empty. */
OcError oc_wasm_bridge_drain_one(OcWasmBridge *br);

/* Current queue depth (0 if `br` is NULL). */
uint32_t oc_wasm_bridge_queue_depth(const OcWasmBridge *br);

/* ─── TypeScript interface ───────────────────────────────────────────────── */

/* Format the TypeScript interface contract for this bridge into `buf` (up
 * to `cap-1` chars + NUL). This mirrors the Rust
 * `WASM_WORKER_TYPESCRIPT_BINDINGS` string literal: a single block of
 * `export interface`/`export type` declarations describing the request,
 * response, stream chunk, cache, and stats payloads. Returns the number of
 * bytes written excluding NUL; if `buf` is NULL or `cap == 0`, returns the
 * length that would have been written. */
size_t oc_wasm_bridge_format_interface(char *buf, size_t cap);

/* Returns a pointer to a static, NUL-terminated string containing the full
 * TypeScript interface contract. Never returns NULL. The pointed-to memory
 * is valid for the lifetime of the process and must not be freed. */
const char *oc_wasm_bridge_interface_string(void);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_WASM_H */
