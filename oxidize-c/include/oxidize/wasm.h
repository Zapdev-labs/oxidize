/* wasm.h — WebAssembly bridge for oxidize-c. */
#ifndef OXIDIZE_WASM_H
#define OXIDIZE_WASM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif


/* Default cap on the in-bridge message queue (must be a power of two). */
#define OC_WASM_MSG_QUEUE_CAP   16

/* Default max heap memory the bridge will attempt to reserve for weights
 * and KV cache before falling back to embedded/empty weights. Mirrors the
 * default browser WASM memory limit of 2 GiB. */
#define OC_WASM_DEFAULT_MAX_MEMORY (2ULL * 1024ULL * 1024ULL * 1024ULL)

/* Default cap on generated tokens per generate call when the caller passes
 * 0 for `max_tokens`. Matches the Rust WorkerModelConfig::default context. */
#define OC_WASM_DEFAULT_MAX_TOKENS 4096


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

/* A single queued message. */
typedef struct OcWasmMessage {
    OcWasmMessageType type;
    const char       *payload;       /* not owned by the queue                 */
    uint32_t          max_tokens;    /* GENERATE only                           */
    uint32_t          token_id;      /* sequence id assigned by enqueue         */
} OcWasmMessage;


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


/* Called for each generated token during oc_wasm_bridge_generate(). `token` */
typedef bool (*OcWasmTokenCallback)(uint32_t token, uint32_t index,
                                    void *userdata);


/* A host hook table is invoked by the bridge to perform the actual model */
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


/* Opaque bridge state. The full struct is defined in src/util/wasm_bridge.c;
 * callers always hold a pointer. */
typedef struct OcWasmBridge OcWasmBridge;


/* Allocate and initialize a new bridge with the given config. If `cfg` is
 * NULL, defaults are used (oc_wasm_bridge_config_default). Returns NULL on
 * OOM. The returned pointer must be freed with oc_wasm_bridge_free(). */
OcWasmBridge *oc_wasm_bridge_init(const OcWasmBridgeConfig *cfg);

/* Release all resources owned by the bridge (queue, weights, KV cache).
 * Safe on NULL. */
void oc_wasm_bridge_free(OcWasmBridge *br);


/* Load a model from `path` (a host path or URL). The bridge records stats */
OcError oc_wasm_bridge_load_model(OcWasmBridge *br, const char *path);

/* Load a model from in-memory `data` (`len` bytes). This mirrors the Rust
 * `cache_downloaded_model` path: the host fetches bytes off-thread and
 * hands them to the bridge. Returns OC_OK or an error code. */
OcError oc_wasm_bridge_load_model_bytes(OcWasmBridge *br,
                                        const uint8_t *data, size_t len);


/* Generate text from `prompt`. For each produced token, `on_token` (if not caller the generation did not fit and must not be treated as complete. */
size_t oc_wasm_bridge_generate(OcWasmBridge *br,
                               const char *prompt,
                               uint32_t max_tokens,
                               OcWasmTokenCallback on_token,
                               void *userdata,
                               char *out_buf, size_t out_cap);

/* Request cancellation of any in-flight generation. */
OcError oc_wasm_bridge_cancel(OcWasmBridge *br);


/* Copy the current stats snapshot into `*out`. Returns OC_OK or
 * OC_ERR_INVALID_ARG if `br`/`out` is NULL. */
OcError oc_wasm_bridge_get_stats(OcWasmBridge *br, OcWasmStats *out);


/* Install a host hook table so model loading and token sampling run the host's real forward-pass implementation instead of the built-in stub. */
OcError oc_wasm_bridge_install_hooks(OcWasmBridge *br,
                                     const OcWasmHostHooks *hooks,
                                     void *userdata);


/* Enqueue a message onto the bridge's internal queue. The caller retains */
OcError oc_wasm_bridge_enqueue(OcWasmBridge *br, OcWasmMessage *msg);

/* Drain one message from the queue and act on it. */
OcError oc_wasm_bridge_drain_one(OcWasmBridge *br);

/* Current queue depth (0 if `br` is NULL). */
uint32_t oc_wasm_bridge_queue_depth(const OcWasmBridge *br);


/* Format the TypeScript interface contract for this bridge into `buf` (up */
size_t oc_wasm_bridge_format_interface(char *buf, size_t cap);

/* Returns a pointer to a static, NUL-terminated string containing the full
 * TypeScript interface contract. Never returns NULL. The pointed-to memory
 * is valid for the lifetime of the process and must not be freed. */
const char *oc_wasm_bridge_interface_string(void);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_WASM_H */
