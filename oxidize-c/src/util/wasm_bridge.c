/* wasm_bridge.c — WebAssembly bridge implementation. */

#define _POSIX_C_SOURCE 200809L

#include "oxidize/wasm.h"

#include "oxidize/mem_util.h"

#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__linux__) || defined(__APPLE__)
#define OC_WASM_HAS_CLOCK 1
#else
#define OC_WASM_HAS_CLOCK 0
#endif


/* The OcWasmHostHooks table and oc_wasm_bridge_install_hooks() are declared
 * in the public header (wasm.h) so downstream WASM users can attach a real
 * forward-pass implementation. */


/* Hidden definition referenced by the public opaque typedef in wasm.h. */
struct OcWasmBridge {
    OcWasmBridgeConfig  cfg;
    OcWasmStats         stats;
    OcWasmHostHooks     hooks;
    void               *hooks_userdata;
    bool                hooks_installed;
    /* Message queue (ring buffer). `queue` is NULL if queue_cap == 0. */
    OcWasmMessage      *queue;
    uint32_t            queue_cap;
    uint32_t            queue_head;    /* pop index                          */
    uint32_t            queue_tail;    /* push index                         */
    uint32_t            next_token_id; /* monotonic seq for enqueued msgs    */
    /* Cancellation flag observed by the generate loop. Atomic so a
     * concurrent cancel (threads enabled) is race-free. */
    atomic_bool         cancel_requested;
    /* Allocated bytes tracked by this bridge (for memory_used in pure WASM). */
    uint64_t            alloc_bytes;
};


static bool  stub_load_model_bytes(const uint8_t *data, size_t len,
                                   const char *path, void *userdata);
static uint32_t stub_sample_token(const char *prompt,
                                 const uint32_t *generated, size_t n_generated,
                                 uint32_t max_tokens, float temperature,
                                 void *userdata);
static void  stub_release(void *userdata);

static const OcWasmHostHooks STUB_HOOKS = {
    .load_model_bytes = stub_load_model_bytes,
    .sample_token     = stub_sample_token,
    .release          = stub_release,
};


static double now_ms(void)
{
#if OC_WASM_HAS_CLOCK
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
#else
    /* Fallback: clock() is process CPU time, not wall clock, but is
     * available on every C11 host including freestanding WASI. */
    return (double)clock() * (1000.0 / (double)CLOCKS_PER_SEC);
#endif
}


void oc_wasm_bridge_config_default(OcWasmBridgeConfig *cfg)
{
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->max_memory    = OC_WASM_DEFAULT_MAX_MEMORY;
    cfg->enable_simd   = true;
    cfg->enable_threads = false;
    cfg->model_path    = NULL;
    cfg->max_tokens    = OC_WASM_DEFAULT_MAX_TOKENS;
    cfg->temperature   = 0.01f;     /* match Rust WorkerInferenceRequest */
    cfg->queue_cap     = OC_WASM_MSG_QUEUE_CAP;
}


/* Track an allocation of `n` bytes against the bridge's accounting. */
static void track_alloc(OcWasmBridge *br, size_t n)
{
    if (!br) return;
    br->alloc_bytes += n;
    if (br->alloc_bytes > br->stats.memory_peak) {
        br->stats.memory_peak = br->alloc_bytes;
    }
}

static void track_free(OcWasmBridge *br, size_t n)
{
    if (!br) return;
    if (n > br->alloc_bytes) {
        br->alloc_bytes = 0;
    } else {
        br->alloc_bytes -= n;
    }
}

/* Refresh `memory_used` from the best available source. On WASM (no
 * /proc) we fall back to the internal malloc accounting. */
static void refresh_memory_used(OcWasmBridge *br)
{
    if (!br) return;
    uint64_t rss = oc_mem_rss_bytes();
    /* If RSS is available and non-zero, prefer it (matches Rust core
     * which uses the platform memory API). Otherwise use the bridge's
     * own accounting. */
    if (rss != 0) {
        br->stats.memory_used = rss;
    } else {
        br->stats.memory_used = br->alloc_bytes;
    }
    if (br->stats.memory_used > br->stats.memory_peak) {
        br->stats.memory_peak = br->stats.memory_used;
    }
}


OcWasmBridge *oc_wasm_bridge_init(const OcWasmBridgeConfig *cfg)
{
    OcWasmBridgeConfig defaults;
    if (!cfg) {
        oc_wasm_bridge_config_default(&defaults);
        cfg = &defaults;
    }

    OcWasmBridge *br = calloc(1, sizeof(*br));
    if (!br) return NULL;
    br->cfg  = *cfg;
    br->stats.memory_used = 0;
    br->stats.memory_peak = 0;
    br->hooks = STUB_HOOKS;
    br->hooks_installed = false;
    br->cancel_requested = false;
    br->alloc_bytes = sizeof(*br);

    /* Normalize queue capacity: must be a power of two, nonzero if the
     * caller wants a queue. 0 means "no queue" (direct-call mode). */
    uint32_t qcap = cfg->queue_cap;
    if (qcap == 0) {
        qcap = OC_WASM_MSG_QUEUE_CAP;
    }
    /* Round up to the next power of two (matches the Rust HashMap's
     * capacity growth semantics: caller-passed arbitrary values are
     * tolerated). */
    if (qcap & (qcap - 1)) {
        uint32_t v = 1;
        while (v < qcap && v != 0) v <<= 1;
        qcap = v == 0 ? OC_WASM_MSG_QUEUE_CAP : v;
    }
    br->queue_cap = qcap;
    /* Enforce the caller's memory budget before allocating the queue. */
    uint64_t budget = br->cfg.max_memory ? br->cfg.max_memory
                                         : OC_WASM_DEFAULT_MAX_MEMORY;
    if (sizeof(*br) + (uint64_t)qcap * sizeof(OcWasmMessage) > budget) {
        free(br);
        return NULL;
    }
    br->queue = calloc(qcap, sizeof(OcWasmMessage));
    if (!br->queue) {
        free(br);
        return NULL;
    }
    track_alloc(br, (size_t)qcap * sizeof(OcWasmMessage));
    br->queue_head = 0;
    br->queue_tail = 0;
    br->next_token_id = 1;
    br->stats.messages_queued = 0;
    br->stats.messages_processed = 0;
    br->stats.model_loaded = false;

    /* If the caller supplied a model_path, attempt to load it eagerly. We
     * tolerate failures here (mirrors Rust's lazy-load semantics: the
     * first generate() will re-attempt). */
    if (cfg->model_path && cfg->model_path[0] != '\0') {
        (void)oc_wasm_bridge_load_model(br, cfg->model_path);
    }

    refresh_memory_used(br);
    return br;
}

void oc_wasm_bridge_free(OcWasmBridge *br)
{
    if (!br) return;
    /* Release host hooks state if installed. */
    if (br->hooks_installed && br->hooks.release) {
        br->hooks.release(br->hooks_userdata);
    }
    if (br->queue) {
        track_free(br, (size_t)br->queue_cap * sizeof(OcWasmMessage));
        free(br->queue);
        br->queue = NULL;
    }
    track_free(br, sizeof(*br));
    free(br);
}


OcError oc_wasm_bridge_load_model(OcWasmBridge *br, const char *path)
{
    if (!br) return OC_ERR_INVALID_ARG;
    if (!path || path[0] == '\0') return OC_ERR_INVALID_ARG;

    /* The host hook is responsible for actually loading bytes from `path`.
     * If no hook is installed, the stub accepts any path (it only needs the
     * non-NULL check). */
    bool ok = false;
    if (br->hooks.load_model_bytes) {
        ok = br->hooks.load_model_bytes(NULL, 0, path, br->hooks_userdata);
    }
    if (!ok) {
        return OC_ERR_IO;
    }
    br->stats.model_loaded = true;
    refresh_memory_used(br);
    return OC_OK;
}

OcError oc_wasm_bridge_load_model_bytes(OcWasmBridge *br,
                                        const uint8_t *data, size_t len)
{
    if (!br) return OC_ERR_INVALID_ARG;
    if (!data || len == 0) return OC_ERR_INVALID_ARG;

    bool ok = false;
    if (br->hooks.load_model_bytes) {
        ok = br->hooks.load_model_bytes(data, len, NULL, br->hooks_userdata);
    }
    if (!ok) {
        return OC_ERR_FORMAT;
    }
    br->stats.model_loaded = true;
    refresh_memory_used(br);
    return OC_OK;
}


size_t oc_wasm_bridge_generate(OcWasmBridge *br,
                               const char *prompt,
                               uint32_t max_tokens,
                               OcWasmTokenCallback on_token,
                               void *userdata,
                               char *out_buf, size_t out_cap)
{
    if (!br) return 0;
    if (!prompt) return 0;
    if (max_tokens == 0) max_tokens = br->cfg.max_tokens;
    if (max_tokens == 0) max_tokens = OC_WASM_DEFAULT_MAX_TOKENS;

    if (!br->stats.model_loaded) {
        /* Try to auto-load via the configured path, if any. */
        if (br->cfg.model_path && br->cfg.model_path[0] != '\0') {
            if (oc_wasm_bridge_load_model(br, br->cfg.model_path) != OC_OK) {
                return 0;
            }
        } else {
            return 0;
        }
    }

    /* Reset cancellation flag for this call. */
    br->cancel_requested = false;

    /* Consumed tokens = prompt length (token-equivalent: we count UTF-8
     * bytes as a coarse proxy, mirroring the Rust worker which counts
     * prompt_tokens.len()). */
    size_t prompt_len = strlen(prompt);
    br->stats.tokens_consumed += prompt_len > 0 ? prompt_len : 1;

    /* Output buffer cursor. */
    size_t out_written = 0;
    bool truncated = false;
    if (out_buf && out_cap > 0) {
        out_buf[0] = '\0';
    }

    /* Generated token ids (we keep a small on-stack buffer; for larger
     * generations we fall back to a heap buffer). */
    uint32_t stack_buf[64];
    uint32_t *generated = stack_buf;
    size_t generated_cap = 64;
    bool heap_buf = false;
    uint64_t budget = br->cfg.max_memory ? br->cfg.max_memory
                                         : OC_WASM_DEFAULT_MAX_MEMORY;
    if ((size_t)max_tokens > generated_cap &&
        br->alloc_bytes + (uint64_t)max_tokens * sizeof(uint32_t) > budget) {
        /* Would exceed the memory budget: stay on the stack buffer. */
        max_tokens = (uint32_t)generated_cap;
    }
    if ((size_t)max_tokens > generated_cap) {
        generated = malloc((size_t)max_tokens * sizeof(uint32_t));
        if (!generated) {
            generated = stack_buf;
            generated_cap = 64;
            if ((uint32_t)max_tokens > generated_cap) {
                max_tokens = (uint32_t)generated_cap;
            }
        } else {
            heap_buf = true;
            generated_cap = max_tokens;
            track_alloc(br, generated_cap * sizeof(uint32_t));
        }
    }

    double t_start = now_ms();
    uint32_t produced = 0;
    float temperature = br->cfg.temperature;

    for (uint32_t i = 0; i < max_tokens; i++) {
        /* Honor cancellation. Mirrors the Rust streaming worker which
         * breaks on Poll::Ready(None) when the host drops the stream. */
        if (br->cancel_requested) break;

        uint32_t tok = 0;
        if (br->hooks.sample_token) {
            tok = br->hooks.sample_token(prompt, generated, produced,
                                         max_tokens, temperature,
                                         br->hooks_userdata);
        }
        if (tok == 0) break;  /* host reported end-of-stream / failure */

        if (produced < generated_cap) {
            generated[produced] = tok;
        }
        produced++;
        br->stats.tokens_generated++;

        /* Append a textual representation of the token to the output */
        if (out_buf && out_cap > 0) {
            char tok_str[16];
            int n = snprintf(tok_str, sizeof(tok_str), "%u ", tok);
            if (n > 0) {
                if (out_written + (size_t)n + 1 <= out_cap) {
                    memcpy(out_buf + out_written, tok_str, (size_t)n);
                    out_written += (size_t)n;
                    out_buf[out_written] = '\0';
                } else {
                    truncated = true;
                }
            }
        }

        /* Invoke the per-token callback. If it returns false, stop. */
        if (on_token && !on_token(tok, i, userdata)) {
            break;
        }
    }

    double t_end = now_ms();
    double elapsed_ms = t_end - t_start;
    br->stats.last_latency_ms = elapsed_ms;
    if (elapsed_ms > 0.0) {
        br->stats.tokens_per_sec = (double)produced * 1000.0 / elapsed_ms;
    } else {
        /* Avoid div-by-zero; if we produced tokens in zero measured time
         * (e.g. clock resolution), report a very high throughput. */
        br->stats.tokens_per_sec = produced > 0 ? 1e9 : 0.0;
    }

    if (heap_buf) {
        track_free(br, generated_cap * sizeof(uint32_t));
        free(generated);
    }
    refresh_memory_used(br);
    /* Per the API contract, overflow (truncated output) returns 0; the
     * buffer still holds the NUL-terminated partial text. */
    return truncated ? 0 : out_written;
}

OcError oc_wasm_bridge_cancel(OcWasmBridge *br)
{
    if (!br) return OC_ERR_INVALID_ARG;
    br->cancel_requested = true;
    return OC_OK;
}


OcError oc_wasm_bridge_get_stats(OcWasmBridge *br, OcWasmStats *out)
{
    if (!br || !out) return OC_ERR_INVALID_ARG;
    refresh_memory_used(br);
    *out = br->stats;
    return OC_OK;
}


OcError oc_wasm_bridge_enqueue(OcWasmBridge *br, OcWasmMessage *msg)
{
    if (!br || !msg) return OC_ERR_INVALID_ARG;
    if (!br->queue || br->queue_cap == 0) return OC_ERR_INTERNAL;
    if (br->stats.messages_queued >= br->queue_cap) return OC_ERR_OOM;

    uint32_t idx = br->queue_tail % br->queue_cap;
    br->queue[idx] = *msg;
    br->queue[idx].token_id = br->next_token_id++;
    br->queue_tail++;
    br->stats.messages_queued++;
    /* Reflect the assigned sequence back to the caller so they can match
     * drain responses to enqueued requests. */
    msg->token_id = br->queue[idx].token_id;
    return OC_OK;
}

OcError oc_wasm_bridge_drain_one(OcWasmBridge *br)
{
    if (!br) return OC_ERR_INVALID_ARG;
    if (!br->queue || br->stats.messages_queued == 0) return OC_ERR_FORMAT;

    uint32_t idx = br->queue_head % br->queue_cap;
    OcWasmMessage msg = br->queue[idx];
    br->queue_head++;
    br->stats.messages_queued--;
    br->stats.messages_processed++;

    switch (msg.type) {
    case OC_WASM_MSG_LOAD:
        if (!msg.payload) return OC_ERR_INVALID_ARG;
        return oc_wasm_bridge_load_model(br, msg.payload);
    case OC_WASM_MSG_GENERATE:
        if (!msg.payload) return OC_ERR_INVALID_ARG;
        oc_wasm_bridge_generate(br, msg.payload, msg.max_tokens,
                                 NULL, NULL, NULL, 0);
        return OC_OK;
    case OC_WASM_MSG_CANCEL:
        return oc_wasm_bridge_cancel(br);
    case OC_WASM_MSG_STATUS:
        /* No-op; caller should poll oc_wasm_bridge_get_stats. */
        return OC_OK;
    case OC_WASM_MSG_NONE:
    default:
        return OC_ERR_FORMAT;
    }
}

uint32_t oc_wasm_bridge_queue_depth(const OcWasmBridge *br)
{
    if (!br) return 0;
    return br->stats.messages_queued;
}


/* The stub generator produces deterministic tokens derived from the prompt */

static bool stub_load_model_bytes(const uint8_t *data, size_t len,
                                  const char *path, void *userdata)
{
    (void)data; (void)len; (void)path; (void)userdata;
    /* The stub unconditionally accepts the model. A real host would parse
     * the GGUF header here. */
    return true;
}

/* Simple FNV-1a hash to deterministically derive a seed from the prompt. */
static uint32_t prompt_hash(const char *s)
{
    uint32_t h = 2166136261u;
    if (!s) return h;
    while (*s) {
        h ^= (uint8_t)*s++;
        h *= 16777619u;
    }
    return h;
}

static uint32_t stub_sample_token(const char *prompt,
                                 const uint32_t *generated, size_t n_generated,
                                 uint32_t max_tokens, float temperature,
                                 void *userdata)
{
    (void)max_tokens; (void)userdata;
    /* Mirror the Rust stub generator which echoes the last prompt token */
    if (temperature <= 0.0f) {
        /* Greedy: always return the same token (the Rust default echoes
         * the last prompt token, e.g. vec![9,9,9] for prompt [7,9]). */
        if (n_generated == 0) {
            uint32_t h = prompt_hash(prompt);
            return (h % 1000) + 1;  /* 1..1000, avoid 0 (EOS sentinel) */
        }
        return generated[n_generated - 1];
    }
    /* Non-greedy: deterministic pseudo-random walk seeded by prompt.
     * Clamp temperature first: NaN/inf/huge values would be undefined
     * behavior in the float→uint32_t conversion. */
    if (!isfinite(temperature) || temperature > 1000.0f)
        temperature = 1000.0f;
    uint32_t h = prompt_hash(prompt);
    uint32_t t = (uint32_t)(temperature * 1000.0f) + 1;
    uint32_t tok = (h ^ (uint32_t)n_generated * t) % 1000;
    return tok + 1;  /* avoid 0 */
}

static void stub_release(void *userdata)
{
    (void)userdata;
}


/* This string mirrors the Rust `WASM_WORKER_TYPESCRIPT_BINDINGS` literal in oxidize-core/src/util/web_worker.rs, augmented with the C-bridge-specific stats interface. */
static const char WASM_INTERFACE_STRING[] =
    "\n"
    "export interface OxidizeWorkerModelConfig {\n"
    "  vocab_size: number;\n"
    "  context_size: number;\n"
    "  layer_count: number;\n"
    "}\n"
    "\n"
    "export interface OxidizeWorkerInferenceRequest {\n"
    "  prompt_tokens: number[];\n"
    "  max_new_tokens: number;\n"
    "  model: OxidizeWorkerModelConfig;\n"
    "}\n"
    "\n"
    "export interface OxidizeWorkerInferenceResponse {\n"
    "  generated_tokens: number[];\n"
    "  consumed_tokens: number;\n"
    "}\n"
    "\n"
    "export interface OxidizeWorkerStreamChunk {\n"
    "  token: number;\n"
    "  index: number;\n"
    "}\n"
    "\n"
    "export interface OxidizeWorkerMessageResponse {\n"
    "  response: OxidizeWorkerInferenceResponse | null;\n"
    "  error: string | null;\n"
    "}\n"
    "\n"
    "export type OxidizeWorkerModelCacheAction =\n"
    "  | \"cache_downloaded_model\"\n"
    "  | \"get_cached_model\"\n"
    "  | \"remove_cached_model\"\n"
    "  | \"clear_cache\"\n"
    "  | \"cache_stats\";\n"
    "\n"
    "export interface OxidizeWorkerModelCacheRequest {\n"
    "  action: OxidizeWorkerModelCacheAction;\n"
    "  model_id?: string;\n"
    "  model_bytes?: number[];\n"
    "}\n"
    "\n"
    "export interface OxidizeWorkerModelCacheResponse {\n"
    "  model_id: string | null;\n"
    "  model_bytes: number[] | null;\n"
    "  cached: boolean | null;\n"
    "  cached_models: number;\n"
    "  cached_bytes: number;\n"
    "}\n"
    "\n"
    "export interface OxidizeWorkerModelCacheMessageResponse {\n"
    "  response: OxidizeWorkerModelCacheResponse | null;\n"
    "  error: string | null;\n"
    "}\n"
    "\n"
    "export interface OxidizeWorkerStreamResponse {\n"
    "  chunks: OxidizeWorkerStreamChunk[];\n"
    "  response: OxidizeWorkerInferenceResponse | null;\n"
    "  error: string | null;\n"
    "}\n"
    "\n"
    "/* C-bridge extensions: config + stats (mirrors OcWasmBridgeConfig and\n"
    "   OcWasmStats in oxidize-c/include/oxidize/wasm.h). */\n"
    "export interface OxidizeWasmBridgeConfig {\n"
    "  max_memory: number;\n"
    "  enable_simd: boolean;\n"
    "  enable_threads: boolean;\n"
    "  model_path?: string;\n"
    "  max_tokens: number;\n"
    "  temperature: number;\n"
    "  queue_cap: number;\n"
    "}\n"
    "\n"
    "export interface OxidizeWasmStats {\n"
    "  tokens_generated: number;\n"
    "  tokens_consumed: number;\n"
    "  tokens_per_sec: number;\n"
    "  last_latency_ms: number;\n"
    "  memory_used: number;\n"
    "  memory_peak: number;\n"
    "  messages_queued: number;\n"
    "  messages_processed: number;\n"
    "  model_loaded: boolean;\n"
    "}\n"
    "\n"
    "export type OxidizeWasmMessageType =\n"
    "  | \"load\"\n"
    "  | \"generate\"\n"
    "  | \"cancel\"\n"
    "  | \"status\";\n"
    "\n"
    "export interface OxidizeWasmMessage {\n"
    "  type: OxidizeWasmMessageType;\n"
    "  payload?: string;\n"
    "  max_tokens?: number;\n"
    "  token_id: number;\n"
    "}\n";

size_t oc_wasm_bridge_format_interface(char *buf, size_t cap)
{
    size_t len = sizeof(WASM_INTERFACE_STRING) - 1;  /* exclude NUL */
    if (!buf || cap == 0) return len;
    size_t to_copy = len < cap - 1 ? len : cap - 1;
    memcpy(buf, WASM_INTERFACE_STRING, to_copy);
    buf[to_copy] = '\0';
    return to_copy;
}

const char *oc_wasm_bridge_interface_string(void)
{
    return WASM_INTERFACE_STRING;
}


OcError oc_wasm_bridge_install_hooks(OcWasmBridge *br,
                                     const OcWasmHostHooks *hooks,
                                     void *userdata)
{
    if (!br || !hooks) return OC_ERR_INVALID_ARG;
    /* Release the previously-installed hook state (if any). */
    if (br->hooks_installed && br->hooks.release) {
        br->hooks.release(br->hooks_userdata);
    }
    br->hooks = *hooks;
    br->hooks_userdata = userdata;
    br->hooks_installed = true;
    return OC_OK;
}


/* Exposed for the test suite to verify the built-in stub is in use by
 * default (no host hooks installed). Returns true if the stub hooks are
 * active (i.e. the caller has not installed custom hooks). */
bool oc_wasm_bridge_using_stub_hooks(const OcWasmBridge *br)
{
    if (!br) return false;
    return !br->hooks_installed;
}

/* Exposed for tests: returns a pointer to the (non-owning) config struct
 * inside the bridge. NULL if br is NULL. */
const OcWasmBridgeConfig *oc_wasm_bridge_peek_config(const OcWasmBridge *br)
{
    return br ? &br->cfg : NULL;
}

/* Internal helper used by tests to clear the model_loaded flag without
 * going through the host hooks (simulates an unload). */
void oc_wasm_bridge_test_unload_model(OcWasmBridge *br)
{
    if (br) br->stats.model_loaded = false;
}
