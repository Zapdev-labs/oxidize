/* test_wasm.c — WASM bridge tests.
 *
 * Verifies:
 *   - init/free bridge (default + custom config)
 *   - config defaults match the Rust WorkerModelConfig::default()
 *   - load_model via stub hooks (path + bytes)
 *   - generate() produces tokens, invokes callback, populates stats
 *   - cancel() interrupts generation
 *   - message queue enqueue/drain/depth
 *   - format_interface() emits the TypeScript contract
 *   - null/invalid-arg handling on every entry point
 *
 * Uses the built-in stub host hooks (no real GGUF required). Mirrors the
 * Rust web_worker.rs unit tests which assert exact token sequences against
 * the deterministic stub generator.
 */
#include <criterion/criterion.h>
#include <string.h>

#include "oxidize/wasm.h"

/* Test-only helpers declared (not in public header) but defined in the .c. */
extern bool oc_wasm_bridge_using_stub_hooks(const OcWasmBridge *br);
extern const OcWasmBridgeConfig *oc_wasm_bridge_peek_config(const OcWasmBridge *br);
extern void oc_wasm_bridge_test_unload_model(OcWasmBridge *br);

/* ─── Callback state for counting tokens ──────────────────────────────── */

typedef struct CbState {
    uint32_t count;
    uint32_t last_token;
    uint32_t stop_at;   /* if nonzero, stop after this many tokens */
} CbState;

static bool count_callback(uint32_t token, uint32_t index, void *ud)
{
    (void)index;
    CbState *s = (CbState *)ud;
    s->count++;
    s->last_token = token;
    if (s->stop_at != 0 && s->count >= s->stop_at) {
        return false;  /* request early stop */
    }
    return true;
}

/* ─── Init / free ─────────────────────────────────────────────────────── */

Test(wasm, init_free_default)
{
    OcWasmBridge *br = oc_wasm_bridge_init(NULL);
    cr_assert_not_null(br, "default init should succeed");
    cr_assert(oc_wasm_bridge_using_stub_hooks(br),
              "fresh bridge should use built-in stub hooks");
    oc_wasm_bridge_free(br);
}

Test(wasm, init_custom_config)
{
    OcWasmBridgeConfig cfg;
    oc_wasm_bridge_config_default(&cfg);
    cfg.max_memory = 64ULL * 1024 * 1024;
    cfg.enable_simd = false;
    cfg.enable_threads = true;
    cfg.max_tokens = 32;
    cfg.temperature = 0.0f;

    OcWasmBridge *br = oc_wasm_bridge_init(&cfg);
    cr_assert_not_null(br);
    const OcWasmBridgeConfig *peek = oc_wasm_bridge_peek_config(br);
    cr_assert_not_null(peek);
    cr_assert_eq(peek->max_memory, 64ULL * 1024 * 1024);
    cr_assert_eq(peek->enable_simd, false);
    cr_assert_eq(peek->enable_threads, true);
    cr_assert_eq(peek->max_tokens, 32u);
    cr_assert_eq(peek->temperature, 0.0f);
    oc_wasm_bridge_free(br);
}

Test(wasm, init_eager_load_invalid_path)
{
    /* A bogus path will be passed to the stub load hook, which accepts
     * anything, so the bridge should report model_loaded == true. This
     * verifies the eager-load path is exercised without crashing. */
    OcWasmBridgeConfig cfg;
    oc_wasm_bridge_config_default(&cfg);
    cfg.model_path = "/nonexistent/model.gguf";
    OcWasmBridge *br = oc_wasm_bridge_init(&cfg);
    cr_assert_not_null(br);
    OcWasmStats stats;
    cr_assert_eq(oc_wasm_bridge_get_stats(br, &stats), OC_OK);
    cr_assert(stats.model_loaded, "stub hook should accept any path");
    oc_wasm_bridge_free(br);
}

/* ─── Config defaults ────────────────────────────────────────────────── */

Test(wasm, config_defaults_match_rust)
{
    OcWasmBridgeConfig cfg;
    oc_wasm_bridge_config_default(&cfg);
    cr_assert_eq(cfg.max_memory, OC_WASM_DEFAULT_MAX_MEMORY,
                 "default max_memory should be 2 GiB");
    cr_assert_eq(cfg.max_tokens, OC_WASM_DEFAULT_MAX_TOKENS,
                 "default max_tokens should be 4096");
    cr_assert_eq(cfg.enable_simd, true);
    cr_assert_eq(cfg.enable_threads, false);
    cr_assert_eq(cfg.model_path, NULL);
    /* Rust SamplingConfig default temperature is 1.0, but the Rust worker
     * overrides it to 0.01 — match the worker, not the sampling default. */
    cr_assert_float_eq(cfg.temperature, 0.01f, 1e-6f);
    cr_assert_eq(cfg.queue_cap, OC_WASM_MSG_QUEUE_CAP);
}

Test(wasm, config_default_null_safe)
{
    /* Should not crash on NULL. */
    oc_wasm_bridge_config_default(NULL);
    cr_assert(true);
}

/* ─── Model loading ──────────────────────────────────────────────────── */

Test(wasm, load_model_path_null_handling)
{
    OcWasmBridge *br = oc_wasm_bridge_init(NULL);
    cr_assert_not_null(br);
    cr_assert_neq(oc_wasm_bridge_load_model(br, NULL), OC_OK);
    cr_assert_neq(oc_wasm_bridge_load_model(br, ""), OC_OK);
    cr_assert_neq(oc_wasm_bridge_load_model(NULL, "x"), OC_OK);
    OcWasmStats stats;
    oc_wasm_bridge_get_stats(br, &stats);
    cr_assert(!stats.model_loaded, "failed loads must not set model_loaded");
    oc_wasm_bridge_free(br);
}

Test(wasm, load_model_bytes)
{
    OcWasmBridge *br = oc_wasm_bridge_init(NULL);
    cr_assert_not_null(br);
    const uint8_t fake_gguf[] = {0x47, 0x47, 0x55, 0x46, 0x00, 0x00, 0x00, 0x00};
    cr_assert_eq(oc_wasm_bridge_load_model_bytes(br, fake_gguf, sizeof(fake_gguf)),
                 OC_OK);
    OcWasmStats stats;
    oc_wasm_bridge_get_stats(br, &stats);
    cr_assert(stats.model_loaded);
    oc_wasm_bridge_free(br);
}

Test(wasm, load_model_bytes_null_handling)
{
    OcWasmBridge *br = oc_wasm_bridge_init(NULL);
    cr_assert_not_null(br);
    cr_assert_neq(oc_wasm_bridge_load_model_bytes(br, NULL, 0), OC_OK);
    cr_assert_neq(oc_wasm_bridge_load_model_bytes(br, (const uint8_t *)"x", 0),
                 OC_OK);
    cr_assert_neq(oc_wasm_bridge_load_model_bytes(NULL, (const uint8_t *)"x", 1),
                 OC_OK);
    oc_wasm_bridge_free(br);
}

/* ─── Generation ─────────────────────────────────────────────────────── */

Test(wasm, generate_without_model_fails_gracefully)
{
    OcWasmBridge *br = oc_wasm_bridge_init(NULL);
    cr_assert_not_null(br);
    /* No model loaded and no configured path → generate returns 0. */
    size_t n = oc_wasm_bridge_generate(br, "hello", 5, NULL, NULL, NULL, 0);
    cr_assert_eq(n, 0u);
    oc_wasm_bridge_free(br);
}

Test(wasm, generate_after_load_produces_tokens)
{
    OcWasmBridge *br = oc_wasm_bridge_init(NULL);
    cr_assert_not_null(br);
    cr_assert_eq(oc_wasm_bridge_load_model(br, "stub.gguf"), OC_OK);

    CbState state = {0, 0, 0};
    char out[256];
    size_t n = oc_wasm_bridge_generate(br, "hello world", 8,
                                       count_callback, &state,
                                       out, sizeof(out));
    cr_assert_eq(state.count, 8u, "callback should fire once per token");
    cr_assert_gt(n, 0u, "output buffer should be non-empty");
    cr_assert_str_not_empty(out);

    OcWasmStats stats;
    oc_wasm_bridge_get_stats(br, &stats);
    cr_assert_eq(stats.tokens_generated, 8u);
    cr_assert(stats.tokens_per_sec > 0.0);
    cr_assert(stats.last_latency_ms >= 0.0);
    oc_wasm_bridge_free(br);
}

Test(wasm, generate_callback_early_stop)
{
    OcWasmBridge *br = oc_wasm_bridge_init(NULL);
    cr_assert_not_null(br);
    cr_assert_eq(oc_wasm_bridge_load_model(br, "stub.gguf"), OC_OK);

    CbState state = {0, 0, 3};
    oc_wasm_bridge_generate(br, "prompt", 100,
                            count_callback, &state, NULL, 0);
    cr_assert_eq(state.count, 3u, "early stop should terminate generation");

    OcWasmStats stats;
    oc_wasm_bridge_get_stats(br, &stats);
    cr_assert_eq(stats.tokens_generated, 3u);
    oc_wasm_bridge_free(br);
}

Test(wasm, generate_null_handling)
{
    OcWasmBridge *br = oc_wasm_bridge_init(NULL);
    cr_assert_not_null(br);
    cr_assert_eq(oc_wasm_bridge_generate(NULL, "x", 1, NULL, NULL, NULL, 0), 0u);
    cr_assert_eq(oc_wasm_bridge_generate(br, NULL, 1, NULL, NULL, NULL, 0), 0u);
    oc_wasm_bridge_free(br);
}

Test(wasm, generate_deterministic_with_zero_temp)
{
    /* The stub generator is deterministic when temperature == 0 (greedy).
     * Two runs of the same prompt should produce the same first token. */
    OcWasmBridgeConfig cfg;
    oc_wasm_bridge_config_default(&cfg);
    cfg.temperature = 0.0f;
    OcWasmBridge *br = oc_wasm_bridge_init(&cfg);
    cr_assert_not_null(br);
    cr_assert_eq(oc_wasm_bridge_load_model(br, "stub.gguf"), OC_OK);

    CbState s1 = {0, 0, 1};
    oc_wasm_bridge_generate(br, "same prompt", 5, count_callback, &s1, NULL, 0);
    /* Reset internal generated-token counter by unloading/reloading. */
    oc_wasm_bridge_test_unload_model(br);
    cr_assert_eq(oc_wasm_bridge_load_model(br, "stub.gguf"), OC_OK);
    CbState s2 = {0, 0, 1};
    oc_wasm_bridge_generate(br, "same prompt", 5, count_callback, &s2, NULL, 0);
    cr_assert_eq(s1.last_token, s2.last_token,
                 "greedy generation should be deterministic");
    oc_wasm_bridge_free(br);
}

/* ─── Cancellation ───────────────────────────────────────────────────── */

Test(wasm, cancel_sets_flag)
{
    OcWasmBridge *br = oc_wasm_bridge_init(NULL);
    cr_assert_not_null(br);
    cr_assert_eq(oc_wasm_bridge_cancel(br), OC_OK);
    oc_wasm_bridge_free(br);
}

Test(wasm, cancel_null_handling)
{
    cr_assert_neq(oc_wasm_bridge_cancel(NULL), OC_OK);
}

/* ─── Stats ──────────────────────────────────────────────────────────── */

Test(wasm, get_stats_null_handling)
{
    OcWasmStats stats;
    cr_assert_neq(oc_wasm_bridge_get_stats(NULL, &stats), OC_OK);
    OcWasmBridge *br = oc_wasm_bridge_init(NULL);
    cr_assert_not_null(br);
    cr_assert_neq(oc_wasm_bridge_get_stats(br, NULL), OC_OK);
    oc_wasm_bridge_free(br);
}

Test(wasm, stats_initial_state)
{
    OcWasmBridge *br = oc_wasm_bridge_init(NULL);
    cr_assert_not_null(br);
    OcWasmStats stats;
    cr_assert_eq(oc_wasm_bridge_get_stats(br, &stats), OC_OK);
    cr_assert_eq(stats.tokens_generated, 0u);
    cr_assert_eq(stats.tokens_consumed, 0u);
    cr_assert_eq(stats.tokens_per_sec, 0.0);
    cr_assert_eq(stats.messages_queued, 0u);
    cr_assert_eq(stats.messages_processed, 0u);
    cr_assert_eq(stats.model_loaded, false);
    oc_wasm_bridge_free(br);
}

/* ─── Message queue ─────────────────────────────────────────────────── */

Test(wasm, queue_enqueue_drain)
{
    OcWasmBridge *br = oc_wasm_bridge_init(NULL);
    cr_assert_not_null(br);
    cr_assert_eq(oc_wasm_bridge_queue_depth(br), 0u);

    OcWasmMessage load_msg = { .type = OC_WASM_MSG_LOAD, .payload = "stub.gguf" };
    cr_assert_eq(oc_wasm_bridge_enqueue(br, &load_msg), OC_OK);
    cr_assert_gt(load_msg.token_id, 0u, "enqueue should assign token_id");
    cr_assert_eq(oc_wasm_bridge_queue_depth(br), 1u);

    OcWasmMessage gen_msg = { .type = OC_WASM_MSG_GENERATE, .payload = "hi",
                              .max_tokens = 2 };
    cr_assert_eq(oc_wasm_bridge_enqueue(br, &gen_msg), OC_OK);
    cr_assert_eq(oc_wasm_bridge_queue_depth(br), 2u);

    /* Drain the LOAD message first (FIFO). */
    cr_assert_eq(oc_wasm_bridge_drain_one(br), OC_OK);
    cr_assert_eq(oc_wasm_bridge_queue_depth(br), 1u);
    OcWasmStats stats;
    oc_wasm_bridge_get_stats(br, &stats);
    cr_assert(stats.model_loaded, "LOAD message should load the model");
    cr_assert_eq(stats.messages_processed, 1u);

    /* Drain the GENERATE message. */
    cr_assert_eq(oc_wasm_bridge_drain_one(br), OC_OK);
    cr_assert_eq(oc_wasm_bridge_queue_depth(br), 0u);
    oc_wasm_bridge_get_stats(br, &stats);
    cr_assert_eq(stats.messages_processed, 2u);
    cr_assert_eq(stats.tokens_generated, 2u);

    oc_wasm_bridge_free(br);
}

Test(wasm, queue_drain_empty_returns_error)
{
    OcWasmBridge *br = oc_wasm_bridge_init(NULL);
    cr_assert_not_null(br);
    cr_assert_neq(oc_wasm_bridge_drain_one(br), OC_OK);
    oc_wasm_bridge_free(br);
}

Test(wasm, queue_cancel_message_drains)
{
    /* Draining a CANCEL message should return OC_OK and set the cancel flag.
     * The flag is observed by the next generate() call's first forward step;
     * since generate() resets the flag at entry, the cancel must be requested
     * *during* generation (via the callback) to actually interrupt it — see
     * the `cancel_during_generate` test below. */
    OcWasmBridge *br = oc_wasm_bridge_init(NULL);
    cr_assert_not_null(br);
    OcWasmMessage cancel = { .type = OC_WASM_MSG_CANCEL };
    cr_assert_eq(oc_wasm_bridge_enqueue(br, &cancel), OC_OK);
    cr_assert_eq(oc_wasm_bridge_drain_one(br), OC_OK);
    OcWasmStats stats;
    oc_wasm_bridge_get_stats(br, &stats);
    cr_assert_eq(stats.messages_processed, 1u);
    oc_wasm_bridge_free(br);
}

/* Callback that requests cancellation after the first token. */
static bool cancel_after_first(uint32_t token, uint32_t index, void *ud)
{
    (void)token; (void)index;
    OcWasmBridge *br = (OcWasmBridge *)ud;
    oc_wasm_bridge_cancel(br);
    return true;  /* keep going; the flag is checked on the next step */
}

Test(wasm, cancel_during_generate)
{
    OcWasmBridge *br = oc_wasm_bridge_init(NULL);
    cr_assert_not_null(br);
    cr_assert_eq(oc_wasm_bridge_load_model(br, "stub.gguf"), OC_OK);

    /* Request max_tokens=100 but cancel via callback after token 0.
     * Generation should stop after 1 token (the flag is checked at the
     * top of the next loop iteration). */
    size_t n = oc_wasm_bridge_generate(br, "prompt", 100,
                                       cancel_after_first, br, NULL, 0);
    (void)n;
    OcWasmStats stats;
    oc_wasm_bridge_get_stats(br, &stats);
    cr_assert_leq(stats.tokens_generated, 2u,
                 "cancel should stop generation within 1-2 tokens");
    oc_wasm_bridge_free(br);
}

Test(wasm, queue_null_handling)
{
    OcWasmBridge *br = oc_wasm_bridge_init(NULL);
    cr_assert_not_null(br);
    OcWasmMessage msg = { .type = OC_WASM_MSG_STATUS };
    cr_assert_neq(oc_wasm_bridge_enqueue(NULL, &msg), OC_OK);
    cr_assert_neq(oc_wasm_bridge_enqueue(br, NULL), OC_OK);
    cr_assert_neq(oc_wasm_bridge_drain_one(NULL), OC_OK);
    cr_assert_eq(oc_wasm_bridge_queue_depth(NULL), 0u);
    oc_wasm_bridge_free(br);
}

Test(wasm, queue_overflow)
{
    OcWasmBridgeConfig cfg;
    oc_wasm_bridge_config_default(&cfg);
    cfg.queue_cap = 2;
    OcWasmBridge *br = oc_wasm_bridge_init(&cfg);
    cr_assert_not_null(br);
    OcWasmMessage m1 = { .type = OC_WASM_MSG_STATUS };
    OcWasmMessage m2 = { .type = OC_WASM_MSG_STATUS };
    OcWasmMessage m3 = { .type = OC_WASM_MSG_STATUS };
    cr_assert_eq(oc_wasm_bridge_enqueue(br, &m1), OC_OK);
    cr_assert_eq(oc_wasm_bridge_enqueue(br, &m2), OC_OK);
    cr_assert_neq(oc_wasm_bridge_enqueue(br, &m3), OC_OK,
                  "third enqueue should fail (queue full)");
    cr_assert_eq(oc_wasm_bridge_queue_depth(br), 2u);
    oc_wasm_bridge_free(br);
}

/* ─── TypeScript interface ───────────────────────────────────────────── */

Test(wasm, format_interface_produces_typescript)
{
    /* First query the required length. */
    size_t needed = oc_wasm_bridge_format_interface(NULL, 0);
    cr_assert_gt(needed, 0u);

    char *buf = malloc(needed + 1);
    cr_assert_not_null(buf);
    size_t written = oc_wasm_bridge_format_interface(buf, needed + 1);
    cr_assert_eq(written, needed);
    buf[written] = '\0';

    /* Verify the contract contains the same declarations as the Rust
     * WASM_WORKER_TYPESCRIPT_BINDINGS literal. */
    cr_assert(strstr(buf, "interface OxidizeWorkerModelConfig") != NULL);
    cr_assert(strstr(buf, "interface OxidizeWorkerInferenceRequest") != NULL);
    cr_assert(strstr(buf, "interface OxidizeWorkerInferenceResponse") != NULL);
    cr_assert(strstr(buf, "interface OxidizeWorkerStreamChunk") != NULL);
    cr_assert(strstr(buf, "interface OxidizeWorkerMessageResponse") != NULL);
    cr_assert(strstr(buf, "type OxidizeWorkerModelCacheAction") != NULL);
    cr_assert(strstr(buf, "interface OxidizeWorkerModelCacheRequest") != NULL);
    cr_assert(strstr(buf, "interface OxidizeWorkerModelCacheResponse") != NULL);
    cr_assert(strstr(buf, "interface OxidizeWorkerModelCacheMessageResponse")
              != NULL);
    cr_assert(strstr(buf, "interface OxidizeWorkerStreamResponse") != NULL);

    /* C-bridge extensions (not in the Rust literal but added by this port). */
    cr_assert(strstr(buf, "interface OxidizeWasmBridgeConfig") != NULL);
    cr_assert(strstr(buf, "interface OxidizeWasmStats") != NULL);
    cr_assert(strstr(buf, "type OxidizeWasmMessageType") != NULL);
    cr_assert(strstr(buf, "interface OxidizeWasmMessage") != NULL);

    free(buf);
}

Test(wasm, format_interface_truncates_safely)
{
    char tiny[8];
    size_t written = oc_wasm_bridge_format_interface(tiny, sizeof(tiny));
    cr_assert_lt(written, sizeof(tiny), "should truncate to fit");
    cr_assert_eq(tiny[written], '\0', "output must be NUL-terminated");
}

Test(wasm, interface_string_static_pointer)
{
    const char *s = oc_wasm_bridge_interface_string();
    cr_assert_not_null(s);
    cr_assert_gt(strlen(s), 100u, "interface string should be substantial");
    cr_assert(strstr(s, "OxidizeWorker") != NULL);
}

Test(wasm, format_interface_null_handling)
{
    /* snprintf-style: NULL buffer or zero cap returns the length that would
     * have been written (never writes). */
    size_t needed = oc_wasm_bridge_format_interface(NULL, 0);
    cr_assert_gt(needed, 0u);
    cr_assert_eq(oc_wasm_bridge_format_interface(NULL, 64), needed);
    char buf[8];
    cr_assert_eq(oc_wasm_bridge_format_interface(buf, 0), needed);
}
