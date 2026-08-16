/*
 * openai.h — OpenAI-compatible HTTP route handlers.
 *
 * Implements the `server-openai-routes` feature on top of server-http-core.
 * Exposes the OpenAI-compatible endpoints:
 *   GET  /v1/models               → list loaded model(s)
 *   POST /v1/completions           → text completion
 *   POST /v1/chat/completions      → chat with message array
 *   POST /v1/embeddings            → embedding vectors
 *   POST /v1/responses             → Responses API
 *   POST /v1/mesh/chat/completions → chat routed across the mesh cluster
 * plus the operational routes documented further down (/healthz, /livez,
 * /readyz, /metrics, /openapi.json).
 *
 * JSON parsing is dependency-free and tailored to the OpenAI request shape.
 *
 * The handler drives oc_llama_forward + oc_sample for generation, so a real
 * loaded model is required. When no model is loaded, /v1/models returns a
 * placeholder and completions return 503.
 */
#ifndef OXIDIZE_OPENAI_H
#define OXIDIZE_OPENAI_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/http.h"
#include "oxidize/llama.h"
#include "oxidize/middleware.h"
#include "oxidize/tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Server-side state shared across requests. The model + tokenizer are
 * loaded once at startup; the session is created per-request (or pooled
 * by a later feature — for now each request gets a fresh session, which
 * is correct but not optimal for concurrent requests). */
typedef struct OcOpenaiState {
    OcLlamaModel *model;
    OcTokenizer *tokenizer;
    char         *model_id;     /* e.g. "qwen2.5-7b-instruct" (owned)    */
    bool          model_loaded;
    /* Optional middleware stack (auth, rate limit, metrics, audit, CORS).
     * NULL disables all of it — the handler then behaves exactly as before.
     * Not owned; the caller initializes and frees it. */
    OcMiddleware *mw;
} OcOpenaiState;

/* ─── Operational endpoints ───────────────────────────────────────────────
 *
 * Beyond the OpenAI surface, the handler serves the probe/observability
 * routes a Kubernetes deployment needs, matching the Rust server's router:
 *
 *   GET /healthz      → 200 always (process is up)
 *   GET /livez        → 200 always (process is not wedged)
 *   GET /readyz       → 200 once a model is loaded, else 503
 *   GET /metrics      → Prometheus text exposition (503 without middleware)
 *   GET /openapi.json → OpenAPI 3.1 description of the above
 *
 * `readyz` is deliberately distinct from `healthz`: it gates traffic until
 * the (slow) model load finishes, so a pod is not sent requests it would
 * answer with 503. */

/* Build the OpenAPI 3.1 spec as a malloc'd JSON string. Caller frees.
 * Returns NULL on OOM. */
char *oc_openai_openapi_json(void);

/* The HTTP handler entry point. Pass this to oc_http_server_start as the
 * handler with an OcOpenaiState* as user_data. */
void oc_openai_handler(const OcHttpRequest *req,
                       int *out_status,
                       const char **out_content_type,
                       const char **out_body,
                       size_t *out_body_len,
                       void *user_data);

void oc_openai_attach_http(OcHttpServer *srv, OcOpenaiState *st);

bool oc_openai_stream_authorize(const OcHttpRequest *req, int *out_status,
                                const char **out_body, void *user_data);

/* Convenience: build a JSON error response body (malloc'd, caller frees). */
char *oc_openai_error_json(const char *message, const char *type);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_OPENAI_H */
