/*
 * openai.h — OpenAI-compatible HTTP route handlers.
 *
 * Implements the `server-openai-routes` feature on top of server-http-core.
 * Exposes the three canonical endpoints:
 *   GET  /v1/models               → list loaded model(s)
 *   POST /v1/completions           → text completion
 *   POST /v1/chat/completions      → chat with message array
 *
 * JSON parsing is intentionally minimal (no full JSON parser): the handlers
 * extract the fields they need (model, prompt, messages, max_tokens,
 * temperature) via simple substring search. This is robust enough for the
 * OpenAI request shape and avoids pulling in a JSON dependency.
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
} OcOpenaiState;

/* The HTTP handler entry point. Pass this to oc_http_server_start as the
 * handler with an OcOpenaiState* as user_data. */
void oc_openai_handler(const OcHttpRequest *req,
                       int *out_status,
                       const char **out_content_type,
                       const char **out_body,
                       size_t *out_body_len,
                       void *user_data);

/* Convenience: build a JSON error response body (malloc'd, caller frees). */
char *oc_openai_error_json(const char *message, const char *type);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_OPENAI_H */
