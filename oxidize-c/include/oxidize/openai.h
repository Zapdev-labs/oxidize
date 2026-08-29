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

/* Server-side state shared across requests. The model + tokenizer are */
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
