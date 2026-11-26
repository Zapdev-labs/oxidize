/*
 * inkling_api.h — OpenAI-compatible chat API for Inkling-2 (`serve` with an
 * Inkling GGUF).
 *
 * POST /v1/chat/completions renders the request with Inkling-2's chat
 * template (messages, tools, reasoning_effort), decodes greedily (with the
 * DSpark drafter when one is loaded) and returns the model's thinking as
 * reasoning_content, its text as content and its tool invocations as
 * tool_calls, streamed as chat.completion.chunk events when "stream" is
 * true. Requests run one at a time on one session; the session keeps the
 * end of the previous prompt, so a request that extends the previous
 * conversation only prefills the new tokens.
 */
#ifndef OXIDIZE_INKLING_API_H
#define OXIDIZE_INKLING_API_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/inkling.h"
#include "oxidize/middleware.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── chat template ─────────────────────────────────────────────────────── */

/* Render the messages / tools / reasoning_effort of an OpenAI chat request
 * (JSON text) as Inkling-2's chat template does, without a generation
 * prompt (the model opens its own message). default_effort applies when
 * the request has no reasoning_effort. Returns a malloc'd string, or NULL
 * with a static message in *err. */
char *oc_inkling_chat_render(const char *request_json, size_t len, float default_effort, const char **err);

/* ─── reply parsing ─────────────────────────────────────────────────────── */

typedef enum {
    OC_INK_PART_REASONING, /* text of a <|content_thinking|> message                  */
    OC_INK_PART_CONTENT,   /* text of a <|content_text|> message                      */
    OC_INK_PART_TOOL_CALL, /* a whole tool call: text = name, args = arguments (JSON) */
} OcInklingPartKind;

/* args: a tool call's arguments as a NUL-terminated JSON object, NULL for
 * other parts; valid only during the callback, so copy it to keep it. */
typedef void (*OcInklingPartCb)(OcInklingPartKind kind, const char *text, size_t len, const char *args, void *user);

/* Splits the generated text into parts as it arrives. push takes the text
 * of one token (special: a control token such as <|message_model|>);
 * reasoning and content arrive as deltas, a tool call once its message
 * ends. done is set once the model ends its turn. */
typedef struct OcInklingReply {
    OcInklingPartCb cb;
    void           *user;
    int             state;
    char           *name, *json;  /* message header, tool call body */
    size_t          name_len, name_cap, json_len, json_cap;
    uint32_t        n_tool_calls;
    bool            done;
} OcInklingReply;

void oc_inkling_reply_init(OcInklingReply *r, OcInklingPartCb cb, void *user);
void oc_inkling_reply_push(OcInklingReply *r, const char *text, size_t len, bool special);
void oc_inkling_reply_free(OcInklingReply *r);

/* ─── server ────────────────────────────────────────────────────────────── */

typedef struct OcInklingServeOptions {
    const char   *host;
    uint16_t      port;
    const char   *model_id;
    uint32_t      n_ctx;      /* session positions                        */
    uint32_t      n_predict;  /* default max_tokens                       */
    uint32_t      draft_max;  /* DSpark tokens per round (with a drafter) */
    float         p_min;
    float         effort;     /* default reasoning effort                 */
    OcMiddleware *mw;         /* optional; not owned                      */
    bool          json;       /* one JSON startup record instead of the banner */
} OcInklingServeOptions;

/* Serve m (with drafter d, may be NULL) until the process is stopped. With
 * gpu the session runs on the Vulkan forward (oc_inkling_gpu_load first). */
OcError oc_inkling_serve(const OcInklingModel *m, const OcInklingDraft *d, bool gpu, const OcInklingServeOptions *o);

#ifdef __cplusplus
}
#endif

#endif
