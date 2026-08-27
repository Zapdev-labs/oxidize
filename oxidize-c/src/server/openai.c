/*
 * openai.c — OpenAI-compatible HTTP route handlers.
 *
 * Implements GET /v1/models, POST /v1/completions, POST /v1/chat/completions
 * on top of the server-http-core. JSON parsing is minimal (substring search
 * for the fields we need) — robust for the OpenAI request shape, no JSON dep.
 *
 * Generation drives oc_llama_forward + oc_sample. Each request gets a fresh
 * OcLlamaSession (correct, not optimal — concurrent-request pooling is a
 * later feature). When no model is loaded, /v1/models returns a placeholder
 * and completions return 503.
 */
#define _POSIX_C_SOURCE 200809L   /* strdup */
#include "oxidize/openai.h"

#include "oxidize/error.h"
#include "oxidize/chat.h"
#include "oxidize/llama.h"
#include "oxidize/log.h"
#include "oxidize/sampling.h"
#include "oxidize/version.h"
#include "oxidize/tokenizer.h"

#include <math.h>
#include <pthread.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    pthread_mutex_t mutex;
    OcLlamaModel *model;
    OcLlamaSession session;
    uint32_t *tokens;
    size_t n_tokens;
    bool valid;
} OcPromptPrefixCache;

static OcPromptPrefixCache g_prompt_prefix_cache = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
};

/* The model workspace and persistent compute pool are process-global. HTTP
 * workers must not enter inference concurrently: a second dispatch replaces
 * the first region's function and stack-backed job pointer while its compute
 * workers are still using them. */
static pthread_mutex_t g_generation_mutex = PTHREAD_MUTEX_INITIALIZER;

#define OC_OPENAI_MAX_PROMPT_BYTES (1024u * 1024u)

/* ─── Minimal JSON field extraction ──────────────────────────────────────
 *
 * Finds "key":"value" or "key":number in a JSON body. Returns a pointer to
 * the start of the value (NUL-terminated via a local copy) or NULL. For
 * strings, handles escaped quotes minimally (no \\uXXXX). For numbers,
 * returns a pointer to the digits.
 */

static const char *find_json_string_field(const char *json, const char *key,
                                          char *out, size_t out_cap)
{
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = json;
    for (;;) {
        p = strstr(p, pattern);
        if (p == NULL) return NULL;
        p += strlen(pattern);
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
        if (*p != ':') continue;
        p++;
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
        if (*p == '"') break;
    }
    p++;   /* skip opening quote */
    size_t i = 0;
    while (*p && i + 1 < out_cap) {
        if (*p == '\\' && p[1] != '\0') {
            char esc = p[1];
            if (esc == '"' || esc == '\\' || esc == '/')
                out[i++] = esc;
            else if (esc == 'n')
                out[i++] = '\n';
            else if (esc == 't')
                out[i++] = '\t';
            else
                out[i++] = esc;
            p += 2;
            continue;
        }
        if (*p == '"') break;
        out[i++] = *p++;
    }
    if (*p != '"') return NULL;
    out[i] = '\0';
    return out;
}

static int find_json_int_field(const char *json, const char *key, int def)
{
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (p == NULL) return def;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t' || *p == ':') p++;
    return (int)strtol(p, NULL, 10);
}

static double find_json_double_field(const char *json, const char *key, double def)
{
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (p == NULL) return def;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t' || *p == ':') p++;
    return strtod(p, NULL);
}

/* Extract the content of the LAST "assistant"/"user"/"system" message's
 * "content" field. For chat completions we render the full message array
 * as plain text (a real chat-template renderer is wired by the tokenizer
 * feature — for now we concatenate contents). */
static bool next_message_object(const char **cursor, const char **start,
                                const char **end)
{
    const char *p = *cursor;
    bool in_string = false;
    bool escaped = false;
    size_t depth = 0;
    const char *object = NULL;
    for (; *p; p++) {
        if (in_string) {
            if (escaped) escaped = false;
            else if (*p == '\\') escaped = true;
            else if (*p == '"') in_string = false;
            continue;
        }
        if (*p == '"') { in_string = true; continue; }
        if (*p == ']' && depth == 0) return false;
        if (*p == '{') {
            if (depth++ == 0) object = p;
        } else if (*p == '}' && depth > 0 && --depth == 0) {
            *start = object;
            *end = p + 1;
            *cursor = p + 1;
            return true;
        }
    }
    return false;
}

static bool find_message_content(const char *object, char *out, size_t out_cap)
{
    if (find_json_string_field(object, "content", out, out_cap) != NULL)
        return true;
    const char *field = strstr(object, "\"content\"");
    const char *cursor = field ? strchr(field, '[') : NULL;
    if (cursor == NULL) return false;
    cursor++;
    size_t used = 0;
    const char *part_start;
    const char *part_end;
    while (next_message_object(&cursor, &part_start, &part_end)) {
        const size_t part_len = (size_t)(part_end - part_start);
        char *part = malloc(part_len + 1u);
        if (part == NULL) return false;
        memcpy(part, part_start, part_len);
        part[part_len] = '\0';
        char type[32];
        char *text = malloc(out_cap - used);
        const bool text_part = text != NULL &&
            find_json_string_field(part, "type", type, sizeof(type)) != NULL &&
            (strcmp(type, "text") == 0 || strcmp(type, "input_text") == 0) &&
            find_json_string_field(part, "text", text, out_cap - used) != NULL;
        free(part);
        if (text_part) {
            const size_t n = strlen(text);
            memcpy(out + used, text, n);
            used += n;
        }
        free(text);
    }
    if (used == 0) return false;
    out[used] = '\0';
    return true;
}

static bool extract_messages_content(const char *json, OcChatTemplate template,
                                     char *out, size_t out_cap,
                                     size_t *system_prefix_chars)
{
    size_t out_i = 0;
    size_t message_count = 0;
    if (system_prefix_chars != NULL) *system_prefix_chars = 0;
    const char *messages = strstr(json, "\"messages\"");
    const char *p = messages ? strchr(messages, '[') : NULL;
    if (!p) return false;
    p++;
    char role[64];
    char *content = malloc(out_cap);
    if (content == NULL) return false;
    const char *object_start;
    const char *object_end;
    while (next_message_object(&p, &object_start, &object_end)) {
        size_t object_len = (size_t)(object_end - object_start);
        char *object = malloc(object_len + 1);
        if (!object) {
            free(content);
            return false;
        }
        memcpy(object, object_start, object_len);
        object[object_len] = '\0';
        bool parsed = find_json_string_field(object, "role", role, sizeof(role)) &&
                      find_message_content(object, content, out_cap);
        free(object);
        if (!parsed) {
            free(content);
            return false;
        }
        const char *lookahead = p;
        const char *next_start;
        const char *next_end;
        bool is_last = !next_message_object(&lookahead, &next_start, &next_end);
        size_t written = oc_chat_render_message(template, role, content,
                                                out + out_i, out_cap - out_i,
                                                message_count == 0,
                                                is_last);
        if (written == 0) {
            free(content);
            return false;
        }
        out_i += written;
        if (!is_last && system_prefix_chars != NULL)
            *system_prefix_chars = out_i;
        message_count++;
    }
    out[out_i] = '\0';
    free(content);
    return message_count > 0;
}

/* ─── Helpers ──────────────────────────────────────────────────────────── */

static char *json_escape(const char *src)
{
    if (src == NULL) src = "";
    size_t cap = 1;
    for (const unsigned char *p = (const unsigned char *)src; *p; p++) {
        if (*p == '"' || *p == '\\' || *p == '\b' || *p == '\f' ||
            *p == '\n' || *p == '\r' || *p == '\t')
            cap += 2;
        else if (*p < 0x20)
            cap += 6;
        else
            cap++;
    }
    char *out = malloc(cap);
    if (out == NULL) return NULL;
    char *dst = out;
    static const char hex[] = "0123456789abcdef";
    for (const unsigned char *p = (const unsigned char *)src; *p; p++) {
        switch (*p) {
        case '"': *dst++ = '\\'; *dst++ = '"'; break;
        case '\\': *dst++ = '\\'; *dst++ = '\\'; break;
        case '\b': *dst++ = '\\'; *dst++ = 'b'; break;
        case '\f': *dst++ = '\\'; *dst++ = 'f'; break;
        case '\n': *dst++ = '\\'; *dst++ = 'n'; break;
        case '\r': *dst++ = '\\'; *dst++ = 'r'; break;
        case '\t': *dst++ = '\\'; *dst++ = 't'; break;
        default:
            if (*p < 0x20) {
                *dst++ = '\\'; *dst++ = 'u'; *dst++ = '0'; *dst++ = '0';
                *dst++ = hex[*p >> 4]; *dst++ = hex[*p & 0x0f];
            } else {
                *dst++ = (char)*p;
            }
        }
    }
    *dst = '\0';
    return out;
}

static char *completion_stream_body(const char *text, const char *model,
                                    bool chat)
{
    char *escaped_text = json_escape(text);
    char *escaped_model = json_escape(model ? model : "unknown");
    if (escaped_text == NULL || escaped_model == NULL) {
        free(escaped_text);
        free(escaped_model);
        return NULL;
    }
    size_t cap = strlen(escaped_text) + strlen(escaped_model) * 2u + 1024u;
    char *body = malloc(cap);
    if (body == NULL) {
        free(escaped_text);
        free(escaped_model);
        return NULL;
    }
    if (chat) {
        snprintf(body, cap,
            "data: {\"id\":\"chatcmpl-oxidize\",\"object\":\"chat.completion.chunk\","
            "\"created\":0,\"model\":\"%s\",\"choices\":[{\"index\":0,"
            "\"delta\":{\"role\":\"assistant\",\"content\":\"%s\"},"
            "\"finish_reason\":null}]}\n\n"
            "data: {\"id\":\"chatcmpl-oxidize\",\"object\":\"chat.completion.chunk\","
            "\"created\":0,\"model\":\"%s\",\"choices\":[{\"index\":0,"
            "\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
            "data: [DONE]\n\n",
            escaped_model, escaped_text, escaped_model);
    } else {
        snprintf(body, cap,
            "data: {\"id\":\"cmpl-oxidize\",\"object\":\"text_completion\","
            "\"created\":0,\"model\":\"%s\",\"choices\":[{\"index\":0,"
            "\"text\":\"%s\",\"finish_reason\":null}]}\n\n"
            "data: {\"id\":\"cmpl-oxidize\",\"object\":\"text_completion\","
            "\"created\":0,\"model\":\"%s\",\"choices\":[{\"index\":0,"
            "\"text\":\"\",\"finish_reason\":\"stop\"}]}\n\n"
            "data: [DONE]\n\n",
            escaped_model, escaped_text, escaped_model);
    }
    free(escaped_text);
    free(escaped_model);
    return body;
}

typedef struct OcCompletionStream {
    const OcHttpRequest *request;
    const char *model;
    bool chat;
    bool sent_role;
    bool connected;
} OcCompletionStream;

static bool send_completion_delta(const char *text, void *context)
{
    OcCompletionStream *stream = context;
    char *escaped_text = json_escape(text);
    char *escaped_model = json_escape(stream->model ? stream->model : "unknown");
    if (escaped_text == NULL || escaped_model == NULL) {
        free(escaped_text);
        free(escaped_model);
        return false;
    }
    size_t cap = strlen(escaped_text) + strlen(escaped_model) + 512u;
    char *event = malloc(cap);
    if (event == NULL) {
        free(escaped_text);
        free(escaped_model);
        return false;
    }
    if (stream->chat) {
        snprintf(event, cap,
            "data: {\"id\":\"chatcmpl-oxidize\",\"object\":\"chat.completion.chunk\","
            "\"created\":0,\"model\":\"%s\",\"choices\":[{\"index\":0,"
            "\"delta\":{%s\"content\":\"%s\"},\"finish_reason\":null}]}\n\n",
            escaped_model,
            stream->sent_role ? "" : "\"role\":\"assistant\",",
            escaped_text);
        stream->sent_role = true;
    } else {
        snprintf(event, cap,
            "data: {\"id\":\"cmpl-oxidize\",\"object\":\"text_completion\","
            "\"created\":0,\"model\":\"%s\",\"choices\":[{\"index\":0,"
            "\"text\":\"%s\",\"finish_reason\":null}]}\n\n",
            escaped_model, escaped_text);
    }
    stream->connected = stream->request->stream_write(
        stream->request->stream_context, event, strlen(event));
    free(event);
    free(escaped_text);
    free(escaped_model);
    return stream->connected;
}

static void finish_completion_stream(OcCompletionStream *stream)
{
    const char *finish = stream->chat
        ? "data: {\"id\":\"chatcmpl-oxidize\",\"object\":\"chat.completion.chunk\","
          "\"created\":0,\"model\":\"oxidize\",\"choices\":[{\"index\":0,"
          "\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
          "data: [DONE]\n\n"
        : "data: {\"id\":\"cmpl-oxidize\",\"object\":\"text_completion\","
          "\"created\":0,\"model\":\"oxidize\",\"choices\":[{\"index\":0,"
          "\"text\":\"\",\"finish_reason\":\"stop\"}]}\n\n"
          "data: [DONE]\n\n";
    if (stream->connected)
        stream->connected = stream->request->stream_write(
            stream->request->stream_context, finish, strlen(finish));
}

char *oc_openai_error_json(const char *message, const char *type)
{
    if (type == NULL) type = "invalid_request_error";
    char *escaped_message = json_escape(message);
    char *escaped_type = json_escape(type);
    if (escaped_message == NULL || escaped_type == NULL) {
        free(escaped_message);
        free(escaped_type);
        return NULL;
    }
    size_t cap = strlen(escaped_message) + strlen(escaped_type) + 64;
    char *buf = malloc(cap);
    if (buf == NULL) {
        free(escaped_message);
        free(escaped_type);
        return NULL;
    }
    snprintf(buf, cap, "{\"error\":{\"message\":\"%s\",\"type\":\"%s\"}}",
             escaped_message, escaped_type);
    free(escaped_message);
    free(escaped_type);
    return buf;
}

/* ─── Generation core ──────────────────────────────────────────────────── */

/* Run generation for `prompt` and return a malloc'd completion string. */
static bool restore_system_prefix(OcOpenaiState *st, OcLlamaSession *sess,
                                  const uint32_t *tokens, size_t n_tokens)
{
    if (n_tokens < 32) return false;
    OcPromptPrefixCache *cache = &g_prompt_prefix_cache;
    pthread_mutex_lock(&cache->mutex);
    bool match = cache->valid && cache->model == st->model &&
                 cache->n_tokens == n_tokens &&
                 memcmp(cache->tokens, tokens,
                        n_tokens * sizeof(*tokens)) == 0;
    if (!match) {
        if (cache->valid) {
            oc_llama_session_free(&cache->session);
            free(cache->tokens);
            memset(&cache->session, 0, sizeof(cache->session));
            cache->tokens = NULL;
            cache->valid = false;
        }
        if (oc_llama_session_init(st->model, &cache->session) == OC_OK &&
            oc_llama_prefill(&cache->session, tokens, n_tokens, 0,
                             cache->session.logits) == OC_OK) {
            cache->tokens = malloc(n_tokens * sizeof(*tokens));
            if (cache->tokens != NULL) {
                memcpy(cache->tokens, tokens, n_tokens * sizeof(*tokens));
                cache->model = st->model;
                cache->n_tokens = n_tokens;
                cache->valid = true;
                match = true;
            }
        }
        if (!match) oc_llama_session_free(&cache->session);
    }
    const bool restored = match &&
        oc_llama_session_copy_prefix(sess, &cache->session) == OC_OK;
    pthread_mutex_unlock(&cache->mutex);
    return restored;
}

static char *generate_completion_unlocked(OcOpenaiState *st,
                                          const char *prompt,
                                          size_t system_prefix_chars,
                                          int max_tokens, float temperature,
                                          float top_p,
                                          bool (*on_text)(const char *, void *),
                                          void *on_text_context)
{
    if (st == NULL || !st->model_loaded || st->model == NULL || st->tokenizer == NULL) {
        return NULL;
    }
    OcLlamaSession sess;
    if (oc_llama_session_init(st->model, &sess) != OC_OK) {
        return NULL;
    }
    uint32_t *ids = NULL;
    size_t n_ids = 0;
    OcSpecialTokenPolicy pol = st->tokenizer->has_add_bos_token && st->tokenizer->add_bos_token
        ? OC_TOK_ADD_BOS : OC_TOK_DEFAULT;
    if (oc_tokenizer_encode(st->tokenizer, prompt, pol, &ids, &n_ids) != OC_OK || n_ids == 0) {
        oc_llama_session_free(&sess);
        return NULL;
    }
    if (max_tokens < 0) max_tokens = 0;
    {
        size_t n_ctx = st->model->cfg.n_ctx;
        size_t remaining = n_ctx > n_ids ? n_ctx - n_ids : 0;
        if ((size_t)max_tokens > remaining) max_tokens = (int)remaining;
        if ((size_t)max_tokens > SIZE_MAX - n_ids ||
            (n_ids + (size_t)max_tokens) > SIZE_MAX / sizeof(*ids)) {
            free(ids);
            oc_llama_session_free(&sess);
            return NULL;
        }
    }
    uint32_t *grown = realloc(ids, (n_ids + (size_t)max_tokens) * sizeof(*ids));
    if (grown == NULL) {
        free(ids);
        oc_llama_session_free(&sess);
        return NULL;
    }
    ids = grown;

    OcSamplerConfig scfg = OC_SAMPLER_DEFAULT;
    scfg.temperature = temperature;
    scfg.repeat_penalty = 1.1f;
    if (temperature > 0.0f) {
        scfg.type = OC_SAMPLER_TOP_P;
        scfg.top_p = top_p > 0.0f && top_p <= 1.0f ? top_p : 0.95f;
    }

    size_t consumed = 0;
    if (system_prefix_chars > 0 && system_prefix_chars <= strlen(prompt)) {
        char *prefix = strndup(prompt, system_prefix_chars);
        uint32_t *prefix_ids = NULL;
        size_t n_prefix_ids = 0;
        if (prefix != NULL &&
            oc_tokenizer_encode(st->tokenizer, prefix, pol, &prefix_ids,
                                &n_prefix_ids) == OC_OK &&
            n_prefix_ids <= n_ids &&
            memcmp(prefix_ids, ids, n_prefix_ids * sizeof(*ids)) == 0 &&
            restore_system_prefix(st, &sess, prefix_ids, n_prefix_ids))
            consumed = n_prefix_ids;
        free(prefix_ids);
        free(prefix);
    }
    if (consumed < n_ids &&
        oc_llama_prefill(&sess, ids + consumed, n_ids - consumed, 0,
                         sess.logits) != OC_OK) {
        free(ids); oc_llama_session_free(&sess);
        return NULL;
    }

    size_t cap = 4096, len = 0;
    char *result = malloc(cap);
    if (result == NULL) { free(ids); oc_llama_session_free(&sess); return NULL; }
    result[0] = '\0';

    size_t n_hist = n_ids;
    for (int t = 0; t < max_tokens; t++) {
        uint32_t tok = oc_sample(sess.logits, st->model->cfg.vocab_size, &scfg,
                                 ids, n_hist);
        if (st->tokenizer->has_eos && tok == st->tokenizer->eos_id) break;
        ids[n_hist++] = tok;
        char *piece = NULL;
        if (oc_tokenizer_decode(st->tokenizer, &tok, 1, &piece) == OC_OK && piece) {
            size_t plen = strlen(piece);
            if (len + plen + 1 >= cap) {
                cap = (len + plen + 1) * 2;
                char *nr = realloc(result, cap);
                if (nr == NULL) { free(piece); break; }
                result = nr;
            }
            memcpy(result + len, piece, plen);
            len += plen;
            result[len] = '\0';
            bool keep_generating = on_text == NULL || plen == 0 ||
                                   on_text(piece, on_text_context);
            free(piece);
            if (!keep_generating) break;
        }
        if (oc_llama_forward(&sess, tok, sess.logits) != OC_OK) break;
    }
    free(ids);
    oc_llama_session_free(&sess);
    return result;
}

static char *generate_completion(OcOpenaiState *st, const char *prompt,
                                 size_t system_prefix_chars,
                                 int max_tokens, float temperature, float top_p,
                                 bool (*on_text)(const char *, void *),
                                 void *on_text_context)
{
    pthread_mutex_lock(&g_generation_mutex);
    char *result = generate_completion_unlocked(st, prompt,
                                                system_prefix_chars,
                                                max_tokens, temperature, top_p,
                                                on_text, on_text_context);
    pthread_mutex_unlock(&g_generation_mutex);
    return result;
}

/* ─── Route handlers ──────────────────────────────────────────────────── */

static void handle_list_models(OcOpenaiState *st, int *out_status,
                               const char **out_body)
{
    const char *model_id = st->model_loaded && st->model_id
        ? st->model_id : "placeholder";
    char *escaped_model_id = json_escape(model_id);
    if (escaped_model_id == NULL) { *out_status = 500; return; }
    size_t cap = strlen(escaped_model_id) + 96;
    char *buf = malloc(cap);
    if (buf == NULL) { free(escaped_model_id); *out_status = 500; return; }
    snprintf(buf, cap,
        "{\"object\":\"list\",\"data\":[{\"id\":\"%s\",\"object\":\"model\",\"owned_by\":\"oxidize\"}]}",
        escaped_model_id);
    free(escaped_model_id);
    *out_status = 200;
    *out_body = buf;
}

bool oc_openai_stream_authorize(const OcHttpRequest *req, int *out_status,
                                const char **out_body, void *user_data)
{
    OcOpenaiState *st = user_data;
    OcRequestContext rctx = {
        .method      = req->method,
        .path        = req->path,
        .auth_header = req->auth_header,
        .client_ip   = req->client_ip,
    };
    bool ok = true;
    if (st != NULL && st->mw != NULL) {
        int reject = oc_middleware_process_request(st->mw, &rctx);
        if (reject != 0) {
            *out_status = reject;
            *out_body = oc_openai_error_json(
                reject == 401 ? "unauthorized" : "rate limit exceeded",
                reject == 401 ? "authentication_error" : "rate_limit_error");
            ok = false;
        }
    }
    if (ok && (st == NULL || !st->model_loaded || st->model == NULL ||
               st->tokenizer == NULL)) {
        *out_status = 503;
        *out_body = oc_openai_error_json("no model loaded", "server_error");
        ok = false;
    }
    if (ok && strcmp(req->path, "/v1/completions") == 0) {
        char prompt[8192];
        if (find_json_string_field(req->body, "prompt", prompt, sizeof(prompt)) == NULL) {
            *out_status = 400;
            *out_body = oc_openai_error_json("missing 'prompt' field", "invalid_request_error");
            ok = false;
        }
    } else if (ok && strcmp(req->path, "/v1/chat/completions") == 0) {
        char *prompt = calloc(OC_OPENAI_MAX_PROMPT_BYTES, 1);
        if (prompt == NULL) {
            *out_status = 500;
            *out_body = oc_openai_error_json("out of memory", "server_error");
            ok = false;
        } else {
            size_t prefix = 0;
            const bool valid = extract_messages_content(req->body,
                oc_chat_detect_named(oc_model_arch_name(st->model->arch),
                                     st->model_id), prompt,
                OC_OPENAI_MAX_PROMPT_BYTES, &prefix);
            free(prompt);
            if (!valid) {
                *out_status = 400;
                *out_body = oc_openai_error_json("invalid or oversized messages", "invalid_request_error");
                ok = false;
            }
        }
    }
    if (!ok && st != NULL && st->mw != NULL) {
        OcResponseContext resp = {
            .status = *out_status,
            .duration_ms = 0,
            .tokens_generated = 0,
        };
        oc_middleware_process_response(st->mw, &rctx, &resp);
    }
    return ok;
}

static void handle_completion(OcOpenaiState *st, const OcHttpRequest *req,
                              int *out_status, const char **out_body)
{
    if (!st || !st->model_loaded || st->model == NULL || st->tokenizer == NULL) {
        *out_body = oc_openai_error_json("no model loaded", "server_error");
        *out_status = 503;
        return;
    }
    char prompt[8192] = {0};
    if (find_json_string_field(req->body, "prompt", prompt, sizeof(prompt)) == NULL) {
        *out_body = oc_openai_error_json("missing 'prompt' field", "invalid_request_error");
        *out_status = 400;
        return;
    }
    int max_tokens = find_json_int_field(req->body, "max_tokens", 128);
    double temp = find_json_double_field(req->body, "temperature", 0.0);
    double top_p = find_json_double_field(req->body, "top_p", 0.95);
    bool stream = oc_http_json_bool_field(req->body, "stream", false);

    OcCompletionStream stream_ctx = {
        .request = req, .model = st->model_id, .chat = false,
        .connected = stream && req->stream_write != NULL,
    };
    char *text = generate_completion(st, prompt, 0, max_tokens, (float)temp, (float)top_p,
                                     stream_ctx.connected
                                         ? send_completion_delta : NULL,
                                     &stream_ctx);
    if (text == NULL) {
        *out_body = oc_openai_error_json("generation failed", "server_error");
        *out_status = 500;
        return;
    }
    if (stream) {
        if (req->stream_write != NULL) {
            finish_completion_stream(&stream_ctx);
            free(text);
            *out_body = NULL;
            *out_status = 200;
            return;
        }
        *out_body = completion_stream_body(text, st->model_id, false);
        free(text);
        *out_status = *out_body != NULL ? 200 : 500;
        return;
    }
    /* Build the OpenAI completion response JSON. */
    char *escaped_text = json_escape(text);
    char *escaped_model = json_escape(st->model_id ? st->model_id : "unknown");
    free(text);
    if (escaped_text == NULL || escaped_model == NULL) {
        free(escaped_text); free(escaped_model); *out_status = 500; return;
    }
    size_t cap = strlen(escaped_text) + strlen(escaped_model) + 512;
    char *buf = malloc(cap);
    if (buf == NULL) { free(escaped_text); free(escaped_model); *out_status = 500; return; }
    snprintf(buf, cap,
        "{\"id\":\"cmpl-oxidize\",\"object\":\"text_completion\",\"created\":0,"
        "\"model\":\"%s\",\"choices\":[{\"text\":\"%s\",\"index\":0,\"finish_reason\":\"stop\"}],"
        "\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":0,\"total_tokens\":0}}",
        escaped_model, escaped_text);
    free(escaped_text);
    free(escaped_model);
    *out_status = 200;
    *out_body = buf;
}

static void handle_chat_completion(OcOpenaiState *st, const OcHttpRequest *req,
                                   int *out_status, const char **out_body)
{
    if (!st || !st->model_loaded || !st->model) {
        *out_body = oc_openai_error_json("no model loaded", "server_error");
        *out_status = 503;
        return;
    }
    bool stream = oc_http_json_bool_field(req->body, "stream", false);
    char *prompt = calloc(OC_OPENAI_MAX_PROMPT_BYTES, 1);
    if (prompt == NULL) {
        *out_body = oc_openai_error_json("out of memory", "server_error");
        *out_status = 500;
        return;
    }
    size_t system_prefix_chars = 0;
    OcChatTemplate template = oc_chat_detect_named(
        oc_model_arch_name(st->model->arch), st->model_id);
    if (!extract_messages_content(req->body, template, prompt,
                                  OC_OPENAI_MAX_PROMPT_BYTES,
                                  &system_prefix_chars)) {
        free(prompt);
        *out_body = oc_openai_error_json("invalid or oversized messages", "invalid_request_error");
        *out_status = 400;
        return;
    }
    int max_tokens = find_json_int_field(req->body, "max_tokens", 128);
    double temp = find_json_double_field(req->body, "temperature", 0.0);
    double top_p = find_json_double_field(req->body, "top_p", 0.95);
    OcCompletionStream stream_ctx = {
        .request = req, .model = st->model_id, .chat = true,
        .connected = stream && req->stream_write != NULL,
    };
    char *text = generate_completion(st, prompt, system_prefix_chars,
                                     max_tokens, (float)temp, (float)top_p,
                                     stream_ctx.connected
                                         ? send_completion_delta : NULL,
                                     &stream_ctx);
    free(prompt);
    if (text == NULL) {
        *out_body = oc_openai_error_json("generation failed", "server_error");
        *out_status = 500;
        return;
    }
    if (stream) {
        if (req->stream_write != NULL) {
            finish_completion_stream(&stream_ctx);
            free(text);
            *out_body = NULL;
            *out_status = 200;
            return;
        }
        *out_body = completion_stream_body(text, st->model_id, true);
        free(text);
        *out_status = *out_body != NULL ? 200 : 500;
        return;
    }
    char *escaped_text = json_escape(text);
    char *escaped_model = json_escape(st->model_id ? st->model_id : "unknown");
    free(text);
    if (escaped_text == NULL || escaped_model == NULL) {
        free(escaped_text); free(escaped_model); *out_status = 500; return;
    }
    size_t cap = strlen(escaped_text) + strlen(escaped_model) + 640;
    char *buf = malloc(cap);
    if (buf == NULL) { free(escaped_text); free(escaped_model); *out_status = 500; return; }
    snprintf(buf, cap,
        "{\"id\":\"chatcmpl-oxidize\",\"object\":\"chat.completion\",\"created\":0,"
        "\"model\":\"%s\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
        "\"content\":\"%s\"},\"finish_reason\":\"stop\"}],"
        "\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":0,\"total_tokens\":0}}",
        escaped_model, escaped_text);
    free(escaped_text);
    free(escaped_model);
    *out_status = 200;
    *out_body = buf;
}

/* ─── POST /v1/embeddings ────────────────────────────────────────────── */

static void handle_embeddings(OcOpenaiState *st, const OcHttpRequest *req,
                              int *out_status, const char **out_body)
{
    if (!st->model_loaded) {
        *out_body = oc_openai_error_json("no model loaded", "server_error");
        *out_status = 503;
        return;
    }
    char input[8192];
    if (!find_json_string_field(req->body, "input", input, sizeof(input))) {
        *out_body = oc_openai_error_json("missing 'input' field", "invalid_request_error");
        *out_status = 400;
        return;
    }

    uint32_t *ids = NULL;
    size_t n_ids = 0;
    OcSpecialTokenPolicy pol = st->tokenizer->has_add_bos_token && st->tokenizer->add_bos_token
        ? OC_TOK_ADD_BOS : OC_TOK_DEFAULT;
    OcError e = oc_tokenizer_encode(st->tokenizer, input, pol, &ids, &n_ids);
    if (e != OC_OK || n_ids == 0) {
        free(ids);
        *out_body = oc_openai_error_json("tokenization failed", "server_error");
        *out_status = 500;
        return;
    }

    pthread_mutex_lock(&g_generation_mutex);
    OcLlamaSession sess;
    e = oc_llama_session_init(st->model, &sess);
    if (e != OC_OK) {
        pthread_mutex_unlock(&g_generation_mutex);
        free(ids);
        *out_body = oc_openai_error_json("session init failed", "server_error");
        *out_status = 500;
        return;
    }

    for (size_t i = 0; i < n_ids; i++) {
        e = oc_llama_forward(&sess, ids[i], NULL);
        if (e != OC_OK) break;
    }
    if (e != OC_OK) {
        oc_llama_session_free(&sess);
        pthread_mutex_unlock(&g_generation_mutex);
        free(ids);
        *out_body = oc_openai_error_json("forward pass failed", "server_error");
        *out_status = 500;
        return;
    }

    uint32_t n_embd = st->model->cfg.n_embd;
    float *embedding = calloc(n_embd, sizeof(float));
    if (!embedding) {
        oc_llama_session_free(&sess); free(ids);
        pthread_mutex_unlock(&g_generation_mutex);
        *out_body = oc_openai_error_json("allocation failed", "server_error");
        *out_status = 500; return;
    }
    for (uint32_t i = 0; i < n_embd; i++) embedding[i] = sess.x[i];
    oc_llama_session_free(&sess);
    pthread_mutex_unlock(&g_generation_mutex);
    free(ids);

    /* L2 normalize. */
    float norm = 0.0f;
    for (uint32_t i = 0; i < n_embd; i++) norm += embedding[i] * embedding[i];
    norm = sqrtf(norm);
    if (norm > 0.0f) {
        float inv = 1.0f / norm;
        for (uint32_t i = 0; i < n_embd; i++) embedding[i] *= inv;
    }

    char *escaped_model = json_escape(st->model_id ? st->model_id : "unknown");
    if (!escaped_model) { free(embedding); *out_status = 500; return; }
    size_t cap = (size_t)n_embd * 16 + strlen(escaped_model) + 512;
    char *buf = malloc(cap);
    if (!buf) { free(embedding); free(escaped_model); *out_status = 500; return; }
    size_t pos = 0;
    pos += snprintf(buf + pos, cap - pos,
        "{\"object\":\"list\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":[");
    for (uint32_t i = 0; i < n_embd && pos + 24 < cap; i++) {
        pos += snprintf(buf + pos, cap - pos, "%s%.7f", i > 0 ? "," : "", embedding[i]);
    }
    pos += snprintf(buf + pos, cap - pos,
        "]}],\"model\":\"%s\",\"usage\":{\"prompt_tokens\":%zu,\"total_tokens\":%zu}}",
        escaped_model, n_ids, n_ids);
    free(escaped_model);
    free(embedding);
    *out_status = 200;
    *out_body = buf;
}

/* ─── POST /v1/responses ──────────────────────────────────────────────── */

static void handle_responses(OcOpenaiState *st, const OcHttpRequest *req,
                              int *out_status, const char **out_body)
{
    if (!st->model_loaded) {
        *out_body = oc_openai_error_json("no model loaded", "server_error");
        *out_status = 503;
        return;
    }
    char prompt[8192];
    if (!find_json_string_field(req->body, "input", prompt, sizeof(prompt))) {
        if (!find_json_string_field(req->body, "prompt", prompt, sizeof(prompt))) {
            *out_body = oc_openai_error_json("missing 'input' field", "invalid_request_error");
            *out_status = 400;
            return;
        }
    }
    int max_tokens = find_json_int_field(req->body, "max_output_tokens", 128);
    if (max_tokens == 128)
        max_tokens = find_json_int_field(req->body, "max_tokens", 128);
    double temp = find_json_double_field(req->body, "temperature", 0.0);
    double top_p = find_json_double_field(req->body, "top_p", 0.95);
    char *text = generate_completion(st, prompt, 0, max_tokens, (float)temp, (float)top_p,
                                     NULL, NULL);
    if (!text) {
        *out_body = oc_openai_error_json("generation failed", "server_error");
        *out_status = 500; return;
    }
    char *escaped = json_escape(text);
    char *escaped_model = json_escape(st->model_id ? st->model_id : "unknown");
    free(text);
    if (!escaped || !escaped_model) {
        free(escaped); free(escaped_model); *out_status = 500; return;
    }
    size_t cap = strlen(escaped) + strlen(escaped_model) + 512;
    char *buf = malloc(cap);
    if (!buf) { free(escaped); free(escaped_model); *out_status = 500; return; }
    snprintf(buf, cap,
        "{\"id\":\"resp-oxidize\",\"object\":\"response\",\"created\":0,"
        "\"model\":\"%s\",\"output\":[{\"type\":\"message\",\"role\":\"assistant\","
        "\"content\":[{\"type\":\"output_text\",\"text\":\"%s\"}]}],"
        "\"status\":\"completed\",\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}",
        escaped_model, escaped);
    free(escaped);
    free(escaped_model);
    *out_status = 200;
    *out_body = buf;
}

/* ─── Operational endpoints ─────────────────────────────────────────────── */

/* A 200/503 probe reply. Bodies are always malloc'd — http.c frees them. */
static void handle_probe(bool ready, int *out_status, const char **out_body)
{
    *out_status = ready ? 200 : 503;
    *out_body = strdup(ready ? "{\"status\":\"ok\"}"
                                : "{\"status\":\"not ready\"}");
}

static void handle_metrics(OcOpenaiState *st, int *out_status,
                           const char **out_content_type,
                           const char **out_body)
{
    if (st->mw == NULL) {
        /* No middleware stack means nothing is recording. Say so rather than
         * serving an all-zero scrape, which would look like a healthy but
         * idle server. */
        *out_status = 503;
        *out_body = oc_openai_error_json("metrics not enabled", "server_error");
        return;
    }
    /* Sized for the fixed set of counters plus 8 histogram buckets. */
    char *buf = malloc(4096);
    if (!buf) { *out_status = 500; return; }
    size_t n = oc_metrics_format_prometheus(&st->mw->metrics, buf, 4096);
    if (n == 0) { free(buf); *out_status = 500; return; }
    *out_status = 200;
    /* The version parameter is part of the Prometheus text format contract;
     * scrapers use it to pick a parser. */
    *out_content_type = "text/plain; version=0.0.4; charset=utf-8";
    *out_body = buf;
}

char *oc_openai_openapi_json(void)
{
    static const char spec[] =
    "{\"openapi\":\"3.1.0\","
    "\"info\":{\"title\":\"oxidize-c API\",\"version\":\"" OC_VERSION "\","
    "\"description\":\"OpenAI-compatible endpoints exposed by oxidize-c.\"},"
    "\"servers\":[{\"url\":\"/\"}],"
    "\"paths\":{"
    "\"/healthz\":{\"get\":{\"summary\":\"Health check\","
        "\"responses\":{\"200\":{\"description\":\"OK\"}}}},"
    "\"/livez\":{\"get\":{\"summary\":\"Liveness check\","
        "\"responses\":{\"200\":{\"description\":\"OK\"}}}},"
    "\"/readyz\":{\"get\":{\"summary\":\"Readiness check\","
        "\"responses\":{\"200\":{\"description\":\"Model loaded\"},"
        "\"503\":{\"description\":\"No model loaded\"}}}},"
    "\"/metrics\":{\"get\":{\"summary\":\"Prometheus metrics\","
        "\"responses\":{\"200\":{\"description\":\"Exposition text\"},"
        "\"503\":{\"description\":\"Metrics not enabled\"}}}},"
    "\"/v1/models\":{\"get\":{\"summary\":\"List models\","
        "\"responses\":{\"200\":{\"description\":\"Model list\"}}}},"
    "\"/v1/completions\":{\"post\":{\"summary\":\"Create completion\","
        "\"responses\":{\"200\":{\"description\":\"Completion\"},"
        "\"503\":{\"description\":\"No model loaded\"}}}},"
    "\"/v1/chat/completions\":{\"post\":{\"summary\":\"Create chat completion\","
        "\"responses\":{\"200\":{\"description\":\"Chat completion\"},"
        "\"503\":{\"description\":\"No model loaded\"}}}},"
    "\"/v1/embeddings\":{\"post\":{\"summary\":\"Create embeddings\","
        "\"responses\":{\"200\":{\"description\":\"Embedding vectors\"}}}},"
    "\"/v1/responses\":{\"post\":{\"summary\":\"Create response\","
        "\"responses\":{\"200\":{\"description\":\"Response\"}}}},"
    "\"/v1/mesh/chat/completions\":{\"post\":{"
        "\"summary\":\"Chat completion routed across the mesh cluster\","
        "\"responses\":{\"200\":{\"description\":\"Chat completion\"},"
        "\"503\":{\"description\":\"Mesh not configured\"}}}}"
    "}}";
    return strdup(spec);
}

void oc_openai_handler(const OcHttpRequest *req,
                       int *out_status,
                       const char **out_content_type,
                       const char **out_body,
                       size_t *out_body_len,
                       void *user_data)
{
    OcOpenaiState *st = (OcOpenaiState *)user_data;
    *out_status = 404;
    *out_content_type = "application/json";
    *out_body = NULL;
    *out_body_len = 0;

    /* Pre-handler middleware: auth, then rate limit. A rejection short-circuits
     * the route table but still falls through to the post-handler block below,
     * so 401s and 429s show up in metrics and the audit log. */
    struct timespec t_start;
    (void)clock_gettime(CLOCK_MONOTONIC, &t_start);
    OcRequestContext rctx = {
        .method      = req->method,
        .path        = req->path,
        .auth_header = req->auth_header,
        .client_ip   = req->client_ip,
    };
    int reject = 0;
    if (st->mw != NULL && req->stream_write == NULL) {
        reject = oc_middleware_process_request(st->mw, &rctx);
    }

    if (reject != 0) {
        *out_status = reject;
        *out_body = oc_openai_error_json(
            reject == 401 ? "unauthorized" : "rate limit exceeded",
            reject == 401 ? "authentication_error" : "rate_limit_error");
    } else if (req->method == OC_HTTP_GET &&
               strcmp(req->path, "/healthz") == 0) {
        handle_probe(true, out_status, out_body);
    } else if (req->method == OC_HTTP_GET &&
               strcmp(req->path, "/livez") == 0) {
        handle_probe(true, out_status, out_body);
    } else if (req->method == OC_HTTP_GET &&
               strcmp(req->path, "/readyz") == 0) {
        handle_probe(st->model_loaded, out_status, out_body);
    } else if (req->method == OC_HTTP_GET &&
               strcmp(req->path, "/metrics") == 0) {
        handle_metrics(st, out_status, out_content_type, out_body);
    } else if (req->method == OC_HTTP_GET &&
               strcmp(req->path, "/openapi.json") == 0) {
        *out_status = 200;
        *out_body = oc_openai_openapi_json();
    } else if (req->method == OC_HTTP_GET && strcmp(req->path, "/v1/models") == 0) {
        handle_list_models(st, out_status, out_body);
    } else if (req->method == OC_HTTP_POST &&
               strcmp(req->path, "/v1/completions") == 0) {
        handle_completion(st, req, out_status, out_body);
    } else if (req->method == OC_HTTP_POST &&
               strcmp(req->path, "/v1/chat/completions") == 0) {
        handle_chat_completion(st, req, out_status, out_body);
    } else if (req->method == OC_HTTP_POST &&
               strcmp(req->path, "/v1/embeddings") == 0) {
        handle_embeddings(st, req, out_status, out_body);
    } else if (req->method == OC_HTTP_POST &&
               strcmp(req->path, "/v1/responses") == 0) {
        handle_responses(st, req, out_status, out_body);
    } else {
        *out_body = oc_openai_error_json("not found", "invalid_request_error");
        *out_status = 404;
    }

    if (*out_status == 200 && req->method == OC_HTTP_POST &&
        oc_http_json_bool_field(req->body, "stream", false) &&
        (strcmp(req->path, "/v1/completions") == 0 ||
         strcmp(req->path, "/v1/chat/completions") == 0))
        *out_content_type = "text/event-stream";

    /* Compute body length if a body was set. */
    if (*out_body != NULL) {
        *out_body_len = strlen(*out_body);
    }

    /* Post-handler middleware: record metrics + audit. Runs for every path
     * including rejections and 404s, so the counters match what the socket
     * actually saw. */
    if (st->mw != NULL) {
        struct timespec t_end;
        (void)clock_gettime(CLOCK_MONOTONIC, &t_end);
        uint64_t ms = (uint64_t)((t_end.tv_sec - t_start.tv_sec) * 1000 +
                                 (t_end.tv_nsec - t_start.tv_nsec) / 1000000);
        OcResponseContext resp = {
            .status           = *out_status,
            .duration_ms      = ms,
            /* Token accounting is owned by the generation path; the handler
             * does not see a count, so this stays 0 until generate_completion
             * reports one. */
            .tokens_generated = 0,
        };
        oc_middleware_process_response(st->mw, &rctx, &resp);
    }
    /* The HTTP server core (http.c) frees the response body after write()
     * when body_len > 0. Handlers always return malloc'd buffers. */
}

void oc_openai_attach_http(OcHttpServer *srv, OcOpenaiState *st)
{
    if (srv == NULL) return;
    oc_http_server_set_stream_authorizer(srv, oc_openai_stream_authorize);
    if (st == NULL || st->mw == NULL) return;
    char cors[384];
    if (oc_middleware_cors_headers(st->mw, cors, sizeof(cors)) > 0)
        oc_http_server_set_extra_headers(srv, cors);
}
