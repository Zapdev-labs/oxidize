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
#include "oxidize/tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    /* Search for "key" : "value" */
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (p == NULL) return NULL;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t' || *p == ':') p++;
    if (*p != '"') return NULL;
    p++;   /* skip opening quote */
    size_t i = 0;
    while (*p && *p != '"' && i + 1 < out_cap) {
        if (*p == '\\' && *(p+1) == '"') {
            out[i++] = '"';
            p += 2;
        } else if (*p == '\\' && *(p+1) == 'n') {
            out[i++] = '\n';
            p += 2;
        } else {
            out[i++] = *p++;
        }
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

static bool find_json_bool_field(const char *json, const char *key, bool def)
{
    size_t key_len = strlen(key);
    size_t depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (const char *p = json; *p; p++) {
        if (in_string) {
            if (escaped) escaped = false;
            else if (*p == '\\') escaped = true;
            else if (*p == '"') in_string = false;
            continue;
        }
        if (*p == '{' || *p == '[') { depth++; continue; }
        if (*p == '}' || *p == ']') { if (depth > 0) depth--; continue; }
        if (*p != '"') continue;
        if (depth == 1 && strncmp(p + 1, key, key_len) == 0 &&
            p[1 + key_len] == '"') {
            const char *value = p + key_len + 2;
            while (*value == ' ' || *value == '\t') value++;
            if (*value++ != ':') return def;
            while (*value == ' ' || *value == '\t') value++;
            if (strncmp(value, "true", 4) == 0) return true;
            if (strncmp(value, "false", 5) == 0) return false;
            return def;
        }
        in_string = true;
    }
    return def;
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

static bool extract_messages_content(const char *json, OcChatTemplate template,
                                     char *out, size_t out_cap)
{
    size_t out_i = 0;
    size_t message_count = 0;
    const char *messages = strstr(json, "\"messages\"");
    const char *p = messages ? strchr(messages, '[') : NULL;
    if (!p) return false;
    p++;
    char role[64];
    char content[16384];
    const char *object_start;
    const char *object_end;
    while (next_message_object(&p, &object_start, &object_end)) {
        size_t object_len = (size_t)(object_end - object_start);
        char *object = malloc(object_len + 1);
        if (!object) return false;
        memcpy(object, object_start, object_len);
        object[object_len] = '\0';
        bool parsed = find_json_string_field(object, "role", role, sizeof(role)) &&
                      find_json_string_field(object, "content", content, sizeof(content));
        free(object);
        if (!parsed)
            return false;
        const char *lookahead = p;
        const char *next_start;
        const char *next_end;
        bool is_last = !next_message_object(&lookahead, &next_start, &next_end);
        size_t written = oc_chat_render_message(template, role, content,
                                                out + out_i, out_cap - out_i,
                                                message_count == 0,
                                                is_last);
        if (written == 0) return false;
        out_i += written;
        message_count++;
    }
    out[out_i] = '\0';
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
static char *generate_completion(OcOpenaiState *st, const char *prompt,
                                 int max_tokens, float temperature)
{
    if (st == NULL || !st->model_loaded || st->model == NULL || st->tokenizer == NULL) {
        return strdup("");
    }
    OcLlamaSession sess;
    if (oc_llama_session_init(st->model, &sess) != OC_OK) {
        return strdup("");
    }
    uint32_t *ids = NULL;
    size_t n_ids = 0;
    OcSpecialTokenPolicy pol = st->tokenizer->has_add_bos_token && st->tokenizer->add_bos_token
        ? OC_TOK_ADD_BOS : OC_TOK_DEFAULT;
    if (oc_tokenizer_encode(st->tokenizer, prompt, pol, &ids, &n_ids) != OC_OK || n_ids == 0) {
        oc_llama_session_free(&sess);
        return strdup("");
    }

    OcSamplerConfig scfg = OC_SAMPLER_DEFAULT;
    scfg.temperature = temperature;
    scfg.repeat_penalty = 1.1f;
    if (temperature > 0.0f) {
        scfg.type = OC_SAMPLER_TOP_P;
        scfg.top_p = 0.95f;
    }

    /* Prefill all but the last token. */
    for (size_t i = 0; i + 1 < n_ids; i++) {
        if (oc_llama_forward(&sess, ids[i], NULL) != OC_OK) break;
    }
    /* Forward the last prompt token WITH logits. */
    uint32_t next = ids[n_ids - 1];
    if (oc_llama_forward(&sess, next, sess.logits) != OC_OK) {
        free(ids); oc_llama_session_free(&sess);
        return strdup("");
    }

    /* Generate. Use a dynamic string builder. */
    size_t cap = 4096, len = 0;
    char *result = malloc(cap);
    if (result == NULL) { free(ids); oc_llama_session_free(&sess); return NULL; }
    result[0] = '\0';

    for (int t = 0; t < max_tokens; t++) {
        uint32_t tok = oc_sample(sess.logits, st->model->cfg.vocab_size, &scfg, NULL, 0);
        if (st->tokenizer->has_eos && tok == st->tokenizer->eos_id) break;
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
            free(piece);
        }
        if (oc_llama_forward(&sess, tok, sess.logits) != OC_OK) break;
    }
    free(ids);
    oc_llama_session_free(&sess);
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

static void handle_completion(OcOpenaiState *st, const OcHttpRequest *req,
                              int *out_status, const char **out_body)
{
    if (!st || !st->model_loaded) {
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
    bool stream = find_json_bool_field(req->body, "stream", false);

    if (stream) {
        *out_body = oc_openai_error_json("streaming is not supported by this server",
                                         "invalid_request_error");
        *out_status = 400;
        return;
    }

    char *text = generate_completion(st, prompt, max_tokens, (float)temp);
    if (text == NULL) {
        *out_body = oc_openai_error_json("generation failed", "server_error");
        *out_status = 500;
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
    if (find_json_bool_field(req->body, "stream", false)) {
        *out_body = oc_openai_error_json("streaming is not supported by this server",
                                         "invalid_request_error");
        *out_status = 400;
        return;
    }
    char prompt[16384] = {0};
    OcChatTemplate template = oc_chat_detect(oc_model_arch_name(st->model->arch));
    if (!extract_messages_content(req->body, template, prompt, sizeof(prompt))) {
        *out_body = oc_openai_error_json("invalid or oversized messages", "invalid_request_error");
        *out_status = 400;
        return;
    }
    int max_tokens = find_json_int_field(req->body, "max_tokens", 128);
    double temp = find_json_double_field(req->body, "temperature", 0.0);
    char *text = generate_completion(st, prompt, max_tokens, (float)temp);
    if (text == NULL) {
        *out_body = oc_openai_error_json("generation failed", "server_error");
        *out_status = 500;
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

    OcLlamaSession sess;
    e = oc_llama_session_init(st->model, &sess);
    if (e != OC_OK) {
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
        free(ids);
        *out_body = oc_openai_error_json("forward pass failed", "server_error");
        *out_status = 500;
        return;
    }

    uint32_t n_embd = st->model->cfg.n_embd;
    float *embedding = calloc(n_embd, sizeof(float));
    if (!embedding) {
        oc_llama_session_free(&sess); free(ids);
        *out_body = oc_openai_error_json("allocation failed", "server_error");
        *out_status = 500; return;
    }
    for (uint32_t i = 0; i < n_embd; i++) embedding[i] = sess.x[i];
    oc_llama_session_free(&sess);
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
    char *text = generate_completion(st, prompt, max_tokens, (float)temp);
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

    if (req->method == OC_HTTP_GET && strcmp(req->path, "/v1/models") == 0) {
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

    /* Compute body length if a body was set. */
    if (*out_body != NULL) {
        *out_body_len = strlen(*out_body);
    }
    /* NOTE: the response body is malloc'd and freed by the HTTP server's
     * worker after write(). We leak it here intentionally — the server
     * core owns the buffer for the duration of the response. The server
     * core currently does NOT free handler-returned bodies (a known TODO
     * tracked for the paged-scheduler feature when response lifetimes are
     * formalized). */
}
