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
#include "oxidize/llama.h"
#include "oxidize/log.h"
#include "oxidize/sampling.h"
#include "oxidize/tokenizer.h"

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
static void extract_messages_content(const char *json, char *out, size_t out_cap)
{
    size_t out_i = 0;
    const char *p = json;
    while ((p = strstr(p, "\"content\"")) != NULL) {
        p += strlen("\"content\"");
        while (*p == ' ' || *p == '\t' || *p == ':') p++;
        if (*p != '"') continue;
        p++;
        while (*p && *p != '"' && out_i + 1 < out_cap) {
            if (*p == '\\' && *(p+1) == 'n') { out[out_i++] = '\n'; p += 2; }
            else if (*p == '\\' && *(p+1) == '"') { out[out_i++] = '"'; p += 2; }
            else out[out_i++] = *p++;
        }
        if (*p == '"') p++;
        if (out_i + 1 < out_cap) out[out_i++] = '\n';
    }
    out[out_i] = '\0';
}

/* ─── Helpers ──────────────────────────────────────────────────────────── */

char *oc_openai_error_json(const char *message, const char *type)
{
    if (type == NULL) type = "invalid_request_error";
    /* Minimal JSON; assume message has no unescaped quotes (caller-controlled). */
    size_t cap = strlen(message) + strlen(type) + 64;
    char *buf = malloc(cap);
    if (buf == NULL) return NULL;
    snprintf(buf, cap, "{\"error\":{\"message\":\"%s\",\"type\":\"%s\"}}",
             message, type);
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
    char *buf = malloc(1024);
    if (buf == NULL) { *out_status = 500; return; }
    if (st->model_loaded && st->model_id) {
        snprintf(buf, 1024,
            "{\"object\":\"list\",\"data\":[{\"id\":\"%s\",\"object\":\"model\",\"owned_by\":\"oxidize\"}]}",
            st->model_id);
    } else {
        snprintf(buf, 1024,
            "{\"object\":\"list\",\"data\":[{\"id\":\"placeholder\",\"object\":\"model\",\"owned_by\":\"oxidize\"}]}");
    }
    *out_status = 200;
    *out_body = buf;
}

static void handle_completion(OcOpenaiState *st, const OcHttpRequest *req,
                              int *out_status, const char **out_body)
{
    if (!st->model_loaded) {
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
    bool stream = false;
    if (strstr(req->body, "\"stream\"")) {
        const char *p = strstr(req->body, "\"stream\"");
        p = strchr(p + 8, ':');
        if (p) { p++; while (*p==' '||*p=='\t') p++; if (*p=='t') stream=true; }
    }

    if (stream) {
        char *text = generate_completion(st, prompt, max_tokens, (float)temp);
        if (!text) { *out_body=oc_openai_error_json("gen failed","server_error"); *out_status=500; return; }
        size_t cap = strlen(text) + 1024;
        char *buf = malloc(cap);
        if (buf) {
            snprintf(buf, cap,
                "data: {\"id\":\"cmpl-oxidize\",\"object\":\"text_completion\",\"choices\":[{\"text\":\"%s\",\"index\":0}]}\n\ndata: [DONE]\n\n", text);
            free(text);
            *out_body = buf; *out_status = 200; return;
        }
        free(text);
    }

    char *text = generate_completion(st, prompt, max_tokens, (float)temp);
    if (text == NULL) {
        *out_body = oc_openai_error_json("generation failed", "server_error");
        *out_status = 500;
        return;
    }
    /* Build the OpenAI completion response JSON. */
    size_t cap = strlen(text) + 512;
    char *buf = malloc(cap);
    if (buf == NULL) { free(text); *out_status = 500; return; }
    /* Escape any quotes/newlines in the generated text for JSON safety. */
    /* (For brevity we trust the tokenizer output is mostly printable; a
     * full escaper is a TODO.) */
    snprintf(buf, cap,
        "{\"id\":\"cmpl-oxidize\",\"object\":\"text_completion\",\"created\":0,"
        "\"model\":\"%s\",\"choices\":[{\"text\":\"%s\",\"index\":0,\"finish_reason\":\"stop\"}],"
        "\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":0,\"total_tokens\":0}}",
        st->model_id ? st->model_id : "unknown", text);
    free(text);
    *out_status = 200;
    *out_body = buf;
}

static void handle_chat_completion(OcOpenaiState *st, const OcHttpRequest *req,
                                   int *out_status, const char **out_body)
{
    if (!st->model_loaded) {
        *out_body = oc_openai_error_json("no model loaded", "server_error");
        *out_status = 503;
        return;
    }
    char prompt[16384] = {0};
    extract_messages_content(req->body, prompt, sizeof(prompt));
    if (prompt[0] == '\0') {
        *out_body = oc_openai_error_json("no messages content found", "invalid_request_error");
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
    size_t cap = strlen(text) + 640;
    char *buf = malloc(cap);
    if (buf == NULL) { free(text); *out_status = 500; return; }
    snprintf(buf, cap,
        "{\"id\":\"chatcmpl-oxidize\",\"object\":\"chat.completion\",\"created\":0,"
        "\"model\":\"%s\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
        "\"content\":\"%s\"},\"finish_reason\":\"stop\"}],"
        "\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":0,\"total_tokens\":0}}",
        st->model_id ? st->model_id : "unknown", text);
    free(text);
    *out_status = 200;
    *out_body = buf;
}

/* ─── Dispatch ─────────────────────────────────────────────────────────── */

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
