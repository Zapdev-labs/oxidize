/*
 * realtime.c — OpenAI Realtime API over WebSocket implementation.
 *
 * Drives a bidirectional WebSocket session: parses incoming JSON commands
 * (input_text_delta, session.update, audio buffer append/commit,
 * response.cancel), forwards text input through the loaded model, and
 * emits typed events (speech.started, response.text.delta,
 * speech.stopped, error) as JSON over the WebSocket.
 *
 * The text streaming path mirrors the generation loop in openai.c:
 * tokenize → forward prompt → sample + decode token-by-token → emit
 * text_delta events. When no model is loaded, the session responds with
 * an error event (matching the OpenAI Realtime error shape).
 */
#define _GNU_SOURCE 1   /* memmem, strdup */
#include "oxidize/realtime.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── JSON field extraction (minimal, tailored to Realtime message shape) ──── */

static const char *find_field_value(const char *json, size_t len,
                                    const char *key)
{
    char pattern[64];
    int pn = snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    if (pn < 0 || (size_t)pn >= sizeof(pattern)) return NULL;
    const char *end = json + len;
    const char *p = json;
    while (p < end) {
        const char *hit = memmem(p, (size_t)(end - p), pattern, (size_t)pn);
        if (hit == NULL) return NULL;
        const char *c = hit + pn;
        while (c < end && (*c == ' ' || *c == '\t' || *c == ':')) c++;
        return c;
    }
    return NULL;
}

/* Extract a quoted string field value into `out` (NUL-terminated). Returns
 * true on success. Handles minimal \" escapes. */
static bool extract_json_string(const char *json, size_t len,
                                const char *key, char *out, size_t cap)
{
    const char *v = find_field_value(json, len, key);
    if (v == NULL) return false;
    const char *end = json + len;
    if (v >= end || *v != '"') return false;
    v++;
    size_t i = 0;
    while (v < end && *v != '"' && i + 1 < cap) {
        if (*v == '\\' && v + 1 < end) {
            if (v[1] == '"')      { out[i++] = '"'; v += 2; }
            else if (v[1] == 'n') { out[i++] = '\n'; v += 2; }
            else if (v[1] == 't') { out[i++] = '\t'; v += 2; }
            else if (v[1] == '\\') { out[i++] = '\\'; v += 2; }
            else                  { out[i++] = v[1]; v += 2; }
        } else {
            out[i++] = *v++;
        }
    }
    out[i] = '\0';
    return (v < end && *v == '"');
}

static bool extract_json_double(const char *json, size_t len,
                                const char *key, double *out)
{
    const char *v = find_field_value(json, len, key);
    if (v == NULL) return false;
    char *endp = NULL;
    double d = strtod(v, &endp);
    if (endp == v) return false;
    *out = d;
    return true;
}

static bool extract_json_uint(const char *json, size_t len,
                             const char *key, uint32_t *out)
{
    const char *v = find_field_value(json, len, key);
    if (v == NULL) return false;
    char *endp = NULL;
    long val = strtol(v, &endp, 10);
    if (endp == v || val < 0) return false;
    *out = (uint32_t)val;
    return true;
}

/* ─── Message type parsing ─────────────────────────────────────────────────── */

OcRealtimeMessageType oc_realtime_parse_type(const char *json, size_t len)
{
    if (json == NULL || len == 0) return OC_RT_MSG_UNKNOWN;
    char type[64];
    if (!extract_json_string(json, len, "type", type, sizeof(type))) {
        return OC_RT_MSG_UNKNOWN;
    }
    if (strcmp(type, "input_text_delta") == 0) {
        return OC_RT_MSG_INPUT_TEXT_DELTA;
    }
    if (strcmp(type, "input_audio_buffer.append") == 0) {
        return OC_RT_MSG_INPUT_AUDIO_BUFFER_APPEND;
    }
    if (strcmp(type, "input_audio_buffer.commit") == 0) {
        return OC_RT_MSG_INPUT_AUDIO_BUFFER_COMMIT;
    }
    if (strcmp(type, "session.update") == 0) {
        return OC_RT_MSG_SESSION_UPDATE;
    }
    if (strcmp(type, "response.cancel") == 0) {
        return OC_RT_MSG_RESPONSE_CANCEL;
    }
    return OC_RT_MSG_UNKNOWN;
}

/* ─── Event formatting ─────────────────────────────────────────────────────── */

size_t oc_realtime_format_event(OcRealtimeEvent ev, const char *data,
                                char *buf, size_t cap)
{
    const char *type;
    switch (ev) {
    case OC_RT_SPEECH_STARTED: type = "speech.started"; break;
    case OC_RT_TEXT_DELTA:     type = "response.text.delta"; break;
    case OC_RT_SPEECH_STOPPED: type = "speech.stopped"; break;
    case OC_RT_ERROR:          type = "error"; break;
    default:                   type = "unknown"; break;
    }
    /* An absent payload serializes as JSON null (an empty value would make
     * the frame invalid JSON). */
    const char *payload = (data != NULL) ? data : "null";
    int n = snprintf(buf, cap, "{\"type\":\"%s\",\"delta\":%s}",
                     type, payload);
    if (n < 0 || (size_t)n >= cap) return 0;
    return (size_t)n;
}

/* ─── Session lifecycle ────────────────────────────────────────────────────── */

OcError oc_realtime_session_init(OcRealtimeSession *sess,
                                 OcWsSession *ws,
                                 OcLlamaModel *model,
                                 OcTokenizer *tokenizer)
{
    if (sess == NULL || ws == NULL) return OC_ERR_INVALID_ARG;
    memset(sess, 0, sizeof(*sess));
    sess->ws = ws;
    sess->model = model;
    sess->tokenizer = tokenizer;
    sess->cfg.temperature = 0.8f;
    strncpy(sess->cfg.voice, "alloy", sizeof(sess->cfg.voice) - 1);
    sess->cfg.max_response_tokens = 512u;
    if (model != NULL) {
        OcError e = oc_llama_session_init(model, &sess->llama_sess);
        if (e != OC_OK) return e;
        sess->has_llama_sess = true;
    }
    sess->history_cap = 4096u;
    sess->history = malloc(sess->history_cap * sizeof(uint32_t));
    if (sess->history == NULL) {
        if (sess->has_llama_sess) {
            oc_llama_session_free(&sess->llama_sess);
            sess->has_llama_sess = false;
        }
        return OC_ERR_OOM;
    }
    return OC_OK;
}

void oc_realtime_session_free(OcRealtimeSession *sess)
{
    if (sess == NULL) return;
    free(sess->history);
    sess->history = NULL;
    sess->history_len = 0;
    sess->history_cap = 0;
    if (sess->has_llama_sess) {
        oc_llama_session_free(&sess->llama_sess);
        sess->has_llama_sess = false;
    }
}

/* ─── Event sending ────────────────────────────────────────────────────────── */

OcError oc_realtime_send_event(OcRealtimeSession *sess,
                               OcRealtimeEvent ev, const char *data)
{
    if (sess == NULL || sess->ws == NULL) return OC_ERR_INVALID_ARG;
    char buf[8192];
    /* For text_delta, wrap data in quotes if it's raw text. Caller may pass
     * already-quoted JSON; we detect by checking the first non-space char. */
    const char *payload = data;
    char quoted[4096];
    if (ev == OC_RT_TEXT_DELTA && data != NULL) {
        const char *p = data;
        while (*p == ' ' || *p == '\t') p++;
        if (*p != '"') {
            /* Escape + quote the raw string (JSON-special + control chars). */
            size_t oi = 0;
            quoted[oi++] = '"';
            for (const char *s = data; *s && oi + 7 < sizeof(quoted); s++) {
                unsigned char c = (unsigned char)*s;
                if (c == '"' || c == '\\') {
                    quoted[oi++] = '\\';
                    quoted[oi++] = (char)c;
                } else if (c == '\n') {
                    quoted[oi++] = '\\'; quoted[oi++] = 'n';
                } else if (c == '\t') {
                    quoted[oi++] = '\\'; quoted[oi++] = 't';
                } else if (c == '\r') {
                    quoted[oi++] = '\\'; quoted[oi++] = 'r';
                } else if (c < 0x20u) {
                    oi += (size_t)snprintf(quoted + oi, sizeof(quoted) - oi,
                                           "\\u%04x", (unsigned)c);
                } else {
                    quoted[oi++] = (char)c;
                }
            }
            quoted[oi++] = '"';
            quoted[oi] = '\0';
            payload = quoted;
        }
    }
    size_t n = oc_realtime_format_event(ev, payload, buf, sizeof(buf));
    if (n == 0) return OC_ERR_INTERNAL;
    return oc_ws_send_text(sess->ws, buf);
}

/* ─── Text generation (mirrors openai.c loop) ──────────────────────────────── */

static OcError generate_text_response(OcRealtimeSession *sess,
                                      const char *prompt_text)
{
    if (sess->model == NULL || sess->tokenizer == NULL ||
        !sess->has_llama_sess) {
        oc_realtime_send_event(sess, OC_RT_ERROR,
                               "{\"message\":\"no model loaded\"}");
        return OC_ERR_MODEL;
    }
    /* Prepend session instructions (session.update) to the prompt so they
     * actually influence the response. */
    char *combined = NULL;
    if (sess->cfg.instructions[0] != '\0') {
        size_t need = strlen(sess->cfg.instructions) + strlen(prompt_text) + 3u;
        combined = malloc(need);
        if (combined != NULL) {
            snprintf(combined, need, "%s\n\n%s", sess->cfg.instructions,
                     prompt_text);
            prompt_text = combined;
        }
    }
    /* Tokenize prompt. */
    uint32_t *ids = NULL;
    size_t n_ids = 0;
    OcError e = oc_tokenizer_encode(sess->tokenizer, prompt_text,
                                    OC_TOK_DEFAULT, &ids, &n_ids);
    free(combined);
    if (e != OC_OK || ids == NULL || n_ids == 0) {
        oc_realtime_send_event(sess, OC_RT_ERROR,
                               "{\"message\":\"tokenization failed\"}");
        free(ids);
        return OC_ERR_TOKENIZER;
    }
    /* Feed prompt tokens (skip logits for all but the last). */
    for (size_t i = 0; i + 1 < n_ids; i++) {
        if (oc_llama_forward(&sess->llama_sess, ids[i], NULL) != OC_OK) {
            free(ids);
            oc_realtime_send_event(sess, OC_RT_ERROR,
                                   "{\"message\":\"forward failed\"}");
            return OC_ERR_MODEL;
        }
    }
    if (oc_llama_forward(&sess->llama_sess, ids[n_ids - 1],
                         sess->llama_sess.logits) != OC_OK) {
        free(ids);
        oc_realtime_send_event(sess, OC_RT_ERROR,
                               "{\"message\":\"forward failed\"}");
        return OC_ERR_MODEL;
    }
    oc_realtime_send_event(sess, OC_RT_SPEECH_STARTED, NULL);

    /* Generate up to max_response_tokens. */
    OcSamplerConfig scfg = OC_SAMPLER_DEFAULT;
    scfg.temperature = sess->cfg.temperature;
    uint32_t eos = sess->tokenizer->eos_id;
    bool has_eos = sess->tokenizer->has_eos;
    for (uint32_t step = 0; step < sess->cfg.max_response_tokens; step++) {
        uint32_t tok = oc_sample(sess->llama_sess.logits,
                                 sess->model->cfg.vocab_size,
                                 &scfg, NULL, 0);
        if (has_eos && tok == eos) break;
        char *piece = NULL;
        if (oc_tokenizer_decode(sess->tokenizer, &tok, 1, &piece) == OC_OK &&
            piece != NULL) {
            oc_realtime_send_event(sess, OC_RT_TEXT_DELTA, piece);
            free(piece);
        }
        if (oc_llama_forward(&sess->llama_sess, tok,
                             sess->llama_sess.logits) != OC_OK) {
            break;
        }
    }
    oc_realtime_send_event(sess, OC_RT_SPEECH_STOPPED, NULL);
    free(ids);
    return OC_OK;
}

/* ─── Message dispatch ─────────────────────────────────────────────────────── */

OcError oc_realtime_process_message(OcRealtimeSession *sess,
                                    const char *json, size_t len)
{
    if (sess == NULL || json == NULL) return OC_ERR_INVALID_ARG;
    OcRealtimeMessageType t = oc_realtime_parse_type(json, len);
    switch (t) {
    case OC_RT_MSG_INPUT_TEXT_DELTA: {
        char delta[8192];
        if (!extract_json_string(json, len, "delta", delta, sizeof(delta))) {
            oc_realtime_send_event(sess, OC_RT_ERROR,
                                   "{\"message\":\"missing delta\"}");
            return OC_ERR_FORMAT;
        }
        return generate_text_response(sess, delta);
    }
    case OC_RT_MSG_SESSION_UPDATE: {
        /* Find the "session" object and apply known fields. */
        double temp;
        if (extract_json_double(json, len, "temperature", &temp)) {
            sess->cfg.temperature = (float)temp;
        }
        char voice[64];
        if (extract_json_string(json, len, "voice", voice, sizeof(voice))) {
            strncpy(sess->cfg.voice, voice, sizeof(sess->cfg.voice) - 1);
            sess->cfg.voice[sizeof(sess->cfg.voice) - 1] = '\0';
        }
        char instr[1024];
        if (extract_json_string(json, len, "instructions", instr,
                                sizeof(instr))) {
            strncpy(sess->cfg.instructions, instr,
                    sizeof(sess->cfg.instructions) - 1);
            sess->cfg.instructions[sizeof(sess->cfg.instructions) - 1] = '\0';
        }
        uint32_t max_resp;
        if (extract_json_uint(json, len, "max_response_output_tokens",
                              &max_resp) && max_resp > 0) {
            sess->cfg.max_response_tokens = max_resp;
        }
        return OC_OK;
    }
    case OC_RT_MSG_INPUT_AUDIO_BUFFER_APPEND:
        /* Audio path is out of scope for the C port; acknowledge. */
        return OC_OK;
    case OC_RT_MSG_INPUT_AUDIO_BUFFER_COMMIT:
        return OC_OK;
    case OC_RT_MSG_RESPONSE_CANCEL:
        /* Generation is synchronous: by the time this frame is read, no
         * response is in flight. Tell the client rather than silently
         * acknowledging an unsupported operation. */
        oc_realtime_send_event(sess, OC_RT_ERROR,
                               "{\"message\":\"no active response to cancel\"}");
        return OC_OK;
    case OC_RT_MSG_UNKNOWN:
    default:
        oc_realtime_send_event(sess, OC_RT_ERROR,
                               "{\"message\":\"unknown message type\"}");
        return OC_ERR_FORMAT;
    }
}

/* ─── Main session loop ────────────────────────────────────────────────────── */

OcError oc_realtime_handle_session(OcRealtimeSession *sess)
{
    if (sess == NULL || sess->ws == NULL) return OC_ERR_INVALID_ARG;
    while (!sess->closed && sess->ws->state != OC_WS_CLOSED) {
        OcWsFrame frame;
        OcError e = oc_ws_read_frame(sess->ws, &frame);
        if (e != OC_OK) {
            sess->closed = true;
            break;
        }
        if (frame.opcode == OC_WS_OPCODE_CLOSE) {
            oc_ws_close(sess->ws, 1000);
            sess->closed = true;
            break;
        }
        if (frame.opcode == OC_WS_OPCODE_PING) {
            oc_ws_send_frame(sess->ws, OC_WS_OPCODE_PONG, true,
                             frame.payload, frame.payload_len);
            continue;
        }
        if (frame.opcode == OC_WS_OPCODE_TEXT ||
            frame.opcode == OC_WS_OPCODE_CONTINUATION) {
            /* Treat as text message; ensure NUL-termination. */
            char *msg = malloc(frame.payload_len + 1u);
            if (msg == NULL) return OC_ERR_OOM;
            memcpy(msg, frame.payload, frame.payload_len);
            msg[frame.payload_len] = '\0';
            OcError pe = oc_realtime_process_message(sess, msg,
                                                     frame.payload_len);
            free(msg);
            if (pe != OC_OK && pe != OC_ERR_FORMAT) {
                return pe;
            }
        }
    }
    return OC_OK;
}
