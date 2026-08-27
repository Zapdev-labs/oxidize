/*
 * realtime.h — OpenAI Realtime API over WebSocket.
 *
 * Implements the `server-realtime` feature on top of server-websocket. A
 * realtime session is a bidirectional WebSocket connection that streams
 * model output as a sequence of typed events (text deltas, speech
 * markers, errors). Incoming client messages are JSON-encoded commands:
 *   {"type":"input_text_delta","delta":"..."}
 *   {"type":"session.update","session":{"temperature":0.7,...}}
 *   {"type":"input_audio_buffer.append","audio":"base64..."}
 *
 * Scope (oxidize-c port): the text streaming path. Audio encoding/decoding
 * is out of scope (the Rust core handles TTS/STT); the C port forwards
 * text input through the loaded model and emits text_delta events.
 */
#ifndef OXIDIZE_REALTIME_H
#define OXIDIZE_REALTIME_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/llama.h"
#include "oxidize/sampling.h"
#include "oxidize/tokenizer.h"
#include "oxidize/websocket.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Event types emitted by the server ───────────────────────────────────── */
typedef enum {
    OC_RT_SPEECH_STARTED = 0,
    OC_RT_TEXT_DELTA,
    OC_RT_SPEECH_STOPPED,
    OC_RT_ERROR,
} OcRealtimeEvent;

/* ─── Incoming client message types (parsed from JSON) ────────────────────── */
typedef enum {
    OC_RT_MSG_UNKNOWN = 0,
    OC_RT_MSG_INPUT_TEXT_DELTA,
    OC_RT_MSG_INPUT_AUDIO_BUFFER_APPEND,
    OC_RT_MSG_INPUT_AUDIO_BUFFER_COMMIT,
    OC_RT_MSG_SESSION_UPDATE,
    OC_RT_MSG_RESPONSE_CANCEL,
} OcRealtimeMessageType;

/* ─── Session config (updatable via session.update) ───────────────────────── */
typedef struct OcRealtimeSessionConfig {
    float temperature;
    char  voice[64];
    char  instructions[1024];
    uint32_t max_response_tokens;
} OcRealtimeSessionConfig;

/* ─── Realtime session ────────────────────────────────────────────────────── */
typedef struct OcRealtimeSession {
    OcWsSession            *ws;
    OcLlamaModel           *model;
    OcTokenizer            *tokenizer;
    OcLlamaSession          llama_sess;
    OcRealtimeSessionConfig cfg;
    /* conversation token history (for context continuity) */
    uint32_t              *history;
    size_t                 history_len;
    size_t                 history_cap;
    bool                   has_llama_sess;
    bool                   closed;
} OcRealtimeSession;

/* Initialize a realtime session bound to an open WebSocket session + a
 * loaded model/tokenizer. `model`/`tokenizer` may be NULL (the session
 * will respond with error events). Returns OC_OK or OC_ERR_OOM. */
OcError oc_realtime_session_init(OcRealtimeSession *sess,
                                 OcWsSession *ws,
                                 OcLlamaModel *model,
                                 OcTokenizer *tokenizer);

/* Release session-owned buffers (history, llama session). Does NOT close
 * the underlying WebSocket. Safe on NULL. */
void oc_realtime_session_free(OcRealtimeSession *sess);

/* Main loop for a WebSocket realtime session. Reads frames, dispatches
 * them via oc_realtime_process_message, and returns when the client closes
 * or an unrecoverable error occurs. Returns OC_OK on clean close. */
OcError oc_realtime_handle_session(OcRealtimeSession *sess);

/* Send a realtime event as JSON over the WebSocket. `data` is the event
 * payload (already JSON-encoded value, or NULL for empty). Returns OC_OK
 * or an error from the WebSocket layer. */
OcError oc_realtime_send_event(OcRealtimeSession *sess,
                               OcRealtimeEvent ev, const char *data);

/* Process one incoming client message (JSON). Parses the type and
 * dispatches: input_text_delta → tokenize + forward through model +
 * emit text_delta events; session.update → update cfg; others →
 * acknowledged or ignored. Returns OC_OK or OC_ERR_FORMAT. */
OcError oc_realtime_process_message(OcRealtimeSession *sess,
                                    const char *json, size_t len);

/* ─── Pure helpers (exposed for testing) ──────────────────────────────────── */

/* Parse the "type" field of a JSON message. Returns the matching enum or
 * OC_RT_MSG_UNKNOWN if unrecognized. */
OcRealtimeMessageType oc_realtime_parse_type(const char *json, size_t len);

/* Format a realtime event as a JSON string into `buf`. Returns bytes
 * written excluding NUL, or 0 on overflow. */
size_t oc_realtime_format_event(OcRealtimeEvent ev, const char *data,
                                char *buf, size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_REALTIME_H */
