/*
 * test_realtime.c — Realtime API session tests.
 *
 * VAL-RT-001..003 cover the pure helpers (no socket I/O, no loaded model):
 *   1. Message type parsing (input_text_delta, session.update, unknown).
 *   2. Event formatting (each event type produces valid JSON).
 *   3. Session init/free without a model (error event path).
 */
#define _GNU_SOURCE 1
#include "framework.h"

#include "oxidize/realtime.h"

#include <string.h>

/* ─── Message type parsing ─────────────────────────────────────────────────── */

Test(realtime, parse_input_text_delta)
{
    const char *msg = "{\"type\":\"input_text_delta\",\"delta\":\"hello\"}";
    cr_assert_eq(oc_realtime_parse_type(msg, strlen(msg)),
                 OC_RT_MSG_INPUT_TEXT_DELTA);
}

Test(realtime, parse_session_update)
{
    const char *msg = "{\"type\":\"session.update\",\"session\":{\"temperature\":0.9}}";
    cr_assert_eq(oc_realtime_parse_type(msg, strlen(msg)),
                 OC_RT_MSG_SESSION_UPDATE);
}

Test(realtime, parse_audio_buffer_append)
{
    const char *msg = "{\"type\":\"input_audio_buffer.append\",\"audio\":\"base64\"}";
    cr_assert_eq(oc_realtime_parse_type(msg, strlen(msg)),
                 OC_RT_MSG_INPUT_AUDIO_BUFFER_APPEND);
}

Test(realtime, parse_response_cancel)
{
    const char *msg = "{\"type\":\"response.cancel\"}";
    cr_assert_eq(oc_realtime_parse_type(msg, strlen(msg)),
                 OC_RT_MSG_RESPONSE_CANCEL);
}

Test(realtime, parse_unknown_type)
{
    const char *msg = "{\"type\":\"something.else\"}";
    cr_assert_eq(oc_realtime_parse_type(msg, strlen(msg)),
                 OC_RT_MSG_UNKNOWN);
}

Test(realtime, parse_missing_type_field)
{
    const char *msg = "{\"delta\":\"no type here\"}";
    cr_assert_eq(oc_realtime_parse_type(msg, strlen(msg)),
                 OC_RT_MSG_UNKNOWN);
}

Test(realtime, parse_null_json)
{
    cr_assert_eq(oc_realtime_parse_type(NULL, 0), OC_RT_MSG_UNKNOWN);
}

/* ─── Event formatting ─────────────────────────────────────────────────────── */

Test(realtime, format_speech_started)
{
    char buf[256];
    size_t n = oc_realtime_format_event(OC_RT_SPEECH_STARTED, NULL,
                                        buf, sizeof(buf));
    cr_assert_gt(n, 0);
    cr_assert_str_eq(buf, "{\"type\":\"speech.started\",\"delta\":null}");
}

Test(realtime, format_text_delta)
{
    char buf[256];
    size_t n = oc_realtime_format_event(OC_RT_TEXT_DELTA, "hello",
                                        buf, sizeof(buf));
    cr_assert_gt(n, 0);
    cr_assert(strstr(buf, "response.text.delta") != NULL);
    cr_assert(strstr(buf, "hello") != NULL);
}

Test(realtime, format_error)
{
    char buf[256];
    size_t n = oc_realtime_format_event(OC_RT_ERROR,
                                        "{\"message\":\"bad\"}",
                                        buf, sizeof(buf));
    cr_assert_gt(n, 0);
    cr_assert(strstr(buf, "\"type\":\"error\"") != NULL);
    cr_assert(strstr(buf, "bad") != NULL);
}

Test(realtime, format_overflow_returns_zero)
{
    char buf[8];   /* too small */
    size_t n = oc_realtime_format_event(OC_RT_ERROR, NULL, buf, sizeof(buf));
    cr_assert_eq(n, 0u);
}

/* ─── Session init/free (no model) ──────────────────────────────────────────── */

Test(realtime, session_init_free_without_model)
{
    OcRealtimeSession sess;
    OcWsSession ws;
    memset(&ws, 0, sizeof(ws));
    ws.fd = -1;
    cr_assert_eq(oc_realtime_session_init(&sess, &ws, NULL, NULL), OC_OK);
    cr_assert(sess.ws == &ws);
    cr_assert(sess.model == NULL);
    cr_assert(sess.tokenizer == NULL);
    cr_assert(!sess.has_llama_sess);
    cr_assert_eq(sess.cfg.max_response_tokens, 512u);
    cr_assert(sess.history != NULL);
    cr_assert(sess.history_cap > 0);
    /* process_message with no model should report an error path but not crash */
    const char *msg = "{\"type\":\"input_text_delta\",\"delta\":\"hi\"}";
    /* Returns OC_ERR_MODEL or OC_ERR_FORMAT — both are non-OK; the send path
     * writes to a closed socket (fd=-1) and best-efforts. We only check it
     * does not crash. */
    (void)oc_realtime_process_message(&sess, msg, strlen(msg));
    oc_realtime_session_free(&sess);
}

Test(realtime, session_init_null_rejected)
{
    cr_assert_eq(oc_realtime_session_init(NULL, NULL, NULL, NULL),
                 OC_ERR_INVALID_ARG);
}

Test(realtime, process_message_unknown_type_sends_error)
{
    OcRealtimeSession sess;
    OcWsSession ws;
    memset(&ws, 0, sizeof(ws));
    ws.fd = -1;
    cr_assert_eq(oc_realtime_session_init(&sess, &ws, NULL, NULL), OC_OK);
    const char *msg = "{\"type\":\"unknown.command\"}";
    OcError e = oc_realtime_process_message(&sess, msg, strlen(msg));
    cr_assert_eq(e, OC_ERR_FORMAT);
    oc_realtime_session_free(&sess);
}

Test(realtime, session_update_applies_config)
{
    OcRealtimeSession sess;
    OcWsSession ws;
    memset(&ws, 0, sizeof(ws));
    ws.fd = -1;
    cr_assert_eq(oc_realtime_session_init(&sess, &ws, NULL, NULL), OC_OK);
    const char *msg = "{\"type\":\"session.update\","
                      "\"session\":{\"temperature\":0.5,"
                      "\"voice\":\"nova\","
                      "\"max_response_output_tokens\":128}}";
    OcError e = oc_realtime_process_message(&sess, msg, strlen(msg));
    cr_assert_eq(e, OC_OK);
    cr_assert_float_eq(sess.cfg.temperature, 0.5f, 0.001f);
    cr_assert_str_eq(sess.cfg.voice, "nova");
    cr_assert_eq(sess.cfg.max_response_tokens, 128u);
    oc_realtime_session_free(&sess);
}
