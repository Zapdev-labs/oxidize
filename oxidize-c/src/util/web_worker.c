/*
 * web_worker.c — Web worker bridge implementation.
 */
#include "oxidize/web_worker.h"

#include <stdlib.h>
#include <string.h>

OcError oc_web_worker_init(OcWebWorker *ww)
{
    if (!ww) return OC_ERR_INVALID_ARG;
    memset(ww, 0, sizeof(*ww));
    ww->running = true;
    ww->busy = false;
    ww->next_request_id = 1;
    return OC_OK;
}

OcError oc_web_worker_send(OcWebWorker *ww, OcWwMsgType type,
                          const char *payload, size_t payload_len,
                          uint64_t *out_request_id)
{
    if (!ww) return OC_ERR_INVALID_ARG;
    if (!ww->running) return OC_ERR_MODEL;
    if (ww->count >= OC_WW_MAX_QUEUE) return OC_ERR_OOM;

    OcWwMessage *msg = &ww->queue[ww->tail];
    memset(msg, 0, sizeof(*msg));
    msg->type = type;
    msg->request_id = ww->next_request_id++;
    if (payload && payload_len > 0) {
        size_t n = payload_len < OC_WW_MAX_MSG ? payload_len : OC_WW_MAX_MSG;
        memcpy(msg->payload, payload, n);
        msg->payload_len = n;
    }

    if (out_request_id) *out_request_id = msg->request_id;

    ww->tail = (ww->tail + 1) % OC_WW_MAX_QUEUE;
    ww->count++;
    return OC_OK;
}

OcError oc_web_worker_receive(OcWebWorker *ww, OcWwMessage *out_msg)
{
    if (!ww || !out_msg) return OC_ERR_INVALID_ARG;
    if (ww->count == 0) return OC_ERR_MODEL;

    *out_msg = ww->queue[ww->head];
    ww->head = (ww->head + 1) % OC_WW_MAX_QUEUE;
    ww->count--;
    return OC_OK;
}

OcError oc_web_worker_send_result(OcWebWorker *ww, uint64_t request_id,
                                 const char *payload, size_t payload_len)
{
    if (!ww) return OC_ERR_INVALID_ARG;
    if (!ww->running) return OC_ERR_MODEL;
    if (ww->count >= OC_WW_MAX_QUEUE) return OC_ERR_OOM;

    OcWwMessage *msg = &ww->queue[ww->tail];
    memset(msg, 0, sizeof(*msg));
    msg->type = OC_WW_MSG_RESULT;
    msg->request_id = request_id;
    if (payload && payload_len > 0) {
        size_t n = payload_len < OC_WW_MAX_MSG ? payload_len : OC_WW_MAX_MSG;
        memcpy(msg->payload, payload, n);
        msg->payload_len = n;
    }

    ww->tail = (ww->tail + 1) % OC_WW_MAX_QUEUE;
    ww->count++;
    ww->busy = false;
    return OC_OK;
}

OcError oc_web_worker_send_error(OcWebWorker *ww, uint64_t request_id,
                                const char *error_msg)
{
    if (!ww) return OC_ERR_INVALID_ARG;
    if (!ww->running) return OC_ERR_MODEL;
    if (ww->count >= OC_WW_MAX_QUEUE) return OC_ERR_OOM;

    OcWwMessage *msg = &ww->queue[ww->tail];
    memset(msg, 0, sizeof(*msg));
    msg->type = OC_WW_MSG_ERROR;
    msg->request_id = request_id;
    if (error_msg) {
        size_t n = strlen(error_msg);
        if (n >= OC_WW_MAX_MSG) n = OC_WW_MAX_MSG - 1;
        memcpy(msg->payload, error_msg, n);
        msg->payload_len = n;
    }

    ww->tail = (ww->tail + 1) % OC_WW_MAX_QUEUE;
    ww->count++;
    ww->busy = false;
    return OC_OK;
}

OcError oc_web_worker_cancel(OcWebWorker *ww, uint64_t request_id)
{
    if (!ww) return OC_ERR_INVALID_ARG;
    /* Mark as not busy - in a real implementation would cancel the operation. */
    (void)request_id;
    ww->busy = false;
    return OC_OK;
}

bool oc_web_worker_is_running(const OcWebWorker *ww)
{
    return ww ? ww->running : false;
}

bool oc_web_worker_is_busy(const OcWebWorker *ww)
{
    return ww ? ww->busy : false;
}

uint32_t oc_web_worker_queue_size(const OcWebWorker *ww)
{
    return ww ? ww->count : 0;
}

uint64_t oc_web_worker_next_id(OcWebWorker *ww)
{
    if (!ww) return 0;
    return ww->next_request_id++;
}

const char *oc_web_worker_msg_type_name(OcWwMsgType type)
{
    switch (type) {
    case OC_WW_MSG_LOAD:     return "load";
    case OC_WW_MSG_GENERATE: return "generate";
    case OC_WW_MSG_ENCODE:   return "encode";
    case OC_WW_MSG_DECODE:   return "decode";
    case OC_WW_MSG_TOKENIZE: return "tokenize";
    case OC_WW_MSG_BENCH:    return "bench";
    case OC_WW_MSG_CANCEL:   return "cancel";
    case OC_WW_MSG_STATUS:   return "status";
    case OC_WW_MSG_RESULT:   return "result";
    case OC_WW_MSG_ERROR:    return "error";
    default: return "unknown";
    }
}

void oc_web_worker_free(OcWebWorker *ww)
{
    if (!ww) return;
    memset(ww, 0, sizeof(*ww));
}
