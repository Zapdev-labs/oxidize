#ifndef OXIDIZE_WEB_WORKER_H
#define OXIDIZE_WEB_WORKER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_WW_MAX_MSG 4096
#define OC_WW_MAX_QUEUE 64

typedef enum {
    OC_WW_MSG_LOAD = 0,
    OC_WW_MSG_GENERATE = 1,
    OC_WW_MSG_ENCODE = 2,
    OC_WW_MSG_DECODE = 3,
    OC_WW_MSG_TOKENIZE = 4,
    OC_WW_MSG_BENCH = 5,
    OC_WW_MSG_CANCEL = 6,
    OC_WW_MSG_STATUS = 7,
    OC_WW_MSG_RESULT = 8,
    OC_WW_MSG_ERROR = 9,
} OcWwMsgType;

typedef struct {
    OcWwMsgType type;
    uint64_t request_id;
    char payload[OC_WW_MAX_MSG];
    size_t payload_len;
    uint32_t token;
    double elapsed_ms;
} OcWwMessage;

typedef struct {
    OcWwMessage queue[OC_WW_MAX_QUEUE];
    uint32_t head;
    uint32_t tail;
    uint32_t count;
    uint64_t next_request_id;
    bool running;
    bool busy;
} OcWebWorker;

OcError oc_web_worker_init(OcWebWorker *ww);
OcError oc_web_worker_send(OcWebWorker *ww, OcWwMsgType type,
                          const char *payload, size_t payload_len,
                          uint64_t *out_request_id);
OcError oc_web_worker_receive(OcWebWorker *ww, OcWwMessage *out_msg);
OcError oc_web_worker_send_result(OcWebWorker *ww, uint64_t request_id,
                                 const char *payload, size_t payload_len);
OcError oc_web_worker_send_error(OcWebWorker *ww, uint64_t request_id,
                                const char *error_msg);
OcError oc_web_worker_cancel(OcWebWorker *ww, uint64_t request_id);
bool oc_web_worker_is_running(const OcWebWorker *ww);
bool oc_web_worker_is_busy(const OcWebWorker *ww);
uint32_t oc_web_worker_queue_size(const OcWebWorker *ww);
uint64_t oc_web_worker_next_id(OcWebWorker *ww);
const char *oc_web_worker_msg_type_name(OcWwMsgType type);
void oc_web_worker_free(OcWebWorker *ww);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_WEB_WORKER_H */
