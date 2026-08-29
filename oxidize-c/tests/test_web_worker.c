/* test_web_worker.c — Web worker tests. */
#include "framework.h"
#include "oxidize/web_worker.h"
#include <string.h>

Test(ww, init)
{
    OcWebWorker ww;
    cr_assert_eq(oc_web_worker_init(&ww), OC_OK);
    cr_assert(ww.running);
    cr_assert(!ww.busy);
    cr_assert_eq(ww.count, 0);
    cr_assert_eq(ww.next_request_id, 1);
    oc_web_worker_free(&ww);
}

OC_TEST_NULL_SAFE(ww, init_null,
        cr_assert_neq(oc_web_worker_init(NULL), OC_OK);)

Test(ww, send)
{
    OcWebWorker ww;
    oc_web_worker_init(&ww);
    uint64_t id;
    cr_assert_eq(oc_web_worker_send(&ww, OC_WW_MSG_GENERATE, "hello", 5, &id), OC_OK);
    cr_assert_eq(id, 1);
    cr_assert_eq(ww.count, 1);
    oc_web_worker_free(&ww);
}

Test(ww, send_null_payload)
{
    OcWebWorker ww;
    oc_web_worker_init(&ww);
    uint64_t id;
    cr_assert_eq(oc_web_worker_send(&ww, OC_WW_MSG_LOAD, NULL, 0, &id), OC_OK);
    cr_assert_eq(id, 1);
    oc_web_worker_free(&ww);
}

Test(ww, receive)
{
    OcWebWorker ww;
    oc_web_worker_init(&ww);
    uint64_t id;
    oc_web_worker_send(&ww, OC_WW_MSG_GENERATE, "hello", 5, &id);
    OcWwMessage msg;
    cr_assert_eq(oc_web_worker_receive(&ww, &msg), OC_OK);
    cr_assert_eq(msg.type, OC_WW_MSG_GENERATE);
    cr_assert_eq(msg.request_id, id);
    cr_assert_eq(msg.payload_len, 5);
    cr_assert(strncmp(msg.payload, "hello", 5) == 0);
    cr_assert_eq(ww.count, 0);
    oc_web_worker_free(&ww);
}

Test(ww, receive_empty)
{
    OcWebWorker ww;
    oc_web_worker_init(&ww);
    OcWwMessage msg;
    cr_assert_neq(oc_web_worker_receive(&ww, &msg), OC_OK);
    oc_web_worker_free(&ww);
}

Test(ww, send_result)
{
    OcWebWorker ww;
    oc_web_worker_init(&ww);
    cr_assert_eq(oc_web_worker_send_result(&ww, 42, "result", 6), OC_OK);
    cr_assert_eq(ww.count, 1);
    OcWwMessage msg;
    oc_web_worker_receive(&ww, &msg);
    cr_assert_eq(msg.type, OC_WW_MSG_RESULT);
    cr_assert_eq(msg.request_id, 42);
    oc_web_worker_free(&ww);
}

Test(ww, send_error)
{
    OcWebWorker ww;
    oc_web_worker_init(&ww);
    cr_assert_eq(oc_web_worker_send_error(&ww, 1, "failed"), OC_OK);
    OcWwMessage msg;
    oc_web_worker_receive(&ww, &msg);
    cr_assert_eq(msg.type, OC_WW_MSG_ERROR);
    cr_assert_str_eq(msg.payload, "failed");
    oc_web_worker_free(&ww);
}

Test(ww, cancel)
{
    OcWebWorker ww;
    oc_web_worker_init(&ww);
    ww.busy = true;
    cr_assert_eq(oc_web_worker_cancel(&ww, 1), OC_OK);
    cr_assert(!ww.busy);
    oc_web_worker_free(&ww);
}

Test(ww, queue_fills)
{
    OcWebWorker ww;
    oc_web_worker_init(&ww);
    for (int i = 0; i < OC_WW_MAX_QUEUE; i++) {
        cr_assert_eq(oc_web_worker_send(&ww, OC_WW_MSG_GENERATE, NULL, 0, NULL), OC_OK);
    }
    cr_assert_eq(ww.count, OC_WW_MAX_QUEUE);
    /* Next send should fail. */
    cr_assert_neq(oc_web_worker_send(&ww, OC_WW_MSG_GENERATE, NULL, 0, NULL), OC_OK);
    oc_web_worker_free(&ww);
}

Test(ww, circular_buffer)
{
    OcWebWorker ww;
    oc_web_worker_init(&ww);
    /* Fill and drain repeatedly. */
    for (int round = 0; round < 3; round++) {
        for (int i = 0; i < 10; i++) {
            oc_web_worker_send(&ww, OC_WW_MSG_GENERATE, NULL, 0, NULL);
        }
        for (int i = 0; i < 10; i++) {
            OcWwMessage msg;
            oc_web_worker_receive(&ww, &msg);
        }
    }
    cr_assert_eq(ww.count, 0);
    oc_web_worker_free(&ww);
}

Test(ww, is_running)
{
    OcWebWorker ww;
    oc_web_worker_init(&ww);
    cr_assert(oc_web_worker_is_running(&ww));
    ww.running = false;
    cr_assert(!oc_web_worker_is_running(&ww));
    oc_web_worker_free(&ww);
}

Test(ww, is_busy)
{
    OcWebWorker ww;
    oc_web_worker_init(&ww);
    cr_assert(!oc_web_worker_is_busy(&ww));
    ww.busy = true;
    cr_assert(oc_web_worker_is_busy(&ww));
    oc_web_worker_free(&ww);
}

Test(ww, queue_size)
{
    OcWebWorker ww;
    oc_web_worker_init(&ww);
    cr_assert_eq(oc_web_worker_queue_size(&ww), 0);
    oc_web_worker_send(&ww, OC_WW_MSG_LOAD, NULL, 0, NULL);
    cr_assert_eq(oc_web_worker_queue_size(&ww), 1);
    oc_web_worker_free(&ww);
}

OC_TEST_NULL_SAFE(ww, queue_size_null,
        cr_assert_eq(oc_web_worker_queue_size(NULL), 0);)

Test(ww, next_id)
{
    OcWebWorker ww;
    oc_web_worker_init(&ww);
    cr_assert_eq(oc_web_worker_next_id(&ww), 1);
    cr_assert_eq(oc_web_worker_next_id(&ww), 2);
    cr_assert_eq(oc_web_worker_next_id(&ww), 3);
    oc_web_worker_free(&ww);
}

Test(ww, msg_type_name)
{
    cr_assert_str_eq(oc_web_worker_msg_type_name(OC_WW_MSG_LOAD), "load");
    cr_assert_str_eq(oc_web_worker_msg_type_name(OC_WW_MSG_GENERATE), "generate");
    cr_assert_str_eq(oc_web_worker_msg_type_name(OC_WW_MSG_RESULT), "result");
    cr_assert_str_eq(oc_web_worker_msg_type_name(OC_WW_MSG_ERROR), "error");
}

OC_TEST_NULL_SAFE(ww, free_null,
        oc_web_worker_free(NULL);)

Test(ww, send_after_free)
{
    OcWebWorker ww;
    oc_web_worker_init(&ww);
    ww.running = false;
    cr_assert_neq(oc_web_worker_send(&ww, OC_WW_MSG_LOAD, NULL, 0, NULL), OC_OK);
    oc_web_worker_free(&ww);
}
