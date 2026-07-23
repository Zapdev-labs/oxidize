/* test_batch_engine.c — Continuous-batching decode engine tests. */
#include <criterion/criterion.h>
#include "oxidize/batch_engine.h"
#include <string.h>

Test(batch, config_init)
{
    OcBatchConfig cfg;
    oc_batch_config_init(&cfg);
    cr_assert_eq(cfg.max_batch, 32);
    cr_assert_eq(cfg.default_capacity_tokens, 2048);
}

Test(batch, engine_init)
{
    OcBatchConfig cfg;
    oc_batch_config_init(&cfg);
    OcBatchEngine *engine = NULL;
    cr_assert_eq(oc_batch_engine_init(&engine, &cfg, 32, 4096), OC_OK);
    cr_assert_not_null(engine);
    cr_assert_eq(oc_batch_active_len(engine), 0);
    cr_assert_eq(oc_batch_pending_len(engine), 0);
    cr_assert(!oc_batch_has_work(engine));
    oc_batch_engine_free(engine);
}

Test(batch, engine_init_null)
{
    cr_assert_neq(oc_batch_engine_init(NULL, NULL, 1, 1), OC_OK);
}

Test(batch, engine_init_bad_kv)
{
    OcBatchEngine *engine = NULL;
    cr_assert_neq(oc_batch_engine_init(&engine, NULL, 0, 0), OC_OK);
}

Test(batch, submit)
{
    OcBatchEngine *engine = NULL;
    oc_batch_engine_init(&engine, NULL, 32, 4096);
    uint32_t prompt[] = {1, 2, 3};
    OcSeqId id;
    cr_assert_eq(oc_batch_submit(engine, prompt, 3, 10, 0, false, &id), OC_OK);
    cr_assert_eq(id, 1);
    cr_assert_eq(oc_batch_pending_len(engine), 1);
    cr_assert(oc_batch_has_work(engine));
    cr_assert_eq(oc_batch_total_submitted(engine), 1);
    oc_batch_engine_free(engine);
}

Test(batch, submit_null)
{
    cr_assert_neq(oc_batch_submit(NULL, NULL, 0, 0, 0, false, NULL), OC_OK);
}

Test(batch, submit_empty_prompt)
{
    OcBatchEngine *engine = NULL;
    oc_batch_engine_init(&engine, NULL, 32, 4096);
    OcSeqId id;
    cr_assert_neq(oc_batch_submit(engine, (uint32_t[]){1}, 0, 10, 0, false, &id), OC_OK);
    oc_batch_engine_free(engine);
}

Test(batch, step_admits_and_decodes)
{
    OcBatchEngine *engine = NULL;
    oc_batch_engine_init(&engine, NULL, 32, 4096);
    uint32_t prompt[] = {1, 2, 3};
    OcSeqId id;
    oc_batch_submit(engine, prompt, 3, 5, 0, false, &id);

    /* First step should admit the pending request and produce one token. */
    OcBatchStepOutput out[16];
    size_t n_out;
    cr_assert_eq(oc_batch_step(engine, out, 16, &n_out), OC_OK);
    cr_assert_eq(n_out, 1);
    cr_assert_eq(out[0].seq_id, id);
    cr_assert(!out[0].finished);
    cr_assert_eq(oc_batch_active_len(engine), 1);
    cr_assert_eq(oc_batch_pending_len(engine), 0);
    oc_batch_engine_free(engine);
}

Test(batch, step_completes_on_max_new)
{
    OcBatchEngine *engine = NULL;
    oc_batch_engine_init(&engine, NULL, 32, 4096);
    uint32_t prompt[] = {1};
    OcSeqId id;
    oc_batch_submit(engine, prompt, 1, 3, 0, false, &id);

    OcBatchStepOutput out[16];
    size_t n_out;
    /* Step 1: admit + decode (generated=2). */
    oc_batch_step(engine, out, 16, &n_out);
    cr_assert_eq(n_out, 1);
    cr_assert(!out[0].finished);
    /* Step 2: decode (generated=3=max_new -> finished). */
    oc_batch_step(engine, out, 16, &n_out);
    cr_assert_eq(n_out, 1);
    cr_assert(out[0].finished);
    /* Step 3: no more work. */
    oc_batch_step(engine, out, 16, &n_out);
    cr_assert_eq(n_out, 0);
    cr_assert(!oc_batch_has_work(engine));
    oc_batch_engine_free(engine);
}

Test(batch, stop_token)
{
    OcBatchEngine *engine = NULL;
    oc_batch_engine_init(&engine, NULL, 32, 4096);
    uint32_t prompt[] = {1};
    OcSeqId id;
    /* stop_token = 2. The stub increments last_token each step. */
    oc_batch_submit(engine, prompt, 1, 100, 2, true, &id);

    OcBatchStepOutput out[16];
    size_t n_out;
    oc_batch_step(engine, out, 16, &n_out);  /* last=2 -> stop! */
    cr_assert_eq(n_out, 1);
    cr_assert(out[0].finished);
    oc_batch_engine_free(engine);
}

Test(batch, cancel_pending)
{
    OcBatchEngine *engine = NULL;
    oc_batch_engine_init(&engine, NULL, 32, 4096);
    uint32_t prompt[] = {1, 2};
    OcSeqId id;
    oc_batch_submit(engine, prompt, 2, 10, 0, false, &id);
    cr_assert_eq(oc_batch_pending_len(engine), 1);
    cr_assert(oc_batch_cancel(engine, id));
    cr_assert_eq(oc_batch_pending_len(engine), 0);
    oc_batch_engine_free(engine);
}

Test(batch, cancel_not_found)
{
    OcBatchEngine *engine = NULL;
    oc_batch_engine_init(&engine, NULL, 32, 4096);
    cr_assert(!oc_batch_cancel(engine, 999));
    oc_batch_engine_free(engine);
}

Test(batch, multiple_sequences)
{
    OcBatchEngine *engine = NULL;
    oc_batch_engine_init(&engine, NULL, 32, 4096);
    uint32_t p1[] = {1};
    uint32_t p2[] = {2};
    uint32_t p3[] = {3};
    OcSeqId id1, id2, id3;
    oc_batch_submit(engine, p1, 1, 5, 0, false, &id1);
    oc_batch_submit(engine, p2, 1, 5, 0, false, &id2);
    oc_batch_submit(engine, p3, 1, 5, 0, false, &id3);
    cr_assert_eq(oc_batch_pending_len(engine), 3);

    OcBatchStepOutput out[32];
    size_t n_out;
    oc_batch_step(engine, out, 32, &n_out);
    cr_assert_eq(n_out, 3);  /* all 3 admitted */
    cr_assert_eq(oc_batch_active_len(engine), 3);
    cr_assert_eq(oc_batch_pending_len(engine), 0);
    oc_batch_engine_free(engine);
}

Test(batch, seq_position)
{
    OcBatchEngine *engine = NULL;
    oc_batch_engine_init(&engine, NULL, 32, 4096);
    uint32_t prompt[] = {1, 2, 3, 4, 5};
    OcSeqId id;
    oc_batch_submit(engine, prompt, 5, 10, 0, false, &id);

    OcBatchStepOutput out[16];
    size_t n_out;
    oc_batch_step(engine, out, 16, &n_out);  /* admit */

    size_t pos;
    cr_assert_eq(oc_batch_seq_position(engine, id, &pos), OC_OK);
    cr_assert_eq(pos, 6);  /* 5 prompt + 1 generated */
    oc_batch_engine_free(engine);
}

Test(batch, seq_position_not_found)
{
    OcBatchEngine *engine = NULL;
    oc_batch_engine_init(&engine, NULL, 32, 4096);
    size_t pos;
    cr_assert_neq(oc_batch_seq_position(engine, 999, &pos), OC_OK);
    oc_batch_engine_free(engine);
}

Test(batch, total_submitted)
{
    OcBatchEngine *engine = NULL;
    oc_batch_engine_init(&engine, NULL, 32, 4096);
    uint32_t p[] = {1};
    OcSeqId id;
    oc_batch_submit(engine, p, 1, 5, 0, false, &id);
    oc_batch_submit(engine, p, 1, 5, 0, false, &id);
    oc_batch_submit(engine, p, 1, 5, 0, false, &id);
    cr_assert_eq(oc_batch_total_submitted(engine), 3);
    oc_batch_engine_free(engine);
}

Test(batch, free_null)
{
    oc_batch_engine_free(NULL);
}
