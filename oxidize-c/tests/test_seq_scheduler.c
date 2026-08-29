/* test_seq_scheduler.c — sequence scheduler tests. */
#define _POSIX_C_SOURCE 200809L
#include <criterion/criterion.h>
#include "oxidize/seq_scheduler.h"
#include "oxidize/kv_page.h"
#include <string.h>

/* Helper: build a request with given id, prompt, and max_tokens. */
static OcSeqRequest make_request(uint64_t id, const uint32_t *prompt,
                                  size_t n_prompt, size_t max_tokens,
                                  float temperature)
{
    OcSeqRequest r;
    r.id = id;
    r.prompt_tokens = prompt;
    r.n_prompt = n_prompt;
    r.max_tokens = max_tokens;
    r.temperature = temperature;
    return r;
}

static uint32_t one_prompt[] = {1};


Test(seq_scheduler, config_default)
{
    OcSeqSchedulerConfig c = oc_seq_sched_config_default();
    cr_assert_eq(c.max_batch_size, OC_SEQ_SCHED_DEFAULT_MAX_BATCH_SIZE);
    cr_assert_eq(c.max_total_tokens, OC_SEQ_SCHED_DEFAULT_MAX_TOTAL_TOKENS);
    cr_assert_float_eq(c.water_level, OC_SEQ_SCHED_DEFAULT_WATER_LEVEL, 1e-6f);
}

Test(seq_scheduler, init_free)
{
    OcSeqScheduler *s = NULL;
    OcSeqSchedulerConfig c = oc_seq_sched_config_default();
    c.max_batch_size = 4;
    cr_assert_eq(oc_seq_sched_init(&s, c, NULL), OC_OK);
    cr_assert_not_null(s);
    cr_assert_eq(s->n_waiting, 0u);
    cr_assert_eq(s->n_running, 0u);
    oc_seq_sched_free(s);
}

Test(seq_scheduler, init_invalid_config)
{
    OcSeqScheduler *s = NULL;
    OcSeqSchedulerConfig c = oc_seq_sched_config_default();
    c.max_batch_size = 0;
    cr_assert_eq(oc_seq_sched_init(&s, c, NULL), OC_ERR_INVALID_ARG);
}

Test(seq_scheduler, init_invalid_water_level)
{
    OcSeqScheduler *s = NULL;
    OcSeqSchedulerConfig c = oc_seq_sched_config_default();
    c.water_level = 2.0f;
    cr_assert_eq(oc_seq_sched_init(&s, c, NULL), OC_ERR_INVALID_ARG);
}


Test(seq_scheduler, add_request)
{
    OcSeqScheduler *s = NULL;
    OcSeqSchedulerConfig c = oc_seq_sched_config_default();
    c.max_batch_size = 4;
    oc_seq_sched_init(&s, c, NULL);
    OcSeqRequest r = make_request(42, one_prompt, 1, 10, 0.8f);
    cr_assert_eq(oc_seq_sched_add(s, &r), OC_OK);
    cr_assert_eq(oc_seq_sched_waiting_count(s), 1u);
    oc_seq_sched_free(s);
}

Test(seq_scheduler, add_null_request)
{
    OcSeqScheduler *s = NULL;
    OcSeqSchedulerConfig c = oc_seq_sched_config_default();
    c.max_batch_size = 4;
    oc_seq_sched_init(&s, c, NULL);
    cr_assert_eq(oc_seq_sched_add(s, NULL), OC_ERR_INVALID_ARG);
    oc_seq_sched_free(s);
}

Test(seq_scheduler, add_duplicate_id)
{
    OcSeqScheduler *s = NULL;
    OcSeqSchedulerConfig c = oc_seq_sched_config_default();
    c.max_batch_size = 4;
    oc_seq_sched_init(&s, c, NULL);
    OcSeqRequest r = make_request(7, one_prompt, 1, 5, 0.0f);
    cr_assert_eq(oc_seq_sched_add(s, &r), OC_OK);
    cr_assert_eq(oc_seq_sched_add(s, &r), OC_ERR_INVALID_ARG);
    oc_seq_sched_free(s);
}

Test(seq_scheduler, add_when_full)
{
    OcSeqScheduler *s = NULL;
    OcSeqSchedulerConfig c = oc_seq_sched_config_default();
    c.max_batch_size = 2;
    oc_seq_sched_init(&s, c, NULL);
    OcSeqRequest r1 = make_request(1, one_prompt, 1, 5, 0.0f);
    OcSeqRequest r2 = make_request(2, one_prompt, 1, 5, 0.0f);
    OcSeqRequest r3 = make_request(3, one_prompt, 1, 5, 0.0f);
    cr_assert_eq(oc_seq_sched_add(s, &r1), OC_OK);
    cr_assert_eq(oc_seq_sched_add(s, &r2), OC_OK);
    cr_assert_eq(oc_seq_sched_add(s, &r3), OC_ERR_OOM);
    oc_seq_sched_free(s);
}


Test(seq_scheduler, schedule_empty)
{
    OcSeqScheduler *s = NULL;
    OcSeqSchedulerConfig c = oc_seq_sched_config_default();
    c.max_batch_size = 4;
    oc_seq_sched_init(&s, c, NULL);
    OcSeqBatch batch;
    cr_assert_eq(oc_seq_sched_schedule(s, &batch), OC_OK);
    cr_assert_eq(batch.n_seqs, 0u);
    oc_seq_sched_free(s);
}

Test(seq_scheduler, schedule_prefill_waiting)
{
    OcSeqScheduler *s = NULL;
    OcSeqSchedulerConfig c = oc_seq_sched_config_default();
    c.max_batch_size = 4;
    c.max_total_tokens = 1024;
    oc_seq_sched_init(&s, c, NULL);
    OcSeqRequest r = make_request(10, one_prompt, 1, 5, 0.0f);
    oc_seq_sched_add(s, &r);
    OcSeqBatch batch;
    oc_seq_sched_schedule(s, &batch);
    cr_assert_eq(batch.n_seqs, 1u);
    cr_assert(batch.is_prefill[0]);
    cr_assert_eq(oc_seq_sched_running_count(s), 1u);
    cr_assert_eq(oc_seq_sched_waiting_count(s), 0u);
    oc_seq_sched_free(s);
}

Test(seq_scheduler, schedule_decode_running)
{
    OcSeqScheduler *s = NULL;
    OcSeqSchedulerConfig c = oc_seq_sched_config_default();
    c.max_batch_size = 4;
    c.max_total_tokens = 1024;
    oc_seq_sched_init(&s, c, NULL);
    OcSeqRequest r = make_request(10, one_prompt, 1, 5, 0.0f);
    oc_seq_sched_add(s, &r);
    /* First schedule: prefill. */
    OcSeqBatch batch;
    oc_seq_sched_schedule(s, &batch);
    cr_assert(batch.is_prefill[0]);
    /* Second schedule: decode (the seq is now RUNNING). */
    oc_seq_sched_schedule(s, &batch);
    cr_assert_eq(batch.n_seqs, 1u);
    cr_assert(!batch.is_prefill[0]);
    oc_seq_sched_free(s);
}

Test(seq_scheduler, schedule_mixed_batch)
{
    OcSeqScheduler *s = NULL;
    OcSeqSchedulerConfig c = oc_seq_sched_config_default();
    c.max_batch_size = 4;
    c.max_total_tokens = 1024;
    oc_seq_sched_init(&s, c, NULL);
    OcSeqRequest r1 = make_request(1, one_prompt, 1, 5, 0.0f);
    OcSeqRequest r2 = make_request(2, one_prompt, 1, 5, 0.0f);
    oc_seq_sched_add(s, &r1);
    oc_seq_sched_add(s, &r2);
    /* First schedule: both should be prefilled. */
    OcSeqBatch batch;
    oc_seq_sched_schedule(s, &batch);
    cr_assert_eq(batch.n_seqs, 2u);
    cr_assert(batch.is_prefill[0]);
    cr_assert(batch.is_prefill[1]);
    /* Second schedule: both should be decode. */
    oc_seq_sched_schedule(s, &batch);
    cr_assert_eq(batch.n_seqs, 2u);
    cr_assert(!batch.is_prefill[0]);
    cr_assert(!batch.is_prefill[1]);
    oc_seq_sched_free(s);
}


Test(seq_scheduler, append_token_transitions_running)
{
    OcSeqScheduler *s = NULL;
    OcSeqSchedulerConfig c = oc_seq_sched_config_default();
    c.max_batch_size = 4;
    c.max_total_tokens = 1024;
    oc_seq_sched_init(&s, c, NULL);
    OcSeqRequest r = make_request(5, one_prompt, 1, 5, 0.0f);
    oc_seq_sched_add(s, &r);
    /* Wait -> Running via append_token (without schedule). */
    cr_assert_eq(oc_seq_sched_append_token(s, 5, 100), OC_OK);
    cr_assert_eq(oc_seq_sched_running_count(s), 1u);
    oc_seq_sched_free(s);
}

Test(seq_scheduler, append_token_auto_finish)
{
    OcSeqScheduler *s = NULL;
    OcSeqSchedulerConfig c = oc_seq_sched_config_default();
    c.max_batch_size = 4;
    c.max_total_tokens = 1024;
    oc_seq_sched_init(&s, c, NULL);
    OcSeqRequest r = make_request(5, one_prompt, 1, 2, 0.0f);
    oc_seq_sched_add(s, &r);
    oc_seq_sched_append_token(s, 5, 100);
    cr_assert_eq(oc_seq_sched_append_token(s, 5, 101), OC_OK);
    /* After 2 generated tokens (max_tokens=2), should be finished. */
    cr_assert_eq(oc_seq_sched_running_count(s), 0u);
    oc_seq_sched_free(s);
}

Test(seq_scheduler, append_token_unknown_id)
{
    OcSeqScheduler *s = NULL;
    OcSeqSchedulerConfig c = oc_seq_sched_config_default();
    c.max_batch_size = 4;
    oc_seq_sched_init(&s, c, NULL);
    cr_assert_eq(oc_seq_sched_append_token(s, 999, 1), OC_ERR_INVALID_ARG);
    oc_seq_sched_free(s);
}


Test(seq_scheduler, finish_request)
{
    OcSeqScheduler *s = NULL;
    OcSeqSchedulerConfig c = oc_seq_sched_config_default();
    c.max_batch_size = 4;
    c.max_total_tokens = 1024;
    oc_seq_sched_init(&s, c, NULL);
    OcSeqRequest r = make_request(5, one_prompt, 1, 5, 0.0f);
    oc_seq_sched_add(s, &r);
    /* Admit it (prefill). */
    OcSeqBatch batch;
    oc_seq_sched_schedule(s, &batch);
    cr_assert_eq(oc_seq_sched_running_count(s), 1u);
    /* Finish it. */
    cr_assert_eq(oc_seq_sched_finish(s, 5), OC_OK);
    cr_assert_eq(oc_seq_sched_running_count(s), 0u);
    oc_seq_sched_free(s);
}

Test(seq_scheduler, finish_idempotent)
{
    OcSeqScheduler *s = NULL;
    OcSeqSchedulerConfig c = oc_seq_sched_config_default();
    c.max_batch_size = 4;
    c.max_total_tokens = 1024;
    oc_seq_sched_init(&s, c, NULL);
    OcSeqRequest r = make_request(5, one_prompt, 1, 5, 0.0f);
    oc_seq_sched_add(s, &r);
    OcSeqBatch batch;
    oc_seq_sched_schedule(s, &batch);
    cr_assert_eq(oc_seq_sched_finish(s, 5), OC_OK);
    /* Second finish should be OK (idempotent). */
    cr_assert_eq(oc_seq_sched_finish(s, 5), OC_OK);
    oc_seq_sched_free(s);
}

Test(seq_scheduler, abort_request)
{
    OcSeqScheduler *s = NULL;
    OcSeqSchedulerConfig c = oc_seq_sched_config_default();
    c.max_batch_size = 4;
    c.max_total_tokens = 1024;
    oc_seq_sched_init(&s, c, NULL);
    OcSeqRequest r = make_request(5, one_prompt, 1, 5, 0.0f);
    oc_seq_sched_add(s, &r);
    OcSeqBatch batch;
    oc_seq_sched_schedule(s, &batch);
    cr_assert_eq(oc_seq_sched_running_count(s), 1u);
    cr_assert_eq(oc_seq_sched_abort(s, 5), OC_OK);
    cr_assert_eq(oc_seq_sched_running_count(s), 0u);
    oc_seq_sched_free(s);
}

Test(seq_scheduler, abort_unknown_id)
{
    OcSeqScheduler *s = NULL;
    OcSeqSchedulerConfig c = oc_seq_sched_config_default();
    c.max_batch_size = 4;
    oc_seq_sched_init(&s, c, NULL);
    cr_assert_eq(oc_seq_sched_abort(s, 999), OC_ERR_INVALID_ARG);
    oc_seq_sched_free(s);
}

Test(seq_scheduler, aborted_excluded_from_schedule)
{
    OcSeqScheduler *s = NULL;
    OcSeqSchedulerConfig c = oc_seq_sched_config_default();
    c.max_batch_size = 4;
    c.max_total_tokens = 1024;
    oc_seq_sched_init(&s, c, NULL);
    OcSeqRequest r = make_request(5, one_prompt, 1, 5, 0.0f);
    oc_seq_sched_add(s, &r);
    oc_seq_sched_abort(s, 5);
    OcSeqBatch batch;
    oc_seq_sched_schedule(s, &batch);
    cr_assert_eq(batch.n_seqs, 0u);
    oc_seq_sched_free(s);
}


Test(seq_scheduler, can_fit_within_capacity)
{
    OcSeqScheduler *s = NULL;
    OcSeqSchedulerConfig c = oc_seq_sched_config_default();
    c.max_batch_size = 4;
    c.max_total_tokens = 100;
    c.water_level = 0.8f;
    oc_seq_sched_init(&s, c, NULL);
    cr_assert(oc_seq_sched_can_fit(s, 80));
    cr_assert(oc_seq_sched_can_fit(s, 50));
    oc_seq_sched_free(s);
}

Test(seq_scheduler, can_fit_exceeds_capacity)
{
    OcSeqScheduler *s = NULL;
    OcSeqSchedulerConfig c = oc_seq_sched_config_default();
    c.max_batch_size = 4;
    c.max_total_tokens = 100;
    c.water_level = 0.8f;
    oc_seq_sched_init(&s, c, NULL);
    cr_assert(!oc_seq_sched_can_fit(s, 81));
    oc_seq_sched_free(s);
}

Test(seq_scheduler, schedule_blocks_when_full)
{
    OcSeqScheduler *s = NULL;
    OcSeqSchedulerConfig c = oc_seq_sched_config_default();
    c.max_batch_size = 4;
    c.max_total_tokens = 10;
    c.water_level = 0.8f;
    oc_seq_sched_init(&s, c, NULL);
    uint32_t prompt9[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    OcSeqRequest r = make_request(1, prompt9, 9, 5, 0.0f);
    oc_seq_sched_add(s, &r);
    OcSeqBatch batch;
    oc_seq_sched_schedule(s, &batch);
    /* The prompt is too large to fit, so it should stay WAITING. */
    cr_assert_eq(batch.n_seqs, 0u);
    cr_assert_eq(oc_seq_sched_waiting_count(s), 1u);
    oc_seq_sched_free(s);
}

Test(seq_scheduler, schedule_with_page_manager)
{
    OcSeqScheduler *s = NULL;
    OcSeqSchedulerConfig c = oc_seq_sched_config_default();
    c.max_batch_size = 4;
    c.max_total_tokens = 1024;
    /* Build a small page manager. */
    OcKvPageConfig pc = oc_kv_page_config_default();
    pc.page_size = 4;
    pc.head_dim = 2;
    pc.n_heads = 2;
    pc.n_layers = 2;
    pc.max_pages = 16;
    OcKvPageManager *mgr = NULL;
    oc_kv_page_init(&mgr, pc);
    oc_seq_sched_init(&s, c, mgr);
    uint32_t prompt6[] = {1, 2, 3, 4, 5, 6};
    OcSeqRequest r = make_request(1, prompt6, 6, 3, 0.0f);
    oc_seq_sched_add(s, &r);
    OcSeqBatch batch;
    oc_seq_sched_schedule(s, &batch);
    /* The sequence should have been admitted (prefill). */
    cr_assert_eq(batch.n_seqs, 1u);
    cr_assert(batch.is_prefill[0]);
    cr_assert_eq(oc_seq_sched_running_count(s), 1u);
    /* With page_size=4 and 6 prompt tokens, we need 2 pages. */
    /* Verify pages were allocated by checking the page manager stats. */
    OcKvPageStats stats;
    oc_kv_page_stats(mgr, &stats);
    cr_assert(stats.used_pages >= 2u);
    oc_seq_sched_free(s);
    oc_kv_page_free_mgr(mgr);
}

Test(seq_scheduler, running_and_waiting_counts)
{
    OcSeqScheduler *s = NULL;
    OcSeqSchedulerConfig c = oc_seq_sched_config_default();
    c.max_batch_size = 4;
    c.max_total_tokens = 1024;
    oc_seq_sched_init(&s, c, NULL);
    OcSeqRequest r1 = make_request(1, one_prompt, 1, 5, 0.0f);
    OcSeqRequest r2 = make_request(2, one_prompt, 1, 5, 0.0f);
    oc_seq_sched_add(s, &r1);
    oc_seq_sched_add(s, &r2);
    cr_assert_eq(oc_seq_sched_waiting_count(s), 2u);
    cr_assert_eq(oc_seq_sched_running_count(s), 0u);
    OcSeqBatch batch;
    oc_seq_sched_schedule(s, &batch);
    cr_assert_eq(oc_seq_sched_waiting_count(s), 0u);
    cr_assert_eq(oc_seq_sched_running_count(s), 2u);
    oc_seq_sched_free(s);
}
