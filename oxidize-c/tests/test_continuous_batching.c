/* test_continuous_batching.c — continuous batching scheduler tests. */
#define _POSIX_C_SOURCE 200809L
#include <criterion/criterion.h>
#include "oxidize/continuous_batching.h"
#include <string.h>
#include <sys/types.h>

/* Helper: build a request with given id, prompt, and max_tokens. */
static OcBatchRequest make_request(uint64_t id, const uint32_t *prompt,
                                    size_t n_prompt, size_t max_tokens,
                                    float temperature)
{
    OcBatchRequest r;
    r.id = id;
    r.prompt_tokens = prompt;
    r.n_prompt = n_prompt;
    r.max_tokens = max_tokens;
    r.temperature = temperature;
    return r;
}

Test(continuous_batching, config_default)
{
    OcBatchConfig c = oc_batch_config_default();
    cr_assert_eq(c.max_batch_size, OC_BATCH_DEFAULT_MAX_BATCH_SIZE);
    cr_assert_eq(c.max_seq_len, OC_BATCH_DEFAULT_MAX_SEQ_LEN);
    cr_assert_eq(c.scheduling_strategy, OC_BATCH_FCFS);
}

Test(continuous_batching, init_free)
{
    OcBatchScheduler *s = NULL;
    OcBatchConfig c = oc_batch_config_default();
    c.max_batch_size = 4;
    cr_assert_eq(oc_batch_scheduler_init(&s, c), OC_OK);
    cr_assert_not_null(s);
    cr_assert_eq(s->slots_used, 0u);
    oc_batch_scheduler_free(s);
}

Test(continuous_batching, init_invalid_config)
{
    OcBatchScheduler *s = NULL;
    OcBatchConfig c = oc_batch_config_default();
    c.max_batch_size = 0;
    cr_assert_eq(oc_batch_scheduler_init(&s, c), OC_ERR_INVALID_ARG);
}

Test(continuous_batching, add_request)
{
    OcBatchScheduler *s = NULL;
    OcBatchConfig c = oc_batch_config_default();
    c.max_batch_size = 4;
    oc_batch_scheduler_init(&s, c);
    uint32_t prompt[] = {1, 2, 3};
    OcBatchRequest r = make_request(42, prompt, 3, 10, 0.8f);
    cr_assert_eq(oc_batch_scheduler_add(s, &r), OC_OK);
    cr_assert_eq(s->slots_used, 1u);
    cr_assert_eq(s->total_requests, 1u);
    oc_batch_scheduler_free(s);
}

Test(continuous_batching, add_null_request)
{
    OcBatchScheduler *s = NULL;
    OcBatchConfig c = oc_batch_config_default();
    c.max_batch_size = 4;
    oc_batch_scheduler_init(&s, c);
    cr_assert_eq(oc_batch_scheduler_add(s, NULL), OC_ERR_INVALID_ARG);
    oc_batch_scheduler_free(s);
}

Test(continuous_batching, add_duplicate_id)
{
    OcBatchScheduler *s = NULL;
    OcBatchConfig c = oc_batch_config_default();
    c.max_batch_size = 4;
    oc_batch_scheduler_init(&s, c);
    uint32_t prompt[] = {1};
    OcBatchRequest r = make_request(7, prompt, 1, 5, 0.0f);
    cr_assert_eq(oc_batch_scheduler_add(s, &r), OC_OK);
    cr_assert_eq(oc_batch_scheduler_add(s, &r), OC_ERR_INVALID_ARG);
    oc_batch_scheduler_free(s);
}

Test(continuous_batching, add_when_full)
{
    OcBatchScheduler *s = NULL;
    OcBatchConfig c = oc_batch_config_default();
    c.max_batch_size = 2;
    oc_batch_scheduler_init(&s, c);
    uint32_t prompt[] = {1};
    OcBatchRequest r1 = make_request(1, prompt, 1, 5, 0.0f);
    OcBatchRequest r2 = make_request(2, prompt, 1, 5, 0.0f);
    OcBatchRequest r3 = make_request(3, prompt, 1, 5, 0.0f);
    cr_assert_eq(oc_batch_scheduler_add(s, &r1), OC_OK);
    cr_assert_eq(oc_batch_scheduler_add(s, &r2), OC_OK);
    cr_assert_eq(oc_batch_scheduler_add(s, &r3), OC_ERR_OOM);
    oc_batch_scheduler_free(s);
}

Test(continuous_batching, next_batch_empty)
{
    OcBatchScheduler *s = NULL;
    OcBatchConfig c = oc_batch_config_default();
    c.max_batch_size = 4;
    oc_batch_scheduler_init(&s, c);
    OcBatchSlot *slots[4];
    size_t count = 999;
    cr_assert_eq(oc_batch_scheduler_next_batch(s, slots, 4, &count), OC_OK);
    cr_assert_eq(count, 0u);
    oc_batch_scheduler_free(s);
}

Test(continuous_batching, next_batch_fcfs_order)
{
    OcBatchScheduler *s = NULL;
    OcBatchConfig c = oc_batch_config_default();
    c.max_batch_size = 4;
    c.scheduling_strategy = OC_BATCH_FCFS;
    oc_batch_scheduler_init(&s, c);
    uint32_t prompt[] = {1};
    OcBatchRequest r1 = make_request(10, prompt, 1, 5, 0.0f);
    OcBatchRequest r2 = make_request(20, prompt, 1, 5, 0.0f);
    cr_assert_eq(oc_batch_scheduler_add(s, &r1), OC_OK);
    cr_assert_eq(oc_batch_scheduler_add(s, &r2), OC_OK);
    OcBatchSlot *slots[4];
    size_t count = 0;
    oc_batch_scheduler_next_batch(s, slots, 4, &count);
    cr_assert_eq(count, 2u);
    cr_assert_eq(slots[0]->request_id, 10u);
    cr_assert_eq(slots[1]->request_id, 20u);
    oc_batch_scheduler_free(s);
}

Test(continuous_batching, next_batch_sjf_order)
{
    OcBatchScheduler *s = NULL;
    OcBatchConfig c = oc_batch_config_default();
    c.max_batch_size = 4;
    c.scheduling_strategy = OC_BATCH_SHORTEST_JOB_FIRST;
    oc_batch_scheduler_init(&s, c);
    uint32_t prompt[] = {1};
    /* r1 has larger max_tokens (10), r2 smaller (3). */
    OcBatchRequest r1 = make_request(10, prompt, 1, 10, 0.0f);
    OcBatchRequest r2 = make_request(20, prompt, 1, 3, 0.0f);
    cr_assert_eq(oc_batch_scheduler_add(s, &r1), OC_OK);
    cr_assert_eq(oc_batch_scheduler_add(s, &r2), OC_OK);
    OcBatchSlot *slots[4];
    size_t count = 0;
    oc_batch_scheduler_next_batch(s, slots, 4, &count);
    cr_assert_eq(count, 2u);
    cr_assert_eq(slots[0]->request_id, 20u);
    cr_assert_eq(slots[1]->request_id, 10u);
    oc_batch_scheduler_free(s);
}

Test(continuous_batching, update_token_transitions_running)
{
    OcBatchScheduler *s = NULL;
    OcBatchConfig c = oc_batch_config_default();
    c.max_batch_size = 4;
    oc_batch_scheduler_init(&s, c);
    uint32_t prompt[] = {1};
    OcBatchRequest r = make_request(5, prompt, 1, 3, 0.0f);
    oc_batch_scheduler_add(s, &r);
    cr_assert_eq(oc_batch_scheduler_update_token(s, 5, 100), OC_OK);
    ssize_t idx = -1;
    for (uint32_t i = 0; i < s->config.max_batch_size; i++) {
        if (s->slots[i] && s->slots[i]->request_id == 5) { idx = i; break; }
    }
    cr_assert(idx >= 0);
    cr_assert_eq(s->slots[idx]->state, OC_BATCH_SLOT_RUNNING);
    cr_assert_eq(s->slots[idx]->n_generated, 1u);
    cr_assert_eq(s->slots[idx]->tokens[0], 100u);
    oc_batch_scheduler_free(s);
}

Test(continuous_batching, update_token_auto_complete)
{
    OcBatchScheduler *s = NULL;
    OcBatchConfig c = oc_batch_config_default();
    c.max_batch_size = 4;
    oc_batch_scheduler_init(&s, c);
    uint32_t prompt[] = {1};
    OcBatchRequest r = make_request(5, prompt, 1, 2, 0.0f);
    oc_batch_scheduler_add(s, &r);
    oc_batch_scheduler_update_token(s, 5, 100);
    cr_assert_eq(oc_batch_scheduler_update_token(s, 5, 101), OC_OK);
    ssize_t idx = -1;
    for (uint32_t i = 0; i < s->config.max_batch_size; i++) {
        if (s->slots[i] && s->slots[i]->request_id == 5) { idx = i; break; }
    }
    cr_assert(idx >= 0);
    cr_assert_eq(s->slots[idx]->state, OC_BATCH_SLOT_COMPLETED);
    cr_assert_eq(s->slots[idx]->n_generated, 2u);
    oc_batch_scheduler_free(s);
}

Test(continuous_batching, update_token_unknown_id)
{
    OcBatchScheduler *s = NULL;
    OcBatchConfig c = oc_batch_config_default();
    c.max_batch_size = 4;
    oc_batch_scheduler_init(&s, c);
    cr_assert_eq(oc_batch_scheduler_update_token(s, 999, 1), OC_ERR_INVALID_ARG);
    oc_batch_scheduler_free(s);
}

Test(continuous_batching, complete_request)
{
    OcBatchScheduler *s = NULL;
    OcBatchConfig c = oc_batch_config_default();
    c.max_batch_size = 4;
    oc_batch_scheduler_init(&s, c);
    uint32_t prompt[] = {1};
    OcBatchRequest r = make_request(5, prompt, 1, 5, 0.0f);
    oc_batch_scheduler_add(s, &r);
    cr_assert_eq(oc_batch_scheduler_complete(s, 5), OC_OK);
    ssize_t idx = -1;
    for (uint32_t i = 0; i < s->config.max_batch_size; i++) {
        if (s->slots[i] && s->slots[i]->request_id == 5) { idx = i; break; }
    }
    cr_assert(idx >= 0);
    cr_assert_eq(s->slots[idx]->state, OC_BATCH_SLOT_COMPLETED);
    cr_assert_eq(s->completed_requests, 1u);
    oc_batch_scheduler_free(s);
}

Test(continuous_batching, complete_unknown_id)
{
    OcBatchScheduler *s = NULL;
    OcBatchConfig c = oc_batch_config_default();
    c.max_batch_size = 4;
    oc_batch_scheduler_init(&s, c);
    cr_assert_eq(oc_batch_scheduler_complete(s, 999), OC_ERR_INVALID_ARG);
    oc_batch_scheduler_free(s);
}

Test(continuous_batching, abort_request)
{
    OcBatchScheduler *s = NULL;
    OcBatchConfig c = oc_batch_config_default();
    c.max_batch_size = 4;
    oc_batch_scheduler_init(&s, c);
    uint32_t prompt[] = {1};
    OcBatchRequest r = make_request(5, prompt, 1, 5, 0.0f);
    oc_batch_scheduler_add(s, &r);
    cr_assert_eq(oc_batch_scheduler_abort(s, 5), OC_OK);
    ssize_t idx = -1;
    for (uint32_t i = 0; i < s->config.max_batch_size; i++) {
        if (s->slots[i] && s->slots[i]->request_id == 5) { idx = i; break; }
    }
    cr_assert(idx >= 0);
    cr_assert_eq(s->slots[idx]->state, OC_BATCH_SLOT_ABORTED);
    cr_assert_eq(s->aborted_requests, 1u);
    oc_batch_scheduler_free(s);
}

Test(continuous_batching, aborted_excluded_from_next_batch)
{
    OcBatchScheduler *s = NULL;
    OcBatchConfig c = oc_batch_config_default();
    c.max_batch_size = 4;
    oc_batch_scheduler_init(&s, c);
    uint32_t prompt[] = {1};
    OcBatchRequest r = make_request(5, prompt, 1, 5, 0.0f);
    oc_batch_scheduler_add(s, &r);
    oc_batch_scheduler_abort(s, 5);
    OcBatchSlot *slots[4];
    size_t count = 0;
    oc_batch_scheduler_next_batch(s, slots, 4, &count);
    cr_assert_eq(count, 0u);
    oc_batch_scheduler_free(s);
}

Test(continuous_batching, stats)
{
    OcBatchScheduler *s = NULL;
    OcBatchConfig c = oc_batch_config_default();
    c.max_batch_size = 4;
    oc_batch_scheduler_init(&s, c);
    uint32_t prompt[] = {1};
    OcBatchRequest r1 = make_request(1, prompt, 1, 2, 0.0f);
    OcBatchRequest r2 = make_request(2, prompt, 1, 5, 0.0f);
    oc_batch_scheduler_add(s, &r1);
    oc_batch_scheduler_add(s, &r2);
    oc_batch_scheduler_update_token(s, 1, 10);
    oc_batch_scheduler_update_token(s, 1, 11);  /* auto-complete r1 */
    oc_batch_scheduler_abort(s, 2);
    OcBatchStats stats;
    cr_assert_eq(oc_batch_scheduler_stats(s, &stats), OC_OK);
    cr_assert_eq(stats.total_requests, 2u);
    cr_assert_eq(stats.completed_requests, 1u);
    cr_assert_eq(stats.aborted_requests, 1u);
    cr_assert(stats.avg_latency_ticks > 0.0);
    cr_assert(stats.throughput_tok_per_sec > 0.0);
    oc_batch_scheduler_free(s);
}
