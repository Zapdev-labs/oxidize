/* test_scheduler.c — request scheduler tests. */
#define _POSIX_C_SOURCE 200809L
#include <criterion/criterion.h>
#include <string.h>
#include "oxidize/scheduler.h"


static OcSchedRequest make_request(uint64_t id, const uint32_t *prompt,
                                    uint32_t n_prompt, uint32_t max_tokens,
                                    OcSchedPriority priority, uint64_t created_ms)
{
    OcSchedRequest r;
    oc_sched_request_init(&r, id, prompt, n_prompt, max_tokens, priority,
                           created_ms);
    return r;
}

static uint32_t one_prompt[] = {1};


Test(scheduler, config_init_defaults)
{
    OcSchedConfig cfg;
    cr_assert_eq(oc_sched_config_init(&cfg), OC_OK);
    cr_assert_eq(cfg.max_batch_size, OC_SCHED_DEFAULT_MAX_BATCH_SIZE);
    cr_assert_eq(cfg.max_tokens_total, OC_SCHED_DEFAULT_MAX_TOKENS_TOTAL);
    cr_assert_eq(cfg.preempt_mode, OC_SCHED_PREEMPT_RECOMPUTE);
    cr_assert(cfg.enable_continuous_batching);
}

Test(scheduler, config_init_null)
{
    cr_assert_eq(oc_sched_config_init(NULL), OC_ERR_INVALID_ARG);
}


Test(scheduler, init_free)
{
    OcScheduler sched;
    cr_assert_eq(oc_sched_init(&sched, NULL), OC_OK);
    cr_assert_eq(sched.n_requests, 0u);
    cr_assert_eq(sched.n_running, 0u);
    cr_assert_eq(sched.n_completed, 0u);
    oc_sched_free(&sched);
}

Test(scheduler, init_with_custom_config)
{
    OcSchedConfig cfg;
    oc_sched_config_init(&cfg);
    cfg.max_batch_size = 4;
    cfg.max_tokens_total = 1024;
    OcScheduler sched;
    cr_assert_eq(oc_sched_init(&sched, &cfg), OC_OK);
    cr_assert_eq(sched.config.max_batch_size, 4u);
    cr_assert_eq(sched.config.max_tokens_total, 1024u);
    oc_sched_free(&sched);
}

Test(scheduler, init_rejects_zero_batch_size)
{
    OcSchedConfig cfg;
    oc_sched_config_init(&cfg);
    cfg.max_batch_size = 0;
    OcScheduler sched;
    cr_assert_eq(oc_sched_init(&sched, &cfg), OC_ERR_INVALID_ARG);
}

Test(scheduler, init_rejects_zero_tokens_total)
{
    OcSchedConfig cfg;
    oc_sched_config_init(&cfg);
    cfg.max_tokens_total = 0;
    OcScheduler sched;
    cr_assert_eq(oc_sched_init(&sched, &cfg), OC_ERR_INVALID_ARG);
}

Test(scheduler, init_null_sched)
{
    cr_assert_eq(oc_sched_init(NULL, NULL), OC_ERR_INVALID_ARG);
}

Test(scheduler, free_null_is_safe)
{
    oc_sched_free(NULL);
    cr_assert(true);
}


Test(scheduler, request_init_copies_tokens)
{
    uint32_t prompt[] = {10, 20, 30};
    OcSchedRequest r;
    cr_assert_eq(oc_sched_request_init(&r, 42, prompt, 3, 100,
                                       OC_SCHED_PRIORITY_NORMAL, 1000), OC_OK);
    cr_assert_eq(r.id, 42u);
    cr_assert_eq(r.n_prompt, 3u);
    cr_assert_eq(r.max_tokens, 100u);
    cr_assert_eq(r.priority, OC_SCHED_PRIORITY_NORMAL);
    cr_assert_eq(r.status, OC_SCHED_STATUS_PENDING);
    cr_assert_not_null(r.prompt_tokens);
    cr_assert_neq(r.prompt_tokens, prompt);
    cr_assert_eq(r.prompt_tokens[0], 10u);
    cr_assert_eq(r.prompt_tokens[2], 30u);
    oc_sched_request_free(&r);
}

Test(scheduler, request_init_null_prompt_zero_ok)
{
    OcSchedRequest r;
    cr_assert_eq(oc_sched_request_init(&r, 1, NULL, 0, 50,
                                       OC_SCHED_PRIORITY_LOW, 0), OC_OK);
    cr_assert_null(r.prompt_tokens);
    oc_sched_request_free(&r);
}

Test(scheduler, request_init_null_prompt_nonzero_fails)
{
    OcSchedRequest r;
    cr_assert_eq(oc_sched_request_init(&r, 1, NULL, 5, 50,
                                       OC_SCHED_PRIORITY_LOW, 0),
                 OC_ERR_INVALID_ARG);
}


Test(scheduler, add_request_returns_id)
{
    OcScheduler sched;
    oc_sched_init(&sched, NULL);
    OcSchedRequest r = make_request(0, one_prompt, 1, 10,
                                    OC_SCHED_PRIORITY_NORMAL, 100);
    uint64_t id = 0;
    cr_assert_eq(oc_sched_add_request(&sched, &r, &id), OC_OK);
    cr_assert_eq(id, 1u);
    cr_assert_eq(sched.n_requests, 1u);
    cr_assert_eq(oc_sched_n_pending(&sched), 1u);
    oc_sched_request_free(&r);
    oc_sched_free(&sched);
}

Test(scheduler, add_null_request)
{
    OcScheduler sched;
    oc_sched_init(&sched, NULL);
    uint64_t id;
    cr_assert_eq(oc_sched_add_request(&sched, NULL, &id), OC_ERR_INVALID_ARG);
    oc_sched_free(&sched);
}

Test(scheduler, add_multiple_monotonic_ids)
{
    OcScheduler sched;
    oc_sched_init(&sched, NULL);
    OcSchedRequest r1 = make_request(0, one_prompt, 1, 5,
                                     OC_SCHED_PRIORITY_NORMAL, 100);
    OcSchedRequest r2 = make_request(0, one_prompt, 1, 5,
                                     OC_SCHED_PRIORITY_NORMAL, 101);
    uint64_t id1, id2;
    oc_sched_add_request(&sched, &r1, &id1);
    oc_sched_add_request(&sched, &r2, &id2);
    cr_assert_eq(id1, 1u);
    cr_assert_eq(id2, 2u);
    oc_sched_request_free(&r1);
    oc_sched_request_free(&r2);
    oc_sched_free(&sched);
}


Test(scheduler, next_batch_empty)
{
    OcScheduler sched;
    oc_sched_init(&sched, NULL);
    uint64_t ids[8];
    uint32_t n = 0;
    cr_assert_eq(oc_sched_next_batch(&sched, ids, 8, &n), OC_OK);
    cr_assert_eq(n, 0u);
    oc_sched_free(&sched);
}

Test(scheduler, next_batch_admits_pending)
{
    OcScheduler sched;
    oc_sched_init(&sched, NULL);
    OcSchedRequest r = make_request(0, one_prompt, 1, 5,
                                     OC_SCHED_PRIORITY_NORMAL, 100);
    uint64_t id;
    oc_sched_add_request(&sched, &r, &id);
    uint64_t ids[8];
    uint32_t n = 0;
    oc_sched_next_batch(&sched, ids, 8, &n);
    cr_assert_eq(n, 1u);
    cr_assert_eq(ids[0], id);
    cr_assert_eq(oc_sched_n_running(&sched), 1u);
    cr_assert_eq(oc_sched_n_pending(&sched), 0u);
    oc_sched_request_free(&r);
    oc_sched_free(&sched);
}

Test(scheduler, next_batch_respects_max_batch)
{
    OcSchedConfig cfg;
    oc_sched_config_init(&cfg);
    cfg.max_batch_size = 32;
    cfg.max_tokens_total = 8192;
    OcScheduler sched;
    oc_sched_init(&sched, &cfg);
    /* Add 5 requests. */
    for (int i = 0; i < 5; i++) {
        OcSchedRequest r = make_request(0, one_prompt, 1, 5,
                                         OC_SCHED_PRIORITY_NORMAL, 100 + i);
        uint64_t id;
        oc_sched_add_request(&sched, &r, &id);
        oc_sched_request_free(&r);
    }
    /* Request only 3. */
    uint64_t ids[3];
    uint32_t n = 0;
    oc_sched_next_batch(&sched, ids, 3, &n);
    cr_assert_eq(n, 3u);
    cr_assert_eq(oc_sched_n_running(&sched), 3u);
    cr_assert_eq(oc_sched_n_pending(&sched), 2u);
    oc_sched_free(&sched);
}

Test(scheduler, next_batch_priority_order)
{
    OcScheduler sched;
    oc_sched_init(&sched, NULL);
    /* Add: low first (created_ms earlier), then high. */
    OcSchedRequest r_low = make_request(0, one_prompt, 1, 5,
                                        OC_SCHED_PRIORITY_LOW, 100);
    OcSchedRequest r_high = make_request(0, one_prompt, 1, 5,
                                          OC_SCHED_PRIORITY_HIGH, 101);
    uint64_t id_low, id_high;
    oc_sched_add_request(&sched, &r_low, &id_low);
    oc_sched_add_request(&sched, &r_high, &id_high);
    uint64_t ids[4];
    uint32_t n = 0;
    oc_sched_next_batch(&sched, ids, 4, &n);
    cr_assert_eq(n, 2u);
    /* High priority should be first. */
    cr_assert_eq(ids[0], id_high);
    cr_assert_eq(ids[1], id_low);
    oc_sched_request_free(&r_low);
    oc_sched_request_free(&r_high);
    oc_sched_free(&sched);
}

Test(scheduler, next_batch_fifo_same_priority)
{
    OcScheduler sched;
    oc_sched_init(&sched, NULL);
    OcSchedRequest r1 = make_request(0, one_prompt, 1, 5,
                                     OC_SCHED_PRIORITY_NORMAL, 100);
    OcSchedRequest r2 = make_request(0, one_prompt, 1, 5,
                                     OC_SCHED_PRIORITY_NORMAL, 50);
    uint64_t id1, id2;
    oc_sched_add_request(&sched, &r1, &id1);
    oc_sched_add_request(&sched, &r2, &id2);
    uint64_t ids[4];
    uint32_t n = 0;
    oc_sched_next_batch(&sched, ids, 4, &n);
    cr_assert_eq(n, 2u);
    cr_assert_eq(ids[0], id2);
    cr_assert_eq(ids[1], id1);
    oc_sched_request_free(&r1);
    oc_sched_request_free(&r2);
    oc_sched_free(&sched);
}


Test(scheduler, complete_request)
{
    OcScheduler sched;
    oc_sched_init(&sched, NULL);
    OcSchedRequest r = make_request(0, one_prompt, 1, 5,
                                     OC_SCHED_PRIORITY_NORMAL, 100);
    uint64_t id;
    oc_sched_add_request(&sched, &r, &id);
    uint64_t ids[4];
    uint32_t n;
    oc_sched_next_batch(&sched, ids, 4, &n);
    cr_assert_eq(oc_sched_complete_request(&sched, id), OC_OK);
    cr_assert_eq(oc_sched_n_running(&sched), 0u);
    cr_assert_eq(oc_sched_n_completed(&sched), 1u);
    oc_sched_request_free(&r);
    oc_sched_free(&sched);
}

Test(scheduler, complete_unknown_id)
{
    OcScheduler sched;
    oc_sched_init(&sched, NULL);
    cr_assert_eq(oc_sched_complete_request(&sched, 999), OC_ERR_INVALID_ARG);
    oc_sched_free(&sched);
}

Test(scheduler, cancel_request)
{
    OcScheduler sched;
    oc_sched_init(&sched, NULL);
    OcSchedRequest r = make_request(0, one_prompt, 1, 5,
                                     OC_SCHED_PRIORITY_NORMAL, 100);
    uint64_t id;
    oc_sched_add_request(&sched, &r, &id);
    uint64_t ids[4];
    uint32_t n;
    oc_sched_next_batch(&sched, ids, 4, &n);
    cr_assert_eq(oc_sched_cancel_request(&sched, id), OC_OK);
    cr_assert_eq(oc_sched_n_running(&sched), 0u);
    oc_sched_request_free(&r);
    oc_sched_free(&sched);
}

Test(scheduler, cancel_unknown_id)
{
    OcScheduler sched;
    oc_sched_init(&sched, NULL);
    cr_assert_eq(oc_sched_cancel_request(&sched, 999), OC_ERR_INVALID_ARG);
    oc_sched_free(&sched);
}
