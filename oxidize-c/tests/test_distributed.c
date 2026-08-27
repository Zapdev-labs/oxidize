/*
 * test_distributed.c — tests for the distributed inference scheduler.
 *
 * Tests cover:
 *   - Init / free lifecycle
 *   - Config validation (valid + invalid cases)
 *   - Role resolution
 *   - Stats initialization
 *   - Single-node operation (no communication needed)
 *   - Null / edge-case handling
 *   - Stats JSON formatting
 */
#include <criterion/criterion.h>
#include <string.h>
#include <stdio.h>

#include "oxidize/distributed.h"
#include "oxidize/error.h"

#include <pthread.h>

/* Fixed loopback port for the two-node tests (and PORT+1 for the
 * timeout test). Kept out of the ephemeral range to avoid collisions. */
#define OC_TEST_PORT      52930
#define OC_TEST_PORT_STR "52930"
#define OC_TEST_TP_PORT      52932
#define OC_TEST_TP_PORT_STR "52932"

/* ------------------------------------------------------------------ */
/* Helpers                                                            */
/* ------------------------------------------------------------------ */

static OcDistributedConfig make_single_node_config(void)
{
    OcDistributedConfig cfg = OC_DISTRIBUTED_CONFIG_DEFAULT;
    return cfg;
}

static OcDistributedConfig make_multinode_config(uint32_t n_nodes,
                                                   uint32_t rank,
                                                   uint32_t pp,
                                                   uint32_t tp)
{
    OcDistributedConfig cfg = OC_DISTRIBUTED_CONFIG_DEFAULT;
    cfg.n_nodes = n_nodes;
    cfg.node_rank = rank;
    cfg.pipeline_stages = pp;
    cfg.tensor_parallel_size = tp;
    cfg.pipeline_rank = rank / tp;       /* simple linear mapping */
    cfg.tensor_rank = rank % tp;
    cfg.activation_dtype_size = 4;
    return cfg;
}

/* ------------------------------------------------------------------ */
/* Init / Free                                                        */
/* ------------------------------------------------------------------ */

Test(distributed, init_single_node)
{
    OcDistributedScheduler sched;
    OcDistributedConfig cfg = make_single_node_config();
    cr_assert_eq(oc_distributed_init(&sched, &cfg), OC_OK);
    cr_assert(sched.initialized);
    cr_assert_eq(sched.role, OC_NODE_ROLE_PIPELINE_MASTER);
    cr_assert_eq(sched.n_peers, 1);
    cr_assert_eq(sched.peers[0].rank, 0);
    cr_assert(sched.peers[0].online);
    cr_assert_eq(sched.listen_fd, -1);  /* no listen socket in single-node */
    cr_assert_null(sched.send_buf);     /* no buffers allocated */
    cr_assert_null(sched.recv_buf);
    oc_distributed_free(&sched);
    cr_assert(!sched.initialized);
}

Test(distributed, init_and_free_idempotent)
{
    OcDistributedScheduler sched;
    OcDistributedConfig cfg = make_single_node_config();
    cr_assert_eq(oc_distributed_init(&sched, &cfg), OC_OK);
    oc_distributed_free(&sched);
    /* Free again should be safe. */
    oc_distributed_free(&sched);
    cr_assert(!sched.initialized);
}

Test(distributed, init_multi_node_pipeline)
{
    /* 4-node pipeline parallelism, this node is rank 1 (worker). */
    OcDistributedScheduler sched;
    OcDistributedConfig cfg = make_multinode_config(4, 1, 4, 1);
    /* Don't connect to a coordinator; init should still succeed. */
    cfg.coordinator_addr = NULL;
    cfg.listen_port = 0;
    OcError e = oc_distributed_init(&sched, &cfg);
    /* A worker with no coordinator address has nothing to connect to. */
    cr_assert_eq(e, OC_ERR_NETWORK,
                 "worker init without a coordinator address must fail");
    cr_assert(!sched.initialized);
    oc_distributed_free(&sched);
}

/* ------------------------------------------------------------------ */
/* Config validation                                                  */
/* ------------------------------------------------------------------ */

Test(distributed, validate_config_single_node)
{
    OcDistributedConfig cfg = make_single_node_config();
    cr_assert_eq(oc_distributed_validate_config(&cfg), OC_OK);
}

Test(distributed, validate_config_multi_node)
{
    OcDistributedConfig cfg = make_multinode_config(8, 3, 2, 4);
    cr_assert_eq(oc_distributed_validate_config(&cfg), OC_OK);
}

Test(distributed, validate_config_reject_null)
{
    cr_assert_eq(oc_distributed_validate_config(NULL), OC_ERR_INVALID_ARG);
}

Test(distributed, validate_config_reject_zero_nodes)
{
    OcDistributedConfig cfg = make_single_node_config();
    cfg.n_nodes = 0;
    cr_assert_eq(oc_distributed_validate_config(&cfg), OC_ERR_INVALID_ARG);
}

Test(distributed, validate_config_reject_bad_rank)
{
    OcDistributedConfig cfg = make_single_node_config();
    cfg.n_nodes = 4;
    cfg.node_rank = 4; /* == n_nodes */
    cr_assert_eq(oc_distributed_validate_config(&cfg), OC_ERR_INVALID_ARG);
    cfg.node_rank = 10; /* > n_nodes */
    cr_assert_eq(oc_distributed_validate_config(&cfg), OC_ERR_INVALID_ARG);
}

Test(distributed, validate_config_reject_zero_pipeline)
{
    OcDistributedConfig cfg = make_single_node_config();
    cfg.pipeline_stages = 0;
    cr_assert_eq(oc_distributed_validate_config(&cfg), OC_ERR_INVALID_ARG);
}

Test(distributed, validate_config_reject_zero_tp)
{
    OcDistributedConfig cfg = make_single_node_config();
    cfg.tensor_parallel_size = 0;
    cr_assert_eq(oc_distributed_validate_config(&cfg), OC_ERR_INVALID_ARG);
}

Test(distributed, validate_config_reject_zero_dtype)
{
    OcDistributedConfig cfg = make_single_node_config();
    cfg.activation_dtype_size = 0;
    cr_assert_eq(oc_distributed_validate_config(&cfg), OC_ERR_INVALID_ARG);
}

Test(distributed, validate_config_reject_pp_exceeds_nodes)
{
    OcDistributedConfig cfg = make_multinode_config(2, 0, 4, 1);
    cr_assert_eq(oc_distributed_validate_config(&cfg), OC_ERR_INVALID_ARG);
}

Test(distributed, validate_config_reject_pp_tp_exceeds_nodes)
{
    OcDistributedConfig cfg = make_multinode_config(4, 0, 2, 4);
    /* pp * tp = 8 > n_nodes = 4 */
    cr_assert_eq(oc_distributed_validate_config(&cfg), OC_ERR_INVALID_ARG);
}

Test(distributed, validate_config_reject_bad_pipeline_rank)
{
    OcDistributedConfig cfg = make_multinode_config(4, 0, 2, 1);
    cfg.pipeline_rank = 2; /* >= pipeline_stages */
    cr_assert_eq(oc_distributed_validate_config(&cfg), OC_ERR_INVALID_ARG);
}

Test(distributed, validate_config_reject_bad_tensor_rank)
{
    OcDistributedConfig cfg = make_multinode_config(4, 0, 1, 2);
    cfg.tensor_rank = 2; /* >= tensor_parallel_size */
    cr_assert_eq(oc_distributed_validate_config(&cfg), OC_ERR_INVALID_ARG);
}

/* ------------------------------------------------------------------ */
/* Role resolution                                                    */
/* ------------------------------------------------------------------ */

Test(distributed, role_single_node_is_master)
{
    OcDistributedConfig cfg = make_single_node_config();
    cr_assert_eq(oc_distributed_resolve_role(&cfg),
                 OC_NODE_ROLE_PIPELINE_MASTER);
}

Test(distributed, role_pipeline_master)
{
    OcDistributedConfig cfg = make_multinode_config(4, 0, 4, 1);
    cr_assert_eq(oc_distributed_resolve_role(&cfg),
                 OC_NODE_ROLE_PIPELINE_MASTER);
}

Test(distributed, role_pipeline_worker)
{
    OcDistributedConfig cfg = make_multinode_config(4, 2, 4, 1);
    /* In pure pipeline mode (TP=1), non-zero pipeline_rank is a worker. */
    /* But our resolver marks it as PIPELINE_WORKER when TP=1 and rank>0. */
    OcNodeRole r = oc_distributed_resolve_role(&cfg);
    /* With TP=1, the resolver returns PIPELINE_WORKER for rank > 0. */
    cr_assert(r == OC_NODE_ROLE_PIPELINE_WORKER,
              "expected PIPELINE_WORKER for non-zero pipeline rank with TP=1");
}

Test(distributed, role_tensor_parallel)
{
    OcDistributedConfig cfg = make_multinode_config(4, 1, 1, 4);
    OcNodeRole r = oc_distributed_resolve_role(&cfg);
    cr_assert(r == OC_NODE_ROLE_TENSOR_PARALLEL || r == OC_NODE_ROLE_PIPELINE_MASTER,
              "expected TENSOR_PARALLEL or PIPELINE_MASTER for TP>1");
}

Test(distributed, role_null_config)
{
    cr_assert_eq(oc_distributed_resolve_role(NULL), OC_NODE_ROLE_NONE);
}

/* ------------------------------------------------------------------ */
/* Stats                                                              */
/* ------------------------------------------------------------------ */

Test(distributed, stats_init_zero)
{
    OcDistributedScheduler sched;
    OcDistributedConfig cfg = make_single_node_config();
    cr_assert_eq(oc_distributed_init(&sched, &cfg), OC_OK);

    const OcDistributedStats *st = oc_distributed_get_stats(&sched);
    cr_assert_not_null(st);
    cr_assert_eq(st->bytes_sent, 0);
    cr_assert_eq(st->bytes_received, 0);
    cr_assert_eq(st->latency_ms, 0.0);
    cr_assert_eq(st->avg_latency_ms, 0.0);
    cr_assert_eq(st->latency_samples, 0);
    cr_assert_eq(st->tokens_processed, 0);
    cr_assert_eq(st->barriers_hit, 0);
    cr_assert_eq(st->send_calls, 0);
    cr_assert_eq(st->recv_calls, 0);
    cr_assert_eq(st->allreduce_calls, 0);
    cr_assert_eq(st->reconnects, 0);
    cr_assert_eq(st->disconnects, 0);

    oc_distributed_free(&sched);
}

Test(distributed, stats_null_scheduler)
{
    cr_assert_null(oc_distributed_get_stats(NULL));
}

/* ------------------------------------------------------------------ */
/* Single-node operation (no communication needed)                    */
/* ------------------------------------------------------------------ */

Test(distributed, single_node_send_activations_noop)
{
    OcDistributedScheduler sched;
    OcDistributedConfig cfg = make_single_node_config();
    cr_assert_eq(oc_distributed_init(&sched, &cfg), OC_OK);

    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    /* Single-node: send is a no-op, returns OK. */
    cr_assert_eq(oc_distributed_send_activations(&sched, data, 4), OC_OK);

    /* Stats: send_calls incremented, but bytes_sent still 0 (no network). */
    const OcDistributedStats *st = oc_distributed_get_stats(&sched);
    cr_assert_eq(st->send_calls, 1);
    cr_assert_eq(st->bytes_sent, 0);

    oc_distributed_free(&sched);
}

Test(distributed, single_node_recv_activations_noop)
{
    OcDistributedScheduler sched;
    OcDistributedConfig cfg = make_single_node_config();
    cr_assert_eq(oc_distributed_init(&sched, &cfg), OC_OK);

    float out[4] = {0};
    /* Single-node: recv is a no-op, returns OK without writing. */
    cr_assert_eq(oc_distributed_recv_activations(&sched, out, 4), OC_OK);

    const OcDistributedStats *st = oc_distributed_get_stats(&sched);
    cr_assert_eq(st->recv_calls, 1);
    cr_assert_eq(st->bytes_received, 0);

    oc_distributed_free(&sched);
}

Test(distributed, single_node_all_reduce_noop)
{
    OcDistributedScheduler sched;
    OcDistributedConfig cfg = make_single_node_config();
    cr_assert_eq(oc_distributed_init(&sched, &cfg), OC_OK);

    float data[] = {1.0f, 2.0f, 3.0f};
    /* Single-node: all-reduce is a no-op; data unchanged. */
    cr_assert_eq(oc_distributed_all_reduce(&sched, data, 3), OC_OK);
    cr_assert_float_eq(data[0], 1.0f, 1e-6f);
    cr_assert_float_eq(data[1], 2.0f, 1e-6f);
    cr_assert_float_eq(data[2], 3.0f, 1e-6f);

    const OcDistributedStats *st = oc_distributed_get_stats(&sched);
    cr_assert_eq(st->allreduce_calls, 1);

    oc_distributed_free(&sched);
}

Test(distributed, single_node_barrier_noop)
{
    OcDistributedScheduler sched;
    OcDistributedConfig cfg = make_single_node_config();
    cr_assert_eq(oc_distributed_init(&sched, &cfg), OC_OK);

    cr_assert_eq(oc_distributed_barrier(&sched), OC_OK);

    const OcDistributedStats *st = oc_distributed_get_stats(&sched);
    cr_assert_eq(st->barriers_hit, 1);

    oc_distributed_free(&sched);
}

Test(distributed, single_node_get_latency)
{
    OcDistributedScheduler sched;
    OcDistributedConfig cfg = make_single_node_config();
    cr_assert_eq(oc_distributed_init(&sched, &cfg), OC_OK);

    double lat = oc_distributed_get_latency(&sched);
    cr_assert_geq(lat, 0.0, "latency should be non-negative");

    const OcDistributedStats *st = oc_distributed_get_stats(&sched);
    cr_assert_eq(st->latency_samples, 1);
    cr_assert_geq(st->latency_ms, 0.0);

    oc_distributed_free(&sched);
}

/* ------------------------------------------------------------------ */
/* Null handling                                                      */
/* ------------------------------------------------------------------ */

Test(distributed, null_scheduler_init)
{
    OcDistributedConfig cfg = make_single_node_config();
    cr_assert_eq(oc_distributed_init(NULL, &cfg), OC_ERR_INVALID_ARG);
}

Test(distributed, null_config_init)
{
    OcDistributedScheduler sched;
    cr_assert_eq(oc_distributed_init(&sched, NULL), OC_ERR_INVALID_ARG);
}

Test(distributed, free_null)
{
    /* Should not crash. */
    oc_distributed_free(NULL);
}

Test(distributed, send_null_data_nonzero_count)
{
    OcDistributedScheduler sched;
    OcDistributedConfig cfg = make_single_node_config();
    cr_assert_eq(oc_distributed_init(&sched, &cfg), OC_OK);
    /* count > 0 but data == NULL: invalid even in single-node. */
    cr_assert_eq(oc_distributed_send_activations(&sched, NULL, 4),
                 OC_ERR_INVALID_ARG);
    oc_distributed_free(&sched);
}

Test(distributed, send_zero_count_ok)
{
    OcDistributedScheduler sched;
    OcDistributedConfig cfg = make_single_node_config();
    cr_assert_eq(oc_distributed_init(&sched, &cfg), OC_OK);
    /* count == 0 with NULL data is fine. */
    cr_assert_eq(oc_distributed_send_activations(&sched, NULL, 0), OC_OK);
    oc_distributed_free(&sched);
}

Test(distributed, recv_null_data_nonzero_count)
{
    OcDistributedScheduler sched;
    OcDistributedConfig cfg = make_single_node_config();
    cr_assert_eq(oc_distributed_init(&sched, &cfg), OC_OK);
    cr_assert_eq(oc_distributed_recv_activations(&sched, NULL, 4),
                 OC_ERR_INVALID_ARG);
    oc_distributed_free(&sched);
}

Test(distributed, all_reduce_null_data_nonzero_count)
{
    OcDistributedScheduler sched;
    OcDistributedConfig cfg = make_single_node_config();
    cr_assert_eq(oc_distributed_init(&sched, &cfg), OC_OK);
    cr_assert_eq(oc_distributed_all_reduce(&sched, NULL, 4),
                 OC_ERR_INVALID_ARG);
    oc_distributed_free(&sched);
}

Test(distributed, barrier_null_scheduler)
{
    cr_assert_eq(oc_distributed_barrier(NULL), OC_ERR_INVALID_ARG);
}

Test(distributed, operations_on_uninit_scheduler)
{
    OcDistributedScheduler sched;
    memset(&sched, 0, sizeof(sched));
    float data[4] = {1, 2, 3, 4};
    cr_assert_eq(oc_distributed_send_activations(&sched, data, 4),
                 OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_distributed_recv_activations(&sched, data, 4),
                 OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_distributed_all_reduce(&sched, data, 4),
                 OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_distributed_barrier(&sched), OC_ERR_INVALID_ARG);
}

/* ------------------------------------------------------------------ */
/* Stats JSON format                                                  */
/* ------------------------------------------------------------------ */

Test(distributed, stats_json_basic)
{
    OcDistributedScheduler sched;
    OcDistributedConfig cfg = make_single_node_config();
    cr_assert_eq(oc_distributed_init(&sched, &cfg), OC_OK);

    /* Do some operations to populate stats. */
    float data[] = {1.0f, 2.0f, 3.0f};
    oc_distributed_send_activations(&sched, data, 3);
    oc_distributed_recv_activations(&sched, data, 3);
    oc_distributed_all_reduce(&sched, data, 3);
    oc_distributed_barrier(&sched);
    oc_distributed_get_latency(&sched);

    char buf[1024];
    size_t n = oc_distributed_stats_json(&sched, buf, sizeof(buf));
    cr_assert_gt(n, 0, "JSON output should be non-empty");
    cr_assert_lt(n, sizeof(buf), "JSON should fit in buffer");

    /* Verify JSON starts and ends with braces. */
    cr_assert_eq(buf[0], '{');
    cr_assert_eq(buf[n], '\0');  /* NUL-terminated */
    /* Find last non-null char. */
    cr_assert_eq(buf[n - 1], '}', "JSON should end with }");

    /* Verify key fields are present. */
    cr_assert_not_null(strstr(buf, "\"bytes_sent\""));
    cr_assert_not_null(strstr(buf, "\"bytes_received\""));
    cr_assert_not_null(strstr(buf, "\"latency_ms\""));
    cr_assert_not_null(strstr(buf, "\"tokens_processed\""));
    cr_assert_not_null(strstr(buf, "\"barriers_hit\""));
    cr_assert_not_null(strstr(buf, "\"send_calls\""));
    cr_assert_not_null(strstr(buf, "\"recv_calls\""));
    cr_assert_not_null(strstr(buf, "\"allreduce_calls\""));
    cr_assert_not_null(strstr(buf, "\"reconnects\""));
    cr_assert_not_null(strstr(buf, "\"disconnects\""));

    /* Verify call counts in JSON. */
    cr_assert_not_null(strstr(buf, "\"send_calls\":1"));
    cr_assert_not_null(strstr(buf, "\"recv_calls\":1"));
    cr_assert_not_null(strstr(buf, "\"allreduce_calls\":1"));
    cr_assert_not_null(strstr(buf, "\"barriers_hit\":1"));

    oc_distributed_free(&sched);
}

Test(distributed, stats_json_null_scheduler)
{
    char buf[64];
    size_t n = oc_distributed_stats_json(NULL, buf, sizeof(buf));
    cr_assert_eq(n, 2, "null scheduler should produce \"{}\"");
    cr_assert_str_eq(buf, "{}");
}

Test(distributed, stats_json_null_buffer)
{
    OcDistributedScheduler sched;
    OcDistributedConfig cfg = make_single_node_config();
    cr_assert_eq(oc_distributed_init(&sched, &cfg), OC_OK);

    /* Should return required length without writing. */
    size_t n = oc_distributed_stats_json(&sched, NULL, 0);
    cr_assert_gt(n, 0);

    oc_distributed_free(&sched);
}

Test(distributed, stats_json_small_buffer_truncates)
{
    OcDistributedScheduler sched;
    OcDistributedConfig cfg = make_single_node_config();
    cr_assert_eq(oc_distributed_init(&sched, &cfg), OC_OK);

    char buf[8];
    size_t n = oc_distributed_stats_json(&sched, buf, sizeof(buf));
    cr_assert_gt(n, sizeof(buf), "required length should exceed small buffer");
    /* Buffer should be NUL-terminated within its capacity. */
    cr_assert_eq(buf[sizeof(buf) - 1], '\0');

    oc_distributed_free(&sched);
}

/* ------------------------------------------------------------------ */
/* Reconnect                                                          */
/* ------------------------------------------------------------------ */

Test(distributed, reconnect_single_node_no_peers)
{
    OcDistributedScheduler sched;
    OcDistributedConfig cfg = make_single_node_config();
    cr_assert_eq(oc_distributed_init(&sched, &cfg), OC_OK);

    /* In single-node mode, there are no remote peers; reconnect to any
     * rank >= n_peers is invalid. */
    cr_assert_eq(oc_distributed_reconnect(&sched, 5), OC_ERR_INVALID_ARG);

    oc_distributed_free(&sched);
}

Test(distributed, reconnect_null_scheduler)
{
    cr_assert_eq(oc_distributed_reconnect(NULL, 0), OC_ERR_INVALID_ARG);
}

/* ------------------------------------------------------------------ */
/* Multi-node pipeline edge cases                                     */
/* ------------------------------------------------------------------ */

Test(distributed, multi_node_init_times_out_without_peers)
{
    /* Rank 0 listens but nobody connects: init must give up at the
     * configured deadline rather than blocking forever. */
    OcDistributedScheduler sched;
    OcDistributedConfig cfg = make_multinode_config(4, 0, 4, 1);
    cfg.coordinator_addr = NULL;
    cfg.listen_port = OC_TEST_PORT + 1;
    cfg.connect_timeout_ms = 200;
    cr_assert_eq(oc_distributed_validate_config(&cfg), OC_OK);
    cr_assert_eq(oc_distributed_init(&sched, &cfg), OC_ERR_NETWORK);
    cr_assert(!sched.initialized);
    oc_distributed_free(&sched);
}

/* Worker half of the two-node round-trip test below. */
static void *worker_thread_main(void *arg)
{
    (void)arg;
    static OcError result;
    OcDistributedScheduler sched;
    OcDistributedConfig cfg = make_multinode_config(2, 1, 2, 1);
    cfg.coordinator_addr = "127.0.0.1:" OC_TEST_PORT_STR;
    cfg.connect_timeout_ms = 5000;

    result = oc_distributed_init(&sched, &cfg);
    if (result != OC_OK) return &result;

    float buf[4] = {0};
    result = oc_distributed_recv_activations(&sched, buf, 4);
    if (result == OC_OK) {
        for (int i = 0; i < 4; i++) {
            if (buf[i] != (float)(i + 1)) result = OC_ERR_NETWORK;
        }
    }
    oc_distributed_free(&sched);
    return &result;
}

Test(distributed, multi_node_pipeline_round_trip)
{
    /* Real two-node pipeline: rank 0 accepts, rank 1 connects, and one
     * activation vector crosses the wire intact. */
    pthread_t worker;
    cr_assert_eq(pthread_create(&worker, NULL, worker_thread_main, NULL), 0);

    OcDistributedScheduler sched;
    OcDistributedConfig cfg = make_multinode_config(2, 0, 2, 1);
    cfg.listen_port = OC_TEST_PORT;
    cfg.connect_timeout_ms = 5000;

    cr_assert_eq(oc_distributed_init(&sched, &cfg), OC_OK);
    cr_assert(sched.initialized);
    cr_assert(sched.peers[1].online);

    float send[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    cr_assert_eq(oc_distributed_send_activations(&sched, send, 4), OC_OK);
    cr_assert_eq(sched.stats.bytes_sent, sizeof(send));

    void *wres = NULL;
    cr_assert_eq(pthread_join(worker, &wres), 0);
    cr_assert_not_null(wres);
    cr_assert_eq(*(OcError *)wres, OC_OK, "worker side failed");

    oc_distributed_free(&sched);
}

typedef struct TpWorkerArgs {
    uint32_t rank;
    float data[4];
    OcError result;
} TpWorkerArgs;

static void *tp_worker_thread_main(void *arg)
{
    TpWorkerArgs *worker = arg;
    OcDistributedScheduler sched;
    OcDistributedConfig cfg = make_multinode_config(3, worker->rank, 1, 3);
    cfg.coordinator_addr = "127.0.0.1:" OC_TEST_TP_PORT_STR;
    cfg.connect_timeout_ms = 5000;

    worker->result = oc_distributed_init(&sched, &cfg);
    if (worker->result == OC_OK) {
        worker->result = oc_distributed_all_reduce(&sched, worker->data, 4);
        oc_distributed_free(&sched);
    }
    return worker;
}

Test(distributed, three_node_all_reduce)
{
    TpWorkerArgs workers[2] = {
        {.rank = 1, .data = {10.0f, 20.0f, 30.0f, 40.0f},
         .result = OC_ERR_NETWORK},
        {.rank = 2, .data = {100.0f, 200.0f, 300.0f, 400.0f},
         .result = OC_ERR_NETWORK},
    };
    pthread_t threads[2];
    cr_assert_eq(pthread_create(&threads[0], NULL, tp_worker_thread_main,
                                &workers[0]), 0);
    cr_assert_eq(pthread_create(&threads[1], NULL, tp_worker_thread_main,
                                &workers[1]), 0);

    OcDistributedScheduler sched;
    OcDistributedConfig cfg = make_multinode_config(3, 0, 1, 3);
    cfg.listen_port = OC_TEST_TP_PORT;
    cfg.connect_timeout_ms = 5000;
    cr_assert_eq(oc_distributed_init(&sched, &cfg), OC_OK);

    float master[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    cr_assert_eq(oc_distributed_all_reduce(&sched, master, 4), OC_OK);

    for (size_t i = 0; i < 2; i++) {
        void *thread_result = NULL;
        cr_assert_eq(pthread_join(threads[i], &thread_result), 0);
        cr_assert_eq(thread_result, &workers[i]);
        cr_assert_eq(workers[i].result, OC_OK,
                     "tensor-parallel worker %zu failed", i + 1);
    }

    const float expected[4] = {111.0f, 222.0f, 333.0f, 444.0f};
    for (size_t i = 0; i < 4; i++) {
        cr_assert_float_eq(master[i], expected[i], 1e-6f,
                           "master all-reduce mismatch at %zu", i);
        cr_assert_float_eq(workers[0].data[i], expected[i], 1e-6f,
                           "rank 1 all-reduce mismatch at %zu", i);
        cr_assert_float_eq(workers[1].data[i], expected[i], 1e-6f,
                           "rank 2 all-reduce mismatch at %zu", i);
    }

    cr_assert_eq(sched.stats.allreduce_calls, 1);
    oc_distributed_free(&sched);
}

Test(distributed, validate_config_reject_too_many_nodes)
{
    OcDistributedConfig cfg = make_multinode_config(4, 0, 4, 1);
    cfg.n_nodes = OC_DIST_MAX_NODES + 1;
    cr_assert_eq(oc_distributed_validate_config(&cfg), OC_ERR_INVALID_ARG);
}

Test(distributed, validate_config_reject_inconsistent_rank_tuple)
{
    /* rank 3 claiming pipeline/tensor rank 0/0 must be rejected. */
    OcDistributedConfig cfg = make_multinode_config(4, 3, 4, 1);
    cfg.pipeline_rank = 0;
    cfg.tensor_rank = 0;
    cr_assert_eq(oc_distributed_validate_config(&cfg), OC_ERR_INVALID_ARG);
}
