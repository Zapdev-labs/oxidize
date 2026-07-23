/*
 * spinpool.h — Thread pool with spin-waiting for low-latency parallel compute.
 *
 * Ports the spin-pool concept from oxidize-core/src/compute/spinpool.rs.
 * Workers stay resident and spin between regions (with condvar fallback for
 * truly idle periods) so per-region handoff cost is minimal. Tasks are
 * submitted into a bounded queue and processed by the worker threads.
 */
#ifndef OXIDIZE_SPINPOOL_H
#define OXIDIZE_SPINPOOL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <pthread.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Constants ──────────────────────────────────────────────────────── */

#define OC_SP_DEFAULT_SPIN_ITERATIONS 1000u
#define OC_SP_DEFAULT_QUEUE_SIZE      256u

/* ─── Types ─────────────────────────────────────────────────────────── */

/* Configuration. Use oc_spinpool_config_default() for sensible defaults. */
typedef struct OcSpinPoolConfig {
    size_t n_threads;        /* 0 = auto (available_parallelism) */
    size_t spin_iterations;  /* spin loops before parking (1000)  */
    size_t queue_size;       /* task queue capacity (256)          */
} OcSpinPoolConfig;

/* A task: function pointer + argument + result storage. */
typedef struct OcSpinPoolTask {
    void *(*fn)(void *arg);   /* task function                      */
    void  *arg;               /* argument passed to fn              */
    void  *result;             /* result returned by fn (set async) */
} OcSpinPoolTask;

/* The thread pool. All fields are internal; use the public API. */
typedef struct OcSpinPool {
    OcSpinPoolConfig  config;
    pthread_t         *threads;
    size_t             n_threads;
    OcSpinPoolTask    *task_queue;
    size_t             queue_capacity;
    size_t             head;          /* next slot to write               */
    size_t             tail;          /* next slot to read                */
    size_t             n_pending;     /* tasks queued but not done         */
    size_t             n_active;      /* tasks currently executing          */
    bool               shutdown;
    pthread_mutex_t    lock;
    pthread_cond_t     not_empty;     /* signaled when a task is enqueued  */
    pthread_cond_t     not_full;      /* signaled when a slot frees up     */
    pthread_cond_t     idle;          /* signaled when all workers idle    */
} OcSpinPool;

/* ─── Public API ─────────────────────────────────────────────────────── */

/* Produce a default config: n_threads=0 (auto), spin=1000, queue=256. */
OcSpinPoolConfig oc_spinpool_config_default(void);

/* Create a thread pool. Returns OC_OK on success. The caller owns the
 * struct and must free it with oc_spinpool_free(). */
OcError oc_spinpool_init(OcSpinPool *pool, OcSpinPoolConfig config);

/* Submit a task. The task function `task_fn` is called with `arg` by a
 * worker thread. Returns OC_OK on success, OC_ERR_OOM if the queue is
 * full (caller should wait and retry), OC_ERR_INVALID_ARG on bad args. */
OcError oc_spinpool_submit(OcSpinPool *pool, void *(*task_fn)(void *),
                           void *arg);

/* Wait until all pending tasks complete and all workers are idle. */
OcError oc_spinpool_wait(OcSpinPool *pool);

/* Parallel map: apply `fn` to each of `n_items` items, storing results
 * into `out_results` (length n_items). Blocks until all complete. */
OcError oc_spinpool_map(OcSpinPool *pool, void *(*fn)(void *),
                        void **items, size_t n_items, void **out_results);

/* Get the number of worker threads. */
size_t oc_spinpool_n_threads(const OcSpinPool *pool);

/* Get the number of pending (queued, not yet completed) tasks. */
size_t oc_spinpool_n_pending(const OcSpinPool *pool);

/* Shutdown the pool and free all resources. Joins all worker threads. */
void oc_spinpool_free(OcSpinPool *pool);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_SPINPOOL_H */
