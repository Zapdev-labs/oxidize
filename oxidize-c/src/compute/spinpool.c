#define _POSIX_C_SOURCE 200809L
/* macOS hides _SC_NPROCESSORS_ONLN under strict _POSIX_C_SOURCE; restore it. */
#ifdef __APPLE__
#define _DARWIN_C_SOURCE 1
#endif
#include "oxidize/spinpool.h"

#include <unistd.h>
#include <stdlib.h>
#include <string.h>


static void *worker_loop(void *arg)
{
    OcSpinPool *pool = (OcSpinPool *)arg;
    for (;;) {
        pthread_mutex_lock(&pool->lock);

        /* Spin briefly before parking. */
        size_t spins = 0;
        while (pool->n_pending == 0 && !pool->shutdown) {
            if (spins < pool->config.spin_iterations) {
                spins++;
                /* Release and re-acquire to allow progress; use a short
                 * yield-style retry. */
                pthread_mutex_unlock(&pool->lock);
                /* Spin loop hint: sched_yield lets other threads run. */
                sched_yield();
                pthread_mutex_lock(&pool->lock);
                continue;
            }
            /* Signal idle before parking. */
            if (pool->n_active == 0) {
                pthread_cond_broadcast(&pool->idle);
            }
            pthread_cond_wait(&pool->not_empty, &pool->lock);
            spins = 0;
        }

        if (pool->shutdown && pool->n_pending == 0) {
            pthread_mutex_unlock(&pool->lock);
            break;
        }

        /* Dequeue one task. */
        OcSpinPoolTask task = pool->task_queue[pool->tail];
        pool->tail = (pool->tail + 1) % pool->queue_capacity;
        pool->n_pending--;
        pool->n_active++;

        /* Signal that a queue slot freed up. */
        pthread_cond_signal(&pool->not_full);
        pthread_mutex_unlock(&pool->lock);

        /* Execute the task outside the lock. */
        if (task.fn) {
            task.result = task.fn(task.arg);
        }

        pthread_mutex_lock(&pool->lock);
        pool->n_active--;
        if (pool->n_pending == 0 && pool->n_active == 0) {
            pthread_cond_broadcast(&pool->idle);
        }
        pthread_mutex_unlock(&pool->lock);
    }
    return NULL;
}


OcSpinPoolConfig oc_spinpool_config_default(void)
{
    OcSpinPoolConfig cfg;
    cfg.n_threads       = 0;  /* auto */
    cfg.spin_iterations = OC_SP_DEFAULT_SPIN_ITERATIONS;
    cfg.queue_size      = OC_SP_DEFAULT_QUEUE_SIZE;
    return cfg;
}

OcError oc_spinpool_init(OcSpinPool *pool, OcSpinPoolConfig config)
{
    if (!pool) return OC_ERR_INVALID_ARG;
    if (config.queue_size == 0) return OC_ERR_INVALID_ARG;

    memset(pool, 0, sizeof(*pool));
    pool->config          = config;
    pool->queue_capacity  = config.queue_size;
    pool->head            = 0;
    pool->tail            = 0;
    pool->n_pending       = 0;
    pool->n_active        = 0;
    pool->shutdown        = false;

    /* Determine thread count. */
    size_t n_threads = config.n_threads;
    if (n_threads == 0) {
        long cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
        n_threads = (cpu_count > 0) ? (size_t)cpu_count : 4;
    }
    if (n_threads == 0) n_threads = 1;
    pool->n_threads = n_threads;

    /* Allocate the task queue. */
    pool->task_queue = calloc(pool->queue_capacity, sizeof(OcSpinPoolTask));
    if (!pool->task_queue) return OC_ERR_OOM;

    /* Allocate thread array. */
    pool->threads = calloc(n_threads, sizeof(pthread_t));
    if (!pool->threads) {
        free(pool->task_queue);
        pool->task_queue = NULL;
        return OC_ERR_OOM;
    }

    /* Initialize synchronization primitives. */
    if (pthread_mutex_init(&pool->lock, NULL) != 0) {
        free(pool->task_queue);
        free(pool->threads);
        pool->task_queue = NULL;
        pool->threads    = NULL;
        return OC_ERR_INTERNAL;
    }
    if (pthread_cond_init(&pool->not_empty, NULL) != 0) {
        pthread_mutex_destroy(&pool->lock);
        free(pool->task_queue);
        free(pool->threads);
        pool->task_queue = NULL;
        pool->threads    = NULL;
        return OC_ERR_INTERNAL;
    }
    if (pthread_cond_init(&pool->not_full, NULL) != 0) {
        pthread_cond_destroy(&pool->not_empty);
        pthread_mutex_destroy(&pool->lock);
        free(pool->task_queue);
        free(pool->threads);
        pool->task_queue = NULL;
        pool->threads    = NULL;
        return OC_ERR_INTERNAL;
    }
    if (pthread_cond_init(&pool->idle, NULL) != 0) {
        pthread_cond_destroy(&pool->not_full);
        pthread_cond_destroy(&pool->not_empty);
        pthread_mutex_destroy(&pool->lock);
        free(pool->task_queue);
        free(pool->threads);
        pool->task_queue = NULL;
        pool->threads    = NULL;
        return OC_ERR_INTERNAL;
    }

    /* Spawn worker threads. */
    for (size_t i = 0; i < n_threads; i++) {
        if (pthread_create(&pool->threads[i], NULL, worker_loop, pool) != 0) {
            /* Failed to spawn: shut down any already-started workers. */
            pthread_mutex_lock(&pool->lock);
            pool->shutdown = true;
            pthread_cond_broadcast(&pool->not_empty);
            pthread_mutex_unlock(&pool->lock);
            for (size_t j = 0; j < i; j++) {
                pthread_join(pool->threads[j], NULL);
            }
            pthread_cond_destroy(&pool->idle);
            pthread_cond_destroy(&pool->not_full);
            pthread_cond_destroy(&pool->not_empty);
            pthread_mutex_destroy(&pool->lock);
            free(pool->task_queue);
            free(pool->threads);
            pool->task_queue = NULL;
            pool->threads    = NULL;
            return OC_ERR_INTERNAL;
        }
    }

    return OC_OK;
}

OcError oc_spinpool_submit(OcSpinPool *pool, void *(*task_fn)(void *),
                           void *arg)
{
    if (!pool || !task_fn) return OC_ERR_INVALID_ARG;

    pthread_mutex_lock(&pool->lock);

    /* Wait for a free queue slot. */
    while (pool->n_pending >= pool->queue_capacity && !pool->shutdown) {
        pthread_cond_wait(&pool->not_full, &pool->lock);
    }
    if (pool->shutdown) {
        pthread_mutex_unlock(&pool->lock);
        return OC_ERR_INVALID_ARG;
    }

    /* Enqueue the task. */
    pool->task_queue[pool->head].fn     = task_fn;
    pool->task_queue[pool->head].arg    = arg;
    pool->task_queue[pool->head].result = NULL;
    pool->head = (pool->head + 1) % pool->queue_capacity;
    pool->n_pending++;

    pthread_cond_signal(&pool->not_empty);
    pthread_mutex_unlock(&pool->lock);
    return OC_OK;
}

OcError oc_spinpool_wait(OcSpinPool *pool)
{
    if (!pool) return OC_ERR_INVALID_ARG;
    pthread_mutex_lock(&pool->lock);
    while ((pool->n_pending > 0 || pool->n_active > 0) && !pool->shutdown) {
        pthread_cond_wait(&pool->idle, &pool->lock);
    }
    pthread_mutex_unlock(&pool->lock);
    return OC_OK;
}

OcError oc_spinpool_map(OcSpinPool *pool, void *(*fn)(void *),
                        void **items, size_t n_items, void **out_results)
{
    if (!pool || !fn) return OC_ERR_INVALID_ARG;
    if (n_items == 0) return OC_OK;
    if (!items) return OC_ERR_INVALID_ARG;

    /* Build a task list. We submit all tasks, then wait. Each task writes
     * its result into the out_results slot. We use a per-task context that
     * carries the item and the output slot. */
    typedef struct {
        void *(*fn)(void *);
        void  *item;
        void **out_slot;
    } MapContext;

    MapContext *ctxs = calloc(n_items, sizeof(MapContext));
    if (!ctxs) return OC_ERR_OOM;

    /* Trampoline that extracts the context, calls fn, stores result. */
    /* We need a stable function pointer; use a local function. */
    /* Note: this is a static trampoline pattern. */
    for (size_t i = 0; i < n_items; i++) {
        ctxs[i].fn       = fn;
        ctxs[i].item     = items[i];
        ctxs[i].out_slot = out_results ? &out_results[i] : NULL;
    }

    /* We cannot capture ctx in a plain function pointer easily, so we use
     * a different approach: submit tasks that each run fn(item) and store
     * the result. We use a thread-local-ish approach via the arg. */
    /* Since the task fn signature is void *(*fn)(void *arg), and we need C function pointers don't capture closures. So we use a static */

    /* Actually, the simplest correct approach: use the arg as the item, */

    /* Allocate a results array if caller didn't provide one. */
    void **results = out_results;
    bool allocated_results = false;
    if (!results) {
        results = calloc(n_items, sizeof(void *));
        if (!results) {
            free(ctxs);
            return OC_ERR_OOM;
        }
        allocated_results = true;
    }

    /* Submit tasks using the context as arg and a trampoline. */
    /* Define trampoline as a local function — C11 allows nested function
     * definitions? No, standard C does not. Use a file-scope function. */

    /* Use ctxs[i].out_slot to store the result. */
    for (size_t i = 0; i < n_items; i++) {
        ctxs[i].out_slot = &results[i];
    }

    /* File-scope trampoline defined below; we pass &ctxs[i] as arg. */
    extern void *oc_spinpool_map_trampoline(void *arg);

    for (size_t i = 0; i < n_items; i++) {
        OcError err = oc_spinpool_submit(pool, oc_spinpool_map_trampoline,
                                         &ctxs[i]);
        if (err != OC_OK) {
            free(ctxs);
            if (allocated_results) free(results);
            return err;
        }
    }

    OcError werr = oc_spinpool_wait(pool);
    free(ctxs);
    if (allocated_results) free(results);
    return werr;
}

/* Trampoline for oc_spinpool_map: calls ctx->fn(ctx->item) and stores the
 * result into *ctx->out_slot. */
void *oc_spinpool_map_trampoline(void *arg)
{
    typedef struct {
        void *(*fn)(void *);
        void  *item;
        void **out_slot;
    } MapContext;
    MapContext *ctx = (MapContext *)arg;
    void *result = ctx->fn(ctx->item);
    if (ctx->out_slot) *ctx->out_slot = result;
    return result;
}

size_t oc_spinpool_n_threads(const OcSpinPool *pool)
{
    if (!pool) return 0;
    return pool->n_threads;
}

size_t oc_spinpool_n_pending(const OcSpinPool *pool)
{
    if (!pool) return 0;
    return pool->n_pending;
}

void oc_spinpool_free(OcSpinPool *pool)
{
    if (!pool) return;
    if (!pool->threads) {
        memset(pool, 0, sizeof(*pool));
        return;
    }

    /* Signal shutdown. */
    pthread_mutex_lock(&pool->lock);
    pool->shutdown = true;
    pthread_cond_broadcast(&pool->not_empty);
    pthread_cond_broadcast(&pool->not_full);
    pthread_mutex_unlock(&pool->lock);

    /* Join all workers. */
    for (size_t i = 0; i < pool->n_threads; i++) {
        pthread_join(pool->threads[i], NULL);
    }

    /* Destroy sync primitives. */
    pthread_cond_destroy(&pool->idle);
    pthread_cond_destroy(&pool->not_full);
    pthread_cond_destroy(&pool->not_empty);
    pthread_mutex_destroy(&pool->lock);

    free(pool->task_queue);
    free(pool->threads);
    memset(pool, 0, sizeof(*pool));
}
