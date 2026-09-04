/*
 * parallel.c — persistent worker pool (see parallel.h).
 *
 * Dispatch is a generation counter rather than a queue. Publishing a region
 * is one release-store; each worker compares the counter against the last
 * generation it ran and, when it moves, executes its fixed slice. Completion
 * is an atomic decrement the caller spins on. No allocation and no mutex on
 * the dispatch path — the region body is often only microseconds of work, so
 * a futex round trip per region would cost more than the work itself.
 *
 * Workers spin for a bounded number of iterations before parking on a condvar.
 * Generation and decode are back-to-back regions, so a spinning worker
 * normally finds the next region before it ever parks; the condvar only
 * matters between tokens or while the model is loading.
 */
#define _POSIX_C_SOURCE 200809L
#ifdef __APPLE__
#define _DARWIN_C_SOURCE 1  /* _SC_NPROCESSORS_ONLN is BSD-surface on macOS */
#endif
#include "oxidize/parallel.h"

#include "oxidize/log.h"
#include "oxidize/numa.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

/* Spin iterations before a worker parks. Tuned to comfortably cover the gap
 * between consecutive regions in a forward pass without burning a core
 * through the seconds-long pauses while a model loads. */
#define SPIN_BEFORE_PARK 20000u

/* Below this many iterations a region runs inline: splitting costs a
 * dispatch plus a completion wait, which is not worth it for a handful of
 * rows (e.g. a 32-row attention-head loop). */
#define MIN_ITEMS_TO_SPLIT 8u

typedef struct Worker {
    pthread_t thread;
    size_t    tid;
    void     *scratch;
    size_t    scratch_cap;
} Worker;

static struct {
    Worker   workers[OC_PARALLEL_MAX_THREADS];
    size_t   n_threads;          /* including the calling thread; 1 = inline */
    bool     started;

    /* Current region. Written by the caller before publishing `generation`,
     * read by workers after observing it — the release/acquire pair on
     * `generation` is what makes these safe to read unsynchronized. */
    OcParallelFn fn;
    void        *user_data;
    size_t       n_items;

    _Atomic uint64_t generation;   /* bumped once per region                */
    _Atomic size_t   n_remaining;  /* slices not yet finished               */
    _Atomic bool     shutdown;
    /* Workers currently blocked on the condvar. The dispatch path reads this
     * to decide whether waking anyone is necessary at all — see the comment
     * in oc_parallel_for(). */
    _Atomic uint32_t n_parked;

    /* Parking. Only touched once a worker has spun without seeing work. */
    pthread_mutex_t lock;
    pthread_cond_t  wake;
} g_pool;

/* Caller's scratch, for the slice the calling thread runs itself (tid 0). */
static void  *g_main_scratch;
static size_t g_main_scratch_cap;

/* Slice [begin, end) of `n` for thread `tid` of `n_threads`. Remainder is
 * spread one item at a time over the low threads rather than dumped on the
 * last one, which would leave it running alone at the end of every region. */
static void slice_for(size_t n, size_t tid, size_t n_threads,
                      size_t *begin, size_t *end)
{
    const size_t base = n / n_threads;
    const size_t rem  = n % n_threads;
    const size_t b = tid * base + (tid < rem ? tid : rem);
    const size_t e = b + base + (tid < rem ? 1u : 0u);
    *begin = b;
    *end   = e;
}

static void run_slice(size_t tid)
{
    size_t begin, end;
    slice_for(g_pool.n_items, tid, g_pool.n_threads, &begin, &end);
    if (begin < end) {
        g_pool.fn(begin, end, tid, g_pool.user_data);
    }
    atomic_fetch_sub_explicit(&g_pool.n_remaining, 1u, memory_order_release);
}

static void *worker_main(void *arg)
{
    Worker *w = (Worker *)arg;
    uint64_t seen = 0;

    /* SMT-aware pinning: when the pool size equals the machine's physical
     * core count, land this worker on its own core (see numa.c — measured
     * 19% faster than SMT siblings sharing cores on µop-bound phases).
     * Advice is cached topology state; no-op anywhere else. */
    {
        uint32_t cpu;
        if (oc_numa_distinct_core_for_worker(w->tid, g_pool.n_threads, &cpu)) {
            OcError pe = oc_numa_pin_cpu(cpu);
            if (pe != OC_OK)
                oc_log(OC_LOG_WARN,
                       "parallel: worker %zu pin to cpu %u failed", w->tid,
                       cpu);
        }
    }

    for (;;) {
        uint64_t gen;
        size_t spins = 0;
        /* Spin first, then park. */
        for (;;) {
            if (atomic_load_explicit(&g_pool.shutdown, memory_order_acquire)) {
                return NULL;
            }
            gen = atomic_load_explicit(&g_pool.generation, memory_order_acquire);
            if (gen != seen) break;
            if (++spins < SPIN_BEFORE_PARK) {
#if defined(__x86_64__) || defined(__i386__)
                __builtin_ia32_pause();
#elif defined(__aarch64__)
                __asm__ __volatile__("yield" ::: "memory");
#endif
                continue;
            }
            pthread_mutex_lock(&g_pool.lock);
            /* Publish that we are about to park BEFORE re-checking, so a
             * dispatcher that sees a zero count cannot have missed us: it
             * would then also have published its generation, which the
             * re-check below observes while we still hold the lock. */
            atomic_fetch_add_explicit(&g_pool.n_parked, 1u, memory_order_seq_cst);
            /* Re-check under the lock: the region may have been published
             * between the load above and taking the lock, and the signal
             * would then already have been sent. */
            while (!atomic_load_explicit(&g_pool.shutdown, memory_order_acquire) &&
                   atomic_load_explicit(&g_pool.generation,
                                        memory_order_acquire) == seen) {
                pthread_cond_wait(&g_pool.wake, &g_pool.lock);
            }
            atomic_fetch_sub_explicit(&g_pool.n_parked, 1u, memory_order_seq_cst);
            pthread_mutex_unlock(&g_pool.lock);
            spins = 0;
        }
        seen = gen;
        if (atomic_load_explicit(&g_pool.shutdown, memory_order_acquire)) {
            return NULL;
        }
        run_slice(w->tid);
    }
}

void oc_parallel_shutdown(void)
{
    if (!g_pool.started) return;

    atomic_store_explicit(&g_pool.shutdown, true, memory_order_seq_cst);
    /* Unconditional here, unlike the dispatch path: shutdown must reach
     * workers that are already parked. */
    pthread_mutex_lock(&g_pool.lock);
    pthread_cond_broadcast(&g_pool.wake);
    pthread_mutex_unlock(&g_pool.lock);

    /* Worker i owns g_pool.workers[i + 1]; index 0 is the calling thread. */
    for (size_t i = 1; i < g_pool.n_threads; i++) {
        pthread_join(g_pool.workers[i].thread, NULL);
        free(g_pool.workers[i].scratch);
        g_pool.workers[i].scratch = NULL;
        g_pool.workers[i].scratch_cap = 0;
    }
    pthread_mutex_destroy(&g_pool.lock);
    pthread_cond_destroy(&g_pool.wake);

    free(g_main_scratch);
    g_main_scratch = NULL;
    g_main_scratch_cap = 0;

    g_pool.started = false;
    g_pool.n_threads = 1;
    atomic_store_explicit(&g_pool.shutdown, false, memory_order_relaxed);
    atomic_store_explicit(&g_pool.generation, 0, memory_order_relaxed);
}

OcError oc_parallel_set_threads(size_t n_threads)
{
    /* Undo the calling thread's worker-0 pin before resizing the pool:
     * the affinity belongs to the thread and survives
     * oc_parallel_shutdown, and every worker of the new pool inherits
     * the calling thread's mask — the resize path would otherwise
     * collapse all replacement workers onto worker 0's single core.
     * No-op when this thread was never pinned. */
    oc_numa_pin_restore();
    oc_parallel_shutdown();

    if (n_threads == 0) {
        long cpus = sysconf(_SC_NPROCESSORS_ONLN);
        n_threads = (cpus > 0) ? (size_t)cpus : 1u;
    }
    if (n_threads > OC_PARALLEL_MAX_THREADS) n_threads = OC_PARALLEL_MAX_THREADS;

    g_pool.n_threads = n_threads;
    if (n_threads <= 1) {
        g_pool.n_threads = 1;
        return OC_OK;   /* inline; nothing to start */
    }

    if (pthread_mutex_init(&g_pool.lock, NULL) != 0) {
        g_pool.n_threads = 1;
        return OC_ERR_INTERNAL;
    }
    if (pthread_cond_init(&g_pool.wake, NULL) != 0) {
        pthread_mutex_destroy(&g_pool.lock);
        g_pool.n_threads = 1;
        return OC_ERR_INTERNAL;
    }
    atomic_store_explicit(&g_pool.shutdown, false, memory_order_relaxed);
    atomic_store_explicit(&g_pool.generation, 0, memory_order_relaxed);
    atomic_store_explicit(&g_pool.n_remaining, 0, memory_order_relaxed);
    atomic_store_explicit(&g_pool.n_parked, 0u, memory_order_relaxed);
    g_pool.started = true;

    for (size_t i = 1; i < n_threads; i++) {
        g_pool.workers[i].tid = i;
        g_pool.workers[i].scratch = NULL;
        g_pool.workers[i].scratch_cap = 0;
        if (pthread_create(&g_pool.workers[i].thread, NULL, worker_main,
                           &g_pool.workers[i]) != 0) {
            /* Keep the threads that did start rather than failing outright:
             * fewer workers is a slowdown, not a wrong answer. */
            oc_log(OC_LOG_WARN,
                   "parallel: could only start %zu of %zu threads",
                   i, n_threads);
            g_pool.n_threads = i;
            return (i > 1) ? OC_OK : OC_ERR_INTERNAL;
        }
    }
    /* The calling thread is worker 0: same one-core-per-worker advice.
     * Snapshot its pre-pin mask first so a later oc_parallel_set_threads
     * can restore (see the top of this function). */
    {
        uint32_t cpu;
        oc_numa_pin_save_orig();
        if (oc_numa_distinct_core_for_worker(0, n_threads, &cpu)) {
            OcError pe = oc_numa_pin_cpu(cpu);
            if (pe != OC_OK)
                oc_log(OC_LOG_WARN,
                       "parallel: worker 0 pin to cpu %u failed", cpu);
        }
    }
    return OC_OK;
}

size_t oc_parallel_n_threads(void)
{
    return g_pool.n_threads ? g_pool.n_threads : 1u;
}

void oc_parallel_for(size_t n, OcParallelFn fn, void *user_data)
{
    if (n == 0 || fn == NULL) return;

    const size_t nt = oc_parallel_n_threads();
    if (nt <= 1 || !g_pool.started || n < MIN_ITEMS_TO_SPLIT) {
        fn(0, n, 0, user_data);
        return;
    }

    /* Publish the region, then bump the generation with release ordering so
     * a worker that observes the new generation also observes these fields. */
    g_pool.fn = fn;
    g_pool.user_data = user_data;
    g_pool.n_items = n;
    atomic_store_explicit(&g_pool.n_remaining, nt, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_pool.generation, 1u, memory_order_seq_cst);

    /* Wake anyone who parked — but only if anyone actually did.
     *
     * A forward pass opens on the order of 200 regions per token, and between
     * consecutive regions the workers are still spinning, so the common case
     * is that nothing is parked and the mutex round trip is pure overhead.
     * Taking a lock and broadcasting 200 times per token across 48 threads was
     * limiting how far this scaled: throughput peaked at 40 threads and fell
     * off at 48, where llama.cpp's stays flat.
     *
     * The generation store above is seq_cst, and a parking worker increments
     * n_parked under the lock before re-checking the generation, so the two
     * orderings interleave safely: either we see its increment and signal, or
     * it sees our new generation and does not sleep. */
    if (atomic_load_explicit(&g_pool.n_parked, memory_order_seq_cst) != 0) {
        pthread_mutex_lock(&g_pool.lock);
        pthread_cond_broadcast(&g_pool.wake);
        pthread_mutex_unlock(&g_pool.lock);
    }

    /* The calling thread is participant 0 rather than a spectator. */
    run_slice(0);

    while (atomic_load_explicit(&g_pool.n_remaining, memory_order_acquire) != 0) {
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#elif defined(__aarch64__)
        __asm__ __volatile__("yield" ::: "memory");
#endif
    }
}

void *oc_parallel_scratch(size_t tid, size_t bytes)
{
    void **slot;
    size_t *cap;
    if (tid == 0) {
        slot = &g_main_scratch;
        cap  = &g_main_scratch_cap;
    } else if (tid < OC_PARALLEL_MAX_THREADS) {
        slot = &g_pool.workers[tid].scratch;
        cap  = &g_pool.workers[tid].scratch_cap;
    } else {
        return NULL;
    }

    if (*cap >= bytes && *slot != NULL) return *slot;

    void *p = realloc(*slot, bytes);
    if (p == NULL) return NULL;
    *slot = p;
    *cap = bytes;
    return p;
}
