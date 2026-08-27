/*
 * parallel.h — persistent worker pool for the inference hot path.
 *
 * The forward pass dispatches a parallel region several times per layer (Q/K/V,
 * attention output, gate/up, down — so on the order of 200 regions per token
 * for a 32-layer model). That rules out both thread-per-call and the queue in
 * spinpool.h, which allocates and takes a mutex per submitted item: the
 * dispatch overhead would swamp the work. Here a region is published by
 * bumping one atomic generation counter, workers pick up a statically-assigned
 * slice, and the calling thread does a slice itself rather than idling.
 *
 * The pool is process-global and lazily started. `oc_parallel_set_threads(1)`
 * makes every region run inline on the calling thread with no synchronization
 * at all, which is both the default and the way the tests keep results
 * deterministic.
 *
 * Not reentrant: a region body must not itself open a region.
 */
#ifndef OXIDIZE_PARALLEL_H
#define OXIDIZE_PARALLEL_H

#include <stddef.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Upper bound on worker threads. Sized for large dual-socket machines; the
 * effective count is min(this, what the caller asks for). */
#define OC_PARALLEL_MAX_THREADS 256u

/* Body of a parallel region. Called once per participating thread with a
 * half-open slice [begin, end) of the iteration space and a stable thread
 * index `tid` in [0, n_threads). Slices never overlap, so a body that writes
 * only within its own slice needs no locking. */
typedef void (*OcParallelFn)(size_t begin, size_t end, size_t tid,
                             void *user_data);

/* Set the worker count. 1 disables threading entirely (regions run inline).
 * 0 means "auto": use the number of online CPUs. Starting or resizing the
 * pool joins any existing workers first, so this is not cheap — call it once
 * at startup, not per token. Returns OC_OK, or OC_ERR_INTERNAL if threads
 * could not be created (in which case the pool falls back to inline). */
OcError oc_parallel_set_threads(size_t n_threads);

/* Current worker count, including the calling thread. 1 means inline. */
size_t oc_parallel_n_threads(void);

/* Run `fn` over [0, n) split across the pool, and return once every slice has
 * finished. With one thread, or n small enough that splitting cannot pay for
 * itself, `fn` is called once inline on the caller. */
void oc_parallel_for(size_t n, OcParallelFn fn, void *user_data);

/* Per-thread scratch of at least `bytes`, valid until the next call with a
 * larger size on the same `tid`. Exists because the quantized matvec needs a
 * dequantization buffer per thread: the single caller-provided `temp` cannot
 * be shared once rows are split. Returns NULL on allocation failure.
 * Only valid for `tid` values handed to an OcParallelFn. */
void *oc_parallel_scratch(size_t tid, size_t bytes);

/* Join workers and release the pool. Safe to call when not started. Mainly
 * for tests and leak checkers; a process exiting need not call it. */
void oc_parallel_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_PARALLEL_H */
