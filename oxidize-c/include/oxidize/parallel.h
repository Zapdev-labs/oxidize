/* parallel.h — persistent worker pool for the inference hot path. Not reentrant: a region body must not itself open a region. */
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

/* Body of a parallel region. */
typedef void (*OcParallelFn)(size_t begin, size_t end, size_t tid,
                             void *user_data);

/* Set the worker count. 1 disables threading (regions run inline). 0 means auto (online CPU count). Call once at startup. If workers cannot be created, falls back to inline and returns OC_ERR_INTERNAL. */
OcError oc_parallel_set_threads(size_t n_threads);

/* Current worker count, including the calling thread. 1 means inline. */
size_t oc_parallel_n_threads(void);

/* Run `fn` over [0, n) split across the pool, and return once every slice has
 * finished. With one thread, or n small enough that splitting cannot pay for
 * itself, `fn` is called once inline on the caller. */
void oc_parallel_for(size_t n, OcParallelFn fn, void *user_data);

/* Per-thread scratch of at least `bytes`, valid until the next larger call on the same `tid`. Returns NULL on allocation failure. `tid` must be a value handed to OcParallelFn. */
void *oc_parallel_scratch(size_t tid, size_t bytes);

/* Join workers and release the pool. Safe to call when not started. Mainly
 * for tests and leak checkers; a process exiting need not call it. */
void oc_parallel_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_PARALLEL_H */
