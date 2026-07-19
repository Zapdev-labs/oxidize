/*
 * arena.h — OcArena bump-pointer allocator.
 *
 * Allocates from a single contiguous backing buffer; `oc_arena_free()` frees
 * everything at once (no per-allocation free tracking). Used for model-lifetime
 * allocations: config structs, tensor name strings, temp buffers. mmap'd
 * weights are NOT arena-allocated (their lifetime is tied to mmap).
 *
 * Port concept: architecture.md §3.6.
 */
#ifndef OXIDIZE_ARENA_H
#define OXIDIZE_ARENA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OcArena OcArena;

/* Default initial capacity if `initial_cap == 0`. */
#define OC_ARENA_DEFAULT_CAP (1024 * 1024)  /* 1 MiB */

/* Create a new arena with at least `initial_cap` bytes of backing storage.
 * If `initial_cap == 0`, uses OC_ARENA_DEFAULT_CAP. Returns NULL on OOM. */
OcArena *oc_arena_new(size_t initial_cap);

/* Allocate `n * sz` bytes aligned to `align` (must be a power of two and
 * <= 256). `sz == 0` or `n == 0` returns a non-NULL unique pointer. Returns
 * NULL on OOM (the arena grows automatically up to a hard cap). The returned
 * pointer is valid until `oc_arena_free()` is called. */
void *oc_arena_alloc(OcArena *a, size_t n, size_t align);

/* Convenience: allocate `n_bytes` bytes with default alignment (16). */
void *oc_arena_alloc_bytes(OcArena *a, size_t n_bytes);

/* Duplicate a NUL-terminated string into the arena. Returns NULL on OOM or
 * NULL input. The returned pointer is arena-owned. */
char *oc_arena_dup(OcArena *a, const char *s);

/* Duplicate `n` bytes of `s` into the arena (no NUL required). Returns a
 * NUL-terminated copy (allocates `n+1` bytes). Returns NULL on OOM/NULL. */
char *oc_arena_dup_n(OcArena *a, const char *s, size_t n);

/* Format a printf-style string into the arena. Returns the arena-owned
 * NUL-terminated result, or NULL on OOM. */
char *oc_arena_printf(OcArena *a, const char *fmt, ...);

/* Total bytes currently allocated from this arena (sum of all live allocations
 * excluding bookkeeping). */
size_t oc_arena_used(const OcArena *a);

/* Current backing capacity. */
size_t oc_arena_capacity(const OcArena *a);

/* Reset the arena to empty WITHOUT freeing the backing storage. Useful for
 * scratch arenas that get reused. All previously-returned pointers are
 * invalidated. */
void oc_arena_reset(OcArena *a);

/* Free the arena and all allocations it owns. Safe on NULL. */
void oc_arena_free(OcArena *a);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_ARENA_H */
