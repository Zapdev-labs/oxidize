/*
 * mmap.h — OcMmap: read-only memory mapping with madvise/mlock/prefault.
 *
 * Port of oxidize-core::MappedGgufFile mmap machinery (memmap2-based) to C.
 * The single place in the C port where `mmap`/`madvise`/`mlock` syscalls
 * appear. Higher-level GGUF loading (src/format/gguf.c) uses OcMmap to back
 * tensor data; weights live outside the arena and are freed via munmap.
 *
 * Lifecycle (heap-allocated, mirroring OcArena's pattern):
 *   OcMmap *m = NULL;
 *   if (oc_mmap_open_readonly("model.gguf", &m) == OC_OK) {
 *       oc_mmap_advise_hugepage(m);      // MADV_HUGEPAGE (best-effort)
 *       oc_mmap_advise_sequential(m);    // MADV_SEQUENTIAL (best-effort)
 *       const uint8_t *bytes = oc_mmap_bytes(m);
 *       size_t           len   = oc_mmap_len(m);
 *       ...
 *       oc_mmap_close(m);                // munmap + close fd + free(m)
 *   }
 *
 * `oc_mmap_mlock_with_headroom()` only calls `mlock` when the mapping fits
 * in `MemAvailable` with >= 30% headroom (model_bytes < available * 70%),
 * matching Rust's `prefault_pages_locked` policy. On success, every page is
 * also touched via volatile reads to fault it in.
 *
 * All syscalls here are best-effort: a failure to madvise/mlock is logged at
 * WARN level but does NOT return an error (the mapping is still usable).
 */
#ifndef OXIDIZE_UTIL_MMAP_H
#define OXIDIZE_UTIL_MMAP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A read-only memory mapping of a file. Owns the file descriptor and the
 * mapping; `oc_mmap_close()` releases both and frees the OcMmap itself.
 * The struct is opaque (defined in src/util/mmap.c): callers interact via
 * the accessor functions below. This keeps <sys/mman.h> out of the public
 * header surface. */
typedef struct OcMmap OcMmap;

/* Memory-map `path` for read-only access (PROT_READ, MAP_PRIVATE).
 *
 * On Linux, also applies MADV_SEQUENTIAL + MADV_WILLNEED (best-effort,
 * matching Rust `load_mapped_gguf`). The caller may additionally apply
 * `oc_mmap_advise_hugepage()` once it has decided hugepages are appropriate
 * (Rust enables THP only when the model fits in RAM with >= 2x headroom).
 *
 * On success, writes a heap-allocated OcMmap* to `*out`. Returns OC_OK,
 * OC_ERR_IO (open/stat), OC_ERR_INVALID_ARG (NULL args), or OC_ERR_OOM
 * (mmap/malloc failure). On error, `*out` is set to NULL. */
OcError oc_mmap_open_readonly(const char *path, OcMmap **out);

/* Map an already-open file descriptor for read-only access. Takes ownership
 * of `fd` (closes it on `oc_mmap_close`). Used by multi-shard GGUF loading
 * where the caller pre-opens shards for validation. */
OcError oc_mmap_open_fd(int fd, size_t len, OcMmap **out);

/* Apply MADV_HUGEPAGE to the mapping (Linux only). Best-effort: on success
 * sets the internal hugepage flag; on failure logs WARN and leaves it false.
 * No-op (returns OC_OK) on non-Linux platforms. The caller is responsible for
 * headroom policy (only enable THP when model fits in RAM with >= 2x
 * headroom — see Rust `MappedGgufFile::advise_huge_pages`). */
OcError oc_mmap_advise_hugepage(OcMmap *m);

/* Apply MADV_SEQUENTIAL (best-effort, Linux only). */
OcError oc_mmap_advise_sequential(OcMmap *m);

/* Apply MADV_RANDOM (best-effort, Linux only) — for random-access patterns. */
OcError oc_mmap_advise_random(OcMmap *m);

/* Apply MADV_WILLNEED (best-effort, Linux only) — queues async readahead. */
OcError oc_mmap_advise_willneed(OcMmap *m);

/* Returns true if MADV_HUGEPAGE was applied successfully. */
bool oc_mmap_hugepage(const OcMmap *m);

/* Returns true if mlock() succeeded for this mapping. */
bool oc_mmap_mlocked(const OcMmap *m);

/* Lock the mapping into physical RAM with `mlock(2)`. Only attempts the lock
 * when the mapping fits in `MemAvailable` (Linux) with >= 30% headroom
 * (model_bytes < available * 7 / 10). Returns true if mlock succeeded; false
 * if it was skipped (headroom too tight) or failed. Either way also runs a
 * sequential prefault sweep (volatile reads) so every page is resident.
 *
 * On Linux, raises RLIMIT_MEMLOCK to RLIM_INFINITY first (requires
 * CAP_IPC_LOCK or root); if that fails, mlock may still succeed for small
 * mappings under the default limit. */
bool oc_mmap_mlock_with_headroom(OcMmap *m);

/* Sequential prefault sweep: touch every 4 KiB page via volatile reads so
 * they are faulted into the page cache. Returns a checksum (XOR of all
 * touched bytes) — useful as a sanity check that the mapping is readable.
 * Single-threaded; for parallel prefault use `oc_mmap_prefault_parallel`. */
uint8_t oc_mmap_prefault(const OcMmap *m);

/* Parallel prefault sweep using `n_threads` pthreads. Each thread faults a
 * contiguous chunk of the mapping. Returns the XOR checksum of all touched
 * bytes. `n_threads` is clamped to [1, 64]. */
uint8_t oc_mmap_prefault_parallel(const OcMmap *m, size_t n_threads);

/* Read `MemAvailable` from /proc/meminfo (Linux only). Returns true and
 * writes `*out_bytes` on success; false on any failure (callers should treat
 * false as "unlimited" to be safe). Mirrors Rust `linux_mem_available_bytes`. */
bool oc_linux_mem_available_bytes(uint64_t *out_bytes);

/* Pointer to the mapped bytes (PROT_READ). Returns NULL if `m` is closed. */
const uint8_t *oc_mmap_bytes(const OcMmap *m);

/* Length of the mapping in bytes. Returns 0 if `m` is closed. */
size_t oc_mmap_len(const OcMmap *m);

/* Close the mapping: munmap + close(fd) + free(m). Safe on NULL. After this
 * call, the OcMmap* is invalid. */
void oc_mmap_close(OcMmap *m);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_UTIL_MMAP_H */
