
/* Expose POSIX/Linux extensions: O_CLOEXEC, madvise, MADV_*, mlock, setrlimit,
 * pthread_create, etc. Must be the very first non-comment line so system
 * headers pick it up. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "oxidize/util/mmap.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef __linux__
#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#endif

#include "oxidize/log.h"
#include "oxidize/util/bytes.h"

struct OcMmap {
    void   *addr;       /* mmap'd region (PROT_READ, MAP_PRIVATE); NULL if closed */
    size_t  len;        /* length of the mapping in bytes */
    int     fd;         /* underlying file descriptor, or -1 if not owned */
    bool    hugepage;   /* true if MADV_HUGEPAGE was applied successfully */
    bool    mlocked;    /* true if mlock() succeeded */
    bool    heap_alloc; /* true if addr was malloc'd (non-Linux fallback) */
};

/* Allocate an OcMmap on the heap and zero-initialize it. */
static OcMmap *mmap_alloc(void)
{
    OcMmap *m = (OcMmap *)calloc(1, sizeof(OcMmap));
    if (!m) {
        oc_log(OC_LOG_ERROR, "mmap: OOM allocating OcMmap");
        return NULL;
    }
    m->fd = -1;
    return m;
}

OcError oc_mmap_open_readonly(const char *path, OcMmap **out)
{
    if (!path || !out) return OC_ERR_INVALID_ARG;
    *out = NULL;

    OcMmap *m = mmap_alloc();
    if (!m) return OC_ERR_OOM;

#ifdef __linux__
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        oc_log(OC_LOG_ERROR, "mmap: open(%s) failed: %s", path, strerror(errno));
        free(m);
        return OC_ERR_IO;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        oc_log(OC_LOG_ERROR, "mmap: fstat(%s) failed: %s", path, strerror(errno));
        close(fd);
        free(m);
        return OC_ERR_IO;
    }
    if (!S_ISREG(st.st_mode)) {
        oc_log(OC_LOG_ERROR, "mmap: %s is not a regular file", path);
        close(fd);
        free(m);
        return OC_ERR_IO;
    }
    if (st.st_size <= 0) {
        oc_log(OC_LOG_ERROR, "mmap: %s is empty (size=%lld)", path,
                (long long)st.st_size);
        close(fd);
        free(m);
        return OC_ERR_IO;
    }
    size_t len = (size_t)st.st_size;

    /* PROT_READ, MAP_PRIVATE: page-cache backed, copy-on-write for writes
     * (we never write). Mirrors Rust `memmap2::Mmap::map(file)`. */
    void *addr = mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
        oc_log(OC_LOG_ERROR, "mmap: mmap(%s, %zu) failed: %s", path, len,
                strerror(errno));
        close(fd);
        free(m);
        return OC_ERR_OOM;
    }

    m->addr = addr;
    m->len  = len;
    m->fd   = fd;

    /* Best-effort: tell the kernel we'll read sequentially + will need the
     * pages. Matches Rust `load_mapped_gguf`. */
    oc_mmap_advise_sequential(m);
    oc_mmap_advise_willneed(m);
    *out = m;
    return OC_OK;
#else
    /* Non-Linux: fall back to read() into a malloc'd buffer so the API is
     * still usable (tests don't need true mmap). The "mapping" is freed via
     * oc_mmap_close() -> free(). */
    FILE *f = fopen(path, "rb");
    if (!f) {
        oc_log(OC_LOG_ERROR, "mmap: fopen(%s) failed: %s", path, strerror(errno));
        free(m);
        return OC_ERR_IO;
    }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); free(m); return OC_ERR_IO; }
    long sz = ftell(f);
    if (sz <= 0) { fclose(f); free(m); return OC_ERR_IO; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); free(m); return OC_ERR_IO; }
    size_t len = (size_t)sz;
    void *addr = malloc(len);
    if (!addr) { fclose(f); free(m); return OC_ERR_OOM; }
    if (fread(addr, 1, len, f) != len) {
        oc_log(OC_LOG_ERROR, "mmap: fread(%s) failed", path);
        free(addr);
        fclose(f);
        free(m);
        return OC_ERR_IO;
    }
    fclose(f);
    m->addr      = addr;
    m->len       = len;
    m->fd        = -1;
    m->heap_alloc = true;
    *out = m;
    return OC_OK;
#endif
}

OcError oc_mmap_open_fd(int fd, size_t len, OcMmap **out)
{
    if (!out || fd < 0 || len == 0) return OC_ERR_INVALID_ARG;
    *out = NULL;
    OcMmap *m = mmap_alloc();
    if (!m) return OC_ERR_OOM;

#ifdef __linux__
    /* Note: oc_mmap_open_fd takes ownership of `fd` (m->fd = fd on success, and oc_mmap_close() will close it). */
    /* Reject len > file size: mmap would succeed but touching pages past EOF
     * SIGBUSes. Only enforced for regular files (the only mappable case here). */
    struct stat st;
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)
        || st.st_size < 0 || len > (size_t)st.st_size) {
        oc_log(OC_LOG_ERROR, "mmap: fd=%d not a regular file of >= %zu bytes",
                fd, len);
        close(fd);
        free(m);
        return OC_ERR_INVALID_ARG;
    }
    void *addr = mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
        oc_log(OC_LOG_ERROR, "mmap: mmap(fd=%d, %zu) failed: %s", fd, len,
                strerror(errno));
        close(fd);
        free(m);
        return OC_ERR_OOM;
    }
    m->addr = addr;
    m->len  = len;
    m->fd   = fd;
    oc_mmap_advise_sequential(m);
    oc_mmap_advise_willneed(m);
    *out = m;
    return OC_OK;
#else
    /* Non-Linux: oc_mmap_open_fd is unsupported (no mmap syscall). Close the
     * caller-provided fd so it doesn't leak, then return OC_ERR_INVALID_ARG. */
    (void)close(fd);
    free(m);
    return OC_ERR_INVALID_ARG;
#endif
}

OcError oc_mmap_advise_hugepage(OcMmap *m)
{
    if (!m || !m->addr) return OC_ERR_INVALID_ARG;
#ifdef __linux__
    /* MADV_HUGEPAGE: hint khugepaged to collapse small pages into 2 MiB
     * transparent huge pages. Best-effort. */
    if (madvise(m->addr, m->len, MADV_HUGEPAGE) != 0) {
        oc_log(OC_LOG_WARN, "mmap: MADV_HUGEPAGE failed: %s", strerror(errno));
        m->hugepage = false;
    } else {
        m->hugepage = true;
    }
    /* MADV_COLLAPSE (kernel >= 6.1): synchronously collapse page-cache folios
     * into huge pages. Best-effort: older kernels return EINVAL. */
#ifndef MADV_COLLAPSE
#define MADV_COLLAPSE 25
#endif
    if (madvise(m->addr, m->len, MADV_COLLAPSE) != 0) {
        /* Expected on older kernels; the MADV_HUGEPAGE hint still stands. */
    }
    return OC_OK;
#else
    return OC_OK;
#endif
}

OcError oc_mmap_advise_sequential(OcMmap *m)
{
    if (!m || !m->addr) return OC_ERR_INVALID_ARG;
#ifdef __linux__
    if (madvise(m->addr, m->len, MADV_SEQUENTIAL) != 0) {
        oc_log(OC_LOG_WARN, "mmap: MADV_SEQUENTIAL failed: %s", strerror(errno));
    }
#endif
    return OC_OK;
}

OcError oc_mmap_advise_random(OcMmap *m)
{
    if (!m || !m->addr) return OC_ERR_INVALID_ARG;
#ifdef __linux__
    if (madvise(m->addr, m->len, MADV_RANDOM) != 0) {
        oc_log(OC_LOG_WARN, "mmap: MADV_RANDOM failed: %s", strerror(errno));
    }
#endif
    return OC_OK;
}

OcError oc_mmap_advise_willneed(OcMmap *m)
{
    if (!m || !m->addr) return OC_ERR_INVALID_ARG;
#ifdef __linux__
    if (madvise(m->addr, m->len, MADV_WILLNEED) != 0) {
        oc_log(OC_LOG_WARN, "mmap: MADV_WILLNEED failed: %s", strerror(errno));
    }
#endif
    return OC_OK;
}

bool oc_mmap_hugepage(const OcMmap *m)
{
    return m ? m->hugepage : false;
}

bool oc_mmap_mlocked(const OcMmap *m)
{
    return m ? m->mlocked : false;
}

bool oc_linux_mem_available_bytes(uint64_t *out_bytes)
{
    if (!out_bytes) return false;
#ifdef __linux__
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return false;
    char line[256];
    bool found = false;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "MemAvailable:", 13) == 0) {
            /* Parse the number after "MemAvailable:" using strtoull (sscanf
             * doesn't report conversion errors reliably). */
            errno = 0;
            char *endp = NULL;
            unsigned long long kb = strtoull(line + 13, &endp, 10);
            if (errno == 0 && endp != line + 13) {
                *out_bytes = (uint64_t)kb * 1024ull;
                found = true;
                break;
            }
        }
    }
    fclose(f);
    return found;
#else
    return false;
#endif
}

bool oc_mmap_mlock_with_headroom(OcMmap *m)
{
    if (!m || !m->addr) return false;
#ifdef __linux__
    /* Raise RLIMIT_MEMLOCK to RLIM_INFINITY (requires CAP_IPC_LOCK or root).
     * Best-effort: if it fails, mlock may still succeed for small mappings
     * under the default 64 KiB limit. */
    struct rlimit unlimited = {
        .rlim_cur = RLIM_INFINITY,
        .rlim_max = RLIM_INFINITY,
    };
    (void)setrlimit(RLIMIT_MEMLOCK, &unlimited);

    uint64_t available = 0;
    bool has_avail = oc_linux_mem_available_bytes(&available);
    /* If we can't read MemAvailable, treat as unlimited to be safe (mirrors
     * Rust which uses `unwrap_or(u64::MAX)`). */
    bool safe_to_lock = true;
    if (has_avail) {
        /* Only mlock when the mapping fits with >= 30% headroom:
         * model_bytes < available * 7 / 10. */
        safe_to_lock = ((uint64_t)m->len) < (available * 7ull) / 10ull;
    }

    bool mlocked = false;
    if (safe_to_lock) {
        if (mlock(m->addr, m->len) == 0) {
            mlocked = true;
        } else {
            oc_log(OC_LOG_WARN, "mmap: mlock(%zu bytes) failed: %s",
                    m->len, strerror(errno));
        }
    } else {
        oc_log(OC_LOG_INFO,
                "mmap: skipping mlock (headroom too tight: %zu bytes vs %llu avail)",
                m->len, (unsigned long long)available);
        /* Fall back to async readahead only. */
        oc_mmap_advise_willneed(m);
    }

    m->mlocked = mlocked;
    /* Always prefault (single-threaded) so every page is resident — even if
     * mlock was skipped, this warms the page cache. */
    oc_mmap_prefault(m);
    return mlocked;
#else
    /* Non-Linux: no mlock; just prefault. */
    oc_mmap_prefault(m);
    return false;
#endif
}

uint8_t oc_mmap_prefault(const OcMmap *m)
{
    if (!m || !m->addr) return 0;
    const uint8_t *bytes = (const uint8_t *)m->addr;
    size_t len = m->len;
    uint8_t checksum = 0;
    for (size_t off = 0; off < len; off += 4096) {
        checksum ^= oc_read_volatile_byte(bytes, len, off);
    }
    /* Touch the last byte too (mirrors Rust `if let Some(last) = bytes.last()`). */
    if (len > 0) {
        checksum ^= oc_read_volatile_byte(bytes, len, len - 1);
    }
    return checksum;
}

/* Parallel prefault: split the page offsets into `n_threads` contiguous
 * chunks, spawn pthreads, each XORs its chunk. Reduces wall-clock when the
 * mapping is multi-GB and memory channels saturate single-threaded reads. */

#ifdef __linux__
typedef struct {
    const uint8_t *bytes;
    size_t         len;
    size_t        *offsets;     /* array of page offsets for this chunk */
    size_t         n_offsets;
    uint8_t        checksum;
} PrefaultChunk;

static void *prefault_worker(void *arg)
{
    PrefaultChunk *c = (PrefaultChunk *)arg;
    uint8_t checksum = 0;
    for (size_t i = 0; i < c->n_offsets; i++) {
        checksum ^= oc_read_volatile_byte(c->bytes, c->len, c->offsets[i]);
    }
    c->checksum = checksum;
    return NULL;
}
#endif

uint8_t oc_mmap_prefault_parallel(const OcMmap *m, size_t n_threads)
{
    if (!m || !m->addr) return 0;
    if (n_threads == 0) n_threads = 1;
#ifdef __linux__
    if (n_threads > 64) n_threads = 64;
    if (n_threads == 1) {
        return oc_mmap_prefault(m);
    }

    const uint8_t *bytes = (const uint8_t *)m->addr;
    size_t len = m->len;

    /* Collect page offsets. */
    size_t n_offsets = (len + 4095) / 4096;
    size_t *offsets = (size_t *)malloc(n_offsets * sizeof(size_t));
    if (!offsets) return oc_mmap_prefault(m);   /* fall back to single-threaded */
    for (size_t i = 0, off = 0; i < n_offsets; i++, off += 4096) {
        offsets[i] = off;
    }

    /* Split into chunks. */
    size_t per_chunk = (n_offsets + n_threads - 1) / n_threads;
    if (per_chunk == 0) per_chunk = 1;

    pthread_t *threads = (pthread_t *)malloc(n_threads * sizeof(pthread_t));
    /* calloc (not malloc) so unused chunks have checksum=0 — the final XOR
     * loop iterates over all `n_threads` chunks, so any uninitialized field
     * would be UB. */
    PrefaultChunk *chunks = (PrefaultChunk *)calloc(n_threads, sizeof(PrefaultChunk));
    if (!threads || !chunks) {
        free(offsets); free(threads); free(chunks);
        return oc_mmap_prefault(m);
    }

    size_t spawned = 0;
    for (size_t t = 0; t < n_threads; t++) {
        size_t start = t * per_chunk;
        if (start >= n_offsets) break;
        size_t end = start + per_chunk;
        if (end > n_offsets) end = n_offsets;
        chunks[t].bytes     = bytes;
        chunks[t].len       = len;
        chunks[t].offsets   = offsets + start;
        chunks[t].n_offsets = end - start;
        chunks[t].checksum  = 0;
        if (pthread_create(&threads[spawned], NULL, prefault_worker, &chunks[t]) == 0) {
            spawned++;
        } else {
            /* Fallback: run this chunk inline. */
            prefault_worker(&chunks[t]);
        }
    }

    uint8_t checksum = 0;
    for (size_t t = 0; t < spawned; t++) {
        pthread_join(threads[t], NULL);
    }
    for (size_t t = 0; t < n_threads; t++) {
        checksum ^= chunks[t].checksum;
    }
    /* Touch last byte (mirrors single-threaded path). */
    if (len > 0) {
        checksum ^= oc_read_volatile_byte(bytes, len, len - 1);
    }

    free(offsets); free(threads); free(chunks);
    return checksum;
#else
    /* Non-Linux: no pthread; fall back to single-threaded. */
    return oc_mmap_prefault(m);
#endif
}

const uint8_t *oc_mmap_bytes(const OcMmap *m)
{
    if (!m) return NULL;
    return (const uint8_t *)m->addr;
}

size_t oc_mmap_len(const OcMmap *m)
{
    if (!m) return 0;
    return m->len;
}

void oc_mmap_close(OcMmap *m)
{
    if (!m) return;
#ifdef __linux__
    if (m->mlocked && m->addr) {
        /* Best-effort unlock; ignore failure (process exit will release). */
        (void)munlock(m->addr, m->len);
    }
    if (m->addr) {
        if (munmap(m->addr, m->len) != 0) {
            oc_log(OC_LOG_WARN, "mmap: munmap(%p, %zu) failed: %s",
                    m->addr, m->len, strerror(errno));
        }
    }
    if (m->fd >= 0) {
        close(m->fd);
    }
#else
    /* Non-Linux fallback: addr was malloc'd. */
    if (m->addr && m->heap_alloc) free(m->addr);
#endif
    free(m);
}
