#include "oxidize/arena.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OC_ARENA_MAX_CHUNK   (64u * 1024u * 1024u)  /* 64 MiB max chunk */
#define OC_ARENA_MIN_CHUNK   (4096u)
#define OC_ARENA_ALIGN_MAX   256u

typedef struct OcArenaChunk {
    struct OcArenaChunk *next;
    size_t cap;          /* capacity of this chunk's data area              */
    size_t off;          /* current bump offset into data                   */
    /* data follows immediately after this header, aligned to OC_ARENA_ALIGN_MAX.
     * The raw allocation is `sizeof(OcArenaChunk) + cap + OC_ARENA_ALIGN_MAX`
     * so we can always find an aligned base within it. */
} OcArenaChunk;

struct OcArena {
    OcArenaChunk *head;
    OcArenaChunk *tail;
    size_t        total_used;    /* sum of all chunk offs (live bytes)   */
    size_t        next_chunk_cap; /* size of the next chunk to allocate  */
};

/* Return the aligned data base of a chunk. */
static uint8_t *chunk_data(OcArenaChunk *c)
{
    /* Place data immediately after the header, aligned up to the max alignment
     * the arena supports. */
    uintptr_t base = (uintptr_t)(c + 1);
    uintptr_t aligned = (base + OC_ARENA_ALIGN_MAX - 1) & ~((uintptr_t)OC_ARENA_ALIGN_MAX - 1);
    return (uint8_t *)aligned;
}

static size_t round_up_pow2(size_t n)
{
    size_t r = 1;
    while (r < n) {
        if (r > SIZE_MAX / 2) return 0; /* would overflow */
        r <<= 1;
    }
    return r;
}

static size_t align_up(size_t v, size_t align)
{
    if (align == 0 || (align & (align - 1)) != 0) align = 1;
    return (v + align - 1) & ~(align - 1);
}

static OcArenaChunk *chunk_new(size_t cap)
{
    if (cap < OC_ARENA_MIN_CHUNK) cap = OC_ARENA_MIN_CHUNK;
    /* Reject caps that would wrap the malloc size below. */
    if (cap > SIZE_MAX - sizeof(OcArenaChunk) - OC_ARENA_ALIGN_MAX) return NULL;
    /* Allocate extra space for alignment slack (OC_ARENA_ALIGN_MAX bytes). */
    OcArenaChunk *c = (OcArenaChunk *)malloc(sizeof(OcArenaChunk) + cap + OC_ARENA_ALIGN_MAX);
    if (!c) return NULL;
    c->next = NULL;
    c->cap  = cap;
    c->off  = 0;
    return c;
}

OcArena *oc_arena_new(size_t initial_cap)
{
    if (initial_cap == 0) initial_cap = OC_ARENA_DEFAULT_CAP;
    initial_cap = round_up_pow2(initial_cap);
    if (initial_cap == 0) return NULL; /* requested cap too large */
    if (initial_cap < OC_ARENA_MIN_CHUNK) initial_cap = OC_ARENA_MIN_CHUNK;

    OcArena *a = (OcArena *)malloc(sizeof(OcArena));
    if (!a) return NULL;
    a->head = a->tail = NULL;
    a->total_used = 0;
    a->next_chunk_cap = initial_cap;

    OcArenaChunk *c = chunk_new(initial_cap);
    if (!c) { free(a); return NULL; }
    a->head = a->tail = c;
    return a;
}

void *oc_arena_alloc(OcArena *a, size_t n, size_t align)
{
    if (!a) return NULL;
    if (n == 0) n = 1;
    if (align == 0 || (align & (align - 1)) != 0) align = 1;
    if (align > OC_ARENA_ALIGN_MAX) align = OC_ARENA_ALIGN_MAX;

    if (n > SIZE_MAX - align) return NULL; /* n + align below would wrap */

    /* Try the current tail chunk, then any already-allocated chunks after it
     * (present after oc_arena_reset()). Overflow-safe fit check. */
    while (a->tail) {
        OcArenaChunk *c = a->tail;
        size_t aligned_off = align_up(c->off, align);
        if (aligned_off <= c->cap && n <= c->cap - aligned_off) {
            void *p = chunk_data(c) + aligned_off;
            c->off = aligned_off + n;
            a->total_used += n;
            return p;
        }
        if (!c->next) break;
        a->tail = c->next;
    }

    /* Need a new chunk. Size it to fit the allocation with headroom. */
    size_t want = n + align;
    if (want < a->next_chunk_cap) want = a->next_chunk_cap;
    if (want > OC_ARENA_MAX_CHUNK) want = OC_ARENA_MAX_CHUNK;
    /* If even the max chunk can't fit this single allocation, allocate a
     * dedicated oversized chunk. */
    if (want < n + align) want = n + align;

    OcArenaChunk *nc = chunk_new(want);
    if (!nc) return NULL;

    /* Append. */
    if (a->tail) a->tail->next = nc;
    else         a->head = nc;
    a->tail = nc;

    /* Grow the next-chunk cap geometrically for amortized O(1) growth. */
    if (a->next_chunk_cap < OC_ARENA_MAX_CHUNK) {
        a->next_chunk_cap <<= 1;
        if (a->next_chunk_cap == 0 || a->next_chunk_cap > OC_ARENA_MAX_CHUNK)
            a->next_chunk_cap = OC_ARENA_MAX_CHUNK;
    }

    /* Bump in the new chunk. */
    size_t aligned_off = align_up(nc->off, align);
    void *p = chunk_data(nc) + aligned_off;
    nc->off = aligned_off + n;
    a->total_used += n;
    return p;
}

void *oc_arena_alloc_bytes(OcArena *a, size_t n_bytes)
{
    return oc_arena_alloc(a, n_bytes, 16);
}

char *oc_arena_dup(OcArena *a, const char *s)
{
    if (!s) return NULL;
    return oc_arena_dup_n(a, s, strlen(s));
}

char *oc_arena_dup_n(OcArena *a, const char *s, size_t n)
{
    if (!s) return NULL;
    if (n == SIZE_MAX) return NULL; /* n + 1 would wrap */
    char *dst = (char *)oc_arena_alloc(a, n + 1, 1);
    if (!dst) return NULL;
    memcpy(dst, s, n);
    dst[n] = '\0';
    return dst;
}

char *oc_arena_printf(OcArena *a, const char *fmt, ...)
{
    if (!a || !fmt) return NULL;

    va_list ap1, ap2;
    va_start(ap1, fmt);
    va_copy(ap2, ap1);
    int n = vsnprintf(NULL, 0, fmt, ap1);
    va_end(ap1);
    if (n < 0) { va_end(ap2); return NULL; }

    char *dst = (char *)oc_arena_alloc(a, (size_t)n + 1, 1);
    if (!dst) { va_end(ap2); return NULL; }

    vsnprintf(dst, (size_t)n + 1, fmt, ap2);
    va_end(ap2);
    return dst;
}

size_t oc_arena_used(const OcArena *a)
{
    return a ? a->total_used : 0;
}

size_t oc_arena_capacity(const OcArena *a)
{
    if (!a) return 0;
    size_t total = 0;
    for (OcArenaChunk *c = a->head; c; c = c->next) total += c->cap;
    return total;
}

void oc_arena_reset(OcArena *a)
{
    if (!a) return;
    for (OcArenaChunk *c = a->head; c; c = c->next) c->off = 0;
    a->tail = a->head; /* reuse chunks from the start; alloc walks forward */
    a->total_used = 0;
}

void oc_arena_free(OcArena *a)
{
    if (!a) return;
    OcArenaChunk *c = a->head;
    while (c) {
        OcArenaChunk *next = c->next;
        free(c);
        c = next;
    }
    free(a);
}
