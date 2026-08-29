#ifndef OXIDIZE_KV_PAGE_H
#define OXIDIZE_KV_PAGE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif


#define OC_KV_PAGE_DEFAULT_PAGE_SIZE  16
#define OC_KV_PAGE_DEFAULT_HEAD_DIM   128
#define OC_KV_PAGE_DEFAULT_N_HEADS    32
#define OC_KV_PAGE_DEFAULT_N_LAYERS   32
#define OC_KV_PAGE_DEFAULT_MAX_PAGES  4096


typedef struct OcKvPageConfig {
    uint32_t page_size;   /* tokens per page (default 16)                */
    uint32_t head_dim;   /* dimension per head (default 128)            */
    uint32_t n_heads;    /* number of attention heads (default 32)     */
    uint32_t n_layers;   /* number of model layers (default 32)         */
    uint32_t max_pages;  /* total physical pages (default 4096)         */
} OcKvPageConfig;

/* Returns a sensible default config. */
OcKvPageConfig oc_kv_page_config_default(void);


/* Per-page byte size for one of K or V:
 *   page_size * n_layers * head_dim * n_heads * sizeof(float)
 * Total per-page K+V storage is 2x this value. */
size_t oc_kv_page_kv_bytes(const OcKvPageConfig *cfg);

typedef struct OcKvPage {
    uint32_t page_id;     /* index into the manager's pages array         */
    uint32_t ref_count;   /* number of sequences referencing this page    */
    uint64_t seq_id;      /* owning sequence id (0 = free)                */
    uint32_t start_pos;   /* starting token position within the sequence  */
    /* K/V storage: layer_k[layer][token][head_dim * n_heads] */
    float *layer_k;       /* [n_layers * page_size * head_dim * n_heads] */
    float *layer_v;       /* [n_layers * page_size * head_dim * n_heads] */
} OcKvPage;


typedef struct OcKvPageStats {
    uint32_t total_pages;       /* total physical pages              */
    uint32_t free_pages;        /* pages available for allocation    */
    uint32_t used_pages;        /* pages currently allocated        */
    uint64_t allocation_count;  /* cumulative allocations           */
    uint64_t eviction_count;    /* cumulative evictions             */
} OcKvPageStats;


typedef struct OcKvPageManager {
    OcKvPageConfig config;
    OcKvPage *pages;       /* max_pages entries                       */
    uint32_t *free_list;   /* LIFO stack of free page ids            */
    uint32_t n_free;       /* number of pages on the free list       */
    uint32_t n_total;      /* total pages in the pool                */
    OcKvPageStats stats;   /* running counters                       */
} OcKvPageManager;

/* Allocate and initialize a page manager. Returns OC_ERR_OOM on failure.
 * The caller owns the result and must call oc_kv_page_free_mgr. */
OcError oc_kv_page_init(OcKvPageManager **out, OcKvPageConfig config);

/* Allocate a page for a sequence at the given start position.
 * Returns the page_id via out_page_id. Returns OC_ERR_OOM if no pages
 * are available. */
OcError oc_kv_page_alloc(OcKvPageManager *mgr, uint64_t seq_id,
                          uint32_t start_pos, uint32_t *out_page_id);

/* Free a page (decrement ref count). The page returns to the free list
 * when the ref count reaches zero. Returns OC_ERR_INVALID_ARG for an
 * invalid page_id or a page that is already free. */
OcError oc_kv_page_free(OcKvPageManager *mgr, uint32_t page_id);

/* Get a page by ID. Returns NULL for an invalid or free page. */
OcKvPage *oc_kv_page_get(OcKvPageManager *mgr, uint32_t page_id);

/* Write K/V vectors to a page slot at (layer, token_offset).
 * key and value must each have head_dim * n_heads floats.
 * token_offset must be < page_size; layer must be < n_layers. */
OcError oc_kv_page_write_kv(OcKvPageManager *mgr, uint32_t page_id,
                              uint32_t layer, uint32_t token_offset,
                              const float *key, const float *value);

/* Read K/V vectors from a page slot at (layer, token_offset).
 * out_key and out_value must each have head_dim * n_heads floats. */
OcError oc_kv_page_read_kv(OcKvPageManager *mgr, uint32_t page_id,
                             uint32_t layer, uint32_t token_offset,
                             float *out_key, float *out_value);

/* Collect all page ids belonging to a sequence. Writes up to max ids into
 * out_page_ids and the actual count into out_count. */
OcError oc_kv_page_get_seq_pages(OcKvPageManager *mgr, uint64_t seq_id,
                                  uint32_t *out_page_ids, size_t max,
                                  size_t *out_count);

/* Populate out_stats with current statistics. */
OcError oc_kv_page_stats(const OcKvPageManager *mgr, OcKvPageStats *out_stats);

/* Free the page manager and all owned pages. Safe on NULL. */
void oc_kv_page_free_mgr(OcKvPageManager *mgr);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_KV_PAGE_H */
