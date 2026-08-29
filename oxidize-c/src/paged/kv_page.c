/* kv_page.c — paged KV cache management implementation. */
#include "oxidize/kv_page.h"

#include <stdlib.h>
#include <string.h>


OcKvPageConfig oc_kv_page_config_default(void)
{
    OcKvPageConfig c;
    c.page_size  = OC_KV_PAGE_DEFAULT_PAGE_SIZE;
    c.head_dim   = OC_KV_PAGE_DEFAULT_HEAD_DIM;
    c.n_heads    = OC_KV_PAGE_DEFAULT_N_HEADS;
    c.n_layers   = OC_KV_PAGE_DEFAULT_N_LAYERS;
    c.max_pages  = OC_KV_PAGE_DEFAULT_MAX_PAGES;
    return c;
}

size_t oc_kv_page_kv_bytes(const OcKvPageConfig *cfg)
{
    if (!cfg) return 0;
    return (size_t)cfg->page_size *
           (size_t)cfg->n_layers *
           (size_t)cfg->head_dim *
           (size_t)cfg->n_heads *
           sizeof(float);
}


/* Per-slot element count for layer_k / layer_v:
 *   n_layers * page_size * head_dim * n_heads
 */
static size_t page_slot_count(const OcKvPageConfig *cfg)
{
    return (size_t)cfg->n_layers *
           (size_t)cfg->page_size *
           (size_t)cfg->head_dim *
           (size_t)cfg->n_heads;
}

OcError oc_kv_page_init(OcKvPageManager **out, OcKvPageConfig config)
{
    if (!out) return OC_ERR_INVALID_ARG;
    if (config.max_pages == 0 || config.page_size == 0 ||
        config.head_dim == 0 || config.n_heads == 0 ||
        config.n_layers == 0) {
        return OC_ERR_INVALID_ARG;
    }
    OcKvPageManager *mgr = malloc(sizeof(*mgr));
    if (!mgr) return OC_ERR_OOM;
    memset(mgr, 0, sizeof(*mgr));
    mgr->config = config;
    mgr->n_total = config.max_pages;
    mgr->n_free = config.max_pages;

    mgr->pages = calloc(config.max_pages, sizeof(*mgr->pages));
    if (!mgr->pages) {
        free(mgr);
        return OC_ERR_OOM;
    }
    mgr->free_list = malloc((size_t)config.max_pages * sizeof(*mgr->free_list));
    if (!mgr->free_list) {
        free(mgr->pages);
        free(mgr);
        return OC_ERR_OOM;
    }
    /* LIFO: push pages in reverse so page 0 is at the top of the stack. */
    for (uint32_t i = 0; i < config.max_pages; i++) {
        mgr->free_list[i] = config.max_pages - 1 - i;
        mgr->pages[i].page_id = i;
        mgr->pages[i].ref_count = 0;
        mgr->pages[i].seq_id = 0;
        mgr->pages[i].start_pos = 0;
        mgr->pages[i].layer_k = NULL;
        mgr->pages[i].layer_v = NULL;
    }
    mgr->stats.total_pages = config.max_pages;
    mgr->stats.free_pages = config.max_pages;
    mgr->stats.used_pages = 0;
    mgr->stats.allocation_count = 0;
    mgr->stats.eviction_count = 0;
    *out = mgr;
    return OC_OK;
}

void oc_kv_page_free_mgr(OcKvPageManager *mgr)
{
    if (!mgr) return;
    if (mgr->pages) {
        for (uint32_t i = 0; i < mgr->n_total; i++) {
            free(mgr->pages[i].layer_k);
            free(mgr->pages[i].layer_v);
        }
        free(mgr->pages);
    }
    free(mgr->free_list);
    free(mgr);
}

OcError oc_kv_page_alloc(OcKvPageManager *mgr, uint64_t seq_id,
                          uint32_t start_pos, uint32_t *out_page_id)
{
    if (!mgr || !out_page_id) return OC_ERR_INVALID_ARG;
    if (mgr->n_free == 0) return OC_ERR_OOM;
    /* Pop from the top of the LIFO stack. */
    uint32_t idx = mgr->free_list[mgr->n_free - 1];
    mgr->n_free--;
    OcKvPage *p = &mgr->pages[idx];
    /* Lazily allocate K/V storage on first use. */
    if (!p->layer_k) {
        size_t count = page_slot_count(&mgr->config);
        p->layer_k = malloc(count * sizeof(float));
        p->layer_v = malloc(count * sizeof(float));
        if (!p->layer_k || !p->layer_v) {
            free(p->layer_k);
            free(p->layer_v);
            p->layer_k = NULL;
            p->layer_v = NULL;
            /* Push back onto the free list. */
            mgr->free_list[mgr->n_free] = idx;
            mgr->n_free++;
            return OC_ERR_OOM;
        }
        memset(p->layer_k, 0, count * sizeof(float));
        memset(p->layer_v, 0, count * sizeof(float));
    }
    p->ref_count = 1;
    p->seq_id = seq_id;
    p->start_pos = start_pos;
    *out_page_id = idx;
    mgr->stats.free_pages = mgr->n_free;
    mgr->stats.used_pages = mgr->n_total - mgr->n_free;
    mgr->stats.allocation_count++;
    return OC_OK;
}

OcError oc_kv_page_free(OcKvPageManager *mgr, uint32_t page_id)
{
    if (!mgr) return OC_ERR_INVALID_ARG;
    if (page_id >= mgr->n_total) return OC_ERR_INVALID_ARG;
    OcKvPage *p = &mgr->pages[page_id];
    if (p->ref_count == 0) return OC_ERR_INVALID_ARG;
    p->ref_count--;
    if (p->ref_count == 0) {
        p->seq_id = 0;
        p->start_pos = 0;
        /* Return to the free list. */
        mgr->free_list[mgr->n_free] = page_id;
        mgr->n_free++;
        mgr->stats.eviction_count++;
        mgr->stats.free_pages = mgr->n_free;
        mgr->stats.used_pages = mgr->n_total - mgr->n_free;
    }
    return OC_OK;
}

OcKvPage *oc_kv_page_get(OcKvPageManager *mgr, uint32_t page_id)
{
    if (!mgr) return NULL;
    if (page_id >= mgr->n_total) return NULL;
    if (mgr->pages[page_id].ref_count == 0) return NULL;
    return &mgr->pages[page_id];
}

/* Offset into layer_k/layer_v for (layer, token_offset):
 *   layer * (page_size * head_dim * n_heads) + token_offset * (head_dim * n_heads)
 */
static size_t slot_offset(const OcKvPageConfig *cfg, uint32_t layer,
                           uint32_t token_offset)
{
    size_t per_token = (size_t)cfg->head_dim * (size_t)cfg->n_heads;
    return (size_t)layer * (size_t)cfg->page_size * per_token +
           (size_t)token_offset * per_token;
}

static size_t per_token_count(const OcKvPageConfig *cfg)
{
    return (size_t)cfg->head_dim * (size_t)cfg->n_heads;
}

OcError oc_kv_page_write_kv(OcKvPageManager *mgr, uint32_t page_id,
                              uint32_t layer, uint32_t token_offset,
                              const float *key, const float *value)
{
    if (!mgr || !key || !value) return OC_ERR_INVALID_ARG;
    if (page_id >= mgr->n_total) return OC_ERR_INVALID_ARG;
    OcKvPage *p = &mgr->pages[page_id];
    if (p->ref_count == 0 || !p->layer_k || !p->layer_v) {
        return OC_ERR_INVALID_ARG;
    }
    if (layer >= mgr->config.n_layers) return OC_ERR_INVALID_ARG;
    if (token_offset >= mgr->config.page_size) return OC_ERR_INVALID_ARG;
    size_t offset = slot_offset(&mgr->config, layer, token_offset);
    size_t n = per_token_count(&mgr->config);
    memcpy(p->layer_k + offset, key, n * sizeof(float));
    memcpy(p->layer_v + offset, value, n * sizeof(float));
    return OC_OK;
}

OcError oc_kv_page_read_kv(OcKvPageManager *mgr, uint32_t page_id,
                             uint32_t layer, uint32_t token_offset,
                             float *out_key, float *out_value)
{
    if (!mgr || !out_key || !out_value) return OC_ERR_INVALID_ARG;
    if (page_id >= mgr->n_total) return OC_ERR_INVALID_ARG;
    OcKvPage *p = &mgr->pages[page_id];
    if (p->ref_count == 0 || !p->layer_k || !p->layer_v) {
        return OC_ERR_INVALID_ARG;
    }
    if (layer >= mgr->config.n_layers) return OC_ERR_INVALID_ARG;
    if (token_offset >= mgr->config.page_size) return OC_ERR_INVALID_ARG;
    size_t offset = slot_offset(&mgr->config, layer, token_offset);
    size_t n = per_token_count(&mgr->config);
    memcpy(out_key, p->layer_k + offset, n * sizeof(float));
    memcpy(out_value, p->layer_v + offset, n * sizeof(float));
    return OC_OK;
}

OcError oc_kv_page_get_seq_pages(OcKvPageManager *mgr, uint64_t seq_id,
                                  uint32_t *out_page_ids, size_t max,
                                  size_t *out_count)
{
    if (!mgr || !out_count) return OC_ERR_INVALID_ARG;
    if (max > 0 && !out_page_ids) return OC_ERR_INVALID_ARG;
    size_t count = 0;
    for (uint32_t i = 0; i < mgr->n_total && count < max; i++) {
        if (mgr->pages[i].ref_count > 0 && mgr->pages[i].seq_id == seq_id) {
            if (out_page_ids) {
                out_page_ids[count] = i;
            }
            count++;
        }
    }
    *out_count = count;
    return OC_OK;
}

OcError oc_kv_page_stats(const OcKvPageManager *mgr, OcKvPageStats *out_stats)
{
    if (!mgr || !out_stats) return OC_ERR_INVALID_ARG;
    *out_stats = mgr->stats;
    return OC_OK;
}
