#include "oxidize/block_pool.h"

#include <stdlib.h>
#include <string.h>


OcError oc_blkpool_init(OcBlkPool *pool, uint32_t n_blocks, size_t block_size)
{
    if (!pool) return OC_ERR_INVALID_ARG;
    if (n_blocks == 0 || block_size == 0) return OC_ERR_INVALID_ARG;

    memset(pool, 0, sizeof(*pool));
    pool->n_blocks   = n_blocks;
    pool->block_size = block_size;
    pool->n_free     = n_blocks;

    /* Contiguous backing slab. */
    pool->slab = malloc((size_t)n_blocks * block_size);
    if (!pool->slab) {
        return OC_ERR_OOM;
    }
    memset(pool->slab, 0, (size_t)n_blocks * block_size);

    /* Per-block pointer array. */
    pool->blocks = malloc((size_t)n_blocks * sizeof(*pool->blocks));
    if (!pool->blocks) {
        free(pool->slab);
        pool->slab = NULL;
        return OC_ERR_OOM;
    }
    for (uint32_t i = 0; i < n_blocks; i++) {
        pool->blocks[i] = (char *)pool->slab + (size_t)i * block_size;
    }

    /* LIFO free list: push in reverse so index 0 is on top. */
    pool->free_list = malloc((size_t)n_blocks * sizeof(*pool->free_list));
    if (!pool->free_list) {
        free(pool->blocks);
        pool->blocks = NULL;
        free(pool->slab);
        pool->slab = NULL;
        return OC_ERR_OOM;
    }
    for (uint32_t i = 0; i < n_blocks; i++) {
        pool->free_list[i] = n_blocks - 1 - i;
    }

    /* Allocated flags. */
    pool->allocated = calloc(n_blocks, sizeof(*pool->allocated));
    if (!pool->allocated) {
        free(pool->free_list);
        pool->free_list = NULL;
        free(pool->blocks);
        pool->blocks = NULL;
        free(pool->slab);
        pool->slab = NULL;
        return OC_ERR_OOM;
    }
    return OC_OK;
}

void oc_blkpool_free_pool(OcBlkPool *pool)
{
    if (!pool) return;
    free(pool->allocated);
    free(pool->free_list);
    free(pool->blocks);
    free(pool->slab);
    memset(pool, 0, sizeof(*pool));
}


uint32_t oc_blkpool_alloc(OcBlkPool *pool)
{
    if (!pool || pool->n_free == 0) return OC_BLKPOOL_INVALID;
    uint32_t idx = pool->free_list[--pool->n_free];
    pool->allocated[idx] = true;
    return idx;
}

OcError oc_blkpool_free(OcBlkPool *pool, uint32_t block_idx)
{
    if (!pool) return OC_ERR_INVALID_ARG;
    if (block_idx >= pool->n_blocks) return OC_ERR_INVALID_ARG;
    if (!pool->allocated[block_idx]) return OC_ERR_INVALID_ARG;
    pool->allocated[block_idx] = false;
    pool->free_list[pool->n_free++] = block_idx;
    return OC_OK;
}

void *oc_blkpool_get(const OcBlkPool *pool, uint32_t block_idx)
{
    if (!pool) return NULL;
    if (block_idx >= pool->n_blocks) return NULL;
    if (!pool->allocated[block_idx]) return NULL;
    return pool->blocks[block_idx];
}


void oc_blkpool_reset(OcBlkPool *pool)
{
    if (!pool) return;
    for (uint32_t i = 0; i < pool->n_blocks; i++) {
        pool->allocated[i] = false;
        pool->free_list[i]  = pool->n_blocks - 1 - i;
    }
    pool->n_free = pool->n_blocks;
    memset(pool->slab, 0, (size_t)pool->n_blocks * pool->block_size);
}


uint32_t oc_blkpool_n_free(const OcBlkPool *pool)
{
    if (!pool) return 0;
    return pool->n_free;
}

uint32_t oc_blkpool_n_used(const OcBlkPool *pool)
{
    if (!pool) return 0;
    return pool->n_blocks - pool->n_free;
}

bool oc_blkpool_is_allocated(const OcBlkPool *pool, uint32_t block_idx)
{
    if (!pool) return false;
    if (block_idx >= pool->n_blocks) return false;
    return pool->allocated[block_idx];
}
