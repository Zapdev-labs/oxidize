/* block_pool.h — memory block pool allocator for efficient tensor allocation. */
#ifndef OXIDIZE_BLOCK_POOL_H
#define OXIDIZE_BLOCK_POOL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Sentinel returned by oc_blkpool_alloc when the pool is exhausted. */
#define OC_BLKPOOL_INVALID ((uint32_t)0xFFFFFFFFu)

typedef struct OcBlkPool {
    void   **blocks;      /* n_blocks pointers into the backing slab       */
    uint32_t n_blocks;    /* total blocks in the pool                       */
    size_t   block_size;  /* bytes per block                                */
    uint32_t *free_list;  /* LIFO stack of free block indices              */
    uint32_t  n_free;     /* number of entries on the free list             */
    bool     *allocated;  /* per-block allocation flag                     */
    void     *slab;       /* contiguous backing allocation (owned)          */
} OcBlkPool;

/* Allocate and initialize a pool of n_blocks blocks each of block_size
 * bytes. Returns OC_OK on success, OC_ERR_INVALID_ARG for bad arguments,
 * OC_ERR_OOM on allocation failure. */
OcError oc_blkpool_init(OcBlkPool *pool, uint32_t n_blocks, size_t block_size);

/* Allocate a block from the pool. Returns the block index, or
 * OC_BLKPOOL_INVALID if the pool is exhausted. */
uint32_t oc_blkpool_alloc(OcBlkPool *pool);

/* Free a previously allocated block. Returns OC_ERR_INVALID_ARG for an
 * invalid index or a block that is not allocated. Returns OC_OK on
 * success. */
OcError oc_blkpool_free(OcBlkPool *pool, uint32_t block_idx);

/* Get a pointer to the block at block_idx. Returns NULL for an invalid
 * index (including a free block). */
void *oc_blkpool_get(const OcBlkPool *pool, uint32_t block_idx);

/* Number of free blocks. Returns 0 on NULL. */
uint32_t oc_blkpool_n_free(const OcBlkPool *pool);

/* Number of used blocks. Returns 0 on NULL. */
uint32_t oc_blkpool_n_used(const OcBlkPool *pool);

/* True if block_idx is currently allocated. Returns false on NULL or
 * invalid index. */
bool oc_blkpool_is_allocated(const OcBlkPool *pool, uint32_t block_idx);

/* Mark all blocks as free (does not free memory). */
void oc_blkpool_reset(OcBlkPool *pool);

/* Free the pool's backing memory and zero the struct. Safe on NULL. */
void oc_blkpool_free_pool(OcBlkPool *pool);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_BLOCK_POOL_H */
