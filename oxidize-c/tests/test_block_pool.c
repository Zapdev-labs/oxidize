/* test_block_pool.c — memory block pool allocator tests. */
#define _POSIX_C_SOURCE 200809L
#include "framework.h"
#include "oxidize/block_pool.h"
#include <stdlib.h>
#include <string.h>

Test(block_pool, init_free)
{
    OcBlkPool pool;
    cr_assert_eq(oc_blkpool_init(&pool, 8, 64), OC_OK);
    cr_assert_eq(pool.n_blocks, 8u);
    cr_assert_eq(pool.block_size, 64u);
    cr_assert_eq(pool.n_free, 8u);
    cr_assert_not_null(pool.blocks);
    cr_assert_not_null(pool.free_list);
    cr_assert_not_null(pool.allocated);
    cr_assert_not_null(pool.slab);
    oc_blkpool_free_pool(&pool);
    cr_assert_null(pool.blocks);
    cr_assert_null(pool.free_list);
    cr_assert_null(pool.allocated);
    cr_assert_null(pool.slab);
}

Test(block_pool, init_null)
{
    cr_assert_eq(oc_blkpool_init(NULL, 8, 64), OC_ERR_INVALID_ARG);
}

Test(block_pool, init_invalid_args)
{
    OcBlkPool pool;
    cr_assert_eq(oc_blkpool_init(&pool, 0, 64), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_blkpool_init(&pool, 8, 0), OC_ERR_INVALID_ARG);
}

Test(block_pool, alloc_single)
{
    OcBlkPool pool;
    oc_blkpool_init(&pool, 4, 32);
    uint32_t idx = oc_blkpool_alloc(&pool);
    cr_assert_neq(idx, OC_BLKPOOL_INVALID);
    cr_assert_eq(pool.n_free, 3u);
    cr_assert(oc_blkpool_is_allocated(&pool, idx));
    oc_blkpool_free_pool(&pool);
}

Test(block_pool, alloc_multiple)
{
    OcBlkPool pool;
    oc_blkpool_init(&pool, 4, 32);
    uint32_t a = oc_blkpool_alloc(&pool);
    uint32_t b = oc_blkpool_alloc(&pool);
    uint32_t c = oc_blkpool_alloc(&pool);
    cr_assert_neq(a, OC_BLKPOOL_INVALID);
    cr_assert_neq(b, OC_BLKPOOL_INVALID);
    cr_assert_neq(c, OC_BLKPOOL_INVALID);
    /* Indices should be distinct. */
    cr_assert_neq(a, b);
    cr_assert_neq(b, c);
    cr_assert_neq(a, c);
    cr_assert_eq(pool.n_free, 1u);
    cr_assert_eq(oc_blkpool_n_used(&pool), 3u);
    oc_blkpool_free_pool(&pool);
}

Test(block_pool, alloc_exhaustion)
{
    OcBlkPool pool;
    oc_blkpool_init(&pool, 2, 16);
    uint32_t a = oc_blkpool_alloc(&pool);
    uint32_t b = oc_blkpool_alloc(&pool);
    uint32_t c = oc_blkpool_alloc(&pool);
    cr_assert_neq(a, OC_BLKPOOL_INVALID);
    cr_assert_neq(b, OC_BLKPOOL_INVALID);
    cr_assert_eq(c, OC_BLKPOOL_INVALID);
    cr_assert_eq(pool.n_free, 0u);
    oc_blkpool_free_pool(&pool);
}

Test(block_pool, alloc_on_null)
{
    cr_assert_eq(oc_blkpool_alloc(NULL), OC_BLKPOOL_INVALID);
}

Test(block_pool, free_returns_to_pool)
{
    OcBlkPool pool;
    oc_blkpool_init(&pool, 4, 32);
    uint32_t idx = oc_blkpool_alloc(&pool);
    cr_assert_eq(pool.n_free, 3u);
    cr_assert_eq(oc_blkpool_free(&pool, idx), OC_OK);
    cr_assert_eq(pool.n_free, 4u);
    cr_assert_not(oc_blkpool_is_allocated(&pool, idx));
    /* Can reallocate. */
    uint32_t idx2 = oc_blkpool_alloc(&pool);
    cr_assert_neq(idx2, OC_BLKPOOL_INVALID);
    oc_blkpool_free_pool(&pool);
}

Test(block_pool, free_invalid)
{
    OcBlkPool pool;
    oc_blkpool_init(&pool, 4, 32);
    cr_assert_eq(oc_blkpool_free(&pool, 99), OC_ERR_INVALID_ARG);
    /* Freeing an already-free block. */
    cr_assert_eq(oc_blkpool_free(&pool, 0), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_blkpool_free(NULL, 0), OC_ERR_INVALID_ARG);
    oc_blkpool_free_pool(&pool);
}

Test(block_pool, get_valid)
{
    OcBlkPool pool;
    oc_blkpool_init(&pool, 4, 32);
    uint32_t idx = oc_blkpool_alloc(&pool);
    void *ptr = oc_blkpool_get(&pool, idx);
    cr_assert_not_null(ptr);
    /* Write to the block to verify it is usable. */
    memset(ptr, 0xAB, 32);
    unsigned char *bytes = (unsigned char *)ptr;
    cr_assert_eq(bytes[0], 0xAB);
    cr_assert_eq(bytes[31], 0xAB);
    oc_blkpool_free_pool(&pool);
}

Test(block_pool, get_invalid_returns_null)
{
    OcBlkPool pool;
    oc_blkpool_init(&pool, 4, 32);
    cr_assert_null(oc_blkpool_get(&pool, 99));
    /* Block 0 is free (never allocated). */
    cr_assert_null(oc_blkpool_get(&pool, 0));
    cr_assert_null(oc_blkpool_get(NULL, 0));
    oc_blkpool_free_pool(&pool);
}

Test(block_pool, is_allocated)
{
    OcBlkPool pool;
    oc_blkpool_init(&pool, 4, 32);
    cr_assert_not(oc_blkpool_is_allocated(&pool, 0));
    uint32_t idx = oc_blkpool_alloc(&pool);
    cr_assert(oc_blkpool_is_allocated(&pool, idx));
    cr_assert_not(oc_blkpool_is_allocated(&pool, 99));
    cr_assert_not(oc_blkpool_is_allocated(NULL, 0));
    oc_blkpool_free(&pool, idx);
    cr_assert_not(oc_blkpool_is_allocated(&pool, idx));
    oc_blkpool_free_pool(&pool);
}

Test(block_pool, n_free_and_n_used)
{
    OcBlkPool pool;
    oc_blkpool_init(&pool, 10, 16);
    cr_assert_eq(oc_blkpool_n_free(&pool), 10u);
    cr_assert_eq(oc_blkpool_n_used(&pool), 0u);
    uint32_t a = oc_blkpool_alloc(&pool);
    uint32_t b = oc_blkpool_alloc(&pool);
    (void)a;
    (void)b;
    cr_assert_eq(oc_blkpool_n_free(&pool), 8u);
    cr_assert_eq(oc_blkpool_n_used(&pool), 2u);
    cr_assert_eq(oc_blkpool_n_free(NULL), 0u);
    cr_assert_eq(oc_blkpool_n_used(NULL), 0u);
    oc_blkpool_free_pool(&pool);
}

Test(block_pool, reset)
{
    OcBlkPool pool;
    oc_blkpool_init(&pool, 4, 32);
    oc_blkpool_alloc(&pool);
    oc_blkpool_alloc(&pool);
    cr_assert_eq(pool.n_free, 2u);
    /* Write to a block, then reset and verify it is zeroed. */
    uint32_t idx = oc_blkpool_alloc(&pool);
    void *ptr = oc_blkpool_get(&pool, idx);
    memset(ptr, 0xCD, 32);
    oc_blkpool_reset(&pool);
    cr_assert_eq(pool.n_free, 4u);
    cr_assert_eq(oc_blkpool_n_used(&pool), 0u);
    /* After reset all blocks are free. */
    for (uint32_t i = 0; i < pool.n_blocks; i++) {
        cr_assert_not(oc_blkpool_is_allocated(&pool, i));
    }
    /* Slab should be zeroed. */
    unsigned char *slab = (unsigned char *)pool.slab;
    for (uint32_t i = 0; i < pool.n_blocks * (uint32_t)pool.block_size; i++) {
        cr_assert_eq(slab[i], 0);
    }
    oc_blkpool_free_pool(&pool);
}

Test(block_pool, reset_null_is_safe)
{
    oc_blkpool_reset(NULL);
    cr_assert(true);
}

Test(block_pool, free_pool_null_is_safe)
{
    oc_blkpool_free_pool(NULL);
    cr_assert(true);
}

Test(block_pool, distinct_block_addresses)
{
    OcBlkPool pool;
    oc_blkpool_init(&pool, 4, 32);
    uint32_t a = oc_blkpool_alloc(&pool);
    uint32_t b = oc_blkpool_alloc(&pool);
    void *pa = oc_blkpool_get(&pool, a);
    void *pb = oc_blkpool_get(&pool, b);
    cr_assert_not_null(pa);
    cr_assert_not_null(pb);
    cr_assert_neq(pa, pb);
    /* Blocks do not overlap. */
    cr_assert_geq((size_t)((char *)pb - (char *)pa), 32u);
    oc_blkpool_free_pool(&pool);
}
