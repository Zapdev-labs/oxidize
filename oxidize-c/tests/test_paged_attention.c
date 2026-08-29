/*
 * test_paged_attention.c — paged KV cache management + scheduling tests.
 */
#include "framework.h"

#include "oxidize/paged_attention.h"

#include <stdint.h>
#include <string.h>

/* ─── BlockPool ────────────────────────────────────────────────────────── */

Test(paged, block_pool_alloc_free)
{
    OcBlockPool pool;
    OcBlockPoolConfig cfg = { .block_size = 16, .num_blocks = 8,
        .num_layers = 2, .num_kv_heads = 4, .head_dim = 64, .dtype_size = 4 };
    cr_assert_eq(oc_block_pool_init(&pool, &cfg), OC_OK);
    cr_assert_eq(pool.free_count, 8);

    OcBlockId a, b;
    cr_assert_eq(oc_block_pool_allocate(&pool, &a), OC_OK);
    cr_assert_eq(oc_block_pool_allocate(&pool, &b), OC_OK);
    cr_assert_eq(pool.free_count, 6);
    cr_assert_neq(a, b);

    cr_assert_eq(oc_block_pool_free_block(&pool, a), OC_OK);
    cr_assert_eq(pool.free_count, 7);
    /* Re-allocating should give back block a (LIFO). */
    OcBlockId c;
    cr_assert_eq(oc_block_pool_allocate(&pool, &c), OC_OK);
    cr_assert_eq(c, a);

    oc_block_pool_free(&pool);
}

Test(paged, block_pool_ref_counting)
{
    OcBlockPool pool;
    OcBlockPoolConfig cfg = { .block_size = 16, .num_blocks = 4,
        .num_layers = 1, .num_kv_heads = 1, .head_dim = 64, .dtype_size = 4 };
    cr_assert_eq(oc_block_pool_init(&pool, &cfg), OC_OK);

    OcBlockId id;
    cr_assert_eq(oc_block_pool_allocate(&pool, &id), OC_OK);
    cr_assert_eq(pool.blocks[id].ref_count, 1);

    cr_assert_eq(oc_block_pool_inc_ref(&pool, id), OC_OK);
    cr_assert_eq(pool.blocks[id].ref_count, 2);

    cr_assert_eq(oc_block_pool_dec_ref(&pool, id), OC_OK);
    cr_assert_eq(pool.blocks[id].ref_count, 1);
    cr_assert_eq(pool.free_count, 3); /* not freed yet */

    cr_assert_eq(oc_block_pool_dec_ref(&pool, id), OC_OK);
    cr_assert_eq(pool.blocks[id].ref_count, 0);
    cr_assert_eq(pool.free_count, 4); /* freed */

    oc_block_pool_free(&pool);
}

Test(paged, block_pool_cow)
{
    OcBlockPool pool;
    OcBlockPoolConfig cfg = { .block_size = 16, .num_blocks = 4,
        .num_layers = 1, .num_kv_heads = 1, .head_dim = 64, .dtype_size = 4 };
    cr_assert_eq(oc_block_pool_init(&pool, &cfg), OC_OK);

    OcBlockId id;
    cr_assert_eq(oc_block_pool_allocate(&pool, &id), OC_OK);
    oc_block_pool_inc_ref(&pool, id); /* ref_count = 2 */

    OcBlockId new_id;
    bool did_copy;
    cr_assert_eq(oc_block_pool_cow(&pool, id, &new_id, &did_copy), OC_OK);
    cr_assert(did_copy, "should copy when ref_count > 1");
    cr_assert_neq(new_id, id);
    cr_assert_eq(pool.blocks[id].ref_count, 1);
    cr_assert_eq(pool.blocks[new_id].ref_count, 1);

    /* COW with ref_count=1 should not copy. */
    cr_assert_eq(oc_block_pool_cow(&pool, new_id, &new_id, &did_copy), OC_OK);
    cr_assert(!did_copy, "should not copy when ref_count == 1");

    oc_block_pool_free(&pool);
}

Test(paged, block_pool_prefix_cache)
{
    OcBlockPool pool;
    OcBlockPoolConfig cfg = { .block_size = 16, .num_blocks = 8,
        .num_layers = 1, .num_kv_heads = 1, .head_dim = 64, .dtype_size = 4 };
    cr_assert_eq(oc_block_pool_init(&pool, &cfg), OC_OK);

    uint32_t tokens[] = {1, 2, 3, 4};
    OcBlockHash h = oc_compute_block_hash(tokens, 4);
    cr_assert_neq(h, 0);

    OcBlockId id;
    cr_assert_eq(oc_block_pool_allocate(&pool, &id), OC_OK);
    oc_block_pool_insert_prefix(&pool, h, id);

    OcBlockId found = oc_block_pool_lookup_prefix(&pool, h);
    cr_assert_eq(found, id, "prefix cache hit should return the block");

    /* Miss with different tokens. */
    uint32_t tokens2[] = {5, 6, 7, 8};
    OcBlockHash h2 = oc_compute_block_hash(tokens2, 4);
    cr_assert_neq(h2, h);
    cr_assert_eq(oc_block_pool_lookup_prefix(&pool, h2), (OcBlockId)-1,
                 "should miss for different hash");

    oc_block_pool_free(&pool);
}

Test(paged, allocating_cached_free_block_invalidates_hash)
{
    OcBlockPool pool;
    OcBlockPoolConfig cfg = { .block_size = 4, .num_blocks = 2,
        .num_layers = 1, .num_kv_heads = 1, .head_dim = 2, .dtype_size = 4 };
    cr_assert_eq(oc_block_pool_init(&pool, &cfg), OC_OK);
    OcBlockId id;
    cr_assert_eq(oc_block_pool_allocate(&pool, &id), OC_OK);
    OcBlockHash hash = oc_compute_block_hash((uint32_t[]){1, 2, 3, 4}, 4);
    oc_block_pool_insert_prefix(&pool, hash, id);
    cr_assert_eq(oc_block_pool_dec_ref(&pool, id), OC_OK);
    OcBlockId reused;
    cr_assert_eq(oc_block_pool_allocate(&pool, &reused), OC_OK);
    cr_assert_eq(reused, id);
    cr_assert_eq(oc_block_pool_lookup_prefix(&pool, hash), (OcBlockId)-1);
    oc_block_pool_free(&pool);
}

Test(paged, prefix_insert_rejects_invalid_block_id)
{
    OcBlockPool pool;
    OcBlockPoolConfig cfg = { .block_size = 4, .num_blocks = 2,
        .num_layers = 1, .num_kv_heads = 1, .head_dim = 2, .dtype_size = 4 };
    cr_assert_eq(oc_block_pool_init(&pool, &cfg), OC_OK);
    oc_block_pool_insert_prefix(&pool, 123, 99);
    cr_assert_eq(pool.cache_count, 0);
    oc_block_pool_free(&pool);
}

/* ─── BlockTable ────────────────────────────────────────────────────────── */

Test(paged, block_table_basic)
{
    OcBlockTable bt;
    cr_assert_eq(oc_block_table_init(&bt, 4), OC_OK);
    cr_assert_eq(bt.block_size, 4);
    cr_assert_eq(bt.num_blocks, 0);
    cr_assert_eq(bt.num_tokens, 0);

    /* First token always needs a block (no blocks allocated yet). */
    bool needs;
    needs = oc_block_table_append_token(&bt); cr_assert(needs);
    /* Allocate a block. */
    cr_assert_eq(oc_block_table_append_block(&bt, 10), OC_OK);
    cr_assert_eq(bt.num_blocks, 1);

    /* Tokens 2-4 fit in block 0 (no new block needed). */
    needs = oc_block_table_append_token(&bt); cr_assert(!needs);
    needs = oc_block_table_append_token(&bt); cr_assert(!needs);
    needs = oc_block_table_append_token(&bt); cr_assert(!needs);
    cr_assert_eq(bt.num_tokens, 4);

    /* 5th token: needs block 1. */
    needs = oc_block_table_append_token(&bt); cr_assert(needs);
    cr_assert_eq(bt.num_tokens, 5);

    /* Allocate second block. */
    cr_assert_eq(oc_block_table_append_block(&bt, 20), OC_OK);
    cr_assert_eq(bt.num_blocks, 2);

    /* Check slot mapping. */
    OcBlockId block;
    uint32_t slot;
    cr_assert_eq(oc_block_table_get_slot(&bt, 0, &block, &slot), OC_OK);
    cr_assert_eq(block, 10);
    cr_assert_eq(slot, 0);
    cr_assert_eq(oc_block_table_get_slot(&bt, 3, &block, &slot), OC_OK);
    cr_assert_eq(block, 10);
    cr_assert_eq(slot, 3);
    cr_assert_eq(oc_block_table_get_slot(&bt, 4, &block, &slot), OC_OK);
    cr_assert_eq(block, 20);
    cr_assert_eq(slot, 0);

    oc_block_table_free(&bt);
}

Test(paged, block_table_blocks_needed)
{
    OcBlockTable bt;
    cr_assert_eq(oc_block_table_init(&bt, 16), OC_OK);
    /* 0 tokens, need 16 more → 1 block. */
    cr_assert_eq(oc_block_table_blocks_needed(&bt, 16), 1);
    /* 0 tokens, need 17 → 2 blocks. */
    cr_assert_eq(oc_block_table_blocks_needed(&bt, 17), 2);
    /* Append a block + 16 tokens. */
    oc_block_table_append_block(&bt, 0);
    for (int i = 0; i < 16; i++) oc_block_table_append_token(&bt);
    /* 16 tokens, need 1 more → 1 block (for token 16). */
    cr_assert_eq(oc_block_table_blocks_needed(&bt, 1), 1);
    /* 16 tokens, need 16 more → 1 block (fills block 1). */
    cr_assert_eq(oc_block_table_blocks_needed(&bt, 16), 1);
    oc_block_table_free(&bt);
}

/* ─── Sequence ──────────────────────────────────────────────────────────── */

Test(paged, sequence_basic)
{
    OcPagedSequence seq;
    uint32_t prompt[] = {1, 2, 3, 4, 5};
    OcSamplerConfig scfg = OC_SAMPLER_DEFAULT;
    cr_assert_eq(oc_seq_init(&seq, 42, prompt, 5, 10, 0xFFFF, scfg, 0), OC_OK);
    cr_assert_eq(seq.seq_id, 42);
    cr_assert_eq(seq.prompt_len, 5);
    cr_assert_eq(seq.status, OC_SEQ_WAITING);
    cr_assert(!oc_seq_is_finished(&seq));

    /* Append generated tokens. */
    cr_assert_eq(oc_seq_append_token(&seq, 10), OC_OK);
    cr_assert_eq(seq.generated_len, 1);
    cr_assert_eq(seq.generated_tokens[0], 10);
    cr_assert_eq(oc_seq_total_tokens(&seq), 1); /* num_prefilled=0, gen=1 */

    /* Finish on stop token. */
    cr_assert_eq(oc_seq_append_token(&seq, 0xFFFF), OC_OK);
    cr_assert(oc_seq_is_finished(&seq));

    oc_seq_free(&seq);
}

/* ─── Scheduler ─────────────────────────────────────────────────────────── */

Test(paged, scheduler_single_seq)
{
    OcScheduler sched;
    OcSchedulerConfig scfg = OC_SCHEDULER_DEFAULT;
    OcBlockPoolConfig bcfg = { .block_size = 4, .num_blocks = 32,
        .num_layers = 2, .num_kv_heads = 4, .head_dim = 64, .dtype_size = 4 };
    cr_assert_eq(oc_scheduler_init(&sched, &scfg, &bcfg), OC_OK);

    uint32_t prompt[] = {1, 2, 3, 4, 5, 6, 7, 8};
    OcPagedSequence *seq = malloc(sizeof(OcPagedSequence));
    cr_assert_eq(oc_seq_init(seq, 1, prompt, 8, 4, 0xFFFF,
                             OC_SAMPLER_DEFAULT, 0), OC_OK);
    cr_assert_eq(oc_scheduler_add_sequence(&sched, seq), OC_OK);

    /* Step 1: should prefill up to prefill_chunk_size (16) tokens = all 8. */
    OcSchedulerStepResult res;
    cr_assert_eq(oc_scheduler_step(&sched, &res), OC_OK);
    cr_assert_eq(res.n_scheduled, 1, "one sequence scheduled");
    cr_assert_eq(res.prefill_counts[0], 8, "prefill all 8 prompt tokens");
    cr_assert_eq(res.decode_counts[0], 0);
    cr_assert_eq(res.total_tokens, 8);
    oc_scheduler_step_result_free(&res);

    /* Postprocess: sampled token 10. */
    OcSeqId ids[] = {1};
    uint32_t tokens[] = {10};
    cr_assert_eq(oc_scheduler_postprocess(&sched, ids, tokens, 1), OC_OK);

    /* Step 2: should decode 1 token. */
    cr_assert_eq(oc_scheduler_step(&sched, &res), OC_OK);
    cr_assert_eq(res.n_scheduled, 1);
    cr_assert_eq(res.decode_counts[0], 1, "decode 1 token");
    cr_assert_eq(res.prefill_counts[0], 0);
    oc_scheduler_step_result_free(&res);

    oc_scheduler_free(&sched);
}

Test(paged, scheduler_preemption)
{
    OcScheduler sched;
    OcSchedulerConfig scfg = OC_SCHEDULER_DEFAULT;
    scfg.max_num_running_seqs = 2;
    OcBlockPoolConfig bcfg = { .block_size = 4, .num_blocks = 4,
        .num_layers = 1, .num_kv_heads = 1, .head_dim = 64, .dtype_size = 4 };
    cr_assert_eq(oc_scheduler_init(&sched, &scfg, &bcfg), OC_OK);

    /* Add a sequence, preempt it. */
    uint32_t prompt[] = {1, 2, 3, 4};
    OcPagedSequence *seq = malloc(sizeof(OcPagedSequence));
    cr_assert_eq(oc_seq_init(seq, 1, prompt, 4, 10, 0xFFFF,
                             OC_SAMPLER_DEFAULT, 0), OC_OK);
    oc_scheduler_add_sequence(&sched, seq);

    /* Prefill. */
    OcSchedulerStepResult res;
    oc_scheduler_step(&sched, &res);
    oc_scheduler_step_result_free(&res);

    /* Preempt. */
    cr_assert_eq(oc_scheduler_preempt(&sched, 1), OC_OK);
    /* Sequence should be back in waiting. */
    cr_assert_eq(seq->status, OC_SEQ_WAITING);
    cr_assert_eq(seq->num_prefilled_tokens, 0);

    oc_scheduler_free(&sched);
}

Test(paged, scheduler_uses_pool_block_size_and_limits_partial_prefill)
{
    OcScheduler sched;
    OcSchedulerConfig scfg = OC_SCHEDULER_DEFAULT;
    OcBlockPoolConfig bcfg = { .block_size = 4, .num_blocks = 1,
        .num_layers = 1, .num_kv_heads = 1, .head_dim = 2, .dtype_size = 4 };
    cr_assert_eq(oc_scheduler_init(&sched, &scfg, &bcfg), OC_OK);
    uint32_t prompt[] = {1, 2, 3, 4, 5, 6, 7, 8};
    OcPagedSequence *seq = malloc(sizeof(*seq));
    cr_assert_eq(oc_seq_init(seq, 1, prompt, 8, 4, UINT32_MAX,
                             OC_SAMPLER_DEFAULT, 0), OC_OK);
    cr_assert_eq(oc_scheduler_add_sequence(&sched, seq), OC_OK);
    cr_assert_eq(seq->block_table.block_size, 4);
    OcSchedulerStepResult res;
    cr_assert_eq(oc_scheduler_step(&sched, &res), OC_OK);
    cr_assert_eq(res.prefill_counts[0], 4);
    cr_assert_eq(seq->num_prefilled_tokens, 4);
    cr_assert_eq(seq->block_table.num_tokens, 4);
    cr_assert_eq(seq->block_table.num_blocks, 1);
    oc_scheduler_step_result_free(&res);
    oc_scheduler_free(&sched);
}

Test(paged, scheduler_grows_full_waiting_queue)
{
    OcScheduler sched;
    OcSchedulerConfig scfg = OC_SCHEDULER_DEFAULT;
    scfg.max_num_running_seqs = 1;
    OcBlockPoolConfig bcfg = { .block_size = 4, .num_blocks = 32,
        .num_layers = 1, .num_kv_heads = 1, .head_dim = 2, .dtype_size = 4 };
    cr_assert_eq(oc_scheduler_init(&sched, &scfg, &bcfg), OC_OK);
    size_t initial_capacity = sched.waiting_cap;
    uint32_t prompt = 1;
    for (size_t i = 0; i <= initial_capacity; i++) {
        OcPagedSequence *seq = malloc(sizeof(*seq));
        cr_assert_eq(oc_seq_init(seq, i + 1, &prompt, 1, 1, UINT32_MAX,
                                 OC_SAMPLER_DEFAULT, i), OC_OK);
        cr_assert_eq(oc_scheduler_add_sequence(&sched, seq), OC_OK);
    }
    cr_assert_gt(sched.waiting_cap, initial_capacity);
    cr_assert_eq(sched.waiting_count, initial_capacity + 1);
    OcSchedulerStepResult res;
    cr_assert_eq(oc_scheduler_step(&sched, &res), OC_OK);
    cr_assert_eq(res.scheduled_ids[0], 1);
    cr_assert_eq(sched.waiting_count, initial_capacity);
    oc_scheduler_step_result_free(&res);
    oc_scheduler_free(&sched);
}

Test(paged, finished_sequence_releases_running_slot)
{
    OcScheduler sched;
    OcSchedulerConfig scfg = OC_SCHEDULER_DEFAULT;
    scfg.max_num_running_seqs = 1;
    OcBlockPoolConfig bcfg = { .block_size = 4, .num_blocks = 4,
        .num_layers = 1, .num_kv_heads = 1, .head_dim = 2, .dtype_size = 4 };
    cr_assert_eq(oc_scheduler_init(&sched, &scfg, &bcfg), OC_OK);
    uint32_t prompt[] = {1};
    for (OcSeqId id = 1; id <= 2; id++) {
        OcPagedSequence *seq = malloc(sizeof(*seq));
        cr_assert_eq(oc_seq_init(seq, id, prompt, 1, 1, 10,
                                 OC_SAMPLER_DEFAULT, (size_t)id), OC_OK);
        cr_assert_eq(oc_scheduler_add_sequence(&sched, seq), OC_OK);
    }
    OcSchedulerStepResult res;
    cr_assert_eq(oc_scheduler_step(&sched, &res), OC_OK);
    oc_scheduler_step_result_free(&res);
    cr_assert_eq(oc_scheduler_postprocess(&sched, (OcSeqId[]){1},
                                          (uint32_t[]){10}, 1), OC_OK);
    cr_assert_eq(sched.running_count, 0);
    cr_assert_eq(oc_scheduler_step(&sched, &res), OC_OK);
    cr_assert_eq(res.scheduled_ids[0], 2);
    oc_scheduler_step_result_free(&res);
    oc_scheduler_free(&sched);
}

Test(paged, decode_reserves_one_position_and_preserves_boundary_page)
{
    OcScheduler sched;
    OcSchedulerConfig scfg = OC_SCHEDULER_DEFAULT;
    OcBlockPoolConfig bcfg = { .block_size = 4, .num_blocks = 4,
        .num_layers = 1, .num_kv_heads = 1, .head_dim = 2, .dtype_size = 4 };
    cr_assert_eq(oc_scheduler_init(&sched, &scfg, &bcfg), OC_OK);
    uint32_t prompt[] = {1, 2, 3, 4};
    OcPagedSequence *seq = malloc(sizeof(*seq));
    cr_assert_eq(oc_seq_init(seq, 1, prompt, 4, 4, UINT32_MAX,
                             OC_SAMPLER_DEFAULT, 0), OC_OK);
    cr_assert_eq(oc_scheduler_add_sequence(&sched, seq), OC_OK);
    OcSchedulerStepResult res;
    cr_assert_eq(oc_scheduler_step(&sched, &res), OC_OK);
    oc_scheduler_step_result_free(&res);
    OcBlockId first = seq->block_table.logical_to_physical[0];
    cr_assert_eq(oc_block_pool_inc_ref(&sched.block_pool, first), OC_OK);
    cr_assert_eq(oc_scheduler_postprocess(&sched, (OcSeqId[]){1},
                                          (uint32_t[]){10}, 1), OC_OK);
    cr_assert_eq(oc_scheduler_step(&sched, &res), OC_OK);
    cr_assert_eq(seq->block_table.num_tokens, 5);
    cr_assert_eq(seq->block_table.num_blocks, 2);
    cr_assert_eq(seq->block_table.logical_to_physical[0], first);
    cr_assert_eq(sched.block_pool.blocks[first].ref_count, 2);
    oc_scheduler_step_result_free(&res);
    cr_assert_eq(oc_scheduler_postprocess(&sched, (OcSeqId[]){1},
                                          (uint32_t[]){11}, 1), OC_OK);
    cr_assert_eq(seq->block_table.num_tokens, 5);
    cr_assert_eq(oc_block_pool_dec_ref(&sched.block_pool, first), OC_OK);
    oc_scheduler_free(&sched);
}

Test(paged, decode_reports_copy_on_write_for_shared_partial_page)
{
    OcScheduler sched;
    OcSchedulerConfig scfg = OC_SCHEDULER_DEFAULT;
    OcBlockPoolConfig bcfg = { .block_size = 4, .num_blocks = 4,
        .num_layers = 1, .num_kv_heads = 1, .head_dim = 2, .dtype_size = 4 };
    cr_assert_eq(oc_scheduler_init(&sched, &scfg, &bcfg), OC_OK);
    uint32_t prompt[] = {1, 2, 3};
    OcPagedSequence *seq = malloc(sizeof(*seq));
    cr_assert_eq(oc_seq_init(seq, 1, prompt, 3, 4, UINT32_MAX,
                             OC_SAMPLER_DEFAULT, 0), OC_OK);
    cr_assert_eq(oc_scheduler_add_sequence(&sched, seq), OC_OK);
    OcSchedulerStepResult res;
    cr_assert_eq(oc_scheduler_step(&sched, &res), OC_OK);
    oc_scheduler_step_result_free(&res);
    OcBlockId shared = seq->block_table.logical_to_physical[0];
    cr_assert_eq(oc_block_pool_inc_ref(&sched.block_pool, shared), OC_OK);
    cr_assert_eq(oc_scheduler_postprocess(&sched, (OcSeqId[]){1},
                                          (uint32_t[]){10}, 1), OC_OK);
    cr_assert_eq(oc_scheduler_step(&sched, &res), OC_OK);
    cr_assert_eq(res.n_cow_copies, 1);
    cr_assert_eq(res.cow_src_blocks[0], shared);
    cr_assert_neq(res.cow_dst_blocks[0], shared);
    cr_assert_eq(seq->block_table.logical_to_physical[0], res.cow_dst_blocks[0]);
    oc_scheduler_step_result_free(&res);
    cr_assert_eq(oc_block_pool_dec_ref(&sched.block_pool, shared), OC_OK);
    oc_scheduler_free(&sched);
}

/* ─── Block hash ────────────────────────────────────────────────────────── */

Test(paged, block_hash_deterministic)
{
    uint32_t tokens[] = {1, 2, 3};
    OcBlockHash h1 = oc_compute_block_hash(tokens, 3);
    OcBlockHash h2 = oc_compute_block_hash(tokens, 3);
    cr_assert_eq(h1, h2, "same tokens → same hash");

    uint32_t tokens2[] = {1, 2, 4};
    OcBlockHash h3 = oc_compute_block_hash(tokens2, 3);
    cr_assert_neq(h1, h3, "different tokens → different hash");

    /* Cumulative: prefix [1,2] vs [1,2,3] should differ. */
    OcBlockHash h_prefix = oc_compute_block_hash(tokens, 2);
    cr_assert_neq(h_prefix, h1, "prefix hash differs from full hash");
}

/* ─── Paged attention kernel ────────────────────────────────────────────── */

Test(paged, attention_kernel_basic)
{
    /* 2 blocks × 4 slots × 1 kv_head × 2 head_dim.
     * K cache: block 0 = [[1,0],[0,1],[0,0],[0,0]]
     *          block 1 = [[0,0],[0,0],[0,0],[0,0]]
     * V cache: block 0 = [[10,0],[0,20],[0,0],[0,0]]
     *          block 1 = [[0,0],[0,0],[0,0],[0,0]]
     * Query = [1, 0] (head_dim=2)
     * Only token 0 has non-zero K·Q dot product → output should be V[0] = [10, 0]. */
    uint32_t block_size = 4, n_kv_heads = 1, head_dim = 2;
    size_t block_stride = block_size * n_kv_heads * head_dim; /* 8 */
    size_t n_blocks = 2;
    float kv[32]; /* 2 blocks × 2 (K+V) × 4 slots × 1 head × 2 dim = 32 */
    memset(kv, 0, sizeof(kv));

    /* K for block 0, slot 0: [1, 0] */
    kv[0] = 1.0f; kv[1] = 0.0f;
    /* K for block 0, slot 1: [0, 1] */
    kv[8] = 0.0f; /* wait, slot stride = n_kv_heads * head_dim = 2 */
    /* Actually layout: kv[block * block_stride + slot * slot_stride + kv_head * head_dim + d]
     * block_stride = 4 * 1 * 2 = 8
     * slot_stride = 1 * 2 = 2
     * K block 0: [1,0, 0,1, 0,0, 0,0] at offset 0
     * V block 0: [10,0, 0,20, 0,0, 0,0] at offset 16 (after K for 2 blocks)
     */
    /* K block 0 slot 0 = [1, 0] */
    kv[0 * 8 + 0 * 2 + 0 * 2 + 0] = 1.0f;
    kv[0 * 8 + 0 * 2 + 0 * 2 + 1] = 0.0f;
    /* K block 0 slot 1 = [0, 1] */
    kv[0 * 8 + 1 * 2 + 0 * 2 + 0] = 0.0f;
    kv[0 * 8 + 1 * 2 + 0 * 2 + 1] = 1.0f;

    /* V block 0 slot 0 = [10, 0] */
    size_t total_k = n_blocks * block_stride; /* 16 */
    kv[total_k + 0 * 8 + 0 * 2 + 0 * 2 + 0] = 10.0f;
    kv[total_k + 0 * 8 + 0 * 2 + 0 * 2 + 1] = 0.0f;
    /* V block 0 slot 1 = [0, 20] */
    kv[total_k + 0 * 8 + 1 * 2 + 0 * 2 + 0] = 0.0f;
    kv[total_k + 0 * 8 + 1 * 2 + 0 * 2 + 1] = 20.0f;

    OcBlockId block_table[] = {0, 1};
    float q[] = {1.0f, 0.0f};
    float out[2] = {0.0f, 0.0f};

    /* Attend to 2 past tokens (slot 0 and 1 in block 0). */
    oc_paged_attention_head(kv, block_table, 2, q, 0, 1, 2, 4, 2, out);

    /* Q·K[0] = 1*1 + 0*0 = 1 → exp(1/sqrt(2))
     * Q·K[1] = 1*0 + 0*1 = 0 → exp(0/sqrt(2))
     * softmax: weight[0] = e^0.707/(e^0.707+1) ≈ 0.669
     *          weight[1] = 1/(e^0.707+1) ≈ 0.331
     * out = 0.669 * [10, 0] + 0.331 * [0, 20] = [6.69, 6.62] */
    cr_assert_float_eq(out[0], 10.0f * 0.669f, 0.1f, "out[0] approx 6.69");
    cr_assert_float_eq(out[1], 20.0f * 0.331f, 0.1f, "out[1] approx 6.62");
}
