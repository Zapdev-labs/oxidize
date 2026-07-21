/*
 * paged_attention.h — vLLM-style paged KV cache management + scheduling.
 *
 * Port of oxidize-core/src/paged_attention/. The Rust module provides block
 * allocation, prefix caching, and FCFS scheduling but no attention kernel.
 * This C port adds a paged attention scatter/gather kernel that uses block
 * tables to compute attention over physically scattered KV cache pages.
 *
 * Design:
 *   - BlockPool: fixed pool of physical KV blocks with ref-counting, free
 *     list, and prefix-cache hash map (FNV-1a cumulative token hash).
 *   - BlockTable: per-sequence logical→physical block mapping.
 *   - Scheduler: FCFS queue with prefill chunking, decode, prefix cache hits,
 *     COW on shared decode blocks, and preemption.
 *   - Paged attention kernel: gathers KV from physical blocks via the block
 *     table, computes online softmax attention per head.
 */
#ifndef OXIDIZE_PAGED_ATTENTION_H
#define OXIDIZE_PAGED_ATTENTION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/sampling.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Types ────────────────────────────────────────────────────────────── */

typedef uint32_t OcBlockId;    /* physical block identifier (0..num_blocks-1) */
typedef uint64_t OcBlockHash;  /* FNV-1a cumulative hash                      */
typedef uint64_t OcSeqId;      /* sequence identifier                         */

/* ─── BlockPool ────────────────────────────────────────────────────────── */

typedef struct OcBlockPoolConfig {
    uint32_t block_size;     /* tokens per block (default 16)               */
    uint32_t num_blocks;     /* total physical blocks                       */
    uint32_t num_layers;     /* model layer count                           */
    uint32_t num_kv_heads;   /* KV heads per layer                          */
    uint32_t head_dim;       /* dimension per head                          */
    uint32_t dtype_size;     /* bytes per element (4 = f32)                  */
} OcBlockPoolConfig;

/* Per-block byte size = block_size * num_layers * 2 (K+V) * num_kv_heads
 * * head_dim * dtype_size. */
size_t oc_block_bytes(const OcBlockPoolConfig *cfg);

typedef struct OcPhysicalBlock {
    OcBlockId id;
    uint32_t  ref_count;
    OcBlockHash block_hash;   /* 0 = not in prefix cache                    */
    bool     has_hash;
    uint64_t last_accessed;   /* LRU counter                                 */
} OcPhysicalBlock;

typedef struct OcBlockPool {
    OcBlockPoolConfig config;
    OcPhysicalBlock *blocks;       /* num_blocks entries                     */
    OcBlockId *free_list;          /* LIFO stack                             */
    size_t free_count;
    /* Prefix cache: hash → block_id. Simple linear-probe hash table. */
    OcBlockHash *cache_keys;       /* 0 = empty slot                         */
    OcBlockId   *cache_vals;
    size_t cache_cap;              /* power-of-2                             */
    size_t cache_count;
    uint64_t access_counter;        /* monotonic LRU clock                    */
} OcBlockPool;

OcError oc_block_pool_init(OcBlockPool *pool, const OcBlockPoolConfig *cfg);
void    oc_block_pool_free(OcBlockPool *pool);

OcError oc_block_pool_allocate(OcBlockPool *pool, OcBlockId *out);
OcError oc_block_pool_allocate_n(OcBlockPool *pool, size_t n, OcBlockId *out);
OcError oc_block_pool_free_block(OcBlockPool *pool, OcBlockId id);
OcError oc_block_pool_inc_ref(OcBlockPool *pool, OcBlockId id);
OcError oc_block_pool_dec_ref(OcBlockPool *pool, OcBlockId id);
OcError oc_block_pool_cow(OcBlockPool *pool, OcBlockId id, OcBlockId *new_id,
                          bool *did_copy);

OcBlockHash oc_compute_block_hash(const uint32_t *tokens, size_t n);
OcBlockId oc_block_pool_lookup_prefix(OcBlockPool *pool, OcBlockHash hash);
void oc_block_pool_insert_prefix(OcBlockPool *pool, OcBlockHash hash, OcBlockId id);
bool oc_block_pool_evict_lru(OcBlockPool *pool);

/* ─── BlockTable ───────────────────────────────────────────────────────── */

typedef struct OcBlockTable {
    OcBlockId *logical_to_physical;  /* dynamic array                        */
    size_t num_blocks;               /* logical block count                   */
    size_t capacity;
    uint32_t block_size;
    size_t num_tokens;               /* total tokens mapped                    */
} OcBlockTable;

OcError oc_block_table_init(OcBlockTable *bt, uint32_t block_size);
void    oc_block_table_free(OcBlockTable *bt);
OcError oc_block_table_append_block(OcBlockTable *bt, OcBlockId physical);
bool    oc_block_table_append_token(OcBlockTable *bt);  /* true = needs new block */
OcError oc_block_table_get_slot(const OcBlockTable *bt, size_t token_pos,
                                 OcBlockId *block, uint32_t *slot);
size_t  oc_block_table_blocks_needed(const OcBlockTable *bt, size_t n_tokens);
OcError oc_block_table_truncate(OcBlockTable *bt, size_t n_tokens,
                                OcBlockId *freed, size_t *n_freed);

/* ─── Sequence ──────────────────────────────────────────────────────────── */

typedef enum {
    OC_SEQ_WAITING = 0,
    OC_SEQ_RUNNING = 1,
    OC_SEQ_FINISHED = 2,
} OcSeqStatus;

typedef struct OcPagedSequence {
    OcSeqId seq_id;
    OcSeqStatus status;
    uint32_t *prompt_tokens;
    size_t prompt_len;
    uint32_t *generated_tokens;
    size_t generated_len;
    size_t generated_cap;
    OcBlockTable block_table;
    size_t arrival_order;
    size_t max_new_tokens;
    uint32_t stop_token;       /* 0xFFFFFFFF = no stop                    */
    OcSamplerConfig sampling;
    size_t num_prefilled_tokens;
} OcPagedSequence;

OcError oc_seq_init(OcPagedSequence *seq, OcSeqId id,
                    const uint32_t *prompt, size_t prompt_len,
                    size_t max_new_tokens, uint32_t stop_token,
                    OcSamplerConfig sampling, size_t arrival_order);
void oc_seq_free(OcPagedSequence *seq);
bool oc_seq_is_finished(const OcPagedSequence *seq);
size_t oc_seq_total_tokens(const OcPagedSequence *seq);
OcError oc_seq_append_token(OcPagedSequence *seq, uint32_t token);

/* ─── Scheduler ────────────────────────────────────────────────────────── */

typedef struct OcSchedulerConfig {
    size_t max_num_batched_tokens;  /* default 512                         */
    size_t prefill_chunk_size;       /* default 16                          */
    size_t max_num_running_seqs;    /* default 8                           */
} OcSchedulerConfig;

#define OC_SCHEDULER_DEFAULT ((OcSchedulerConfig){ 512, 16, 8 })

typedef struct OcSchedulerStepResult {
    OcSeqId *scheduled_ids;      /* sequences scheduled this step           */
    size_t n_scheduled;
    size_t *prefill_counts;      /* per-seq prefill token counts           */
    size_t *decode_counts;       /* per-seq decode token counts (0 or 1)   */
    size_t total_tokens;
} OcSchedulerStepResult;

typedef struct OcScheduler {
    OcSchedulerConfig config;
    OcBlockPool block_pool;
    OcPagedSequence **sequences;  /* hash table: seq_id → sequence          */
    size_t seq_cap;
    size_t seq_count;
    OcSeqId *waiting;             /* FCFS queue                             */
    size_t waiting_head, waiting_tail, waiting_cap;
    OcSeqId *running;
    size_t running_count, running_cap;
    size_t next_arrival_order;
} OcScheduler;

OcError oc_scheduler_init(OcScheduler *sched,
                          const OcSchedulerConfig *scfg,
                          const OcBlockPoolConfig *bcfg);
void    oc_scheduler_free(OcScheduler *sched);
OcError oc_scheduler_add_sequence(OcScheduler *sched, OcPagedSequence *seq);
OcError oc_scheduler_step(OcScheduler *sched, OcSchedulerStepResult *out);
OcError oc_scheduler_postprocess(OcScheduler *sched,
                                  const OcSeqId *ids, const uint32_t *tokens,
                                  size_t n);
OcError oc_scheduler_preempt(OcScheduler *sched, OcSeqId id);
void    oc_scheduler_invalidate_prefix_cache(OcScheduler *sched);

/* ─── Paged attention kernel ──────────────────────────────────────────────
 *
 * Computes attention for a single query position, gathering KV cache entries
 * from physical blocks via the block table. Online softmax (numerically
 * stable, single pass).
 *
 *   kv_cache    : flat array containing a complete K region followed by a
 *                 complete V region; each region is
 *                 [num_blocks][block_size][num_kv_heads][head_dim] of f32,
 *                 so allocate 2 * num_blocks * block_size * num_kv_heads *
 *                 head_dim floats
 *   block_table : maps logical → physical blocks
 *   q           : query vector [head_dim]
 *   n_kv_heads  : number of KV heads (for GQA, q_head maps to kv_head = q_head / group_size)
 *   kv_head     : which KV head to attend to
 *   n_past      : number of past tokens to attend to
 *   head_dim    : dimension per head
 *   out         : output [head_dim]
 */
void oc_paged_attention_head(
    const float *kv_cache,
    const OcBlockId *block_table, /* logical → physical                       */
    size_t n_blocks,
    const float *q,               /* [head_dim]                              */
    uint32_t kv_head,
    uint32_t n_kv_heads,
    uint32_t head_dim,
    uint32_t block_size,
    size_t n_past,
    float *out);                  /* [head_dim]                              */

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_PAGED_ATTENTION_H */
