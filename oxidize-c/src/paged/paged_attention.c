/*
 * paged_attention.c — vLLM-style paged KV cache management + scheduling.
 *
 * Port of oxidize-core/src/paged_attention/. Implements:
 *   - BlockPool: physical block allocator with ref-counting + prefix cache
 *   - BlockTable: per-sequence logical→physical mapping
 *   - Scheduler: FCFS with prefill chunking, decode, prefix cache, COW
 *   - Paged attention kernel: scatter/gather KV from physical blocks
 */
#include "oxidize/paged_attention.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ─── Helpers ──────────────────────────────────────────────────────────── */

static void *xcalloc(size_t n, size_t sz)
{
    void *p = calloc(n, sz);
    return p;
}

static void *xrealloc(void *ptr, size_t n, size_t sz)
{
    return realloc(ptr, n * sz);
}

/* ─── BlockPoolConfig ──────────────────────────────────────────────────── */

size_t oc_block_bytes(const OcBlockPoolConfig *cfg)
{
    return (size_t)cfg->block_size * cfg->num_layers * 2 *
           cfg->num_kv_heads * cfg->head_dim * cfg->dtype_size;
}

/* ─── BlockPool ────────────────────────────────────────────────────────── */

OcError oc_block_pool_init(OcBlockPool *pool, const OcBlockPoolConfig *cfg)
{
    if (!pool || !cfg || cfg->num_blocks == 0 || cfg->block_size == 0)
        return OC_ERR_INVALID_ARG;
    pool->config = *cfg;
    pool->blocks = xcalloc(cfg->num_blocks, sizeof(OcPhysicalBlock));
    pool->free_list = xcalloc(cfg->num_blocks, sizeof(OcBlockId));
    if (!pool->blocks || !pool->free_list) {
        free(pool->blocks); free(pool->free_list);
        return OC_ERR_OOM;
    }
    /* LIFO: push in reverse so pop gives 0, 1, 2, ... */
    for (size_t i = 0; i < cfg->num_blocks; i++) {
        pool->blocks[i].id = (OcBlockId)i;
        pool->blocks[i].ref_count = 0;
        pool->blocks[i].has_hash = false;
        pool->blocks[i].block_hash = 0;
        pool->blocks[i].last_accessed = 0;
        pool->free_list[cfg->num_blocks - 1 - i] = (OcBlockId)i;
    }
    pool->free_count = cfg->num_blocks;
    /* Prefix cache: power-of-2 hash table, ~2x num_blocks. */
    pool->cache_cap = 1;
    while (pool->cache_cap < cfg->num_blocks * 2) pool->cache_cap <<= 1;
    if (pool->cache_cap < 16) pool->cache_cap = 16;
    pool->cache_keys = xcalloc(pool->cache_cap, sizeof(OcBlockHash));
    pool->cache_vals = xcalloc(pool->cache_cap, sizeof(OcBlockId));
    if (!pool->cache_keys || !pool->cache_vals) {
        free(pool->blocks); free(pool->free_list);
        free(pool->cache_keys); free(pool->cache_vals);
        return OC_ERR_OOM;
    }
    pool->cache_count = 0;
    pool->access_counter = 0;
    return OC_OK;
}

void oc_block_pool_free(OcBlockPool *pool)
{
    if (!pool) return;
    free(pool->blocks); free(pool->free_list);
    free(pool->cache_keys); free(pool->cache_vals);
    memset(pool, 0, sizeof(*pool));
}

OcError oc_block_pool_allocate(OcBlockPool *pool, OcBlockId *out)
{
    if (!pool || !out) return OC_ERR_INVALID_ARG;
    if (pool->free_count == 0) return OC_ERR_OOM;
    pool->free_count--;
    OcBlockId id = pool->free_list[pool->free_count];
    if (pool->blocks[id].has_hash) {
        for (size_t i = 0; i < pool->cache_cap; i++) {
            if (pool->cache_keys[i] != 0 && pool->cache_vals[i] == id) {
                pool->cache_keys[i] = 0;
                pool->cache_count--;
            }
        }
    }
    pool->blocks[id].ref_count = 1;
    pool->blocks[id].has_hash = false;
    pool->blocks[id].block_hash = 0;
    *out = id;
    return OC_OK;
}

OcError oc_block_pool_allocate_n(OcBlockPool *pool, size_t n, OcBlockId *out)
{
    if (!pool || !out) return OC_ERR_INVALID_ARG;
    if (n == 0) return OC_OK;
    if (pool->free_count < n) return OC_ERR_OOM;
    for (size_t i = 0; i < n; i++) {
        OcError e = oc_block_pool_allocate(pool, &out[i]);
        if (e != OC_OK) {
            /* Rollback. */
            for (size_t j = 0; j < i; j++)
                oc_block_pool_free_block(pool, out[j]);
            return e;
        }
    }
    return OC_OK;
}

OcError oc_block_pool_free_block(OcBlockPool *pool, OcBlockId id)
{
    if (!pool || id >= pool->config.num_blocks) return OC_ERR_INVALID_ARG;
    pool->blocks[id].ref_count = 0;
    pool->free_list[pool->free_count++] = id;
    return OC_OK;
}

OcError oc_block_pool_inc_ref(OcBlockPool *pool, OcBlockId id)
{
    if (!pool || id >= pool->config.num_blocks) return OC_ERR_INVALID_ARG;
    pool->blocks[id].ref_count++;
    return OC_OK;
}

OcError oc_block_pool_dec_ref(OcBlockPool *pool, OcBlockId id)
{
    if (!pool || id >= pool->config.num_blocks) return OC_ERR_INVALID_ARG;
    if (pool->blocks[id].ref_count == 0) return OC_ERR_INVALID_ARG;
    pool->blocks[id].ref_count--;
    if (pool->blocks[id].ref_count == 0) {
        pool->free_list[pool->free_count++] = id;
    }
    return OC_OK;
}

OcError oc_block_pool_cow(OcBlockPool *pool, OcBlockId id,
                          OcBlockId *new_id, bool *did_copy)
{
    if (!pool || !new_id || !did_copy) return OC_ERR_INVALID_ARG;
    if (id >= pool->config.num_blocks) return OC_ERR_INVALID_ARG;
    *did_copy = false;
    *new_id = id;
    if (pool->blocks[id].ref_count <= 1) return OC_OK; /* no copy needed */
    /* Allocate a fresh block and dec-ref the original. */
    OcError e = oc_block_pool_allocate(pool, new_id);
    if (e != OC_OK) return e;
    oc_block_pool_dec_ref(pool, id);
    *did_copy = true;
    return OC_OK;
}

/* FNV-1a cumulative hash. */
OcBlockHash oc_compute_block_hash(const uint32_t *tokens, size_t n)
{
    OcBlockHash h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= (OcBlockHash)tokens[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

/* Linear-probe hash lookup in the prefix cache. */
static size_t cache_slot(const OcBlockPool *pool, OcBlockHash key)
{
    /* Simple modulo hash; key=0 means empty, so we skip 0 keys. */
    if (key == 0) key = 1;
    return (size_t)(key & (pool->cache_cap - 1));
}

OcBlockId oc_block_pool_lookup_prefix(OcBlockPool *pool, OcBlockHash hash)
{
    if (!pool || hash == 0) return (OcBlockId)-1;
    size_t start = cache_slot(pool, hash);
    for (size_t i = 0; i < pool->cache_cap; i++) {
        size_t idx = (start + i) & (pool->cache_cap - 1);
        if (pool->cache_keys[idx] == 0) continue;
        if (pool->cache_keys[idx] == hash) {
            OcBlockId bid = pool->cache_vals[idx];
            if (pool->blocks[bid].ref_count > 0) {
                pool->blocks[bid].last_accessed = ++pool->access_counter;
                return bid;
            }
            /* Stale entry: remove it. */
            pool->cache_keys[idx] = 0;
            pool->cache_count--;
            return (OcBlockId)-1;
        }
    }
    return (OcBlockId)-1;
}

void oc_block_pool_insert_prefix(OcBlockPool *pool, OcBlockHash hash, OcBlockId id)
{
    if (!pool || hash == 0 || id >= pool->config.num_blocks) return;
    if (pool->blocks[id].ref_count == 0) return;
    /* Evict if > 75% full. */
    if (pool->cache_count * 4 > pool->cache_cap * 3) {
        oc_block_pool_evict_lru(pool);
    }
    size_t start = cache_slot(pool, hash);
    for (size_t i = 0; i < pool->cache_cap; i++) {
        size_t idx = (start + i) & (pool->cache_cap - 1);
        if (pool->cache_keys[idx] == 0) {
            pool->cache_keys[idx] = hash;
            pool->cache_vals[idx] = id;
            pool->cache_count++;
            pool->blocks[id].block_hash = hash;
            pool->blocks[id].has_hash = true;
            pool->blocks[id].last_accessed = ++pool->access_counter;
            return;
        }
        if (pool->cache_keys[idx] == hash) return; /* already cached */
    }
}

bool oc_block_pool_evict_lru(OcBlockPool *pool)
{
    if (!pool || pool->cache_count == 0) return false;
    uint64_t min_access = UINT64_MAX;
    size_t min_idx = (size_t)-1;
    for (size_t i = 0; i < pool->cache_cap; i++) {
        if (pool->cache_keys[i] == 0) continue;
        OcBlockId bid = pool->cache_vals[i];
        if (pool->blocks[bid].ref_count > 0) continue; /* can't evict in-use */
        if (pool->blocks[bid].last_accessed < min_access) {
            min_access = pool->blocks[bid].last_accessed;
            min_idx = i;
        }
    }
    if (min_idx == (size_t)-1) return false;
    pool->blocks[pool->cache_vals[min_idx]].has_hash = false;
    pool->cache_keys[min_idx] = 0;
    pool->cache_count--;
    return true;
}

/* ─── BlockTable ────────────────────────────────────────────────────────── */

OcError oc_block_table_init(OcBlockTable *bt, uint32_t block_size)
{
    if (!bt || block_size == 0) return OC_ERR_INVALID_ARG;
    bt->block_size = block_size;
    bt->num_blocks = 0;
    bt->capacity = 4;
    bt->logical_to_physical = xcalloc(bt->capacity, sizeof(OcBlockId));
    bt->num_tokens = 0;
    if (!bt->logical_to_physical) return OC_ERR_OOM;
    return OC_OK;
}

void oc_block_table_free(OcBlockTable *bt)
{
    if (!bt) return;
    free(bt->logical_to_physical);
    memset(bt, 0, sizeof(*bt));
}

OcError oc_block_table_append_block(OcBlockTable *bt, OcBlockId physical)
{
    if (!bt) return OC_ERR_INVALID_ARG;
    if (bt->num_blocks >= bt->capacity) {
        size_t new_cap = bt->capacity * 2;
        OcBlockId *np = xrealloc(bt->logical_to_physical, new_cap, sizeof(OcBlockId));
        if (!np) return OC_ERR_OOM;
        bt->logical_to_physical = np;
        bt->capacity = new_cap;
    }
    bt->logical_to_physical[bt->num_blocks++] = physical;
    return OC_OK;
}

bool oc_block_table_append_token(OcBlockTable *bt)
{
    if (!bt) return false;
    size_t block_pos = bt->num_tokens / bt->block_size;
    bt->num_tokens++;
    return (block_pos >= bt->num_blocks); /* need a new block */
}

OcError oc_block_table_get_slot(const OcBlockTable *bt, size_t token_pos,
                                 OcBlockId *block, uint32_t *slot)
{
    if (!bt || !block || !slot) return OC_ERR_INVALID_ARG;
    if (token_pos >= bt->num_tokens) return OC_ERR_INVALID_ARG;
    size_t logical = token_pos / bt->block_size;
    if (logical >= bt->num_blocks) return OC_ERR_INVALID_ARG;
    *block = bt->logical_to_physical[logical];
    *slot = (uint32_t)(token_pos % bt->block_size);
    return OC_OK;
}

size_t oc_block_table_blocks_needed(const OcBlockTable *bt, size_t n_tokens)
{
    if (!bt || n_tokens == 0) return 0;
    size_t total = bt->num_tokens + n_tokens;
    size_t needed = (total + bt->block_size - 1) / bt->block_size;
    return (needed > bt->num_blocks) ? (needed - bt->num_blocks) : 0;
}

OcError oc_block_table_truncate(OcBlockTable *bt, size_t n_tokens,
                                OcBlockId *freed, size_t *n_freed)
{
    if (!bt) return OC_ERR_INVALID_ARG;
    size_t new_blocks = (n_tokens + bt->block_size - 1) / bt->block_size;
    if (new_blocks > bt->num_blocks) new_blocks = bt->num_blocks;
    if (freed && n_freed) {
        *n_freed = bt->num_blocks - new_blocks;
        for (size_t i = new_blocks; i < bt->num_blocks; i++)
            freed[i - new_blocks] = bt->logical_to_physical[i];
    }
    bt->num_blocks = new_blocks;
    bt->num_tokens = n_tokens;
    return OC_OK;
}

/* ─── Sequence ──────────────────────────────────────────────────────────── */

OcError oc_seq_init(OcPagedSequence *seq, OcSeqId id,
                    const uint32_t *prompt, size_t prompt_len,
                    size_t max_new_tokens, uint32_t stop_token,
                    OcSamplerConfig sampling, size_t arrival_order)
{
    if (!seq || !prompt || prompt_len == 0) return OC_ERR_INVALID_ARG;
    memset(seq, 0, sizeof(*seq));
    seq->seq_id = id;
    seq->status = OC_SEQ_WAITING;
    seq->prompt_tokens = xcalloc(prompt_len, sizeof(uint32_t));
    if (!seq->prompt_tokens) return OC_ERR_OOM;
    memcpy(seq->prompt_tokens, prompt, prompt_len * sizeof(uint32_t));
    seq->prompt_len = prompt_len;
    seq->generated_cap = max_new_tokens > 0 ? max_new_tokens : 256;
    seq->generated_tokens = xcalloc(seq->generated_cap, sizeof(uint32_t));
    if (!seq->generated_tokens) { free(seq->prompt_tokens); return OC_ERR_OOM; }
    seq->generated_len = 0;
    OcError e = oc_block_table_init(&seq->block_table, 16);
    if (e != OC_OK) { free(seq->prompt_tokens); free(seq->generated_tokens); return e; }
    seq->arrival_order = arrival_order;
    seq->max_new_tokens = max_new_tokens;
    seq->stop_token = stop_token;
    seq->sampling = sampling;
    seq->num_prefilled_tokens = 0;
    return OC_OK;
}

void oc_seq_free(OcPagedSequence *seq)
{
    if (!seq) return;
    free(seq->prompt_tokens);
    free(seq->generated_tokens);
    oc_block_table_free(&seq->block_table);
    memset(seq, 0, sizeof(*seq));
}

bool oc_seq_is_finished(const OcPagedSequence *seq)
{
    if (!seq) return true;
    if (seq->status == OC_SEQ_FINISHED) return true;
    if (seq->max_new_tokens > 0 && seq->generated_len >= seq->max_new_tokens)
        return true;
    if (seq->generated_len > 0 &&
        seq->generated_tokens[seq->generated_len - 1] == seq->stop_token)
        return true;
    return false;
}

size_t oc_seq_total_tokens(const OcPagedSequence *seq)
{
    return seq->num_prefilled_tokens + seq->generated_len;
}

OcError oc_seq_append_token(OcPagedSequence *seq, uint32_t token)
{
    if (!seq) return OC_ERR_INVALID_ARG;
    if (seq->generated_len >= seq->generated_cap) {
        size_t nc = seq->generated_cap * 2;
        uint32_t *nt = xrealloc(seq->generated_tokens, nc, sizeof(uint32_t));
        if (!nt) return OC_ERR_OOM;
        seq->generated_tokens = nt;
        seq->generated_cap = nc;
    }
    seq->generated_tokens[seq->generated_len++] = token;
    return OC_OK;
}

/* ─── Scheduler ─────────────────────────────────────────────────────────── */

OcError oc_scheduler_init(OcScheduler *sched,
                          const OcSchedulerConfig *scfg,
                          const OcBlockPoolConfig *bcfg)
{
    if (!sched || !scfg || !bcfg) return OC_ERR_INVALID_ARG;
    memset(sched, 0, sizeof(*sched));
    sched->config = *scfg;
    OcError e = oc_block_pool_init(&sched->block_pool, bcfg);
    if (e != OC_OK) return e;
    sched->seq_cap = scfg->max_num_running_seqs * 4 + 16;
    sched->sequences = xcalloc(sched->seq_cap, sizeof(OcPagedSequence *));
    sched->waiting_cap = scfg->max_num_running_seqs * 2 + 16;
    sched->waiting = xcalloc(sched->waiting_cap, sizeof(OcSeqId));
    sched->running_cap = scfg->max_num_running_seqs + 1;
    sched->running = xcalloc(sched->running_cap, sizeof(OcSeqId));
    if (!sched->sequences || !sched->waiting || !sched->running) {
        oc_block_pool_free(&sched->block_pool);
        free(sched->sequences); free(sched->waiting); free(sched->running);
        return OC_ERR_OOM;
    }
    sched->waiting_head = sched->waiting_tail = sched->waiting_count = 0;
    sched->running_count = 0;
    sched->next_arrival_order = 0;
    return OC_OK;
}

void oc_scheduler_free(OcScheduler *sched)
{
    if (!sched) return;
    /* Free all sequences. */
    for (size_t i = 0; i < sched->seq_cap; i++) {
        if (sched->sequences[i]) {
            oc_seq_free(sched->sequences[i]);
            free(sched->sequences[i]);
        }
    }
    oc_block_pool_free(&sched->block_pool);
    free(sched->sequences); free(sched->waiting); free(sched->running);
    memset(sched, 0, sizeof(*sched));
}

static OcError grow_waiting_queue(OcScheduler *sched)
{
    size_t new_cap = sched->waiting_cap * 2;
    OcSeqId *waiting = xcalloc(new_cap, sizeof(OcSeqId));
    if (!waiting) return OC_ERR_OOM;
    for (size_t i = 0; i < sched->waiting_count; i++)
        waiting[i] = sched->waiting[(sched->waiting_head + i) % sched->waiting_cap];
    free(sched->waiting);
    sched->waiting = waiting;
    sched->waiting_cap = new_cap;
    sched->waiting_head = 0;
    sched->waiting_tail = sched->waiting_count;
    return OC_OK;
}

OcError oc_scheduler_add_sequence(OcScheduler *sched, OcPagedSequence *seq)
{
    if (!sched || !seq) return OC_ERR_INVALID_ARG;
    if (seq->block_table.num_tokens != 0) return OC_ERR_INVALID_ARG;
    seq->block_table.block_size = sched->block_pool.config.block_size;
    if (sched->waiting_count == sched->waiting_cap &&
        grow_waiting_queue(sched) != OC_OK) return OC_ERR_OOM;
    bool stored = false;
    size_t slot = (size_t)(seq->seq_id % sched->seq_cap);
    for (size_t i = 0; i < sched->seq_cap; i++) {
        size_t idx = (slot + i) % sched->seq_cap;
        if (sched->sequences[idx] &&
            sched->sequences[idx]->seq_id == seq->seq_id)
            return OC_ERR_INVALID_ARG;
        if (sched->sequences[idx] == NULL) {
            sched->sequences[idx] = seq;
            sched->seq_count++;
            stored = true;
            break;
        }
    }
    if (!stored) return OC_ERR_OOM;
    sched->waiting[sched->waiting_tail] = seq->seq_id;
    sched->waiting_tail = (sched->waiting_tail + 1) % sched->waiting_cap;
    sched->waiting_count++;
    return OC_OK;
}

static OcError reserve_blocks(OcScheduler *sched, OcPagedSequence *seq,
                              size_t n_tokens)
{
    size_t needed = oc_block_table_blocks_needed(&seq->block_table, n_tokens);
    if (needed == 0) return OC_OK;
    if (needed > sched->block_pool.free_count) return OC_ERR_OOM;
    size_t required = seq->block_table.num_blocks + needed;
    if (required > seq->block_table.capacity) {
        size_t new_cap = seq->block_table.capacity;
        while (new_cap < required) new_cap *= 2;
        OcBlockId *blocks = xrealloc(seq->block_table.logical_to_physical,
                                     new_cap, sizeof(OcBlockId));
        if (!blocks) return OC_ERR_OOM;
        seq->block_table.logical_to_physical = blocks;
        seq->block_table.capacity = new_cap;
    }
    OcBlockId *dst = seq->block_table.logical_to_physical
                   + seq->block_table.num_blocks;
    OcError e = oc_block_pool_allocate_n(&sched->block_pool, needed, dst);
    if (e != OC_OK) return e;
    seq->block_table.num_blocks += needed;
    return OC_OK;
}

static size_t schedulable_tokens(const OcScheduler *sched,
                                 const OcPagedSequence *seq, size_t requested)
{
    size_t mapped = seq->block_table.num_blocks * seq->block_table.block_size;
    size_t available = mapped > seq->block_table.num_tokens
        ? mapped - seq->block_table.num_tokens : 0;
    size_t free_capacity = sched->block_pool.free_count
                         * seq->block_table.block_size;
    size_t capacity = available + free_capacity;
    return requested < capacity ? requested : capacity;
}

OcError oc_scheduler_step(OcScheduler *sched, OcSchedulerStepResult *out)
{
    if (!sched || !out) return OC_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));
    size_t budget = sched->config.max_num_batched_tokens;
    out->scheduled_ids = xcalloc(sched->running_cap + 4, sizeof(OcSeqId));
    out->prefill_counts = xcalloc(sched->running_cap + 4, sizeof(size_t));
    out->decode_counts = xcalloc(sched->running_cap + 4, sizeof(size_t));
    if (!out->scheduled_ids || !out->prefill_counts || !out->decode_counts) {
        free(out->scheduled_ids); free(out->prefill_counts); free(out->decode_counts);
        return OC_ERR_OOM;
    }

    /* Phase 1: Decode running sequences (1 token each). */
    for (size_t i = 0; i < sched->running_count && budget > 0; i++) {
        OcSeqId sid = sched->running[i];
        /* Find the sequence. */
        OcPagedSequence *seq = NULL;
        for (size_t j = 0; j < sched->seq_cap; j++) {
            if (sched->sequences[j] && sched->sequences[j]->seq_id == sid) {
                seq = sched->sequences[j];
                break;
            }
        }
        if (!seq || seq->status != OC_SEQ_RUNNING) continue;
        if (seq->num_prefilled_tokens < seq->prompt_len) continue; /* still prefilling */

        size_t logical = seq->block_table.num_tokens / seq->block_table.block_size;
        if (logical < seq->block_table.num_blocks) {
            OcBlockId old_id = seq->block_table.logical_to_physical[logical];
            if (sched->block_pool.blocks[old_id].ref_count > 1) {
                return OC_ERR_INVALID_ARG;
            }
        } else if (reserve_blocks(sched, seq, 1) != OC_OK) {
            break;
        }
        (void)oc_block_table_append_token(&seq->block_table);
        out->scheduled_ids[out->n_scheduled] = sid;
        out->decode_counts[out->n_scheduled] = 1;
        out->n_scheduled++;
        out->total_tokens++;
        budget--;
    }

    /* Phase 2: Continue prefill of partially-prefilled running sequences. */
    for (size_t i = 0; i < sched->running_count && budget > 0; i++) {
        OcSeqId sid = sched->running[i];
        OcPagedSequence *seq = NULL;
        for (size_t j = 0; j < sched->seq_cap; j++) {
            if (sched->sequences[j] && sched->sequences[j]->seq_id == sid) {
                seq = sched->sequences[j];
                break;
            }
        }
        if (!seq || seq->status != OC_SEQ_RUNNING) continue;
        size_t remaining = seq->prompt_len - seq->num_prefilled_tokens;
        if (remaining == 0) continue;

        size_t chunk = remaining;
        if (chunk > sched->config.prefill_chunk_size)
            chunk = sched->config.prefill_chunk_size;
        if (chunk > budget) chunk = budget;

        chunk = schedulable_tokens(sched, seq, chunk);
        if (chunk == 0 || reserve_blocks(sched, seq, chunk) != OC_OK) continue;
        /* Advance token cursor. */
        seq->num_prefilled_tokens += chunk;
        for (size_t t = 0; t < chunk; t++)
            oc_block_table_append_token(&seq->block_table);

        out->scheduled_ids[out->n_scheduled] = sid;
        out->prefill_counts[out->n_scheduled] = chunk;
        out->n_scheduled++;
        out->total_tokens += chunk;
        budget -= chunk;
    }

    /* Phase 3: Pull from waiting queue (FCFS). */
    while (budget > 0 && sched->waiting_count > 0 &&
           sched->running_count < sched->config.max_num_running_seqs) {
        OcSeqId sid = sched->waiting[sched->waiting_head];
        sched->waiting_head = (sched->waiting_head + 1) % sched->waiting_cap;
        sched->waiting_count--;

        OcPagedSequence *seq = NULL;
        for (size_t j = 0; j < sched->seq_cap; j++) {
            if (sched->sequences[j] && sched->sequences[j]->seq_id == sid) {
                seq = sched->sequences[j];
                break;
            }
        }
        if (!seq) continue;

        size_t remaining = seq->prompt_len - seq->num_prefilled_tokens;
        size_t chunk = remaining;
        if (chunk > sched->config.prefill_chunk_size)
            chunk = sched->config.prefill_chunk_size;
        if (chunk > budget) chunk = budget;

        chunk = schedulable_tokens(sched, seq, chunk);
        if (chunk == 0 || reserve_blocks(sched, seq, chunk) != OC_OK) {
            /* Put back in waiting. */
            sched->waiting_head = (sched->waiting_head - 1 + sched->waiting_cap) % sched->waiting_cap;
            sched->waiting[sched->waiting_head] = sid;
            sched->waiting_count++;
            break;
        }

        seq->status = OC_SEQ_RUNNING;
        seq->num_prefilled_tokens += chunk;
        for (size_t t = 0; t < chunk; t++)
            oc_block_table_append_token(&seq->block_table);

        if (sched->running_count < sched->running_cap)
            sched->running[sched->running_count++] = sid;

        out->scheduled_ids[out->n_scheduled] = sid;
        out->prefill_counts[out->n_scheduled] = chunk;
        out->n_scheduled++;
        out->total_tokens += chunk;
        budget -= chunk;
    }

    return OC_OK;
}

OcError oc_scheduler_postprocess(OcScheduler *sched,
                                  const OcSeqId *ids, const uint32_t *tokens,
                                  size_t n)
{
    if (!sched || !ids || !tokens) return OC_ERR_INVALID_ARG;
    for (size_t i = 0; i < n; i++) {
        OcPagedSequence *seq = NULL;
        for (size_t j = 0; j < sched->seq_cap; j++) {
            if (sched->sequences[j] && sched->sequences[j]->seq_id == ids[i]) {
                seq = sched->sequences[j];
                break;
            }
        }
        if (!seq || seq->status != OC_SEQ_RUNNING) continue;
        oc_seq_append_token(seq, tokens[i]);
        if (oc_seq_is_finished(seq)) {
            seq->status = OC_SEQ_FINISHED;
            /* Free blocks. */
            for (size_t b = 0; b < seq->block_table.num_blocks; b++) {
                oc_block_pool_dec_ref(&sched->block_pool,
                    seq->block_table.logical_to_physical[b]);
            }
            for (size_t r = 0; r < sched->running_count; r++) {
                if (sched->running[r] == seq->seq_id) {
                    sched->running[r] = sched->running[--sched->running_count];
                    break;
                }
            }
        }
    }
    return OC_OK;
}

OcError oc_scheduler_preempt(OcScheduler *sched, OcSeqId id)
{
    if (!sched) return OC_ERR_INVALID_ARG;
    OcPagedSequence *seq = NULL;
    size_t run_idx = (size_t)-1;
    for (size_t i = 0; i < sched->running_count; i++) {
        if (sched->running[i] == id) { run_idx = i; break; }
    }
    for (size_t j = 0; j < sched->seq_cap; j++) {
        if (sched->sequences[j] && sched->sequences[j]->seq_id == id) {
            seq = sched->sequences[j];
            break;
        }
    }
    if (!seq) return OC_ERR_INVALID_ARG;
    if (sched->waiting_count == sched->waiting_cap &&
        grow_waiting_queue(sched) != OC_OK) return OC_ERR_OOM;

    /* Free all blocks. */
    for (size_t b = 0; b < seq->block_table.num_blocks; b++) {
        oc_block_pool_dec_ref(&sched->block_pool,
            seq->block_table.logical_to_physical[b]);
    }
    seq->block_table.num_blocks = 0;
    seq->block_table.num_tokens = 0;
    seq->num_prefilled_tokens = 0;
    seq->status = OC_SEQ_WAITING;

    /* Remove from running, push front of waiting. */
    if (run_idx != (size_t)-1) {
        sched->running[run_idx] = sched->running[--sched->running_count];
    }
    sched->waiting_head = (sched->waiting_head - 1 + sched->waiting_cap) % sched->waiting_cap;
    sched->waiting[sched->waiting_head] = id;
    sched->waiting_count++;
    return OC_OK;
}

void oc_scheduler_invalidate_prefix_cache(OcScheduler *sched)
{
    if (!sched) return;
    memset(sched->block_pool.cache_keys, 0,
           sched->block_pool.cache_cap * sizeof(OcBlockHash));
    sched->block_pool.cache_count = 0;
}

/* ─── Paged attention kernel ────────────────────────────────────────────── */

void oc_paged_attention_head(
    const float *kv_cache,
    const OcBlockId *block_table,
    size_t n_blocks,
    const float *q,
    uint32_t kv_head,
    uint32_t n_kv_heads,
    uint32_t head_dim,
    uint32_t block_size,
    size_t n_past,
    float *out)
{
    /* Online softmax: single pass, numerically stable.
     * For each past token at position p:
     *   block = block_table[p / block_size]
     *   slot  = p % block_size
     *   k_ptr = kv_cache + block * block_size * n_kv_heads * head_dim
     *                   + slot * n_kv_heads * head_dim
     *                   + kv_head * head_dim
     *   score = dot(q, k_ptr, head_dim) / sqrt(head_dim)
     * Then softmax over all scores, weighted sum of v_ptr. */
    size_t block_stride = (size_t)block_size * n_kv_heads * head_dim;
    size_t slot_stride  = (size_t)n_kv_heads * head_dim;
    float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);
    memset(out, 0, (size_t)head_dim * sizeof(*out));

    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    /* First pass: find max score. */
    for (size_t p = 0; p < n_past; p++) {
        size_t logical = p / block_size;
        if (logical >= n_blocks) break;
        OcBlockId phys = block_table[logical];
        uint32_t slot = (uint32_t)(p % block_size);
        const float *k = kv_cache + (size_t)phys * block_stride
                       + (size_t)slot * slot_stride
                       + (size_t)kv_head * head_dim;
        float score = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++)
            score += q[d] * k[d];
        score *= inv_sqrt_d;
        if (score > max_score) max_score = score;
    }

    if (max_score == -INFINITY) max_score = 0.0f;

    /* Second pass: compute softmax weights and weighted sum of V. */
    for (size_t p = 0; p < n_past; p++) {
        size_t logical = p / block_size;
        if (logical >= n_blocks) break;
        OcBlockId phys = block_table[logical];
        uint32_t slot = (uint32_t)(p % block_size);
        const float *k = kv_cache + (size_t)phys * block_stride
                       + (size_t)slot * slot_stride
                       + (size_t)kv_head * head_dim;
        const float *v_ptr = kv_cache + (size_t)phys * block_stride
                           + (size_t)slot * slot_stride
                           + (size_t)kv_head * head_dim;
        float score = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++)
            score += q[d] * k[d];
        score *= inv_sqrt_d;
        float weight = expf(score - max_score);
        sum_exp += weight;

        /* V pointer: V follows K in the same flat buffer.
         * total_k = n_blocks * block_stride (size of all K data). */
        size_t total_k = n_blocks * block_stride;
        const float *v = kv_cache + total_k
                       + (size_t)phys * block_stride
                       + (size_t)slot * slot_stride
                       + (size_t)kv_head * head_dim;
        (void)v_ptr; /* legacy alias */
        for (uint32_t d = 0; d < head_dim; d++)
            out[d] += weight * v[d];
    }

    if (sum_exp > 0.0f) {
        float inv = 1.0f / sum_exp;
        for (uint32_t d = 0; d < head_dim; d++)
            out[d] *= inv;
    }
}
