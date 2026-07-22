/*
 * seq_scheduler.h — vLLM-style sequence scheduler.
 *
 * Manages incoming inference requests, allocates KV cache pages, and
 * schedules batches for processing. Implements the standard vLLM
 * scheduling loop:
 *   - Prefill: process prompts of WAITING sequences if capacity available.
 *   - Decode: generate next token for RUNNING sequences.
 *   - Swap: move sequences to/from SWAPPED based on memory pressure.
 *
 * See include/oxidize/kv_page.h for the underlying page manager.
 */
#ifndef OXIDIZE_SEQ_SCHEDULER_H
#define OXIDIZE_SEQ_SCHEDULER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/kv_page.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Constants ────────────────────────────────────────────────────────── */

#define OC_SEQ_SCHED_DEFAULT_MAX_BATCH_SIZE    32
#define OC_SEQ_SCHED_DEFAULT_MAX_TOTAL_TOKENS  8192
#define OC_SEQ_SCHED_DEFAULT_WATER_LEVEL       0.85f
#define OC_SEQ_SCHED_MAX_PAGES_PER_SEQ         64
#define OC_SEQ_SCHED_MAX_BATCH                 32

/* ─── Sequence state ───────────────────────────────────────────────────── */

typedef enum {
    OC_SEQ_STATE_WAITING  = 0,
    OC_SEQ_STATE_RUNNING  = 1,
    OC_SEQ_STATE_SWAPPED  = 2,
    OC_SEQ_STATE_FINISHED = 3,
    OC_SEQ_STATE_ABORTED  = 4,
} OcSeqState;

/* ─── Request ──────────────────────────────────────────────────────────── */

/* A single inference request submitted by the caller. The scheduler copies
 * prompt_tokens (caller retains ownership of the input array). */
typedef struct OcSeqRequest {
    uint64_t id;                          /* caller-supplied unique id      */
    const uint32_t *prompt_tokens;        /* prompt token ids (borrowed)    */
    size_t   n_prompt;                    /* number of prompt tokens        */
    size_t   max_tokens;                  /* max tokens to generate         */
    float    temperature;                 /* sampling temperature (>= 0)    */
} OcSeqRequest;

/* ─── Sequence info ────────────────────────────────────────────────────── */

typedef struct OcSeqInfo {
    OcSeqRequest request;        /* copied request (owns prompt_tokens)    */
    OcSeqState   state;          /* current scheduling state               */
    uint32_t     page_ids[OC_SEQ_SCHED_MAX_PAGES_PER_SEQ];
    size_t       n_pages;        /* number of allocated pages              */
    size_t       last_pos;       /* last token position (prompt + generated) */
    uint64_t     arrival_tick;   /* scheduler tick at arrival              */
    size_t       n_generated;    /* number of tokens generated so far      */
} OcSeqInfo;

/* ─── Config ───────────────────────────────────────────────────────────── */

typedef struct OcSeqSchedulerConfig {
    uint32_t max_batch_size;     /* max concurrent running sequences       */
    uint32_t max_total_tokens;   /* max total tokens across all sequences  */
    float    water_level;        /* memory pressure threshold (0..1)       */
} OcSeqSchedulerConfig;

/* Returns a sensible default config. */
OcSeqSchedulerConfig oc_seq_sched_config_default(void);

/* ─── Batch ────────────────────────────────────────────────────────────── */

typedef struct OcSeqBatch {
    uint64_t seq_ids[OC_SEQ_SCHED_MAX_BATCH]; /* sequence ids in this batch */
    bool     is_prefill[OC_SEQ_SCHED_MAX_BATCH];
    size_t   n_seqs;                            /* number of sequences      */
} OcSeqBatch;

/* ─── Scheduler ────────────────────────────────────────────────────────── */

typedef struct OcSeqScheduler {
    OcSeqSchedulerConfig config;
    OcSeqInfo **sequences;      /* max_batch_size slots                    */
    uint32_t    n_waiting;
    uint32_t    n_running;
    uint32_t    n_swapped;
    uint64_t    tick;            /* monotonic scheduler tick               */
    OcKvPageManager *page_mgr;   /* optional page manager for KV cache     */
    size_t      total_tokens;    /* total tokens currently in use          */
} OcSeqScheduler;

/* Allocate and initialize a scheduler. If page_mgr is non-NULL, the
 * scheduler will allocate/free pages from it; otherwise it just tracks
 * token counts. Returns OC_ERR_OOM on failure. The caller owns the result
 * and must call oc_seq_sched_free. */
OcError oc_seq_sched_init(OcSeqScheduler **out, OcSeqSchedulerConfig config,
                           OcKvPageManager *page_mgr);

/* Add a request to the waiting queue. Returns OC_ERR_INVALID_ARG if the
 * request is malformed or the id already exists, OC_ERR_OOM if full. */
OcError oc_seq_sched_add(OcSeqScheduler *sched, const OcSeqRequest *request);

/* Schedule the next batch (prefill + decode).
 *   - Prefill: process prompts of WAITING sequences if capacity available.
 *   - Decode: generate next token for RUNNING sequences.
 *   - Swap: move sequences to/from SWAPPED based on memory pressure.
 * Returns OC_OK even if out_batch.n_seqs == 0 (nothing to do). */
OcError oc_seq_sched_schedule(OcSeqScheduler *sched, OcSeqBatch *out_batch);

/* Append a generated token to a sequence. Transitions WAITING -> RUNNING
 * on the first token. Returns OC_ERR_INVALID_ARG for an unknown id or
 * terminal state. */
OcError oc_seq_sched_append_token(OcSeqScheduler *sched, uint64_t seq_id,
                                   uint32_t token);

/* Mark a sequence as finished. Idempotent on already-finished sequences. */
OcError oc_seq_sched_finish(OcSeqScheduler *sched, uint64_t seq_id);

/* Abort a sequence and free its pages. Idempotent on already-aborted
 * sequences. */
OcError oc_seq_sched_abort(OcSeqScheduler *sched, uint64_t seq_id);

/* Check if there's room for n more tokens (considering max_total_tokens
 * and the water level). */
bool oc_seq_sched_can_fit(OcSeqScheduler *sched, size_t n_tokens);

/* Number of running sequences. */
size_t oc_seq_sched_running_count(OcSeqScheduler *sched);

/* Number of waiting sequences. */
size_t oc_seq_sched_waiting_count(OcSeqScheduler *sched);

/* Free the scheduler and all owned sequences. Safe on NULL. */
void oc_seq_sched_free(OcSeqScheduler *sched);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_SEQ_SCHEDULER_H */
