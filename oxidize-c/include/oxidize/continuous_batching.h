/*
 * continuous_batching.h — continuous batching scheduler for inference requests.
 *
 * Port of oxidize-core/src/paged_attention/ continuous batching concepts.
 * Batches multiple inference requests with different sequence lengths
 * together (vLLM-style), allowing prefill of new requests to interleave
 * with decoding of in-progress requests.
 *
 * Design:
 *   - OcBatchConfig: tuning knobs (max_batch_size, max_seq_len, strategy).
 *   - OcBatchRequest: a single inference request (prompt tokens + max_tokens).
 *   - OcBatchSlot: per-request runtime state (generated tokens, status).
 *   - OcBatchScheduler: holds slots, selects the next batch to run, tracks
 *     throughput stats.
 *   - Scheduling strategies: FCFS (arrival order) or SHORTEST_JOB_FIRST
 *     (smallest remaining work first).
 */
#ifndef OXIDIZE_CONTINUOUS_BATCHING_H
#define OXIDIZE_CONTINUOUS_BATCHING_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Constants ────────────────────────────────────────────────────────── */

#define OC_BATCH_DEFAULT_MAX_BATCH_SIZE 32
#define OC_BATCH_DEFAULT_MAX_SEQ_LEN    4096
#define OC_BATCH_MAX_TOKENS_PER_REQUEST  8192

/* ─── Scheduling strategy ─────────────────────────────────────────────── */

typedef enum {
    OC_BATCH_FCFS = 0,            /* first-come, first-served             */
    OC_BATCH_SHORTEST_JOB_FIRST,  /* smallest remaining work first        */
} OcBatchStrategy;

/* ─── Slot state ───────────────────────────────────────────────────────── */

typedef enum {
    OC_BATCH_SLOT_WAITING    = 0,  /* queued, not yet running             */
    OC_BATCH_SLOT_RUNNING    = 1,  /* actively generating                 */
    OC_BATCH_SLOT_COMPLETED  = 2,  /* finished normally                   */
    OC_BATCH_SLOT_ABORTED    = 3,  /* cancelled by caller                 */
} OcBatchSlotState;

/* ─── Request / Slot / Stats ──────────────────────────────────────────── */

/* A single inference request submitted by the caller. The scheduler copies
 * prompt_tokens (caller retains ownership of the input array). */
typedef struct OcBatchRequest {
    uint64_t id;                          /* caller-supplied unique id      */
    const uint32_t *prompt_tokens;        /* prompt token ids (borrowed)    */
    size_t   n_prompt;                    /* number of prompt tokens        */
    size_t   max_tokens;                  /* max tokens to generate         */
    float    temperature;                 /* sampling temperature (>= 0)     */
} OcBatchRequest;

/* Per-request runtime slot. Owned by the scheduler. */
typedef struct OcBatchSlot {
    uint64_t          request_id;         /* mirrors OcBatchRequest.id       */
    OcBatchSlotState  state;              /* current slot state              */
    size_t            n_generated;        /* tokens generated so far         */
    size_t            n_processed;        /* tokens processed (prefill+gen)  */
    size_t            max_tokens;         /* cap on generated tokens         */
    float             temperature;        /* per-request temperature         */
    uint32_t         *tokens;             /* generated token buffer          */
    size_t            tokens_cap;         /* allocated capacity of tokens    */
    uint64_t          arrival_tick;       /* scheduler tick at arrival       */
    uint64_t          start_tick;         /* tick when entering RUNNING      */
    uint64_t          end_tick;           /* tick when entering COMPLETED    */
} OcBatchSlot;

/* Throughput statistics. */
typedef struct OcBatchStats {
    uint64_t total_requests;              /* requests ever added             */
    uint64_t completed_requests;          /* requests reaching COMPLETED     */
    uint64_t aborted_requests;           /* requests aborted                */
    double   avg_latency_ticks;          /* avg end_tick - arrival_tick     */
    double   throughput_tok_per_sec;      /* tokens / elapsed seconds        */
} OcBatchStats;

/* ─── Config ───────────────────────────────────────────────────────────── */

typedef struct OcBatchConfig {
    uint32_t          max_batch_size;     /* max concurrent slots            */
    uint32_t          max_seq_len;        /* max prompt + generated tokens   */
    OcBatchStrategy   scheduling_strategy;
} OcBatchConfig;

/* Returns a sensible default config. */
OcBatchConfig oc_batch_config_default(void);

/* ─── Scheduler ────────────────────────────────────────────────────────── */

typedef struct OcBatchScheduler {
    OcBatchConfig   config;
    OcBatchSlot   **slots;                /* max_batch_size entries          */
    uint32_t        slots_used;
    uint64_t        tick;                 /* monotonic scheduler tick        */
    /* Stats accumulators. */
    uint64_t        total_requests;
    uint64_t        completed_requests;
    uint64_t        aborted_requests;
    uint64_t        total_latency_ticks;
    uint64_t        total_tokens_generated;
    /* Next request_id lookup is linear; max_batch_size is small (<=32). */
} OcBatchScheduler;

/* Allocate and initialize a scheduler. Returns OC_ERR_OOM on allocation
 * failure. The caller owns the result and must call oc_batch_scheduler_free. */
OcError oc_batch_scheduler_init(OcBatchScheduler **out, OcBatchConfig config);

/* Add a request. Returns OC_ERR_INVALID_ARG if request is malformed,
 * OC_ERR_OOM if the scheduler is full or allocation fails. */
OcError oc_batch_scheduler_add(OcBatchScheduler *s, const OcBatchRequest *req);

/* Select the next batch of slots to process. Writes up to max_slots slot
 * pointers into out_slots (borrowed; owned by scheduler) and the count
 * into out_count. Only WAITING and RUNNING slots are returned. Returns
 * OC_OK even if out_count == 0 (nothing to do). */
OcError oc_batch_scheduler_next_batch(OcBatchScheduler *s,
                                       OcBatchSlot **out_slots,
                                       size_t max_slots,
                                       size_t *out_count);

/* Append a generated token to the slot matching request_id. Transitions
 * WAITING -> RUNNING on the first token. Transitions to COMPLETED when
 * n_generated reaches max_tokens. Returns OC_ERR_INVALID_ARG if the
 * request_id is unknown or the slot is already terminal. */
OcError oc_batch_scheduler_update_token(OcBatchScheduler *s,
                                        uint64_t request_id,
                                        uint32_t token);

/* Mark a request as completed (e.g., EOS reached). Idempotent on already
 * completed slots; returns OC_ERR_INVALID_ARG for unknown ids. */
OcError oc_batch_scheduler_complete(OcBatchScheduler *s, uint64_t request_id);

/* Abort a request. The slot becomes ABORTED and is no longer returned by
 * next_batch. Returns OC_ERR_INVALID_ARG for unknown ids. */
OcError oc_batch_scheduler_abort(OcBatchScheduler *s, uint64_t request_id);

/* Populate out_stats with current throughput statistics. */
OcError oc_batch_scheduler_stats(const OcBatchScheduler *s,
                                  OcBatchStats *out_stats);

/* Free the scheduler and all owned slots. Safe on NULL. */
void oc_batch_scheduler_free(OcBatchScheduler *s);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_CONTINUOUS_BATCHING_H */
