/*
 * batch_engine.h — Continuous-batching decode engine.
 *
 * Port of oxidize-core/src/model/inference/batch_engine.rs.
 * Manages per-sequence KV state, admits new requests up to a batch cap,
 * prefills each one, then issues a single batched decode per step.
 * The engine is model-agnostic: it manages bookkeeping only.
 */
#ifndef OXIDIZE_BATCH_ENGINE_H
#define OXIDIZE_BATCH_ENGINE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_BATCH_MAX_SEQS     256
#define OC_BATCH_MAX_PROMPT   32768
#define OC_BATCH_INVALID_ID   0

typedef uint64_t OcSeqId;

typedef struct {
    uint32_t max_batch;             /* max sequences decoded together (default 32) */
    uint32_t default_capacity_tokens; /* per-sequence KV capacity (default 2048) */
} OcBatchConfig;

typedef struct {
    OcSeqId  seq_id;
    uint32_t token;
    bool     finished;
} OcBatchStepOutput;

typedef struct OcBatchEngine OcBatchEngine;

/* Initialize config with defaults. */
void oc_batch_config_init(OcBatchConfig *cfg);

/* Create a new engine. */
OcError oc_batch_engine_init(OcBatchEngine **out, const OcBatchConfig *cfg,
                             uint32_t kv_layers, uint32_t kv_row_len);

/* Destroy the engine. */
void oc_batch_engine_free(OcBatchEngine *engine);

/* Submit a prompt for processing. Returns the SeqId for tracking. */
OcError oc_batch_submit(OcBatchEngine *engine,
                        const uint32_t *prompt, size_t prompt_len,
                        size_t max_new, uint32_t stop_token, bool has_stop,
                        OcSeqId *out_id);

/* Number of actively decoding sequences. */
size_t oc_batch_active_len(const OcBatchEngine *engine);

/* Number of pending (not yet admitted) requests. */
size_t oc_batch_pending_len(const OcBatchEngine *engine);

/* True while any sequence is pending or active. */
bool oc_batch_has_work(const OcBatchEngine *engine);

/* Cancel a sequence (pending or active). Returns true if it was present. */
bool oc_batch_cancel(OcBatchEngine *engine, OcSeqId id);

/* Execute one decode step. Outputs up to max_out results in out.
 * Returns OC_OK on success. *n_out is set to the number of outputs produced. */
OcError oc_batch_step(OcBatchEngine *engine,
                      OcBatchStepOutput *out, size_t max_out, size_t *n_out);

/* Get the current position (next token position) for a sequence. */
OcError oc_batch_seq_position(const OcBatchEngine *engine, OcSeqId id, size_t *out_pos);

/* Total number of sequences ever submitted. */
size_t oc_batch_total_submitted(const OcBatchEngine *engine);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_BATCH_ENGINE_H */
