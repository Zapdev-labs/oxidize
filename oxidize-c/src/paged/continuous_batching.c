/* continuous_batching.c — continuous batching scheduler implementation. */
#include "oxidize/continuous_batching.h"

#include <stdlib.h>
#include <string.h>

#define SLOT_NOT_FOUND ((size_t)-1)


OcBatchConfig oc_batch_config_default(void)
{
    OcBatchConfig c;
    c.max_batch_size = OC_BATCH_DEFAULT_MAX_BATCH_SIZE;
    c.max_seq_len    = OC_BATCH_DEFAULT_MAX_SEQ_LEN;
    c.scheduling_strategy = OC_BATCH_FCFS;
    return c;
}


static void slot_free(OcBatchSlot *slot)
{
    if (!slot) return;
    free(slot->tokens);
    free(slot);
}

static OcBatchSlot *slot_new(const OcBatchRequest *req, uint64_t tick)
{
    if (!req) return NULL;
    OcBatchSlot *s = malloc(sizeof(*s));
    if (!s) return NULL;
    memset(s, 0, sizeof(*s));
    s->request_id   = req->id;
    s->state        = OC_BATCH_SLOT_WAITING;
    s->n_generated  = 0;
    s->n_processed  = req->n_prompt;
    s->max_tokens    = req->max_tokens;
    s->temperature   = req->temperature;
    s->arrival_tick  = tick;
    s->start_tick    = 0;
    s->end_tick      = 0;
    if (req->max_tokens > OC_BATCH_MAX_TOKENS_PER_REQUEST) {
        s->tokens_cap = OC_BATCH_MAX_TOKENS_PER_REQUEST;
    } else if (req->max_tokens == 0) {
        s->tokens_cap = 64;  /* default modest buffer */
    } else {
        s->tokens_cap = req->max_tokens;
    }
    s->tokens = malloc(s->tokens_cap * sizeof(*s->tokens));
    if (!s->tokens) {
        free(s);
        return NULL;
    }
    return s;
}

static int slot_is_terminal(const OcBatchSlot *s)
{
    return s->state == OC_BATCH_SLOT_COMPLETED || s->state == OC_BATCH_SLOT_ABORTED;
}

/* Find the slot index for a request_id. Returns SLOT_NOT_FOUND if not found. */
static size_t scheduler_find(const OcBatchScheduler *s, uint64_t request_id)
{
    for (uint32_t i = 0; i < s->config.max_batch_size; i++) {
        if (s->slots[i] && s->slots[i]->request_id == request_id) {
            return (size_t)i;
        }
    }
    return SLOT_NOT_FOUND;
}

OcError oc_batch_scheduler_init(OcBatchScheduler **out, OcBatchConfig config)
{
    if (!out) return OC_ERR_INVALID_ARG;
    if (config.max_batch_size == 0) return OC_ERR_INVALID_ARG;
    OcBatchScheduler *s = malloc(sizeof(*s));
    if (!s) return OC_ERR_OOM;
    memset(s, 0, sizeof(*s));
    s->config = config;
    s->slots = calloc(config.max_batch_size, sizeof(*s->slots));
    if (!s->slots) {
        free(s);
        return OC_ERR_OOM;
    }
    s->slots_used = 0;
    s->tick = 0;
    s->total_requests = 0;
    s->completed_requests = 0;
    s->aborted_requests = 0;
    s->total_latency_ticks = 0;
    s->total_tokens_generated = 0;
    *out = s;
    return OC_OK;
}

OcError oc_batch_scheduler_add(OcBatchScheduler *s, const OcBatchRequest *req)
{
    if (!s || !req) return OC_ERR_INVALID_ARG;
    if (req->n_prompt == 0 && req->prompt_tokens) {
        return OC_ERR_INVALID_ARG;
    }
    if (req->n_prompt > 0 && !req->prompt_tokens) {
        return OC_ERR_INVALID_ARG;
    }
    if (req->max_tokens > s->config.max_seq_len) {
        return OC_ERR_INVALID_ARG;
    }
    if (scheduler_find(s, req->id) != SLOT_NOT_FOUND) {
        return OC_ERR_INVALID_ARG;
    }
    if (s->slots_used >= s->config.max_batch_size) {
        return OC_ERR_OOM;
    }
    OcBatchSlot *slot = slot_new(req, s->tick);
    if (!slot) return OC_ERR_OOM;
    /* Insert into the first free slot. */
    for (uint32_t i = 0; i < s->config.max_batch_size; i++) {
        if (s->slots[i] == NULL) {
            s->slots[i] = slot;
            s->slots_used++;
            s->total_requests++;
            return OC_OK;
        }
    }
    /* Unreachable (slots_used < max_batch_size). */
    slot_free(slot);
    return OC_ERR_INTERNAL;
}

/* Comparison for SJF: smallest remaining work (max_tokens - n_generated)
 * first. */
static int sjf_cmp(const OcBatchSlot *a, const OcBatchSlot *b)
{
    size_t ra = a->max_tokens - a->n_generated;
    size_t rb = b->max_tokens - b->n_generated;
    if (ra < rb) return -1;
    if (ra > rb) return 1;
    return 0;
}

OcError oc_batch_scheduler_next_batch(OcBatchScheduler *s,
                                       OcBatchSlot **out_slots,
                                       size_t max_slots,
                                       size_t *out_count)
{
    if (!s || !out_slots || !out_count) return OC_ERR_INVALID_ARG;
    size_t count = 0;
    /* Gather candidates: WAITING and RUNNING slots. */
    /* For small max_batch_size (<=32) a linear scan is fine. */
    size_t candidate_idx[256];
    size_t n_candidates = 0;
    for (uint32_t i = 0; i < s->config.max_batch_size; i++) {
        OcBatchSlot *slot = s->slots[i];
        if (!slot) continue;
        if (slot->state == OC_BATCH_SLOT_WAITING ||
            slot->state == OC_BATCH_SLOT_RUNNING) {
            if (n_candidates < 256) {
                candidate_idx[n_candidates++] = i;
            }
        }
    }
    /* Sort by strategy. */
    if (s->config.scheduling_strategy == OC_BATCH_SHORTEST_JOB_FIRST) {
        /* Simple insertion sort. */
        for (size_t i = 1; i < n_candidates; i++) {
            size_t key = candidate_idx[i];
            size_t j = i;
            while (j > 0 &&
                   sjf_cmp(s->slots[candidate_idx[j - 1]],
                           s->slots[key]) > 0) {
                candidate_idx[j] = candidate_idx[j - 1];
                j--;
            }
            candidate_idx[j] = key;
        }
    }
    /* FCFS: insertion sort by arrival_tick (stable). */
    if (s->config.scheduling_strategy == OC_BATCH_FCFS) {
        for (size_t i = 1; i < n_candidates; i++) {
            size_t key = candidate_idx[i];
            size_t j = i;
            while (j > 0 &&
                   s->slots[candidate_idx[j - 1]]->arrival_tick >
                       s->slots[key]->arrival_tick) {
                candidate_idx[j] = candidate_idx[j - 1];
                j--;
            }
            candidate_idx[j] = key;
        }
    }
    /* Emit up to max_slots. */
    for (size_t i = 0; i < n_candidates && count < max_slots; i++) {
        out_slots[count++] = s->slots[candidate_idx[i]];
    }
    *out_count = count;
    s->tick++;
    return OC_OK;
}

OcError oc_batch_scheduler_update_token(OcBatchScheduler *s,
                                        uint64_t request_id,
                                        uint32_t token)
{
    if (!s) return OC_ERR_INVALID_ARG;
    size_t idx = scheduler_find(s, request_id);
    if (idx == SLOT_NOT_FOUND) return OC_ERR_INVALID_ARG;
    OcBatchSlot *slot = s->slots[idx];
    if (slot_is_terminal(slot)) return OC_ERR_INVALID_ARG;
    /* Transition WAITING -> RUNNING on the first generated token. */
    if (slot->state == OC_BATCH_SLOT_WAITING) {
        slot->state = OC_BATCH_SLOT_RUNNING;
        slot->start_tick = s->tick;
    }
    /* Grow tokens buffer if needed. */
    if (slot->n_generated >= slot->tokens_cap) {
        size_t new_cap = slot->tokens_cap * 2;
        if (new_cap > OC_BATCH_MAX_TOKENS_PER_REQUEST) {
            new_cap = OC_BATCH_MAX_TOKENS_PER_REQUEST;
        }
        if (new_cap <= slot->tokens_cap) {
            /* Hard cap reached; cannot store more tokens. */
            return OC_ERR_OOM;
        }
        uint32_t *new_tokens = realloc(slot->tokens,
                                       new_cap * sizeof(*new_tokens));
        if (!new_tokens) return OC_ERR_OOM;
        slot->tokens = new_tokens;
        slot->tokens_cap = new_cap;
    }
    slot->tokens[slot->n_generated++] = token;
    slot->n_processed++;
    s->total_tokens_generated++;
    s->tick++;  /* advance scheduler tick on each token generation */
    /* Auto-complete when max_tokens reached. */
    if (slot->max_tokens > 0 && slot->n_generated >= slot->max_tokens) {
        slot->state = OC_BATCH_SLOT_COMPLETED;
        slot->end_tick = s->tick;
        s->completed_requests++;
        s->total_latency_ticks += (slot->end_tick - slot->arrival_tick);
    }
    return OC_OK;
}

OcError oc_batch_scheduler_complete(OcBatchScheduler *s, uint64_t request_id)
{
    if (!s) return OC_ERR_INVALID_ARG;
    size_t idx = scheduler_find(s, request_id);
    if (idx == SLOT_NOT_FOUND) return OC_ERR_INVALID_ARG;
    OcBatchSlot *slot = s->slots[idx];
    if (slot->state == OC_BATCH_SLOT_COMPLETED) return OC_OK;
    if (slot->state == OC_BATCH_SLOT_ABORTED) return OC_ERR_INVALID_ARG;
    slot->state = OC_BATCH_SLOT_COMPLETED;
    slot->end_tick = s->tick;
    s->completed_requests++;
    s->total_latency_ticks += (slot->end_tick - slot->arrival_tick);
    return OC_OK;
}

OcError oc_batch_scheduler_abort(OcBatchScheduler *s, uint64_t request_id)
{
    if (!s) return OC_ERR_INVALID_ARG;
    size_t idx = scheduler_find(s, request_id);
    if (idx == SLOT_NOT_FOUND) return OC_ERR_INVALID_ARG;
    OcBatchSlot *slot = s->slots[idx];
    if (slot->state == OC_BATCH_SLOT_ABORTED) return OC_OK;
    if (slot->state == OC_BATCH_SLOT_COMPLETED) return OC_ERR_INVALID_ARG;
    slot->state = OC_BATCH_SLOT_ABORTED;
    slot->end_tick = s->tick;
    s->aborted_requests++;
    s->total_latency_ticks += (slot->end_tick - slot->arrival_tick);
    return OC_OK;
}

OcError oc_batch_scheduler_stats(const OcBatchScheduler *s,
                                  OcBatchStats *out_stats)
{
    if (!s || !out_stats) return OC_ERR_INVALID_ARG;
    out_stats->total_requests = s->total_requests;
    out_stats->completed_requests = s->completed_requests;
    out_stats->aborted_requests = s->aborted_requests;
    if (s->completed_requests + s->aborted_requests > 0) {
        out_stats->avg_latency_ticks =
            (double)s->total_latency_ticks /
            (double)(s->completed_requests + s->aborted_requests);
    } else {
        out_stats->avg_latency_ticks = 0.0;
    }
    if (s->tick > 0) {
        out_stats->throughput_tok_per_sec =
            (double)s->total_tokens_generated / (double)s->tick;
    } else {
        out_stats->throughput_tok_per_sec = 0.0;
    }
    return OC_OK;
}

void oc_batch_scheduler_free(OcBatchScheduler *s)
{
    if (!s) return;
    for (uint32_t i = 0; i < s->config.max_batch_size; i++) {
        slot_free(s->slots[i]);
        s->slots[i] = NULL;
    }
    free(s->slots);
    free(s);
}
