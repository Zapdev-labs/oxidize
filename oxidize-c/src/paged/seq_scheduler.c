#include "oxidize/seq_scheduler.h"

#include <stdlib.h>
#include <string.h>


OcSeqSchedulerConfig oc_seq_sched_config_default(void)
{
    OcSeqSchedulerConfig c;
    c.max_batch_size   = OC_SEQ_SCHED_DEFAULT_MAX_BATCH_SIZE;
    c.max_total_tokens = OC_SEQ_SCHED_DEFAULT_MAX_TOTAL_TOKENS;
    c.water_level      = OC_SEQ_SCHED_DEFAULT_WATER_LEVEL;
    return c;
}


#define SEQ_NOT_FOUND ((size_t)-1)

static bool seq_is_terminal(const OcSeqInfo *s)
{
    return s->state == OC_SEQ_STATE_FINISHED ||
           s->state == OC_SEQ_STATE_ABORTED;
}

/* Find the slot index for a seq_id. Returns SEQ_NOT_FOUND if not found. */
static size_t sched_find(const OcSeqScheduler *s, uint64_t seq_id)
{
    for (uint32_t i = 0; i < s->config.max_batch_size; i++) {
        if (s->sequences[i] && s->sequences[i]->request.id == seq_id) {
            return (size_t)i;
        }
    }
    return SEQ_NOT_FOUND;
}

static void seq_free(OcSeqInfo *s, OcKvPageManager *page_mgr)
{
    if (!s) return;
    /* Free any allocated pages. */
    if (page_mgr) {
        for (size_t i = 0; i < s->n_pages; i++) {
            oc_kv_page_free(page_mgr, s->page_ids[i]);
        }
    }
    free((void *)s->request.prompt_tokens);
    free(s);
}

static OcSeqInfo *seq_new(const OcSeqRequest *req, uint64_t tick)
{
    if (!req) return NULL;
    OcSeqInfo *s = malloc(sizeof(*s));
    if (!s) return NULL;
    memset(s, 0, sizeof(*s));
    s->request.id = req->id;
    s->request.n_prompt = req->n_prompt;
    s->request.max_tokens = req->max_tokens;
    s->request.temperature = req->temperature;
    if (req->n_prompt > 0 && req->prompt_tokens) {
        s->request.prompt_tokens =
            malloc(req->n_prompt * sizeof(*req->prompt_tokens));
        if (!s->request.prompt_tokens) {
            free(s);
            return NULL;
        }
        memcpy((void *)s->request.prompt_tokens, req->prompt_tokens,
               req->n_prompt * sizeof(*req->prompt_tokens));
    } else {
        s->request.prompt_tokens = NULL;
    }
    s->state = OC_SEQ_STATE_WAITING;
    s->n_pages = 0;
    s->last_pos = req->n_prompt;
    s->arrival_tick = tick;
    s->n_generated = 0;
    return s;
}

/* Allocate pages for n_tokens worth of KV cache, starting at start_pos.
 * Returns OC_OK or OC_ERR_OOM. */
static OcError alloc_pages_for(OcSeqScheduler *sched, OcSeqInfo *s,
                                size_t n_tokens, uint32_t start_pos)
{
    if (!sched->page_mgr || n_tokens == 0) return OC_OK;
    uint32_t page_size = sched->page_mgr->config.page_size;
    /* Number of pages needed to cover n_tokens starting at start_pos
     * within the current page. */
    uint32_t offset_in_page = start_pos % page_size;
    size_t needed = (offset_in_page + n_tokens + page_size - 1) / page_size;
    if (s->n_pages + needed > OC_SEQ_SCHED_MAX_PAGES_PER_SEQ) {
        return OC_ERR_OOM;
    }
    for (size_t i = 0; i < needed; i++) {
        if (s->n_pages >= OC_SEQ_SCHED_MAX_PAGES_PER_SEQ) {
            return OC_ERR_OOM;
        }
        uint32_t pid = 0;
        uint32_t pos = start_pos + (uint32_t)(i * page_size);
        OcError e = oc_kv_page_alloc(sched->page_mgr, s->request.id,
                                      pos, &pid);
        if (e != OC_OK) return e;
        s->page_ids[s->n_pages++] = pid;
    }
    return OC_OK;
}

/* Free pages held by a sequence (does not free the sequence struct). */
static void free_seq_pages(OcSeqScheduler *sched, OcSeqInfo *s)
{
    if (!sched->page_mgr || !s) return;
    for (size_t i = 0; i < s->n_pages; i++) {
        oc_kv_page_free(sched->page_mgr, s->page_ids[i]);
    }
    s->n_pages = 0;
}


OcError oc_seq_sched_init(OcSeqScheduler **out, OcSeqSchedulerConfig config,
                           OcKvPageManager *page_mgr)
{
    if (!out) return OC_ERR_INVALID_ARG;
    if (config.max_batch_size == 0 || config.max_total_tokens == 0) {
        return OC_ERR_INVALID_ARG;
    }
    if (config.water_level < 0.0f || config.water_level > 1.0f) {
        return OC_ERR_INVALID_ARG;
    }
    OcSeqScheduler *s = malloc(sizeof(*s));
    if (!s) return OC_ERR_OOM;
    memset(s, 0, sizeof(*s));
    s->config = config;
    s->sequences = calloc(config.max_batch_size, sizeof(*s->sequences));
    if (!s->sequences) {
        free(s);
        return OC_ERR_OOM;
    }
    s->n_waiting = 0;
    s->n_running = 0;
    s->n_swapped = 0;
    s->tick = 0;
    s->page_mgr = page_mgr;
    s->total_tokens = 0;
    *out = s;
    return OC_OK;
}

void oc_seq_sched_free(OcSeqScheduler *s)
{
    if (!s) return;
    for (uint32_t i = 0; i < s->config.max_batch_size; i++) {
        if (s->sequences[i]) {
            seq_free(s->sequences[i], s->page_mgr);
            s->sequences[i] = NULL;
        }
    }
    free(s->sequences);
    free(s);
}

OcError oc_seq_sched_add(OcSeqScheduler *s, const OcSeqRequest *req)
{
    if (!s || !req) return OC_ERR_INVALID_ARG;
    if (req->n_prompt > 0 && !req->prompt_tokens) return OC_ERR_INVALID_ARG;
    if (sched_find(s, req->id) != SEQ_NOT_FOUND) return OC_ERR_INVALID_ARG;
    /* Check if there's a free slot. */
    for (uint32_t i = 0; i < s->config.max_batch_size; i++) {
        if (s->sequences[i] == NULL) {
            OcSeqInfo *seq = seq_new(req, s->tick);
            if (!seq) return OC_ERR_OOM;
            s->sequences[i] = seq;
            s->n_waiting++;
            return OC_OK;
        }
    }
    return OC_ERR_OOM;
}

OcError oc_seq_sched_schedule(OcSeqScheduler *s, OcSeqBatch *out_batch)
{
    if (!s || !out_batch) return OC_ERR_INVALID_ARG;
    memset(out_batch, 0, sizeof(*out_batch));
    size_t batch_count = 0;

    for (uint32_t i = 0; i < s->config.max_batch_size &&
                         batch_count < OC_SEQ_SCHED_MAX_BATCH; i++) {
        OcSeqInfo *seq = s->sequences[i];
        if (!seq) continue;
        if (seq->state == OC_SEQ_STATE_RUNNING) {
            out_batch->seq_ids[batch_count] = seq->request.id;
            out_batch->is_prefill[batch_count] = false;
            batch_count++;
        }
    }

    for (uint32_t i = 0; i < s->config.max_batch_size &&
                         batch_count < OC_SEQ_SCHED_MAX_BATCH; i++) {
        OcSeqInfo *seq = s->sequences[i];
        if (!seq) continue;
        if (seq->state != OC_SEQ_STATE_WAITING) continue;

        /* Check capacity: can we fit the prompt? */
        if (!oc_seq_sched_can_fit(s, seq->request.n_prompt)) {
            /* Under memory pressure: swap out running sequences if possible. */
            continue;
        }

        /* Allocate pages for the prompt. */
        if (s->page_mgr) {
            OcError e = alloc_pages_for(s, seq, seq->request.n_prompt, 0);
            if (e != OC_OK) continue;
        }
        s->total_tokens += seq->request.n_prompt;
        seq->state = OC_SEQ_STATE_RUNNING;
        s->n_waiting--;
        s->n_running++;
        out_batch->seq_ids[batch_count] = seq->request.id;
        out_batch->is_prefill[batch_count] = true;
        batch_count++;
    }

    uint32_t water_mark = (uint32_t)((float)s->config.max_total_tokens *
                                     s->config.water_level);
    if (s->total_tokens > water_mark && s->n_running > 0) {
        /* Swap out the most recently admitted running sequence. */
        for (int32_t i = (int32_t)s->config.max_batch_size - 1; i >= 0; i--) {
            OcSeqInfo *seq = s->sequences[i];
            if (!seq) continue;
            if (seq->state != OC_SEQ_STATE_RUNNING) continue;
            /* Free the pages (swap to CPU / drop KV). */
            free_seq_pages(s, seq);
            s->total_tokens -= seq->last_pos;
            seq->state = OC_SEQ_STATE_SWAPPED;
            s->n_running--;
            s->n_swapped++;
            break;
        }
    }

    if (s->n_swapped > 0 && s->total_tokens < water_mark) {
        for (uint32_t i = 0; i < s->config.max_batch_size; i++) {
            OcSeqInfo *seq = s->sequences[i];
            if (!seq) continue;
            if (seq->state != OC_SEQ_STATE_SWAPPED) continue;
            if (!oc_seq_sched_can_fit(s, seq->last_pos)) break;
            /* Re-allocate pages for the restored sequence. */
            if (s->page_mgr) {
                OcError e = alloc_pages_for(s, seq, seq->last_pos, 0);
                if (e != OC_OK) break;
            }
            s->total_tokens += seq->last_pos;
            seq->state = OC_SEQ_STATE_RUNNING;
            s->n_swapped--;
            s->n_running++;
            /* Add to decode batch if there's room. */
            if (batch_count < OC_SEQ_SCHED_MAX_BATCH) {
                out_batch->seq_ids[batch_count] = seq->request.id;
                out_batch->is_prefill[batch_count] = false;
                batch_count++;
            }
            /* Only restore one per step. */
            break;
        }
    }

    out_batch->n_seqs = batch_count;
    s->tick++;
    return OC_OK;
}

OcError oc_seq_sched_append_token(OcSeqScheduler *s, uint64_t seq_id,
                                   uint32_t token)
{
    (void)token; /* token value is tracked by the caller; we only count. */
    if (!s) return OC_ERR_INVALID_ARG;
    size_t idx = sched_find(s, seq_id);
    if (idx == SEQ_NOT_FOUND) return OC_ERR_INVALID_ARG;
    OcSeqInfo *seq = s->sequences[idx];
    if (seq_is_terminal(seq)) return OC_ERR_INVALID_ARG;
    /* Transition WAITING -> RUNNING on the first token (if not already). */
    if (seq->state == OC_SEQ_STATE_WAITING) {
        seq->state = OC_SEQ_STATE_RUNNING;
        s->n_waiting--;
        s->n_running++;
    }
    /* SWAPPED sequences can also receive tokens (they stay SWAPPED). */
    seq->n_generated++;
    seq->last_pos++;
    s->total_tokens++;
    /* Allocate a new page if we've crossed a page boundary. */
    if (s->page_mgr) {
        uint32_t page_size = s->page_mgr->config.page_size;
        if (seq->last_pos % page_size == 0) {
            /* Need a new page for the next token. */
            if (seq->n_pages < OC_SEQ_SCHED_MAX_PAGES_PER_SEQ) {
                uint32_t pid = 0;
                if (oc_kv_page_alloc(s->page_mgr, seq->request.id,
                                      (uint32_t)seq->last_pos, &pid) == OC_OK) {
                    seq->page_ids[seq->n_pages++] = pid;
                }
            }
        }
    }
    /* Auto-finish when max_tokens reached. */
    if (seq->request.max_tokens > 0 &&
        seq->n_generated >= seq->request.max_tokens) {
        seq->state = OC_SEQ_STATE_FINISHED;
        s->n_running--;
        s->total_tokens -= seq->last_pos;
        free_seq_pages(s, seq);
    }
    return OC_OK;
}

OcError oc_seq_sched_finish(OcSeqScheduler *s, uint64_t seq_id)
{
    if (!s) return OC_ERR_INVALID_ARG;
    size_t idx = sched_find(s, seq_id);
    if (idx == SEQ_NOT_FOUND) return OC_ERR_INVALID_ARG;
    OcSeqInfo *seq = s->sequences[idx];
    if (seq->state == OC_SEQ_STATE_FINISHED) return OC_OK;
    if (seq->state == OC_SEQ_STATE_ABORTED) return OC_ERR_INVALID_ARG;
    /* Update counters based on the previous state. */
    switch (seq->state) {
    case OC_SEQ_STATE_WAITING:
        s->n_waiting--;
        break;
    case OC_SEQ_STATE_RUNNING:
        s->n_running--;
        s->total_tokens -= seq->last_pos;
        break;
    case OC_SEQ_STATE_SWAPPED:
        s->n_swapped--;
        break;
    default:
        break;
    }
    seq->state = OC_SEQ_STATE_FINISHED;
    free_seq_pages(s, seq);
    return OC_OK;
}

OcError oc_seq_sched_abort(OcSeqScheduler *s, uint64_t seq_id)
{
    if (!s) return OC_ERR_INVALID_ARG;
    size_t idx = sched_find(s, seq_id);
    if (idx == SEQ_NOT_FOUND) return OC_ERR_INVALID_ARG;
    OcSeqInfo *seq = s->sequences[idx];
    if (seq->state == OC_SEQ_STATE_ABORTED) return OC_OK;
    if (seq->state == OC_SEQ_STATE_FINISHED) return OC_ERR_INVALID_ARG;
    /* Update counters based on the previous state. */
    switch (seq->state) {
    case OC_SEQ_STATE_WAITING:
        s->n_waiting--;
        break;
    case OC_SEQ_STATE_RUNNING:
        s->n_running--;
        s->total_tokens -= seq->last_pos;
        break;
    case OC_SEQ_STATE_SWAPPED:
        s->n_swapped--;
        break;
    default:
        break;
    }
    seq->state = OC_SEQ_STATE_ABORTED;
    free_seq_pages(s, seq);
    return OC_OK;
}

bool oc_seq_sched_can_fit(OcSeqScheduler *s, size_t n_tokens)
{
    if (!s) return false;
    /* Check against water level, not the hard cap. This leaves headroom
     * for the swap-out mechanism. */
    size_t water_mark = (size_t)((float)s->config.max_total_tokens *
                                 s->config.water_level);
    return (s->total_tokens + n_tokens) <= water_mark;
}

size_t oc_seq_sched_running_count(OcSeqScheduler *s)
{
    if (!s) return 0;
    return s->n_running;
}

size_t oc_seq_sched_waiting_count(OcSeqScheduler *s)
{
    if (!s) return 0;
    return s->n_waiting;
}
