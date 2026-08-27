/*
 * scheduler.c — Higher-level request scheduler implementation.
 *
 * See include/oxidize/scheduler.h for design notes.
 */
#include "oxidize/scheduler.h"

#include <stdlib.h>
#include <string.h>

/* ─── Config ───────────────────────────────────────────────────────────── */

OcError oc_sched_config_init(OcSchedConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    cfg->max_batch_size             = OC_SCHED_DEFAULT_MAX_BATCH_SIZE;
    cfg->max_tokens_total           = OC_SCHED_DEFAULT_MAX_TOKENS_TOTAL;
    cfg->preempt_mode               = OC_SCHED_PREEMPT_RECOMPUTE;
    cfg->enable_continuous_batching = true;
    return OC_OK;
}

/* ─── Request helpers ─────────────────────────────────────────────────── */

OcError oc_sched_request_init(OcSchedRequest *req, uint64_t id,
                               const uint32_t *prompt_tokens, uint32_t n_prompt,
                               uint32_t max_tokens, OcSchedPriority priority,
                               uint64_t created_ms)
{
    if (!req) return OC_ERR_INVALID_ARG;
    if (n_prompt > 0 && !prompt_tokens) return OC_ERR_INVALID_ARG;

    memset(req, 0, sizeof(*req));
    req->id          = id;
    req->n_prompt    = n_prompt;
    req->max_tokens  = max_tokens;
    req->priority    = priority;
    req->created_ms  = created_ms;
    req->status      = OC_SCHED_STATUS_PENDING;

    if (n_prompt > 0) {
        req->prompt_tokens = malloc((size_t)n_prompt * sizeof(uint32_t));
        if (!req->prompt_tokens) return OC_ERR_OOM;
        memcpy(req->prompt_tokens, prompt_tokens,
               (size_t)n_prompt * sizeof(uint32_t));
    } else {
        req->prompt_tokens = NULL;
    }
    return OC_OK;
}

void oc_sched_request_free(OcSchedRequest *req)
{
    if (!req) return;
    free(req->prompt_tokens);
    req->prompt_tokens = NULL;
    req->n_prompt = 0;
}

/* ─── Helpers ──────────────────────────────────────────────────────────── */

static OcSchedRequest *request_clone(const OcSchedRequest *req, uint64_t id)
{
    if (!req) return NULL;
    OcSchedRequest *copy = malloc(sizeof(*copy));
    if (!copy) return NULL;
    memset(copy, 0, sizeof(*copy));
    copy->id          = id;
    copy->n_prompt    = req->n_prompt;
    copy->max_tokens  = req->max_tokens;
    copy->priority    = req->priority;
    copy->created_ms  = req->created_ms;
    copy->status      = OC_SCHED_STATUS_PENDING;
    if (req->n_prompt > 0 && req->prompt_tokens) {
        copy->prompt_tokens =
            malloc((size_t)req->n_prompt * sizeof(uint32_t));
        if (!copy->prompt_tokens) {
            free(copy);
            return NULL;
        }
        memcpy(copy->prompt_tokens, req->prompt_tokens,
               (size_t)req->n_prompt * sizeof(uint32_t));
    } else {
        copy->prompt_tokens = NULL;
    }
    return copy;
}

static size_t sched_find(const OcScheduler *s, uint64_t id)
{
    for (uint32_t i = 0; i < s->n_requests; i++) {
        if (s->requests[i] && s->requests[i]->id == id) {
            return (size_t)i;
        }
    }
    return (size_t)-1;
}

static OcError sched_grow(OcScheduler *s)
{
    uint32_t new_cap = s->capacity * 2;
    if (new_cap == 0) new_cap = OC_SCHED_INITIAL_CAPACITY;
    OcSchedRequest **arr =
        realloc(s->requests, (size_t)new_cap * sizeof(*arr));
    if (!arr) return OC_ERR_OOM;
    /* Zero new slots. */
    memset(arr + s->capacity, 0,
           (size_t)(new_cap - s->capacity) * sizeof(*arr));
    s->requests = arr;
    s->capacity = new_cap;
    return OC_OK;
}

/* ─── Scheduler lifecycle ──────────────────────────────────────────────── */

OcError oc_sched_init(OcScheduler *sched, const OcSchedConfig *cfg)
{
    if (!sched) return OC_ERR_INVALID_ARG;
    OcSchedConfig c;
    if (cfg) {
        c = *cfg;
    } else {
        oc_sched_config_init(&c);
    }
    if (c.max_batch_size == 0 || c.max_tokens_total == 0) {
        return OC_ERR_INVALID_ARG;
    }
    memset(sched, 0, sizeof(*sched));
    sched->config   = c;
    sched->capacity = OC_SCHED_INITIAL_CAPACITY;
    sched->requests = calloc(sched->capacity, sizeof(*sched->requests));
    if (!sched->requests) {
        return OC_ERR_OOM;
    }
    sched->n_requests  = 0;
    sched->n_running   = 0;
    sched->n_completed  = 0;
    sched->next_id      = 1;
    return OC_OK;
}

OcError oc_sched_add_request(OcScheduler *sched, const OcSchedRequest *req,
                               uint64_t *out_id)
{
    if (!sched || !req) return OC_ERR_INVALID_ARG;
    if (req->n_prompt > 0 && !req->prompt_tokens) return OC_ERR_INVALID_ARG;

    /* Check if at capacity (max_batch_size limits concurrent RUNNING + the
     * total array is dynamic, but admission uses max_batch_size). */
    if (sched->n_running >= sched->config.max_batch_size &&
        sched->config.enable_continuous_batching == false) {
        return OC_ERR_OOM;
    }

    /* Grow if needed. */
    if (sched->n_requests >= sched->capacity) {
        OcError e = sched_grow(sched);
        if (e != OC_OK) return e;
    }

    uint64_t id = sched->next_id++;
    OcSchedRequest *copy = request_clone(req, id);
    if (!copy) return OC_ERR_OOM;

    sched->requests[sched->n_requests++] = copy;
    if (out_id) *out_id = id;
    return OC_OK;
}

OcError oc_sched_cancel_request(OcScheduler *sched, uint64_t id)
{
    if (!sched) return OC_ERR_INVALID_ARG;
    size_t idx = sched_find(sched, id);
    if (idx == (size_t)-1) return OC_ERR_INVALID_ARG;
    OcSchedRequest *r = sched->requests[idx];
    if (r->status == OC_SCHED_STATUS_CANCELLED) return OC_OK;
    if (r->status == OC_SCHED_STATUS_COMPLETED) return OC_ERR_INVALID_ARG;
    if (r->status == OC_SCHED_STATUS_RUNNING) {
        sched->n_running--;
    }
    r->status = OC_SCHED_STATUS_CANCELLED;
    return OC_OK;
}

OcError oc_sched_complete_request(OcScheduler *sched, uint64_t id)
{
    if (!sched) return OC_ERR_INVALID_ARG;
    size_t idx = sched_find(sched, id);
    if (idx == (size_t)-1) return OC_ERR_INVALID_ARG;
    OcSchedRequest *r = sched->requests[idx];
    if (r->status == OC_SCHED_STATUS_COMPLETED) return OC_OK;
    if (r->status == OC_SCHED_STATUS_CANCELLED) return OC_ERR_INVALID_ARG;
    if (r->status == OC_SCHED_STATUS_RUNNING) {
        sched->n_running--;
    }
    r->status = OC_SCHED_STATUS_COMPLETED;
    sched->n_completed++;
    return OC_OK;
}

/* ─── Next batch ───────────────────────────────────────────────────────── */

/* Sort pending requests by priority (desc), then created_ms (asc, FIFO).
 * Uses simple selection: we build an index array of pending slots. */
OcError oc_sched_next_batch(OcScheduler *sched, uint64_t *out_ids,
                             uint32_t max_batch, uint32_t *out_n)
{
    if (!sched || !out_ids || !out_n) return OC_ERR_INVALID_ARG;
    *out_n = 0;
    if (max_batch == 0) return OC_OK;

    uint32_t selected = 0;
    uint32_t tokens_used = 0;

    /* Iterate: pick the best pending request each round (greedy).
     * Best = highest priority, then earliest created_ms. */
    while (selected < max_batch) {
        int best_idx = -1;
        for (uint32_t i = 0; i < sched->n_requests; i++) {
            OcSchedRequest *r = sched->requests[i];
            if (!r || r->status != OC_SCHED_STATUS_PENDING) continue;
            if (best_idx == -1) {
                best_idx = (int)i;
                continue;
            }
            OcSchedRequest *best = sched->requests[best_idx];
            if (r->priority > best->priority) {
                best_idx = (int)i;
            } else if (r->priority == best->priority &&
                       r->created_ms < best->created_ms) {
                best_idx = (int)i;
            }
        }
        if (best_idx == -1) break; /* no more pending */

        OcSchedRequest *r = sched->requests[best_idx];
        /* Check token budget. */
        uint32_t prompt_tokens = r->n_prompt;
        if (tokens_used + prompt_tokens > sched->config.max_tokens_total &&
            selected > 0) {
            break; /* would exceed budget; stop. */
        }

        out_ids[selected++] = r->id;
        tokens_used += prompt_tokens;
        r->status = OC_SCHED_STATUS_RUNNING;
        sched->n_running++;
    }

    *out_n = selected;
    return OC_OK;
}

/* ─── Accessors ─────────────────────────────────────────────────────────── */

uint32_t oc_sched_n_pending(const OcScheduler *sched)
{
    if (!sched) return 0;
    uint32_t count = 0;
    for (uint32_t i = 0; i < sched->n_requests; i++) {
        if (sched->requests[i] &&
            sched->requests[i]->status == OC_SCHED_STATUS_PENDING) {
            count++;
        }
    }
    return count;
}

uint32_t oc_sched_n_running(const OcScheduler *sched)
{
    if (!sched) return 0;
    return sched->n_running;
}

uint32_t oc_sched_n_completed(const OcScheduler *sched)
{
    if (!sched) return 0;
    return sched->n_completed;
}

/* ─── Free ──────────────────────────────────────────────────────────────── */

void oc_sched_free(OcScheduler *sched)
{
    if (!sched) return;
    for (uint32_t i = 0; i < sched->n_requests; i++) {
        if (sched->requests[i]) {
            oc_sched_request_free(sched->requests[i]);
            free(sched->requests[i]);
            sched->requests[i] = NULL;
        }
    }
    free(sched->requests);
    memset(sched, 0, sizeof(*sched));
}
