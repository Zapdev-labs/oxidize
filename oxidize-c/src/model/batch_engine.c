#define _POSIX_C_SOURCE 200809L
#include "oxidize/batch_engine.h"

#include <stdlib.h>
#include <string.h>


typedef struct {
    OcSeqId  id;
    uint32_t *prompt;
    size_t   prompt_len;
    size_t   max_new;
    uint32_t stop_token;
    bool     has_stop;
} OcPending;

typedef struct {
    OcSeqId  id;
    size_t   pos;        /* absolute position of next token */
    uint32_t last_token; /* last produced token */
    size_t   generated;
    size_t   max_new;
    uint32_t stop_token;
    bool     has_stop;
    bool     finished;
} OcActiveMeta;

struct OcBatchEngine {
    OcBatchConfig  cfg;
    uint32_t       kv_layers;
    uint32_t       kv_row_len;
    OcPending     *pending;
    size_t         pending_count;
    size_t         pending_cap;
    OcActiveMeta  *active_meta;
    size_t         active_count;
    size_t         active_cap;
    OcSeqId        next_id;
    size_t         total_submitted;
    OcBatchForward forward;  /* optional forward callback */
};


void oc_batch_config_init(OcBatchConfig *cfg)
{
    if (!cfg) return;
    cfg->max_batch = 32;
    cfg->default_capacity_tokens = 2048;
}


OcError oc_batch_engine_init(OcBatchEngine **out, const OcBatchConfig *cfg,
                             uint32_t kv_layers, uint32_t kv_row_len)
{
    if (!out) return OC_ERR_INVALID_ARG;
    if (kv_layers == 0 || kv_row_len == 0) return OC_ERR_INVALID_ARG;

    OcBatchEngine *e = calloc(1, sizeof(*e));
    if (!e) return OC_ERR_OOM;

    if (cfg) e->cfg = *cfg;
    else oc_batch_config_init(&e->cfg);

    if (e->cfg.max_batch == 0) e->cfg.max_batch = 1;
    if (e->cfg.max_batch > OC_BATCH_MAX_SEQS) e->cfg.max_batch = OC_BATCH_MAX_SEQS;

    e->kv_layers = kv_layers;
    e->kv_row_len = kv_row_len;
    e->next_id = 1;
    e->total_submitted = 0;

    e->pending_cap = 64;
    e->pending = calloc(e->pending_cap, sizeof(OcPending));
    e->active_cap = e->cfg.max_batch;
    e->active_meta = calloc(e->active_cap, sizeof(OcActiveMeta));

    if (!e->pending || !e->active_meta) {
        free(e->pending);
        free(e->active_meta);
        free(e);
        return OC_ERR_OOM;
    }
    e->pending_count = 0;
    e->active_count = 0;

    *out = e;
    return OC_OK;
}

void oc_batch_engine_free(OcBatchEngine *engine)
{
    if (!engine) return;
    for (size_t i = 0; i < engine->pending_count; i++)
        free(engine->pending[i].prompt);
    free(engine->pending);
    free(engine->active_meta);
    free(engine);
}


OcError oc_batch_submit(OcBatchEngine *engine,
                        const uint32_t *prompt, size_t prompt_len,
                        size_t max_new, uint32_t stop_token, bool has_stop,
                        OcSeqId *out_id)
{
    if (!engine || !prompt || !out_id) return OC_ERR_INVALID_ARG;
    if (prompt_len == 0 || prompt_len > OC_BATCH_MAX_PROMPT) return OC_ERR_INVALID_ARG;

    /* Grow pending array if needed. */
    if (engine->pending_count >= engine->pending_cap) {
        size_t new_cap = engine->pending_cap * 2;
        OcPending *np = realloc(engine->pending, new_cap * sizeof(OcPending));
        if (!np) return OC_ERR_OOM;
        engine->pending = np;
        engine->pending_cap = new_cap;
    }

    OcPending *p = &engine->pending[engine->pending_count];
    p->id = engine->next_id++;
    p->prompt_len = prompt_len;
    p->max_new = max_new;
    p->stop_token = stop_token;
    p->has_stop = has_stop;
    p->prompt = malloc(prompt_len * sizeof(uint32_t));
    if (!p->prompt) return OC_ERR_OOM;
    memcpy(p->prompt, prompt, prompt_len * sizeof(uint32_t));

    engine->pending_count++;
    engine->total_submitted++;
    *out_id = p->id;
    return OC_OK;
}


size_t oc_batch_active_len(const OcBatchEngine *engine)
{
    return engine ? engine->active_count : 0;
}

size_t oc_batch_pending_len(const OcBatchEngine *engine)
{
    return engine ? engine->pending_count : 0;
}

bool oc_batch_has_work(const OcBatchEngine *engine)
{
    if (!engine) return false;
    return engine->pending_count > 0 || engine->active_count > 0;
}


bool oc_batch_cancel(OcBatchEngine *engine, OcSeqId id)
{
    if (!engine) return false;

    /* Check pending. */
    for (size_t i = 0; i < engine->pending_count; i++) {
        if (engine->pending[i].id == id) {
            free(engine->pending[i].prompt);
            for (size_t j = i; j < engine->pending_count - 1; j++)
                engine->pending[j] = engine->pending[j + 1];
            engine->pending_count--;
            return true;
        }
    }
    /* Check active. */
    for (size_t i = 0; i < engine->active_count; i++) {
        if (engine->active_meta[i].id == id) {
            engine->active_meta[i].finished = true;
            return true;
        }
    }
    return false;
}


static void admit_pending(OcBatchEngine *engine)
{
    while (engine->pending_count > 0 &&
           engine->active_count < engine->cfg.max_batch) {
        OcPending *p = &engine->pending[0];
        OcActiveMeta *a = &engine->active_meta[engine->active_count];
        a->id = p->id;
        a->pos = p->prompt_len;  /* next position after prefill */
        a->last_token = p->prompt[p->prompt_len - 1];
        a->generated = 1;       /* prefill produces first token */
        a->max_new = p->max_new;
        a->stop_token = p->stop_token;
        a->has_stop = p->has_stop;
        a->finished = false;
        engine->active_count++;

        /* Remove from pending (shift left). */
        free(p->prompt);
        for (size_t j = 0; j < engine->pending_count - 1; j++)
            engine->pending[j] = engine->pending[j + 1];
        engine->pending_count--;
    }
}

static void compact_active(OcBatchEngine *engine)
{
    size_t w = 0;
    for (size_t r = 0; r < engine->active_count; r++) {
        if (!engine->active_meta[r].finished) {
            if (w != r)
                engine->active_meta[w] = engine->active_meta[r];
            w++;
        }
    }
    engine->active_count = w;
}

OcError oc_batch_set_forward(OcBatchEngine *engine, OcBatchForward forward)
{
    if (!engine) return OC_ERR_INVALID_ARG;
    engine->forward = forward;
    return OC_OK;
}

OcError oc_batch_step(OcBatchEngine *engine,
                      OcBatchStepOutput *out, size_t max_out, size_t *n_out)
{
    if (!engine || !out || !n_out) return OC_ERR_INVALID_ARG;
    if (max_out == 0) return OC_ERR_INVALID_ARG;

    /* Admit pending requests. */
    admit_pending(engine);

    *n_out = 0;

    /* Decode each active sequence. */
    for (size_t i = 0; i < engine->active_count && *n_out < max_out; i++) {
        OcActiveMeta *a = &engine->active_meta[i];
        if (a->finished) continue;

        if (engine->forward.fn) {
            /* Real token generation via callback. */
            uint32_t next_token = 0;
            OcError e = engine->forward.fn(engine->forward.ctx, a->last_token,
                                           a->pos, a->id, &next_token);
            if (e != OC_OK) {
                a->finished = true;
                a->last_token = 0;
            } else {
                a->last_token = next_token;
            }
        } else {
            /* Fallback: simulate token generation (increment last token). */
            a->last_token = a->last_token + 1;
        }
        a->pos++;
        a->generated++;

        /* Check termination. */
        if (a->generated >= a->max_new) {
            a->finished = true;
        } else if (a->has_stop && a->last_token == a->stop_token) {
            a->finished = true;
        }

        out[*n_out].seq_id = a->id;
        out[*n_out].token = a->last_token;
        out[*n_out].finished = a->finished;
        (*n_out)++;
    }

    /* Remove finished sequences. */
    compact_active(engine);
    return OC_OK;
}


OcError oc_batch_seq_position(const OcBatchEngine *engine, OcSeqId id, size_t *out_pos)
{
    if (!engine || !out_pos) return OC_ERR_INVALID_ARG;
    for (size_t i = 0; i < engine->active_count; i++) {
        if (engine->active_meta[i].id == id) {
            *out_pos = engine->active_meta[i].pos;
            return OC_OK;
        }
    }
    return OC_ERR_INVALID_ARG;
}

size_t oc_batch_total_submitted(const OcBatchEngine *engine)
{
    return engine ? engine->total_submitted : 0;
}
