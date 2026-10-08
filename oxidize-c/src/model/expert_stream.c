#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include "oxidize/expert_stream.h"

#include "oxidize/llama.h"
#include "oxidize/log.h"
#include "oxidize/parallel.h"
#include "oxidize/util/mmap.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>

typedef struct {
    const uint8_t *base;
    size_t expert_bytes;
    OcMmap *mmap;
    size_t mmap_offset;
} OcExpertTensorSpan;

typedef struct {
    OcMmap *mmap;
    size_t  off;
    size_t  len;
} OcExpertFaultJob;

#define OC_ES_MAX_DENSE 24

typedef struct {
    uint32_t n_experts;
    OcExpertTensorSpan gate;
    OcExpertTensorSpan up;
    OcExpertTensorSpan down;
    /* Per-token dense weights (attention/delta/router/shared-expert) that are
     * re-read every decode step but never explicitly faulted: under page-cache
     * pressure they inline-fault inside matvecs at single-thread speed. */
    OcExpertFaultJob dense[OC_ES_MAX_DENSE];
    uint32_t n_dense;
} OcExpertLayerLayout;

typedef struct {
    uint32_t layer;
    uint32_t expert;
    uint64_t bytes;
} OcExpertHot;

struct OcExpertStreamPool {
    OcExpertStreamConfig cfg;
    OcLlamaModel *model;
    OcExpertLayerLayout *layers;
    uint32_t n_layers;
    uint32_t n_experts;
    uint8_t *cached;
    OcExpertHot *hot;
    size_t hot_len;
    size_t hot_cap;
    size_t hot_head;
    uint64_t resident_bytes;
    uint64_t prefetch_bytes;
    uint64_t reclaim_bytes;
    double   fault_ms;      /* wall time inside the blocking fault-in */
    uint64_t fault_bytes;   /* spans populated through the blocking path */
};

void oc_expert_stream_config_init(OcExpertStreamConfig *cfg)
{
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->cache_bytes = OC_EXPERT_STREAM_DEFAULT_CACHE;
    cfg->prefetch = true;
    cfg->reclaim = true;
}

static bool mul_size(size_t a, size_t b, size_t *out)
{
    if (b != 0 && a > SIZE_MAX / b) return false;
    *out = a * b;
    return true;
}

static bool span_from_view(const OcGgufMmappedFile *gguf, const OcWeightView *view,
                           size_t n_experts, size_t n_embd, size_t inter,
                           OcExpertTensorSpan *out)
{
    memset(out, 0, sizeof(*out));
    if (!view || !view->data || n_experts == 0) return false;
    size_t packed = 0;
    if (!mul_size((size_t)view->rows, view->row_bytes, &packed)) return false;
    size_t expert_bytes;
    if ((inter > 0 && view->rows == inter) ||
        (n_embd > 0 && view->rows == n_embd)) {
        expert_bytes = packed;
    } else if (view->rows % n_experts == 0) {
        expert_bytes = packed / n_experts;
    } else {
        expert_bytes = packed;
    }
    if (expert_bytes == 0) return false;
    size_t total = 0;
    if (!mul_size(expert_bytes, n_experts, &total)) return false;
    out->base = view->data;
    out->expert_bytes = expert_bytes;
    if (!gguf || !gguf->shards) return true;
    for (size_t i = 0; i < gguf->n_shards; i++) {
        const uint8_t *bytes = gguf->shards[i].bytes;
        size_t len = gguf->shards[i].len;
        if (!bytes || !gguf->shards[i].mmap) continue;
        if ((const uint8_t *)view->data < bytes) continue;
        size_t off = (size_t)((const uint8_t *)view->data - bytes);
        if (off > len || total > len - off) continue;
        out->mmap = gguf->shards[i].mmap;
        out->mmap_offset = off;
        return true;
    }
    return true;
}

static void advise_span(const OcExpertTensorSpan *span, uint32_t expert,
                        OcMmapAdvice advice)
{
    if (!span || !span->mmap || span->expert_bytes == 0) return;
    size_t off = span->mmap_offset + (size_t)expert * span->expert_bytes;
    (void)oc_mmap_advise_range(span->mmap, off, span->expert_bytes, advice);
}

static void fault_span(const OcExpertTensorSpan *span, uint32_t expert)
{
    if (!span || !span->mmap || span->expert_bytes == 0) return;
    size_t off = span->mmap_offset + (size_t)expert * span->expert_bytes;
    (void)oc_mmap_fault_range(span->mmap, off, span->expert_bytes);
}

static size_t cache_key(const OcExpertStreamPool *pool, uint32_t layer,
                        uint32_t expert)
{
    return (size_t)layer * pool->n_experts + expert;
}

static uint64_t expert_bytes(const OcExpertLayerLayout *L)
{
    if (!L) return 0;
    return (uint64_t)L->gate.expert_bytes + L->up.expert_bytes +
           L->down.expert_bytes;
}

static void reclaim_one(OcExpertStreamPool *pool)
{
    if (!pool || pool->hot_len == 0) return;
    OcExpertHot hot = pool->hot[pool->hot_head];
    pool->hot_head = (pool->hot_head + 1) % pool->hot_cap;
    pool->hot_len--;
    if (hot.layer >= pool->n_layers || hot.expert >= pool->n_experts) return;
    const OcExpertLayerLayout *L = &pool->layers[hot.layer];
    advise_span(&L->gate, hot.expert, OC_MMAP_ADVICE_DONTNEED);
    advise_span(&L->up, hot.expert, OC_MMAP_ADVICE_DONTNEED);
    advise_span(&L->down, hot.expert, OC_MMAP_ADVICE_DONTNEED);
    if (pool->cached)
        pool->cached[cache_key(pool, hot.layer, hot.expert)] = 0;
    if (pool->resident_bytes >= hot.bytes)
        pool->resident_bytes -= hot.bytes;
    else
        pool->resident_bytes = 0;
    pool->reclaim_bytes += hot.bytes;
}

static OcError record_hot(OcExpertStreamPool *pool, uint32_t layer,
                          uint32_t expert, uint64_t bytes)
{
    if (!pool->hot || pool->hot_cap == 0) return OC_OK;
    if (layer >= pool->n_layers || expert >= pool->n_experts) return OC_OK;
    if (pool->cached && pool->cached[cache_key(pool, layer, expert)]) {
        for (size_t i = 0; i < pool->hot_len; i++) {
            size_t idx = (pool->hot_head + i) % pool->hot_cap;
            if (pool->hot[idx].layer != layer || pool->hot[idx].expert != expert)
                continue;
            OcExpertHot h = pool->hot[idx];
            for (size_t j = i; j + 1 < pool->hot_len; j++) {
                size_t a = (pool->hot_head + j) % pool->hot_cap;
                size_t b = (pool->hot_head + j + 1) % pool->hot_cap;
                pool->hot[a] = pool->hot[b];
            }
            pool->hot[(pool->hot_head + pool->hot_len - 1) % pool->hot_cap] = h;
            return OC_OK;
        }
    }
    if (pool->hot_len == pool->hot_cap) reclaim_one(pool);
    size_t idx = (pool->hot_head + pool->hot_len) % pool->hot_cap;
    pool->hot[idx].layer = layer;
    pool->hot[idx].expert = expert;
    pool->hot[idx].bytes = bytes;
    pool->hot_len++;
    pool->resident_bytes += bytes;
    if (pool->cached)
        pool->cached[cache_key(pool, layer, expert)] = 1;
    while (pool->cfg.reclaim && pool->resident_bytes > pool->cfg.cache_bytes &&
           pool->hot_len > 1) {
        reclaim_one(pool);
    }
    return OC_OK;
}

static void push_dense(const OcGgufMmappedFile *gguf, const OcWeightView *view,
                       OcExpertLayerLayout *L)
{
    if (!view || !view->data || L->n_dense >= OC_ES_MAX_DENSE) return;
    OcExpertTensorSpan span;
    if (!span_from_view(gguf, view, 1, 0, 0, &span) || !span.mmap) return;
    size_t packed = (size_t)view->rows * view->row_bytes;
    if (packed == 0) return;
    OcExpertFaultJob *j = &L->dense[L->n_dense++];
    j->mmap = span.mmap;
    j->off  = span.mmap_offset;
    j->len  = packed;
}

/* Small per-layer weights are kept hot by normal LRU pressure; the dense set
 * targets the multi-MB tensors that otherwise inline-fault during decode. */
static void collect_dense(const OcGgufMmappedFile *gguf, const OcLlamaLayer *L,
                          OcExpertLayerLayout *out)
{
    push_dense(gguf, &L->attn_q, out);
    push_dense(gguf, &L->attn_k, out);
    push_dense(gguf, &L->attn_v, out);
    push_dense(gguf, &L->attn_output, out);
    push_dense(gguf, &L->attn_qkv, out);
    push_dense(gguf, &L->attn_gate, out);
    push_dense(gguf, &L->ssm_out, out);
    push_dense(gguf, &L->ssm_alpha, out);
    push_dense(gguf, &L->ssm_beta, out);
    push_dense(gguf, &L->ssm_conv1d, out);
    push_dense(gguf, &L->ssm_dt_bias, out);
    push_dense(gguf, &L->ffn_gate_inp, out);
    push_dense(gguf, &L->ffn_gate_inp_shexp, out);
    push_dense(gguf, &L->ffn_gate_shexp, out);
    push_dense(gguf, &L->ffn_up_shexp, out);
    push_dense(gguf, &L->ffn_down_shexp, out);
    /* qwen4exp hyper-connection + PLE tensors are also per-token dense. */
    push_dense(gguf, &L->hc_attn_down, out);
    push_dense(gguf, &L->hc_attn_up, out);
    push_dense(gguf, &L->hc_attn_inject, out);
    push_dense(gguf, &L->hc_ffn_down, out);
    push_dense(gguf, &L->hc_ffn_up, out);
    push_dense(gguf, &L->hc_ffn_inject, out);
    push_dense(gguf, &L->ple_key, out);
    push_dense(gguf, &L->ple_value, out);
}

OcError oc_expert_stream_new(OcLlamaModel *model, const OcExpertStreamConfig *cfg,
                             OcExpertStreamPool **out)
{
    if (!model || !out) return OC_ERR_INVALID_ARG;
    *out = NULL;
    OcExpertStreamPool *pool = calloc(1, sizeof(*pool));
    if (!pool) return OC_ERR_OOM;
    if (cfg) pool->cfg = *cfg;
    else oc_expert_stream_config_init(&pool->cfg);
    if (pool->cfg.cache_bytes == 0)
        pool->cfg.cache_bytes = OC_EXPERT_STREAM_DEFAULT_CACHE;
    pool->model = model;
    if (model->cfg.num_experts == 0) {
        *out = pool;
        return OC_OK;
    }
    pool->n_layers = model->cfg.n_layer;
    /* The MTP/nextn block carries its own expert pool — give it the slot
     * one past the last main layer (see moe_layer_index). */
    if (model->mtp.layer.ffn_gate_exps.data != NULL)
        pool->n_layers += 1;
    pool->layers = calloc(pool->n_layers, sizeof(*pool->layers));
    if (!pool->layers) {
        free(pool);
        return OC_ERR_OOM;
    }
    uint32_t n_exp = model->cfg.num_experts;
    pool->n_experts = n_exp;
    for (uint32_t i = 0; i < pool->n_layers; i++) {
        const OcLlamaLayer *L = (i < model->cfg.n_layer) ? &model->layers[i]
                                                       : &model->mtp.layer;
        pool->layers[i].n_experts = n_exp;
        size_t n_embd = model->cfg.n_embd;
        size_t inter = model->cfg.expert_intermediate_size;
        (void)span_from_view(&model->gguf, &L->ffn_gate_exps, n_exp, n_embd,
                             inter, &pool->layers[i].gate);
        (void)span_from_view(&model->gguf, &L->ffn_up_exps, n_exp, n_embd,
                             inter, &pool->layers[i].up);
        (void)span_from_view(&model->gguf, &L->ffn_down_exps, n_exp, n_embd,
                             inter, &pool->layers[i].down);
        collect_dense(&model->gguf, L, &pool->layers[i]);
    }
    size_t nslot = (size_t)pool->n_layers * (size_t)n_exp;
    pool->cached = calloc(nslot ? nslot : 1, 1);
    if (!pool->cached) {
        free(pool->layers);
        free(pool);
        return OC_ERR_OOM;
    }
    size_t cap = nslot;
    if (cap < 256) cap = 256;
    pool->hot = calloc(cap, sizeof(*pool->hot));
    if (!pool->hot) {
        free(pool->cached);
        free(pool->layers);
        free(pool);
        return OC_ERR_OOM;
    }
    pool->hot_cap = cap;
    *out = pool;
    return OC_OK;
}

void oc_expert_stream_free(OcExpertStreamPool *pool)
{
    if (!pool) return;
    free(pool->layers);
    free(pool->cached);
    free(pool->hot);
    free(pool);
}

static void advise_view(const OcGgufMmappedFile *gguf, const OcWeightView *view,
                        OcMmapAdvice advice)
{
    if (!view || !view->data) return;
    OcExpertTensorSpan span;
    if (!span_from_view(gguf, view, 1, 0, 0, &span) || !span.mmap) return;
    size_t bytes = (size_t)view->rows * view->row_bytes;
    (void)oc_mmap_advise_range(span.mmap, span.mmap_offset, bytes, advice);
}

OcError oc_expert_stream_prepare_mapping(OcExpertStreamPool *pool)
{
    if (!pool || !pool->model) return OC_ERR_INVALID_ARG;
    OcGgufMmappedFile *gguf = &pool->model->gguf;
    if (gguf->shards) {
        for (size_t i = 0; i < gguf->n_shards; i++) {
            if (gguf->shards[i].mmap)
                oc_mmap_advise_random(gguf->shards[i].mmap);
        }
    }
    advise_view(gguf, &pool->model->tok_embeddings, OC_MMAP_ADVICE_WILLNEED);
    advise_view(gguf, &pool->model->output, OC_MMAP_ADVICE_WILLNEED);
    for (uint32_t i = 0; i < pool->n_layers; i++) {
        const OcLlamaLayer *L = &pool->model->layers[i];
        advise_view(gguf, &L->attn_q, OC_MMAP_ADVICE_WILLNEED);
        advise_view(gguf, &L->attn_k, OC_MMAP_ADVICE_WILLNEED);
        advise_view(gguf, &L->attn_v, OC_MMAP_ADVICE_WILLNEED);
        advise_view(gguf, &L->attn_output, OC_MMAP_ADVICE_WILLNEED);
        advise_view(gguf, &L->attn_qkv, OC_MMAP_ADVICE_WILLNEED);
        advise_view(gguf, &L->attn_gate, OC_MMAP_ADVICE_WILLNEED);
        advise_view(gguf, &L->ssm_out, OC_MMAP_ADVICE_WILLNEED);
        advise_view(gguf, &L->ffn_gate_inp, OC_MMAP_ADVICE_WILLNEED);
        advise_view(gguf, &L->ffn_gate_shexp, OC_MMAP_ADVICE_WILLNEED);
        advise_view(gguf, &L->ffn_up_shexp, OC_MMAP_ADVICE_WILLNEED);
        advise_view(gguf, &L->ffn_down_shexp, OC_MMAP_ADVICE_WILLNEED);
        if (pool->layers[i].gate.mmap)
            advise_span(&pool->layers[i].gate, 0, OC_MMAP_ADVICE_RANDOM);
        if (pool->layers[i].up.mmap)
            advise_span(&pool->layers[i].up, 0, OC_MMAP_ADVICE_RANDOM);
        if (pool->layers[i].down.mmap)
            advise_span(&pool->layers[i].down, 0, OC_MMAP_ADVICE_RANDOM);
        /* RANDOM on expert 0 only covers that expert; mark the whole stacked
         * tensor random by advising the full span. */
        OcWeightView whole = L->ffn_gate_exps;
        advise_view(gguf, &whole, OC_MMAP_ADVICE_RANDOM);
        advise_view(gguf, &L->ffn_up_exps, OC_MMAP_ADVICE_RANDOM);
        advise_view(gguf, &L->ffn_down_exps, OC_MMAP_ADVICE_RANDOM);
    }
    oc_log(OC_LOG_INFO,
           "expert-stream: cache=%llu MiB layers=%u experts=%u",
           (unsigned long long)(pool->cfg.cache_bytes / (1024ull * 1024ull)),
           pool->n_layers, pool->model->cfg.num_experts);
    return OC_OK;
}

static void fault_jobs(size_t begin, size_t end, size_t tid, void *user)
{
    const OcExpertFaultJob *jobs = (const OcExpertFaultJob *)user;
    (void)tid;
    for (size_t i = begin; i < end; i++)
        (void)oc_mmap_fault_range(jobs[i].mmap, jobs[i].off, jobs[i].len);
}

static void push_fault_job(OcExpertFaultJob *jobs, size_t *n_jobs,
                           const OcExpertTensorSpan *span, uint32_t expert)
{
    if (!span->mmap || span->expert_bytes == 0) return;
    jobs[*n_jobs].mmap = span->mmap;
    jobs[*n_jobs].off  = span->mmap_offset + (size_t)expert * span->expert_bytes;
    jobs[*n_jobs].len  = span->expert_bytes;
    (*n_jobs)++;
}

OcError oc_expert_stream_touch(OcExpertStreamPool *pool, uint32_t layer,
                               const uint32_t *experts, uint32_t n,
                               bool prefetch)
{
    if (!pool || !experts) return OC_ERR_INVALID_ARG;
    if (layer >= pool->n_layers || n == 0) return OC_OK;
    if (prefetch && !pool->cfg.prefetch) return OC_OK;
    const OcExpertLayerLayout *L = &pool->layers[layer];
    uint64_t bytes = expert_bytes(L);
    OcMmapAdvice advice = OC_MMAP_ADVICE_WILLNEED;
    /* Issue all WILLNEED hints first so the kernel's async readahead pipeline
     * is primed for every selected expert before we start faulting. */
    for (uint32_t i = 0; i < n; i++) {
        uint32_t idx = experts[i];
        if (idx >= L->n_experts) continue;
        advise_span(&L->gate, idx, advice);
        advise_span(&L->up, idx, advice);
        advise_span(&L->down, idx, advice);
    }
    /* Fault-in is spread across the compute pool: each expert's gate/up/down
     * span populates on a different thread so the SSD's queue depth does the
     * overlapping instead of serializing on one fault stream. The layer's
     * dense weights ride along — they are re-read every token but are never
     * touched elsewhere, so without this they inline-fault one page at a
     * time inside the matvec threads (~5x slower than POPULATE's batched
     * readahead). */
    OcExpertFaultJob *jobs = NULL;
    if (!prefetch) {
        jobs = (OcExpertFaultJob *)malloc(((size_t)n * 3 +
                                          (size_t)OC_ES_MAX_DENSE) *
                                         sizeof(*jobs));
    }
    size_t n_jobs = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t idx = experts[i];
        if (idx >= L->n_experts) continue;
        if (jobs) {
            push_fault_job(jobs, &n_jobs, &L->gate, idx);
            push_fault_job(jobs, &n_jobs, &L->up, idx);
            push_fault_job(jobs, &n_jobs, &L->down, idx);
        } else if (!prefetch) {
            fault_span(&L->gate, idx);
            fault_span(&L->up, idx);
            fault_span(&L->down, idx);
        }
        if (prefetch) pool->prefetch_bytes += bytes;
        record_hot(pool, layer, idx, bytes);
    }
    if (jobs) {
        /* Fault this layer's dense weights (the not-yet-read FFN-side ones)
         * and async-prefetch the NEXT layer's dense set so its attention
         * tensors arrive while this layer's MoE compute runs. */
        for (uint32_t i = 0; i < L->n_dense; i++)
            jobs[n_jobs++] = L->dense[i];
        uint32_t next = (layer + 1) % pool->n_layers;
        const OcExpertLayerLayout *NL = &pool->layers[next];
        for (uint32_t i = 0; i < NL->n_dense; i++) {
            const OcExpertFaultJob *j = &NL->dense[i];
            (void)oc_mmap_advise_range(j->mmap, j->off, j->len,
                                     OC_MMAP_ADVICE_WILLNEED);
        }
    }
    if (n_jobs > 0) {
        struct timespec ts0, ts1;
        clock_gettime(CLOCK_MONOTONIC, &ts0);
        if (n_jobs > 1) oc_parallel_for(n_jobs, fault_jobs, jobs);
        else (void)oc_mmap_fault_range(jobs[0].mmap, jobs[0].off,
                                       jobs[0].len);
        clock_gettime(CLOCK_MONOTONIC, &ts1);
        pool->fault_ms += (double)(ts1.tv_sec - ts0.tv_sec) * 1e3 +
                          (double)(ts1.tv_nsec - ts0.tv_nsec) * 1e-6;
        pool->fault_bytes += (uint64_t)n_jobs * bytes / 3u;
    }
    free(jobs);
    return OC_OK;
}

OcError oc_expert_stream_touch_layer(OcExpertStreamPool *pool, uint32_t layer)
{
    if (!pool || layer >= pool->n_layers) return OC_ERR_INVALID_ARG;
    uint32_t n = pool->layers[layer].n_experts;
    if (n == 0) return OC_OK;
    uint32_t *idx = malloc((size_t)n * sizeof(*idx));
    if (!idx) return OC_ERR_OOM;
    for (uint32_t i = 0; i < n; i++) idx[i] = i;
    OcError e = oc_expert_stream_touch(pool, layer, idx, n, true);
    free(idx);
    return e;
}

uint64_t oc_expert_stream_resident_bytes(const OcExpertStreamPool *pool)
{
    return pool ? pool->resident_bytes : 0;
}

uint64_t oc_expert_stream_prefetch_bytes(const OcExpertStreamPool *pool)
{
    return pool ? pool->prefetch_bytes : 0;
}

uint64_t oc_expert_stream_reclaim_bytes(const OcExpertStreamPool *pool)
{
    return pool ? pool->reclaim_bytes : 0;
}

double oc_expert_stream_fault_ms(const OcExpertStreamPool *pool)
{
    return pool ? pool->fault_ms : 0.0;
}

uint64_t oc_expert_stream_fault_bytes(const OcExpertStreamPool *pool)
{
    return pool ? pool->fault_bytes : 0;
}
