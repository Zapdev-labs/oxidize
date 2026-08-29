/*
 * rotorquant_cache.c — fused RotorQuant KV cache (3D rotors + int4).
 *
 * Inverse rotation is folded into the query/output: logits rotate Q once
 * and int4-dot; attention unrotates the value accumulator once.
 */
#include "oxidize/rotorquant_cache.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct OcRotorQuantPage {
    size_t layer;
    size_t head;
    size_t tokens;
    size_t first_position;
    float *key_scales;
    uint8_t *key_codes;
    float *value_scales;
    uint8_t *value_codes;
};

static float lcg_unit(uint64_t *state)
{
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)(*state >> 40) / (float)(1u << 24);
}

static void make_rotor(uint64_t *state, float *m)
{
    float ax = lcg_unit(state) * 2.0f - 1.0f;
    float ay = lcg_unit(state) * 2.0f - 1.0f;
    float az = lcg_unit(state) * 2.0f - 1.0f;
    const float n = sqrtf(ax * ax + ay * ay + az * az) + 1.0e-9f;
    ax /= n;
    ay /= n;
    az /= n;
    const float angle = lcg_unit(state) * 6.28318530718f;
    const float c = cosf(angle);
    const float s = sinf(angle);
    const float t = 1.0f - c;
    m[0] = t * ax * ax + c;
    m[1] = t * ax * ay - s * az;
    m[2] = t * ax * az + s * ay;
    m[3] = t * ax * ay + s * az;
    m[4] = t * ay * ay + c;
    m[5] = t * ay * az - s * ax;
    m[6] = t * ax * az - s * ay;
    m[7] = t * ay * az + s * ax;
    m[8] = t * az * az + c;
}

static int clampi(int v, int lo, int hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static void page_free(OcRotorQuantPage *p)
{
    if (!p) return;
    free(p->key_scales);
    free(p->key_codes);
    free(p->value_scales);
    free(p->value_codes);
    memset(p, 0, sizeof(*p));
}

static size_t blocks_per_row(const OcRotorQuantCache *c)
{
    return (c->config.head_dim + c->config.block_size - 1) / c->config.block_size;
}

static size_t code_bytes_row(const OcRotorQuantCache *c)
{
    return (c->config.head_dim + 1) / 2;
}

static int token_visible(size_t pos, size_t query_position)
{
    return query_position == (size_t)-1 || pos <= query_position;
}

static OcError rotate_into(const OcRotorQuantCache *c, const float *v, float *out)
{
    const size_t d = c->config.head_dim;
    const size_t groups = c->n_groups;
    size_t g;
    for (g = 0; g < groups; g++) {
        const float *m = c->rotors + g * 9;
        const float x = v[g * 3];
        const float y = v[g * 3 + 1];
        const float z = v[g * 3 + 2];
        out[g * 3]     = m[0] * x + m[1] * y + m[2] * z;
        out[g * 3 + 1] = m[3] * x + m[4] * y + m[5] * z;
        out[g * 3 + 2] = m[6] * x + m[7] * y + m[8] * z;
    }
    for (g = groups * 3; g < d; g++) out[g] = v[g];
    return OC_OK;
}

static OcError unrotate_into(const OcRotorQuantCache *c, const float *v, float *out)
{
    const size_t d = c->config.head_dim;
    const size_t groups = c->n_groups;
    size_t g;
    for (g = 0; g < groups; g++) {
        const float *m = c->rotors + g * 9;
        const float x = v[g * 3];
        const float y = v[g * 3 + 1];
        const float z = v[g * 3 + 2];
        out[g * 3]     = m[0] * x + m[3] * y + m[6] * z;
        out[g * 3 + 1] = m[1] * x + m[4] * y + m[7] * z;
        out[g * 3 + 2] = m[2] * x + m[5] * y + m[8] * z;
    }
    for (g = groups * 3; g < d; g++) out[g] = v[g];
    return OC_OK;
}

static OcError quantize_rows(OcRotorQuantCache *c, const float *src, size_t tokens,
                             float **scales_out, uint8_t **codes_out)
{
    const size_t d = c->config.head_dim;
    const size_t bpr = blocks_per_row(c);
    const size_t cb = code_bytes_row(c);
    float *scales = (float *)calloc(tokens * bpr, sizeof(float));
    uint8_t *codes = (uint8_t *)calloc(tokens * cb, 1);
    float *rotated = (float *)malloc(d * sizeof(float));
    size_t t, b, i;
    if (!scales || !codes || !rotated) {
        free(scales);
        free(codes);
        free(rotated);
        return OC_ERR_OOM;
    }
    for (t = 0; t < tokens; t++) {
        rotate_into(c, src + t * d, rotated);
        for (b = 0; b < bpr; b++) {
            const size_t start = b * c->config.block_size;
            const size_t end = start + c->config.block_size < d
                                   ? start + c->config.block_size
                                   : d;
            float max_abs = 0.0f;
            for (i = start; i < end; i++) {
                float a = fabsf(rotated[i]);
                if (a > max_abs) max_abs = a;
            }
            const float scale = max_abs > 0.0f ? max_abs / 7.0f : 0.0f;
            const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
            scales[t * bpr + b] = scale;
            for (i = start; i < end; i++) {
                const float v = rotated[i] * inv;
                const int q = clampi((int)(v + (v >= 0.0f ? 0.5f : -0.5f)),
                                     -7, 7) + 7;
                uint8_t *byte = &codes[t * cb + i / 2];
                if ((i & 1u) == 0)
                    *byte = (uint8_t)((*byte & 0xF0u) | (unsigned)q);
                else
                    *byte = (uint8_t)((*byte & 0x0Fu) | ((unsigned)q << 4));
            }
        }
    }
    free(rotated);
    *scales_out = scales;
    *codes_out = codes;
    return OC_OK;
}

void oc_rotorquant_cache_config_init(OcRotorQuantCacheConfig *cfg)
{
    if (!cfg) return;
    cfg->head_dim = 128;
    cfg->block_size = OC_ROTORQUANT_CACHE_DEFAULT_BLOCK;
    cfg->seed = OC_ROTORQUANT_CACHE_DEFAULT_SEED;
}

OcError oc_rotorquant_cache_init(OcRotorQuantCache *cache,
                                 const OcRotorQuantCacheConfig *cfg)
{
    size_t g;
    uint64_t state;
    if (!cache || !cfg) return OC_ERR_INVALID_ARG;
    if (cfg->head_dim == 0 || cfg->block_size == 0) return OC_ERR_INVALID_ARG;
    memset(cache, 0, sizeof(*cache));
    cache->config = *cfg;
    cache->n_groups = cfg->head_dim / 3;
    if (cache->n_groups > 0) {
        cache->rotors = (float *)malloc(cache->n_groups * 9 * sizeof(float));
        if (!cache->rotors) return OC_ERR_OOM;
        state = cfg->seed;
        for (g = 0; g < cache->n_groups; g++)
            make_rotor(&state, cache->rotors + g * 9);
    }
    return OC_OK;
}

void oc_rotorquant_cache_free(OcRotorQuantCache *cache)
{
    size_t i;
    if (!cache) return;
    for (i = 0; i < cache->n_pages; i++) page_free(&cache->pages[i]);
    free(cache->pages);
    free(cache->rotors);
    memset(cache, 0, sizeof(*cache));
}

void oc_rotorquant_cache_clear(OcRotorQuantCache *cache)
{
    size_t i;
    if (!cache) return;
    for (i = 0; i < cache->n_pages; i++) page_free(&cache->pages[i]);
    cache->n_pages = 0;
}

OcError oc_rotorquant_cache_store_page(OcRotorQuantCache *cache,
                                       size_t layer, size_t kv_head,
                                       const float *keys, const float *values,
                                       size_t n_tokens, size_t first_position)
{
    OcRotorQuantPage *np = NULL;
    OcError e;
    size_t i;
    int reused = 0;
    float *key_scales = NULL, *value_scales = NULL;
    uint8_t *key_codes = NULL, *value_codes = NULL;
    if (!cache || !keys || !values || n_tokens == 0) return OC_ERR_INVALID_ARG;
    e = quantize_rows(cache, keys, n_tokens, &key_scales, &key_codes);
    if (e != OC_OK) return e;
    e = quantize_rows(cache, values, n_tokens, &value_scales, &value_codes);
    if (e != OC_OK) {
        free(key_scales);
        free(key_codes);
        return e;
    }
    for (i = 0; i < cache->n_pages; i++) {
        if (cache->pages[i].layer == layer && cache->pages[i].head == kv_head &&
            cache->pages[i].first_position == first_position) {
            np = &cache->pages[i];
            reused = 1;
            break;
        }
    }
    if (!np) {
        if (cache->n_pages == cache->cap_pages) {
            size_t ncap = cache->cap_pages ? cache->cap_pages * 2 : 8;
            OcRotorQuantPage *grown = (OcRotorQuantPage *)realloc(
                cache->pages, ncap * sizeof(*grown));
            if (!grown) {
                free(key_scales); free(key_codes);
                free(value_scales); free(value_codes);
                return OC_ERR_OOM;
            }
            cache->pages = grown;
            cache->cap_pages = ncap;
        }
        np = &cache->pages[cache->n_pages];
        memset(np, 0, sizeof(*np));
    }
    if (reused) page_free(np);
    np->layer = layer;
    np->head = kv_head;
    np->tokens = n_tokens;
    np->first_position = first_position;
    np->key_scales = key_scales;
    np->key_codes = key_codes;
    np->value_scales = value_scales;
    np->value_codes = value_codes;
    if (!reused) cache->n_pages += 1;
    return OC_OK;
}

OcError oc_rotorquant_cache_rotate(const OcRotorQuantCache *cache,
                                   const float *v, size_t n, float *out)
{
    if (!cache || !v || !out) return OC_ERR_INVALID_ARG;
    if (n != cache->config.head_dim) return OC_ERR_INVALID_ARG;
    return rotate_into(cache, v, out);
}

OcError oc_rotorquant_cache_unrotate(const OcRotorQuantCache *cache,
                                     const float *v, size_t n, float *out)
{
    if (!cache || !v || !out) return OC_ERR_INVALID_ARG;
    if (n != cache->config.head_dim) return OC_ERR_INVALID_ARG;
    return unrotate_into(cache, v, out);
}

size_t oc_rotorquant_cache_n_logits(const OcRotorQuantCache *cache,
                                    size_t layer, size_t kv_head)
{
    size_t n = 0, i;
    if (!cache) return 0;
    for (i = 0; i < cache->n_pages; i++) {
        if (cache->pages[i].layer == layer && cache->pages[i].head == kv_head)
            n += cache->pages[i].tokens;
    }
    return n;
}

size_t oc_rotorquant_cache_page_count(const OcRotorQuantCache *cache)
{
    return cache ? cache->n_pages : 0;
}

bool oc_rotorquant_cache_page_view(const OcRotorQuantCache *cache, size_t index,
                                   OcRotorQuantPageView *view)
{
    const OcRotorQuantPage *p;
    if (!cache || !view || index >= cache->n_pages) return false;
    p = &cache->pages[index];
    view->layer = p->layer;
    view->kv_head = p->head;
    view->tokens = p->tokens;
    view->first_position = p->first_position;
    view->key_codes = p->key_codes;
    view->key_scales = p->key_scales;
    view->value_codes = p->value_codes;
    view->value_scales = p->value_scales;
    return true;
}

OcError oc_rotorquant_cache_logits(const OcRotorQuantCache *cache,
                                   size_t layer, size_t kv_head,
                                   const float *query, size_t query_n,
                                   size_t query_position,
                                   float *out, size_t out_cap, size_t *n_out)
{
    float *rq;
    size_t need, written = 0, pi, t, b, i;
    size_t d, bpr, cb;
    if (!cache || !query || !n_out) return OC_ERR_INVALID_ARG;
    if (query_n != cache->config.head_dim) return OC_ERR_INVALID_ARG;
    need = oc_rotorquant_cache_n_logits(cache, layer, kv_head);
    *n_out = need;
    if (need == 0) return OC_OK;
    if (!out || out_cap < need) return OC_ERR_INVALID_ARG;
    d = cache->config.head_dim;
    bpr = blocks_per_row(cache);
    cb = code_bytes_row(cache);
    rq = (float *)malloc(d * sizeof(float));
    if (!rq) return OC_ERR_OOM;
    rotate_into(cache, query, rq);
    for (pi = 0; pi < cache->n_pages; pi++) {
        const OcRotorQuantPage *p = &cache->pages[pi];
        if (p->layer != layer || p->head != kv_head) continue;
        for (t = 0; t < p->tokens; t++) {
            const uint8_t *codes;
            const float *scales;
            float logit = 0.0f;
            if (!token_visible(p->first_position + t, query_position)) continue;
            codes = p->key_codes + t * cb;
            scales = p->key_scales + t * bpr;
            for (b = 0; b < bpr; b++) {
                const size_t start = b * cache->config.block_size;
                const size_t end = start + cache->config.block_size < d
                                       ? start + cache->config.block_size
                                       : d;
                float sum = 0.0f;
                for (i = start; i < end; i++) {
                    const uint8_t byte = codes[i / 2];
                    const int q = (i & 1u) == 0 ? (byte & 0x0F) : (byte >> 4);
                    sum += rq[i] * (float)(q - 7);
                }
                logit += sum * scales[b];
            }
            out[written++] = logit;
        }
    }
    free(rq);
    *n_out = written;
    return OC_OK;
}

OcError oc_rotorquant_cache_attention(const OcRotorQuantCache *cache,
                                      size_t layer, size_t kv_head,
                                      const float *query, size_t query_n,
                                      size_t query_position,
                                      float *out)
{
    size_t n = 0, pi, t, b, i, score_index = 0;
    float *scores, *acc;
    float scale, max_score, sum;
    size_t d, bpr, cb;
    OcError e;
    if (!cache || !query || !out) return OC_ERR_INVALID_ARG;
    if (query_n != cache->config.head_dim) return OC_ERR_INVALID_ARG;
    d = cache->config.head_dim;

    n = oc_rotorquant_cache_n_logits(cache, layer, kv_head);
    for (i = 0; i < d; i++) out[i] = 0.0f;
    if (n == 0) return OC_OK;

    scores = (float *)malloc(n * sizeof(float));
    if (!scores) return OC_ERR_OOM;
    e = oc_rotorquant_cache_logits(cache, layer, kv_head, query, query_n,
                                   query_position, scores, n, &n);
    if (e != OC_OK) {
        free(scores);
        return e;
    }
    if (n == 0) {
        free(scores);
        return OC_OK;
    }
    scale = 1.0f / sqrtf((float)d);
    max_score = -INFINITY;
    for (i = 0; i < n; i++) {
        scores[i] *= scale;
        if (scores[i] > max_score) max_score = scores[i];
    }
    sum = 0.0f;
    for (i = 0; i < n; i++) {
        scores[i] = expf(scores[i] - max_score);
        sum += scores[i];
    }
    if (sum > 0.0f) {
        for (i = 0; i < n; i++) scores[i] /= sum;
    }

    acc = (float *)calloc(d, sizeof(float));
    if (!acc) {
        free(scores);
        return OC_ERR_OOM;
    }
    bpr = blocks_per_row(cache);
    cb = code_bytes_row(cache);
    for (pi = 0; pi < cache->n_pages; pi++) {
        const OcRotorQuantPage *p = &cache->pages[pi];
        if (p->layer != layer || p->head != kv_head) continue;
        for (t = 0; t < p->tokens; t++) {
            float weight;
            const uint8_t *codes;
            const float *scales;
            if (!token_visible(p->first_position + t, query_position)) continue;
            weight = scores[score_index++];
            if (weight < 1.0e-12f) continue;
            codes = p->value_codes + t * cb;
            scales = p->value_scales + t * bpr;
            for (b = 0; b < bpr; b++) {
                const size_t start = b * cache->config.block_size;
                const size_t end = start + cache->config.block_size < d
                                       ? start + cache->config.block_size
                                       : d;
                const float ws = weight * scales[b];
                for (i = start; i < end; i++) {
                    const uint8_t byte = codes[i / 2];
                    const int q = (i & 1u) == 0 ? (byte & 0x0F) : (byte >> 4);
                    acc[i] += ws * (float)(q - 7);
                }
            }
        }
    }
    unrotate_into(cache, acc, out);
    free(acc);
    free(scores);
    return OC_OK;
}

OcError oc_rotorquant_cache_stats(const OcRotorQuantCache *cache,
                                  OcRotorQuantCacheStats *out)
{
    size_t i;
    float coords;
    size_t bpr, cb;
    if (!cache || !out) return OC_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));
    bpr = blocks_per_row(cache);
    cb = code_bytes_row(cache);
    for (i = 0; i < cache->n_pages; i++) {
        const OcRotorQuantPage *p = &cache->pages[i];
        out->token_count += p->tokens;
        out->key_bytes += p->tokens * cb;
        out->value_bytes += p->tokens * cb;
        out->metadata_bytes += (p->tokens * bpr * 2) * sizeof(float);
    }
    out->f32_baseline_bytes =
        out->token_count * cache->config.head_dim * 2 * sizeof(float);
    coords = (float)(2 * out->token_count * cache->config.head_dim);
    if (coords > 0.0f) {
        out->total_bits_per_coord =
            (float)((out->key_bytes + out->value_bytes + out->metadata_bytes) * 8) /
            coords;
    }
    return OC_OK;
}

float oc_rotorquant_cache_compression_ratio(const OcRotorQuantCacheStats *st)
{
    size_t bytes;
    if (!st) return 1.0f;
    bytes = st->key_bytes + st->value_bytes + st->metadata_bytes;
    return bytes == 0 ? 1.0f
                      : (float)st->f32_baseline_bytes / (float)bytes;
}

OcError oc_rotorquant_cache_rewind(OcRotorQuantCache *cache, size_t n_keep)
{
    size_t i, w;
    if (!cache) return OC_ERR_INVALID_ARG;
    w = 0;
    for (i = 0; i < cache->n_pages; i++) {
        if (cache->pages[i].first_position >= n_keep) {
            page_free(&cache->pages[i]);
            continue;
        }
        if (cache->pages[i].first_position + cache->pages[i].tokens > n_keep)
            cache->pages[i].tokens = n_keep - cache->pages[i].first_position;
        if (w != i) {
            cache->pages[w] = cache->pages[i];
            memset(&cache->pages[i], 0, sizeof(cache->pages[i]));
        }
        w += 1;
    }
    cache->n_pages = w;
    return OC_OK;
}
