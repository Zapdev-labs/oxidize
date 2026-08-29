/*
 * helix_cache.c — polar 4-bit keys + Hadamard 3-bit values.
 *
 * RoPE relative term is freq * (query_pos - key_pos) so the rotated
 * pre-RoPE dot matches R_qpos q · R_kpos k. The C++ PR used the opposite
 * sign, which is invisible when phases are zero (cos is even) and wrong
 * otherwise.
 */
#include "oxidize/helix_cache.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define OC_HELIX_PI        3.14159265358979323846f
#define OC_HELIX_PHI_STEP  (2.0f * OC_HELIX_PI / 16.0f)
#define OC_HELIX_VALUE_PAD 8

typedef struct {
    size_t layer;
    size_t head;
    size_t page;
} OcHelixPageKey;

typedef struct {
    float *mu_phi;
    float *log_rho_min;
    float *log_rho_step;
    uint8_t *active_mask;
    uint8_t *codes;
    float *rho_lut;
    size_t n_pairs;
} OcHelixColdKey;

typedef struct {
    float *scales;
    uint8_t *codes;
    size_t code_bytes; /* excluding padding */
} OcHelixColdValue;

typedef struct {
    uint32_t uncertainty_counter;
    float recent_max_overlap;
    uint64_t access_count;
} OcHelixPromotion;

struct OcHelixPage {
    OcHelixPageKey key;
    OcHelixPageTier tier;
    size_t *positions;
    size_t n_tokens;
    OcHelixColdKey cold_key;
    OcHelixColdValue cold_value;
    float *hot_keys;
    float *hot_values;
    OcHelixPromotion promotion;
};

/* cos/sin of (k-8)*2π/16. k<8 have negative sine (k=1 is sin(-7π/8)). */
static const float k_phase_c[16] = {
    -1.00000000f, -0.92387953f, -0.70710678f, -0.38268343f,
     0.00000000f,  0.38268343f,  0.70710678f,  0.92387953f,
     1.00000000f,  0.92387953f,  0.70710678f,  0.38268343f,
     0.00000000f, -0.38268343f, -0.70710678f, -0.92387953f,
};
static const float k_phase_s[16] = {
     0.00000000f, -0.38268343f, -0.70710678f, -0.92387953f,
    -1.00000000f, -0.92387953f, -0.70710678f, -0.38268343f,
     0.00000000f,  0.38268343f,  0.70710678f,  0.92387953f,
     1.00000000f,  0.92387953f,  0.70710678f,  0.38268343f,
};

static size_t packed_bits_bytes(size_t bit_count)
{
    return (bit_count + 7) / 8;
}

static void set_bit(uint8_t *data, size_t index, int value)
{
    const size_t byte = index / 8;
    const uint8_t mask = (uint8_t)(1u << (index % 8));
    if (value)
        data[byte] = (uint8_t)(data[byte] | mask);
    else
        data[byte] = (uint8_t)(data[byte] & ~mask);
}

static int get_bit(const uint8_t *data, size_t index)
{
    return (data[index / 8] & (uint8_t)(1u << (index % 8))) != 0;
}

static float wrap_angle(float value)
{
    while (value <= -OC_HELIX_PI) value += 2.0f * OC_HELIX_PI;
    while (value > OC_HELIX_PI) value -= 2.0f * OC_HELIX_PI;
    return value;
}

static float rope_frequency(size_t pair, size_t head_dim, float theta)
{
    return powf(theta, -2.0f * (float)pair / (float)head_dim);
}

static void hadamard8(const float *src, float *dst)
{
    const float a0 = src[0] + src[1];
    const float a1 = src[0] - src[1];
    const float a2 = src[2] + src[3];
    const float a3 = src[2] - src[3];
    const float a4 = src[4] + src[5];
    const float a5 = src[4] - src[5];
    const float a6 = src[6] + src[7];
    const float a7 = src[6] - src[7];
    const float b0 = a0 + a2;
    const float b1 = a1 + a3;
    const float b2 = a0 - a2;
    const float b3 = a1 - a3;
    const float b4 = a4 + a6;
    const float b5 = a5 + a7;
    const float b6 = a4 - a6;
    const float b7 = a5 - a7;
    dst[0] = b0 + b4;
    dst[1] = b1 + b5;
    dst[2] = b2 + b6;
    dst[3] = b3 + b7;
    dst[4] = b0 - b4;
    dst[5] = b1 - b5;
    dst[6] = b2 - b6;
    dst[7] = b3 - b7;
}

static int same_key(const OcHelixPageKey *a, size_t layer, size_t head,
                    size_t page)
{
    return a->layer == layer && a->head == head && a->page == page;
}

static int clampi(int v, int lo, int hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static void cold_key_free(OcHelixColdKey *k)
{
    if (!k) return;
    free(k->mu_phi);
    free(k->log_rho_min);
    free(k->log_rho_step);
    free(k->active_mask);
    free(k->codes);
    free(k->rho_lut);
    memset(k, 0, sizeof(*k));
}

static void cold_value_free(OcHelixColdValue *v)
{
    if (!v) return;
    free(v->scales);
    free(v->codes);
    memset(v, 0, sizeof(*v));
}

static void page_reset(OcHelixPage *p)
{
    if (!p) return;
    free(p->positions);
    cold_key_free(&p->cold_key);
    cold_value_free(&p->cold_value);
    free(p->hot_keys);
    free(p->hot_values);
    memset(p, 0, sizeof(*p));
}

static OcHelixPage *find_page(OcHelixCache *cache, size_t layer, size_t head,
                              size_t page_id)
{
    size_t i;
    for (i = 0; i < cache->n_pages; i++) {
        if (same_key(&cache->pages[i].key, layer, head, page_id))
            return &cache->pages[i];
    }
    return NULL;
}

static const OcHelixPage *find_page_c(const OcHelixCache *cache, size_t layer,
                                      size_t head, size_t page_id)
{
    size_t i;
    for (i = 0; i < cache->n_pages; i++) {
        if (same_key(&cache->pages[i].key, layer, head, page_id))
            return &cache->pages[i];
    }
    return NULL;
}

static OcError grow_pages(OcHelixCache *cache)
{
    size_t ncap;
    OcHelixPage *grown;
    if (cache->n_pages < cache->cap_pages) return OC_OK;
    ncap = cache->cap_pages ? cache->cap_pages * 2 : 8;
    grown = (OcHelixPage *)realloc(cache->pages, ncap * sizeof(*grown));
    if (!grown) return OC_ERR_OOM;
    cache->pages = grown;
    cache->cap_pages = ncap;
    return OC_OK;
}

/* Steal `built` into the matching slot, or append a new slot. n_pages grows
 * only after the new page is fully constructed. An existing page is reset
 * only after `built` is ready. */
static OcError install_page(OcHelixCache *cache, OcHelixPage *built)
{
    OcHelixPage *existing;
    OcError e;
    if (!cache || !built) return OC_ERR_INVALID_ARG;
    existing = find_page(cache, built->key.layer, built->key.head,
                         built->key.page);
    if (existing) {
        page_reset(existing);
        *existing = *built;
        memset(built, 0, sizeof(*built));
        return OC_OK;
    }
    e = grow_pages(cache);
    if (e != OC_OK) return e;
    cache->pages[cache->n_pages] = *built;
    memset(built, 0, sizeof(*built));
    cache->n_pages += 1;
    return OC_OK;
}

static OcError decode_cold_page(const OcHelixCache *cache, const OcHelixPage *page,
                                float *keys, float *values)
{
    size_t d, pairs, groups, t, p, g, i;
    if (!cache || !page || !keys || !values) return OC_ERR_INVALID_ARG;
    if (page->tier != OC_HELIX_PAGE_COLD) return OC_ERR_INVALID_ARG;
    d = cache->config.head_dim;
    pairs = d / 2;
    groups = d / 8;
    for (t = 0; t < page->n_tokens; t++) {
        float *krow = keys + t * d;
        float *vrow = values + t * d;
        size_t row_byte = t * d * 3 / 8;
        for (p = 0; p < pairs; p++) {
            const size_t idx = t * pairs + p;
            uint8_t code;
            float rho, mu, c, s, cmu, smu;
            if (!get_bit(page->cold_key.active_mask, idx)) {
                krow[2 * p] = 0.0f;
                krow[2 * p + 1] = 0.0f;
                continue;
            }
            code = page->cold_key.codes[idx];
            rho = page->cold_key.rho_lut[p * 16 + (code >> 4)];
            mu = page->cold_key.mu_phi[p];
            c = k_phase_c[code & 15];
            s = k_phase_s[code & 15];
            cmu = cosf(mu);
            smu = sinf(mu);
            krow[2 * p] = rho * (c * cmu - s * smu);
            krow[2 * p + 1] = rho * (s * cmu + c * smu);
        }
        for (g = 0; g < groups; g++) {
            uint32_t word = 0;
            float transformed[8], inverse[8];
            memcpy(&word, page->cold_value.codes + row_byte + g * 3, 3);
            for (i = 0; i < 8; i++)
                transformed[i] =
                    (float)((int)((word >> (3 * i)) & 7u) - 3) *
                    page->cold_value.scales[g];
            hadamard8(transformed, inverse);
            for (i = 0; i < 8; i++)
                vrow[g * 8 + i] = inverse[i] * 0.125f;
        }
    }
    return OC_OK;
}

/* Decode a truncated cold page to f32 so append can extend the prefix. */
static OcError thaw_page_for_append(OcHelixCache *cache, OcHelixPage *page)
{
    float *keys, *values;
    size_t *pos;
    OcHelixPageKey key;
    size_t d, n;
    OcError e;
    if (!cache || !page) return OC_ERR_INVALID_ARG;
    if (page->tier != OC_HELIX_PAGE_COLD) return OC_OK;
    d = cache->config.head_dim;
    n = page->n_tokens;
    keys = (float *)malloc(n * d * sizeof(float));
    values = (float *)malloc(n * d * sizeof(float));
    if (!keys || !values) {
        free(keys);
        free(values);
        return OC_ERR_OOM;
    }
    e = decode_cold_page(cache, page, keys, values);
    if (e != OC_OK) {
        free(keys);
        free(values);
        return e;
    }
    key = page->key;
    pos = page->positions;
    page->positions = NULL;
    page_reset(page);
    page->key = key;
    page->tier = OC_HELIX_PAGE_HOT;
    page->positions = pos;
    page->n_tokens = n;
    page->hot_keys = keys;
    page->hot_values = values;
    return OC_OK;
}

static void softmax_inplace(float *scores, size_t n, float scale)
{
    size_t i;
    float max_score = -INFINITY;
    float sum = 0.0f;
    for (i = 0; i < n; i++) {
        scores[i] *= scale;
        if (scores[i] > max_score) max_score = scores[i];
    }
    for (i = 0; i < n; i++) {
        scores[i] = expf(scores[i] - max_score);
        sum += scores[i];
    }
    if (sum <= 0.0f) return;
    for (i = 0; i < n; i++) scores[i] /= sum;
}

void oc_helix_cache_config_init(OcHelixCacheConfig *cfg)
{
    if (!cfg) return;
    cfg->page_size = 64;
    cfg->head_dim = 128;
    cfg->rope_dim = 0;
    cfg->key_radius_bits = 4;
    cfg->key_phase_bits = 4;
    cfg->value_bits = 3;
    cfg->inactive_threshold = 0.0f;
    cfg->promotion_epsilon = 0.1f;
    cfg->promotion_budget = 3;
}

OcError oc_helix_cache_init(OcHelixCache *cache, const OcHelixCacheConfig *cfg)
{
    if (!cache || !cfg) return OC_ERR_INVALID_ARG;
    if (cfg->page_size == 0) return OC_ERR_INVALID_ARG;
    if (cfg->head_dim == 0 || (cfg->head_dim % 2) != 0 ||
        (cfg->head_dim % 8) != 0)
        return OC_ERR_INVALID_ARG;
    if (cfg->key_radius_bits != 4 || cfg->key_phase_bits != 4 ||
        cfg->value_bits != 3)
        return OC_ERR_INVALID_ARG;
    if (cfg->rope_dim != 0 && (cfg->rope_dim % 2) != 0)
        return OC_ERR_INVALID_ARG;
    memset(cache, 0, sizeof(*cache));
    cache->config = *cfg;
    if (cache->config.rope_dim == 0 || cache->config.rope_dim > cfg->head_dim)
        cache->config.rope_dim = cfg->head_dim;
    return OC_OK;
}

void oc_helix_cache_free(OcHelixCache *cache)
{
    size_t i;
    if (!cache) return;
    for (i = 0; i < cache->n_pages; i++) page_reset(&cache->pages[i]);
    free(cache->pages);
    memset(cache, 0, sizeof(*cache));
}

void oc_helix_cache_clear(OcHelixCache *cache)
{
    size_t i;
    if (!cache) return;
    for (i = 0; i < cache->n_pages; i++) page_reset(&cache->pages[i]);
    cache->n_pages = 0;
}

static OcError copy_positions(OcHelixPage *page, const size_t *positions,
                              size_t n)
{
    page->positions = (size_t *)malloc(n * sizeof(size_t));
    if (!page->positions) return OC_ERR_OOM;
    memcpy(page->positions, positions, n * sizeof(size_t));
    page->n_tokens = n;
    return OC_OK;
}

OcError oc_helix_cache_store_hot_page(OcHelixCache *cache,
                                      size_t layer, size_t kv_head,
                                      size_t page_id,
                                      const float *pre_rope_keys,
                                      const float *values,
                                      const size_t *positions,
                                      size_t n_tokens)
{
    OcHelixPage built;
    OcHelixPage *page = &built;
    OcError e;
    size_t d;
    if (!cache || !pre_rope_keys || !values || !positions) return OC_ERR_INVALID_ARG;
    if (n_tokens == 0 || n_tokens > cache->config.page_size)
        return OC_ERR_INVALID_ARG;
    d = cache->config.head_dim;
    memset(&built, 0, sizeof(built));
    page->key.layer = layer;
    page->key.head = kv_head;
    page->key.page = page_id;
    page->tier = OC_HELIX_PAGE_HOT;
    e = copy_positions(page, positions, n_tokens);
    if (e != OC_OK) {
        page_reset(page);
        return e;
    }
    page->hot_keys = (float *)malloc(n_tokens * d * sizeof(float));
    page->hot_values = (float *)malloc(n_tokens * d * sizeof(float));
    if (!page->hot_keys || !page->hot_values) {
        page_reset(page);
        return OC_ERR_OOM;
    }
    memcpy(page->hot_keys, pre_rope_keys, n_tokens * d * sizeof(float));
    memcpy(page->hot_values, values, n_tokens * d * sizeof(float));
    e = install_page(cache, &built);
    if (e != OC_OK) {
        page_reset(&built);
        return e;
    }
    return OC_OK;
}

OcError oc_helix_cache_store_cold_page(OcHelixCache *cache,
                                       size_t layer, size_t kv_head,
                                       size_t page_id,
                                       const float *pre_rope_keys,
                                       const float *values,
                                       const size_t *positions,
                                       size_t n_tokens)
{
    OcHelixPage built;
    OcHelixPage *page = &built;
    OcError e;
    const size_t d = cache ? cache->config.head_dim : 0;
    const size_t pairs = d / 2;
    size_t codes;
    size_t t, p, g, i, k;
    float *sc_log_rho = NULL, *sc_phi = NULL;
    float *sin_sum = NULL, *cos_sum = NULL;
    float *min_log = NULL, *max_log = NULL, *inv_step = NULL;
    float *transformed = NULL;
    if (!cache || !pre_rope_keys || !values || !positions)
        return OC_ERR_INVALID_ARG;
    if (n_tokens == 0 || n_tokens > cache->config.page_size)
        return OC_ERR_INVALID_ARG;
    codes = n_tokens * pairs;
    memset(&built, 0, sizeof(built));
    page->key.layer = layer;
    page->key.head = kv_head;
    page->key.page = page_id;
    page->tier = OC_HELIX_PAGE_COLD;
    e = copy_positions(page, positions, n_tokens);
    if (e != OC_OK) {
        page_reset(page);
        return e;
    }

    page->cold_key.n_pairs = pairs;
    page->cold_key.mu_phi = (float *)calloc(pairs, sizeof(float));
    page->cold_key.log_rho_min = (float *)calloc(pairs, sizeof(float));
    page->cold_key.log_rho_step = (float *)calloc(pairs, sizeof(float));
    page->cold_key.active_mask = (uint8_t *)calloc(packed_bits_bytes(codes), 1);
    page->cold_key.codes = (uint8_t *)calloc(codes, 1);
    page->cold_key.rho_lut = (float *)calloc(pairs * 16, sizeof(float));
    if (!page->cold_key.mu_phi || !page->cold_key.log_rho_min ||
        !page->cold_key.log_rho_step || !page->cold_key.active_mask ||
        !page->cold_key.codes || !page->cold_key.rho_lut) {
        page_reset(page);
        return OC_ERR_OOM;
    }

    sc_log_rho = (float *)calloc(codes, sizeof(float));
    sc_phi = (float *)calloc(codes, sizeof(float));
    sin_sum = (float *)calloc(pairs, sizeof(float));
    cos_sum = (float *)calloc(pairs, sizeof(float));
    min_log = (float *)malloc(pairs * sizeof(float));
    max_log = (float *)malloc(pairs * sizeof(float));
    inv_step = (float *)calloc(pairs, sizeof(float));
    if (!sc_log_rho || !sc_phi || !sin_sum || !cos_sum || !min_log ||
        !max_log || !inv_step) {
        free(sc_log_rho); free(sc_phi); free(sin_sum); free(cos_sum);
        free(min_log); free(max_log); free(inv_step);
        page_reset(page);
        return OC_ERR_OOM;
    }
    for (p = 0; p < pairs; p++) {
        min_log[p] = INFINITY;
        max_log[p] = -INFINITY;
    }

    for (t = 0; t < n_tokens; t++) {
        const float *row = pre_rope_keys + t * d;
        for (p = 0; p < pairs; p++) {
            const float x = row[2 * p];
            const float y = row[2 * p + 1];
            const float rho = sqrtf(x * x + y * y);
            const size_t idx = t * pairs + p;
            if (rho <= 0.0f || rho < cache->config.inactive_threshold) continue;
            set_bit(page->cold_key.active_mask, idx, 1);
            sc_phi[idx] = atan2f(y, x);
            sc_log_rho[idx] = logf(rho + 1.0e-12f);
            if (rho > 0.0f) {
                sin_sum[p] += y / rho;
                cos_sum[p] += x / rho;
            }
            if (sc_log_rho[idx] < min_log[p]) min_log[p] = sc_log_rho[idx];
            if (sc_log_rho[idx] > max_log[p]) max_log[p] = sc_log_rho[idx];
        }
    }

    for (p = 0; p < pairs; p++) {
        float mu, step;
        if (!isfinite(min_log[p])) {
            min_log[p] = 0.0f;
            continue;
        }
        mu = atan2f(sin_sum[p], cos_sum[p]);
        step = (max_log[p] > min_log[p]) ? (max_log[p] - min_log[p]) / 15.0f
                                         : 0.0f;
        page->cold_key.mu_phi[p] = mu;
        page->cold_key.log_rho_min[p] = min_log[p];
        page->cold_key.log_rho_step[p] = step;
        for (k = 0; k < 16; k++)
            page->cold_key.rho_lut[p * 16 + k] =
                expf(min_log[p] + step * (float)k);
        inv_step[p] = step == 0.0f ? 0.0f : 1.0f / step;
    }

    for (t = 0; t < n_tokens; t++) {
        for (p = 0; p < pairs; p++) {
            const size_t idx = t * pairs + p;
            int rho_code, phi_code;
            float delta;
            if (!get_bit(page->cold_key.active_mask, idx)) continue;
            rho_code = clampi((int)((sc_log_rho[idx] - min_log[p]) * inv_step[p] +
                                    0.5f),
                              0, 15);
            delta = wrap_angle(sc_phi[idx] - page->cold_key.mu_phi[p]);
            phi_code = clampi((int)lroundf(delta / OC_HELIX_PHI_STEP) + 8, 0, 15);
            page->cold_key.codes[idx] =
                (uint8_t)((rho_code << 4) | phi_code);
        }
    }

    free(sc_log_rho); free(sc_phi); free(sin_sum); free(cos_sum);
    free(min_log); free(max_log); free(inv_step);

    {
        const size_t groups = d / 8;
        page->cold_value.scales = (float *)calloc(groups, sizeof(float));
        page->cold_value.code_bytes = packed_bits_bytes(n_tokens * d * 3);
        page->cold_value.codes = (uint8_t *)calloc(
            page->cold_value.code_bytes + OC_HELIX_VALUE_PAD, 1);
        transformed = (float *)malloc(n_tokens * 8 * sizeof(float));
        if (!page->cold_value.scales || !page->cold_value.codes ||
            !transformed) {
            free(transformed);
            page_reset(page);
            return OC_ERR_OOM;
        }
        for (g = 0; g < groups; g++) {
            float max_abs = 0.0f;
            float scale, inv_scale;
            for (t = 0; t < n_tokens; t++) {
                hadamard8(values + t * d + g * 8, transformed + t * 8);
                for (i = 0; i < 8; i++) {
                    float a = fabsf(transformed[t * 8 + i]);
                    if (a > max_abs) max_abs = a;
                }
            }
            scale = max_abs > 0.0f ? max_abs / 3.0f : 0.0f;
            inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
            page->cold_value.scales[g] = scale;
            for (t = 0; t < n_tokens; t++) {
                uint32_t word = 0;
                size_t byte;
                for (i = 0; i < 8; i++) {
                    const float v = transformed[t * 8 + i] * inv_scale;
                    const int q = clampi(
                        (int)(v + (v >= 0.0f ? 0.5f : -0.5f)), -3, 3);
                    word |= (uint32_t)(q + 3) << (3 * i);
                }
                byte = (t * d + g * 8) * 3 / 8;
                page->cold_value.codes[byte] = (uint8_t)word;
                page->cold_value.codes[byte + 1] = (uint8_t)(word >> 8);
                page->cold_value.codes[byte + 2] = (uint8_t)(word >> 16);
            }
        }
        free(transformed);
    }
    e = install_page(cache, &built);
    if (e != OC_OK) {
        page_reset(&built);
        return e;
    }
    return OC_OK;
}

static OcError extend_hot_page(OcHelixPage *page, size_t d, size_t add,
                               const float *keys, const float *values,
                               const size_t *positions)
{
    size_t n = page->n_tokens + add;
    size_t *pos = (size_t *)realloc(page->positions, n * sizeof(size_t));
    float *hk = (float *)realloc(page->hot_keys, n * d * sizeof(float));
    float *hv = (float *)realloc(page->hot_values, n * d * sizeof(float));
    if (!pos || !hk || !hv) {
        page->positions = pos ? pos : page->positions;
        page->hot_keys = hk ? hk : page->hot_keys;
        page->hot_values = hv ? hv : page->hot_values;
        return OC_ERR_OOM;
    }
    memcpy(pos + page->n_tokens, positions, add * sizeof(size_t));
    memcpy(hk + page->n_tokens * d, keys, add * d * sizeof(float));
    memcpy(hv + page->n_tokens * d, values, add * d * sizeof(float));
    page->positions = pos;
    page->hot_keys = hk;
    page->hot_values = hv;
    page->n_tokens = n;
    return OC_OK;
}

static OcHelixPage *find_open_hot(OcHelixCache *cache, size_t layer,
                                  size_t kv_head, size_t page_id)
{
    size_t i;
    for (i = 0; i < cache->n_pages; i++) {
        OcHelixPage *p = &cache->pages[i];
        if (p->key.layer == layer && p->key.head == kv_head &&
            p->key.page == page_id &&
            p->tier == OC_HELIX_PAGE_HOT &&
            p->n_tokens < cache->config.page_size)
            return p;
    }
    return NULL;
}

static OcError freeze_hot_page(OcHelixCache *cache, OcHelixPage *page)
{
    float *keys, *values;
    size_t *pos;
    size_t n, layer, head, pid;
    OcError e;
    if (!page || page->tier != OC_HELIX_PAGE_HOT || page->n_tokens == 0)
        return OC_OK;
    keys = page->hot_keys;
    values = page->hot_values;
    pos = page->positions;
    n = page->n_tokens;
    layer = page->key.layer;
    head = page->key.head;
    pid = page->key.page;
    page->hot_keys = NULL;
    page->hot_values = NULL;
    page->positions = NULL;
    page->n_tokens = 0;
    e = oc_helix_cache_store_cold_page(cache, layer, head, pid, keys, values,
                                       pos, n);
    free(keys);
    free(values);
    free(pos);
    return e;
}

OcError oc_helix_cache_append(OcHelixCache *cache,
                              size_t layer, size_t kv_head,
                              const float *pre_rope_keys,
                              const float *values,
                              const size_t *positions,
                              size_t n_tokens)
{
    const size_t d = cache ? cache->config.head_dim : 0;
    size_t offset = 0;
    if (!cache || !pre_rope_keys || !values || !positions || n_tokens == 0)
        return OC_ERR_INVALID_ARG;
    while (offset < n_tokens) {
        size_t page_id = positions[offset] / cache->config.page_size;
        OcHelixPage *page = find_open_hot(cache, layer, kv_head, page_id);
        size_t room, take;
        int is_new = 0;
        OcError e;
        if (!page) {
            OcHelixPage *existing = find_page(cache, layer, kv_head, page_id);
            if (existing && existing->n_tokens < cache->config.page_size) {
                e = thaw_page_for_append(cache, existing);
                if (e != OC_OK) return e;
                page = existing;
            }
        }
        if (!page) {
            e = grow_pages(cache);
            if (e != OC_OK) return e;
            page = &cache->pages[cache->n_pages];
            memset(page, 0, sizeof(*page));
            page->key.layer = layer;
            page->key.head = kv_head;
            page->key.page = page_id;
            page->tier = OC_HELIX_PAGE_HOT;
            page->n_tokens = 0;
            is_new = 1;
        }
        room = cache->config.page_size - page->n_tokens;
        take = n_tokens - offset;
        if (take > room) take = room;
        if (take == 0) {
            if (is_new) page_reset(page);
            return OC_ERR_INVALID_ARG;
        }
        e = extend_hot_page(page, d, take, pre_rope_keys + offset * d,
                            values + offset * d, positions + offset);
        if (e != OC_OK) {
            if (is_new) page_reset(page);
            return e;
        }
        if (is_new) cache->n_pages += 1;
        offset += take;
        if (page->n_tokens >= cache->config.page_size) {
            e = freeze_hot_page(cache, page);
            if (e != OC_OK) return e;
        }
    }
    return OC_OK;
}

size_t oc_helix_cache_n_logits(const OcHelixCache *cache, size_t layer,
                               size_t kv_head)
{
    size_t n = 0, i;
    if (!cache) return 0;
    for (i = 0; i < cache->n_pages; i++) {
        if (cache->pages[i].key.layer == layer &&
            cache->pages[i].key.head == kv_head)
            n += cache->pages[i].n_tokens;
    }
    return n;
}

size_t oc_helix_cache_page_count(const OcHelixCache *cache)
{
    return cache ? cache->n_pages : 0;
}

bool oc_helix_cache_cold_page_view(const OcHelixCache *cache, size_t index,
                                   OcHelixColdPageView *view)
{
    const OcHelixPage *p;
    if (!cache || !view || index >= cache->n_pages) return false;
    p = &cache->pages[index];
    if (p->tier != OC_HELIX_PAGE_COLD) return false;
    view->layer = p->key.layer;
    view->kv_head = p->key.head;
    view->page_id = p->key.page;
    view->tokens = p->n_tokens;
    view->positions = p->positions;
    view->key_codes = p->cold_key.codes;
    view->active_mask = p->cold_key.active_mask;
    view->mu_phi = p->cold_key.mu_phi;
    view->log_rho_min = p->cold_key.log_rho_min;
    view->log_rho_step = p->cold_key.log_rho_step;
    view->value_codes = p->cold_value.codes;
    view->value_scales = p->cold_value.scales;
    return true;
}

static void seed_query(const OcHelixPage *page, const float *query,
                       size_t pairs, const float *freq, float rel, int cold,
                       float *qx, float *qy)
{
    size_t p;
    for (p = 0; p < pairs; p++) {
        const float theta = freq[p] * rel - (cold ? page->cold_key.mu_phi[p] : 0.0f);
        const float c = cosf(theta);
        const float s = sinf(theta);
        const float x = query[2 * p];
        const float y = query[2 * p + 1];
        qx[p] = x * c - y * s;
        qy[p] = x * s + y * c;
    }
}

OcError oc_helix_cache_logits(const OcHelixCache *cache,
                              size_t layer, size_t kv_head,
                              const float *query_pre_rope, size_t query_n,
                              size_t query_position, float rope_theta,
                              float *out, size_t out_cap, size_t *n_out)
{
    size_t d, pairs;
    size_t need, written = 0, pi, t, p;
    float *freq = NULL, *step_c = NULL, *step_s = NULL, *qx = NULL, *qy = NULL;
    if (!cache || !query_pre_rope || !n_out) return OC_ERR_INVALID_ARG;
    if (query_n != cache->config.head_dim) return OC_ERR_INVALID_ARG;
    if (rope_theta <= 0.0f) return OC_ERR_INVALID_ARG;
    d = cache->config.head_dim;
    pairs = d / 2;
    {
        const size_t rope_dim = cache->config.rope_dim ? cache->config.rope_dim : d;
        const size_t rope_pairs = rope_dim / 2;
        need = oc_helix_cache_n_logits(cache, layer, kv_head);
        *n_out = need;
        if (need == 0) return OC_OK;
        if (!out || out_cap < need) return OC_ERR_INVALID_ARG;

        freq = (float *)malloc(pairs * sizeof(float));
        step_c = (float *)malloc(pairs * sizeof(float));
        step_s = (float *)malloc(pairs * sizeof(float));
        qx = (float *)malloc(pairs * sizeof(float));
        qy = (float *)malloc(pairs * sizeof(float));
        if (!freq || !step_c || !step_s || !qx || !qy) {
            free(freq); free(step_c); free(step_s); free(qx); free(qy);
            return OC_ERR_OOM;
        }
        for (p = 0; p < pairs; p++) {
            if (p < rope_pairs) {
                freq[p] = rope_frequency(p, rope_dim, rope_theta);
                step_c[p] = cosf(freq[p]);
                step_s[p] = sinf(freq[p]);
            } else {
                freq[p] = 0.0f;
                step_c[p] = 1.0f;
                step_s[p] = 0.0f;
            }
        }
    }

    for (pi = 0; pi < cache->n_pages; pi++) {
        const OcHelixPage *page = &cache->pages[pi];
        const int cold = page->tier == OC_HELIX_PAGE_COLD;
        if (page->key.layer != layer || page->key.head != kv_head) continue;
        if (page->n_tokens == 0) continue;
        /* P1: relative RoPE is query_pos - key_pos, not the reverse. */
        seed_query(page, query_pre_rope, pairs, freq,
                   (float)query_position - (float)page->positions[0], cold,
                   qx, qy);
        for (t = 0; t < page->n_tokens; t++) {
            float logit = 0.0f;
            if (page->positions[t] > query_position) continue;
            if (t > 0) {
                if (page->positions[t] == page->positions[t - 1] + 1) {
                    for (p = 0; p < pairs; p++) {
                        const float nx = qx[p] * step_c[p] + qy[p] * step_s[p];
                        qy[p] = -qx[p] * step_s[p] + qy[p] * step_c[p];
                        qx[p] = nx;
                    }
                } else {
                    seed_query(page, query_pre_rope, pairs, freq,
                               (float)query_position -
                                   (float)page->positions[t],
                               cold, qx, qy);
                }
            }
            if (cold) {
                const size_t base = t * pairs;
                for (p = 0; p < pairs; p++) {
                    const size_t idx = base + p;
                    uint8_t code;
                    float rho;
                    if (!get_bit(page->cold_key.active_mask, idx)) continue;
                    code = page->cold_key.codes[idx];
                    rho = page->cold_key.rho_lut[p * 16 + (code >> 4)];
                    logit += rho * (qx[p] * k_phase_c[code & 15] +
                                    qy[p] * k_phase_s[code & 15]);
                }
            } else {
                const float *row = page->hot_keys + t * d;
                for (p = 0; p < pairs; p++)
                    logit += qx[p] * row[2 * p] + qy[p] * row[2 * p + 1];
            }
            out[written++] = logit;
        }
    }
    free(freq); free(step_c); free(step_s); free(qx); free(qy);
    *n_out = written;
    return OC_OK;
}

OcError oc_helix_cache_attention(const OcHelixCache *cache,
                                 size_t layer, size_t kv_head,
                                 const float *query_pre_rope, size_t query_n,
                                 size_t query_position, float rope_theta,
                                 float *out)
{
    size_t n, i, pi, t, g, score_index = 0;
    float *scores, *acc;
    size_t d, groups;
    OcError e;
    if (!cache || !query_pre_rope || !out) return OC_ERR_INVALID_ARG;
    if (query_n != cache->config.head_dim) return OC_ERR_INVALID_ARG;
    if (rope_theta <= 0.0f) return OC_ERR_INVALID_ARG;
    d = cache->config.head_dim;
    groups = d / 8;
    for (i = 0; i < d; i++) out[i] = 0.0f;
    n = oc_helix_cache_n_logits(cache, layer, kv_head);
    if (n == 0) return OC_OK;
    scores = (float *)malloc(n * sizeof(float));
    if (!scores) return OC_ERR_OOM;
    e = oc_helix_cache_logits(cache, layer, kv_head, query_pre_rope, query_n,
                              query_position, rope_theta, scores, n, &n);
    if (e != OC_OK) {
        free(scores);
        return e;
    }
    if (n == 0) {
        free(scores);
        return OC_OK;
    }
    softmax_inplace(scores, n, 1.0f / sqrtf((float)d));
    acc = (float *)malloc(d * sizeof(float));
    if (!acc) {
        free(scores);
        return OC_ERR_OOM;
    }
    for (pi = 0; pi < cache->n_pages; pi++) {
        const OcHelixPage *page = &cache->pages[pi];
        if (page->key.layer != layer || page->key.head != kv_head) continue;
        if (page->tier == OC_HELIX_PAGE_HOT) {
            for (t = 0; t < page->n_tokens; t++) {
                float weight;
                if (page->positions[t] > query_position) continue;
                weight = scores[score_index++];
                for (i = 0; i < d; i++)
                    out[i] += weight * page->hot_values[t * d + i];
            }
            continue;
        }
        memset(acc, 0, d * sizeof(float));
        {
            float sum_w = 0.0f;
            const uint8_t *codes = page->cold_value.codes;
            for (t = 0; t < page->n_tokens; t++) {
                float weight;
                float w8;
                size_t row_byte;
                if (page->positions[t] > query_position) continue;
                weight = scores[score_index++];
                if (weight < 1.0e-12f) continue;
                w8 = weight * 0.125f;
                sum_w += w8;
                row_byte = t * d * 3 / 8;
                for (g = 0; g < groups; g++) {
                    uint32_t word = 0;
                    memcpy(&word, codes + row_byte + g * 3, 3);
                    for (i = 0; i < 8; i++)
                        acc[g * 8 + i] += w8 * (float)((word >> (3 * i)) & 7u);
                }
            }
            for (g = 0; g < groups; g++) {
                float centered[8];
                float inverse[8];
                const float s = page->cold_value.scales[g];
                for (i = 0; i < 8; i++)
                    centered[i] = acc[g * 8 + i] - 3.0f * sum_w;
                hadamard8(centered, inverse);
                for (i = 0; i < 8; i++)
                    out[g * 8 + i] += s * inverse[i];
            }
        }
    }
    free(acc);
    free(scores);
    return OC_OK;
}

OcError oc_helix_cache_bump_uncertainty(OcHelixCache *cache,
                                        size_t layer, size_t kv_head,
                                        size_t page_id,
                                        float interval_overlap)
{
    OcHelixPage *page;
    if (!cache) return OC_ERR_INVALID_ARG;
    page = find_page(cache, layer, kv_head, page_id);
    if (!page) return OC_ERR_INVALID_ARG;
    if (interval_overlap > page->promotion.recent_max_overlap)
        page->promotion.recent_max_overlap = interval_overlap;
    if (interval_overlap >= cache->config.promotion_epsilon)
        page->promotion.uncertainty_counter += 1;
    return OC_OK;
}

OcError oc_helix_cache_should_promote(const OcHelixCache *cache,
                                      size_t layer, size_t kv_head,
                                      size_t page_id, bool *out)
{
    const OcHelixPage *page;
    if (!cache || !out) return OC_ERR_INVALID_ARG;
    page = find_page_c(cache, layer, kv_head, page_id);
    if (!page) return OC_ERR_INVALID_ARG;
    *out = page->tier == OC_HELIX_PAGE_COLD &&
           page->promotion.uncertainty_counter >= cache->config.promotion_budget;
    return OC_OK;
}

OcError oc_helix_cache_stats(const OcHelixCache *cache, OcHelixCacheStats *out)
{
    size_t i;
    float key_coords;
    if (!cache || !out) return OC_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));
    for (i = 0; i < cache->n_pages; i++) {
        const OcHelixPage *page = &cache->pages[i];
        out->token_count += page->n_tokens;
        out->page_metadata_bytes += page->n_tokens * sizeof(size_t);
        if (page->tier == OC_HELIX_PAGE_HOT) {
            out->hot_pages += 1;
            out->hot_bytes += page->n_tokens * cache->config.head_dim *
                              sizeof(float) * 2;
            continue;
        }
        out->cold_pages += 1;
        out->key_metadata_bytes += page->cold_key.n_pairs * sizeof(float) * 3;
        out->key_metadata_bytes += page->cold_key.n_pairs * 16u * sizeof(float);
        out->value_metadata_bytes +=
            (cache->config.head_dim / 8) * sizeof(float);
        out->key_bytes += packed_bits_bytes(page->n_tokens * page->cold_key.n_pairs);
        out->key_bytes += page->n_tokens * page->cold_key.n_pairs;
        out->value_bytes += page->cold_value.code_bytes;
    }
    out->metadata_bytes = out->key_metadata_bytes + out->value_metadata_bytes +
                          out->page_metadata_bytes;
    key_coords = (float)(out->token_count * cache->config.head_dim);
    out->f32_baseline_bytes =
        out->token_count * cache->config.head_dim * 2 * sizeof(float);
    if (key_coords > 0.0f) {
        out->key_bits_per_coord =
            (float)((out->key_bytes + out->key_metadata_bytes) * 8) / key_coords;
        out->value_bits_per_coord =
            (float)((out->value_bytes + out->value_metadata_bytes) * 8) /
            key_coords;
        out->total_bits_per_coord =
            (float)((out->key_bytes + out->value_bytes + out->hot_bytes +
                     out->metadata_bytes) *
                    8) /
            (2.0f * key_coords);
    }
    return OC_OK;
}

float oc_helix_cache_compression_ratio(const OcHelixCacheStats *st)
{
    size_t bytes;
    if (!st) return 1.0f;
    bytes = st->key_bytes + st->value_bytes + st->hot_bytes + st->metadata_bytes;
    return bytes == 0 ? 1.0f
                      : (float)st->f32_baseline_bytes / (float)bytes;
}

OcError oc_helix_cache_rewind(OcHelixCache *cache, size_t n_keep)
{
    size_t i, w;
    if (!cache) return OC_ERR_INVALID_ARG;
    w = 0;
    for (i = 0; i < cache->n_pages; i++) {
        OcHelixPage *p = &cache->pages[i];
        size_t keep = 0, t;
        for (t = 0; t < p->n_tokens; t++) {
            if (p->positions[t] < n_keep) keep = t + 1;
        }
        if (keep == 0) {
            page_reset(p);
            continue;
        }
        p->n_tokens = keep;
        if (w != i) {
            cache->pages[w] = cache->pages[i];
            memset(&cache->pages[i], 0, sizeof(cache->pages[i]));
        }
        w += 1;
    }
    cache->n_pages = w;
    return OC_OK;
}
