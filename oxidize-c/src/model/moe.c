/* moe.c — Mixture-of-Experts routing and expert forward pass implementation. */
#include "oxidize/moe.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static bool config_valid(const OcMoeConfig *c)
{
    if (!c) return false;
    if (c->n_experts == 0) return false;
    if (c->hidden_dim == 0) return false;
    if (c->n_active_experts > c->n_experts) return false;
    if (c->n_active_experts > OC_MOE_MAX_EXPERTS_PER_TOKEN) return false;
    /* TOP_P/SOFTMAX may select every expert, but OcMoeRouteResult only
     * holds OC_MOE_MAX_EXPERTS_PER_TOKEN entries; reject configs that
     * would silently drop experts. */
    if (c->routing_method != OC_MOE_ROUTE_TOP_K &&
        c->n_experts > OC_MOE_MAX_EXPERTS_PER_TOKEN) return false;
    if (c->expert_size == 0) return false;
    if (c->routing_method >= OC_MOE_ROUTE__COUNT) return false;
    if (c->top_p < 0.0f || c->top_p > 1.0f) return false;
    return true;
}

static float silu_f(float x)
{
    return x / (1.0f + expf(-x));
}

/* In-place softmax over `n` values. Numerically stable (subtract max). */
static void softmax_inplace(float *v, size_t n)
{
    if (n == 0) return;
    float max = v[0];
    for (size_t i = 1; i < n; i++) {
        if (v[i] > max) max = v[i];
    }
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        v[i] = expf(v[i] - max);
        sum += v[i];
    }
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (size_t i = 0; i < n; i++) v[i] *= inv;
    }
}

/* Shannon entropy (in nats) of a probability distribution. */
static float shannon_entropy(const float *p, size_t n)
{
    float h = 0.0f;
    for (size_t i = 0; i < n; i++) {
        if (p[i] > 1e-12f) {
            h -= p[i] * logf(p[i]);
        }
    }
    return h;
}

/* Simple insertion sort of (index, value) pairs descending by value.
 * Used for top-k / top-p selection. n is small (typically <= 64). */
typedef struct { uint32_t idx; float val; } IdxVal;

static void sort_descending(IdxVal *arr, size_t n)
{
    for (size_t i = 1; i < n; i++) {
        IdxVal cur = arr[i];
        size_t j = i;
        while (j > 0 && arr[j - 1].val < cur.val) {
            arr[j] = arr[j - 1];
            j--;
        }
        arr[j] = cur;
    }
}


const char *oc_moe_routing_method_name(OcMoeRoutingMethod method)
{
    switch (method) {
        case OC_MOE_ROUTE_TOP_K:   return "top_k";
        case OC_MOE_ROUTE_TOP_P:   return "top_p";
        case OC_MOE_ROUTE_SOFTMAX: return "softmax";
        default:                   return "unknown";
    }
}


OcError oc_moe_router_init(OcMoeRouter *r, const OcMoeConfig *config)
{
    if (!r) return OC_ERR_INVALID_ARG;
    memset(r, 0, sizeof(*r));
    if (!config_valid(config)) return OC_ERR_INVALID_ARG;

    r->config = *config;
    /* Documented default: n_active_experts == 0 means 1. */
    if (r->config.n_active_experts == 0) r->config.n_active_experts = 1;
    /* Sensible defaults for top_p when using TOP_P without explicit setting. */
    if (r->config.routing_method == OC_MOE_ROUTE_TOP_P && r->config.top_p == 0.0f) {
        r->config.top_p = 0.9f;
    }

    size_t gate_elems = (size_t)config->n_experts * config->hidden_dim;
    r->gate_weights = calloc(gate_elems, sizeof(float));
    if (!r->gate_weights) {
        oc_moe_router_free(r);
        return OC_ERR_OOM;
    }

    size_t exp_elems = (size_t)config->n_experts * config->expert_size * config->hidden_dim;
    r->expert_weights = calloc(exp_elems, sizeof(float));
    if (!r->expert_weights) {
        oc_moe_router_free(r);
        return OC_ERR_OOM;
    }

    r->expert_up   = calloc(exp_elems, sizeof(float));
    r->expert_down = calloc(exp_elems, sizeof(float));
    if (!r->expert_up || !r->expert_down) {
        oc_moe_router_free(r);
        return OC_ERR_OOM;
    }

    r->expert_usage_counts = calloc(config->n_experts, sizeof(uint64_t));
    if (!r->expert_usage_counts) {
        oc_moe_router_free(r);
        return OC_ERR_OOM;
    }

    r->total_tokens       = 0;
    r->routing_entropy_sum = 0.0;
    return OC_OK;
}

OcError oc_moe_router_set_gate(OcMoeRouter *r, const float *weights)
{
    if (!r || !r->gate_weights) return OC_ERR_INVALID_ARG;
    size_t n = (size_t)r->config.n_experts * r->config.hidden_dim;
    if (weights) {
        memcpy(r->gate_weights, weights, n * sizeof(float));
    } else {
        memset(r->gate_weights, 0, n * sizeof(float));
    }
    return OC_OK;
}

OcError oc_moe_router_set_experts(OcMoeRouter *r,
                                  const float *gate_proj,
                                  const float *up_proj,
                                  const float *down_proj)
{
    if (!r || !r->expert_weights) return OC_ERR_INVALID_ARG;
    size_t n = (size_t)r->config.n_experts * r->config.expert_size * r->config.hidden_dim;

    if (gate_proj) memcpy(r->expert_weights, gate_proj, n * sizeof(float));

    /* Optional projections: an omitted projection drops its buffer so
     * expert_forward selects the single-projection path instead of a
     * zero-weight SwiGLU. */
    if (up_proj) {
        if (!r->expert_up) {
            r->expert_up = malloc(n * sizeof(float));
            if (!r->expert_up) return OC_ERR_OOM;
        }
        memcpy(r->expert_up, up_proj, n * sizeof(float));
    } else {
        free(r->expert_up);
        r->expert_up = NULL;
    }
    if (down_proj) {
        if (!r->expert_down) {
            r->expert_down = malloc(n * sizeof(float));
            if (!r->expert_down) return OC_ERR_OOM;
        }
        memcpy(r->expert_down, down_proj, n * sizeof(float));
    } else {
        free(r->expert_down);
        r->expert_down = NULL;
    }
    return OC_OK;
}

void oc_moe_router_free(OcMoeRouter *r)
{
    if (!r) return;
    free(r->gate_weights);
    free(r->expert_weights);
    free(r->expert_up);
    free(r->expert_down);
    free(r->expert_usage_counts);
    memset(r, 0, sizeof(*r));
}


/* Compute gate logits = gate_weights @ hidden. logits[e] = dot(row_e, hidden). */
static void compute_gate_logits(const OcMoeRouter *r,
                                const float *hidden,
                                float *logits)
{
    size_t hd = r->config.hidden_dim;
    for (uint32_t e = 0; e < r->config.n_experts; e++) {
        const float *row = r->gate_weights + (size_t)e * hd;
        float dot = 0.0f;
        for (size_t i = 0; i < hd; i++) {
            dot += row[i] * hidden[i];
        }
        logits[e] = dot;
    }
}

OcError oc_moe_route(OcMoeRouter *r,
                     const float *hidden,
                     OcMoeRouteResult *out,
                     float *temp_logits)
{
    if (!r || !hidden || !out) return OC_ERR_INVALID_ARG;
    if (!r->gate_weights || !r->expert_usage_counts) return OC_ERR_INVALID_ARG;

    memset(out, 0, sizeof(*out));

    float *logits = temp_logits;
    bool allocated = false;
    if (!logits) {
        logits = malloc((size_t)r->config.n_experts * sizeof(float));
        if (!logits) return OC_ERR_OOM;
        allocated = true;
    }

    compute_gate_logits(r, hidden, logits);

    /* Build probability distribution via softmax. */
    float *probs = malloc((size_t)r->config.n_experts * sizeof(float));
    if (!probs) {
        if (allocated) free(logits);
        return OC_ERR_OOM;
    }
    memcpy(probs, logits, (size_t)r->config.n_experts * sizeof(float));
    softmax_inplace(probs, r->config.n_experts);

    out->entropy = shannon_entropy(probs, r->config.n_experts);

    /* Select experts based on routing method. */
    IdxVal *pairs = malloc((size_t)r->config.n_experts * sizeof(IdxVal));
    if (!pairs) {
        free(probs);
        if (allocated) free(logits);
        return OC_ERR_OOM;
    }
    for (uint32_t i = 0; i < r->config.n_experts; i++) {
        pairs[i].idx = i;
        pairs[i].val = probs[i];
    }
    sort_descending(pairs, r->config.n_experts);

    uint32_t k;
    switch (r->config.routing_method) {
        case OC_MOE_ROUTE_TOP_K:
            k = r->config.n_active_experts;
            break;
        case OC_MOE_ROUTE_SOFTMAX:
            k = r->config.n_experts;
            break;
        case OC_MOE_ROUTE_TOP_P: {
            /* Nucleus: keep minimal set whose cumulative prob >= top_p. */
            float cum = 0.0f;
            uint32_t min_k = r->config.n_active_experts;
            k = 0;
            for (uint32_t i = 0; i < r->config.n_experts; i++) {
                cum += pairs[i].val;
                k++;
                if (cum >= r->config.top_p && k >= min_k) break;
            }
            if (k == 0) k = 1;
            break;
        }
        default:
            k = r->config.n_active_experts;
            break;
    }
    if (k > r->config.n_experts) k = r->config.n_experts;
    if (k > OC_MOE_MAX_EXPERTS_PER_TOKEN) k = OC_MOE_MAX_EXPERTS_PER_TOKEN;

    out->n_selected = k;
    float weight_sum = 0.0f;
    for (uint32_t i = 0; i < k; i++) {
        out->expert_indices[i]  = pairs[i].idx;
        out->expert_weights[i]  = pairs[i].val;
        out->gate_logits[i]     = logits[pairs[i].idx];
        weight_sum += pairs[i].val;
    }

    /* Renormalize weights if configured. */
    if (r->config.normalize_weights && weight_sum > 0.0f) {
        float inv = 1.0f / weight_sum;
        for (uint32_t i = 0; i < k; i++) {
            out->expert_weights[i] *= inv;
        }
    }

    /* Update stats: per-expert usage counts + entropy accumulation. */
    for (uint32_t i = 0; i < k; i++) {
        r->expert_usage_counts[out->expert_indices[i]]++;
    }
    r->total_tokens++;
    r->routing_entropy_sum += (double)out->entropy;

    free(pairs);
    free(probs);
    if (allocated) free(logits);
    return OC_OK;
}


OcError oc_moe_expert_forward(const OcMoeRouter *r,
                              uint32_t expert_idx,
                              const float *x,
                              float *out, size_t *out_len,
                              float *temp)
{
    if (!r || !x || !out || !out_len) return OC_ERR_INVALID_ARG;
    if (!r->expert_weights) return OC_ERR_INVALID_ARG;
    if (expert_idx >= r->config.n_experts) return OC_ERR_MODEL;

    size_t hd = r->config.hidden_dim;
    size_t es = r->config.expert_size;
    size_t expert_off = (size_t)expert_idx * es * hd;

    const float *gate_w = r->expert_weights + expert_off;

    /* SwiGLU path: requires up + down weights + temp scratch. */
    if (r->expert_up && r->expert_down && temp) {
        const float *up_w   = r->expert_up   + expert_off;
        const float *down_w = r->expert_down + (size_t)expert_idx * hd * es;

        /* Single-projection path: out = expert_weights @ x [length es]. */
        /* Single-projection path: out = expert_weights @ x [length es]. */
        /* Single-projection path: out = expert_weights @ x [length es]. */
        for (size_t j = 0; j < es; j++) {
            const float *gw_row = gate_w + j * hd;
            const float *uw_row = up_w   + j * hd;
            float g = 0.0f, u = 0.0f;
            for (size_t i = 0; i < hd; i++) {
                g += gw_row[i] * x[i];
                u += uw_row[i] * x[i];
            }
            temp[j] = silu_f(g) * u;
        }

        for (size_t j = 0; j < hd; j++) {
            const float *dw_row = down_w + j * es;
            float acc = 0.0f;
            for (size_t i = 0; i < es; i++) {
                acc += dw_row[i] * temp[i];
            }
            out[j] = acc;
        }
        *out_len = hd;
        return OC_OK;
    }

    /* Single-projection path: out = expert_weights @ x  [length es]. */
    for (size_t j = 0; j < es; j++) {
        const float *row = gate_w + j * hd;
        float acc = 0.0f;
        for (size_t i = 0; i < hd; i++) {
            acc += row[i] * x[i];
        }
        out[j] = acc;
    }
    *out_len = es;
    return OC_OK;
}


OcError oc_moe_combine(const OcMoeRouteResult *result,
                       const float *const *expert_outs,
                       size_t n_selected,
                       size_t out_len,
                       float *combined)
{
    if (!result || !expert_outs || !combined) return OC_ERR_INVALID_ARG;
    if (n_selected == 0) return OC_ERR_INVALID_ARG;
    if (n_selected != result->n_selected) return OC_ERR_INVALID_ARG;

    memset(combined, 0, out_len * sizeof(float));
    for (size_t i = 0; i < n_selected; i++) {
        if (!expert_outs[i]) return OC_ERR_INVALID_ARG;
        float w = result->expert_weights[i];
        const float *eo = expert_outs[i];
        for (size_t j = 0; j < out_len; j++) {
            combined[j] += w * eo[j];
        }
    }
    return OC_OK;
}


OcError oc_moe_get_stats(const OcMoeRouter *r, OcMoeStats *stats)
{
    if (!r || !stats) return OC_ERR_INVALID_ARG;
    memset(stats, 0, sizeof(*stats));
    stats->total_tokens      = r->total_tokens;
    stats->n_experts         = r->config.n_experts;
    stats->n_active_experts  = r->config.n_active_experts;
    stats->expert_usage_counts = malloc((size_t)r->config.n_experts * sizeof(uint64_t));
    if (!stats->expert_usage_counts) return OC_ERR_OOM;
    memcpy(stats->expert_usage_counts, r->expert_usage_counts,
           (size_t)r->config.n_experts * sizeof(uint64_t));
    stats->routing_entropy = (r->total_tokens > 0)
        ? (r->routing_entropy_sum / (double)r->total_tokens)
        : 0.0;
    return OC_OK;
}

void oc_moe_stats_free(OcMoeStats *stats)
{
    if (!stats) return;
    free(stats->expert_usage_counts);
    memset(stats, 0, sizeof(*stats));
}


/* Write a uint64 array as JSON into buf. Returns chars written. */
static size_t format_u64_array(char *buf, size_t cap, size_t *off,
                               const uint64_t *arr, size_t n)
{
    size_t o = *off;
    if (o + 1 >= cap) { *off = o; return o; }
    buf[o++] = '[';
    for (size_t i = 0; i < n; i++) {
        if (i > 0) {
            if (o + 2 >= cap) { *off = o; return o; }
            buf[o++] = ',';
            buf[o++] = ' ';
        }
        int written = snprintf(buf + o, cap - o, "%llu",
                               (unsigned long long)arr[i]);
        if (written < 0 || (size_t)written >= cap - o) {
            *off = o;
            return o;
        }
        o += (size_t)written;
    }
    if (o + 1 >= cap) { *off = o; return o; }
    buf[o++] = ']';
    *off = o;
    return o;
}

size_t oc_moe_stats_format(const OcMoeStats *stats, char *buf, size_t cap)
{
    if (!stats) return 0;

    /* Compute the length we would write by formatting into a temp buffer
     * or directly into buf. We use a two-pass approach: first compute the
     * size needed, then write. */
    size_t off = 0;

    /* Helper to append a string. */
    #define APPEND_STR(s) do { \
        size_t len = strlen(s); \
        if (buf) { \
            size_t copy = (off + len + 1 <= cap) ? len : (cap > off ? cap - off - 1 : 0); \
            memcpy(buf + off, s, copy); \
        } \
        off += len; \
    } while (0)

    #define APPEND_FMT(...) do { \
        if (buf && off + 1 < cap) { \
            int w = snprintf(buf + off, cap - off, __VA_ARGS__); \
            if (w > 0) off += (size_t)w; \
        } else { \
            char tmp[64]; \
            int w = snprintf(tmp, sizeof(tmp), __VA_ARGS__); \
            if (w > 0) off += (size_t)w; \
        } \
    } while (0)

    APPEND_STR("{\n");
    APPEND_FMT("  \"total_tokens\": %llu,\n", (unsigned long long)stats->total_tokens);
    APPEND_FMT("  \"n_experts\": %u,\n", stats->n_experts);
    APPEND_FMT("  \"n_active_experts\": %u,\n", stats->n_active_experts);

    APPEND_STR("  \"expert_usage_counts\": ");
    if (stats->expert_usage_counts && stats->n_experts > 0) {
        if (buf) {
            format_u64_array(buf, cap, &off,
                            stats->expert_usage_counts, stats->n_experts);
        } else {
            /* Compute length without buf. */
            size_t needed = 2; /* [] */
            for (size_t i = 0; i < stats->n_experts; i++) {
                if (i > 0) needed += 2;
                char tmp[32];
                int w = snprintf(tmp, sizeof(tmp), "%llu",
                                 (unsigned long long)stats->expert_usage_counts[i]);
                if (w > 0) needed += (size_t)w;
            }
            off += needed;
        }
    } else {
        if (buf && off + 2 < cap) { buf[off++] = '['; buf[off++] = ']'; }
        else off += 2;
    }
    if (buf && off + 2 < cap) { buf[off++] = ','; buf[off++] = '\n'; }
    else off += 2;

    APPEND_FMT("  \"routing_entropy\": %.6f\n", stats->routing_entropy);
    APPEND_STR("}");

    if (buf && off < cap) buf[off] = '\0';
    return off;

    #undef APPEND_STR
    #undef APPEND_FMT
}
