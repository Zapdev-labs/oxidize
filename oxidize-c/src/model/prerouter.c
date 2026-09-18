#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include "oxidize/prerouter.h"

#include "oxidize/activation.h"
#include "oxidize/expert_stream.h"
#include "oxidize/flash_attention.h"
#include "oxidize/log.h"
#include "oxidize/safetensors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t owner;
    uint32_t hidden_width;
    uint32_t input_dim;
    uint32_t n_experts;
    float *fc1;
    float *fc2;
    float *linear_init;
} OcPrerouterHead;

struct OcPrerouter {
    uint32_t n_layers;
    uint32_t n_experts;
    uint32_t hidden_dim;
    uint32_t top_k;
    uint32_t input_dim;
    bool replace_routing;
    OcPrerouterHead *heads;
    uint32_t n_heads;
    OcExpertStreamPool *stream;
    float *concat;
    float *mlp_h;
    float *logits;
    float *resid;
    float *scratch_w;
    uint32_t *order;
    uint32_t *prev_sel;
    uint32_t *curr_sel;
    uint32_t *pred_sel;
    float *pred_w;
    uint32_t *staged_sel;
    float *staged_w;
    uint8_t *has_pred;
    uint8_t *has_staged;
    uint8_t *has_curr;
    uint8_t *has_prev;
};

static void softmax_topk(const float *logits, uint32_t n, uint32_t k,
                         uint32_t *sel, float *weights)
{
    if (!logits || !sel || n == 0 || k == 0) return;
    if (k > n) k = n;
    float mx = logits[0];
    for (uint32_t i = 1; i < n; i++)
        if (logits[i] > mx) mx = logits[i];
    double sum = 0.0;
    /* weights used as scratch for exp */
    for (uint32_t i = 0; i < n; i++) {
        weights[i] = expf(logits[i] - mx);
        sum += (double)weights[i];
    }
    if (sum > 0.0) {
        float inv = (float)(1.0 / sum);
        for (uint32_t i = 0; i < n; i++) weights[i] *= inv;
    }
    for (uint32_t i = 0; i < n; i++) sel[i] = i;
    for (uint32_t i = 0; i < k; i++) {
        uint32_t best = i;
        for (uint32_t j = i + 1; j < n; j++) {
            if (weights[sel[j]] > weights[sel[best]]) best = j;
        }
        uint32_t tmp = sel[i];
        sel[i] = sel[best];
        sel[best] = tmp;
    }
    double wsum = 0.0;
    for (uint32_t i = 0; i < k; i++) wsum += (double)weights[sel[i]];
    if (wsum <= 0.0) wsum = 1.0;
    for (uint32_t i = 0; i < k; i++)
        weights[sel[i]] = (float)(weights[sel[i]] / wsum);
}

static void matvec_f32(const float *w, const float *x, float *out,
                       uint32_t rows, uint32_t cols)
{
    for (uint32_t r = 0; r < rows; r++) {
        const float *row = w + (size_t)r * cols;
        float acc = 0.0f;
        for (uint32_t c = 0; c < cols; c++) acc += row[c] * x[c];
        out[r] = acc;
    }
}

static OcError copy_weight(const OcSafetensorsFile *st,
                           const OcSafetensorsTensor *t, float **out,
                           uint32_t expect_rows, uint32_t expect_cols)
{
    if (!t || t->n_dims != 2) return OC_ERR_FORMAT;
    if (t->shape[0] != expect_rows || t->shape[1] != expect_cols)
        return OC_ERR_TENSOR;
    const void *raw = NULL;
    OcError e = oc_safetensors_get_tensor_data(st, t, &raw);
    if (e != OC_OK) return e;
    size_t n = (size_t)expect_rows * expect_cols;
    float *buf = malloc(n * sizeof(float));
    if (!buf) return OC_ERR_OOM;
    if (strcmp(t->dtype, "F32") == 0) {
        memcpy(buf, raw, n * sizeof(float));
    } else if (strcmp(t->dtype, "F16") == 0) {
        const uint16_t *h = (const uint16_t *)raw;
        for (size_t i = 0; i < n; i++) buf[i] = oc_f16_to_f32_bits(h[i]);
    } else {
        free(buf);
        return OC_ERR_FORMAT;
    }
    *out = buf;
    return OC_OK;
}

OcError oc_prerouter_new(uint32_t n_layers, uint32_t n_experts,
                         uint32_t hidden_dim, uint32_t top_k,
                         OcPrerouter **out)
{
    if (!out || n_layers == 0 || n_experts == 0 || hidden_dim == 0)
        return OC_ERR_INVALID_ARG;
    *out = NULL;
    OcPrerouter *p = calloc(1, sizeof(*p));
    if (!p) return OC_ERR_OOM;
    p->n_layers = n_layers;
    p->n_experts = n_experts;
    p->hidden_dim = hidden_dim;
    p->top_k = top_k > 0 ? top_k : 1;
    p->input_dim = hidden_dim + 2u * n_experts;
    p->replace_routing = true;
    p->concat = calloc(p->input_dim, sizeof(float));
    p->logits = calloc(n_experts, sizeof(float));
    p->resid = calloc(n_experts, sizeof(float));
    p->scratch_w = calloc(n_experts, sizeof(float));
    p->order = calloc(n_experts, sizeof(uint32_t));
    p->prev_sel = calloc((size_t)n_layers * p->top_k, sizeof(uint32_t));
    p->curr_sel = calloc((size_t)n_layers * p->top_k, sizeof(uint32_t));
    p->pred_sel = calloc((size_t)n_layers * p->top_k, sizeof(uint32_t));
    p->pred_w = calloc((size_t)n_layers * p->top_k, sizeof(float));
    p->staged_sel = calloc((size_t)n_layers * p->top_k, sizeof(uint32_t));
    p->staged_w = calloc((size_t)n_layers * p->top_k, sizeof(float));
    p->has_pred = calloc(n_layers, 1);
    p->has_staged = calloc(n_layers, 1);
    p->has_curr = calloc(n_layers, 1);
    p->has_prev = calloc(n_layers, 1);
    if (!p->concat || !p->logits || !p->resid || !p->scratch_w || !p->order ||
        !p->prev_sel || !p->curr_sel || !p->pred_sel || !p->pred_w ||
        !p->staged_sel || !p->staged_w ||
        !p->has_pred || !p->has_staged || !p->has_curr || !p->has_prev) {
        oc_prerouter_free(p);
        return OC_ERR_OOM;
    }
    *out = p;
    return OC_OK;
}

static int parse_layer_key(const char *name, uint32_t *layer, const char **kind)
{
    unsigned l = 0;
    char kindbuf[32];
    if (sscanf(name, "layers.%u.%31[^.].weight", &l, kindbuf) != 2)
        return -1;
    *layer = l;
    if (strcmp(kindbuf, "fc1") == 0) *kind = "fc1";
    else if (strcmp(kindbuf, "fc2") == 0) *kind = "fc2";
    else if (strcmp(kindbuf, "linear_init") == 0) *kind = "linear_init";
    else return -1;
    return 0;
}

static OcPrerouterHead *ensure_head(OcPrerouter *p, uint32_t owner)
{
    for (uint32_t i = 0; i < p->n_heads; i++) {
        if (p->heads[i].owner == owner) return &p->heads[i];
    }
    OcPrerouterHead *n = realloc(p->heads, (p->n_heads + 1) * sizeof(*n));
    if (!n) return NULL;
    p->heads = n;
    OcPrerouterHead *h = &p->heads[p->n_heads++];
    memset(h, 0, sizeof(*h));
    h->owner = owner;
    h->n_experts = p->n_experts;
    h->input_dim = p->input_dim;
    return h;
}

OcError oc_prerouter_load_safetensors(OcPrerouter *p, const char *path)
{
    if (!p || !path) return OC_ERR_INVALID_ARG;
    OcSafetensorsFile st;
    OcError e = oc_safetensors_open(path, &st);
    if (e != OC_OK) return e;

    for (size_t i = 0; i < st.n_tensors; i++) {
        const OcSafetensorsTensor *t = &st.tensors[i];
        uint32_t layer = 0;
        const char *kind = NULL;
        if (parse_layer_key(t->name, &layer, &kind) != 0) continue;
        if (layer >= p->n_layers) continue;
        OcPrerouterHead *h = ensure_head(p, layer);
        if (!h) { oc_safetensors_close(&st); return OC_ERR_OOM; }
        if (strcmp(kind, "fc1") == 0) {
            if (t->n_dims != 2) continue;
            h->hidden_width = (uint32_t)t->shape[0];
            if (h->hidden_width == 0) continue;
            if (!p->mlp_h) {
                p->mlp_h = calloc(h->hidden_width, sizeof(float));
                if (!p->mlp_h) { oc_safetensors_close(&st); return OC_ERR_OOM; }
            }
            free(h->fc1);
            h->fc1 = NULL;
            e = copy_weight(&st, t, &h->fc1, h->hidden_width, p->input_dim);
        } else if (strcmp(kind, "fc2") == 0) {
            free(h->fc2);
            h->fc2 = NULL;
            e = copy_weight(&st, t, &h->fc2, p->n_experts,
                            h->hidden_width ? h->hidden_width : (uint32_t)t->shape[1]);
            if (h->hidden_width == 0 && t->n_dims == 2)
                h->hidden_width = (uint32_t)t->shape[1];
        } else {
            free(h->linear_init);
            h->linear_init = NULL;
            e = copy_weight(&st, t, &h->linear_init, p->n_experts, p->input_dim);
        }
        if (e != OC_OK) {
            oc_safetensors_close(&st);
            return e;
        }
    }
    oc_safetensors_close(&st);
    if (p->n_heads == 0) return OC_ERR_MODEL;
    oc_log(OC_LOG_INFO, "prerouter: loaded %u heads from %s (K=%u, replace=%d)",
           p->n_heads, path, p->top_k, (int)p->replace_routing);
    return OC_OK;
}

void oc_prerouter_set_replace_routing(OcPrerouter *p, bool replace)
{
    if (p) p->replace_routing = replace;
}

void oc_prerouter_set_stream(OcPrerouter *p, OcExpertStreamPool *stream)
{
    if (p) p->stream = stream;
}

bool oc_prerouter_replace_routing(const OcPrerouter *p)
{
    return p ? p->replace_routing : false;
}

uint32_t oc_prerouter_n_heads(const OcPrerouter *p)
{
    return p ? p->n_heads : 0;
}

uint32_t oc_prerouter_top_k(const OcPrerouter *p)
{
    return p ? p->top_k : 0;
}

bool oc_prerouter_has_prediction(const OcPrerouter *p, uint32_t layer)
{
    if (!p || layer >= p->n_layers) return false;
    return p->replace_routing && p->n_heads > 0 && p->has_pred[layer];
}

OcError oc_prerouter_consume(OcPrerouter *p, uint32_t layer,
                             uint32_t *sel, uint32_t k, float *weights_out)
{
    if (!p || !sel || !weights_out) return OC_ERR_INVALID_ARG;
    if (layer >= p->n_layers || !p->has_pred[layer]) return OC_ERR_MODEL;
    if (k > p->top_k) k = p->top_k;
    memset(weights_out, 0, (size_t)p->n_experts * sizeof(float));
    for (uint32_t i = 0; i < k; i++) {
        sel[i] = p->pred_sel[(size_t)layer * p->top_k + i];
        weights_out[sel[i]] = p->pred_w[(size_t)layer * p->top_k + i];
    }
    p->has_pred[layer] = 0;
    return OC_OK;
}

static const OcPrerouterHead *head_for_owner(const OcPrerouter *p, uint32_t owner)
{
    for (uint32_t i = 0; i < p->n_heads; i++) {
        if (p->heads[i].owner == owner) return &p->heads[i];
    }
    return NULL;
}

static void fill_onehot(float *dst, uint32_t n, const uint32_t *sel, uint32_t k)
{
    memset(dst, 0, (size_t)n * sizeof(float));
    for (uint32_t i = 0; i < k; i++) {
        if (sel[i] < n) dst[sel[i]] = 1.0f;
    }
}

static OcError run_head(OcPrerouter *p, const OcPrerouterHead *h,
                        const float *hidden, const uint32_t *curr,
                        const uint32_t *prev, uint32_t k,
                        uint32_t *out_sel, float *out_w)
{
    memset(p->concat, 0, (size_t)p->input_dim * sizeof(float));
    memcpy(p->concat, hidden, (size_t)p->hidden_dim * sizeof(float));
    fill_onehot(p->concat + p->hidden_dim, p->n_experts, curr, k);
    fill_onehot(p->concat + p->hidden_dim + p->n_experts, p->n_experts, prev, k);

    uint32_t width = h->hidden_width;
    if (!h->fc1 || !h->fc2 || !h->linear_init || !p->mlp_h || width == 0)
        return OC_ERR_MODEL;
    matvec_f32(h->fc1, p->concat, p->mlp_h, width, p->input_dim);
    for (uint32_t i = 0; i < width; i++)
        p->mlp_h[i] = oc_gelu_exact_f32(p->mlp_h[i]);
    matvec_f32(h->fc2, p->mlp_h, p->logits, p->n_experts, width);
    matvec_f32(h->linear_init, p->concat, p->resid, p->n_experts, p->input_dim);
    for (uint32_t i = 0; i < p->n_experts; i++) p->logits[i] += p->resid[i];

    softmax_topk(p->logits, p->n_experts, p->top_k, p->order, p->scratch_w);
    uint32_t tk = p->top_k;
    for (uint32_t i = 0; i < tk; i++) {
        out_sel[i] = p->order[i];
        out_w[i] = p->scratch_w[p->order[i]];
    }
    return OC_OK;
}

OcError oc_prerouter_commit(OcPrerouter *p, uint32_t layer,
                            const float *hidden,
                            const uint32_t *sel, uint32_t k)
{
    if (!p || !hidden || !sel) return OC_ERR_INVALID_ARG;
    if (layer >= p->n_layers) return OC_OK;
    if (k > p->top_k) k = p->top_k;

    if (p->has_curr[layer]) {
        memcpy(p->prev_sel + (size_t)layer * p->top_k,
               p->curr_sel + (size_t)layer * p->top_k,
               (size_t)p->top_k * sizeof(uint32_t));
        p->has_prev[layer] = 1;
    }
    memcpy(p->curr_sel + (size_t)layer * p->top_k, sel, (size_t)k * sizeof(uint32_t));
    for (uint32_t i = k; i < p->top_k; i++)
        p->curr_sel[(size_t)layer * p->top_k + i] = 0;
    p->has_curr[layer] = 1;

    const OcPrerouterHead *h = head_for_owner(p, layer);
    if (!h) return OC_OK;
    uint32_t consumer = layer + 1;
    if (consumer >= p->n_layers) return OC_OK;

    uint32_t *pred = p->staged_sel + (size_t)consumer * p->top_k;
    float *pw = p->staged_w + (size_t)consumer * p->top_k;
    OcError e = run_head(p, h, hidden,
                         p->curr_sel + (size_t)layer * p->top_k,
                         p->has_prev[layer]
                             ? p->prev_sel + (size_t)layer * p->top_k
                             : p->curr_sel + (size_t)layer * p->top_k,
                         p->top_k, pred, pw);
    if (e != OC_OK) return e;
    p->has_staged[consumer] = 1;
    if (p->stream) {
        (void)oc_expert_stream_touch(p->stream, consumer, pred, p->top_k, true);
    }
    return OC_OK;
}

void oc_prerouter_advance(OcPrerouter *p)
{
    if (!p || !p->has_staged || !p->has_pred) return;
    size_t nsel = (size_t)p->n_layers * p->top_k;
    if (p->pred_sel && p->staged_sel)
        memcpy(p->pred_sel, p->staged_sel, nsel * sizeof(uint32_t));
    if (p->pred_w && p->staged_w)
        memcpy(p->pred_w, p->staged_w, nsel * sizeof(float));
    memcpy(p->has_pred, p->has_staged, p->n_layers);
    memset(p->has_staged, 0, p->n_layers);
}

void oc_prerouter_reset(OcPrerouter *p)
{
    if (!p) return;
    if (p->has_pred) memset(p->has_pred, 0, p->n_layers);
    if (p->has_staged) memset(p->has_staged, 0, p->n_layers);
    if (p->has_curr) memset(p->has_curr, 0, p->n_layers);
    if (p->has_prev) memset(p->has_prev, 0, p->n_layers);
}

void oc_prerouter_free(OcPrerouter *p)
{
    if (!p) return;
    if (p->heads) {
        for (uint32_t i = 0; i < p->n_heads; i++) {
            free(p->heads[i].fc1);
            free(p->heads[i].fc2);
            free(p->heads[i].linear_init);
        }
        free(p->heads);
    }
    free(p->concat);
    free(p->mlp_h);
    free(p->logits);
    free(p->resid);
    free(p->scratch_w);
    free(p->order);
    free(p->prev_sel);
    free(p->curr_sel);
    free(p->pred_sel);
    free(p->pred_w);
    free(p->staged_sel);
    free(p->staged_w);
    free(p->has_pred);
    free(p->has_staged);
    free(p->has_curr);
    free(p->has_prev);
    free(p);
}
