#include "oxidize/eagle4.h"

#include "oxidize/log.h"
#include "oxidize/safetensors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void oc_eagle4_config_init(OcEagle4Config *cfg)
{
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->hidden_size = 256;
    cfg->target_hidden_size = 10240;
    cfg->n_aux = 3;
    cfg->extract_layers[0] = 2;
    cfg->extract_layers[1] = 24;
    cfg->extract_layers[2] = 46;
    cfg->vocab_size = 248320;
    cfg->n_heads = 4;
    cfg->n_kv_heads = 4;
    cfg->head_dim = 64;
    cfg->intermediate_size = 1024;
    cfg->sliding_window = 4096;
    cfg->lm_rank = 64;
    cfg->n_candidates = 128;
    cfg->rms_eps = 1e-6f;
    cfg->rope_theta = 10000.0f;
    cfg->use_rope = 0;
}

static int mul_ok(size_t a, size_t b, size_t *out)
{
    if (a != 0 && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static void weight_free(OcEagle4Weight *w)
{
    if (!w) return;
    free(w->data);
    w->data = NULL;
    w->rows = w->cols = 0;
}

static OcError weight_alloc(OcEagle4Weight *w, size_t rows, size_t cols)
{
    size_t n = 0;
    weight_free(w);
    if (!mul_ok(rows, cols, &n) || n == 0) return OC_ERR_INVALID_ARG;
    w->data = calloc(n, sizeof(float));
    if (!w->data) return OC_ERR_OOM;
    w->rows = rows;
    w->cols = cols;
    return OC_OK;
}

static void gemv(const OcEagle4Weight *w, const float *x, float *y)
{
    for (size_t r = 0; r < w->rows; r++) {
        const float *row = w->data + r * w->cols;
        float dot = 0.0f;
        for (size_t c = 0; c < w->cols; c++) dot += row[c] * x[c];
        y[r] = dot;
    }
}

static float row_dot(const float *row, const float *x, size_t n)
{
    float dot = 0.0f;
    for (size_t i = 0; i < n; i++) dot += row[i] * x[i];
    return dot;
}

static void rms_norm(const float *x, const float *w, float *y, size_t n,
                     float eps)
{
    float ss = 0.0f;
    for (size_t i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    if (w) {
        for (size_t i = 0; i < n; i++) y[i] = x[i] * inv * w[i];
    } else {
        for (size_t i = 0; i < n; i++) y[i] = x[i] * inv;
    }
}

static float silu(float x)
{
    return x / (1.0f + expf(-x));
}

static void apply_rope(float *q, size_t n_heads, size_t head_dim, size_t pos,
                       float theta)
{
    for (size_t h = 0; h < n_heads; h++) {
        float *qh = q + h * head_dim;
        for (size_t d = 0; d + 1 < head_dim; d += 2) {
            float freq = (float)pos / powf(theta, (float)d / (float)head_dim);
            float c = cosf(freq);
            float s = sinf(freq);
            float q0 = qh[d];
            float q1 = qh[d + 1];
            qh[d] = q0 * c - q1 * s;
            qh[d + 1] = q0 * s + q1 * c;
        }
    }
}

static int cfg_ok(const OcEagle4Config *c)
{
    if (!c) return 0;
    if (c->hidden_size == 0 || c->target_hidden_size == 0 || c->n_aux == 0 ||
        c->n_aux > OC_EAGLE4_MAX_AUX || c->vocab_size == 0 ||
        c->n_heads == 0 || c->n_kv_heads == 0 || c->head_dim == 0 ||
        c->intermediate_size == 0 || c->lm_rank == 0 || c->n_candidates == 0)
        return 0;
    if (c->n_heads % c->n_kv_heads != 0) return 0;
    if (c->n_candidates > c->vocab_size) return 0;
    return 1;
}

OcError oc_eagle4_init(OcEagle4Model *m, const OcEagle4Config *cfg)
{
    if (!m || !cfg_ok(cfg)) return OC_ERR_INVALID_ARG;
    memset(m, 0, sizeof(*m));
    m->cfg = *cfg;
    const size_t h = cfg->hidden_size;
    const size_t t = cfg->target_hidden_size;
    const size_t q = cfg->n_heads * cfg->head_dim;
    const size_t kv = cfg->n_kv_heads * cfg->head_dim;
    const size_t inter = cfg->intermediate_size;
    OcError e;
    if ((e = weight_alloc(&m->fc_norm, 1, t)) != OC_OK ||
        (e = weight_alloc(&m->fc, h, cfg->n_aux * t)) != OC_OK ||
        (e = weight_alloc(&m->embed, cfg->vocab_size, h)) != OC_OK ||
        (e = weight_alloc(&m->eh_proj, h, 2 * h)) != OC_OK ||
        (e = weight_alloc(&m->pre_norm, 1, h)) != OC_OK ||
        (e = weight_alloc(&m->q_proj, q, h)) != OC_OK ||
        (e = weight_alloc(&m->k_proj, kv, h)) != OC_OK ||
        (e = weight_alloc(&m->v_proj, kv, h)) != OC_OK ||
        (e = weight_alloc(&m->o_proj, h, q)) != OC_OK ||
        (e = weight_alloc(&m->ffn_norm, 1, h)) != OC_OK ||
        (e = weight_alloc(&m->gate, inter, h)) != OC_OK ||
        (e = weight_alloc(&m->up, inter, h)) != OC_OK ||
        (e = weight_alloc(&m->down, h, inter)) != OC_OK ||
        (e = weight_alloc(&m->post_norm, 1, h)) != OC_OK ||
        (e = weight_alloc(&m->lm_head, cfg->vocab_size, h)) != OC_OK ||
        (e = weight_alloc(&m->lr_down, cfg->lm_rank, h)) != OC_OK ||
        (e = weight_alloc(&m->lr_up, cfg->vocab_size, cfg->lm_rank)) != OC_OK) {
        oc_eagle4_free(m);
        return e;
    }
    m->feedback = calloc(h, sizeof(float));
    if (!m->feedback) {
        oc_eagle4_free(m);
        return OC_ERR_OOM;
    }
    for (size_t i = 0; i < t; i++) m->fc_norm.data[i] = 1.0f;
    for (size_t i = 0; i < h; i++) {
        m->pre_norm.data[i] = 1.0f;
        m->ffn_norm.data[i] = 1.0f;
        m->post_norm.data[i] = 1.0f;
    }
    m->loaded = 1;
    return OC_OK;
}

static char *read_file(const char *path, size_t *n_out)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t n = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[n] = '\0';
    if (n_out) *n_out = n;
    return buf;
}

static const char *find_key(const char *json, const char *key)
{
    char pat[128];
    int n = snprintf(pat, sizeof pat, "\"%s\"", key);
    if (n <= 0 || (size_t)n >= sizeof pat) return NULL;
    const char *p = strstr(json, pat);
    if (!p) return NULL;
    p += (size_t)n;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    if (*p != ':') return NULL;
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

static int json_size(const char *json, const char *key, size_t *out)
{
    const char *p = find_key(json, key);
    if (!p) return 0;
    char *end = NULL;
    double v = strtod(p, &end);
    if (end == p || v < 0 || v > (double)SIZE_MAX) return 0;
    *out = (size_t)v;
    return 1;
}

static int json_f32(const char *json, const char *key, float *out)
{
    const char *p = find_key(json, key);
    if (!p) return 0;
    char *end = NULL;
    float v = strtof(p, &end);
    if (end == p) return 0;
    *out = v;
    return 1;
}

static int json_bool(const char *json, const char *key, int *out)
{
    const char *p = find_key(json, key);
    if (!p) return 0;
    if (strncmp(p, "true", 4) == 0) { *out = 1; return 1; }
    if (strncmp(p, "false", 5) == 0) { *out = 0; return 1; }
    return 0;
}

static int json_u32_array(const char *json, const char *key, uint32_t *dst,
                          size_t maxn, size_t *n_out)
{
    const char *p = find_key(json, key);
    if (!p || *p != '[') return 0;
    p++;
    size_t n = 0;
    while (*p && *p != ']') {
        while (*p == ' ' || *p == ',' || *p == '\n') p++;
        if (*p == ']') break;
        char *end = NULL;
        unsigned long v = strtoul(p, &end, 10);
        if (end == p || n >= maxn || v > UINT32_MAX) return 0;
        dst[n++] = (uint32_t)v;
        p = end;
    }
    *n_out = n;
    return n > 0;
}

static OcError copy_f32(const OcSafetensorsFile *st, const char *name,
                        OcEagle4Weight *w, size_t rows, size_t cols)
{
    const OcSafetensorsTensor *t = NULL;
    OcError e = oc_safetensors_get_tensor(st, name, &t);
    if (e != OC_OK) {
        oc_log(OC_LOG_ERROR, "eagle4: missing tensor %s", name);
        return e;
    }
    if (strcmp(t->dtype, "F32") != 0 || t->n_dims != 2 ||
        t->shape[0] != rows || t->shape[1] != cols) {
        oc_log(OC_LOG_ERROR, "eagle4: %s shape/dtype mismatch", name);
        return OC_ERR_FORMAT;
    }
    const void *raw = NULL;
    e = oc_safetensors_get_tensor_data(st, t, &raw);
    if (e != OC_OK) return e;
    size_t n = 0;
    if (!mul_ok(rows, cols, &n)) return OC_ERR_INVALID_ARG;
    memcpy(w->data, raw, n * sizeof(float));
    return OC_OK;
}

static OcError copy_vec(const OcSafetensorsFile *st, const char *name,
                        OcEagle4Weight *w, size_t n)
{
    const OcSafetensorsTensor *t = NULL;
    OcError e = oc_safetensors_get_tensor(st, name, &t);
    if (e != OC_OK) {
        oc_log(OC_LOG_ERROR, "eagle4: missing tensor %s", name);
        return e;
    }
    int ok = strcmp(t->dtype, "F32") == 0 &&
             ((t->n_dims == 1 && t->shape[0] == n) ||
              (t->n_dims == 2 && t->shape[0] == 1 && t->shape[1] == n) ||
              (t->n_dims == 2 && t->shape[0] == n && t->shape[1] == 1));
    if (!ok) {
        oc_log(OC_LOG_ERROR, "eagle4: %s shape/dtype mismatch", name);
        return OC_ERR_FORMAT;
    }
    const void *raw = NULL;
    e = oc_safetensors_get_tensor_data(st, t, &raw);
    if (e != OC_OK) return e;
    memcpy(w->data, raw, n * sizeof(float));
    return OC_OK;
}

OcError oc_eagle4_load(OcEagle4Model *m, const char *dir)
{
    if (!m || !dir) return OC_ERR_INVALID_ARG;
    char cfg_path[1024];
    char st_path[1024];
    if (snprintf(cfg_path, sizeof cfg_path, "%s/config.json", dir) >=
            (int)sizeof cfg_path ||
        snprintf(st_path, sizeof st_path, "%s/model.safetensors", dir) >=
            (int)sizeof st_path)
        return OC_ERR_INVALID_ARG;
    char *json = read_file(cfg_path, NULL);
    if (!json) return OC_ERR_IO;
    OcEagle4Config cfg;
    oc_eagle4_config_init(&cfg);
    size_t n_aux = cfg.n_aux;
    if (!json_size(json, "hidden_size", &cfg.hidden_size) ||
        !json_size(json, "target_hidden_size", &cfg.target_hidden_size) ||
        !json_size(json, "vocab_size", &cfg.vocab_size) ||
        !json_size(json, "num_attention_heads", &cfg.n_heads) ||
        !json_size(json, "num_key_value_heads", &cfg.n_kv_heads) ||
        !json_size(json, "head_dim", &cfg.head_dim) ||
        !json_size(json, "intermediate_size", &cfg.intermediate_size) ||
        !json_size(json, "lm_rank", &cfg.lm_rank) ||
        !json_size(json, "n_candidates", &cfg.n_candidates)) {
        free(json);
        return OC_ERR_FORMAT;
    }
    json_size(json, "num_aux_hidden_states", &n_aux);
    cfg.n_aux = n_aux;
    json_size(json, "sliding_window", &cfg.sliding_window);
    json_f32(json, "rms_norm_eps", &cfg.rms_eps);
    json_f32(json, "rope_theta", &cfg.rope_theta);
    json_bool(json, "use_rope", &cfg.use_rope);
    size_t n_layers = 0;
    uint32_t layers[OC_EAGLE4_MAX_AUX];
    if (json_u32_array(json, "extract_layers", layers, OC_EAGLE4_MAX_AUX,
                       &n_layers)) {
        cfg.n_aux = n_layers;
        for (size_t i = 0; i < n_layers; i++) cfg.extract_layers[i] = layers[i];
    }
    free(json);
    if (cfg.rms_eps <= 0.0f) cfg.rms_eps = 1e-6f;
    OcError e = oc_eagle4_init(m, &cfg);
    if (e != OC_OK) return e;
    OcSafetensorsFile st;
    memset(&st, 0, sizeof st);
    e = oc_safetensors_open(st_path, &st);
    if (e != OC_OK) {
        oc_eagle4_free(m);
        return e;
    }
    const size_t h = cfg.hidden_size;
    const size_t t = cfg.target_hidden_size;
    const size_t q = cfg.n_heads * cfg.head_dim;
    const size_t kv = cfg.n_kv_heads * cfg.head_dim;
    e = copy_vec(&st, "fc_norm.weight", &m->fc_norm, t);
    if (e == OC_OK) e = copy_f32(&st, "fc.weight", &m->fc, h, cfg.n_aux * t);
    if (e == OC_OK) e = copy_f32(&st, "embed_tokens.weight", &m->embed, cfg.vocab_size, h);
    if (e == OC_OK) e = copy_f32(&st, "eh_proj.weight", &m->eh_proj, h, 2 * h);
    if (e == OC_OK) e = copy_vec(&st, "pre_norm.weight", &m->pre_norm, h);
    if (e == OC_OK) e = copy_f32(&st, "q_proj.weight", &m->q_proj, q, h);
    if (e == OC_OK) e = copy_f32(&st, "k_proj.weight", &m->k_proj, kv, h);
    if (e == OC_OK) e = copy_f32(&st, "v_proj.weight", &m->v_proj, kv, h);
    if (e == OC_OK) e = copy_f32(&st, "o_proj.weight", &m->o_proj, h, q);
    if (e == OC_OK) e = copy_vec(&st, "ffn_norm.weight", &m->ffn_norm, h);
    if (e == OC_OK) e = copy_f32(&st, "gate_proj.weight", &m->gate, cfg.intermediate_size, h);
    if (e == OC_OK) e = copy_f32(&st, "up_proj.weight", &m->up, cfg.intermediate_size, h);
    if (e == OC_OK) e = copy_f32(&st, "down_proj.weight", &m->down, h, cfg.intermediate_size);
    if (e == OC_OK) e = copy_vec(&st, "post_norm.weight", &m->post_norm, h);
    if (e == OC_OK) e = copy_f32(&st, "lm_head.weight", &m->lm_head, cfg.vocab_size, h);
    if (e == OC_OK) e = copy_f32(&st, "lr_down.weight", &m->lr_down, cfg.lm_rank, h);
    if (e == OC_OK) e = copy_f32(&st, "lr_up.weight", &m->lr_up, cfg.vocab_size, cfg.lm_rank);
    oc_safetensors_close(&st);
    if (e != OC_OK) {
        oc_eagle4_free(m);
        return e;
    }
    oc_log(OC_LOG_INFO,
           "eagle4: hidden=%zu target=%zu aux=%zu vocab=%zu window=%zu rank=%zu",
           h, t, cfg.n_aux, cfg.vocab_size, cfg.sliding_window, cfg.lm_rank);
    return OC_OK;
}

static OcError ensure_cache(OcEagle4Model *m, size_t need)
{
    if (need <= m->cache_cap) return OC_OK;
    size_t cap = m->cache_cap ? m->cache_cap * 2 : 64;
    while (cap < need) {
        if (cap > SIZE_MAX / 2) return OC_ERR_OOM;
        cap *= 2;
    }
    size_t kv = m->cfg.n_kv_heads * m->cfg.head_dim;
    size_t n = 0;
    if (!mul_ok(cap, kv, &n)) return OC_ERR_OOM;
    float *k = calloc(n, sizeof(float));
    float *v = calloc(n, sizeof(float));
    if (!k || !v) {
        free(k);
        free(v);
        return OC_ERR_OOM;
    }
    if (m->cache_len > 0 && m->keys) {
        size_t old = 0;
        if (!mul_ok(m->cache_len, kv, &old)) {
            free(k);
            free(v);
            return OC_ERR_OOM;
        }
        memcpy(k, m->keys, old * sizeof(float));
        memcpy(v, m->values, old * sizeof(float));
    }
    free(m->keys);
    free(m->values);
    m->keys = k;
    m->values = v;
    m->cache_cap = cap;
    return OC_OK;
}

void oc_eagle4_reset(OcEagle4Model *m)
{
    if (!m) return;
    m->cache_len = 0;
    m->position = 0;
    m->has_feedback = 0;
    if (m->feedback && m->cfg.hidden_size)
        memset(m->feedback, 0, m->cfg.hidden_size * sizeof(float));
}

void oc_eagle4_free(OcEagle4Model *m)
{
    if (!m) return;
    weight_free(&m->fc_norm);
    weight_free(&m->fc);
    weight_free(&m->embed);
    weight_free(&m->eh_proj);
    weight_free(&m->pre_norm);
    weight_free(&m->q_proj);
    weight_free(&m->k_proj);
    weight_free(&m->v_proj);
    weight_free(&m->o_proj);
    weight_free(&m->ffn_norm);
    weight_free(&m->gate);
    weight_free(&m->up);
    weight_free(&m->down);
    weight_free(&m->post_norm);
    weight_free(&m->lm_head);
    weight_free(&m->lr_down);
    weight_free(&m->lr_up);
    free(m->feedback);
    free(m->keys);
    free(m->values);
    memset(m, 0, sizeof(*m));
}

static void attend(const OcEagle4Model *m, const float *q, float *out)
{
    const size_t hd = m->cfg.head_dim;
    const size_t n_heads = m->cfg.n_heads;
    const size_t n_kv = m->cfg.n_kv_heads;
    const size_t group = n_heads / n_kv;
    const size_t kv_stride = n_kv * hd;
    size_t begin = 0;
    if (m->cfg.sliding_window > 0 && m->cache_len > m->cfg.sliding_window)
        begin = m->cache_len - m->cfg.sliding_window;
    const size_t len = m->cache_len - begin;
    const float scale = 1.0f / sqrtf((float)hd);
    float *scores = malloc(len ? len * sizeof(float) : sizeof(float));
    memset(out, 0, n_heads * hd * sizeof(float));
    if (!scores || len == 0) {
        free(scores);
        return;
    }
    for (size_t h = 0; h < n_heads; h++) {
        size_t kv_h = h / group;
        const float *qh = q + h * hd;
        float maxs = -INFINITY;
        for (size_t t = 0; t < len; t++) {
            const float *kh = m->keys + (begin + t) * kv_stride + kv_h * hd;
            scores[t] = row_dot(qh, kh, hd) * scale;
            if (scores[t] > maxs) maxs = scores[t];
        }
        float sum = 0.0f;
        for (size_t t = 0; t < len; t++) {
            scores[t] = expf(scores[t] - maxs);
            sum += scores[t];
        }
        float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        float *oh = out + h * hd;
        for (size_t t = 0; t < len; t++) {
            const float *vh = m->values + (begin + t) * kv_stride + kv_h * hd;
            float p = scores[t] * inv;
            for (size_t d = 0; d < hd; d++) oh[d] += p * vh[d];
        }
    }
    free(scores);
}

typedef struct {
    float score;
    uint32_t index;
} OcEagleCand;

static void heap_swap(OcEagleCand *a, OcEagleCand *b)
{
    OcEagleCand t = *a;
    *a = *b;
    *b = t;
}

static void heap_sift_up(OcEagleCand *h, size_t i)
{
    while (i > 0) {
        size_t p = (i - 1) / 2;
        if (h[p].score <= h[i].score) break;
        heap_swap(&h[p], &h[i]);
        i = p;
    }
}

static void heap_sift_down(OcEagleCand *h, size_t n, size_t i)
{
    for (;;) {
        size_t l = i * 2 + 1;
        size_t r = l + 1;
        size_t s = i;
        if (l < n && h[l].score < h[s].score) s = l;
        if (r < n && h[r].score < h[s].score) s = r;
        if (s == i) break;
        heap_swap(&h[s], &h[i]);
        i = s;
    }
}

static uint32_t pick_token(const OcEagle4Model *m, const float *hidden)
{
    const size_t rank = m->cfg.lm_rank;
    const size_t h = m->cfg.hidden_size;
    const size_t vocab = m->cfg.vocab_size;
    size_t k = m->cfg.n_candidates;
    if (k > vocab) k = vocab;
    float *low = malloc(rank * sizeof(float));
    OcEagleCand *heap = malloc(k * sizeof(*heap));
    if (!low || !heap) {
        free(low);
        free(heap);
        return 0;
    }
    gemv(&m->lr_down, hidden, low);
    size_t n = 0;
    for (size_t v = 0; v < vocab; v++) {
        float s = row_dot(m->lr_up.data + v * rank, low, rank);
        if (n < k) {
            heap[n].score = s;
            heap[n].index = (uint32_t)v;
            heap_sift_up(heap, n);
            n++;
        } else if (s > heap[0].score) {
            heap[0].score = s;
            heap[0].index = (uint32_t)v;
            heap_sift_down(heap, n, 0);
        }
    }
    uint32_t best = heap[0].index;
    float best_logit = -INFINITY;
    for (size_t i = 0; i < n; i++) {
        uint32_t idx = heap[i].index;
        float logit = row_dot(m->lm_head.data + (size_t)idx * h, hidden, h);
        if (logit > best_logit) {
            best_logit = logit;
            best = idx;
        }
    }
    free(low);
    free(heap);
    return best;
}

OcError oc_eagle4_step(OcEagle4Model *m, uint32_t token, const float *aux,
                       float *out_hidden, uint32_t *out_token, float *out_logit)
{
    if (!m || !m->loaded) return OC_ERR_INVALID_ARG;
    if (token >= m->cfg.vocab_size) return OC_ERR_INVALID_ARG;
    if (!aux && !m->has_feedback) return OC_ERR_INVALID_ARG;
    const OcEagle4Config *c = &m->cfg;
    const size_t h = c->hidden_size;
    const size_t t = c->target_hidden_size;
    const size_t qn = c->n_heads * c->head_dim;
    const size_t kvn = c->n_kv_heads * c->head_dim;
    const size_t inter = c->intermediate_size;
    float *cat = malloc((c->n_aux * t) * sizeof(float));
    float *g = malloc(h * sizeof(float));
    float *cat2 = malloc((2 * h) * sizeof(float));
    float *u = malloc(h * sizeof(float));
    float *q = malloc(qn * sizeof(float));
    float *k = malloc(kvn * sizeof(float));
    float *v = malloc(kvn * sizeof(float));
    float *attn = malloc(qn * sizeof(float));
    float *proj = malloc(h * sizeof(float));
    float *normed = malloc(h * sizeof(float));
    float *gate = malloc(inter * sizeof(float));
    float *up = malloc(inter * sizeof(float));
    float *hidden = malloc(h * sizeof(float));
    if (!cat || !g || !cat2 || !u || !q || !k || !v || !attn || !proj ||
        !normed || !gate || !up || !hidden) {
        free(cat); free(g); free(cat2); free(u); free(q); free(k); free(v);
        free(attn); free(proj); free(normed); free(gate); free(up); free(hidden);
        return OC_ERR_OOM;
    }
    if (aux) {
        for (size_t i = 0; i < c->n_aux; i++) {
            rms_norm(aux + i * t, m->fc_norm.data, cat + i * t, t, c->rms_eps);
        }
        gemv(&m->fc, cat, g);
    } else {
        memcpy(g, m->feedback, h * sizeof(float));
    }
    const float *emb = m->embed.data + (size_t)token * h;
    memcpy(cat2, emb, h * sizeof(float));
    memcpy(cat2 + h, g, h * sizeof(float));
    gemv(&m->eh_proj, cat2, u);
    rms_norm(u, m->pre_norm.data, u, h, c->rms_eps);
    gemv(&m->q_proj, u, q);
    gemv(&m->k_proj, u, k);
    gemv(&m->v_proj, u, v);
    if (c->use_rope) {
        apply_rope(q, c->n_heads, c->head_dim, m->position, c->rope_theta);
        apply_rope(k, c->n_kv_heads, c->head_dim, m->position, c->rope_theta);
    }
    OcError ce = ensure_cache(m, m->cache_len + 1);
    if (ce != OC_OK) {
        free(cat); free(g); free(cat2); free(u); free(q); free(k); free(v);
        free(attn); free(proj); free(normed); free(gate); free(up); free(hidden);
        return ce;
    }
    size_t stride = kvn;
    memcpy(m->keys + m->cache_len * stride, k, kvn * sizeof(float));
    memcpy(m->values + m->cache_len * stride, v, kvn * sizeof(float));
    m->cache_len++;
    attend(m, q, attn);
    gemv(&m->o_proj, attn, proj);
    for (size_t i = 0; i < h; i++) hidden[i] = g[i] + proj[i];
    rms_norm(hidden, m->ffn_norm.data, normed, h, c->rms_eps);
    gemv(&m->gate, normed, gate);
    gemv(&m->up, normed, up);
    for (size_t i = 0; i < inter; i++) gate[i] = silu(gate[i]) * up[i];
    gemv(&m->down, gate, proj);
    for (size_t i = 0; i < h; i++) hidden[i] += proj[i];
    rms_norm(hidden, m->post_norm.data, m->feedback, h, c->rms_eps);
    m->has_feedback = 1;
    m->position++;
    if (out_hidden) memcpy(out_hidden, m->feedback, h * sizeof(float));
    uint32_t picked = pick_token(m, m->feedback);
    if (out_token) *out_token = picked;
    if (out_logit) {
        *out_logit = row_dot(m->lm_head.data + (size_t)picked * h,
                             m->feedback, h);
    }
    free(cat); free(g); free(cat2); free(u); free(q); free(k); free(v);
    free(attn); free(proj); free(normed); free(gate); free(up); free(hidden);
    return OC_OK;
}

OcError oc_eagle4_draft(OcEagle4Model *m, uint32_t token, const float *aux,
                        uint32_t n_draft, uint32_t *out_tokens)
{
    if (!m || !aux || !out_tokens || n_draft == 0) return OC_ERR_INVALID_ARG;
    if (n_draft > OC_EAGLE4_MAX_DRAFT) n_draft = OC_EAGLE4_MAX_DRAFT;
    oc_eagle4_reset(m);
    for (uint32_t i = 0; i < n_draft; i++) {
        uint32_t next = 0;
        OcError e = oc_eagle4_step(m, token, i == 0 ? aux : NULL, NULL, &next,
                                   NULL);
        if (e != OC_OK) return e;
        out_tokens[i] = next;
        token = next;
    }
    return OC_OK;
}

OcEagleFeatureCapture *oc_eagle4_capture_new(const uint32_t *layers,
                                            uint32_t n_layers,
                                            uint32_t hidden,
                                            uint32_t cap)
{
    if (!layers || n_layers == 0 || n_layers > OC_EAGLE4_MAX_AUX ||
        hidden == 0 || cap == 0)
        return NULL;
    OcEagleFeatureCapture *c = calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->n_layers = n_layers;
    c->hidden = hidden;
    c->cap = cap;
    for (uint32_t i = 0; i < n_layers; i++) c->layers[i] = layers[i];
    c->tokens = calloc(cap, sizeof(uint32_t));
    size_t plane = 0;
    size_t n = 0;
    if (!c->tokens || !mul_ok(cap, n_layers, &plane) ||
        !mul_ok(plane, hidden, &n)) {
        oc_eagle4_capture_free(c);
        return NULL;
    }
    c->hiddens = calloc(n, sizeof(float));
    if (!c->hiddens) {
        oc_eagle4_capture_free(c);
        return NULL;
    }
    return c;
}

void oc_eagle4_capture_free(OcEagleFeatureCapture *cap)
{
    if (!cap) return;
    free(cap->tokens);
    free(cap->hiddens);
    free(cap);
}

void oc_eagle4_capture_at(OcEagleFeatureCapture *cap, uint32_t layer,
                          uint32_t index, uint32_t token,
                          const float *hidden, size_t n)
{
    if (!cap || !hidden || index >= cap->cap || n != cap->hidden) return;
    uint32_t slot = UINT32_MAX;
    for (uint32_t i = 0; i < cap->n_layers; i++) {
        if (cap->layers[i] == layer) { slot = i; break; }
    }
    if (slot == UINT32_MAX) return;
    cap->tokens[index] = token;
    size_t off = ((size_t)index * cap->n_layers + slot) * cap->hidden;
    memcpy(cap->hiddens + off, hidden, n * sizeof(float));
    if (index + 1 > cap->n_tokens) cap->n_tokens = index + 1;
}

OcError oc_eagle4_capture_concat(const OcEagleFeatureCapture *cap,
                                 uint32_t index, float *dst, size_t dst_n)
{
    if (!cap || !dst || index >= cap->n_tokens) return OC_ERR_INVALID_ARG;
    size_t need = (size_t)cap->n_layers * cap->hidden;
    if (dst_n < need) return OC_ERR_INVALID_ARG;
    for (uint32_t s = 0; s < cap->n_layers; s++) {
        size_t off = ((size_t)index * cap->n_layers + s) * cap->hidden;
        memcpy(dst + (size_t)s * cap->hidden, cap->hiddens + off,
               cap->hidden * sizeof(float));
    }
    return OC_OK;
}

OcError oc_eagle4_capture_save(const OcEagleFeatureCapture *cap,
                               const char *path)
{
    if (!cap || !path || cap->n_tokens == 0) return OC_ERR_INVALID_ARG;
    FILE *f = fopen(path, "wb");
    if (!f) return OC_ERR_IO;
    uint32_t hdr[3] = { cap->n_tokens, cap->n_layers, cap->hidden };
    int ok = fwrite("E4H1", 1, 4, f) == 4 &&
             fwrite(hdr, sizeof hdr, 1, f) == 1 &&
             fwrite(cap->layers, sizeof(uint32_t), cap->n_layers, f) ==
                 cap->n_layers &&
             fwrite(cap->tokens, sizeof(uint32_t), cap->n_tokens, f) ==
                 cap->n_tokens;
    size_t nfloat = (size_t)cap->n_tokens * cap->n_layers * cap->hidden;
    ok = ok && fwrite(cap->hiddens, sizeof(float), nfloat, f) == nfloat;
    if (fclose(f) != 0) ok = 0;
    return ok ? OC_OK : OC_ERR_IO;
}
