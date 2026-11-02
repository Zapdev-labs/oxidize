#include "oxidize/exl3_overlay.h"
#include "oxidize/exl3.h"
#include "oxidize/hf_metadata.h"
#include "oxidize/llama.h"
#include "oxidize/safetensors.h"
#include "oxidize/oxk.h"
#include "oxidize/log.h"

#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

typedef struct OcExl3Entry {
    OcExl3Matrix matrix;
    OcWeightView *view;
} OcExl3Entry;

struct OcExl3Overlay {
    OcSafetensorsFile *files;
    size_t n_files;
    OcExl3Entry *entries;
    size_t n_entries;
    void **owned;
    size_t n_owned;
};

void oc_exl3_overlay_free(OcExl3Overlay *o)
{
    if (!o) return;
    for (size_t i = 0; i < o->n_entries; i++) {
        free((void *)o->entries[i].matrix.suh);
        free((void *)o->entries[i].matrix.svh);
    }
    for (size_t i = 0; i < o->n_files; i++) oc_safetensors_close(&o->files[i]);
    for (size_t i = 0; i < o->n_owned; i++) free(o->owned[i]);
    free(o->owned);
    free(o->entries);
    free(o->files);
    free(o);
}

static bool ends_with(const char *s, const char *suffix)
{
    size_t n = strlen(s), k = strlen(suffix);
    return n >= k && strcmp(s + n - k, suffix) == 0;
}

static OcError add_file(OcExl3Overlay *o, const char *path)
{
    if (o->n_files == SIZE_MAX / sizeof(*o->files)) return OC_ERR_OOM;
    OcSafetensorsFile *p = realloc(o->files, (o->n_files + 1) * sizeof(*p));
    if (!p) return OC_ERR_OOM;
    o->files = p;
    OcError e = oc_safetensors_open(path, &p[o->n_files]);
    if (e == OC_OK) o->n_files++;
    return e;
}

static OcError open_files(OcExl3Overlay *o, const char *path)
{
    struct stat st;
    if (stat(path, &st)) return OC_ERR_IO;
    if (!S_ISDIR(st.st_mode)) return add_file(o, path);
    DIR *dir = opendir(path);
    if (!dir) return OC_ERR_IO;
    OcError e = OC_OK;
    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        if (!ends_with(ent->d_name, ".safetensors")) continue;
        size_t n = strlen(path) + strlen(ent->d_name) + 2;
        char *p = malloc(n);
        if (!p) { e = OC_ERR_OOM; break; }
        snprintf(p, n, "%s/%s", path, ent->d_name);
        e = add_file(o, p);
        free(p);
        if (e != OC_OK) break;
    }
    closedir(dir);
    return e == OC_OK && !o->n_files ? OC_ERR_FORMAT : e;
}

static const OcSafetensorsTensor *find_tensor(const OcExl3Overlay *o,
                                             const char *name, const void **data)
{
    for (size_t f = 0; f < o->n_files; f++) {
        const OcSafetensorsTensor *t;
        if (oc_safetensors_get_tensor(&o->files[f], name, &t) == OC_OK) {
            if (oc_safetensors_get_tensor_data(&o->files[f], t, data) != OC_OK)
                return NULL;
            return t;
        }
    }
    return NULL;
}

static const OcSafetensorsTensor *companion(const OcExl3Overlay *o,
                                            const char *key, const char *suffix,
                                            const void **data)
{
    char name[OC_SAFETENSORS_NAME_LEN];
    int n = snprintf(name, sizeof(name), "%s.%s", key, suffix);
    if (n < 0 || (size_t)n >= sizeof(name)) return NULL;
    return find_tensor(o, name, data);
}

static OcError scale(const OcExl3Overlay *o, const char *key, bool input,
                     size_t dim, const float **out)
{
    const void *a = NULL, *b = NULL;
    const OcSafetensorsTensor *h = companion(o, key, input ? "suh" : "svh", &a);
    const OcSafetensorsTensor *p = companion(o, key, input ? "su" : "sv", &b);
    if ((h != NULL) == (p != NULL)) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
    const OcSafetensorsTensor *t = h ? h : p;
    size_t count = h ? dim : dim / 16;
    if (t->n_dims != 1 || t->shape[0] != count ||
        t->data_length != count * 2 || strcmp(t->dtype, h ? "F16" : "I16"))
        do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
    float *v = malloc(dim * sizeof(*v));
    if (!v) return OC_ERR_OOM;
    const uint8_t *raw = h ? a : b;
    for (size_t i = 0; i < dim; i++) {
        v[i] = h ? oc_oxk_f16_le_to_f32(raw + i * 2)
                 : ((raw[i / 8] >> (i % 8)) & 1) ? -1.0f : 1.0f;
        if (!isfinite(v[i])) { free(v); do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0); }
    }
    *out = v;
    return OC_OK;
}

static OcWeightView *target(OcLlamaModel *m, const char *key)
{
    if (!strcmp(key, "lm_head")) return &m->output;
    unsigned layer;
    int used = 0;
    if (sscanf(key, "model.layers.%u.%n", &layer, &used) != 1 ||
        used == 0 || layer >= m->cfg.n_layer) return NULL;
    char canonical[64];
    snprintf(canonical, sizeof(canonical), "model.layers.%u.", layer);
    if (strncmp(key, canonical, strlen(canonical))) return NULL;
    OcLlamaLayer *l = &m->layers[layer];
    const char *s = key + used;
    if (!strcmp(s, "self_attn.q_proj")) return &l->attn_q;
    if (!strcmp(s, "self_attn.k_proj")) return &l->attn_k;
    if (!strcmp(s, "self_attn.v_proj")) return &l->attn_v;
    if (!strcmp(s, "self_attn.o_proj")) return &l->attn_output;
    if (!strcmp(s, "mlp.gate_proj")) return &l->ffn_gate;
    if (!strcmp(s, "mlp.up_proj")) return &l->ffn_up;
    if (!strcmp(s, "mlp.down_proj")) return &l->ffn_down;
    return NULL;
}

static OcError attach_exl3(OcLlamaModel *m, OcExl3Overlay *o, bool native)
{
    OcError e = OC_OK;
    size_t count = 0;
    /* Duplicate names, including companions in different shards, are ambiguous. */
    for (size_t f = 0; f < o->n_files; f++) {
        for (size_t i = 0; i < o->files[f].n_tensors; i++) {
            const void *checked = NULL;
            e = oc_safetensors_get_tensor_data(&o->files[f],
                    &o->files[f].tensors[i], &checked);
            if (e != OC_OK) goto fail;
            const char *name = o->files[f].tensors[i].name;
            for (size_t g = 0; g <= f; g++) {
                size_t end = g == f ? i : o->files[g].n_tensors;
                for (size_t j = 0; j < end; j++)
                    if (!strcmp(name, o->files[g].tensors[j].name)) {
                        e = OC_ERR_FORMAT; goto fail;
                    }
            }
            if (ends_with(name, ".trellis")) count++;
        }
    }
    if (!count) { e = OC_ERR_FORMAT; goto fail; }
    o->entries = calloc(count, sizeof(*o->entries));
    if (!o->entries) { e = OC_ERR_OOM; goto fail; }
    for (size_t f = 0; f < o->n_files; f++) {
        for (size_t i = 0; i < o->files[f].n_tensors; i++) {
            const OcSafetensorsTensor *t = &o->files[f].tensors[i];
            if (!ends_with(t->name, ".trellis")) continue;
            char key[OC_SAFETENSORS_NAME_LEN];
            size_t len = strlen(t->name) - strlen(".trellis");
            memcpy(key, t->name, len); key[len] = 0;
            OcExl3Entry *entry = &o->entries[o->n_entries++];
            entry->view = target(m, key);
            if (!entry->view || !entry->view->data) { e = OC_ERR_MODEL; goto fail; }
            OcExl3Matrix *q = &entry->matrix;
            q->input_dim = entry->view->cols;
            q->output_dim = entry->view->rows;
            if (!q->input_dim || !q->output_dim ||
                q->input_dim % 128 || q->output_dim % 128 ||
                q->input_dim > SIZE_MAX / sizeof(float) ||
                q->output_dim > SIZE_MAX / sizeof(float)) {
                e = OC_ERR_FORMAT; goto fail;
            }
            if (strcmp(t->dtype, "I16") || t->n_dims != 3 ||
                t->shape[0] != q->input_dim / 16 ||
                t->shape[1] != q->output_dim / 16 ||
                t->shape[2] < 8 || t->shape[2] > 128 || t->shape[2] % 8) {
                e = OC_ERR_FORMAT; goto fail;
            }
            const void *data = NULL;
            e = oc_safetensors_get_tensor_data(&o->files[f], t, &data);
            if (e != OC_OK) goto fail;
            q->packed = data; q->packed_size = t->data_length;
            q->bits_twice = (unsigned)(t->shape[2] / 8);
            bool mcg = companion(o, key, "mcg", &data) != NULL;
            bool mul1 = companion(o, key, "mul1", &data) != NULL;
            if ((mcg && mul1) || (!native && companion(o, key, "bias", &data)) ||
                companion(o, key, "scale", &data)) { e = OC_ERR_FORMAT; goto fail; }
            q->codebook = mul1 ? 2u :
                          mcg ? 1u : 0u;
            e = scale(o, key, true, q->input_dim, &q->suh);
            if (e == OC_OK) e = scale(o, key, false, q->output_dim, &q->svh);
            if (e == OC_OK) e = oc_exl3_validate(q);
            if (e != OC_OK) goto fail;
        }
    }
    for (size_t i = 0; i < o->n_entries; i++) {
        o->entries[i].view->exl3 = &o->entries[i].matrix;
        if (o->entries[i].view == &m->output) m->cfg.tied_embeddings = false;
    }
    m->exl3_overlay = o;
    oc_log(OC_LOG_INFO, "EXL3 overlay: attached %zu matrices from %zu shards",
           o->n_entries, o->n_files);
    return OC_OK;
fail:
    return e;
}

OcError oc_llama_load_exl3(OcLlamaModel *m, const char *path)
{
    if (!m || !path || !m->layers || !m->output.data ||
        m->live_sessions || m->exl3_overlay) return OC_ERR_INVALID_ARG;
    if ((m->arch != OC_ARCH_LLAMA && m->arch != OC_ARCH_MISTRAL &&
         m->arch != OC_ARCH_QWEN) || m->cfg.num_experts || m->cfg.uses_mla ||
        m->cfg.is_qwen35 || m->cfg.is_k2 || m->cfg.is_qwen4exp ||
        m->mtp.present || m->cfg.nextn_predict_layers) return OC_ERR_MODEL;
    OcExl3Overlay *o = calloc(1, sizeof(*o));
    if (!o) return OC_ERR_OOM;
    OcError e = open_files(o, path);
    if (e == OC_OK) e = attach_exl3(m, o, false);
    if (e != OC_OK) oc_exl3_overlay_free(o);
    return e;
}


static OcError native_vector(const OcExl3Overlay *o, const char *name,
                             size_t dim, bool optional, float **out)
{
    const void *data = NULL;
    const OcSafetensorsTensor *t = find_tensor(o, name, &data);
    if (!t) return optional ? OC_OK : OC_ERR_TENSOR;
    size_t width = !strcmp(t->dtype, "F32") ? 4 :
        (!strcmp(t->dtype, "F16") || !strcmp(t->dtype, "BF16")) ? 2 : 0;
    if (!width || t->n_dims != 1 || t->shape[0] != dim ||
        dim > SIZE_MAX / sizeof(float) || t->data_length != dim * width)
        do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
    float *v = malloc(dim * sizeof(*v));
    if (!v) return OC_ERR_OOM;
    const uint8_t *p = data;
    for (size_t i = 0; i < dim; i++) {
        if (width == 4) memcpy(&v[i], p + i * 4, 4);
        else if (!strcmp(t->dtype, "F16")) v[i] = oc_oxk_f16_le_to_f32(p + i * 2);
        else {
            uint32_t bits = ((uint32_t)p[i*2] | (uint32_t)p[i*2+1] << 8) << 16;
            memcpy(&v[i], &bits, 4);
        }
        if (!isfinite(v[i])) { free(v); do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0); }
    }
    *out = v;
    return OC_OK;
}

static OcError native_matrix(OcExl3Overlay *o, const char *key,
                             size_t rows, size_t cols, bool packed,
                             OcWeightView *view)
{
    const void *data = NULL, *trellis = NULL;
    const OcSafetensorsTensor *t = companion(o, key, "weight", &data);
    const OcSafetensorsTensor *q = companion(o, key, "trellis", &trellis);
    if (q) {
        if (!packed || t) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
        *view = (OcWeightView){.data=trellis, .rows=rows, .cols=cols};
        return OC_OK;
    }
    if (!t) { oc_log_error("Q4EMISS %s", key); return OC_ERR_TENSOR; }
    size_t width = !strcmp(t->dtype,"F32") ? 4 :
        (!strcmp(t->dtype,"F16") || !strcmp(t->dtype,"BF16")) ? 2 : 0;
    if (!width || !rows || !cols || cols > SIZE_MAX / width ||
        rows > SIZE_MAX / (cols * width) || t->n_dims != 2 ||
        t->shape[0] != rows || t->shape[1] != cols ||
        t->data_length != rows * cols * width) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
    /* SafeTensors permits unaligned payload offsets. Keep ordinary views
     * aligned for CPU kernels without expanding their storage dtype. */
    if ((uintptr_t)data % width) {
        void *copy = malloc(t->data_length);
        if (!copy) return OC_ERR_OOM;
        void **owned = realloc(o->owned, (o->n_owned + 1) * sizeof(*owned));
        if (!owned) { free(copy); return OC_ERR_OOM; }
        o->owned = owned; o->owned[o->n_owned++] = copy;
        memcpy(copy, data, t->data_length); data = copy;
    }
    *view = (OcWeightView){.data=data, .rows=rows, .cols=cols,
        .row_bytes=cols*width, .qtype=width == 4 ? OC_QUANT_F32 :
            !strcmp(t->dtype,"F16") ? OC_QUANT_F16 : OC_QUANT_BF16};
    return OC_OK;
}

static bool native_known_tensor(OcLlamaModel *m, const char *name)
{
    if (!strcmp(name,"model.embed_tokens.weight") || !strcmp(name,"model.norm.weight"))
        return true;
    char key[OC_SAFETENSORS_NAME_LEN];
    size_t n = strlen(name);
    if (n >= sizeof(key)) return false;
    memcpy(key,name,n+1);
    char *suffix = strrchr(key,'.');
    if (!suffix) return false;
    *suffix++ = 0;
    OcWeightView *w = target(m,key);
    if (w) {
        if (!strcmp(suffix,"bias")) {
            for (size_t i=0;i<m->cfg.n_layer;i++)
                if (w == &m->layers[i].attn_q || w == &m->layers[i].attn_k ||
                    w == &m->layers[i].attn_v) return true;
            return false;
        }
        if (!strcmp(suffix,"weight")) return true;
        const void *data;
        if (!companion(m->exl3_overlay,key,"trellis",&data)) return false;
        return !strcmp(suffix,"trellis") || !strcmp(suffix,"suh") ||
            !strcmp(suffix,"svh") || !strcmp(suffix,"su") ||
            !strcmp(suffix,"sv") || !strcmp(suffix,"mcg") || !strcmp(suffix,"mul1");
    }
    for (uint32_t i=0;i<m->cfg.n_layer;i++) {
        const char *norms[]={"input_layernorm", "post_attention_layernorm",
            "self_attn.q_norm", "self_attn.k_norm"};
        for(size_t j=0;j<4;j++) {
            char expected[128];
            snprintf(expected,sizeof(expected),"model.layers.%u.%s.weight",i,norms[j]);
            if (!strcmp(name,expected)) return true;
        }
    }
    return false;
}

/* ── Native qwen4exp (CYBER-FROST class: Qwen3.8-Flash-Next EXL3) ────────
 *
 * Direct load of an HF EXL3 SafeTensors directory. The checkpoint names its
 * tensors model.language_model.* and keeps experts as INDIVIDUAL per-expert
 * trellis tensors (512 × 3 per layer, not stacked), so the loader:
 *   - indexes every tensor once and resolves names by binary search,
 *   - binds full-attention q_proj directly (HF already fuses the sigmoid
 *     output gate per head: rows are [q_h | gate_h] interleaved),
 *   - relabels the linear-attention V heads from HF's grouped-by-K-head
 *     order to the tiled order the forward expects, via 128-block
 *     permutation views (no packed bytes are copied or requantized),
 *   - dequantizes the small native (F16/BF16) tensors to owned f32,
 *   - folds the zero-centred norms (+1) exactly as the GGUF converter does,
 *   - fuses mtp.fc_embedding/fc_hidden into a single F32 eh_proj,
 *   - leaves the PLE n-gram table unbound for now: the checkpoint packs it
 *     in a custom exl3_ngram_trellis format that is not decodable here yet,
 *     so ple_enabled is turned off with a warning.
 *
 * The vision tower and the QSA indexer tensors are allowed but skipped: the
 * dense attention path is exact for every context length. */

typedef struct {
    size_t file, idx;       /* into the overlay's files / tensors arrays */
    const char *name;       /* points into the tensor struct             */
} Q4eName;

typedef struct {
    Q4eName *items;
    size_t n;
    unsigned char *used;
} Q4eIndex;

static int q4e_name_cmp(const void *a, const void *b)
{
    return strcmp(((const Q4eName *)a)->name, ((const Q4eName *)b)->name);
}

static OcError q4e_index_build(OcExl3Overlay *o, Q4eIndex *ni)
{
    size_t total = 0;
    for (size_t f = 0; f < o->n_files; f++) total += o->files[f].n_tensors;
    ni->items = malloc((total ? total : 1) * sizeof(*ni->items));
    ni->used = calloc(total ? total : 1, 1);
    if (!ni->items || !ni->used) return OC_ERR_OOM;
    ni->n = 0;
    for (size_t f = 0; f < o->n_files; f++)
        for (size_t i = 0; i < o->files[f].n_tensors; i++) {
            Q4eName *e = &ni->items[ni->n++];
            e->file = f; e->idx = i;
            e->name = o->files[f].tensors[i].name;
        }
    qsort(ni->items, ni->n, sizeof(*ni->items), q4e_name_cmp);
    /* Duplicate names across shards are ambiguous — reject them here once,
     * O(n log n), instead of the dense path's quadratic scan. */
    for (size_t i = 1; i < ni->n; i++)
        if (!strcmp(ni->items[i].name, ni->items[i - 1].name))
            do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
    return OC_OK;
}

/* Resolve `name`; marks it consumed. NULL when absent. */
static const OcSafetensorsTensor *q4e_find(Q4eIndex *ni, OcExl3Overlay *o,
                                           const char *name, const void **data,
                                           size_t *mark)
{
    Q4eName key = {0, 0, name};
    Q4eName *hit = bsearch(&key, ni->items, ni->n, sizeof(*ni->items),
                           q4e_name_cmp);
    if (!hit) return NULL;
    ni->used[hit - ni->items] = 1;
    if (mark) *mark = (size_t)(hit - ni->items);
    const OcSafetensorsTensor *t = &o->files[hit->file].tensors[hit->idx];
    if (data && oc_safetensors_get_tensor_data(&o->files[hit->file], t, data))
        return NULL;
    return t;
}

static void *q4e_owned(OcExl3Overlay *o, void *p)
{
    void **owned = realloc(o->owned, (o->n_owned + 1) * sizeof(*owned));
    if (!owned) return NULL;
    o->owned = owned;
    o->owned[o->n_owned++] = p;
    return p;
}

/* 1-D F16/BF16/F32 vector → owned f32 with an optional zero-centre fold
 * (+1), optional gather by `row_perm`, and an optional v[i] = -exp(v[i])
 * transform (A_log → A) applied BEFORE the fold. */
static OcError q4e_vector(Q4eIndex *ni, OcExl3Overlay *o, const char *name,
                          size_t dim, float **out, float add,
                          const uint32_t *row_perm, bool neg_exp)
{
    const void *data = NULL;
    const OcSafetensorsTensor *t = q4e_find(ni, o, name, &data, NULL);
    if (!t) { oc_log_error("Q4EMISS %s", name); return OC_ERR_TENSOR; }
    size_t width = !strcmp(t->dtype, "F32") ? 4 :
        (!strcmp(t->dtype, "F16") || !strcmp(t->dtype, "BF16")) ? 2 : 0;
    if (!width || t->n_dims != 1 || t->shape[0] != dim ||
        dim > SIZE_MAX / sizeof(float) / 2 || t->data_length != dim * width)
        do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
    float *v = malloc((dim ? dim : 1) * sizeof(*v));
    if (!v) return OC_ERR_OOM;
    for (size_t i = 0; i < dim; i++) {
        size_t s = row_perm ? row_perm[i] : i;
        if (s >= dim) { free(v); do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0); }
        float x;
        const uint8_t *p = (const uint8_t *)data + s * width;
        if (width == 4) memcpy(&x, p, 4);
        else if (!strcmp(t->dtype, "F16")) x = oc_oxk_f16_le_to_f32(p);
        else {
            uint32_t bits = ((uint32_t)p[0] | (uint32_t)p[1] << 8) << 16;
            memcpy(&x, &bits, 4);
        }
        if (neg_exp) x = -expf(x);
        v[i] = x + add;
        if (!isfinite(v[i])) { free(v); do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0); }
    }
    /* NOT registered with the overlay: q4e_vector outputs are assigned to
     * model fields that oc_llama_free() releases (norms, ssm_*_f32, …). */
    *out = v;
    return OC_OK;
}

/* 2-D F16/BF16/F32 matrix → owned F32 view with optional row permutation.
 * rows/cols are the logical view dims (rows = outputs, cols = inputs). */
static OcError q4e_matrix_f32(Q4eIndex *ni, OcExl3Overlay *o, const char *name,
                              size_t rows, size_t cols,
                              const uint32_t *row_perm, OcWeightView *view)
{
    const void *data = NULL;
    char wname[OC_SAFETENSORS_NAME_LEN];
    if (snprintf(wname, sizeof(wname), "%s.weight", name) >=
        (int)sizeof(wname)) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
    const OcSafetensorsTensor *t = q4e_find(ni, o, wname, &data, NULL);
    if (!t) { oc_log_error("Q4EMISS %s", wname); return OC_ERR_TENSOR; }
    size_t width = !strcmp(t->dtype, "F32") ? 4 :
        (!strcmp(t->dtype, "F16") || !strcmp(t->dtype, "BF16")) ? 2 : 0;
    if (!width || !rows || !cols || t->n_dims != 2 ||
        t->shape[0] != rows || t->shape[1] != cols ||
        cols > SIZE_MAX / width || rows > SIZE_MAX / (cols * width) ||
        t->data_length != rows * cols * width)
        do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
    float *v = malloc(rows * cols * sizeof(*v));
    if (!v) return OC_ERR_OOM;
    for (size_t r = 0; r < rows; r++) {
        size_t s = row_perm ? row_perm[r] : r;
        if (s >= rows) { free(v); do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0); }
        const uint8_t *p = (const uint8_t *)data + s * cols * width;
        for (size_t i = 0; i < cols; i++) {
            float x;
            const uint8_t *q = p + i * width;
            if (width == 4) memcpy(&x, q, 4);
            else if (!strcmp(t->dtype, "F16")) x = oc_oxk_f16_le_to_f32(q);
            else {
                uint32_t bits = ((uint32_t)q[0] | (uint32_t)q[1] << 8) << 16;
                memcpy(&x, &bits, 4);
            }
            v[r * cols + i] = x;
            if (!isfinite(x)) { free(v); do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0); }
        }
    }
    float *owned = q4e_owned(o, v);
    if (!owned) { free(v); return OC_ERR_OOM; }
    *view = (OcWeightView){.data = (const uint8_t *)owned, .rows = rows, .cols = cols,
        .row_bytes = cols * sizeof(float), .qtype = OC_QUANT_F32};
    return OC_OK;
}

/* The dense path's scale() resolves companions by linear scan; the qwen4exp
 * loader resolves thousands of matrices, so look their scales up through
 * the name index instead. Same on-disk forms: suh/svh = F16 [dim], or
 * su/sv = packed sign bits ([dim/16] I16). */
static OcError q4e_scale(Q4eIndex *ni, OcExl3Overlay *o, const char *key,
                         bool input, size_t dim, const float **out)
{
    char name[OC_SAFETENSORS_NAME_LEN];
    const void *h = NULL, *p = NULL;
    if (snprintf(name, sizeof(name), "%s.%s", key, input ? "suh" : "svh") >=
        (int)sizeof(name)) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
    const OcSafetensorsTensor *th = q4e_find(ni, o, name, &h, NULL);
    if (snprintf(name, sizeof(name), "%s.%s", key, input ? "su" : "sv") >=
        (int)sizeof(name)) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
    const OcSafetensorsTensor *tp = q4e_find(ni, o, name, &p, NULL);
    if ((th != NULL) == (tp != NULL)) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
    const OcSafetensorsTensor *t = th ? th : tp;
    size_t count = th ? dim : dim / 16;
    if (t->n_dims != 1 || t->shape[0] != count ||
        t->data_length != count * 2 || strcmp(t->dtype, th ? "F16" : "I16"))
        do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
    float *v = malloc((dim ? dim : 1) * sizeof(*v));
    if (!v) return OC_ERR_OOM;
    const uint8_t *raw = th ? h : p;
    for (size_t i = 0; i < dim; i++) {
        v[i] = th ? oc_oxk_f16_le_to_f32(raw + i * 2)
                  : ((raw[i / 8] >> (i % 8)) & 1) ? -1.0f : 1.0f;
        if (!isfinite(v[i])) { free(v); do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0); }
    }
    float *owned = q4e_owned(o, v);
    if (!owned) { free(v); return OC_ERR_OOM; }
    *out = owned;
    return OC_OK;
}

/* EXL3 trellis matrix + scales, resolved by name. The matrix is
 * heap-allocated (stable address for OcWeightView.exl3) and owned by the
 * overlay. `in_perm`/`out_perm` attach 128-block relabeling (stored on the
 * matrix, packed bytes untouched). */
static OcError q4e_exl3_matrix(Q4eIndex *ni, OcExl3Overlay *o, const char *key,
                               size_t in_dim, size_t out_dim,
                               const uint32_t *in_perm,
                               const uint32_t *out_perm, OcExl3Matrix **out)
{
    char name[OC_SAFETENSORS_NAME_LEN];
    if (snprintf(name, sizeof(name), "%s.trellis", key) >= (int)sizeof(name))
        do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
    const void *data = NULL;
    const OcSafetensorsTensor *t = q4e_find(ni, o, name, &data, NULL);
    if (!t) { oc_log_error("Q4EMISS %s", name); return OC_ERR_TENSOR; }
    if (!in_dim || !out_dim || in_dim % 128 || out_dim % 128 ||
        strcmp(t->dtype, "I16") || t->n_dims != 3 ||
        t->shape[0] != in_dim / 16 || t->shape[1] != out_dim / 16 ||
        t->shape[2] < 8 || t->shape[2] > 128 || t->shape[2] % 8)
        do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
    OcExl3Matrix *q = malloc(sizeof(*q));
    if (!q) return OC_ERR_OOM;
    memset(q, 0, sizeof(*q));
    q->input_dim = in_dim; q->output_dim = out_dim;
    q->in_block_perm = in_perm; q->out_block_perm = out_perm;
    q->packed = data;
    q->packed_size = t->data_length;
    q->bits_twice = (unsigned)(t->shape[2] / 8);
    char cname[OC_SAFETENSORS_NAME_LEN];
    const void *m = NULL;
    if (snprintf(cname, sizeof(cname), "%s.mul1", key) < (int)sizeof(cname) &&
        q4e_find(ni, o, cname, &m, NULL)) q->codebook = 2u;
    else if (snprintf(cname, sizeof(cname), "%s.mcg", key) <
                 (int)sizeof(cname) && q4e_find(ni, o, cname, &m, NULL))
        q->codebook = 1u;
    OcError e = q4e_scale(ni, o, key, true, in_dim, &q->suh);
    if (e == OC_OK) e = q4e_scale(ni, o, key, false, out_dim, &q->svh);
    if (e == OC_OK) e = oc_exl3_validate(q);
    if (e != OC_OK) { free(q); return e; }
    if (!q4e_owned(o, q)) { free(q); return OC_ERR_OOM; }
    *out = q;
    return OC_OK;
}

/* OcWeightView over a trellis matrix: rows = outputs, cols = inputs. */
static OcWeightView exl3_view(const OcExl3Matrix *q)
{
    return (OcWeightView){.data = q->packed, .rows = q->output_dim,
        .cols = q->input_dim, .exl3 = q};
}

/* Bind a plain (F16/BF16/F32) 2-D tensor as a quantized view without
 * touching its bytes (hyper-connection projections, router, shared gate). */
static OcError q4e_view(Q4eIndex *ni, OcExl3Overlay *o, const char *key,
                        size_t rows, size_t cols, OcWeightView *view)
{
    char name[OC_SAFETENSORS_NAME_LEN];
    if (snprintf(name, sizeof(name), "%s.weight", key) >= (int)sizeof(name))
        do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
    const void *data = NULL;
    const OcSafetensorsTensor *t = q4e_find(ni, o, name, &data, NULL);
    if (!t) { oc_log_error("Q4EMISS %s", name); return OC_ERR_TENSOR; }
    size_t width = !strcmp(t->dtype, "F32") ? 4 :
        (!strcmp(t->dtype, "F16") || !strcmp(t->dtype, "BF16")) ? 2 : 0;
    if (!width || !rows || !cols || t->n_dims != 2 ||
        t->shape[0] != rows || t->shape[1] != cols ||
        cols > SIZE_MAX / width || rows > SIZE_MAX / (cols * width) ||
        t->data_length != rows * cols * width)
        do { oc_log_error("Q4EFMT %s: got shape %zux%zu dims %zu len %zu want %zux%zu", name, (size_t)t->shape[0], (size_t)t->shape[1], (size_t)t->n_dims, (size_t)t->data_length, rows, cols); return OC_ERR_FORMAT; } while (0);
    if ((uintptr_t)data % width) {
        /* SafeTensors permits unaligned payload offsets; keep views aligned
         * for the CPU kernels. */
        void *copy = malloc(t->data_length);
        if (!copy) return OC_ERR_OOM;
        if (!q4e_owned(o, copy)) { free(copy); return OC_ERR_OOM; }
        memcpy(copy, data, t->data_length);
        data = copy;
    }
    *view = (OcWeightView){.data = data, .rows = rows, .cols = cols,
        .row_bytes = cols * width, .qtype = width == 4 ? OC_QUANT_F32 :
            !strcmp(t->dtype, "F16") ? OC_QUANT_F16 : OC_QUANT_BF16};
    return OC_OK;
}

/* grouped→tiled V-head relabeling at 128-block granularity: HF stores V
 * heads grouped by K head ([G0 v0 v1 v2 | G1 v0 v1 v2 | …]), ggml broadcast
 * wants them tiled ([G0v0, G1v0, …, G0v1, …]). Each V head is head_v_dim =
 * 128 dims wide, i.e. exactly one EXL3 output block, so the whole reorder
 * is one permutation of the 128-block index space. */
static uint32_t *q4e_v_perm(OcExl3Overlay *o, uint32_t n_k, uint32_t n_v)
{
    uint32_t per_k = n_v / n_k;
    uint32_t *perm = malloc(n_v * sizeof(*perm));
    if (!perm) return NULL;
    for (uint32_t g = 0; g < n_k; g++)
        for (uint32_t r = 0; r < per_k; r++)
            perm[r * n_k + g] = g * per_k + r;
    return q4e_owned(o, perm);
}

/* The qwen4exp forward expects layer-kind split, state and KV indices and
 * the layer counts — the GGUF path derives them in resolve_weights; do it
 * here for the bound layers. */
static OcError q4e_finalize(OcLlamaModel *m, uint32_t n_state, uint32_t n_kv)
{
    m->cfg.n_recurrent_layers = n_state;
    m->cfg.n_full_attention_layers = n_kv;
    if (m->tok_embeddings.rows == 0 ||
        m->tok_embeddings.cols != m->cfg.n_embd ||
        m->output.rows != m->tok_embeddings.rows ||
        m->output.cols != m->cfg.n_embd)
        return OC_ERR_MODEL;
    m->cfg.vocab_size = (uint32_t)m->tok_embeddings.rows;
    return OC_OK;
}

/* MTP draft head: a full-attention HC block (mtp.layers.0.*) plus the
 * nextn side tensors. The draft input projects [enorm(emb) ‖ hnorm(h_c)]
 * through eh_proj — HF splits that into fc_embedding + fc_hidden summed,
 * which is the same linear map as a column-concatenated fused matrix, so
 * dequantize both and fuse them into one F32 eh_proj. */
static OcError q4e_load_mtp(OcLlamaModel *m, OcExl3Overlay *o, Q4eIndex *ni,
                            uint32_t kv_index)
{
    OcLlamaConfig *c = &m->cfg;
    char key[OC_SAFETENSORS_NAME_LEN];
    OcError e;
    const size_t D = c->n_embd;
    const size_t hc_dim = (size_t)c->hc_mult * D;
    const size_t q_rows = (size_t)c->n_head * c->head_dim;
    const size_t kv_rows = (size_t)c->n_head_kv * c->head_dim;

    if (!c->nextn_predict_layers) return OC_OK;
    OcLlamaLayer *M = &m->mtp.layer;
    M->kind = OC_LLAMA_LAYER_FULL_ATTENTION;
    M->head_dim = c->head_dim;
    M->n_head_kv = c->n_head_kv;
    M->rope_dim = c->rope_dim;
    M->rope_theta = c->rope_theta;
    M->use_rope = true;
    M->kv_cache_index = kv_index;

    OcExl3Matrix *mx = NULL;
    if (snprintf(key, sizeof(key), "mtp.layers.0.self_attn.q_proj") >=
        (int)sizeof(key)) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
    if ((e = q4e_exl3_matrix(ni, o, key, D, 2 * q_rows, NULL, NULL, &mx)))
        return e;
    M->attn_q = exl3_view(mx);
    if (snprintf(key, sizeof(key), "mtp.layers.0.self_attn.k_proj") >=
        (int)sizeof(key)) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
    if ((e = q4e_exl3_matrix(ni, o, key, D, kv_rows, NULL, NULL, &mx)))
        return e;
    M->attn_k = exl3_view(mx);
    if (snprintf(key, sizeof(key), "mtp.layers.0.self_attn.v_proj") >=
        (int)sizeof(key)) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
    if ((e = q4e_exl3_matrix(ni, o, key, D, kv_rows, NULL, NULL, &mx)))
        return e;
    M->attn_v = exl3_view(mx);
    if (snprintf(key, sizeof(key), "mtp.layers.0.self_attn.o_proj") >=
        (int)sizeof(key)) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
    if ((e = q4e_exl3_matrix(ni, o, key, q_rows, D, NULL, NULL, &mx)))
        return e;
    M->attn_output = exl3_view(mx);
    if (snprintf(key, sizeof(key), "mtp.layers.0.self_attn.q_norm.weight") >=
        (int)sizeof(key)) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
    if ((e = q4e_vector(ni, o, key, c->head_dim, &M->attn_q_norm, 1.0f, NULL,
                        false))) return e;
    if (snprintf(key, sizeof(key), "mtp.layers.0.self_attn.k_norm.weight") >=
        (int)sizeof(key)) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
    if ((e = q4e_vector(ni, o, key, c->head_dim, &M->attn_k_norm, 1.0f, NULL,
                        false))) return e;

    /* HC blocks (attention + FFN sides) — same tensors as a base layer. */
    static const char *const hc_sides[2] = {"attn_hyper_connection",
                                            "mlp_hyper_connection"};
    for (size_t side = 0; side < 2; side++) {
        float **norm = side == 0 ? &M->hc_attn_norm : &M->hc_ffn_norm;
        OcWeightView *down = side == 0 ? &M->hc_attn_down : &M->hc_ffn_down;
        OcWeightView *up = side == 0 ? &M->hc_attn_up : &M->hc_ffn_up;
        OcWeightView *inject = side == 0 ? &M->hc_attn_inject
                                         : &M->hc_ffn_inject;
        if (snprintf(key, sizeof(key), "mtp.layers.0.%s.hc_norm.weight",
                     hc_sides[side]) >= (int)sizeof(key))
            do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
        if ((e = q4e_vector(ni, o, key, hc_dim, norm, 1.0f, NULL, false)))
            return e;
        if (snprintf(key, sizeof(key),
                     "mtp.layers.0.%s.input_mix_weight_down",
                     hc_sides[side]) >= (int)sizeof(key))
            do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
        if ((e = q4e_view(ni, o, key, c->hc_low_rank, hc_dim, down)))
            return e;
        if (snprintf(key, sizeof(key),
                     "mtp.layers.0.%s.input_mix_weight_up",
                     hc_sides[side]) >= (int)sizeof(key))
            do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
        if ((e = q4e_view(ni, o, key, hc_dim, c->hc_low_rank, up)))
            return e;
        if (snprintf(key, sizeof(key),
                     "mtp.layers.0.%s.block_inject_weight",
                     hc_sides[side]) >= (int)sizeof(key))
            do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
        /* Same contract as the layer path above: a malformed
         * block_inject must not silently zero the inject projection. */
        if ((e = q4e_view(ni, o, key, c->hc_mult, hc_dim, inject)))
            return e;
    }

    /* MoE: router + shared expert + per-expert EXL3. */
    if ((e = q4e_view(ni, o, "mtp.layers.0.mlp.gate", c->num_experts, D,
                      &M->ffn_gate_inp))) return e;
    if ((e = q4e_exl3_matrix(ni, o, "mtp.layers.0.mlp.shared_expert.gate_proj",
                             D, c->shared_expert_intermediate_size, NULL,
                             NULL, &mx))) return e;
    M->ffn_gate_shexp = exl3_view(mx);
    if ((e = q4e_exl3_matrix(ni, o, "mtp.layers.0.mlp.shared_expert.up_proj",
                             D, c->shared_expert_intermediate_size, NULL,
                             NULL, &mx))) return e;
    M->ffn_up_shexp = exl3_view(mx);
    if ((e = q4e_exl3_matrix(ni, o, "mtp.layers.0.mlp.shared_expert.down_proj",
                             c->shared_expert_intermediate_size, D, NULL,
                             NULL, &mx))) return e;
    M->ffn_down_shexp = exl3_view(mx);
    if ((e = q4e_view(ni, o, "mtp.layers.0.mlp.shared_expert_gate", 1, D,
                      &M->ffn_gate_inp_shexp))) return e;
    M->exl3_gate = malloc((size_t)c->num_experts * sizeof(*M->exl3_gate));
    M->exl3_up   = malloc((size_t)c->num_experts * sizeof(*M->exl3_up));
    M->exl3_down = malloc((size_t)c->num_experts * sizeof(*M->exl3_down));
    if (!M->exl3_gate || !M->exl3_up || !M->exl3_down) return OC_ERR_OOM;
    for (uint32_t ex = 0; ex < c->num_experts; ex++) {
        if (snprintf(key, sizeof(key), "mtp.layers.0.mlp.experts.%u.gate_proj",
                     ex) >= (int)sizeof(key)) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
        if ((e = q4e_exl3_matrix(ni, o, key, D,
                                 c->expert_intermediate_size, NULL, NULL,
                                 &mx))) return e;
        M->exl3_gate[ex] = *mx;
        key[strlen(key) - strlen("gate_proj")] = 0;
        strcat(key, "up_proj");
        if ((e = q4e_exl3_matrix(ni, o, key, D,
                                 c->expert_intermediate_size, NULL, NULL,
                                 &mx))) return e;
        M->exl3_up[ex] = *mx;
        key[strlen(key) - strlen("up_proj")] = 0;
        strcat(key, "down_proj");
        if ((e = q4e_exl3_matrix(ni, o, key, c->expert_intermediate_size, D,
                                 NULL, NULL, &mx))) return e;
        M->exl3_down[ex] = *mx;
    }

    /* nextn side: enorm (emb) / hnorm (hc streams) / the hc_head mixer. */
    if ((e = q4e_vector(ni, o, "mtp.pre_fc_norm_embedding.weight", D,
                        &m->mtp.enorm, 1.0f, NULL, false))) return e;
    if ((e = q4e_vector(ni, o, "mtp.pre_fc_norm_hidden.weight", hc_dim,
                        &m->mtp.hnorm, 1.0f, NULL, false))) return e;
    if ((e = q4e_vector(ni, o, "mtp.hyper_connection_mixer.hc_norm.weight",
                        hc_dim, &m->mtp.hc_head_norm, 1.0f, NULL, false)))
        return e;
    if ((e = q4e_view(ni, o, "mtp.hyper_connection_mixer.input_mix_weight_down",
                      c->hc_low_rank, hc_dim, &m->mtp.hc_head_down)))
        return e;
    if ((e = q4e_view(ni, o, "mtp.hyper_connection_mixer.input_mix_weight_up",
                      hc_dim, c->hc_low_rank, &m->mtp.hc_head_up)))
        return e;

    /* Fused eh_proj: [W_emb | W_h] over [enorm(emb) ‖ hnorm(h_c)].
     * Both fc tensors ship as trellis — dequantize, then concat columns. */
    OcExl3Matrix *fc_emb = NULL, *fc_hid = NULL;
    if ((e = q4e_exl3_matrix(ni, o, "mtp.fc_embedding", D, D, NULL, NULL,
                             &fc_emb)))
        return e;
    /* fc_emb/fc_hid stay owned by the overlay — no frees here or model
     * cleanup would double-free them. */
    if ((e = q4e_exl3_matrix(ni, o, "mtp.fc_hidden", D, D, NULL, NULL,
                             &fc_hid)))
        return e;
    float *emb = malloc(D * D * sizeof(float));
    float *hid = malloc(D * D * sizeof(float));
    float *eh = malloc(D * 2 * D * sizeof(float));
    if (!emb || !hid || !eh) {
        free(emb); free(hid); free(eh);
        return OC_ERR_OOM;
    }
    e = oc_exl3_dequant_f32(fc_emb, emb);
    if (e == OC_OK)
        e = oc_exl3_dequant_f32(fc_hid, hid);
    if (e != OC_OK) { free(emb); free(hid); free(eh); return e; }
    /* eh_proj is row-major D×2D = [W_emb | W_h]: the two D×D halves must
     * be interleaved column-wise per output row, not concatenated. */
    for (size_t r = 0; r < D; r++) {
        memcpy(eh + r * 2 * D, emb + r * D, D * sizeof(float));
        memcpy(eh + r * 2 * D + D, hid + r * D, D * sizeof(float));
    }
    free(emb); free(hid);
    if (!q4e_owned(o, eh)) { free(eh); return OC_ERR_OOM; }
    m->mtp.eh_proj = (OcWeightView){.data = (const uint8_t *)eh, .rows = D, .cols = 2 * D,
        .row_bytes = 2 * D * sizeof(float), .qtype = OC_QUANT_F32};

    m->mtp.present = true;
    oc_log(OC_LOG_INFO, "qwen4exp: MTP draft head loaded (draft k=%u)",
           OC_MTP_DEFAULT_DRAFT);
    return OC_OK;
}

static OcError q4e_load_native_body(OcLlamaModel *m, OcExl3Overlay *o,
                                    Q4eIndex ni)
{
    OcLlamaConfig *c = &m->cfg;
    char key[OC_SAFETENSORS_NAME_LEN];
    OcError e = OC_OK;

    const size_t D = c->n_embd;
    const size_t hc_dim = (size_t)c->hc_mult * D;
    const size_t q_rows = (size_t)c->n_head * c->head_dim;
    const size_t kv_rows = (size_t)c->n_head_kv * c->head_dim;
    const size_t i_size = c->expert_intermediate_size;
    const size_t key_dim = (size_t)c->ssm_group_count * c->ssm_state_size;
    const size_t conv_dim = 2 * key_dim + c->ssm_inner_size;
    const uint32_t n_exp = c->num_experts;

    /* V-head permutation for the linear-attention layers. Owned by the
     * overlay so every error path below frees it. */
    /* q4e_v_perm registers its buffer with q4e_owned on success. */
    uint32_t *vperm = q4e_v_perm(o, c->ssm_group_count, c->ssm_value_heads);
    if (!vperm) return OC_ERR_OOM;
    /* The trellis row/block perms below treat one V head as exactly one
     * 128 block (true for every shipped qwen4exp checkpoint). */
    if (c->ssm_value_heads == 0 ||
        c->ssm_inner_size / c->ssm_value_heads != 128)
        return OC_ERR_MODEL;

    /* Heap-allocate every EXL3 matrix so OcWeightView.exl3 pointers stay
     * stable; the pointer goes on the overlay's owned list. */
    OcExl3Matrix *mx = NULL;

    /* Token embedding (BF16) and the 8-bit lm_head. */
    e = q4e_view(&ni, o, "model.language_model.embed_tokens",
                 c->vocab_size, D, &m->tok_embeddings);
    if (e == OC_OK)
        e = q4e_exl3_matrix(&ni, o, "lm_head", D, c->vocab_size, NULL, NULL,
                            &mx);
    if (e != OC_OK) return e;
    m->output = exl3_view(mx);
    c->tied_embeddings = false;

    /* The final hyper-connection head replaces output_norm. */
    if ((e = q4e_vector(&ni, o,
                        "model.language_model.hyper_connection_mixer.hc_norm.weight",
                        hc_dim, &m->output_hc_norm, 1.0f, NULL, false)))
        return e;
    if ((e = q4e_view(&ni, o,
                      "model.language_model.hyper_connection_mixer.input_mix_weight_down",
                      c->hc_low_rank, hc_dim, &m->output_hc_down)))
        return e;
    if ((e = q4e_view(&ni, o,
                      "model.language_model.hyper_connection_mixer.input_mix_weight_up",
                      hc_dim, c->hc_low_rank, &m->output_hc_up)))
        return e;

    /* Per-expert EXL3 tables: one trellis matrix per (layer, expert, proj). */
    for (uint32_t l = 0; l < c->n_layer; l++) {
        OcLlamaLayer *L = &m->layers[l];
        L->exl3_gate = malloc((size_t)n_exp * sizeof(*L->exl3_gate));
        L->exl3_up   = malloc((size_t)n_exp * sizeof(*L->exl3_up));
        L->exl3_down = malloc((size_t)n_exp * sizeof(*L->exl3_down));
        if (!L->exl3_gate || !L->exl3_up || !L->exl3_down) return OC_ERR_OOM;
        for (uint32_t ex = 0; ex < n_exp; ex++) {
            if (snprintf(key, sizeof(key),
                         "model.language_model.layers.%u.mlp.experts.%u.gate_proj",
                         l, ex) >= (int)sizeof(key)) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
            if ((e = q4e_exl3_matrix(&ni, o, key, D, i_size, NULL, NULL,
                                     &mx))) return e;
            L->exl3_gate[ex] = *mx;
            key[strlen(key) - strlen("gate_proj")] = 0;
            strcat(key, "up_proj");
            if ((e = q4e_exl3_matrix(&ni, o, key, D, i_size, NULL, NULL,
                                     &mx))) return e;
            L->exl3_up[ex] = *mx;
            key[strlen(key) - strlen("up_proj")] = 0;
            strcat(key, "down_proj");
            if ((e = q4e_exl3_matrix(&ni, o, key, i_size, D, NULL, NULL,
                                     &mx))) return e;
            L->exl3_down[ex] = *mx;
        }
    }
    oc_log(OC_LOG_INFO, "qwen4exp: %u layers x %u experts x 3 trellis "
           "matrices bound", c->n_layer, n_exp);

    /* Layer stack. */
    uint32_t n_state = 0, n_kv = 0;
    for (uint32_t l = 0; l < c->n_layer; l++) {
        OcLlamaLayer *L = &m->layers[l];
        L->head_dim = c->head_dim;
        L->n_head_kv = c->n_head_kv;
        L->rope_dim = c->rope_dim;
        L->rope_theta = c->rope_theta;
        L->use_rope = true;

        /* Router + shared expert + its gate (every layer). */
        if (snprintf(key, sizeof(key),
                     "model.language_model.layers.%u.mlp.gate", l) >=
            (int)sizeof(key)) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
        if ((e = q4e_view(&ni, o, key, n_exp, D, &L->ffn_gate_inp)))
            return e;
        if (snprintf(key, sizeof(key),
                     "model.language_model.layers.%u.mlp.shared_expert.gate_proj",
                     l) >= (int)sizeof(key)) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
        if ((e = q4e_exl3_matrix(&ni, o, key, D,
                                 c->shared_expert_intermediate_size, NULL,
                                 NULL, &mx))) return e;
        L->ffn_gate_shexp = exl3_view(mx);
        key[strlen(key) - strlen("gate_proj")] = 0;
        strcat(key, "up_proj");
        if ((e = q4e_exl3_matrix(&ni, o, key, D,
                                 c->shared_expert_intermediate_size, NULL,
                                 NULL, &mx))) return e;
        L->ffn_up_shexp = exl3_view(mx);
        key[strlen(key) - strlen("up_proj")] = 0;
        strcat(key, "down_proj");
        if ((e = q4e_exl3_matrix(&ni, o, key,
                                 c->shared_expert_intermediate_size, D,
                                 NULL, NULL, &mx))) return e;
        L->ffn_down_shexp = exl3_view(mx);
        if (snprintf(key, sizeof(key),
                     "model.language_model.layers.%u.mlp.shared_expert_gate",
                     l) >= (int)sizeof(key)) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
        if ((e = q4e_view(&ni, o, key, 1, D, &L->ffn_gate_inp_shexp)))
            return e;

        /* Hyper-connection blocks, attention and FFN sides. Zero-centred
         * gammas fold to (1 + w) exactly as the GGUF converter does. */
        static const char *const hc_sides[2] = {"attn_hyper_connection",
                                                "mlp_hyper_connection"};
        for (size_t side = 0; side < 2; side++) {
            float **norm = side == 0 ? &L->hc_attn_norm : &L->hc_ffn_norm;
            OcWeightView *down = side == 0 ? &L->hc_attn_down : &L->hc_ffn_down;
            OcWeightView *up = side == 0 ? &L->hc_attn_up : &L->hc_ffn_up;
            OcWeightView *inject = side == 0 ? &L->hc_attn_inject
                                             : &L->hc_ffn_inject;
            if (snprintf(key, sizeof(key),
                         "model.language_model.layers.%u.%s.hc_norm.weight",
                         l, hc_sides[side]) >= (int)sizeof(key))
                do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
            if ((e = q4e_vector(&ni, o, key, hc_dim, norm, 1.0f, NULL,
                                false))) return e;
            if (snprintf(key, sizeof(key),
                         "model.language_model.layers.%u.%s.input_mix_weight_down",
                         l, hc_sides[side]) >= (int)sizeof(key))
                do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
            if ((e = q4e_view(&ni, o, key, c->hc_low_rank, hc_dim, down)))
                return e;
            if (snprintf(key, sizeof(key),
                         "model.language_model.layers.%u.%s.input_mix_weight_up",
                         l, hc_sides[side]) >= (int)sizeof(key))
                do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
            if ((e = q4e_view(&ni, o, key, hc_dim, c->hc_low_rank, up)))
                return e;
            if (snprintf(key, sizeof(key),
                         "model.language_model.layers.%u.%s.block_inject_weight",
                         l, hc_sides[side]) >= (int)sizeof(key))
                do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
            /* block_inject is present on every layer of this checkpoint;
             * discarding a view failure would silently drop the inject
             * projection, so propagate it. */
            if ((e = q4e_view(&ni, o, key, c->hc_mult, hc_dim, inject)))
                return e;
        }

        if (c->recurrent_layers && c->recurrent_layers[l]) {
            /* Gated delta-net layer. */
            L->kind = OC_LLAMA_LAYER_QWEN35_RECURRENT;
            L->state_index = n_state++;
            /* out_perm: blocks 0..31 are Q/K (identity); the 48 V blocks
             * relabel grouped→tiled. */
            uint32_t *qkv_perm = malloc((conv_dim / 128) * sizeof(*qkv_perm));
            if (!qkv_perm) return OC_ERR_OOM;
            for (size_t b = 0; b < conv_dim / 128; b++)
                qkv_perm[b] = b < 2 * key_dim / 128 ? (uint32_t)b
                              : 2 * key_dim / 128 + vperm[b - 2 * key_dim / 128];
            if (!q4e_owned(o, qkv_perm)) { free(qkv_perm); return OC_ERR_OOM; }
            if (snprintf(key, sizeof(key),
                         "model.language_model.layers.%u.linear_attn.in_proj_qkv",
                         l) >= (int)sizeof(key)) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
            if ((e = q4e_exl3_matrix(&ni, o, key, D, conv_dim, NULL,
                                     qkv_perm, &mx))) return e;
            L->attn_qkv = exl3_view(mx);
            if (snprintf(key, sizeof(key),
                         "model.language_model.layers.%u.linear_attn.in_proj_z",
                         l) >= (int)sizeof(key)) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
            if ((e = q4e_exl3_matrix(&ni, o, key, D, c->ssm_inner_size, NULL,
                                     vperm, &mx))) return e;
            L->attn_gate = exl3_view(mx);
            if (snprintf(key, sizeof(key),
                         "model.language_model.layers.%u.linear_attn.out_proj",
                         l) >= (int)sizeof(key)) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
            if ((e = q4e_exl3_matrix(&ni, o, key, c->ssm_inner_size, D,
                                     vperm, NULL, &mx))) return e;
            L->ssm_out = exl3_view(mx);

            /* dt_bias and A: HF stores 48 grouped-ordered heads; the forward
             * wants tiled order and A = -exp(A_log). */
            if (snprintf(key, sizeof(key),
                         "model.language_model.layers.%u.linear_attn.dt_bias",
                         l) >= (int)sizeof(key)) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
            if ((e = q4e_vector(&ni, o, key, c->ssm_value_heads,
                                &L->ssm_dt_f32, 0.0f, vperm, false)))
                return e;
            L->ssm_dt_bias = (OcWeightView){.data = (const uint8_t *)L->ssm_dt_f32, .rows = 1,
                .cols = c->ssm_value_heads,
                .row_bytes = 4 * (size_t)c->ssm_value_heads,
                .qtype = OC_QUANT_F32};
            if (snprintf(key, sizeof(key),
                         "model.language_model.layers.%u.linear_attn.A_log",
                         l) >= (int)sizeof(key)) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
            if ((e = q4e_vector(&ni, o, key, c->ssm_value_heads,
                                &L->ssm_a_f32, 0.0f, vperm, true)))
                return e;
            L->ssm_a = (OcWeightView){.data = (const uint8_t *)L->ssm_a_f32, .rows = 1,
                .cols = c->ssm_value_heads,
                .row_bytes = 4 * (size_t)c->ssm_value_heads,
                .qtype = OC_QUANT_F32};

            /* Conv kernel: [conv_dim, 1, kernel] BF16 → F32 with the V
             * channels relabeled. */
            if (snprintf(key, sizeof(key),
                         "model.language_model.layers.%u.linear_attn.conv1d.weight",
                         l) >= (int)sizeof(key)) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
            const void *cdata = NULL;
            const OcSafetensorsTensor *ct = q4e_find(&ni, o, key, &cdata, NULL);
            if (!ct || strcmp(ct->dtype, "BF16") || ct->n_dims != 3 ||
                ct->shape[0] != conv_dim || ct->shape[1] != 1 ||
                ct->shape[2] != c->ssm_conv_kernel ||
                ct->data_length != conv_dim * c->ssm_conv_kernel * 2)
                do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
            float *cw = malloc(conv_dim * c->ssm_conv_kernel * sizeof(float));
            if (!cw) return OC_ERR_OOM;
            const uint16_t *cr = cdata;
            /* vperm is a per-HEAD relabel (length ssm_value_heads); V
             * channels advance value_head_dim rows per head. */
            const size_t vhd = c->ssm_inner_size / c->ssm_value_heads;
            for (size_t r = 0; r < conv_dim; r++) {
                size_t s;
                if (r >= 2 * key_dim) {
                    size_t off = r - 2 * key_dim;
                    s = 2 * key_dim + (size_t)vperm[off / vhd] * vhd
                      + off % vhd;
                } else {
                    s = r;
                }
                for (size_t kk = 0; kk < c->ssm_conv_kernel; kk++) {
                    uint32_t bits =
                        (uint32_t)cr[s * c->ssm_conv_kernel + kk] << 16;
                    float x;
                    memcpy(&x, &bits, 4);
                    cw[r * c->ssm_conv_kernel + kk] = x;
                    if (!isfinite(x)) { free(cw); do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0); }
                }
            }
            if (!q4e_owned(o, cw)) { free(cw); return OC_ERR_OOM; }
            L->ssm_conv1d = (OcWeightView){.data = (const uint8_t *)cw, .rows = conv_dim,
                .cols = c->ssm_conv_kernel,
                .row_bytes = 4 * c->ssm_conv_kernel, .qtype = OC_QUANT_F32};

            /* in_proj_a → ssm_alpha, in_proj_b → ssm_beta ([48, D] each,
             * grouped rows relabeled). The forward requires f32. */
            if (snprintf(key, sizeof(key),
                         "model.language_model.layers.%u.linear_attn.in_proj_a",
                         l) >= (int)sizeof(key)) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
            if ((e = q4e_matrix_f32(&ni, o, key, c->ssm_value_heads, D,
                                    vperm, &L->ssm_alpha))) return e;
            if (snprintf(key, sizeof(key),
                         "model.language_model.layers.%u.linear_attn.in_proj_b",
                         l) >= (int)sizeof(key)) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
            if ((e = q4e_matrix_f32(&ni, o, key, c->ssm_value_heads, D,
                                    vperm, &L->ssm_beta))) return e;
            /* Per-head delta state norm over the value head (no fold —
             * the converter keeps this one zero-centred). */
            if (snprintf(key, sizeof(key),
                         "model.language_model.layers.%u.linear_attn.norm.weight",
                         l) >= (int)sizeof(key)) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
            if ((e = q4e_vector(&ni, o, key, c->ssm_state_size,
                                &L->ssm_norm_f32, 0.0f, NULL, false)))
                return e;
            L->ssm_norm = (OcWeightView){.data = (const uint8_t *)L->ssm_norm_f32, .rows = 1,
                .cols = c->ssm_state_size,
                .row_bytes = 4 * c->ssm_state_size, .qtype = OC_QUANT_F32};
        } else {
            /* Full attention: HF q_proj already carries [q | gate] per head
             * (svh = 2 * n_head * head_dim) — bind it directly. */
            L->kind = OC_LLAMA_LAYER_FULL_ATTENTION;
            L->kv_cache_index = n_kv++;
            if (snprintf(key, sizeof(key),
                         "model.language_model.layers.%u.self_attn.q_proj",
                         l) >= (int)sizeof(key)) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
            if ((e = q4e_exl3_matrix(&ni, o, key, D, 2 * q_rows, NULL, NULL,
                                     &mx))) return e;
            L->attn_q = exl3_view(mx);
            if (snprintf(key, sizeof(key),
                         "model.language_model.layers.%u.self_attn.k_proj",
                         l) >= (int)sizeof(key)) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
            if ((e = q4e_exl3_matrix(&ni, o, key, D, kv_rows, NULL, NULL,
                                     &mx))) return e;
            L->attn_k = exl3_view(mx);
            if (snprintf(key, sizeof(key),
                         "model.language_model.layers.%u.self_attn.v_proj",
                         l) >= (int)sizeof(key)) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
            if ((e = q4e_exl3_matrix(&ni, o, key, D, kv_rows, NULL, NULL,
                                     &mx))) return e;
            L->attn_v = exl3_view(mx);
            if (snprintf(key, sizeof(key),
                         "model.language_model.layers.%u.self_attn.o_proj",
                         l) >= (int)sizeof(key)) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
            if ((e = q4e_exl3_matrix(&ni, o, key, q_rows, D, NULL, NULL,
                                     &mx))) return e;
            L->attn_output = exl3_view(mx);
            /* Per-head Q/K RMSNorms, zero-centred → fold to (1 + w). */
            if (snprintf(key, sizeof(key),
                         "model.language_model.layers.%u.self_attn.q_norm.weight",
                         l) >= (int)sizeof(key)) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
            if ((e = q4e_vector(&ni, o, key, c->head_dim, &L->attn_q_norm,
                                1.0f, NULL, false))) return e;
            if (snprintf(key, sizeof(key),
                         "model.language_model.layers.%u.self_attn.k_norm.weight",
                         l) >= (int)sizeof(key)) do { oc_log_error("Q4EFMT %s:%d", __func__, __LINE__); return OC_ERR_FORMAT; } while (0);
            if ((e = q4e_vector(&ni, o, key, c->head_dim, &L->attn_k_norm,
                                1.0f, NULL, false))) return e;
        }
    }

    /* MTP draft head. */
    e = q4e_load_mtp(m, o, &ni, n_kv);
    if (e != OC_OK) return e;

    /* PLE stays unbound: the checkpoint packs the n-gram table in the
     * custom exl3_ngram_trellis format, which is not decodable here yet. */
    if (c->ple_enabled) {
        c->ple_enabled = false;
        oc_log(OC_LOG_WARN, "qwen4exp: the PLE n-gram table ships in the "
               "custom exl3_ngram_trellis format, which is not decodable "
               "here yet — running without it degrades output quality");
    }

    e = q4e_finalize(m, n_state, n_kv);
    if (e != OC_OK) return e;

    /* Everything not consumed must be an explicitly skippable tensor. */
    for (size_t i = 0; i < ni.n; i++) {
        if (ni.used[i]) continue;
        const char *nme = ni.items[i].name;
        if (!strncmp(nme, "model.visual.", 13)) continue;   /* vision tower */
        if (strstr(nme, ".ple.")) continue;                 /* PLE block    */
        if (strstr(nme, "ngram_embedding.")) continue;      /* PLE table    */
        if (strstr(nme, "self_attn.indexer.")) {
            /* QSA indexer tensors mean dense attention is only exact
             * below indexer_top_k; beyond it the logits attend the wrong
             * keys. Refuse rather than silently run inexact. */
            oc_log(OC_LOG_ERROR,
                   "qwen4exp: QSA indexer tensor %s is unsupported — "
                   "checkpoint requires sparse attention", nme);
            return OC_ERR_MODEL;
        }
        oc_log(OC_LOG_ERROR, "qwen4exp: unconsumed checkpoint tensor %s",
               nme);
        return OC_ERR_MODEL;
    }
    oc_log(OC_LOG_INFO, "qwen4exp native EXL3 checkpoint loaded: "
           "%u layers (%u recurrent, %u full), %u experts, ctx %u",
           c->n_layer, n_state, n_kv, n_exp, c->n_ctx);
    return OC_OK;
}

/* Wrapper: owns the name index so every body error path releases it. */
static OcError load_native_qwen4exp(OcLlamaModel *m, OcExl3Overlay *o)
{
    Q4eIndex ni = {0};
    OcError e = q4e_index_build(o, &ni);
    if (e != OC_OK) {
        free(ni.items);
        free(ni.used);
        return e;
    }
    e = q4e_load_native_body(m, o, ni);
    free(ni.items);
    free(ni.used);
    return e;
}

OcError oc_llama_load_exl3_native(const char *directory, OcLlamaModel *out)
{
    if (!directory || !out) return OC_ERR_INVALID_ARG;
    memset(out,0,sizeof(*out));
    OcError e = oc_hf_metadata_load(directory,&out->cfg,&out->arch,&out->gguf.unified);
    if (e != OC_OK) goto fail;
    OcLlamaConfig *c = &out->cfg;
    const bool q4e = c->is_qwen4exp;
    if ((out->arch != OC_ARCH_LLAMA && out->arch != OC_ARCH_MISTRAL &&
         out->arch != OC_ARCH_QWEN) ||
        (!q4e && (c->num_experts || c->uses_mla || c->is_qwen35 || c->is_k2 ||
                  c->nextn_predict_layers))) {
        e = OC_ERR_MODEL; goto fail;
    }
    out->layers = calloc(c->n_layer,sizeof(*out->layers));
    out->exl3_overlay = calloc(1,sizeof(*out->exl3_overlay));
    if (!out->layers || !out->exl3_overlay) { e=OC_ERR_OOM; goto fail; }
    OcExl3Overlay *o = out->exl3_overlay;
    e = open_files(o,directory);
    if (e != OC_OK) goto fail;
    if (q4e) {
        e = load_native_qwen4exp(out, o);
        if (e != OC_OK) goto fail;
        return OC_OK;
    }
    for(size_t f=0;f<o->n_files;f++) for(size_t i=0;i<o->files[f].n_tensors;i++)
        if (!native_known_tensor(out,o->files[f].tensors[i].name)) {
            e=OC_ERR_MODEL; goto fail;
        }
    e = native_matrix(o,"model.embed_tokens",c->vocab_size,c->n_embd,false,&out->tok_embeddings);
    if (e != OC_OK) goto fail;
    e = native_vector(o,"model.norm.weight",c->n_embd,false,&out->final_norm);
    if (e != OC_OK) goto fail;
    for (uint32_t i=0;i<c->n_layer;i++) {
        OcLlamaLayer *l=&out->layers[i];
        l->head_dim=c->head_dim; l->n_head_kv=c->n_head_kv;
        l->rope_dim=c->rope_dim; l->rope_theta=c->rope_theta;
        l->use_rope=true; l->kv_cache_index=i;
        l->sliding_window=c->sliding_window;
        const char *names[]={"self_attn.q_proj","self_attn.k_proj","self_attn.v_proj",
            "self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"};
        OcWeightView *views[]={&l->attn_q,&l->attn_k,&l->attn_v,&l->attn_output,
            &l->ffn_gate,&l->ffn_up,&l->ffn_down};
        size_t q=(size_t)c->n_head*c->head_dim, kv=(size_t)c->n_head_kv*c->kv_head_dim;
        size_t rows[]={q,kv,kv,c->n_embd,c->n_ff,c->n_ff,c->n_embd};
        size_t cols[]={c->n_embd,c->n_embd,c->n_embd,q,c->n_embd,c->n_embd,c->n_ff};
        char key[128];
        for(size_t j=0;j<7;j++) {
            snprintf(key,sizeof(key),"model.layers.%u.%s",i,names[j]);
            e=native_matrix(o,key,rows[j],cols[j],true,views[j]);
            if(e != OC_OK) goto fail;
        }
        const char *vectors[]={"input_layernorm.weight","post_attention_layernorm.weight",
            "self_attn.q_proj.bias","self_attn.k_proj.bias","self_attn.v_proj.bias",
            "self_attn.q_norm.weight","self_attn.k_norm.weight"};
        float **values[]={&l->attn_norm,&l->ffn_norm,&l->attn_q_bias,&l->attn_k_bias,
            &l->attn_v_bias,&l->attn_q_norm,&l->attn_k_norm};
        size_t dims[]={c->n_embd,c->n_embd,q,kv,kv,c->head_dim,c->head_dim};
        for(size_t j=0;j<7;j++) {
            snprintf(key,sizeof(key),"model.layers.%u.%s",i,vectors[j]);
            e=native_vector(o,key,dims[j],j>=2,values[j]);
            if(e != OC_OK) goto fail;
        }
    }
    e=native_matrix(o,"lm_head",c->vocab_size,c->n_embd,true,&out->output);
    if(e == OC_ERR_TENSOR && c->tied_embeddings) { out->output=out->tok_embeddings; e=OC_OK; }
    else if(e == OC_OK) c->tied_embeddings=false;
    if(e != OC_OK) goto fail;
    e=attach_exl3(out,o,true);
    if(e != OC_OK) goto fail;
    return OC_OK;
fail:
    oc_llama_free(out);
    return e;
}
