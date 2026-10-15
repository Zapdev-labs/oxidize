#include "oxidize/exl3_overlay.h"
#include "oxidize/exl3.h"
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
};

void oc_exl3_overlay_free(OcExl3Overlay *o)
{
    if (!o) return;
    for (size_t i = 0; i < o->n_entries; i++) {
        free((void *)o->entries[i].matrix.suh);
        free((void *)o->entries[i].matrix.svh);
    }
    for (size_t i = 0; i < o->n_files; i++) oc_safetensors_close(&o->files[i]);
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
    if ((h != NULL) == (p != NULL)) return OC_ERR_FORMAT;
    const OcSafetensorsTensor *t = h ? h : p;
    size_t count = h ? dim : dim / 16;
    if (t->n_dims != 1 || t->shape[0] != count ||
        t->data_length != count * 2 || strcmp(t->dtype, h ? "F16" : "I16"))
        return OC_ERR_FORMAT;
    float *v = malloc(dim * sizeof(*v));
    if (!v) return OC_ERR_OOM;
    const uint8_t *raw = h ? a : b;
    for (size_t i = 0; i < dim; i++) {
        v[i] = h ? oc_oxk_f16_le_to_f32(raw + i * 2)
                 : ((raw[i / 8] >> (i % 8)) & 1) ? -1.0f : 1.0f;
        if (!isfinite(v[i])) { free(v); return OC_ERR_FORMAT; }
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
    if (e != OC_OK) goto fail;
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
            if ((mcg && mul1) || companion(o, key, "bias", &data) ||
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
    oc_exl3_overlay_free(o);
    return e;
}
