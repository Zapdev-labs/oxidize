#include "oxidize/exl3_overlay.h"
#include "oxidize/exl3.h"
#include "oxidize/hf_metadata.h"
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
        return OC_ERR_FORMAT;
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
        if (!isfinite(v[i])) { free(v); return OC_ERR_FORMAT; }
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
        if (!packed || t) return OC_ERR_FORMAT;
        *view = (OcWeightView){.data=trellis, .rows=rows, .cols=cols};
        return OC_OK;
    }
    if (!t) return OC_ERR_TENSOR;
    size_t width = !strcmp(t->dtype,"F32") ? 4 :
        (!strcmp(t->dtype,"F16") || !strcmp(t->dtype,"BF16")) ? 2 : 0;
    if (!width || !rows || !cols || cols > SIZE_MAX / width ||
        rows > SIZE_MAX / (cols * width) || t->n_dims != 2 ||
        t->shape[0] != rows || t->shape[1] != cols ||
        t->data_length != rows * cols * width) return OC_ERR_FORMAT;
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

OcError oc_llama_load_exl3_native(const char *directory, OcLlamaModel *out)
{
    if (!directory || !out) return OC_ERR_INVALID_ARG;
    memset(out,0,sizeof(*out));
    OcError e = oc_hf_metadata_load(directory,&out->cfg,&out->arch,&out->gguf.unified);
    if (e != OC_OK) goto fail;
    OcLlamaConfig *c = &out->cfg;
    if ((out->arch != OC_ARCH_LLAMA && out->arch != OC_ARCH_MISTRAL &&
         out->arch != OC_ARCH_QWEN) || c->num_experts || c->uses_mla ||
        c->is_qwen35 || c->is_k2 || c->is_qwen4exp || c->nextn_predict_layers) {
        e = OC_ERR_MODEL; goto fail;
    }
    out->layers = calloc(c->n_layer,sizeof(*out->layers));
    out->exl3_overlay = calloc(1,sizeof(*out->exl3_overlay));
    if (!out->layers || !out->exl3_overlay) { e=OC_ERR_OOM; goto fail; }
    OcExl3Overlay *o = out->exl3_overlay;
    e = open_files(o,directory);
    if (e != OC_OK) goto fail;
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
