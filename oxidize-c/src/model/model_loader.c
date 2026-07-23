/*
 * model_loader.c — Universal model loader implementation.
 *
 * Uses the existing GGUF parser to load and inspect model files.
 */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/model_loader.h"
#include "oxidize/gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static void copy_str(char *dst, size_t cap, const char *src)
{
    if (!dst || cap == 0 || !src) return;
    size_t n = strlen(src);
    if (n >= cap) n = cap - 1;
    memcpy(dst, src, n);
    dst[n] = '\0';
}

OcError oc_model_loader_init(OcModelLoader *loader, const char *path)
{
    if (!loader || !path) return OC_ERR_INVALID_ARG;
    memset(loader, 0, sizeof(*loader));
    copy_str(loader->path, sizeof(loader->path), path);
    loader->arch = OC_ARCH_UNKNOWN;
    return OC_OK;
}

OcError oc_model_loader_load(OcModelLoader *loader)
{
    if (!loader || loader->path[0] == '\0') return OC_ERR_INVALID_ARG;
    if (loader->loaded) return OC_OK;

    /* Open the GGUF file. */
    OcGgufFile gguf;
    OcError e = oc_gguf_open(loader->path, &gguf);
    if (e != OC_OK) return e;

    /* Get file size. */
    struct stat st;
    if (stat(loader->path, &st) == 0)
        loader->file_size = st.st_size;

    /* Extract architecture. */
    const char *arch_str = NULL;
    size_t arch_len = 0;
    if (oc_gguf_metadata_get_str(&gguf, "general.architecture", &arch_str, &arch_len)) {
        loader->arch = oc_model_arch_from_str(arch_str);
        copy_str(loader->metadata.arch, sizeof(loader->metadata.arch), arch_str);
    }

    /* Extract common metadata. */
    oc_gguf_metadata_get_u32(&gguf, "llama.block_count", &loader->metadata.n_layers);
    oc_gguf_metadata_get_u32(&gguf, "llama.embedding_length", &loader->metadata.n_embd);
    oc_gguf_metadata_get_u32(&gguf, "llama.attention.head_count", &loader->metadata.n_heads);
    oc_gguf_metadata_get_u32(&gguf, "llama.attention.head_count_kv", &loader->metadata.n_kv_heads);
    oc_gguf_metadata_get_u32(&gguf, "llama.context_length", &loader->metadata.context_length);
    oc_gguf_metadata_get_u32(&gguf, "tokenizer.ggml.tokens", &loader->metadata.vocab_size);

    /* Try Qwen-specific keys. */
    if (loader->metadata.n_layers == 0)
        oc_gguf_metadata_get_u32(&gguf, "qwen.block_count", &loader->metadata.n_layers);
    if (loader->metadata.n_embd == 0)
        oc_gguf_metadata_get_u32(&gguf, "qwen.embedding_length", &loader->metadata.n_embd);

    /* Try GPT-2 keys. */
    if (loader->metadata.n_layers == 0)
        oc_gguf_metadata_get_u32(&gguf, "gpt2.block_count", &loader->metadata.n_layers);
    if (loader->metadata.n_embd == 0)
        oc_gguf_metadata_get_u32(&gguf, "gpt2.embedding_length", &loader->metadata.n_embd);

    /* Try Falcon keys. */
    if (loader->metadata.n_layers == 0)
        oc_gguf_metadata_get_u32(&gguf, "falcon.block_count", &loader->metadata.n_layers);

    /* Compute hidden_dim (n_embd * 4 for Llama family). */
    loader->metadata.hidden_dim = loader->metadata.n_embd * 4;

    /* Compute param count estimate. */
    if (loader->metadata.n_layers > 0 && loader->metadata.n_embd > 0) {
        /* Rough estimate: 12 * n_layers * n_embd^2 for a standard transformer. */
        loader->metadata.n_params = (uint64_t)loader->metadata.n_layers *
                                     loader->metadata.n_embd * loader->metadata.n_embd * 12;
    }

    /* Copy tensor info. */
    size_t n = gguf.tensor_count;
    if (n > OC_LOADER_MAX_TENSORS) n = OC_LOADER_MAX_TENSORS;
    for (size_t i = 0; i < n; i++) {
        OcLoadedTensor *t = &loader->tensors[i];
        copy_str(t->name, sizeof(t->name), gguf.tensors[i].name);
        t->n_dims = gguf.tensors[i].n_dims;
        for (uint32_t d = 0; d < t->n_dims && d < 4; d++)
            t->dims[d] = gguf.tensors[i].dims[d];
        t->type = gguf.tensors[i].ggml_type;
        t->offset = gguf.tensors[i].absolute_offset;
        t->size = gguf.tensors[i].relative_offset;
    }
    loader->n_tensors = n;

    oc_gguf_free(&gguf);
    loader->loaded = true;
    return OC_OK;
}

OcError oc_model_loader_get_tensor(const OcModelLoader *loader,
                                   const char *name,
                                   const OcLoadedTensor **out)
{
    if (!loader || !name || !out) return OC_ERR_INVALID_ARG;
    for (size_t i = 0; i < loader->n_tensors; i++) {
        if (strcmp(loader->tensors[i].name, name) == 0) {
            *out = &loader->tensors[i];
            return OC_OK;
        }
    }
    return OC_ERR_MODEL;
}

OcError oc_model_loader_list_tensors(const OcModelLoader *loader,
                                    const OcLoadedTensor **out_array,
                                    size_t *out_count)
{
    if (!loader || !out_array || !out_count) return OC_ERR_INVALID_ARG;
    *out_array = loader->tensors;
    *out_count = loader->n_tensors;
    return OC_OK;
}

/* oc_model_arch_name and oc_model_arch_from_str are in model.h */

uint64_t oc_model_loader_param_count(const OcModelLoader *loader)
{
    if (!loader) return 0;
    return loader->metadata.n_params;
}

void oc_model_loader_free(OcModelLoader *loader)
{
    if (!loader) return;
    memset(loader, 0, sizeof(*loader));
}
