/*
 * model_loader.c — Universal model loader implementation.
 *
 * Uses the existing GGUF parser to load and inspect model files.
 */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/model_loader.h"
#include "oxidize/gguf.h"
#include "oxidize/quant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void copy_str(char *dst, size_t cap, const char *src)
{
    if (!dst || cap == 0 || !src) return;
    size_t n = strlen(src);
    if (n >= cap) n = cap - 1;
    memcpy(dst, src, n);
    dst[n] = '\0';
}

static bool tensor_byte_size(const OcGgufTensorInfo *info, uint64_t *out)
{
    if (!info || !out || info->n_dims == 0 ||
        info->n_dims > OC_GGUF_MAX_DIMS) return false;

    uint64_t elements = 1;
    for (uint32_t i = 0; i < info->n_dims; i++) {
        if (info->dims[i] != 0 && elements > UINT64_MAX / info->dims[i])
            return false;
        elements *= info->dims[i];
    }

    OcGgufQuantizationType type =
        oc_quant_type_from_ggml_id(info->ggml_type);
    OcQuantBlockLayout layout = oc_quant_block_size(type);
    if (layout.elements_per_block == 0 || layout.bytes_per_block == 0 ||
        elements % layout.elements_per_block != 0) return false;

    uint64_t blocks = elements / layout.elements_per_block;
    if (blocks > UINT64_MAX / layout.bytes_per_block) return false;
    *out = blocks * layout.bytes_per_block;
    return true;
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

    OcError e = oc_gguf_map_open(loader->path, &loader->mapped);
    if (e != OC_OK) return e;
    const OcGgufFile *gguf = &loader->mapped.unified;

    loader->file_size = oc_gguf_map_total_bytes(&loader->mapped);

    /* Extract architecture. */
    const char *arch_str = NULL;
    size_t arch_len = 0;
    if (oc_gguf_metadata_get_str(gguf, "general.architecture", &arch_str, &arch_len)) {
        loader->arch = oc_model_arch_from_str(arch_str);
        copy_str(loader->metadata.arch, sizeof(loader->metadata.arch), arch_str);
    }

    /* Extract common metadata. */
    oc_gguf_metadata_get_u32(gguf, "llama.block_count", &loader->metadata.n_layers);
    oc_gguf_metadata_get_u32(gguf, "llama.embedding_length", &loader->metadata.n_embd);
    oc_gguf_metadata_get_u32(gguf, "llama.attention.head_count", &loader->metadata.n_heads);
    oc_gguf_metadata_get_u32(gguf, "llama.attention.head_count_kv", &loader->metadata.n_kv_heads);
    oc_gguf_metadata_get_u32(gguf, "llama.context_length", &loader->metadata.context_length);
    oc_gguf_metadata_get_u32(gguf, "tokenizer.ggml.tokens", &loader->metadata.vocab_size);

    /* Try Qwen-specific keys. */
    if (loader->metadata.n_layers == 0)
        oc_gguf_metadata_get_u32(gguf, "qwen.block_count", &loader->metadata.n_layers);
    if (loader->metadata.n_embd == 0)
        oc_gguf_metadata_get_u32(gguf, "qwen.embedding_length", &loader->metadata.n_embd);

    /* Try GPT-2 keys. */
    if (loader->metadata.n_layers == 0)
        oc_gguf_metadata_get_u32(gguf, "gpt2.block_count", &loader->metadata.n_layers);
    if (loader->metadata.n_embd == 0)
        oc_gguf_metadata_get_u32(gguf, "gpt2.embedding_length", &loader->metadata.n_embd);

    /* Try Falcon keys. */
    if (loader->metadata.n_layers == 0)
        oc_gguf_metadata_get_u32(gguf, "falcon.block_count", &loader->metadata.n_layers);

    /* Compute hidden_dim (n_embd * 4 for Llama family). */
    loader->metadata.hidden_dim = loader->metadata.n_embd * 4;

    /* Compute param count estimate. */
    if (loader->metadata.n_layers > 0 && loader->metadata.n_embd > 0) {
        /* Rough estimate: 12 * n_layers * n_embd^2 for a standard transformer. */
        loader->metadata.n_params = (uint64_t)loader->metadata.n_layers *
                                     loader->metadata.n_embd * loader->metadata.n_embd * 12;
    }

    /* Copy tensor info. */
    size_t n = gguf->tensor_count;
    if (n > OC_LOADER_MAX_TENSORS) n = OC_LOADER_MAX_TENSORS;
    for (size_t i = 0; i < n; i++) {
        OcLoadedTensor *t = &loader->tensors[i];
        const OcGgufTensorInfo *info = &gguf->tensors[i];
        copy_str(t->name, sizeof(t->name), info->name);
        t->n_dims = info->n_dims;
        for (uint32_t d = 0; d < t->n_dims && d < 4; d++)
            t->dims[d] = info->dims[d];
        t->type = info->ggml_type;
        t->shard_index = info->shard_index;
        t->offset = info->absolute_offset;
        if (!tensor_byte_size(info, &t->size)) {
            oc_gguf_map_free(&loader->mapped);
            return OC_ERR_QUANT;
        }
    }
    loader->n_tensors = n;

    loader->loaded = true;
    return OC_OK;
}

OcError oc_model_loader_get_tensor_data(const OcModelLoader *loader,
                                        const char *name,
                                        const uint8_t **out_data,
                                        size_t *out_size)
{
    if (!loader || !loader->loaded || !name || !out_data || !out_size)
        return OC_ERR_INVALID_ARG;

    const OcLoadedTensor *loaded = NULL;
    OcError e = oc_model_loader_get_tensor(loader, name, &loaded);
    if (e != OC_OK) return e;
    if (loaded->shard_index >= loader->mapped.n_shards)
        return OC_ERR_FORMAT;

    const OcGgufTensorInfo *info =
        oc_gguf_map_tensor_get(&loader->mapped, name);
    const uint8_t *data =
        oc_gguf_map_tensor_data(&loader->mapped, info);
    if (!info || !data) return OC_ERR_FORMAT;

    const OcGgufShard *shard = &loader->mapped.shards[loaded->shard_index];
    if (loaded->offset > shard->len ||
        loaded->size > shard->len - loaded->offset)
        return OC_ERR_FORMAT;

    *out_data = data;
    *out_size = (size_t)loaded->size;
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
    oc_gguf_map_free(&loader->mapped);
    memset(loader, 0, sizeof(*loader));
}
