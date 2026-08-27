/*
 * llama_swap.c — Llama-family model swap implementation.
 *
 * Uses mmap to load GGUF files into memory for fast model switching.
 */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/llama_swap.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* mmap headers */
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

OcError oc_model_swap_init(OcModelSwap *sw)
{
    if (!sw) return OC_ERR_INVALID_ARG;
    memset(sw, 0, sizeof(*sw));
    sw->active_idx = -1;
    return OC_OK;
}

OcError oc_model_swap_register(OcModelSwap *sw, const char *path,
                               const char *name, int *out_idx)
{
    if (!sw || !path) return OC_ERR_INVALID_ARG;
    if (sw->n_models >= OC_SWAP_MAX_MODELS) return OC_ERR_OOM;

    int idx = (int)sw->n_models;
    OcSwapModelEntry *e = &sw->models[idx];
    memset(e, 0, sizeof(*e));

    size_t plen = strlen(path);
    if (plen >= sizeof(e->path)) plen = sizeof(e->path) - 1;
    memcpy(e->path, path, plen);
    e->path[plen] = '\0';

    if (name) {
        size_t nlen = strlen(name);
        if (nlen >= sizeof(e->name)) nlen = sizeof(e->name) - 1;
        memcpy(e->name, name, nlen);
        e->name[nlen] = '\0';
    }

    e->loaded = false;
    sw->n_models++;
    if (out_idx) *out_idx = idx;
    return OC_OK;
}

OcError oc_model_swap_load(OcModelSwap *sw, int idx)
{
    if (!sw || idx < 0 || (size_t)idx >= sw->n_models)
        return OC_ERR_INVALID_ARG;

    OcSwapModelEntry *e = &sw->models[idx];
    if (e->loaded) return OC_OK;

    /* Open the file. */
    int fd = open(e->path, O_RDONLY);
    if (fd < 0) {
        return OC_ERR_IO;
    }

    /* Get file size. */
    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return OC_ERR_IO;
    }
    if (st.st_size <= 0) {
        close(fd);
        return OC_ERR_MODEL;
    }

    /* mmap the file. */
    void *mapped = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        close(fd);
        return OC_ERR_OOM;
    }

    /* Store the mapping. The fd stays open for the lifetime of the mapping. */
    e->model_data = mapped;
    e->model_data_size = (size_t)st.st_size;
    e->size_bytes = (uint64_t)st.st_size;
    e->loaded = true;
    sw->total_loaded_bytes += e->model_data_size;

    /* We can close fd after mmap on most Unix systems, but keeping it open
     * is safer for portability. Store fd in name field padding? No, use
     * a separate approach: close fd since MAP_PRIVATE keeps the mapping. */
    close(fd);

    return OC_OK;
}

OcError oc_model_swap_unload(OcModelSwap *sw, int idx)
{
    if (!sw || idx < 0 || (size_t)idx >= sw->n_models)
        return OC_ERR_INVALID_ARG;

    OcSwapModelEntry *e = &sw->models[idx];
    if (!e->loaded) return OC_OK;

    /* Unmap the file from memory. */
    if (e->model_data) {
        munmap(e->model_data, e->model_data_size);
    }
    e->model_data = NULL;
    sw->total_loaded_bytes -= e->model_data_size;
    e->model_data_size = 0;
    e->loaded = false;
    if (sw->active_idx == idx) sw->active_idx = -1;
    return OC_OK;
}

OcError oc_model_swap_activate(OcModelSwap *sw, int idx)
{
    if (!sw || idx < 0 || (size_t)idx >= sw->n_models)
        return OC_ERR_INVALID_ARG;

    /* Load if needed. */
    OcError e = oc_model_swap_load(sw, idx);
    if (e != OC_OK) return e;

    sw->active_idx = idx;
    return OC_OK;
}

int oc_model_swap_active(const OcModelSwap *sw)
{
    if (!sw) return -1;
    return sw->active_idx;
}

OcError oc_model_swap_info(const OcModelSwap *sw, int idx,
                           const OcSwapModelEntry **out_info)
{
    if (!sw || !out_info || idx < 0 || (size_t)idx >= sw->n_models)
        return OC_ERR_INVALID_ARG;
    *out_info = &sw->models[idx];
    return OC_OK;
}

size_t oc_model_swap_list(const OcModelSwap *sw,
                          const OcSwapModelEntry **out_array)
{
    if (!sw || !out_array) return 0;
    *out_array = sw->models;
    return sw->n_models;
}

uint64_t oc_model_swap_loaded_bytes(const OcModelSwap *sw)
{
    if (!sw) return 0;
    return sw->total_loaded_bytes;
}

void oc_model_swap_free(OcModelSwap *sw)
{
    if (!sw) return;
    for (size_t i = 0; i < sw->n_models; i++) {
        if (sw->models[i].loaded && sw->models[i].model_data) {
            munmap(sw->models[i].model_data, sw->models[i].model_data_size);
        }
    }
    memset(sw, 0, sizeof(*sw));
    sw->active_idx = -1;
}
