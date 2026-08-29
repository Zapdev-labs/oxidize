#define _POSIX_C_SOURCE 200809L
#include "oxidize/loader.h"
#include "oxidize/gguf.h"
#include "oxidize/model.h"
#include "oxidize/model_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static const char *const OC_LOADER_ARCHS[] = {
    "llama", "mistral", "mixtral", "deepseek", "qwen", "gemma",
    "phi", "falcon", "gpt2", "gptj", "gptneox", "minimax",
    "lfm2", "lfm2moe", "glm_moe_dsa", "hunyuan_moe", "longcat",
    "qwen35moe", "qwen35", "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen35_text",
    "qwen3_5_moe_text", "qwen36", "qwen3_6",
    "muse-glimmer", "muse_glimmer",
};
#define OC_LOADER_N_ARCHS \
    (sizeof(OC_LOADER_ARCHS) / sizeof(OC_LOADER_ARCHS[0]))


static void copy_str(char *dst, size_t cap, const char *src)
{
    if (!dst || cap == 0 || !src) return;
    size_t n = strlen(src);
    if (n >= cap) n = cap - 1;
    memcpy(dst, src, n);
    dst[n] = '\0';
}

static void set_error(OcLoaderResult *result, const char *msg)
{
    if (!result || !msg) return;
    copy_str(result->error_msg, sizeof(result->error_msg), msg);
    result->loaded = false;
}

static void zero_result(OcLoaderResult *result)
{
    memset(result, 0, sizeof(*result));
}


OcError oc_loader_init(void)
{
    /* In the dependency-free C11 port, loader state is process-wide and requires no heap allocation. */
    return OC_OK;
}


OcError oc_loader_load(const char *path, OcLoaderResult *result)
{
    if (!path || !result) return OC_ERR_INVALID_ARG;
    zero_result(result);

    /* Use the existing OcModelLoader to parse the GGUF. */
    OcModelLoader ml;
    OcError e = oc_model_loader_init(&ml, path);
    if (e != OC_OK) {
        set_error(result, "loader init failed");
        return e;
    }

    e = oc_model_loader_load(&ml);
    if (e != OC_OK) {
        set_error(result, "failed to load GGUF");
        oc_model_loader_free(&ml);
        return e;
    }

    /* Populate the result from the loaded metadata. */
    copy_str(result->arch_name, sizeof(result->arch_name), ml.metadata.arch);
    result->n_tensors = (uint32_t)ml.n_tensors;
    result->n_layers = ml.metadata.n_layers;
    result->model_size = ml.file_size;
    result->loaded = true;

    oc_model_loader_free(&ml);
    return OC_OK;
}

OcError oc_loader_load_with_arch(const char *path, const char *arch,
                                 OcLoaderResult *result)
{
    if (!path || !arch || !result) return OC_ERR_INVALID_ARG;
    zero_result(result);

    /* Validate the requested architecture. */
    if (!oc_loader_is_supported(arch)) {
        set_error(result, "unsupported architecture");
        return OC_ERR_MODEL;
    }

    /* Load and verify the detected architecture matches the request. */
    OcError e = oc_loader_load(path, result);
    if (e != OC_OK) return e;

    /* Compare normalized forms: if the detected arch is non-empty and does
     * not normalize to the same OcModelArchitecture as the request, reject. */
    if (result->arch_name[0] != '\0') {
        OcModelArchitecture requested = oc_model_arch_from_str(arch);
        OcModelArchitecture detected = oc_model_arch_from_str(result->arch_name);
        if (requested != OC_ARCH_UNKNOWN && detected != OC_ARCH_UNKNOWN &&
            requested != detected) {
            set_error(result, "architecture mismatch");
            return OC_ERR_MODEL;
        }
    }

    /* If the detected arch was empty (no general.architecture key), stamp
     * the requested architecture into the result. */
    if (result->arch_name[0] == '\0') {
        copy_str(result->arch_name, sizeof(result->arch_name), arch);
    }

    return OC_OK;
}


OcError oc_loader_detect_arch(const char *path, char *out_arch,
                              size_t out_size)
{
    if (!path || !out_arch || out_size == 0) return OC_ERR_INVALID_ARG;
    out_arch[0] = '\0';

    OcGgufFile gguf;
    OcError e = oc_gguf_open(path, &gguf);
    if (e != OC_OK) return e;

    const char *arch_str = NULL;
    size_t arch_len = 0;
    if (oc_gguf_metadata_get_str(&gguf, "general.architecture",
                                  &arch_str, &arch_len) && arch_str) {
        copy_str(out_arch, out_size, arch_str);
    }

    oc_gguf_free(&gguf);
    return OC_OK;
}


OcError oc_loader_supported_archs(const char **out, uint32_t *n)
{
    if (!n) return OC_ERR_INVALID_ARG;
    if (out) {
        for (uint32_t i = 0; i < (uint32_t)OC_LOADER_N_ARCHS; i++) {
            out[i] = OC_LOADER_ARCHS[i];
        }
    }
    *n = (uint32_t)OC_LOADER_N_ARCHS;
    return OC_OK;
}

bool oc_loader_is_supported(const char *arch)
{
    if (!arch || !*arch) return false;
    OcModelArchitecture a = oc_model_arch_from_str(arch);
    return a != OC_ARCH_UNKNOWN;
}


void oc_loader_unload(OcLoaderResult *result)
{
    if (!result) return;
    zero_result(result);
}


const char *oc_loader_arch_name(uint32_t idx)
{
    if (idx >= (uint32_t)OC_LOADER_N_ARCHS) return NULL;
    return OC_LOADER_ARCHS[idx];
}
