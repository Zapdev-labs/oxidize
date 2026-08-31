/* model_loader.h — Universal model loader for GGUF files. */
#ifndef OXIDIZE_MODEL_LOADER_H
#define OXIDIZE_MODEL_LOADER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/gguf.h"
#include "oxidize/model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_LOADER_MAX_TENSORS 2048
#define OC_LOADER_MAX_NAME 256
#define OC_LOADER_MAX_ARCH 64

/* Uses OcModelArchitecture from model.h */

typedef struct OcLoadedTensor {
    char name[OC_LOADER_MAX_NAME];
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type;
    uint32_t shard_index;
    uint64_t offset;
    uint64_t size;
} OcLoadedTensor;

typedef struct OcModelMetadata {
    char arch[OC_LOADER_MAX_ARCH];
    uint32_t n_layers;
    uint32_t n_embd;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t vocab_size;
    uint32_t context_length;
    uint32_t hidden_dim;
    float rope_freq_base;
    float rope_scale;
    uint64_t n_params;
} OcModelMetadata;

typedef struct OcModelLoader {
    char path[512];
    OcModelArchitecture arch;
    OcModelMetadata metadata;
    OcLoadedTensor tensors[OC_LOADER_MAX_TENSORS];
    size_t n_tensors;
    uint64_t file_size;
    OcGgufMmappedFile mapped;
    bool loaded;
} OcModelLoader;

OcError oc_model_loader_init(OcModelLoader *loader, const char *path);
OcError oc_model_loader_load(OcModelLoader *loader);
OcError oc_model_loader_get_tensor(const OcModelLoader *loader,
                                   const char *name,
                                   const OcLoadedTensor **out);
/* Return a read-only view of a tensor's exact encoded bytes. The pointer is
 * valid until oc_model_loader_free(loader); no tensor copy is performed. */
OcError oc_model_loader_get_tensor_data(const OcModelLoader *loader,
                                        const char *name,
                                        const uint8_t **out_data,
                                        size_t *out_size);
OcError oc_model_loader_list_tensors(const OcModelLoader *loader,
                                    const OcLoadedTensor **out_array,
                                    size_t *out_count);
/* Uses oc_model_arch_name and oc_model_arch_parse from model.h */
uint64_t oc_model_loader_param_count(const OcModelLoader *loader);
void oc_model_loader_free(OcModelLoader *loader);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_MODEL_LOADER_H */
