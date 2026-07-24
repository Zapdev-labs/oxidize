/*
 * safetensors_to_gguf.h — SafeTensors to GGUF conversion utility.
 *
 * Reads a SafeTensors checkpoint (HuggingFace format) and converts it
 * to GGUF format for use with the oxidize inference engine.
 *
 * Port of oxidize-convert/ Rust crate + oxidize-core/src/format/safetensors_to_gguf.rs.
 */
#ifndef OXIDIZE_SAFETENSORS_TO_GGUF_H
#define OXIDIZE_SAFETENSORS_TO_GGUF_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OcConvertConfig {
    const char *input_path;     /* SafeTensors file or model directory     */
    const char *output_path;    /* output GGUF path                        */
    const char *target_type;    /* target quant type (Q4_K_M, F16, etc.)   */
    const char *arch;           /* architecture override (llama, qwen2, etc.) */
    bool        verbose;
} OcConvertConfig;

/* Convert a SafeTensors checkpoint to GGUF format. */
OcError oc_safetensors_to_gguf(const OcConvertConfig *cfg);

/* Parse SafeTensors metadata header (JSON at the start of the file).
 * On success, `*out_json` receives a heap-allocated, NUL-terminated copy of
 * the metadata JSON and `*out_len` its length. The caller owns the buffer
 * and must free() it. */
OcError oc_safetensors_parse_header(const char *path,
                                     char **out_json, size_t *out_len);

/* Detect the model architecture from SafeTensors tensor names.
 * Returns a string like "llama", "qwen2", "mistral", etc. */
const char *oc_detect_arch_from_tensors(const char *const *tensor_names,
                                         size_t n_tensors);

/* Map a SafeTensors tensor name to a GGUF canonical name.
 * e.g. "model.layers.0.self_attn.q_proj.weight" → "blk.0.attn_q.weight"
 * For per-layer names, the returned pointer aliases a thread-local buffer
 * that is overwritten by the next call on the same thread — copy the result
 * before mapping another name. Unmapped names return `st_name` itself. */
const char *oc_map_tensor_name(const char *st_name, const char *arch);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_SAFETENSORS_TO_GGUF_H */
