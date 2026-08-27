/*
 * quantize_tool.h — offline GGUF weight quantization tool.
 *
 * Re-quantizes a GGUF model from one quantization type to another. Reads
 * the input GGUF, dequantizes each tensor to f32, then re-quantizes to
 * the target type and writes a new GGUF file.
 */
#ifndef OXIDIZE_QUANTIZE_TOOL_H
#define OXIDIZE_QUANTIZE_TOOL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/gguf.h"
#include "oxidize/quant.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OcQuantizeConfig {
    const char *input_path;
    const char *output_path;
    const char *target_type; /* "Q4_0", "Q4_K_M", "Q4_K_S", "Q8_0", "F16" */
    bool verbose;
} OcQuantizeConfig;

/* Quantize a GGUF model from one type to another.
 * Reads input_path, dequantizes all weight tensors, re-quantizes to the
 * target type, and writes to output_path. */
OcError oc_quantize_model(const OcQuantizeConfig *cfg);

/* Parse a quantization type string into an OcGgufQuantizationType. */
OcError oc_quantize_parse_type(const char *str, OcGgufQuantizationType *out);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_QUANTIZE_TOOL_H */
