/* conversion.h — Model format conversion. */
#ifndef OXIDIZE_CONVERSION_H
#define OXIDIZE_CONVERSION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OC_CONV_Q_F32 = 0,
    OC_CONV_Q_F16 = 1,
    OC_CONV_Q_BF16 = 2,
    OC_CONV_Q_Q8_0 = 3,
    OC_CONV_Q_Q4_0 = 4,
    OC_CONV_Q_Q4_K = 5,
    OC_CONV_Q_Q5_K = 6,
    OC_CONV_Q_Q6_K = 7,
    OC_CONV_Q_Q4_K_M = 8,
} OcConvQuantType;

typedef struct {
    const char *input_path;     /* SafeTensors file or directory */
    const char *output_path;    /* GGUF output path */
    OcConvQuantType target;     /* Target quantization type */
    bool verbose;
    bool copy_metadata;
    uint32_t n_threads;
} OcConvConfig;

typedef struct {
    size_t n_tensors;
    size_t n_bytes;
    double elapsed_sec;
    char arch_name[64];
    OcConvQuantType target;
} OcConvResult;

OcError oc_conv_config_init(OcConvConfig *cfg);
OcError oc_conv_run(const OcConvConfig *cfg, OcConvResult *result);
OcError oc_conv_quant_type_from_str(const char *str, OcConvQuantType *out);
const char *oc_conv_quant_type_name(OcConvQuantType type);
const char *oc_conv_quant_type_gguf_name(OcConvQuantType type);
bool oc_conv_is_valid_quant_type(OcConvQuantType type);
uint32_t oc_conv_bits_per_weight(OcConvQuantType type);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_CONVERSION_H */
