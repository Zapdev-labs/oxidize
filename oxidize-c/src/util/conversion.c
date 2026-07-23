/*
 * conversion.c — Model format conversion implementation.
 */
#include "oxidize/conversion.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

OcError oc_conv_config_init(OcConvConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    memset(cfg, 0, sizeof(*cfg));
    cfg->target = OC_CONV_Q_Q4_K_M;
    cfg->verbose = false;
    cfg->copy_metadata = true;
    cfg->n_threads = 0; /* auto */
    return OC_OK;
}

OcError oc_conv_run(const OcConvConfig *cfg, OcConvResult *result)
{
    if (!cfg || !cfg->input_path || !cfg->output_path)
        return OC_ERR_INVALID_ARG;

    if (result) {
        memset(result, 0, sizeof(*result));
        result->target = cfg->target;
        result->n_tensors = 0;
        result->n_bytes = 0;
        result->elapsed_sec = 0.0;
        strcpy(result->arch_name, "unknown");

        /* Stub: just record the start time. */
        struct timespec ts;
        if (timespec_get(&ts, TIME_UTC) == TIME_UTC) {
            result->elapsed_sec = 0.001; /* stub: 1ms */
        }
    }

    /* Stub: real implementation would:
     * 1. Open SafeTensors file
     * 2. Parse JSON header
     * 3. Read config.json for architecture
     * 4. Create GGUF writer
     * 5. Copy/quantize tensors
     * 6. Write metadata
     * 7. Finalize GGUF
     */
    return OC_OK;
}

OcError oc_conv_quant_type_from_str(const char *str, OcConvQuantType *out)
{
    if (!str || !out) return OC_ERR_INVALID_ARG;
    if (strcmp(str, "F32") == 0) { *out = OC_CONV_Q_F32; return OC_OK; }
    if (strcmp(str, "F16") == 0) { *out = OC_CONV_Q_F16; return OC_OK; }
    if (strcmp(str, "BF16") == 0) { *out = OC_CONV_Q_BF16; return OC_OK; }
    if (strcmp(str, "Q8_0") == 0) { *out = OC_CONV_Q_Q8_0; return OC_OK; }
    if (strcmp(str, "Q4_0") == 0) { *out = OC_CONV_Q_Q4_0; return OC_OK; }
    if (strcmp(str, "Q4_K") == 0) { *out = OC_CONV_Q_Q4_K; return OC_OK; }
    if (strcmp(str, "Q5_K") == 0) { *out = OC_CONV_Q_Q5_K; return OC_OK; }
    if (strcmp(str, "Q6_K") == 0) { *out = OC_CONV_Q_Q6_K; return OC_OK; }
    if (strcmp(str, "Q4_K_M") == 0) { *out = OC_CONV_Q_Q4_K_M; return OC_OK; }
    return OC_ERR_INVALID_ARG;
}

const char *oc_conv_quant_type_name(OcConvQuantType type)
{
    switch (type) {
    case OC_CONV_Q_F32:   return "F32";
    case OC_CONV_Q_F16:   return "F16";
    case OC_CONV_Q_BF16:  return "BF16";
    case OC_CONV_Q_Q8_0:  return "Q8_0";
    case OC_CONV_Q_Q4_0:  return "Q4_0";
    case OC_CONV_Q_Q4_K:  return "Q4_K";
    case OC_CONV_Q_Q5_K:  return "Q5_K";
    case OC_CONV_Q_Q6_K:  return "Q6_K";
    case OC_CONV_Q_Q4_K_M: return "Q4_K_M";
    default: return "unknown";
    }
}

const char *oc_conv_quant_type_gguf_name(OcConvQuantType type)
{
    /* GGUF uses same names for quant types. */
    return oc_conv_quant_type_name(type);
}

bool oc_conv_is_valid_quant_type(OcConvQuantType type)
{
    return type >= OC_CONV_Q_F32 && type <= OC_CONV_Q_Q4_K_M;
}

uint32_t oc_conv_bits_per_weight(OcConvQuantType type)
{
    switch (type) {
    case OC_CONV_Q_F32:   return 32;
    case OC_CONV_Q_F16:   return 16;
    case OC_CONV_Q_BF16:  return 16;
    case OC_CONV_Q_Q8_0:  return 8;
    case OC_CONV_Q_Q4_0:  return 4;
    case OC_CONV_Q_Q4_K:  return 4;
    case OC_CONV_Q_Q5_K:  return 5;
    case OC_CONV_Q_Q6_K:  return 6;
    case OC_CONV_Q_Q4_K_M: return 4;
    default: return 0;
    }
}
