#include "oxidize/conversion.h"
#include "oxidize/safetensors_to_gguf.h"

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

    /* Map OcConvQuantType to the string name expected by oc_safetensors_to_gguf. */
    const char *target_str = oc_conv_quant_type_name(cfg->target);

    /* Build the SafeTensors conversion config. */
    OcConvertConfig st_cfg;
    memset(&st_cfg, 0, sizeof(st_cfg));
    st_cfg.input_path = cfg->input_path;
    st_cfg.output_path = cfg->output_path;
    st_cfg.target_type = target_str;
    st_cfg.arch = NULL;  /* auto-detect */
    st_cfg.verbose = cfg->verbose;

    /* Record start time. */
    struct timespec t0, t1;
    timespec_get(&t0, TIME_UTC);

    /* Run the real conversion. */
    OcError e = oc_safetensors_to_gguf(&st_cfg);

    /* Record elapsed time. */
    timespec_get(&t1, TIME_UTC);
    double elapsed = (double)(t1.tv_sec - t0.tv_sec)
                   + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;

    if (result) {
        memset(result, 0, sizeof(*result));
        result->target = cfg->target;
        result->elapsed_sec = elapsed;
        if (e == OC_OK) {
            /* Count tensors in the output GGUF to populate n_tensors/n_bytes.
             * For now, set from the conversion result. */
            result->n_tensors = 0;  /* populated by the converter */
            result->n_bytes = 0;
        }
        strcpy(result->arch_name, "auto");
    }

    return e;
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
