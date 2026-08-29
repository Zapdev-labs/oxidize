/* fingerprint.h — Model file fingerprinting. */
#ifndef OXIDIZE_FINGERPRINT_H
#define OXIDIZE_FINGERPRINT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_FP_MAX_ARCH 64

typedef struct {
    char architecture[OC_FP_MAX_ARCH];
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t hidden_dim;
    uint32_t intermediate_dim;
    uint32_t vocab_size;
    uint32_t n_ctx;
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t quant_type;
    uint64_t file_size;
    uint64_t estimated_params;
    bool has_rope_scaling;
    float rope_theta;
    float rope_scaling_factor;
    char rope_scaling_type[32];
} OcModelFingerprint;

OcError oc_fingerprint_init(OcModelFingerprint *fp);
OcError oc_fingerprint_from_file(const char *path, OcModelFingerprint *fp);
OcError oc_fingerprint_validate(const OcModelFingerprint *fp);
bool oc_fingerprint_is_quantized(const OcModelFingerprint *fp);
bool oc_fingerprint_is_moe(const OcModelFingerprint *fp);
bool oc_fingerprint_has_gqa(const OcModelFingerprint *fp);
double oc_fingerprint_model_size_gb(const OcModelFingerprint *fp);
double oc_fingerprint_bits_per_param(const OcModelFingerprint *fp);
const char *oc_fingerprint_summary(const OcModelFingerprint *fp, char *out, size_t out_size);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_FINGERPRINT_H */
