/*
 * fingerprint.c — Model file fingerprinting implementation.
 */
#include "oxidize/fingerprint.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

OcError oc_fingerprint_init(OcModelFingerprint *fp)
{
    if (!fp) return OC_ERR_INVALID_ARG;
    memset(fp, 0, sizeof(*fp));
    strcpy(fp->architecture, "unknown");
    fp->rope_theta = 10000.0f;
    fp->quant_type = 0; /* F32 */
    return OC_OK;
}

OcError oc_fingerprint_from_file(const char *path, OcModelFingerprint *fp)
{
    if (!path || !fp) return OC_ERR_INVALID_ARG;
    oc_fingerprint_init(fp);

    /* Stub: real implementation would mmap the GGUF file and read metadata.
     * For now, just set the file size from stat. */
    FILE *f = fopen(path, "rb");
    if (!f) return OC_ERR_IO;
    fseek(f, 0, SEEK_END);
    fp->file_size = (uint64_t)ftell(f);
    fclose(f);

    /* Estimate params from file size (very rough). */
    if (fp->file_size > 0) {
        fp->estimated_params = fp->file_size / 2; /* rough: ~2 bytes per param for Q4 */
    }

    return OC_OK;
}

OcError oc_fingerprint_validate(const OcModelFingerprint *fp)
{
    if (!fp) return OC_ERR_INVALID_ARG;
    if (fp->n_layers == 0 && fp->file_size == 0) return OC_ERR_INVALID_ARG;
    return OC_OK;
}

bool oc_fingerprint_is_quantized(const OcModelFingerprint *fp)
{
    if (!fp) return false;
    return fp->quant_type > 0;
}

bool oc_fingerprint_is_moe(const OcModelFingerprint *fp)
{
    if (!fp) return false;
    return fp->n_expert > 1;
}

bool oc_fingerprint_has_gqa(const OcModelFingerprint *fp)
{
    if (!fp) return false;
    return fp->n_kv_heads > 0 && fp->n_kv_heads < fp->n_heads;
}

double oc_fingerprint_model_size_gb(const OcModelFingerprint *fp)
{
    if (!fp) return 0.0;
    return (double)fp->file_size / (1024.0 * 1024 * 1024);
}

double oc_fingerprint_bits_per_param(const OcModelFingerprint *fp)
{
    if (!fp || fp->estimated_params == 0) return 0.0;
    return (double)fp->file_size * 8.0 / (double)fp->estimated_params;
}

const char *oc_fingerprint_summary(const OcModelFingerprint *fp, char *out, size_t out_size)
{
    if (!fp || !out || out_size == 0) return "";
    snprintf(out, out_size,
        "arch=%s, layers=%u, hidden=%u, heads=%u/%u, vocab=%u, "
        "quant=%u, size=%.1fGB, params=%lluM, bpw=%.1f, moe=%s",
        fp->architecture, fp->n_layers, fp->hidden_dim,
        fp->n_heads, fp->n_kv_heads, fp->vocab_size,
        fp->quant_type, oc_fingerprint_model_size_gb(fp),
        (unsigned long long)(fp->estimated_params / 1000000),
        oc_fingerprint_bits_per_param(fp),
        oc_fingerprint_is_moe(fp) ? "yes" : "no");
    return out;
}
