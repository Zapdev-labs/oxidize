/*
 * fingerprint.c — Model file fingerprinting implementation.
 *
 * Reads GGUF metadata to populate the fingerprint with real architecture
 * information (layers, heads, hidden dim, vocab, quant type, etc.).
 */
#include "oxidize/fingerprint.h"
#include "oxidize/gguf.h"

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

    /* Get file size. */
    FILE *f = fopen(path, "rb");
    if (!f) return OC_ERR_IO;
    fseek(f, 0, SEEK_END);
    fp->file_size = (uint64_t)ftell(f);
    fclose(f);

    /* Parse GGUF metadata for real architecture info. */
    OcGgufFile gguf;
    OcError e = oc_gguf_open(path, &gguf);
    if (e != OC_OK) {
        /* Not a valid GGUF — fall back to file-size estimate. */
        if (fp->file_size > 0)
            fp->estimated_params = fp->file_size / 2;
        return OC_OK;
    }

    /* Read architecture string. */
    const char *arch = NULL;
    size_t arch_len = 0;
    if (oc_gguf_metadata_get_str(&gguf, "general.architecture", &arch, &arch_len) &&
        arch_len > 0 && arch_len < OC_FP_MAX_ARCH - 1) {
        memcpy(fp->architecture, arch, arch_len);
        fp->architecture[arch_len] = '\0';
    }

    /* Build prefix for architecture-specific keys (e.g. "llama.block_count"). */
    char key[128];
    const char *prefix = fp->architecture;

    /* Helper macro for reading metadata. */
    #define FP_GET_U32(name, field) do { \
        snprintf(key, sizeof(key), "%s.%s", prefix, name); \
        uint32_t val; \
        if (oc_gguf_metadata_get_u32(&gguf, key, &val)) \
            fp->field = val; \
    } while (0)

    #define FP_GET_F32(name, field) do { \
        snprintf(key, sizeof(key), "%s.%s", prefix, name); \
        float val; \
        if (oc_gguf_metadata_get_f32(&gguf, key, &val)) \
            fp->field = val; \
    } while (0)

    FP_GET_U32("block_count", n_layers);
    FP_GET_U32("attention.head_count", n_heads);
    FP_GET_U32("attention.head_count_kv", n_kv_heads);
    FP_GET_U32("attention.key_length", head_dim);
    FP_GET_U32("embedding_length", hidden_dim);
    FP_GET_U32("feed_forward_length", intermediate_dim);
    FP_GET_U32("vocab_size", vocab_size);
    FP_GET_U32("context_length", n_ctx);
    FP_GET_U32("expert_count", n_expert);
    FP_GET_U32("expert_used_count", n_expert_used);
    FP_GET_F32("rope_freq_base", rope_theta);

    /* Read rope scaling. */
    snprintf(key, sizeof(key), "%s.%s", prefix, "rope_scaling.type");
    const char *rs_type = NULL;
    size_t rs_len = 0;
    if (oc_gguf_metadata_get_str(&gguf, key, &rs_type, &rs_len) && rs_len > 0) {
        size_t copy = rs_len < sizeof(fp->rope_scaling_type) - 1 ? rs_len : sizeof(fp->rope_scaling_type) - 1;
        memcpy(fp->rope_scaling_type, rs_type, copy);
        fp->rope_scaling_type[copy] = '\0';
        fp->has_rope_scaling = true;
    }
    snprintf(key, sizeof(key), "%s.%s", prefix, "rope_scaling.factor");
    float rs_factor;
    if (oc_gguf_metadata_get_f32(&gguf, key, &rs_factor))
        fp->rope_scaling_factor = rs_factor;

    #undef FP_GET_U32
    #undef FP_GET_F32

    /* Determine quant type from first tensor. */
    if (gguf.tensor_count > 0) {
        fp->quant_type = gguf.tensors[0].ggml_type;
    }

    /* Estimate params: hidden * (n_layers * (intermediate + 4*hidden) + vocab). */
    if (fp->hidden_dim > 0 && fp->n_layers > 0) {
        uint64_t inter = fp->intermediate_dim > 0 ? fp->intermediate_dim : fp->hidden_dim * 4;
        fp->estimated_params = (uint64_t)fp->n_layers * (
            (uint64_t)fp->hidden_dim * (inter + 4 * fp->hidden_dim)
        );
        if (fp->vocab_size > 0)
            fp->estimated_params += (uint64_t)fp->vocab_size * fp->hidden_dim;
        /* MoE: add expert FFN params. */
        if (fp->n_expert > 1) {
            fp->estimated_params += (uint64_t)(fp->n_expert - 1) * fp->n_layers *
                                   inter * fp->hidden_dim * 3;
        }
    } else if (fp->file_size > 0) {
        fp->estimated_params = fp->file_size / 2;
    }

    oc_gguf_free(&gguf);
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
