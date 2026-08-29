/* quantize_tool.c — offline GGUF weight quantization tool. */
#include "oxidize/quantize_tool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "oxidize/log.h"
#include "oxidize/quant.h"


OcError oc_quantize_parse_type(const char *str, OcGgufQuantizationType *out)
{
    if (!str || !out) return OC_ERR_INVALID_ARG;
    /* Use the existing name table from quant.h. We match by name. */
    for (uint32_t i = 0; i < (uint32_t)OC_QUANT__COUNT; i++) {
        const char *name = oc_quant_type_name((OcGgufQuantizationType)i);
        if (name && strcmp(name, str) == 0) {
            *out = (OcGgufQuantizationType)i;
            return OC_OK;
        }
    }
    return OC_ERR_INVALID_ARG;
}

static bool target_type_supported(OcGgufQuantizationType type)
{
    switch (type) {
    case OC_QUANT_F32:
    case OC_QUANT_F16:
    case OC_QUANT_BF16:
    case OC_QUANT_I8:
    case OC_QUANT_I16:
    case OC_QUANT_I32:
    case OC_QUANT_I64:
    case OC_QUANT_F64:
    case OC_QUANT_Q4_0:
    case OC_QUANT_Q4_1:
    case OC_QUANT_Q5_0:
    case OC_QUANT_Q5_1:
    case OC_QUANT_Q8_0:
    case OC_QUANT_Q4_K_S:
    case OC_QUANT_Q4_K_M:
    case OC_QUANT_AL5:
    case OC_QUANT_AL5_XS:
    case OC_QUANT_AL6:
    case OC_QUANT_AL8:
        return true;
    default:
        return false;
    }
}


static void write_u32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void write_u64(FILE *f, uint64_t v) { fwrite(&v, 8, 1, f); }
static void write_f32(FILE *f, float v) { fwrite(&v, 4, 1, f); }
static void write_f64(FILE *f, double v) { fwrite(&v, 8, 1, f); }

static void write_str(FILE *f, const char *s, size_t len)
{
    write_u64(f, len);
    fwrite(s, 1, len, f);
}

static void write_metadata_payload(FILE *f, const OcGgufMetadataValue *v, OcGgufMetadataType type)
{
    switch (type) {
    case OC_GGUF_MT_UINT8:
        fwrite(&v->v.u8, 1, 1, f);
        break;
    case OC_GGUF_MT_INT8:
        fwrite(&v->v.i8, 1, 1, f);
        break;
    case OC_GGUF_MT_UINT16:
        fwrite(&v->v.u16, 2, 1, f);
        break;
    case OC_GGUF_MT_INT16:
        fwrite(&v->v.i16, 2, 1, f);
        break;
    case OC_GGUF_MT_UINT32:
        write_u32(f, v->v.u32);
        break;
    case OC_GGUF_MT_INT32:
        fwrite(&v->v.i32, 4, 1, f);
        break;
    case OC_GGUF_MT_FLOAT32:
        write_f32(f, v->v.f32);
        break;
    case OC_GGUF_MT_BOOL: {
        uint8_t value = v->v.b ? 1 : 0;
        fwrite(&value, 1, 1, f);
        break;
    }
    case OC_GGUF_MT_STRING:
        write_str(f, v->v.str.data, v->v.str.len);
        break;
    case OC_GGUF_MT_ARRAY:
        write_u32(f, v->v.arr.elem_type);
        write_u64(f, v->v.arr.len);
        for (size_t i = 0; i < v->v.arr.len; i++) {
            write_metadata_payload(f, &v->v.arr.values[i], v->v.arr.elem_type);
        }
        break;
    case OC_GGUF_MT_UINT64:
        write_u64(f, v->v.u64);
        break;
    case OC_GGUF_MT_INT64:
        fwrite(&v->v.i64, 8, 1, f);
        break;
    case OC_GGUF_MT_FLOAT64:
        write_f64(f, v->v.f64);
        break;
    default:
        break;
    }
}

static void write_metadata_value(FILE *f, const OcGgufMetadataValue *v)
{
    write_u32(f, v->type);
    write_metadata_payload(f, v, v->type);
}

static void discard_output(FILE *out, const char *path)
{
    if (out) fclose(out);
    remove(path);
}


OcError oc_quantize_model(const OcQuantizeConfig *cfg)
{
    if (!cfg || !cfg->input_path || !cfg->output_path || !cfg->target_type) return OC_ERR_INVALID_ARG;

    OcGgufQuantizationType target_qtype;
    OcError e = oc_quantize_parse_type(cfg->target_type, &target_qtype);
    if (e != OC_OK) {
        fprintf(stderr, "error: unknown quantization type '%s'\n", cfg->target_type);
        return OC_ERR_INVALID_ARG;
    }
    if (!target_type_supported(target_qtype)) {
        fprintf(stderr, "error: quantization encoder for '%s' is not implemented\n", cfg->target_type);
        return OC_ERR_QUANT;
    }
    if (strcmp(cfg->input_path, cfg->output_path) == 0) {
        fprintf(stderr, "error: input and output paths must differ\n");
        return OC_ERR_INVALID_ARG;
    }
    struct stat input_stat;
    struct stat output_stat;
    if (stat(cfg->input_path, &input_stat) == 0 &&
        stat(cfg->output_path, &output_stat) == 0 &&
        input_stat.st_dev == output_stat.st_dev &&
        input_stat.st_ino == output_stat.st_ino) {
        fprintf(stderr, "error: input and output paths refer to the same file\n");
        return OC_ERR_INVALID_ARG;
    }

    /* Open input GGUF via mmap. */
    OcGgufMmappedFile mf;
    e = oc_gguf_map_open(cfg->input_path, &mf);
    if (e != OC_OK) {
        fprintf(stderr, "error: failed to open input GGUF (%s)\n", oc_error_msg(e));
        return e;
    }

    const OcGgufFile *gf = &mf.unified;
    if (cfg->verbose) {
        fprintf(stderr, "quantize: %s -> %s\n", cfg->input_path, cfg->output_path);
        fprintf(stderr, "  tensors: %llu\n", (unsigned long long)gf->tensor_count);
        fprintf(stderr, "  metadata: %llu KV pairs\n", (unsigned long long)gf->metadata_kv_count);
        fprintf(stderr, "  target type: %s\n", oc_quant_type_name(target_qtype));
    }

    /* Open output file. */
    FILE *out = fopen(cfg->output_path, "wb");
    if (!out) {
        fprintf(stderr, "error: cannot open output file '%s'\n", cfg->output_path);
        oc_gguf_map_free(&mf);
        return OC_ERR_IO;
    }

    /* Write GGUF header. */
    write_u32(out, 0x46554747); /* "GGUF" */
    write_u32(out, 3);          /* version 3 */
    write_u64(out, gf->tensor_count);
    write_u64(out, gf->metadata_kv_count);

    /* Copy metadata KV pairs. */
    for (uint64_t i = 0; i < gf->metadata_kv_count; i++) {
        const OcGgufMetadataKV *kv = &gf->metadata[i];
        write_str(out, kv->key, strlen(kv->key));
        write_metadata_value(out, &kv->value);
    }

    /* Write tensor info (patching ggml_type for weight tensors). */
    uint64_t data_offset = 0;
    for (uint64_t i = 0; i < gf->tensor_count; i++) {
        const OcGgufTensorInfo *t = &gf->tensors[i];
        write_str(out, t->name, strlen(t->name));
        write_u32(out, t->n_dims);
        for (uint32_t d = 0; d < t->n_dims; d++) write_u64(out, t->dims[d]);

        OcGgufQuantizationType src_qtype = oc_quant_type_from_ggml_id(t->ggml_type);
        if (src_qtype == OC_QUANT_UNKNOWN) {
            fprintf(stderr, "error: unsupported source type %u for tensor %s\n", t->ggml_type, t->name);
            discard_output(out, cfg->output_path);
            oc_gguf_map_free(&mf);
            return OC_ERR_QUANT;
        }
        size_t n_elements = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) n_elements *= t->dims[d];
        size_t target_size = oc_quantized_size(target_qtype, n_elements);
        bool quantize = target_size != 0;
        uint32_t out_ggml_type = t->ggml_type;
        if (quantize) {
            out_ggml_type = oc_quant_type_to_ggml_id(target_qtype);
        }
        write_u32(out, out_ggml_type);

        write_u64(out, data_offset);

        size_t data_size = quantize ? target_size : oc_quantized_size(src_qtype, n_elements);
        if (data_size == 0) {
            fprintf(stderr, "error: invalid block length for tensor %s\n", t->name);
            discard_output(out, cfg->output_path);
            oc_gguf_map_free(&mf);
            return OC_ERR_QUANT;
        }
        data_offset += data_size;
        /* Align to GGUF_ALIGNMENT (default 32). */
        uint64_t align = gf->alignment ? gf->alignment : 32;
        data_offset = (data_offset + align - 1) & ~(align - 1);
    }

    /* Pad to alignment. */
    uint64_t align = gf->alignment ? gf->alignment : 32;
    long pos = ftell(out);
    uint64_t padded = (pos + align - 1) & ~(align - 1);
    while (pos < (long)padded) {
        fputc(0, out);
        pos++;
    }

    /* Second pass: write tensor data. */
    for (uint64_t i = 0; i < gf->tensor_count; i++) {
        const OcGgufTensorInfo *t = &gf->tensors[i];
        const uint8_t *src_data = oc_gguf_map_tensor_data(&mf, t);
        if (!src_data) {
            fprintf(stderr, "error: tensor %s has no data\n", t->name);
            discard_output(out, cfg->output_path);
            oc_gguf_map_free(&mf);
            return OC_ERR_FORMAT;
        }

        size_t n_elements = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) n_elements *= t->dims[d];
        OcGgufQuantizationType src_qtype = oc_quant_type_from_ggml_id(t->ggml_type);

        size_t dst_size = oc_quantized_size(target_qtype, n_elements);
        if (dst_size != 0) {
            float *f32_buf = malloc(n_elements * sizeof(float));
            if (!f32_buf) {
                discard_output(out, cfg->output_path);
                oc_gguf_map_free(&mf);
                return OC_ERR_OOM;
            }

            e = oc_quant_dequant_row(src_qtype, src_data, oc_quantized_size(src_qtype, n_elements), f32_buf,
                                     n_elements);
            if (e != OC_OK) {
                fprintf(stderr, "error: dequantization failed for %s (%s)\n", t->name, oc_error_msg(e));
                free(f32_buf);
                discard_output(out, cfg->output_path);
                oc_gguf_map_free(&mf);
                return e;
            }

            uint8_t *dst_buf = malloc(dst_size);
            if (!dst_buf) {
                free(f32_buf);
                discard_output(out, cfg->output_path);
                oc_gguf_map_free(&mf);
                return OC_ERR_OOM;
            }
            e = oc_quant_pack_row(target_qtype, f32_buf, n_elements, dst_buf, dst_size);
            free(f32_buf);
            if (e != OC_OK) {
                fprintf(stderr, "error: quantization failed for %s (%s)\n", t->name, oc_error_msg(e));
                free(dst_buf);
                discard_output(out, cfg->output_path);
                oc_gguf_map_free(&mf);
                return e;
            }
            fwrite(dst_buf, 1, dst_size, out);
            free(dst_buf);
        } else {
            size_t raw_size = oc_quantized_size(src_qtype, n_elements);
            fwrite(src_data, 1, raw_size, out);
        }

        /* Pad to alignment. */
        pos = ftell(out);
        padded = (pos + align - 1) & ~(align - 1);
        while (pos < (long)padded) {
            fputc(0, out);
            pos++;
        }

        if (cfg->verbose && (i % 10 == 0 || i == gf->tensor_count - 1)) {
            fprintf(stderr, "  tensor %llu/%llu: %s (%u dims, %zu elements)\n", (unsigned long long)(i + 1),
                    (unsigned long long)gf->tensor_count, t->name, t->n_dims, n_elements);
        }
    }

    bool write_failed = ferror(out) != 0;
    if (fclose(out) != 0) write_failed = true;
    if (write_failed) {
        remove(cfg->output_path);
        oc_gguf_map_free(&mf);
        return OC_ERR_IO;
    }
    oc_gguf_map_free(&mf);

    if (cfg->verbose) {
        fprintf(stderr, "quantize: done, output written to %s\n", cfg->output_path);
    }
    return OC_OK;
}
