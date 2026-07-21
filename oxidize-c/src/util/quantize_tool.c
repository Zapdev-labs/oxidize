/*
 * quantize_tool.c — offline GGUF weight quantization tool.
 *
 * Reads an input GGUF, dequantizes each weight tensor to f32, re-quantizes
 * to the target type, and writes a new GGUF file.
 *
 * The writer is a minimal GGUF v3 writer that copies metadata from the
 * input file, patches tensor type ids, and writes the re-quantized data.
 */
#include "oxidize/quantize_tool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oxidize/quant.h"
#include "oxidize/log.h"

/* ─── Type parsing ────────────────────────────────────────────────────── */

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

size_t oc_quantize_block_size(OcGgufQuantizationType type, size_t n_elements)
{
    return oc_quantized_size(type, n_elements);
}

/* ─── Minimal GGUF writer ───────────────────────────────────────────────
 *
 * Writes a GGUF v3 file with:
 *   - Copied metadata KV pairs from the source file
 *   - Patched tensor type ids (to the target quant type)
 *   - Re-quantized tensor data
 *
 * Layout: magic | version | tensor_count | kv_count | KV pairs | tensor info
 *         | padding | tensor data
 */

static void write_u32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void write_u64(FILE *f, uint64_t v) { fwrite(&v, 8, 1, f); }
static void write_f32(FILE *f, float v)    { fwrite(&v, 4, 1, f); }

static void write_str(FILE *f, const char *s, size_t len)
{
    write_u64(f, len);
    fwrite(s, 1, len, f);
}

/* Write a metadata value. Only handles common types used in GGUF model files. */
static void write_metadata_value(FILE *f, const OcGgufMetadataValue *v)
{
    switch (v->type) {
    case OC_GGUF_MT_UINT8:   write_u32(f, 0); write_u32(f, v->v.u8); break;
    case OC_GGUF_MT_INT8:   write_u32(f, 1); write_u32(f, (uint32_t)v->v.i8); break;
    case OC_GGUF_MT_UINT16: write_u32(f, 2); write_u32(f, v->v.u16); break;
    case OC_GGUF_MT_INT16:  write_u32(f, 3); write_u32(f, (uint32_t)v->v.i16); break;
    case OC_GGUF_MT_UINT32: write_u32(f, 4); write_u32(f, v->v.u32); break;
    case OC_GGUF_MT_INT32:  write_u32(f, 5); write_u32(f, (uint32_t)v->v.i32); break;
    case OC_GGUF_MT_FLOAT32: write_u32(f, 6); write_f32(f, v->v.f32); break;
    case OC_GGUF_MT_BOOL:   write_u32(f, 7); write_u32(f, v->v.b ? 1 : 0); break;
    case OC_GGUF_MT_STRING:
        write_u32(f, 8);
        write_str(f, v->v.str.data, v->v.str.len);
        break;
    case OC_GGUF_MT_ARRAY:
        write_u32(f, 9);
        write_u32(f, v->v.arr.elem_type);
        write_u64(f, v->v.arr.len);
        for (size_t i = 0; i < v->v.arr.len; i++) {
            write_metadata_value(f, &v->v.arr.values[i]);
        }
        break;
    case OC_GGUF_MT_UINT64: write_u32(f, 10); write_u64(f, v->v.u64); break;
    case OC_GGUF_MT_INT64:  write_u32(f, 11); write_u64(f, (uint64_t)v->v.i64); break;
    case OC_GGUF_MT_FLOAT64: write_u32(f, 12); { uint64_t bits; memcpy(&bits, &v->v.f64, 8); write_u64(f, bits); } break;
    default:
        write_u32(f, 0); write_u32(f, 0); break;
    }
}

/* ─── Main quantize function ──────────────────────────────────────────── */

OcError oc_quantize_model(const OcQuantizeConfig *cfg)
{
    if (!cfg || !cfg->input_path || !cfg->output_path || !cfg->target_type)
        return OC_ERR_INVALID_ARG;

    OcGgufQuantizationType target_qtype;
    OcError e = oc_quantize_parse_type(cfg->target_type, &target_qtype);
    if (e != OC_OK) {
        fprintf(stderr, "error: unknown quantization type '%s'\n", cfg->target_type);
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
    write_u32(out, 0x46554747);  /* "GGUF" */
    write_u32(out, 3);           /* version 3 */
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
    uint64_t *tensor_offsets = calloc(gf->tensor_count, sizeof(uint64_t));
    if (!tensor_offsets) { fclose(out); oc_gguf_map_free(&mf); return OC_ERR_OOM; }

    for (uint64_t i = 0; i < gf->tensor_count; i++) {
        const OcGgufTensorInfo *t = &gf->tensors[i];
        write_str(out, t->name, strlen(t->name));
        write_u32(out, t->n_dims);
        for (uint32_t d = 0; d < t->n_dims; d++)
            write_u64(out, t->dims[d]);

        /* Determine if this is a weight tensor (quantizable) or a
         * non-quantizable tensor (e.g. token embeddings for some types).
         * For simplicity, we re-quantize all tensors that have a recognized
         * quant type. */
        OcGgufQuantizationType src_qtype = oc_quant_type_from_ggml_id(t->ggml_type);
        uint32_t out_ggml_type = t->ggml_type;
        if (src_qtype != OC_QUANT_UNKNOWN && src_qtype != OC_QUANT_F32 && src_qtype != OC_QUANT_F16) {
            out_ggml_type = oc_quant_type_to_ggml_id(target_qtype);
        }
        write_u32(out, out_ggml_type);

        tensor_offsets[i] = data_offset;
        write_u64(out, data_offset);

        /* Compute data size for this tensor. */
        size_t n_elements = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) n_elements *= t->dims[d];
        size_t data_size;
        if (src_qtype != OC_QUANT_UNKNOWN && src_qtype != OC_QUANT_F32 && src_qtype != OC_QUANT_F16) {
            data_size = oc_quantized_size(target_qtype, n_elements);
        } else if (src_qtype == OC_QUANT_F32) {
            data_size = n_elements * sizeof(float);
        } else if (src_qtype == OC_QUANT_F16) {
            data_size = n_elements * 2;
        } else {
            /* Unknown: copy raw bytes. */
            data_size = n_elements * sizeof(float);
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
    while (pos < (long)padded) { fputc(0, out); pos++; }

    /* Second pass: write tensor data. */
    for (uint64_t i = 0; i < gf->tensor_count; i++) {
        const OcGgufTensorInfo *t = &gf->tensors[i];
        const uint8_t *src_data = oc_gguf_map_tensor_data(&mf, t);
        if (!src_data) {
            fprintf(stderr, "warning: tensor %s has no data, skipping\n", t->name);
            continue;
        }

        size_t n_elements = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) n_elements *= t->dims[d];
        OcGgufQuantizationType src_qtype = oc_quant_type_from_ggml_id(t->ggml_type);

        if (src_qtype != OC_QUANT_UNKNOWN && src_qtype != OC_QUANT_F32 && src_qtype != OC_QUANT_F16) {
            /* Dequantize → re-quantize. */
            float *f32_buf = malloc(n_elements * sizeof(float));
            if (!f32_buf) { fclose(out); free(tensor_offsets); oc_gguf_map_free(&mf); return OC_ERR_OOM; }

            e = oc_quant_dequant_row(src_qtype, src_data,
                                     oc_quantized_size(src_qtype, n_elements),
                                     f32_buf, n_elements);
            if (e != OC_OK) {
                fprintf(stderr, "warning: dequant failed for %s, copying raw\n", t->name);
                free(f32_buf);
                fwrite(src_data, 1, oc_quantized_size(src_qtype, n_elements), out);
            } else {
                size_t dst_size = oc_quantized_size(target_qtype, n_elements);
                uint8_t *dst_buf = malloc(dst_size);
                if (!dst_buf) {
                    free(f32_buf);
                    fclose(out); free(tensor_offsets); oc_gguf_map_free(&mf);
                    return OC_ERR_OOM;
                }
                e = oc_quant_pack_row(target_qtype, f32_buf, n_elements,
                                      dst_buf, dst_size);
                if (e != OC_OK) {
                    fprintf(stderr, "warning: pack failed for %s, writing f32\n", t->name);
                    fwrite(f32_buf, sizeof(float), n_elements, out);
                } else {
                    fwrite(dst_buf, 1, dst_size, out);
                }
                free(dst_buf);
                free(f32_buf);
            }
        } else if (src_qtype == OC_QUANT_F32) {
            /* F32: optionally quantize to target if target is F16. */
            if (target_qtype == OC_QUANT_F16) {
                /* Convert f32 → f16 (simplified: just write f32 for now). */
                fwrite(src_data, sizeof(float), n_elements, out);
            } else {
                fwrite(src_data, sizeof(float), n_elements, out);
            }
        } else {
            /* F16 or unknown: copy raw. */
            size_t raw_size = n_elements * (t->ggml_type == 1 ? 2 : 4);
            fwrite(src_data, 1, raw_size, out);
        }

        /* Pad to alignment. */
        pos = ftell(out);
        padded = (pos + align - 1) & ~(align - 1);
        while (pos < (long)padded) { fputc(0, out); pos++; }

        if (cfg->verbose && (i % 10 == 0 || i == gf->tensor_count - 1)) {
            fprintf(stderr, "  tensor %llu/%llu: %s (%u dims, %zu elements)\n",
                    (unsigned long long)(i + 1), (unsigned long long)gf->tensor_count,
                    t->name, t->n_dims, n_elements);
        }
    }

    free(tensor_offsets);
    fclose(out);
    oc_gguf_map_free(&mf);

    if (cfg->verbose) {
        fprintf(stderr, "quantize: done, output written to %s\n", cfg->output_path);
    }
    return OC_OK;
}
