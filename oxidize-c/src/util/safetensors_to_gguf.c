/* safetensors_to_gguf.c — SafeTensors to GGUF conversion implementation. */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/safetensors_to_gguf.h"

#include "oxidize/quant.h"
#include "oxidize/log.h"
#include "oxidize/gguf_writer.h"
#include "oxidize/quantize_tool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>


/* SafeTensors format: 8-byte little-endian header length + JSON header + raw tensor data. */

OcError oc_safetensors_parse_header(const char *path,
                                     char **out_json, size_t *out_len)
{
    if (!path || !out_json || !out_len) return OC_ERR_INVALID_ARG;
    *out_json = NULL;
    *out_len = 0;

    FILE *f = fopen(path, "rb");
    if (!f) return OC_ERR_IO;

    /* Read 8-byte header length (explicit little-endian decode for
     * portability to big-endian hosts). */
    uint8_t len_bytes[8];
    if (fread(len_bytes, 1, 8, f) != 8) { fclose(f); return OC_ERR_FORMAT; }
    uint64_t header_len = 0;
    for (int i = 7; i >= 0; i--)
        header_len = (header_len << 8) | len_bytes[i];
    if (header_len == 0 || header_len > (1ULL << 30)) {
        fclose(f);
        return OC_ERR_FORMAT;
    }

    /* Read JSON header. */
    char *json = malloc(header_len + 1);
    if (!json) { fclose(f); return OC_ERR_OOM; }
    if (fread(json, 1, header_len, f) != header_len) {
        free(json);
        fclose(f);
        return OC_ERR_FORMAT;
    }
    json[header_len] = '\0';

    /* Get the data offset (where tensor data starts). */
    /* The data section starts at 8 + header_len. */

    *out_json = json;
    *out_len = header_len;
    fclose(f);
    return OC_OK;
}


/* Simple substring match helpers. */
static bool contains(const char *s, const char *sub)
{
    return s && sub && strstr(s, sub) != NULL;
}

const char *oc_detect_arch_from_tensors(const char *const *names, size_t n)
{
    if (!names || n == 0) return "unknown";
    /* Check first few tensor names for architecture-specific patterns. */
    for (size_t i = 0; i < n && i < 50; i++) {
        if (!names[i]) continue;
        if (contains(names[i], "qwen")) return "qwen2";
        if (contains(names[i], "gemma")) return "gemma";
        if (contains(names[i], "phi")) return "phi3";
        if (contains(names[i], "deepseek")) return "deepseek2";
        if (contains(names[i], "falcon")) return "falcon";
        if (contains(names[i], "gpt2")) return "gpt2";
        if (contains(names[i], "mistral")) return "mistral";
        if (contains(names[i], "mixtral")) return "mixtral";
        if (contains(names[i], "llama")) return "llama";
        if (contains(names[i], "model.layers") && contains(names[i], "self_attn")) return "llama";
        if (contains(names[i], "model.layers") && contains(names[i], "mlp")) return "llama";
    }
    return "unknown"; /* no recognized pattern — surface detection failure */
}


const char *oc_map_tensor_name(const char *st_name, const char *arch)
{
    if (!st_name) return NULL;
    (void)arch; /* arch can affect mapping in some cases */

    /* Common HuggingFace → GGUF name mappings.
     * These work for Llama/Mistral/Qwen architectures. */

    /* Global tensors. */
    if (strcmp(st_name, "model.embed_tokens.weight") == 0 ||
        strcmp(st_name, "embed_tokens.weight") == 0) return "token_embd.weight";
    if (strcmp(st_name, "model.norm.weight") == 0 ||
        strcmp(st_name, "norm.weight") == 0) return "output_norm.weight";
    if (strcmp(st_name, "lm_head.weight") == 0) return "output.weight";

    /* Per-layer: model.layers.{N}.{suffix} → blk.{N}.{gguf_suffix} */
    if (strncmp(st_name, "model.layers.", 13) == 0) {
        const char *p = st_name + 13;
        char *end = NULL;
        unsigned long layer = strtoul(p, &end, 10);
        if (end && *end == '.') {
            /* Thread-local: results are stable across threads, but a second
             * call on the same thread reuses this buffer (see header). */
            static _Thread_local char buf[256];
            const char *suffix = end + 1;

            /* Attention weights. */
            if (strcmp(suffix, "input_layernorm.weight") == 0)
                snprintf(buf, sizeof(buf), "blk.%lu.attn_norm.weight", layer);
            else if (strcmp(suffix, "self_attn.q_proj.weight") == 0)
                snprintf(buf, sizeof(buf), "blk.%lu.attn_q.weight", layer);
            else if (strcmp(suffix, "self_attn.k_proj.weight") == 0)
                snprintf(buf, sizeof(buf), "blk.%lu.attn_k.weight", layer);
            else if (strcmp(suffix, "self_attn.v_proj.weight") == 0)
                snprintf(buf, sizeof(buf), "blk.%lu.attn_v.weight", layer);
            else if (strcmp(suffix, "self_attn.o_proj.weight") == 0)
                snprintf(buf, sizeof(buf), "blk.%lu.attn_output.weight", layer);
            /* Post-attention norm. */
            else if (strcmp(suffix, "post_attention_layernorm.weight") == 0)
                snprintf(buf, sizeof(buf), "blk.%lu.ffn_norm.weight", layer);
            /* FFN weights. */
            else if (strcmp(suffix, "mlp.gate_proj.weight") == 0)
                snprintf(buf, sizeof(buf), "blk.%lu.ffn_gate.weight", layer);
            else if (strcmp(suffix, "mlp.up_proj.weight") == 0)
                snprintf(buf, sizeof(buf), "blk.%lu.ffn_up.weight", layer);
            else if (strcmp(suffix, "mlp.down_proj.weight") == 0)
                snprintf(buf, sizeof(buf), "blk.%lu.ffn_down.weight", layer);
            /* MoE weights. */
            else if (strcmp(suffix, "mlp.gate.weight") == 0)
                snprintf(buf, sizeof(buf), "blk.%lu.ffn_gate_inp.weight", layer);
            else if (strcmp(suffix, "mlp.experts.gate_proj.weight") == 0)
                snprintf(buf, sizeof(buf), "blk.%lu.ffn_gate_exps.weight", layer);
            else if (strcmp(suffix, "mlp.experts.up_proj.weight") == 0)
                snprintf(buf, sizeof(buf), "blk.%lu.ffn_up_exps.weight", layer);
            else if (strcmp(suffix, "mlp.experts.down_proj.weight") == 0)
                snprintf(buf, sizeof(buf), "blk.%lu.ffn_down_exps.weight", layer);
            /* DeepSeek MLA weights. */
            else if (strcmp(suffix, "self_attn.q_a_proj.weight") == 0)
                snprintf(buf, sizeof(buf), "blk.%lu.attn_q_a.weight", layer);
            else if (strcmp(suffix, "self_attn.q_a_layernorm.weight") == 0)
                snprintf(buf, sizeof(buf), "blk.%lu.attn_q_a_norm.weight", layer);
            else if (strcmp(suffix, "self_attn.q_b_proj.weight") == 0)
                snprintf(buf, sizeof(buf), "blk.%lu.attn_q_b.weight", layer);
            else if (strcmp(suffix, "self_attn.kv_a_proj_with_mqa.weight") == 0)
                snprintf(buf, sizeof(buf), "blk.%lu.attn_kv_a_mqa.weight", layer);
            else if (strcmp(suffix, "self_attn.kv_a_layernorm.weight") == 0)
                snprintf(buf, sizeof(buf), "blk.%lu.attn_kv_a_norm.weight", layer);
            else if (strcmp(suffix, "self_attn.k_b_proj.weight") == 0)
                snprintf(buf, sizeof(buf), "blk.%lu.attn_k_b.weight", layer);
            else if (strcmp(suffix, "self_attn.v_b_proj.weight") == 0)
                snprintf(buf, sizeof(buf), "blk.%lu.attn_v_b.weight", layer);
            else
                return st_name; /* unmapped, keep original */
            return buf;
        }
    }
    return st_name; /* unmapped, keep original */
}


OcError oc_safetensors_to_gguf(const OcConvertConfig *cfg)
{
    if (!cfg || !cfg->input_path || !cfg->output_path)
        return OC_ERR_INVALID_ARG;

    /* Parse the SafeTensors header. */
    char *header_json = NULL;
    size_t header_len = 0;
    OcError e = oc_safetensors_parse_header(cfg->input_path, &header_json, &header_len);
    if (e != OC_OK) {
        fprintf(stderr, "error: failed to parse SafeTensors header (%s)\n", oc_error_msg(e));
        return e;
    }

    if (cfg->verbose) {
        fprintf(stderr, "convert: %s -> %s\n", cfg->input_path, cfg->output_path);
        fprintf(stderr, "  header size: %zu bytes\n", header_len);
    }

    /* Data section starts after 8-byte length + header_len. */
    size_t data_offset = 8 + header_len;

    /* Open the SafeTensors file for reading tensor data. */
    FILE *st_file = fopen(cfg->input_path, "rb");
    if (!st_file) {
        free(header_json);
        return OC_ERR_IO;
    }

    /* Seek past header to data section. */
    if (fseek(st_file, (long)data_offset, SEEK_SET) != 0) {
        fclose(st_file);
        free(header_json);
        return OC_ERR_IO;
    }

    /* Parse JSON header: extract tensor entries.
     * SafeTensors JSON format: {"tensor_name": {"dtype": "F32", "shape": [d0,d1], "data_offsets": [start, end]}, ...}
     * We do a minimal JSON parse to extract tensor names, dtypes, shapes, and data offsets. */

    /* Collect tensor info. */
    typedef struct {
        char name[256];
        char dtype[16];
        size_t offset_start;
        size_t offset_end;
        size_t n_dims;
        size_t dims[4];
    } TensorInfo;

    TensorInfo tensors[512];
    size_t n_tensors = 0;
    const char *p = header_json;

    /* Skip opening brace. */
    while (*p && *p != '{') p++;
    if (*p == '{') p++;

    while (*p && *p != '}' && n_tensors < 512) {
        /* Skip whitespace and commas. */
        while (*p && (*p == ' ' || *p == ',' || *p == '\n' || *p == '\t')) p++;
        if (*p == '}' || !*p) break;

        /* Expect a string key (tensor name). */
        if (*p != '"') break;
        p++; /* skip opening quote */
        size_t name_len = 0;
        while (*p && *p != '"' && name_len < 255) {
            tensors[n_tensors].name[name_len++] = *p++;
        }
        tensors[n_tensors].name[name_len] = '\0';
        if (*p == '"') p++;

        /* Skip whitespace and colon. */
        while (*p && (*p == ' ' || *p == ':')) p++;

        /* Expect opening brace for the tensor info object. */
        if (*p != '{') break;
        p++;

        /* Parse fields: dtype, shape, data_offsets. */
        while (*p && *p != '}') {
            /* Skip whitespace and commas. */
            while (*p && (*p == ' ' || *p == ',' || *p == '\n' || *p == '\t')) p++;
            if (*p == '}' || !*p) break;

            /* Parse field name. */
            if (*p != '"') break;
            p++;
            char field_name[32];
            size_t fn_len = 0;
            while (*p && *p != '"' && fn_len < 31) {
                field_name[fn_len++] = *p++;
            }
            field_name[fn_len] = '\0';
            if (*p == '"') p++;

            /* Skip whitespace and colon. */
            while (*p && (*p == ' ' || *p == ':')) p++;

            if (strcmp(field_name, "dtype") == 0) {
                /* Parse string value. */
                if (*p == '"') {
                    p++;
                    size_t dt_len = 0;
                    while (*p && *p != '"' && dt_len < 15) {
                        tensors[n_tensors].dtype[dt_len++] = *p++;
                    }
                    tensors[n_tensors].dtype[dt_len] = '\0';
                    if (*p == '"') p++;
                }
            } else if (strcmp(field_name, "shape") == 0) {
                /* Parse array of integers. */
                if (*p == '[') p++;
                size_t dim_count = 0;
                while (*p && *p != ']') {
                    while (*p && (*p == ' ' || *p == ',')) p++;
                    if (*p == ']') break;
                    /* Parse number. */
                    size_t val = 0;
                    while (*p && isdigit((unsigned char)*p)) {
                        val = val * 10 + (size_t)(*p - '0');
                        p++;
                    }
                    if (dim_count < 4) {
                        tensors[n_tensors].dims[dim_count] = val;
                    }
                    dim_count++;
                    while (*p && *p != ',' && *p != ']') p++;
                }
                tensors[n_tensors].n_dims = dim_count;
                if (*p == ']') p++;
            } else if (strcmp(field_name, "data_offsets") == 0) {
                /* Parse array of two integers [start, end]. */
                if (*p == '[') p++;
                size_t off_idx = 0;
                while (*p && *p != ']') {
                    while (*p && (*p == ' ' || *p == ',')) p++;
                    if (*p == ']') break;
                    size_t val = 0;
                    while (*p && isdigit((unsigned char)*p)) {
                        val = val * 10 + (size_t)(*p - '0');
                        p++;
                    }
                    if (off_idx == 0) tensors[n_tensors].offset_start = val;
                    else tensors[n_tensors].offset_end = val;
                    off_idx++;
                    while (*p && *p != ',' && *p != ']') p++;
                }
                if (*p == ']') p++;
            } else {
                /* Skip unknown field value. */
                if (*p == '"') {
                    p++;
                    while (*p && *p != '"') p++;
                    if (*p == '"') p++;
                } else if (*p == '[') {
                    int depth = 1;
                    p++;
                    while (*p && depth > 0) {
                        if (*p == '[') depth++;
                        else if (*p == ']') depth--;
                        p++;
                    }
                } else if (*p == '{') {
                    int depth = 1;
                    p++;
                    while (*p && depth > 0) {
                        if (*p == '{') depth++;
                        else if (*p == '}') depth--;
                        p++;
                    }
                } else {
                    while (*p && *p != ',' && *p != '}') p++;
                }
            }
        }
        /* Skip closing brace of tensor info. */
        if (*p == '}') p++;

        /* Skip "__metadata__" entries (they have dict values). */
        if (strcmp(tensors[n_tensors].name, "__metadata__") != 0) {
            n_tensors++;
        }
    }

    if (cfg->verbose) {
        fprintf(stderr, "  parsed %zu tensors from header\n", n_tensors);
    }

    /* Determine architecture. */
    const char *names[512] = {0};  /* zero-init: silences -Werror=maybe-uninitialized when n_tensors==0 */
    size_t n_names = n_tensors < 512 ? n_tensors : 512;
    for (size_t i = 0; i < n_names; i++)
        names[i] = tensors[i].name;
    const char *arch = oc_detect_arch_from_tensors(names, n_names);

    if (cfg->verbose) {
        fprintf(stderr, "  detected architecture: %s\n", arch);
    }

    /* Determine target quantization type (default F32). */
    OcGgufQuantizationType target_qtype = OC_QUANT_F32;
    if (cfg->target_type) {
        e = oc_quantize_parse_type(cfg->target_type, &target_qtype);
        if (e != OC_OK) target_qtype = OC_QUANT_F32;
    }
    uint32_t target_ggml_type = oc_quant_type_to_ggml_id(target_qtype);

    /* Create GGUF writer. */
    OcGgufWriter writer;
    e = oc_gguf_writer_init(cfg->output_path, arch, &writer);
    if (e != OC_OK) {
        fclose(st_file);
        free(header_json);
        return e;
    }

    /* Write metadata. */
    oc_gguf_writer_add_string(&writer, "general.architecture", arch);
    oc_gguf_writer_add_uint32(&writer, "general.file_type", target_ggml_type);

    /* Write tensors. */
    for (size_t i = 0; i < n_tensors; i++) {
        const char *gguf_name = oc_map_tensor_name(tensors[i].name, arch);
        size_t n_elements = 1;
        for (size_t d = 0; d < tensors[i].n_dims; d++)
            n_elements *= tensors[i].dims[d];
        if (n_elements == 0) n_elements = 1;

        /* Read tensor data from SafeTensors file. */
        size_t tsize = tensors[i].offset_end - tensors[i].offset_start;
        float *f32_data = malloc(n_elements * sizeof(float));
        if (!f32_data) {
            oc_gguf_writer_free(&writer);
            fclose(st_file);
            free(header_json);
            return OC_ERR_OOM;
        }

        /* Seek to tensor data. */
        if (fseek(st_file, (long)(data_offset + tensors[i].offset_start), SEEK_SET) != 0) {
            free(f32_data);
            continue;
        }

        /* Read and convert to F32 based on dtype. */
        if (strcmp(tensors[i].dtype, "F32") == 0) {
            if (fread(f32_data, sizeof(float), n_elements, st_file) != n_elements) {
                free(f32_data);
                continue;
            }
        } else if (strcmp(tensors[i].dtype, "F16") == 0) {
            uint8_t *raw = malloc(tsize);
            /* A short read would leave `raw` partly uninitialized and write
             * that garbage into the GGUF, so skip the tensor instead — same
             * behavior as the F32 branch above. */
            if (!raw || fread(raw, 1, tsize, st_file) != tsize) {
                free(raw);
                free(f32_data);
                continue;
            }
            {
                for (size_t j = 0; j < n_elements; j++) {
                    uint16_t bits = (uint16_t)raw[2*j] | ((uint16_t)raw[2*j+1] << 8);
                    uint32_t f32_bits = ((uint32_t)(bits & 0x8000) << 16) |
                                        (((uint32_t)(bits & 0x7C00) + 0x3800) << 13) |
                                        ((uint32_t)(bits & 0x03FF) << 13);
                    memcpy(&f32_data[j], &f32_bits, sizeof(float));
                }
                free(raw);
            }
        } else if (strcmp(tensors[i].dtype, "BF16") == 0) {
            uint8_t *raw = malloc(tsize);
            if (!raw || fread(raw, 1, tsize, st_file) != tsize) {
                free(raw);
                free(f32_data);
                continue;
            }
            {
                for (size_t j = 0; j < n_elements; j++) {
                    uint16_t bits = (uint16_t)raw[2*j] | ((uint16_t)raw[2*j+1] << 8);
                    uint32_t f32_bits = (uint32_t)bits << 16;
                    memcpy(&f32_data[j], &f32_bits, sizeof(float));
                }
                free(raw);
            }
        } else {
            memset(f32_data, 0, n_elements * sizeof(float));
        }

        /* Write to GGUF as F32. */
        uint64_t gguf_dims[4];
        uint32_t gguf_ndim = (uint32_t)tensors[i].n_dims;
        for (size_t d = 0; d < gguf_ndim && d < 4; d++)
            gguf_dims[d] = (uint64_t)tensors[i].dims[d];

        oc_gguf_writer_add_tensor(&writer, gguf_name, gguf_ndim, gguf_dims,
                                  0 /* F32 */, f32_data,
                                  (uint64_t)(n_elements * sizeof(float)));
        free(f32_data);
    }

    /* Finalize and write the GGUF file. */
    e = oc_gguf_writer_finalize(&writer);
    oc_gguf_writer_free(&writer);
    fclose(st_file);
    free(header_json);

    if (e != OC_OK) {
        fprintf(stderr, "error: failed to finalize GGUF file (%s)\n", oc_error_msg(e));
        return e;
    }

    if (cfg->verbose) {
        fprintf(stderr, "  conversion complete: %zu tensors written\n", n_tensors);
    }

    (void)target_ggml_type;
    return OC_OK;
}
