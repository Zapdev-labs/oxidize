/*
 * safetensors_to_gguf.c — SafeTensors to GGUF conversion implementation.
 */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/safetensors_to_gguf.h"

#include "oxidize/quant.h"
#include "oxidize/log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── SafeTensors header parsing ──────────────────────────────────────── */

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

/* ─── Architecture detection ──────────────────────────────────────────── */

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
    return "llama"; /* default */
}

/* ─── Tensor name mapping ─────────────────────────────────────────────── */

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

/* ─── Main conversion ──────────────────────────────────────────────────── */

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
        fprintf(stderr, "convert: %s → %s\n", cfg->input_path, cfg->output_path);
        fprintf(stderr, "  header size: %zu bytes\n", header_len);
    }

    /* Full conversion (JSON tensor-table parsing, name mapping, quantization,
     * GGUF write via oc_gguf_writer) is not implemented yet. Return an error
     * rather than reporting success without producing an output file. */
    oc_log(OC_LOG_ERROR,
           "safetensors_to_gguf: conversion not implemented; no output written");
    free(header_json);
    return OC_ERR_MODEL;
}
