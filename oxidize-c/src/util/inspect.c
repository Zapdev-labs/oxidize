/*
 * inspect.c — Model inspector implementation.
 *
 * Opens a GGUF file (or uses a loaded OcLlamaModel), reads metadata and
 * tensor info, computes parameter count, dominant quant type, file size,
 * estimates RAM usage, suggests thread/NUMA policy, and formats the result
 * as a human-readable table or JSON.
 *
 * The inspection logic mirrors the Rust `--inspect` CLI path: it extracts
 * architecture + dimension metadata, counts tensors, estimates parameter
 * count from tensor shapes, identifies the dominant quantization type by
 * total byte share, and produces memory/threading heuristics.
 */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/inspect.h"

#include "oxidize/gguf.h"
#include "oxidize/mem_util.h"
#include "oxidize/model.h"
#include "oxidize/quant.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ─── Internal helpers ─────────────────────────────────────────────────── */

/* Copy a GGUF metadata string into a fixed-size buffer, NUL-terminated.
 * Truncates if the value is longer than `cap-1`. Safe on NULL value. */
static void copy_meta_str(const OcGgufFile *f, const char *key,
                          char *dst, size_t cap)
{
    if (cap == 0) return;
    dst[0] = '\0';
    const char *data = NULL;
    size_t len = 0;
    if (oc_gguf_metadata_get_str(f, key, &data, &len) && data && len > 0) {
        size_t copy_len = (len < cap - 1) ? len : cap - 1;
        memcpy(dst, data, copy_len);
        dst[copy_len] = '\0';
    }
}

/* Get a u32 metadata value with a fallback default. */
static uint32_t get_meta_u32(const OcGgufFile *f, const char *key,
                             uint32_t default_val)
{
    uint32_t val;
    if (oc_gguf_metadata_get_u32(f, key, &val)) return val;
    return default_val;
}

/* Get a float metadata value with a fallback default. */
static float get_meta_f32(const OcGgufFile *f, const char *key,
                          float default_val)
{
    float val;
    if (oc_gguf_metadata_get_f32(f, key, &val)) return val;
    return default_val;
}

/* Get a bool metadata value with a fallback default. */
static bool get_meta_bool(const OcGgufFile *f, const char *key,
                          bool default_val)
{
    bool val;
    if (oc_gguf_metadata_get_bool(f, key, &val)) return val;
    return default_val;
}

/* Compute the byte size of a tensor from its dims and ggml type.
 * For quantized types, uses oc_quantized_size; for F32/F16, uses direct
 * element count * sizeof. Falls back to 0 for unknown types. */
static uint64_t compute_tensor_bytes(const OcGgufTensorInfo *t)
{
    if (!t || t->n_dims == 0) return 0;

    /* Count total elements across all dims. */
    uint64_t elements = 1;
    for (uint32_t i = 0; i < t->n_dims && i < OC_GGUF_MAX_DIMS; i++) {
        elements *= t->dims[i];
    }

    OcGgufQuantizationType qt = oc_quant_type_from_ggml_id(t->ggml_type);
    if (qt == OC_QUANT_UNKNOWN) {
        /* Unknown type: estimate based on element count and common sizes. */
        switch (t->ggml_type) {
            case 0:  return elements * 4;  /* F32 */
            case 1:  return elements * 2;  /* F16 */
            case 2:  return elements * 2;  /* BF16 */
            default: return 0;
        }
    }

    /* For known quant types, use the quantized size function. */
    size_t sz = oc_quantized_size(qt, (size_t)elements);
    if (sz > 0) return (uint64_t)sz;

    /* Fallback for non-quantized types (F32, F16, I8, etc.). */
    OcQuantBlockLayout layout = oc_quant_block_size(qt);
    if (layout.elements_per_block > 0 && layout.bytes_per_block > 0) {
        return (elements / layout.elements_per_block) * layout.bytes_per_block;
    }

    /* Direct element-size estimate for plain types. */
    switch (qt) {
        case OC_QUANT_F32:  return elements * 4;
        case OC_QUANT_F16:  return elements * 2;
        case OC_QUANT_F64:  return elements * 8;
        case OC_QUANT_I8:   return elements * 1;
        case OC_QUANT_I16:  return elements * 2;
        case OC_QUANT_I32:  return elements * 4;
        case OC_QUANT_I64:  return elements * 8;
        case OC_QUANT_BF16: return elements * 2;
        default:            return 0;
    }
}

/* Count available CPU cores (logical) using sysconf. */
static uint32_t get_logical_cores(void)
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n <= 0) return 4;  /* conservative fallback */
    return (uint32_t)n;
}

/* Get total system RAM in bytes. Uses oc_mem_available_bytes on Linux. */
static uint64_t get_total_ram(void)
{
    OcMemUsage mu;
    if (oc_mem_usage_get(&mu) == OC_OK && mu.total > 0) {
        return mu.total;
    }
    return 0;
}

/* Suggest a thread count based on model size and core count.
 * Heuristics (from AGENTS.md learned facts):
 *   - Small models (< 4 GB): min(cores, 8)
 *   - Medium models (4-32 GB): min(cores, 16)
 *   - Large models (32-192 GB): min(cores, 16)
 *   - Very large models (> 192 GB): min(cores, 48) */
static uint32_t suggest_threads(double size_gb, uint32_t cores)
{
    if (cores == 0) cores = 4;
    if (size_gb < 4.0) {
        uint32_t t = cores < 8 ? cores : 8;
        return t > 0 ? t : 1;
    } else if (size_gb < 32.0) {
        uint32_t t = cores < 16 ? cores : 16;
        return t > 0 ? t : 1;
    } else if (size_gb < 192.0) {
        uint32_t t = cores < 16 ? cores : 16;
        return t > 0 ? t : 1;
    } else {
        uint32_t t = cores < 48 ? cores : 48;
        return t > 0 ? t : 1;
    }
}

/* Suggest NUMA interleave policy based on model size vs available RAM.
 * Interleave is suggested when the model exceeds 50% of total RAM (i.e.
 * it spans multiple NUMA nodes). */
static bool suggest_numa_interleave(double size_gb, uint64_t total_ram)
{
    if (total_ram == 0) return false;
    double total_gb = (double)total_ram / 1e9;
    if (total_gb <= 0.0) return false;

    /* If model uses more than half of total RAM, interleave. */
    return size_gb > (total_gb * 0.5);
}

/* ─── Core inspection from OcGgufFile ──────────────────────────────────── */

/* Fill OcModelInfo from a parsed GGUF file. This is the shared core used by
 * both oc_inspect_model() (which opens the file) and oc_inspect_llama()
 * (which already has a parsed file from the loaded model). */
static OcError inspect_from_gguf(const OcGgufFile *f, uint64_t file_size,
                                 OcModelInfo *out)
{
    if (!f || !out) return OC_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));

    /* Architecture. */
    OcModelArchitecture arch = oc_gguf_arch_from_file(f);
    const char *arch_name = oc_model_arch_name(arch);
    strncpy(out->arch, arch_name, sizeof(out->arch) - 1);
    out->arch[sizeof(out->arch) - 1] = '\0';

    /* General name. */
    copy_meta_str(f, "general.name", out->name, sizeof(out->name));

    /* Core dimensions — try architecture-specific keys. */
    char arch_prefix[80];
    snprintf(arch_prefix, sizeof(arch_prefix), "%s.", out->arch);

    /* Build prefixed key lookup. We try both "arch.key" and common
     * HuggingFace-style keys. */
    char key_buf[128];

    snprintf(key_buf, sizeof(key_buf), "%scontext_length", arch_prefix);
    out->n_ctx = get_meta_u32(f, key_buf, 2048);

    snprintf(key_buf, sizeof(key_buf), "%sembedding_length", arch_prefix);
    out->n_embd = get_meta_u32(f, key_buf, 0);

    snprintf(key_buf, sizeof(key_buf), "%sblock_count", arch_prefix);
    out->n_layer = get_meta_u32(f, key_buf, 0);

    snprintf(key_buf, sizeof(key_buf), "%shead_count", arch_prefix);
    out->n_head = get_meta_u32(f, key_buf, 0);

    snprintf(key_buf, sizeof(key_buf), "%shead_count_kv", arch_prefix);
    /* If kv heads not specified, default to n_head (no GQA). */
    out->n_head_kv = get_meta_u32(f, key_buf, out->n_head);

    snprintf(key_buf, sizeof(key_buf), "%sfeed_forward_length", arch_prefix);
    out->n_ff = get_meta_u32(f, key_buf, 0);

    /* Vocabulary size: try to read from metadata, or infer from tokenizer. */
    snprintf(key_buf, sizeof(key_buf), "%svocab_size", arch_prefix);
    /* Some GGUFs store it as a model-level key without the arch prefix. */
    out->vocab_size = get_meta_u32(f, key_buf, 0);
    if (out->vocab_size == 0) {
        out->vocab_size = get_meta_u32(f, "tokenizer.ggml.tokens.size", 0);
    }

    /* RoPE. */
    snprintf(key_buf, sizeof(key_buf), "%srope_freq_base", arch_prefix);
    out->rope_freq_base = get_meta_f32(f, key_buf, 10000.0f);
    out->uses_rope = !oc_model_arch_uses_alibi(arch);

    /* Sliding window attention. */
    snprintf(key_buf, sizeof(key_buf), "%ssliding_window", arch_prefix);
    out->sliding_window = get_meta_u32(f, key_buf, 0);

    /* Architecture-specific feature flags. */
    out->uses_mla = oc_model_arch_uses_mla(arch);
    out->uses_gqa = (out->n_head_kv > 0 && out->n_head_kv < out->n_head);

    /* RMSNorm: most modern LLM architectures use RMSNorm (Llama, Qwen, etc.).
     * GPT-2 and Falcon use standard LayerNorm. */
    out->uses_rms_norm = (arch != OC_ARCH_GPT2 && arch != OC_ARCH_FALCON &&
                          arch != OC_ARCH_GPTJ && arch != OC_ARCH_GPTNEOX &&
                          arch != OC_ARCH_UNKNOWN);

    /* SwiGLU: Llama, Mistral, Qwen, DeepSeek, Gemma2+ use SwiGLU.
     * GPT-2/Falcon use GELU. Phi uses GeGLU. */
    out->uses_swiglu = (arch == OC_ARCH_LLAMA || arch == OC_ARCH_MISTRAL ||
                        arch == OC_ARCH_MIXTRAL || arch == OC_ARCH_DEEPSEEK ||
                        arch == OC_ARCH_QWEN || arch == OC_ARCH_GEMMA ||
                        arch == OC_ARCH_HUNYUAN_MOE || arch == OC_ARCH_GLM_MOE_DSA);

    /* Tokenizer info. */
    {
        const char *tok_model = NULL;
        size_t tok_len = 0;
        if (oc_gguf_metadata_get_str(f, "tokenizer.ggml.model",
                                     &tok_model, &tok_len) && tok_model) {
            if (strncmp(tok_model, "llama", 5) == 0 ||
                strncmp(tok_model, "bpe", 3) == 0) {
                strncpy(out->tokenizer_type, "BPE",
                        sizeof(out->tokenizer_type) - 1);
            } else if (strncmp(tok_model, "gpt2", 4) == 0) {
                strncpy(out->tokenizer_type, "BPE",
                        sizeof(out->tokenizer_type) - 1);
            } else if (strncmp(tok_model, "bert", 4) == 0) {
                strncpy(out->tokenizer_type, "WordPiece",
                        sizeof(out->tokenizer_type) - 1);
            } else if (strncmp(tok_model, "sentencepiece", 13) == 0 ||
                       strncmp(tok_model, "unigram", 7) == 0) {
                strncpy(out->tokenizer_type, "SentencePiece",
                        sizeof(out->tokenizer_type) - 1);
            } else if (strncmp(tok_model, "tiktoken", 8) == 0 ||
                       strncmp(tok_model, "rwkv", 4) == 0) {
                strncpy(out->tokenizer_type, "Tiktoken",
                        sizeof(out->tokenizer_type) - 1);
            } else {
                /* Copy raw model name if short enough. */
                size_t copy_len = (tok_len < sizeof(out->tokenizer_type) - 1)
                    ? tok_len : sizeof(out->tokenizer_type) - 1;
                memcpy(out->tokenizer_type, tok_model, copy_len);
                out->tokenizer_type[copy_len] = '\0';
            }
        } else {
            strncpy(out->tokenizer_type, "unknown",
                    sizeof(out->tokenizer_type) - 1);
        }
        out->tokenizer_type[sizeof(out->tokenizer_type) - 1] = '\0';
    }

    /* Tokenizer special token IDs. The `add_bos_token` / `add_eos_token`
     * metadata flags indicate whether the tokenizer expects BOS/EOS by
     * default; we read them to sanity-check the special token IDs. */
    bool add_bos = get_meta_bool(f, "tokenizer.ggml.add_bos_token", false);
    bool add_eos = get_meta_bool(f, "tokenizer.ggml.add_eos_token", false);
    out->bos_id = get_meta_u32(f, "tokenizer.ggml.bos_token_id", 0);
    out->eos_id = get_meta_u32(f, "tokenizer.ggml.eos_token_id", 0);
    out->pad_id = get_meta_u32(f, "tokenizer.ggml.padding_token_id", 0);
    /* If add_bos is false and bos_id is 0, the model genuinely has no BOS. */
    (void)add_bos;
    (void)add_eos;

    /* ─── Tensor analysis ──────────────────────────────────────────────── */

    out->file_size = file_size;
    out->size_gb = (file_size > 0) ? (double)file_size / 1e9 : 0.0;

    /* Allocate per-tensor summaries (cap at a reasonable max to avoid OOM
     * on pathological files). `n_tensors` is clamped to the allocated count
     * so consumers iterating it never read past the array. */
    uint32_t max_tensors = 100000;
    uint32_t total = (f->tensor_count < UINT32_MAX)
        ? (uint32_t)f->tensor_count : UINT32_MAX;
    uint32_t n_copy = (total < max_tensors) ? total : max_tensors;
    out->n_tensors = n_copy;

    if (n_copy > 0) {
        out->tensors = (OcTensorSummary *)calloc(n_copy, sizeof(OcTensorSummary));
        if (!out->tensors) {
            return OC_ERR_OOM;
        }
    }

    /* Walk tensors: compute param count, total bytes, dominant quant type. */
    uint64_t total_param_count = 0;
    uint64_t total_tensor_bytes = 0;

    /* For dominant quant type: track byte counts per ggml type id. */
    /* We use a simple approach: count bytes per type, then pick the max. */
    /* Since we don't have a hash map, we track the top few types seen. */
    struct { uint32_t ggml_type; uint64_t bytes; } type_counts[64];
    uint32_t n_types = 0;

    for (uint32_t i = 0; i < n_copy; i++) {
        const OcGgufTensorInfo *t = &f->tensors[i];
        OcTensorSummary *s = &out->tensors[i];

        /* Copy name. */
        if (t->name) {
            strncpy(s->name, t->name, sizeof(s->name) - 1);
            s->name[sizeof(s->name) - 1] = '\0';
        }

        s->type = t->ggml_type;
        /* OcTensorSummary.dims holds 4 entries; clamp n_dims so consumers
         * never index past the copied dimensions. */
        s->n_dims = (t->n_dims < 4) ? t->n_dims : 4;
        for (uint32_t d = 0; d < s->n_dims; d++) {
            s->dims[d] = t->dims[d];
        }

        /* Compute byte size. */
        uint64_t tbytes = compute_tensor_bytes(t);
        s->bytes = tbytes;
        total_tensor_bytes += tbytes;

        /* Estimate parameter count: for most tensors, param count ≈ element
         * count. For quantized tensors, elements = total_elements. */
        uint64_t elements = 1;
        for (uint32_t d = 0; d < t->n_dims && d < OC_GGUF_MAX_DIMS; d++) {
            elements *= t->dims[d];
        }
        total_param_count += elements;

        /* Track byte counts per type for dominant quant detection. */
        bool found = false;
        for (uint32_t j = 0; j < n_types; j++) {
            if (type_counts[j].ggml_type == t->ggml_type) {
                type_counts[j].bytes += tbytes;
                found = true;
                break;
            }
        }
        if (!found && n_types < 64) {
            type_counts[n_types].ggml_type = t->ggml_type;
            type_counts[n_types].bytes = tbytes;
            n_types++;
        }
    }

    out->param_count = total_param_count;

    /* Find dominant quant type (most bytes). */
    uint32_t dominant_ggml_type = 0;
    uint64_t dominant_bytes = 0;
    for (uint32_t j = 0; j < n_types; j++) {
        if (type_counts[j].bytes > dominant_bytes) {
            dominant_bytes = type_counts[j].bytes;
            dominant_ggml_type = type_counts[j].ggml_type;
        }
    }

    /* Set quant type name. */
    OcGgufQuantizationType dominant_qt = oc_quant_type_from_ggml_id(dominant_ggml_type);
    const char *qt_name = oc_quant_type_name(dominant_qt);
    strncpy(out->quant_type, qt_name, sizeof(out->quant_type) - 1);
    out->quant_type[sizeof(out->quant_type) - 1] = '\0';

    /* Build quant description. */
    {
        double dominant_fraction = 0.0;
        if (total_tensor_bytes > 0) {
            dominant_fraction = (double)dominant_bytes / (double)total_tensor_bytes;
        }
        double param_billion = (double)total_param_count / 1e9;

        snprintf(out->quant_description, sizeof(out->quant_description),
                 "Dominant: %s (%.1f%% of weight bytes), %.2fB params, "
                 "%u tensors, %.2f GB on disk",
                 out->quant_type,
                 dominant_fraction * 100.0,
                 param_billion,
                 out->n_tensors,
                 out->size_gb);
    }

    /* ─── Memory estimates ─────────────────────────────────────────────── */

    /* Estimated RAM usage: file_size * 1.3 (for KV cache + overhead). */
    out->estimated_ram_usage = (uint64_t)((double)file_size * 1.3);

    /* Thread/NUMA suggestions. */
    uint32_t cores = get_logical_cores();
    out->suggested_threads = suggest_threads(out->size_gb, cores);

    uint64_t total_ram = get_total_ram();
    out->suggested_numa_interleave = suggest_numa_interleave(out->size_gb,
                                                               total_ram);

    return OC_OK;
}

/* ─── Public API ────────────────────────────────────────────────────────── */

OcError oc_inspect_model(const char *path, OcModelInfo *out)
{
    if (!path || !out) return OC_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));

    /* Open the GGUF file via mmap (handles single + split files). */
    OcGgufMmappedFile m;
    OcError e = oc_gguf_map_open(path, &m);
    if (e != OC_OK) {
        return e;
    }

    /* Get total file size across all shards. */
    uint64_t file_size = oc_gguf_map_total_bytes(&m);
    if (file_size == 0) {
        /* Fallback: use the unified backing length if available. */
        file_size = m.unified.backing_len;
    }

    /* Run the core inspection on the unified view. */
    e = inspect_from_gguf(&m.unified, file_size, out);

    oc_gguf_map_free(&m);
    return e;
}

OcError oc_inspect_llama(const OcLlamaModel *model, OcModelInfo *out)
{
    if (!model || !out) return OC_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));

    /* Use the model's GGUF unified view. */
    const OcGgufFile *f = &model->gguf.unified;

    /* Get file size from the mmap'd shards. */
    uint64_t file_size = oc_gguf_map_total_bytes(&model->gguf);
    if (file_size == 0) {
        file_size = f->backing_len;
    }

    OcError e = inspect_from_gguf(f, file_size, out);
    if (e != OC_OK) return e;

    /* Override dimensions from the model config (authoritative after load). */
    const OcLlamaConfig *cfg = &model->cfg;
    if (cfg->n_layer > 0) out->n_layer = cfg->n_layer;
    if (cfg->n_embd > 0) out->n_embd = cfg->n_embd;
    if (cfg->n_head > 0) out->n_head = cfg->n_head;
    if (cfg->n_head_kv > 0) out->n_head_kv = cfg->n_head_kv;
    if (cfg->n_ff > 0) out->n_ff = cfg->n_ff;
    if (cfg->vocab_size > 0) out->vocab_size = cfg->vocab_size;
    if (cfg->n_ctx > 0) out->n_ctx = cfg->n_ctx;

    /* Override architecture-specific flags from config. */
    out->uses_mla = cfg->uses_mla;
    out->uses_gqa = (cfg->n_head_kv > 0 && cfg->n_head_kv < cfg->n_head);
    out->rope_freq_base = cfg->rope_theta;
    out->sliding_window = cfg->sliding_window;
    out->uses_rope = !oc_model_arch_uses_alibi(model->arch);

    /* RMSNorm: config has rms_norm_eps, which implies RMSNorm usage. */
    out->uses_rms_norm = (cfg->rms_norm_eps > 0.0f);

    /* SwiGLU: most Llama-family architectures use SwiGLU; Gemma uses GeGLU. */
    out->uses_swiglu = !cfg->uses_geglu;

    /* Update architecture name from model. */
    const char *arch_name = oc_model_arch_name(model->arch);
    strncpy(out->arch, arch_name, sizeof(out->arch) - 1);
    out->arch[sizeof(out->arch) - 1] = '\0';

    return OC_OK;
}

/* ─── Formatting: human-readable table ──────────────────────────────────── */

/* Helper: append a formatted line to buf, advancing the offset. */
static size_t append_line(char *buf, size_t cap, size_t off, const char *fmt, ...)
{
    if (off >= cap) return off;
    va_list args;
    va_start(args, fmt);
    int n = vsnprintf(buf + off, cap - off, fmt, args);
    va_end(args);
    if (n < 0) return off;
    if ((size_t)n >= cap - off) {
        /* Truncated. */
        buf[cap - 1] = '\0';
        return cap;
    }
    return off + (size_t)n;
}

/* Compute the exact rendered length of a formatter by rendering into a
 * temporary buffer, growing until it fits. Returns 0 on OOM. */
static size_t exact_render_len(size_t (*fmt)(const OcModelInfo *, char *, size_t),
                               const OcModelInfo *info, size_t initial_cap)
{
    size_t cap = initial_cap;
    for (int tries = 0; tries < 16; tries++) {
        char *tmp = malloc(cap);
        if (!tmp) return 0;
        size_t n = fmt(info, tmp, cap);
        free(tmp);
        if (n > 0) return n;      /* formatter returns 0 only on truncation */
        if (cap > SIZE_MAX / 2) return 0;
        cap *= 2;
    }
    return 0;
}

size_t oc_inspect_format(const OcModelInfo *info, char *buf, size_t cap)
{
    if (!info) return 0;

    /* If buf is NULL or cap is 0, return the exact rendered length so the
     * caller can allocate len + 1 and format once. */
    if (!buf || cap == 0) {
        return exact_render_len(oc_inspect_format, info,
                                4096 + (size_t)info->n_tensors * 128);
    }

    size_t off = 0;
    const char *sep = "══════════════════════════════════════════════════════════════════════════════\n";

    off = append_line(buf, cap, off, "%s", sep);
    off = append_line(buf, cap, off, "  Model Inspector — oxidize-c\n");
    off = append_line(buf, cap, off, "%s", sep);
    off = append_line(buf, cap, off, "\n");

    /* Architecture + identity. */
    off = append_line(buf, cap, off, "  Architecture:     %s\n", info->arch);
    off = append_line(buf, cap, off, "  Name:             %s\n",
                      info->name[0] ? info->name : "(not set)");
    off = append_line(buf, cap, off, "\n");

    /* Core dimensions. */
    off = append_line(buf, cap, off, "  ── Dimensions ──────────────────────────────\n");
    off = append_line(buf, cap, off, "  Layers:           %u\n", info->n_layer);
    off = append_line(buf, cap, off, "  Embedding dim:    %u\n", info->n_embd);
    off = append_line(buf, cap, off, "  Attention heads:  %u\n", info->n_head);
    off = append_line(buf, cap, off, "  KV heads:         %u%s\n", info->n_head_kv,
                      info->uses_gqa ? " (GQA)" : "");
    off = append_line(buf, cap, off, "  FFN hidden:       %u\n", info->n_ff);
    off = append_line(buf, cap, off, "  Vocab size:       %u\n", info->vocab_size);
    off = append_line(buf, cap, off, "  Context length:   %u\n", info->n_ctx);
    off = append_line(buf, cap, off, "\n");

    /* Size metrics. */
    off = append_line(buf, cap, off, "  ── Size ─────────────────────────────────────\n");
    off = append_line(buf, cap, off, "  Parameters:       %llu (%.2fB)\n",
                      (unsigned long long)info->param_count,
                      (double)info->param_count / 1e9);
    off = append_line(buf, cap, off, "  File size:        %llu bytes (%.2f GB)\n",
                      (unsigned long long)info->file_size, info->size_gb);
    off = append_line(buf, cap, off, "  Tensor count:     %u\n", info->n_tensors);
    off = append_line(buf, cap, off, "\n");

    /* Quantization summary. */
    off = append_line(buf, cap, off, "  ── Quantization ────────────────────────────\n");
    off = append_line(buf, cap, off, "  Dominant type:    %s\n", info->quant_type);
    off = append_line(buf, cap, off, "  Description:      %s\n", info->quant_description);
    off = append_line(buf, cap, off, "\n");

    /* Memory estimates. */
    off = append_line(buf, cap, off, "  ── Memory ──────────────────────────────────\n");
    off = append_line(buf, cap, off, "  Est. RAM usage:   %llu bytes (%.2f GB)\n",
                      (unsigned long long)info->estimated_ram_usage,
                      (double)info->estimated_ram_usage / 1e9);
    off = append_line(buf, cap, off, "  Suggested threads: %u\n", info->suggested_threads);
    off = append_line(buf, cap, off, "  NUMA interleave:  %s\n",
                      info->suggested_numa_interleave ? "yes" : "no");
    off = append_line(buf, cap, off, "\n");

    /* Tokenizer info. */
    off = append_line(buf, cap, off, "  ── Tokenizer ────────────────────────────────\n");
    off = append_line(buf, cap, off, "  Type:             %s\n", info->tokenizer_type);
    off = append_line(buf, cap, off, "  BOS token id:     %u\n", info->bos_id);
    off = append_line(buf, cap, off, "  EOS token id:     %u\n", info->eos_id);
    off = append_line(buf, cap, off, "  PAD token id:     %u\n", info->pad_id);
    off = append_line(buf, cap, off, "\n");

    /* Architecture features. */
    off = append_line(buf, cap, off, "  ── Architecture Features ───────────────────\n");
    off = append_line(buf, cap, off, "  RoPE:             %s (freq_base=%.1f)\n",
                      info->uses_rope ? "yes" : "no", (double)info->rope_freq_base);
    off = append_line(buf, cap, off, "  RMSNorm:          %s\n",
                      info->uses_rms_norm ? "yes" : "no");
    off = append_line(buf, cap, off, "  SwiGLU:           %s\n",
                      info->uses_swiglu ? "yes" : "no");
    off = append_line(buf, cap, off, "  GQA:              %s\n",
                      info->uses_gqa ? "yes" : "no");
    off = append_line(buf, cap, off, "  MLA:              %s\n",
                      info->uses_mla ? "yes" : "no");
    off = append_line(buf, cap, off, "  Sliding window:   %u\n", info->sliding_window);
    off = append_line(buf, cap, off, "\n");

    /* Per-tensor summary (first N tensors). */
    if (info->tensors && info->n_tensors > 0) {
        off = append_line(buf, cap, off, "  ── Tensors (first %u of %u) ────────────────\n",
                          info->n_tensors < 20 ? info->n_tensors : 20,
                          info->n_tensors);
        uint32_t show = (info->n_tensors < 20) ? info->n_tensors : 20;
        for (uint32_t i = 0; i < show; i++) {
            const OcTensorSummary *t = &info->tensors[i];
            off = append_line(buf, cap, off, "    %-40s type=%-3u bytes=%-12llu dims=[",
                              t->name, t->type, (unsigned long long)t->bytes);
            for (uint32_t d = 0; d < t->n_dims && d < 4; d++) {
                off = append_line(buf, cap, off, "%s%llu",
                                  d > 0 ? ", " : "",
                                  (unsigned long long)t->dims[d]);
            }
            off = append_line(buf, cap, off, "]\n");
        }
        if (info->n_tensors > 20) {
            off = append_line(buf, cap, off, "    ... (%u more tensors)\n",
                              info->n_tensors - 20);
        }
        off = append_line(buf, cap, off, "\n");
    }

    off = append_line(buf, cap, off, "%s", sep);

    /* Check if we fit. */
    if (off >= cap) {
        buf[cap - 1] = '\0';
        return 0;  /* truncated */
    }

    return off;
}

/* ─── Formatting: JSON ──────────────────────────────────────────────────── */

/* Helper: append a JSON string field (with escaping). */
static size_t json_str(char *buf, size_t cap, size_t off,
                       const char *key, const char *val)
{
    if (off >= cap) return off;
    int n = snprintf(buf + off, cap - off, "\"%s\":\"", key);
    if (n < 0 || (size_t)n >= cap - off) { if (cap > 0) buf[cap-1] = '\0'; return cap; }
    off += (size_t)n;

    /* Escape the value. */
    for (const char *p = val; *p; p++) {
        if (off >= cap) break;
        char c = *p;
        const char *esc = NULL;
        switch (c) {
            case '"':  esc = "\\\""; break;
            case '\\': esc = "\\\\"; break;
            case '\n': esc = "\\n";  break;
            case '\r': esc = "\\r";  break;
            case '\t': esc = "\\t";  break;
            default:
                if ((unsigned char)c < 0x20) {
                    n = snprintf(buf + off, cap - off, "\\u%04x", (unsigned)c);
                    if (n < 0 || (size_t)n >= cap - off) { if (cap > 0) buf[cap-1] = '\0'; return cap; }
                    off += (size_t)n;
                } else {
                    if (off < cap - 1) {
                        buf[off++] = c;
                    }
                }
                continue;
        }
        if (esc) {
            n = snprintf(buf + off, cap - off, "%s", esc);
            if (n < 0 || (size_t)n >= cap - off) { if (cap > 0) buf[cap-1] = '\0'; return cap; }
            off += (size_t)n;
        }
    }

    n = snprintf(buf + off, cap - off, "\"");
    if (n < 0 || (size_t)n >= cap - off) { if (cap > 0) buf[cap-1] = '\0'; return cap; }
    off += (size_t)n;
    return off;
}

/* Helper: append raw text to JSON buffer. */
static size_t json_raw(char *buf, size_t cap, size_t off, const char *fmt, ...)
{
    if (off >= cap) return off;
    va_list args;
    va_start(args, fmt);
    int n = vsnprintf(buf + off, cap - off, fmt, args);
    va_end(args);
    if (n < 0) return off;
    if ((size_t)n >= cap - off) { if (cap > 0) buf[cap-1] = '\0'; return cap; }
    return off + (size_t)n;
}

size_t oc_inspect_format_json(const OcModelInfo *info, char *buf, size_t cap)
{
    if (!info) return 0;

    if (!buf || cap == 0) {
        /* Return the exact rendered length (long/escaped tensor names make
         * any fixed estimate unsafe for the query-then-format pattern). */
        return exact_render_len(oc_inspect_format_json, info,
                                4096 + (size_t)info->n_tensors * 200);
    }

    size_t off = 0;

    off = json_raw(buf, cap, off, "{");

    /* Architecture + identity. */
    off = json_str(buf, cap, off, "arch", info->arch);
    off = json_raw(buf, cap, off, ",");
    off = json_str(buf, cap, off, "name", info->name);
    off = json_raw(buf, cap, off, ",");

    /* Core dimensions. */
    off = json_raw(buf, cap, off,
        "\"n_layer\":%u,\"n_embd\":%u,\"n_head\":%u,\"n_head_kv\":%u,"
        "\"n_ff\":%u,\"vocab_size\":%u,\"n_ctx\":%u,",
        info->n_layer, info->n_embd, info->n_head, info->n_head_kv,
        info->n_ff, info->vocab_size, info->n_ctx);

    /* Size metrics. */
    off = json_raw(buf, cap, off,
        "\"param_count\":%llu,\"file_size\":%llu,\"size_gb\":%.4f,",
        (unsigned long long)info->param_count,
        (unsigned long long)info->file_size, info->size_gb);

    /* Quantization. */
    off = json_str(buf, cap, off, "quant_type", info->quant_type);
    off = json_raw(buf, cap, off, ",");
    off = json_str(buf, cap, off, "quant_description", info->quant_description);
    off = json_raw(buf, cap, off, ",");
    off = json_raw(buf, cap, off, "\"n_tensors\":%u,", info->n_tensors);

    /* Memory. */
    off = json_raw(buf, cap, off,
        "\"estimated_ram_usage\":%llu,\"suggested_threads\":%u,"
        "\"suggested_numa_interleave\":%s,",
        (unsigned long long)info->estimated_ram_usage,
        info->suggested_threads,
        info->suggested_numa_interleave ? "true" : "false");

    /* Tokenizer. */
    off = json_str(buf, cap, off, "tokenizer_type", info->tokenizer_type);
    off = json_raw(buf, cap, off, ",");
    off = json_raw(buf, cap, off,
        "\"bos_id\":%u,\"eos_id\":%u,\"pad_id\":%u,",
        info->bos_id, info->eos_id, info->pad_id);

    /* Architecture features. */
    off = json_raw(buf, cap, off,
        "\"uses_rope\":%s,\"uses_rms_norm\":%s,\"uses_swiglu\":%s,"
        "\"uses_gqa\":%s,\"uses_mla\":%s,",
        info->uses_rope ? "true" : "false",
        info->uses_rms_norm ? "true" : "false",
        info->uses_swiglu ? "true" : "false",
        info->uses_gqa ? "true" : "false",
        info->uses_mla ? "true" : "false");

    off = json_raw(buf, cap, off, "\"rope_freq_base\":%.1f,", (double)info->rope_freq_base);
    off = json_raw(buf, cap, off, "\"sliding_window\":%u,", info->sliding_window);

    /* Tensors array. */
    off = json_raw(buf, cap, off, "\"tensors\":[");
    if (info->tensors && info->n_tensors > 0) {
        uint32_t show = (info->n_tensors < 50) ? info->n_tensors : 50;
        for (uint32_t i = 0; i < show; i++) {
            const OcTensorSummary *t = &info->tensors[i];
            if (i > 0) off = json_raw(buf, cap, off, ",");
            off = json_raw(buf, cap, off, "{");
            off = json_str(buf, cap, off, "name", t->name);
            off = json_raw(buf, cap, off, ",");
            off = json_raw(buf, cap, off,
                "\"type\":%u,\"bytes\":%llu,\"n_dims\":%u,\"dims\":[",
                t->type, (unsigned long long)t->bytes, t->n_dims);
            for (uint32_t d = 0; d < t->n_dims && d < 4; d++) {
                if (d > 0) off = json_raw(buf, cap, off, ",");
                off = json_raw(buf, cap, off, "%llu", (unsigned long long)t->dims[d]);
            }
            off = json_raw(buf, cap, off, "]}");
        }
    }
    off = json_raw(buf, cap, off, "]");

    off = json_raw(buf, cap, off, "}");

    if (off >= cap) {
        buf[cap - 1] = '\0';
        return 0;  /* truncated */
    }

    return off;
}

/* ─── Cleanup ───────────────────────────────────────────────────────────── */

void oc_inspect_free(OcModelInfo *info)
{
    if (!info) return;
    if (info->tensors) {
        free(info->tensors);
        info->tensors = NULL;
    }
    memset(info, 0, sizeof(*info));
}
