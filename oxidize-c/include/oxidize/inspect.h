/*
 * inspect.h — Model inspector: detailed analysis of a loaded GGUF model.
 *
 * Port of the `--inspect` CLI feature from oxidize-core. Provides a
 * structured `OcModelInfo` containing: architecture metadata, layer/embd/
 * head counts, parameter count, file size, dominant quant type, per-tensor
 * summary, memory estimates, thread/NUMA suggestions, tokenizer info, and
 * architecture-specific feature flags (RoPE, RMSNorm, SwiGLU, GQA, MLA).
 *
 * The inspector can open a GGUF file directly via `oc_inspect_model()`, or
 * analyze an already-loaded `OcLlamaModel` via `oc_inspect_llama()`. Output
 * is formatted as a human-readable table (`oc_inspect_format`) or JSON
 * (`oc_inspect_format_json`).
 */
#ifndef OXIDIZE_INSPECT_H
#define OXIDIZE_INSPECT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/llama.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Per-tensor summary ──────────────────────────────────────────────────
 *
 * A compact view of a single GGUF tensor. `name` is NUL-terminated (copied
 * from the GGUF tensor table). `type` is the raw ggml dtype id (0=F32, 1=F16,
 * 8=Q8_0, ...). `bytes` is the on-disk byte size. `dims[i]` follows GGUF
 * order (dims[0] innermost); `n_dims` is the count of valid entries. */
typedef struct OcTensorSummary {
    char     name[128];
    uint32_t type;
    uint64_t bytes;
    uint32_t n_dims;
    uint64_t dims[4];
} OcTensorSummary;

/* ─── Model info ──────────────────────────────────────────────────────────
 *
 * The full inspection result. Strings are NUL-terminated; numeric fields
 * default to 0. `tensors` is a heap-allocated array of `n_tensors` entries
 * (freed by `oc_inspect_free()`). All other fields are by-value. */
typedef struct OcModelInfo {
    /* Architecture + identity. */
    char     arch[64];        /* "llama", "qwen2", "deepseek2", ...        */
    char     name[128];       /* general.name (may be empty)                */

    /* Core dimensions. */
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_ff;
    uint32_t vocab_size;
    uint32_t n_ctx;

    /* Size metrics. */
    uint64_t param_count;     /* estimated total parameters                 */
    uint64_t file_size;       /* bytes on disk                              */
    double   size_gb;         /* file_size / 1e9                            */

    /* Quantization summary. */
    char     quant_type[32];        /* dominant quant type name             */
    char     quant_description[256];/* human-readable quant summary         */
    uint32_t n_tensors;             /* count of tensors in the file         */

    /* Per-tensor summary (heap-allocated, freed by oc_inspect_free). */
    OcTensorSummary *tensors;

    /* Memory estimates. */
    uint64_t estimated_ram_usage;      /* bytes (file_size * 1.3)         */
    uint32_t suggested_threads;        /* heuristic thread count           */
    bool     suggested_numa_interleave; /* true → interleave across sockets */

    /* Tokenizer info. */
    char     tokenizer_type[32];  /* "BPE", "SentencePiece", "WordPiece", "Tiktoken", "unknown" */
    uint32_t bos_id;
    uint32_t eos_id;
    uint32_t pad_id;

    /* Architecture-specific feature flags. */
    bool     uses_rope;
    bool     uses_rms_norm;
    bool     uses_swiglu;
    bool     uses_gqa;
    bool     uses_mla;
    float    rope_freq_base;
    uint32_t sliding_window;
} OcModelInfo;

/* Inspect a GGUF file on disk: open, parse metadata + tensor table, compute
 * all summary fields, fill `*out`. Returns OC_OK, OC_ERR_IO (file open),
 * OC_ERR_FORMAT (bad GGUF), OC_ERR_OOM, or OC_ERR_INVALID_ARG (NULL args).
 * On error, `*out` is zeroed. On success, caller owns `out->tensors` and
 * must call `oc_inspect_free(out)` when done. */
OcError oc_inspect_model(const char *path, OcModelInfo *out);

/* Inspect an already-loaded OcLlamaModel: derive all summary fields from
 * the model's config + GGUF metadata. Does NOT re-open the file. Returns
 * OC_OK or OC_ERR_INVALID_ARG (NULL args). Caller owns `out->tensors` and
 * must call `oc_inspect_free(out)`. */
OcError oc_inspect_llama(const OcLlamaModel *model, OcModelInfo *out);

/* Format `info` as a human-readable table into `buf` (up to `cap-1` chars,
 * NUL-terminated). Returns the full length of the formatted output
 * (excluding NUL), like snprintf: if the return value is >= `cap`, the
 * output was truncated. If `buf` is NULL or `cap` is 0, nothing is
 * written and the required length is returned. */
size_t oc_inspect_format(const OcModelInfo *info, char *buf, size_t cap);

/* Format `info` as a single-line JSON object into `buf` (up to `cap-1`
 * chars, NUL-terminated). Returns the full length of the formatted output
 * (excluding NUL), like snprintf: if the return value is >= `cap`, the
 * output was truncated. If `buf` is NULL or `cap` is 0, nothing is
 * written and the required length is returned. */
size_t oc_inspect_format_json(const OcModelInfo *info, char *buf, size_t cap);

/* Free the heap-allocated fields of `info` (currently `info->tensors`).
 * Safe on NULL, zeroed, or already-freed OcModelInfo. Does NOT free `info`
 * itself (it is typically stack or caller-owned). Zeros the struct. */
void oc_inspect_free(OcModelInfo *info);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_INSPECT_H */
