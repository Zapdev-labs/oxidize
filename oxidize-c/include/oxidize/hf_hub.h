#ifndef OXIDIZE_HF_HUB_H
#define OXIDIZE_HF_HUB_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif


/* Default cache root: ~/.cache/oxidize/hf (computed at runtime; the macro
 * is the relative suffix under $HOME). */
#define OC_HF_DEFAULT_CACHE_SUFFIX ".cache/oxidize/hf"

/* Default HuggingFace API base. Plain HTTP — see file-level comment about
 * TLS. A production caller typically sets api_base to an HTTPS-capable
 * proxy or mirror. */
#define OC_HF_DEFAULT_API_BASE "https://huggingface.co"

/* Default revision (branch) used when none is specified. */
#define OC_HF_DEFAULT_REVISION "main"

/* Maximum number of .gguf siblings returned by oc_hf_list_models(). */
#define OC_HF_MAX_MODELS 64

/* Maximum lengths for string fields in OcHfModel/OcHfConfig. */
#define OC_HF_MAX_REPO_ID   256
#define OC_HF_MAX_FILENAME  512
#define OC_HF_MAX_QUANT_TYPE 32
#define OC_HF_MAX_SHA256    65   /* 64 hex chars + NUL */
#define OC_HF_MAX_URL       1024
#define OC_HF_MAX_CACHE_DIR 1024

/* Buffer size for the HTTP response header parser. */
#define OC_HF_HTTP_HEADER_BUF 8192

/* Chunk size for download reads (64 KiB). */
#define OC_HF_DOWNLOAD_CHUNK 65536u


/* Quantization-type filter for oc_hf_list_models(). Pass NULL or "" to
 * disable filtering (return all .gguf files). */
typedef struct OcHfConfig {
    char cache_dir[OC_HF_MAX_CACHE_DIR]; /* "" → default ~/.cache/oxidize/hf */
    char api_token[256];                 /* Bearer token; "" → anonymous   */
    char repo_id[OC_HF_MAX_REPO_ID];     /* e.g. "Qwen/Qwen2-7B-Instruct"  */
    char revision[64];                   /* "" → "main"                    */
    char quant_type[OC_HF_MAX_QUANT_TYPE]; /* e.g. "Q4_K_M"; "" → any     */
    char api_base[OC_HF_MAX_URL];        /* "" → default HF base           */
} OcHfConfig;

/* A discovered or resolved model file. All strings are NUL-terminated. */
typedef struct OcHfModel {
    char     repo_id[OC_HF_MAX_REPO_ID];
    char     filename[OC_HF_MAX_FILENAME];
    uint64_t size_bytes;                 /* 0 if unknown                  */
    char     quant_type[OC_HF_MAX_QUANT_TYPE]; /* parsed from filename; "" */
    char     sha256[OC_HF_MAX_SHA256];   /* hex digest; "" if unknown     */
    char     download_url[OC_HF_MAX_URL]; /* resolve URL; "" until resolved */
} OcHfModel;

/* Progress info passed to the download callback. */
typedef struct OcHfDownloadProgress {
    uint64_t downloaded_bytes;
    uint64_t total_bytes;     /* 0 if unknown (no Content-Length)         */
    double   speed;           /* bytes/sec                               */
    double   eta;             /* seconds remaining; -1 if unknown        */
} OcHfDownloadProgress;

/* Callback invoked periodically during download. May be NULL.
 * Returning non-zero aborts the download with OC_ERR_IO. */
typedef int (*OcHfProgressCb)(const OcHfDownloadProgress *prog, void *user);


/* Initialize a config with defaults. `cache_dir` may be NULL (uses the
 * default ~/.cache/oxidize/hf). Returns OC_OK or OC_ERR_INVALID_ARG. */
OcError oc_hf_config_init(OcHfConfig *cfg, const char *cache_dir);

/* List .gguf models in a HuggingFace repo that match the config's */
OcError oc_hf_list_models(const OcHfConfig *cfg,
                          OcHfModel *out_models, size_t *inout_count);

/* Resolve a repo_id + filename to a HuggingFace download URL. If */
OcError oc_hf_resolve(const OcHfConfig *cfg, OcHfModel *out_model);

/* Get the local cache path for a repo_id + filename. Writes a NUL- */
OcError oc_hf_cache_path(const OcHfConfig *cfg,
                         const char *repo_id, const char *filename,
                         char *out_path, size_t cap);

/* Download a model file to cache_dir. Resumes partial downloads if a Range header. Acquires a global mutex for the duration. */
OcError oc_hf_download(const OcHfConfig *cfg, const OcHfModel *model,
                       OcHfProgressCb cb, void *user);

/* List all cached .gguf files under cache_dir. Writes up to
 * *inout_count entries. Returns OC_OK even if the cache is empty
 * (*inout_count == 0). */
OcError oc_hf_cache_list(const OcHfConfig *cfg,
                         OcHfModel *out_models, size_t *inout_count);

/* Total size in bytes of all files under cache_dir. Returns OC_OK and
 * writes *out_bytes (0 on empty/missing cache). */
OcError oc_hf_cache_size(const OcHfConfig *cfg, uint64_t *out_bytes);

/* Remove cached models older than `max_age_seconds` (0 = remove all).
 * Writes the count of removed files to *out_removed (may be NULL).
 * Never removes the cache_dir itself. */
OcError oc_hf_cache_clean(const OcHfConfig *cfg, uint64_t max_age_seconds,
                          size_t *out_removed);


/* Parse a quant type from a .gguf filename. Looks for the Q\d[_-]\w+ */
bool oc_hf_parse_quant_type(const char *filename, char *out, size_t cap);

/* True if `filename` ends with ".gguf" (case-insensitive). */
bool oc_hf_is_gguf(const char *filename);

/* Sanitize a repo_id for use as a directory name: replace '/' with '_'.
 * Writes into out (cap bytes). Returns OC_OK or OC_ERR_INVALID_ARG. */
OcError oc_hf_sanitize_repo_id(const char *repo_id, char *out, size_t cap);

/* Compute the default cache dir from $HOME (or /tmp fallback). Writes a
 * NUL-terminated path into out (cap bytes). */
OcError oc_hf_default_cache_dir(char *out, size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_HF_HUB_H */
