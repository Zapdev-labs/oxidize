/*
 * pretokenize.h — regex-equivalent BPE pre-tokenizers (hand-written splitters).
 *
 * A pre-tokenizer cuts text into "words"; byte-level BPE merges then run
 * inside each word only. The splitters here are exact ports of llama.cpp's
 * custom split functions, selected by GGUF `tokenizer.ggml.pre`.
 */
#ifndef OXIDIZE_PRETOKENIZE_H
#define OXIDIZE_PRETOKENIZE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    /* No pre-tokenization: merge over the whole segment (legacy Qwen path). */
    OC_PRETOK_NONE = 0,
    /* K2-Horizon (`tokenizer.ggml.pre == "k2-horizon"`):
     *   (?i:'s|'t|'re|'ve|'m|'ll|'d)
     *   |[^\r\n\p{L}\p{N}]?(?:\p{L}|\p{M}|‌|‍)+
     *   |\p{N}{1,3}
     *   | ?[^\s\p{L}\p{N}]+[\r\n]*
     *   |\s*[\r\n]+|\s+(?!\S)|\s+ */
    OC_PRETOK_K2_HORIZON = 1,
} OcPretokType;

/* Map a `tokenizer.ggml.pre` string to a splitter. NULL / unknown strings map
 * to OC_PRETOK_NONE (legacy behavior). */
OcPretokType oc_pretok_type_from_name(const char *pre);

/* Split `cpts[0..n)` into words. Each word's length (in code points) is
 * written to `out_lens` (capacity >= n); the lengths sum to n. Returns the
 * number of words. OC_PRETOK_NONE yields a single word (when n > 0). */
size_t oc_pretok_split(OcPretokType type, const uint32_t *cpts, size_t n,
                       uint32_t *out_lens);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_PRETOKENIZE_H */
