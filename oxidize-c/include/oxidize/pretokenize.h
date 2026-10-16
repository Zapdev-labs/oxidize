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
    /* GPT-2 (`pre == "gpt2"`): standard 's/'t/... + letter/number runs. */
    OC_PRETOK_GPT2 = 2,
    /* Llama-3 BPE (`pre == "llama-bpe"` or "llama3"): like GPT-2 but
     * numbers split in 1-3 digit chunks and contractions lowercase. */
    OC_PRETOK_LLAMA3 = 3,
    /* Qwen2 (`pre == "qwen2"`): letters run without combining marks;
     * numbers are matched one Unicode number at a time. */
    OC_PRETOK_QWEN2 = 4,
    /* Qwen2 multimodal variant (`pre == "qwen_m"`): combining marks
     * (\p{M}) stay inside letter runs. */
    OC_PRETOK_QWEN2_M = 5,
} OcPretokType;

/* Map a `tokenizer.ggml.pre` string to a splitter. NULL / unknown strings map
 * to OC_PRETOK_NONE (legacy behavior). */
OcPretokType oc_pretok_type_from_name(const char *pre);

/* Split `cpts[0..n)` into words. Each word's length (in code points) is
 * written to `out_lens` (capacity >= n); the lengths sum to n. Returns the
 * number of words. OC_PRETOK_NONE yields a single word (when n > 0).
 * Lengths are size_t so an input longer than UINT32_MAX is not truncated. */
size_t oc_pretok_split(OcPretokType type, const uint32_t *cpts, size_t n,
                       size_t *out_lens);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_PRETOKENIZE_H */
