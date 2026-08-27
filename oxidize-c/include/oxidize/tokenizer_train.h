/*
 * tokenizer_train.h — BPE tokenizer trainer.
 *
 * Learns Byte-Pair Encoding merge rules from a text corpus. The resulting
 * vocab + merge rules can be saved to a JSON file for later loading by a
 * tokenizer encoder.
 *
 * Algorithm (standard BPE training):
 *   1. Pre-tokenize the corpus: split on whitespace, then split each token
 *      into individual characters (UTF-8 codepoint boundaries).
 *   2. Initialize the vocabulary with all individual characters seen.
 *   3. Count all adjacent symbol pairs across all words.
 *   4. Find the most frequent pair (above min_frequency threshold).
 *   5. Merge that pair in all words, add the merged symbol to the vocab,
 *      and record the merge rule.
 *   6. Repeat until: no pair has count >= min_frequency, or max_merges
 *      reached, or max_vocab_size reached.
 *
 * This is a straightforward C implementation; it is NOT a port of a
 * specific Rust module (oxidize-core does not currently include a BPE
 * trainer — only loaders/encoders). The API mirrors the Config + Error +
 * Struct trinity convention used across the oxidize-c codebase.
 *
 * Usage:
 *   OcBpeTrainConfig cfg = { .max_vocab_size = 1000, .min_frequency = 2,
 *                            .max_merges = 500 };
 *   OcBpeTrainer *t = oc_bpe_trainer_init(cfg);
 *   oc_bpe_trainer_train(t, corpus, strlen(corpus));
 *   oc_bpe_trainer_save(t, "vocab.json");
 *   oc_bpe_trainer_free(t);
 */
#ifndef OXIDIZE_TOKENIZER_TRAIN_H
#define OXIDIZE_TOKENIZER_TRAIN_H

#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Config ────────────────────────────────────────────────────────────── */

typedef struct OcBpeTrainConfig {
    size_t max_vocab_size;   /* hard cap on total vocab entries (0 = no cap) */
    size_t min_frequency;    /* minimum pair frequency to consider merging  */
    size_t max_merges;       /* max number of merge rules (0 = no limit)    */
} OcBpeTrainConfig;

/* Sensible defaults for small corpora. */
#define OC_BPE_DEFAULT_MAX_VOCAB   10000
#define OC_BPE_DEFAULT_MIN_FREQ    2
#define OC_BPE_DEFAULT_MAX_MERGES  5000

/* ─── Merge rule ───────────────────────────────────────────────────────── */

/* A single merge rule: "left right" -> "merged". The tokens are indices into
 * the vocab array. */
typedef struct OcBpeMerge {
    uint32_t left;     /* vocab index of the left symbol                   */
    uint32_t right;    /* vocab index of the right symbol                  */
    uint32_t merged;   /* vocab index of the resulting merged symbol       */
} OcBpeMerge;

/* ─── Vocab entry ──────────────────────────────────────────────────────── */

typedef struct OcBpeVocabEntry {
    char    *token;    /* NUL-terminated token string (heap-owned)          */
    uint32_t id;       /* vocab id (index into the vocab array)             */
} OcBpeVocabEntry;

/* ─── Trainer ──────────────────────────────────────────────────────────── */

typedef struct OcBpeTrainer OcBpeTrainer;

/* Create a new trainer with the given config. If max_vocab_size == 0,
 * uses OC_BPE_DEFAULT_MAX_VOCAB; similarly for the other fields.
 * Returns NULL on OOM. */
OcBpeTrainer *oc_bpe_trainer_init(OcBpeTrainConfig config);

/* Train BPE on the given corpus text. The corpus is a NUL-terminated UTF-8
 * string (corpus_len is the byte length, or 0 to use strlen(corpus)).
 *
 * After training, the vocab and merge rules are available via the getter
 * functions. Calling train() twice on the same trainer resets the state.
 *
 * Returns OC_OK, OC_ERR_INVALID_ARG (NULL trainer / NULL corpus),
 * OC_ERR_OOM. */
OcError oc_bpe_trainer_train(OcBpeTrainer *t, const char *corpus, size_t corpus_len);

/* Get the learned vocabulary. Writes a pointer to the internal vocab array
 * (owned by the trainer, valid until free() or next train()) and the count
 * to `*out_count`. Returns OC_OK, OC_ERR_INVALID_ARG. */
OcError oc_bpe_trainer_vocab(const OcBpeTrainer *t,
                             const OcBpeVocabEntry **out_entries,
                             size_t *out_count);

/* Get the learned merge rules. Writes a pointer to the internal merge array
 * and the count. Returns OC_OK, OC_ERR_INVALID_ARG. */
OcError oc_bpe_trainer_merges(const OcBpeTrainer *t,
                              const OcBpeMerge **out_merges, size_t *out_count);

/* Save the vocab + merge rules to a JSON file at `path`.
 *
 * JSON format:
 *   {
 *     "vocab": [ {"id": 0, "token": "a"}, ... ],
 *     "merges": [ {"left": 5, "right": 3, "merged": 10}, ... ]
 *   }
 *
 * Returns OC_OK, OC_ERR_IO, OC_ERR_INVALID_ARG. */
OcError oc_bpe_trainer_save(const OcBpeTrainer *t, const char *path);

/* Number of vocab entries. Returns 0 if t is NULL or untrained. */
size_t oc_bpe_trainer_vocab_size(const OcBpeTrainer *t);

/* Number of merge rules. Returns 0 if t is NULL or untrained. */
size_t oc_bpe_trainer_merge_count(const OcBpeTrainer *t);

/* Free the trainer and all owned allocations. Safe on NULL. */
void oc_bpe_trainer_free(OcBpeTrainer *t);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_TOKENIZER_TRAIN_H */
