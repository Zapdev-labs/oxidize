#ifndef OXIDIZE_TOKENIZER_TRAIN_H
#define OXIDIZE_TOKENIZER_TRAIN_H

#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef struct OcBpeTrainConfig {
    size_t max_vocab_size;   /* hard cap on total vocab entries (0 = no cap) */
    size_t min_frequency;    /* minimum pair frequency to consider merging  */
    size_t max_merges;       /* max number of merge rules (0 = no limit)    */
} OcBpeTrainConfig;

/* Sensible defaults for small corpora. */
#define OC_BPE_DEFAULT_MAX_VOCAB   10000
#define OC_BPE_DEFAULT_MIN_FREQ    2
#define OC_BPE_DEFAULT_MAX_MERGES  5000


/* A single merge rule: "left right" -> "merged". The tokens are indices into
 * the vocab array. */
typedef struct OcBpeMerge {
    uint32_t left;     /* vocab index of the left symbol                   */
    uint32_t right;    /* vocab index of the right symbol                  */
    uint32_t merged;   /* vocab index of the resulting merged symbol       */
} OcBpeMerge;


typedef struct OcBpeVocabEntry {
    char    *token;    /* NUL-terminated token string (heap-owned)          */
    uint32_t id;       /* vocab id (index into the vocab array)             */
} OcBpeVocabEntry;


typedef struct OcBpeTrainer OcBpeTrainer;

/* Create a new trainer with the given config. If max_vocab_size == 0,
 * uses OC_BPE_DEFAULT_MAX_VOCAB; similarly for the other fields.
 * Returns NULL on OOM. */
OcBpeTrainer *oc_bpe_trainer_init(OcBpeTrainConfig config);

/* Train BPE on the given corpus text. The corpus is a NUL-terminated UTF-8 */
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

/* Save the vocab + merge rules to a JSON file at `path`. */
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
