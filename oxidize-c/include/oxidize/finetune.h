#ifndef OXIDIZE_FINETUNE_H
#define OXIDIZE_FINETUNE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/llama.h"
#include "oxidize/lora.h"
#include "oxidize/tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OC_FT_SFT       = 0,  /* supervised fine-tuning                  */
    OC_FT_SELF_TRAIN = 1, /* self-training with synthetic data         */
    OC_FT_DPO       = 2,  /* direct preference optimization (stub)    */
    OC_FT_PPO       = 3,  /* proximal policy optimization (stub)      */
} OcFtStrategy;

typedef struct OcFtConfig {
    OcFtStrategy strategy;
    const char *model_path;       /* base model GGUF path              */
    const char *dataset_path;     /* JSONL training data                */
    const char *output_dir;       /* output directory for adapters       */
    const char *resume_from;      /* checkpoint to resume from           */
    uint32_t    lora_rank;        /* LoRA rank (default 8)               */
    uint32_t    lora_alpha;       /* LoRA alpha (default 16)             */
    uint32_t    epochs;           /* number of training epochs            */
    uint32_t    batch_size;       /* batch size                            */
    float       learning_rate;   /* learning rate                         */
    float       warmup_ratio;     /* warmup fraction of total steps        */
    float       weight_decay;    /* weight decay                          */
    uint32_t    max_seq_length;   /* maximum sequence length               */
    bool        verbose;
} OcFtConfig;

/* Run a fine-tuning job. */
OcError oc_finetune_run(const OcFtConfig *cfg);

/* Save LoRA adapters to a file. */
OcError oc_lora_save(const OcLoraModel *lm, const char *path);

/* Load LoRA adapters from a file. */
OcError oc_lora_load(const char *path, OcLoraModel *lm);

/* Generate synthetic training data from the model (for self-training). */
OcError oc_finetune_generate_synthetic(OcLlamaModel *model, OcTokenizer *tok,
                                        const char *prompt, uint32_t n_samples,
                                        const char *output_path);

/* Format a conversation as SFT input (prompt + response pairs). */
OcError oc_finetune_format_sft(const char *system, const char *user,
                               const char *assistant, char *out, size_t out_cap);

/* Get the name of a fine-tuning strategy. */
const char *oc_ft_strategy_name(OcFtStrategy s);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_FINETUNE_H */
