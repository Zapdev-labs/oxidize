/*
 * dspark.h — DSpark-style block drafting on native Qwen MTP.
 *
 * llama.cpp's draft-dspark is a DFlash variant with an anchor-first noise
 * block and a Markov/confidence head. This Qwen3.5/3.8 GGUF has no sidecar
 * DSpark drafter (those are DeepSeek-V4 4096-d). It does ship in-GGUF MTP
 * (`blk.N.nextn.*`). This module applies the DSpark draft policy to that
 * head: draft a full block from the last committed token, stop early when
 * pairwise confidence falls below p_min, then greedy-verify on the target.
 */
#ifndef OXIDIZE_DSPARK_H
#define OXIDIZE_DSPARK_H

#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/llama.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_DSPARK_MAX_BLOCK 8u

typedef struct OcDsparkConfig {
    uint32_t block_size;
    float    p_min;
} OcDsparkConfig;

typedef struct OcDsparkStats {
    uint32_t drafted;
    uint32_t accepted;
    uint32_t emitted;
} OcDsparkStats;

void oc_dspark_config_init(OcDsparkConfig *cfg);
float oc_dspark_pairwise_conf(const float *logits, size_t vocab);
OcError oc_dspark_advance(OcLlamaSession *sess, float *logits,
                          const OcDsparkConfig *cfg,
                          uint32_t *out_tokens, size_t max_out, size_t *n_out,
                          OcDsparkStats *stats);

#ifdef __cplusplus
}
#endif

#endif
