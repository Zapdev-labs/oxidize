/* dspark.h — DSpark-style block drafting on native Qwen MTP. */
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
