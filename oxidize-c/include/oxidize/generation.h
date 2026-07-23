/*
 * generation.h — Token generation engine.
 *
 * Wraps the model forward pass with sampling, tokenizer, and timing
 * to provide a clean generation API. Port from oxidize-core/src/model/generation.rs.
 */
#ifndef OXIDIZE_GENERATION_H
#define OXIDIZE_GENERATION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/sampling.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OC_GEN_MAX_TOKENS 8192

typedef struct {
    uint32_t max_tokens;
    float temperature;
    float top_p;
    int32_t top_k;
    float repeat_penalty;
    uint32_t repeat_last_n;
    uint32_t seed;
    bool stream;
    bool stop_on_eos;
} OcGenConfig;

typedef struct {
    uint32_t *tokens;
    size_t n_tokens;
    double eval_time_sec;
    double tokens_per_sec;
    double prefill_time_sec;
    double prefill_tokens_per_sec;
    bool stopped_on_eos;
    size_t n_prompt_tokens;
} OcGenResult;

typedef struct {
    uint32_t *context;
    size_t n_context;
    uint32_t recent_tokens[64];
    size_t n_recent;
    OcSamplerConfig sampler;
    uint32_t pos;
} OcGenState;

typedef int (*OcGenTokenCb)(uint32_t token, const char *piece, void *user);

OcError oc_gen_config_init(OcGenConfig *cfg);
OcError oc_gen_state_init(OcGenState *state, const uint32_t *context, size_t n);
OcError oc_gen_state_add_token(OcGenState *state, uint32_t token);
OcError oc_gen_result_init(OcGenResult *result, size_t max_tokens);
void oc_gen_result_free(OcGenResult *result);
OcError oc_gen_config_from_cli(const OcGenConfig *cfg, OcSamplerConfig *out);
const char *oc_gen_stop_reason(const OcGenResult *result);
uint64_t oc_gen_total_tokens(const OcGenResult *result);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_GENERATION_H */
