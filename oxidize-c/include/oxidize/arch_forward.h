/* arch_forward.h — architecture-specific forward passes (GPT-2, GPT-NeoX, Falcon). */
#ifndef OXIDIZE_ARCH_FORWARD_H
#define OXIDIZE_ARCH_FORWARD_H

#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/llama.h"

#ifdef __cplusplus
extern "C" {
#endif

/* GPT-2 forward pass for a single token. `logits_out` may be NULL to skip the lm_head projection (prefill). */
OcError oc_arch_forward_gpt2(OcLlamaSession *sess, uint32_t token,
                              float *logits_out);

/* GPT-J forward pass for a single token. */
OcError oc_arch_forward_gptj(OcLlamaSession *sess, uint32_t token,
                              float *logits_out);

/* GPT-NeoX forward pass for a single token. */
OcError oc_arch_forward_gpt_neox(OcLlamaSession *sess, uint32_t token,
                                  float *logits_out);

/* Falcon forward pass for a single token. */
OcError oc_arch_forward_falcon(OcLlamaSession *sess, uint32_t token,
                                float *logits_out);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_ARCH_FORWARD_H */
