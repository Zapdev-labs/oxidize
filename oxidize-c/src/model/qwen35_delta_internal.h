#pragma once

#include "oxidize/qwen35_delta.h"

OcError oc_qwen35_delta_single_token_step(OcQwen35DeltaState *state,
                                          const OcQwen35DeltaParams *params,
                                          const OcQwen35DeltaInput *input,
                                          float *conv_output,
                                          size_t conv_output_len,
                                          float *output, size_t output_len);
