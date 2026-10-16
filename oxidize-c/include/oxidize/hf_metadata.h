#ifndef OXIDIZE_HF_METADATA_H
#define OXIDIZE_HF_METADATA_H
#include "oxidize/llama.h"
/* Load config.json and tokenizer.json directly. tokenizer_config.json is
 * optional. Caller zero-initializes outputs; metadata owns its arena and
 * is freed with oc_gguf_free(). No GGUF file is read or written. */
OcError oc_hf_metadata_load(const char *directory, OcLlamaConfig *config,
                          OcModelArchitecture *arch, OcGgufFile *metadata);
#endif
