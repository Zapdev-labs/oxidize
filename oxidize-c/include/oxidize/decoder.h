/*
 * decoder.h — Text decoder for token IDs to text.
 *
 * Converts token IDs to text with special token handling, BOS/EOS
 * awareness, and streaming support. Port from oxidize-core/src/model/.
 */
#ifndef OXIDIZE_DECODER_H
#define OXIDIZE_DECODER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Config ─────────────────────────────────────────────────────────── */

typedef struct OcDecoderConfig {
    bool    skip_special_tokens;   /* default true  */
    bool    add_bos;               /* default false */
    bool    add_eos;               /* default false */
    bool    add_space_prefix;      /* default true  */
} OcDecoderConfig;

/* ─── Decoder ────────────────────────────────────────────────────────── */

typedef struct OcDecoder {
    OcDecoderConfig config;
    size_t          n_decoded;
    bool            last_was_space;
} OcDecoder;

/* ─── Config helpers ────────────────────────────────────────────────── */

/* Initialize config with defaults. Returns OC_ERR_INVALID_ARG if NULL. */
OcError oc_decoder_config_init(OcDecoderConfig *cfg);

/* ─── Lifecycle ──────────────────────────────────────────────────────── */

/* Initialize a decoder with the given config (or defaults if NULL). */
OcError oc_decoder_init(OcDecoder *dec, const OcDecoderConfig *cfg);

/* Decode a single token to text. `token_text` is the raw token string
 * (e.g. from a tokenizer vocab). Writes decoded text into `out` (up to
 * out_size - 1 chars, NUL-terminated). */
OcError oc_decoder_decode_token(OcDecoder *dec, const char *token_text,
                                 char *out, size_t out_size);

/* Decode multiple tokens into a single output string. Writes up to
 * out_size - 1 chars, NUL-terminated. */
OcError oc_decoder_decode_tokens(OcDecoder *dec, const char **token_texts,
                                  size_t n, char *out, size_t out_size);

/* Reset decoder state (n_decoded, last_was_space). Does not change config. */
OcError oc_decoder_reset(OcDecoder *dec);

/* Get the number of tokens decoded since init/reset. */
size_t oc_decoder_n_decoded(const OcDecoder *dec);

/* Free the decoder (no-op for now, but provided for future-proofing). */
void oc_decoder_free(OcDecoder *dec);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_DECODER_H */
