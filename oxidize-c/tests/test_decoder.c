/* test_decoder.c — text decoder tests. */
#define _POSIX_C_SOURCE 200809L
#include "framework.h"
#include <string.h>
#include "oxidize/decoder.h"

/* ─── Config tests ─────────────────────────────────────────────────────── */

Test(decoder, config_init_defaults)
{
    OcDecoderConfig cfg;
    cr_assert_eq(oc_decoder_config_init(&cfg), OC_OK);
    cr_assert(cfg.skip_special_tokens);
    cr_assert(!cfg.add_bos);
    cr_assert(!cfg.add_eos);
    cr_assert(cfg.add_space_prefix);
}

OC_TEST_REJECTS_NULL(decoder, config_init_null, oc_decoder_config_init(NULL))

/* ─── Init tests ────────────────────────────────────────────────────────── */

Test(decoder, init_default)
{
    OcDecoder dec;
    cr_assert_eq(oc_decoder_init(&dec, NULL), OC_OK);
    cr_assert(dec.config.skip_special_tokens);
    cr_assert_eq(dec.n_decoded, 0u);
    cr_assert(!dec.last_was_space);
    oc_decoder_free(&dec);
}

Test(decoder, init_custom_config)
{
    OcDecoderConfig cfg;
    oc_decoder_config_init(&cfg);
    cfg.add_bos = true;
    cfg.add_eos = true;
    OcDecoder dec;
    cr_assert_eq(oc_decoder_init(&dec, &cfg), OC_OK);
    cr_assert(dec.config.add_bos);
    cr_assert(dec.config.add_eos);
    oc_decoder_free(&dec);
}

OC_TEST_REJECTS_NULL(decoder, init_null, oc_decoder_init(NULL, NULL))

Test(decoder, free_null_safe)
{
    oc_decoder_free(NULL);
    cr_assert(true);
}

/* ─── Decode single token tests ─────────────────────────────────────────── */

Test(decoder, decode_simple_token)
{
    OcDecoder dec;
    oc_decoder_init(&dec, NULL);
    char out[256];
    cr_assert_eq(oc_decoder_decode_token(&dec, "hello", out, sizeof(out)),
                 OC_OK);
    cr_assert_str_eq(out, "hello");
    cr_assert_eq(dec.n_decoded, 1u);
    oc_decoder_free(&dec);
}

Test(decoder, decode_space_prefix_token)
{
    /* SentencePiece-style: ▁hello means " hello" */
    OcDecoder dec;
    oc_decoder_init(&dec, NULL);
    char out[256];
    cr_assert_eq(oc_decoder_decode_token(&dec, "\xe2\x96\x81hello", out,
                                          sizeof(out)), OC_OK);
    cr_assert_str_eq(out, " hello");
    oc_decoder_free(&dec);
}

Test(decoder, decode_special_token_skipped)
{
    OcDecoder dec;
    oc_decoder_init(&dec, NULL);
    char out[256];
    /* <s> is a special token; should be skipped by default. */
    cr_assert_eq(oc_decoder_decode_token(&dec, "<s>", out, sizeof(out)),
                 OC_OK);
    cr_assert_str_eq(out, "");
    cr_assert_eq(dec.n_decoded, 1u);
    oc_decoder_free(&dec);
}

Test(decoder, decode_special_token_not_skipped)
{
    OcDecoderConfig cfg;
    oc_decoder_config_init(&cfg);
    cfg.skip_special_tokens = false;
    OcDecoder dec;
    oc_decoder_init(&dec, &cfg);
    char out[256];
    cr_assert_eq(oc_decoder_decode_token(&dec, "<pad>", out, sizeof(out)),
                 OC_OK);
    cr_assert_str_eq(out, "<pad>");
    oc_decoder_free(&dec);
}

Test(decoder, decode_eos_token)
{
    OcDecoder dec;
    oc_decoder_init(&dec, NULL);
    char out[256];
    cr_assert_eq(oc_decoder_decode_token(&dec, "</s>", out, sizeof(out)),
                 OC_OK);
    /* EOS is a special token, skipped by default. */
    cr_assert_str_eq(out, "");
    oc_decoder_free(&dec);
}

/* ─── Decode multiple tokens tests ──────────────────────────────────────── */

Test(decoder, decode_tokens_basic)
{
    OcDecoder dec;
    oc_decoder_init(&dec, NULL);
    const char *tokens[] = {"\xe2\x96\x81hello", "\xe2\x96\x81world"};
    char out[256];
    cr_assert_eq(oc_decoder_decode_tokens(&dec, tokens, 2, out, sizeof(out)),
                 OC_OK);
    cr_assert_str_eq(out, " hello world");
    cr_assert_eq(dec.n_decoded, 2u);
    oc_decoder_free(&dec);
}

Test(decoder, decode_tokens_with_bos_eos)
{
    OcDecoderConfig cfg;
    oc_decoder_config_init(&cfg);
    cfg.add_bos = true;
    cfg.add_eos = true;
    OcDecoder dec;
    oc_decoder_init(&dec, &cfg);
    const char *tokens[] = {"\xe2\x96\x81test"};
    char out[256];
    cr_assert_eq(oc_decoder_decode_tokens(&dec, tokens, 1, out, sizeof(out)),
                 OC_OK);
    cr_assert_str_eq(out, "<s> test</s>");
    oc_decoder_free(&dec);
}

Test(decoder, decode_tokens_empty)
{
    OcDecoder dec;
    oc_decoder_init(&dec, NULL);
    char out[256];
    cr_assert_eq(oc_decoder_decode_tokens(&dec, NULL, 0, out, sizeof(out)),
                 OC_ERR_INVALID_ARG);
    oc_decoder_free(&dec);
}

/* ─── Reset / state tests ───────────────────────────────────────────────── */

Test(decoder, reset_clears_state)
{
    OcDecoder dec;
    oc_decoder_init(&dec, NULL);
    char out[256];
    oc_decoder_decode_token(&dec, "hi", out, sizeof(out));
    cr_assert_eq(dec.n_decoded, 1u);
    cr_assert_eq(oc_decoder_reset(&dec), OC_OK);
    cr_assert_eq(dec.n_decoded, 0u);
    cr_assert(!dec.last_was_space);
    oc_decoder_free(&dec);
}

Test(decoder, n_decoded)
{
    OcDecoder dec;
    oc_decoder_init(&dec, NULL);
    char out[256];
    cr_assert_eq(oc_decoder_n_decoded(&dec), 0u);
    oc_decoder_decode_token(&dec, "a", out, sizeof(out));
    oc_decoder_decode_token(&dec, "b", out, sizeof(out));
    cr_assert_eq(oc_decoder_n_decoded(&dec), 2u);
    oc_decoder_free(&dec);
}

OC_TEST_NULL_SAFE(decoder, n_decoded_null,
        cr_assert_eq(oc_decoder_n_decoded(NULL), 0u);)
