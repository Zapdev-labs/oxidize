/* decoder.c — Text decoder implementation. */
#include "oxidize/decoder.h"

#include <stdlib.h>
#include <string.h>


/* Common special token prefixes used by SentencePiece, Tiktoken, etc. */
static bool is_special_token(const char *text)
{
    if (!text) return false;
    /* <...> pattern: BPE/sentencepiece special tokens like <s>, </s>,
     * <pad>, <unk>, <mask>, etc. */
    if (text[0] == '<') {
        const char *close = strchr(text, '>');
        if (close) return true;
    }
    /* <|...|> pattern: tiktoken special tokens like <|endoftext|>,
     * <|im_start|>, <|im_end|> */
    if (text[0] == '<' && text[1] == '|') {
        return true;
    }
    return false;
}

/* Check if a token text is the BOS token. */
static bool is_bos_token(const char *text)
{
    if (!text) return false;
    return strcmp(text, "<s>") == 0;
}

/* Check if a token text is the EOS token. */
static bool is_eos_token(const char *text)
{
    if (!text) return false;
    return strcmp(text, "</s>") == 0 ||
           strcmp(text, "<|endoftext|>") == 0 ||
           strcmp(text, "<|im_end|>") == 0;
}


__attribute__((unused))
static size_t str_len(const char *s)
{
    return s ? strlen(s) : 0;
}

/* Append src to dst at position *pos, up to out_size - 1 chars. Updates *pos.
 * Returns OC_ERR_OOM if not enough space (still writes what fits). */
static OcError append_str(char *dst, size_t out_size, size_t *pos,
                           const char *src)
{
    if (!dst || out_size == 0 || !src) return OC_ERR_INVALID_ARG;
    size_t src_len = strlen(src);
    size_t avail = out_size - 1 - *pos;
    size_t to_copy = src_len < avail ? src_len : avail;
    if (to_copy > 0) {
        memcpy(dst + *pos, src, to_copy);
        *pos += to_copy;
    }
    dst[*pos] = '\0';
    if (src_len > avail) return OC_ERR_OOM;
    return OC_OK;
}


OcError oc_decoder_config_init(OcDecoderConfig *cfg)
{
    if (!cfg) return OC_ERR_INVALID_ARG;
    cfg->skip_special_tokens = true;
    cfg->add_bos             = false;
    cfg->add_eos             = false;
    cfg->add_space_prefix    = true;
    return OC_OK;
}


OcError oc_decoder_init(OcDecoder *dec, const OcDecoderConfig *cfg)
{
    if (!dec) return OC_ERR_INVALID_ARG;
    if (cfg) {
        dec->config = *cfg;
    } else {
        oc_decoder_config_init(&dec->config);
    }
    dec->n_decoded       = 0;
    dec->last_was_space  = false;
    return OC_OK;
}


OcError oc_decoder_decode_token(OcDecoder *dec, const char *token_text,
                                 char *out, size_t out_size)
{
    if (!dec || !token_text || !out || out_size == 0) {
        return OC_ERR_INVALID_ARG;
    }
    out[0] = '\0';
    size_t pos = 0;

    /* First token: optionally add BOS. */
    if (dec->n_decoded == 0 && dec->config.add_bos) {
        OcError e = append_str(out, out_size, &pos, "<s>");
        if (e != OC_OK) return e;
        dec->last_was_space = false;
    }

    /* Handle special tokens. */
    if (is_special_token(token_text)) {
        if (is_bos_token(token_text)) {
            if (dec->config.skip_special_tokens) {
                dec->n_decoded++;
                return OC_OK;
            }
        }
        if (is_eos_token(token_text)) {
            if (dec->config.skip_special_tokens) {
                dec->n_decoded++;
                dec->last_was_space = true;
                return OC_OK;
            }
            /* Not skipping: append the EOS text. */
        }
        if (dec->config.skip_special_tokens) {
            dec->n_decoded++;
            return OC_OK;
        }
        /* Not skipping special tokens: append as-is. */
        OcError e = append_str(out, out_size, &pos, token_text);
        if (e != OC_OK) return e;
        dec->n_decoded++;
        dec->last_was_space = false;
        return OC_OK;
    }

    /* Regular token: handle space prefix. */
    /* SentencePiece-style tokens start with '▁' (U+2581) for word boundary. */
    const char *text_to_write = token_text;
    bool starts_with_space = false;

    if (token_text[0] == '\xe2' && token_text[1] == '\x96' &&
        token_text[2] == '\x81') {
        /* SentencePiece space marker: replace with actual space. */
        starts_with_space = true;
        text_to_write = token_text + 3;
    } else if (token_text[0] == ' ') {
        starts_with_space = true;
        text_to_write = token_text + 1;
    }

    /* Add space prefix logic. */
    if (starts_with_space) {
        /* If we should add a space and last wasn't a space, add one. */
        if (!dec->last_was_space && dec->config.add_space_prefix) {
            if (pos < out_size - 1) {
                out[pos++] = ' ';
                out[pos] = '\0';
            }
        }
        dec->last_was_space = false;
    }

    /* Append the (possibly trimmed) token text. */
    OcError e = append_str(out, out_size, &pos, text_to_write);
    if (e != OC_OK) return e;

    /* Update space state: check if token ends with a space. */
    size_t written_len = strlen(text_to_write);
    if (written_len > 0 && text_to_write[written_len - 1] == ' ') {
        dec->last_was_space = true;
    } else {
        dec->last_was_space = false;
    }

    dec->n_decoded++;
    return OC_OK;
}

OcError oc_decoder_decode_tokens(OcDecoder *dec, const char **token_texts,
                                  size_t n, char *out, size_t out_size)
{
    if (!dec || !token_texts || !out || out_size == 0) {
        return OC_ERR_INVALID_ARG;
    }
    out[0] = '\0';
    size_t pos = 0;

    /* Optionally add BOS at the start. */
    if (dec->config.add_bos) {
        OcError e = append_str(out, out_size, &pos, "<s>");
        if (e != OC_OK) return e;
        dec->last_was_space = false;
        /* Set n_decoded so decode_token doesn't add BOS again. */
        dec->n_decoded = 1;
    }

    for (size_t i = 0; i < n; i++) {
        char buf[4096];
        buf[0] = '\0';
        /* Decode this token into a temp buffer. */
        /* Save state to restore after decoding this token. */
        size_t saved_n = dec->n_decoded;
        bool saved_space = dec->last_was_space;

        OcError e = oc_decoder_decode_token(dec, token_texts[i], buf,
                                             sizeof(buf));
        if (e != OC_OK) return e;

        /* Restore n_decoded (we'll count at the end) and use the
         * produced text. */
        dec->n_decoded = saved_n;
        dec->last_was_space = saved_space;

        /* Now apply the token's effect manually using the produced text. */
        if (strlen(buf) > 0) {
            e = append_str(out, out_size, &pos, buf);
            if (e != OC_OK) return e;
        }

        /* Update state based on the produced text. */
        dec->n_decoded++;
        size_t blen = strlen(buf);
        if (blen > 0 && buf[blen - 1] == ' ') {
            dec->last_was_space = true;
        } else {
            dec->last_was_space = false;
        }
    }

    /* Optionally add EOS at the end. */
    if (dec->config.add_eos) {
        OcError e2 = append_str(out, out_size, &pos, "</s>");
        if (e2 != OC_OK) return e2;
    }

    return OC_OK;
}


OcError oc_decoder_reset(OcDecoder *dec)
{
    if (!dec) return OC_ERR_INVALID_ARG;
    dec->n_decoded = 0;
    dec->last_was_space = false;
    return OC_OK;
}

size_t oc_decoder_n_decoded(const OcDecoder *dec)
{
    if (!dec) return 0;
    return dec->n_decoded;
}

void oc_decoder_free(OcDecoder *dec)
{
    if (!dec) return;
    memset(dec, 0, sizeof(*dec));
}
