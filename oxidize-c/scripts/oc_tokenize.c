/* oc_tokenize.c — tokenizer-only driver for parity checks (`make oc-tokenize`).
 *
 * Maps the GGUF lazily (no readahead, no tensor loads — only the metadata
 * pages are touched), builds the tokenizer once, then reads one
 * hex-encoded UTF-8 text per stdin line and prints its token ids
 * space-separated, one line per input. Used by scripts/tokenizer_parity.py.
 *
 *   oc-tokenize MODEL.gguf [--no-special] [--bos] [--chat]
 *
 *   --no-special  OC_TOK_DISALLOW_SPECIAL (llama.cpp parse_special=false)
 *   --bos         prepend BOS (OC_TOK_ADD_BOS)
 *   --chat        wrap each text as one user turn with the detected
 *                 template + generation prompt before encoding
 *   --info        print tokenizer summary (pre-tokenizer, EOG ids) and exit
 */
#define _POSIX_C_SOURCE 200809L  /* clock_gettime */

#include "oxidize/gguf.h"
#include "oxidize/tokenizer.h"
#include "oxidize/util/mmap.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int hexval(int c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.gguf [--no-special] [--bos] [--chat] [--info]\n",
                argv[0]);
        return 2;
    }
    bool no_special = false, bos = false, chat = false, info = false;
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--no-special") == 0) no_special = true;
        else if (strcmp(argv[i], "--bos") == 0) bos = true;
        else if (strcmp(argv[i], "--chat") == 0) chat = true;
        else if (strcmp(argv[i], "--info") == 0) info = true;
        else { fprintf(stderr, "unknown flag %s\n", argv[i]); return 2; }
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    OcGgufMmappedFile map;
    OcError e = oc_gguf_map_open_flags(argv[1], OC_MMAP_F_NO_READAHEAD, &map);
    if (e != OC_OK) {
        fprintf(stderr, "gguf open failed: %s\n", oc_error_msg(e));
        return 1;
    }
    OcTokenizer tok;
    e = oc_tokenizer_load_from_gguf(&map.unified, &tok);
    if (e != OC_OK) {
        fprintf(stderr, "tokenizer load failed: %s\n", oc_error_msg(e));
        oc_gguf_map_free(&map);
        return 1;
    }
    OcTemplateKind tmpl = oc_tokenizer_detect_template(&map.unified);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    fprintf(stderr, "tokenizer loaded in %.3fs\n",
            (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9);

    if (info) {
        const char *pre = NULL;
        size_t pre_len = 0;
        oc_gguf_metadata_get_str(&map.unified, "tokenizer.ggml.pre", &pre, &pre_len);
        printf("pre=%s template=%d bos=%u eos=%u add_bos=%d eog=[", pre ? pre : "(none)",
               (int)tmpl, tok.bos_id, tok.eos_id, (int)tok.add_bos_token);
        for (size_t i = 0; i < tok.n_eog; ++i) printf(i ? ",%u" : "%u", tok.eog_ids[i]);
        printf("]\n");
        oc_tokenizer_free(&tok);
        oc_gguf_map_free(&map);
        return 0;
    }

    OcSpecialTokenPolicy pol = no_special ? OC_TOK_DISALLOW_SPECIAL
                             : bos ? OC_TOK_ADD_BOS : OC_TOK_DEFAULT;
    size_t cap = 1 << 16;
    char *line = malloc(cap);
    char *text = malloc(cap);
    if (!line || !text) return 1;
    double enc_s = 0.0;
    size_t total_ids = 0;
    int rc = 0;
    for (;;) {
        /* Read a full line of arbitrary length. */
        size_t n = 0;
        int c;
        while ((c = getchar()) != EOF && c != '\n') {
            if (n + 1 >= cap) {
                cap *= 2;
                line = realloc(line, cap);
                text = realloc(text, cap);
                if (!line || !text) return 1;
            }
            line[n++] = (char)c;
        }
        if (c == EOF && n == 0) break;
        size_t tl = 0;
        for (size_t i = 0; i + 1 < n; i += 2) {
            int hi = hexval(line[i]), lo = hexval(line[i + 1]);
            if (hi < 0 || lo < 0) break;
            text[tl++] = (char)(hi * 16 + lo);
        }
        text[tl] = '\0';

        char *rendered = NULL;
        const char *input = text;
        if (chat) {
            OcChatMessage msg = { "user", text };
            if (oc_tokenizer_apply_chat_template(&msg, 1, tmpl, true, &rendered) != OC_OK) {
                rc = 1;
                break;
            }
            input = rendered;
        }
        uint32_t *ids = NULL;
        size_t n_ids = 0;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        e = oc_tokenizer_encode(&tok, input, pol, &ids, &n_ids);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        enc_s += (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
        free(rendered);
        if (e != OC_OK) {
            printf("ERROR %s\n", oc_error_msg(e));
        } else {
            for (size_t i = 0; i < n_ids; ++i) printf(i ? " %u" : "%u", ids[i]);
            printf("\n");
            total_ids += n_ids;
        }
        free(ids);
        if (c == EOF) break;
    }
    fprintf(stderr, "encoded %zu ids in %.3fs\n", total_ids, enc_s);
    free(line);
    free(text);
    oc_tokenizer_free(&tok);
    oc_gguf_map_free(&map);
    return rc;
}
