/*
 * perplexity.c — perplexity evaluation implementation.
 */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/perplexity.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double wall_now(void)
{
    struct timespec ts;
    if (timespec_get(&ts, TIME_UTC) != TIME_UTC) return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

OcError oc_perplexity_evaluate(OcLlamaModel *model, OcTokenizer *tok,
                                const char *text, size_t max_tokens,
                                OcPerplexityResult *out)
{
    if (!model || !tok || !text || !out) return OC_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));

    /* Tokenize the text. */
    uint32_t *ids = NULL;
    size_t n_ids = 0;
    OcSpecialTokenPolicy pol = tok->has_add_bos_token && tok->add_bos_token
        ? OC_TOK_ADD_BOS : OC_TOK_DEFAULT;
    OcError e = oc_tokenizer_encode(tok, text, pol, &ids, &n_ids);
    if (e != OC_OK) return e;
    if (n_ids < 2) { free(ids); return OC_ERR_INVALID_ARG; }

    /* Limit tokens if requested. */
    if (max_tokens > 0 && max_tokens < n_ids) n_ids = max_tokens;

    /* Initialize session. */
    OcLlamaSession sess;
    e = oc_llama_session_init(model, &sess);
    if (e != OC_OK) { free(ids); return e; }

    /* Process tokens: for each position i, the loss is -log2(p(token[i] | tokens[0..i-1])).
     * We forward tokens[0..i-1] (building KV cache), then compute logits at position i-1,
     * and evaluate the log-probability of token[i]. */
    double total_nll = 0.0;
    size_t n_evaluated = 0;
    double start = wall_now();

    if (ids[0] >= model->cfg.vocab_size) {
        oc_llama_session_free(&sess);
        free(ids);
        return OC_ERR_INVALID_ARG;
    }
    e = oc_llama_forward(&sess, ids[0], sess.logits);
    if (e != OC_OK) {
        oc_llama_session_free(&sess);
        free(ids);
        return e;
    }

    for (size_t i = 1; i < n_ids; i++) {
        if (ids[i] >= model->cfg.vocab_size) {
            oc_llama_session_free(&sess);
            free(ids);
            return OC_ERR_INVALID_ARG;
        }

        /* Compute softmax to get probabilities. */
        uint32_t vocab = model->cfg.vocab_size;
        float max_logit = sess.logits[0];
        for (uint32_t v = 1; v < vocab; v++) {
            if (sess.logits[v] > max_logit) max_logit = sess.logits[v];
        }

        double sum_exp = 0.0;
        for (uint32_t v = 0; v < vocab; v++) {
            sum_exp += exp((double)sess.logits[v] - max_logit);
        }

        /* Probability of the actual next token. */
        double logit_target = (double)sess.logits[ids[i]];
        double log_prob = logit_target - max_logit - log(sum_exp);
        total_nll += -log_prob / log(2.0);  /* convert to log2 */
        n_evaluated++;

        if (i + 1 < n_ids) {
            e = oc_llama_forward(&sess, ids[i], sess.logits);
            if (e != OC_OK) {
                if (e == OC_ERR_INVALID_ARG) break;
                oc_llama_session_free(&sess);
                free(ids);
                return e;
            }
        }
    }

    double elapsed = wall_now() - start;
    oc_llama_session_free(&sess);
    free(ids);

    out->n_tokens = n_evaluated;
    out->total_nll = total_nll;
    out->avg_nll = (n_evaluated > 0) ? total_nll / (double)n_evaluated : 0.0;
    out->ppl = (n_evaluated > 0) ? pow(2.0, out->avg_nll) : 0.0;
    out->eval_time_sec = elapsed;
    out->tokens_per_sec = (elapsed > 0) ? (double)n_evaluated / elapsed : 0.0;

    return OC_OK;
}

OcError oc_perplexity_evaluate_file(OcLlamaModel *model, OcTokenizer *tok,
                                    const char *file_path, size_t max_tokens,
                                    OcPerplexityResult *out)
{
    if (!file_path || !out) return OC_ERR_INVALID_ARG;
    FILE *f = fopen(file_path, "rb");
    if (!f) return OC_ERR_IO;

    /* Read the entire file. */
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return OC_ERR_IO; }
    long fsize = ftell(f);
    if (fsize < 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return OC_ERR_IO; }

    if (fsize <= 0 || fsize > (1L << 28)) { /* 256 MB max */
        fclose(f);
        return OC_ERR_INVALID_ARG;
    }

    char *text = malloc(fsize + 1);
    if (!text) { fclose(f); return OC_ERR_OOM; }
    size_t nread = fread(text, 1, fsize, f);
    if (nread != (size_t)fsize) {
        free(text);
        fclose(f);
        return OC_ERR_IO;
    }
    fclose(f);
    text[nread] = '\0';

    OcError e = oc_perplexity_evaluate(model, tok, text, max_tokens, out);
    free(text);
    return e;
}

void oc_perplexity_format(const OcPerplexityResult *r, char *buf, size_t buf_len)
{
    if (!r || !buf || buf_len == 0) return;
    snprintf(buf, buf_len,
             "perplexity: %.4f\n"
             "avg_nll: %.6f\n"
             "tokens: %zu\n"
             "time: %.2fs (%.1f tok/s)\n",
             r->ppl, r->avg_nll, r->n_tokens,
             r->eval_time_sec, r->tokens_per_sec);
}
