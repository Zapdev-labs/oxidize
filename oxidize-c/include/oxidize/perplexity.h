#ifndef OXIDIZE_PERPLEXITY_H
#define OXIDIZE_PERPLEXITY_H

#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"
#include "oxidize/llama.h"
#include "oxidize/tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OcPerplexityResult {
    double ppl;              /* perplexity (2^avg_nll)                 */
    double avg_nll;         /* average negative log-likelihood        */
    double total_nll;       /* sum of negative log-likelihoods         */
    size_t n_tokens;        /* number of tokens evaluated             */
    double eval_time_sec;   /* wall-clock time for evaluation          */
    double tokens_per_sec;  /* processing speed                       */
} OcPerplexityResult;

/* Compute perplexity of `model` on `text`.
 * Tokenizes the text, runs forward over each token, computes the
 * cross-entropy loss at each position, and returns the geometric mean. */
OcError oc_perplexity_evaluate(OcLlamaModel *model, OcTokenizer *tok,
                                const char *text, size_t max_tokens,
                                OcPerplexityResult *out);

/* Compute perplexity from a file (reads the file, passes contents to
 * oc_perplexity_evaluate). */
OcError oc_perplexity_evaluate_file(OcLlamaModel *model, OcTokenizer *tok,
                                    const char *file_path, size_t max_tokens,
                                    OcPerplexityResult *out);

/* Format perplexity results as a human-readable string. */
void oc_perplexity_format(const OcPerplexityResult *r, char *buf, size_t buf_len);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_PERPLEXITY_H */
