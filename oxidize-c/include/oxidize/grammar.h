/*
 * grammar.h — grammar constraints for structured output (JSON, regex, CFG).
 *
 * Provides a simple grammar constraint system that filters logits during
 * sampling to ensure generated text conforms to a specified grammar (e.g.
 * JSON, specific field names, or a character-level CFG).
 *
 * This is a lightweight implementation focused on JSON schema constraints
 * and character-set filtering. Full CFG parsing (like llama.cpp's GBNF) is
 * left as a future enhancement.
 */
#ifndef OXIDIZE_GRAMMAR_H
#define OXIDIZE_GRAMMAR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OC_GRAMMAR_NONE    = 0,  /* no constraint                          */
    OC_GRAMMAR_JSON    = 1,  /* valid JSON output                      */
    OC_GRAMMAR_CHOICE  = 2,  /* one of a set of literal strings        */
} OcGrammarType;

typedef struct OcGrammarConstraint {
    OcGrammarType type;
    /* For OC_GRAMMAR_CHOICE: array of allowed strings. */
    const char *const *choices;
    size_t n_choices;
    /* Current state: how many characters of the current choice have been
     * matched so far (for choice mode). */
    size_t matched_pos;
    size_t active_choice;   /* index into choices array              */
    bool in_string;         /* for JSON: are we inside a string?      */
    bool escaped;           /* for JSON: was the previous char \?     */
    bool started;           /* has any output been generated yet?     */
    bool finished;         /* has the grammar been satisfied?        */
    char json_stack[64];
    size_t json_depth;
} OcGrammarConstraint;

/* Initialize a grammar constraint. */
void oc_grammar_init(OcGrammarConstraint *g, OcGrammarType type);

/* Initialize a choice grammar with the given allowed strings. */
void oc_grammar_init_choice(OcGrammarConstraint *g,
                            const char *const *choices, size_t n_choices);

/* Check if a token is allowed by the grammar given the current state.
 * `token_bytes` is the decoded token text (UTF-8), `token_len` is its length.
 * Returns true if the token is allowed. */
bool oc_grammar_allows_token(OcGrammarConstraint *g,
                             const char *token_bytes, size_t token_len);

/* Update grammar state after a token has been accepted. */
void oc_grammar_advance(OcGrammarConstraint *g,
                        const char *token_bytes, size_t token_len);

/* Check if a single character is allowed at the current grammar state.
 * Used for character-level filtering during sampling. */
bool oc_grammar_allows_char(OcGrammarConstraint *g, char c);

/* Reset the grammar state (start a new generation). */
void oc_grammar_reset(OcGrammarConstraint *g);

/* Check if the grammar is satisfied (e.g., valid JSON complete). */
bool oc_grammar_is_satisfied(const OcGrammarConstraint *g);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_GRAMMAR_H */
