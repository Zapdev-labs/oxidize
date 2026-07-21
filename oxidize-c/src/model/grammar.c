/*
 * grammar.c — grammar constraint implementation.
 *
 * Provides JSON validation and choice-based constraints for structured output.
 */
#include "oxidize/grammar.h"

#include <stdlib.h>
#include <string.h>

void oc_grammar_init(OcGrammarConstraint *g, OcGrammarType type)
{
    if (!g) return;
    memset(g, 0, sizeof(*g));
    g->type = type;
}

void oc_grammar_init_choice(OcGrammarConstraint *g,
                            const char *const *choices, size_t n_choices)
{
    if (!g) return;
    memset(g, 0, sizeof(*g));
    g->type = OC_GRAMMAR_CHOICE;
    g->choices = choices;
    g->n_choices = n_choices;
}

/* Check if a character is valid in JSON outside a string. */
static bool json_structural_char(char c)
{
    /* Outside strings, JSON allows: { } [ ] : , whitespace, digits, true/false/null */
    switch (c) {
    case '{': case '}': case '[': case ']': case ':': case ',':
    case ' ': case '\t': case '\n': case '\r':
    case 't': case 'r': case 'u':  /* true */
    case 'f': case 'a': case 'l': case 's':  /* false */
    case 'n':  /* null */
    case '0': case '1': case '2': case '3': case '4':
    case '5': case '6': case '7': case '8': case '9':
    case '-': case '+': case '.':
    case 'E':
    case '"':
        return true;
    default:
        return false;
    }
}

/* Check if a character is valid inside a JSON string. */
static bool json_string_char(char c, bool escaped)
{
    if (escaped) {
        /* After backslash, only specific escapes are valid. */
        switch (c) {
        case '"': case '\\': case '/': case 'b': case 'f':
        case 'n': case 'r': case 't': case 'u':
            return true;
        default:
            return false;
        }
    }
    /* Unescaped control characters (0x00-0x1F) are invalid in JSON strings. */
    if ((unsigned char)c < 0x20) return false;
    return true;
}

bool oc_grammar_allows_char(OcGrammarConstraint *g, char c)
{
    if (!g || g->type == OC_GRAMMAR_NONE) return true;
    if (g->finished) return false;

    switch (g->type) {
    case OC_GRAMMAR_JSON:
        if (g->in_string) {
            if (g->escaped) {
                return json_string_char(c, true);
            }
            if (c == '\\') return true;
            if (c == '"') return true;
            return json_string_char(c, false);
        }
        return json_structural_char(c);

    case OC_GRAMMAR_CHOICE: {
        /* Check if c matches any choice at the current position. */
        for (size_t i = 0; i < g->n_choices; i++) {
            if (g->choices[i] == NULL) continue;
            size_t len = strlen(g->choices[i]);
            if (g->matched_pos < len && g->choices[i][g->matched_pos] == c) {
                return true;
            }
            /* Also allow matching from the start of any choice if we haven't
             * started matching yet. */
            if (!g->started && len > 0 && g->choices[i][0] == c) {
                return true;
            }
        }
        return false;
    }

    case OC_GRAMMAR_REGEX:
        /* Simplified: allow all characters (full regex not implemented). */
        return true;

    default:
        return true;
    }
}

bool oc_grammar_allows_token(OcGrammarConstraint *g,
                             const char *token_bytes, size_t token_len)
{
    if (!g || g->type == OC_GRAMMAR_NONE) return true;
    if (g->finished) return false;
    if (!token_bytes || token_len == 0) return true;

    /* Simulate advancing through the token to check if all chars are allowed. */
    OcGrammarConstraint sim = *g;
    for (size_t i = 0; i < token_len; i++) {
        if (!oc_grammar_allows_char(&sim, token_bytes[i])) {
            return false;
        }
        /* Update sim state. */
        switch (sim.type) {
        case OC_GRAMMAR_JSON:
            if (sim.in_string) {
                if (sim.escaped) {
                    sim.escaped = false;
                } else if (token_bytes[i] == '\\') {
                    sim.escaped = true;
                } else if (token_bytes[i] == '"') {
                    sim.in_string = false;
                }
            } else {
                if (token_bytes[i] == '"') {
                    sim.in_string = true;
                }
            }
            break;
        case OC_GRAMMAR_CHOICE:
            sim.matched_pos++;
            sim.started = true;
            break;
        default:
            break;
        }
    }
    return true;
}

void oc_grammar_advance(OcGrammarConstraint *g,
                        const char *token_bytes, size_t token_len)
{
    if (!g || !token_bytes) return;
    g->started = true;
    for (size_t i = 0; i < token_len; i++) {
        char c = token_bytes[i];
        switch (g->type) {
        case OC_GRAMMAR_JSON:
            if (g->in_string) {
                if (g->escaped) {
                    g->escaped = false;
                } else if (c == '\\') {
                    g->escaped = true;
                } else if (c == '"') {
                    g->in_string = false;
                }
            } else {
                if (c == '"') {
                    g->in_string = true;
                }
            }
            break;
        case OC_GRAMMAR_CHOICE:
            g->matched_pos++;
            /* Check if we've completed a choice. */
            if (g->active_choice < g->n_choices && g->choices[g->active_choice]) {
                if (g->matched_pos >= strlen(g->choices[g->active_choice])) {
                    g->finished = true;
                }
            }
            break;
        default:
            break;
        }
    }
}

void oc_grammar_reset(OcGrammarConstraint *g)
{
    if (!g) return;
    g->matched_pos = 0;
    g->active_choice = 0;
    g->in_string = false;
    g->escaped = false;
    g->started = false;
    g->finished = false;
}

bool oc_grammar_is_satisfied(const OcGrammarConstraint *g)
{
    if (!g) return true;
    if (g->type == OC_GRAMMAR_NONE) return true;
    if (g->type == OC_GRAMMAR_JSON) {
        /* JSON is satisfied when we're not inside a string and not escaped. */
        return !g->in_string && !g->escaped && g->started;
    }
    return g->finished;
}
