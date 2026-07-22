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
    if (g->finished && g->type != OC_GRAMMAR_CHOICE) return false;

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
        if (c == '{' || c == '[') return g->json_depth < sizeof(g->json_stack);
        if (c == '}' || c == ']') {
            if (g->json_depth == 0) return false;
            char open = g->json_stack[g->json_depth - 1];
            return (open == '{' && c == '}') || (open == '[' && c == ']');
        }
        return json_structural_char(c);

    case OC_GRAMMAR_CHOICE: {
        for (size_t i = 0; i < g->n_choices; i++) {
            if (g->choices[i] == NULL) continue;
            size_t len = strlen(g->choices[i]);
            if (g->matched_pos < len &&
                memcmp(g->choices[i], g->choice_prefix, g->matched_pos) == 0 &&
                g->choices[i][g->matched_pos] == c) {
                return true;
            }
        }
        return false;
    }

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
        oc_grammar_advance(&sim, &token_bytes[i], 1);
    }
    return true;
}

void oc_grammar_advance(OcGrammarConstraint *g,
                        const char *token_bytes, size_t token_len)
{
    if (!g || !token_bytes) return;
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
                    if (g->json_depth == 0) g->finished = true;
                }
            } else {
                if (c == '"') {
                    g->in_string = true;
                    if (g->json_depth == 0) g->started = true;
                } else if (c == '{' || c == '[') {
                    if (g->json_depth < sizeof(g->json_stack)) {
                        g->json_stack[g->json_depth++] = c;
                        g->started = true;
                    }
                } else if ((c == '}' || c == ']') && g->json_depth > 0) {
                    char open = g->json_stack[g->json_depth - 1];
                    if ((open == '{' && c == '}') || (open == '[' && c == ']')) {
                        g->json_depth--;
                        if (g->json_depth == 0) g->finished = true;
                    }
                } else if (g->json_depth == 0 && c != ' ' && c != '\t' &&
                           c != '\n' && c != '\r') {
                    g->started = true;
                    g->root_primitive = true;
                }
            }
            break;
        case OC_GRAMMAR_CHOICE:
            if (g->matched_pos >= sizeof(g->choice_prefix)) break;
            g->choice_prefix[g->matched_pos] = c;
            g->started = true;
            g->matched_pos++;
            g->finished = false;
            for (size_t choice = 0; choice < g->n_choices; choice++) {
                if (g->choices[choice] && strlen(g->choices[choice]) == g->matched_pos &&
                    memcmp(g->choices[choice], g->choice_prefix, g->matched_pos) == 0) {
                    g->active_choice = choice;
                    g->finished = true;
                    break;
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
    g->root_primitive = false;
    g->json_depth = 0;
    memset(g->json_stack, 0, sizeof(g->json_stack));
    memset(g->choice_prefix, 0, sizeof(g->choice_prefix));
}

bool oc_grammar_is_satisfied(const OcGrammarConstraint *g)
{
    if (!g) return true;
    if (g->type == OC_GRAMMAR_NONE) return true;
    if (g->type == OC_GRAMMAR_JSON) {
        return (g->finished || g->root_primitive) && !g->in_string &&
               !g->escaped && g->json_depth == 0;
    }
    return g->finished;
}
