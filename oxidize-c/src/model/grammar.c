/*
 * grammar.c — grammar constraint implementation.
 *
 * Provides JSON validation and choice-based constraints for structured output.
 */
#include "oxidize/grammar.h"

#include <stdlib.h>
#include <string.h>

enum {
    JSON_ROOT_NONE,
    JSON_ROOT_TRUE,
    JSON_ROOT_FALSE,
    JSON_ROOT_NULL,
    JSON_ROOT_NUMBER,
};

enum {
    JSON_NUM_SIGN,
    JSON_NUM_ZERO,
    JSON_NUM_INT,
    JSON_NUM_DOT,
    JSON_NUM_FRAC,
    JSON_NUM_EXP,
    JSON_NUM_EXP_SIGN,
    JSON_NUM_EXP_DIGITS,
};

static bool json_space(char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

static const char *json_root_literal(uint8_t kind)
{
    if (kind == JSON_ROOT_TRUE) return "true";
    if (kind == JSON_ROOT_FALSE) return "false";
    if (kind == JSON_ROOT_NULL) return "null";
    return NULL;
}

static bool json_number_complete(uint8_t state)
{
    return state == JSON_NUM_ZERO || state == JSON_NUM_INT ||
           state == JSON_NUM_FRAC || state == JSON_NUM_EXP_DIGITS;
}

static bool json_number_allows(uint8_t state, char c)
{
    switch (state) {
    case JSON_NUM_SIGN: return c == '0' || (c >= '1' && c <= '9');
    case JSON_NUM_ZERO: return c == '.' || c == 'e' || c == 'E' || json_space(c);
    case JSON_NUM_INT: return (c >= '0' && c <= '9') || c == '.' ||
                              c == 'e' || c == 'E' || json_space(c);
    case JSON_NUM_DOT: return c >= '0' && c <= '9';
    case JSON_NUM_FRAC: return (c >= '0' && c <= '9') || c == 'e' ||
                               c == 'E' || json_space(c);
    case JSON_NUM_EXP: return c == '+' || c == '-' || (c >= '0' && c <= '9');
    case JSON_NUM_EXP_SIGN: return c >= '0' && c <= '9';
    case JSON_NUM_EXP_DIGITS: return (c >= '0' && c <= '9') || json_space(c);
    default: return false;
    }
}

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
    g->viable_choices = n_choices >= 64 ? UINT64_MAX : (UINT64_C(1) << n_choices) - 1;
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
    if (g->finished && g->type != OC_GRAMMAR_CHOICE &&
        !(g->type == OC_GRAMMAR_JSON && g->json_root_kind == JSON_ROOT_NUMBER))
        return g->type == OC_GRAMMAR_JSON &&
               (c == ' ' || c == '\t' || c == '\n' || c == '\r');

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
        if (g->json_depth == 0 && g->json_root_kind != JSON_ROOT_NONE) {
            const char *literal = json_root_literal(g->json_root_kind);
            if (literal) {
                size_t len = strlen(literal);
                return g->json_root_state < len
                    ? c == literal[g->json_root_state] : json_space(c);
            }
            return json_number_allows(g->json_root_state, c);
        }
        if (c == '{' || c == '[') return g->json_depth < sizeof(g->json_stack);
        if (c == '}' || c == ']') {
            if (g->json_depth == 0) return false;
            char open = g->json_stack[g->json_depth - 1];
            return (open == '{' && c == '}') || (open == '[' && c == ']');
        }
        if (g->json_depth == 0 && !g->started)
            return json_space(c) || c == '"' || c == 't' || c == 'f' ||
                   c == 'n' || c == '-' || (c >= '0' && c <= '9');
        return json_structural_char(c);

    case OC_GRAMMAR_CHOICE: {
        for (size_t i = 0; i < g->n_choices; i++) {
            if (i >= 64 || (g->viable_choices & (UINT64_C(1) << i)) == 0 ||
                g->choices[i] == NULL) continue;
            size_t len = strlen(g->choices[i]);
            if (g->matched_pos < len && g->choices[i][g->matched_pos] == c) {
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
    if (g->finished && g->type != OC_GRAMMAR_CHOICE &&
        !(g->type == OC_GRAMMAR_JSON && g->json_root_kind == JSON_ROOT_NUMBER))
        return false;
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
                } else if (g->json_depth == 0 && !g->started && !json_space(c)) {
                    g->started = true;
                    if (c == 't') g->json_root_kind = JSON_ROOT_TRUE;
                    else if (c == 'f') g->json_root_kind = JSON_ROOT_FALSE;
                    else if (c == 'n') g->json_root_kind = JSON_ROOT_NULL;
                    else {
                        g->json_root_kind = JSON_ROOT_NUMBER;
                        if (c == '-') g->json_root_state = JSON_NUM_SIGN;
                        else if (c == '0') g->json_root_state = JSON_NUM_ZERO;
                        else g->json_root_state = JSON_NUM_INT;
                        g->finished = json_number_complete(g->json_root_state);
                        break;
                    }
                    g->json_root_state = 1;
                } else if (g->json_depth == 0 && g->json_root_kind != JSON_ROOT_NONE &&
                           !json_space(c)) {
                    const char *literal = json_root_literal(g->json_root_kind);
                    if (literal) {
                        g->json_root_state++;
                        g->finished = g->json_root_state == strlen(literal);
                    } else {
                        switch (g->json_root_state) {
                        case JSON_NUM_SIGN: g->json_root_state = c == '0' ? JSON_NUM_ZERO : JSON_NUM_INT; break;
                        case JSON_NUM_ZERO:
                        case JSON_NUM_INT:
                            if (c == '.') g->json_root_state = JSON_NUM_DOT;
                            else if (c == 'e' || c == 'E') g->json_root_state = JSON_NUM_EXP;
                            break;
                        case JSON_NUM_DOT: g->json_root_state = JSON_NUM_FRAC; break;
                        case JSON_NUM_FRAC: if (c == 'e' || c == 'E') g->json_root_state = JSON_NUM_EXP; break;
                        case JSON_NUM_EXP: g->json_root_state = (c == '+' || c == '-') ? JSON_NUM_EXP_SIGN : JSON_NUM_EXP_DIGITS; break;
                        case JSON_NUM_EXP_SIGN: g->json_root_state = JSON_NUM_EXP_DIGITS; break;
                        case JSON_NUM_EXP_DIGITS: break;
                        }
                        g->finished = json_number_complete(g->json_root_state);
                    }
                }
            }
            break;
        case OC_GRAMMAR_CHOICE:
            g->started = true;
            for (size_t choice = 0; choice < g->n_choices && choice < 64; choice++) {
                if ((g->viable_choices & (UINT64_C(1) << choice)) != 0 &&
                    (!g->choices[choice] ||
                     g->choices[choice][g->matched_pos] != c))
                    g->viable_choices &= ~(UINT64_C(1) << choice);
            }
            g->matched_pos++;
            g->finished = false;
            for (size_t choice = 0; choice < g->n_choices && choice < 64; choice++) {
                if ((g->viable_choices & (UINT64_C(1) << choice)) != 0 &&
                    g->choices[choice] &&
                    strlen(g->choices[choice]) == g->matched_pos) {
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
    g->json_root_kind = 0;
    g->json_root_state = 0;
    g->json_depth = 0;
    memset(g->json_stack, 0, sizeof(g->json_stack));
    g->viable_choices = g->n_choices >= 64 ? UINT64_MAX :
                        (UINT64_C(1) << g->n_choices) - 1;
}

bool oc_grammar_is_satisfied(const OcGrammarConstraint *g)
{
    if (!g) return true;
    if (g->type == OC_GRAMMAR_NONE) return true;
    if (g->type == OC_GRAMMAR_JSON) {
        return g->finished && !g->in_string && !g->escaped && g->json_depth == 0;
    }
    return g->finished;
}
