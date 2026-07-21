/* test_grammar.c — grammar constraint tests. */
#include <criterion/criterion.h>
#include "oxidize/grammar.h"

Test(grammar, init_none)
{
    OcGrammarConstraint g;
    oc_grammar_init(&g, OC_GRAMMAR_NONE);
    cr_assert_eq(g.type, OC_GRAMMAR_NONE);
    cr_assert(oc_grammar_allows_char(&g, 'x'));
    cr_assert(oc_grammar_allows_char(&g, '{'));
}

Test(grammar, json_allows_structural)
{
    OcGrammarConstraint g;
    oc_grammar_init(&g, OC_GRAMMAR_JSON);
    cr_assert(oc_grammar_allows_char(&g, '{'));
    cr_assert(oc_grammar_allows_char(&g, '"'));
    cr_assert(oc_grammar_allows_char(&g, '}'));
    cr_assert(oc_grammar_allows_char(&g, ':'));
    cr_assert(oc_grammar_allows_char(&g, ','));
    cr_assert(!oc_grammar_allows_char(&g, 'x'));  /* x is not structural outside string */
}

Test(grammar, json_string_mode)
{
    OcGrammarConstraint g;
    oc_grammar_init(&g, OC_GRAMMAR_JSON);
    /* Enter string mode. */
    oc_grammar_advance(&g, "\"", 1);
    cr_assert(g.in_string);
    /* Inside string, any printable char is allowed. */
    cr_assert(oc_grammar_allows_char(&g, 'x'));
    cr_assert(oc_grammar_allows_char(&g, 'h'));
    cr_assert(oc_grammar_allows_char(&g, 'e'));
    /* Control chars not allowed. */
    cr_assert(!oc_grammar_allows_char(&g, '\x01'));
    /* Escape sequence. */
    oc_grammar_advance(&g, "\\", 1);
    cr_assert(g.escaped);
    cr_assert(oc_grammar_allows_char(&g, 'n'));
    cr_assert(!oc_grammar_allows_char(&g, 'x'));
    oc_grammar_advance(&g, "n", 1);
    cr_assert(!g.escaped);
    /* Close string. */
    oc_grammar_advance(&g, "\"", 1);
    cr_assert(!g.in_string);
}

Test(grammar, choice_mode)
{
    OcGrammarConstraint g;
    const char *choices[] = {"yes", "no"};
    oc_grammar_init_choice(&g, choices, 2);
    /* 'y' should be allowed (starts "yes"). */
    cr_assert(oc_grammar_allows_char(&g, 'y'));
    /* 'n' should be allowed (starts "no"). */
    cr_assert(oc_grammar_allows_char(&g, 'n'));
    /* 'x' should not be allowed. */
    cr_assert(!oc_grammar_allows_char(&g, 'x'));
    /* Advance with 'y'. */
    oc_grammar_advance(&g, "y", 1);
    cr_assert(oc_grammar_allows_char(&g, 'e'));
    cr_assert(!oc_grammar_allows_char(&g, 'n'));
    /* Complete "yes". */
    oc_grammar_advance(&g, "es", 2);
    cr_assert(g.finished);

    oc_grammar_reset(&g);
    cr_assert(oc_grammar_allows_token(&g, "no", 2));
    oc_grammar_advance(&g, "no", 2);
    cr_assert(g.finished);
}

Test(grammar, reset)
{
    OcGrammarConstraint g;
    oc_grammar_init(&g, OC_GRAMMAR_JSON);
    g.in_string = true;
    g.escaped = true;
    g.started = true;
    oc_grammar_reset(&g);
    cr_assert(!g.in_string);
    cr_assert(!g.escaped);
    cr_assert(!g.started);
}

Test(grammar, is_satisfied)
{
    OcGrammarConstraint g;
    oc_grammar_init(&g, OC_GRAMMAR_NONE);
    cr_assert(oc_grammar_is_satisfied(&g));
    
    oc_grammar_init(&g, OC_GRAMMAR_JSON);
    cr_assert(!oc_grammar_is_satisfied(&g));  /* not started */
    oc_grammar_advance(&g, "{", 1);
    cr_assert(!oc_grammar_is_satisfied(&g));
    oc_grammar_advance(&g, "}", 1);
    cr_assert(oc_grammar_is_satisfied(&g));
}

Test(grammar, token_allows)
{
    OcGrammarConstraint g;
    oc_grammar_init(&g, OC_GRAMMAR_JSON);
    /* Token '{"key":' should be allowed. */
    cr_assert(oc_grammar_allows_token(&g, "{\"key\":", 7));
    /* Token with invalid char outside string should be rejected. */
    cr_assert(!oc_grammar_allows_token(&g, "xyz", 3));
}
