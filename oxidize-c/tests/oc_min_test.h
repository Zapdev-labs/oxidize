/*
 * oc_min_test.h — minimal Criterion-style test harness.
 *
 * NOTE: This is a temporary shim so the `core-types-error-arena` feature can
 * follow TDD before the `criterion-test-infra` feature vendors the real
 * Criterion (MIT) library. The `criterion-test-infra` worker will replace
 * this header + the test_runner glue with the real Criterion API. Tests
 * written against this header use a Criterion-shaped API (Test(name, case),
 * cr_assert_*) so they can be migrated with minimal edits.
 *
 * License: MIT (will be removed once real Criterion is vendored).
 */
#ifndef OC_MIN_TEST_H
#define OC_MIN_TEST_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Counters exposed for the runner. */
extern int oc_test_total;
extern int oc_test_failed;
extern int oc_test_asserts;

/* A test function pointer. */
typedef void (*oc_test_fn)(void);

/* Criterion-compatible macro: Test(name, desc) { body }
 * Declares a static function `oc_test_<name>_<desc>` which the runner
 * references by name via the OC_TEST_ENTRY() macro in a per-suite array. */
#define Test(name, desc)                                                        \
    static void oc_test_##name##_##desc(void)

/* Assert primitives (Criterion-shaped). On failure, prints to stderr and
 * early-returns from the test function (counts as a failure but does not
 * abort the whole suite). */
#define cr_assert(cond, ...)                                                    \
    do {                                                                       \
        oc_test_asserts++;                                                     \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL: %s:%d: %s: ", __FILE__, __LINE__, #cond);   \
            fprintf(stderr, __VA_ARGS__);                                      \
            fprintf(stderr, "\n");                                             \
            oc_test_failed++;                                                  \
            return;                                                            \
        }                                                                      \
    } while (0)

#define cr_assert_eq(a, b, ...) cr_assert((a) == (b), __VA_ARGS__)
#define cr_assert_ne(a, b, ...) cr_assert((a) != (b), __VA_ARGS__)
#define cr_assert_str_eq(a, b, ...)                                            \
    cr_assert(strcmp((a), (b)) == 0, __VA_ARGS__)
#define cr_assert_null(ptr, ...) cr_assert((ptr) == NULL, __VA_ARGS__)
#define cr_assert_not_null(ptr, ...) cr_assert((ptr) != NULL, __VA_ARGS__)
#define cr_expect(cond, ...) cr_assert(cond, __VA_ARGS__)
#define cr_expect_eq(a, b, ...) cr_assert_eq(a, b, __VA_ARGS__)
#define cr_expect_ne(a, b, ...) cr_assert_ne(a, b, __VA_ARGS__)

/* Test suite: a per-TU array of test function pointers, plus a register
 * function the runner calls. Usage in each test_<module>.c:
 *
 *   OC_TEST_SUITE_DEF(error)
 *   OC_TEST_ENTRY(error, msg_returns_ok)
 *   OC_TEST_ENTRY(error, msg_returns_io)
 *   ...
 *   OC_TEST_SUITE_END(error)
 *
 * The runner declares `OC_TEST_SUITE_REG(<suite>)` for each suite and calls
 * `oc_register_<suite>()` in main.
 */
#define OC_TEST_SUITE_REG(name)                                                \
    void oc_register_##name(void);

#define OC_TEST_SUITE_DEF(name)                                                \
    static oc_test_fn oc_suite_##name[] = {

#define OC_TEST_ENTRY(name, desc)                                              \
    oc_test_##name##_##desc,

#define OC_TEST_SUITE_END(name)                                                 \
    NULL                                                                        \
    };                                                                          \
    void oc_register_##name(void)                                              \
    {                                                                           \
        const oc_test_fn *p = oc_suite_##name;                                 \
        while (*p) {                                                           \
            oc_test_total++;                                                  \
            (*p)();                                                            \
            p++;                                                               \
        }                                                                       \
    }

#endif /* OC_MIN_TEST_H */
