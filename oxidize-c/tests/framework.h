/*
 * framework.h — minimal in-repo test framework.
 *
 * Drop-in replacement for the subset of Criterion (MIT, v2.4.3) that the
 * oxidize-c test suite actually uses, replacing 9k lines of vendored
 * headers plus a 2.6 MB prebuilt libcriterion.a (which also pinned the
 * CI runners to glibc >= 2.38). Behavior contract:
 *
 *   - Test(suite, case) auto-registration; framework_main.c provides main()
 *   - fork-per-test isolation: a crash kills one test, not the run
 *   - cr_assert* aborts the current test; cr_expect* records and continues
 *   - .description / .disabled extras; cr_skip_test()
 *   - CLI: --filter/--pattern glob, --list, --jobs N, --xml FILE, --help
 *   - exit code 0 iff no test failed or crashed (skips are non-failures)
 *
 * Output mirrors Criterion's summary line:
 *   [====] Synthesis: Tested: N | Passing: N | Failing: n | Crashing: n
 *                    [| Skipped: n] [| Disabled: n]
 *
 * This file is test infrastructure only — never installed, never part of
 * liboxidize-c.a.
 */
#ifndef OXIDIZE_C_TESTS_FRAMEWORK_H
#define OXIDIZE_C_TESTS_FRAMEWORK_H

#include <math.h>
#include <setjmp.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Test registry ──────────────────────────────────────────────────── */

typedef void (*OcTestFn)(void);

typedef struct OcTest {
    const char *suite;
    const char *case_name;
    const char *description;
    int disabled;
    OcTestFn fn;
    struct OcTest *next;
} OcTest;

extern OcTest *oc_tests_head;

void oc_test_register(OcTest *t);

/* Per-test state, owned by framework_main.c's runner. */
extern jmp_buf oc_test_abort_jmp;
extern int oc_test_failed;        /* any soft failure so far */
extern int oc_test_can_skip;      /* nonzero between runner setup and body */

/* Hard failure: prints, then longjmps to the runner (test FAIL). */
void oc_test_fail(const char *file, int line, const char *fmt, ...)
    __attribute__((noreturn));

/* Soft failure: prints, marks the test failed, and RETURNS (expect). */
void oc_test_soft_fail(const char *file, int line, const char *fmt, ...);

/* Skip: longjmps to the runner (test SKIP). Only valid inside a test. */
void oc_test_skip(const char *fmt, ...) __attribute__((noreturn));

/* Comparison failure: prints `lhs op rhs [sa vs sb]` then the optional
 * caller format. Fixed operands are emitted before `__VA_ARGS__` so a
 * message like `cr_assert_eq(a, b, "n=%d", n)` cannot bind `n` to `%s`. */
void oc_test_fail_cmp(const char *file, int line, const char *lhs,
                      const char *op, const char *rhs, const char *sa,
                      const char *sb, const char *fmt, ...)
    __attribute__((noreturn, format(printf, 8, 9)));
void oc_test_soft_fail_cmp(const char *file, int line, const char *lhs,
                           const char *op, const char *rhs, const char *sa,
                           const char *sb, const char *fmt, ...)
    __attribute__((format(printf, 8, 9)));

/* Filter matching, also exercised by test_framework.c. A slash-less
 * filter that already names a suite or case does not rewrite '_' → '/'. */
int oc_filter_use_underscore_rewrite(const char *filter, const OcTest *head);
int oc_filter_selects(const char *filter, const OcTest *t,
                      int rewrite_underscores);

/* Used by the value-printing assert macros; not for direct calls. */
const char *oc_test_vstr(const void *p);
const char *oc_test_vstr_ll(long long v);
const char *oc_test_vstr_ull(unsigned long long v);
const char *oc_test_vstr_d(double v);
const char *oc_test_vstr_ld(long double v);

/* ─── Test() definition ──────────────────────────────────────────────── */

/* Extras (`.description = "..."`, `.disabled = true`) are appended after
 * the positional initializer and use designated initializers, which C11
 * guarantees override earlier positional fields (6.7.9p19). An empty
 * __VA_ARGS__ leaves a trailing comma, also valid since C99. */
#define OC_TEST_REGISTER_(suite, case_name, fn, ...)                     \
    static void oc_test_ctor_##suite##_##case_name(void)                 \
        __attribute__((used, constructor));                              \
    static void oc_test_ctor_##suite##_##case_name(void)                 \
    {                                                                    \
        static OcTest oc_test_entry_##suite##_##case_name = {            \
            #suite, #case_name, NULL, 0, fn, NULL, __VA_ARGS__ };        \
        oc_test_register(&oc_test_entry_##suite##_##case_name);         \
    }

#define Test(suite, case_name, ...)                                      \
    static void oc_test_body_##suite##_##case_name(void);                \
    OC_TEST_REGISTER_(suite, case_name,                                  \
                      oc_test_body_##suite##_##case_name, __VA_ARGS__)   \
    static void oc_test_body_##suite##_##case_name(void)

/* ─── Assertion plumbing ─────────────────────────────────────────────── */

/* Type-selecting stringifier: (x) + 0 promotes enums/small ints to int,
 * keeps float/double/long double, and leaves pointer types (void* + 0 is
 * a GNU extension but compiles clean under -Wall -Wextra). */
#define OC_VSTR_(x)                                                       \
    _Generic((x) + 0,                                                     \
        int: oc_test_vstr_ll,                                             \
        unsigned int: oc_test_vstr_ull,                                   \
        long: oc_test_vstr_ll,                                            \
        unsigned long: oc_test_vstr_ull,                                  \
        long long: oc_test_vstr_ll,                                       \
        unsigned long long: oc_test_vstr_ull,                             \
        float: oc_test_vstr_d,                                            \
        double: oc_test_vstr_d,                                           \
        long double: oc_test_vstr_ld,                                     \
        default: oc_test_vstr)((x))

/* Two-value comparison. The predicate evaluates each operand once; the
 * failure stringifier may evaluate again (failure path only). */
#define OC_ASSERT_OP(a, b, op, ...)                                      \
    do {                                                                  \
        if (!((a) op (b)))                                                \
            oc_test_fail_cmp(__FILE__, __LINE__, #a, #op, #b,             \
                OC_VSTR_(a), OC_VSTR_(b), "" __VA_ARGS__);                \
    } while (0)

#define OC_EXPECT_OP(a, b, op, ...)                                      \
    do {                                                                  \
        if (!((a) op (b)))                                                \
            oc_test_soft_fail_cmp(__FILE__, __LINE__, #a, #op, #b,        \
                OC_VSTR_(a), OC_VSTR_(b), "" __VA_ARGS__);                \
    } while (0)

/* ─── Assertions: hard (abort test on failure) ───────────────────────── */

#define cr_fail(...) oc_test_fail(__FILE__, __LINE__, "" __VA_ARGS__)
#define cr_assert(cond, ...)                                              \
    do {                                                                  \
        if (!(cond))                                                      \
            oc_test_fail(__FILE__, __LINE__,                              \
                "assertion failed: %s " __VA_ARGS__, #cond);              \
    } while (0)
#define cr_assert_not(cond, ...) cr_assert(!(cond), __VA_ARGS__)

#define cr_assert_eq(a, b, ...)    OC_ASSERT_OP(a, b, ==, __VA_ARGS__)
#define cr_assert_neq(a, b, ...)   OC_ASSERT_OP(a, b, !=, __VA_ARGS__)
#define cr_assert_lt(a, b, ...)    OC_ASSERT_OP(a, b, <, __VA_ARGS__)
#define cr_assert_leq(a, b, ...)   OC_ASSERT_OP(a, b, <=, __VA_ARGS__)
#define cr_assert_gt(a, b, ...)    OC_ASSERT_OP(a, b, >, __VA_ARGS__)
#define cr_assert_geq(a, b, ...)   OC_ASSERT_OP(a, b, >=, __VA_ARGS__)

/* ─── Assertions: soft (record failure, keep running the test) ───────── */

#define cr_expect(cond, ...)                                              \
    do {                                                                  \
        if (!(cond))                                                      \
            oc_test_soft_fail(__FILE__, __LINE__,                         \
                "expectation failed: %s " __VA_ARGS__, #cond);           \
    } while (0)
#define cr_expect_not(cond, ...) cr_expect(!(cond), __VA_ARGS__)

#define cr_expect_eq(a, b, ...)    OC_EXPECT_OP(a, b, ==, __VA_ARGS__)
#define cr_expect_neq(a, b, ...)   OC_EXPECT_OP(a, b, !=, __VA_ARGS__)
#define cr_expect_lt(a, b, ...)    OC_EXPECT_OP(a, b, <, __VA_ARGS__)
#define cr_expect_leq(a, b, ...)   OC_EXPECT_OP(a, b, <=, __VA_ARGS__)
#define cr_expect_gt(a, b, ...)    OC_EXPECT_OP(a, b, >, __VA_ARGS__)
#define cr_expect_geq(a, b, ...)   OC_EXPECT_OP(a, b, >=, __VA_ARGS__)

/* ─── Floating point ─────────────────────────────────────────────────── */

#define cr_assert_float_eq(a, b, eps, ...)                               \
    do {                                                                  \
        double oc_fa_ = (double)(a), oc_fb_ = (double)(b),                \
               oc_eps_ = (double)(eps);                                   \
        if (!(fabs(oc_fa_ - oc_fb_) <= oc_eps_))                          \
            oc_test_fail(__FILE__, __LINE__,                              \
                "floats not within %s: [%s vs %s] " __VA_ARGS__,          \
                #eps, oc_test_vstr_d(oc_fa_), oc_test_vstr_d(oc_fb_));    \
    } while (0)

#define cr_expect_float_eq(a, b, eps, ...)                               \
    do {                                                                  \
        double oc_fa_ = (double)(a), oc_fb_ = (double)(b),                \
               oc_eps_ = (double)(eps);                                   \
        if (!(fabs(oc_fa_ - oc_fb_) <= oc_eps_))                          \
            oc_test_soft_fail(__FILE__, __LINE__,                         \
                "floats not within %s: [%s vs %s] " __VA_ARGS__,          \
                #eps, oc_test_vstr_d(oc_fa_), oc_test_vstr_d(oc_fb_));    \
    } while (0)

#define cr_assert_float_neq(a, b, eps, ...)                              \
    do {                                                                  \
        double oc_fa_ = (double)(a), oc_fb_ = (double)(b),                \
               oc_eps_ = (double)(eps);                                   \
        if (fabs(oc_fa_ - oc_fb_) <= oc_eps_)                             \
            oc_test_fail(__FILE__, __LINE__,                              \
                "floats equal within %s: [%s vs %s] " __VA_ARGS__,        \
                #eps, oc_test_vstr_d(oc_fa_), oc_test_vstr_d(oc_fb_));    \
    } while (0)

/* ─── Strings ────────────────────────────────────────────────────────── */

const char *oc_test_str_or_null(const char *s);

#define cr_assert_str_eq(a, b, ...)                                       \
    do {                                                                  \
        const char *oc_sa_ = (a), *oc_sb_ = (b);                          \
        if (!oc_sa_ || !oc_sb_ || strcmp(oc_sa_, oc_sb_) != 0)            \
            oc_test_fail(__FILE__, __LINE__,                              \
                "strings differ: [%s vs %s] " __VA_ARGS__,                \
                oc_test_str_or_null(oc_sa_), oc_test_str_or_null(oc_sb_)); \
    } while (0)

#define cr_expect_str_eq(a, b, ...)                                       \
    do {                                                                  \
        const char *oc_sa_ = (a), *oc_sb_ = (b);                          \
        if (!oc_sa_ || !oc_sb_ || strcmp(oc_sa_, oc_sb_) != 0)            \
            oc_test_soft_fail(__FILE__, __LINE__,                        \
                "strings differ: [%s vs %s] " __VA_ARGS__,               \
                oc_test_str_or_null(oc_sa_), oc_test_str_or_null(oc_sb_)); \
    } while (0)

#define cr_assert_str_neq(a, b, ...)                                      \
    do {                                                                  \
        const char *oc_sa_ = (a), *oc_sb_ = (b);                          \
        if (oc_sa_ && oc_sb_ && strcmp(oc_sa_, oc_sb_) == 0)              \
            oc_test_fail(__FILE__, __LINE__,                              \
                "strings equal: [%s vs %s] " __VA_ARGS__,                \
                oc_test_str_or_null(oc_sa_), oc_test_str_or_null(oc_sb_)); \
    } while (0)

#define cr_assert_str_not_empty(a, ...)                                   \
    do {                                                                  \
        const char *oc_sa_ = (a);                                         \
        if (!oc_sa_ || oc_sa_[0] == '\0')                                 \
            oc_test_fail(__FILE__, __LINE__,                              \
                "string empty: %s " __VA_ARGS__, #a);                     \
    } while (0)

/* ─── Bytes / memory ─────────────────────────────────────────────────── */

#define cr_assert_arr_eq(a, b, size, ...)                                \
    do {                                                                  \
        if (memcmp((a), (b), (size)) != 0)                                \
            oc_test_fail(__FILE__, __LINE__,                              \
                "arrays differ: %s vs %s (%zu bytes) " __VA_ARGS__,       \
                #a, #b, (size_t)(size));                                  \
    } while (0)

#define cr_assert_arr_neq(a, b, size, ...)                                \
    do {                                                                  \
        if (memcmp((a), (b), (size)) == 0)                                \
            oc_test_fail(__FILE__, __LINE__,                              \
                "arrays equal: %s vs %s (%zu bytes) " __VA_ARGS__,       \
                #a, #b, (size_t)(size));                                  \
    } while (0)

/* ─── Pointers ───────────────────────────────────────────────────────── */

#define cr_assert_null(p, ...)       cr_assert((p) == NULL, "" __VA_ARGS__)
#define cr_assert_not_null(p, ...)   cr_assert((p) != NULL, "" __VA_ARGS__)
#define cr_expect_null(p, ...)       cr_expect((p) == NULL, "" __VA_ARGS__)
#define cr_expect_not_null(p, ...)   cr_expect((p) != NULL, "" __VA_ARGS__)

/* ─── Flow control ───────────────────────────────────────────────────── */

#define cr_skip_test(...) oc_test_skip("" __VA_ARGS__)
#define cr_skip(...)      cr_skip_test(__VA_ARGS__)

/* ─── One-line regression cases ──────────────────────────────────────── */
/* These keep the exercised expression visible at the call site; they are
 * plain Test() bodies that happen to fit one line. */

/* fn(NULL) and friends must be no-ops: common C-port free/null-safety
 * contract; a crash or sanitizer failure fails the test. */
#define OC_TEST_NULL_SAFE(suite, name, ...)                               \
    Test(suite, name)                                                      \
    {                                                                     \
        __VA_ARGS__;                                                      \
    }

/* The call must be rejected with OC_ERR_INVALID_ARG on NULL input. */
#define OC_TEST_REJECTS_NULL(suite, name, call)                            \
    Test(suite, name)                                                      \
    {                                                                     \
        cr_assert_eq(call, OC_ERR_INVALID_ARG);                           \
    }

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_C_TESTS_FRAMEWORK_H */
