/*
 * framework_main.c — runner for the in-repo test framework (see framework.h).
 *
 * One process per test (fork) so a segfault or sanitizer abort kills only
 * that test, matching Criterion's isolation model. CLI flags:
 *
 *   --filter 'suite/case*'   Criterion-style glob (fnmatch); also accepts
 *                            a bare suite name, a bare case name, and
 *                            underscore form (kv_cache_init → kv_cache/init).
 *   --pattern GLOB           alias for --filter
 *   --list                   Print "suite/case" lines and exit 0.
 *   --jobs N                 Accepted for compatibility; tests always run
 *                            serially (fork-per-test is the isolation
 *                            boundary).
 *   --xml FILE               JUnit-style results for CI dashboards.
 *   --verbose N              Level >= 1 prints per-test PASS lines.
 *   --help                   Flag reference.
 *
 * Exit code: 0 iff every enabled selected test passed or skipped (no
 * failures, no crashes). Exit 2 on CLI errors.
 */
#if defined(__unix__) || defined(__APPLE__)
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#endif

#include "framework.h"

#include <errno.h>
#include <fnmatch.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

OcTest *oc_tests_head = NULL;
static size_t oc_tests_registered = 0;

size_t oc_test_count(void);

jmp_buf oc_test_abort_jmp;
int oc_test_failed = 0;
int oc_test_can_skip = 0;

static const char *g_filter = NULL;
static int g_verbose = 0;

/* ─── Registry ───────────────────────────────────────────────────────── */

size_t oc_test_count(void)
{
    return oc_tests_registered;
}

void oc_test_register(OcTest *t)
{
    oc_tests_registered++;
    /* Constructors run in link order; keep registration order stable. */
    t->next = NULL;
    static OcTest **tail = &oc_tests_head;
    *tail = t;
    tail = &t->next;
}

/* ─── Per-test state helpers ─────────────────────────────────────────── */

static __thread char g_vstr_bufs[4][40];
static __thread int g_vstr_idx;

static const char *vstr_take(void)
{
    g_vstr_idx = (g_vstr_idx + 1) & 3;
    return g_vstr_bufs[g_vstr_idx];
}

const char *oc_test_vstr(const void *p)
{
    char *b = (char *)vstr_take();
    snprintf(b, sizeof(g_vstr_bufs[0]), "%p", p);
    return b;
}

const char *oc_test_vstr_ll(long long v)
{
    char *b = (char *)vstr_take();
    snprintf(b, sizeof(g_vstr_bufs[0]), "%lld", v);
    return b;
}

const char *oc_test_vstr_ull(unsigned long long v)
{
    char *b = (char *)vstr_take();
    snprintf(b, sizeof(g_vstr_bufs[0]), "%llu", v);
    return b;
}

const char *oc_test_vstr_d(double v)
{
    char *b = (char *)vstr_take();
    snprintf(b, sizeof(g_vstr_bufs[0]), "%g", v);
    return b;
}

const char *oc_test_vstr_ld(long double v)
{
    char *b = (char *)vstr_take();
    snprintf(b, sizeof(g_vstr_bufs[0]), "%Lg", v);
    return b;
}

const char *oc_test_str_or_null(const char *s)
{
    return s ? s : "(null)";
}

/* ─── Failure paths ──────────────────────────────────────────────────── */

static void vreport(const char *file, int line, const char *kind,
                    const char *fmt, va_list ap)
{
    fprintf(stderr, "[----] %s:%d: %s: ", file, line, kind);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    fflush(stderr);
}

void oc_test_fail(const char *file, int line, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vreport(file, line, "FAIL", fmt, ap);
    va_end(ap);
    oc_test_failed = 1;
    if (oc_test_can_skip)
        longjmp(oc_test_abort_jmp, 1);
    exit(1);
}

void oc_test_soft_fail(const char *file, int line, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vreport(file, line, "EXPECT", fmt, ap);
    va_end(ap);
    oc_test_failed = 1;
}

void oc_test_skip(const char *fmt, ...)
{
    va_list ap;
    fprintf(stderr, "[SKIP] ");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    if (!oc_test_can_skip)
        exit(77);
    longjmp(oc_test_abort_jmp, 2);
}

/* ─── Selection ──────────────────────────────────────────────────────── */

static int match_full(const char *pat, const OcTest *t)
{
    char full[512];
    snprintf(full, sizeof(full), "%s/%s", t->suite, t->case_name);
    if (fnmatch(pat, full, 0) == 0)
        return 1;
    if (fnmatch(pat, t->case_name, 0) == 0)
        return 1;
    if (fnmatch(pat, t->suite, 0) == 0)
        return 1;
    return 0;
}

static int selected(const OcTest *t)
{
    if (!g_filter || !g_filter[0])
        return 1;
    if (match_full(g_filter, t))
        return 1;
    /* TEST_FILTER=test_kv_cache_init (after stripping test_) should match
     * kv_cache/init. Try each '_' → '/' rewrite; also allow a trailing glob
     * so test_gguf_v3_header matches gguf/v3_header*. */
    if (strchr(g_filter, '/') || strpbrk(g_filter, "*?["))
        return 0;
    size_t n = strlen(g_filter);
    if (n == 0 || n >= 255)
        return 0;
    char alt[256];
    memcpy(alt, g_filter, n + 1);
    for (char *p = alt; *p; p++) {
        if (*p != '_')
            continue;
        *p = '/';
        if (match_full(alt, t))
            return 1;
        char glob[260];
        snprintf(glob, sizeof(glob), "%s*", alt);
        if (match_full(glob, t))
            return 1;
        *p = '_';
    }
    return 0;
}

/* ─── Fork-per-test runner ───────────────────────────────────────────── */

typedef enum {
    RESULT_PASS,
    RESULT_FAIL,
    RESULT_SKIP,
    RESULT_CRASH
} OcResult;

static volatile sig_atomic_t g_child_sig;

static void child_signal_handler(int sig)
{
    g_child_sig = sig;
    _exit(128 + sig);
}

static OcResult run_one(OcTest *t)
{
    fflush(NULL);

    pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "fork failed: %s\n", strerror(errno));
        return RESULT_CRASH;
    }
    if (pid == 0) {
        /* ASan owns SEGV/ABRT; chaining our handler would swallow reports. */
#if !defined(__SANITIZE_ADDRESS__)
        struct sigaction sa;
        memset(&sa, 0, sizeof(sa));
        sa.sa_handler = child_signal_handler;
        sigaction(SIGSEGV, &sa, NULL);
        sigaction(SIGBUS, &sa, NULL);
        sigaction(SIGILL, &sa, NULL);
        sigaction(SIGFPE, &sa, NULL);
        sigaction(SIGABRT, &sa, NULL);
#else
        (void)child_signal_handler;
        (void)g_child_sig;
#endif

        oc_test_failed = 0;
        oc_test_can_skip = 1;
        int longjmp_rc = setjmp(oc_test_abort_jmp);
        if (longjmp_rc == 0)
            t->fn();
        oc_test_can_skip = 0;
        /* 2 = skip. A prior cr_expect failure still fails the test. */
        if (longjmp_rc == 2 && !oc_test_failed)
            _exit(77);
        _exit(oc_test_failed || longjmp_rc == 1 ? 1 : 0);
    }

    int st = 0;
    pid_t waited;
    do {
        waited = waitpid(pid, &st, 0);
    } while (waited < 0 && errno == EINTR);
    if (waited < 0) {
        fprintf(stderr, "waitpid failed: %s\n", strerror(errno));
        return RESULT_CRASH;
    }
    if (WIFSIGNALED(st))
        return RESULT_CRASH;
    if (WIFEXITED(st)) {
        int code = WEXITSTATUS(st);
        if (code == 0)
            return RESULT_PASS;
        if (code == 1 || code == 2)
            return RESULT_FAIL;
        if (code == 77)
            return RESULT_SKIP;
        return RESULT_CRASH;
    }
    return RESULT_CRASH;
}

/* ─── Reporting ───────────────────────────────────────────────────────── */

typedef struct {
    size_t tested, passing, failing, crashing, skipped, disabled;
} OcStats;

typedef struct {
    const OcTest *test;
    OcResult result;
} OcRunRecord;

static void print_synthesis(const OcStats *s)
{
    printf("[====] Synthesis: Tested: %zu | Passing: %zu | Failing: %zu | "
           "Crashing: %zu",
           s->tested, s->passing, s->failing, s->crashing);
    if (s->skipped)
        printf(" | Skipped: %zu", s->skipped);
    if (s->disabled)
        printf(" | Disabled: %zu", s->disabled);
    printf("\n");
}

static void xml_text(FILE *f, const char *s)
{
    for (; s && *s; s++) {
        switch (*s) {
        case '&':
            fputs("&amp;", f);
            break;
        case '<':
            fputs("&lt;", f);
            break;
        case '>':
            fputs("&gt;", f);
            break;
        case '"':
            fputs("&quot;", f);
            break;
        default:
            fputc((unsigned char)*s, f);
            break;
        }
    }
}

static int suite_already_emitted(const OcRunRecord *runs, size_t i)
{
    const char *suite = runs[i].test->suite;
    for (size_t j = 0; j < i; j++) {
        if (strcmp(runs[j].test->suite, suite) == 0)
            return 1;
    }
    return 0;
}

static void write_xml(const char *path, const OcStats *s,
                      const OcRunRecord *runs, size_t n_runs)
{
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "failed to write XML to %s: %s\n", path,
                strerror(errno));
        return;
    }
    fprintf(f, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    fprintf(f, "<testsuites tests=\"%zu\" failures=\"%zu\" "
               "errors=\"%zu\" skipped=\"%zu\" time=\"0\">\n",
            s->tested, s->failing, s->crashing, s->skipped);

    for (size_t i = 0; i < n_runs; i++) {
        if (suite_already_emitted(runs, i))
            continue;
        const char *suite = runs[i].test->suite;
        size_t suite_tests = 0, suite_fails = 0, suite_errors = 0,
               suite_skips = 0;
        for (size_t k = 0; k < n_runs; k++) {
            if (strcmp(runs[k].test->suite, suite) != 0)
                continue;
            suite_tests++;
            if (runs[k].result == RESULT_FAIL)
                suite_fails++;
            else if (runs[k].result == RESULT_CRASH)
                suite_errors++;
            else if (runs[k].result == RESULT_SKIP)
                suite_skips++;
        }
        fprintf(f, "  <testsuite name=\"");
        xml_text(f, suite);
        fprintf(f, "\" tests=\"%zu\" failures=\"%zu\" errors=\"%zu\" "
                   "skipped=\"%zu\">\n",
                suite_tests, suite_fails, suite_errors, suite_skips);
        for (size_t k = 0; k < n_runs; k++) {
            if (strcmp(runs[k].test->suite, suite) != 0)
                continue;
            const OcTest *t = runs[k].test;
            fprintf(f, "    <testcase classname=\"");
            xml_text(f, t->suite);
            fprintf(f, "\" name=\"");
            xml_text(f, t->case_name);
            fprintf(f, "\"");
            if (runs[k].result == RESULT_SKIP)
                fprintf(f, "><skipped/></testcase>\n");
            else if (runs[k].result == RESULT_FAIL)
                fprintf(f, "><failure/></testcase>\n");
            else if (runs[k].result == RESULT_CRASH)
                fprintf(f, "><error/></testcase>\n");
            else
                fprintf(f, "/>\n");
        }
        fprintf(f, "  </testsuite>\n");
    }
    fprintf(f, "</testsuites>\n");
    fclose(f);
}

/* ─── main ───────────────────────────────────────────────────────────── */

static void usage(FILE *out)
{
    fprintf(out,
            "Usage: test_runner [options] [filter]\n"
            "  --filter GLOB    suite/case glob (fnmatch). A bare suite\n"
            "                   name runs that suite. Underscores may stand\n"
            "                   in for the suite/case slash.\n"
            "  --pattern GLOB   alias for --filter\n"
            "  --list           print suite/case lines and exit 0\n"
            "  --xml FILE       JUnit XML results\n"
            "  --jobs N         accepted; runs stay serial\n"
            "  --verbose N      0=quiet (default), >=1 print passes\n"
            "  --help           this message\n");
}

static const char *require_arg(int argc, char **argv, int *i, const char *flag)
{
    if (*i + 1 >= argc) {
        fprintf(stderr, "%s requires a value\n", flag);
        usage(stderr);
        return NULL;
    }
    return argv[++(*i)];
}

int main(int argc, char **argv)
{
    const char *xml_path = NULL;
    int jobs = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--list") == 0) {
            for (OcTest *t = oc_tests_head; t; t = t->next)
                printf("%s/%s%s\n", t->suite, t->case_name,
                       t->disabled ? " (disabled)" : "");
            return 0;
        } else if (strcmp(argv[i], "--help") == 0 ||
                   strcmp(argv[i], "-h") == 0) {
            usage(stdout);
            return 0;
        } else if (strcmp(argv[i], "--filter") == 0 ||
                   strcmp(argv[i], "--pattern") == 0) {
            const char *v = require_arg(argc, argv, &i, argv[i]);
            if (!v)
                return 2;
            g_filter = v;
        } else if (strcmp(argv[i], "--xml") == 0) {
            const char *v = require_arg(argc, argv, &i, argv[i]);
            if (!v)
                return 2;
            xml_path = v;
        } else if (strcmp(argv[i], "--jobs") == 0) {
            const char *v = require_arg(argc, argv, &i, argv[i]);
            if (!v)
                return 2;
            jobs = atoi(v);
            (void)jobs;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            const char *v = require_arg(argc, argv, &i, argv[i]);
            if (!v)
                return 2;
            g_verbose = atoi(v);
        } else if (strcmp(argv[i], "--short-filename") == 0 ||
                   strcmp(argv[i], "--full-statistics") == 0 ||
                   strcmp(argv[i], "--always-exit-0") == 0 ||
                   strcmp(argv[i], "--no-early-exit") == 0) {
            /* Criterion leftovers: accepted and ignored. */
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "unknown flag: %s\n", argv[i]);
            usage(stderr);
            return 2;
        } else {
            g_filter = argv[i];
        }
    }

    OcStats stats = {0};
    OcRunRecord *runs = calloc(oc_test_count(), sizeof(OcRunRecord));
    size_t n_runs = 0;
    for (OcTest *t = oc_tests_head; t; t = t->next) {
        if (!selected(t))
            continue;
        if (t->disabled) {
            stats.disabled++;
            continue;
        }
        stats.tested++;
        OcResult r = run_one(t);
        if (runs) {
            runs[n_runs].test = t;
            runs[n_runs].result = r;
            n_runs++;
        }
        switch (r) {
        case RESULT_PASS:
            stats.passing++;
            if (g_verbose)
                printf("[PASS] %s/%s\n", t->suite, t->case_name);
            break;
        case RESULT_FAIL:
            stats.failing++;
            printf("[FAIL] %s::%s\n", t->suite, t->case_name);
            break;
        case RESULT_SKIP:
            stats.skipped++;
            if (g_verbose)
                printf("[SKIP] %s/%s\n", t->suite, t->case_name);
            break;
        case RESULT_CRASH:
            stats.crashing++;
            printf("[CRSH] %s::%s\n", t->suite, t->case_name);
            break;
        }
    }

    print_synthesis(&stats);
    if (xml_path)
        write_xml(xml_path, &stats, runs, n_runs);
    free(runs);
    return (stats.failing || stats.crashing) ? 1 : 0;
}
