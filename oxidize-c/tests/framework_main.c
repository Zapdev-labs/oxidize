/*
 * framework_main.c — runner for the in-repo test framework (see framework.h).
 *
 * One process per test (fork) so a segfault or sanitizer abort kills only
 * that test, matching Criterion's isolation model. CLI flags:
 *
 *   --filter 'suite/case*'   Criterion-style glob (fnmatch), also accepts
 *                            bare patterns; when omitted, run everything.
 *   --list                   Print "suite/case" lines and exit 0.
 *   --jobs N                 Accepted for compatibility; tests always run
 *                            serially (fork-per-test is the parallelism
 *                            boundary Criterion used too).
 *   --xml FILE               JUnit-style results for CI dashboards.
 *   --verbose N              Accepted; level >= 1 prints per-test lines.
 *
 * Exit code: 0 iff every enabled selected test passed and none crashed.
 */
#if defined(__unix__) || defined(__APPLE__)
/* sigaction/fnmatch need POSIX visibility under strict -std=c11. */
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
    vreport(file, line, "assertion failed", fmt, ap);
    va_end(ap);
    oc_test_failed = 1;
    if (oc_test_can_skip)
        longjmp(oc_test_abort_jmp, 1);
    /* Outside a managed test (shouldn't happen): abort the process. */
    exit(1);
}

void oc_test_soft_fail(const char *file, int line, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vreport(file, line, "expectation failed", fmt, ap);
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
        exit(77); /* skip outside managed runner: mirror automake */
    longjmp(oc_test_abort_jmp, 2);
}

/* ─── Selection ──────────────────────────────────────────────────────── */

static int selected(const OcTest *t)
{
    if (!g_filter || !g_filter[0])
        return 1;
    char pat[256];
    snprintf(pat, sizeof(pat), "%s", g_filter);
    char full[512];
    snprintf(full, sizeof(full), "%s/%s", t->suite, t->case_name);
    if (fnmatch(pat, full, 0) == 0)
        return 1;
    /* Criterion also matches a bare case name. */
    if (fnmatch(pat, t->case_name, 0) == 0)
        return 1;
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
        struct sigaction sa;
        memset(&sa, 0, sizeof(sa));
        sa.sa_handler = child_signal_handler;
        sigaction(SIGSEGV, &sa, NULL);
        sigaction(SIGBUS, &sa, NULL);
        sigaction(SIGILL, &sa, NULL);
        sigaction(SIGFPE, &sa, NULL);
        sigaction(SIGABRT, &sa, NULL);

        oc_test_failed = 0;
        oc_test_can_skip = 1;
        int longjmp_rc = setjmp(oc_test_abort_jmp);
        if (longjmp_rc == 0)
            t->fn();
        oc_test_can_skip = 0;
        /* longjmp value 2 = skip (cr_skip_test), 1 = hard failure. */
        if (longjmp_rc == 2)
            _exit(77);
        _exit(oc_test_failed ? 1 : 0);
    }

    int st = 0;
    pid_t w;
    do {
        w = waitpid(pid, &st, 0);
    } while (w < 0 && errno == EINTR);
    if (w < 0) {
        /* Never reaped the child: treat as a crash, never a pass. */
        fprintf(stderr, "waitpid(%d) failed: %s\n", (int)pid, strerror(errno));
        return RESULT_CRASH;
    }
    if (WIFSIGNALED(st))
        return RESULT_CRASH;
    if (WIFEXITED(st)) {
        int code = WEXITSTATUS(st);
        if (code == 0)
            return RESULT_PASS;
        if (code == 1 || code == 2)
            return RESULT_FAIL; /* 2 = longjmp skip path used as fail marker */
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

/* Per-test XML outcome, parallel to the registry list. */
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
    printf("\n");
}

static void write_xml(const char *path, const OcStats *s,
                      const OcRunRecord *runs, size_t n_runs)
{
    FILE *f = fopen(path, "w");
    if (!f)
        return;
    fprintf(f, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    fprintf(f, "<testsuites tests=\"%zu\" failures=\"%zu\" "
               "errors=\"%zu\" time=\"0\">\n",
            s->tested, s->failing, s->crashing);
    /* Walk runs in registry order: registration is grouped per suite
     * (constructors emit per object file), so open a <testsuite> on every
     * suite-name change and close it before the next — one row per test,
     * no duplicates. */
    const char *cur_suite = NULL;
    size_t suite_tests = 0, suite_fails = 0, suite_crashes = 0;
    for (size_t i = 0; i < n_runs; i++) {
        const OcTest *t = runs[i].test;
        if (!cur_suite || strcmp(t->suite, cur_suite) != 0) {
            if (cur_suite)
                fprintf(f, "  </testsuite>\n");
            cur_suite = t->suite;
            suite_tests = 0;
            suite_fails = 0;
            suite_crashes = 0;
            for (size_t j = i; j < n_runs; j++) {
                const OcTest *u = runs[j].test;
                if (strcmp(u->suite, cur_suite) != 0)
                    break;
                suite_tests++;
                if (runs[j].result == RESULT_FAIL)
                    suite_fails++;
                else if (runs[j].result == RESULT_CRASH)
                    suite_crashes++;
            }
            fprintf(f, "  <testsuite name=\"%s\" tests=\"%zu\" "
                       "failures=\"%zu\" errors=\"%zu\">\n",
                    cur_suite, suite_tests, suite_fails, suite_crashes);
        }
        fprintf(f, "    <testcase classname=\"%s\" name=\"%s\"", t->suite,
                t->case_name);
        if (runs[i].result == RESULT_SKIP)
            fprintf(f, "><skipped/></testcase>\n");
        else if (runs[i].result == RESULT_FAIL)
            fprintf(f, "><failure/></testcase>\n");
        else if (runs[i].result == RESULT_CRASH)
            fprintf(f, "><error/></testcase>\n");
        else
            fprintf(f, "/>\n");
    }
    if (cur_suite)
        fprintf(f, "  </testsuite>\n");
    fprintf(f, "</testsuites>\n");
    fclose(f);
}

/* ─── main ───────────────────────────────────────────────────────────── */

static void usage(FILE *out, const char *prog)
{
    fprintf(out,
        "usage: %s [options] [filter]\n"
        "  --filter PAT   run only tests matching 'suite/case*' (fnmatch)\n"
        "  --pattern PAT  alias for --filter (Criterion compatibility)\n"
        "  --list         list tests and exit\n"
        "  --xml FILE     write JUnit-style results\n"
        "  --jobs N       accepted; tests run serially either way\n"
        "  --verbose N    0=quiet, 1=print passes, higher=more\n",
        prog);
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
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(stdout, argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--filter") == 0 ||
                   strcmp(argv[i], "--pattern") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s: %s requires a value\n", argv[0], argv[i]);
                usage(stderr, argv[0]);
                return 2;
            }
            g_filter = argv[++i];
        } else if (strcmp(argv[i], "--xml") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s: --xml requires a value\n", argv[0]);
                usage(stderr, argv[0]);
                return 2;
            }
            xml_path = argv[++i];
        } else if (strcmp(argv[i], "--jobs") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s: --jobs requires a value\n", argv[0]);
                usage(stderr, argv[0]);
                return 2;
            }
            jobs = atoi(argv[++i]);
            (void)jobs;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s: --verbose requires a value\n", argv[0]);
                usage(stderr, argv[0]);
                return 2;
            }
            g_verbose = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--short-filename") == 0 ||
                   strcmp(argv[i], "--full-statistics") == 0 ||
                   strcmp(argv[i], "--always-exit-0") == 0 ||
                   strcmp(argv[i], "--no-early-exit") == 0) {
            /* Accepted no-ops for Criterion compatibility. */
        } else if (argv[i][0] == '-' && argv[i][1] == '-' &&
                   strcmp(argv[i], "--") != 0) {
            /* Unknown long flags are rejected rather than silently becoming
             * a filter that matches nothing (a false green). */
            fprintf(stderr, "%s: unknown option: %s\n", argv[0], argv[i]);
            usage(stderr, argv[0]);
            return 2;
        } else {
            /* Positional: treat as a filter for convenience. */
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
            /* Skip is a non-failure, not a pass: keep it out of Passing. */
            stats.skipped++;
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
