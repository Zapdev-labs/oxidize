/* runner.c — minimal test runner.
 *
 * Calls the per-suite register functions (oc_register_<name>) declared by each
 * tests/test_<module>.c via OC_TEST_SUITE_DEF / OC_TEST_SUITE_END. Prints a
 * summary and exits 0 on success, 1 on any failure.
 *
 * The criterion-test-infra feature will replace this with the real Criterion
 * runner that links against libcriterion.a.
 */
#include "oc_min_test.h"

#include <stdio.h>
#include <stdlib.h>

int oc_test_total   = 0;
int oc_test_failed  = 0;
int oc_test_asserts = 0;

/* Per-suite registration entrypoints (one per test_<module>.c). */
OC_TEST_SUITE_REG(error)
OC_TEST_SUITE_REG(dtype)
OC_TEST_SUITE_REG(arena)
OC_TEST_SUITE_REG(hashtable)
OC_TEST_SUITE_REG(vector)
OC_TEST_SUITE_REG(log)
OC_TEST_SUITE_REG(string)
OC_TEST_SUITE_REG(bytes)

int main(void)
{
    fputs("=== oxidize-c test_runner (minimal harness) ===\n", stderr);

    oc_register_error();
    oc_register_dtype();
    oc_register_arena();
    oc_register_hashtable();
    oc_register_vector();
    oc_register_log();
    oc_register_string();
    oc_register_bytes();

    fprintf(stderr, "=== %d/%d tests passed, %d asserts ===\n",
            oc_test_total - oc_test_failed, oc_test_total, oc_test_asserts);
    return oc_test_failed == 0 ? 0 : 1;
}
