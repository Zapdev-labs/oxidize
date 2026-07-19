/* test_arena.c — OcArena bump allocator tests.
 *
 * Verifies bump-pointer semantics, alignment, growth, dup/printf helpers, and
 * 1000 alloc/free cycles (ASan-clean substitutes for valgrind which is not
 * installed locally).
 */
#include "oc_min_test.h"
#include "oxidize/arena.h"

#include <stdint.h>
#include <string.h>

Test(arena, new_and_free)
{
    OcArena *a = oc_arena_new(0);
    cr_assert_not_null(a, "arena should allocate");
    oc_arena_free(a);
}

Test(arena, alloc_returns_aligned)
{
    OcArena *a = oc_arena_new(4096);
    cr_assert_not_null(a, "");
    void *p1 = oc_arena_alloc(a, 1, 16);
    cr_assert_not_null(p1, "");
    cr_assert_eq((uintptr_t)p1 & 15, 0, "should be 16-aligned");
    void *p2 = oc_arena_alloc(a, 1, 256);
    cr_assert_not_null(p2, "");
    cr_assert_eq((uintptr_t)p2 & 255, 0, "should be 256-aligned");
    oc_arena_free(a);
}

Test(arena, alloc_writes_dont_crash)
{
    OcArena *a = oc_arena_new(4096);
    char *p = (char *)oc_arena_alloc(a, 256, 1);
    cr_assert_not_null(p, "");
    memset(p, 'A', 256);
    for (int i = 0; i < 256; i++) {
        cr_assert_eq(p[i], 'A', "byte %d", i);
    }
    oc_arena_free(a);
}

Test(arena, grows_past_initial_cap)
{
    OcArena *a = oc_arena_new(4096);  /* small initial */
    cr_assert_not_null(a, "");
    /* Allocate 1 MiB total in 1 KiB chunks — forces multiple grows. */
    for (int i = 0; i < 1024; i++) {
        void *p = oc_arena_alloc(a, 1024, 16);
        cr_assert_not_null(p, "alloc %d", i);
        memset(p, i & 0xff, 1024);
    }
    cr_assert(oc_arena_used(a) >= 1024 * 1024, "should have grown");
    oc_arena_free(a);
}

Test(arena, dup_and_dup_n)
{
    OcArena *a = oc_arena_new(4096);
    char *s1 = oc_arena_dup(a, "hello");
    cr_assert_str_eq(s1, "hello", "");
    char *s2 = oc_arena_dup_n(a, "world\0hidden", 5);
    cr_assert_str_eq(s2, "world", "");
    oc_arena_free(a);
}

Test(arena, printf)
{
    OcArena *a = oc_arena_new(4096);
    char *s = oc_arena_printf(a, "n=%d s=%s", 42, "hi");
    cr_assert_str_eq(s, "n=42 s=hi", "");
    oc_arena_free(a);
}

Test(arena, reset_reuses_buffer)
{
    OcArena *a = oc_arena_new(4096);
    void *p1 = oc_arena_alloc(a, 128, 16);
    cr_assert_not_null(p1, "");
    size_t used_before = oc_arena_used(a);
    cr_assert(used_before >= 128, "");
    oc_arena_reset(a);
    cr_assert_eq(oc_arena_used(a), 0, "reset should zero used");
    void *p2 = oc_arena_alloc(a, 128, 16);
    cr_assert_not_null(p2, "");
    oc_arena_free(a);
}

Test(arena, thousand_alloc_free_cycles)
{
    /* Mirrors the "valgrind-clean on 1000 alloc/free cycles" requirement.
     * ASan substitutes for valgrind (not installed). */
    for (int cycle = 0; cycle < 1000; cycle++) {
        OcArena *a = oc_arena_new(1024);
        cr_assert_not_null(a, "cycle %d", cycle);
        for (int i = 0; i < 32; i++) {
            void *p = oc_arena_alloc(a, 64, 16);
            cr_assert_not_null(p, "cycle %d alloc %d", cycle, i);
            memset(p, i, 64);
        }
        oc_arena_free(a);
    }
}

Test(arena, null_arena_returns_null)
{
    cr_assert_null(oc_arena_alloc(NULL, 16, 16), "");
    cr_assert_null(oc_arena_alloc_bytes(NULL, 16), "");
    cr_assert_null(oc_arena_dup(NULL, "x"), "");
    cr_assert_eq(oc_arena_used(NULL), 0, "");
    oc_arena_free(NULL);  /* must not crash */
}

OC_TEST_SUITE_DEF(arena)
OC_TEST_ENTRY(arena, new_and_free)
OC_TEST_ENTRY(arena, alloc_returns_aligned)
OC_TEST_ENTRY(arena, alloc_writes_dont_crash)
OC_TEST_ENTRY(arena, grows_past_initial_cap)
OC_TEST_ENTRY(arena, dup_and_dup_n)
OC_TEST_ENTRY(arena, printf)
OC_TEST_ENTRY(arena, reset_reuses_buffer)
OC_TEST_ENTRY(arena, thousand_alloc_free_cycles)
OC_TEST_ENTRY(arena, null_arena_returns_null)
OC_TEST_SUITE_END(arena)
