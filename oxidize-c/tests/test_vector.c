/* test_vector.c — OcVector dynamic array tests. */
#include <criterion/criterion.h>
#include "oxidize/vector.h"

#include <string.h>

Test(vector, init_free)
{
    OcVector v;
    cr_assert_eq(oc_vector_init(&v, sizeof(int)), OC_OK, "");
    cr_assert_eq(v.elem_size, sizeof(int), "");
    cr_assert_eq(v.len, 0, "");
    cr_assert_eq(v.cap, 0, "");
    cr_assert_null(v.data, "");
    oc_vector_free(&v);
    cr_assert_null(v.data, "");
}

Test(vector, push_get)
{
    OcVector v;
    oc_vector_init(&v, sizeof(int));
    for (int i = 0; i < 100; i++) {
        cr_assert_eq(oc_vector_push(&v, &i), OC_OK, "push %d", i);
    }
    cr_assert_eq(oc_vector_len(&v), 100, "len");
    for (int i = 0; i < 100; i++) {
        int *p = (int *)oc_vector_get(&v, (size_t)i);
        cr_assert_not_null(p, "get %d", i);
        cr_assert_eq(*p, i, "value %d", i);
    }
    cr_assert_null(oc_vector_get(&v, 100), "out-of-range returns NULL");
    oc_vector_free(&v);
}

Test(vector, push_n)
{
    OcVector v;
    oc_vector_init(&v, sizeof(int));
    int arr[50];
    for (int i = 0; i < 50; i++) arr[i] = i * 2;
    cr_assert_eq(oc_vector_push_n(&v, arr, 50), OC_OK, "");
    cr_assert_eq(oc_vector_len(&v), 50, "");
    cr_assert_eq(*(int *)oc_vector_get(&v, 49), 98, "");
    oc_vector_free(&v);
}

Test(vector, push_n_overlapping_without_growth)
{
    OcVector v;
    cr_assert_eq(oc_vector_init(&v, sizeof(int)), OC_OK);
    cr_assert_eq(oc_vector_reserve(&v, 8), OC_OK);
    int values[] = { 1, 2, 3, 4 };
    cr_assert_eq(oc_vector_push_n(&v, values, 4), OC_OK);
    const int *overlap = oc_vector_get(&v, 1);
    cr_assert_eq(oc_vector_push_n(&v, overlap, 3), OC_OK);
    const int expected[] = { 1, 2, 3, 4, 2, 3, 4 };
    cr_assert_arr_eq(v.data, expected, sizeof(expected));
    oc_vector_free(&v);
}

Test(vector, pop)
{
    OcVector v;
    oc_vector_init(&v, sizeof(int));
    int x = 7;
    oc_vector_push(&v, &x);
    int out = 0;
    cr_assert(oc_vector_pop(&v, &out), "pop");
    cr_assert_eq(out, 7, "");
    cr_assert_eq(oc_vector_len(&v), 0, "");
    cr_assert(!oc_vector_pop(&v, NULL), "pop empty");
    oc_vector_free(&v);
}

Test(vector, clear)
{
    OcVector v;
    oc_vector_init(&v, sizeof(int));
    for (int i = 0; i < 10; i++) oc_vector_push(&v, &i);
    oc_vector_clear(&v);
    cr_assert_eq(oc_vector_len(&v), 0, "");
    /* Should still be able to push after clear (capacity retained). */
    int x = 99;
    oc_vector_push(&v, &x);
    cr_assert_eq(*(int *)oc_vector_get(&v, 0), 99, "");
    oc_vector_free(&v);
}

Test(vector, reserve)
{
    OcVector v;
    oc_vector_init(&v, sizeof(int));
    cr_assert_eq(oc_vector_reserve(&v, 1000), OC_OK, "");
    cr_assert(v.cap >= 1000, "cap should be >= 1000");
    oc_vector_free(&v);
}

Test(vector, invalid_args)
{
    cr_assert_eq(oc_vector_init(NULL, sizeof(int)), OC_ERR_INVALID_ARG, "");
    OcVector v;
    cr_assert_eq(oc_vector_init(&v, 0), OC_ERR_INVALID_ARG, "");
    cr_assert_eq(oc_vector_push(NULL, NULL), OC_ERR_INVALID_ARG, "");
}

Test(vector, thousand_push_pop_cycles)
{
    /* Stress: 1000 push/pop cycles — ASan-clean. */
    for (int cycle = 0; cycle < 1000; cycle++) {
        OcVector v;
        oc_vector_init(&v, sizeof(long long));
        for (int i = 0; i < 100; i++) {
            long long x = (long long)i * cycle;
            oc_vector_push(&v, &x);
        }
        oc_vector_free(&v);
    }
}
