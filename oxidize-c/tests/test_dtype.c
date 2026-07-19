/* test_dtype.c — OcDType tests. */
#include "oc_min_test.h"
#include "oxidize/dtype.h"

Test(dtype, sizes)
{
    cr_assert_eq(oc_dtype_size(OC_DTYPE_F32), 4, "F32 size");
    cr_assert_eq(oc_dtype_size(OC_DTYPE_F16), 2, "F16 size");
    cr_assert_eq(oc_dtype_size(OC_DTYPE_BF16), 2, "BF16 size");
    cr_assert_eq(oc_dtype_size(OC_DTYPE_I8),  1, "I8 size");
    cr_assert_eq(oc_dtype_size(OC_DTYPE_I16), 2, "I16 size");
    cr_assert_eq(oc_dtype_size(OC_DTYPE_I32), 4, "I32 size");
    cr_assert_eq(oc_dtype_size(OC_DTYPE_I64), 8, "I64 size");
    cr_assert_eq(oc_dtype_size(OC_DTYPE_F64), 8, "F64 size");
}

Test(dtype, names)
{
    cr_assert_str_eq(oc_dtype_name(OC_DTYPE_F32), "F32", "");
    cr_assert_str_eq(oc_dtype_name(OC_DTYPE_BF16), "BF16", "");
    cr_assert_str_eq(oc_dtype_name(OC_DTYPE_I8),  "I8",  "");
    cr_assert_str_eq(oc_dtype_name(OC_DTYPE_I64), "I64", "");
}

Test(dtype, round_trip)
{
    OcDType ds[] = { OC_DTYPE_F32, OC_DTYPE_F16, OC_DTYPE_BF16, OC_DTYPE_I8,
                     OC_DTYPE_I16, OC_DTYPE_I32, OC_DTYPE_I64, OC_DTYPE_F64 };
    for (size_t i = 0; i < sizeof(ds)/sizeof(ds[0]); i++) {
        const char *name = oc_dtype_name(ds[i]);
        OcDType back = oc_dtype_from_str(name);
        cr_assert_eq(back, ds[i], "round-trip failed for %s", name);
    }
}

Test(dtype, unknown)
{
    cr_assert_eq(oc_dtype_size(OC_DTYPE_UNKNOWN), 0, "unknown size");
    cr_assert_eq(oc_dtype_from_str("ZZZ"), OC_DTYPE_UNKNOWN, "unknown parse");
    cr_assert_eq(oc_dtype_from_str(NULL),  OC_DTYPE_UNKNOWN, "NULL parse");
}

OC_TEST_SUITE_DEF(dtype)
OC_TEST_ENTRY(dtype, sizes)
OC_TEST_ENTRY(dtype, names)
OC_TEST_ENTRY(dtype, round_trip)
OC_TEST_ENTRY(dtype, unknown)
OC_TEST_SUITE_END(dtype)
