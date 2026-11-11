/* test_clock.c — clock util tests. */
#include <criterion/criterion.h>
#include "oxidize/util/clock.h"

Test(clock, monotonic_readings_never_go_backwards)
{
    uint64_t ns0 = oc_clock_mono_ns();
    double s0 = oc_clock_mono_sec();
    uint64_t ns1 = oc_clock_mono_ns();
    double s1 = oc_clock_mono_sec();
    cr_assert_geq(ns1, ns0, "");
    cr_assert_geq(s1, s0, "");
}

Test(clock, units_agree)
{
    double sec = oc_clock_mono_sec();
    double ms = oc_clock_mono_ms();
    uint64_t ms_u64 = oc_clock_mono_ms_u64();
    uint64_t ns = oc_clock_mono_ns();
    cr_assert_float_eq(ms, sec * 1e3, 1e3, "");
    cr_assert_float_eq((double)ms_u64, sec * 1e3, 1e3, "");
    cr_assert_float_eq((double)ns * 1e-9, sec, 1.0, "");
}

Test(clock, wall_clock_is_after_2020)
{
    cr_assert_gt(oc_clock_wall_sec(), 1577836800.0, "");
}
