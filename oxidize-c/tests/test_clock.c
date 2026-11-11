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
    uint64_t ns0 = oc_clock_mono_ns();
    double sec = oc_clock_mono_sec();
    double ms = oc_clock_mono_ms();
    uint64_t ms_u64 = oc_clock_mono_ms_u64();
    uint64_t ns1 = oc_clock_mono_ns();

    double lo = (double)ns0, hi = (double)ns1;
    cr_assert(sec * 1e9 >= lo - 1e3 && sec * 1e9 <= hi + 1e3, "");
    cr_assert(ms * 1e6 >= lo - 1e3 && ms * 1e6 <= hi + 1e3, "");
    cr_assert(ms_u64 >= ns0 / 1000000u && ms_u64 <= ns1 / 1000000u, "");
}

Test(clock, wall_clock_is_never_negative)
{
    cr_assert_geq(oc_clock_wall_sec(), 0.0, "");
}
