/*
 * clock.c — monotonic and wall clock readings.
 */
#define _POSIX_C_SOURCE 200809L

#include "oxidize/util/clock.h"

#include <time.h>

static struct timespec mono_now(void)
{
    struct timespec ts = {0};
    (void)clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts;
}

uint64_t oc_clock_mono_ns(void)
{
    struct timespec ts = mono_now();
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

uint64_t oc_clock_mono_ms_u64(void)
{
    struct timespec ts = mono_now();
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u;
}

double oc_clock_mono_ms(void)
{
    struct timespec ts = mono_now();
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

double oc_clock_mono_sec(void)
{
    struct timespec ts = mono_now();
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

double oc_clock_wall_sec(void)
{
    struct timespec ts;
    if (timespec_get(&ts, TIME_UTC) != TIME_UTC) return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
