/*
 * clock.h — the process clocks, in one place.
 *
 * Monotonic readings are for measuring elapsed time. The wall reading is
 * UTC seconds since the epoch and can jump when the system clock is set.
 */
#ifndef OXIDIZE_UTIL_CLOCK_H
#define OXIDIZE_UTIL_CLOCK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint64_t oc_clock_mono_ns(void);
uint64_t oc_clock_mono_ms_u64(void);
double oc_clock_mono_ms(void);
double oc_clock_mono_sec(void);

/* Returns 0.0 if the wall clock is unavailable. */
double oc_clock_wall_sec(void);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_UTIL_CLOCK_H */
