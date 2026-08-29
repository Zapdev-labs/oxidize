/* mem_util.h — memory usage reporting utilities. */
#ifndef OXIDIZE_MEM_UTIL_H
#define OXIDIZE_MEM_UTIL_H

#include <stddef.h>
#include <stdint.h>

#include "oxidize/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Memory usage snapshot (in bytes). */
typedef struct OcMemUsage {
    uint64_t rss;            /* resident set size (physical memory)      */
    uint64_t virtual;        /* virtual memory size                      */
    uint64_t peak_rss;       /* peak RSS (high-water mark)               */
    uint64_t available;      /* system memory available                  */
    uint64_t total;          /* total system memory                      */
} OcMemUsage;

/* Get the current process's memory usage. Returns OC_OK or OC_ERR_IO. */
OcError oc_mem_usage_get(OcMemUsage *out);

/* Get the current process's RSS in bytes (convenience function). */
uint64_t oc_mem_rss_bytes(void);

/* Get available system memory in bytes. */
uint64_t oc_mem_available_bytes(void);

/* Check if `bytes` of memory can be allocated with headroom.
 * Returns true if available >= bytes * (1 + headroom_fraction). */
bool oc_mem_can_allocate(size_t bytes, double headroom_fraction);

/* Format memory usage as a human-readable string. */
void oc_mem_usage_format(const OcMemUsage *mu, char *buf, size_t buf_len);

/* Human-readable byte size (e.g. "1.5 GB"). */
void oc_mem_format_bytes(uint64_t bytes, char *buf, size_t buf_len);

#ifdef __cplusplus
}
#endif

#endif /* OXIDIZE_MEM_UTIL_H */
