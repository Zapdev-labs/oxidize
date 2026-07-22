/*
 * mem_util.c — memory usage reporting implementation.
 */
#define _POSIX_C_SOURCE 200809L
#include "oxidize/mem_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined(__linux__)
#include <sys/sysinfo.h>
#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#endif

OcError oc_mem_usage_get(OcMemUsage *out)
{
    if (!out) return OC_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));

#if defined(__linux__)
    /* Parse /proc/self/status for RSS, VmSize, VmPeak. */
    FILE *f = fopen("/proc/self/status", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "VmRSS:", 6) == 0) {
                unsigned long kb = 0;
                sscanf(line + 6, "%lu", &kb);
                out->rss = (uint64_t)kb * 1024;
            } else if (strncmp(line, "VmSize:", 7) == 0) {
                unsigned long kb = 0;
                sscanf(line + 7, "%lu", &kb);
                out->virtual = (uint64_t)kb * 1024;
            } else if (strncmp(line, "VmPeak:", 7) == 0) {
                unsigned long kb = 0;
                sscanf(line + 7, "%lu", &kb);
                out->peak_rss = (uint64_t)kb * 1024;
            }
        }
        fclose(f);
    }
    /* System memory via sysinfo. */
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        out->total = (uint64_t)si.totalram * si.mem_unit;
        out->available = (uint64_t)si.freeram * si.mem_unit;
    }
#elif defined(__APPLE__)
    /* macOS: use mach API. */
    struct task_basic_info t_info;
    mach_msg_type_number_t t_count = TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_BASIC_INFO,
                  (task_info_t)&t_info, &t_count) == KERN_SUCCESS) {
        out->rss = t_info.resident_size;
        out->virtual = t_info.virtual_size;
    }
    /* System memory via sysctl. */
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    uint64_t memsize = 0;
    size_t len = sizeof(memsize);
    if (sysctl(mib, 2, &memsize, &len, NULL, 0) == 0) {
        out->total = memsize;
    }
    /* vm_statistics for available. */
    vm_statistics_data_t vm_stat;
    mach_msg_type_number_t count = HOST_VM_INFO_COUNT;
    if (host_statistics(mach_host_self(), HOST_VM_INFO,
                        (host_info_t)&vm_stat, &count) == KERN_SUCCESS) {
        out->available = (uint64_t)vm_stat.free_count * (uint64_t)getpagesize();
    }
#else
    /* Fallback: not available. */
    out->rss = 0;
    out->available = 0;
#endif

    return OC_OK;
}

uint64_t oc_mem_rss_bytes(void)
{
    OcMemUsage mu;
    if (oc_mem_usage_get(&mu) != OC_OK) return 0;
    return mu.rss;
}

uint64_t oc_mem_available_bytes(void)
{
    OcMemUsage mu;
    if (oc_mem_usage_get(&mu) != OC_OK) return 0;
    return mu.available;
}

bool oc_mem_can_allocate(size_t bytes, double headroom_fraction)
{
    uint64_t avail = oc_mem_available_bytes();
    if (avail == 0) return true; /* can't determine, allow */
    double needed = (double)bytes * (1.0 + headroom_fraction);
    return (double)avail >= needed;
}

void oc_mem_usage_format(const OcMemUsage *mu, char *buf, size_t buf_len)
{
    if (!mu || !buf || buf_len == 0) return;
    char rss_str[32], virt_str[32], avail_str[32], total_str[32];
    oc_mem_format_bytes(mu->rss, rss_str, sizeof(rss_str));
    oc_mem_format_bytes(mu->virtual, virt_str, sizeof(virt_str));
    oc_mem_format_bytes(mu->available, avail_str, sizeof(avail_str));
    oc_mem_format_bytes(mu->total, total_str, sizeof(total_str));
    snprintf(buf, buf_len,
             "RSS: %s | Virtual: %s | Available: %s / %s",
             rss_str, virt_str, avail_str, total_str);
}

void oc_mem_format_bytes(uint64_t bytes, char *buf, size_t buf_len)
{
    if (!buf || buf_len == 0) return;
    const char *units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_idx = 0;
    double val = (double)bytes;
    while (val >= 1024.0 && unit_idx < 4) {
        val /= 1024.0;
        unit_idx++;
    }
    snprintf(buf, buf_len, "%.1f %s", val, units[unit_idx]);
}
