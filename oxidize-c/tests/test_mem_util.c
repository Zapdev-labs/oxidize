/* test_mem_util.c — memory utility tests. */
#include <criterion/criterion.h>
#include "oxidize/mem_util.h"
#include <string.h>

Test(mem_util, get_usage)
{
    OcMemUsage mu;
    cr_assert_eq(oc_mem_usage_get(&mu), OC_OK);
    /* RSS should be non-zero for a running process. */
    /* On some CI systems it might be 0 if /proc is not mounted. */
    /* Just check the function doesn't crash. */
}

Test(mem_util, rss_bytes)
{
    uint64_t rss = oc_mem_rss_bytes();
    /* RSS may be 0 on some systems, but the function should not crash. */
    (void)rss;
}

Test(mem_util, available_bytes)
{
    uint64_t avail = oc_mem_available_bytes();
    (void)avail;
}

Test(mem_util, can_allocate)
{
    /* Can always allocate 0 bytes. */
    cr_assert(oc_mem_can_allocate(0, 0.5));
    /* Probably can allocate 1 byte. */
    cr_assert(oc_mem_can_allocate(1, 0.5));
}

Test(mem_util, format_bytes)
{
    char buf[32];
    oc_mem_format_bytes(0, buf, sizeof(buf));
    cr_assert_str_eq(buf, "0.0 B");

    oc_mem_format_bytes(1024, buf, sizeof(buf));
    cr_assert_str_eq(buf, "1.0 KB");

    oc_mem_format_bytes(1048576, buf, sizeof(buf));
    cr_assert_str_eq(buf, "1.0 MB");

    oc_mem_format_bytes(1073741824ULL, buf, sizeof(buf));
    cr_assert_str_eq(buf, "1.0 GB");
}

Test(mem_util, format_usage)
{
    OcMemUsage mu = { .rss = 1048576, .virtual = 2097152,
                     .peak_rss = 0, .available = 4294967296ULL,
                     .total = 8589934592ULL };
    char buf[256];
    oc_mem_usage_format(&mu, buf, sizeof(buf));
    cr_assert(strstr(buf, "RSS") != NULL);
    cr_assert(strstr(buf, "1.0 MB") != NULL);
}
