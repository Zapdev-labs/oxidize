/* test_rdma.c — RDMA stub tests. */
#include <criterion/criterion.h>
#include "oxidize/rdma.h"
#include <string.h>

Test(rdma, init)
{
    OcRdmaDevice dev;
    cr_assert_eq(oc_rdma_init(&dev, "ib0"), OC_OK);
    cr_assert_str_eq(dev.device_name, "ib0");
    cr_assert(!dev.active);
    cr_assert_eq(dev.n_regions, 0);
    oc_rdma_free(&dev);
}

Test(rdma, init_null_name)
{
    OcRdmaDevice dev;
    cr_assert_eq(oc_rdma_init(&dev, NULL), OC_OK);
    cr_assert_str_eq(dev.device_name, "ib0");
    oc_rdma_free(&dev);
}

Test(rdma, init_null)
{
    cr_assert_neq(oc_rdma_init(NULL, "ib0"), OC_OK);
}

Test(rdma, register_memory)
{
    OcRdmaDevice dev;
    oc_rdma_init(&dev, "ib0");
    char buf[1024];
    OcRdmaRegion region;
    cr_assert_eq(oc_rdma_register_memory(&dev, buf, 1024, &region), OC_OK);
    cr_assert_eq(region.length, 1024);
    cr_assert(region.valid);
    cr_assert_eq(region.lkey, 1);
    cr_assert_eq(region.rkey, 1);
    cr_assert_eq(dev.n_regions, 1);
    oc_rdma_free(&dev);
}

Test(rdma, register_null)
{
    cr_assert_neq(oc_rdma_register_memory(NULL, NULL, 0, NULL), OC_OK);
}

Test(rdma, deregister)
{
    OcRdmaDevice dev;
    oc_rdma_init(&dev, "ib0");
    char buf[256];
    OcRdmaRegion region;
    oc_rdma_register_memory(&dev, buf, 256, &region);
    cr_assert_eq(oc_rdma_deregister_memory(&dev, &region), OC_OK);
    cr_assert(!region.valid);
    oc_rdma_free(&dev);
}

Test(rdma, send)
{
    OcRdmaDevice dev;
    oc_rdma_init(&dev, "ib0");
    char buf[1024];
    OcRdmaRegion region;
    oc_rdma_register_memory(&dev, buf, 1024, &region);
    cr_assert_eq(oc_rdma_send(&dev, &region, 0, 512), OC_OK);
    cr_assert_eq(dev.bytes_sent, 512);
    oc_rdma_free(&dev);
}

Test(rdma, send_oob)
{
    OcRdmaDevice dev;
    oc_rdma_init(&dev, "ib0");
    char buf[256];
    OcRdmaRegion region;
    oc_rdma_register_memory(&dev, buf, 256, &region);
    cr_assert_neq(oc_rdma_send(&dev, &region, 0, 512), OC_OK);
    oc_rdma_free(&dev);
}

Test(rdma, receive)
{
    OcRdmaDevice dev;
    oc_rdma_init(&dev, "ib0");
    char buf[1024];
    OcRdmaRegion region;
    oc_rdma_register_memory(&dev, buf, 1024, &region);
    cr_assert_eq(oc_rdma_receive(&dev, &region, 0, 256), OC_OK);
    cr_assert_eq(dev.bytes_received, 256);
    oc_rdma_free(&dev);
}

Test(rdma, receive_oob)
{
    OcRdmaDevice dev;
    oc_rdma_init(&dev, "ib0");
    char buf[64];
    OcRdmaRegion region;
    oc_rdma_register_memory(&dev, buf, 64, &region);
    cr_assert_neq(oc_rdma_receive(&dev, &region, 32, 64), OC_OK);
    oc_rdma_free(&dev);
}

Test(rdma, is_active)
{
    OcRdmaDevice dev;
    oc_rdma_init(&dev, "ib0");
    cr_assert(!oc_rdma_is_active(&dev));
    dev.active = true;
    cr_assert(oc_rdma_is_active(&dev));
    oc_rdma_free(&dev);
}

Test(rdma, n_regions)
{
    OcRdmaDevice dev;
    oc_rdma_init(&dev, "ib0");
    cr_assert_eq(oc_rdma_n_regions(&dev), 0);
    char buf[128];
    OcRdmaRegion r;
    oc_rdma_register_memory(&dev, buf, 128, &r);
    cr_assert_eq(oc_rdma_n_regions(&dev), 1);
    oc_rdma_free(&dev);
}

Test(rdma, bytes_sent)
{
    OcRdmaDevice dev;
    oc_rdma_init(&dev, "ib0");
    char buf[1024];
    OcRdmaRegion r;
    oc_rdma_register_memory(&dev, buf, 1024, &r);
    oc_rdma_send(&dev, &r, 0, 100);
    oc_rdma_send(&dev, &r, 100, 200);
    cr_assert_eq(oc_rdma_bytes_sent(&dev), 300);
    oc_rdma_free(&dev);
}

Test(rdma, bytes_received)
{
    OcRdmaDevice dev;
    oc_rdma_init(&dev, "ib0");
    char buf[1024];
    OcRdmaRegion r;
    oc_rdma_register_memory(&dev, buf, 1024, &r);
    oc_rdma_receive(&dev, &r, 0, 50);
    cr_assert_eq(oc_rdma_bytes_received(&dev), 50);
    oc_rdma_free(&dev);
}

Test(rdma, free_null)
{
    oc_rdma_free(NULL);
}
