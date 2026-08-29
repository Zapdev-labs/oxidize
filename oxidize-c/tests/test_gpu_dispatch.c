/* test_gpu_dispatch.c — GPU dispatch tests. */
#include "framework.h"
#include "oxidize/gpu_dispatch.h"
#include <string.h>

Test(gpu, config_init)
{
    OcGpuDispatchConfig cfg;
    cr_assert_eq(oc_gpu_dispatch_config_init(&cfg), OC_OK);
    cr_assert_eq(cfg.n_devices, 0);
    cr_assert_eq(cfg.max_queue_size, 256);
    cr_assert_eq(cfg.batch_timeout_ms, 10);
}

Test(gpu, config_init_null)
{
    cr_assert_neq(oc_gpu_dispatch_config_init(NULL), OC_OK);
}

Test(gpu, task_type_name)
{
    cr_assert_str_eq(oc_gpu_task_type_name(OC_GPU_TASK_MATMUL), "MATMUL");
    cr_assert_str_eq(oc_gpu_task_type_name(OC_GPU_TASK_ATTENTION), "ATTENTION");
    cr_assert_str_eq(oc_gpu_task_type_name(OC_GPU_TASK_SAMPLING), "SAMPLING");
    cr_assert_str_eq(oc_gpu_task_type_name(OC_GPU_TASK_OTHER), "OTHER");
}

Test(gpu, init_free)
{
    OcGpuDispatch *d = NULL;
    cr_assert_eq(oc_gpu_dispatch_init(NULL, &d), OC_OK);
    cr_assert_not_null(d);
    cr_assert_eq(d->queue_cap, 256);
    cr_assert_eq(oc_gpu_dispatch_n_pending(d), 0);
    oc_gpu_dispatch_free(d);
}

Test(gpu, init_null_out)
{
    cr_assert_neq(oc_gpu_dispatch_init(NULL, NULL), OC_OK);
}

Test(gpu, detect_no_cuda)
{
    OcGpuDispatch *d = NULL;
    oc_gpu_dispatch_init(NULL, &d);
    /* In dependency-free C11 port, no CUDA -> 0 devices. */
    cr_assert_eq(oc_gpu_dispatch_n_devices(d), 0);
    cr_assert_eq(oc_gpu_dispatch_detect(d), OC_OK);
    cr_assert_eq(oc_gpu_dispatch_n_devices(d), 0);
    oc_gpu_dispatch_free(d);
}

Test(gpu, submit_task)
{
    OcGpuDispatch *d = NULL;
    oc_gpu_dispatch_init(NULL, &d);
    OcGpuTask task = {
        .id = 1,
        .type = OC_GPU_TASK_MATMUL,
        .priority = 5,
        .data_ptr = NULL,
        .data_size = 0,
    };
    cr_assert_eq(oc_gpu_dispatch_submit(d, task), OC_OK);
    cr_assert_eq(oc_gpu_dispatch_n_pending(d), 1);
    oc_gpu_dispatch_free(d);
}

Test(gpu, submit_priority_order)
{
    OcGpuDispatch *d = NULL;
    oc_gpu_dispatch_init(NULL, &d);
    OcGpuTask low  = { .id = 1, .type = OC_GPU_TASK_OTHER,    .priority = 1 };
    OcGpuTask high = { .id = 2, .type = OC_GPU_TASK_MATMUL,   .priority = 10 };
    OcGpuTask mid  = { .id = 3, .type = OC_GPU_TASK_ATTENTION, .priority = 5 };
    oc_gpu_dispatch_submit(d, low);
    oc_gpu_dispatch_submit(d, high);
    oc_gpu_dispatch_submit(d, mid);
    cr_assert_eq(oc_gpu_dispatch_n_pending(d), 3);
    /* Highest priority first. */
    cr_assert_eq(d->task_queue[0].id, 2);
    cr_assert_eq(d->task_queue[1].id, 3);
    cr_assert_eq(d->task_queue[2].id, 1);
    oc_gpu_dispatch_free(d);
}

Test(gpu, submit_stable_same_priority)
{
    OcGpuDispatch *d = NULL;
    oc_gpu_dispatch_init(NULL, &d);
    OcGpuTask a = { .id = 1, .type = OC_GPU_TASK_OTHER, .priority = 5 };
    OcGpuTask b = { .id = 2, .type = OC_GPU_TASK_OTHER, .priority = 5 };
    OcGpuTask c = { .id = 3, .type = OC_GPU_TASK_OTHER, .priority = 5 };
    oc_gpu_dispatch_submit(d, a);
    oc_gpu_dispatch_submit(d, b);
    oc_gpu_dispatch_submit(d, c);
    /* Same priority: insertion order preserved. */
    cr_assert_eq(d->task_queue[0].id, 1);
    cr_assert_eq(d->task_queue[1].id, 2);
    cr_assert_eq(d->task_queue[2].id, 3);
    oc_gpu_dispatch_free(d);
}

Test(gpu, submit_to_full_queue)
{
    OcGpuDispatchConfig cfg;
    oc_gpu_dispatch_config_init(&cfg);
    cfg.max_queue_size = 2;
    OcGpuDispatch *d = NULL;
    oc_gpu_dispatch_init(&cfg, &d);
    OcGpuTask t = { .id = 0, .type = OC_GPU_TASK_OTHER, .priority = 1 };
    cr_assert_eq(oc_gpu_dispatch_submit(d, t), OC_OK);
    cr_assert_eq(oc_gpu_dispatch_submit(d, t), OC_OK);
    cr_assert_neq(oc_gpu_dispatch_submit(d, t), OC_OK);
    cr_assert_eq(oc_gpu_dispatch_n_pending(d), 2);
    oc_gpu_dispatch_free(d);
}

Test(gpu, submit_null)
{
    OcGpuTask t = { .id = 0, .type = OC_GPU_TASK_OTHER, .priority = 0 };
    cr_assert_neq(oc_gpu_dispatch_submit(NULL, t), OC_OK);
}

Test(gpu, n_pending)
{
    OcGpuDispatch *d = NULL;
    oc_gpu_dispatch_init(NULL, &d);
    cr_assert_eq(oc_gpu_dispatch_n_pending(d), 0);
    OcGpuTask t = { .id = 1, .type = OC_GPU_TASK_MATMUL, .priority = 0 };
    oc_gpu_dispatch_submit(d, t);
    cr_assert_eq(oc_gpu_dispatch_n_pending(d), 1);
    oc_gpu_dispatch_free(d);
}

Test(gpu, get_device_none)
{
    OcGpuDispatch *d = NULL;
    oc_gpu_dispatch_init(NULL, &d);
    OcGpuDevice dev;
    cr_assert_neq(oc_gpu_dispatch_get_device(d, 0, &dev), OC_OK);
    oc_gpu_dispatch_free(d);
}

Test(gpu, get_device_invalid_args)
{
    cr_assert_neq(oc_gpu_dispatch_get_device(NULL, 0, NULL), OC_OK);
}

Test(gpu, n_devices)
{
    OcGpuDispatch *d = NULL;
    oc_gpu_dispatch_init(NULL, &d);
    cr_assert_eq(oc_gpu_dispatch_n_devices(d), 0);
    oc_gpu_dispatch_free(d);
}

Test(gpu, free_null_safe)
{
    oc_gpu_dispatch_free(NULL);
}

Test(gpu, init_custom_queue_size)
{
    OcGpuDispatchConfig cfg;
    oc_gpu_dispatch_config_init(&cfg);
    cfg.max_queue_size = 8;
    OcGpuDispatch *d = NULL;
    cr_assert_eq(oc_gpu_dispatch_init(&cfg, &d), OC_OK);
    cr_assert_eq(d->queue_cap, 8);
    /* Fill exactly. */
    for (uint32_t i = 0; i < 8; i++) {
        OcGpuTask t = { .id = i, .type = OC_GPU_TASK_OTHER, .priority = 1 };
        cr_assert_eq(oc_gpu_dispatch_submit(d, t), OC_OK);
    }
    cr_assert_eq(oc_gpu_dispatch_n_pending(d), 8);
    oc_gpu_dispatch_free(d);
}

Test(gpu, detect_null)
{
    cr_assert_neq(oc_gpu_dispatch_detect(NULL), OC_OK);
}
