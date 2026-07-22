/* test_offload.c — CPU/GPU offload pipeline tests. */
#include <criterion/criterion.h>
#include "oxidize/offload.h"

Test(offload, cuda_available_check)
{
    /* Just verify the function doesn't crash. */
    bool avail = oc_offload_cuda_available();
    (void)avail;
}

Test(offload, suggest_gpu_layers)
{
    /* 10 GB model, 8 GB VRAM → should suggest some layers. */
    uint32_t n = oc_offload_suggest_gpu_layers(
        10ULL * 1024 * 1024 * 1024,
        8ULL * 1024 * 1024 * 1024);
    cr_assert(n > 0, "Should suggest >0 layers for 8GB VRAM on 10GB model");

    /* 0 VRAM → 0 layers. */
    cr_assert_eq(oc_offload_suggest_gpu_layers(1000000, 0), 0);
}

Test(offload, suggest_gpu_layers_zero_model)
{
    /* 0-size model → 0 layers. */
    cr_assert_eq(oc_offload_suggest_gpu_layers(0, 1000000000), 0);
}

Test(offload, init_null_safety)
{
    OcOffloadPipeline pipe;
    cr_assert_neq(oc_offload_init(&pipe, NULL, NULL), OC_OK);
    cr_assert_neq(oc_offload_init(NULL, NULL, NULL), OC_OK);
}

Test(offload, forward_uninitialized)
{
    OcOffloadPipeline pipe;
    memset(&pipe, 0, sizeof(pipe));
    float logits[10] = {0};
    cr_assert_neq(oc_offload_forward(&pipe, 0, logits), OC_OK);
}

Test(offload, free_null_safety)
{
    /* Should not crash. */
    oc_offload_free(NULL);
}
