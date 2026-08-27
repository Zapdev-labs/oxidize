/* test_offload_plan.c — Multi-GPU offload planning tests. */
#include <criterion/criterion.h>
#include "oxidize/offload.h"
#include <stdlib.h>
#include <string.h>

static OcPlanTensorInfo make_tensor(const char *name)
{
    OcPlanTensorInfo t;
    t.name = name;
    t.layer_index = SIZE_MAX;  /* let the planner infer */
    return t;
}

Test(offload_plan, single_gpu_caps_layers)
{
    OcPlanTensorInfo tensors[] = {
        make_tensor("blk.0.attn_q.weight"),
        make_tensor("blk.1.attn_q.weight"),
        make_tensor("blk.2.attn_q.weight"),
        make_tensor("tok_embeddings.weight"),
    };
    OcLayerOffloadPlan plan = oc_plan_layer_offload(tensors, 4, 99);
    cr_assert_eq(plan.total_layers, 3);
    cr_assert_eq(plan.n_gpu_layers, 3);
    cr_assert_eq(plan.gpu_tensor_count, 3);
    cr_assert_eq(plan.cpu_tensor_count, 1);
}

Test(offload_plan, zero_gpu_layers)
{
    OcPlanTensorInfo tensors[] = {
        make_tensor("blk.0.attn_q.weight"),
        make_tensor("blk.0.attn_k.weight"),
    };
    OcLayerOffloadPlan plan = oc_plan_layer_offload(tensors, 2, 0);
    cr_assert_eq(plan.n_gpu_layers, 0);
    cr_assert_eq(plan.total_layers, 1);
    cr_assert_eq(plan.gpu_tensor_count, 0);
    cr_assert_eq(plan.cpu_tensor_count, 2);
    cr_assert_eq(oc_layer_offload_has_gpu_tensors(&plan), false);
}

Test(offload_plan, hf_layer_names)
{
    OcPlanTensorInfo tensors[] = {
        make_tensor("model.layers.0.self_attn.q_proj.weight"),
        make_tensor("model.layers.1.self_attn.q_proj.weight"),
        make_tensor("lm_head.weight"),
    };
    OcLayerOffloadPlan plan = oc_plan_layer_offload(tensors, 3, 1);
    cr_assert_eq(plan.total_layers, 2);
    cr_assert_eq(plan.n_gpu_layers, 1);
    cr_assert_eq(plan.gpu_tensor_count, 1);
    cr_assert_eq(plan.cpu_tensor_count, 2);
    cr_assert(oc_layer_offload_has_gpu_tensors(&plan));
}

Test(offload_plan, empty)
{
    OcLayerOffloadPlan plan = oc_plan_layer_offload(NULL, 0, 5);
    cr_assert_eq(plan.total_layers, 0);
    cr_assert_eq(plan.n_gpu_layers, 0);
}

Test(multi_gpu, pipeline_basic)
{
    OcPlanTensorInfo tensors[] = {
        make_tensor("blk.0.attn_q.weight"),
        make_tensor("blk.0.attn_k.weight"),
        make_tensor("blk.1.attn_q.weight"),
        make_tensor("blk.1.attn_k.weight"),
        make_tensor("blk.2.attn_q.weight"),
        make_tensor("blk.2.attn_k.weight"),
        make_tensor("blk.3.attn_q.weight"),
        make_tensor("blk.3.attn_k.weight"),
        make_tensor("output.weight"),
    };
    OcMultiGpuConfig cfg = {
        .gpu_count = 2,
        .n_gpu_layers = 4,
        .strategy = OC_PARALLELISM_PIPELINE,
    };
    OcMultiGpuOffloadPlan plan;
    OcError e = oc_plan_multi_gpu_offload(tensors, 9, &cfg, &plan);
    cr_assert_eq(e, OC_OK);
    cr_assert_eq(plan.total_layers, 4);
    cr_assert_eq(plan.n_gpu_layers, 4);
    cr_assert_eq(plan.total_gpu_tensor_count, 8);
    cr_assert_eq(plan.cpu_tensor_count, 1);
    cr_assert_eq(plan.n_gpu_assignments, 2);
    cr_assert_eq(plan.n_pipeline_stages, 2);

    /* Each GPU should have 2 layers, 4 tensors. */
    cr_assert_eq(plan.gpu_assignments[0].layer_count, 2);
    cr_assert_eq(plan.gpu_assignments[1].layer_count, 2);

    /* Pipeline stages should have correct ranges. */
    cr_assert_eq(plan.pipeline_stages[0].start_layer, 0);
    cr_assert_eq(plan.pipeline_stages[0].end_layer, 1);
    cr_assert_eq(plan.pipeline_stages[1].start_layer, 2);
    cr_assert_eq(plan.pipeline_stages[1].end_layer, 3);

    oc_multi_gpu_plan_free(&plan);
}

Test(multi_gpu, tensor_parallel)
{
    OcPlanTensorInfo tensors[] = {
        make_tensor("blk.0.attn_q.weight"),
        make_tensor("blk.0.attn_k.weight"),
        make_tensor("blk.1.attn_q.weight"),
        make_tensor("blk.1.attn_k.weight"),
    };
    OcMultiGpuConfig cfg = {
        .gpu_count = 2,
        .n_gpu_layers = 2,
        .strategy = OC_PARALLELISM_TENSOR,
    };
    OcMultiGpuOffloadPlan plan;
    OcError e = oc_plan_multi_gpu_offload(tensors, 4, &cfg, &plan);
    cr_assert_eq(e, OC_OK);
    cr_assert_eq(plan.total_gpu_tensor_count, 4);
    cr_assert_eq(plan.n_gpu_assignments, 2);
    cr_assert_eq(plan.n_pipeline_stages, 0);  /* no pipeline stages for tensor parallel */

    /* Tensors distributed across 2 GPUs (hash-based, so just check total). */
    size_t total_tensors = plan.gpu_assignments[0].tensor_count +
                           plan.gpu_assignments[1].tensor_count;
    cr_assert_eq(total_tensors, 4);

    oc_multi_gpu_plan_free(&plan);
}

Test(multi_gpu, zero_gpu_count)
{
    OcPlanTensorInfo tensors[] = { make_tensor("blk.0.attn_q.weight") };
    OcMultiGpuConfig cfg = { .gpu_count = 0, .n_gpu_layers = 1, .strategy = OC_PARALLELISM_PIPELINE };
    OcMultiGpuOffloadPlan plan;
    cr_assert_neq(oc_plan_multi_gpu_offload(tensors, 1, &cfg, &plan), OC_OK);
}

Test(multi_gpu, uneven_split)
{
    OcPlanTensorInfo tensors[] = {
        make_tensor("blk.0.w"),
        make_tensor("blk.1.w"),
        make_tensor("blk.2.w"),
    };
    OcMultiGpuConfig cfg = {
        .gpu_count = 2,
        .n_gpu_layers = 3,
        .strategy = OC_PARALLELISM_PIPELINE,
    };
    OcMultiGpuOffloadPlan plan;
    OcError e = oc_plan_multi_gpu_offload(tensors, 3, &cfg, &plan);
    cr_assert_eq(e, OC_OK);
    /* 3 layers / 2 GPUs: GPU0=2, GPU1=1. */
    cr_assert_eq(plan.gpu_assignments[0].layer_count, 2);
    cr_assert_eq(plan.gpu_assignments[1].layer_count, 1);

    oc_multi_gpu_plan_free(&plan);
}

Test(multi_gpu, zero_gpu_layers)
{
    OcPlanTensorInfo tensors[] = { make_tensor("blk.0.w") };
    OcMultiGpuConfig cfg = {
        .gpu_count = 2,
        .n_gpu_layers = 0,
        .strategy = OC_PARALLELISM_PIPELINE,
    };
    OcMultiGpuOffloadPlan plan;
    OcError e = oc_plan_multi_gpu_offload(tensors, 1, &cfg, &plan);
    cr_assert_eq(e, OC_OK);
    cr_assert_eq(plan.n_gpu_layers, 0);
    cr_assert_eq(plan.total_gpu_tensor_count, 0);

    oc_multi_gpu_plan_free(&plan);
}
