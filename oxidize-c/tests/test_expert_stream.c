#include <criterion/criterion.h>

#include "oxidize/error.h"
#include "oxidize/expert_stream.h"
#include "oxidize/llama.h"

#include <stdlib.h>
#include <string.h>

Test(expert_stream, new_on_dense_model)
{
    OcLlamaModel model;
    memset(&model, 0, sizeof(model));
    model.cfg.n_layer = 2;
    model.cfg.num_experts = 0;
    OcLlamaLayer layers[2];
    memset(layers, 0, sizeof(layers));
    model.layers = layers;

    OcExpertStreamPool *pool = NULL;
    cr_assert_eq(oc_expert_stream_new(&model, NULL, &pool), OC_OK);
    cr_assert_not_null(pool);
    cr_assert_eq(oc_expert_stream_resident_bytes(pool), 0);
    oc_expert_stream_free(pool);
}

Test(expert_stream, touch_selected_experts)
{
    OcLlamaModel model;
    memset(&model, 0, sizeof(model));
    model.cfg.n_layer = 1;
    model.cfg.num_experts = 4;
    OcLlamaLayer layer;
    memset(&layer, 0, sizeof(layer));
    uint8_t blob[256];
    memset(blob, 1, sizeof(blob));
    layer.ffn_gate_exps = (OcWeightView){
        .data = blob, .qtype = OC_QUANT_F32, .rows = 8, .cols = 4,
        .row_bytes = 16,
    };
    layer.ffn_up_exps = layer.ffn_gate_exps;
    layer.ffn_down_exps = layer.ffn_gate_exps;
    model.layers = &layer;

    OcExpertStreamConfig cfg;
    oc_expert_stream_config_init(&cfg);
    cfg.cache_bytes = 64;
    OcExpertStreamPool *pool = NULL;
    cr_assert_eq(oc_expert_stream_new(&model, &cfg, &pool), OC_OK);
    uint32_t experts[] = {0, 2};
    cr_assert_eq(oc_expert_stream_touch(pool, 0, experts, 2, false), OC_OK);
    cr_assert(oc_expert_stream_resident_bytes(pool) > 0);
    oc_expert_stream_free(pool);
}

Test(expert_stream, rejects_null)
{
    OcExpertStreamPool *pool = NULL;
    cr_assert_eq(oc_expert_stream_new(NULL, NULL, &pool), OC_ERR_INVALID_ARG);
    cr_assert_eq(oc_expert_stream_touch(NULL, 0, NULL, 1, false),
                 OC_ERR_INVALID_ARG);
    oc_expert_stream_free(NULL);
}

Test(expert_stream, reclaim_drops_cold_experts)
{
    OcLlamaModel model;
    memset(&model, 0, sizeof(model));
    model.cfg.n_layer = 1;
    model.cfg.num_experts = 4;
    OcLlamaLayer layer;
    memset(&layer, 0, sizeof(layer));
    uint8_t blob[256];
    memset(blob, 1, sizeof(blob));
    layer.ffn_gate_exps = (OcWeightView){
        .data = blob, .qtype = OC_QUANT_F32, .rows = 8, .cols = 4,
        .row_bytes = 16,
    };
    layer.ffn_up_exps = layer.ffn_gate_exps;
    layer.ffn_down_exps = layer.ffn_gate_exps;
    model.layers = &layer;

    OcExpertStreamConfig cfg;
    oc_expert_stream_config_init(&cfg);
    cfg.cache_bytes = 64;
    OcExpertStreamPool *pool = NULL;
    cr_assert_eq(oc_expert_stream_new(&model, &cfg, &pool), OC_OK);
    uint32_t experts[] = {0, 1, 2, 3};
    cr_assert_eq(oc_expert_stream_touch(pool, 0, experts, 4, false), OC_OK);
    cr_assert(oc_expert_stream_reclaim_bytes(pool) > 0);
    oc_expert_stream_free(pool);
}

Test(expert_stream, retouch_does_not_double_count)
{
    OcLlamaModel model;
    memset(&model, 0, sizeof(model));
    model.cfg.n_layer = 1;
    model.cfg.num_experts = 4;
    OcLlamaLayer layer;
    memset(&layer, 0, sizeof(layer));
    uint8_t blob[256];
    memset(blob, 1, sizeof(blob));
    layer.ffn_gate_exps = (OcWeightView){
        .data = blob, .qtype = OC_QUANT_F32, .rows = 8, .cols = 4,
        .row_bytes = 16,
    };
    layer.ffn_up_exps = layer.ffn_gate_exps;
    layer.ffn_down_exps = layer.ffn_gate_exps;
    model.layers = &layer;

    OcExpertStreamConfig cfg;
    oc_expert_stream_config_init(&cfg);
    cfg.cache_bytes = 1u << 20;
    OcExpertStreamPool *pool = NULL;
    cr_assert_eq(oc_expert_stream_new(&model, &cfg, &pool), OC_OK);
    uint32_t experts[] = {0, 2};
    cr_assert_eq(oc_expert_stream_touch(pool, 0, experts, 2, false), OC_OK);
    uint64_t once = oc_expert_stream_resident_bytes(pool);
    cr_assert(once > 0);
    cr_assert_eq(oc_expert_stream_touch(pool, 0, experts, 2, false), OC_OK);
    cr_assert_eq(oc_expert_stream_resident_bytes(pool), once);
    oc_expert_stream_free(pool);
}
