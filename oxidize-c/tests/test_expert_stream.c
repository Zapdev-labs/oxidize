#include <criterion/criterion.h>

#include "oxidize/error.h"
#include "oxidize/expert_stream.h"
#include "oxidize/llama.h"
#include "oxidize/util/mmap.h"

#include <stdio.h>
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
    cfg.cache_bytes = 4096;
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
    cfg.cache_bytes = 80;
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

Test(expert_stream, unlimited_cache_leaves_residency_to_kernel)
{
    uint8_t blob[256] = {0};
    OcLlamaLayer layer = {0};
    layer.ffn_gate_exps = (OcWeightView){
        .data = blob, .qtype = OC_QUANT_F32, .rows = 8, .cols = 4,
        .row_bytes = 16,
    };
    layer.ffn_up_exps = layer.ffn_down_exps = layer.ffn_gate_exps;
    OcLlamaModel model = {0};
    model.cfg.n_layer = 1;
    model.cfg.num_experts = 4;
    model.layers = &layer;
    OcExpertStreamConfig cfg;
    oc_expert_stream_config_init(&cfg);
    cfg.cache_bytes = UINT64_MAX;
    OcExpertStreamPool *pool = NULL;
    cr_assert_eq(oc_expert_stream_new(&model, &cfg, &pool), OC_OK);
    uint32_t experts[] = {0, 1, 2, 3};
    for (size_t i = 0; i < 2; i++)
        cr_assert_eq(oc_expert_stream_touch(pool, 0, experts, 4, false), OC_OK);
    cr_assert_eq(oc_expert_stream_resident_bytes(pool), 0);
    cr_assert_eq(oc_expert_stream_reclaim_bytes(pool), 0);
    oc_expert_stream_free(pool);
}

Test(expert_stream, prepare_mapping_includes_mtp_layer)
{
    uint8_t blob[256] = {0};
    OcLlamaModel model = {0};
    model.cfg.n_layer = 1;
    model.cfg.num_experts = 4;
    model.layers = calloc(1, sizeof(*model.layers));
    cr_assert_not_null(model.layers);
    model.mtp.layer.ffn_gate_exps = (OcWeightView){
        .data = blob, .qtype = OC_QUANT_F32, .rows = 8, .cols = 4,
        .row_bytes = 16,
    };
    OcExpertStreamPool *pool = NULL;
    cr_assert_eq(oc_expert_stream_new(&model, NULL, &pool), OC_OK);
    cr_assert_eq(oc_expert_stream_prepare_mapping(pool), OC_OK);
    oc_expert_stream_free(pool);
    free(model.layers);
}

Test(expert_stream, fault_bytes_include_dense_and_unequal_expert_spans)
{
    const char *path = "/tmp/oxidize-c-expert-stream-fault.bin";
    uint8_t blob[1024] = {0};
    FILE *file = fopen(path, "wb");
    cr_assert_not_null(file);
    cr_assert_eq(fwrite(blob, 1, sizeof(blob), file), sizeof(blob));
    cr_assert_eq(fclose(file), 0);
    OcMmap *mapping = NULL;
    cr_assert_eq(oc_mmap_open_readonly(path, &mapping), OC_OK);
    cr_assert_eq(remove(path), 0);
    const uint8_t *data = oc_mmap_bytes(mapping);
    OcGgufShard shard = {
        .mmap = mapping, .bytes = (uint8_t *)data, .len = sizeof(blob),
    };
    OcLlamaLayer layer = {0};
    layer.ffn_gate_exps = (OcWeightView){
        .data = data, .qtype = OC_QUANT_F32, .rows = 8, .cols = 4,
        .row_bytes = 16,
    };
    layer.ffn_up_exps = layer.ffn_down_exps = layer.ffn_gate_exps;
    layer.ffn_up_exps.data = data + 128;
    layer.ffn_up_exps.rows = 12;
    layer.ffn_down_exps.data = data + 320;
    layer.ffn_down_exps.rows = 16;
    layer.attn_q = (OcWeightView){
        .data = data + 576, .qtype = OC_QUANT_F32, .rows = 4, .cols = 4,
        .row_bytes = 16,
    };
    OcLlamaModel model = {0};
    model.cfg.n_layer = 1;
    model.cfg.num_experts = 4;
    model.layers = &layer;
    model.gguf.shards = &shard;
    model.gguf.n_shards = 1;
    OcExpertStreamPool *pool = NULL;
    cr_assert_eq(oc_expert_stream_new(&model, NULL, &pool), OC_OK);
    uint32_t experts[] = {0, 2};
    cr_assert_eq(oc_expert_stream_touch(pool, 0, experts, 2, false), OC_OK);
    cr_assert_eq(oc_expert_stream_fault_bytes(pool), 2 * (32 + 48 + 64) + 64);
    oc_expert_stream_free(pool);
    oc_mmap_close(mapping);
}
