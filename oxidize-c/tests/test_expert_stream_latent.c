#define _POSIX_C_SOURCE 200809L
#include <criterion/criterion.h>
#include "oxidize/expert_stream.h"
#include "oxidize/llama.h"
#include "oxidize/util/mmap.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

Test(expert_stream_latent, faults_full_latent_down_projection)
{
    char path[] = "/tmp/oxidize-latent-stream-XXXXXX";
    int fd = mkstemp(path);
    cr_assert_geq(fd, 0);
    cr_assert_eq(ftruncate(fd, 1024), 0);
    close(fd);
    OcMmap *mapping = NULL;
    cr_assert_eq(oc_mmap_open_readonly(path, &mapping), OC_OK);
    cr_assert_eq(unlink(path), 0);
    const uint8_t *data = oc_mmap_bytes(mapping);
    OcGgufShard shard = {
        .mmap = mapping, .bytes = (uint8_t *)data, .len = 1024,
    };
    OcLlamaLayer layer = {0};
    layer.expert_latent_dim = 8;
    layer.ffn_gate_exps = (OcWeightView){
        .data = data, .qtype = OC_QUANT_F32,
        .rows = 2, .cols = 8, .row_bytes = 32,
    };
    layer.ffn_up_exps = layer.ffn_gate_exps;
    layer.ffn_up_exps.data = data + 256;
    layer.ffn_down_exps = (OcWeightView){
        .data = data + 512, .qtype = OC_QUANT_F32,
        .rows = 8, .cols = 2, .row_bytes = 8,
    };
    OcLlamaModel model = {0};
    model.cfg.n_layer = 1;
    model.cfg.n_embd = 16;
    model.cfg.num_experts = 4;
    model.cfg.expert_intermediate_size = 2;
    model.layers = &layer;
    model.gguf.shards = &shard;
    model.gguf.n_shards = 1;
    OcExpertStreamPool *pool = NULL;
    cr_assert_eq(oc_expert_stream_new(&model, NULL, &pool), OC_OK);
    uint32_t selected[] = {1, 3};
    cr_assert_eq(oc_expert_stream_touch(pool, 0, selected, 2, false), OC_OK);
    cr_assert_eq(oc_expert_stream_fault_bytes(pool), 384);
    cr_assert_eq(oc_expert_stream_resident_bytes(pool), 384);
    cr_assert_eq(oc_expert_stream_touch(pool, 0, selected, 2, false), OC_OK);
    cr_assert_eq(oc_expert_stream_resident_bytes(pool), 384);
    oc_expert_stream_free(pool);
    oc_mmap_close(mapping);
}

#ifdef __linux__
static bool mapping_is_random(const uint8_t *address)
{
    FILE *file = fopen("/proc/self/smaps", "r");
    cr_assert_not_null(file);
    char line[1024];
    bool matched = false, found = false, random = false;
    while (fgets(line, sizeof(line), file)) {
        unsigned long begin, end;
        if (sscanf(line, "%lx-%lx", &begin, &end) == 2)
            matched = (uintptr_t)address >= begin && (uintptr_t)address < end;
        else if (matched && strncmp(line, "VmFlags:", 8) == 0) {
            found = true;
            random = strstr(line, " rr ") != NULL;
            break;
        }
    }
    fclose(file);
    cr_assert(found);
    return random;
}
#endif

Test(expert_stream_latent, restores_readahead_for_every_latent_group_only)
{
#ifndef __linux__
    cr_skip_test("Linux mmap advice is checked through smaps");
#else
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    char path[] = "/tmp/oxidize-latent-advice-XXXXXX";
    int fd = mkstemp(path);
    cr_assert_geq(fd, 0);
    cr_assert_eq(ftruncate(fd, (off_t)(10 * page)), 0);
    close(fd);
    OcMmap *mapping = NULL;
    cr_assert_eq(oc_mmap_open_readonly(path, &mapping), OC_OK);
    cr_assert_eq(unlink(path), 0);
    const uint8_t *data = oc_mmap_bytes(mapping);
    OcGgufShard shard = {
        .mmap = mapping, .bytes = (uint8_t *)data, .len = 10 * page,
    };
    OcLlamaLayer layer = {0};
    layer.expert_latent_dim = 4;
    layer.ffn_latent_d = (OcWeightView){
        .data = data + 2 * page, .qtype = OC_QUANT_F32,
        .rows = page / 16, .cols = 4, .row_bytes = 16,
    };
    layer.ffn_latent_u = layer.ffn_latent_d;
    layer.ffn_latent_u.data = data + 6 * page;
    OcLlamaModel model = {0};
    model.cfg.n_layer = 1;
    model.cfg.n_embd = 16;
    model.cfg.num_experts = 4;
    model.cfg.expert_latent_group_count = 2;
    model.layers = &layer;
    model.gguf.shards = &shard;
    model.gguf.n_shards = 1;
    OcExpertStreamPool *pool = NULL;
    cr_assert_eq(oc_expert_stream_new(&model, NULL, &pool), OC_OK);
    cr_assert_eq(oc_expert_stream_prepare_mapping(pool), OC_OK);
    for (size_t i = 0; i < 10; i++) {
        const bool latent = (i >= 2 && i < 4) || (i >= 6 && i < 8);
        cr_assert_eq(mapping_is_random(data + i * page), !latent,
                     "page=%zu", i);
    }
    oc_expert_stream_free(pool);
    oc_mmap_close(mapping);
#endif
}
