#include "oxidize/autotune.hpp"

#include <cassert>
#include <cstdio>
#include <string>

static void test_small_dense_dual_numa() {
  oxidize::HardwareInventory inv;
  inv.physical_cores = 32;
  inv.logical_cores = 64;
  inv.numa_nodes = 2;
  inv.total_ram_bytes = 192ULL << 30;

  oxidize::ModelFingerprint model;
  model.file_size_bytes = 500ULL << 20;  // 500 MiB Qwen 0.5B

  auto plan = oxidize::plan_cpu(inv, model);
  assert(plan.numa_mode == "single");
  assert(plan.threads == 32);
  assert(!plan.mmap_hugepages);
}

static void test_huge_model_interleave() {
  oxidize::HardwareInventory inv;
  inv.physical_cores = 48;
  inv.logical_cores = 96;
  inv.numa_nodes = 2;
  inv.total_ram_bytes = 192ULL << 30;

  oxidize::ModelFingerprint model;
  model.file_size_bytes = 193ULL << 30;

  auto plan = oxidize::plan_cpu(inv, model);
  assert(plan.numa_mode == "interleave");
  assert(plan.threads == 48);
}

static void test_h100_31b_throughput_plan() {
  oxidize::HardwareInventory inv;
  inv.physical_cores = 32;
  inv.logical_cores = 64;
  inv.numa_nodes = 1;
  inv.total_ram_bytes = 256ULL << 30;
  inv.has_cuda = true;
  inv.gpu_name = "NVIDIA H100 80GB HBM3";
  inv.gpu_vram_bytes = 80ULL << 30;

  oxidize::ModelFingerprint model;
  model.file_size_bytes = 18ULL * 1000ULL * 1000ULL * 1000ULL;
  model.layer_count = 60;
  model.num_kv_heads = 8;
  model.head_dim = 128;

  auto plan = oxidize::plan_cpu(inv, model);
  assert(plan.use_cuda);
  assert(plan.pipeline == "paged");
  assert(plan.kv_cache_dtype == "q4");
  assert(plan.kv_quantization == "turboquant");
  assert(plan.weight_plan == "w4a16");
  assert(plan.attention_kernel == "flash_attention_3");
  assert(plan.cuda_graphs);
  assert(plan.persistent_decode_kernels);
  assert(plan.tensor_parallelism == 1);
  assert(plan.pipeline_parallelism == 1);
  assert(plan.chunked_prefill_tokens >= 512);
  assert(plan.max_decode_batch >= 16);
  assert(plan.expected_decode_tps >= 1000.0);
}

static void test_exceeds_ram_threshold() {
  oxidize::HardwareInventory inv;
  inv.physical_cores = 16;
  inv.logical_cores = 32;
  inv.numa_nodes = 2;
  inv.total_ram_bytes = 120ULL << 30;

  oxidize::ModelFingerprint model;
  model.file_size_bytes = 100ULL << 30;  // > 80% of 120 GiB

  auto plan = oxidize::plan_cpu(inv, model);
  assert(plan.numa_mode == "interleave");
}

static void test_json_nonempty() {
  oxidize::HardwareInventory inv;
  inv.logical_cores = 16;
  inv.numa_nodes = 1;
  inv.total_ram_bytes = 64ULL << 30;
  oxidize::ModelFingerprint model;
  model.file_size_bytes = 1ULL << 30;
  auto plan = oxidize::plan_cpu(inv, model);
  std::string json = oxidize::plan_to_json(plan);
  assert(json.find("\"numa_mode\"") != std::string::npos);
  assert(json.find("\"threads\"") != std::string::npos);
  assert(json.find("\"kv_cache_dtype\"") != std::string::npos);
  assert(json.find("\"attention_kernel\"") != std::string::npos);
}

int main() {
  test_small_dense_dual_numa();
  test_huge_model_interleave();
  test_h100_31b_throughput_plan();
  test_exceeds_ram_threshold();
  test_json_nonempty();
  std::printf("autotune_test: ok\n");
  return 0;
}
