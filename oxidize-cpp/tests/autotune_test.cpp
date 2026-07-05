#include "oxidize/autotune.hpp"

#include <cstdlib>
#include <cstdio>
#include <string>

static void require(bool ok, const char* message) {
  if (!ok) {
    std::fprintf(stderr, "autotune_test failed: %s\n", message);
    std::abort();
  }
}

static void test_small_dense_dual_numa() {
  oxidize::HardwareInventory inv;
  inv.physical_cores = 32;
  inv.logical_cores = 64;
  inv.numa_nodes = 2;
  inv.total_ram_bytes = 192ULL << 30;

  oxidize::ModelFingerprint model;
  model.file_size_bytes = 500ULL << 20;  // 500 MiB Qwen 0.5B

  auto plan = oxidize::plan_cpu(inv, model);
  require(plan.numa_mode == "single", "plan.numa_mode == \"single\"");
  require(plan.threads == 32, "plan.threads == 32");
  require(!plan.mmap_hugepages, "!plan.mmap_hugepages");
}

static void test_large_dense_dual_numa_interleave() {
  oxidize::HardwareInventory inv;
  inv.physical_cores = 48;
  inv.logical_cores = 96;
  inv.numa_nodes = 2;
  inv.total_ram_bytes = 376ULL << 30;

  oxidize::ModelFingerprint model;
  model.file_size_bytes = 18ULL << 30;  // ~18 GiB Qwen 32B Q4_K_M

  auto plan = oxidize::plan_cpu(inv, model);
  require(plan.numa_mode == "interleave",
          "large dense dual-socket model should use interleave");
  require(plan.threads == 96,
          "large dense dual-socket model should use all logical cores");
  require(plan.mmap_policy == "prefetch",
          "large dense model should still use prefetch mmap policy");
  require(plan.prefetch_layers == 0,
          "in-RAM model should not need async layer prefetch");
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
  require(plan.numa_mode == "interleave", "plan.numa_mode == \"interleave\"");
  require(plan.threads == 48, "plan.threads == 48");
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
  require(plan.numa_mode == "interleave", "plan.numa_mode == \"interleave\"");
  require(plan.mmap_policy == "demand",
          "huge model should use demand mmap policy");
  require(plan.prefetch_layers >= 1,
          "model exceeding RAM should enable layer prefetch");
}

static void test_way_over_ram_prefetch_depth() {
  oxidize::HardwareInventory inv;
  inv.physical_cores = 48;
  inv.logical_cores = 96;
  inv.numa_nodes = 2;
  inv.total_ram_bytes = 192ULL << 30;

  oxidize::ModelFingerprint model;
  model.file_size_bytes = 300ULL << 30;  // > 1.5x RAM

  auto plan = oxidize::plan_cpu(inv, model);
  require(plan.prefetch_layers >= 2,
          "model much larger than RAM should use deeper prefetch");
}

static void test_small_model_prefetch_mmap_policy() {
  oxidize::HardwareInventory inv;
  inv.logical_cores = 16;
  inv.numa_nodes = 1;
  inv.total_ram_bytes = 64ULL << 30;

  oxidize::ModelFingerprint model;
  model.file_size_bytes = 2ULL << 30;

  auto plan = oxidize::plan_cpu(inv, model);
  require(plan.mmap_policy == "prefetch",
          "small model should use prefetch mmap policy");
  require(plan.prefetch_layers == 0,
          "small model should not need layer prefetch");
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
  require(json.find("\"numa_mode\"") != std::string::npos, "json.find(\"\\\"numa_mode\\\"\") != std::string::npos");
  require(json.find("\"threads\"") != std::string::npos, "json.find(\"\\\"threads\\\"\") != std::string::npos");
  require(json.find("\"mmap_policy\"") != std::string::npos,
          "JSON plan should include mmap_policy");
  require(json.find("\"prefetch_layers\"") != std::string::npos,
          "JSON plan should include prefetch_layers");
}

int main() {
  test_small_dense_dual_numa();
  test_large_dense_dual_numa_interleave();
  test_huge_model_interleave();
  test_exceeds_ram_threshold();
  test_way_over_ram_prefetch_depth();
  test_small_model_prefetch_mmap_policy();
  test_json_nonempty();
  std::printf("autotune_test: ok\n");
  return 0;
}
