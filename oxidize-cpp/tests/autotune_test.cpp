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
}

int main() {
  test_small_dense_dual_numa();
  test_huge_model_interleave();
  test_exceeds_ram_threshold();
  test_json_nonempty();
  std::printf("autotune_test: ok\n");
  return 0;
}
