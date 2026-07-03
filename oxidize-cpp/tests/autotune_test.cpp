#include "oxidize/autotune.hpp"

#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <string>
#include <unistd.h>

static void require(bool cond, const char* message) {
  if (!cond) {
    std::fprintf(stderr, "autotune_test: %s\n", message);
    std::exit(1);
  }
}

static void test_small_dense_dual_numa() {
  oxidize::HardwareInventory inv;
  inv.physical_cores = 48;
  inv.logical_cores = 96;
  inv.numa_nodes = 2;
  inv.total_ram_bytes = 376ULL << 30;

  oxidize::ModelFingerprint model;
  model.file_size_bytes = 8ULL << 30;

  auto plan = oxidize::plan_cpu(inv, model);
  require(plan.numa_mode == "single", "small dense plan should use single NUMA");
  require(plan.threads == 16, "small dense plan should use 16 threads");
  require(!plan.mmap_hugepages, "small dense plan should not request hugepages");
  require(plan.mmap_advice == "sequential_prefetch",
          "small dense plan should prefetch sequential mmap");
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
  require(plan.numa_mode == "interleave", "huge plan should interleave NUMA");
  require(plan.threads == 48, "huge plan should use 48 threads");
  require(plan.mmap_advice == "random", "huge plan should use random mmap advice");
}

static void test_exceeds_ram_threshold() {
  oxidize::HardwareInventory inv;
  inv.physical_cores = 16;
  inv.logical_cores = 32;
  inv.numa_nodes = 2;
  inv.total_ram_bytes = 120ULL << 30;

  oxidize::ModelFingerprint model;
  model.file_size_bytes = 100ULL << 30;

  auto plan = oxidize::plan_cpu(inv, model);
  require(plan.numa_mode == "interleave", "RAM-pressure plan should interleave NUMA");
  require(plan.mmap_advice == "random", "RAM-pressure plan should use random mmap advice");
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
  require(json.find("\"numa_mode\"") != std::string::npos,
          "plan JSON should include numa_mode");
  require(json.find("\"threads\"") != std::string::npos,
          "plan JSON should include threads");
  require(json.find("\"mmap_advice\"") != std::string::npos,
          "plan JSON should include mmap_advice");
}

static void write_temp_file(const std::string& path, size_t bytes) {
  std::ofstream f(path, std::ios::binary);
  std::string payload(bytes, 'x');
  f.write(payload.data(), static_cast<std::streamsize>(payload.size()));
}

static void test_split_fingerprint_sums_shards() {
  const std::string prefix =
      "/tmp/oxidize-autotune-split-" + std::to_string(getpid());
  const std::string shard1 = prefix + "-00001-of-00003.gguf";
  const std::string shard2 = prefix + "-00002-of-00003.gguf";
  const std::string shard3 = prefix + "-00003-of-00003.gguf";
  std::remove(shard1.c_str());
  std::remove(shard2.c_str());
  std::remove(shard3.c_str());

  write_temp_file(shard1, 5);
  write_temp_file(shard2, 7);
  write_temp_file(shard3, 11);

  auto fp = oxidize::fingerprint_model_file(shard1);
  require(fp.exists, "split first shard should exist");
  require(fp.file_size_bytes == 23,
          "split fingerprint should sum all sibling shards");

  std::remove(shard1.c_str());
  std::remove(shard2.c_str());
  std::remove(shard3.c_str());
}

static void test_missing_fingerprint_marks_absent() {
  const std::string path =
      "/tmp/oxidize-autotune-missing-" + std::to_string(getpid()) + ".gguf";
  std::remove(path.c_str());
  auto fp = oxidize::fingerprint_model_file(path);
  require(!fp.exists, "missing model should be marked absent");
  require(fp.file_size_bytes == 0, "missing model should have zero fingerprint size");
}

int main() {
  test_small_dense_dual_numa();
  test_huge_model_interleave();
  test_exceeds_ram_threshold();
  test_json_nonempty();
  test_split_fingerprint_sums_shards();
  test_missing_fingerprint_marks_absent();
  std::printf("autotune_test: ok\n");
  return 0;
}
