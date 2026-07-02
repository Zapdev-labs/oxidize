#include "oxidize/gguf.hpp"

#include <cassert>
#include <cstdio>
#include <string>

static void test_mmap_hugepages_load() {
  // The fixture is tiny; this mainly verifies the option is accepted and the
  // madvise(MADV_HUGEPAGE) hint does not fail the load.
  oxidize::GgufLoadOptions opts;
  opts.mmap_policy = oxidize::MmapPolicy::Prefetch;
  opts.mmap_hugepages = true;
  auto model = oxidize::GgufModel::load(
      "../tests/fixtures/valid-v3.gguf", opts);
  assert(model.size() > 0);
}

static void test_parse_mmap_policy_rejects_invalid_value() {
  auto parsed = oxidize::parse_mmap_policy("nonsense");
  assert(!parsed.has_value());
}

static void test_parse_mmap_policy_names() {
  auto demand = oxidize::parse_mmap_policy("demand");
  assert(demand.has_value());
  assert(*demand == oxidize::MmapPolicy::Demand);
  assert(std::string(oxidize::mmap_policy_name(*demand)) == "demand");

  auto prefetch = oxidize::parse_mmap_policy("prefetch");
  assert(prefetch.has_value());
  assert(*prefetch == oxidize::MmapPolicy::Prefetch);
  assert(std::string(oxidize::mmap_policy_name(*prefetch)) == "prefetch");
}

int main() {
  test_mmap_hugepages_load();
  test_parse_mmap_policy_rejects_invalid_value();
  test_parse_mmap_policy_names();
  std::printf("gguf_mmap_policy_test: ok\n");
  return 0;
}
