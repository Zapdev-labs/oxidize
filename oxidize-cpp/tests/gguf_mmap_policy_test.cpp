#include "oxidize/gguf.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

static void require(bool ok, const char* message) {
  if (!ok) {
    std::fprintf(stderr, "gguf_mmap_policy_test failed: %s\n", message);
    std::abort();
  }
}

static void test_parse_mmap_policy_rejects_invalid_value() {
  require(!oxidize::parse_mmap_policy("nonsense").has_value(),
          "invalid policy name must not parse");
  require(!oxidize::parse_mmap_policy("").has_value(),
          "empty policy name must not parse");
}

static void check_round_trip(const char* name, oxidize::MmapPolicy expected) {
  auto parsed = oxidize::parse_mmap_policy(name);
  require(parsed.has_value(), name);
  require(*parsed == expected, name);
  require(std::string(oxidize::mmap_policy_name(*parsed)) == name, name);
}

static void test_parse_mmap_policy_names() {
  check_round_trip("demand", oxidize::MmapPolicy::Demand);
  check_round_trip("prefetch", oxidize::MmapPolicy::Prefetch);
  check_round_trip("sequential", oxidize::MmapPolicy::Sequential);
  check_round_trip("random", oxidize::MmapPolicy::Random);
}

int main() {
  test_parse_mmap_policy_rejects_invalid_value();
  test_parse_mmap_policy_names();
  std::printf("gguf_mmap_policy_test: ok\n");
  return 0;
}
