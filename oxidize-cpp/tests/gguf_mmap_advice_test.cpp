#include "oxidize/gguf.hpp"

#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <string>

static void require(bool cond, const char* message) {
  if (!cond) {
    std::fprintf(stderr, "gguf_mmap_advice_test: %s\n", message);
    std::exit(1);
  }
}

int main() {
  require(oxidize::gguf_mmap_advice_name(
              oxidize::gguf_mmap_advice_from_name("sequential_prefetch")) ==
              std::string("sequential_prefetch"),
          "sequential_prefetch should round-trip");
  require(oxidize::gguf_mmap_advice_name(
              oxidize::gguf_mmap_advice_from_name("random")) ==
              std::string("random"),
          "random should round-trip");
  require(oxidize::gguf_mmap_advice_from_numa_mode("interleave") ==
              oxidize::GgufMmapAdvice::Random,
          "interleave NUMA should imply random mmap advice");
  require(oxidize::gguf_mmap_advice_from_numa_mode("single") ==
              oxidize::GgufMmapAdvice::SequentialPrefetch,
          "single NUMA should imply sequential prefetch mmap advice");

  bool rejected = false;
  try {
    (void)oxidize::gguf_mmap_advice_from_name("garbage");
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "unknown mmap advice should be rejected");

  std::printf("gguf_mmap_advice_test: ok\n");
  return 0;
}
