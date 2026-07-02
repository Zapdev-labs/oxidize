#include "oxidize/gguf.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

void write_u8(std::vector<uint8_t>& out, uint8_t v) { out.push_back(v); }
void write_u32(std::vector<uint8_t>& out, uint32_t v) {
  out.push_back(static_cast<uint8_t>(v));
  out.push_back(static_cast<uint8_t>(v >> 8));
  out.push_back(static_cast<uint8_t>(v >> 16));
  out.push_back(static_cast<uint8_t>(v >> 24));
}
void write_u64(std::vector<uint8_t>& out, uint64_t v) {
  for (int i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}
void write_string(std::vector<uint8_t>& out, const std::string& s) {
  write_u64(out, s.size());
  out.insert(out.end(), s.begin(), s.end());
}
void write_f32(std::vector<uint8_t>& out, float v) {
  uint32_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  write_u32(out, bits);
}

std::string make_temp_gguf_path() {
  char path[] = "/tmp/prefetch_layer_map_test_XXXXXX";
  int fd = ::mkstemp(path);
  assert(fd >= 0);
  ::close(fd);
  return std::string(path);
}

void write_file(const std::string& path, const std::vector<uint8_t>& data) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(data.data()),
          static_cast<std::streamsize>(data.size()));
  assert(f.good());
}

std::vector<uint8_t> build_two_layer_gguf() {
  std::vector<uint8_t> out;
  // Magic + version.
  out.insert(out.end(), {'G', 'G', 'U', 'F'});
  write_u32(out, 3);  // version
  write_u64(out, 2);  // tensor_count
  write_u64(out, 1);  // metadata_count

  // Metadata: general.alignment = 32 (uint32).
  write_string(out, "general.alignment");
  write_u32(out, 4);  // Uint32
  write_u32(out, 32);

  // Tensor info size before padding.
  size_t info_start = out.size();

  // Tensor 0: blk.0.attn_q.weight, [32,32] F32, data at offset 0.
  write_string(out, "blk.0.attn_q.weight");
  write_u32(out, 2);
  write_u64(out, 32);
  write_u64(out, 32);
  write_u32(out, 0);  // F32
  write_u64(out, 0);  // relative offset

  // Tensor 1: blk.1.attn_q.weight, [32,32] F32, data after tensor0 data.
  write_string(out, "blk.1.attn_q.weight");
  write_u32(out, 2);
  write_u64(out, 32);
  write_u64(out, 32);
  write_u32(out, 0);  // F32
  write_u64(out, 32 * 32 * sizeof(float));  // relative offset

  // Align to 32 bytes.
  while (out.size() % 32 != 0) write_u8(out, 0);
  size_t data_start = out.size();

  // Write tensor data: two 32x32 F32 matrices of zeros.
  for (size_t i = 0; i < 2 * 32 * 32; ++i) write_f32(out, 0.0f);

  // Patch relative offsets to be absolute from data_start. Our helper wrote
  // offsets relative to data_start, but GGUF expects them relative to the data
  // section start, which is exactly data_start. Since data_start is where the
  // data section begins, the offsets are already correct.
  (void)data_start;
  return out;
}

}  // namespace

static void test_layer_ranges_group_by_index() {
  std::string path = make_temp_gguf_path();
  std::vector<uint8_t> data = build_two_layer_gguf();
  write_file(path, data);

  oxidize::GgufModel model = oxidize::GgufModel::load(path);
  const auto& ranges = model.layer_ranges();

  assert(ranges.size() == 2);
  assert(ranges.count(0) == 1);
  assert(ranges.count(1) == 1);

  const auto& layer0 = ranges.at(0);
  const auto& layer1 = ranges.at(1);

  assert(layer0.size() == 1);
  assert(layer1.size() == 1);

  assert(layer0[0].shard_index == 0);
  assert(layer0[0].length == 32 * 32 * sizeof(float));

  assert(layer1[0].shard_index == 0);
  assert(layer1[0].offset > layer0[0].offset);
  assert(layer1[0].length == 32 * 32 * sizeof(float));

  std::remove(path.c_str());
}

int main() {
  test_layer_ranges_group_by_index();
  std::printf("prefetch_layer_map_test: ok\n");
  return 0;
}
