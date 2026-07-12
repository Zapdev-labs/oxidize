// oxidize-cpp-merge: SafeTensors checkpoint merger (linear / SLERP).
//
// C++20 port of oxidize-merge (Rust); semantics mirror
// oxidize-merge/src/{merge,blend,index,recipe,writer}.rs:
//   - blendable dtypes F32 / F16 / BF16, everything else copies from A
//   - SLERP over the whole tensor as one vector, f64 accumulation,
//     linear fallback for tiny / antipodal angles
//   - missing-tensor policy error | a | b
//   - sharded output model-NNNNN-of-?????.safetensors renamed at finish,
//     plus model.safetensors.index.json
//
// Reuses oxidize::json (src/server/json.{hpp,cpp}) and quant.hpp's
// f16_le_to_f32. Verified byte-exact against the Rust and C ports.
//
// Usage mirrors the Rust CLI; run with --self-test for built-in checks.

// Self-test asserts must survive Release (-DNDEBUG) builds.
#undef NDEBUG

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "oxidize/quant.hpp"
#include "../server/json.hpp"

namespace fs = std::filesystem;
using oxidize::json::Value;

namespace {

[[noreturn]] void die(const std::string& msg) {
  std::cerr << "error: " << msg << "\n";
  std::exit(1);
}

// ------------------------------------------------------------- f16 encode --
// Round-to-nearest-even f32 -> f16, matching oxidize-merge/src/blend.rs.
uint16_t f32_to_f16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  uint16_t sign = static_cast<uint16_t>((bits >> 31) & 1);
  int exp = static_cast<int>((bits >> 23) & 0xff);
  uint32_t frac = bits & 0x7fffff;
  if (exp == 255)
    return static_cast<uint16_t>((sign << 15) | (0x1f << 10) |
                                 ((frac != 0 ? 1 : 0) << 9));
  int new_exp = exp - 127 + 15;
  uint32_t new_frac = frac >> 13;
  if (new_exp <= 0) {
    if (new_exp < -10) return static_cast<uint16_t>(sign << 15);
    new_frac |= 0x400;
    new_frac >>= 1 - new_exp;
    return static_cast<uint16_t>((sign << 15) | new_frac);
  }
  if (new_exp >= 0x1f) return static_cast<uint16_t>((sign << 15) | (0x1f << 10));
  if (((frac >> 12) & 1) == 1 && ((frac & 0xfff) != 0 || (new_frac & 1) == 1)) {
    new_frac += 1;
    if (new_frac == 0x400) {
      new_frac = 0;
      new_exp += 1;
      if (new_exp >= 0x1f) return static_cast<uint16_t>((sign << 15) | (0x1f << 10));
    }
  }
  return static_cast<uint16_t>((sign << 15) |
                               (static_cast<uint32_t>(new_exp) << 10) | new_frac);
}

// ------------------------------------------------------------ model index --

struct MappedFile {
  const uint8_t* data = nullptr;
  size_t len = 0;
  MappedFile() = default;
  explicit MappedFile(const fs::path& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) die("failed to open " + path.string());
    struct stat st{};
    if (fstat(fd, &st) != 0) die("failed to stat " + path.string());
    len = static_cast<size_t>(st.st_size);
    void* p = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) die("failed to mmap " + path.string());
    data = static_cast<const uint8_t*>(p);
  }
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  MappedFile(MappedFile&& o) noexcept : data(o.data), len(o.len) {
    o.data = nullptr;
    o.len = 0;
  }
  MappedFile& operator=(MappedFile&& o) noexcept {
    if (this != &o) {
      if (data) ::munmap(const_cast<uint8_t*>(data), len);
      data = o.data;
      len = o.len;
      o.data = nullptr;
      o.len = 0;
    }
    return *this;
  }
  ~MappedFile() {
    if (data) ::munmap(const_cast<uint8_t*>(data), len);
  }
};

struct TensorRef {
  std::string dtype;
  std::vector<uint64_t> shape;
  size_t shard = 0;  // index into Model::shards
  uint64_t begin = 0, end = 0;
};

struct Shard {
  fs::path path;
  MappedFile map;
  size_t data_off = 0;
};

struct Model {
  std::vector<Shard> shards;
  std::map<std::string, TensorRef> tensors;       // sorted, like BTreeMap
  std::map<std::string, std::string> metadata;
};

bool is_blendable(const std::string& dtype) {
  return dtype == "F32" || dtype == "F16" || dtype == "BF16";
}

// Merge per-shard metadata, erroring on conflicting values for the same key.
void merge_metadata_strict(std::map<std::string, std::string>& into,
                           const std::map<std::string, std::string>& from,
                           const std::string& ctx) {
  for (const auto& [k, v] : from) {
    auto it = into.find(k);
    if (it != into.end() && it->second != v)
      die(ctx + ": conflicting metadata for key \"" + k + "\": \"" + it->second +
          "\" vs \"" + v + "\"");
    into.emplace(k, v);
  }
}

size_t open_shard(Model& m, const fs::path& path) {
  Shard shard;
  shard.path = path;
  shard.map = MappedFile(path);
  if (shard.map.len < 8) die(path.string() + ": too small for a safetensors file");
  uint64_t header_len;
  std::memcpy(&header_len, shard.map.data, 8);  // little-endian host
  if (8 + header_len > shard.map.len)
    die(path.string() + ": header length exceeds file size");
  shard.data_off = static_cast<size_t>(8 + header_len);

  Value header;
  try {
    header = oxidize::json::parse(std::string(
        reinterpret_cast<const char*>(shard.map.data) + 8, header_len));
  } catch (const std::exception& e) {
    die("failed to parse SafeTensors " + path.string() + ": " + e.what());
  }
  if (!header.is_object()) die(path.string() + ": header is not a JSON object");

  size_t shard_idx = m.shards.size();
  std::map<std::string, std::string> shard_meta;
  for (const auto& [name, info] : *header.obj) {
    if (name == "__metadata__") {
      if (info.is_object())
        for (const auto& [mk, mv] : *info.obj)
          if (mv.is_string()) shard_meta[mk] = mv.str;
      continue;
    }
    if (!info.is_object()) die(path.string() + ": bad tensor entry " + name);
    TensorRef t;
    t.shard = shard_idx;
    t.dtype = info.get_str("dtype");
    if (t.dtype.empty()) die(path.string() + ": tensor " + name + " missing dtype");
    if (const Value* sh = info.find("shape"); sh && sh->is_array())
      for (const Value& d : *sh->arr) t.shape.push_back(static_cast<uint64_t>(d.num));
    const Value* off = info.find("data_offsets");
    if (!off || !off->is_array() || off->arr->size() != 2)
      die(path.string() + ": tensor " + name + " missing data_offsets");
    t.begin = static_cast<uint64_t>((*off->arr)[0].num);
    t.end = static_cast<uint64_t>((*off->arr)[1].num);
    if (t.end < t.begin || 8 + header_len + t.end > shard.map.len)
      die(path.string() + ": tensor " + name + " data_offsets out of range");
    m.tensors.emplace(name, std::move(t));
  }
  merge_metadata_strict(m.metadata, shard_meta, path.string());
  m.shards.push_back(std::move(shard));
  return shard_idx;
}

// Reject shard names that are not a plain file name (path traversal guard).
void validate_shard_name(const std::string& name) {
  if (name.empty() || name.find('/') != std::string::npos || name == "." ||
      name == "..")
    die("invalid shard name \"" + name +
        "\" in weight index (must be a plain file name)");
}

Model open_model(const fs::path& path) {
  Model m;
  if (fs::is_regular_file(path)) {
    open_shard(m, path);
    return m;
  }
  if (!fs::is_directory(path))
    die("model path " + path.string() + " is neither a file nor a directory");

  std::optional<fs::path> index_path;
  std::vector<fs::path> shard_files;
  for (const auto& e : fs::directory_iterator(path)) {
    std::string n = e.path().filename().string();
    if (n.ends_with(".safetensors.index.json")) {
      if (!index_path || n < index_path->filename().string()) index_path = e.path();
    } else if (n.ends_with(".safetensors")) {
      shard_files.push_back(e.path());
    }
  }

  if (index_path) {
    std::ifstream f(*index_path);
    if (!f) die("failed to read " + index_path->string());
    std::stringstream ss;
    ss << f.rdbuf();
    Value index;
    try {
      index = oxidize::json::parse(ss.str());
    } catch (const std::exception& e) {
      die("invalid safetensors index JSON: " + std::string(e.what()));
    }
    if (const Value* meta = index.find("metadata"); meta && meta->is_object())
      for (const auto& [k, v] : *meta->obj)
        if (v.is_string()) m.metadata[k] = v.str;
    const Value* wm = index.find("weight_map");
    if (!wm || !wm->is_object()) die("weight index missing weight_map");

    std::map<std::string, size_t> opened;
    for (const auto& [tensor_name, shard_val] : *wm->obj) {
      if (!shard_val.is_string())
        die("weight_map entry for " + tensor_name + " is not a string");
      const std::string& sn = shard_val.str;
      if (!opened.count(sn)) {
        validate_shard_name(sn);
        opened[sn] = open_shard(m, path / sn);
      }
      if (!m.tensors.count(tensor_name))
        die("tensor " + tensor_name + " missing from shard " + sn);
    }
    return m;
  }

  if (shard_files.empty())
    die("no .safetensors files found in " + path.string());
  std::sort(shard_files.begin(), shard_files.end());
  for (const auto& sp : shard_files) {
    size_t before = m.tensors.size();
    Model tmp;
    size_t idx = open_shard(tmp, sp);
    (void)idx;
    for (auto& [name, t] : tmp.tensors) {
      if (m.tensors.count(name))
        die("duplicate tensor " + name + " in directory " + path.string());
      t.shard = m.shards.size();
      m.tensors.emplace(name, std::move(t));
    }
    merge_metadata_strict(m.metadata, tmp.metadata, sp.string());
    m.shards.push_back(std::move(tmp.shards[0]));
    (void)before;
  }
  return m;
}

std::pair<const uint8_t*, size_t> tensor_bytes(const Model& m, const TensorRef& t) {
  const Shard& s = m.shards[t.shard];
  return {s.map.data + s.data_off + t.begin, static_cast<size_t>(t.end - t.begin)};
}

// ----------------------------------------------------------------- recipe --

struct Recipe {
  float attention_t = 0.3f, mlp_t = 0.5f, other_t = 0.4f;
  std::optional<float> default_t;
};

float t_for_tensor(const Recipe& r, const std::string& name) {
  if (r.default_t) return *r.default_t;
  std::string lower(name);
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  auto has = [&](const char* s) { return lower.find(s) != std::string::npos; };
  if (has("self_attn") || has(".attn.") || has("attention") || has("q_proj") ||
      has("k_proj") || has("v_proj") || has("o_proj") || has("qkv") ||
      has("query_proj") || has("key_proj") || has("value_proj"))
    return r.attention_t;
  if (has("mlp") || has("ffn") || has("feed_forward") || has("expert") ||
      has("gate_proj") || has("up_proj") || has("down_proj") || has("w1") ||
      has("w2") || has("w3"))
    return r.mlp_t;
  return r.other_t;
}

// ------------------------------------------------------------------ blend --

void linear_f32(const float* a, const float* b, size_t n, float t, float* out) {
  float one_minus_t = 1.0f - t;
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; i++) out[i] = std::fma(a[i], one_minus_t, b[i] * t);
}

void slerp_f32(const float* a, const float* b, size_t n, float t, float* out) {
  if (n == 0) return;
  double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : dot, norm_a, norm_b)
  for (size_t i = 0; i < n; i++) {
    double l = a[i], r = b[i];
    dot += l * r;
    norm_a += l * l;
    norm_b += r * r;
  }
  if (norm_a == 0.0 && norm_b == 0.0) {
    std::fill_n(out, n, 0.0f);
    return;
  }
  if (norm_a == 0.0) { std::copy_n(b, n, out); return; }
  if (norm_b == 0.0) { std::copy_n(a, n, out); return; }
  double cos_theta = std::clamp(dot / (std::sqrt(norm_a) * std::sqrt(norm_b)), -1.0, 1.0);
  double theta = std::acos(cos_theta);
  if (theta < 1e-8) { linear_f32(a, b, n, t, out); return; }
  double sin_theta = std::sin(theta);
  // Near-antipodal: slerp weights blow up; fall back to linear.
  if (sin_theta < 1e-8) { linear_f32(a, b, n, t, out); return; }
  double w0 = std::sin((1.0 - static_cast<double>(t)) * theta) / sin_theta;
  double w1 = std::sin(static_cast<double>(t) * theta) / sin_theta;
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; i++)
    out[i] = static_cast<float>(w0 * static_cast<double>(a[i]) +
                                w1 * static_cast<double>(b[i]));
}

enum class Method { Linear, Slerp };

std::vector<uint8_t> blend_bytes(const std::string& dtype, const uint8_t* a,
                                 const uint8_t* b, size_t len, float t,
                                 Method method, const std::string& name) {
  size_t elem = dtype == "F32" ? 4 : 2;
  if (len % elem != 0)
    die("tensor " + name + " byte length not a multiple of element size");
  size_t n = len / elem;
  std::vector<float> fa(n), fb(n), fo(n);
  if (dtype == "F32") {
    std::memcpy(fa.data(), a, len);
    std::memcpy(fb.data(), b, len);
  } else if (dtype == "F16") {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
      fa[i] = oxidize::f16_le_to_f32(a + 2 * i);
      fb[i] = oxidize::f16_le_to_f32(b + 2 * i);
    }
  } else {  // BF16
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
      uint16_t ua, ub;
      std::memcpy(&ua, a + 2 * i, 2);
      std::memcpy(&ub, b + 2 * i, 2);
      uint32_t xa = static_cast<uint32_t>(ua) << 16;
      uint32_t xb = static_cast<uint32_t>(ub) << 16;
      std::memcpy(&fa[i], &xa, 4);
      std::memcpy(&fb[i], &xb, 4);
    }
  }
  if (method == Method::Linear) linear_f32(fa.data(), fb.data(), n, t, fo.data());
  else slerp_f32(fa.data(), fb.data(), n, t, fo.data());
  std::vector<uint8_t> out(len);
  if (dtype == "F32") {
    std::memcpy(out.data(), fo.data(), len);
  } else if (dtype == "F16") {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
      uint16_t u = f32_to_f16(fo[i]);
      std::memcpy(out.data() + 2 * i, &u, 2);
    }
  } else {  // BF16: truncate like the Rust port
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
      uint32_t bits;
      std::memcpy(&bits, &fo[i], 4);
      uint16_t u = static_cast<uint16_t>(bits >> 16);
      std::memcpy(out.data() + 2 * i, &u, 2);
    }
  }
  return out;
}

// ----------------------------------------------------------------- writer --

struct OutTensor {
  std::string name;
  std::string dtype;
  std::vector<uint64_t> shape;
  std::vector<uint8_t> data;
};

void write_safetensors_file(const fs::path& path,
                            const std::vector<OutTensor>& tensors,
                            const std::map<std::string, std::string>& metadata) {
  std::string hdr = "{";
  bool first = true;
  if (!metadata.empty()) {
    hdr += "\"__metadata__\":{";
    bool mf = true;
    for (const auto& [k, v] : metadata) {
      if (!mf) hdr += ",";
      hdr += oxidize::json::escape(k) + ":" + oxidize::json::escape(v);
      mf = false;
    }
    hdr += "}";
    first = false;
  }
  uint64_t off = 0;
  for (const auto& t : tensors) {
    if (!first) hdr += ",";
    hdr += oxidize::json::escape(t.name) + ":{\"dtype\":\"" + t.dtype +
           "\",\"shape\":[";
    for (size_t i = 0; i < t.shape.size(); i++) {
      if (i) hdr += ",";
      hdr += std::to_string(t.shape[i]);
    }
    hdr += "],\"data_offsets\":[" + std::to_string(off) + "," +
           std::to_string(off + t.data.size()) + "]}";
    off += t.data.size();
    first = false;
  }
  hdr += "}";
  hdr.resize((hdr.size() + 7) & ~size_t{7}, ' ');  // pad to 8-byte multiple

  std::ofstream f(path, std::ios::binary);
  if (!f) die("failed to create " + path.string());
  uint64_t hlen = hdr.size();
  f.write(reinterpret_cast<const char*>(&hlen), 8);
  f.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));
  for (const auto& t : tensors)
    f.write(reinterpret_cast<const char*>(t.data.data()),
            static_cast<std::streamsize>(t.data.size()));
  if (!f) die("failed to write " + path.string());
}

class Writer {
 public:
  Writer(fs::path output, uint64_t max_shard_bytes,
         std::map<std::string, std::string> metadata)
      : output_(std::move(output)),
        max_shard_bytes_(max_shard_bytes),
        metadata_(std::move(metadata)) {
    single_ = output_.extension() == ".safetensors";
    if (single_) {
      if (output_.has_parent_path()) fs::create_directories(output_.parent_path());
    } else {
      if (max_shard_bytes_ == 0) die("max shard size must be greater than zero");
      fs::create_directories(output_);
    }
  }

  void push(OutTensor t) {
    if (!single_ && !pending_.empty() &&
        pending_bytes_ + t.data.size() > max_shard_bytes_)
      flush_shard();
    pending_bytes_ += t.data.size();
    pending_.push_back(std::move(t));
  }

  size_t finish() {
    if (single_) {
      if (pending_.empty()) die("no tensors were written");
      write_safetensors_file(output_, pending_, metadata_);
      return pending_.size();
    }
    if (!pending_.empty()) flush_shard();
    if (weight_map_.empty()) die("no tensors were written");

    size_t total = shard_index_;
    char buf[64];
    for (size_t i = 0; i < total; i++) {
      std::snprintf(buf, sizeof(buf), "model-%05zu-of-?????.safetensors", i);
      fs::path oldp = output_ / buf;
      std::snprintf(buf, sizeof(buf), "model-%05zu-of-%05zu.safetensors", i, total);
      fs::path newp = output_ / buf;
      if (fs::exists(oldp)) {
        fs::rename(oldp, newp);
      } else if (!fs::exists(newp)) {
        die("shard missing while finalizing index (expected " + oldp.string() +
            " or " + newp.string() + ")");
      }
    }

    std::ofstream f(output_ / "model.safetensors.index.json");
    if (!f) die("failed to create index json");
    f << "{\n  \"metadata\": {\n";
    for (auto it = metadata_.begin(); it != metadata_.end(); ++it) {
      f << "    " << oxidize::json::escape(it->first) << ": "
        << oxidize::json::escape(it->second)
        << (std::next(it) != metadata_.end() ? "," : "") << "\n";
    }
    f << "  },\n  \"weight_map\": {\n";
    for (auto it = weight_map_.begin(); it != weight_map_.end(); ++it) {
      std::string shard = it->second;
      if (auto pos = shard.find("of-?????"); pos != std::string::npos) {
        std::snprintf(buf, sizeof(buf), "of-%05zu", total);
        shard.replace(pos, 8, buf);
      }
      f << "    " << oxidize::json::escape(it->first) << ": "
        << oxidize::json::escape(shard)
        << (std::next(it) != weight_map_.end() ? "," : "") << "\n";
    }
    f << "  }\n}\n";
    if (!f) die("failed to write index json");
    return total_tensors_;
  }

 private:
  void flush_shard() {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "model-%05zu-of-?????.safetensors",
                  shard_index_);
    write_safetensors_file(output_ / buf, pending_, metadata_);
    for (const auto& t : pending_) {
      weight_map_[t.name] = buf;
      total_tensors_++;
    }
    shard_index_++;
    pending_.clear();
    pending_bytes_ = 0;
  }

  fs::path output_;
  uint64_t max_shard_bytes_;
  std::map<std::string, std::string> metadata_;
  bool single_ = false;
  std::vector<OutTensor> pending_;
  uint64_t pending_bytes_ = 0;
  size_t shard_index_ = 0;
  std::map<std::string, std::string> weight_map_;
  size_t total_tensors_ = 0;
};

// ------------------------------------------------------------------ merge --

enum class MissingPolicy { Error, A, B };

void resolve_single_side(MissingPolicy policy, bool missing_from_b,
                         const std::string& name) {
  if (policy == MissingPolicy::Error) {
    if (missing_from_b) die("tensor " + name + " exists only in model A");
    die("tensor " + name + " exists only in model B");
  }
  if (policy == MissingPolicy::A && !missing_from_b)
    die("tensor " + name + " missing from model A");
  if (policy == MissingPolicy::B && missing_from_b)
    die("tensor " + name + " missing from model B");
}

void validate_compatible(const std::string& name, const TensorRef& a,
                         const TensorRef& b) {
  if (a.dtype != b.dtype)
    die("dtype mismatch for " + name + ": " + a.dtype + " vs " + b.dtype);
  if (a.shape != b.shape) die("shape mismatch for " + name);
}

std::string fmt_float(float v) {
  std::ostringstream ss;
  ss << v;
  return ss.str();
}

void self_test();

void usage() {
  std::cerr <<
      "oxidize-cpp-merge: merge two SafeTensors checkpoints (linear or SLERP)\n"
      "usage: oxidize-cpp-merge --a <model> --b <model> --output <path>\n"
      "  --a / --b        .safetensors file or HuggingFace model directory\n"
      "  --output         .safetensors file or directory for sharded output\n"
      "  --method         linear | slerp            (default slerp)\n"
      "  --preset         kimi-k275\n"
      "  --t              global blend weight in [0,1] toward B\n"
      "  --attention-t    attention blend weight     (default 0.3)\n"
      "  --mlp-t          MLP/expert blend weight    (default 0.5)\n"
      "  --other-t        other-tensor blend weight  (default 0.4)\n"
      "  --missing        error | a | b              (default error)\n"
      "  --max-shard-gib  max shard size in GiB      (default 5)\n"
      "  --dry-run        validate without writing\n"
      "  --self-test      run built-in correctness checks\n";
  std::exit(1);
}

}  // namespace

int main(int argc, char** argv) {
  std::string a_path, b_path, out_path;
  Method method = Method::Slerp;
  MissingPolicy missing = MissingPolicy::Error;
  Recipe recipe;
  bool dry_run = false;
  uint64_t max_shard_gib = 5;
  std::optional<float> global_t;
  bool preset_kimi = false;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) usage();
      return argv[++i];
    };
    if (arg == "--self-test") { self_test(); return 0; }
    else if (arg == "--dry-run") dry_run = true;
    else if (arg == "--a") a_path = val();
    else if (arg == "--b") b_path = val();
    else if (arg == "--output") out_path = val();
    else if (arg == "--method") {
      std::string v = val();
      if (v == "linear") method = Method::Linear;
      else if (v == "slerp") method = Method::Slerp;
      else die("--method must be linear or slerp");
    } else if (arg == "--preset") {
      if (val() != "kimi-k275") die("unknown preset");
      preset_kimi = true;
    } else if (arg == "--t") global_t = std::stof(val());
    else if (arg == "--attention-t") recipe.attention_t = std::stof(val());
    else if (arg == "--mlp-t") recipe.mlp_t = std::stof(val());
    else if (arg == "--other-t") recipe.other_t = std::stof(val());
    else if (arg == "--missing") {
      std::string v = val();
      if (v == "error") missing = MissingPolicy::Error;
      else if (v == "a") missing = MissingPolicy::A;
      else if (v == "b") missing = MissingPolicy::B;
      else die("--missing must be error, a, or b");
    } else if (arg == "--max-shard-gib") max_shard_gib = std::stoull(val());
    else usage();
  }
  if (a_path.empty() || b_path.empty() || out_path.empty()) usage();
  if (global_t && (*global_t < 0.0f || *global_t > 1.0f))
    die("--t must be in [0, 1]");
  if (recipe.attention_t < 0.0f || recipe.attention_t > 1.0f)
    die("--attention-t must be in [0, 1]");
  if (recipe.mlp_t < 0.0f || recipe.mlp_t > 1.0f) die("--mlp-t must be in [0, 1]");
  if (recipe.other_t < 0.0f || recipe.other_t > 1.0f)
    die("--other-t must be in [0, 1]");

  if (global_t) {
    recipe.attention_t = recipe.mlp_t = recipe.other_t = *global_t;
    recipe.default_t = *global_t;
  } else if (preset_kimi) {
    recipe = Recipe{};  // kimi-k275 == the defaults (0.3 / 0.5 / 0.4)
  }

  Model ma = open_model(a_path);
  Model mb = open_model(b_path);

  size_t merged = 0, copied_a = 0, copied_b = 0;
  std::optional<Writer> writer;
  if (!dry_run) {
    std::map<std::string, std::string> meta = ma.metadata;
    for (const auto& [k, v] : mb.metadata) meta[k] = v;
    meta["oxidize-merge.method"] = method == Method::Linear ? "linear" : "slerp";
    meta["oxidize-merge.attention_t"] = fmt_float(recipe.attention_t);
    meta["oxidize-merge.mlp_t"] = fmt_float(recipe.mlp_t);
    meta["oxidize-merge.other_t"] = fmt_float(recipe.other_t);
    if (recipe.default_t)
      meta["oxidize-merge.default_t"] = fmt_float(*recipe.default_t);
    meta["oxidize-merge.model_a"] = a_path;
    meta["oxidize-merge.model_b"] = b_path;
    uint64_t max_bytes =
        max_shard_gib > UINT64_MAX / (1024ull * 1024 * 1024)
            ? UINT64_MAX
            : max_shard_gib * 1024ull * 1024 * 1024;
    writer.emplace(fs::path(out_path), max_bytes, std::move(meta));
  }

  // Union walk over two sorted maps.
  auto ita = ma.tensors.begin();
  auto itb = mb.tensors.begin();
  while (ita != ma.tensors.end() || itb != mb.tensors.end()) {
    int cmp = ita == ma.tensors.end()   ? 1
              : itb == mb.tensors.end() ? -1
                                        : ita->first.compare(itb->first) < 0 ? -1
                                          : ita->first == itb->first         ? 0
                                                                             : 1;
    if (cmp == 0) {
      const std::string& name = ita->first;
      validate_compatible(name, ita->second, itb->second);
      if (is_blendable(ita->second.dtype)) {
        merged++;
        if (!dry_run) {
          auto [ba, la] = tensor_bytes(ma, ita->second);
          auto [bb, lb] = tensor_bytes(mb, itb->second);
          if (la != lb) die("tensor " + name + " byte length mismatch");
          writer->push({name, ita->second.dtype, ita->second.shape,
                        blend_bytes(ita->second.dtype, ba, bb, la,
                                    t_for_tensor(recipe, name), method, name)});
        }
      } else {
        copied_a++;
        if (!dry_run) {
          auto [ba, la] = tensor_bytes(ma, ita->second);
          writer->push({name, ita->second.dtype, ita->second.shape,
                        std::vector<uint8_t>(ba, ba + la)});
        }
      }
      ++ita;
      ++itb;
    } else if (cmp < 0) {
      resolve_single_side(missing, true, ita->first);
      copied_a++;
      if (!dry_run) {
        auto [ba, la] = tensor_bytes(ma, ita->second);
        writer->push({ita->first, ita->second.dtype, ita->second.shape,
                      std::vector<uint8_t>(ba, ba + la)});
      }
      ++ita;
    } else {
      resolve_single_side(missing, false, itb->first);
      copied_b++;
      if (!dry_run) {
        auto [bb, lb] = tensor_bytes(mb, itb->second);
        writer->push({itb->first, itb->second.dtype, itb->second.shape,
                      std::vector<uint8_t>(bb, bb + lb)});
      }
      ++itb;
    }
  }

  if (dry_run) {
    std::cout << "Dry run: would blend " << merged << " tensors, copy "
              << copied_a << " from A, copy " << copied_b << " from B -> "
              << out_path << "\n";
  } else {
    writer->finish();
    std::cout << "Merged " << merged << " tensors (" << copied_a
              << " copied from A, " << copied_b << " copied from B) -> "
              << out_path << "\n";
  }
  return 0;
}

// -------------------------------------------------------------- self-test --

namespace {

void self_test() {
  fs::path dir = fs::temp_directory_path() / "oxidize-cpp-merge-selftest";
  fs::create_directories(dir);

  auto write_f32_model = [&](const fs::path& p, const std::string& name,
                             std::vector<float> vals) {
    OutTensor t{name, "F32", {vals.size()},
                std::vector<uint8_t>(vals.size() * 4)};
    std::memcpy(t.data.data(), vals.data(), t.data.size());
    write_safetensors_file(p, {t}, {});
  };

  // linear midpoint
  write_f32_model(dir / "a.safetensors", "weight", {0.0f, 2.0f});
  write_f32_model(dir / "b.safetensors", "weight", {2.0f, 4.0f});
  Model ma = open_model(dir / "a.safetensors");
  Model mb = open_model(dir / "b.safetensors");
  assert(ma.tensors.size() == 1 && mb.tensors.size() == 1);
  auto [ba, la] = tensor_bytes(ma, ma.tensors.at("weight"));
  auto [bb, lb] = tensor_bytes(mb, mb.tensors.at("weight"));
  assert(la == 8 && lb == 8);
  auto out = blend_bytes("F32", ba, bb, 8, 0.5f, Method::Linear, "weight");
  float fo[2];
  std::memcpy(fo, out.data(), 8);
  assert(std::fabs(fo[0] - 1.0f) < 1e-5f && std::fabs(fo[1] - 3.0f) < 1e-5f);

  // slerp endpoints + 45° midpoint
  float ua[2] = {1.0f, 0.0f}, ub[2] = {0.0f, 1.0f}, so[2];
  slerp_f32(ua, ub, 2, 0.0f, so);
  assert(std::fabs(so[0] - 1.0f) < 1e-5f && std::fabs(so[1]) < 1e-5f);
  slerp_f32(ua, ub, 2, 0.5f, so);
  float half = 0.70710678f;
  assert(std::fabs(so[0] - half) < 1e-4f && std::fabs(so[1] - half) < 1e-4f);

  // bf16 exact midpoints
  uint16_t b16a[2] = {0x3f80, 0x4040}, b16b[2] = {0x4040, 0x40a0};
  auto bo = blend_bytes("BF16", reinterpret_cast<uint8_t*>(b16a),
                        reinterpret_cast<uint8_t*>(b16b), 4, 0.5f,
                        Method::Linear, "t");
  uint16_t r0, r1;
  std::memcpy(&r0, bo.data(), 2);
  std::memcpy(&r1, bo.data() + 2, 2);
  assert(r0 == 0x4000 && r1 == 0x4080);

  // f16 roundtrip
  for (float v = -4.0f; v <= 4.0f; v += 0.25f) {
    uint16_t h = f32_to_f16(v);
    uint8_t le[2];
    std::memcpy(le, &h, 2);
    assert(oxidize::f16_le_to_f32(le) == v);
  }

  // sharded writer roundtrip: force 2 shards, reopen via index
  fs::path sharded = dir / "sharded";
  fs::remove_all(sharded);
  Writer w(sharded, 8, {{"format", "pt"}});
  OutTensor o1{"alpha", "F32", {2}, std::vector<uint8_t>(8)};
  OutTensor o2{"beta", "F32", {2}, std::vector<uint8_t>(8)};
  float t1[2] = {1.0f, 2.0f}, t2[2] = {3.0f, 4.0f};
  std::memcpy(o1.data.data(), t1, 8);
  std::memcpy(o2.data.data(), t2, 8);
  w.push(std::move(o1));
  w.push(std::move(o2));
  assert(w.finish() == 2);
  Model ms = open_model(sharded);
  assert(ms.tensors.size() == 2);
  auto [pb, pl] = tensor_bytes(ms, ms.tensors.at("beta"));
  float rv[2];
  std::memcpy(rv, pb, 8);
  assert(pl == 8 && rv[0] == 3.0f && rv[1] == 4.0f);
  assert(ms.metadata.at("format") == "pt");

  // recipe classification
  Recipe r;
  assert(std::fabs(t_for_tensor(r, "model.layers.0.self_attn.q_proj.weight") - 0.3f) < 1e-6f);
  assert(std::fabs(t_for_tensor(r, "model.layers.3.mlp.experts.0.gate_proj.weight") - 0.5f) < 1e-6f);
  assert(std::fabs(t_for_tensor(r, "model.embed_tokens.weight") - 0.4f) < 1e-6f);

  std::cout << "oxidize-cpp-merge self-test: all checks passed\n";
}

}  // namespace
