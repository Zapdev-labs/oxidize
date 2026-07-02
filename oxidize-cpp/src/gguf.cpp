// Ported from oxidize-core/src/format/gguf.rs (parser, ByteReader, metadata
// store, tensor info table, alignment + data-section resolution, architecture
// detection, tensor-name mapping) and oxidize-core/src/model/inference.rs
// (InferenceConfig::from_gguf + metadata_u32/f32_lookup, metadata_u32_array_max,
// first_tensor_dims, first_layer_tensor_dims, ModelArchitecture::from_gguf).
//
// Faithful semantics: little-endian readers, GGUF v2/v3 only, default
// alignment 32 (must be power of two), align_up of the cursor to the data
// section, per-tensor absolute_offset = data_section_start + relative_offset.

#include "oxidize/gguf.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace oxidize {

const char* mmap_policy_name(MmapPolicy policy) {
  switch (policy) {
    case MmapPolicy::Demand: return "demand";
    case MmapPolicy::Prefetch: return "prefetch";
    case MmapPolicy::Sequential: return "sequential";
    case MmapPolicy::Random: return "random";
  }
  return "prefetch";
}

std::optional<MmapPolicy> parse_mmap_policy(std::string_view value) {
  if (value == "demand") return MmapPolicy::Demand;
  if (value == "prefetch") return MmapPolicy::Prefetch;
  if (value == "sequential") return MmapPolicy::Sequential;
  if (value == "random") return MmapPolicy::Random;
  return std::nullopt;
}

namespace {

constexpr uint64_t kDefaultAlignment = 32;
const unsigned char kGgufMagic[4] = {'G', 'G', 'U', 'F'};

// ── ByteReader (mirror of gguf.rs::ByteReader) ────────────────────────────────
class ByteReader {
 public:
  ByteReader(const uint8_t* bytes, size_t len) : bytes_(bytes), len_(len) {}

  size_t position() const { return cursor_; }

  const uint8_t* read_exact(size_t len) {
    // checked_add for overflow, then bounds check (UnexpectedEof).
    if (cursor_ > len_ || len > len_ - cursor_) {
      throw std::runtime_error("gguf: unexpected end of file");
    }
    const uint8_t* out = bytes_ + cursor_;
    cursor_ += len;
    return out;
  }

  uint8_t read_u8() { return read_exact(1)[0]; }
  int8_t read_i8() { return static_cast<int8_t>(read_u8()); }

  uint16_t read_u16() {
    const uint8_t* p = read_exact(2);
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
  }
  int16_t read_i16() { return static_cast<int16_t>(read_u16()); }

  uint32_t read_u32() {
    const uint8_t* p = read_exact(4);
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
  }
  int32_t read_i32() { return static_cast<int32_t>(read_u32()); }

  uint64_t read_u64() {
    const uint8_t* p = read_exact(8);
    uint64_t v = 0;
    for (int b = 0; b < 8; ++b) v |= static_cast<uint64_t>(p[b]) << (8 * b);
    return v;
  }
  int64_t read_i64() { return static_cast<int64_t>(read_u64()); }

  float read_f32() {
    uint32_t bits = read_u32();
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
  }
  double read_f64() {
    uint64_t bits = read_u64();
    double out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
  }

  bool read_bool() { return read_u8() != 0; }

  std::string read_string() {
    uint64_t len = read_u64();
    if (len > static_cast<uint64_t>(SIZE_MAX)) {
      throw std::runtime_error("gguf: integer overflow while parsing");
    }
    const uint8_t* p = read_exact(static_cast<size_t>(len));
    return std::string(reinterpret_cast<const char*>(p), static_cast<size_t>(len));
  }

  GgufMetadataValue read_value_of_type(GgufMetadataType t) {
    using K = GgufMetadataValue::Kind;
    GgufMetadataValue v;
    switch (t) {
      case GgufMetadataType::Uint8:
        v.kind = K::Uint8;
        v.u = read_u8();
        return v;
      case GgufMetadataType::Int8:
        v.kind = K::Int8;
        v.i = read_i8();
        return v;
      case GgufMetadataType::Uint16:
        v.kind = K::Uint16;
        v.u = read_u16();
        return v;
      case GgufMetadataType::Int16:
        v.kind = K::Int16;
        v.i = read_i16();
        return v;
      case GgufMetadataType::Uint32:
        v.kind = K::Uint32;
        v.u = read_u32();
        return v;
      case GgufMetadataType::Int32:
        v.kind = K::Int32;
        v.i = read_i32();
        return v;
      case GgufMetadataType::Float32:
        v.kind = K::Float32;
        v.f = static_cast<double>(read_f32());
        return v;
      case GgufMetadataType::Bool:
        v.kind = K::Bool;
        v.u = read_bool() ? 1 : 0;
        return v;
      case GgufMetadataType::String:
        v.kind = K::String;
        v.str = read_string();
        return v;
      case GgufMetadataType::Array: {
        v.kind = K::Array;
        v.array_element_type = metadata_type_from_u32(read_u32());
        uint64_t n = read_u64();
        v.array.reserve(static_cast<size_t>(n));
        for (uint64_t k = 0; k < n; ++k) {
          v.array.push_back(read_value_of_type(v.array_element_type));
        }
        return v;
      }
      case GgufMetadataType::Uint64:
        v.kind = K::Uint64;
        v.u = read_u64();
        return v;
      case GgufMetadataType::Int64:
        v.kind = K::Int64;
        v.i = read_i64();
        return v;
      case GgufMetadataType::Float64:
        v.kind = K::Float64;
        v.f = read_f64();
        return v;
    }
    throw std::runtime_error("gguf: unknown metadata type");
  }

  static GgufMetadataType metadata_type_from_u32(uint32_t value) {
    if (value > 12) {
      throw std::runtime_error("gguf: unknown metadata type: " +
                               std::to_string(value));
    }
    return static_cast<GgufMetadataType>(value);
  }

 private:
  const uint8_t* bytes_;
  size_t len_;
  size_t cursor_ = 0;
};

// gguf.rs::align_up — round `value` up to a multiple of `alignment` (pow2).
uint64_t align_up(uint64_t value, uint64_t alignment) {
  uint64_t mask = alignment - 1;
  // checked_add(mask)
  if (value > UINT64_MAX - mask) {
    throw std::runtime_error("gguf: integer overflow while parsing");
  }
  return (value + mask) & ~mask;
}

bool is_power_of_two(uint64_t v) { return v != 0 && (v & (v - 1)) == 0; }

void advise_mmap(void* map, size_t size, const GgufLoadOptions& options) {
  switch (options.mmap_policy) {
    case MmapPolicy::Demand:
      ::madvise(map, size, MADV_RANDOM);
      break;
    case MmapPolicy::Prefetch:
      ::madvise(map, size, MADV_SEQUENTIAL);
      ::madvise(map, size, MADV_WILLNEED);
      break;
    case MmapPolicy::Sequential:
      ::madvise(map, size, MADV_SEQUENTIAL);
      break;
    case MmapPolicy::Random:
      ::madvise(map, size, MADV_RANDOM);
      break;
  }
}

// gguf.rs::alignment_from_metadata
uint64_t alignment_from_metadata(const GgufMetadataValue& v) {
  using K = GgufMetadataValue::Kind;
  switch (v.kind) {
    case K::Uint8:
    case K::Uint16:
    case K::Uint32:
    case K::Uint64:
      return v.u;
    case K::Int8:
    case K::Int16:
    case K::Int32:
    case K::Int64:
      if (v.i > 0) return static_cast<uint64_t>(v.i);
      break;
    default:
      break;
  }
  throw std::runtime_error("gguf: invalid alignment: 0");
}

// inference.rs::metadata_u32_lookup widening rules.
std::optional<uint32_t> metadata_as_u32(const GgufMetadataValue& v) {
  using K = GgufMetadataValue::Kind;
  switch (v.kind) {
    case K::Bool:  // bool flags (e.g. expert_weights_norm) widen to 0/1
    case K::Uint8:
    case K::Uint16:
    case K::Uint32:
      return static_cast<uint32_t>(v.u);
    case K::Uint64:
      if (v.u <= UINT32_MAX) return static_cast<uint32_t>(v.u);
      return std::nullopt;
    case K::Int8:
    case K::Int16:
      if (v.i >= 0) return static_cast<uint32_t>(v.i);
      return std::nullopt;
    case K::Int32:
    case K::Int64:
      if (v.i >= 0 && v.i <= static_cast<int64_t>(UINT32_MAX))
        return static_cast<uint32_t>(v.i);
      return std::nullopt;
    default:
      return std::nullopt;
  }
}

// inference.rs::metadata_f32_lookup widening rules.
std::optional<float> metadata_as_f32(const GgufMetadataValue& v) {
  using K = GgufMetadataValue::Kind;
  switch (v.kind) {
    case K::Float32:
    case K::Float64:
      return static_cast<float>(v.f);
    case K::Int8:
    case K::Int16:
    case K::Int32:
    case K::Int64:
      return static_cast<float>(v.i);
    case K::Uint8:
    case K::Uint16:
    case K::Uint32:
    case K::Uint64:
      return static_cast<float>(v.u);
    default:
      return std::nullopt;
  }
}

std::string to_ascii_lower(const std::string& s) {
  std::string out(s);
  for (char& c : out)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

// gguf.rs::detect_architecture_from_metadata_keys — first metadata key whose
// namespace (before the first '.') is a known architecture.
std::string detect_architecture_from_metadata_keys(
    const std::map<std::string, GgufMetadataValue>& metadata) {
  static const char* kKnown[] = {
      "llama",       "mistral",      "mixtral",     "qwen",     "qwen2",
      "qwen2moe",    "qwen35",       "deepseek",    "deepseek2", "deepseek_v2",
      "deepseek_v3", "deepseek_moe", "gemma",       "phi",      "falcon",
      "gpt2",        "gptj",         "gptneox",     "dflash",   "dflash-draft"};
  for (const auto& kv : metadata) {
    auto dot = kv.first.find('.');
    if (dot == std::string::npos) continue;
    std::string ns = kv.first.substr(0, dot);
    for (const char* k : kKnown) {
      if (ns == k) return ns;
    }
  }
  return std::string();
}

// ── tensor name mapping (gguf.rs::map_*) ─────────────────────────────────────

bool split_once(const std::string& s, char delim, std::string& left,
                std::string& right) {
  auto pos = s.find(delim);
  if (pos == std::string::npos) return false;
  left = s.substr(0, pos);
  right = s.substr(pos + 1);
  return true;
}

std::optional<std::string> map_hf_decoder_name(const std::string& name) {
  if (name == "model.embed_tokens.weight") return std::string("tok_embeddings.weight");
  if (name == "lm_head.weight") return std::string("output.weight");
  if (name == "model.norm.weight") return std::string("norm.weight");

  const std::string prefix = "model.layers.";
  if (name.rfind(prefix, 0) != 0) return std::nullopt;
  std::string rest = name.substr(prefix.size());
  std::string layer, suffix;
  if (!split_once(rest, '.', layer, suffix)) return std::nullopt;

  const std::string moe1 = "block_sparse_moe.experts.";
  if (suffix.rfind(moe1, 0) == 0) {
    std::string r = suffix.substr(moe1.size());
    std::string expert, expert_weight;
    if (!split_once(r, '.', expert, expert_weight)) return std::nullopt;
    const char* mapped;
    if (expert_weight == "w1.weight") mapped = "ffn_gate";
    else if (expert_weight == "w2.weight") mapped = "ffn_down";
    else if (expert_weight == "w3.weight") mapped = "ffn_up";
    else return std::nullopt;
    return "blk." + layer + "." + mapped + "." + expert + ".weight";
  }

  const std::string moe2 = "mlp.experts.";
  if (suffix.rfind(moe2, 0) == 0) {
    std::string r = suffix.substr(moe2.size());
    std::string expert, expert_weight;
    if (!split_once(r, '.', expert, expert_weight)) return std::nullopt;
    const char* mapped;
    if (expert_weight == "gate_proj.weight") mapped = "ffn_gate";
    else if (expert_weight == "up_proj.weight") mapped = "ffn_up";
    else if (expert_weight == "down_proj.weight") mapped = "ffn_down";
    else return std::nullopt;
    return "blk." + layer + "." + mapped + "." + expert + ".weight";
  }

  const char* mapped_suffix = nullptr;
  if (suffix == "input_layernorm.weight") mapped_suffix = "attn_norm.weight";
  else if (suffix == "post_attention_layernorm.weight") mapped_suffix = "ffn_norm.weight";
  else if (suffix == "self_attn.q_proj.weight") mapped_suffix = "attn_q.weight";
  else if (suffix == "self_attn.k_proj.weight") mapped_suffix = "attn_k.weight";
  else if (suffix == "self_attn.v_proj.weight") mapped_suffix = "attn_v.weight";
  else if (suffix == "self_attn.o_proj.weight") mapped_suffix = "attn_output.weight";
  else if (suffix == "self_attn.q_a_proj.weight") mapped_suffix = "attn_q_a.weight";
  else if (suffix == "self_attn.q_a_layernorm.weight") mapped_suffix = "attn_q_a_norm.weight";
  else if (suffix == "self_attn.q_b_proj.weight") mapped_suffix = "attn_q_b.weight";
  else if (suffix == "self_attn.kv_a_proj_with_mqa.weight") mapped_suffix = "attn_kv_a_mqa.weight";
  else if (suffix == "self_attn.kv_a_layernorm.weight") mapped_suffix = "attn_kv_a_norm.weight";
  else if (suffix == "mlp.up_proj.weight") mapped_suffix = "ffn_up.weight";
  else if (suffix == "mlp.gate_proj.weight") mapped_suffix = "ffn_gate.weight";
  else if (suffix == "mlp.down_proj.weight") mapped_suffix = "ffn_down.weight";
  else if (suffix == "mlp.gate.weight") mapped_suffix = "ffn_gate_inp.weight";
  else if (suffix == "mlp.shared_expert.gate_proj.weight") mapped_suffix = "ffn_gate_shexp.weight";
  else if (suffix == "mlp.shared_expert.up_proj.weight") mapped_suffix = "ffn_up_shexp.weight";
  else if (suffix == "mlp.shared_expert.down_proj.weight") mapped_suffix = "ffn_down_shexp.weight";
  else if (suffix == "mlp.shared_expert_gate.weight") mapped_suffix = "ffn_gate_inp_shexp.weight";
  else if (suffix == "block_sparse_moe.gate.weight") mapped_suffix = "ffn_gate_inp.weight";
  else return std::nullopt;

  return "blk." + layer + "." + mapped_suffix;
}

std::optional<std::string> map_falcon_name(const std::string& name) {
  if (name == "transformer.word_embeddings.weight") return std::string("tok_embeddings.weight");
  if (name == "lm_head.weight") return std::string("output.weight");
  if (name == "transformer.ln_f.weight") return std::string("norm.weight");
  return std::nullopt;
}

std::optional<std::string> map_gpt2_name(const std::string& name) {
  if (name == "transformer.wte.weight") return std::string("tok_embeddings.weight");
  if (name == "lm_head.weight") return std::string("output.weight");
  if (name == "transformer.ln_f.weight") return std::string("norm.weight");
  return std::nullopt;
}

std::optional<std::string> map_gptj_name(const std::string& name) {
  return map_gpt2_name(name);  // identical rules in gguf.rs::map_gptj_name
}

std::optional<std::string> map_gpt_neox_name(const std::string& name) {
  if (name == "gpt_neox.embed_in.weight") return std::string("tok_embeddings.weight");
  if (name == "embed_out.weight" || name == "lm_head.weight") return std::string("output.weight");
  if (name == "gpt_neox.final_layer_norm.weight") return std::string("norm.weight");
  return std::nullopt;
}

std::string map_tensor_name(const std::string& architecture, const std::string& name) {
  std::string arch = to_ascii_lower(architecture);
  std::optional<std::string> mapped;
  if (arch == "llama" || arch == "mistral" || arch == "mixtral" || arch == "qwen" ||
      arch == "qwen2" || arch == "qwen2moe" || arch == "qwen35" || arch == "deepseek" ||
      arch == "deepseek2" || arch == "deepseek_v2" || arch == "deepseek_v3" ||
      arch == "deepseek_moe" || arch == "gemma" || arch == "phi") {
    mapped = map_hf_decoder_name(name);
  } else if (arch == "falcon") {
    mapped = map_falcon_name(name);
  } else if (arch == "gpt2") {
    mapped = map_gpt2_name(name);
  } else if (arch == "gptj") {
    mapped = map_gptj_name(name);
  } else if (arch == "gptneox") {
    mapped = map_gpt_neox_name(name);
  }
  return mapped.value_or(name);
}

}  // namespace

// ── GgufFile::architecture ───────────────────────────────────────────────────
std::string GgufFile::architecture() const {
  auto it = metadata.find("general.architecture");
  if (it != metadata.end() && it->second.kind == GgufMetadataValue::Kind::String) {
    return it->second.str;
  }
  return detect_architecture_from_metadata_keys(metadata);
}

// ── architecture_from_name (config.hpp contract) ─────────────────────────────
// Mirrors ModelArchitecture::from_gguf string matching in inference.rs.
Architecture architecture_from_name(const std::string& name) {
  const std::string& a = name;
  if (a == "llama") return Architecture::Llama;
  if (a == "mistral") return Architecture::Mistral;
  if (a == "mixtral") return Architecture::Mixtral;
  if (a == "deepseek" || a == "deepseek2" || a == "deepseek_v2" ||
      a == "deepseek_v3" || a == "deepseek_moe")
    return Architecture::DeepSeek;
  if (a == "qwen" || a == "qwen2" || a == "qwen2moe" || a == "qwen3" ||
      a == "qwen3moe" || a == "qwen35" || a == "qwen3_5" || a == "qwen3_5_text" ||
      a == "qwen35_text" || a == "qwen3_5_moe" || a == "qwen3_5_moe_text" ||
      a == "qwen35moe")
    return Architecture::Qwen;
  if (a == "gemma" || a == "gemma2" || a == "gemma3" || a == "gemma4")
    return Architecture::Gemma;
  if (a == "phi" || a == "phi3") return Architecture::Phi;
  if (a == "falcon") return Architecture::Falcon;
  if (a == "gpt2") return Architecture::Gpt2;
  if (a == "gptj") return Architecture::GptJ;
  if (a == "gptneox") return Architecture::GptNeoX;
  if (a == "minimax" || a == "minimax-m2" || a == "minimax-text-01")
    return Architecture::MiniMax;
  if (a == "lfm2") return Architecture::Lfm2;
  if (a == "lfm2moe") return Architecture::Lfm2Moe;
  if (a == "glm-dsa" || a == "glm_dsa" || a == "glmdsa") return Architecture::GlmDsa;
  return Architecture::Llama;  // gguf.rs default
}

// ── GgufModel::parse ─────────────────────────────────────────────────────────
GgufFile GgufModel::parse(const uint8_t* bytes, size_t len) {
  ByteReader reader(bytes, len);

  const uint8_t* magic = reader.read_exact(4);
  if (std::memcmp(magic, kGgufMagic, 4) != 0) {
    throw std::runtime_error("gguf: invalid gguf magic");
  }

  uint32_t version = reader.read_u32();
  if (version != 2 && version != 3) {
    throw std::runtime_error("gguf: unsupported gguf version: " +
                             std::to_string(version));
  }

  uint64_t tensor_count = reader.read_u64();
  uint64_t metadata_count = reader.read_u64();

  GgufFile file;
  file.version = version;
  file.tensor_count = tensor_count;

  for (uint64_t m = 0; m < metadata_count; ++m) {
    std::string key = reader.read_string();
    GgufMetadataType vt = ByteReader::metadata_type_from_u32(reader.read_u32());
    GgufMetadataValue value = reader.read_value_of_type(vt);
    file.metadata.insert_or_assign(std::move(key), std::move(value));
  }

  file.tensor_infos.reserve(static_cast<size_t>(tensor_count));
  for (uint64_t t = 0; t < tensor_count; ++t) {
    GgufTensorInfo info;
    info.raw_name = reader.read_string();
    uint32_t n_dims = reader.read_u32();
    info.dimensions.reserve(n_dims);
    for (uint32_t d = 0; d < n_dims; ++d) {
      info.dimensions.push_back(reader.read_u64());
    }
    info.ggml_type = reader.read_u32();
    info.quant = from_ggml_type(info.ggml_type);
    info.relative_offset = reader.read_u64();
    info.absolute_offset = 0;
    file.tensor_infos.push_back(std::move(info));
  }

  // Alignment: from general.alignment metadata (must be power of two), else 32.
  uint64_t alignment = kDefaultAlignment;
  {
    auto it = file.metadata.find("general.alignment");
    if (it != file.metadata.end()) {
      alignment = alignment_from_metadata(it->second);
    }
  }
  if (!is_power_of_two(alignment)) {
    throw std::runtime_error("gguf: invalid alignment: " +
                             std::to_string(alignment));
  }
  file.alignment = alignment;

  uint64_t data_section_start =
      align_up(static_cast<uint64_t>(reader.position()), alignment);
  if (data_section_start > static_cast<uint64_t>(len)) {
    throw std::runtime_error("gguf: unexpected end of file");
  }
  file.data_section_start = data_section_start;

  // Resolve absolute offsets and the mapped tensor name.
  std::string arch = file.architecture();
  if (arch.empty()) arch = "llama";  // gguf.rs mapped_tensor_infos default
  for (auto& info : file.tensor_infos) {
    // checked_add
    if (info.relative_offset > UINT64_MAX - data_section_start) {
      throw std::runtime_error("gguf: integer overflow while parsing");
    }
    info.absolute_offset = data_section_start + info.relative_offset;
    if (info.absolute_offset > static_cast<uint64_t>(len)) {
      throw std::runtime_error("gguf: unexpected end of file");
    }
    info.name = map_tensor_name(arch, info.raw_name);
  }

  return file;
}

// ── GgufModel::load (POSIX mmap) ─────────────────────────────────────────────
GgufModel GgufModel::load(const std::string& path) {
  return load(path, GgufLoadOptions{});
}

GgufModel GgufModel::load(const std::string& path,
                          const GgufLoadOptions& options) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error("gguf: io error: cannot open " + path);
  }
  struct stat st;
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    throw std::runtime_error("gguf: io error: cannot stat " + path);
  }
  size_t size = static_cast<size_t>(st.st_size);
  if (size == 0) {
    ::close(fd);
    throw std::runtime_error("gguf: unexpected end of file");
  }
  void* map = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (map == MAP_FAILED) {
    throw std::runtime_error("gguf: io error: mmap failed for " + path);
  }
  advise_mmap(map, size, options);

  GgufModel model;
  model.map_ = map;
  model.base_ = static_cast<const uint8_t*>(map);
  model.size_ = size;
  try {
    model.parsed_ = parse(model.base_, model.size_);
  } catch (...) {
    ::munmap(map, size);
    model.map_ = nullptr;
    model.base_ = nullptr;
    model.size_ = 0;
    throw;
  }
  // shard 0 view (all tensor_infos default shard_index = 0).
  model.shards_.push_back(Shard{map, model.base_, size});

  // Split (sharded) GGUF: mmap + merge every sibling shard. split.count and
  // split.no are global keys (see llama.cpp llama-arch.cpp). A single-file GGUF
  // has split.count == 0 or 1.
  uint32_t split_count = 0;
  if (auto sc = model.get_u32("split.count")) split_count = *sc;
  if (split_count > 1) {
    return load_split(path, std::move(model), split_count, options);
  }
  return model;
}

// gguf.rs has no split support; this mirrors llama.cpp's llama_split_path
// ("%s-%05d-of-%05d.gguf") + gguf_meta merge for multi-file models.
GgufModel GgufModel::load_split(const std::string& first_path,
                               GgufModel first_model, uint32_t split_count,
                               const GgufLoadOptions& options) {
  // Derive the split prefix by stripping the "-NNNNN-of-NNNNN.gguf" suffix from
  // the supplied first-shard path (llama.cpp llama_split_prefix).
  std::string prefix = first_path;
  {
    char postfix[64];
    std::snprintf(postfix, sizeof(postfix), "-%05u-of-%05u.gguf",
                  1u, split_count);
    std::string str_post(postfix);
    if (first_path.size() > str_post.size() &&
        first_path.compare(first_path.size() - str_post.size(), str_post.size(),
                           str_post) == 0) {
      prefix = first_path.substr(0, first_path.size() - str_post.size());
    } else {
      throw std::runtime_error(
          "gguf: split model first shard path does not match "
          "'-00001-of-%05d.gguf' naming: " + first_path);
    }
  }

  // shard 0 is already mmap'd + parsed in first_model. Merge in shards 1..N-1.
  for (uint32_t i = 1; i < split_count; ++i) {
    char path_buf[1024];
    std::snprintf(path_buf, sizeof(path_buf), "%s-%05u-of-%05u.gguf",
                  prefix.c_str(), i + 1u, split_count);
    std::string shard_path(path_buf);

    int fd = ::open(shard_path.c_str(), O_RDONLY);
    if (fd < 0) {
      throw std::runtime_error("gguf: io error: cannot open shard " + shard_path);
    }
    struct stat st;
    if (::fstat(fd, &st) != 0) {
      ::close(fd);
      throw std::runtime_error("gguf: io error: cannot stat shard " + shard_path);
    }
    size_t ssize = static_cast<size_t>(st.st_size);
    if (ssize == 0) {
      ::close(fd);
      throw std::runtime_error("gguf: empty shard: " + shard_path);
    }
    void* smap = ::mmap(nullptr, ssize, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (smap == MAP_FAILED) {
      throw std::runtime_error("gguf: io error: mmap failed for shard " + shard_path);
    }
    advise_mmap(smap, ssize, options);

    const uint8_t* sbase = static_cast<const uint8_t*>(smap);
    GgufFile shard_hdr;
    try {
      shard_hdr = parse(sbase, ssize);
    } catch (...) {
      ::munmap(smap, ssize);
      throw;
    }

    size_t shard_idx = first_model.shards_.size();
    first_model.shards_.push_back(Shard{smap, sbase, ssize});

    // Append this shard's tensor infos, tagging their owning shard so tensor()
    // resolves the data pointer against the right mmap base.
    for (auto& info : shard_hdr.tensor_infos) {
      info.shard_index = shard_idx;
      first_model.parsed_.tensor_infos.push_back(std::move(info));
    }
  }
  return first_model;
}

GgufModel::GgufModel(GgufModel&& other) noexcept
    : map_(other.map_),
      base_(other.base_),
      size_(other.size_),
      shards_(std::move(other.shards_)),
      parsed_(std::move(other.parsed_)) {
  other.map_ = nullptr;
  other.base_ = nullptr;
  other.size_ = 0;
  other.shards_.clear();
}

GgufModel& GgufModel::operator=(GgufModel&& other) noexcept {
  if (this != &other) {
    for (auto& s : shards_) {
      if (s.map != nullptr) ::munmap(s.map, s.size);
    }
    map_ = other.map_;
    base_ = other.base_;
    size_ = other.size_;
    shards_ = std::move(other.shards_);
    parsed_ = std::move(other.parsed_);
    other.map_ = nullptr;
    other.base_ = nullptr;
    other.size_ = 0;
    other.shards_.clear();
  }
  return *this;
}

GgufModel::~GgufModel() {
  // shards_[0] aliases map_/base_/size_, so munmap via shards_ only (no separate
  // munmap of map_) to avoid a double-unmap.
  for (auto& s : shards_) {
    if (s.map != nullptr) ::munmap(s.map, s.size);
  }
}

// ── metadata accessors ───────────────────────────────────────────────────────
std::optional<uint32_t> GgufModel::get_u32(const std::string& key) const {
  auto it = parsed_.metadata.find(key);
  if (it == parsed_.metadata.end()) return std::nullopt;
  return metadata_as_u32(it->second);
}

std::optional<float> GgufModel::get_f32(const std::string& key) const {
  auto it = parsed_.metadata.find(key);
  if (it == parsed_.metadata.end()) return std::nullopt;
  return metadata_as_f32(it->second);
}

std::optional<std::string> GgufModel::get_string(const std::string& key) const {
  auto it = parsed_.metadata.find(key);
  if (it == parsed_.metadata.end()) return std::nullopt;
  if (it->second.kind != GgufMetadataValue::Kind::String) return std::nullopt;
  return it->second.str;
}

const GgufMetadataValue* GgufModel::get_array(const std::string& key) const {
  auto it = parsed_.metadata.find(key);
  if (it == parsed_.metadata.end()) return nullptr;
  if (it->second.kind != GgufMetadataValue::Kind::Array) return nullptr;
  return &it->second;
}

Architecture GgufModel::architecture() const {
  std::string arch = parsed_.architecture();
  if (arch.empty()) arch = "llama";
  return architecture_from_name(arch);
}

bool GgufModel::has_tensor(const std::string& name) const {
  for (const auto& t : parsed_.tensor_infos) {
    if (t.name == name) return true;
  }
  return false;
}

TensorView GgufModel::tensor(const std::string& name) const {
  for (const auto& t : parsed_.tensor_infos) {
    if (t.name == name) {
      const uint8_t* shard_base = base_;
      if (t.shard_index < shards_.size()) shard_base = shards_[t.shard_index].base;
      TensorView v;
      v.data = shard_base + t.absolute_offset;
      v.dims = t.dimensions;
      v.quant = t.quant;
      v.byte_offset = t.absolute_offset;
      return v;
    }
  }
  throw std::runtime_error("gguf: tensor not found: " + name);
}

// ── build_inference_config (inference.rs::InferenceConfig::from_gguf) ─────────
namespace {

// gguf_metadata_prefix: canonicalize qwen3.5 variants to "qwen35".
std::string gguf_metadata_prefix(const std::string& arch) {
  if (arch == "qwen3_5_moe_text" || arch == "qwen3_5_moe" || arch == "qwen35moe" ||
      arch == "qwen3_5" || arch == "qwen3_5_text" || arch == "qwen35_text") {
    return "qwen35";
  }
  return arch;
}

bool is_qwen35_family(const std::string& arch) {
  return arch == "qwen3_5_moe_text" || arch == "qwen3_5_moe" || arch == "qwen35moe" ||
         arch == "qwen3_5" || arch == "qwen35";
}

// first_tensor_dims: exact mapped-name match.
std::optional<std::vector<uint64_t>> first_tensor_dims(const GgufFile& f,
                                                       const std::string& name) {
  for (const auto& t : f.tensor_infos) {
    if (t.name == name) return t.dimensions;
  }
  return std::nullopt;
}

// first_layer_tensor_dims: name starts with "blk.0." and ends with suffix.
std::optional<std::vector<uint64_t>> first_layer_tensor_dims(
    const GgufFile& f, const std::string& suffix) {
  for (const auto& t : f.tensor_infos) {
    const std::string& n = t.name;
    if (n.rfind("blk.0.", 0) == 0 && n.size() >= suffix.size() &&
        n.compare(n.size() - suffix.size(), suffix.size(), suffix) == 0) {
      return t.dimensions;
    }
  }
  return std::nullopt;
}

bool ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// metadata_u32_array_max: largest integer in an array-typed field.
std::optional<uint32_t> metadata_u32_array_max(const GgufFile& f,
                                               const std::string& key) {
  auto it = f.metadata.find(key);
  if (it == f.metadata.end() || it->second.kind != GgufMetadataValue::Kind::Array)
    return std::nullopt;
  std::optional<uint32_t> best;
  for (const auto& v : it->second.array) {
    auto u = metadata_as_u32(v);
    if (u) {
      if (!best || *u > *best) best = *u;
    }
  }
  return best;
}

}  // namespace

InferenceConfig build_inference_config(const GgufModel& model) {
  const GgufFile& meta = model.parsed();
  std::string raw_arch = meta.architecture();
  if (raw_arch.empty()) raw_arch = "llama";
  Architecture architecture = architecture_from_name(raw_arch);

  std::string metadata_prefix = gguf_metadata_prefix(raw_arch);
  const std::string arch = metadata_prefix;  // canonical arch string

  // arch_u32 / arch_f32: try the metadata_prefix key first, then (only if the
  // prefix was rewritten) the canonical arch key.
  auto arch_u32 = [&](const std::string& suffix) -> std::optional<uint32_t> {
    auto v = model.get_u32(metadata_prefix + "." + suffix);
    if (v) return v;
    if (metadata_prefix == arch) return std::nullopt;
    return model.get_u32(arch + "." + suffix);
  };
  auto arch_f32 = [&](const std::string& suffix) -> std::optional<float> {
    auto v = model.get_f32(metadata_prefix + "." + suffix);
    if (v) return v;
    if (metadata_prefix == arch) return std::nullopt;
    return model.get_f32(arch + "." + suffix);
  };

  bool uses_mla = (architecture == Architecture::DeepSeek ||
                   architecture == Architecture::GlmDsa);
  bool is_glm_dsa = (architecture == Architecture::GlmDsa);

  // token embedding dims, used for fallbacks.
  std::optional<std::vector<uint64_t>> token_embd_dims =
      first_tensor_dims(meta, "tok_embeddings.weight");
  if (!token_embd_dims)
    token_embd_dims = first_tensor_dims(meta, "token_embd.weight");

  // hidden_size.
  size_t hidden_size;
  {
    std::optional<uint32_t> v = arch_u32("embedding_length");
    if (!v && token_embd_dims) {
      const auto& d = *token_embd_dims;
      if (d.size() == 1) v = static_cast<uint32_t>(d[0]);
      else if (d.size() >= 2) v = static_cast<uint32_t>(d[1]);
    }
    hidden_size = v ? static_cast<size_t>(*v) : 4096;
  }

  // vocab_size.
  size_t vocab_size;
  {
    std::optional<uint32_t> v = arch_u32("vocab_size");
    if (!v) v = model.get_u32("general.vocab_size");
    if (!v) v = model.get_u32("tokenizer.ggml.tokens.count");
    if (!v && token_embd_dims) {
      const auto& d = *token_embd_dims;
      if (d.size() >= 2) {
        uint64_t mx = 0;
        for (uint64_t x : d) mx = std::max(mx, x);
        v = static_cast<uint32_t>(mx);
      }
    }
    vocab_size = v ? static_cast<size_t>(*v) : 32000;
  }

  // context_size.
  size_t context_size;
  {
    auto v = arch_u32("context_length");
    context_size = v ? static_cast<size_t>(*v) : 4096;
  }

  // layer_count (excludes MTP/nextn heads).
  size_t nextn_layers = static_cast<size_t>(arch_u32("nextn_predict_layers").value_or(0));
  size_t block_count = static_cast<size_t>(arch_u32("block_count").value_or(32));
  size_t layer_count = block_count > nextn_layers ? block_count - nextn_layers : 0;

  // intermediate_size.
  size_t intermediate_size;
  {
    std::optional<uint32_t> v = arch_u32("feed_forward_length");
    if (v) {
      intermediate_size = static_cast<size_t>(*v);
    } else {
      auto d = first_layer_tensor_dims(meta, "ffn_gate.weight");
      if (!d) d = first_layer_tensor_dims(meta, "ffn_up.weight");
      if (d && d->size() >= 2) intermediate_size = static_cast<size_t>((*d)[1]);
      else intermediate_size = 11008;
    }
  }

  size_t num_attention_heads = static_cast<size_t>(arch_u32("attention.head_count").value_or(32));

  // attn_k output width: blk.0.attn_k.weight dim[1], else any *.attn_k.weight.
  std::optional<uint64_t> attn_k_out;
  {
    auto d = first_layer_tensor_dims(meta, "attn_k.weight");
    if (!d) {
      for (const auto& t : meta.tensor_infos) {
        if (ends_with(t.name, ".attn_k.weight")) {
          d = t.dimensions;
          break;
        }
      }
    }
    if (d && d->size() >= 2) attn_k_out = (*d)[1];
  }
  size_t head_dim_guess = num_attention_heads ? hidden_size / num_attention_heads : 0;

  // num_key_value_heads.
  size_t num_key_value_heads;
  {
    std::optional<size_t> v;
    if (auto u = arch_u32("attention.head_count_kv")) v = static_cast<size_t>(*u);
    if (!v) {
      if (auto u = metadata_u32_array_max(meta, metadata_prefix + ".attention.head_count_kv"))
        v = static_cast<size_t>(*u);
    }
    if (v && *v == 0) v.reset();  // .filter(|&v| v > 0)
    if (!v && attn_k_out && head_dim_guess > 0) {
      size_t inferred = static_cast<size_t>(*attn_k_out) / head_dim_guess;
      if (inferred > 0) v = inferred;
    }
    num_key_value_heads = v ? *v : num_attention_heads;
  }

  // key_value_head_dim.
  size_t key_value_head_dim;
  {
    std::optional<size_t> v;
    if (auto u = arch_u32("attention.key_length")) v = static_cast<size_t>(*u);
    if (!v && attn_k_out && num_key_value_heads > 0)
      v = static_cast<size_t>(*attn_k_out) / num_key_value_heads;
    if (!v)
      v = num_attention_heads ? hidden_size / num_attention_heads : 0;
    key_value_head_dim = *v;
  }
  // DeepSeek-V2 MLA stores the per-head MLA key dim in key_value_head_dim so the
  // KV cache slots size to the (small) MLA key. GLM-5.2 (glm-dsa), however,
  // caches the *compressed latent* (kv_lora_rank + rope) of width key_length=576
  // per token (kv_heads=1). For glm-dsa we therefore KEEP key_value_head_dim at
  // the cached width (576) and expose the MLA key/val dims (256) separately via
  // mla_key_dim / mla_val_dim below.
  if (uses_mla && !is_glm_dsa) {
    std::optional<size_t> mla_k;
    if (auto u = arch_u32("attention.key_length_mla")) mla_k = static_cast<size_t>(*u);
    if (!mla_k) {
      auto d = first_layer_tensor_dims(meta, "attn_q_b.weight");
      if (d && d->size() >= 2) {
        size_t denom = std::max<size_t>(num_attention_heads, 1);
        mla_k = static_cast<size_t>((*d)[1]) / denom;
      }
    }
    if (mla_k && *mla_k > 0) key_value_head_dim = *mla_k;
  }

  float rms_norm_eps = arch_f32("attention.layer_norm_rms_epsilon").value_or(1e-5f);

  float rope_theta = arch_f32("rope.freq_base").value_or(10000.0f);
  if (is_qwen35_family(arch) && rope_theta <= 10000.0f) {
    rope_theta = 10000000.0f;
  }

  size_t sliding_window = static_cast<size_t>(arch_u32("attention.sliding_window").value_or(0));

  // num_experts.
  size_t num_experts;
  {
    std::optional<uint32_t> v = arch_u32("expert_count");
    if (!v) v = model.get_u32("expert_count");
    num_experts = v ? static_cast<size_t>(*v) : 0;
    if (num_experts == 0) {
      for (const auto& t : meta.tensor_infos) {
        if (t.name == "blk.1.ffn_gate_inp.weight") {
          if (t.dimensions.size() >= 2)
            num_experts = static_cast<size_t>(t.dimensions[1]);
          break;
        }
      }
    }
  }

  // num_experts_per_tok.
  size_t num_experts_per_tok;
  {
    std::optional<uint32_t> v = arch_u32("expert_used_count");
    if (!v) v = model.get_u32("expert_used_count");
    num_experts_per_tok = v ? static_cast<size_t>(*v) : 0;
    if (num_experts_per_tok == 0 && uses_mla && num_experts > 0)
      num_experts_per_tok = 8;
  }

  // expert_intermediate_size.
  size_t expert_intermediate_size;
  {
    std::optional<uint32_t> v = arch_u32("expert_feed_forward_length");
    if (!v) v = model.get_u32("expert_feed_forward_length");
    expert_intermediate_size = v ? static_cast<size_t>(*v) : 0;
    if (expert_intermediate_size == 0) {
      auto d = first_layer_tensor_dims(meta, "ffn_gate_shexp.weight");
      if (!d) {
        for (const auto& t : meta.tensor_infos) {
          if (ends_with(t.name, ".ffn_gate_shexp.weight")) {
            d = t.dimensions;
            break;
          }
        }
      }
      if (d && d->size() >= 2)
        expert_intermediate_size = static_cast<size_t>((*d)[1]);
    }
  }

  size_t shortconv_l_cache = static_cast<size_t>(arch_u32("shortconv.l_cache").value_or(0));
  size_t leading_dense_layers = static_cast<size_t>(arch_u32("leading_dense_block_count").value_or(0));

  bool expert_gating_sigmoid;
  {
    std::optional<uint32_t> v = arch_u32("expert_gating_func");
    if (!v) v = model.get_u32("expert_gating_func");
    expert_gating_sigmoid = v ? (*v == 2) : false;
  }

  float expert_weights_scale;
  {
    std::optional<float> v = arch_f32("expert_weights_scale");
    if (!v) v = model.get_f32("expert_weights_scale");
    expert_weights_scale = (v && *v > 0.0f) ? *v : 1.0f;
  }

  size_t expert_group_count;
  {
    std::optional<uint32_t> v = arch_u32("expert_group_count");
    if (!v) v = model.get_u32("expert_group_count");
    expert_group_count = v ? static_cast<size_t>(*v) : 0;
  }
  size_t expert_group_used_count;
  {
    std::optional<uint32_t> v = arch_u32("expert_group_used_count");
    if (!v) v = model.get_u32("expert_group_used_count");
    expert_group_used_count = v ? static_cast<size_t>(*v) : 0;
  }

  // MLA (glm-dsa / DeepSeek-V2) ranks and per-head MLA key/value dims.
  size_t q_lora_rank = static_cast<size_t>(arch_u32("attention.q_lora_rank").value_or(0));
  size_t kv_lora_rank = static_cast<size_t>(arch_u32("attention.kv_lora_rank").value_or(0));
  size_t mla_key_dim = static_cast<size_t>(arch_u32("attention.key_length_mla").value_or(0));
  size_t mla_val_dim = static_cast<size_t>(arch_u32("attention.value_length_mla").value_or(0));

  // Shared (always-on) experts and routed-expert weight normalization.
  size_t num_shared_experts;
  {
    std::optional<uint32_t> v = arch_u32("expert_shared_count");
    if (!v) v = model.get_u32("expert_shared_count");
    num_shared_experts = v ? static_cast<size_t>(*v) : 0;
  }
  bool expert_weights_norm;
  {
    std::optional<uint32_t> v = arch_u32("expert_weights_norm");
    if (!v) v = model.get_u32("expert_weights_norm");
    expert_weights_norm = v ? (*v != 0) : false;
  }

  // rope_dim (partial RoPE).
  size_t rope_dim = static_cast<size_t>(arch_u32("rope.dimension_count").value_or(0));
  if (rope_dim == 0 && is_qwen35_family(arch) && key_value_head_dim > 0) {
    rope_dim = key_value_head_dim / 4;
  }

  // Gemma-family specifics.
  bool is_gemma = (architecture == Architecture::Gemma);
  size_t sliding_window_pattern = 0;
  float rope_theta_swa = 0.0f;
  float embedding_scale = 1.0f;
  bool gelu_ffn = false;
  bool sandwich_norm = false;
  if (is_gemma) {
    std::optional<uint32_t> v = arch_u32("attention.sliding_window_pattern");
    if (v && *v > 0) {
      sliding_window_pattern = static_cast<size_t>(*v);
    } else {
      sliding_window_pattern = (arch == "gemma2") ? 2 : 6;
    }
    float swa = arch_f32("rope.freq_base_swa").value_or(0.0f);
    rope_theta_swa = (swa > 0.0f) ? swa : 10000.0f;
    embedding_scale = std::sqrt(static_cast<float>(hidden_size));
    gelu_ffn = true;
    sandwich_norm = true;
  }

  // rms_norm_weight_plus_one: only qwen35/qwen35moe lineage. (OXIDIZE_RMS_PLUS_ONE
  // env override is a debug hook in Rust and intentionally omitted here.)
  bool rms_norm_weight_plus_one =
      (arch == "qwen35" || arch == "qwen35moe" || arch == "qwen3_5_moe" ||
       arch == "qwen3_5_moe_text");

  InferenceConfig cfg;
  cfg.vocab_size = vocab_size;
  cfg.context_size = context_size;
  cfg.layer_count = layer_count;
  cfg.hidden_size = hidden_size;
  cfg.intermediate_size = intermediate_size;
  cfg.num_attention_heads = num_attention_heads;
  cfg.num_key_value_heads = num_key_value_heads;
  cfg.key_value_head_dim = key_value_head_dim;
  cfg.kv_cache_dtype = DType::F32;
  cfg.rms_norm_eps = rms_norm_eps;
  cfg.rope_theta = rope_theta;
  cfg.architecture = architecture;
  cfg.sliding_window = sliding_window;
  cfg.num_experts = num_experts;
  cfg.num_experts_per_tok = num_experts_per_tok;
  cfg.expert_intermediate_size = expert_intermediate_size;
  cfg.alibi_num_heads = 0;
  cfg.shortconv_l_cache = shortconv_l_cache;
  cfg.leading_dense_layers = leading_dense_layers;
  cfg.expert_gating_sigmoid = expert_gating_sigmoid;
  cfg.rope_dim = rope_dim;
  cfg.rope_theta_swa = rope_theta_swa;
  cfg.sliding_window_pattern = sliding_window_pattern;
  cfg.embedding_scale = embedding_scale;
  cfg.gelu_ffn = gelu_ffn;
  cfg.sandwich_norm = sandwich_norm;
  cfg.rms_norm_weight_plus_one = rms_norm_weight_plus_one;
  cfg.nextn_predict_layers = nextn_layers;
  cfg.expert_weights_scale = expert_weights_scale;
  cfg.expert_group_count = expert_group_count;
  cfg.expert_group_used_count = expert_group_used_count;
  cfg.q_lora_rank = q_lora_rank;
  cfg.kv_lora_rank = kv_lora_rank;
  cfg.mla_key_dim = mla_key_dim;
  cfg.mla_val_dim = mla_val_dim;
  cfg.num_shared_experts = num_shared_experts;
  cfg.expert_weights_norm = expert_weights_norm;
  return cfg;
}

}  // namespace oxidize
