#pragma once
// Ported from oxidize-core/src/format/gguf.rs (parser, metadata KV store,
// tensor info table, alignment + data-section offset) and the GGUF-driven
// InferenceConfig derivation in oxidize-core/src/model/inference.rs
// (InferenceConfig::from_gguf and its metadata helpers).
//
// Byte/bit-faithful port: little-endian readers, GGUF v2/v3, default
// alignment 32, llama.cpp metadata key conventions ("${arch}.block_count",
// ".embedding_length", etc.). QuantType + from_ggml_type come from quant.hpp.

#include <cstdint>
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "oxidize/config.hpp"
#include "oxidize/quant.hpp"

namespace oxidize {

enum class GgufMmapAdvice {
  SequentialPrefetch,
  Random,
};

const char* gguf_mmap_advice_name(GgufMmapAdvice advice);
GgufMmapAdvice gguf_mmap_advice_from_name(const std::string& name);
GgufMmapAdvice gguf_mmap_advice_from_numa_mode(const std::string& numa_mode);

// Mirror of gguf.rs::GgufMetadataType (repr(u32) discriminants).
enum class GgufMetadataType : uint32_t {
  Uint8 = 0,
  Int8 = 1,
  Uint16 = 2,
  Int16 = 3,
  Uint32 = 4,
  Int32 = 5,
  Float32 = 6,
  Bool = 7,
  String = 8,
  Array = 9,
  Uint64 = 10,
  Int64 = 11,
  Float64 = 12,
};

// Mirror of gguf.rs::GgufMetadataValue. A tagged union over all GGUF scalar
// types plus String and (possibly nested) Array. Arrays store their element
// type and a flat vector of values.
struct GgufMetadataValue {
  enum class Kind : uint8_t {
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Float32,
    Bool,
    String,
    Array,
    Uint64,
    Int64,
    Float64,
  };

  Kind kind = Kind::Uint32;

  // Scalar payloads (only the one matching `kind` is meaningful).
  uint64_t u = 0;   // holds Uint8/Uint16/Uint32/Uint64/Bool
  int64_t i = 0;    // holds Int8/Int16/Int32/Int64
  double f = 0.0;   // holds Float32 (as f32 widened) / Float64
  std::string str;  // holds String

  // Array payload.
  GgufMetadataType array_element_type = GgufMetadataType::Uint8;
  std::vector<GgufMetadataValue> array;

  bool is_array() const { return kind == Kind::Array; }
};

// Mirror of gguf.rs::GgufTensorInfo (post data-section resolution).
struct GgufTensorInfo {
  std::string name;                 // mapped (internal) name, see map_tensor_name
  std::string raw_name;             // original GGUF tensor name
  std::vector<uint64_t> dimensions; // GGUF dim order (as stored)
  uint32_t ggml_type = 0;           // raw ggml_type id
  QuantType quant = QuantType::F32; // from_ggml_type(ggml_type)
  uint64_t relative_offset = 0;     // offset within data section
  uint64_t absolute_offset = 0;     // data_section_start + relative_offset (within its shard)
  size_t shard_index = 0;           // which mmap'd shard this tensor lives in (0 = single file)
};

// Mirror of gguf.rs::GgufFile (the parsed header), parsed from a byte slice.
struct GgufFile {
  uint32_t version = 0;
  uint64_t tensor_count = 0;
  std::map<std::string, GgufMetadataValue> metadata;
  std::vector<GgufTensorInfo> tensor_infos;
  uint64_t alignment = 32;
  uint64_t data_section_start = 0;

  // gguf.rs::GgufFile::architecture — prefer general.architecture, else detect
  // from a known metadata namespace prefix. Returns empty string if unknown.
  std::string architecture() const;
};

// Resolved tensor view returned by GgufModel::tensor.
struct TensorView {
  const uint8_t* data = nullptr;
  std::vector<uint64_t> dims;
  QuantType quant = QuantType::F32;
  uint64_t byte_offset = 0;  // absolute offset into the mmap
};

// Owns a POSIX mmap of a .gguf file and the parsed header. Mirrors
// gguf.rs::MappedGgufFile + accessor surface needed by downstream modules.
class GgufModel {
 public:
  // Parse magic/version/metadata/tensor-table from an in-memory byte slice.
  // Does NOT take ownership of `bytes`. Throws std::runtime_error on any
  // malformed input (mirrors GgufParseError variants).
  static GgufFile parse(const uint8_t* bytes, size_t len);

  // mmap a .gguf file (read-only, MAP_PRIVATE) and parse its header.
  // Throws std::runtime_error on open/mmap/parse failure.
  static GgufModel load(
      const std::string& path,
      GgufMmapAdvice advice = GgufMmapAdvice::SequentialPrefetch);

  GgufModel(const GgufModel&) = delete;
  GgufModel& operator=(const GgufModel&) = delete;
  GgufModel(GgufModel&&) noexcept;
  GgufModel& operator=(GgufModel&&) noexcept;
  ~GgufModel();

  const GgufFile& parsed() const { return parsed_; }
  const uint8_t* bytes() const { return base_; }
  size_t size() const { return size_; }

  // ---- metadata accessors (key = full GGUF key, e.g. "llama.block_count") ----
  // Numeric accessors mirror the widening rules in inference.rs::metadata_*_lookup.
  std::optional<uint32_t> get_u32(const std::string& key) const;
  std::optional<float> get_f32(const std::string& key) const;
  std::optional<std::string> get_string(const std::string& key) const;
  // Returns a pointer to the raw array value (nullptr if missing/not an array).
  const GgufMetadataValue* get_array(const std::string& key) const;

  // Architecture detection -> config.hpp::Architecture via architecture_from_name.
  Architecture architecture() const;

  // Tensor lookup by mapped (internal) name, e.g. "blk.3.attn_q.weight".
  // Throws std::runtime_error if not found.
  TensorView tensor(const std::string& name) const;
  bool has_tensor(const std::string& name) const;

 private:
  GgufModel() = default;

  // One mmap'd file (a single GGUF or one shard of a split GGUF).
  struct Shard {
    void* map = nullptr;
    const uint8_t* base = nullptr;
    size_t size = 0;
  };

  // Load a split GGUF: open shard 0 at `path`, read split.count, then mmap and
  // parse every sibling shard and merge their tensor tables into one model.
  static GgufModel load_split(const std::string& first_path,
                              GgufModel first_model, uint32_t split_count,
                              GgufMmapAdvice advice);

  void* map_ = nullptr;       // mmap base of shard 0 (for munmap); also shards_[0]
  const uint8_t* base_ = nullptr;  // shard 0 base
  size_t size_ = 0;           // shard 0 size
  std::vector<Shard> shards_; // all mmap'd shards (>=1). shards_[0] mirrors map_/base_/size_.
  GgufFile parsed_;           // merged header (metadata from shard 0, all tensor_infos)
};

// Map a GGUF general.architecture / namespace prefix string to a config.hpp
// Architecture. Implements architecture_from_name declared in config.hpp,
// mirroring ModelArchitecture::from_gguf in inference.rs.
// (defined in gguf.cpp)

// Build an InferenceConfig from a parsed GGUF model, mirroring
// InferenceConfig::from_gguf in inference.rs. Reads the standard llama.cpp
// metadata keys under the resolved architecture prefix, falling back to tensor
// dimensions exactly as the Rust does.
InferenceConfig build_inference_config(const GgufModel& model);

}  // namespace oxidize
