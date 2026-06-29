#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace oxidize {

struct VisionConfig {
  int image_size = 336;
  int patch_size = 14;
  int hidden_size = 1536;
  int projection_dim = 4096;
  int num_image_tokens = 576;
  float image_mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
  float image_std[3] = {0.26862954f, 0.26130258f, 0.27577711f};

  int num_patches_per_side() const { return image_size / patch_size; }
  int num_patches() const { return num_patches_per_side() * num_patches_per_side(); }
  int patch_dim() const { return patch_size * patch_size * 3; }
};

struct ImagePatches {
  std::vector<float> data;
  int num_patches = 0;
  int patch_dim = 0;
  int original_width = 0;
  int original_height = 0;
};

class ImagePreprocessor {
 public:
  explicit ImagePreprocessor(VisionConfig cfg) : cfg_(std::move(cfg)) {}
  ImagePatches preprocess_rgb(const uint8_t* pixels, int width, int height) const;
  ImagePatches preprocess_rgba(const uint8_t* pixels, int width, int height) const;

 private:
  VisionConfig cfg_;
};

// Loads mmproj GGUF and runs the vision tower. Encoder matmul stack is WIP;
// preprocess + tensor inventory work today.
class MmProjModel {
 public:
  explicit MmProjModel(std::string path);
  const std::string& path() const { return path_; }
  bool loaded() const { return loaded_; }
  std::vector<float> encode(const ImagePatches& patches) const;

 private:
  std::string path_;
  bool loaded_ = false;
  VisionConfig cfg_;
};

std::vector<float> build_multimodal_embeddings(
    const std::vector<float>& text_embd,
    int hidden_size,
    const std::vector<float>& image_embd,
    int image_token_count,
    int insert_position);

}  // namespace oxidize
