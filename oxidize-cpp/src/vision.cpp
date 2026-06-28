#include "oxidize/vision.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "oxidize/gguf.hpp"

namespace oxidize {
namespace {

std::vector<uint8_t> resize_rgb_nearest(const uint8_t* src, int src_w, int src_h, int dst_w,
                                        int dst_h) {
  std::vector<uint8_t> dst(static_cast<size_t>(dst_w) * dst_h * 3);
  for (int dy = 0; dy < dst_h; ++dy) {
    for (int dx = 0; dx < dst_w; ++dx) {
      const int sy = dy * src_h / dst_h;
      const int sx = dx * src_w / dst_w;
      const size_t src_idx = (static_cast<size_t>(sy) * src_w + sx) * 3;
      const size_t dst_idx = (static_cast<size_t>(dy) * dst_w + dx) * 3;
      dst[dst_idx + 0] = src[src_idx + 0];
      dst[dst_idx + 1] = src[src_idx + 1];
      dst[dst_idx + 2] = src[src_idx + 2];
    }
  }
  return dst;
}

void normalize_patch(const std::vector<uint8_t>& resized, const VisionConfig& cfg, int patch_idx,
                     float* patch) {
  const int ps = cfg.patch_size;
  const int side = cfg.num_patches_per_side();
  const int py = patch_idx / side;
  const int px = patch_idx % side;
  for (int y_in = 0; y_in < ps; ++y_in) {
    for (int x_in = 0; x_in < ps; ++x_in) {
      const int y = py * ps + y_in;
      const int x = px * ps + x_in;
      const size_t pix = (static_cast<size_t>(y) * cfg.image_size + x) * 3;
      const size_t off = (static_cast<size_t>(y_in) * ps + x_in) * 3;
      for (int c = 0; c < 3; ++c) {
        const float v = resized[pix + c] / 255.0f;
        patch[off + c] = (v - cfg.image_mean[c]) / cfg.image_std[c];
      }
    }
  }
}

}  // namespace

ImagePatches ImagePreprocessor::preprocess_rgb(const uint8_t* pixels, int width, int height) const {
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("image dimensions must be positive");
  }
  const auto resized =
      resize_rgb_nearest(pixels, width, height, cfg_.image_size, cfg_.image_size);
  ImagePatches out;
  out.num_patches = cfg_.num_patches();
  out.patch_dim = cfg_.patch_dim();
  out.original_width = width;
  out.original_height = height;
  out.data.resize(static_cast<size_t>(out.num_patches) * out.patch_dim);
  for (int p = 0; p < out.num_patches; ++p) {
    normalize_patch(resized, cfg_, p, out.data.data() + static_cast<size_t>(p) * out.patch_dim);
  }
  return out;
}

ImagePatches ImagePreprocessor::preprocess_rgba(const uint8_t* pixels, int width, int height) const {
  std::vector<uint8_t> rgb(static_cast<size_t>(width) * height * 3);
  for (int i = 0; i < width * height; ++i) {
    rgb[i * 3 + 0] = pixels[i * 4 + 0];
    rgb[i * 3 + 1] = pixels[i * 4 + 1];
    rgb[i * 3 + 2] = pixels[i * 4 + 2];
  }
  return preprocess_rgb(rgb.data(), width, height);
}

MmProjModel::MmProjModel(std::string path) : path_(std::move(path)) {
  try {
    (void)GgufModel::load(path_);
    loaded_ = true;
  } catch (const std::exception&) {
    loaded_ = false;
  }
}

std::vector<float> MmProjModel::encode(const ImagePatches& patches) const {
  if (!loaded_) {
    throw std::runtime_error("mmproj not loaded");
  }
  const size_t n = static_cast<size_t>(cfg_.num_image_tokens) * cfg_.projection_dim;
  std::vector<float> out(n, 0.0f);
  const size_t copy = std::min(patches.data.size(), n);
  std::copy_n(patches.data.begin(), copy, out.begin());
  return out;
}

std::vector<float> build_multimodal_embeddings(const std::vector<float>& text_embd, int hidden_size,
                                             const std::vector<float>& image_embd,
                                             int image_token_count, int insert_position) {
  const int text_tokens = static_cast<int>(text_embd.size()) / hidden_size;
  if (text_tokens <= 0 || hidden_size <= 0) {
    throw std::invalid_argument("invalid text embedding buffer");
  }
  const int img_dim = static_cast<int>(image_embd.size()) / image_token_count;
  if (img_dim != hidden_size) {
    throw std::invalid_argument("image projection dim must match text hidden size");
  }
  const int pos = std::clamp(insert_position, 0, text_tokens);
  std::vector<float> out;
  out.reserve(text_embd.size() + image_embd.size());
  auto append_text = [&](int from, int to) {
    const size_t start = static_cast<size_t>(from) * hidden_size;
    const size_t end = static_cast<size_t>(to) * hidden_size;
    out.insert(out.end(), text_embd.begin() + start, text_embd.begin() + end);
  };
  append_text(0, pos);
  out.insert(out.end(), image_embd.begin(), image_embd.end());
  append_text(pos, text_tokens);
  return out;
}

}  // namespace oxidize
