// CPU-only parity gate for the oxidize C++ port.
//
// This is the numerical correctness gate that MUST pass before any GPU spend.
// It builds small synthetic tensors and asserts the C++ CPU kernels match
// hand-computed reference values within 1e-4:
//   - rms_norm           (oxidize/tensor.hpp)
//   - apply_rope         (oxidize/tensor.hpp)
//   - swiglu_inplace     (oxidize/tensor.hpp)
//   - matvec / gemv      (oxidize/tensor.hpp)
//   - dequantize_row Q4_K and Q8_0 single blocks (oxidize/quant.hpp)
//
// Reference values were derived independently (Python, see commit notes) from
// the same scalar math the Rust kernels implement:
//   oxidize-core/src/compute/tensor/kernels.rs (rms_norm_f32, apply_rope_f32,
//     apply_swiglu_inplace_f32, gemv_f32_cpu)
//   oxidize-core/src/compute/quantization.rs   (dequant_q4_k, dequant_q8_0,
//     get_scale_min_k4, f16_le_to_f32)
//
// Plus a section guarded by the OXIDIZE_TEST_GGUF environment variable: if set
// to a path, it loads that GGUF, runs one forward step, and asserts the logits
// are finite and vocab-sized (mirrors oxidize-core/src/model/inference.rs
// forward semantics via oxidize/model_llama.hpp).
//
// No test framework: plain assert-style checks; non-zero exit on failure.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "oxidize/model.hpp"
#include "oxidize/model_llama.hpp"
#include "oxidize/quant.hpp"
#include "oxidize/tensor.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

void check_bool(bool cond, const char* what, const char* file, int line) {
  ++g_checks;
  if (!cond) {
    ++g_failures;
    std::fprintf(stderr, "FAIL [%s:%d]: %s\n", file, line, what);
  }
}

void check_close(float got, float want, float tol, const char* what,
                 const char* file, int line) {
  ++g_checks;
  const float diff = std::fabs(got - want);
  if (!(diff <= tol) || std::isnan(got)) {
    ++g_failures;
    std::fprintf(stderr,
                 "FAIL [%s:%d]: %s  got=%.9g want=%.9g |diff|=%.9g tol=%g\n",
                 file, line, what, got, want, diff, tol);
  }
}

#define CHECK(cond) check_bool((cond), #cond, __FILE__, __LINE__)
#define CHECK_CLOSE(got, want, tol) \
  check_close((got), (want), (tol), #got " ~= " #want, __FILE__, __LINE__)

constexpr float kTol = 1e-4f;

// x = [1,2,3,4], weight = [1,1,1,1], eps = 0, weight_plus_one = false.
// mean(x^2) = (1+4+9+16)/4 = 7.5 ; inv_rms = 1/sqrt(7.5).
// out[i] = x[i] / sqrt(7.5).
void test_rms_norm() {
  const std::vector<float> x = {1.f, 2.f, 3.f, 4.f};
  const std::vector<float> w = {1.f, 1.f, 1.f, 1.f};
  std::vector<float> out(4, 0.f);
  oxidize::rms_norm(out.data(), x.data(), w.data(), 4, 0.0f, false);
  const float inv = 1.0f / std::sqrt(7.5f);
  for (int i = 0; i < 4; ++i) {
    CHECK_CLOSE(out[i], x[i] * inv, kTol);
  }

  // weight_plus_one path (Gemma / Qwen-(1+w) RMSNorm): scale = (w+1).
  const std::vector<float> w2 = {0.5f, 0.5f, 0.5f, 0.5f};  // -> scale 1.5
  std::vector<float> out2(4, 0.f);
  oxidize::rms_norm(out2.data(), x.data(), w2.data(), 4, 0.0f, true);
  for (int i = 0; i < 4; ++i) {
    CHECK_CLOSE(out2[i], x[i] * inv * 1.5f, kTol);
  }
}

// One head, head_dim = 4, full rotation (rope_dim = 0), pos = 1, theta = 10000.
// half_dim = 2. freq_0 = 1, freq_1 = theta^(-2/4) = theta^-0.5 = 1/100.
// angle_0 = 1*1 = 1 rad ; angle_1 = 1*0.01 = 0.01 rad.
// vec = [a0,a1,b0,b1] where pairs are (h[i], h[half_dim+i]):
//   pair0 = (vec[0], vec[2]) with angle_0
//   pair1 = (vec[1], vec[3]) with angle_1
// out[0]      = a0*cos - b0*sin
// out[2]      = a0*sin + b0*cos
// out[1]      = a1*cos1 - b1*sin1
// out[3]      = a1*sin1 + b1*cos1
void test_rope() {
  std::vector<float> v = {1.f, 2.f, 3.f, 4.f};  // a0=1,a1=2,b0=3,b1=4
  oxidize::apply_rope(v.data(), /*head_dim=*/4, /*num_heads=*/1, /*pos=*/1,
                      /*theta=*/10000.0f, /*rope_dim=*/0);
  const float ang0 = 1.0f;          // freq_0=1, pos=1
  const float ang1 = 0.01f;         // freq_1 = 10000^-0.5 = 0.01
  const float c0 = std::cos(ang0), s0 = std::sin(ang0);
  const float c1 = std::cos(ang1), s1 = std::sin(ang1);
  CHECK_CLOSE(v[0], 1.f * c0 - 3.f * s0, kTol);
  CHECK_CLOSE(v[2], 1.f * s0 + 3.f * c0, kTol);
  CHECK_CLOSE(v[1], 2.f * c1 - 4.f * s1, kTol);
  CHECK_CLOSE(v[3], 2.f * s1 + 4.f * c1, kTol);

  // pos == 0 is an identity (apply_rope_f32 returns input unchanged at pos 0).
  std::vector<float> v0 = {5.f, 6.f, 7.f, 8.f};
  oxidize::apply_rope(v0.data(), 4, 1, 0, 10000.0f, 0);
  CHECK_CLOSE(v0[0], 5.f, kTol);
  CHECK_CLOSE(v0[1], 6.f, kTol);
  CHECK_CLOSE(v0[2], 7.f, kTol);
  CHECK_CLOSE(v0[3], 8.f, kTol);

  // Partial RoPE: rope_dim = 2 rotates only the first 2 of 4 dims, rest pass
  // through unchanged. half_dim = 1, freq_0 = 1, angle = pos*1 = 1.
  std::vector<float> vp = {1.f, 3.f, 9.f, 11.f};  // pair0=(vp[0],vp[1]); tail=vp[2],vp[3]
  oxidize::apply_rope(vp.data(), 4, 1, 1, 10000.0f, /*rope_dim=*/2);
  CHECK_CLOSE(vp[0], 1.f * c0 - 3.f * s0, kTol);
  CHECK_CLOSE(vp[1], 1.f * s0 + 3.f * c0, kTol);
  CHECK_CLOSE(vp[2], 9.f, kTol);   // untouched
  CHECK_CLOSE(vp[3], 11.f, kTol);  // untouched
}

// out = silu(gate) * up, silu(x) = x*sigmoid(x).
void test_swiglu() {
  std::vector<float> gate = {0.f, 1.f, -1.f, 2.f};
  std::vector<float> up = {1.f, 2.f, 3.f, 0.5f};
  std::vector<float> out(4, 0.f);
  oxidize::swiglu_inplace(gate.data(), up.data(), out.data(), 4);
  for (int i = 0; i < 4; ++i) {
    const float g = (i == 0) ? 0.f : (i == 1) ? 1.f : (i == 2) ? -1.f : 2.f;
    const float silu = g * (1.0f / (1.0f + std::exp(-g)));
    CHECK_CLOSE(out[i], silu * up[i], kTol);
  }
  // silu(0) = 0 exactly.
  CHECK_CLOSE(out[0], 0.0f, kTol);
}

// W = [[1,2,3],[4,5,6]] (2x3 row-major), x = [1,1,1].
// y = [1+2+3, 4+5+6] = [6, 15].
void test_matvec() {
  const std::vector<float> W = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  const std::vector<float> x = {1.f, 1.f, 1.f};
  std::vector<float> y(2, 0.f);
  oxidize::matvec(y.data(), W.data(), x.data(), 2, 3);
  CHECK_CLOSE(y[0], 6.0f, kTol);
  CHECK_CLOSE(y[1], 15.0f, kTol);

  // dot with a non-uniform x to catch column-order bugs.
  const std::vector<float> x2 = {1.f, 10.f, 100.f};
  std::vector<float> y2(2, 0.f);
  oxidize::matvec(y2.data(), W.data(), x2.data(), 2, 3);
  CHECK_CLOSE(y2[0], 1.f + 20.f + 300.f, kTol);   // 321
  CHECK_CLOSE(y2[1], 4.f + 50.f + 600.f, kTol);   // 654
}

// f16 little-endian writer for the two scale values we need (exact: these
// round-trip without loss). 0.5=0x3800, 0.25=0x3400, 0.125=0x3000.
void put_f16(std::vector<uint8_t>& b, uint16_t bits) {
  b.push_back(static_cast<uint8_t>(bits & 0xFF));
  b.push_back(static_cast<uint8_t>(bits >> 8));
}

// d = 0.5 ; qs[i] (int8) chosen as a regular ramp; out[i] = qs[i] * d.
void test_dequant_q8_0() {
  std::vector<uint8_t> blk;
  put_f16(blk, 0x3800);  // d = 0.5
  std::vector<int8_t> qs(32);
  for (int i = 0; i < 32; ++i) {
    int v = -128 + i * 8;          // -128 .. 120 step 8
    qs[i] = static_cast<int8_t>(v);
    blk.push_back(static_cast<uint8_t>(static_cast<uint8_t>(qs[i])));
  }
  CHECK(blk.size() == oxidize::BLOCK_Q8_0_SIZE);

  std::vector<float> out(32, 0.f);
  oxidize::dequantize_row(oxidize::QuantType::Q8_0, blk.data(), out.data(), 32);
  for (int i = 0; i < 32; ++i) {
    CHECK_CLOSE(out[i], static_cast<float>(qs[i]) * 0.5f, kTol);
  }
  // Spot-check exact endpoints against the Python reference.
  CHECK_CLOSE(out[0], -64.0f, kTol);
  CHECK_CLOSE(out[31], 60.0f, kTol);
}

// get_scale_min_k4 reference (quantization.rs) for the independent recompute.
void ref_scale_min_k4(int j, const uint8_t* sc, uint8_t& s, uint8_t& m) {
  if (j < 4) {
    s = sc[j] & 63;
    m = sc[j + 4] & 63;
  } else {
    s = (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
    m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
  }
}

// d = 0.25, min = 0.125. scales[12] chosen with low values so high-bit
// borrowing (scales[j-4]>>6) is zero, keeping the math hand-traceable. qs[128]
// is a deterministic nibble pattern. We recompute the full reference here from
// the same algorithm and assert the kernel matches, plus pin a few exact values
// from the independent Python reference (-0.625, 24.25, -0.875, 7.5).
void test_dequant_q4_k() {
  const float d = 0.25f;
  const float mn = 0.125f;
  const uint8_t scales[12] = {10, 20, 30, 40, 5, 6, 7, 8, 12, 13, 14, 15};
  uint8_t qs[128];
  for (int i = 0; i < 128; ++i) {
    const int lo = (i * 7) % 16;
    const int hi = (i * 3 + 5) % 16;
    qs[i] = static_cast<uint8_t>(lo | (hi << 4));
  }

  std::vector<uint8_t> blk;
  put_f16(blk, 0x3400);  // d = 0.25
  put_f16(blk, 0x3000);  // min = 0.125
  for (int i = 0; i < 12; ++i) blk.push_back(scales[i]);
  for (int i = 0; i < 128; ++i) blk.push_back(qs[i]);
  CHECK(blk.size() == oxidize::BLOCK_Q4_K_SIZE);

  std::vector<float> out(256, 0.f);
  oxidize::dequantize_row(oxidize::QuantType::Q4_K_M, blk.data(), out.data(),
                          256);

  // Independent reference recompute (mirrors dequant_q4_k loop structure).
  float ref[256];
  size_t out_ptr = 0;
  int is = 0;
  for (int gp = 0; gp < 4; ++gp) {
    const size_t q_base = static_cast<size_t>(gp) * 32;
    uint8_t sc1, m1, sc2, m2;
    ref_scale_min_k4(is, scales, sc1, m1);
    ref_scale_min_k4(is + 1, scales, sc2, m2);
    const float d1 = d * static_cast<float>(sc1);
    const float min1 = mn * static_cast<float>(m1);
    const float d2 = d * static_cast<float>(sc2);
    const float min2 = mn * static_cast<float>(m2);
    for (int l = 0; l < 32; ++l) {
      ref[out_ptr + l] = d1 * static_cast<float>(qs[q_base + l] & 0xF) - min1;
    }
    for (int l = 0; l < 32; ++l) {
      ref[out_ptr + 32 + l] =
          d2 * static_cast<float>(qs[q_base + l] >> 4) - min2;
    }
    out_ptr += 64;
    is += 2;
  }
  for (int i = 0; i < 256; ++i) {
    CHECK_CLOSE(out[i], ref[i], kTol);
  }

  // Pins from the independent Python derivation.
  CHECK_CLOSE(out[0], -0.625f, kTol);
  CHECK_CLOSE(out[32], 24.25f, kTol);
  CHECK_CLOSE(out[64], -0.875f, kTol);
  CHECK_CLOSE(out[255], 7.5f, kTol);
}

// KVALUES_IQ4NL codebook (quant.cpp) — spot-check IQ4_NL dequant.
constexpr int8_t kIq4NlCodebook[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

void test_dequant_iq4_nl() {
  std::vector<uint8_t> blk(oxidize::BLOCK_IQ4_NL_SIZE, 0);
  blk[0] = 0x00;
  blk[1] = 0x3C;  // f16 1.0
  blk[2] = 0x10;  // nibbles 0 and 1
  CHECK(blk.size() == oxidize::BLOCK_IQ4_NL_SIZE);

  std::vector<float> out(oxidize::QK4_NL, 0.f);
  oxidize::dequantize_row(oxidize::QuantType::IQ4_NL, blk.data(), out.data(),
                          oxidize::QK4_NL);
  CHECK_CLOSE(out[0], static_cast<float>(kIq4NlCodebook[0]), kTol);
  CHECK_CLOSE(out[16], static_cast<float>(kIq4NlCodebook[1]), kTol);
  for (float v : out) {
    CHECK(std::isfinite(v));
  }
}

// Fused int8 GEMV for IQ4_NL must match dequant-then-dot within act-quant tolerance.
void test_gemv_iq4_nl() {
  constexpr size_t cols = 256;
  constexpr size_t nb = cols / oxidize::QK4_NL;
  std::vector<uint8_t> row(nb * oxidize::BLOCK_IQ4_NL_SIZE);
  uint32_t state = 12345u;
  for (size_t i = 0; i < row.size(); ++i)
    row[i] = static_cast<uint8_t>((state = state * 1664525u + 1013904223u) >> 13);
  for (size_t b = 0; b < nb; ++b) {
    uint8_t* blk = row.data() + b * oxidize::BLOCK_IQ4_NL_SIZE;
    blk[0] = static_cast<uint8_t>(0x00 | (state & 0xFF));
    blk[1] = 0x2C;  // ~0.06 f16 scale
  }

  std::vector<float> x(cols);
  for (size_t i = 0; i < cols; ++i)
    x[i] = (static_cast<float>((state = state * 1664525u + 1013904223u) & 0xFFFFu) /
            65536.0f - 0.5f) *
           2.0f;

  std::vector<float> dq(cols);
  oxidize::dequantize_row(oxidize::QuantType::IQ4_NL, row.data(), dq.data(), cols);
  float ref = 0.0f;
  for (size_t i = 0; i < cols; ++i) ref += dq[i] * x[i];

  std::vector<float> got(1, 0.f);
  oxidize::gemv_quantized(got.data(), oxidize::QuantType::IQ4_NL, row.data(), 1,
                          cols, x.data());

  float mag = 0.0f;
  for (size_t i = 0; i < cols; ++i) mag += std::fabs(dq[i] * x[i]);
  const float tol = 0.02f * (mag > 1.0f ? mag : 1.0f);
  CHECK(std::fabs(got[0] - ref) <= tol);
  std::fprintf(stderr, "[iq4_nl gemv] fused=%.5f ref=%.5f tol=%.5f\n", got[0], ref,
               tol);
}

void test_dequant_iq2_xs_finite() {
  std::vector<uint8_t> blk(oxidize::BLOCK_IQ2_XS_SIZE, 0);
  blk[0] = 0x00;
  blk[1] = 0x3C;  // f16 1.0
  for (size_t i = 2; i < blk.size(); ++i) {
    blk[i] = static_cast<uint8_t>(i % 251);
  }
  std::vector<float> out(oxidize::QK_K, 0.f);
  oxidize::dequantize_row(oxidize::QuantType::IQ2_XS, blk.data(), out.data(),
                          oxidize::QK_K);
  for (float v : out) {
    CHECK(std::isfinite(v));
  }
}

void test_dequant_iq2_s_finite() {
  std::vector<uint8_t> blk(oxidize::BLOCK_IQ2_S_SIZE, 0);
  blk[0] = 0x00;
  blk[1] = 0x3C;  // f16 1.0
  for (size_t i = 2; i < blk.size(); ++i) {
    blk[i] = static_cast<uint8_t>(i % 251);
  }
  std::vector<float> out(oxidize::QK_K, 0.f);
  oxidize::dequantize_row(oxidize::QuantType::IQ2_S, blk.data(), out.data(),
                          oxidize::QK_K);
  for (float v : out) {
    CHECK(std::isfinite(v));
  }
}

// f16_le_to_f32 bit-exact spot checks (quantization.rs::f16_le_to_f32).
void test_f16() {
  uint8_t half[2] = {0x00, 0x3C};  // 1.0
  CHECK_CLOSE(oxidize::f16_le_to_f32(half), 1.0f, 0.0f);
  uint8_t half2[2] = {0x00, 0xC0};  // -2.0
  CHECK_CLOSE(oxidize::f16_le_to_f32(half2), -2.0f, 0.0f);
  uint8_t half0[2] = {0x00, 0x00};  // 0.0
  CHECK_CLOSE(oxidize::f16_le_to_f32(half0), 0.0f, 0.0f);
}

// Guarded by OXIDIZE_TEST_GGUF=<path>. Loads the model, runs one forward step
// on a single token, and asserts the logits vector is vocab-sized and finite.
void test_gguf_forward_if_present() {
  const char* path = std::getenv("OXIDIZE_TEST_GGUF");
  if (path == nullptr || path[0] == '\0') {
    std::fprintf(stderr,
                 "[skip] OXIDIZE_TEST_GGUF not set; skipping GGUF forward.\n");
    return;
  }
  std::fprintf(stderr, "[gguf] loading %s\n", path);

  // Hard errors (parse failure, unsupported architecture, missing tensors) are
  // surfaced as a parity failure rather than an uncaught throw. Note: pointing
  // this at a header-only parser fixture (no data section) is user error and may
  // SIGBUS via mmap-past-EOF — supply a real model GGUF.
  try {
    std::unique_ptr<oxidize::Model> model = oxidize::load_llama_gguf(path);
    CHECK(model != nullptr);
    const size_t vocab = model->vocab_size();
    CHECK(vocab > 0);

    oxidize::Session session;
    // Token 0 (BOS-ish) is always a valid index into the embedding table.
    const std::vector<oxidize::Token> tokens = {0u};
    oxidize::Logits logits = model->forward(tokens, session);

    CHECK(logits.size() == vocab);
    bool all_finite = true;
    for (float v : logits) {
      if (!std::isfinite(v)) {
        all_finite = false;
        break;
      }
    }
    CHECK(all_finite);
    CHECK(session.consumed_tokens() == 1);
    std::fprintf(stderr, "[gguf] forward ok: vocab=%zu logits[0]=%.6g\n", vocab,
                 logits.empty() ? 0.0f : logits[0]);
  } catch (const std::exception& e) {
    ++g_failures;
    std::fprintf(stderr, "FAIL [gguf]: %s\n", e.what());
  }
}


void test_quant_al5() {
  constexpr size_t N = 256;
  std::array<float, N> x{};
  for (size_t i = 0; i < N; ++i)
    x[i] = (static_cast<float>((i * 1103515245u + 12345u) & 0xffffu) / 65536.0f -
            0.5f) *
           2.0f;

  std::array<uint8_t, N / 32 * 18> buf40{};
  std::array<uint8_t, N / 32 * 18> buf4o{};
  std::array<float, N> dq40{};
  std::array<float, N> dq4o{};

  oxidize::quantize_row_q4_0(x.data(), buf40.data(), N);
  oxidize::quantize_row_al5(x.data(), buf4o.data(), N);
  oxidize::dequantize_row(oxidize::QuantType::Q4_0, buf40.data(), dq40.data(), N);
  oxidize::dequantize_row(oxidize::QuantType::AL5, buf4o.data(), dq4o.data(), N);

  double se40 = 0.0;
  double se4o = 0.0;
  for (size_t i = 0; i < N; ++i) {
    double e40 = static_cast<double>(dq40[i] - x[i]);
    double e4o = static_cast<double>(dq4o[i] - x[i]);
    se40 += e40 * e40;
    se4o += e4o * e4o;
  }
  const double rmse40 = std::sqrt(se40 / static_cast<double>(N));
  const double rmse4o = std::sqrt(se4o / static_cast<double>(N));
  CHECK(rmse4o < rmse40);
  std::fprintf(stderr, "[al5] rmse=%.5f vs q4_0 rmse=%.5f (%.1f%% lower)\n",
               rmse4o, rmse40, 100.0 * (rmse40 - rmse4o) / rmse40);
}

}  // namespace

int main() {
  test_rms_norm();
  test_rope();
  test_swiglu();
  test_matvec();
  test_dequant_q8_0();
  test_dequant_q4_k();
  test_dequant_iq4_nl();
  test_gemv_iq4_nl();
  test_dequant_iq2_xs_finite();
  test_dequant_iq2_s_finite();
  test_quant_al5();
  test_f16();
  test_gguf_forward_if_present();

  if (g_failures == 0) {
    std::fprintf(stderr, "PARITY OK: %d checks passed.\n", g_checks);
    return 0;
  }
  std::fprintf(stderr, "PARITY FAILED: %d/%d checks failed.\n", g_failures,
               g_checks);
  return 1;
}
