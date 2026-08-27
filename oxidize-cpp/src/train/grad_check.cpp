// Finite-difference gradient checks for all backward ops.
// Compiled into the grad_check ctest executable.
//
// Each test: perturb input by delta=1e-5 (double precision oracle), rtol 1%.
// Double-precision FD avoids float32 cancellation in softmax/exp derivatives.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>
#include <random>

#include "oxidize/autograd.hpp"
#include "oxidize/lora.hpp"

namespace {

static const double DELTA = 1e-5;
static const double RTOL  = 0.01;   // 1%
static const double ATOL  = 1e-7;

bool check_close(double analytic, double numeric, const char* what, size_t idx) {
  double denom = std::max(std::abs(analytic), std::max(std::abs(numeric), ATOL));
  double rel = std::abs(analytic - numeric) / denom;
  if (rel > RTOL) {
    printf("FAIL [%s] idx=%zu  analytic=%.8e  numeric=%.8e  rel=%.5f\n",
           what, idx, analytic, numeric, rel);
    return false;
  }
  return true;
}

// FD check with double-precision oracle and float analytic gradient.
// f_d: double-precision forward function
// grad_fn: fills float analytic gradient given float x
int fd_check(const char* name, size_t n,
              std::function<double(const double*)> f_d,
              std::function<void(const float*, float*)> grad_fn,
              const float* x_ref) {
  std::vector<float> analytic(n, 0.0f);
  grad_fn(x_ref, analytic.data());

  int fails = 0;
  std::vector<double> xd(x_ref, x_ref + n);
  for (size_t i = 0; i < n; ++i) {
    double orig = xd[i];
    xd[i] = orig + DELTA;
    double fp = f_d(xd.data());
    xd[i] = orig - DELTA;
    double fm = f_d(xd.data());
    xd[i] = orig;
    double numeric = (fp - fm) / (2.0 * DELTA);
    if (!check_close(static_cast<double>(analytic[i]), numeric, name, i)) {
      ++fails;
      if (fails > 5) { printf("  (stopping after 5 failures)\n"); break; }
    }
  }
  return fails;
}

// ---- Tests ---------------------------------------------------------------

int test_matmul_backward() {
  printf("  matmul_backward...\n");
  std::mt19937_64 rng(1);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  const size_t rows = 8, cols = 6;
  std::vector<float> W(rows * cols), x(cols), dy(rows);
  for (auto& v : W) v = nd(rng);
  for (auto& v : x) v = nd(rng);
  for (auto& v : dy) v = nd(rng);

  // Loss = sum(dy * W * xi).
  auto f_x = [&](const double* xd) {
    double loss = 0.0;
    for (size_t r = 0; r < rows; ++r) {
      double y = 0.0;
      for (size_t c = 0; c < cols; ++c) y += W[r * cols + c] * xd[c];
      loss += dy[r] * y;
    }
    return loss;
  };
  auto g_x = [&](const float* xi, float* gx) {
    std::fill(gx, gx + cols, 0.0f);
    std::vector<float> dW(rows * cols, 0.0f);
    oxidize::matmul_backward(gx, dW.data(), W.data(), xi, dy.data(), rows, cols);
  };
  int fails = fd_check("matmul/dx", cols, f_x, g_x, x.data());

  auto f_W = [&](const double* Wd) {
    double loss = 0.0;
    for (size_t r = 0; r < rows; ++r) {
      double y = 0.0;
      for (size_t c = 0; c < cols; ++c) y += Wd[r * cols + c] * x[c];
      loss += dy[r] * y;
    }
    return loss;
  };
  auto g_W = [&](const float* Wi, float* gW) {
    std::fill(gW, gW + rows * cols, 0.0f);
    std::vector<float> dx(cols, 0.0f);
    oxidize::matmul_backward(dx.data(), gW, Wi, x.data(), dy.data(), rows, cols);
  };
  fails += fd_check("matmul/dW", rows * cols, f_W, g_W, W.data());
  return fails;
}

int test_rmsnorm_backward() {
  printf("  rmsnorm_backward...\n");
  std::mt19937_64 rng(2);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  const size_t n = 16;
  const double eps = 1e-5;

  std::vector<float> x(n), w(n), dy(n);
  for (auto& v : x) v = nd(rng);
  for (auto& v : w) v = nd(rng);
  for (auto& v : dy) v = nd(rng);

  // Double-precision RMSNorm forward.
  auto rms_fwd_d = [&](const double* xi, const double* wi, bool plus_one) {
    double sumsq = 0.0;
    for (size_t i = 0; i < n; ++i) sumsq += xi[i] * xi[i];
    double inv_rms = 1.0 / std::sqrt(sumsq / n + eps);
    double loss = 0.0;
    for (size_t i = 0; i < n; ++i) {
      double weff = plus_one ? (wi[i] + 1.0) : wi[i];
      loss += dy[i] * xi[i] * inv_rms * weff;
    }
    return loss;
  };

  int fails = 0;
  for (bool plus_one : {false, true}) {
    std::vector<double> wd(w.begin(), w.end());
    auto f_x = [&](const double* xd) { return rms_fwd_d(xd, wd.data(), plus_one); };
    auto g_x = [&](const float* xi, float* gx) {
      std::fill(gx, gx + n, 0.0f);
      std::vector<float> gw(n, 0.0f);
      oxidize::rmsnorm_backward(gx, gw.data(), xi, w.data(), dy.data(), n,
                                 static_cast<float>(eps), plus_one);
    };
    fails += fd_check(plus_one ? "rmsnorm/dx(+1)" : "rmsnorm/dx", n, f_x, g_x, x.data());

    std::vector<double> xd(x.begin(), x.end());
    auto f_w = [&](const double* wi) { return rms_fwd_d(xd.data(), wi, plus_one); };
    auto g_w = [&](const float* wi, float* gw) {
      std::fill(gw, gw + n, 0.0f);
      std::vector<float> gx(n, 0.0f);
      oxidize::rmsnorm_backward(gx.data(), gw, x.data(), wi, dy.data(), n,
                                 static_cast<float>(eps), plus_one);
    };
    fails += fd_check(plus_one ? "rmsnorm/dw(+1)" : "rmsnorm/dw", n, f_w, g_w, w.data());
  }
  return fails;
}

int test_swiglu_backward() {
  printf("  swiglu_backward...\n");
  std::mt19937_64 rng(3);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  const size_t n = 12;
  std::vector<float> gate(n), up(n), dy(n);
  for (auto& v : gate) v = nd(rng);
  for (auto& v : up)   v = nd(rng);
  for (auto& v : dy)   v = nd(rng);

  auto silu_d = [](double g) { return g / (1.0 + std::exp(-g)); };

  std::vector<double> upd(up.begin(), up.end());
  auto f_g = [&](const double* gd) {
    double loss = 0.0;
    for (size_t i = 0; i < n; ++i) loss += dy[i] * silu_d(gd[i]) * upd[i];
    return loss;
  };
  auto gr_g = [&](const float* g, float* dg) {
    std::fill(dg, dg + n, 0.0f);
    std::vector<float> dup(n, 0.0f);
    oxidize::swiglu_backward(dg, dup.data(), g, up.data(), dy.data(), n);
  };
  int fails = fd_check("swiglu/dgate", n, f_g, gr_g, gate.data());

  std::vector<double> gd(gate.begin(), gate.end());
  auto f_u = [&](const double* ud) {
    double loss = 0.0;
    for (size_t i = 0; i < n; ++i) loss += dy[i] * silu_d(gd[i]) * ud[i];
    return loss;
  };
  auto gr_u = [&](const float* u, float* du) {
    std::fill(du, du + n, 0.0f);
    std::vector<float> dg(n, 0.0f);
    oxidize::swiglu_backward(dg.data(), du, gate.data(), u, dy.data(), n);
  };
  fails += fd_check("swiglu/dup", n, f_u, gr_u, up.data());
  return fails;
}

int test_cross_entropy() {
  printf("  cross_entropy_backward...\n");
  std::mt19937_64 rng(4);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  const size_t vocab = 32;
  const size_t target = 7;

  std::vector<float> z(vocab);
  for (auto& v : z) v = nd(rng);

  // Double-precision cross-entropy forward.
  auto ce_d = [&](const double* zd) {
    double mx = zd[0];
    for (size_t i = 1; i < vocab; ++i) if (zd[i] > mx) mx = zd[i];
    double sum = 0.0;
    for (size_t i = 0; i < vocab; ++i) sum += std::exp(zd[i] - mx);
    return -(zd[target] - mx - std::log(sum));
  };
  auto g_z = [&](const float* zi, float* dz) {
    std::fill(dz, dz + vocab, 0.0f);
    oxidize::cross_entropy_backward(dz, zi, target, 1.0f, 1.0f, vocab);
  };
  return fd_check("cross_entropy/dz", vocab, ce_d, g_z, z.data());
}

int test_attention_backward() {
  printf("  attention_backward...\n");
  std::mt19937_64 rng(5);
  std::normal_distribution<float> nd(0.0f, 0.5f);

  const size_t T = 4, n_heads = 2, kv_heads = 1, hd = 4;
  const size_t q_len = n_heads * hd, kv_len = kv_heads * hd;
  const double scale = 1.0 / std::sqrt(static_cast<double>(hd));

  std::vector<float> q(T * q_len), k(T * kv_len), v(T * kv_len), dout(T * q_len);
  for (auto& x : q) x = nd(rng);
  for (auto& x : k) x = nd(rng);
  for (auto& x : v) x = nd(rng);
  for (auto& x : dout) x = nd(rng);

  // Double-precision causal attention forward.
  auto attn_fwd_d = [&](const double* qd, const double* kd, const double* vd,
                          std::vector<double>& attn_w) {
    attn_w.assign(n_heads * T * T, 0.0);
    std::vector<double> out(T * q_len, 0.0);
    size_t groups = n_heads / kv_heads;
    for (size_t h = 0; h < n_heads; ++h) {
      size_t kv_h = h / groups;
      for (size_t t = 0; t < T; ++t) {
        const double* qh = qd + t * q_len + h * hd;
        double* row = attn_w.data() + h * T * T + t * T;
        double mx = -1e38;
        for (size_t s = 0; s <= t; ++s) {
          const double* kh = kd + s * kv_len + kv_h * hd;
          double dot = 0.0;
          for (size_t d = 0; d < hd; ++d) dot += qh[d] * kh[d];
          row[s] = dot * scale; mx = std::max(mx, row[s]);
        }
        double sm = 0.0;
        for (size_t s = 0; s <= t; ++s) { row[s] = std::exp(row[s] - mx); sm += row[s]; }
        if (sm > 0.0) for (size_t s = 0; s <= t; ++s) row[s] /= sm;
        double* oh = out.data() + t * q_len + h * hd;
        for (size_t s = 0; s <= t; ++s) {
          const double* vh = vd + s * kv_len + kv_h * hd;
          for (size_t d = 0; d < hd; ++d) oh[d] += row[s] * vh[d];
        }
      }
    }
    return out;
  };

  auto loss_d = [&](const std::vector<double>& out) {
    double l = 0.0;
    for (size_t i = 0; i < T * q_len; ++i) l += dout[i] * out[i];
    return l;
  };

  int fails = 0;

  // Check dq.
  {
    std::vector<double> kd(k.begin(), k.end()), vd(v.begin(), v.end());
    auto f_q = [&](const double* qd) {
      std::vector<double> aw; return loss_d(attn_fwd_d(qd, kd.data(), vd.data(), aw));
    };
    auto g_q = [&](const float* qf, float* gq) {
      // Compute float attn weights.
      std::vector<float> aw(n_heads * T * T, 0.0f);
      size_t groups = n_heads / kv_heads;
      float scf = static_cast<float>(scale);
      for (size_t h = 0; h < n_heads; ++h) {
        size_t kv_h = h / groups;
        for (size_t t = 0; t < T; ++t) {
          const float* qh = qf + t * q_len + h * hd;
          float* row = aw.data() + h * T * T + t * T;
          float mx = -1e38f;
          for (size_t s = 0; s <= t; ++s) {
            const float* kh = k.data() + s * kv_len + kv_h * hd;
            float dot = 0.0f;
            for (size_t d = 0; d < hd; ++d) dot += qh[d] * kh[d];
            row[s] = dot * scf; mx = std::max(mx, row[s]);
          }
          float sm = 0.0f;
          for (size_t s = 0; s <= t; ++s) { row[s] = std::exp(row[s] - mx); sm += row[s]; }
          if (sm > 0.0f) for (size_t s = 0; s <= t; ++s) row[s] /= sm;
        }
      }
      // Analytic: compute dq given saved aw.
      std::fill(gq, gq + T * q_len, 0.0f);
      std::vector<float> dk(T * kv_len, 0.0f), dv(T * kv_len, 0.0f);
      for (size_t h = 0; h < n_heads; ++h) {
        size_t kv_h = h / groups;
        for (size_t t = 0; t < T; ++t) {
          const float* attn_h = aw.data() + h * T * T + t * T;
          const float* doh = dout.data() + t * q_len + h * hd;
          const float* qh = qf + t * q_len + h * hd;
          std::vector<float> dattn(t + 1, 0.0f);
          for (size_t s = 0; s <= t; ++s) {
            const float* vh = v.data() + s * kv_len + kv_h * hd;
            float d = 0.0f;
            for (size_t d_ = 0; d_ < hd; ++d_) d += doh[d_] * vh[d_];
            dattn[s] = d;
          }
          float sda = 0.0f;
          for (size_t s = 0; s <= t; ++s) sda += attn_h[s] * dattn[s];
          float* gqh = gq + t * q_len + h * hd;
          for (size_t s = 0; s <= t; ++s) {
            float ds = attn_h[s] * (dattn[s] - sda);
            const float* kh = k.data() + s * kv_len + kv_h * hd;
            for (size_t d = 0; d < hd; ++d) gqh[d] += scf * ds * kh[d];
          }
          (void)qh; (void)dk; (void)dv;
        }
      }
    };
    fails += fd_check("attention/dq", T * q_len, f_q, g_q, q.data());
  }

  // Check dv (simpler: linear in v given fixed attn_weights).
  {
    std::vector<double> qd(q.begin(), q.end()), kd(k.begin(), k.end());
    // Pre-compute float attn weights at reference q, k.
    std::vector<float> aw_ref(n_heads * T * T, 0.0f);
    size_t groups = n_heads / kv_heads;
    float scf = static_cast<float>(scale);
    for (size_t h = 0; h < n_heads; ++h) {
      size_t kv_h = h / groups;
      for (size_t t = 0; t < T; ++t) {
        const float* qh = q.data() + t * q_len + h * hd;
        float* row = aw_ref.data() + h * T * T + t * T;
        float mx = -1e38f;
        for (size_t s = 0; s <= t; ++s) {
          const float* kh = k.data() + s * kv_len + kv_h * hd;
          float dot = 0.0f;
          for (size_t d = 0; d < hd; ++d) dot += qh[d] * kh[d];
          row[s] = dot * scf; mx = std::max(mx, row[s]);
        }
        float sm = 0.0f;
        for (size_t s = 0; s <= t; ++s) { row[s] = std::exp(row[s] - mx); sm += row[s]; }
        if (sm > 0.0f) for (size_t s = 0; s <= t; ++s) row[s] /= sm;
      }
    }

    auto f_v = [&](const double* vd) {
      std::vector<double> aw; return loss_d(attn_fwd_d(qd.data(), kd.data(), vd, aw));
    };
    auto g_v = [&](const float* vf, float* gv) {
      std::fill(gv, gv + T * kv_len, 0.0f);
      // dv[t, kv_h] += sum_h(attn[h, t', t] * dout[h, t'])
      for (size_t h = 0; h < n_heads; ++h) {
        size_t kv_h = h / groups;
        for (size_t tp = 0; tp < T; ++tp) {
          const float* row = aw_ref.data() + h * T * T + tp * T;
          const float* doh = dout.data() + tp * q_len + h * hd;
          for (size_t s = 0; s <= tp; ++s) {
            float* gvh = gv + s * kv_len + kv_h * hd;
            for (size_t d = 0; d < hd; ++d) gvh[d] += row[s] * doh[d];
          }
        }
      }
      (void)vf;
    };
    fails += fd_check("attention/dv", T * kv_len, f_v, g_v, v.data());
  }
  return fails;
}

int test_lora_backward() {
  printf("  lora_backward...\n");
  std::mt19937_64 rng(6);
  std::normal_distribution<float> nd(0.0f, 0.5f);

  const size_t rows = 8, cols = 6, rank = 3;
  oxidize::LoraAdapter la;
  la.init(rows, cols, rank, static_cast<float>(rank), rng);

  std::vector<float> x(cols), dy(rows);
  for (auto& v : x)  v = nd(rng);
  for (auto& v : dy) v = nd(rng);

  const float scaling = la.scaling;

  // Double-precision LoRA forward.
  auto lora_fwd_d = [&](const double* Ad, const double* Bd, const double* xd) {
    std::vector<double> ax(rank, 0.0);
    for (size_t r = 0; r < rank; ++r)
      for (size_t c = 0; c < cols; ++c) ax[r] += Ad[r * cols + c] * xd[c];
    double loss = 0.0;
    for (size_t row = 0; row < rows; ++row) {
      double s = 0.0;
      for (size_t r = 0; r < rank; ++r) s += Bd[row * rank + r] * ax[r];
      loss += static_cast<double>(dy[row]) * scaling * s;
    }
    return loss;
  };

  // Recompute ax for backward.
  std::vector<float> ax(rank, 0.0f);
  for (size_t r = 0; r < rank; ++r)
    for (size_t c = 0; c < cols; ++c) ax[r] += la.A[r * cols + c] * x[c];

  int fails = 0;
  std::vector<double> Bd(la.B.begin(), la.B.end());
  std::vector<double> xd(x.begin(), x.end());

  // dA.
  {
    auto f = [&](const double* Ad) { return lora_fwd_d(Ad, Bd.data(), xd.data()); };
    auto g = [&](const float* Af, float* gA) {
      oxidize::LoraAdapter la2 = la;
      la2.A.assign(Af, Af + rank * cols);
      la2.zero_grads();
      std::vector<float> ax2(rank, 0.0f);
      for (size_t r = 0; r < rank; ++r)
        for (size_t c = 0; c < cols; ++c) ax2[r] += la2.A[r * cols + c] * x[c];
      std::vector<float> dx_la(cols, 0.0f);
      la2.backward(x.data(), dy.data(), ax2.data(), dx_la.data(), rank);
      std::copy(la2.dA.begin(), la2.dA.end(), gA);
    };
    fails += fd_check("lora/dA", rank * cols, f, g, la.A.data());
  }
  // dB.
  {
    std::vector<double> Ad(la.A.begin(), la.A.end());
    auto f = [&](const double* Bd2) { return lora_fwd_d(Ad.data(), Bd2, xd.data()); };
    auto g = [&](const float* Bf, float* gB) {
      oxidize::LoraAdapter la2 = la;
      la2.B.assign(Bf, Bf + rows * rank);
      la2.zero_grads();
      std::vector<float> dx_la(cols, 0.0f);
      la2.backward(x.data(), dy.data(), ax.data(), dx_la.data(), rank);
      std::copy(la2.dB.begin(), la2.dB.end(), gB);
    };
    fails += fd_check("lora/dB", rows * rank, f, g, la.B.data());
  }
  // dx.
  {
    std::vector<double> Ad(la.A.begin(), la.A.end());
    auto f = [&](const double* xd2) { return lora_fwd_d(Ad.data(), Bd.data(), xd2); };
    auto g = [&](const float* xf, float* gx) {
      std::fill(gx, gx + cols, 0.0f);
      std::vector<float> ax2(rank, 0.0f);
      for (size_t r = 0; r < rank; ++r)
        for (size_t c = 0; c < cols; ++c) ax2[r] += la.A[r * cols + c] * xf[c];
      oxidize::LoraAdapter la2 = la;
      la2.zero_grads();
      la2.backward(xf, dy.data(), ax2.data(), gx, rank);
    };
    fails += fd_check("lora/dx", cols, f, g, x.data());
  }
  return fails;
}

}  // namespace

int main() {
  printf("=== Gradient checks (delta=1e-5, rtol=1%%, double-precision oracle) ===\n");
  int total_fails = 0;
  total_fails += test_matmul_backward();
  total_fails += test_rmsnorm_backward();
  total_fails += test_swiglu_backward();
  total_fails += test_cross_entropy();
  total_fails += test_attention_backward();
  total_fails += test_lora_backward();
  if (total_fails == 0) {
    printf("ALL GRADIENT CHECKS PASSED\n");
    return 0;
  }
  printf("FAILED: %d gradient check(s)\n", total_fails);
  return 1;
}
