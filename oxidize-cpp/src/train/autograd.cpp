// Autograd tape implementation + backward op kernels.

#include "oxidize/autograd.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace oxidize {


float* Tape::alloc(size_t n) {
  BufEntry e;
  e.data.assign(n, 0.0f);
  // grad deferred until grad() is called
  bufs_.push_back(std::move(e));
  float* ptr = bufs_.back().data.data();
  ptr_to_idx_[ptr] = bufs_.size() - 1;
  return ptr;
}

float* Tape::grad(const float* ptr) {
  auto it = ptr_to_idx_.find(ptr);
  if (it == ptr_to_idx_.end()) {
    throw std::runtime_error("Tape::grad: unknown buffer");
  }
  BufEntry& e = bufs_[it->second];
  if (e.grad.empty()) {
    e.grad.assign(e.data.size(), 0.0f);
  }
  return e.grad.data();
}

void Tape::push(std::function<void()> fn) {
  ops_.push_back(std::move(fn));
}

void Tape::backward() {
  for (int i = static_cast<int>(ops_.size()) - 1; i >= 0; --i) {
    ops_[i]();
  }
}

void Tape::zero_grads() {
  for (auto& e : bufs_) {
    if (!e.grad.empty()) std::fill(e.grad.begin(), e.grad.end(), 0.0f);
  }
}

void Tape::reset() {
  bufs_.clear();
  ptr_to_idx_.clear();
  ops_.clear();
}

size_t Tape::size_of(const float* ptr) const {
  auto it = ptr_to_idx_.find(ptr);
  if (it == ptr_to_idx_.end()) return 0;
  return bufs_[it->second].data.size();
}


void matmul_backward(float* dx, float* dW,
                     const float* W, const float* x, const float* dy,
                     size_t rows, size_t cols) {
  // dx += W^T * dy  (dx may be nullptr to skip)
  if (dx) {
    for (size_t c = 0; c < cols; ++c) {
      float s = 0.0f;
      for (size_t r = 0; r < rows; ++r) s += W[r * cols + c] * dy[r];
      dx[c] += s;
    }
  }
  // dW += dy outer x  (dW may be nullptr to skip, e.g. frozen weights in LoRA)
  if (dW) {
    for (size_t r = 0; r < rows; ++r) {
      const float dy_r = dy[r];
      for (size_t c = 0; c < cols; ++c) {
        dW[r * cols + c] += dy_r * x[c];
      }
    }
  }
}

void rmsnorm_backward(float* dx, float* dw,
                      const float* x, const float* w, const float* dy,
                      size_t n, float eps, bool weight_plus_one) {
  // Compute inv_rms.
  float sumsq = 0.0f;
  for (size_t i = 0; i < n; ++i) sumsq += x[i] * x[i];
  float inv_rms = 1.0f / std::sqrt(sumsq / static_cast<float>(n) + eps);
  float inv_rms2 = inv_rms * inv_rms;

  // Effective weights (wi = w[i] + 1 if plus_one else w[i]).
  // sum_term = (1/n) * sum(dy * wi * x)
  float sum_term = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float wi = weight_plus_one ? (w[i] + 1.0f) : w[i];
    sum_term += dy[i] * wi * x[i];
  }
  sum_term *= inv_rms2 / static_cast<float>(n);

  // dx[i] += inv_rms * (dy[i]*wi - x[i] * sum_term)
  // dw[i] += dy[i] * x[i] * inv_rms
  for (size_t i = 0; i < n; ++i) {
    float wi = weight_plus_one ? (w[i] + 1.0f) : w[i];
    dx[i] += inv_rms * (dy[i] * wi - x[i] * sum_term);
    dw[i] += dy[i] * x[i] * inv_rms;
  }
}

void rope_backward(float* dx, const float* dy,
                   size_t head_dim, size_t num_heads, size_t pos,
                   float theta, size_t rope_dim) {
  size_t eff_rope = (rope_dim == 0) ? head_dim : std::min(rope_dim, head_dim);
  if (eff_rope % 2 != 0) {
    throw std::runtime_error("rope_backward: effective rope dim must be even");
  }
  size_t half = eff_rope / 2;

  for (size_t h = 0; h < num_heads; ++h) {
    const float* dyh = dy + h * head_dim;
    float*       dxh = dx + h * head_dim;

    for (size_t i = 0; i < half; ++i) {
      float angle = static_cast<float>(pos) /
                    std::pow(theta, static_cast<float>(2 * i) /
                                        static_cast<float>(eff_rope));
      float c = std::cos(angle);
      float s = std::sin(angle);
      // Forward: y0 = x0*c - x1*s;  y1 = x0*s + x1*c
      // Backward: dx0 = dy0*c + dy1*s;  dx1 = -dy0*s + dy1*c
      dxh[i]        = dyh[i] * c + dyh[i + half] * s;
      dxh[i + half] = -dyh[i] * s + dyh[i + half] * c;
    }
    // Pass-through dims.
    for (size_t i = eff_rope; i < head_dim; ++i) dxh[i] = dyh[i];
  }
}

void swiglu_backward(float* dgate, float* dup,
                     const float* gate, const float* up, const float* dout,
                     size_t n) {
  for (size_t i = 0; i < n; ++i) {
    float g = gate[i];
    float s = 1.0f / (1.0f + std::exp(-g));  // sigmoid(gate)
    // out = silu(g) * up = g*s*up
    // dout[i] -> dgate[i], dup[i]
    // d(silu)/dg = s*(1 + g*(1-s))
    dgate[i] += dout[i] * up[i] * s * (1.0f + g * (1.0f - s));
    dup[i]   += dout[i] * g * s;
  }
}

void geglu_backward(float* dgate, float* dup,
                    const float* gate, const float* up, const float* dout,
                    size_t n) {
  constexpr float kSqrt2OverPi = 0.7978845608028654f;  // sqrt(2/pi)
  constexpr float kCoeff = 0.044715f;
  for (size_t i = 0; i < n; ++i) {
    float x = gate[i];
    float x3 = x * x * x;
    float inner = kSqrt2OverPi * (x + kCoeff * x3);
    float tanh_inner = std::tanh(inner);
    float gelu = 0.5f * x * (1.0f + tanh_inner);
    float sech2 = 1.0f - tanh_inner * tanh_inner;
    // dgelu/dx = 0.5*(1+tanh) + 0.5*x*sech^2*d_inner/dx
    // d_inner/dx = sqrt(2/pi)*(1 + 3*coeff*x^2)
    float d_inner_dx = kSqrt2OverPi * (1.0f + 3.0f * kCoeff * x * x);
    float dgelu_dx = 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech2 * d_inner_dx;
    dgate[i] += dout[i] * up[i] * dgelu_dx;
    dup[i]   += dout[i] * gelu;
  }
}

void attention_backward(float* dq, float* dk, float* dv,
                        const float* q, const float* k, const float* v,
                        const float* attn_weights,
                        const float* dout,
                        size_t T, size_t n_heads, size_t kv_heads, size_t hd) {
  // attn_weights: [n_heads x T]  (flat, causal: only first T values meaningful)
  // q:    [n_heads x hd]   (single query position)
  // k, v: [T x kv_heads x hd]
  // dout: [n_heads x hd]
  // dq, dk, dv: same shapes as q, k, v (accumulated).
  float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  size_t groups = n_heads / kv_heads;

  for (size_t h = 0; h < n_heads; ++h) {
    size_t kv_h = h / groups;
    const float* attn_h = attn_weights + h * T;
    const float* dout_h = dout         + h * hd;
    const float* q_h    = q            + h * hd;
    float* dq_h         = dq           + h * hd;

    // dattn[t] = dot(dout_h, v[t, kv_h])
    std::vector<float> dattn(T, 0.0f);
    for (size_t t = 0; t < T; ++t) {
      const float* v_t = v + (t * kv_heads + kv_h) * hd;
      float s = 0.0f;
      for (size_t d = 0; d < hd; ++d) s += dout_h[d] * v_t[d];
      dattn[t] = s;
    }

    float sum_dattn = 0.0f;
    for (size_t t = 0; t < T; ++t) sum_dattn += attn_h[t] * dattn[t];

    for (size_t t = 0; t < T; ++t) {
      float ds = attn_h[t] * (dattn[t] - sum_dattn);
      const float* k_t = k + (t * kv_heads + kv_h) * hd;

      // dq[h] += scale * ds * k[t, kv_h]
      for (size_t d = 0; d < hd; ++d) dq_h[d] += scale * ds * k_t[d];

      // dk[t, kv_h] += scale * ds * q[h]
      float* dk_t = dk + (t * kv_heads + kv_h) * hd;
      for (size_t d = 0; d < hd; ++d) dk_t[d] += scale * ds * q_h[d];

      // dv[t, kv_h] += attn[t] * dout_h
      float* dv_t = dv + (t * kv_heads + kv_h) * hd;
      for (size_t d = 0; d < hd; ++d) dv_t[d] += attn_h[t] * dout_h[d];
    }
  }
}

void embedding_backward(float* demb, const float* dout,
                        size_t token_id, size_t h) {
  float* row = demb + token_id * h;
  for (size_t i = 0; i < h; ++i) row[i] += dout[i];
}

float cross_entropy_forward(const float* z, size_t target, float mask,
                             size_t vocab) {
  if (mask == 0.0f) return 0.0f;
  float mx = z[0];
  for (size_t i = 1; i < vocab; ++i) if (z[i] > mx) mx = z[i];
  float sum = 0.0f;
  for (size_t i = 0; i < vocab; ++i) sum += std::exp(z[i] - mx);
  float log_p_target = z[target] - mx - std::log(sum);
  return -log_p_target;
}

void cross_entropy_backward(float* dz, const float* z, size_t target,
                             float mask, float active_count, size_t vocab) {
  if (mask == 0.0f || active_count <= 0.0f) return;
  // Compute softmax.
  float mx = z[0];
  for (size_t i = 1; i < vocab; ++i) if (z[i] > mx) mx = z[i];
  float sum = 0.0f;
  for (size_t i = 0; i < vocab; ++i) sum += std::exp(z[i] - mx);

  float inv_sum = 1.0f / sum;
  float inv_active = mask / active_count;
  for (size_t i = 0; i < vocab; ++i) {
    float p = std::exp(z[i] - mx) * inv_sum;
    dz[i] += (p - (i == target ? 1.0f : 0.0f)) * inv_active;
  }
}

}  // namespace oxidize
