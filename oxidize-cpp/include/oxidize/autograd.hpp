#pragma once
// Minimal reverse-mode autograd tape over fp32 buffers.
//
// Usage:
//   Tape tape;
//   float* y = tape.alloc(n);           // allocate forward buffer
//   float* dy = tape.grad(y);           // get/create gradient buffer
//   tape.push([=]{ ... backward ... }); // register backward op
//   tape.backward();                    // run backward ops in reverse
//   tape.zero_grads();                  // zero all grad buffers

#include <cstddef>
#include <functional>
#include <unordered_map>
#include <vector>

namespace oxidize {

class Tape {
 public:
  // Allocate a forward-pass buffer of `n` floats (zeroed). Owned by the tape.
  float* alloc(size_t n);

  // Get (or create) the gradient buffer for a forward buffer `ptr`.
  // The gradient buffer is the same size as the forward buffer.
  // Calling grad() implicitly registers `ptr` as a differentiable node.
  float* grad(const float* ptr);

  // Register a backward closure. Will be called in LIFO order during backward().
  void push(std::function<void()> fn);

  // Run all backward closures in reverse (LIFO) order.
  void backward();

  // Zero all gradient buffers (call before each accumulation step).
  void zero_grads();

  // Release all forward + grad buffers and clear the tape.
  void reset();

  // Number of elements in the buffer starting at `ptr`.
  size_t size_of(const float* ptr) const;

 private:
  struct BufEntry {
    std::vector<float> data;   // forward values
    std::vector<float> grad;   // gradient (same size, lazily allocated)
  };

  // Map from raw pointer (data.data()) -> entry index for O(1) lookup.
  std::vector<BufEntry> bufs_;
  std::unordered_map<const float*, size_t> ptr_to_idx_;

  std::vector<std::function<void()>> ops_;  // backward closures (push order)
};


// MatMul backward. y = W * x.
//   dx += W^T * dy        (x: [cols], W: [rows x cols], y: [rows])
//   dW += dy outer x      (dW += dy[:,None] * x[None,:])
// Accumulates into dx and dW (does NOT zero first).
void matmul_backward(float* dx, float* dW,
                     const float* W, const float* x, const float* dy,
                     size_t rows, size_t cols);

// RMSNorm backward. out = x * inv_rms * w  (weight_plus_one: w -> w+1).
//   inv_rms = 1 / sqrt(mean(x^2) + eps)
//   dx += inv_rms * (dy*w - x * inv_rms^2 * (1/n) * sum(dy*w*x))
//   dw += dy * x * inv_rms     (always; the +1 doesn't change dw formula)
// Accumulates into dx, dw.
void rmsnorm_backward(float* dx, float* dw,
                      const float* x, const float* w, const float* dy,
                      size_t n, float eps, bool weight_plus_one);

// RoPE backward (no parameters). Inverse rotation:
//   dx0 = dy0*cos + dy1*sin
//   dx1 = -dy0*sin + dy1*cos
// Applies to all heads. Writes into dx (overwrites, not accumulates).
void rope_backward(float* dx, const float* dy,
                   size_t head_dim, size_t num_heads, size_t pos,
                   float theta, size_t rope_dim);

// SwiGLU backward. out = silu(gate) * up, silu(g) = g * sigmoid(g).
//   s = sigmoid(gate);  dgate = dout * up * s * (1 + gate*(1-s));
//   dup = dout * gate * s
// Accumulates into dgate, dup.
void swiglu_backward(float* dgate, float* dup,
                     const float* gate, const float* up, const float* dout,
                     size_t n);

// GeGLU backward. out = gelu_tanh(gate) * up.
//   gelu'(x) = 0.5*(1+tanh(k*(x+0.044715*x^3))) + x*sech^2(k*(x+0.044715*x^3))*k*(1+3*0.044715*x^2)
//   k = sqrt(2/pi)
// Accumulates into dgate, dup.
void geglu_backward(float* dgate, float* dup,
                    const float* gate, const float* up, const float* dout,
                    size_t n);

// Materialized attention backward (causal, GQA).
// Saved: q [n_heads*hd], k [T*kv_heads*hd], v [T*kv_heads*hd], attn [n_heads*T].
//   dv += attn^T * dout     (per head group)
//   dattn = dout * v^T
//   dscores = attn * (dattn - sum(attn * dattn))   (softmax backward)
//   dq += scale * sum(dscores * k)
//   dk += scale * dscores^T * q
// Accumulates into dq, dk, dv.
void attention_backward(float* dq, float* dk, float* dv,
                        const float* q, const float* k, const float* v,
                        const float* attn_weights,  // [n_heads * T]
                        const float* dout,           // [n_heads * hd]
                        size_t T, size_t n_heads, size_t kv_heads, size_t hd);

// Embedding backward: scatter-add dout[token_id*h : (token_id+1)*h] into demb.
// demb: [vocab * h].
void embedding_backward(float* demb, const float* dout,
                        size_t token_id, size_t h);

// CrossEntropy+softmax backward. z: [vocab], target: token index, mask: 0 or 1.
//   p = softmax(z); L = -log(p[target])
//   dz[i] = (p[i] - (i==target)) * mask / active_count
// Accumulates into dz.
void cross_entropy_backward(float* dz, const float* z, size_t target,
                             float mask, float active_count, size_t vocab);

// CrossEntropy+softmax forward. Returns -log(p[target]). mask=0 -> loss=0.
float cross_entropy_forward(const float* z, size_t target, float mask,
                             size_t vocab);

}  // namespace oxidize
