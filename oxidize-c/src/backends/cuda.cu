/*
 * cuda.cu — CUDA kernels for GPU-accelerated LLM inference.
 *
 * Kernels:
 *   - dequantize_upload: dequantize + upload weight rows to device f32
 *   - matvec_f32: matrix-vector multiply (the main bottleneck)
 *   - rms_norm: RMSNorm with optional Gemma scaling
 *   - apply_rope: rotary positional embedding (split-halves)
 *   - swiglu: SwiGLU / GeGLU activation
 *   - attention_head: per-head attention with online softmax
 *   - embed_lookup: token embedding lookup
 *
 * Build: make cuda (uses nvcc, defines OC_CUDA)
 *
 * The CUDA forward path replaces the CPU forward loop in oc_llama_forward
 * when OC_CUDA is defined and a GPU is available.
 */
#include "oxidize/cuda.h"
#include "oxidize/activation.h"
#include "oxidize/llama.h"

#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

/* ─── CUDA error checking ──────────────────────────────────────────────── */

#define OC_CUDA_CHECK(call, label) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            return OC_ERR_UNSUPPORTED; \
        } \
    } while (0)

/* ─── CUDA kernels ──────────────────────────────────────────────────────── */

/* Embedding lookup: copy row `token` from [vocab, n_embd] embedding matrix. */
__global__ void k_embed_lookup(const float *embeddings, uint32_t token,
                                size_t n_embd, float *out)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_embd) {
        out[i] = embeddings[(size_t)token * n_embd + i];
    }
}

/* RMSNorm: out[i] = x[i] * inv_rms * weight[i] * norm_scale. */
__global__ void k_rms_norm(const float *x, const float *weight, float *out,
                           size_t n, float eps, float norm_scale)
{
    /* Shared memory for reduction. */
    extern __shared__ float sdata[];
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x + tid;
    size_t blockSize = blockDim.x;

    float val = 0.0f;
    if (i < n) val = x[i] * x[i];
    sdata[tid] = val;
    __syncthreads();

    /* Tree reduction. */
    for (size_t s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float inv_rms = 1.0f / sqrtf(sdata[0] / (float)n + eps);
    if (i < n) out[i] = x[i] * inv_rms * weight[i] * norm_scale;
}

/* Matvec: out[row] = sum(W[row, col] * x[col]) for col in [0, cols).
 * One block per row; each thread accumulates a partial sum. */
__global__ void k_matvec_f32(const float *W, size_t rows, size_t cols,
                              const float *x, float *out)
{
    size_t row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float sdata[];
    size_t tid = threadIdx.x;
    size_t blockSize = blockDim.x;

    float sum = 0.0f;
    for (size_t col = tid; col < cols; col += blockSize) {
        sum += W[row * cols + col] * x[col];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (size_t s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[row] = sdata[0];
}

/* RoPE (split-halves NeoX-style). One block per head. */
__global__ void k_apply_rope(float *q, size_t head_dim, size_t rope_dim,
                              int64_t pos, float theta, uint32_t n_heads)
{
    size_t head = blockIdx.x;
    if (head >= n_heads) return;

    float *hq = q + head * head_dim;
    size_t half = rope_dim / 2;
    float freq_mul = powf(theta, -2.0f / (float)rope_dim);
    float freq = 1.0f;

    for (size_t i = threadIdx.x; i < half; i += blockDim.x) {
        float angle = (float)pos * freq;
        float c = cosf(angle);
        float s = sinf(angle);
        float x0 = hq[i];
        float x1 = hq[half + i];
        hq[i] = x0 * c - x1 * s;
        hq[half + i] = x0 * s + x1 * c;
        freq *= freq_mul;
    }
}

/* SwiGLU in-place: gate[i] = silu(gate[i]) * up[i]. */
__global__ void k_swiglu(float *gate, const float *up, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float sig = 1.0f / (1.0f + expf(-g));
        gate[i] = g * sig * up[i];
    }
}

/* GeGLU in-place: gate[i] = gelu(gate[i]) * up[i]. */
__global__ void k_geglu(float *gate, const float *up, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float gelu = 0.5f * g * (1.0f + erff(g / 1.41421356f));
        gate[i] = gelu * up[i];
    }
}

/* Attention per head: online softmax over past KV cache.
 * One block per head, threads cooperate on reduction. */
__global__ void k_attention_head(
    const float *q,           /* [head_dim]                             */
    const float *kv_k,        /* [n_ctx, n_head_kv, head_dim]            */
    const float *kv_v,        /* [n_ctx, n_head_kv, head_dim]            */
    float *out,               /* [head_dim]                              */
    uint32_t kv_head,
    uint32_t n_head_kv,
    uint32_t head_dim,
    size_t n_past)
{
    size_t head = blockIdx.x;
    (void)head;

    float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);

    /* Thread-local max for online softmax. */
    float max_score = -INFINITY;

    /* First pass: find max. */
    for (size_t p = 0; p < n_past; p++) {
        const float *k = kv_k + p * n_head_kv * head_dim + kv_head * head_dim;
        float score = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++)
            score += q[d] * k[d];
        score *= inv_sqrt_d;
        if (score > max_score) max_score = score;
    }

    /* Second pass: compute weighted sum. */
    float sum_exp = 0.0f;
    /* Use shared memory for output accumulation. */
    extern __shared__ float sdata[];
    for (uint32_t d = 0; d < head_dim; d++) sdata[d] = 0.0f;
    __syncthreads();

    for (size_t p = 0; p < n_past; p++) {
        const float *k = kv_k + p * n_head_kv * head_dim + kv_head * head_dim;
        const float *v = kv_v + p * n_head_kv * head_dim + kv_head * head_dim;
        float score = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++)
            score += q[d] * k[d];
        score *= inv_sqrt_d;
        float weight = expf(score - max_score);
        sum_exp += weight;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x)
            sdata[d] += weight * v[d];
        __syncthreads();
    }

    /* Normalize and write output. */
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x)
        out[d] = sdata[d] * inv_sum;
}

/* Copy KV to cache: write K/V at position `pos`. */
__global__ void k_kv_cache_write(float *kv_k, float *kv_v,
                                  const float *k, const float *v,
                                  size_t pos, size_t n_head_kv,
                                  uint32_t head_dim, size_t n_ctx)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)n_head_kv * head_dim;
    if (i < total) {
        size_t offset = pos * n_head_kv * head_dim + i;
        kv_k[offset] = k[i];
        kv_v[offset] = v[i];
    }
}

/* Residual add: x[i] += y[i]. */
__global__ void k_residual_add(float *x, const float *y, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

/* Scale: x[i] *= scale. */
__global__ void k_scale(float *x, float scale, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= scale;
}

/* ─── Host-side helper: upload a dequantized weight row ─────────────────── */

static OcError upload_weight_view(const OcWeightView *view, float **d_out,
                                  float *host_temp)
{
    size_t rows = view->rows;
    size_t cols = view->cols;
    size_t total = rows * cols;

    OC_CUDA_CHECK(cudaMalloc((void **)d_out, total * sizeof(float)), err);

    for (size_t r = 0; r < rows; r++) {
        if (view->qtype == OC_QUANT_F32) {
            const float *src = (const float *)view->data + r * view->cols;
            cudaMemcpyAsync(*d_out + r * cols, src, cols * sizeof(float),
                           cudaMemcpyHostToDevice, 0);
        } else {
            /* Dequantize on host then upload. */
            oc_quant_dequant_row(view->qtype,
                view->data + r * view->row_bytes,
                view->row_bytes, host_temp, cols);
            cudaMemcpyAsync(*d_out + r * cols, host_temp, cols * sizeof(float),
                           cudaMemcpyHostToDevice, 0);
        }
    }
    cudaDeviceSynchronize();
    return OC_OK;
err:
    return OC_ERR_UNSUPPORTED;
}

/* ─── Public API ────────────────────────────────────────────────────────── */

bool oc_cuda_available(void)
{
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    return (e == cudaSuccess && n > 0);
}

OcError oc_cuda_init(OcCudaContext *ctx, const OcLlamaModel *model)
{
    if (!ctx || !model) return OC_ERR_INVALID_ARG;
    memset(ctx, 0, sizeof(*ctx));

    if (!oc_cuda_available()) return OC_ERR_UNSUPPORTED;

    const OcLlamaConfig *c = &model->cfg;
    ctx->n_embd = c->n_embd;
    ctx->n_head = c->n_head;
    ctx->n_head_kv = c->n_head_kv;
    ctx->n_ff = c->n_ff;
    ctx->head_dim = c->head_dim;
    ctx->n_layer = c->n_layer;
    ctx->vocab_size = c->vocab_size;
    ctx->n_ctx = c->n_ctx;

    /* Allocate workspace. */
    size_t embd = c->n_embd;
    size_t q_size = (size_t)c->n_head * c->head_dim;
    size_t kv_row = (size_t)c->n_head_kv * c->head_dim;
    size_t kv_total = (size_t)c->n_layer * c->n_ctx * kv_row;

    OC_CUDA_CHECK(cudaMalloc(&ctx->d_x, embd * sizeof(float)), err);
    OC_CUDA_CHECK(cudaMalloc(&ctx->d_normed, embd * sizeof(float)), err);
    OC_CUDA_CHECK(cudaMalloc(&ctx->d_q, q_size * sizeof(float)), err);
    OC_CUDA_CHECK(cudaMalloc(&ctx->d_k, kv_row * sizeof(float)), err);
    OC_CUDA_CHECK(cudaMalloc(&ctx->d_v, kv_row * sizeof(float)), err);
    OC_CUDA_CHECK(cudaMalloc(&ctx->d_attn_out, q_size * sizeof(float)), err);
    OC_CUDA_CHECK(cudaMalloc(&ctx->d_ffn_gate_buf, c->n_ff * sizeof(float)), err);
    OC_CUDA_CHECK(cudaMalloc(&ctx->d_ffn_up_buf, c->n_ff * sizeof(float)), err);
    OC_CUDA_CHECK(cudaMalloc(&ctx->d_logits, c->vocab_size * sizeof(float)), err);
    OC_CUDA_CHECK(cudaMalloc(&ctx->d_kv_k, kv_total * sizeof(float)), err);
    OC_CUDA_CHECK(cudaMalloc(&ctx->d_kv_v, kv_total * sizeof(float)), err);

    /* Upload embeddings. */
    float *host_temp = (float *)malloc(embd * sizeof(float));
    if (!host_temp) return OC_ERR_OOM;

    OcError e = upload_weight_view(&model->tok_embeddings, &ctx->d_tok_embeddings,
                                   host_temp);
    if (e != OC_OK) { free(host_temp); return e; }

    /* Upload final norm. */
    OC_CUDA_CHECK(cudaMalloc(&ctx->d_final_norm, embd * sizeof(float)), err);
    cudaMemcpy(ctx->d_final_norm, model->final_norm, embd * sizeof(float),
              cudaMemcpyHostToDevice);

    /* Upload output (or alias tok_embeddings if tied). */
    if (model->cfg.tied_embeddings) {
        ctx->d_output = ctx->d_tok_embeddings;
    } else {
        e = upload_weight_view(&model->output, &ctx->d_output, host_temp);
        if (e != OC_OK) { free(host_temp); return e; }
    }

    /* Upload per-layer weights. */
    ctx->d_attn_q = (float **)calloc(c->n_layer, sizeof(float *));
    ctx->d_attn_k = (float **)calloc(c->n_layer, sizeof(float *));
    ctx->d_attn_v = (float **)calloc(c->n_layer, sizeof(float *));
    ctx->d_attn_output = (float **)calloc(c->n_layer, sizeof(float *));
    ctx->d_ffn_gate = (float **)calloc(c->n_layer, sizeof(float *));
    ctx->d_ffn_up = (float **)calloc(c->n_layer, sizeof(float *));
    ctx->d_ffn_down = (float **)calloc(c->n_layer, sizeof(float *));
    ctx->d_attn_norm = (float **)calloc(c->n_layer, sizeof(float *));
    ctx->d_ffn_norm = (float **)calloc(c->n_layer, sizeof(float *));

    for (uint32_t l = 0; l < c->n_layer; l++) {
        const OcLlamaLayer *L = &model->layers[l];
        upload_weight_view(&L->attn_q, &ctx->d_attn_q[l], host_temp);
        upload_weight_view(&L->attn_k, &ctx->d_attn_k[l], host_temp);
        upload_weight_view(&L->attn_v, &ctx->d_attn_v[l], host_temp);
        upload_weight_view(&L->attn_output, &ctx->d_attn_output[l], host_temp);
        upload_weight_view(&L->ffn_gate, &ctx->d_ffn_gate[l], host_temp);
        upload_weight_view(&L->ffn_up, &ctx->d_ffn_up[l], host_temp);
        upload_weight_view(&L->ffn_down, &ctx->d_ffn_down[l], host_temp);
        OC_CUDA_CHECK(cudaMalloc(&ctx->d_attn_norm[l], embd * sizeof(float)), err);
        cudaMemcpy(ctx->d_attn_norm[l], L->attn_norm, embd * sizeof(float),
                  cudaMemcpyHostToDevice);
        OC_CUDA_CHECK(cudaMalloc(&ctx->d_ffn_norm[l], embd * sizeof(float)), err);
        cudaMemcpy(ctx->d_ffn_norm[l], L->ffn_norm, embd * sizeof(float),
                  cudaMemcpyHostToDevice);
    }

    free(host_temp);
    ctx->initialized = true;
    return OC_OK;

err:
    free(host_temp);
    oc_cuda_free(ctx);
    return OC_ERR_UNSUPPORTED;
}

OcError oc_cuda_forward(OcCudaContext *ctx, uint32_t token, uint32_t pos,
                        float *logits_out)
{
    if (!ctx || !ctx->initialized) return OC_ERR_INVALID_ARG;

    const uint32_t embd = ctx->n_embd;
    const uint32_t head_dim = ctx->head_dim;
    const uint32_t n_head = ctx->n_head;
    const uint32_t n_head_kv = ctx->n_head_kv;
    const uint32_t n_ff = ctx->n_ff;
    const uint32_t group = n_head / n_head_kv;
    const float eps = 1e-5f; /* TODO: get from config */

    int block = 256;

    /* 1. Embedding lookup. */
    k_embed_lookup<<<(embd + block - 1) / block, block>>>(
        ctx->d_tok_embeddings, token, embd, ctx->d_x);

    /* 2. Per-layer forward. */
    for (uint32_t l = 0; l < ctx->n_layer; l++) {
        /* Pre-attention RMSNorm. */
        k_rms_norm<<<1, block, block * sizeof(float)>>>(
            ctx->d_x, ctx->d_attn_norm[l], ctx->d_normed, embd, eps, 1.0f);

        /* Q/K/V projections. */
        k_matvec_f32<<<ctx->n_head * head_dim, block, block * sizeof(float)>>>(
            ctx->d_attn_q[l], (size_t)n_head * head_dim, embd,
            ctx->d_normed, ctx->d_q);
        k_matvec_f32<<<n_head_kv * head_dim, block, block * sizeof(float)>>>(
            ctx->d_attn_k[l], (size_t)n_head_kv * head_dim, embd,
            ctx->d_normed, ctx->d_k);
        k_matvec_f32<<<n_head_kv * head_dim, block, block * sizeof(float)>>>(
            ctx->d_attn_v[l], (size_t)n_head_kv * head_dim, embd,
            ctx->d_normed, ctx->d_v);

        /* RoPE on Q. */
        k_apply_rope<<<n_head, block>>>(
            ctx->d_q, head_dim, head_dim, pos, 10000.0f, n_head);
        /* RoPE on K. */
        k_apply_rope<<<n_head_kv, block>>>(
            ctx->d_k, head_dim, head_dim, pos, 10000.0f, n_head_kv);

        /* KV cache write. */
        size_t kv_total = (size_t)n_head_kv * head_dim;
        k_kv_cache_write<<<(kv_total + block - 1) / block, block>>>(
            ctx->d_kv_k + (size_t)l * ctx->n_ctx * kv_total,
            ctx->d_kv_v + (size_t)l * ctx->n_ctx * kv_total,
            ctx->d_k, ctx->d_v, pos, n_head_kv, head_dim, ctx->n_ctx);

        /* Attention per head. */
        for (uint32_t h = 0; h < n_head; h++) {
            uint32_t kv_head = h / group;
            k_attention_head<<<1, block, head_dim * sizeof(float)>>>(
                ctx->d_q + h * head_dim,
                ctx->d_kv_k + (size_t)l * ctx->n_ctx * kv_total,
                ctx->d_kv_v + (size_t)l * ctx->n_ctx * kv_total,
                ctx->d_attn_out + h * head_dim,
                kv_head, n_head_kv, head_dim, pos + 1);
        }

        /* Output projection. */
        k_matvec_f32<<<embd, block, block * sizeof(float)>>>(
            ctx->d_attn_output[l], embd, (size_t)n_head * head_dim,
            ctx->d_attn_out, ctx->d_normed);
        k_residual_add<<<(embd + block - 1) / block, block>>>(
            ctx->d_x, ctx->d_normed, embd);

        /* Pre-FFN RMSNorm. */
        k_rms_norm<<<1, block, block * sizeof(float)>>>(
            ctx->d_x, ctx->d_ffn_norm[l], ctx->d_normed, embd, eps, 1.0f);

        /* FFN: gate, up, SwiGLU, down. */
        k_matvec_f32<<<n_ff, block, block * sizeof(float)>>>(
            ctx->d_ffn_gate[l], n_ff, embd, ctx->d_normed, ctx->d_ffn_gate_buf);
        k_matvec_f32<<<n_ff, block, block * sizeof(float)>>>(
            ctx->d_ffn_up[l], n_ff, embd, ctx->d_normed, ctx->d_ffn_up_buf);
        k_swiglu<<<(n_ff + block - 1) / block, block>>>(
            ctx->d_ffn_gate_buf, ctx->d_ffn_up_buf, n_ff);
        k_matvec_f32<<<embd, block, block * sizeof(float)>>>(
            ctx->d_ffn_down[l], embd, n_ff, ctx->d_ffn_gate_buf, ctx->d_normed);
        k_residual_add<<<(embd + block - 1) / block, block>>>(
            ctx->d_x, ctx->d_normed, embd);
    }

    /* 3. Final norm + lm_head. */
    k_rms_norm<<<1, block, block * sizeof(float)>>>(
        ctx->d_x, ctx->d_final_norm, ctx->d_normed, embd, eps, 1.0f);

    if (logits_out != NULL) {
        k_matvec_f32<<<ctx->vocab_size, block, block * sizeof(float)>>>(
            ctx->d_output, ctx->vocab_size, embd,
            ctx->d_normed, ctx->d_logits);
        cudaMemcpy(logits_out, ctx->d_logits,
                  ctx->vocab_size * sizeof(float),
                  cudaMemcpyDeviceToHost);
    }

    return OC_OK;
}

void oc_cuda_reset(OcCudaContext *ctx)
{
    /* KV cache is overwritten on subsequent forwards; no need to zero. */
    (void)ctx;
}

void oc_cuda_free(OcCudaContext *ctx)
{
    if (!ctx || !ctx->initialized) return;

    cudaFree(ctx->d_tok_embeddings);
    if (!ctx->d_output || ctx->d_output != ctx->d_tok_embeddings)
        cudaFree(ctx->d_output);
    cudaFree(ctx->d_final_norm);
    cudaFree(ctx->d_x); cudaFree(ctx->d_normed);
    cudaFree(ctx->d_q); cudaFree(ctx->d_k); cudaFree(ctx->d_v);
    cudaFree(ctx->d_attn_out);
    cudaFree(ctx->d_ffn_gate_buf); cudaFree(ctx->d_ffn_up_buf);
    cudaFree(ctx->d_logits);
    cudaFree(ctx->d_kv_k); cudaFree(ctx->d_kv_v);

    for (uint32_t l = 0; l < ctx->n_layer; l++) {
        if (ctx->d_attn_q) cudaFree(ctx->d_attn_q[l]);
        if (ctx->d_attn_k) cudaFree(ctx->d_attn_k[l]);
        if (ctx->d_attn_v) cudaFree(ctx->d_attn_v[l]);
        if (ctx->d_attn_output) cudaFree(ctx->d_attn_output[l]);
        if (ctx->d_ffn_gate) cudaFree(ctx->d_ffn_gate[l]);
        if (ctx->d_ffn_up) cudaFree(ctx->d_ffn_up[l]);
        if (ctx->d_ffn_down) cudaFree(ctx->d_ffn_down[l]);
        if (ctx->d_attn_norm) cudaFree(ctx->d_attn_norm[l]);
        if (ctx->d_ffn_norm) cudaFree(ctx->d_ffn_norm[l]);
    }
    free(ctx->d_attn_q); free(ctx->d_attn_k); free(ctx->d_attn_v);
    free(ctx->d_attn_output);
    free(ctx->d_ffn_gate); free(ctx->d_ffn_up); free(ctx->d_ffn_down);
    free(ctx->d_attn_norm); free(ctx->d_ffn_norm);

    ctx->initialized = false;
}
