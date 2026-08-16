/*
 * cuda_selftest.cu — device-side checks for Qwen3.5 kernels and Q4_0 GEMV.
 *
 * Invoked by `oxidize-c --cuda-selftest`. No GGUF required; compares GPU
 * results against the CPU reference (activation.c / qwen35_delta.c / quant).
 */
#include "oxidize/activation.h"
#include "oxidize/cuda.h"
#include "oxidize/cuda_kernels.h"
#include "oxidize/cuda_mmq.h"
#include "oxidize/quant.h"
#include "oxidize/qwen35_delta.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ST_TOL 2e-4f

static bool st_close(const float *a, const float *b, size_t n, float tol)
{
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        float s = fabsf(a[i]) + fabsf(b[i]) + 1.0f;
        if (d > tol * s) return false;
    }
    return true;
}

static OcError st_fail(const char *msg)
{
    fprintf(stderr, "cuda-selftest: FAIL %s\n", msg);
    return OC_ERR_BACKEND;
}

OcError oc_cuda_selftest(void)
{
    if (!oc_cuda_available()) return OC_ERR_BACKEND;

    /* ── unpack Q/gate ──────────────────────────────────────────────── */
    {
        const uint32_t n_heads = 24, head_dim = 256;
        const size_t n = (size_t)n_heads * head_dim;
        float *h_pack = (float *)malloc(2u * n * sizeof(float));
        float *h_q = (float *)malloc(n * sizeof(float));
        float *h_g = (float *)malloc(n * sizeof(float));
        float *want_q = (float *)malloc(n * sizeof(float));
        float *want_g = (float *)malloc(n * sizeof(float));
        if (!h_pack || !h_q || !h_g || !want_q || !want_g) return OC_ERR_OOM;
        for (size_t i = 0; i < 2u * n; i++)
            h_pack[i] = 0.01f * (float)((int)(i % 97) - 48);
        for (uint32_t h = 0; h < n_heads; h++) {
            const float *src = h_pack + (size_t)h * 2u * head_dim;
            memcpy(want_q + (size_t)h * head_dim, src, head_dim * sizeof(float));
            memcpy(want_g + (size_t)h * head_dim, src + head_dim,
                   head_dim * sizeof(float));
        }
        float *d_pack = NULL, *d_q = NULL, *d_g = NULL;
        cudaMalloc(&d_pack, 2u * n * sizeof(float));
        cudaMalloc(&d_q, n * sizeof(float));
        cudaMalloc(&d_g, n * sizeof(float));
        cudaMemcpy(d_pack, h_pack, 2u * n * sizeof(float), cudaMemcpyHostToDevice);
        if (!oc_cuda_qwen35_unpack_qgate(d_pack, d_q, d_g, n_heads, head_dim,
                                         NULL)) {
            free(h_pack); free(h_q); free(h_g); free(want_q); free(want_g);
            cudaFree(d_pack); cudaFree(d_q); cudaFree(d_g);
            return st_fail("unpack_qgate launch");
        }
        cudaMemcpy(h_q, d_q, n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_g, d_g, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = st_close(h_q, want_q, n, 0.0f) && st_close(h_g, want_g, n, 0.0f);
        free(h_pack); free(h_q); free(h_g); free(want_q); free(want_g);
        cudaFree(d_pack); cudaFree(d_q); cudaFree(d_g);
        if (!ok) return st_fail("unpack_qgate");
        fprintf(stderr, "cuda-selftest: unpack_qgate 24x256 OK\n");
    }

    /* ── QK-norm + partial RoPE (Qwen3.8-27B: head 256, rope 64) ───── */
    {
        const uint32_t n_heads = 4, head_dim = 256, rope_dim = 64;
        const size_t n = (size_t)n_heads * head_dim;
        float *h = (float *)malloc(n * sizeof(float));
        float *w = (float *)malloc(head_dim * sizeof(float));
        float *want = (float *)malloc(n * sizeof(float));
        if (!h || !w || !want) return OC_ERR_OOM;
        for (size_t i = 0; i < n; i++)
            h[i] = 0.07f * (float)((int)(i % 61) - 30);
        for (uint32_t i = 0; i < head_dim; i++)
            w[i] = 0.85f + 0.001f * (float)i;
        memcpy(want, h, n * sizeof(float));
        for (uint32_t hd = 0; hd < n_heads; hd++) {
            float *row = want + (size_t)hd * head_dim;
            oc_rms_norm_f32(row, w, row, head_dim, 1e-6f);
            oc_apply_rope_f32(row, row, head_dim, rope_dim, 11, 1e7f);
        }
        float *d_x = NULL, *d_w = NULL;
        cudaMalloc(&d_x, n * sizeof(float));
        cudaMalloc(&d_w, head_dim * sizeof(float));
        cudaMemcpy(d_x, h, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w, w, head_dim * sizeof(float), cudaMemcpyHostToDevice);
        if (!oc_cuda_qk_norm_rope(d_x, d_w, n_heads, head_dim, rope_dim, 11,
                                  1e7f, 1e-6f, 0.0f, 0u, NULL)) {
            free(h); free(w); free(want);
            cudaFree(d_x); cudaFree(d_w);
            return st_fail("qk_norm_rope launch");
        }
        cudaMemcpy(h, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = st_close(h, want, n, ST_TOL);
        free(h); free(w); free(want);
        cudaFree(d_x); cudaFree(d_w);
        if (!ok) return st_fail("qk_norm_rope");
        fprintf(stderr, "cuda-selftest: qk_norm_rope hd=256 rope=64 OK\n");
    }

    /* ── Gated DeltaNet vs CPU ──────────────────────────────────────── */
    {
        enum { NK = 2, NV = 2, DK = 8, DV = 8, KW = 4 };
        const size_t key_dim = (size_t)NK * DK;
        const size_t conv_dim = 2u * key_dim + (size_t)NV * DV;
        const size_t conv_state_len = conv_dim * (KW - 1u);
        const size_t rec_len = (size_t)NV * DV * DK;
        float *conv_state = (float *)calloc(conv_state_len, sizeof(float));
        float *recurrent = (float *)calloc(rec_len, sizeof(float));
        float *conv_w = (float *)malloc(conv_dim * KW * sizeof(float));
        float *qkv = (float *)malloc(conv_dim * sizeof(float));
        float *conv_out = (float *)malloc(conv_dim * sizeof(float));
        float ssm_a[NV], dt_bias[NV], norm_w[DV];
        float gate[NV * DV], beta[NV], alpha[NV];
        float cpu_out[NV * DV], gpu_out[NV * DV];
        if (!conv_state || !recurrent || !conv_w || !qkv || !conv_out)
            return OC_ERR_OOM;
        for (size_t i = 0; i < conv_dim * KW; i++)
            conv_w[i] = 0.04f * (float)((int)i % 9 - 4);
        for (int i = 0; i < NV; i++) {
            ssm_a[i] = -0.15f - 0.02f * (float)i;
            dt_bias[i] = 0.08f;
            beta[i] = 0.3f + 0.1f * (float)i;
            alpha[i] = -0.25f;
        }
        for (int i = 0; i < DV; i++) norm_w[i] = 0.9f + 0.01f * (float)i;
        for (size_t i = 0; i < conv_dim; i++)
            qkv[i] = 0.06f * (float)((int)i - 9);
        for (int i = 0; i < NV * DV; i++) gate[i] = 0.12f * (float)(i - 3);

        OcQwen35DeltaGeometry geometry = {NK, NV, DK, DV, KW};
        OcQwen35DeltaState state;
        if (oc_qwen35_delta_state_init(&state, &geometry, conv_state,
                                       conv_state_len, recurrent, rec_len)
            != OC_OK) {
            return st_fail("cpu delta init");
        }
        OcQwen35DeltaParams params = {
            conv_w, conv_dim * KW, ssm_a, NV, dt_bias, NV, norm_w, DV, 1e-6f,
        };
        OcQwen35DeltaInput input = { qkv, conv_dim, gate, NV * DV, beta, NV,
                                     alpha, NV };
        if (oc_qwen35_delta_step(&state, &params, &input, conv_out, conv_dim,
                                 cpu_out, NV * DV) != OC_OK)
            return st_fail("cpu delta step");

        float *d_cs = NULL, *d_rec = NULL, *d_qkv = NULL, *d_gate = NULL;
        float *d_beta = NULL, *d_alpha = NULL, *d_cw = NULL, *d_a = NULL;
        float *d_dt = NULL, *d_nw = NULL, *d_co = NULL, *d_out = NULL;
        cudaMalloc(&d_cs, conv_state_len * sizeof(float));
        cudaMalloc(&d_rec, rec_len * sizeof(float));
        cudaMalloc(&d_qkv, conv_dim * sizeof(float));
        cudaMalloc(&d_gate, (size_t)NV * DV * sizeof(float));
        cudaMalloc(&d_beta, NV * sizeof(float));
        cudaMalloc(&d_alpha, NV * sizeof(float));
        cudaMalloc(&d_cw, conv_dim * KW * sizeof(float));
        cudaMalloc(&d_a, NV * sizeof(float));
        cudaMalloc(&d_dt, NV * sizeof(float));
        cudaMalloc(&d_nw, DV * sizeof(float));
        cudaMalloc(&d_co, conv_dim * sizeof(float));
        cudaMalloc(&d_out, (size_t)NV * DV * sizeof(float));
        cudaMemset(d_cs, 0, conv_state_len * sizeof(float));
        cudaMemset(d_rec, 0, rec_len * sizeof(float));
        cudaMemcpy(d_qkv, qkv, conv_dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gate, gate, (size_t)NV * DV * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta, beta, NV * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_alpha, alpha, NV * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cw, conv_w, conv_dim * KW * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_a, ssm_a, NV * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dt, dt_bias, NV * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nw, norm_w, DV * sizeof(float), cudaMemcpyHostToDevice);
        if (!oc_cuda_qwen35_delta_step(d_cs, d_rec, d_qkv, d_gate, d_beta,
                                       d_alpha, d_cw, d_a, d_dt, d_nw, d_co,
                                       d_out, NK, NV, DK, DV, KW, 1e-6f,
                                       NULL)) {
            return st_fail("delta launch");
        }
        cudaMemcpy(gpu_out, d_out, (size_t)NV * DV * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaFree(d_cs); cudaFree(d_rec); cudaFree(d_qkv); cudaFree(d_gate);
        cudaFree(d_beta); cudaFree(d_alpha); cudaFree(d_cw); cudaFree(d_a);
        cudaFree(d_dt); cudaFree(d_nw); cudaFree(d_co); cudaFree(d_out);
        free(conv_state); free(recurrent); free(conv_w); free(qkv); free(conv_out);
        if (!st_close(gpu_out, cpu_out, NV * DV, 5e-4f))
            return st_fail("delta_step vs CPU");
        fprintf(stderr, "cuda-selftest: gated delta vs CPU OK\n");
    }

    /* ── Q4_0 packed GEMV (A10G path for Qwen3.8-27B Q4_0) ─────────── */
    {
        const size_t rows = 32, cols = 256;
        float *src = (float *)malloc(rows * cols * sizeof(float));
        float *x = (float *)malloc(cols * sizeof(float));
        float *want = (float *)malloc(rows * sizeof(float));
        float *got = (float *)malloc(rows * sizeof(float));
        if (!src || !x || !want || !got) return OC_ERR_OOM;
        for (size_t i = 0; i < rows * cols; i++)
            src[i] = 0.02f * (float)((int)(i % 19) - 9);
        for (size_t i = 0; i < cols; i++)
            x[i] = 0.03f * (float)((int)(i % 11) - 5);

        const size_t row_bytes = oc_quantized_size(OC_QUANT_Q4_0, cols);
        uint8_t *packed = (uint8_t *)malloc(rows * row_bytes);
        float *deq = (float *)malloc(cols * sizeof(float));
        if (!packed || !deq) return OC_ERR_OOM;
        for (size_t r = 0; r < rows; r++) {
            if (oc_quant_pack_row(OC_QUANT_Q4_0, src + r * cols, cols,
                                  packed + r * row_bytes, row_bytes) != OC_OK)
                return st_fail("q4_0 pack");
            if (oc_quant_dequant_row_scalar(OC_QUANT_Q4_0,
                                            packed + r * row_bytes, row_bytes,
                                            deq, cols) != OC_OK)
                return st_fail("q4_0 dequant");
            float acc = 0.0f;
            for (size_t c = 0; c < cols; c++) acc += deq[c] * x[c];
            want[r] = acc;
        }
        uint8_t *d_w = NULL;
        float *d_x = NULL, *d_out = NULL;
        cudaMalloc(&d_w, rows * row_bytes);
        cudaMalloc(&d_x, cols * sizeof(float));
        cudaMalloc(&d_out, rows * sizeof(float));
        cudaMemcpy(d_w, packed, rows * row_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x, cols * sizeof(float), cudaMemcpyHostToDevice);
        if (!oc_cuda_mmq_matvec(OC_QUANT_Q4_0, d_w, d_x, d_out, rows, cols,
                                NULL)) {
            return st_fail("q4_0 matvec launch");
        }
        cudaMemcpy(got, d_out, rows * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = st_close(got, want, rows, 5e-4f);
        cudaFree(d_w); cudaFree(d_x); cudaFree(d_out);
        free(src); free(x); free(want); free(got); free(packed); free(deq);
        if (!ok) return st_fail("q4_0 matvec vs dequant");
        fprintf(stderr, "cuda-selftest: Q4_0 GEMV OK\n");
    }

    /* ── Q4_K packed GEMV (A10G path for Qwen3.5-27B Q4_K_M) ───────── */
    {
        const size_t rows = 32, cols = 256;
        float *src = (float *)malloc(rows * cols * sizeof(float));
        float *x = (float *)malloc(cols * sizeof(float));
        float *want = (float *)malloc(rows * sizeof(float));
        float *got = (float *)malloc(rows * sizeof(float));
        if (!src || !x || !want || !got) return OC_ERR_OOM;
        for (size_t i = 0; i < rows * cols; i++)
            src[i] = 0.02f * (float)((int)(i % 19) - 9);
        for (size_t i = 0; i < cols; i++)
            x[i] = 0.03f * (float)((int)(i % 11) - 5);

        const size_t row_bytes = oc_quantized_size(OC_QUANT_Q4_K_M, cols);
        uint8_t *packed = (uint8_t *)malloc(rows * row_bytes);
        float *deq = (float *)malloc(cols * sizeof(float));
        if (!packed || !deq) return OC_ERR_OOM;
        for (size_t r = 0; r < rows; r++) {
            if (oc_quant_pack_row(OC_QUANT_Q4_K_M, src + r * cols, cols,
                                  packed + r * row_bytes, row_bytes) != OC_OK)
                return st_fail("q4_k pack");
            if (oc_quant_dequant_row_scalar(OC_QUANT_Q4_K_M,
                                            packed + r * row_bytes, row_bytes,
                                            deq, cols) != OC_OK)
                return st_fail("q4_k dequant");
            float acc = 0.0f;
            for (size_t c = 0; c < cols; c++) acc += deq[c] * x[c];
            want[r] = acc;
        }
        uint8_t *d_w = NULL;
        float *d_x = NULL, *d_out = NULL;
        cudaMalloc(&d_w, rows * row_bytes);
        cudaMalloc(&d_x, cols * sizeof(float));
        cudaMalloc(&d_out, rows * sizeof(float));
        cudaMemcpy(d_w, packed, rows * row_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x, cols * sizeof(float), cudaMemcpyHostToDevice);
        if (!oc_cuda_mmq_matvec(OC_QUANT_Q4_K_M, d_w, d_x, d_out, rows, cols,
                                NULL)) {
            return st_fail("q4_k matvec launch");
        }
        cudaMemcpy(got, d_out, rows * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = st_close(got, want, rows, 5e-4f);
        cudaFree(d_w); cudaFree(d_x); cudaFree(d_out);
        free(src); free(x); free(want); free(got); free(packed); free(deq);
        if (!ok) return st_fail("q4_k matvec vs dequant");
        fprintf(stderr, "cuda-selftest: Q4_K GEMV OK\n");
    }

    fprintf(stderr, "cuda-selftest: all checks passed\n");
    return OC_OK;
}
