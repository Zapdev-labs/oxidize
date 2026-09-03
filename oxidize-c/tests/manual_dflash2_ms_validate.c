/*
 * manual_dflash2_ms_validate.c — multi-step propose validation.
 *
 * Drives 3 consecutive propose steps with the same KV-ring contract as
 * dflash_generate: each step sets a fresh target context for positions
 * [start - n_ctx, start), runs a block forward from the carried ring,
 * proposes, and advances next_noise_pos by the fixed accepted count.
 * Compares per-step token paths + candidate sets against the PyTorch
 * reference (dflash2_ms_ref.py gold in /tmp/dflash2_ms_*.npy).
 *
 * Build: cc -Iinclude -std=c11 -O2 -o /tmp/dflash2_ms_validate \
 *          tests/manual_dflash2_ms_validate.c src/model/dflash2.o \
 *          src/core/simd.o src/core/simd_avx2.o src/core/simd_avx512.o \
 *          src/compute/parallel.o src/core/error.o src/core/log.o \
 *          src/format/safetensors.o src/util/bytes.o src/util/mmap.o \
 *          src/util/file.o src/core/dtype.o -lm -lpthread
 */
#include "oxidize/dflash2.h"
#include "oxidize/parallel.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Minimal .npy (v1/v2) reader for float32 / int64 arrays. */
static void *npy_load(const char *path, char *dtype, size_t *n_elems,
                      int *fortran_order)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char magic[6];
    if (fread(magic, 1, 6, f) != 6) { fclose(f); return NULL; }
    if (memcmp(magic, "\x93NUMPY", 6) != 0) { fclose(f); return NULL; }
    uint8_t ver[2];
    if (fread(ver, 1, 2, f) != 2) { fclose(f); return NULL; }
    uint32_t hlen;
    if (ver[0] == 1) {
        uint8_t hb[2];
        if (fread(hb, 1, 2, f) != 2) { fclose(f); return NULL; }
        hlen = (uint16_t)(hb[0] | (hb[1] << 8));
    } else {
        uint8_t hb[4];
        if (fread(hb, 1, 4, f) != 4) { fclose(f); return NULL; }
        hlen = (uint32_t)(hb[0] | (hb[1] << 8) | (hb[2] << 16) | (hb[3] << 24));
    }
    char *hdr = malloc((size_t)hlen + 1);
    if (!hdr) { fclose(f); return NULL; }
    if (fread(hdr, 1, hlen, f) != hlen) { free(hdr); fclose(f); return NULL; }
    hdr[hlen] = '\0';

    char *dd = strstr(hdr, "'descr'");
    char *fo = strstr(hdr, "'fortran_order'");
    char *sh = strstr(hdr, "'shape'");
    if (!dd || !fo || !sh) { free(hdr); fclose(f); return NULL; }
    *fortran_order = strstr(fo, "True") != NULL;

    const char *val = strchr(dd + 7, '\'');
    if (!val) { free(hdr); fclose(f); return NULL; }
    val++;
    const char *vend = strchr(val, '\'');
    if (!vend) { free(hdr); fclose(f); return NULL; }
    size_t dlen = (size_t)(vend - val);
    if (dlen > 7) dlen = 7;
    memcpy(dtype, val, dlen);
    dtype[dlen] = '\0';

    size_t shape[8], nd = 0;
    char *p = strchr(sh, '(');
    if (!p) { free(hdr); fclose(f); return NULL; }
    p++;
    while (*p && *p != ')') {
        while (*p == ' ' || *p == ',') p++;
        if (*p == ')') break;
        if (nd >= 8) break;
        shape[nd++] = (size_t)strtol(p, &p, 10);
    }
    size_t total = 1;
    for (size_t i = 0; i < nd; i++) {
        /* Guard the shape product: a crafted header could otherwise wrap
         * the element count and under-allocate below. */
        if (shape[i] != 0 && total > SIZE_MAX / shape[i]) {
            free(hdr); fclose(f); return NULL;
        }
        total *= shape[i];
    }
    *n_elems = total;

    size_t elem_sz = strstr(dtype, "f4") ? 4
                        : (strstr(dtype, "i8") ? 8 : 0);
    if (elem_sz == 0) { free(hdr); fclose(f); return NULL; }
    if (total > SIZE_MAX / elem_sz) {
        free(hdr); fclose(f); return NULL;
    }

    void *data = malloc(total * elem_sz);
    if (!data) { free(hdr); fclose(f); return NULL; }
    if (fread(data, elem_sz, total, f) != total) {
        free(data); free(hdr); fclose(f); return NULL;
    }
    free(hdr);
    fclose(f);
    return data;
}

int main(int argc, char **argv)
{
    /* Validation dir keeps argv[1] (the Makefile manual-tests contract);
     * the checkpoint moves off the hard-coded dev-box path: argv[2] or
     * $OXIDIZE_DFLASH2_CKPT, falling back to the local HF cache. */
    const char *val_dir = argc > 1 ? argv[1] : "/tmp";
    const char *ckpt_dir = argc > 2 ? argv[2] : getenv("OXIDIZE_DFLASH2_CKPT");
    if (!ckpt_dir) {
        ckpt_dir = "/home/dih/.cache/huggingface/hub/models--incoai--"
                   "GLM-5.3-Flash-DFlash2/snapshots/"
                   "bf582e4eacc1810f76656d1811693ff6c6737d2a";
        fprintf(stderr, "no checkpoint given (argv[2] or "
                        "OXIDIZE_DFLASH2_CKPT); defaulting to %s\n", ckpt_dir);
    }
    char path[512], dtype[8];
    size_t n;
    (void)n;

    oc_parallel_set_threads(1); /* determinism */

    OcDFlash2Model m;
    snprintf(path, sizeof(path), "%s/model.safetensors", ckpt_dir);
    OcError e = oc_dflash2_model_load(&m, path, NULL);
    if (e != OC_OK) {
        fprintf(stderr, "load failed: %s\n", oc_error_msg(e));
        return 1;
    }
    const size_t H = m.cfg.hidden_size;
    const size_t block = m.cfg.block_size;
    const size_t top_k = m.cfg.selector_top_k;
    const size_t n_draft = block - 1;
    const size_t n_tl = m.cfg.n_target_layer_ids;

    char p2[512];
    snprintf(p2, sizeof(p2), "%s/dflash2_ms_lm.npy", val_dir);
    int fo;
    size_t n_lm;
    float *lm = npy_load(p2, dtype, &n_lm, &fo);
    if (!lm || n_lm != m.cfg.vocab_size * H) {
        fprintf(stderr, "lm load failed\n");
        return 1;
    }
    OcDFlash2Weight lmw = { lm, NULL, m.cfg.vocab_size, H, 0, NULL, NULL };

    int all_ok = 1;
    int64_t start = 4;                      /* matches dflash2_ms_ref.py */
    const int n_ctx_steps[3] = { 4, 3, 3 }; /* prefill then per-verify rows */
    const int accepted_per_step = 3;

    for (int s = 0; s < 3; s++) {
        size_t n_ctx = (size_t)n_ctx_steps[s];
        snprintf(p2, sizeof(p2), "%s/dflash2_ms_noise%d.npy", val_dir, s);
        size_t n_noise;
        float *noise = npy_load(p2, dtype, &n_noise, &fo);
        snprintf(p2, sizeof(p2), "%s/dflash2_ms_ctx%d.npy", val_dir, s);
        size_t n_ctx_elems;
        float *ctx = npy_load(p2, dtype, &n_ctx_elems, &fo);
        snprintf(p2, sizeof(p2), "%s/dflash2_ms_path%d.npy", val_dir, s);
        size_t n_path;
        char dtype_path[8];
        int64_t *gold_path = npy_load(p2, dtype_path, &n_path, &fo);
        snprintf(p2, sizeof(p2), "%s/dflash2_ms_cand%d.npy", val_dir, s);
        size_t n_cand;
        char dtype_cand[8];
        int64_t *gold_cand = npy_load(p2, dtype_cand, &n_cand, &fo);
        if (!noise || !ctx || !gold_path || !gold_cand ||
            n_noise != block * H || n_ctx_elems != n_ctx * n_tl * H ||
            n_path != n_draft || n_cand != n_draft * top_k) {
            fprintf(stderr, "step %d artifact load failed\n", s);
            return 1;
        }
        /* Token-id artifacts must be int64; anything else reads as
         * garbage 8-byte values in the comparisons below. */
        if (strcmp(dtype_path, "<i8") != 0 || strcmp(dtype_cand, "<i8") != 0) {
            fprintf(stderr, "step %d path/cand must be int64 (got %s / %s)\n",
                    s, dtype_path, dtype_cand);
            return 1;
        }

        /* The reference consumes raw target hidden states (pre-fc); our
         * set_context takes the same raw concat. */
        e = oc_dflash2_set_context(&m, ctx, n_ctx);
        if (e != OC_OK) { fprintf(stderr, "set_context failed\n"); return 1; }

        m.next_noise_pos = start;
        uint32_t anchor = 7 + (uint32_t)s;  /* matches the reference */
        uint32_t block_ids[64];
        block_ids[0] = anchor;
        for (size_t i = 1; i < block; i++)
            block_ids[i] = m.cfg.mask_token_id;
        uint32_t *out_tok = malloc(n_draft * sizeof(uint32_t));
        uint32_t *out_cand = malloc(n_draft * top_k * sizeof(uint32_t));
        float *out_probs = malloc(n_draft * top_k * sizeof(float));
        e = oc_dflash2_propose(&m, &anchor, 1, noise, block, block_ids,
                               &lmw, 0.0f, out_tok, out_cand, out_probs);
        if (e != OC_OK) {
            fprintf(stderr, "propose %d failed: %s\n", s, oc_error_msg(e));
            return 1;
        }

        int path_ok = 1;
        for (size_t i = 0; i < n_draft; i++)
            if ((int64_t)out_tok[i] != gold_path[i]) { path_ok = 0; break; }
        int cand_ok = 1;
        for (size_t p = 0; p < n_draft && cand_ok; p++) {
            for (size_t k = 0; k < top_k; k++) {
                int found = 0;
                for (size_t j = 0; j < top_k; j++)
                    if ((uint32_t)gold_cand[p * top_k + j] ==
                        out_cand[p * top_k + k]) { found = 1; break; }
                if (!found) { cand_ok = 0; break; }
            }
        }
        printf("step%d: path %s (ours:", s, path_ok ? "ok" : "MISMATCH");
        for (size_t i = 0; i < n_draft; i++) printf(" %u", out_tok[i]);
        printf(" | gold:");
        for (size_t i = 0; i < n_draft; i++) printf(" %lld", (long long)gold_path[i]);
        printf(") candidates %s\n", cand_ok ? "ok" : "MISMATCH");
        if (!path_ok || !cand_ok) all_ok = 0;

        start += accepted_per_step;

        free(noise); free(ctx); free(gold_path); free(gold_cand);
        free(out_tok); free(out_cand); free(out_probs);
    }

    printf("%s: multi-step ring contract %s\n",
           all_ok ? "PASS" : "FAIL", all_ok ? "holds" : "BROKEN");
    return all_ok ? 0 : 1;
}
