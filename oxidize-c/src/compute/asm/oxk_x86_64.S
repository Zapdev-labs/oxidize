/*
 * oxk_x86_64.S: hand-written x86-64 AVX2/FMA/F16C decode GEMV kernels.
 *
 * These replace the C intrinsics "facc" row kernels in oxk_avx2.c on the
 * single-activation decode path (matvec.c matvec_fused_slice and friends):
 * Q4_0 and Q8_0 (facc contract, below) and the K-quants Q4_K and Q6_K that
 * Q4_K_M files are made of (bit-exact, see the K-quant section).
 *
 * Q4_0 / Q8_0: same contract as the C facc kernels: per-block integer dots are exact, the
 * per-block scale products accumulate in f32 lanes, one horizontal sum per
 * row. Results agree with the scalar reference to f32 reassociation
 * tolerance (the same tolerance the facc C kernels already document).
 *
 * What the hand-written version does differently:
 *   - 4-row tiles. Each activation block (its 32 codes, its f16 scale and
 *     its precomputed sum) is loaded/converted once and reused against four
 *     weight rows, instead of once per row.
 *   - Q4_0's "-8 * sum(a)" correction is subtracted in the int32 lanes
 *     (each of the 8 lanes takes off sum(a), so the lane total drops by
 *     exactly 8*sum(a)). That deletes the scalar float correction chain and
 *     is exact, so it is slightly *more* accurate than the C version.
 *   - The weight scale is broadcast straight from memory into eight f16
 *     lanes and converted with one vcvtph2ps; no GPR round trip.
 *   - One vpand masks both nibble halves after the vinserti128.
 *
 * ABI: System V AMD64. Leaf functions, ymm0-15 are all caller-saved; we
 * push the callee-saved GPRs we use and issue vzeroupper on return.
 *
 * Block layouts (GGUF):
 *   Q4_0: 18 B = f16 d | 16 B nibbles (low nibbles = codes 0..15,
 *         high nibbles = codes 16..31), value = (code - 8) * d
 *   Q8_0: 34 B = f16 d | 32 x int8
 */

#if defined(__x86_64__) && defined(__ELF__)

        .intel_syntax noprefix
        .text

#define Q4_0_BYTES 18
#define Q8_0_BYTES 34
/* Software prefetch runs one 4-row tile ahead, linearly. The four rows of a
 * tile are contiguous (row_bytes apart), so while tile k is being reduced we
 * walk tile k+1 at 4 * (block offset): the whole next tile is requested by
 * the time we reach it, as one sequential stream the hardware prefetcher
 * also likes. A fixed "row + 1 KiB" distance is useless here: decode rows
 * are often under 1 KiB (896-wide Q8_0 = 952 B), so it lands inside the
 * tile already being loaded. PF_TILE_EXTRA adds distance beyond one tile. */
#ifndef PF_TILE_EXTRA
#define PF_TILE_EXTRA 0
#endif

/* ------------------------------------------------------------------------
 * Q4_0 x Q8_0, one weight row, one block.
 *   acc   : f32 accumulator ymm
 *   prow  : GPR holding the row base; r11 = byte offset of this block
 * Shared per-block state (set by Q4_BLOCK_SETUP):
 *   ymm8  = activation codes (32 x s8)
 *   ymm9  = broadcast int32 sum(a) of this block
 *   ymm10 = broadcast f32 activation scale dq
 *   ymm14 = 16-bit ones, ymm15 = 0x0F bytes
 * Clobbers ymm11, ymm12.
 * ---------------------------------------------------------------------- */
.macro Q4_ROW acc, prow
        vmovdqu         xmm11, xmmword ptr [\prow + r11 + 2]
        vpsrlw          xmm12, xmm11, 4
        vinserti128     ymm11, ymm11, xmm12, 1
        vpand           ymm11, ymm11, ymm15
        vpmaddubsw      ymm11, ymm11, ymm8
        vpmaddwd        ymm11, ymm11, ymm14
        vpsubd          ymm11, ymm11, ymm9
        vcvtdq2ps       ymm11, ymm11
        vpbroadcastw    xmm12, word ptr [\prow + r11]
        vcvtph2ps       ymm12, xmm12
        vmulps          ymm12, ymm12, ymm10
        vfmadd231ps     \acc, ymm11, ymm12
.endm

/* r12 -> activation block, r13 -> qsums[b] */
.macro Q4_BLOCK_SETUP
        vmovdqu         ymm8, ymmword ptr [r12 + 2]
        vpbroadcastd    ymm9, dword ptr [r13]
        vpbroadcastw    xmm10, word ptr [r12]
        vcvtph2ps       ymm10, xmm10
.endm

/* ------------------------------------------------------------------------
 * Q8_0 x Q8_0, one weight row, one block.
 * Shared per-block state (Q8_BLOCK_SETUP):
 *   ymm8  = activation codes 0..15 sign-extended to int16
 *   ymm9  = activation codes 16..31 sign-extended to int16
 *   ymm10 = broadcast f32 activation scale dq
 * Widening both operands to i16 keeps every input exact (see oxk_avx2.c:
 * maddubs saturates for some Q8_0 code pairs).
 * Clobbers ymm11, ymm12, ymm13.
 * ---------------------------------------------------------------------- */
.macro Q8_ROW acc, prow
        vpmovsxbw       ymm11, xmmword ptr [\prow + r11 + 2]
        vpmovsxbw       ymm12, xmmword ptr [\prow + r11 + 18]
        vpmaddwd        ymm11, ymm11, ymm8
        vpmaddwd        ymm12, ymm12, ymm9
        vpaddd          ymm11, ymm11, ymm12
        vcvtdq2ps       ymm11, ymm11
        vpbroadcastw    xmm13, word ptr [\prow + r11]
        vcvtph2ps       ymm13, xmm13
        vmulps          ymm13, ymm13, ymm10
        vfmadd231ps     \acc, ymm11, ymm13
.endm

.macro Q8_BLOCK_SETUP
        vpmovsxbw       ymm8, xmmword ptr [r12 + 2]
        vpmovsxbw       ymm9, xmmword ptr [r12 + 18]
        vpbroadcastw    xmm10, word ptr [r12]
        vcvtph2ps       ymm10, xmm10
.endm

/* Reduce ymm0..ymm3 (one accumulator per row) into four floats at [r10]. */
.macro STORE4
        vhaddps         ymm0, ymm0, ymm1
        vhaddps         ymm2, ymm2, ymm3
        vhaddps         ymm0, ymm0, ymm2
        vextractf128    xmm1, ymm0, 1
        vaddps          xmm0, xmm0, xmm1
        vmovups         xmmword ptr [r10], xmm0
.endm

/* Reduce ymm0 into one float at [r10]. Must use the exact summation tree
 * STORE4 applies to each row, ((l0+l1)+(l2+l3)) + ((l4+l5)+(l6+l7)), so a
 * row's result does not depend on whether it landed in a 4-row tile or the
 * tail; that in turn depends on how the matvec split rows across threads. */
.macro STORE1
        vhaddps         ymm0, ymm0, ymm0
        vhaddps         ymm0, ymm0, ymm0
        vextractf128    xmm1, ymm0, 1
        vaddss          xmm0, xmm0, xmm1
        vmovss          dword ptr [r10], xmm0
.endm

/* ========================================================================
 * void oc_asm_dot_rows_q4_0_q8_0_facc(const uint8_t *rows,   rdi
 *                                     size_t row_bytes,      rsi
 *                                     size_t n_rows,         rdx
 *                                     size_t blocks,         rcx
 *                                     const uint8_t *q8,     r8
 *                                     const int32_t *qsums,  r9
 *                                     float *out);           [rsp+8]
 * ====================================================================== */
        .globl  oc_asm_dot_rows_q4_0_q8_0_facc
        .type   oc_asm_dot_rows_q4_0_q8_0_facc, @function
        .p2align 5
oc_asm_dot_rows_q4_0_q8_0_facc:
        .cfi_startproc
        push    rbx
        .cfi_adjust_cfa_offset 8
        push    rbp
        .cfi_adjust_cfa_offset 8
        push    r12
        .cfi_adjust_cfa_offset 8
        push    r13
        .cfi_adjust_cfa_offset 8
        push    r14
        .cfi_adjust_cfa_offset 8
        push    r15
        .cfi_adjust_cfa_offset 8
        mov     r10, qword ptr [rsp + 56]       /* out */

        mov     eax, 0x0F0F0F0F
        vmovd   xmm15, eax
        vpbroadcastd ymm15, xmm15
        mov     eax, 0x00010001
        vmovd   xmm14, eax
        vpbroadcastd ymm14, xmm14

.Lq4_tile:
        cmp     rdx, 4
        jb      .Lq4_tail
        lea     r15, [rdi + rsi]                /* row 1 */
        lea     rbx, [rdi + rsi*2]              /* row 2 */
        lea     rbp, [rbx + rsi]                /* row 3 */
        lea     rax, [rbp + rsi + PF_TILE_EXTRA] /* next tile: prefetch base */
        vxorps  ymm0, ymm0, ymm0
        vxorps  ymm1, ymm1, ymm1
        vxorps  ymm2, ymm2, ymm2
        vxorps  ymm3, ymm3, ymm3
        xor     r11d, r11d
        mov     r12, r8
        mov     r13, r9
        mov     r14, rcx
        test    r14, r14
        jz      .Lq4_tile_store
        .p2align 4
.Lq4_tile_blk:
        prefetcht0 [rax + r11*4]
        prefetcht0 [rax + r11*4 + 64]
        Q4_BLOCK_SETUP
        Q4_ROW  ymm0, rdi
        Q4_ROW  ymm1, r15
        Q4_ROW  ymm2, rbx
        Q4_ROW  ymm3, rbp
        add     r11, Q4_0_BYTES
        add     r12, Q8_0_BYTES
        add     r13, 4
        dec     r14
        jnz     .Lq4_tile_blk
.Lq4_tile_store:
        STORE4
        add     r10, 16
        lea     rdi, [rbp + rsi]
        sub     rdx, 4
        jmp     .Lq4_tile

.Lq4_tail:
        test    rdx, rdx
        jz      .Lq4_done
        lea     rax, [rdi + rsi]
        vxorps  ymm0, ymm0, ymm0
        xor     r11d, r11d
        mov     r12, r8
        mov     r13, r9
        mov     r14, rcx
        test    r14, r14
        jz      .Lq4_tail_store
.Lq4_tail_blk:
        prefetcht0 [rax + r11]                  /* next row, same block */
        Q4_BLOCK_SETUP
        Q4_ROW  ymm0, rdi
        add     r11, Q4_0_BYTES
        add     r12, Q8_0_BYTES
        add     r13, 4
        dec     r14
        jnz     .Lq4_tail_blk
.Lq4_tail_store:
        STORE1
        add     r10, 4
        add     rdi, rsi
        dec     rdx
        jmp     .Lq4_tail

.Lq4_done:
        vzeroupper
        pop     r15
        .cfi_adjust_cfa_offset -8
        pop     r14
        .cfi_adjust_cfa_offset -8
        pop     r13
        .cfi_adjust_cfa_offset -8
        pop     r12
        .cfi_adjust_cfa_offset -8
        pop     rbp
        .cfi_adjust_cfa_offset -8
        pop     rbx
        .cfi_adjust_cfa_offset -8
        ret
        .cfi_endproc
        .size   oc_asm_dot_rows_q4_0_q8_0_facc, .-oc_asm_dot_rows_q4_0_q8_0_facc

/* ========================================================================
 * void oc_asm_dot_rows_q8_0_q8_0_facc(const uint8_t *rows,   rdi
 *                                     size_t row_bytes,      rsi
 *                                     size_t n_rows,         rdx
 *                                     size_t blocks,         rcx
 *                                     const uint8_t *q8,     r8
 *                                     float *out);           r9
 * ====================================================================== */
        .globl  oc_asm_dot_rows_q8_0_q8_0_facc
        .type   oc_asm_dot_rows_q8_0_q8_0_facc, @function
        .p2align 5
oc_asm_dot_rows_q8_0_q8_0_facc:
        .cfi_startproc
        push    rbx
        .cfi_adjust_cfa_offset 8
        push    rbp
        .cfi_adjust_cfa_offset 8
        push    r12
        .cfi_adjust_cfa_offset 8
        push    r14
        .cfi_adjust_cfa_offset 8
        push    r15
        .cfi_adjust_cfa_offset 8
        mov     r10, r9                         /* out */

.Lq8_tile:
        cmp     rdx, 4
        jb      .Lq8_tail
        lea     r15, [rdi + rsi]
        lea     rbx, [rdi + rsi*2]
        lea     rbp, [rbx + rsi]
        lea     rax, [rbp + rsi + PF_TILE_EXTRA] /* next tile: prefetch base */
        vxorps  ymm0, ymm0, ymm0
        vxorps  ymm1, ymm1, ymm1
        vxorps  ymm2, ymm2, ymm2
        vxorps  ymm3, ymm3, ymm3
        xor     r11d, r11d
        mov     r12, r8
        mov     r14, rcx
        test    r14, r14
        jz      .Lq8_tile_store
        .p2align 4
.Lq8_tile_blk:
        prefetcht0 [rax + r11*4]
        prefetcht0 [rax + r11*4 + 64]
        prefetcht0 [rax + r11*4 + 128]
        Q8_BLOCK_SETUP
        Q8_ROW  ymm0, rdi
        Q8_ROW  ymm1, r15
        Q8_ROW  ymm2, rbx
        Q8_ROW  ymm3, rbp
        add     r11, Q8_0_BYTES
        add     r12, Q8_0_BYTES
        dec     r14
        jnz     .Lq8_tile_blk
.Lq8_tile_store:
        STORE4
        add     r10, 16
        lea     rdi, [rbp + rsi]
        sub     rdx, 4
        jmp     .Lq8_tile

.Lq8_tail:
        test    rdx, rdx
        jz      .Lq8_done
        lea     rax, [rdi + rsi]
        vxorps  ymm0, ymm0, ymm0
        xor     r11d, r11d
        mov     r12, r8
        mov     r14, rcx
        test    r14, r14
        jz      .Lq8_tail_store
.Lq8_tail_blk:
        prefetcht0 [rax + r11]                  /* next row, same block */
        Q8_BLOCK_SETUP
        Q8_ROW  ymm0, rdi
        add     r11, Q8_0_BYTES
        add     r12, Q8_0_BYTES
        dec     r14
        jnz     .Lq8_tail_blk
.Lq8_tail_store:
        STORE1
        add     r10, 4
        add     rdi, rsi
        dec     rdx
        jmp     .Lq8_tail

.Lq8_done:
        vzeroupper
        pop     r15
        .cfi_adjust_cfa_offset -8
        pop     r14
        .cfi_adjust_cfa_offset -8
        pop     r12
        .cfi_adjust_cfa_offset -8
        pop     rbp
        .cfi_adjust_cfa_offset -8
        pop     rbx
        .cfi_adjust_cfa_offset -8
        ret
        .cfi_endproc
        .size   oc_asm_dot_rows_q8_0_q8_0_facc, .-oc_asm_dot_rows_q8_0_q8_0_facc

/* ========================================================================
 * K-quant x Q8_K row kernels (Q4_K, Q6_K).
 *
 * Unlike the facc kernels above these are bit-exact against the scalar
 * reference (oc_oxk_dot_q4_k_q8_k_scalar / _q6_k_), as the C bodies are:
 * every block's integer dot is exact, and the per-block float epilogue is
 * the reference expression evaluated lane-wise, one lane per row, with the
 * same operation order. Four rows run as a tile so the expensive parts that
 * do not depend on the row are paid once per tile instead of once per row:
 * the horizontal reductions (one vphaddd tree reduces all four rows' dot
 * and min vectors together), the activation scale broadcast and the whole
 * float epilogue. Tail rows (n_rows % 4) reuse the same tile code with the
 * missing row pointers aliased to the last real row, so a row's bits never
 * depend on where the thread split put the tile boundary.
 *
 * Q4_K block (144 B): f16 d | f16 dmin | 12 B packed 6-bit scales/mins |
 *                     128 B nibbles
 * Q6_K block (210 B): 128 B ql | 64 B qh | 16 x int8 scales | f16 d
 * Q8_K block (292 B): f32 d | 256 x int8 | 16 x i16 block sums
 *
 * void oc_asm_dot_rows_qX_k_q8_k(const uint8_t *rows,  rdi
 *                                size_t row_bytes,     rsi
 *                                size_t n_rows,        rdx
 *                                size_t blocks,        rcx
 *                                const uint8_t *q8,    r8
 *                                float *out);          r9
 * ====================================================================== */

#define Q4_K_BYTES 144
#define Q6_K_BYTES 210
#define Q8_K_BYTES 292

/* Q4_K scale/min unpack into ymm5 (both 128-bit lanes identical), bytes:
 *   [0..3] sc0..3  [4..7] m0..3  [8..11] sc4..7  [12..15] m4..7
 * With q = u0 | u1 << 32 (bytes 4..11) and g = u2 (bytes 12..15):
 *   s0 | m0 << 32 = q & 0x3f..                          (low 6 bits)
 *   s1 | m1 << 32 = ((g | g << 28) & 0x0f..) | ((q >> 2) & 0x30..)
 * which is the reference's
 *   s1 = (u2 & 0x0f0f0f0f) | (((u0 >> 6) & 0x03030303) << 4)
 *   m1 = ((u2 >> 4) & 0x0f0f0f0f) | (((u1 >> 6) & 0x03030303) << 4)
 * done as two 64-bit halves. Clobbers r8, r9, r13, r14. */
.macro Q4K_SCALES prow
        mov             r9, qword ptr [\prow + r11 + 4]
        mov             r13, r9
        shr             r13, 2
        and             r13, qword ptr [rip + .Lk_m30]
        and             r9, qword ptr [rip + .Lk_m3f]
        mov             r14d, dword ptr [\prow + r11 + 12]
        mov             r8, r14
        shl             r8, 28
        or              r8, r14
        and             r8, qword ptr [rip + .Lk_m0f]
        or              r8, r13
        vmovq           xmm5, r9
        vpinsrq         xmm5, xmm5, r8, 1
        vinserti128     ymm5, ymm5, xmm5, 1
.endm

/* One 64-weight group pair gp of a Q4_K row: sub-blocks 2gp (low nibbles)
 * and 2gp+1 (high nibbles). Adds its scaled int32 pair sums into \acc
 * (first=1 initialises \acc instead). ymm15 = 0x0F bytes. */
.macro Q4K_GP prow, acc, gp, first
        vmovdqu         ymm6, ymmword ptr [\prow + r11 + 16 + 32*\gp]
        vpsrlw          ymm7, ymm6, 4
        vpand           ymm6, ymm6, ymm15
        vpand           ymm7, ymm7, ymm15
        vpmaddubsw      ymm6, ymm6, ymmword ptr [r12 + 4 + 64*\gp]
        vpmaddubsw      ymm7, ymm7, ymmword ptr [r12 + 36 + 64*\gp]
        vpshufb         ymm8, ymm5, ymmword ptr [rip + .Lk4_scsel + 64*\gp]
        vpshufb         ymm9, ymm5, ymmword ptr [rip + .Lk4_scsel + 64*\gp + 32]
        vpmaddwd        ymm6, ymm6, ymm8
        vpmaddwd        ymm7, ymm7, ymm9
.if \first
        vpaddd          \acc, ymm6, ymm7
.else
        vpaddd          ymm6, ymm6, ymm7
        vpaddd          \acc, \acc, ymm6
.endif
.endm

/* One Q4_K row block: \acc = 8 x int32 partial dot (sum of scale * q * a),
 * \mv = 8 x int32 partial mins term (sum of m_j * bsum pairs). */
.macro Q4K_ROW prow, acc, mv
        Q4K_SCALES      \prow
        vpshufb         ymm6, ymm5, ymmword ptr [rip + .Lk4_minsel]
        vpmaddwd        \mv, ymm6, ymmword ptr [r12 + 260]
        Q4K_GP          \prow, \acc, 0, 1
        Q4K_GP          \prow, \acc, 1, 0
        Q4K_GP          \prow, \acc, 2, 0
        Q4K_GP          \prow, \acc, 3, 0
.endm

/* Q6_K: 32 quants of one slot K (0..3) in half n. \q holds the unpacked
 * 6-bit values (0..63); ymm5 holds the eight int16 scales of this half in
 * both lanes, and the selector broadcasts groups 2K / 2K+1 onto the low /
 * high eight int16 lanes of the maddubs result. */
.macro Q6K_SLOT acc, q, n, k, first
        vpmaddubsw      \q, \q, ymmword ptr [r12 + 4 + 128*\n + 32*\k]
        vpshufb         ymm9, ymm5, ymmword ptr [rip + .Lk6_scsel + 32*\k]
        vpmaddwd        \q, \q, ymm9
.if \first
        vmovdqa         \acc, \q
.else
        vpaddd          \acc, \acc, \q
.endif
.endm

/* Half n (128 quants) of a Q6_K row block. The high-bit selection
 * ((qh >> 2k) & 3) << 4 is done as one 16-bit shift and a 0x30 mask; bits
 * bled across bytes land outside 0x30 and drop, so the values are the
 * reference's exactly. ymm15 = 0x0F bytes, ymm14 = 0x30 bytes. */
.macro Q6K_HALF prow, acc, n, first
        vmovdqu         ymm8, ymmword ptr [\prow + r11 + 128 + 32*\n]  /* qh */
        vmovdqu         ymm6, ymmword ptr [\prow + r11 + 64*\n]        /* lo */
        vmovdqu         ymm7, ymmword ptr [\prow + r11 + 64*\n + 32]   /* hi */
        vpsllw          ymm1, ymm8, 4
        vpand           ymm1, ymm1, ymm14
        vpand           ymm4, ymm6, ymm15
        vpor            ymm4, ymm4, ymm1
        Q6K_SLOT        \acc, ymm4, \n, 0, \first
        vpsllw          ymm1, ymm8, 2
        vpand           ymm1, ymm1, ymm14
        vpand           ymm4, ymm7, ymm15
        vpor            ymm4, ymm4, ymm1
        Q6K_SLOT        \acc, ymm4, \n, 1, 0
        vpand           ymm1, ymm8, ymm14
        vpsrlw          ymm4, ymm6, 4
        vpand           ymm4, ymm4, ymm15
        vpor            ymm4, ymm4, ymm1
        Q6K_SLOT        \acc, ymm4, \n, 2, 0
        vpsrlw          ymm1, ymm8, 2
        vpand           ymm1, ymm1, ymm14
        vpsrlw          ymm4, ymm7, 4
        vpand           ymm4, ymm4, ymm15
        vpor            ymm4, ymm4, ymm1
        Q6K_SLOT        \acc, ymm4, \n, 3, 0
.endm

/* One Q6_K row block: \acc = 8 x int32 partial dot, \mv = 8 x int32
 * partial sum of sc[g] * bsum[g]. Only in-block bytes are read: the scale
 * halves are 8-byte loads (192..199, 200..207). */
.macro Q6K_ROW prow, acc, mv
        vpmovsxbw       ymm9, xmmword ptr [\prow + r11 + 192]
        vpmaddwd        \mv, ymm9, ymmword ptr [r12 + 260]
        vpmovsxbw       xmm5, qword ptr [\prow + r11 + 192]
        vinserti128     ymm5, ymm5, xmm5, 1
        Q6K_HALF        \prow, \acc, 0, 1
        vpmovsxbw       xmm5, qword ptr [\prow + r11 + 200]
        vinserti128     ymm5, ymm5, xmm5, 1
        Q6K_HALF        \prow, \acc, 1, 0
.endm

/* Final step of the tile reduction. In: ymm12 / ymm13 = the vphaddd trees
 * of the four rows' dot / min vectors, each 128-bit lane holding one
 * partial per row. Out: ymm2 = int32 [P0 P1 P2 P3 | M0 M1 M2 M3]. Integer,
 * so any tree is exact. */
.macro K_REDUCE
        vperm2i128      ymm2, ymm12, ymm13, 0x20
        vperm2i128      ymm3, ymm12, ymm13, 0x31
        vpaddd          ymm2, ymm2, ymm3
.endm

/* One block of a 4-row tile: rows rdi, r15, rbx, rbp; r11 = weight block
 * offset; r12 -> Q8_K block. Leaves ymm2 = [P0..3 | M0..3] int32. */
.macro K_TILE_BLOCK rowmac
        \rowmac         rdi, ymm12, ymm13
        \rowmac         r15, ymm10, ymm11
        vphaddd         ymm12, ymm12, ymm10
        vphaddd         ymm13, ymm13, ymm11
        \rowmac         rbx, ymm2, ymm3
        \rowmac         rbp, ymm10, ymm11
        vphaddd         ymm2, ymm2, ymm10
        vphaddd         ymm3, ymm3, ymm11
        vphaddd         ymm12, ymm12, ymm2
        vphaddd         ymm13, ymm13, ymm3
        K_REDUCE
.endm

/* Q4_K epilogue, lane r = row r:
 *   sum += (dw*dq)*(float)P - (dmin*dq)*(float)M     (reference order)
 * dw/dmin are the first 4 bytes of each row's block; one pshufb sorts the
 * eight f16 into [dw0..3 | dmin0..3]. */
.macro Q4K_EPILOGUE
        vcvtdq2ps       ymm2, ymm2
        vmovd           xmm6, dword ptr [rdi + r11]
        vmovd           xmm7, dword ptr [r15 + r11]
        vmovd           xmm8, dword ptr [rbx + r11]
        vmovd           xmm9, dword ptr [rbp + r11]
        vpunpckldq      xmm6, xmm6, xmm7
        vpunpckldq      xmm8, xmm8, xmm9
        vpunpcklqdq     xmm6, xmm6, xmm8
        vpshufb         xmm6, xmm6, xmmword ptr [rip + .Lk4_dsel]
        vcvtph2ps       ymm6, xmm6
        vbroadcastss    ymm1, dword ptr [r12]
        vmulps          ymm6, ymm6, ymm1
        vmulps          ymm6, ymm6, ymm2
        vextractf128    xmm7, ymm6, 1
        vsubps          xmm6, xmm6, xmm7
        vaddps          xmm0, xmm0, xmm6
.endm

/* Q6_K epilogue, lane r = row r:
 *   sum += (dw*dq)*(float)(P - 32*M)                 (reference order)
 * d is the f16 at byte 208; the dword at 206 keeps the load in-block and
 * a pshufb picks its high halves. */
.macro Q6K_EPILOGUE
        vextracti128    xmm3, ymm2, 1
        vpslld          xmm3, xmm3, 5
        vpsubd          xmm2, xmm2, xmm3
        vcvtdq2ps       xmm2, xmm2
        vmovd           xmm6, dword ptr [rdi + r11 + 206]
        vmovd           xmm7, dword ptr [r15 + r11 + 206]
        vmovd           xmm8, dword ptr [rbx + r11 + 206]
        vmovd           xmm9, dword ptr [rbp + r11 + 206]
        vpunpckldq      xmm6, xmm6, xmm7
        vpunpckldq      xmm8, xmm8, xmm9
        vpunpcklqdq     xmm6, xmm6, xmm8
        vpshufb         xmm6, xmm6, xmmword ptr [rip + .Lk6_dsel]
        vcvtph2ps       xmm6, xmm6
        vbroadcastss    xmm1, dword ptr [r12]
        vmulps          xmm6, xmm6, xmm1
        vmulps          xmm6, xmm6, xmm2
        vaddps          xmm0, xmm0, xmm6
.endm

/* Full row loop for one K-quant type. Stack: [rsp] = q8 base.
 *   rcx = blocks * BYTES (end offset), rdx = rows left, r10 = out.
 * NPF = cache lines of the next tile to prefetch per block (4*BYTES/64). */
.macro K_ROWS_FN name, BYTES, rowmac, epimac, NPF
        .globl  \name
        .type   \name, @function
        .p2align 5
\name:
        .cfi_startproc
        push    rbx
        .cfi_adjust_cfa_offset 8
        push    rbp
        .cfi_adjust_cfa_offset 8
        push    r12
        .cfi_adjust_cfa_offset 8
        push    r13
        .cfi_adjust_cfa_offset 8
        push    r14
        .cfi_adjust_cfa_offset 8
        push    r15
        .cfi_adjust_cfa_offset 8
        push    r8
        .cfi_adjust_cfa_offset 8
        mov     r10, r9
        imul    rcx, rcx, \BYTES
        mov     eax, 0x0F0F0F0F
        vmovd   xmm15, eax
        vpbroadcastd ymm15, xmm15
        mov     eax, 0x30303030
        vmovd   xmm14, eax
        vpbroadcastd ymm14, xmm14

1:      /* tile */
        cmp     rdx, 4
        jb      5f
        lea     r15, [rdi + rsi]
        lea     rbx, [rdi + rsi*2]
        lea     rbp, [rbx + rsi]
        lea     rax, [rbp + rsi + PF_TILE_EXTRA]
        vxorps  xmm0, xmm0, xmm0
        xor     r11d, r11d
        mov     r12, qword ptr [rsp]
        test    rcx, rcx
        jz      3f
        .p2align 4
2:
        .irp k, 0,1,2,3,4,5,6,7,8,9,10,11,12,13
        .if \k < \NPF
        prefetcht0 [rax + r11*4 + 64*\k]
        .endif
        .endr
        K_TILE_BLOCK \rowmac
        \epimac
        add     r11, \BYTES
        add     r12, Q8_K_BYTES
        cmp     r11, rcx
        jb      2b
3:
        vmovups xmmword ptr [r10], xmm0
        add     r10, 16
        lea     rdi, [rbp + rsi]
        sub     rdx, 4
        jmp     1b

5:      /* tail: 1..3 rows, missing rows alias the last real row */
        test    rdx, rdx
        jz      9f
        mov     r15, rdi
        cmp     rdx, 2
        jb      6f
        lea     r15, [rdi + rsi]
6:      mov     rbx, r15
        cmp     rdx, 3
        jb      7f
        lea     rbx, [rdi + rsi*2]
7:      mov     rbp, rbx
        vxorps  xmm0, xmm0, xmm0
        xor     r11d, r11d
        mov     r12, qword ptr [rsp]
        test    rcx, rcx
        jz      8f
        .p2align 4
4:
        K_TILE_BLOCK \rowmac
        \epimac
        add     r11, \BYTES
        add     r12, Q8_K_BYTES
        cmp     r11, rcx
        jb      4b
8:
        vmovss  dword ptr [r10], xmm0
        cmp     rdx, 2
        jb      9f
        vextractps dword ptr [r10 + 4], xmm0, 1
        cmp     rdx, 3
        jb      9f
        vextractps dword ptr [r10 + 8], xmm0, 2

9:
        vzeroupper
        pop     r8
        .cfi_adjust_cfa_offset -8
        pop     r15
        .cfi_adjust_cfa_offset -8
        pop     r14
        .cfi_adjust_cfa_offset -8
        pop     r13
        .cfi_adjust_cfa_offset -8
        pop     r12
        .cfi_adjust_cfa_offset -8
        pop     rbp
        .cfi_adjust_cfa_offset -8
        pop     rbx
        .cfi_adjust_cfa_offset -8
        ret
        .cfi_endproc
        .size   \name, .-\name
.endm

        K_ROWS_FN oc_asm_dot_rows_q4_k_q8_k, Q4_K_BYTES, Q4K_ROW, Q4K_EPILOGUE, 9
        K_ROWS_FN oc_asm_dot_rows_q6_k_q8_k, Q6_K_BYTES, Q6K_ROW, Q6K_EPILOGUE, 14

        .section .rodata
        .p2align 5
/* Q4_K sub-block scale broadcasts: selector j puts sc[j] (byte j for
 * j < 4, byte j + 4 for j >= 4) zero-extended into all 16 int16 lanes. */
.Lk4_scsel:
        .irp j, 0,1,2,3,4,5,6,7
        .if \j < 4
        .rept 16
        .byte \j, 0x80
        .endr
        .else
        .rept 16
        .byte \j + 4, 0x80
        .endr
        .endif
        .endr
/* Q4_K mins duplicated onto bsum pairs: int16 lanes 2k, 2k+1 = m[k]
 * (m0..3 at bytes 4..7 in the low lane, m4..7 at bytes 12..15 high). */
.Lk4_minsel:
        .byte 4,0x80,4,0x80, 5,0x80,5,0x80, 6,0x80,6,0x80, 7,0x80,7,0x80
        .byte 12,0x80,12,0x80, 13,0x80,13,0x80, 14,0x80,14,0x80, 15,0x80,15,0x80
/* Q6_K: slot K takes int16 scale 2K (low lane) and 2K+1 (high lane). */
.Lk6_scsel:
        .irp k, 0,1,2,3
        .rept 8
        .byte 4*\k, 4*\k + 1
        .endr
        .rept 8
        .byte 4*\k + 2, 4*\k + 3
        .endr
        .endr
        .p2align 4
/* [dw0 dmin0 dw1 dmin1 ...] f16 words -> [dw0..3 | dmin0..3]. */
.Lk4_dsel:
        .byte 0,1, 4,5, 8,9, 12,13, 2,3, 6,7, 10,11, 14,15
/* Q6_K: the f16 d is the high word of each gathered dword. */
.Lk6_dsel:
        .byte 2,3, 6,7, 10,11, 14,15, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80
        .p2align 3
.Lk_m3f: .quad 0x3f3f3f3f3f3f3f3f
.Lk_m30: .quad 0x3030303030303030
.Lk_m0f: .quad 0x0f0f0f0f0f0f0f0f

#endif /* __x86_64__ && __ELF__ */

        .section .note.GNU-stack,"",@progbits
