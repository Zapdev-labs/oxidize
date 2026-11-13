/*
 * oxk_x86_64.S: hand-written x86-64 AVX2/FMA/F16C decode GEMV kernels.
 *
 * These replace the C intrinsics "facc" row kernels in oxk_avx2.c on the
 * single-activation decode path (matvec.c matvec_fused_slice and friends).
 * Same contract as the C kernels: per-block integer dots are exact, the
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

#endif /* __x86_64__ && __ELF__ */

        .section .note.GNU-stack,"",@progbits
