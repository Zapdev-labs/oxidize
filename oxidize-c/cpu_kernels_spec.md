# CPU fused GEMV for AL-family weights

This is the plan for decode-time matrix-vector products in `oxidize-c`.
It is an explanation of why the change exists and how the types fit together.

## Who this is for

Someone running a local GGUF on CPU with AL5, AL5_XS, AL6, or AL8 weights.
Before this work, those types dequantized every weight row to f32 and then dotted.
Q4_0 and Q8_0 already quantized the activation once and kept the integer product.

## Why

AL5, AL6, and AL8 store the same bytes as Q4_0, Q5_0, and Q8_0.
Only the encoder differs. The decoder is the same bitstream.
AL5_XS is a 14-byte 3-bit block. It has no fused integer kernel yet.

`oc_matvec_quantized` in `src/compute/matvec.c` only fused Q4_0, Q4_1, Q8_0, and the K-quants.
AL5 therefore paid the slow path on the same layout Q4_0 already fused.

## Data shape

One weight row is `blocks` packed blocks. One activation is packed Q8_0 with the same block count.

Each block produces one int32 product `isum` and one f32 scale product `dw * dq`.
The row result is `sum_b (dw_b * dq_b * (float)isum_b)`.

Layouts:

- AL5 and Q4_0. 18 bytes. f16 `d`, 16 nibble bytes. Code `nibble - 8`.
- AL8 and Q8_0. 34 bytes. f16 `d`, 32 int8. Code is the stored int8.
- AL6 and Q5_0. 22 bytes. f16 `d`, 4-byte `qh`, 16 nibble bytes. Code is 5-bit minus 16.
- AL5_XS. 14 bytes. f16 `d`, 12 bytes of LSB-first 3-bit codes. Code is `v - 4`.

## Work units

1. Alias AL5 to the Q4_0 fused kernel and AL8 to the Q8_0 fused kernel.
2. Add a Q5_0 × Q8_0 integer row-dot and alias AL6 to it. Add Q5_1 the same way.
3. Add an AL5_XS × Q8_0 integer row-dot.
4. Replace the AVX2 Q4_0 stub with an integer `madd_epi16` kernel and install it in `oc_oxk_init`.
5. Tests compare each integer kernel to dequantized weights dotted with the same dequantized Q8 activation.

## Out of scope

New on-disk quant types.
GPU kernels.
Changing AL packers.
Porting this into Rust or Go in the same change.

## Done when

`make -C oxidize-c test` passes.
`tests/test_al_fused.c` shows AL5, AL5_XS, AL6, AL8, Q5_0, and Q5_1 kernels agree with GGUF dequant of the same packed row.
On AVX2 hosts, `oc_oxk_dot_q4_0_q8_0_avx2` matches the scalar Q4_0 kernel at raw float bits.
