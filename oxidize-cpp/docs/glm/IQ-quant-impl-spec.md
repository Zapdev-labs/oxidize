# IQ-quant dequant spec for GLM-5.2 UD-IQ1_M (oxidize-cpp)

GLM-5.2 UD-IQ1_M uses these quant types (must all dequant correctly): F32, Q8_0, Q5_K, Q6_K, Q4_K, **IQ1_M, IQ2_XXS, IQ3_XXS, IQ4_XS** (+ rare Q2_K/Q3_K). Q*_K + IQ4_XS already work. The IQ1/IQ2/IQ3 paths need work.

## CRITICAL: IQ1_M & IQ1_S already exist in src/quant.cpp but are BUGGY (2 bugs)

### Bug 1 (IQ1_M + IQ1_S): `iq1s_grid_decode` (quant.cpp ~line 162) is an APPROXIMATION, not the real table.
The real grid is `static const uint64_t iq1s_grid[2048]` in `/home/dih/llama.cpp-upstream/ggml/src/ggml-common.h` lines 1124–1637 (16 KB; each uint64 packs 8 int8 values from {-1,0,+1}, little-endian). IQ1_S and IQ1_M SHARE this table. Fix: copy the 2048-entry table verbatim into quant.cpp; replace `iq1s_grid_decode(idx,...)` with `const int8_t* g=(const int8_t*)(IQ1S_GRID+idx);` and use g[0..7].

### Bug 2 (IQ1_M only): scale read uses uint8 not uint16 (quant.cpp ~line 639 `scales[ib/2]`).
`scales[8]` is logically 4×uint16_t LE. For odd sub-blocks the 3-bit scale fields straddle/overflow the byte → half the sub-blocks get wrong scale (dl2 collapses to d*1.0). Fix: read uint16 words via `read_u16_le(scales+2*k)`.

### Correct IQ1_M block (56 bytes, QK_K=256): qs[0..31] (4B/subblock), qh[32..47] (2B/subblock), scales[48..55] (4×u16).
1. Global F16 scale d from top nibbles of the 4 u16 words:
   `u16 s = (sc0>>12) | ((sc1>>8)&0x00f0) | ((sc2>>4)&0x0f00) | (sc3&0xf000); d=f16(s);`
2. Per sub-block ib in 0..7: `w=sc[ib/2]; sh=6*(ib%2); dl1=d*(2*((w>>sh)&7)+1); dl2=d*(2*((w>>(sh+3))&7)+1);`
3. 4 grid groups: `qs_b=qs+ib*4; qh_b=qh+ib*2;`
   `idx0=qs_b[0]|((qh_b[0]<<8)&0x700); idx1=qs_b[1]|((qh_b[0]<<4)&0x700); idx2=qs_b[2]|((qh_b[1]<<8)&0x700); idx3=qs_b[3]|((qh_b[1]<<4)&0x700);`
   delta sign: qh_b[0]&0x08 →g0, &0x80→g1, qh_b[1]&0x08→g2, &0x80→g3; `±IQ1S_DELTA (0.125)`.
4. out: groups 0,1 use dl1; 2,3 use dl2: `for j in 0..8: out = dl*((float)g[j] + delta);`
IQ1_S: d is a direct F16 in first 2 bytes (no nibble trick) — only Bug 1 applies there.

## IQ2_XXS / IQ3_XXS — likely NOT in the dequantize_row dispatch switch (mapped in from_ggml_type only). MUST add.
Port from `/home/dih/llama.cpp-upstream/ggml/src/ggml-quants.c` (`dequantize_row_iq2_xxs`, `dequantize_row_iq3_xxs`) + grid tables `iq2xxs_grid[256]` (uint64), `iq3xxs_grid[256]` (uint32), and the `ksigns_iq2xs[128]`/`kmask_iq2xs[8]` sign tables, all in ggml-common.h. These use a sign-encoding scheme (8 bytes from grid + signs from ksigns). Implement scalar dequant matching ggml; wire into dequantize_row + is_supported_quant_gemv (dequant-then-dot fallback is fine for correctness).

## IQ4_XS — already mapped+dispatched per R1; verify it actually runs on a GLM tensor.

## Verify: dequant one known block and compare against llama.cpp's dequantize_row for the same bytes (write a tiny harness, or compare first-row logits). ctest must stay green.
Files: src/quant.cpp (primary), grid tables from ggml-common.h. dispatcher/block-size/ggml-type-map for IQ1_M already correct.
