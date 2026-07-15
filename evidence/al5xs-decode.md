# AL5_XS (ggml type 243) — empirically verified decode

Verified 2026-07-09 against `unsloth/gemma-4-31B-it-GGUF` BF16 source weights
(token_embd.weight and blk.0.ffn_down.weight slices fetched via HTTP range
requests and compared to `freakyskittle/gemma-4-31B-it-AL-GGUF` AL5_XS).

## Layout

Block = 32 weights = **14 bytes** (3.5 bpw):

| bytes | content |
|-------|---------|
| 0–1   | little-endian IEEE f16 scale `d` |
| 2–13  | 96-bit LSB-first bitstream; code `i` occupies bits `[3i, 3i+3)` |

## Decode

```
w[i] = (q[i] - 4) * d      // q in 0..7
```

Uniform integer levels; a fitted per-code LUT gave medians `[-4.07, -2.92,
-1.92, -0.92, 0.00, 0.92, 1.91, 3.15]` — i.e. integer levels, no LUT needed.

## Evidence

- token_embd rows 0–9: cosine(dequant, bf16) = **0.9811** with (q−4)·d,
  0.891 with (q−3.5)·d, 0.13 with MSB-first packing, NaN scales if the f16
  is taken from the block tail.
- blk.0.ffn_down row 0: cosine = **0.9803**.
- Residual ~0.98 (not ~1.0) is inherent 3-bit reconstruction error
  (RMSE 0.0029 vs mean |w| 0.0103), consistent across decoders tried.

File-level facts (gemma-4-31B-it-AL5_XS.gguf, 13,450,666,816 bytes):
- 833 tensors: 422 F32 (norms, scalars), 411 type 243 — every matmul weight
  including token_embd is AL5_XS. `general.file_type = 243`.
- arch `gemma4`: 60 layers, sliding_window_pattern 5×SWA:1×full,
  head_count_kv per layer [16 (swa) / 4 (full)], key/value_length 512 full /
  256 swa, rope freq 1e6 full / 1e4 swa, final_logit_softcapping 30,
  per-block `layer_output_scale` scalar tensors.

## Update 2026-07-11 — the upstream AL5_XS file is functionally broken

The decode above is correct, but the *model* in
`freakyskittle/gemma-4-31B-it-AL-GGUF` degenerates to ` de de de...` under
greedy decode — in oxidize-c AND in llama.cpp b9966 (fed a Q8 dequant of the
same weights, chat-templated or raw). The official
`unsloth/gemma-4-31B-it-GGUF` Q4_K_M is coherent in both llama.cpp and
oxidize-c (near token-identical greedy outputs). Conclusion: oxidize-c's
gemma4 forward pass is correct; the AL upload (or its quantization run) is
bad. Local `Q8_0-converted` / `AL5_XS-fixed` files derive from the broken
file and inherit the failure. Replacement: `gemma-4-31B-it-AL5_XS-own.gguf`,
requantized from the official Q4_K_M with `oxidize-c-requant`.
