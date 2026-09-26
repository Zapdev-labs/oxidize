#!/usr/bin/env python3
"""Convert golden_mtp.npz (export_mtp_gguf.py --golden) to the flat binary
read by `oc-eval --mtp-golden` (layout documented in scripts/oc_eval.c)."""
import struct
import sys

import numpy as np

z = np.load(sys.argv[1])
toks, h, s1, s2, k1, v1 = (z[k] for k in ("tokens", "h", "s1", "s2", "k1", "v1"))
T, D = s1.shape
n_kv, _, hd = k1.shape
assert toks.shape == (T + 2,) and h.shape == (T + 1, D) and s2.shape == (T - 1, D)
with open(sys.argv[2], "wb") as f:
    f.write(struct.pack("<5I", 0x4750544D, T, D, n_kv, hd))
    f.write(toks.astype("<i4").tobytes())
    for a in (h, s1, s2, k1, v1):
        f.write(np.ascontiguousarray(a, dtype="<f4").tobytes())
print(f"T={T} D={D} n_kv={n_kv} hd={hd} -> {sys.argv[2]}")
