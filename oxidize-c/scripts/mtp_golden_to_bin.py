#!/usr/bin/env python3
"""Convert golden_mtp.npz (export_mtp_gguf.py --golden) to the flat binary
read by `oc-eval --mtp-golden` (layout documented in scripts/oc_eval.c)."""
import struct
import sys

import numpy as np

def need(name, arr, shape):
    if getattr(arr, "shape", None) != shape:
        raise SystemExit(f"{name} shape {getattr(arr, 'shape', None)} != {shape}")

z = np.load(sys.argv[1])
missing = [k for k in ("tokens", "h", "s1", "s2", "k1", "v1") if k not in z]
if missing:
    raise SystemExit(f"golden npz missing {missing}")
toks, h, s1, s2, k1, v1 = (z[k] for k in ("tokens", "h", "s1", "s2", "k1", "v1"))
if s1.ndim != 2 or k1.ndim != 3 or v1.ndim != 3:
    raise SystemExit(f"s1/k1/v1 rank {s1.ndim}/{k1.ndim}/{v1.ndim}, want 2/3/3")
T, D = s1.shape
n_kv, t_k, hd = k1.shape
need("tokens", toks, (T + 2,))
need("h", h, (T + 1, D))
need("s2", s2, (T - 1, D))
need("k1", k1, (n_kv, T, hd))
need("v1", v1, (n_kv, T, hd))
with open(sys.argv[2], "wb") as f:
    f.write(struct.pack("<5I", 0x4750544D, T, D, n_kv, hd))
    f.write(toks.astype("<i4").tobytes())
    for a in (h, s1, s2, k1, v1):
        f.write(np.ascontiguousarray(a, dtype="<f4").tobytes())
print(f"T={T} D={D} n_kv={n_kv} hd={hd} -> {sys.argv[2]}")
