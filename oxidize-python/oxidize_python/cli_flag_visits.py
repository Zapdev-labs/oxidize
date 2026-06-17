"""Track which CLI flags were explicitly set on the command line."""

from __future__ import annotations

_FLAG_NAMES = {
  "threads": ("--threads",),
  "ctx_size": ("--ctx-size",),
  "n_gpu_layers": ("--n-gpu-layers",),
  "layer_cache": ("--layer-cache",),
  "layer_wise": ("--layer-wise",),
  "paged": ("--paged",),
  "ram_offload": ("--ram-offload",),
  "dflash_fusion": ("--dflash-fusion",),
}


def flag_visits(argv: list[str]) -> set[str]:
    visited: set[str] = set()
    args = list(argv)
    i = 0
    while i < len(args):
        token = args[i]
        for name, flags in _FLAG_NAMES.items():
            if token in flags:
                visited.add(name)
        i += 1
    return visited
