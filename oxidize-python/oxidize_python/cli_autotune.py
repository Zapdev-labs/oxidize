"""Apply autotune to CLI run options."""

from __future__ import annotations

import json
import sys
from typing import Any

from oxidize_python.core import autotune
from oxidize_python.core.ggufcore import gguf as ggufcore
from oxidize_python.cli_flags import RunOptions


def apply_autotune(model_path: str, opts: RunOptions, visited: set[str]) -> None:
    if not opts.auto_tune:
        return
    mapped = ggufcore.load_mapped(model_path)
    inv = autotune.detect()
    fp = autotune.fingerprint(mapped)
    plan = autotune.plan(inv, fp)
    if _should_print_plan(opts.print_plan):
        if opts.print_plan == "json":
            payload: dict[str, Any] = {
                "threads": plan.threads,
                "ctx_size": plan.ctx_size,
                "n_gpu_layers": plan.n_gpu_layers,
                "layer_wise": plan.layer_wise,
                "layer_cache": plan.layer_cache,
                "pipeline": plan.pipeline.name,
                "rationale": plan.rationale,
            }
            print(json.dumps(payload, indent=2), file=sys.stderr)
        else:
            print(f"\n[oxidize auto-tune plan]\n{plan.summary()}", file=sys.stderr)
    overrides = autotune.overrides_from_plan(plan)
    if "threads" not in visited and overrides.threads:
        opts.threads = overrides.threads
    if "ctx_size" not in visited and overrides.ctx_size:
        opts.ctx_size = overrides.ctx_size
    if "n_gpu_layers" not in visited and overrides.n_gpu_layers is not None:
        opts.n_gpu_layers = overrides.n_gpu_layers
    if "layer_cache" not in visited and overrides.layer_cache:
        opts.layer_cache = overrides.layer_cache
    if "layer_wise" not in visited and overrides.layer_wise:
        opts.layer_wise = overrides.layer_wise
    if "paged" not in visited and overrides.paged:
        opts.use_paged = True
    if plan.speculative.name == "DFLASH" and "dflash_fusion" not in visited and not opts.draft_model:
        opts.dflash_fusion = True
    print(
        f"[oxidize auto-tune] applied: threads={opts.threads} ctx={opts.ctx_size} "
        f"n_gpu_layers={opts.n_gpu_layers} layer_wise={opts.layer_wise}",
        file=sys.stderr,
    )


def _should_print_plan(mode: str) -> bool:
    m = (mode or "auto").lower()
    if m in ("json", "yes", "true", "1"):
        return True
    if m in ("no", "false", "0"):
        return False
    return sys.stderr.isatty()
