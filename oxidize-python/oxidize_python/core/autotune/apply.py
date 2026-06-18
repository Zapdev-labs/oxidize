"""Apply autotune plans to CLI options."""

from __future__ import annotations

from dataclasses import dataclass

from oxidize_python.core.autotune.rules import PipelineMode, TuningPlan
from oxidize_python.core.kv_cache import Quantization as KvQuant


@dataclass
class PlanOverrides:
    threads: int | None = None
    ctx_size: int | None = None
    n_gpu_layers: int | None = None
    layer_cache: int | None = None
    layer_wise: bool | None = None
    mmap: bool | None = None
    paged: bool | None = None
    turboquant: bool | None = None
    pipeline: str | None = None


def overrides_from_plan(plan: TuningPlan) -> PlanOverrides:
    pipeline = {
        PipelineMode.SEQUENTIAL: "sequential",
        PipelineMode.CONTINUOUS: "continuous",
        PipelineMode.PAGED: "paged",
        PipelineMode.ASYMMETRIC: "asymmetric",
    }[plan.pipeline]
    return PlanOverrides(
        threads=plan.threads,
        ctx_size=plan.ctx_size,
        n_gpu_layers=plan.n_gpu_layers,
        layer_cache=plan.layer_cache,
        layer_wise=plan.layer_wise,
        mmap=plan.mmap,
        paged=plan.pipeline == PipelineMode.PAGED,
        turboquant=plan.kv_quantization == KvQuant.TURBOQUANT,
        pipeline=pipeline,
    )
