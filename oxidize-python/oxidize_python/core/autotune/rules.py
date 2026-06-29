"""Autotune rule table (mirrors oxidize-golang/core/autotune/rules.go)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from oxidize_python.core.autotune.detect import HardwareInventory, is_skylake_sp
from oxidize_python.core.autotune.fingerprint import (
    ModelFingerprint,
    kv_bytes_per_token,
    per_layer_weight_bytes,
)
from oxidize_python.core.kv_cache import Quantization as KvQuant
from oxidize_python.core.simd.simd import Backend
from oxidize_python.gpucluster import GpuFamily


class PipelineMode(Enum):
    SEQUENTIAL = auto()
    CONTINUOUS = auto()
    PAGED = auto()
    ASYMMETRIC = auto()


class SpeculativeSpec(Enum):
    NONE = auto()
    DFLASH = auto()
    MTP = auto()


@dataclass
class TuningPlan:
    threads: int = 0
    ctx_size: int = 0
    kv_cache_dtype: str = "f16"
    kv_quantization: KvQuant = KvQuant.ASYMMETRIC
    n_gpu_layers: int = 0
    mmap: bool = True
    mlock: bool = False
    layer_wise: bool = False
    layer_cache: int = 0
    pipeline: PipelineMode = PipelineMode.SEQUENTIAL
    speculative: SpeculativeSpec = SpeculativeSpec.NONE
    decode_tile_tokens: int = 0
    expected_prompt_tps: float = 0.0
    expected_decode_tps: float = 0.0
    rationale: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"threads           : {self.threads}",
            f"ctx_size          : {self.ctx_size}",
            f"kv_cache_dtype    : {self.kv_cache_dtype} (quantization: {self.kv_quantization})",
            f"n_gpu_layers      : {self.n_gpu_layers}",
            f"layer_wise={self.layer_wise} layer_cache={self.layer_cache}",
            f"pipeline          : {self.pipeline.name}",
            f"speculative       : {self.speculative.name}",
            "expected t/s      : "
            f"prompt ≈ {self.expected_prompt_tps:.1f}  "
            f"decode ≈ {self.expected_decode_tps:.1f}",
        ]
        if self.rationale:
            lines.append("\nRationale:")
            lines.extend(f"  - {r}" for r in self.rationale)
        return "\n".join(lines) + "\n"


def plan(inv: HardwareInventory, model: ModelFingerprint) -> TuningPlan:
    p = TuningPlan()
    ram = _effective_ram(inv)
    if ram < model.file_size_bytes * 12 // 10:
        p.layer_wise = True
        p.layer_cache = max(inv.physical_cores // 4, 1)
        p.rationale.append("model exceeds 1.2× RAM → layer_wise streaming")
    if inv.simd == Backend.AVX512F and not is_skylake_sp():
        p.rationale.append("AVX-512 available")
    elif inv.simd == Backend.AVX2:
        p.rationale.append("AVX2 path")
    if inv.has_gpu:
        per_layer = per_layer_weight_bytes(model)
        if per_layer:
            usable = int(inv.gpu_vram_bytes * 0.85)
            n = min(model.layer_count, usable // per_layer) if per_layer else 0
            if inv.gpu_vram_bytes < model.file_size_bytes // 4:
                n = 0
            p.n_gpu_layers = n
            if n == model.layer_count:
                p.mmap = False
    p.kv_cache_dtype = "f16"
    p.kv_quantization = (
        KvQuant.TURBOQUANT
        if inv.gpu_vram_bytes // (1 << 30) < 8 or model.layer_count >= 60
        else KvQuant.ASYMMETRIC
    )
    kv_budget = max(ram - model.file_size_bytes - (8 << 30), 0)
    kv_b = kv_bytes_per_token(model, 2)
    ctx_cap = min(131072, kv_budget // kv_b) if kv_b else 4096
    p.ctx_size = min(max(4096, ctx_cap), 8192 if model.num_kv_heads <= 4 else 4096)
    if p.layer_cache == 0:
        p.layer_cache = max(2, min(inv.physical_cores, 8))
    if inv.has_gpu and model.has_mtp:
        p.speculative = SpeculativeSpec.MTP
    elif inv.has_gpu and model.architecture in ("qwen2", "qwen3", "llama", "lfm2"):
        p.speculative = SpeculativeSpec.DFLASH
    if inv.has_gpu and p.n_gpu_layers > 0:
        p.threads = max(inv.physical_cores // 8, 4)
        p.pipeline = PipelineMode.PAGED
    else:
        p.threads = inv.physical_cores
        if inv.physical_cores >= 8 and inv.total_ram_bytes >= (64 << 30) and not model.is_moe:
            p.pipeline = PipelineMode.CONTINUOUS
    if p.ctx_size > 8192:
        p.decode_tile_tokens = 1024
    elif p.ctx_size > 4096 and inv.simd == Backend.AVX2:
        p.decode_tile_tokens = 512
    p.expected_decode_tps = _estimate_tps(inv, model, p)
    p.expected_prompt_tps = p.expected_decode_tps * 6
    return p


def _effective_ram(inv: HardwareInventory) -> int:
    if inv.container_mem_limit is not None:
        return min(inv.container_mem_limit, inv.total_ram_bytes)
    return inv.total_ram_bytes


def _estimate_tps(inv: HardwareInventory, model: ModelFingerprint, p: TuningPlan) -> float:
    if inv.has_gpu and p.n_gpu_layers > 0 and inv.gpu_family is not None:
        match inv.gpu_family:
            case GpuFamily.B200:
                return 200.0
            case GpuFamily.A100:
                return 90.0
            case GpuFamily.RTX_PRO_6000:
                return 70.0
        return 30.0
    return float(inv.physical_cores) * 0.6
