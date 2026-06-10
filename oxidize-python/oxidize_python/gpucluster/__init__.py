"""GPU cluster modeling, Kubernetes/Helm manifest generation, and runtime detection.

This is a feature-parity port of the Rust ``oxidize_core::gpu_cluster`` module
and the Go ``core/gpucluster`` package, implementing the Oxidize GPU Cluster
specification (``docs/gpu_cluster_spec.md``).

YAML is emitted via string building to keep the package dependency-free (the
pure-Python port ships no YAML library).
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "GpuFamily",
    "GpuProfile",
    "NodePoolSpec",
    "MigProfile",
    "DetectedGpu",
    "profile",
    "all_profiles",
    "node_pool_yaml",
    "node_pools_yaml",
    "node_labels",
    "device_plugin_config_yaml",
    "mig_profiles",
    "mig_config_yaml",
    "gpu_tolerations_yaml",
    "helm_values_yaml",
    "prometheus_rules_yaml",
    "classify_product",
    "parse_nvidia_smi_csv",
    "detect_gpus",
    "summarize",
]


class GpuFamily(Enum):
    """The three GPU tiers the Oxidize cluster targets."""

    B200 = "b200"
    A100 = "a100"
    RTX_PRO_6000 = "rtx-pro-6000"

    @property
    def slug(self) -> str:
        return self.value

    @classmethod
    def from_slug(cls, s: str) -> "GpuFamily | None":
        key = s.strip().lower()
        if key in ("rtx-pro6000", "rtxpro6000"):
            return cls.RTX_PRO_6000
        for f in cls:
            if f.value == key:
                return f
        return None

    @classmethod
    def all(cls) -> list["GpuFamily"]:
        return [cls.B200, cls.A100, cls.RTX_PRO_6000]


@dataclass(frozen=True)
class GpuProfile:
    """Static hardware/scheduling profile for a GPU tier."""

    family: GpuFamily
    product: str
    generation: str
    memory_mib: int
    tdp_watts: int
    nvlink: bool
    mig_capable: bool
    time_slice_replicas: int
    network_class: str
    workload_type: str


_PROFILES: dict[GpuFamily, GpuProfile] = {
    GpuFamily.B200: GpuProfile(
        GpuFamily.B200, "NVIDIA-B200", "blackwell", 196608, 1000,
        True, False, 1, "infiniband", "training",
    ),
    GpuFamily.A100: GpuProfile(
        GpuFamily.A100, "NVIDIA-A100-SXM4-80GB", "ampere", 81920, 400,
        True, True, 2, "infiniband", "mixed",
    ),
    GpuFamily.RTX_PRO_6000: GpuProfile(
        GpuFamily.RTX_PRO_6000, "NVIDIA-RTX-Pro-6000", "ada", 98304, 300,
        False, False, 8, "ethernet", "workstation",
    ),
}


def profile(family: GpuFamily) -> GpuProfile:
    """Return the canonical profile for a family."""
    return _PROFILES[family]


def all_profiles() -> list[GpuProfile]:
    return [_PROFILES[f] for f in GpuFamily.all()]


@dataclass(frozen=True)
class NodePoolSpec:
    family: GpuFamily
    node_count: int
    gpu_per_node: int


_POOL_NAMES = {
    GpuFamily.B200: "b200-training",
    GpuFamily.A100: "a100-mixed",
    GpuFamily.RTX_PRO_6000: "rtx-pro6000",
}


def node_pool_yaml(spec: NodePoolSpec) -> str:
    """Render the node-pool stanza for a single pool (spec §3.1)."""
    p = profile(spec.family)
    return (
        f"  {_POOL_NAMES[spec.family]}:\n"
        f"    count: {spec.node_count}\n"
        f"    gpuPerNode: {spec.gpu_per_node}\n"
        f"    labels:\n"
        f"      oxidize.io/gpu-family: {p.family.slug}\n"
        f"      oxidize.io/gpu-arch: {p.generation}\n"
        f"      oxidize.io/workload-type: {p.workload_type}\n"
        f"      oxidize.io/network-class: {p.network_class}\n"
        f"    taints:\n"
        f"      - key: oxidize.io/gpu\n"
        f"        value: {p.family.slug}\n"
        f"        effect: NoSchedule\n"
    )


def node_pools_yaml(specs: list[NodePoolSpec]) -> str:
    return "nodePools:\n" + "".join(node_pool_yaml(s) for s in specs)


def node_labels(family: GpuFamily, gpu_count: int) -> list[tuple[str, str]]:
    """Render the GFD/scheduling labels a node must carry (spec §3.2)."""
    p = profile(family)
    return [
        ("nvidia.com/gpu.present", "true"),
        ("nvidia.com/gpu.product", p.product),
        ("nvidia.com/gpu.count", str(gpu_count)),
        ("nvidia.com/gpu.memory", str(p.memory_mib)),
        ("nvidia.com/mig.capable", "true" if p.mig_capable else "false"),
        ("oxidize.io/gpu-family", p.family.slug),
        ("oxidize.io/gpu-generation", p.generation),
        ("oxidize.io/network-class", p.network_class),
    ]


def device_plugin_config_yaml(families: list[GpuFamily]) -> str:
    """Render the device-plugin time-slicing ConfigMap (spec §4.3)."""
    out = (
        "apiVersion: v1\n"
        "kind: ConfigMap\n"
        "metadata:\n"
        "  name: nvidia-device-plugin-config\n"
        "  namespace: kube-system\n"
        "data:\n"
        "  config.yaml: |\n"
        "    version: v1\n"
        "    sharing:\n"
        "      timeSlicing:\n"
        "        renameByDefault: false\n"
        "        failRequestsGreaterThanOne: true\n"
        "        resources:\n"
        "          - name: nvidia.com/gpu\n"
        "            replicas: 1\n"
    )
    overrides = [f for f in families if profile(f).time_slice_replicas > 1]
    if overrides:
        out += "    nodes:\n"
        for f in overrides:
            p = profile(f)
            out += (
                "      - match:\n"
                "          - key: oxidize.io/gpu-family\n"
                "            operator: In\n"
                "            values:\n"
                f"              - {p.family.slug}\n"
                "        sharing:\n"
                "          timeSlicing:\n"
                "            renameByDefault: true\n"
                "            resources:\n"
                "              - name: nvidia.com/gpu\n"
                f"                replicas: {p.time_slice_replicas}\n"
            )
    return out


@dataclass(frozen=True)
class MigProfile:
    name: str
    memory_gb: int
    compute_sms: int
    best_for: str


def mig_profiles() -> list[MigProfile]:
    """Recommended A100 MIG geometries (spec §4.4)."""
    return [
        MigProfile("1g.10gb", 10, 14, "light inference, micro-services"),
        MigProfile("2g.20gb", 20, 28, "medium inference, small training"),
        MigProfile("3g.40gb", 40, 42, "large model inference, fine-tuning"),
        MigProfile("4g.40gb", 40, 56, "heavy inference, data processing"),
        MigProfile("7g.80gb", 80, 108, "large training (disable MIG)"),
    ]


def mig_config_yaml(family: GpuFamily) -> str | None:
    """Render the MIG strategy ConfigMap, or ``None`` if not MIG-capable."""
    p = profile(family)
    if not p.mig_capable:
        return None
    return (
        "apiVersion: v1\n"
        "kind: ConfigMap\n"
        "metadata:\n"
        "  name: nvidia-mig-config\n"
        "  namespace: kube-system\n"
        "data:\n"
        "  config.yaml: |\n"
        "    version: v1\n"
        "    flags:\n"
        "      migStrategy: mixed\n"
        "    nodes:\n"
        "      - match:\n"
        "          - key: nvidia.com/gpu.product\n"
        "            operator: In\n"
        "            values:\n"
        f"              - {p.product}\n"
        "        mig:\n"
        "          strategy: mixed\n"
    )


def gpu_tolerations_yaml() -> str:
    """Standard pod tolerations required for any GPU workload (spec §3.3)."""
    return (
        "tolerations:\n"
        '  - key: "oxidize.io/gpu"\n'
        '    operator: "Exists"\n'
        '    effect: "NoSchedule"\n'
        '  - key: "nvidia.com/gpu"\n'
        '    operator: "Exists"\n'
        '    effect: "NoSchedule"\n'
    )


def helm_values_yaml(family: GpuFamily) -> str:
    """Render the GPU-Operator Helm values for a family (spec §5.2)."""
    p = profile(family)
    driver = "550.54.15" if p.generation == "blackwell" else "535.161.08"
    open_modules = "true" if p.generation == "blackwell" else "false"
    mig = "true" if p.mig_capable else "false"
    return (
        f"# GPU-Operator values for the {p.family.slug} ({p.generation}) node pool\n"
        "driver:\n"
        "  enabled: true\n"
        f'  version: "{driver}"\n'
        f"  useOpenKernelModules: {open_modules}\n"
        "toolkit:\n"
        "  enabled: true\n"
        "devicePlugin:\n"
        "  enabled: true\n"
        "  config:\n"
        "    name: nvidia-device-plugin-config\n"
        "dcgmExporter:\n"
        "  enabled: true\n"
        "  serviceMonitor:\n"
        "    enabled: true\n"
        "migManager:\n"
        f"  enabled: {mig}\n"
    )


def prometheus_rules_yaml() -> str:
    """Render the DCGM PrometheusRule for GPU health alerts (spec §8.1)."""
    return (
        "apiVersion: monitoring.coreos.com/v1\n"
        "kind: PrometheusRule\n"
        "metadata:\n"
        "  name: oxidize-gpu-alerts\n"
        "spec:\n"
        "  groups:\n"
        "    - name: gpu-health\n"
        "      rules:\n"
        "        - alert: GPUHighTemperature\n"
        "          expr: dcgm_gpu_temp > 85\n"
        "          for: 5m\n"
        "          labels:\n"
        "            severity: critical\n"
        "        - alert: GPUMemoryNearExhaustion\n"
        "          expr: dcgm_fb_used / (dcgm_fb_free + dcgm_fb_used) > 0.95\n"
        "          for: 10m\n"
        "          labels:\n"
        "            severity: warning\n"
        "        - alert: NVLinkError\n"
        "          expr: dcgm_nvlink_replay_error_count_total > 0\n"
        "          for: 1m\n"
        "          labels:\n"
        "            severity: critical\n"
    )


@dataclass(frozen=True)
class DetectedGpu:
    index: int
    name: str
    memory_total_mib: int
    mig_enabled: bool
    family: GpuFamily | None


def classify_product(name: str) -> GpuFamily | None:
    """Classify an NVML/nvidia-smi product name into a :class:`GpuFamily`."""
    n = name.lower()
    if "b200" in n:
        return GpuFamily.B200
    if "a100" in n:
        return GpuFamily.A100
    if "rtx" in n and "pro" in n and "6000" in n:
        return GpuFamily.RTX_PRO_6000
    return None


def parse_nvidia_smi_csv(output: str) -> list[DetectedGpu]:
    """Parse the CSV output of ``nvidia-smi --query-gpu=index,name,memory.total,mig.mode.current``.

    Lines that don't parse are skipped rather than aborting the whole probe.
    """
    gpus: list[DetectedGpu] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < 3:
            continue
        try:
            index = int(fields[0])
            memory_total_mib = int(fields[2])
        except ValueError:
            continue
        mig_enabled = len(fields) > 3 and fields[3].lower() == "enabled"
        gpus.append(
            DetectedGpu(
                index=index,
                name=fields[1],
                memory_total_mib=memory_total_mib,
                mig_enabled=mig_enabled,
                family=classify_product(fields[1]),
            )
        )
    return gpus


def detect_gpus() -> list[DetectedGpu]:
    """Probe the local node for NVIDIA GPUs via ``nvidia-smi``.

    Returns an empty list when ``nvidia-smi`` is unavailable or fails.
    """
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,mig.mode.current",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if out.returncode != 0:
        return []
    return parse_nvidia_smi_csv(out.stdout)


def summarize(gpus: list[DetectedGpu]) -> list[tuple[GpuFamily, int]]:
    """Count detected GPUs by family, in spec order."""
    out: list[tuple[GpuFamily, int]] = []
    for family in GpuFamily.all():
        n = sum(1 for g in gpus if g.family is family)
        if n > 0:
            out.append((family, n))
    return out
