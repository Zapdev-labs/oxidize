from oxidize_python import gpucluster as gc


def test_family_slug_roundtrips():
    for f in gc.GpuFamily.all():
        assert gc.GpuFamily.from_slug(f.slug) is f
    assert gc.GpuFamily.from_slug("RtxPro6000") is gc.GpuFamily.RTX_PRO_6000
    assert gc.GpuFamily.from_slug("unknown") is None


def test_profiles_match_spec():
    b = gc.profile(gc.GpuFamily.B200)
    assert b.memory_mib == 196608
    assert b.tdp_watts == 1000
    assert b.nvlink and not b.mig_capable and b.time_slice_replicas == 1

    a = gc.profile(gc.GpuFamily.A100)
    assert a.mig_capable and a.time_slice_replicas == 2
    assert a.product == "NVIDIA-A100-SXM4-80GB"

    r = gc.profile(gc.GpuFamily.RTX_PRO_6000)
    assert not r.nvlink and r.time_slice_replicas == 8 and r.network_class == "ethernet"


def test_node_pool_yaml():
    y = gc.node_pool_yaml(gc.NodePoolSpec(gc.GpuFamily.B200, 8, 8))
    for want in (
        "b200-training:",
        "count: 8",
        "oxidize.io/gpu-family: b200",
        "oxidize.io/gpu-arch: blackwell",
        "key: oxidize.io/gpu",
        "effect: NoSchedule",
    ):
        assert want in y


def test_node_pools_yaml_lists_all():
    specs = [
        gc.NodePoolSpec(gc.GpuFamily.B200, 8, 8),
        gc.NodePoolSpec(gc.GpuFamily.A100, 16, 8),
        gc.NodePoolSpec(gc.GpuFamily.RTX_PRO_6000, 4, 2),
    ]
    y = gc.node_pools_yaml(specs)
    assert y.startswith("nodePools:\n")
    for want in ("b200-training:", "a100-mixed:", "rtx-pro6000:"):
        assert want in y


def test_node_labels():
    m = dict(gc.node_labels(gc.GpuFamily.A100, 8))
    assert m["nvidia.com/gpu.product"] == "NVIDIA-A100-SXM4-80GB"
    assert m["nvidia.com/gpu.count"] == "8"
    assert m["nvidia.com/gpu.memory"] == "81920"
    assert m["nvidia.com/mig.capable"] == "true"
    assert m["oxidize.io/gpu-generation"] == "ampere"


def test_device_plugin_config_overrides():
    b = gc.device_plugin_config_yaml([gc.GpuFamily.B200])
    assert "failRequestsGreaterThanOne: true" in b
    assert "nodes:" not in b

    y = gc.device_plugin_config_yaml([gc.GpuFamily.RTX_PRO_6000, gc.GpuFamily.A100])
    for want in ("nodes:", "- rtx-pro-6000", "replicas: 8", "- a100", "replicas: 2"):
        assert want in y


def test_mig_config_only_for_a100():
    assert gc.mig_config_yaml(gc.GpuFamily.A100) is not None
    assert gc.mig_config_yaml(gc.GpuFamily.B200) is None
    assert gc.mig_config_yaml(gc.GpuFamily.RTX_PRO_6000) is None
    y = gc.mig_config_yaml(gc.GpuFamily.A100)
    assert "migStrategy: mixed" in y
    assert "NVIDIA-A100-SXM4-80GB" in y


def test_mig_profiles():
    names = [p.name for p in gc.mig_profiles()]
    assert names == ["1g.10gb", "2g.20gb", "3g.40gb", "4g.40gb", "7g.80gb"]


def test_helm_values_blackwell_driver():
    b = gc.helm_values_yaml(gc.GpuFamily.B200)
    assert "550.54.15" in b
    assert "useOpenKernelModules: true" in b
    assert "migManager:\n  enabled: false" in b

    a = gc.helm_values_yaml(gc.GpuFamily.A100)
    assert "migManager:\n  enabled: true" in a


def test_tolerations_and_prometheus():
    assert "oxidize.io/gpu" in gc.gpu_tolerations_yaml()
    rules = gc.prometheus_rules_yaml()
    assert "GPUHighTemperature" in rules
    assert "dcgm_nvlink_replay_error_count_total > 0" in rules


def test_classify_product():
    assert gc.classify_product("NVIDIA B200") is gc.GpuFamily.B200
    assert gc.classify_product("NVIDIA A100-SXM4-80GB") is gc.GpuFamily.A100
    assert gc.classify_product("NVIDIA A100-PCIE-40GB") is gc.GpuFamily.A100
    assert gc.classify_product("NVIDIA RTX PRO 6000") is gc.GpuFamily.RTX_PRO_6000
    assert gc.classify_product("Tesla V100") is None


def test_parse_nvidia_smi_csv():
    out = (
        "0, NVIDIA A100-SXM4-80GB, 81920, Enabled\n"
        "1, NVIDIA B200, 196608, Disabled\n"
        "garbage line\n"
        "2, Tesla V100, 16384, [N/A]\n"
    )
    gpus = gc.parse_nvidia_smi_csv(out)
    assert len(gpus) == 3
    assert gpus[0].family is gc.GpuFamily.A100 and gpus[0].mig_enabled
    assert gpus[1].family is gc.GpuFamily.B200 and not gpus[1].mig_enabled
    assert gpus[2].family is None and gpus[2].memory_total_mib == 16384


def test_summarize():
    gpus = gc.parse_nvidia_smi_csv(
        "0, NVIDIA A100-SXM4-80GB, 81920, Disabled\n"
        "1, NVIDIA A100-SXM4-80GB, 81920, Disabled\n"
        "2, NVIDIA B200, 196608, Disabled\n"
    )
    assert gc.summarize(gpus) == [(gc.GpuFamily.B200, 1), (gc.GpuFamily.A100, 2)]


def test_detect_gpus_safe_without_hardware():
    # Must not raise regardless of whether nvidia-smi exists.
    assert isinstance(gc.detect_gpus(), list)
