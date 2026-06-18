//! GPU cluster modeling, Kubernetes manifest generation, and runtime detection.
//!
//! This module implements the Oxidize GPU Cluster specification
//! (`docs/gpu_cluster_spec.md`) as code. It provides two cooperating halves:
//!
//! 1. **Manifest generation** — typed [`GpuProfile`]s for the three target GPU
//!    tiers (B200 / A100 / RTX Pro 6000) and pure functions that render the
//!    Kubernetes / Helm YAML the spec describes (node pools, taints & labels,
//!    NVIDIA device-plugin time-slicing, MIG strategy, Prometheus rules, and
//!    GPU-Operator Helm values).
//! 2. **Runtime detection** — [`detect_gpus`] queries `nvidia-smi` to enumerate
//!    physical GPUs present on the node, classifying each into a [`GpuFamily`].
//!    All parsing/classification logic is pure and unit-tested without
//!    requiring NVIDIA hardware; only the live probe needs a real GPU.
//!
//! YAML is emitted via string building on purpose: the workspace pulls in no
//! YAML serializer, and hand-emission keeps this module dependency-free while
//! producing output that matches the spec verbatim.

use std::fmt;
use std::process::Command;

/// The three GPU tiers the Oxidize cluster targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuFamily {
    /// NVIDIA B200 (Blackwell) — HPC / large-scale training.
    B200,
    /// NVIDIA A100 (Ampere) — datacenter inference & training, MIG-capable.
    A100,
    /// NVIDIA RTX Pro 6000 — professional workstation / edge inference.
    RtxPro6000,
}

impl GpuFamily {
    /// All known families, in spec order.
    pub fn all() -> [GpuFamily; 3] {
        [GpuFamily::B200, GpuFamily::A100, GpuFamily::RtxPro6000]
    }

    /// Relative capability rank (higher = higher-end). Used to pick the
    /// best GPU on mixed-family hosts independent of enumeration order.
    pub fn rank(self) -> u8 {
        match self {
            GpuFamily::B200 => 3,
            GpuFamily::A100 => 2,
            GpuFamily::RtxPro6000 => 1,
        }
    }

    /// The `oxidize.io/gpu-family` label value.
    pub fn slug(self) -> &'static str {
        match self {
            GpuFamily::B200 => "b200",
            GpuFamily::A100 => "a100",
            GpuFamily::RtxPro6000 => "rtx-pro-6000",
        }
    }

    /// Parse a family from its slug (label value), case-insensitively.
    pub fn from_slug(s: &str) -> Option<GpuFamily> {
        match s.trim().to_ascii_lowercase().as_str() {
            "b200" => Some(GpuFamily::B200),
            "a100" => Some(GpuFamily::A100),
            "rtx-pro-6000" | "rtx-pro6000" | "rtxpro6000" => Some(GpuFamily::RtxPro6000),
            _ => None,
        }
    }
}

impl fmt::Display for GpuFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.slug())
    }
}

/// Static hardware/scheduling profile for a GPU tier.
///
/// Values mirror the spec's "Target GPU Hardware" and device-plugin sections.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuProfile {
    pub family: GpuFamily,
    /// Exact NVML product name, e.g. `NVIDIA-A100-SXM4-80GB`.
    pub product: &'static str,
    /// Architecture shorthand for the `oxidize.io/gpu-generation` label.
    pub generation: &'static str,
    /// Onboard memory in MiB (the unit GFD reports via `nvidia.com/gpu.memory`).
    pub memory_mib: u32,
    /// Thermal design power (max) in watts.
    pub tdp_watts: u32,
    /// Whether NVLink is present.
    pub nvlink: bool,
    /// Whether the GPU supports MIG partitioning.
    pub mig_capable: bool,
    /// Device-plugin time-slicing replica count (1 == sharing disabled).
    pub time_slice_replicas: u32,
    /// Interconnect class for the `oxidize.io/network-class` label.
    pub network_class: &'static str,
    /// Default workload-type label.
    pub workload_type: &'static str,
}

/// Return the canonical [`GpuProfile`] for a family.
pub fn profile(family: GpuFamily) -> GpuProfile {
    match family {
        GpuFamily::B200 => GpuProfile {
            family,
            product: "NVIDIA-B200",
            generation: "blackwell",
            memory_mib: 196_608, // 192 GiB HBM3e
            tdp_watts: 1000,
            nvlink: true,
            mig_capable: false,
            time_slice_replicas: 1, // full-GPU only; failRequestsGreaterThanOne
            network_class: "infiniband",
            workload_type: "training",
        },
        GpuFamily::A100 => GpuProfile {
            family,
            product: "NVIDIA-A100-SXM4-80GB",
            generation: "ampere",
            memory_mib: 81_920, // 80 GiB HBM2e
            tdp_watts: 400,
            nvlink: true,
            mig_capable: true,
            time_slice_replicas: 2, // conservative for mixed workloads
            network_class: "infiniband",
            workload_type: "mixed",
        },
        GpuFamily::RtxPro6000 => GpuProfile {
            family,
            product: "NVIDIA-RTX-Pro-6000",
            generation: "ada",
            memory_mib: 98_304, // up to 96 GiB GDDR6
            tdp_watts: 300,
            nvlink: false,
            mig_capable: false,
            time_slice_replicas: 8, // dense inference sharing
            network_class: "ethernet",
            workload_type: "workstation",
        },
    }
}

/// Profiles for every family.
pub fn all_profiles() -> Vec<GpuProfile> {
    GpuFamily::all().into_iter().map(profile).collect()
}

// ---------------------------------------------------------------------------
// Manifest generation
// ---------------------------------------------------------------------------

/// A request to size a node pool of a given GPU family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodePoolSpec {
    pub family: GpuFamily,
    /// Number of nodes in the pool.
    pub node_count: u32,
    /// Physical GPUs per node.
    pub gpu_per_node: u32,
}

impl NodePoolSpec {
    pub fn new(family: GpuFamily, node_count: u32, gpu_per_node: u32) -> Self {
        Self {
            family,
            node_count,
            gpu_per_node,
        }
    }
}

/// Render the node-pool YAML stanza for a pool (matches spec §3.1).
pub fn node_pool_yaml(spec: &NodePoolSpec) -> String {
    let p = profile(spec.family);
    let pool_name = match spec.family {
        GpuFamily::B200 => "b200-training",
        GpuFamily::A100 => "a100-mixed",
        GpuFamily::RtxPro6000 => "rtx-pro6000",
    };
    format!(
        "  {pool_name}:\n\
         \x20   count: {count}\n\
         \x20   gpuPerNode: {gpn}\n\
         \x20   labels:\n\
         \x20     oxidize.io/gpu-family: {family}\n\
         \x20     oxidize.io/gpu-arch: {arch}\n\
         \x20     oxidize.io/workload-type: {workload}\n\
         \x20     oxidize.io/network-class: {net}\n\
         \x20   taints:\n\
         \x20     - key: oxidize.io/gpu\n\
         \x20       value: {family}\n\
         \x20       effect: NoSchedule\n",
        pool_name = pool_name,
        count = spec.node_count,
        gpn = spec.gpu_per_node,
        family = p.family.slug(),
        arch = p.generation,
        workload = p.workload_type,
        net = p.network_class,
    )
}

/// Render the full `nodePools:` document for a set of pools.
pub fn node_pools_yaml(specs: &[NodePoolSpec]) -> String {
    let mut out = String::from("nodePools:\n");
    for spec in specs {
        out.push_str(&node_pool_yaml(spec));
    }
    out
}

/// Render the GFD/scheduling labels a node of this family must carry (§3.2).
pub fn node_labels(family: GpuFamily, gpu_count: u32) -> Vec<(String, String)> {
    let p = profile(family);
    vec![
        ("nvidia.com/gpu.present".into(), "true".into()),
        ("nvidia.com/gpu.product".into(), p.product.into()),
        ("nvidia.com/gpu.count".into(), gpu_count.to_string()),
        ("nvidia.com/gpu.memory".into(), p.memory_mib.to_string()),
        ("nvidia.com/mig.capable".into(), p.mig_capable.to_string()),
        ("oxidize.io/gpu-family".into(), p.family.slug().into()),
        ("oxidize.io/gpu-generation".into(), p.generation.into()),
        ("oxidize.io/network-class".into(), p.network_class.into()),
    ]
}

/// Render the NVIDIA device-plugin time-slicing ConfigMap (§4.3).
///
/// The default `sharing` block is conservative (no sharing); per-family
/// overrides are emitted only for families whose `time_slice_replicas > 1`.
pub fn device_plugin_config_yaml(families: &[GpuFamily]) -> String {
    let mut out = String::new();
    out.push_str(
        "apiVersion: v1\n\
         kind: ConfigMap\n\
         metadata:\n\
         \x20 name: nvidia-device-plugin-config\n\
         \x20 namespace: kube-system\n\
         data:\n\
         \x20 config.yaml: |\n\
         \x20   version: v1\n\
         \x20   sharing:\n\
         \x20     timeSlicing:\n\
         \x20       renameByDefault: false\n\
         \x20       failRequestsGreaterThanOne: true\n\
         \x20       resources:\n\
         \x20         - name: nvidia.com/gpu\n\
         \x20           replicas: 1\n",
    );

    let overrides: Vec<GpuFamily> = families
        .iter()
        .copied()
        .filter(|f| profile(*f).time_slice_replicas > 1)
        .collect();

    if !overrides.is_empty() {
        out.push_str("    nodes:\n");
        for family in overrides {
            let p = profile(family);
            out.push_str(&format!(
                "      - match:\n\
                 \x20         - key: oxidize.io/gpu-family\n\
                 \x20           operator: In\n\
                 \x20           values:\n\
                 \x20             - {family}\n\
                 \x20       sharing:\n\
                 \x20         timeSlicing:\n\
                 \x20           renameByDefault: true\n\
                 \x20           resources:\n\
                 \x20             - name: nvidia.com/gpu\n\
                 \x20               replicas: {replicas}\n",
                family = p.family.slug(),
                replicas = p.time_slice_replicas,
            ));
        }
    }
    out
}

/// MIG profile recommendation (§4.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MigProfile {
    pub name: &'static str,
    pub memory_gb: u32,
    pub compute_sms: u32,
    pub best_for: &'static str,
}

/// The MIG geometries the spec recommends for the A100.
pub fn mig_profiles() -> Vec<MigProfile> {
    vec![
        MigProfile {
            name: "1g.10gb",
            memory_gb: 10,
            compute_sms: 14,
            best_for: "light inference, micro-services",
        },
        MigProfile {
            name: "2g.20gb",
            memory_gb: 20,
            compute_sms: 28,
            best_for: "medium inference, small training",
        },
        MigProfile {
            name: "3g.40gb",
            memory_gb: 40,
            compute_sms: 42,
            best_for: "large model inference, fine-tuning",
        },
        MigProfile {
            name: "4g.40gb",
            memory_gb: 40,
            compute_sms: 56,
            best_for: "heavy inference, data processing",
        },
        MigProfile {
            name: "7g.80gb",
            memory_gb: 80,
            compute_sms: 108,
            best_for: "large training (disable MIG)",
        },
    ]
}

/// Render the MIG strategy ConfigMap for A100 nodes (§4.4).
///
/// Returns `None` for families that are not MIG-capable.
pub fn mig_config_yaml(family: GpuFamily) -> Option<String> {
    let p = profile(family);
    if !p.mig_capable {
        return None;
    }
    Some(format!(
        "apiVersion: v1\n\
         kind: ConfigMap\n\
         metadata:\n\
         \x20 name: nvidia-mig-config\n\
         \x20 namespace: kube-system\n\
         data:\n\
         \x20 config.yaml: |\n\
         \x20   version: v1\n\
         \x20   flags:\n\
         \x20     migStrategy: mixed\n\
         \x20   nodes:\n\
         \x20     - match:\n\
         \x20         - key: nvidia.com/gpu.product\n\
         \x20           operator: In\n\
         \x20           values:\n\
         \x20             - {product}\n\
         \x20       mig:\n\
         \x20         strategy: mixed\n",
        product = p.product,
    ))
}

/// Render the standard pod tolerations required for any GPU workload (§3.3).
pub fn gpu_tolerations_yaml() -> String {
    String::from(
        "tolerations:\n\
         \x20 - key: \"oxidize.io/gpu\"\n\
         \x20   operator: \"Exists\"\n\
         \x20   effect: \"NoSchedule\"\n\
         \x20 - key: \"nvidia.com/gpu\"\n\
         \x20   operator: \"Exists\"\n\
         \x20   effect: \"NoSchedule\"\n",
    )
}

/// Render the GPU-Operator Helm `values.yaml` for a family (§5.2).
///
/// The driver version floor is raised for Blackwell (B200).
pub fn helm_values_yaml(family: GpuFamily) -> String {
    let p = profile(family);
    // Blackwell requires the 550.x driver series.
    let driver_version = if p.generation == "blackwell" {
        "550.54.15"
    } else {
        "535.161.08"
    };
    let open_modules = p.generation == "blackwell";
    format!(
        "# GPU-Operator values for the {family} ({arch}) node pool\n\
         driver:\n\
         \x20 enabled: true\n\
         \x20 version: \"{driver}\"\n\
         \x20 useOpenKernelModules: {open}\n\
         toolkit:\n\
         \x20 enabled: true\n\
         devicePlugin:\n\
         \x20 enabled: true\n\
         \x20 config:\n\
         \x20   name: nvidia-device-plugin-config\n\
         dcgmExporter:\n\
         \x20 enabled: true\n\
         \x20 serviceMonitor:\n\
         \x20   enabled: true\n\
         migManager:\n\
         \x20 enabled: {mig}\n",
        family = p.family.slug(),
        arch = p.generation,
        driver = driver_version,
        open = open_modules,
        mig = p.mig_capable,
    )
}

/// Render the DCGM `PrometheusRule` for GPU health alerts (§8.1).
pub fn prometheus_rules_yaml() -> String {
    String::from(
        "apiVersion: monitoring.coreos.com/v1\n\
         kind: PrometheusRule\n\
         metadata:\n\
         \x20 name: oxidize-gpu-alerts\n\
         spec:\n\
         \x20 groups:\n\
         \x20   - name: gpu-health\n\
         \x20     rules:\n\
         \x20       - alert: GPUHighTemperature\n\
         \x20         expr: dcgm_gpu_temp > 85\n\
         \x20         for: 5m\n\
         \x20         labels:\n\
         \x20           severity: critical\n\
         \x20       - alert: GPUMemoryNearExhaustion\n\
         \x20         expr: dcgm_fb_used / (dcgm_fb_free + dcgm_fb_used) > 0.95\n\
         \x20         for: 10m\n\
         \x20         labels:\n\
         \x20           severity: warning\n\
         \x20       - alert: NVLinkError\n\
         \x20         expr: dcgm_nvlink_replay_error_count_total > 0\n\
         \x20         for: 1m\n\
         \x20         labels:\n\
         \x20           severity: critical\n",
    )
}

// ---------------------------------------------------------------------------
// Runtime detection
// ---------------------------------------------------------------------------

/// A physical GPU discovered at runtime via `nvidia-smi`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetectedGpu {
    pub index: u32,
    pub name: String,
    pub memory_total_mib: u32,
    /// Whether MIG mode is currently enabled (best-effort).
    pub mig_enabled: bool,
    /// Family classification, if the product name is recognized.
    pub family: Option<GpuFamily>,
}

/// Classify an NVML/`nvidia-smi` product name into a [`GpuFamily`].
///
/// Matching is substring- and case-insensitive so that SKU variants
/// (e.g. `A100-SXM4-80GB`, `A100-PCIE-40GB`) all resolve correctly.
pub fn classify_product(name: &str) -> Option<GpuFamily> {
    let n = name.to_ascii_lowercase();
    if n.contains("b200") {
        Some(GpuFamily::B200)
    } else if n.contains("a100") {
        Some(GpuFamily::A100)
    } else if n.contains("rtx") && n.contains("pro") && n.contains("6000") {
        Some(GpuFamily::RtxPro6000)
    } else {
        None
    }
}

/// Parse the CSV output of
/// `nvidia-smi --query-gpu=index,name,memory.total,mig.mode.current
///  --format=csv,noheader,nounits`.
///
/// Lines that don't parse are skipped rather than aborting the whole probe.
pub fn parse_nvidia_smi_csv(output: &str) -> Vec<DetectedGpu> {
    let mut gpus = Vec::new();
    for line in output.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        if fields.len() < 3 {
            continue;
        }
        let Ok(index) = fields[0].parse::<u32>() else {
            continue;
        };
        let name = fields[1].to_string();
        let Ok(memory_total_mib) = fields[2].parse::<u32>() else {
            continue;
        };
        let mig_enabled = fields
            .get(3)
            .map(|s| s.eq_ignore_ascii_case("enabled"))
            .unwrap_or(false);
        let family = classify_product(&name);
        gpus.push(DetectedGpu {
            index,
            name,
            memory_total_mib,
            mig_enabled,
            family,
        });
    }
    gpus
}

/// Probe the local node for NVIDIA GPUs by invoking `nvidia-smi`.
///
/// Returns an empty vector if `nvidia-smi` is unavailable or fails — callers
/// running off a GPU node get a clean empty result rather than an error.
pub fn detect_gpus() -> Vec<DetectedGpu> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total,mig.mode.current",
            "--format=csv,noheader,nounits",
        ])
        .output();
    match output {
        Ok(out) if out.status.success() => {
            parse_nvidia_smi_csv(&String::from_utf8_lossy(&out.stdout))
        }
        _ => Vec::new(),
    }
}

/// Summarize detected GPUs into per-family counts (for labeling / capacity).
pub fn summarize(gpus: &[DetectedGpu]) -> Vec<(GpuFamily, u32)> {
    let mut counts: Vec<(GpuFamily, u32)> = Vec::new();
    for family in GpuFamily::all() {
        let n = gpus.iter().filter(|g| g.family == Some(family)).count() as u32;
        if n > 0 {
            counts.push((family, n));
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_slug_roundtrips() {
        for f in GpuFamily::all() {
            assert_eq!(GpuFamily::from_slug(f.slug()), Some(f));
        }
        assert_eq!(
            GpuFamily::from_slug("RtxPro6000"),
            Some(GpuFamily::RtxPro6000)
        );
        assert_eq!(GpuFamily::from_slug("unknown"), None);
    }

    #[test]
    fn profiles_match_spec() {
        let b200 = profile(GpuFamily::B200);
        assert_eq!(b200.memory_mib, 196_608);
        assert_eq!(b200.tdp_watts, 1000);
        assert!(b200.nvlink);
        assert!(!b200.mig_capable);
        assert_eq!(b200.time_slice_replicas, 1);

        let a100 = profile(GpuFamily::A100);
        assert!(a100.mig_capable);
        assert_eq!(a100.time_slice_replicas, 2);
        assert_eq!(a100.product, "NVIDIA-A100-SXM4-80GB");

        let rtx = profile(GpuFamily::RtxPro6000);
        assert!(!rtx.nvlink);
        assert_eq!(rtx.time_slice_replicas, 8);
        assert_eq!(rtx.network_class, "ethernet");
    }

    #[test]
    fn node_pool_yaml_has_taint_and_labels() {
        let y = node_pool_yaml(&NodePoolSpec::new(GpuFamily::B200, 8, 8));
        assert!(y.contains("b200-training:"));
        assert!(y.contains("count: 8"));
        assert!(y.contains("oxidize.io/gpu-family: b200"));
        assert!(y.contains("oxidize.io/gpu-arch: blackwell"));
        assert!(y.contains("key: oxidize.io/gpu"));
        assert!(y.contains("effect: NoSchedule"));
    }

    #[test]
    fn node_pools_yaml_lists_all() {
        let specs = vec![
            NodePoolSpec::new(GpuFamily::B200, 8, 8),
            NodePoolSpec::new(GpuFamily::A100, 16, 8),
            NodePoolSpec::new(GpuFamily::RtxPro6000, 4, 2),
        ];
        let y = node_pools_yaml(&specs);
        assert!(y.starts_with("nodePools:\n"));
        assert!(y.contains("b200-training:"));
        assert!(y.contains("a100-mixed:"));
        assert!(y.contains("rtx-pro6000:"));
    }

    #[test]
    fn node_labels_report_memory_and_mig() {
        let labels = node_labels(GpuFamily::A100, 8);
        let map: std::collections::HashMap<_, _> = labels.into_iter().collect();
        assert_eq!(map["nvidia.com/gpu.product"], "NVIDIA-A100-SXM4-80GB");
        assert_eq!(map["nvidia.com/gpu.count"], "8");
        assert_eq!(map["nvidia.com/gpu.memory"], "81920");
        assert_eq!(map["nvidia.com/mig.capable"], "true");
        assert_eq!(map["oxidize.io/gpu-generation"], "ampere");
    }

    #[test]
    fn device_plugin_config_emits_overrides_only_for_sharing_families() {
        // B200 alone -> no per-node overrides.
        let b = device_plugin_config_yaml(&[GpuFamily::B200]);
        assert!(b.contains("failRequestsGreaterThanOne: true"));
        assert!(!b.contains("nodes:"));

        // RTX + A100 -> overrides with their replica counts.
        let y = device_plugin_config_yaml(&[GpuFamily::RtxPro6000, GpuFamily::A100]);
        assert!(y.contains("nodes:"));
        assert!(y.contains("- rtx-pro-6000"));
        assert!(y.contains("replicas: 8"));
        assert!(y.contains("- a100"));
        assert!(y.contains("replicas: 2"));
    }

    #[test]
    fn mig_config_only_for_a100() {
        assert!(mig_config_yaml(GpuFamily::A100).is_some());
        assert!(mig_config_yaml(GpuFamily::B200).is_none());
        assert!(mig_config_yaml(GpuFamily::RtxPro6000).is_none());
        let y = mig_config_yaml(GpuFamily::A100).unwrap();
        assert!(y.contains("migStrategy: mixed"));
        assert!(y.contains("NVIDIA-A100-SXM4-80GB"));
    }

    #[test]
    fn mig_profiles_cover_spec_geometries() {
        let names: Vec<_> = mig_profiles().into_iter().map(|p| p.name).collect();
        assert_eq!(
            names,
            vec!["1g.10gb", "2g.20gb", "3g.40gb", "4g.40gb", "7g.80gb"]
        );
    }

    #[test]
    fn helm_values_raises_driver_for_blackwell() {
        let b = helm_values_yaml(GpuFamily::B200);
        assert!(b.contains("550.54.15"));
        assert!(b.contains("useOpenKernelModules: true"));
        assert!(b.contains("migManager:\n  enabled: false"));

        let a = helm_values_yaml(GpuFamily::A100);
        assert!(a.contains("migManager:\n  enabled: true"));
    }

    #[test]
    fn tolerations_and_prometheus_rules_render() {
        assert!(gpu_tolerations_yaml().contains("oxidize.io/gpu"));
        let rules = prometheus_rules_yaml();
        assert!(rules.contains("GPUHighTemperature"));
        assert!(rules.contains("dcgm_nvlink_replay_error_count_total > 0"));
    }

    #[test]
    fn classify_product_handles_sku_variants() {
        assert_eq!(classify_product("NVIDIA B200"), Some(GpuFamily::B200));
        assert_eq!(
            classify_product("NVIDIA A100-SXM4-80GB"),
            Some(GpuFamily::A100)
        );
        assert_eq!(
            classify_product("NVIDIA A100-PCIE-40GB"),
            Some(GpuFamily::A100)
        );
        assert_eq!(
            classify_product("NVIDIA RTX PRO 6000"),
            Some(GpuFamily::RtxPro6000)
        );
        // Regression: plain RTX 6000 (non-Pro) must NOT match.
        assert_eq!(classify_product("NVIDIA RTX 6000"), None);
        assert_eq!(classify_product("Tesla V100"), None);
    }

    #[test]
    fn parse_nvidia_smi_csv_parses_rows_and_skips_garbage() {
        let out = "0, NVIDIA A100-SXM4-80GB, 81920, Enabled\n\
                   1, NVIDIA B200, 196608, Disabled\n\
                   garbage line\n\
                   2, Tesla V100, 16384, [N/A]\n";
        let gpus = parse_nvidia_smi_csv(out);
        assert_eq!(gpus.len(), 3);
        assert_eq!(gpus[0].family, Some(GpuFamily::A100));
        assert!(gpus[0].mig_enabled);
        assert_eq!(gpus[1].family, Some(GpuFamily::B200));
        assert!(!gpus[1].mig_enabled);
        assert_eq!(gpus[2].family, None);
        assert_eq!(gpus[2].memory_total_mib, 16384);
    }

    #[test]
    fn summarize_counts_by_family() {
        let gpus = parse_nvidia_smi_csv(
            "0, NVIDIA A100-SXM4-80GB, 81920, Disabled\n\
             1, NVIDIA A100-SXM4-80GB, 81920, Disabled\n\
             2, NVIDIA B200, 196608, Disabled\n",
        );
        let summary = summarize(&gpus);
        assert_eq!(summary, vec![(GpuFamily::B200, 1), (GpuFamily::A100, 2)]);
    }

    #[test]
    fn detect_gpus_is_safe_without_hardware() {
        // Must not panic regardless of whether nvidia-smi exists.
        let _ = detect_gpus();
    }
}
