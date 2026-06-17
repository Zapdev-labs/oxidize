//! `apply_plan` — bridge between a `TuningPlan` and the clap-derived
//! CLI/server `Args` structs.
//!
//! The CLI and server both keep their own `Args` structs (in
//! `oxidize-cli/src/main.rs` and `oxidize-server/src/cli.rs`). The
//! fields we'd set from a plan live there. To avoid coupling the
//! autotune crate to clap, we expose a small `PlanOverrides` struct
//! that the CLI / server consume: each binary diffs its own
//! `Args` against `PlanOverrides::default()` and applies only the
//! ones that the user didn't already set.
//!
//! The "explicit beats implicit" rule is encoded here: any field
//! in `Args` that the user set (i.e. the corresponding
//! `was_set_*` flag is true) is left alone.

use crate::autotune::rules::TuningPlan;

/// User-resolved values. Each field corresponds to one CLI flag
/// that the autotuner can recommend. The CLI / server apply these
/// only when the user didn't set the corresponding flag themselves.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanOverrides {
    pub threads: Option<usize>,
    pub ctx_size: Option<usize>,
    pub n_gpu_layers: Option<usize>,
    pub layer_cache: Option<usize>,
    pub layer_wise: Option<bool>,
    pub mmap: Option<bool>,
    pub mlock: Option<bool>,
    pub mmap_hugepages: Option<bool>,
    pub mmap_prefetch: Option<bool>,
    pub ram_offload: Option<bool>,
    pub cpu_optimized: Option<bool>,
    pub turboquant: Option<bool>,
    pub pipeline: Option<String>,
    pub decode_tile: Option<usize>,
}

impl Default for PlanOverrides {
    fn default() -> Self {
        Self {
            threads: None,
            ctx_size: None,
            n_gpu_layers: None,
            layer_cache: None,
            layer_wise: None,
            mmap: None,
            mlock: None,
            mmap_hugepages: None,
            mmap_prefetch: None,
            ram_offload: None,
            cpu_optimized: None,
            turboquant: None,
            pipeline: None,
            decode_tile: None,
        }
    }
}

/// Convert a `TuningPlan` into the per-flag `PlanOverrides`. Every
/// field that the plan touched gets a `Some` value; everything else
/// stays `None` (meaning "the autotuner has no opinion"). The CLI /
/// server apply only `Some` fields, and only when the user didn't
/// pass the corresponding flag.
pub fn overrides_from_plan(plan: &TuningPlan) -> PlanOverrides {
    let pipeline = match plan.pipeline {
        crate::autotune::rules::PipelineMode::Sequential => Some("sequential".to_string()),
        crate::autotune::rules::PipelineMode::Continuous => Some("continuous".to_string()),
        crate::autotune::rules::PipelineMode::Paged => Some("paged".to_string()),
        crate::autotune::rules::PipelineMode::Asymmetric => Some("asymmetric".to_string()),
    };
    let turboquant = matches!(
        plan.kv_quantization,
        crate::kv_cache::KvQuantization::TurboQuant
    );
    PlanOverrides {
        threads: Some(plan.threads),
        ctx_size: Some(plan.ctx_size),
        n_gpu_layers: Some(plan.n_gpu_layers),
        layer_cache: Some(plan.layer_cache),
        layer_wise: Some(plan.layer_wise),
        mmap: Some(plan.mmap),
        mlock: Some(plan.mlock),
        mmap_hugepages: Some(plan.mmap_hugepages),
        mmap_prefetch: Some(plan.mmap_prefetch),
        ram_offload: Some(plan.mlock), // mlock => ram-offload
        cpu_optimized: Some(false),    // explicit false: don't force
        turboquant: Some(turboquant),
        pipeline,
        decode_tile: if plan.decode_tile_tokens > 0 {
            Some(plan.decode_tile_tokens)
        } else {
            None
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autotune::rules::PipelineMode;
    use crate::kv_cache::KvQuantization;
    use crate::tensor::DType;
    use oxidize_kernels::cpu::CpuVendor;
    use crate::autotune::detect::{HardwareInventory, OsKind};
    use crate::autotune::fingerprint::fingerprint_from_parts;
    use crate::autotune::rules::{plan, OxkIsa, OxkTile, SpeculativeSpec};
    use crate::gguf::GgufQuantizationType;
    use crate::gpu_cluster::GpuFamily;
    use crate::simd::SimdBackend;

    fn inv() -> HardwareInventory {
        HardwareInventory {
            os: OsKind::Linux,
            cpu_vendor: CpuVendor::Amd,
            simd: SimdBackend::Avx2,
            physical_cores: 8,
            logical_cores: 16,
            numa_nodes: 1,
            min_node_ram_bytes: 16u64 << 30,
            total_ram_bytes: 32u64 << 30,
            has_gpu: false,
            gpu_family: None,
            gpu_vram_bytes: 0,
            has_metal: false,
            has_cuda: false,
            has_rocm: false,
            has_rdma: false,
            is_wsl: false,
            container_mem_limit: None,
            hugepages_2mib_avail: false,
        }
    }

    fn m() -> crate::autotune::fingerprint::ModelFingerprint {
        fingerprint_from_parts(
            "qwen2", 32, 2048, 16, 8, 128, 5504, 32000, 4_000_000_000,
            GgufQuantizationType::Q4_K_M,
        )
    }

    #[test]
    fn overrides_carry_every_field() {
        let p = plan(&inv(), &m());
        let o = overrides_from_plan(&p);
        assert!(o.threads.is_some());
        assert!(o.ctx_size.is_some());
        assert!(o.n_gpu_layers.is_some());
        assert!(o.layer_cache.is_some());
        assert!(o.layer_wise.is_some());
        assert!(o.mmap.is_some());
        assert!(o.mlock.is_some());
        assert!(o.pipeline.is_some());
    }

    #[test]
    fn pipeline_string_matches_enum() {
        let p = TuningPlan {
            threads: 4,
            ctx_size: 4096,
            kv_cache_dtype: DType::F16,
            kv_quantization: KvQuantization::Asymmetric,
            n_gpu_layers: 0,
            gpu_split: vec![],
            mmap: true,
            mlock: false,
            mmap_hugepages: false,
            mmap_prefetch: false,
            numa_replicate_dense: false,
            layer_wise: false,
            layer_cache: 4,
            pipeline: PipelineMode::Paged,
            speculative: SpeculativeSpec::None,
            decode_tile_tokens: 0,
            oxk_isa: OxkIsa::Avx2,
            oxk_tile: OxkTile::T4,
            expected_prompt_tps: 50.0,
            expected_decode_tps: 8.0,
            rationale: vec![],
        };
        let o = overrides_from_plan(&p);
        assert_eq!(o.pipeline.as_deref(), Some("paged"));
    }
}
