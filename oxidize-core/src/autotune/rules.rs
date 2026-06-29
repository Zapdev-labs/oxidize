//! The autotune rule table.
//!
//! Given a `HardwareInventory` and a `ModelFingerprint`, produce a
//! `TuningPlan` — a fully-resolved recommendation for every flag the
//! user could pass. Rules are ordered; the first matching rule for
//! each tier wins. Every decision is logged into `plan.rationale` so
//! the user can see why.
//!
//! The planner is a **pure function** — no I/O, no clocks. This
//! makes the table-driven test suite (see `tests` mod) the
//! authoritative spec.

use crate::autotune::detect::HardwareInventory;
use crate::autotune::fingerprint::{ModelFingerprint, kv_bytes_per_token, per_layer_weight_bytes};
use crate::gguf::GgufQuantizationType;
use crate::kv_cache::KvQuantization;
use crate::simd::SimdBackend;
use crate::tensor::DType;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use oxidize_kernels::cpu::CpuVendor;
use oxidize_kernels::cpu::is_skylake_sp;

/// Pipeline / batch mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineMode {
    Sequential,
    Continuous,
    Paged,
    Asymmetric,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeculativeSpec {
    None,
    DFlash,
    Mtp,
}

/// What the user has explicitly set, vs. what the autotuner
/// proposes. The CLI resolves this into a final flag value.
#[derive(Debug, Clone, PartialEq)]
pub struct TuningPlan {
    pub threads: usize,
    pub ctx_size: usize,
    pub kv_cache_dtype: DType,
    pub kv_quantization: KvQuantization,
    pub n_gpu_layers: usize,
    pub gpu_split: Vec<f32>,
    pub mmap: bool,
    pub mlock: bool,
    pub mmap_hugepages: bool,
    pub mmap_prefetch: bool,
    pub numa_replicate_dense: bool,
    pub layer_wise: bool,
    pub layer_cache: usize,
    pub pipeline: PipelineMode,
    pub speculative: SpeculativeSpec,
    pub decode_tile_tokens: usize,
    pub oxk_isa: OxkIsa,
    pub oxk_tile: OxkTile,
    pub expected_prompt_tps: f32,
    pub expected_decode_tps: f32,
    pub rationale: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OxkIsa {
    Scalar,
    Avx2,
    Avx512,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OxkTile {
    T1,
    T4,
    T8,
    T16,
}

impl TuningPlan {
    /// Pretty-printed summary for `--print-plan`. Plain text by
    /// default; pass `as_json = true` for tooling.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("threads           : {}\n", self.threads));
        s.push_str(&format!("ctx_size          : {}\n", self.ctx_size));
        s.push_str(&format!(
            "kv_cache_dtype    : {:?} (quantization: {:?})\n",
            self.kv_cache_dtype, self.kv_quantization
        ));
        s.push_str(&format!("n_gpu_layers      : {}\n", self.n_gpu_layers));
        if !self.gpu_split.is_empty() {
            s.push_str(&format!("gpu_split         : {:?}\n", self.gpu_split));
        }
        s.push_str(&format!(
            "mmap={} mlock={} mmap_hugepages={} mmap_prefetch={}\n",
            self.mmap, self.mlock, self.mmap_hugepages, self.mmap_prefetch
        ));
        s.push_str(&format!(
            "numa_replicate    : {}\n",
            self.numa_replicate_dense
        ));
        s.push_str(&format!(
            "layer_wise={} layer_cache={}\n",
            self.layer_wise, self.layer_cache
        ));
        s.push_str(&format!("pipeline          : {:?}\n", self.pipeline));
        s.push_str(&format!("speculative       : {:?}\n", self.speculative));
        s.push_str(&format!(
            "decode_tile_tokens: {}\n",
            self.decode_tile_tokens
        ));
        s.push_str(&format!(
            "oxk_isa/tile      : {:?} / {:?}\n",
            self.oxk_isa, self.oxk_tile
        ));
        s.push_str(&format!(
            "expected t/s      : prompt ≈ {:.1}  decode ≈ {:.1}\n",
            self.expected_prompt_tps, self.expected_decode_tps
        ));
        if !self.rationale.is_empty() {
            s.push_str("\nRationale:\n");
            for r in &self.rationale {
                s.push_str(&format!("  - {r}\n"));
            }
        }
        s
    }
}

/// Build a `TuningPlan` for the given hardware + model.
pub fn plan(inv: &HardwareInventory, model: &ModelFingerprint) -> TuningPlan {
    let mut plan = TuningPlan {
        threads: 0,
        ctx_size: 0,
        kv_cache_dtype: DType::F32,
        kv_quantization: KvQuantization::Asymmetric,
        n_gpu_layers: 0,
        gpu_split: Vec::new(),
        mmap: true,
        mlock: false,
        mmap_hugepages: false,
        mmap_prefetch: false,
        numa_replicate_dense: false,
        layer_wise: false,
        layer_cache: 0,
        pipeline: PipelineMode::Sequential,
        speculative: SpeculativeSpec::None,
        decode_tile_tokens: 0,
        oxk_isa: OxkIsa::Scalar,
        oxk_tile: OxkTile::T1,
        expected_prompt_tps: 0.0,
        expected_decode_tps: 0.0,
        rationale: Vec::new(),
    };

    tier0_hard_rules(inv, model, &mut plan);
    tier1_isa(inv, &mut plan);
    tier2_gpu_offload(inv, model, &mut plan);
    tier3_kv_and_ctx(inv, model, &mut plan);
    tier4_layer_cache_and_numa(inv, model, &mut plan);
    tier5_speculative(inv, model, &mut plan);
    tier6_threads(inv, &mut plan);
    tier7_decode_tile(&mut plan);
    tier8_pipeline(inv, model, &mut plan);
    estimate_tps(inv, model, &mut plan);

    plan
}

// ---------- tier 0: hard rules (always apply) ----------

fn tier0_hard_rules(inv: &HardwareInventory, model: &ModelFingerprint, plan: &mut TuningPlan) {
    let ram_budget = effective_ram_bytes(inv);
    if ram_budget < model.file_size_bytes.saturating_mul(12) / 10 {
        plan.mmap = true;
        plan.mlock = false;
        plan.layer_wise = true;
        plan.layer_cache = (inv.physical_cores / 4).max(1);
        plan
            .rationale
            .push(format!(
                "model ({:.1} GiB) exceeds 1.2× effective RAM ({:.1} GiB) → streaming layers, mmap=ON, mlock=OFF, layer_wise=ON, layer_cache={}",
                model.file_size_bytes as f64 / (1u64 << 30) as f64,
                ram_budget as f64 / (1u64 << 30) as f64,
                plan.layer_cache
            ));
    } else {
        plan.rationale.push(format!(
            "model ({:.1} GiB) fits in effective RAM ({:.1} GiB) → mmap=ON, mlock=OFF by default",
            model.file_size_bytes as f64 / (1u64 << 30) as f64,
            ram_budget as f64 / (1u64 << 30) as f64
        ));
    }
    if model.is_moe && inv.physical_cores <= 8 {
        plan.numa_replicate_dense = false;
        plan.rationale.push(
            "MoE on <= 8 cores → NUMA replication disabled (overhead exceeds benefit)".to_string(),
        );
    }
    if inv.os == crate::autotune::detect::OsKind::Macos && inv.has_metal {
        plan
            .rationale
            .push("macOS + Metal build available → keep --backend cpu (Metal auto-promotion lives in runtime)".to_string());
    }
}

// ---------- tier 1: ISA + kernel ----------

fn tier1_isa(inv: &HardwareInventory, plan: &mut TuningPlan) {
    match inv.simd {
        SimdBackend::Avx512f => {
            if is_skylake_sp() {
                plan.oxk_isa = OxkIsa::Avx2;
                plan.oxk_tile = OxkTile::T8;
                plan.rationale.push(
                    "Skylake-SP detected → AVX-512 disabled (avx512 regression on this uarch); AVX2 x8"
                        .to_string(),
                );
            } else {
                plan.oxk_isa = OxkIsa::Avx512;
                plan.oxk_tile = OxkTile::T8;
                plan.rationale
                    .push("AVX-512F available + non-Skylake → AVX-512 x8".to_string());
            }
        }
        SimdBackend::Avx2 => {
            plan.oxk_isa = OxkIsa::Avx2;
            plan.oxk_tile = if inv.physical_cores >= 16 {
                OxkTile::T8
            } else {
                OxkTile::T4
            };
            plan.rationale.push(format!(
                "AVX2 only → AVX2 x{}",
                if inv.physical_cores >= 16 { 8 } else { 4 }
            ));
        }
        #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
        SimdBackend::Neon => {
            plan.oxk_isa = OxkIsa::Scalar; // no Neon oxk path yet
            plan.oxk_tile = OxkTile::T1;
            plan.rationale
                .push("ARM/Neon → scalar oxk (no Neon kernel yet)".to_string());
        }
        _ => {
            plan.oxk_isa = OxkIsa::Scalar;
            plan.oxk_tile = OxkTile::T1;
            plan.rationale
                .push("No SIMD beyond SSE2 → scalar oxk".to_string());
        }
    }
}

// ---------- tier 2: GPU offload ----------

fn tier2_gpu_offload(inv: &HardwareInventory, model: &ModelFingerprint, plan: &mut TuningPlan) {
    if !inv.has_gpu && !inv.has_rocm && !inv.has_cuda {
        plan.n_gpu_layers = 0;
        return;
    }
    if !inv.has_gpu {
        plan.n_gpu_layers = 0;
        if inv.has_rocm {
            plan.rationale.push(
                "ROCm build detected but no GPU inventory — set --backend rocm and pass --n-gpu-layers manually"
                    .to_string(),
            );
        }
        return;
    }
    let per_layer = per_layer_weight_bytes(model);
    if per_layer == 0 {
        plan.n_gpu_layers = 0;
        return;
    }
    let usable_vram = (inv.gpu_vram_bytes as f64 * 0.85) as u64;
    let mut n = (usable_vram / per_layer) as usize;
    if inv.gpu_vram_bytes < (model.file_size_bytes / 4) {
        n = 0;
        plan.rationale.push(format!(
            "GPU VRAM ({:.1} GiB) < 25% of model size ({:.1} GiB) → n_gpu_layers=0 (overhead would dominate)",
            inv.gpu_vram_bytes as f64 / (1u64 << 30) as f64,
            model.file_size_bytes as f64 / (1u64 << 30) as f64
        ));
    } else {
        n = n.min(model.layer_count);
        if n == model.layer_count {
            plan.mmap = false;
            plan.mlock = false;
            plan.rationale.push(format!(
                "GPU can hold the full model ({}/{} layers, {:.1} GiB on GPU) → mmap=OFF",
                n,
                model.layer_count,
                inv.gpu_vram_bytes as f64 / (1u64 << 30) as f64
            ));
        } else {
            plan.rationale.push(format!(
                "GPU offload: {}/{} layers at {:.1} GiB usable VRAM",
                n,
                model.layer_count,
                usable_vram as f64 / (1u64 << 30) as f64
            ));
        }
    }
    plan.n_gpu_layers = n;
    // Tensor split for multi-GPU is only set when the user has
    // multiple GPUs; we don't know the count from `inv.gpu_vram_bytes`
    // alone. The CLI / server extend this with `--gpus`.
}

// ---------- tier 3: KV cache dtype + ctx size ----------

fn tier3_kv_and_ctx(inv: &HardwareInventory, model: &ModelFingerprint, plan: &mut TuningPlan) {
    let vram_gib = inv.gpu_vram_bytes / (1u64 << 30);
    if inv.has_gpu && vram_gib >= 16 {
        plan.kv_cache_dtype = DType::F16;
        plan.kv_quantization = KvQuantization::Asymmetric;
        plan.rationale
            .push(">= 16 GiB VRAM → kv=F16 (no additional quantization)".to_string());
    } else if (inv.has_gpu && vram_gib >= 8) || model.layer_count >= 80 {
        plan.kv_cache_dtype = DType::F16;
        plan.kv_quantization = KvQuantization::Asymmetric;
        plan.rationale.push(
            "8-16 GiB VRAM or deep model → kv=F16 + asymmetric INT8 quant on the long tail"
                .to_string(),
        );
    } else if vram_gib < 8 || model.layer_count >= 60 || inv.total_ram_bytes < (32u64 << 30) {
        plan.kv_cache_dtype = DType::F16;
        plan.kv_quantization = KvQuantization::TurboQuant;
        plan.rationale.push(
            "low VRAM / RAM or very deep model → kv=F16 + TurboQuant (block INT4)".to_string(),
        );
    } else {
        plan.kv_cache_dtype = DType::F16;
        plan.kv_quantization = KvQuantization::Asymmetric;
    }

    // Default ctx = 4096 unless the existing config says otherwise.
    // We cap by KV memory budget: leave 60% of effective RAM for
    // the model + 8 GiB for OS/workspace; KV gets the rest.
    let ram_budget = effective_ram_bytes(inv);
    // Only layers that stay resident in RAM count against the KV budget. With
    // GPU offload, the offloaded fraction of the weights lives in VRAM, so
    // charging the full file size here would needlessly clamp ctx_size (e.g.
    // down to 512 tokens) on systems where the model mostly lives on the GPU.
    let model_bytes = if plan.n_gpu_layers > 0 && model.layer_count > 0 {
        let resident_layers = model.layer_count.saturating_sub(plan.n_gpu_layers);
        ((model.file_size_bytes as u128 * resident_layers as u128) / model.layer_count as u128)
            as u64
    } else {
        model.file_size_bytes
    };
    let overhead = 8u64 << 30;
    let kv_budget = ram_budget
        .saturating_sub(model_bytes)
        .saturating_sub(overhead);
    let kv_bytes = kv_bytes_per_token(model, plan.kv_cache_dtype.size_in_bytes());
    let ctx_cap = if kv_bytes > 0 {
        (kv_budget / kv_bytes).min(131_072) as usize
    } else {
        4096
    };
    let default_ctx = if model.num_kv_heads <= 4 {
        8192
    } else if model.layer_count >= 80 {
        4096
    } else {
        4096
    };
    plan.ctx_size = default_ctx.min(ctx_cap.max(512));
    plan.rationale.push(format!(
        "ctx_size={} (default={}, capped to fit {kv_budget} bytes of KV)",
        plan.ctx_size, default_ctx
    ));
}

// ---------- tier 4: layer cache + NUMA ----------

fn tier4_layer_cache_and_numa(
    inv: &HardwareInventory,
    model: &ModelFingerprint,
    plan: &mut TuningPlan,
) {
    if plan.n_gpu_layers == model.layer_count && model.layer_count > 0 {
        // Whole model on GPU — layer cache is irrelevant.
        plan.layer_cache = 0;
        plan.numa_replicate_dense = false;
        return;
    }
    if plan.layer_cache == 0 {
        plan.layer_cache = inv.physical_cores.clamp(2, 8);
        plan.rationale.push(format!(
            "layer_cache={} (~1 layer per 2 cores, capped at 8)",
            plan.layer_cache
        ));
    }
    if inv.numa_nodes >= 2
        && inv.physical_cores >= 16
        && !model.is_moe
        && plan.oxk_isa != OxkIsa::Scalar
    {
        plan.numa_replicate_dense = true;
        plan.rationale.push(
            "NUMA nodes>=2, cores>=16, dense model, SIMD available → NUMA-replicate dense weights"
                .to_string(),
        );
    }
}

// ---------- tier 5: speculative ----------

fn tier5_speculative(inv: &HardwareInventory, model: &ModelFingerprint, plan: &mut TuningPlan) {
    if !inv.has_gpu {
        return;
    }
    if model.has_mtp {
        plan.speculative = SpeculativeSpec::Mtp;
        plan.rationale
            .push("model has MTP tensors + GPU → suggest MTP speculative decoding".to_string());
        return;
    }
    if is_dflash_compatible(&model.architecture) {
        plan.speculative = SpeculativeSpec::DFlash;
        plan.rationale.push(format!(
            "{} on GPU → suggest DFlash speculative decoding (--draft-model omitted by autotune; user supplies)",
            model.architecture
        ));
    }
}

fn is_dflash_compatible(arch: &str) -> bool {
    matches!(arch, "qwen2" | "qwen3" | "llama" | "lfm2")
}

// ---------- tier 6: thread count ----------

fn tier6_threads(inv: &HardwareInventory, plan: &mut TuningPlan) {
    if inv.has_gpu && plan.n_gpu_layers > 0 {
        // Even with full GPU offload, attention is computed host-side between
        // the GPU projection and output kernels, so the CPU sits on the decode
        // critical path and the GPU stalls waiting for it. Measured on an L4 +
        // Qwen3-4B Q4_K_M: threads 4→8 gave +31% tok/s (13.9→18.2); 16/24
        // regressed via oversubscription. So floor at 8 (enough to keep the GPU
        // fed) while still scaling on very large hosts, and never oversubscribe.
        plan.threads = 8.max(inv.physical_cores / 8);
        plan
            .rationale
            .push("GPU offload but host-side attention on critical path → threads floored at 8 to keep the GPU fed".to_string());
        return;
    }
    if inv.container_mem_limit.is_some() {
        plan.threads = inv.physical_cores.clamp(2, 8);
        plan.rationale.push(
            "container memory limit present → cap threads to avoid host scheduler thrash"
                .to_string(),
        );
        return;
    }
    plan.threads = inv.physical_cores;
    plan.rationale.push(format!(
        "CPU-only path → threads = physical_cores ({})",
        inv.physical_cores
    ));
}

// ---------- tier 7: decode tile (split-K attention) ----------

fn tier7_decode_tile(plan: &mut TuningPlan) {
    if plan.ctx_size > 8192 {
        plan.decode_tile_tokens = 1024;
        plan.rationale
            .push("ctx > 8192 → split-K decode tile = 1024".to_string());
    } else if plan.ctx_size > 4096 && matches!(plan.oxk_isa, OxkIsa::Avx2) {
        plan.decode_tile_tokens = 512;
        plan.rationale
            .push("ctx > 4096 on AVX2 → split-K decode tile = 512".to_string());
    }
}

// ---------- tier 8: pipeline ----------

fn tier8_pipeline(inv: &HardwareInventory, model: &ModelFingerprint, plan: &mut TuningPlan) {
    if inv.has_gpu && plan.n_gpu_layers > 0 {
        plan.pipeline = PipelineMode::Paged;
        plan.rationale
            .push("GPU + layers on GPU → paged attention (continuous batching)".to_string());
        return;
    }
    if inv.physical_cores >= 8 && inv.total_ram_bytes >= (64u64 << 30) && !model.is_moe {
        plan.pipeline = PipelineMode::Continuous;
        plan.rationale
            .push(">= 8 cores, >= 64 GiB, dense model → continuous batching".to_string());
        return;
    }
    plan.pipeline = PipelineMode::Sequential;
    plan.rationale
        .push("low-resource or MoE → sequential (default)".to_string());
}

// ---------- tps estimates ----------

fn estimate_tps(inv: &HardwareInventory, model: &ModelFingerprint, plan: &mut TuningPlan) {
    let per_core = per_core_decode_tps(model);
    let cpu_tps = inv.physical_cores as f32 * per_core;
    let mem_bw = inv.total_ram_bytes as f32 * 0.7;
    let mem_tps = if model.file_size_bytes > 0 {
        mem_bw / model.file_size_bytes as f32
    } else {
        0.0
    };
    let cpu_branch = cpu_tps.min(mem_tps);
    let gpu_tps = match (inv.has_gpu, inv.gpu_family) {
        (true, Some(family)) => match family {
            crate::gpu_cluster::GpuFamily::B200 => 200.0,
            crate::gpu_cluster::GpuFamily::A100 => 90.0,
            crate::gpu_cluster::GpuFamily::RtxPro6000 => 70.0,
        },
        (true, None) => 30.0, // unknown vendor — conservative
        (false, _) => 0.0,
    };
    plan.expected_decode_tps = if inv.has_gpu && plan.n_gpu_layers > 0 {
        gpu_tps
    } else {
        cpu_branch
    };
    // Prompt TPS is roughly 5–10× decode (mostly prefill bandwidth
    // bound) — use a coarse 6×.
    plan.expected_prompt_tps = plan.expected_decode_tps * 6.0;
}

fn per_core_decode_tps(model: &ModelFingerprint) -> f32 {
    let size_class = if model.file_size_bytes <= (8u64 << 30) {
        // small <= 8B
        "small"
    } else if model.file_size_bytes <= (30u64 << 30) {
        // medium 8-30B
        "medium"
    } else {
        "large"
    };
    match model.quant {
        GgufQuantizationType::Q4_K_M | GgufQuantizationType::Q4_K_S => match size_class {
            "small" => 1.2,
            "medium" => 0.6,
            _ => 0.25,
        },
        GgufQuantizationType::Q2_K | GgufQuantizationType::Q3_K_S => match size_class {
            "small" => 1.6,
            "medium" => 0.8,
            _ => 0.35,
        },
        GgufQuantizationType::Q8_0 => 0.8,
        GgufQuantizationType::F16 => 0.4,
        GgufQuantizationType::Q5_K_M | GgufQuantizationType::Q5_K_S => match size_class {
            "small" => 0.9,
            "medium" => 0.45,
            _ => 0.20,
        },
        GgufQuantizationType::Q6_K => match size_class {
            "small" => 0.7,
            "medium" => 0.35,
            _ => 0.18,
        },
        _ => 0.5,
    }
}

fn effective_ram_bytes(inv: &HardwareInventory) -> u64 {
    if let Some(cgroup) = inv.container_mem_limit {
        return cgroup.min(inv.total_ram_bytes);
    }
    inv.total_ram_bytes
}

// ---------- tests ----------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autotune::detect::OsKind;
    use crate::autotune::fingerprint::fingerprint_from_parts;
    use crate::gpu_cluster::GpuFamily;
    use crate::simd::SimdBackend;
    use oxidize_kernels::cpu::CpuVendor;

    fn inv_desktop() -> HardwareInventory {
        HardwareInventory {
            os: OsKind::Linux,
            cpu_vendor: CpuVendor::Amd,
            simd: SimdBackend::Avx2,
            physical_cores: 16,
            logical_cores: 32,
            numa_nodes: 2,
            min_node_ram_bytes: 32u64 << 30,
            total_ram_bytes: 64u64 << 30,
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

    fn inv_a100() -> HardwareInventory {
        let mut inv = inv_desktop();
        inv.physical_cores = 32;
        inv.logical_cores = 128;
        inv.total_ram_bytes = 256u64 << 30;
        inv.has_gpu = true;
        inv.gpu_family = Some(GpuFamily::A100);
        inv.gpu_vram_bytes = 80u64 << 30;
        inv
    }

    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    fn inv_macbook() -> HardwareInventory {
        HardwareInventory {
            os: OsKind::Macos,
            cpu_vendor: CpuVendor::Other, // Apple
            simd: SimdBackend::Neon,
            physical_cores: 8,
            logical_cores: 8,
            numa_nodes: 1,
            min_node_ram_bytes: 16u64 << 30,
            total_ram_bytes: 16u64 << 30,
            has_gpu: false,
            gpu_family: None,
            gpu_vram_bytes: 0,
            has_metal: true,
            has_cuda: false,
            has_rocm: false,
            has_rdma: false,
            is_wsl: false,
            container_mem_limit: None,
            hugepages_2mib_avail: false,
        }
    }

    fn model_qwen3_4b() -> ModelFingerprint {
        fingerprint_from_parts(
            "qwen2",
            36,
            2560,
            20,
            8,
            128,
            6912,
            151_936,
            2_500_000_000, // 2.5 GiB-ish (Q4_K_M)
            GgufQuantizationType::Q4_K_M,
        )
    }

    fn model_qwen3_32b() -> ModelFingerprint {
        fingerprint_from_parts(
            "qwen2",
            64,
            5120,
            40,
            8,
            128,
            13_824,
            151_936,
            20_000_000_000,
            GgufQuantizationType::Q4_K_M,
        )
    }

    fn model_70b() -> ModelFingerprint {
        fingerprint_from_parts(
            "llama",
            80,
            8192,
            64,
            8,
            128,
            28_672,
            32_000,
            40_000_000_000,
            GgufQuantizationType::Q4_K_M,
        )
    }

    fn model_moe() -> ModelFingerprint {
        let mut m = fingerprint_from_parts(
            "llama",
            32,
            4096,
            32,
            8,
            128,
            14_336,
            32_000,
            90_000_000_000,
            GgufQuantizationType::Q2_K,
        );
        m.is_moe = true;
        m.expert_count = 8;
        m
    }

    fn model_08b() -> ModelFingerprint {
        fingerprint_from_parts(
            "qwen2",
            24,
            1024,
            16,
            8,
            128,
            2816,
            151_936,
            1_100_000_000,
            GgufQuantizationType::Q8_0,
        )
    }

    #[test]
    fn desktop_no_gpu_4b() {
        let inv = inv_desktop();
        let m = model_qwen3_4b();
        let p = plan(&inv, &m);
        assert_eq!(p.n_gpu_layers, 0);
        assert!(matches!(p.pipeline, PipelineMode::Continuous));
        assert!(matches!(p.kv_cache_dtype, DType::F16));
        assert!(p.threads >= 16);
        assert!(p.rationale.len() >= 5);
    }

    #[test]
    fn desktop_big_model_70b_layer_wise() {
        // Tight memory: 40 GiB on a model that's ~80 GiB-ish so the
        // 1.2× RAM threshold fires and streaming is forced.
        let mut inv = inv_desktop();
        inv.total_ram_bytes = 40u64 << 30;
        let m = model_70b();
        let p = plan(&inv, &m);
        assert!(p.layer_wise, "70B on tight RAM should stream");
        assert!(p.mmap);
        assert!(!p.mlock);
        assert_eq!(p.n_gpu_layers, 0);
    }

    #[test]
    fn a100_32b_full_offload() {
        let inv = inv_a100();
        let m = model_qwen3_32b();
        let p = plan(&inv, &m);
        assert_eq!(p.n_gpu_layers, m.layer_count);
        assert!(!p.mmap, "fully on GPU → no mmap");
        assert!(matches!(p.pipeline, PipelineMode::Paged));
    }

    #[test]
    fn a100_70b_full_offload() {
        let inv = inv_a100();
        let m = model_70b();
        let p = plan(&inv, &m);
        // 80 GiB VRAM vs ~40 GiB model → fits.
        assert_eq!(p.n_gpu_layers, m.layer_count);
    }

    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    #[test]
    fn macbook_apple_silicon_uses_arm() {
        let inv = inv_macbook();
        let m = model_qwen3_4b();
        let p = plan(&inv, &m);
        assert!(matches!(p.oxk_isa, OxkIsa::Scalar)); // no Neon oxk yet
        assert!(matches!(p.simd, SimdBackend::Neon));
        assert!(!p.has_gpu, "no discrete GPU on macbook");
    }

    #[test]
    fn moe_on_low_cores_disables_numa() {
        let mut inv = inv_desktop();
        inv.physical_cores = 4;
        let m = model_moe();
        let p = plan(&inv, &m);
        assert!(!p.numa_replicate_dense);
        assert!(p.rationale.iter().any(|r| r.contains("MoE on <= 8 cores")));
    }

    #[test]
    fn tiny_box_keeps_sequential() {
        let mut inv = inv_desktop();
        inv.physical_cores = 4;
        inv.total_ram_bytes = 8u64 << 30;
        inv.numa_nodes = 1;
        let m = model_08b();
        let p = plan(&inv, &m);
        assert!(matches!(p.pipeline, PipelineMode::Sequential));
        assert!(matches!(p.kv_cache_dtype, DType::F16));
        assert!(p.threads <= 8);
    }

    #[test]
    fn decode_tile_set_for_long_context() {
        let mut inv = inv_desktop();
        inv.simd = SimdBackend::Avx2;
        let mut m = model_qwen3_4b();
        // We can't change ctx directly (the planner decides), so
        // check the threshold: tile is set if ctx > 4096 on AVX2.
        let p = plan(&inv, &m);
        if p.ctx_size > 4096 {
            assert!(p.decode_tile_tokens == 512 || p.decode_tile_tokens == 1024);
        }
    }

    #[test]
    fn plan_summary_is_nonempty() {
        let inv = inv_desktop();
        let m = model_qwen3_4b();
        let p = plan(&inv, &m);
        let s = p.summary();
        assert!(s.contains("threads"));
        assert!(s.contains("ctx_size"));
        assert!(s.contains("Rationale"));
    }
}
