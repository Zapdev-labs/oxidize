use super::*;

pub(super) fn optimize_mapped_model_memory(mapped: &MappedGgufFile, args: &Args) {
    let apply_hints =
        args.cpu_optimized || args.ram_offload || args.mmap_prefetch || args.mmap_hugepages;
    if !apply_hints {
        return;
    }

    if let Err(error) = mapped.advise_random_access() {
        eprintln!("mmap random-access hint failed: {error}");
    }
    if (args.cpu_optimized || args.ram_offload || args.mmap_prefetch)
        && let Err(error) = mapped.advise_will_need()
    {
        eprintln!("mmap prefetch hint failed: {error}");
    }
    if (args.cpu_optimized || args.mmap_hugepages)
        && let Err(error) = mapped.advise_huge_pages()
    {
        eprintln!("mmap hugepage hint failed: {error}");
    }
    if args.ram_offload {
        let n_threads = if args.ram_offload_threads > 0 {
            args.ram_offload_threads
        } else {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8)
        };
        let gib = mapped.bytes().len() as f64 / (1024.0 * 1024.0 * 1024.0);
        eprintln!(
            "ram offload: locking {gib:.2} GiB with {n_threads} threads (mlock + parallel prefault)"
        );
        let (mlocked, checksum, ms) = mapped.prefault_pages_locked(n_threads);
        let throughput = gib / (ms as f64 / 1000.0);
        let lock_status = if mlocked {
            "mlocked (pages pinned, eviction-proof)"
        } else {
            "MADV_WILLNEED only (model too large to mlock safely, or no CAP_IPC_LOCK)"
        };
        eprintln!(
            "ram offload: done in {ms}ms ({throughput:.1} GB/s) checksum={checksum:#04x} | {lock_status}"
        );
    }
}

/// Apply the autotune plan to `args`. Only fills in fields the user
/// didn't explicitly set. Designed to be safe to call even when
/// the user has set most flags (those are left untouched).
pub(super) fn apply_plan_to_args(
    args: &mut Args,
    plan: &oxidize_core::autotune::TuningPlan,
    inv: &oxidize_core::autotune::HardwareInventory,
) {
    let overrides = oxidize_core::autotune::overrides_from_plan(plan);
    // Threads: always fill in if user didn't pass --threads.
    if args.threads.is_none() {
        if let Some(t) = overrides.threads {
            if t > 0 {
                args.threads = Some(t);
            }
        }
    }
    // Ctx size: only if user didn't pass --ctx-size.
    if args.ctx_size.is_none() {
        if let Some(c) = overrides.ctx_size {
            if c > 0 {
                args.ctx_size = Some(c);
            }
        }
    }
    // n_gpu_layers: only if user didn't pass --n-gpu-layers.
    if !args.n_gpu_layers_set {
        if let Some(n) = overrides.n_gpu_layers {
            args.n_gpu_layers = n;
        }
    }
    // kv_cache_dtype: only if user didn't pass --kv-cache-dtype.
    if !args.kv_cache_dtype_set {
        use oxidize_core::tensor::DType;
        let desired = match plan.kv_cache_dtype {
            DType::F16 => KvCacheDType::F16,
            DType::F32 => KvCacheDType::F32,
            DType::I8 => KvCacheDType::Q8,
            DType::I16 => KvCacheDType::Q4,
            _ => KvCacheDType::F16,
        };
        args.kv_cache_dtype = desired;
    }
    // TurboQuant: only if user didn't pass either turboquant flag.
    if !args.turboquant && !args.no_turboquant {
        if let Some(true) = overrides.turboquant {
            args.turboquant = true;
        }
    }
    // layer_cache: only if user kept the default of 1.
    if args.layer_cache == 1 {
        if let Some(c) = overrides.layer_cache {
            if c > 0 && c != 1 {
                args.layer_cache = c;
            }
        }
    }
    // layer_wise: only if user kept the default of false AND the plan
    // recommends it. Documented as best-effort: we can't distinguish
    // `--no-layer-wise` from "user didn't set", so a user who
    // explicitly wants to disable layer_wise should use --no-auto.
    if !args.layer_wise {
        if let Some(true) = overrides.layer_wise {
            args.layer_wise = true;
        }
    }
    // cpu_optimized: never auto-enable (it caps ctx to 2048 and
    // disables the existing auto-cap; it would silently override
    // a lot of user intent). The plan still hints via rationale.
    // ram_offload + mmap hints: best-effort, same caveat.
    if !args.ram_offload {
        if let Some(true) = overrides.ram_offload {
            args.ram_offload = true;
        }
    }
    if !args.mmap_hugepages {
        if let Some(true) = overrides.mmap_hugepages {
            args.mmap_hugepages = true;
        }
    }
    if !args.mmap_prefetch {
        if let Some(true) = overrides.mmap_prefetch {
            args.mmap_prefetch = true;
        }
    }
    eprintln!(
        "[oxidize auto-tune] applied: threads={:?} ctx={:?} n_gpu_layers={} kv={:?} layer_wise={} layer_cache={} turboquant={} (cores={} ram={} GiB gpu={} MiB)",
        args.threads,
        args.ctx_size,
        args.n_gpu_layers,
        args.kv_cache_dtype,
        args.layer_wise,
        args.layer_cache,
        args.turboquant,
        inv.physical_cores,
        inv.total_ram_bytes / (1u64 << 30),
        inv.gpu_vram_bytes / (1024 * 1024),
    );
}

/// JSON-friendly snapshot of a `TuningPlan` for tooling.
pub(super) fn plan_to_json(plan: &oxidize_core::autotune::TuningPlan) -> serde_json::Value {
    use oxidize_core::autotune::{
        AttentionKernel, OxkIsa, OxkTile, PipelineMode, SpeculativeSpec, WeightPlan,
    };
    let isa = match plan.oxk_isa {
        OxkIsa::Scalar => "scalar",
        OxkIsa::Avx2 => "avx2",
        OxkIsa::Avx512 => "avx512",
    };
    let tile = match plan.oxk_tile {
        OxkTile::T1 => 1,
        OxkTile::T4 => 4,
        OxkTile::T8 => 8,
        OxkTile::T16 => 16,
    };
    let pipe = match plan.pipeline {
        PipelineMode::Sequential => "sequential",
        PipelineMode::Continuous => "continuous",
        PipelineMode::Paged => "paged",
        PipelineMode::Asymmetric => "asymmetric",
    };
    let spec = match plan.speculative {
        SpeculativeSpec::None => "none",
        SpeculativeSpec::DFlash => "dflash",
        SpeculativeSpec::Mtp => "mtp",
    };
    let weight_plan = match plan.weight_plan {
        WeightPlan::Native => "native",
        WeightPlan::Fp8 => "fp8",
        WeightPlan::W8A8 => "w8a8",
        WeightPlan::W4A16 => "w4a16",
        WeightPlan::W4A8Kv4 => "w4a8kv4",
    };
    let attention_kernel = match plan.attention_kernel {
        AttentionKernel::Default => "default",
        AttentionKernel::FlashAttention => "flash_attention",
        AttentionKernel::FlashAttention3 => "flash_attention_3",
    };
    serde_json::json!({
        "threads": plan.threads,
        "ctx_size": plan.ctx_size,
        "kv_cache_dtype": format!("{:?}", plan.kv_cache_dtype),
        "kv_quantization": format!("{:?}", plan.kv_quantization),
        "n_gpu_layers": plan.n_gpu_layers,
        "mmap": plan.mmap,
        "mlock": plan.mlock,
        "mmap_hugepages": plan.mmap_hugepages,
        "mmap_prefetch": plan.mmap_prefetch,
        "numa_replicate_dense": plan.numa_replicate_dense,
        "layer_wise": plan.layer_wise,
        "layer_cache": plan.layer_cache,
        "pipeline": pipe,
        "speculative": spec,
        "weight_plan": weight_plan,
        "attention_kernel": attention_kernel,
        "cuda_graphs": plan.cuda_graphs,
        "persistent_decode_kernels": plan.persistent_decode_kernels,
        "tensor_parallelism": plan.tensor_parallelism,
        "pipeline_parallelism": plan.pipeline_parallelism,
        "chunked_prefill_tokens": plan.chunked_prefill_tokens,
        "max_decode_batch": plan.max_decode_batch,
        "decode_tile_tokens": plan.decode_tile_tokens,
        "oxk_isa": isa,
        "oxk_tile": tile,
        "expected_prompt_tps": plan.expected_prompt_tps,
        "expected_decode_tps": plan.expected_decode_tps,
        "rationale": plan.rationale,
    })
}

/// True if stdout is attached to a terminal (best-effort: uses
/// `std::io::IsTerminal` from stdlib).
pub(super) fn atty_stdout() -> bool {
    std::io::stdout().is_terminal()
}
