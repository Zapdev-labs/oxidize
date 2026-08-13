//! PagedAttention runtime: scheduler + block pool wrapping a [`ModelRuntime`].

use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use tokio::sync::Mutex;

use oxidize_core::{
    autotune::TuningPlan,
    model::Model,
    paged_attention::{BlockPool, BlockPoolConfig, Scheduler, SchedulerConfig},
    tensor::DType,
};

use crate::cli::Args;
use crate::runtime::batched_engine::BatchedEngineHandle;
use crate::runtime::model::{LoadedModel, ModelRuntime};

/// Runtime state for PagedAttention-based generation.
///
/// Holds a [`Scheduler`] alongside the loaded model so that each request
/// becomes a [`oxidize_core::paged_attention::Sequence`] with tracked block
/// allocation. The scheduler enforces token budgets, handles block reclamation,
/// and provides accurate usage counts.
pub struct PagedModelRuntime {
    pub runtime: Arc<ModelRuntime>,
    pub scheduler: Mutex<Scheduler>,
    pub next_seq_id: AtomicU64,
    pub block_size: usize,
    /// Shared continuous-batching engine. `Some` when `OX_BATCHED_DECODE` is set
    /// and the backend supports batched `forward_batch`; requests are then routed
    /// through it so concurrent decode is batched into one forward per step.
    pub engine: Option<BatchedEngineHandle>,
}

pub fn build_paged_runtime(
    args: &Args,
    runtime: Arc<ModelRuntime>,
    plan: Option<&TuningPlan>,
) -> Arc<PagedModelRuntime> {
    let inference_model = runtime.model.blocking_lock();
    let context_size = inference_model.context_size();
    let layer_count = inference_model.layer_count();
    drop(inference_model);

    let (num_kv_heads, head_dim) = {
        let model_guard = runtime.model.blocking_lock();
        match &*model_guard {
            LoadedModel::Inference(m) => {
                let cfg = m.config();
                (cfg.num_key_value_heads, cfg.kv_head_dim())
            }
            LoadedModel::LayerWise(m) => {
                let cfg = m.config();
                (cfg.num_key_value_heads, cfg.kv_head_dim())
            }
            LoadedModel::DFlash(m) => {
                let cfg = &m.config;
                (cfg.num_key_value_heads, cfg.kv_head_dim())
            }
            #[cfg(target_os = "macos")]
            LoadedModel::Mlx(m) => {
                let cfg = m.config();
                (cfg.num_key_value_heads, cfg.kv_head_dim())
            }
            #[cfg(not(target_os = "macos"))]
            LoadedModel::Mlx(m) => {
                let cfg = m.config();
                (cfg.num_key_value_heads, cfg.kv_head_dim())
            }
        }
    };

    let config =
        block_pool_config_from_plan(context_size, layer_count, num_kv_heads, head_dim, plan);

    let block_pool = BlockPool::new(config);
    let scheduler_config = scheduler_config_from_plan(args, plan);
    let scheduler = Scheduler::new(scheduler_config, block_pool);

    let engine = BatchedEngineHandle::spawn_if_enabled(Arc::clone(&runtime));

    Arc::new(PagedModelRuntime {
        runtime,
        scheduler: Mutex::new(scheduler),
        next_seq_id: AtomicU64::new(1),
        block_size: config.block_size,
        engine,
    })
}

fn block_pool_config_from_plan(
    context_size: usize,
    layer_count: usize,
    num_kv_heads: usize,
    head_dim: usize,
    plan: Option<&TuningPlan>,
) -> BlockPoolConfig {
    let block_size = 16;
    let dtype = plan.map(|plan| plan.kv_cache_dtype).unwrap_or(DType::F32);
    match context_size.checked_div(block_size).unwrap_or(0) {
        0 => BlockPoolConfig {
            block_size,
            num_layers: layer_count,
            num_kv_heads,
            head_dim,
            dtype,
            ..BlockPoolConfig::default()
        },
        blocks => BlockPoolConfig {
            block_size,
            num_blocks: blocks * 4,
            num_layers: layer_count,
            num_kv_heads,
            head_dim,
            dtype,
        },
    }
}

fn scheduler_config_from_plan(args: &Args, plan: Option<&TuningPlan>) -> SchedulerConfig {
    let planned_prefill_chunk = plan
        .and_then(|plan| (plan.chunked_prefill_tokens > 0).then_some(plan.chunked_prefill_tokens))
        .unwrap_or(args.prefill_chunk_size);
    let max_num_running_seqs = plan
        .and_then(|plan| (plan.max_decode_batch > 0).then_some(plan.max_decode_batch))
        .unwrap_or(8);

    SchedulerConfig {
        max_num_batched_tokens: args.prefill_batch_size,
        prefill_chunk_size: planned_prefill_chunk.min(args.prefill_batch_size.max(1)),
        max_num_running_seqs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use oxidize_core::autotune::{
        AttentionKernel, OxkIsa, OxkTile, PipelineMode, SpeculativeSpec, TuningPlan, WeightPlan,
    };
    use oxidize_core::kv_cache::KvQuantization;

    fn h100_plan() -> TuningPlan {
        TuningPlan {
            threads: 8,
            ctx_size: 4096,
            kv_cache_dtype: DType::I16,
            kv_quantization: KvQuantization::TurboQuant,
            n_gpu_layers: 60,
            gpu_split: Vec::new(),
            mmap: false,
            mlock: false,
            mmap_hugepages: false,
            mmap_prefetch: false,
            numa_replicate_dense: false,
            layer_wise: false,
            layer_cache: 0,
            pipeline: PipelineMode::Paged,
            speculative: SpeculativeSpec::DFlash,
            decode_tile_tokens: 0,
            oxk_isa: OxkIsa::Avx2,
            oxk_tile: OxkTile::T8,
            expected_prompt_tps: 6_000.0,
            expected_decode_tps: 1_150.0,
            weight_plan: WeightPlan::W4A16,
            attention_kernel: AttentionKernel::FlashAttention3,
            cuda_graphs: true,
            persistent_decode_kernels: true,
            tensor_parallelism: 1,
            pipeline_parallelism: 1,
            chunked_prefill_tokens: 512,
            max_decode_batch: 16,
            rationale: Vec::new(),
        }
    }

    #[test]
    fn scheduler_config_uses_throughput_plan_when_available() {
        let args = Args::parse_from(["oxidize-server"]);
        let config = scheduler_config_from_plan(&args, Some(&h100_plan()));

        assert_eq!(config.max_num_batched_tokens, 512);
        assert_eq!(config.prefill_chunk_size, 512);
        assert_eq!(config.max_num_running_seqs, 16);
    }

    #[test]
    fn scheduler_config_uses_explicit_prefill_chunk_without_plan() {
        let args = Args::parse_from(["oxidize-server", "--prefill-chunk-size", "32"]);
        let config = scheduler_config_from_plan(&args, None);

        assert_eq!(config.prefill_chunk_size, 32);
    }

    #[test]
    fn block_pool_config_uses_plan_block_size_and_kv_dtype() {
        let config = block_pool_config_from_plan(4096, 60, 8, 128, Some(&h100_plan()));

        assert_eq!(config.block_size, 16);
        assert_eq!(config.num_blocks, 1024);
        assert_eq!(config.num_layers, 60);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.dtype, DType::I16);
    }
}
