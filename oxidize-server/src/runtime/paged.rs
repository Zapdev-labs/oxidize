//! PagedAttention runtime: scheduler + block pool wrapping a [`ModelRuntime`].

use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::AtomicU64;

use oxidize_core::{
    model::Model,
    paged_attention::{BlockPool, BlockPoolConfig, Scheduler, SchedulerConfig},
    tensor::DType,
};

use crate::cli::Args;
use crate::runtime::model::{LoadedModel, ModelRuntime};

/// Runtime state for PagedAttention-based generation.
///
/// Holds a [`Scheduler`] alongside the loaded model so that each request
/// becomes a [`oxidize_core::paged_attention::Sequence`] with tracked block
/// allocation. The scheduler enforces token budgets, handles block reclamation,
/// and provides accurate usage counts.
pub struct PagedModelRuntime {
    pub runtime: Arc<ModelRuntime>,
    pub scheduler: StdMutex<Scheduler>,
    pub next_seq_id: AtomicU64,
    pub block_size: usize,
}

pub fn build_paged_runtime(args: &Args, runtime: Arc<ModelRuntime>) -> Arc<PagedModelRuntime> {
    let inference_model = runtime.model.lock().expect("model lock poisoned");
    let config = match inference_model.context_size().checked_div(16).unwrap_or(0) {
        0 => BlockPoolConfig::default(),
        blocks => BlockPoolConfig {
            block_size: 16,
            num_blocks: blocks * 4, // heuristic: 4x the context-size-in-blocks
            num_layers: inference_model.layer_count(),
            num_kv_heads: 0,
            head_dim: 0,
            dtype: DType::F32,
        },
    };
    drop(inference_model);

    let (num_kv_heads, head_dim) = {
        let model_guard = runtime.model.lock().expect("model lock poisoned");
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

    let config = BlockPoolConfig {
        num_kv_heads,
        head_dim,
        ..config
    };

    let block_pool = BlockPool::new(config);
    let scheduler_config = SchedulerConfig {
        max_num_batched_tokens: args.prefill_batch_size,
        prefill_chunk_size: 16,
        max_num_running_seqs: 8,
    };
    let scheduler = Scheduler::new(scheduler_config, block_pool);

    Arc::new(PagedModelRuntime {
        runtime,
        scheduler: StdMutex::new(scheduler),
        next_seq_id: AtomicU64::new(1),
        block_size: config.block_size,
    })
}
