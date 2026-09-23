#![allow(
    unused_parens,
    clippy::doc_overindented_list_items,
    clippy::excessive_precision
)]

mod arch;
mod config;
mod dataset;
pub mod dpo;
mod error;
mod export;
mod fused;
mod generate;
mod lora;
pub mod merge;
pub mod qlora;
pub mod rlhf;
mod runtime;
pub mod self_train;
pub mod telemetry;
mod trainer;

pub use arch::{ArchPlan, ForwardFidelity, continuation_span};
pub use config::FinetuneConfig;
pub use dataset::{SftExample, load_jsonl_sft, pack_chunks};
pub use dpo::{DpoConfig, DpoExample, DpoReport, DpoTrainer, load_jsonl_dpo};
pub use error::FinetuneError;
pub use export::{
    export_lora_gguf, load_adapter_manifest, load_output_head_lora, manifest_to_lora_adapters,
};
pub use generate::{GenerateConfig, generate_with_lora};
pub use lora::{LoRAAdapter, LoRATarget};
pub use merge::{AdapterMerger, MergeStrategy, linear_merge, slerp_merge, ties_merge};
pub use qlora::{NF4Block, QLoRAAdapter};
pub use rlhf::{PpoConfig, PpoReport, PpoStepReport, PpoTrainer, RewardModel, RolloutBuffer};
pub use runtime::{
    CausalModel, DpoRun, PpoRun, ScoredSpan, clip_prompt_continuation, load_causal_model,
    score_continuation, train_dpo_on_model, train_ppo_on_model,
};
pub use self_train::{
    SelfTrainConfig, SelfTrainLoop, SelfTrainReport, SelfTrainRoundReport, load_prompts_file,
    sample_prompts,
};
pub use telemetry::{EarlyStopping, MetricsLog, ProgressReporter, TrainingMetrics};
pub use trainer::{FinetuneReport, SftTrainer};
