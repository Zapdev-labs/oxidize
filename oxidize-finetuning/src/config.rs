use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinetuneConfig {
    pub rank: usize,
    pub alpha: f32,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub epochs: usize,
    pub batch_size: usize,
    pub max_seq_len: usize,
    pub gradient_accumulation_steps: usize,
    pub gradient_checkpointing: bool,
    pub warmup_steps: usize,
    pub seed: u64,
    pub output_lora_scale: bool,
}

impl Default for FinetuneConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 32.0,
            learning_rate: 2e-4,
            weight_decay: 0.0,
            epochs: 1,
            batch_size: 1,
            max_seq_len: 2048,
            gradient_accumulation_steps: 4,
            gradient_checkpointing: true,
            warmup_steps: 10,
            seed: 42,
            output_lora_scale: true,
        }
    }
}

impl FinetuneConfig {
    pub fn lora_scale(&self) -> f32 {
        if self.output_lora_scale {
            self.alpha / self.rank.max(1) as f32
        } else {
            1.0
        }
    }
}
