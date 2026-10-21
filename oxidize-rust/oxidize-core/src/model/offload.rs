use std::collections::BTreeSet;

use crate::gguf::GgufTensorInfo;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerOffloadPlan {
    pub n_gpu_layers: usize,
    pub total_layers: usize,
    pub gpu_tensor_count: usize,
    pub cpu_tensor_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelismStrategy {
    Tensor,
    Pipeline,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiGpuConfig {
    pub gpu_count: usize,
    pub n_gpu_layers: usize,
    pub strategy: ParallelismStrategy,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuAssignment {
    pub gpu_index: usize,
    pub layer_count: usize,
    pub tensor_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineStage {
    pub gpu_index: usize,
    pub start_layer: Option<usize>,
    pub end_layer: Option<usize>,
    pub layer_count: usize,
    pub tensor_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiGpuOffloadPlan {
    pub strategy: ParallelismStrategy,
    pub total_layers: usize,
    pub n_gpu_layers: usize,
    pub total_gpu_tensor_count: usize,
    pub cpu_tensor_count: usize,
    pub gpu_assignments: Vec<GpuAssignment>,
    pub pipeline_stages: Vec<PipelineStage>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultiGpuPlanError {
    InvalidGpuCount,
}

impl LayerOffloadPlan {
    pub fn has_gpu_tensors(&self) -> bool {
        self.gpu_tensor_count > 0
    }
}

pub fn plan_layer_offload(tensors: &[GgufTensorInfo], n_gpu_layers: usize) -> LayerOffloadPlan {
    let layers = collect_layer_indices(tensors);
    let total_layers = layers.len();
    let selected_layers = layers
        .into_iter()
        .take(n_gpu_layers.min(total_layers))
        .collect::<BTreeSet<_>>();

    let gpu_tensor_count = tensors
        .iter()
        .filter(|tensor| {
            layer_index_from_name(&tensor.name)
                .map(|layer| selected_layers.contains(&layer))
                .unwrap_or(false)
        })
        .count();
    let cpu_tensor_count = tensors.len().saturating_sub(gpu_tensor_count);

    LayerOffloadPlan {
        n_gpu_layers: selected_layers.len(),
        total_layers,
        gpu_tensor_count,
        cpu_tensor_count,
    }
}

pub fn plan_multi_gpu_offload(
    tensors: &[GgufTensorInfo],
    config: &MultiGpuConfig,
) -> Result<MultiGpuOffloadPlan, MultiGpuPlanError> {
    if config.gpu_count == 0 {
        return Err(MultiGpuPlanError::InvalidGpuCount);
    }

    let layers = collect_layer_indices(tensors);
    let total_layers = layers.len();
    let selected_layers = layers
        .into_iter()
        .take(config.n_gpu_layers.min(total_layers))
        .collect::<Vec<_>>();
    let selected_layer_set = selected_layers.iter().copied().collect::<BTreeSet<_>>();

    let mut layer_counts = vec![0_usize; config.gpu_count];
    let mut tensor_counts = vec![0_usize; config.gpu_count];
    let mut total_gpu_tensor_count = 0_usize;
    let pipeline_stage_for_layer =
        build_pipeline_stage_for_layer(&selected_layers, config.gpu_count);

    for tensor in tensors {
        let Some(layer_index) = layer_index_from_name(&tensor.name) else {
            continue;
        };
        if !selected_layer_set.contains(&layer_index) {
            continue;
        }

        let gpu_index = match config.strategy {
            ParallelismStrategy::Tensor => {
                tensor_parallel_gpu_index(&tensor.name, config.gpu_count)
            }
            ParallelismStrategy::Pipeline => pipeline_stage_for_layer
                .get(&layer_index)
                .copied()
                .unwrap_or(0),
        };
        tensor_counts[gpu_index] += 1;
        total_gpu_tensor_count += 1;
    }

    for layer_index in &selected_layers {
        let gpu_index = match config.strategy {
            ParallelismStrategy::Tensor => layer_index % config.gpu_count,
            ParallelismStrategy::Pipeline => pipeline_stage_for_layer
                .get(layer_index)
                .copied()
                .unwrap_or(0),
        };
        layer_counts[gpu_index] += 1;
    }

    let gpu_assignments = (0..config.gpu_count)
        .map(|gpu_index| GpuAssignment {
            gpu_index,
            layer_count: layer_counts[gpu_index],
            tensor_count: tensor_counts[gpu_index],
        })
        .collect::<Vec<_>>();
    let pipeline_stages = if config.strategy == ParallelismStrategy::Pipeline {
        build_pipeline_stages(&selected_layers, &tensor_counts, config.gpu_count)
    } else {
        Vec::new()
    };

    let cpu_tensor_count = tensors.len().saturating_sub(total_gpu_tensor_count);
    Ok(MultiGpuOffloadPlan {
        strategy: config.strategy,
        total_layers,
        n_gpu_layers: selected_layers.len(),
        total_gpu_tensor_count,
        cpu_tensor_count,
        gpu_assignments,
        pipeline_stages,
    })
}

fn tensor_parallel_gpu_index(name: &str, gpu_count: usize) -> usize {
    let mut hash = 0_u64;
    for byte in name.as_bytes() {
        hash = hash.wrapping_mul(16777619).wrapping_add(u64::from(*byte));
    }
    (hash as usize) % gpu_count
}

fn build_pipeline_stage_for_layer(
    selected_layers: &[usize],
    gpu_count: usize,
) -> std::collections::HashMap<usize, usize> {
    let mut mapping = std::collections::HashMap::with_capacity(selected_layers.len());
    let stage_ranges = pipeline_stage_ranges(selected_layers.len(), gpu_count);
    for (gpu_index, (start, end)) in stage_ranges.into_iter().enumerate() {
        for layer in &selected_layers[start..end] {
            mapping.insert(*layer, gpu_index);
        }
    }
    mapping
}

fn build_pipeline_stages(
    selected_layers: &[usize],
    tensor_counts: &[usize],
    gpu_count: usize,
) -> Vec<PipelineStage> {
    let stage_ranges = pipeline_stage_ranges(selected_layers.len(), gpu_count);
    stage_ranges
        .into_iter()
        .enumerate()
        .map(|(gpu_index, (start, end))| {
            let stage_layers = &selected_layers[start..end];
            PipelineStage {
                gpu_index,
                start_layer: stage_layers.first().copied(),
                end_layer: stage_layers.last().copied(),
                layer_count: stage_layers.len(),
                tensor_count: tensor_counts[gpu_index],
            }
        })
        .collect()
}

fn pipeline_stage_ranges(layer_count: usize, gpu_count: usize) -> Vec<(usize, usize)> {
    let mut ranges = Vec::with_capacity(gpu_count);
    let base = layer_count / gpu_count;
    let remainder = layer_count % gpu_count;
    let mut start = 0_usize;
    for gpu in 0..gpu_count {
        let width = base + usize::from(gpu < remainder);
        let end = start + width;
        ranges.push((start, end));
        start = end;
    }
    ranges
}

fn collect_layer_indices(tensors: &[GgufTensorInfo]) -> BTreeSet<usize> {
    tensors
        .iter()
        .filter_map(|tensor| layer_index_from_name(&tensor.name))
        .collect()
}

fn layer_index_from_name(name: &str) -> Option<usize> {
    if let Some(rest) = name.strip_prefix("blk.") {
        return rest.split('.').next()?.parse::<usize>().ok();
    }
    if let Some(rest) = name.strip_prefix("model.layers.") {
        return rest.split('.').next()?.parse::<usize>().ok();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offload_plan_caps_requested_layers_to_model_layer_count() {
        let tensors = vec![
            tensor("blk.0.attn_q.weight"),
            tensor("blk.1.attn_q.weight"),
            tensor("blk.2.attn_q.weight"),
            tensor("tok_embeddings.weight"),
        ];
        let plan = plan_layer_offload(&tensors, 99);
        assert_eq!(plan.total_layers, 3);
        assert_eq!(plan.n_gpu_layers, 3);
        assert_eq!(plan.gpu_tensor_count, 3);
        assert_eq!(plan.cpu_tensor_count, 1);
    }

    #[test]
    fn offload_plan_handles_zero_requested_gpu_layers() {
        let tensors = vec![tensor("blk.0.attn_q.weight"), tensor("blk.0.attn_k.weight")];
        let plan = plan_layer_offload(&tensors, 0);
        assert_eq!(plan.n_gpu_layers, 0);
        assert_eq!(plan.total_layers, 1);
        assert_eq!(plan.gpu_tensor_count, 0);
        assert_eq!(plan.cpu_tensor_count, 2);
        assert!(!plan.has_gpu_tensors());
    }

    #[test]
    fn offload_plan_supports_hf_layer_names() {
        let tensors = vec![
            tensor("model.layers.0.self_attn.q_proj.weight"),
            tensor("model.layers.1.self_attn.q_proj.weight"),
            tensor("lm_head.weight"),
        ];
        let plan = plan_layer_offload(&tensors, 1);
        assert_eq!(plan.total_layers, 2);
        assert_eq!(plan.n_gpu_layers, 1);
        assert_eq!(plan.gpu_tensor_count, 1);
        assert_eq!(plan.cpu_tensor_count, 2);
        assert!(plan.has_gpu_tensors());
    }

    #[test]
    fn multi_gpu_pipeline_assigns_contiguous_layers() {
        let tensors = vec![
            tensor("blk.0.attn_q.weight"),
            tensor("blk.1.attn_q.weight"),
            tensor("blk.2.attn_q.weight"),
            tensor("blk.3.attn_q.weight"),
            tensor("tok_embeddings.weight"),
        ];
        let config = MultiGpuConfig {
            gpu_count: 2,
            n_gpu_layers: 4,
            strategy: ParallelismStrategy::Pipeline,
        };

        let plan = plan_multi_gpu_offload(&tensors, &config).expect("plan should build");
        assert_eq!(plan.n_gpu_layers, 4);
        assert_eq!(plan.total_layers, 4);
        assert_eq!(plan.total_gpu_tensor_count, 4);
        assert_eq!(plan.cpu_tensor_count, 1);
        assert_eq!(plan.gpu_assignments[0].layer_count, 2);
        assert_eq!(plan.gpu_assignments[1].layer_count, 2);
        assert_eq!(plan.gpu_assignments[0].tensor_count, 2);
        assert_eq!(plan.gpu_assignments[1].tensor_count, 2);
        assert_eq!(plan.pipeline_stages.len(), 2);
        assert_eq!(plan.pipeline_stages[0].start_layer, Some(0));
        assert_eq!(plan.pipeline_stages[0].end_layer, Some(1));
        assert_eq!(plan.pipeline_stages[1].start_layer, Some(2));
        assert_eq!(plan.pipeline_stages[1].end_layer, Some(3));
    }

    #[test]
    fn multi_gpu_tensor_distributes_tensors() {
        let tensors = vec![
            tensor("blk.0.attn_q.weight"),
            tensor("blk.0.attn_k.weight"),
            tensor("blk.1.attn_q.weight"),
            tensor("blk.1.attn_k.weight"),
        ];
        let config = MultiGpuConfig {
            gpu_count: 2,
            n_gpu_layers: 2,
            strategy: ParallelismStrategy::Tensor,
        };

        let plan = plan_multi_gpu_offload(&tensors, &config).expect("plan should build");
        assert_eq!(plan.total_gpu_tensor_count, 4);
        assert_eq!(plan.cpu_tensor_count, 0);
        assert_eq!(plan.gpu_assignments.len(), 2);
        assert!(plan.gpu_assignments[0].tensor_count > 0);
        assert!(plan.gpu_assignments[1].tensor_count > 0);
        assert!(plan.pipeline_stages.is_empty());
    }

    #[test]
    fn multi_gpu_pipeline_balances_remainder_layers() {
        let tensors = vec![
            tensor("blk.0.attn_q.weight"),
            tensor("blk.1.attn_q.weight"),
            tensor("blk.2.attn_q.weight"),
            tensor("blk.3.attn_q.weight"),
            tensor("blk.4.attn_q.weight"),
        ];
        let config = MultiGpuConfig {
            gpu_count: 3,
            n_gpu_layers: 5,
            strategy: ParallelismStrategy::Pipeline,
        };

        let plan = plan_multi_gpu_offload(&tensors, &config).expect("plan should build");
        assert_eq!(
            plan.pipeline_stages
                .iter()
                .map(|stage| stage.layer_count)
                .collect::<Vec<_>>(),
            vec![2, 2, 1]
        );
        assert_eq!(plan.pipeline_stages[0].start_layer, Some(0));
        assert_eq!(plan.pipeline_stages[0].end_layer, Some(1));
        assert_eq!(plan.pipeline_stages[1].start_layer, Some(2));
        assert_eq!(plan.pipeline_stages[1].end_layer, Some(3));
        assert_eq!(plan.pipeline_stages[2].start_layer, Some(4));
        assert_eq!(plan.pipeline_stages[2].end_layer, Some(4));
    }

    #[test]
    fn multi_gpu_pipeline_handles_more_gpus_than_layers() {
        let tensors = vec![tensor("blk.3.attn_q.weight"), tensor("blk.7.attn_q.weight")];
        let config = MultiGpuConfig {
            gpu_count: 4,
            n_gpu_layers: 2,
            strategy: ParallelismStrategy::Pipeline,
        };

        let plan = plan_multi_gpu_offload(&tensors, &config).expect("plan should build");
        assert_eq!(plan.pipeline_stages.len(), 4);
        assert_eq!(plan.pipeline_stages[0].start_layer, Some(3));
        assert_eq!(plan.pipeline_stages[0].end_layer, Some(3));
        assert_eq!(plan.pipeline_stages[1].start_layer, Some(7));
        assert_eq!(plan.pipeline_stages[1].end_layer, Some(7));
        assert_eq!(plan.pipeline_stages[2].start_layer, None);
        assert_eq!(plan.pipeline_stages[2].end_layer, None);
        assert_eq!(plan.pipeline_stages[3].start_layer, None);
        assert_eq!(plan.pipeline_stages[3].end_layer, None);
    }

    #[test]
    fn multi_gpu_rejects_zero_devices() {
        let config = MultiGpuConfig {
            gpu_count: 0,
            n_gpu_layers: 2,
            strategy: ParallelismStrategy::Pipeline,
        };
        let err = plan_multi_gpu_offload(&[tensor("blk.0.attn_q.weight")], &config)
            .expect_err("plan must reject zero GPUs");
        assert_eq!(err, MultiGpuPlanError::InvalidGpuCount);
    }

    fn tensor(name: &str) -> GgufTensorInfo {
        GgufTensorInfo {
            name: name.to_owned(),
            dimensions: vec![1],
            ggml_type: 0,
            relative_offset: 0,
            absolute_offset: 0,
            mmap_index: 0,
        }
    }
}
