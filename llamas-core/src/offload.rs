use std::collections::BTreeSet;

use crate::gguf::GgufTensorInfo;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerOffloadPlan {
    pub n_gpu_layers: usize,
    pub total_layers: usize,
    pub gpu_tensor_count: usize,
    pub cpu_tensor_count: usize,
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

    fn tensor(name: &str) -> GgufTensorInfo {
        GgufTensorInfo {
            name: name.to_owned(),
            dimensions: vec![1],
            ggml_type: 0,
            relative_offset: 0,
            absolute_offset: 0,
        }
    }
}
