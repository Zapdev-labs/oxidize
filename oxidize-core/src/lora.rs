use std::collections::{BTreeMap, BTreeSet};

use crate::gguf::{GgufQuantizationType, GgufTensorInfo};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterKind {
    Lora,
    Qlora,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoraTarget {
    pub base_tensor: String,
    pub lora_a_tensor: String,
    pub lora_b_tensor: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoraPlan {
    pub kind: AdapterKind,
    pub targets: Vec<LoraTarget>,
    pub missing_base_tensors: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoraPlanError {
    MissingPairForLoraA(String),
    MissingPairForLoraB(String),
    DuplicatePair(String),
}

pub fn plan_lora_application(
    base_tensors: &[GgufTensorInfo],
    adapter_tensors: &[GgufTensorInfo],
    base_quantization: Option<GgufQuantizationType>,
) -> Result<LoraPlan, LoraPlanError> {
    let kind = match base_quantization {
        Some(GgufQuantizationType::F16) | Some(GgufQuantizationType::F32) | None => {
            AdapterKind::Lora
        }
        Some(_) => AdapterKind::Qlora,
    };

    let mut lora_a = BTreeMap::new();
    let mut lora_b = BTreeMap::new();
    for tensor in adapter_tensors {
        if let Some(base_name) = tensor.name.strip_suffix(".lora_a.weight") {
            if lora_a
                .insert(base_name.to_owned(), tensor.name.clone())
                .is_some()
            {
                return Err(LoraPlanError::DuplicatePair(base_name.to_owned()));
            }
        } else if let Some(base_name) = tensor.name.strip_suffix(".lora_b.weight")
            && lora_b
                .insert(base_name.to_owned(), tensor.name.clone())
                .is_some()
        {
            return Err(LoraPlanError::DuplicatePair(base_name.to_owned()));
        }
    }

    let all_keys = lora_a
        .keys()
        .chain(lora_b.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut targets = Vec::new();
    for key in &all_keys {
        let Some(a_name) = lora_a.get(key) else {
            return Err(LoraPlanError::MissingPairForLoraB(key.clone()));
        };
        let Some(b_name) = lora_b.get(key) else {
            return Err(LoraPlanError::MissingPairForLoraA(key.clone()));
        };
        targets.push(LoraTarget {
            base_tensor: key.clone(),
            lora_a_tensor: a_name.clone(),
            lora_b_tensor: b_name.clone(),
        });
    }

    let base_tensor_names = base_tensors
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect::<BTreeSet<_>>();
    let missing_base_tensors = targets
        .iter()
        .filter(|target| !base_tensor_names.contains(&target.base_tensor))
        .map(|target| target.base_tensor.clone())
        .collect::<Vec<_>>();

    Ok(LoraPlan {
        kind,
        targets,
        missing_base_tensors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_lora_for_fp16_base_models() {
        let base_tensors = vec![tensor("blk.0.attn_q.weight"), tensor("blk.0.attn_v.weight")];
        let adapter_tensors = vec![
            tensor("blk.0.attn_q.weight.lora_a.weight"),
            tensor("blk.0.attn_q.weight.lora_b.weight"),
        ];

        let plan = plan_lora_application(
            &base_tensors,
            &adapter_tensors,
            Some(GgufQuantizationType::F16),
        )
        .expect("plan should build");
        assert_eq!(plan.kind, AdapterKind::Lora);
        assert_eq!(plan.targets.len(), 1);
        assert_eq!(plan.targets[0].base_tensor, "blk.0.attn_q.weight");
        assert!(plan.missing_base_tensors.is_empty());
    }

    #[test]
    fn plans_qlora_for_quantized_base_models() {
        let base_tensors = vec![tensor("blk.0.attn_q.weight")];
        let adapter_tensors = vec![
            tensor("blk.0.attn_q.weight.lora_a.weight"),
            tensor("blk.0.attn_q.weight.lora_b.weight"),
        ];

        let plan = plan_lora_application(
            &base_tensors,
            &adapter_tensors,
            Some(GgufQuantizationType::Q4_K_M),
        )
        .expect("plan should build");
        assert_eq!(plan.kind, AdapterKind::Qlora);
    }

    #[test]
    fn reports_missing_base_tensors() {
        let base_tensors = vec![tensor("blk.0.attn_q.weight")];
        let adapter_tensors = vec![
            tensor("blk.1.attn_q.weight.lora_a.weight"),
            tensor("blk.1.attn_q.weight.lora_b.weight"),
        ];

        let plan = plan_lora_application(
            &base_tensors,
            &adapter_tensors,
            Some(GgufQuantizationType::F32),
        )
        .expect("plan should build");
        assert_eq!(plan.missing_base_tensors, vec!["blk.1.attn_q.weight"]);
    }

    #[test]
    fn rejects_unpaired_lora_tensors() {
        let err = plan_lora_application(
            &[tensor("blk.0.attn_q.weight")],
            &[tensor("blk.0.attn_q.weight.lora_a.weight")],
            None,
        )
        .expect_err("plan should fail");
        assert_eq!(
            err,
            LoraPlanError::MissingPairForLoraA("blk.0.attn_q.weight".to_owned())
        );
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
