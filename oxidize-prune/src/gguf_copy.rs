use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};
use oxidize_core::gguf::{GgufQuantizationType, GgufTensorInfo, parse_gguf};
use oxidize_core::quantization::quantized_size;

use crate::filter::PruneFilter;
use crate::writer::{OutputTensor, write_gguf};

#[derive(Debug)]
pub struct PruneOptions {
    pub input: PathBuf,
    pub output: PathBuf,
    pub filter: PruneFilter,
    pub dry_run: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub struct PruneSummary {
    pub output: PathBuf,
    pub total: usize,
    pub kept: Vec<String>,
    pub removed: Vec<String>,
    pub dry_run: bool,
}

pub fn prune_gguf(options: PruneOptions) -> Result<PruneSummary> {
    let input = fs::read(&options.input)
        .with_context(|| format!("failed to read input file: {}", options.input.display()))?;
    let parsed = parse_gguf(&input).map_err(|err| anyhow!(err))?;
    let tensors = copy_selected_tensors(&input, &parsed.tensor_infos, &options.filter)?;
    let kept = tensors
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect::<Vec<_>>();
    let removed = parsed
        .tensor_infos
        .iter()
        .filter(|tensor| !options.filter.keeps(&tensor.name))
        .map(|tensor| tensor.name.clone())
        .collect::<Vec<_>>();

    if !options.dry_run {
        let output = write_gguf(parsed.version, &parsed.metadata, &tensors, parsed.alignment)?;
        fs::write(&options.output, &output).with_context(|| {
            format!("failed to write output file: {}", options.output.display())
        })?;
    }

    Ok(PruneSummary {
        output: options.output,
        total: parsed.tensor_infos.len(),
        kept,
        removed,
        dry_run: options.dry_run,
    })
}

fn copy_selected_tensors(
    input: &[u8],
    tensors: &[GgufTensorInfo],
    filter: &PruneFilter,
) -> Result<Vec<OutputTensor>> {
    let mut output = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        if !filter.keeps(&tensor.name) {
            continue;
        }
        let value_count = tensor_value_count(tensor)?;
        let source = GgufQuantizationType::from_ggml_type(tensor.ggml_type);
        let input_size = quantized_size(source, value_count)
            .map_err(|err| anyhow!(err))
            .with_context(|| format!("unsupported input tensor type for {}", tensor.name))?;
        let start = usize::try_from(tensor.absolute_offset)
            .with_context(|| format!("tensor {} offset overflows usize", tensor.name))?;
        let end = start
            .checked_add(input_size)
            .ok_or_else(|| anyhow!("tensor {} byte range overflows", tensor.name))?;
        if end > input.len() {
            bail!("tensor {} extends past end of input GGUF", tensor.name);
        }
        output.push(OutputTensor {
            name: tensor.name.clone(),
            dimensions: tensor.dimensions.clone(),
            ggml_type: tensor.ggml_type,
            data: input[start..end].to_vec(),
        });
    }
    if output.is_empty() {
        bail!("prune filter removed every tensor");
    }
    Ok(output)
}

fn tensor_value_count(tensor: &GgufTensorInfo) -> Result<usize> {
    tensor.dimensions.iter().try_fold(1_usize, |acc, dim| {
        let dim: usize = (*dim)
            .try_into()
            .map_err(|_| anyhow!("tensor {} dimension overflows usize", tensor.name))?;
        acc.checked_mul(dim)
            .ok_or_else(|| anyhow!("tensor {} value count overflows", tensor.name))
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use oxidize_core::gguf::{GgufMetadataValue, parse_gguf};

    #[test]
    fn prunes_tiny_gguf_by_tensor_name() {
        let temp_dir = unique_temp_dir();
        let input_path = temp_dir.join("tiny.gguf");
        let output_path = temp_dir.join("pruned.gguf");
        fs::write(&input_path, tiny_gguf()).expect("tiny GGUF should be written");

        let summary = prune_gguf(PruneOptions {
            input: input_path,
            output: output_path.clone(),
            filter: PruneFilter::new(Vec::new(), vec!["ffn".to_owned()]),
            dry_run: false,
        })
        .expect("prune should succeed");

        assert_eq!(summary.total, 2);
        assert_eq!(summary.kept, vec!["blk.0.attn_q.weight"]);
        assert_eq!(summary.removed, vec!["blk.0.ffn_gate.weight"]);

        let output = fs::read(output_path).expect("output GGUF should exist");
        let parsed = parse_gguf(&output).expect("output GGUF should parse");
        assert_eq!(parsed.tensor_infos.len(), 1);
        assert_eq!(parsed.tensor_infos[0].name, "blk.0.attn_q.weight");
        assert_eq!(parsed.tensor_infos[0].relative_offset, 0);
    }

    #[test]
    fn dry_run_does_not_write_output() {
        let temp_dir = unique_temp_dir();
        let input_path = temp_dir.join("tiny.gguf");
        let output_path = temp_dir.join("dry-run.gguf");
        fs::write(&input_path, tiny_gguf()).expect("tiny GGUF should be written");

        let summary = prune_gguf(PruneOptions {
            input: input_path,
            output: output_path.clone(),
            filter: PruneFilter::new(vec!["attn".to_owned()], Vec::new()),
            dry_run: true,
        })
        .expect("dry run should succeed");

        assert!(summary.dry_run);
        assert!(!output_path.exists());
        assert_eq!(summary.kept, vec!["blk.0.attn_q.weight"]);
    }

    fn tiny_gguf() -> Vec<u8> {
        let metadata = BTreeMap::from([
            (
                "general.architecture".to_owned(),
                GgufMetadataValue::String("llama".to_owned()),
            ),
            (
                "general.alignment".to_owned(),
                GgufMetadataValue::Uint32(32),
            ),
            ("general.file_type".to_owned(), GgufMetadataValue::Uint32(0)),
        ]);
        write_gguf(
            3,
            &metadata,
            &[
                OutputTensor {
                    name: "blk.0.attn_q.weight".to_owned(),
                    dimensions: vec![2, 2],
                    ggml_type: 0,
                    data: f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
                },
                OutputTensor {
                    name: "blk.0.ffn_gate.weight".to_owned(),
                    dimensions: vec![2, 2],
                    ggml_type: 0,
                    data: f32_bytes(&[5.0, 6.0, 7.0, 8.0]),
                },
            ],
            32,
        )
        .expect("tiny GGUF should encode")
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn unique_temp_dir() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before epoch")
            .as_nanos();
        let root = if PathBuf::from("/dev/shm").is_dir() {
            PathBuf::from("/dev/shm")
        } else {
            std::env::temp_dir()
        };
        let dir = root.join(format!("oxidize-prune-test-{nanos}"));
        fs::create_dir_all(&dir).expect("temp dir should be created");
        dir
    }
}
