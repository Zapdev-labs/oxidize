use std::collections::BTreeSet;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};

use crate::blend::{linear_bytes, slerp_bytes};
use crate::index::{ModelIndex, ShardCache, is_blendable};
use crate::recipe::{MergeRecipe, recipe_metadata};
use crate::writer::{MergeWriter, OutputTensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeMethod {
    Linear,
    Slerp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingTensorPolicy {
    Error,
    A,
    B,
}

#[derive(Debug, Clone)]
pub struct MergeOptions {
    pub model_a: PathBuf,
    pub model_b: PathBuf,
    pub output: PathBuf,
    pub method: MergeMethod,
    pub recipe: MergeRecipe,
    pub missing: MissingTensorPolicy,
    pub max_shard_bytes: u64,
    pub dry_run: bool,
}

#[derive(Debug, Clone)]
pub struct MergeReport {
    pub merged_tensors: usize,
    pub copied_from_a: usize,
    pub copied_from_b: usize,
    pub output: PathBuf,
    pub dry_run: bool,
}

pub fn merge_models(opts: MergeOptions) -> Result<MergeReport> {
    let index_a = ModelIndex::open(&opts.model_a)
        .with_context(|| format!("failed to open model A at {}", opts.model_a.display()))?;
    let index_b = ModelIndex::open(&opts.model_b)
        .with_context(|| format!("failed to open model B at {}", opts.model_b.display()))?;

    let names: Vec<String> = index_a
        .tensor_names()
        .chain(index_b.tensor_names())
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    if opts.dry_run {
        let mut merged = 0usize;
        let mut copied_a = 0usize;
        let mut copied_b = 0usize;
        for name in &names {
            match (index_a.tensors.get(name), index_b.tensors.get(name)) {
                (Some(a), Some(b)) => {
                    validate_compatible(a, b)?;
                    if is_blendable(a.dtype) {
                        merged += 1;
                    } else {
                        copied_a += 1;
                    }
                }
                (Some(_), None) => match opts.missing {
                    MissingTensorPolicy::Error => {
                        bail!("tensor {name} exists only in model A");
                    }
                    MissingTensorPolicy::A => copied_a += 1,
                    MissingTensorPolicy::B => bail!("tensor {name} missing from model B"),
                },
                (None, Some(_)) => match opts.missing {
                    MissingTensorPolicy::Error => {
                        bail!("tensor {name} exists only in model B");
                    }
                    MissingTensorPolicy::A => bail!("tensor {name} missing from model A"),
                    MissingTensorPolicy::B => copied_b += 1,
                },
                (None, None) => unreachable!("name came from union"),
            }
        }
        return Ok(MergeReport {
            merged_tensors: merged,
            copied_from_a: copied_a,
            copied_from_b: copied_b,
            output: opts.output.clone(),
            dry_run: true,
        });
    }

    let method_name = match opts.method {
        MergeMethod::Linear => "linear",
        MergeMethod::Slerp => "slerp",
    };
    let mut metadata = index_a.metadata.clone();
    metadata.extend(index_b.metadata);
    metadata.extend(recipe_metadata(&opts.recipe, method_name));
    metadata.insert(
        "oxidize-merge.model_a".to_owned(),
        opts.model_a.display().to_string(),
    );
    metadata.insert(
        "oxidize-merge.model_b".to_owned(),
        opts.model_b.display().to_string(),
    );

    let mut writer = MergeWriter::new(&opts.output, opts.max_shard_bytes, metadata)?;
    let mut cache_a = ShardCache::new();
    let mut cache_b = ShardCache::new();

    let mut merged = 0usize;
    let mut copied_a = 0usize;
    let mut copied_b = 0usize;

    for name in names {
        match (index_a.tensors.get(&name), index_b.tensors.get(&name)) {
            (Some(a), Some(b)) => {
                validate_compatible(a, b)?;
                let out = if is_blendable(a.dtype) {
                    let t = opts.recipe.t_for_tensor(&name);
                    let a_bytes = cache_a.tensor_bytes(a)?.to_vec();
                    let b_bytes = cache_b.tensor_bytes(b)?.to_vec();
                    let mut out_bytes = vec![0_u8; a_bytes.len()];
                    match opts.method {
                        MergeMethod::Linear => {
                            linear_bytes(a.dtype, &a_bytes, &b_bytes, t, &mut out_bytes)?;
                        }
                        MergeMethod::Slerp => {
                            slerp_bytes(a.dtype, &a_bytes, &b_bytes, t, &mut out_bytes)?;
                        }
                    }
                    merged += 1;
                    out_bytes
                } else {
                    copied_a += 1;
                    cache_a.tensor_bytes(a)?.to_vec()
                };
                writer.push(OutputTensor {
                    name: name.clone(),
                    dtype: a.dtype,
                    shape: a.shape.clone(),
                    data: out,
                })?;
            }
            (Some(a), None) => {
                resolve_single_side(&opts.missing, true, &name)?;
                copied_a += 1;
                let data = cache_a.tensor_bytes(a)?.to_vec();
                writer.push(OutputTensor {
                    name,
                    dtype: a.dtype,
                    shape: a.shape.clone(),
                    data,
                })?;
            }
            (None, Some(b)) => {
                resolve_single_side(&opts.missing, false, &name)?;
                copied_b += 1;
                let data = cache_b.tensor_bytes(b)?.to_vec();
                writer.push(OutputTensor {
                    name,
                    dtype: b.dtype,
                    shape: b.shape.clone(),
                    data,
                })?;
            }
            (None, None) => unreachable!("name came from union"),
        }
    }

    writer.finish()?;
    Ok(MergeReport {
        merged_tensors: merged,
        copied_from_a: copied_a,
        copied_from_b: copied_b,
        output: opts.output,
        dry_run: false,
    })
}

fn resolve_single_side(
    policy: &MissingTensorPolicy,
    missing_from_b: bool,
    name: &str,
) -> Result<()> {
    match (policy, missing_from_b) {
        (MissingTensorPolicy::Error, true) => {
            bail!("tensor {name} exists only in model A");
        }
        (MissingTensorPolicy::Error, false) => {
            bail!("tensor {name} exists only in model B");
        }
        (MissingTensorPolicy::A, false) => bail!("tensor {name} missing from model A"),
        (MissingTensorPolicy::B, true) => bail!("tensor {name} missing from model B"),
        (MissingTensorPolicy::A, true) | (MissingTensorPolicy::B, false) => Ok(()),
    }
}

fn validate_compatible(
    a: &crate::index::TensorRef,
    b: &crate::index::TensorRef,
) -> Result<()> {
    if a.dtype != b.dtype {
        bail!(
            "dtype mismatch for {}: {:?} vs {:?}",
            a.name,
            a.dtype,
            b.dtype
        );
    }
    if a.shape != b.shape {
        bail!(
            "shape mismatch for {}: {:?} vs {:?}",
            a.name,
            a.shape,
            b.shape
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{Dtype, TensorView};
    use std::collections::HashMap;
    use std::io::Write;
    use std::path::Path;

    fn write_tensor(path: &Path, name: &str, values: &[f32]) {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tensor = TensorView::new(Dtype::F32, vec![values.len()], &bytes).unwrap();
        let mut tensors = HashMap::new();
        tensors.insert(name.to_owned(), tensor);
        let st = safetensors::tensor::serialize(&tensors, &None).unwrap();
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&st).unwrap();
    }

    #[test]
    fn merges_two_single_file_models() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.safetensors");
        let b = dir.path().join("b.safetensors");
        let out = dir.path().join("merged.safetensors");
        write_tensor(&a, "weight", &[0.0, 2.0]);
        write_tensor(&b, "weight", &[2.0, 4.0]);

        let report = merge_models(MergeOptions {
            model_a: a,
            model_b: b,
            output: out.clone(),
            method: MergeMethod::Linear,
            recipe: MergeRecipe::uniform(0.5),
            missing: MissingTensorPolicy::Error,
            max_shard_bytes: u64::MAX,
            dry_run: false,
        })
        .unwrap();

        assert_eq!(report.merged_tensors, 1);
        let mapped = crate::index::MappedShard::open(&out).unwrap();
        let data = mapped.tensor_bytes("weight").unwrap();
        let vals: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert!((vals[0] - 1.0).abs() < 1e-5);
        assert!((vals[1] - 3.0).abs() < 1e-5);
    }
}
