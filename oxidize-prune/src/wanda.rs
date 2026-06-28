//! Wanda-style and magnitude pruning with optional joint quantize.
//!
//! Top-level entry: [`wanda_prune`] / [`magnitude_prune`] (the latter
//! is a Wanda-style structured mask using the magnitude metric — see
//! `mask.rs`). Both routines:
//!
//! 1. Parse the input GGUF and identify linear-weight tensors
//!    (2-D, `in_dim >= 64`, name matches `*weight` but not embeddings
//!    or the LM head).
//! 2. Dequantize each candidate tensor to f32.
//! 3. Compute the per-row pruning mask.
//! 4. Apply the mask in place (zeros pruned entries).
//! 5. Re-quantize the survivors to the original quantization type
//!    (or to a joint target if `joint_quantize` is set).
//! 6. Emit a new GGUF via `writer::write_gguf`.
//!
//! The activation L2 norms are loaded from a precomputed cache file
//! produced by the calibration runner (see
//! `oxidize_core::activation_stats`). On-disk format: one f32 per line,
//! preceded by `# in_dim <N>`, matching what `l2_norms_to_cache` writes.
//!
//! Reference papers:
//! - Wanda: `arxiv:2306.11695`
//! - SparseGPT: `arxiv:2301.00774`
//! - FlexGen offload / joint prune+quant: `arxiv:2303.06865`

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use oxidize_core::gguf::{GgufQuantizationType, GgufTensorInfo, parse_gguf};
use oxidize_core::quantization::{dequantize_scalar, quantize_scalar, quantized_size};
use oxidize_kernels::dequantize_q4_k_into;

use crate::mask::{
    SparsityPattern, apply_mask_inplace, apply_nm_pattern, magnitude_mask, wanda_mask,
};
use crate::writer::{OutputTensor, write_gguf};

/// Configuration for Wanda pruning.
#[derive(Debug, Clone)]
pub struct WandaOptions {
    pub input: PathBuf,
    pub output: PathBuf,
    /// Path to the L2-norms cache file produced by the calibration
    /// runner. Required for `wanda_prune`; ignored by `magnitude_prune`.
    pub calibration: Option<PathBuf>,
    pub sparsity: f32,
    pub pattern: SparsityPattern,
    /// If set, all linear weights are re-quantized to this type after
    /// masking. If `None`, the original qtype is preserved.
    pub joint_quantize: Option<GgufQuantizationType>,
    /// Tensor names that should never be pruned. Defaults to
    /// embedding + output + token_embd (matched as substrings).
    pub keep_names: Vec<String>,
    pub dry_run: bool,
    pub print_timings: bool,
}

/// Summary of a Wanda/magnitude prune run.
#[derive(Debug, Clone)]
pub struct PruneReport {
    pub total_tensors: usize,
    pub pruned_tensors: usize,
    pub skipped_tensors: usize,
    pub dry_run: bool,
    pub output: PathBuf,
    pub elapsed_ms: u64,
}

/// Run Wanda pruning. Returns a `PruneReport`.
///
/// # Errors
/// - I/O errors reading the input / writing the output.
/// - Parse errors in the input GGUF.
/// - Missing or malformed `calibration` file.
/// - `joint_quantize` types unsupported by the underlying scalar
///   quantizer are surfaced verbatim.
pub fn wanda_prune(options: WandaOptions) -> Result<PruneReport> {
    if !(0.0..1.0).contains(&options.sparsity) {
        bail!("sparsity must be in [0, 1), got {}", options.sparsity);
    }
    let calib_path = options
        .calibration
        .as_ref()
        .context("Wanda requires --calibration <l2_norms.txt>")?;
    let all_norms = load_l2_norms_cache(calib_path)?;
    let start = Instant::now();
    let report = run_inner(options, all_norms)?;
    Ok(PruneReport {
        elapsed_ms: start.elapsed().as_millis() as u64,
        ..report
    })
}

/// Run magnitude pruning (Wanda with the activation norms forced to 1,
/// so the metric collapses to `|W|`). Slightly faster than
/// `wanda_prune` because no per-column lookup is needed.
pub fn magnitude_prune(options: WandaOptions) -> Result<PruneReport> {
    if !(0.0..1.0).contains(&options.sparsity) {
        bail!("sparsity must be in [0, 1), got {}", options.sparsity);
    }
    let start = Instant::now();
    let report = run_inner(options, BTreeMap::new())?;
    Ok(PruneReport {
        elapsed_ms: start.elapsed().as_millis() as u64,
        ..report
    })
}

fn run_inner(
    options: WandaOptions,
    all_norms: BTreeMap<String, Vec<f32>>,
) -> Result<PruneReport> {
    let WandaOptions {
        input,
        output,
        calibration: _,
        sparsity,
        pattern,
        joint_quantize,
        keep_names,
        dry_run,
        print_timings,
    } = options;

    let bytes = fs::read(&input)
        .with_context(|| format!("failed to read input file: {}", input.display()))?;
    let parsed = parse_gguf(&bytes).map_err(|err| anyhow::anyhow!(err))?;

    let default_keep: Vec<String> = vec![
        "token_embd".to_string(),
        "output".to_string(),
        "rope".to_string(),
        "norm".to_string(),
    ];
    let keep_all: Vec<String> = if keep_names.is_empty() {
        default_keep
    } else {
        keep_names
    };

    let mut skipped = 0_usize;
    let mut pruned = 0_usize;
    let mut timing_dequant_ms = 0_u128;
    let mut timing_mask_ms = 0_u128;
    let mut timing_requant_ms = 0_u128;
    let mut results: Vec<OutputTensor> = Vec::with_capacity(parsed.tensor_infos.len());

    for info in &parsed.tensor_infos {
        if !is_linear_weight(info) {
            results.push(pass_through(info, &bytes)?);
            continue;
        }
        if keep_all.iter().any(|k| info.name.contains(k)) {
            results.push(pass_through(info, &bytes)?);
            skipped += 1;
            continue;
        }

        let in_dim = info
            .dimensions
            .last()
            .copied()
            .and_then(|d| usize::try_from(d).ok())
            .context("tensor dimension overflows usize")?;
        let out_dims: Vec<u64> = info
            .dimensions
            .iter()
            .take(info.dimensions.len().saturating_sub(1))
            .copied()
            .collect();
        let out_dim: usize = out_dims
            .iter()
            .try_fold(1_usize, |acc, d| {
                usize::try_from(*d).ok().and_then(|d| acc.checked_mul(d))
            })
            .context("out_dim overflows usize")?;
        let qtype = GgufQuantizationType::from_ggml_type(info.ggml_type);
        let raw = tensor_bytes(info, &bytes)?;
        let norms = all_norms.get(&info.name).cloned();
        if let Some(ref n) = norms
            && n.len() != in_dim
        {
            bail!(
                "{}: calibration norms length {} != in_dim {}",
                info.name,
                n.len(),
                in_dim
            );
        }

        let mut weights_f32 = vec![0.0_f32; out_dim * in_dim];
        let t = Instant::now();
        dequantize_weights(qtype, &raw, &mut weights_f32)?;
        timing_dequant_ms += t.elapsed().as_millis();

        let t = Instant::now();
        let mut mask = if let Some(ref norms) = norms {
            wanda_mask(&weights_f32, norms, out_dim, in_dim, sparsity)
        } else {
            magnitude_mask(&weights_f32, out_dim, in_dim, sparsity)
        };
        if !matches!(pattern, SparsityPattern::Unstructured) {
            let norms_owned;
            let norms_for_score: &[f32] = if let Some(ref n) = norms {
                n.as_slice()
            } else {
                norms_owned = vec![1.0_f32; in_dim];
                norms_owned.as_slice()
            };
            apply_nm_pattern(
                &mut mask,
                out_dim,
                in_dim,
                pattern,
                |r, c| weights_f32[r * in_dim + c].abs() * norms_for_score[c],
            )?;
        }
        apply_mask_inplace(&mut weights_f32, &mask);
        timing_mask_ms += t.elapsed().as_millis();

        let t = Instant::now();
        let target = joint_quantize.unwrap_or(qtype);
        let new_size =
            quantized_size(target, out_dim * in_dim).map_err(|e| anyhow::anyhow!(e))?;
        let mut new_bytes = vec![0u8; new_size];
        let f32_bytes = f32_slice_to_bytes(&weights_f32);
        quantize_scalar(GgufQuantizationType::F32, target, &f32_bytes, &mut new_bytes)
            .map_err(|e| anyhow::anyhow!(e))?;
        timing_requant_ms += t.elapsed().as_millis();

        results.push(OutputTensor {
            name: info.name.clone(),
            dimensions: info.dimensions.clone(),
            ggml_type: ggml_type_for_qtype(target),
            data: new_bytes,
        });
        pruned += 1;
    }

    let out_tensors = results;

    if !dry_run {
        let out_bytes =
            write_gguf(parsed.version, &parsed.metadata, &out_tensors, parsed.alignment)?;
        fs::write(&output, &out_bytes)
            .with_context(|| format!("failed to write output file: {}", output.display()))?;
    }

    if print_timings {
        eprintln!(
            "[oxidize-prune] dequant={}ms mask={}ms requant={}ms pruned={} skipped={} total={}",
            timing_dequant_ms,
            timing_mask_ms,
            timing_requant_ms,
            pruned,
            skipped,
            parsed.tensor_infos.len()
        );
    }

    Ok(PruneReport {
        total_tensors: parsed.tensor_infos.len(),
        pruned_tensors: pruned,
        skipped_tensors: skipped,
        dry_run,
        output,
        elapsed_ms: 0,
    })
}

fn dequantize_weights(
    qtype: GgufQuantizationType,
    raw: &[u8],
    out: &mut [f32],
) -> Result<()> {
    match qtype {
        GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M => {
            dequantize_q4_k_into(raw, out);
            Ok(())
        }
        _ => dequantize_scalar(qtype, raw, out).map_err(|e| anyhow::anyhow!(e)),
    }
}

/// True if this tensor looks like a linear weight matrix
/// (2-D, dimensions product large enough to benefit from pruning).
fn is_linear_weight(info: &GgufTensorInfo) -> bool {
    if info.dimensions.len() < 2 {
        return false;
    }
    if !info.name.ends_with(".weight") {
        return false;
    }
    // Total elements must be large enough for the Wanda mask to be
    // meaningful. The per-row minimum is checked separately inside
    // `wanda_mask`. We use 4 as the floor (a 2x2 weight is the
    // smallest non-trivial linear layer); the real filter is
    // `keep_per_row >= 1` which happens automatically when cols >= 1.
    let total: u64 = info.dimensions.iter().product();
    total >= 4
}

/// Read the raw quantized bytes for a tensor out of the whole-file
/// mmap-style buffer.
fn tensor_bytes(info: &GgufTensorInfo, bytes: &[u8]) -> Result<Vec<u8>> {
    let start = usize::try_from(info.absolute_offset)
        .with_context(|| format!("{}: absolute_offset overflows usize", info.name))?;
    let qtype = GgufQuantizationType::from_ggml_type(info.ggml_type);
    let value_count: usize = info
        .dimensions
        .iter()
        .try_fold(1_usize, |acc, d| {
            usize::try_from(*d).ok().and_then(|d| acc.checked_mul(d))
        })
        .with_context(|| format!("{}: value_count overflows usize", info.name))?;
    let size = quantized_size(qtype, value_count).map_err(|e| anyhow::anyhow!(e))?;
    let end = start
        .checked_add(size)
        .with_context(|| format!("{}: byte range overflows", info.name))?;
    if end > bytes.len() {
        bail!("{}: extends past end of input GGUF", info.name);
    }
    Ok(bytes[start..end].to_vec())
}

/// Copy a tensor's bytes verbatim from input to output (no pruning).
fn pass_through(info: &GgufTensorInfo, bytes: &[u8]) -> Result<OutputTensor> {
    let data = tensor_bytes(info, bytes)?;
    Ok(OutputTensor {
        name: info.name.clone(),
        dimensions: info.dimensions.clone(),
        ggml_type: info.ggml_type,
        data,
    })
}

fn f32_slice_to_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// L2-norms cache format (one file produced by the calibration runner):
/// ```text
/// # in_dim <N>
/// <tensor_name> <f32_0> <f32_1> ... <f32_{N-1}>
/// ...
/// ```
/// Lines starting with `#` are comments. Each data line is a tensor
/// name followed by N space-separated f32 values.
///
/// This is the simplest, most debuggable format; the file is small
/// (one f32 per linear weight column).
pub fn load_l2_norms_cache(path: &Path) -> Result<BTreeMap<String, Vec<f32>>> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read calibration cache: {}", path.display()))?;
    let mut out = BTreeMap::new();
    for (lineno, line) in raw.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut tokens = trimmed.split_whitespace();
        let name = tokens
            .next()
            .with_context(|| format!("{}:{}: missing tensor name", path.display(), lineno + 1))?;
        let values: Result<Vec<f32>> = tokens
            .map(|t| {
                t.parse::<f32>()
                    .with_context(|| format!("{}:{}: bad f32 '{}'", path.display(), lineno + 1, t))
            })
            .collect();
        out.insert(name.to_string(), values?);
    }
    Ok(out)
}

/// Write the L2-norms cache to disk. Used by the calibration runner
/// (typically a CLI subcommand or the server's calibration endpoint).
pub fn write_l2_norms_cache(
    path: &Path,
    norms: &BTreeMap<String, Vec<f32>>,
) -> Result<()> {
    let mut out = String::new();
    out.push_str("# oxidize-prune L2 norms cache\n");
    out.push_str("# one row per linear weight tensor, N f32 values per row\n");
    for (name, values) in norms {
        out.push_str(name);
        out.push(' ');
        for v in values {
            out.push_str(&format!("{v}"));
            out.push(' ');
        }
        out.push('\n');
    }
    fs::write(path, out)
        .with_context(|| format!("failed to write calibration cache: {}", path.display()))?;
    Ok(())
}

/// Sanity-check the calibration cache has the dimensions we expect for
/// the tensors in the input GGUF. Used by the CLI to fail fast.
pub fn validate_calibration(
    cache: &BTreeMap<String, Vec<f32>>,
    gguf_bytes: &[u8],
) -> Result<()> {
    let parsed = parse_gguf(gguf_bytes).map_err(|e| anyhow::anyhow!(e))?;
    for info in &parsed.tensor_infos {
        if !is_linear_weight(info) {
            continue;
        }
        let in_dim = info
            .dimensions
            .last()
            .copied()
            .and_then(|d| usize::try_from(d).ok())
            .unwrap_or(0);
        match cache.get(&info.name) {
            Some(norms) if norms.len() == in_dim => {}
            Some(norms) => bail!(
                "{}: calibration has {} entries, in_dim={}",
                info.name,
                norms.len(),
                in_dim
            ),
            None if in_dim > 0 => eprintln!(
                "warning: no calibration entry for {}; will fall back to magnitude",
                info.name
            ),
            None => {}
        }
    }
    Ok(())
}

/// Inverse of `GgufQuantizationType::from_ggml_type` for the subset we
/// support in joint_quantize. The original qtype is preserved
/// byte-for-byte when joint_quantize is None (see `pass_through`), so
/// this only matters for joint-quantize paths.
fn ggml_type_for_qtype(q: GgufQuantizationType) -> u32 {
    match q {
        GgufQuantizationType::F32 => 0,
        GgufQuantizationType::F16 => 1,
        GgufQuantizationType::Q4_0 => 2,
        GgufQuantizationType::Q4_1 => 3,
        GgufQuantizationType::Q5_0 => 6,
        GgufQuantizationType::Q5_1 => 7,
        GgufQuantizationType::Q8_0 => 8,
        GgufQuantizationType::Q2_K => 10,
        GgufQuantizationType::Q3_K_S | GgufQuantizationType::Q3_K_M | GgufQuantizationType::Q3_K_L => 11,
        GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M => 12,
        GgufQuantizationType::Q5_K_S | GgufQuantizationType::Q5_K_M => 13,
        GgufQuantizationType::Q6_K => 14,
        GgufQuantizationType::BF16 => 30,
        GgufQuantizationType::IQ1_S => 19,
        GgufQuantizationType::IQ1_M => 29,
        GgufQuantizationType::IQ3_S => 21,
        GgufQuantizationType::IQ4_XS => 23,
        GgufQuantizationType::I8 => 24,
        GgufQuantizationType::I16 => 25,
        GgufQuantizationType::I32 => 26,
        GgufQuantizationType::I64 => 27,
        GgufQuantizationType::F64 => 28,
        GgufQuantizationType::NVFP4 => 33,
        GgufQuantizationType::IQ2_XXS
        | GgufQuantizationType::IQ2_XS
        | GgufQuantizationType::IQ3_XXS
        | GgufQuantizationType::IQ4_NL
        | GgufQuantizationType::IQ2_S
        | GgufQuantizationType::Unknown(_) => 0, // fall back to F32 — caller should validate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxidize_core::gguf::GgufMetadataValue;
    use std::collections::BTreeMap;
    use std::time::{SystemTime, UNIX_EPOCH};

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
        let dir = root.join(format!("oxidize-prune-wanda-test-{nanos}"));
        fs::create_dir_all(&dir).expect("temp dir should be created");
        dir
    }

    fn tiny_gguf_with_weights() -> Vec<u8> {
        // 2 linear weights, F32, rows × cols.
        let metadata: BTreeMap<String, GgufMetadataValue> = BTreeMap::from([
            (
                "general.architecture".to_string(),
                GgufMetadataValue::String("llama".to_string()),
            ),
            ("general.alignment".to_string(), GgufMetadataValue::Uint32(32)),
            ("general.file_type".to_string(), GgufMetadataValue::Uint32(0)),
        ]);
        let w1: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let w2: Vec<f32> = (0..32).map(|i| -(i as f32)).collect();
        let f32_bytes = |v: &[f32]| {
            let mut b = Vec::with_capacity(v.len() * 4);
            for x in v {
                b.extend_from_slice(&x.to_le_bytes());
            }
            b
        };
        write_gguf(
            3,
            &metadata,
            &[
                OutputTensor {
                    name: "blk.0.attn_q.weight".to_string(),
                    dimensions: vec![4, 8],
                    ggml_type: 0,
                    data: f32_bytes(&w1),
                },
                OutputTensor {
                    name: "blk.0.ffn_gate.weight".to_string(),
                    dimensions: vec![4, 8],
                    ggml_type: 0,
                    data: f32_bytes(&w2),
                },
            ],
            32,
        )
        .expect("tiny GGUF")
    }

    #[test]
    fn l2_norms_cache_roundtrip() {
        let dir = unique_temp_dir();
        let path = dir.join("norms.txt");
        let mut cache: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        cache.insert("blk.0.attn_q.weight".to_string(), vec![1.0, 2.0, 3.0, 4.0]);
        cache.insert("blk.0.ffn_gate.weight".to_string(), vec![0.5, 0.5, 0.5, 0.5]);
        write_l2_norms_cache(&path, &cache).unwrap();
        let read = load_l2_norms_cache(&path).unwrap();
        assert_eq!(read.len(), 2);
        assert_eq!(read["blk.0.attn_q.weight"], vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn magnitude_prune_drops_bottom_half_per_row() {
        let dir = unique_temp_dir();
        let input = dir.join("in.gguf");
        let output = dir.join("out.gguf");
        fs::write(&input, tiny_gguf_with_weights()).unwrap();
        let opts = WandaOptions {
            input: input.clone(),
            output: output.clone(),
            calibration: None,
            sparsity: 0.5,
            pattern: SparsityPattern::Unstructured,
            joint_quantize: None,
            keep_names: Vec::new(),
            dry_run: false,
            print_timings: false,
        };
        let report = magnitude_prune(opts).unwrap();
        assert_eq!(report.total_tensors, 2);
        assert_eq!(report.pruned_tensors, 2);
        assert!(output.exists());

        // Parse the output and check the kept weights are the larger ones.
        let bytes = fs::read(&output).unwrap();
        let parsed = parse_gguf(&bytes).unwrap();
        let info0 = &parsed.tensor_infos[0];
        let raw0 = tensor_bytes(info0, &bytes).unwrap();
        let mut values = vec![0.0_f32; 32];
        dequantize_scalar(
            GgufQuantizationType::from_ggml_type(info0.ggml_type),
            &raw0,
            &mut values,
        )
        .unwrap();
        // Row 0 had values 0..8; keep top 4 (4,5,6,7) and zero the rest.
        for c in 0..4 {
            assert!(values[c].abs() < 1e-6, "col {c} should be zero, got {}", values[c]);
        }
        for c in 4..8 {
            assert!(
                values[c].abs() > 1e-6,
                "col {c} should be kept, got {}",
                values[c]
            );
        }
    }

    #[test]
    fn wanda_prune_uses_calibration() {
        let dir = unique_temp_dir();
        let input = dir.join("in.gguf");
        let output = dir.join("out.gguf");
        let calib = dir.join("norms.txt");
        fs::write(&input, tiny_gguf_with_weights()).unwrap();
        // Make a Wanda cache that amplifies the right half of each
        // row of `blk.0.attn_q.weight`, so the mask should keep the
        // right half (cols 4..8) even though they are larger in row 0
        // and smaller in row 1.
        let mut cache: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        cache.insert(
            "blk.0.attn_q.weight".to_string(),
            vec![0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0],
        );
        cache.insert(
            "blk.0.ffn_gate.weight".to_string(),
            vec![0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0],
        );
        write_l2_norms_cache(&calib, &cache).unwrap();
        let opts = WandaOptions {
            input: input.clone(),
            output: output.clone(),
            calibration: Some(calib),
            sparsity: 0.5,
            pattern: SparsityPattern::Unstructured,
            joint_quantize: None,
            keep_names: Vec::new(),
            dry_run: false,
            print_timings: false,
        };
        let report = wanda_prune(opts).unwrap();
        assert_eq!(report.pruned_tensors, 2);

        // For blk.0.attn_q.weight (values 0..8 in row-major):
        // Wanda score for col c in row r is |W[r, c]| * 10 for c >= 4,
        // 0 for c < 4. With sparsity 0.5 the top-4 per row are the
        // right half (cols 4..8).
        let bytes = fs::read(&output).unwrap();
        let parsed = parse_gguf(&bytes).unwrap();
        let info0 = &parsed.tensor_infos[0];
        let raw0 = tensor_bytes(info0, &bytes).unwrap();
        let mut values = vec![0.0_f32; 32];
        dequantize_scalar(
            GgufQuantizationType::from_ggml_type(info0.ggml_type),
            &raw0,
            &mut values,
        )
        .unwrap();
        for c in 0..4 {
            assert!(values[c].abs() < 1e-6, "col {c} should be zero, got {}", values[c]);
        }
        for c in 4..8 {
            assert!(values[c].abs() > 1e-6, "col {c} should be kept, got {}", values[c]);
        }
    }

    #[test]
    fn wanda_prune_with_2of4_pattern() {
        let dir = unique_temp_dir();
        let input = dir.join("in.gguf");
        let output = dir.join("out.gguf");
        let calib = dir.join("norms.txt");
        fs::write(&input, tiny_gguf_with_weights()).unwrap();
        let mut cache: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        cache.insert(
            "blk.0.attn_q.weight".to_string(),
            vec![1.0; 8],
        );
        cache.insert(
            "blk.0.ffn_gate.weight".to_string(),
            vec![1.0; 8],
        );
        write_l2_norms_cache(&calib, &cache).unwrap();
        let opts = WandaOptions {
            input,
            output,
            calibration: Some(calib),
            sparsity: 0.5,
            pattern: SparsityPattern::N2of4,
            joint_quantize: None,
            keep_names: Vec::new(),
            dry_run: false,
            print_timings: false,
        };
        wanda_prune(opts).unwrap();
    }

    #[test]
    fn validate_calibration_rejects_wrong_size() {
        let dir = unique_temp_dir();
        let input = dir.join("in.gguf");
        fs::write(&input, tiny_gguf_with_weights()).unwrap();
        let bytes = fs::read(&input).unwrap();
        let mut cache: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        cache.insert("blk.0.attn_q.weight".to_string(), vec![1.0; 4]); // wrong size
        let err = validate_calibration(&cache, &bytes).unwrap_err();
        assert!(err.to_string().contains("calibration has 4 entries"));
    }

    #[test]
    fn oxk_q4k_dequant_matches_core() {
        use oxidize_core::quantization::dequantize_q4_k_scalar;
        use oxidize_kernels::{BLOCK_Q4_K_SIZE, QK_K, dequantize_q4_k_into};
        let mut input = vec![0_u8; 3 * BLOCK_Q4_K_SIZE];
        for (i, b) in input.iter_mut().enumerate() {
            *b = ((i * 17 + 3) % 251) as u8 + 1;
        }
        let mut oxk_out = vec![0.0_f32; 3 * QK_K];
        let mut core_out = vec![0.0_f32; 3 * QK_K];
        dequantize_q4_k_into(&input, &mut oxk_out);
        dequantize_q4_k_scalar(&input, &mut core_out).unwrap();
        for (a, b) in oxk_out.iter().zip(core_out.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}
