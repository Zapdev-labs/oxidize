use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use memmap2::Mmap;
use oxidize_core::gguf::{GgufMetadataArray, GgufMetadataType, GgufMetadataValue};
use safetensors::tensor::{Dtype, SafeTensors};
use serde_json::Value;

#[derive(Debug, Parser)]
#[command(
    name = "oxidize-convert",
    about = "Convert SafeTensors files to GGUF format"
)]
struct Args {
    /// Input SafeTensors file (.safetensors)
    #[arg(long)]
    input: PathBuf,
    /// Output GGUF file (.gguf)
    #[arg(long)]
    output: PathBuf,
    /// Model architecture (e.g. llama, mistral, gpt2). Overrides value from SafeTensors metadata.
    #[arg(long)]
    arch: Option<String>,
}

#[derive(Debug)]
struct OutputTensor {
    name: String,
    dimensions: Vec<u64>,
    ggml_type: u32,
    data: Vec<u8>,
}

fn run(args: Args) -> Result<()> {
    let file =
        File::open(&args.input).with_context(|| format!("failed to open {}", args.input.display()))?;
    // SAFETY: read-only mapping; file handle kept alive via `file` for the mapping's lifetime.
    let mmap = unsafe { Mmap::map(&file) }
        .with_context(|| format!("failed to mmap {}", args.input.display()))?;

    let st = SafeTensors::deserialize(&mmap)
        .map_err(|e| anyhow!("failed to parse SafeTensors: {e:?}"))?;

    let metadata = build_gguf_metadata(&mmap, args.arch.as_deref(), &args.input)?;
    let tensors = build_output_tensors(&st)?;

    let gguf_bytes = write_gguf(3, &metadata, &tensors, 32)?;
    std::fs::write(&args.output, &gguf_bytes)
        .with_context(|| format!("failed to write {}", args.output.display()))?;

    println!(
        "Converted {} tensors → {}",
        tensors.len(),
        args.output.display()
    );
    Ok(())
}

/// Parse the `__metadata__` block from the safetensors file header.
fn read_safetensors_metadata(mmap: &Mmap) -> Result<BTreeMap<String, String>> {
    if mmap.len() < 8 {
        return Ok(BTreeMap::new());
    }
    let header_len = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
    if 8 + header_len > mmap.len() {
        bail!("safetensors header length exceeds file size");
    }
    let header_json: Value = serde_json::from_slice(&mmap[8..8 + header_len])
        .context("failed to parse safetensors header JSON")?;
    let Some(meta_obj) = header_json.get("__metadata__").and_then(|v| v.as_object()) else {
        return Ok(BTreeMap::new());
    };
    Ok(meta_obj
        .iter()
        .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_owned())))
        .collect())
}

fn build_gguf_metadata(
    mmap: &Mmap,
    arch_override: Option<&str>,
    input_path: &Path,
) -> Result<BTreeMap<String, GgufMetadataValue>> {
    let mut meta: BTreeMap<String, GgufMetadataValue> = BTreeMap::new();

    // Pull key-value strings from the SafeTensors __metadata__ header JSON.
    let st_meta = read_safetensors_metadata(mmap)?;

    // Derive architecture: CLI flag > safetensors metadata > filename stem.
    let arch = arch_override
        .map(|s| s.to_owned())
        .or_else(|| {
            st_meta
                .get("model_type")
                .or_else(|| st_meta.get("architecture"))
                .cloned()
        })
        .unwrap_or_else(|| {
            input_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_owned()
        });

    meta.insert(
        "general.architecture".to_owned(),
        GgufMetadataValue::String(arch),
    );

    if let Some(name) = st_meta
        .get("model_name")
        .or_else(|| st_meta.get("name"))
        .cloned()
        .or_else(|| {
            input_path
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.to_owned())
        })
    {
        meta.insert(
            "general.name".to_owned(),
            GgufMetadataValue::String(name),
        );
    }

    // Passthrough any remaining SafeTensors string metadata under a namespaced key.
    for (key, value) in &st_meta {
        if matches!(key.as_str(), "model_type" | "architecture" | "model_name" | "name") {
            continue;
        }
        meta.insert(
            format!("general.safetensors.{key}"),
            GgufMetadataValue::String(value.clone()),
        );
    }

    Ok(meta)
}

fn build_output_tensors(st: &SafeTensors) -> Result<Vec<OutputTensor>> {
    let mut tensors: Vec<OutputTensor> = Vec::with_capacity(st.len());

    for (name, view) in st.tensors() {
        let raw_data = view.data();
        let shape = view.shape();
        let dimensions: Vec<u64> = shape.iter().map(|&d| d as u64).collect();

        let (ggml_type, data) = match view.dtype() {
            Dtype::F32 => (0_u32, raw_data.to_vec()),
            Dtype::F16 => (1_u32, raw_data.to_vec()),
            Dtype::BF16 => {
                // BF16 is not a native GGUF type; promote to F32.
                let f32_data = bf16_to_f32(raw_data);
                (0_u32, f32_data)
            }
            other => bail!("unsupported SafeTensors dtype {other:?} in tensor {name}"),
        };

        tensors.push(OutputTensor {
            name: name.to_owned(),
            dimensions,
            ggml_type,
            data,
        });
    }

    // Stable output order for reproducible files.
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(tensors)
}

/// Reinterpret each pair of bytes as a BF16 value and widen to F32.
fn bf16_to_f32(bytes: &[u8]) -> Vec<u8> {
    assert!(bytes.len() % 2 == 0, "BF16 buffer length must be even");
    let mut out = Vec::with_capacity(bytes.len() * 2);
    for chunk in bytes.chunks_exact(2) {
        // BF16 is the upper 16 bits of an IEEE 754 F32.
        let bits = u32::from(u16::from_le_bytes([chunk[0], chunk[1]])) << 16;
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

// ── GGUF writer ──────────────────────────────────────────────────────────────

fn write_gguf(
    version: u32,
    metadata: &BTreeMap<String, GgufMetadataValue>,
    tensors: &[OutputTensor],
    alignment: u64,
) -> Result<Vec<u8>> {
    if alignment == 0 || !alignment.is_power_of_two() {
        bail!("invalid GGUF alignment: {alignment}");
    }

    // Pre-compute relative tensor data offsets.
    let mut relative_offsets: Vec<u64> = Vec::with_capacity(tensors.len());
    let mut cursor: u64 = 0;
    for tensor in tensors {
        cursor = align_up(cursor, alignment)?;
        relative_offsets.push(cursor);
        cursor = cursor
            .checked_add(tensor.data.len() as u64)
            .ok_or_else(|| anyhow!("tensor data offset overflow"))?;
    }

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

    for (key, value) in metadata {
        write_string(&mut out, key);
        write_metadata_value(&mut out, value)?;
    }

    for (tensor, &rel_offset) in tensors.iter().zip(relative_offsets.iter()) {
        write_string(&mut out, &tensor.name);
        out.extend_from_slice(&(tensor.dimensions.len() as u32).to_le_bytes());
        for dim in &tensor.dimensions {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        out.extend_from_slice(&tensor.ggml_type.to_le_bytes());
        out.extend_from_slice(&rel_offset.to_le_bytes());
    }

    pad_to(&mut out, alignment)?;
    let data_start = out.len() as u64;

    for (tensor, &rel_offset) in tensors.iter().zip(relative_offsets.iter()) {
        let target = (data_start + rel_offset) as usize;
        if out.len() < target {
            out.resize(target, 0);
        }
        out.extend_from_slice(&tensor.data);
        pad_to(&mut out, alignment)?;
    }

    Ok(out)
}

fn write_metadata_value(out: &mut Vec<u8>, value: &GgufMetadataValue) -> Result<()> {
    let vtype = metadata_type(value);
    out.extend_from_slice(&(vtype as u32).to_le_bytes());
    write_metadata_payload(out, value, vtype)
}

fn write_metadata_payload(
    out: &mut Vec<u8>,
    value: &GgufMetadataValue,
    vtype: GgufMetadataType,
) -> Result<()> {
    match (vtype, value) {
        (GgufMetadataType::Uint8, GgufMetadataValue::Uint8(v)) => out.push(*v),
        (GgufMetadataType::Int8, GgufMetadataValue::Int8(v)) => out.push(*v as u8),
        (GgufMetadataType::Uint16, GgufMetadataValue::Uint16(v)) => {
            out.extend_from_slice(&v.to_le_bytes())
        }
        (GgufMetadataType::Int16, GgufMetadataValue::Int16(v)) => {
            out.extend_from_slice(&v.to_le_bytes())
        }
        (GgufMetadataType::Uint32, GgufMetadataValue::Uint32(v)) => {
            out.extend_from_slice(&v.to_le_bytes())
        }
        (GgufMetadataType::Int32, GgufMetadataValue::Int32(v)) => {
            out.extend_from_slice(&v.to_le_bytes())
        }
        (GgufMetadataType::Float32, GgufMetadataValue::Float32(v)) => {
            out.extend_from_slice(&v.to_le_bytes())
        }
        (GgufMetadataType::Bool, GgufMetadataValue::Bool(v)) => out.push(u8::from(*v)),
        (GgufMetadataType::String, GgufMetadataValue::String(v)) => write_string(out, v),
        (GgufMetadataType::Array, GgufMetadataValue::Array(arr)) => {
            write_metadata_array(out, arr)?
        }
        (GgufMetadataType::Uint64, GgufMetadataValue::Uint64(v)) => {
            out.extend_from_slice(&v.to_le_bytes())
        }
        (GgufMetadataType::Int64, GgufMetadataValue::Int64(v)) => {
            out.extend_from_slice(&v.to_le_bytes())
        }
        (GgufMetadataType::Float64, GgufMetadataValue::Float64(v)) => {
            out.extend_from_slice(&v.to_le_bytes())
        }
        _ => bail!("metadata type mismatch"),
    }
    Ok(())
}

fn write_metadata_array(out: &mut Vec<u8>, arr: &GgufMetadataArray) -> Result<()> {
    out.extend_from_slice(&(arr.element_type as u32).to_le_bytes());
    out.extend_from_slice(&(arr.values.len() as u64).to_le_bytes());
    for value in &arr.values {
        write_metadata_payload(out, value, arr.element_type)?;
    }
    Ok(())
}

fn metadata_type(value: &GgufMetadataValue) -> GgufMetadataType {
    match value {
        GgufMetadataValue::Uint8(_) => GgufMetadataType::Uint8,
        GgufMetadataValue::Int8(_) => GgufMetadataType::Int8,
        GgufMetadataValue::Uint16(_) => GgufMetadataType::Uint16,
        GgufMetadataValue::Int16(_) => GgufMetadataType::Int16,
        GgufMetadataValue::Uint32(_) => GgufMetadataType::Uint32,
        GgufMetadataValue::Int32(_) => GgufMetadataType::Int32,
        GgufMetadataValue::Float32(_) => GgufMetadataType::Float32,
        GgufMetadataValue::Bool(_) => GgufMetadataType::Bool,
        GgufMetadataValue::String(_) => GgufMetadataType::String,
        GgufMetadataValue::Array(_) => GgufMetadataType::Array,
        GgufMetadataValue::Uint64(_) => GgufMetadataType::Uint64,
        GgufMetadataValue::Int64(_) => GgufMetadataType::Int64,
        GgufMetadataValue::Float64(_) => GgufMetadataType::Float64,
    }
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn pad_to(out: &mut Vec<u8>, alignment: u64) -> Result<()> {
    let aligned = align_up(out.len() as u64, alignment)? as usize;
    out.resize(aligned, 0);
    Ok(())
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|v| v & !mask)
        .ok_or_else(|| anyhow!("alignment overflow"))
}

fn main() {
    let args = Args::parse();
    if let Err(err) = run(args) {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxidize_core::gguf::parse_gguf;
    use safetensors::tensor::{Dtype, TensorView};
    use std::collections::HashMap;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn write_test_safetensors(path: &Path, tensors: &[(&str, Dtype, Vec<usize>, Vec<u8>)]) {
        let views: HashMap<String, TensorView> = tensors
            .iter()
            .map(|(name, dtype, shape, data)| {
                (
                    name.to_string(),
                    TensorView::new(*dtype, shape.clone(), data).unwrap(),
                )
            })
            .collect();
        let bytes = safetensors::tensor::serialize(&views, &None).unwrap();
        let mut f = File::create(path).unwrap();
        f.write_all(&bytes).unwrap();
    }

    fn tmp_path(suffix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("oxidize-convert-test-{nanos}{suffix}"))
    }

    #[test]
    fn converts_f32_tensor_roundtrip() {
        let input = tmp_path(".safetensors");
        let output = tmp_path(".gguf");

        let data: Vec<f32> = vec![1.0, -2.0, 3.5, -4.25];
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        write_test_safetensors(&input, &[("weight", Dtype::F32, vec![2, 2], bytes)]);

        run(Args {
            input: input.clone(),
            output: output.clone(),
            arch: Some("llama".to_owned()),
        })
        .expect("conversion should succeed");

        let gguf_bytes = std::fs::read(&output).unwrap();
        let parsed = parse_gguf(&gguf_bytes).expect("output should be valid GGUF");
        assert_eq!(parsed.tensor_infos.len(), 1);
        assert_eq!(parsed.tensor_infos[0].name, "weight");
        assert_eq!(parsed.tensor_infos[0].ggml_type, 0); // F32
        assert_eq!(parsed.tensor_infos[0].dimensions, vec![2, 2]);
        assert_eq!(
            parsed.metadata.get("general.architecture"),
            Some(&GgufMetadataValue::String("llama".to_owned()))
        );

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    #[test]
    fn converts_f16_tensor_roundtrip() {
        let input = tmp_path(".safetensors");
        let output = tmp_path(".gguf");

        // Two f16 values: 1.0 and 2.0
        let data: Vec<u8> = vec![0x00, 0x3C, 0x00, 0x40]; // f16 1.0, f16 2.0
        write_test_safetensors(&input, &[("token_embd.weight", Dtype::F16, vec![1, 2], data)]);

        run(Args {
            input: input.clone(),
            output: output.clone(),
            arch: Some("gpt2".to_owned()),
        })
        .expect("f16 conversion should succeed");

        let gguf_bytes = std::fs::read(&output).unwrap();
        let parsed = parse_gguf(&gguf_bytes).expect("output should be valid GGUF");
        assert_eq!(parsed.tensor_infos[0].ggml_type, 1); // F16

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    #[test]
    fn bf16_promoted_to_f32() {
        let input = tmp_path(".safetensors");
        let output = tmp_path(".gguf");

        // Two BF16 values: 1.0 (0x3F80) and -1.0 (0xBF80), little-endian
        let data: Vec<u8> = vec![0x80, 0x3F, 0x80, 0xBF];
        write_test_safetensors(&input, &[("fc.weight", Dtype::BF16, vec![1, 2], data)]);

        run(Args {
            input: input.clone(),
            output: output.clone(),
            arch: None,
        })
        .expect("bf16 conversion should succeed");

        let gguf_bytes = std::fs::read(&output).unwrap();
        let parsed = parse_gguf(&gguf_bytes).expect("output should be valid GGUF");
        assert_eq!(parsed.tensor_infos[0].ggml_type, 0); // F32 (promoted)
        assert_eq!(parsed.tensor_infos[0].dimensions, vec![1, 2]);

        // Verify the promoted values
        let offset = parsed.tensor_infos[0].absolute_offset as usize;
        let f1 = f32::from_le_bytes(gguf_bytes[offset..offset + 4].try_into().unwrap());
        let f2 = f32::from_le_bytes(gguf_bytes[offset + 4..offset + 8].try_into().unwrap());
        assert!((f1 - 1.0).abs() < 1e-6, "expected 1.0, got {f1}");
        assert!((f2 + 1.0).abs() < 1e-6, "expected -1.0, got {f2}");

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    #[test]
    fn multi_tensor_output_sorted_by_name() {
        let input = tmp_path(".safetensors");
        let output = tmp_path(".gguf");

        let data = vec![0u8; 8];
        write_test_safetensors(
            &input,
            &[
                ("z_tensor", Dtype::F32, vec![2], data.clone()),
                ("a_tensor", Dtype::F32, vec![2], data.clone()),
            ],
        );

        run(Args {
            input: input.clone(),
            output: output.clone(),
            arch: None,
        })
        .expect("multi-tensor conversion should succeed");

        let gguf_bytes = std::fs::read(&output).unwrap();
        let parsed = parse_gguf(&gguf_bytes).expect("output should be valid GGUF");
        assert_eq!(parsed.tensor_infos.len(), 2);
        assert_eq!(parsed.tensor_infos[0].name, "a_tensor");
        assert_eq!(parsed.tensor_infos[1].name, "z_tensor");

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }
}
