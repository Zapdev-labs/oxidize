use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use oxidize_core::gguf::{
    GgufFile, GgufMetadataArray, GgufMetadataType, GgufMetadataValue, GgufQuantizationType,
    GgufTensorInfo, parse_gguf,
};
use oxidize_core::quantization::{quantize_scalar, quantized_size};

#[derive(Debug, Parser)]
#[command(name = "oxidize-quantize")]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, value_parser = parse_quantization_type)]
    source: Option<GgufQuantizationType>,
    #[arg(long, value_parser = parse_quantization_type)]
    target: Option<GgufQuantizationType>,
    /// Append an already-encoded tensor to an input GGUF without requantizing
    /// existing tensors. Format: name:path:dim0,dim1:type
    #[arg(long)]
    append_tensor: Vec<String>,
}

fn parse_quantization_type(value: &str) -> Result<GgufQuantizationType, String> {
    match value.to_ascii_uppercase().as_str() {
        "F32" => Ok(GgufQuantizationType::F32),
        "F16" => Ok(GgufQuantizationType::F16),
        "Q4_0" => Ok(GgufQuantizationType::Q4_0),
        "Q4_1" => Ok(GgufQuantizationType::Q4_1),
        "Q5_0" => Ok(GgufQuantizationType::Q5_0),
        "Q5_1" => Ok(GgufQuantizationType::Q5_1),
        "Q8_0" => Ok(GgufQuantizationType::Q8_0),
        "Q2_K" => Ok(GgufQuantizationType::Q2_K),
        "Q3_K_S" => Ok(GgufQuantizationType::Q3_K_S),
        "Q3_K_M" => Ok(GgufQuantizationType::Q3_K_M),
        "Q3_K_L" => Ok(GgufQuantizationType::Q3_K_L),
        "Q4_K_S" => Ok(GgufQuantizationType::Q4_K_S),
        "Q4_K_M" => Ok(GgufQuantizationType::Q4_K_M),
        "Q5_K_S" => Ok(GgufQuantizationType::Q5_K_S),
        "Q5_K_M" => Ok(GgufQuantizationType::Q5_K_M),
        "Q6_K" => Ok(GgufQuantizationType::Q6_K),
        _ => Err(format!("unsupported quantization type: {value}")),
    }
}

fn source_value_count(source: GgufQuantizationType, byte_len: usize) -> Result<usize> {
    let bytes_per_value = match source {
        GgufQuantizationType::F32 => 4,
        GgufQuantizationType::F16 => 2,
        _ => bail!("source must be F16 or F32"),
    };
    if !byte_len.is_multiple_of(bytes_per_value) {
        bail!(
            "input byte length ({byte_len}) is not a multiple of {} for {source:?}",
            bytes_per_value
        );
    }
    Ok(byte_len / bytes_per_value)
}

fn run(args: Args) -> Result<()> {
    quantize_file(
        &args.input,
        &args.output,
        args.source,
        args.target,
        &args.append_tensor,
    )
}

fn quantize_file(
    input_path: &Path,
    output_path: &Path,
    source: Option<GgufQuantizationType>,
    target: Option<GgufQuantizationType>,
    append_specs: &[String],
) -> Result<()> {
    let input = fs::read(input_path)
        .with_context(|| format!("failed to read input file: {}", input_path.display()))?;
    if input.starts_with(b"GGUF") {
        let output = if append_specs.is_empty() {
            let target =
                target.ok_or_else(|| anyhow!("--target is required for GGUF quantization"))?;
            quantize_gguf_bytes(&input, target)?
        } else {
            append_gguf_tensors(&input, append_specs)?
        };
        fs::write(output_path, &output)
            .with_context(|| format!("failed to write output file: {}", output_path.display()))?;
        return Ok(());
    }

    let target = target.ok_or_else(|| anyhow!("--target is required for raw tensor inputs"))?;
    let source = source.ok_or_else(|| anyhow!("--source is required for raw tensor inputs"))?;
    let value_count = source_value_count(source, input.len())?;
    let output_size = quantized_size(target, value_count)
        .map_err(|err| anyhow!(err))
        .context("failed to compute output size")?;
    let mut output = vec![0_u8; output_size];
    quantize_scalar(source, target, &input, &mut output)
        .map_err(|err| anyhow!(err))
        .context("quantization failed")?;
    fs::write(output_path, &output)
        .with_context(|| format!("failed to write output file: {}", output_path.display()))?;
    Ok(())
}

#[derive(Debug, Clone)]
struct OutputTensor {
    name: String,
    dimensions: Vec<u64>,
    ggml_type: u32,
    data: Vec<u8>,
}

fn quantize_gguf_bytes(input: &[u8], target: GgufQuantizationType) -> Result<Vec<u8>> {
    ensure_gguf_target_supported(target)?;
    let parsed = parse_gguf(input).map_err(|err| anyhow!(err))?;
    let mut metadata = parsed.metadata.clone();
    metadata.insert(
        "general.file_type".to_owned(),
        GgufMetadataValue::Uint32(gguf_type_id(target)?),
    );
    let tensors = build_output_tensors(&parsed, input, target)?;
    write_gguf(parsed.version, &metadata, &tensors, parsed.alignment)
}

fn append_gguf_tensors(input: &[u8], append_specs: &[String]) -> Result<Vec<u8>> {
    let parsed = parse_gguf(input).map_err(|err| anyhow!(err))?;
    let mut tensors = copy_existing_tensors(&parsed, input)?;
    for spec in append_specs {
        tensors.push(parse_append_tensor_spec(spec)?);
    }
    write_gguf(parsed.version, &parsed.metadata, &tensors, parsed.alignment)
}

fn copy_existing_tensors(parsed: &GgufFile, input: &[u8]) -> Result<Vec<OutputTensor>> {
    let mut tensors = Vec::with_capacity(parsed.tensor_infos.len());
    for tensor in &parsed.tensor_infos {
        let source = GgufQuantizationType::from_ggml_type(tensor.ggml_type);
        let value_count = tensor_value_count(tensor)?;
        let input_size = quantized_size(source, value_count)
            .map_err(|err| anyhow!(err))
            .with_context(|| format!("unsupported input tensor type for {}", tensor.name))?;
        let start = tensor.absolute_offset as usize;
        let end = start
            .checked_add(input_size)
            .ok_or_else(|| anyhow!("tensor {} byte range overflows", tensor.name))?;
        if end > input.len() {
            bail!("tensor {} extends past end of input GGUF", tensor.name);
        }
        tensors.push(OutputTensor {
            name: tensor.name.clone(),
            dimensions: tensor.dimensions.clone(),
            ggml_type: tensor.ggml_type,
            data: input[start..end].to_vec(),
        });
    }
    Ok(tensors)
}

fn parse_append_tensor_spec(spec: &str) -> Result<OutputTensor> {
    let parts = spec.splitn(4, ':').collect::<Vec<_>>();
    if parts.len() != 4 {
        bail!("append tensor must be name:path:dim0,dim1:type, got {spec}");
    }
    let dimensions = parts[2]
        .split(',')
        .map(|dim| {
            dim.parse::<u64>()
                .with_context(|| format!("invalid tensor dimension '{dim}'"))
        })
        .collect::<Result<Vec<_>>>()?;
    if dimensions.is_empty() {
        bail!("append tensor dimensions must not be empty");
    }
    let qtype = parse_quantization_type(parts[3]).map_err(anyhow::Error::msg)?;
    let data = fs::read(parts[1])
        .with_context(|| format!("failed to read append tensor data: {}", parts[1]))?;
    let value_count = dimensions.iter().try_fold(1_usize, |acc, dim| {
        let dim = usize::try_from(*dim).context("append tensor dimension overflows usize")?;
        acc.checked_mul(dim)
            .ok_or_else(|| anyhow!("append tensor value count overflows"))
    })?;
    let expected = quantized_size(qtype, value_count).map_err(|err| anyhow!(err))?;
    if data.len() != expected {
        bail!(
            "append tensor {} has {} bytes, expected {expected} for {:?} dims {:?}",
            parts[0],
            data.len(),
            qtype,
            dimensions
        );
    }
    Ok(OutputTensor {
        name: parts[0].to_owned(),
        dimensions,
        ggml_type: gguf_type_id(qtype)?,
        data,
    })
}

fn build_output_tensors(
    parsed: &GgufFile,
    input: &[u8],
    target: GgufQuantizationType,
) -> Result<Vec<OutputTensor>> {
    let mut tensors = Vec::with_capacity(parsed.tensor_infos.len());
    for tensor in &parsed.tensor_infos {
        let source = GgufQuantizationType::from_ggml_type(tensor.ggml_type);
        let value_count = tensor_value_count(tensor)?;
        let input_size = quantized_size(source, value_count)
            .map_err(|err| anyhow!(err))
            .with_context(|| format!("unsupported input tensor type for {}", tensor.name))?;
        let start = tensor.absolute_offset as usize;
        let end = start
            .checked_add(input_size)
            .ok_or_else(|| anyhow!("tensor {} byte range overflows", tensor.name))?;
        if end > input.len() {
            bail!("tensor {} extends past end of input GGUF", tensor.name);
        }
        let tensor_bytes = &input[start..end];

        let should_quantize = tensor.dimensions.len() >= 2
            && matches!(
                source,
                GgufQuantizationType::F32 | GgufQuantizationType::F16
            )
            && quantized_size(target, value_count).is_ok();
        let (ggml_type, data) = if should_quantize {
            let output_size = quantized_size(target, value_count).map_err(|err| anyhow!(err))?;
            let mut output = vec![0_u8; output_size];
            quantize_scalar(source, target, tensor_bytes, &mut output)
                .map_err(|err| anyhow!(err))
                .with_context(|| format!("failed to quantize tensor {}", tensor.name))?;
            (gguf_type_id(target)?, output)
        } else {
            (tensor.ggml_type, tensor_bytes.to_vec())
        };

        tensors.push(OutputTensor {
            name: tensor.name.clone(),
            dimensions: tensor.dimensions.clone(),
            ggml_type,
            data,
        });
    }
    Ok(tensors)
}

fn ensure_gguf_target_supported(target: GgufQuantizationType) -> Result<()> {
    match target {
        GgufQuantizationType::F32
        | GgufQuantizationType::F16
        | GgufQuantizationType::Q4_0
        | GgufQuantizationType::Q4_1
        | GgufQuantizationType::Q5_0
        | GgufQuantizationType::Q5_1
        | GgufQuantizationType::Q8_0 => Ok(()),
        other => bail!(
            "GGUF writing currently supports F32/F16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 targets, got {other:?}"
        ),
    }
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

fn gguf_type_id(quantization: GgufQuantizationType) -> Result<u32> {
    match quantization {
        GgufQuantizationType::F32 => Ok(0),
        GgufQuantizationType::F16 => Ok(1),
        GgufQuantizationType::Q4_0 => Ok(2),
        GgufQuantizationType::Q4_1 => Ok(3),
        GgufQuantizationType::Q5_0 => Ok(6),
        GgufQuantizationType::Q5_1 => Ok(7),
        GgufQuantizationType::Q8_0 => Ok(8),
        other => bail!("unsupported GGUF tensor type: {other:?}"),
    }
}

fn write_gguf(
    version: u32,
    metadata: &BTreeMap<String, GgufMetadataValue>,
    tensors: &[OutputTensor],
    alignment: u64,
) -> Result<Vec<u8>> {
    if alignment == 0 || !alignment.is_power_of_two() {
        bail!("invalid GGUF alignment: {alignment}");
    }

    let mut relative_offsets = Vec::with_capacity(tensors.len());
    let mut relative_offset = 0_u64;
    for tensor in tensors {
        relative_offset = align_up_u64(relative_offset, alignment)?;
        relative_offsets.push(relative_offset);
        relative_offset = relative_offset
            .checked_add(tensor.data.len() as u64)
            .ok_or_else(|| anyhow!("GGUF tensor data offset overflow"))?;
    }

    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (key, value) in metadata {
        write_string(&mut out, key);
        write_metadata_value(&mut out, value)?;
    }
    for (tensor, relative_offset) in tensors.iter().zip(relative_offsets.iter().copied()) {
        write_string(&mut out, &tensor.name);
        out.extend_from_slice(&(tensor.dimensions.len() as u32).to_le_bytes());
        for dimension in &tensor.dimensions {
            out.extend_from_slice(&dimension.to_le_bytes());
        }
        out.extend_from_slice(&tensor.ggml_type.to_le_bytes());
        out.extend_from_slice(&relative_offset.to_le_bytes());
    }

    pad_to_alignment(&mut out, alignment)?;
    let data_section_start = out.len() as u64;
    for (tensor, relative_offset) in tensors.iter().zip(relative_offsets.iter().copied()) {
        let expected_len = data_section_start
            .checked_add(relative_offset)
            .ok_or_else(|| anyhow!("GGUF output offset overflow"))?
            as usize;
        if out.len() < expected_len {
            out.resize(expected_len, 0);
        }
        out.extend_from_slice(&tensor.data);
        pad_to_alignment(&mut out, alignment)?;
    }
    Ok(out)
}

fn write_metadata_value(out: &mut Vec<u8>, value: &GgufMetadataValue) -> Result<()> {
    let value_type = metadata_value_type(value);
    out.extend_from_slice(&(value_type as u32).to_le_bytes());
    write_metadata_payload(out, value, value_type)
}

fn write_metadata_payload(
    out: &mut Vec<u8>,
    value: &GgufMetadataValue,
    value_type: GgufMetadataType,
) -> Result<()> {
    match (value_type, value) {
        (GgufMetadataType::Uint8, GgufMetadataValue::Uint8(value)) => out.push(*value),
        (GgufMetadataType::Int8, GgufMetadataValue::Int8(value)) => out.push(*value as u8),
        (GgufMetadataType::Uint16, GgufMetadataValue::Uint16(value)) => {
            out.extend_from_slice(&value.to_le_bytes())
        }
        (GgufMetadataType::Int16, GgufMetadataValue::Int16(value)) => {
            out.extend_from_slice(&value.to_le_bytes())
        }
        (GgufMetadataType::Uint32, GgufMetadataValue::Uint32(value)) => {
            out.extend_from_slice(&value.to_le_bytes())
        }
        (GgufMetadataType::Int32, GgufMetadataValue::Int32(value)) => {
            out.extend_from_slice(&value.to_le_bytes())
        }
        (GgufMetadataType::Float32, GgufMetadataValue::Float32(value)) => {
            out.extend_from_slice(&value.to_le_bytes())
        }
        (GgufMetadataType::Bool, GgufMetadataValue::Bool(value)) => out.push(u8::from(*value)),
        (GgufMetadataType::String, GgufMetadataValue::String(value)) => write_string(out, value),
        (GgufMetadataType::Array, GgufMetadataValue::Array(array)) => {
            write_metadata_array(out, array)?
        }
        (GgufMetadataType::Uint64, GgufMetadataValue::Uint64(value)) => {
            out.extend_from_slice(&value.to_le_bytes())
        }
        (GgufMetadataType::Int64, GgufMetadataValue::Int64(value)) => {
            out.extend_from_slice(&value.to_le_bytes())
        }
        (GgufMetadataType::Float64, GgufMetadataValue::Float64(value)) => {
            out.extend_from_slice(&value.to_le_bytes())
        }
        _ => bail!("metadata array contains value with mismatched type"),
    }
    Ok(())
}

fn write_metadata_array(out: &mut Vec<u8>, array: &GgufMetadataArray) -> Result<()> {
    out.extend_from_slice(&(array.element_type as u32).to_le_bytes());
    out.extend_from_slice(&(array.values.len() as u64).to_le_bytes());
    for value in &array.values {
        write_metadata_payload(out, value, array.element_type)?;
    }
    Ok(())
}

fn metadata_value_type(value: &GgufMetadataValue) -> GgufMetadataType {
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

fn write_string(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn pad_to_alignment(out: &mut Vec<u8>, alignment: u64) -> Result<()> {
    let aligned = align_up_u64(out.len() as u64, alignment)? as usize;
    out.resize(aligned, 0);
    Ok(())
}

fn align_up_u64(value: u64, alignment: u64) -> Result<u64> {
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|value| value & !mask)
        .ok_or_else(|| anyhow!("alignment overflow"))
}

fn main() {
    let args = Args::parse();
    if let Err(err) = run(args) {
        eprintln!("{err:#}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use oxidize_core::quantization::dequantize_scalar;

    #[test]
    fn parses_quantization_type_case_insensitive() {
        assert_eq!(
            parse_quantization_type("q4_0").expect("q4_0 should parse"),
            GgufQuantizationType::Q4_0
        );
        assert_eq!(
            parse_quantization_type("F16").expect("F16 should parse"),
            GgufQuantizationType::F16
        );
    }

    #[test]
    fn quantize_file_writes_expected_size_and_data() {
        let temp_dir = unique_temp_dir();
        let input_path = temp_dir.join("input.bin");
        let output_path = temp_dir.join("output.bin");

        let values = [1.0_f32, -2.0_f32, 0.5_f32, 3.5_f32];
        let mut input = Vec::with_capacity(values.len() * 4);
        for value in values {
            input.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(&input_path, &input).expect("input file should be written");

        quantize_file(
            &input_path,
            &output_path,
            Some(GgufQuantizationType::F32),
            GgufQuantizationType::F16,
        )
        .expect("quantization should succeed");

        let output = fs::read(&output_path).expect("output file should exist");
        assert_eq!(output.len(), 8);

        let mut recovered = vec![0.0_f32; values.len()];
        dequantize_scalar(GgufQuantizationType::F16, &output, &mut recovered)
            .expect("dequantization should succeed");
        assert!(recovered.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn quantize_file_rewrites_tiny_gguf_model() {
        let temp_dir = unique_temp_dir();
        let input_path = temp_dir.join("tiny-f32.gguf");
        let output_path = temp_dir.join("tiny-q8.gguf");

        let matrix_values = (0..32).map(|idx| idx as f32 - 16.0).collect::<Vec<_>>();
        let mut matrix_data = Vec::with_capacity(matrix_values.len() * 4);
        for value in &matrix_values {
            matrix_data.extend_from_slice(&value.to_le_bytes());
        }
        let norm_values = [1.0_f32, 2.0, 3.0, 4.0];
        let mut norm_data = Vec::with_capacity(norm_values.len() * 4);
        for value in norm_values {
            norm_data.extend_from_slice(&value.to_le_bytes());
        }

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
        let input = write_gguf(
            3,
            &metadata,
            &[
                OutputTensor {
                    name: "blk.0.attn_q.weight".to_owned(),
                    dimensions: vec![8, 4],
                    ggml_type: 0,
                    data: matrix_data,
                },
                OutputTensor {
                    name: "blk.0.attn_norm.weight".to_owned(),
                    dimensions: vec![4],
                    ggml_type: 0,
                    data: norm_data,
                },
            ],
            32,
        )
        .expect("tiny GGUF should be written");
        fs::write(&input_path, input).expect("tiny GGUF input should be written");

        quantize_file(&input_path, &output_path, None, GgufQuantizationType::Q8_0)
            .expect("GGUF quantization should succeed");

        let output = fs::read(&output_path).expect("output GGUF should exist");
        let parsed = parse_gguf(&output).expect("output GGUF should parse");
        assert_eq!(
            parsed.metadata.get("general.file_type"),
            Some(&GgufMetadataValue::Uint32(8))
        );
        assert_eq!(parsed.tensor_infos.len(), 2);
        assert_eq!(parsed.tensor_infos[0].ggml_type, 8);
        assert_eq!(parsed.tensor_infos[1].ggml_type, 0);
        assert_eq!(parsed.tensor_infos[0].relative_offset % 32, 0);
        assert_eq!(parsed.tensor_infos[1].relative_offset % 32, 0);

        let matrix_size = quantized_size(GgufQuantizationType::Q8_0, 32).expect("q8 size");
        let matrix_start = parsed.tensor_infos[0].absolute_offset as usize;
        let matrix_end = matrix_start + matrix_size;
        let mut recovered = vec![0.0_f32; 32];
        dequantize_scalar(
            GgufQuantizationType::Q8_0,
            &output[matrix_start..matrix_end],
            &mut recovered,
        )
        .expect("quantized matrix should dequantize");
        assert!(recovered.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn raw_quantization_requires_source_type() {
        let temp_dir = unique_temp_dir();
        let input_path = temp_dir.join("input.bin");
        let output_path = temp_dir.join("output.bin");
        fs::write(&input_path, [0_u8; 8]).expect("input file should be written");

        let err = quantize_file(&input_path, &output_path, None, GgufQuantizationType::F16)
            .expect_err("raw input without source should fail");
        assert!(err.to_string().contains("--source is required"));
    }

    #[test]
    fn source_value_count_rejects_invalid_source_alignment() {
        let err =
            source_value_count(GgufQuantizationType::F32, 3).expect_err("must reject invalid len");
        assert!(err.to_string().contains("not a multiple"));
    }

    fn unique_temp_dir() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("oxidize-quantize-test-{nanos}"));
        fs::create_dir_all(&dir).expect("temp dir should be created");
        dir
    }
}
