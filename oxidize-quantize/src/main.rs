use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use oxidize_core::gguf::{
    GgufFile, GgufMetadataArray, GgufMetadataType, GgufMetadataValue, GgufQuantizationType,
    GgufTensorInfo, load_mapped_gguf, parse_gguf,
};
use oxidize_core::quantization::{quantize_scalar, quantize_scalar_weighted, quantized_size};
use rayon::prelude::*;

/// Per-tensor importance vectors (one entry per input column), keyed by tensor
/// name. Parsed from a llama.cpp `.imatrix`/`.dat` file.
type Imatrix = HashMap<String, Vec<f32>>;

/// Parse a llama.cpp legacy importance-matrix file.
///
/// Layout: `int32 n_entries`, then per entry `int32 name_len`, `name bytes`,
/// `int32 ncall`, `int32 nval`, `f32[nval]`. Stored values are summed squared
/// activations; we divide by `ncall` for a stable per-column mean.
fn load_imatrix(path: &Path) -> Result<Imatrix> {
    let bytes = fs::read(path)
        .with_context(|| format!("failed to read imatrix file: {}", path.display()))?;
    if bytes.len() >= 4 && &bytes[0..4] == b"GGUF" {
        bail!(
            "imatrix {} looks like a GGUF-format imatrix; only the legacy .dat layout is supported",
            path.display()
        );
    }
    let mut cursor = 0usize;
    let read_i32 = |buf: &[u8], at: &mut usize| -> Result<i32> {
        let end = at
            .checked_add(4)
            .ok_or_else(|| anyhow!("imatrix truncated"))?;
        if end > buf.len() {
            bail!("imatrix truncated while reading i32");
        }
        let value = i32::from_le_bytes([buf[*at], buf[*at + 1], buf[*at + 2], buf[*at + 3]]);
        *at = end;
        Ok(value)
    };

    let n_entries = read_i32(&bytes, &mut cursor)?;
    if n_entries < 0 {
        bail!("imatrix has negative entry count: {n_entries}");
    }
    let mut matrix = Imatrix::with_capacity(n_entries as usize);
    for _ in 0..n_entries {
        let name_len = read_i32(&bytes, &mut cursor)?;
        if name_len < 0 {
            bail!("imatrix entry has negative name length");
        }
        let name_len = name_len as usize;
        let name_end = cursor
            .checked_add(name_len)
            .ok_or_else(|| anyhow!("imatrix truncated"))?;
        if name_end > bytes.len() {
            bail!("imatrix truncated while reading name");
        }
        let name = String::from_utf8_lossy(&bytes[cursor..name_end]).into_owned();
        cursor = name_end;

        let ncall = read_i32(&bytes, &mut cursor)?;
        let nval = read_i32(&bytes, &mut cursor)?;
        if nval < 0 {
            bail!("imatrix entry {name} has negative value count");
        }
        let nval = nval as usize;
        let data_end = cursor
            .checked_mul(4)
            .and_then(|bytes| bytes.checked_add(cursor))
            .ok_or_else(|| anyhow!("imatrix entry {name} size overflow"))?;
        if data_end > bytes.len() {
            bail!("imatrix truncated while reading values for {name}");
        }
        let scale = if ncall > 0 { ncall as f32 } else { 1.0 };
        let mut values = Vec::with_capacity(nval);
        for chunk in bytes[cursor..data_end].chunks_exact(4) {
            let raw = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            // Clamp to non-negative finite importance; the encoder requires it.
            let v = if raw.is_finite() && raw > 0.0 {
                raw / scale
            } else {
                0.0
            };
            values.push(v);
        }
        cursor = data_end;
        matrix.insert(name, values);
    }
    Ok(matrix)
}

const STREAM_VALUES_PER_CHUNK: usize = 256 * 4096;

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
    /// Optional llama.cpp importance matrix (.imatrix/.dat) to steer error away
    /// from activation-heavy columns. Currently consumed by IQ4_XS.
    #[arg(long)]
    imatrix: Option<PathBuf>,
    /// Append an already-encoded tensor to an input GGUF without requantizing
    /// existing tensors. Format: name:path:dim0,dim1:type
    #[arg(long)]
    append_tensor: Vec<String>,
    /// Worker threads for GGUF tensor quantization. Defaults to Rayon default.
    #[arg(long)]
    threads: Option<usize>,
    /// Extend output context length with YaRN rope scaling baked into metadata.
    #[arg(long)]
    context_length: Option<u32>,
    /// Training context before YaRN extension. Defaults to the input GGUF context_length.
    #[arg(long)]
    yarn_orig_ctx: Option<u32>,
}

fn parse_quantization_type(value: &str) -> Result<GgufQuantizationType, String> {
    match value.to_ascii_uppercase().as_str() {
        "F32" => Ok(GgufQuantizationType::F32),
        "F16" => Ok(GgufQuantizationType::F16),
        "Q4_0" => Ok(GgufQuantizationType::Q4_0),
        "AL5" => Ok(GgufQuantizationType::AL5),
        "AL8" => Ok(GgufQuantizationType::AL8),
        "AL6" => Ok(GgufQuantizationType::AL6),
        "AL5_XS" => Ok(GgufQuantizationType::AL5_XS),
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
        "IQ4_XS" => Ok(GgufQuantizationType::IQ4_XS),
        "IQ4_NL" => Ok(GgufQuantizationType::IQ4_NL),
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
    if let Some(threads) = args.threads {
        if threads == 0 {
            bail!("--threads must be greater than zero");
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .map_err(|err| anyhow!(err))
            .context("failed to initialize quantization thread pool")?;
    }
    if args.context_length == Some(0) {
        bail!("--context-length must be greater than zero");
    }
    if args.yarn_orig_ctx == Some(0) {
        bail!("--yarn-orig-ctx must be greater than zero");
    }
    let imatrix = match &args.imatrix {
        Some(path) => {
            let loaded = load_imatrix(path)?;
            eprintln!(
                "loaded imatrix {} ({} tensors)",
                path.display(),
                loaded.len()
            );
            Some(loaded)
        }
        None => None,
    };
    quantize_file(
        &args.input,
        &args.output,
        args.source,
        args.target,
        &args.append_tensor,
        imatrix.as_ref(),
        args.context_length,
        args.yarn_orig_ctx,
    )
}

#[allow(clippy::too_many_arguments)]
fn quantize_file(
    input_path: &Path,
    output_path: &Path,
    source: Option<GgufQuantizationType>,
    target: Option<GgufQuantizationType>,
    append_specs: &[String],
    imatrix: Option<&Imatrix>,
    context_length: Option<u32>,
    yarn_orig_ctx: Option<u32>,
) -> Result<()> {
    if input_is_gguf(input_path)? {
        if append_specs.is_empty() {
            let target =
                target.ok_or_else(|| anyhow!("--target is required for GGUF quantization"))?;
            quantize_gguf_stream(
                input_path,
                output_path,
                target,
                imatrix,
                context_length,
                yarn_orig_ctx,
            )?;
        } else {
            let input = fs::read(input_path)
                .with_context(|| format!("failed to read input file: {}", input_path.display()))?;
            let output = append_gguf_tensors(&input, append_specs)?;
            fs::write(output_path, &output).with_context(|| {
                format!("failed to write output file: {}", output_path.display())
            })?;
        }
        return Ok(());
    }

    let input = fs::read(input_path)
        .with_context(|| format!("failed to read input file: {}", input_path.display()))?;
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

fn input_is_gguf(input_path: &Path) -> Result<bool> {
    let mut file = File::open(input_path)
        .with_context(|| format!("failed to open input file: {}", input_path.display()))?;
    let mut magic = [0_u8; 4];
    let read = file
        .read(&mut magic)
        .with_context(|| format!("failed to read input file: {}", input_path.display()))?;
    Ok(read == magic.len() && magic == *b"GGUF")
}

#[derive(Debug, Clone)]
struct OutputTensor {
    name: String,
    dimensions: Vec<u64>,
    ggml_type: u32,
    data: Vec<u8>,
}

#[derive(Debug, Clone)]
struct TensorPlan {
    name: String,
    dimensions: Vec<u64>,
    output_ggml_type: u32,
    absolute_offset: usize,
    input_size: usize,
    output_size: usize,
    source_quantization: GgufQuantizationType,
    output_quantization: GgufQuantizationType,
    quantize: bool,
    /// Per-input-column importance (length == dimensions[0]) when an imatrix
    /// entry exists for this tensor and the target consumes it.
    importance: Option<Vec<f32>>,
}

fn patch_yarn_context(
    metadata: &mut BTreeMap<String, GgufMetadataValue>,
    arch: &str,
    new_ctx: u32,
    orig_ctx: u32,
) {
    let prefix = format!("{arch}.");
    metadata.insert(
        format!("{prefix}context_length"),
        GgufMetadataValue::Uint32(new_ctx),
    );
    metadata.insert(
        format!("{prefix}rope.scaling.type"),
        GgufMetadataValue::String("yarn".to_owned()),
    );
    let factor = new_ctx as f32 / orig_ctx.max(1) as f32;
    metadata.insert(
        format!("{prefix}rope.scaling.factor"),
        GgufMetadataValue::Float32(factor),
    );
    metadata.insert(
        format!("{prefix}rope.scaling.original_context_length"),
        GgufMetadataValue::Uint32(orig_ctx),
    );
}

fn quantize_gguf_stream(
    input_path: &Path,
    output_path: &Path,
    target: GgufQuantizationType,
    imatrix: Option<&Imatrix>,
    context_length: Option<u32>,
    yarn_orig_ctx: Option<u32>,
) -> Result<()> {
    ensure_gguf_target_supported(target)?;
    let mapped = load_mapped_gguf(input_path)
        .map_err(|err| anyhow!(err))
        .with_context(|| format!("failed to mmap GGUF input: {}", input_path.display()))?;
    let parsed = mapped.parsed();
    let input = mapped.bytes();

    let mut metadata = parsed.metadata.clone();
    metadata.insert(
        "general.file_type".to_owned(),
        GgufMetadataValue::Uint32(gguf_type_id(target)?),
    );
    if let Some(new_ctx) = context_length {
        let arch = parsed.architecture().unwrap_or("llama");
        let orig = yarn_orig_ctx
            .or_else(|| {
                metadata
                    .get(&format!("{arch}.context_length"))
                    .and_then(|v| match v {
                        GgufMetadataValue::Uint32(n) => Some(*n),
                        GgufMetadataValue::Uint64(n) => (*n).try_into().ok(),
                        _ => None,
                    })
            })
            .unwrap_or(262144);
        patch_yarn_context(&mut metadata, arch, new_ctx, orig);
        eprintln!(
            "context: YaRN {orig} -> {new_ctx} (factor {:.3})",
            new_ctx as f32 / orig.max(1) as f32
        );
    }
    let plans = build_tensor_plans(parsed, input.len(), target, imatrix)?;

    let mut output = File::create(output_path)
        .with_context(|| format!("failed to create output file: {}", output_path.display()))?;
    write_gguf_stream(
        parsed.version,
        &metadata,
        &plans,
        parsed.alignment,
        input,
        &mut output,
    )
}

fn build_tensor_plans(
    parsed: &GgufFile,
    input_len: usize,
    target: GgufQuantizationType,
    imatrix: Option<&Imatrix>,
) -> Result<Vec<TensorPlan>> {
    parsed
        .tensor_infos
        .iter()
        .map(|tensor| build_tensor_plan(tensor, input_len, target, imatrix))
        .collect()
}

fn build_tensor_plan(
    tensor: &GgufTensorInfo,
    input_len: usize,
    target: GgufQuantizationType,
    imatrix: Option<&Imatrix>,
) -> Result<TensorPlan> {
    let source = GgufQuantizationType::from_ggml_type(tensor.ggml_type);
    let value_count = tensor_value_count(tensor)?;
    let input_size = quantized_size(source, value_count)
        .map_err(|err| anyhow!(err))
        .with_context(|| format!("unsupported input tensor type for {}", tensor.name))?;
    let absolute_offset = usize::try_from(tensor.absolute_offset)
        .with_context(|| format!("tensor {} offset overflows usize", tensor.name))?;
    let end = absolute_offset
        .checked_add(input_size)
        .ok_or_else(|| anyhow!("tensor {} byte range overflows", tensor.name))?;
    if end > input_len {
        bail!("tensor {} extends past end of input GGUF", tensor.name);
    }

    let output_quantization = select_output_quantization(tensor, source, target)?;
    let quantize = output_quantization != source;
    let output_size = if quantize {
        quantized_size(output_quantization, value_count).map_err(|err| anyhow!(err))?
    } else {
        input_size
    };
    let output_ggml_type = if quantize {
        ggml_type_id(output_quantization)?
    } else {
        tensor.ggml_type
    };

    let importance = tensor_importance(tensor, output_quantization, quantize, imatrix);

    Ok(TensorPlan {
        name: tensor.name.clone(),
        dimensions: tensor.dimensions.clone(),
        output_ggml_type,
        absolute_offset,
        input_size,
        output_size,
        source_quantization: source,
        output_quantization,
        quantize,
        importance,
    })
}

/// Look up the per-column importance for a tensor, if the target consumes an
/// imatrix and a length-matching entry exists. Length must equal `dimensions[0]`
/// (the input-column count) so it can be broadcast across rows during encode.
fn tensor_importance(
    tensor: &GgufTensorInfo,
    output_quantization: GgufQuantizationType,
    quantize: bool,
    imatrix: Option<&Imatrix>,
) -> Option<Vec<f32>> {
    if !quantize || output_quantization != GgufQuantizationType::IQ4_XS {
        return None;
    }
    let imatrix = imatrix?;
    let columns = usize::try_from(*tensor.dimensions.first()?).ok()?;
    let values = imatrix.get(&tensor.name)?;
    if values.len() != columns {
        eprintln!(
            "imatrix entry {} has {} values but tensor expects {} columns; quantizing without it",
            tensor.name,
            values.len(),
            columns
        );
        return None;
    }
    Some(values.clone())
}

fn select_output_quantization(
    tensor: &GgufTensorInfo,
    source: GgufQuantizationType,
    requested: GgufQuantizationType,
) -> Result<GgufQuantizationType> {
    if tensor.dimensions.len() < 2 {
        return Ok(source);
    }

    let value_count = tensor_value_count(tensor)?;
    let uniform_dense_target = matches!(
        requested,
        GgufQuantizationType::Q4_0
            | GgufQuantizationType::AL5
            | GgufQuantizationType::AL8
            | GgufQuantizationType::AL6
            | GgufQuantizationType::AL5_XS
            | GgufQuantizationType::Q4_1
            | GgufQuantizationType::Q5_0
            | GgufQuantizationType::Q5_1
            | GgufQuantizationType::Q8_0
    );
    if !uniform_dense_target
        && !matches!(
            source,
            GgufQuantizationType::F32 | GgufQuantizationType::F16 | GgufQuantizationType::BF16
        )
    {
        return Ok(source);
    }

    if requested == GgufQuantizationType::Q4_K_M
        && name_should_stay_unquantized_for_q4_k_m(&tensor.name)
    {
        return Ok(source);
    }
    let mut selected = if requested == GgufQuantizationType::Q4_K_M {
        q4_k_m_mixed_type(&tensor.name)
    } else {
        requested
    };

    if uses_k_quant_blocks(selected) {
        let row_width = tensor
            .dimensions
            .first()
            .copied()
            .and_then(|dim| usize::try_from(dim).ok())
            .ok_or_else(|| anyhow!("tensor {} first dimension overflows usize", tensor.name))?;
        if !row_width.is_multiple_of(k_quant_values_per_block(selected)) {
            selected = if row_width.is_multiple_of(32) {
                GgufQuantizationType::Q5_0
            } else {
                source
            };
        }
    }

    if quantized_size(selected, value_count).is_err() {
        return Ok(source);
    }

    Ok(selected)
}

fn q4_k_m_mixed_type(name: &str) -> GgufQuantizationType {
    // llama.cpp's Q4_K_M is a mixed preset rather than a literal "all Q4_K"
    // conversion.  For Kimi/DeepSeek, llama.cpp keeps output.weight at Q6_K
    // and uses Q4_K for the bulk of the model.  Row-width validation below
    // handles MLA tensors that need Q5_0 fallbacks.
    if name == "output.weight" {
        GgufQuantizationType::Q6_K
    } else {
        GgufQuantizationType::Q4_K_M
    }
}

fn name_should_stay_unquantized_for_q4_k_m(name: &str) -> bool {
    // DeepSeek/Kimi router weights are tiny relative to the model and strongly
    // affect expert choice. llama.cpp keeps these as F32 in its Q4_K_M output.
    name.contains("ffn_gate_inp.weight")
}

fn uses_k_quant_blocks(quantization: GgufQuantizationType) -> bool {
    matches!(
        quantization,
        GgufQuantizationType::Q2_K
            | GgufQuantizationType::Q3_K_S
            | GgufQuantizationType::Q3_K_M
            | GgufQuantizationType::Q3_K_L
            | GgufQuantizationType::Q4_K_S
            | GgufQuantizationType::Q4_K_M
            | GgufQuantizationType::Q5_K_S
            | GgufQuantizationType::Q5_K_M
            | GgufQuantizationType::Q6_K
            | GgufQuantizationType::IQ4_XS
    )
}

fn k_quant_values_per_block(_quantization: GgufQuantizationType) -> usize {
    256
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
        ggml_type: ggml_type_id(qtype)?,
        data,
    })
}

fn ensure_gguf_target_supported(target: GgufQuantizationType) -> Result<()> {
    match target {
        GgufQuantizationType::F32
        | GgufQuantizationType::F16
        | GgufQuantizationType::Q4_0
        | GgufQuantizationType::AL5
        | GgufQuantizationType::AL8
        | GgufQuantizationType::AL6
        | GgufQuantizationType::AL5_XS
        | GgufQuantizationType::Q4_1
        | GgufQuantizationType::Q5_0
        | GgufQuantizationType::Q5_1
        | GgufQuantizationType::Q8_0
        | GgufQuantizationType::Q2_K
        | GgufQuantizationType::Q3_K_S
        | GgufQuantizationType::Q3_K_M
        | GgufQuantizationType::Q3_K_L
        | GgufQuantizationType::Q4_K_S
        | GgufQuantizationType::Q4_K_M
        | GgufQuantizationType::Q5_K_S
        | GgufQuantizationType::Q5_K_M
        | GgufQuantizationType::Q6_K
        | GgufQuantizationType::IQ4_XS
        | GgufQuantizationType::IQ4_NL => Ok(()),
        other => bail!("unsupported quantization target: {other:?}"),
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
        GgufQuantizationType::AL5 => Ok(240),
        GgufQuantizationType::AL8 => Ok(241),
        GgufQuantizationType::AL6 => Ok(242),
        GgufQuantizationType::AL5_XS => Ok(243),
        GgufQuantizationType::Q4_1 => Ok(3),
        GgufQuantizationType::Q5_0 => Ok(6),
        GgufQuantizationType::Q5_1 => Ok(7),
        GgufQuantizationType::Q8_0 => Ok(8),
        GgufQuantizationType::Q2_K => Ok(10),
        GgufQuantizationType::Q3_K_S => Ok(11),
        GgufQuantizationType::Q3_K_M => Ok(12),
        GgufQuantizationType::Q3_K_L => Ok(13),
        GgufQuantizationType::Q4_K_S => Ok(14),
        GgufQuantizationType::Q4_K_M => Ok(15),
        GgufQuantizationType::Q5_K_S => Ok(16),
        GgufQuantizationType::Q5_K_M => Ok(17),
        GgufQuantizationType::Q6_K => Ok(18),
        GgufQuantizationType::IQ4_XS => Ok(30),
        GgufQuantizationType::IQ4_NL => Ok(20),
        other => bail!("unsupported GGUF tensor type: {other:?}"),
    }
}

fn ggml_type_id(quantization: GgufQuantizationType) -> Result<u32> {
    match quantization {
        GgufQuantizationType::F32 => Ok(0),
        GgufQuantizationType::F16 => Ok(1),
        GgufQuantizationType::Q4_0 => Ok(2),
        GgufQuantizationType::AL5 => Ok(240),
        GgufQuantizationType::AL8 => Ok(241),
        GgufQuantizationType::AL6 => Ok(242),
        GgufQuantizationType::AL5_XS => Ok(243),
        GgufQuantizationType::Q4_1 => Ok(3),
        GgufQuantizationType::Q5_0 => Ok(6),
        GgufQuantizationType::Q5_1 => Ok(7),
        GgufQuantizationType::Q8_0 => Ok(8),
        GgufQuantizationType::Q2_K => Ok(10),
        GgufQuantizationType::Q3_K_S
        | GgufQuantizationType::Q3_K_M
        | GgufQuantizationType::Q3_K_L => Ok(11),
        GgufQuantizationType::Q4_K_S | GgufQuantizationType::Q4_K_M => Ok(12),
        GgufQuantizationType::Q5_K_S | GgufQuantizationType::Q5_K_M => Ok(13),
        GgufQuantizationType::Q6_K => Ok(14),
        GgufQuantizationType::IQ4_XS => Ok(23),
        GgufQuantizationType::IQ4_NL => Ok(20),
        GgufQuantizationType::BF16 => Ok(30),
        other => bail!("unsupported GGML tensor type: {other:?}"),
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

fn write_gguf_stream(
    version: u32,
    metadata: &BTreeMap<String, GgufMetadataValue>,
    tensors: &[TensorPlan],
    alignment: u64,
    input: &[u8],
    output: &mut File,
) -> Result<()> {
    if alignment == 0 || !alignment.is_power_of_two() {
        bail!("invalid GGUF alignment: {alignment}");
    }

    let relative_offsets = tensor_relative_offsets(tensors, alignment)?;
    let mut header = Vec::new();
    header.extend_from_slice(b"GGUF");
    header.extend_from_slice(&version.to_le_bytes());
    header.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    header.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (key, value) in metadata {
        write_string(&mut header, key);
        write_metadata_value(&mut header, value)?;
    }
    for (tensor, relative_offset) in tensors.iter().zip(relative_offsets.iter().copied()) {
        write_string(&mut header, &tensor.name);
        header.extend_from_slice(&(tensor.dimensions.len() as u32).to_le_bytes());
        for dimension in &tensor.dimensions {
            header.extend_from_slice(&dimension.to_le_bytes());
        }
        header.extend_from_slice(&tensor.output_ggml_type.to_le_bytes());
        header.extend_from_slice(&relative_offset.to_le_bytes());
    }
    pad_to_alignment(&mut header, alignment)?;
    output.write_all(&header)?;

    let data_section_start = header.len() as u64;
    for (idx, (tensor, relative_offset)) in tensors.iter().zip(relative_offsets.iter()).enumerate()
    {
        let expected_pos = data_section_start
            .checked_add(*relative_offset)
            .ok_or_else(|| anyhow!("GGUF output offset overflow"))?;
        pad_file_to(output, expected_pos)?;
        eprintln!(
            "[{}/{}] {} - {:?} -> {:?} ({} bytes -> {} bytes)",
            idx + 1,
            tensors.len(),
            tensor.name,
            tensor.source_quantization,
            tensor.output_quantization,
            tensor.input_size,
            tensor.output_size
        );
        write_tensor_data_stream(tensor, input, output)?;
        let aligned = align_up_u64(
            expected_pos
                .checked_add(tensor.output_size as u64)
                .ok_or_else(|| anyhow!("GGUF output tensor end overflow"))?,
            alignment,
        )?;
        pad_file_to(output, aligned)?;
    }
    Ok(())
}

fn tensor_relative_offsets(tensors: &[TensorPlan], alignment: u64) -> Result<Vec<u64>> {
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut relative_offset = 0_u64;
    for tensor in tensors {
        relative_offset = align_up_u64(relative_offset, alignment)?;
        offsets.push(relative_offset);
        relative_offset = relative_offset
            .checked_add(tensor.output_size as u64)
            .ok_or_else(|| anyhow!("GGUF tensor data offset overflow"))?;
    }
    Ok(offsets)
}

fn pad_file_to(output: &mut File, target_len: u64) -> Result<()> {
    let current = output.stream_position()?;
    if current > target_len {
        bail!("output position {current} passed expected offset {target_len}");
    }
    let mut remaining = target_len - current;
    const ZEROES: [u8; 4096] = [0; 4096];
    while remaining > 0 {
        let len = usize::try_from(remaining.min(ZEROES.len() as u64))?;
        output.write_all(&ZEROES[..len])?;
        remaining -= len as u64;
    }
    Ok(())
}

fn write_tensor_data_stream(tensor: &TensorPlan, input: &[u8], output: &mut File) -> Result<()> {
    let start = tensor.absolute_offset;
    let end = start
        .checked_add(tensor.input_size)
        .ok_or_else(|| anyhow!("tensor {} byte range overflows", tensor.name))?;
    let input_bytes = &input[start..end];

    if !tensor.quantize {
        output.write_all(input_bytes)?;
        return Ok(());
    }

    let value_count = tensor_value_count_from_dimensions(&tensor.name, &tensor.dimensions)?;
    let chunk_values = stream_chunk_values(tensor.source_quantization, tensor.output_quantization);
    let al5_blocks_parallel = matches!(
        tensor.output_quantization,
        GgufQuantizationType::AL5
            | GgufQuantizationType::AL8
            | GgufQuantizationType::AL6
            | GgufQuantizationType::AL5_XS
    );
    let batch_chunks = if al5_blocks_parallel {
        1
    } else {
        rayon::current_num_threads().max(1) * 2
    };
    let mut processed = 0_usize;
    while processed < value_count {
        let mut batch = Vec::with_capacity(batch_chunks);
        for _ in 0..batch_chunks {
            if processed >= value_count {
                break;
            }
            let values = (value_count - processed).min(chunk_values);
            batch.push((processed, values));
            processed += values;
        }
        let chunks = if al5_blocks_parallel {
            batch
                .into_iter()
                .map(|(start_value, values)| {
                    quantize_tensor_chunk(tensor, input_bytes, start_value, values)
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            batch
                .par_iter()
                .map(|(start_value, values)| {
                    quantize_tensor_chunk(tensor, input_bytes, *start_value, *values)
                })
                .collect::<Result<Vec<_>>>()?
        };
        for chunk in chunks {
            output.write_all(&chunk)?;
        }
    }
    Ok(())
}

fn quantize_tensor_chunk(
    tensor: &TensorPlan,
    input_bytes: &[u8],
    start_value: usize,
    values: usize,
) -> Result<Vec<u8>> {
    let (input_start, input_len) =
        source_byte_range(tensor.source_quantization, start_value, values)?;
    let input_chunk = &input_bytes[input_start..input_start + input_len];
    let output_len =
        quantized_size(tensor.output_quantization, values).map_err(|err| anyhow!(err))?;
    let mut output_chunk = vec![0_u8; output_len];
    match &tensor.importance {
        Some(importance) => {
            // Broadcast per-column importance across rows: the global value index
            // `start_value + i` maps to column `index % columns` (row-major).
            let columns = importance.len();
            let weights = (0..values)
                .map(|i| importance[(start_value + i) % columns])
                .collect::<Vec<_>>();
            quantize_scalar_weighted(
                tensor.source_quantization,
                tensor.output_quantization,
                input_chunk,
                &mut output_chunk,
                &weights,
            )
        }
        None => quantize_scalar(
            tensor.source_quantization,
            tensor.output_quantization,
            input_chunk,
            &mut output_chunk,
        ),
    }
    .map_err(|err| anyhow!(err))
    .with_context(|| format!("failed to quantize tensor {}", tensor.name))?;
    Ok(output_chunk)
}

fn source_byte_range(
    source: GgufQuantizationType,
    start_value: usize,
    values: usize,
) -> Result<(usize, usize)> {
    let input_start = match source {
        GgufQuantizationType::F32 => start_value
            .checked_mul(4)
            .ok_or_else(|| anyhow!("source byte offset overflow"))?,
        GgufQuantizationType::F16 | GgufQuantizationType::BF16 => start_value
            .checked_mul(2)
            .ok_or_else(|| anyhow!("source byte offset overflow"))?,
        other => quantized_size(other, start_value).map_err(|err| anyhow!(err))?,
    };
    let input_len = match source {
        GgufQuantizationType::F32 => values
            .checked_mul(4)
            .ok_or_else(|| anyhow!("source byte length overflow"))?,
        GgufQuantizationType::F16 | GgufQuantizationType::BF16 => values
            .checked_mul(2)
            .ok_or_else(|| anyhow!("source byte length overflow"))?,
        other => quantized_size(other, values).map_err(|err| anyhow!(err))?,
    };
    Ok((input_start, input_len))
}

fn stream_chunk_values(source: GgufQuantizationType, target: GgufQuantizationType) -> usize {
    let source_block = if uses_k_quant_blocks(source) {
        256
    } else if matches!(
        source,
        GgufQuantizationType::Q4_0
            | GgufQuantizationType::AL5
            | GgufQuantizationType::AL8
            | GgufQuantizationType::AL6
            | GgufQuantizationType::AL5_XS
            | GgufQuantizationType::Q4_1
            | GgufQuantizationType::Q5_0
            | GgufQuantizationType::Q5_1
            | GgufQuantizationType::Q8_0
    ) {
        32
    } else {
        1
    };
    let target_block = if uses_k_quant_blocks(target) {
        256
    } else if matches!(
        target,
        GgufQuantizationType::Q4_0
            | GgufQuantizationType::AL5
            | GgufQuantizationType::AL8
            | GgufQuantizationType::AL6
            | GgufQuantizationType::AL5_XS
            | GgufQuantizationType::Q4_1
            | GgufQuantizationType::Q5_0
            | GgufQuantizationType::Q5_1
            | GgufQuantizationType::Q8_0
    ) {
        32
    } else {
        1
    };
    let block = quant_block_lcm(source_block, target_block);
    STREAM_VALUES_PER_CHUNK / block * block
}

fn quant_block_lcm(a: usize, b: usize) -> usize {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    if hi.is_multiple_of(lo) { hi } else { hi * lo }
}

fn tensor_value_count_from_dimensions(name: &str, dimensions: &[u64]) -> Result<usize> {
    dimensions.iter().try_fold(1_usize, |acc, dim| {
        let dim = usize::try_from(*dim)
            .with_context(|| format!("tensor {name} dimension overflows usize"))?;
        acc.checked_mul(dim)
            .ok_or_else(|| anyhow!("tensor {name} value count overflows"))
    })
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
            Some(GgufQuantizationType::F16),
            &[],
            None,
            None,
            None,
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

        quantize_file(
            &input_path,
            &output_path,
            None,
            Some(GgufQuantizationType::Q8_0),
            &[],
            None,
            None,
            None,
        )
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
    fn q4_k_m_policy_uses_mixed_types_and_deepseek_fallbacks() {
        let output = tensor_info("output.weight", vec![256, 256], 1);
        let output_plan =
            build_tensor_plan(&output, 256 * 256 * 2, GgufQuantizationType::Q4_K_M, None)
                .expect("output plan should build");
        assert_eq!(output_plan.output_quantization, GgufQuantizationType::Q6_K);
        assert_eq!(output_plan.output_ggml_type, 14);

        let mla = tensor_info("blk.0.attn_k_b.weight", vec![128, 512, 64, 1], 30);
        let mla_plan =
            build_tensor_plan(&mla, 128 * 512 * 64 * 2, GgufQuantizationType::Q4_K_M, None)
                .expect("MLA plan should build");
        assert_eq!(mla_plan.output_quantization, GgufQuantizationType::Q5_0);
        assert_eq!(mla_plan.output_ggml_type, 6);

        let norm = tensor_info("blk.0.attn_norm.weight", vec![256], 0);
        let norm_plan = build_tensor_plan(&norm, 256 * 4, GgufQuantizationType::Q4_K_M, None)
            .expect("norm plan should build");
        assert_eq!(norm_plan.output_quantization, GgufQuantizationType::F32);
        assert!(!norm_plan.quantize);

        let router = tensor_info("blk.0.ffn_gate_inp.weight", vec![7168, 268], 0);
        let router_plan =
            build_tensor_plan(&router, 7168 * 268 * 4, GgufQuantizationType::Q4_K_M, None)
                .expect("router plan should build");
        assert_eq!(router_plan.output_quantization, GgufQuantizationType::F32);
        assert!(!router_plan.quantize);
    }

    #[test]
    fn quantize_file_streams_q4_k_m_with_ggml_tensor_type() {
        let temp_dir = unique_temp_dir();
        let input_path = temp_dir.join("tiny-f32.gguf");
        let output_path = temp_dir.join("tiny-q4-k-m.gguf");

        let matrix_values = (0..256).map(|idx| idx as f32 / 16.0).collect::<Vec<_>>();
        let mut matrix_data = Vec::with_capacity(matrix_values.len() * 4);
        for value in &matrix_values {
            matrix_data.extend_from_slice(&value.to_le_bytes());
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
            &[OutputTensor {
                name: "blk.0.ffn_gate.weight".to_owned(),
                dimensions: vec![256, 1],
                ggml_type: 0,
                data: matrix_data,
            }],
            32,
        )
        .expect("tiny GGUF should be written");
        fs::write(&input_path, input).expect("tiny GGUF input should be written");

        quantize_file(
            &input_path,
            &output_path,
            None,
            Some(GgufQuantizationType::Q4_K_M),
            &[],
            None,
            None,
            None,
        )
        .expect("GGUF Q4_K_M quantization should succeed");

        let output = fs::read(&output_path).expect("output GGUF should exist");
        let parsed = parse_gguf(&output).expect("output GGUF should parse");
        assert_eq!(
            parsed.metadata.get("general.file_type"),
            Some(&GgufMetadataValue::Uint32(15))
        );
        assert_eq!(parsed.tensor_infos[0].ggml_type, 12);
        assert_eq!(
            output.len() - parsed.tensor_infos[0].absolute_offset as usize,
            align_up_u64(
                quantized_size(GgufQuantizationType::Q4_K_M, 256).expect("q4 size") as u64,
                32,
            )
            .expect("aligned size") as usize
        );
    }

    #[test]
    fn raw_quantization_requires_source_type() {
        let temp_dir = unique_temp_dir();
        let input_path = temp_dir.join("input.bin");
        let output_path = temp_dir.join("output.bin");
        fs::write(&input_path, [0_u8; 8]).expect("input file should be written");

        let err = quantize_file(
            &input_path,
            &output_path,
            None,
            Some(GgufQuantizationType::F16),
            &[],
            None,
            None,
            None,
        )
        .expect_err("raw input without source should fail");
        assert!(err.to_string().contains("--source is required"));
    }

    #[test]
    fn source_value_count_rejects_invalid_source_alignment() {
        let err =
            source_value_count(GgufQuantizationType::F32, 3).expect_err("must reject invalid len");
        assert!(err.to_string().contains("not a multiple"));
    }

    fn tensor_info(name: &str, dimensions: Vec<u64>, ggml_type: u32) -> GgufTensorInfo {
        GgufTensorInfo {
            name: name.to_owned(),
            dimensions,
            ggml_type,
            relative_offset: 0,
            absolute_offset: 0,
            mmap_index: 0,
        }
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
