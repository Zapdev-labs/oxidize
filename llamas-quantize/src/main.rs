use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use llamas_core::gguf::GgufQuantizationType;
use llamas_core::quantization::{quantize_scalar, quantized_size};

#[derive(Debug, Parser)]
#[command(name = "llamas-quantize")]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, value_parser = parse_quantization_type)]
    source: GgufQuantizationType,
    #[arg(long, value_parser = parse_quantization_type)]
    target: GgufQuantizationType,
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
    quantize_file(&args.input, &args.output, args.source, args.target)
}

fn quantize_file(
    input_path: &Path,
    output_path: &Path,
    source: GgufQuantizationType,
    target: GgufQuantizationType,
) -> Result<()> {
    let input = fs::read(input_path)
        .with_context(|| format!("failed to read input file: {}", input_path.display()))?;
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
    use llamas_core::quantization::dequantize_scalar;

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
            GgufQuantizationType::F32,
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
        let dir = std::env::temp_dir().join(format!("llamas-quantize-test-{nanos}"));
        fs::create_dir_all(&dir).expect("temp dir should be created");
        dir
    }
}
