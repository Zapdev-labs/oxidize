use std::collections::BTreeMap;

use anyhow::{Context, Result, anyhow, bail};
use oxidize_core::gguf::{GgufMetadataArray, GgufMetadataType, GgufMetadataValue};

#[derive(Debug, Clone)]
pub struct OutputTensor {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: u32,
    pub data: Vec<u8>,
}

pub fn write_gguf(
    version: u32,
    metadata: &BTreeMap<String, GgufMetadataValue>,
    tensors: &[OutputTensor],
    alignment: u64,
) -> Result<Vec<u8>> {
    if alignment == 0 || !alignment.is_power_of_two() {
        bail!("invalid GGUF alignment: {alignment}");
    }

    let relative_offsets = relative_offsets(tensors, alignment)?;
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
        write_tensor_info(&mut out, tensor, relative_offset);
    }

    pad_to_alignment(&mut out, alignment)?;
    let data_section_start = out.len() as u64;
    for (tensor, relative_offset) in tensors.iter().zip(relative_offsets.iter().copied()) {
        let expected_len = usize::try_from(
            data_section_start
                .checked_add(relative_offset)
                .ok_or_else(|| anyhow!("GGUF output offset overflow"))?,
        )
        .context("GGUF output offset overflows usize")?;
        if out.len() < expected_len {
            out.resize(expected_len, 0);
        }
        out.extend_from_slice(&tensor.data);
        pad_to_alignment(&mut out, alignment)?;
    }
    Ok(out)
}

fn relative_offsets(tensors: &[OutputTensor], alignment: u64) -> Result<Vec<u64>> {
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut offset = 0_u64;
    for tensor in tensors {
        offset = align_up_u64(offset, alignment)?;
        offsets.push(offset);
        offset = offset
            .checked_add(tensor.data.len() as u64)
            .ok_or_else(|| anyhow!("GGUF tensor data offset overflow"))?;
    }
    Ok(offsets)
}

fn write_tensor_info(out: &mut Vec<u8>, tensor: &OutputTensor, relative_offset: u64) {
    write_string(out, &tensor.name);
    out.extend_from_slice(&(tensor.dimensions.len() as u32).to_le_bytes());
    for dimension in &tensor.dimensions {
        out.extend_from_slice(&dimension.to_le_bytes());
    }
    out.extend_from_slice(&tensor.ggml_type.to_le_bytes());
    out.extend_from_slice(&relative_offset.to_le_bytes());
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
        _ => bail!("metadata value has mismatched type"),
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
    let aligned = usize::try_from(align_up_u64(out.len() as u64, alignment)?)
        .context("aligned output length overflows usize")?;
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
