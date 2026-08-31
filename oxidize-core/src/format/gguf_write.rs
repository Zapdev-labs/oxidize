use std::collections::BTreeMap;

use anyhow::{Context, Result, anyhow, bail};

use super::{GgufMetadataArray, GgufMetadataType, GgufMetadataValue};

#[derive(Debug, Clone)]
pub struct GgufOutputTensor {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: u32,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
pub struct GgufTensorHeader<'a> {
    pub name: &'a str,
    pub dimensions: &'a [u64],
    pub ggml_type: u32,
}

pub fn write_gguf(
    version: u32,
    metadata: &BTreeMap<String, GgufMetadataValue>,
    tensors: &[GgufOutputTensor],
    alignment: u64,
) -> Result<Vec<u8>> {
    if alignment == 0 || !alignment.is_power_of_two() {
        bail!("invalid GGUF alignment: {alignment}");
    }

    let relative_offsets = relative_offsets(tensors, alignment)?;
    let headers: Vec<GgufTensorHeader<'_>> = tensors
        .iter()
        .map(|t| GgufTensorHeader {
            name: &t.name,
            dimensions: &t.dimensions,
            ggml_type: t.ggml_type,
        })
        .collect();
    let mut out = encode_gguf_header(version, metadata, &headers, &relative_offsets, alignment)?;
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
        pad_gguf(&mut out, alignment)?;
    }
    Ok(out)
}

pub fn encode_gguf_header(
    version: u32,
    metadata: &BTreeMap<String, GgufMetadataValue>,
    tensors: &[GgufTensorHeader<'_>],
    relative_offsets: &[u64],
    alignment: u64,
) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (key, value) in metadata {
        write_gguf_string(&mut out, key);
        write_metadata_value(&mut out, value)?;
    }
    for (tensor, relative_offset) in tensors.iter().zip(relative_offsets.iter().copied()) {
        write_gguf_string(&mut out, tensor.name);
        out.extend_from_slice(&(tensor.dimensions.len() as u32).to_le_bytes());
        for dimension in tensor.dimensions {
            out.extend_from_slice(&dimension.to_le_bytes());
        }
        out.extend_from_slice(&tensor.ggml_type.to_le_bytes());
        out.extend_from_slice(&relative_offset.to_le_bytes());
    }
    pad_gguf(&mut out, alignment)?;
    Ok(out)
}

pub fn relative_offsets(tensors: &[GgufOutputTensor], alignment: u64) -> Result<Vec<u64>> {
    relative_offsets_from_sizes(tensors.iter().map(|t| t.data.len() as u64), alignment)
}

pub fn relative_offsets_from_sizes(
    sizes: impl IntoIterator<Item = u64>,
    alignment: u64,
) -> Result<Vec<u64>> {
    let mut offsets = Vec::new();
    let mut offset = 0_u64;
    for size in sizes {
        offset = gguf_align_up(offset, alignment)?;
        offsets.push(offset);
        offset = offset
            .checked_add(size)
            .ok_or_else(|| anyhow!("GGUF tensor data offset overflow"))?;
    }
    Ok(offsets)
}

pub fn write_metadata_value(out: &mut Vec<u8>, value: &GgufMetadataValue) -> Result<()> {
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
        (GgufMetadataType::String, GgufMetadataValue::String(value)) => {
            write_gguf_string(out, value)
        }
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

pub fn metadata_value_type(value: &GgufMetadataValue) -> GgufMetadataType {
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

pub fn write_gguf_string(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

pub fn pad_gguf(out: &mut Vec<u8>, alignment: u64) -> Result<()> {
    let aligned = usize::try_from(gguf_align_up(out.len() as u64, alignment)?)
        .context("aligned output length overflows usize")?;
    out.resize(aligned, 0);
    Ok(())
}

pub fn gguf_align_up(value: u64, alignment: u64) -> Result<u64> {
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|value| value & !mask)
        .ok_or_else(|| anyhow!("alignment overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::parse_gguf;

    #[test]
    fn write_then_parse_roundtrip() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".to_owned(),
            GgufMetadataValue::String("llama".to_owned()),
        );
        let tensors = vec![GgufOutputTensor {
            name: "tok.weight".to_owned(),
            dimensions: vec![2, 4],
            ggml_type: 0,
            data: vec![0u8; 32],
        }];
        let bytes = write_gguf(3, &metadata, &tensors, 32).unwrap();
        let parsed = parse_gguf(&bytes).unwrap();
        assert_eq!(parsed.version, 3);
        assert_eq!(parsed.tensor_count, 1);
        assert_eq!(
            parsed.metadata.get("general.architecture"),
            Some(&GgufMetadataValue::String("llama".to_owned()))
        );
        assert_eq!(parsed.tensor_infos[0].name, "tok.weight");
        assert_eq!(parsed.tensor_infos[0].dimensions, vec![2, 4]);
    }
}
