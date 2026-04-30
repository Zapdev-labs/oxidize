use std::collections::BTreeMap;

use thiserror::Error;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone, PartialEq)]
pub struct GgufFile {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata: BTreeMap<String, GgufMetadataValue>,
    pub tensor_infos: Vec<GgufTensorInfo>,
    pub alignment: u64,
    pub data_section_start: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: u32,
    pub relative_offset: u64,
    pub absolute_offset: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GgufMetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(GgufMetadataArray),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgufMetadataArray {
    pub element_type: GgufMetadataType,
    pub values: Vec<GgufMetadataValue>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufMetadataType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl TryFrom<u32> for GgufMetadataType {
    type Error = GgufParseError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Uint8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::Uint16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::Uint32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::Uint64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            _ => Err(GgufParseError::UnknownMetadataType(value)),
        }
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum GgufParseError {
    #[error("invalid gguf magic")]
    InvalidMagic,
    #[error("unsupported gguf version: {0}")]
    UnsupportedVersion(u32),
    #[error("unexpected end of file")]
    UnexpectedEof,
    #[error("invalid utf8 string: {0}")]
    InvalidUtf8(#[from] std::string::FromUtf8Error),
    #[error("unknown metadata type: {0}")]
    UnknownMetadataType(u32),
    #[error("invalid alignment: {0}")]
    InvalidAlignment(u64),
    #[error("integer overflow while parsing")]
    IntegerOverflow,
}

pub fn parse_gguf(bytes: &[u8]) -> Result<GgufFile, GgufParseError> {
    let mut reader = ByteReader::new(bytes);

    let magic = reader.read_exact(4)?;
    if magic != GGUF_MAGIC {
        return Err(GgufParseError::InvalidMagic);
    }

    let version = reader.read_u32()?;
    if version != 2 && version != 3 {
        return Err(GgufParseError::UnsupportedVersion(version));
    }

    let tensor_count = reader.read_u64()?;
    let metadata_count = reader.read_u64()?;

    let mut metadata = BTreeMap::new();
    for _ in 0..metadata_count {
        let key = reader.read_string()?;
        let value_type = GgufMetadataType::try_from(reader.read_u32()?)?;
        let value = reader.read_value_of_type(value_type)?;
        metadata.insert(key, value);
    }

    let mut tensor_infos = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = reader.read_string()?;
        let n_dimensions = reader.read_u32()?;
        let mut dimensions = Vec::with_capacity(n_dimensions as usize);
        for _ in 0..n_dimensions {
            dimensions.push(reader.read_u64()?);
        }
        let ggml_type = reader.read_u32()?;
        let relative_offset = reader.read_u64()?;
        tensor_infos.push(GgufTensorInfo {
            name,
            dimensions,
            ggml_type,
            relative_offset,
            absolute_offset: 0,
        });
    }

    let alignment = metadata
        .get("general.alignment")
        .map_or(Ok(DEFAULT_ALIGNMENT), alignment_from_metadata)?;
    if !alignment.is_power_of_two() {
        return Err(GgufParseError::InvalidAlignment(alignment));
    }

    let data_section_start = align_up(reader.position() as u64, alignment)?;
    if data_section_start > bytes.len() as u64 {
        return Err(GgufParseError::UnexpectedEof);
    }

    for tensor in &mut tensor_infos {
        tensor.absolute_offset = data_section_start
            .checked_add(tensor.relative_offset)
            .ok_or(GgufParseError::IntegerOverflow)?;
        if tensor.absolute_offset > bytes.len() as u64 {
            return Err(GgufParseError::UnexpectedEof);
        }
    }

    Ok(GgufFile {
        version,
        tensor_count,
        metadata,
        tensor_infos,
        alignment,
        data_section_start,
    })
}

fn alignment_from_metadata(value: &GgufMetadataValue) -> Result<u64, GgufParseError> {
    match value {
        GgufMetadataValue::Uint8(v) => Ok((*v).into()),
        GgufMetadataValue::Uint16(v) => Ok((*v).into()),
        GgufMetadataValue::Uint32(v) => Ok((*v).into()),
        GgufMetadataValue::Uint64(v) => Ok(*v),
        GgufMetadataValue::Int8(v) if *v > 0 => Ok(*v as u64),
        GgufMetadataValue::Int16(v) if *v > 0 => Ok(*v as u64),
        GgufMetadataValue::Int32(v) if *v > 0 => Ok(*v as u64),
        GgufMetadataValue::Int64(v) if *v > 0 => Ok(*v as u64),
        _ => Err(GgufParseError::InvalidAlignment(0)),
    }
}

fn align_up(value: u64, alignment: u64) -> Result<u64, GgufParseError> {
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|v| v & !mask)
        .ok_or(GgufParseError::IntegerOverflow)
}

struct ByteReader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> ByteReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }

    fn position(&self) -> usize {
        self.cursor
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8], GgufParseError> {
        let end = self
            .cursor
            .checked_add(len)
            .ok_or(GgufParseError::IntegerOverflow)?;
        if end > self.bytes.len() {
            return Err(GgufParseError::UnexpectedEof);
        }
        let out = &self.bytes[self.cursor..end];
        self.cursor = end;
        Ok(out)
    }

    fn read_u8(&mut self) -> Result<u8, GgufParseError> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_i8(&mut self) -> Result<i8, GgufParseError> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16, GgufParseError> {
        let mut buf = [0_u8; 2];
        buf.copy_from_slice(self.read_exact(2)?);
        Ok(u16::from_le_bytes(buf))
    }

    fn read_i16(&mut self) -> Result<i16, GgufParseError> {
        let mut buf = [0_u8; 2];
        buf.copy_from_slice(self.read_exact(2)?);
        Ok(i16::from_le_bytes(buf))
    }

    fn read_u32(&mut self) -> Result<u32, GgufParseError> {
        let mut buf = [0_u8; 4];
        buf.copy_from_slice(self.read_exact(4)?);
        Ok(u32::from_le_bytes(buf))
    }

    fn read_i32(&mut self) -> Result<i32, GgufParseError> {
        let mut buf = [0_u8; 4];
        buf.copy_from_slice(self.read_exact(4)?);
        Ok(i32::from_le_bytes(buf))
    }

    fn read_u64(&mut self) -> Result<u64, GgufParseError> {
        let mut buf = [0_u8; 8];
        buf.copy_from_slice(self.read_exact(8)?);
        Ok(u64::from_le_bytes(buf))
    }

    fn read_i64(&mut self) -> Result<i64, GgufParseError> {
        let mut buf = [0_u8; 8];
        buf.copy_from_slice(self.read_exact(8)?);
        Ok(i64::from_le_bytes(buf))
    }

    fn read_f32(&mut self) -> Result<f32, GgufParseError> {
        let mut buf = [0_u8; 4];
        buf.copy_from_slice(self.read_exact(4)?);
        Ok(f32::from_le_bytes(buf))
    }

    fn read_f64(&mut self) -> Result<f64, GgufParseError> {
        let mut buf = [0_u8; 8];
        buf.copy_from_slice(self.read_exact(8)?);
        Ok(f64::from_le_bytes(buf))
    }

    fn read_bool(&mut self) -> Result<bool, GgufParseError> {
        Ok(self.read_u8()? != 0)
    }

    fn read_string(&mut self) -> Result<String, GgufParseError> {
        let len = self.read_u64()?;
        let len: usize = len.try_into().map_err(|_| GgufParseError::IntegerOverflow)?;
        let bytes = self.read_exact(len)?;
        Ok(String::from_utf8(bytes.to_vec())?)
    }

    fn read_value_of_type(
        &mut self,
        value_type: GgufMetadataType,
    ) -> Result<GgufMetadataValue, GgufParseError> {
        match value_type {
            GgufMetadataType::Uint8 => Ok(GgufMetadataValue::Uint8(self.read_u8()?)),
            GgufMetadataType::Int8 => Ok(GgufMetadataValue::Int8(self.read_i8()?)),
            GgufMetadataType::Uint16 => Ok(GgufMetadataValue::Uint16(self.read_u16()?)),
            GgufMetadataType::Int16 => Ok(GgufMetadataValue::Int16(self.read_i16()?)),
            GgufMetadataType::Uint32 => Ok(GgufMetadataValue::Uint32(self.read_u32()?)),
            GgufMetadataType::Int32 => Ok(GgufMetadataValue::Int32(self.read_i32()?)),
            GgufMetadataType::Float32 => Ok(GgufMetadataValue::Float32(self.read_f32()?)),
            GgufMetadataType::Bool => Ok(GgufMetadataValue::Bool(self.read_bool()?)),
            GgufMetadataType::String => Ok(GgufMetadataValue::String(self.read_string()?)),
            GgufMetadataType::Array => {
                let element_type = GgufMetadataType::try_from(self.read_u32()?)?;
                let len = self.read_u64()?;
                let mut values = Vec::with_capacity(len as usize);
                for _ in 0..len {
                    values.push(self.read_value_of_type(element_type)?);
                }
                Ok(GgufMetadataValue::Array(GgufMetadataArray {
                    element_type,
                    values,
                }))
            }
            GgufMetadataType::Uint64 => Ok(GgufMetadataValue::Uint64(self.read_u64()?)),
            GgufMetadataType::Int64 => Ok(GgufMetadataValue::Int64(self.read_i64()?)),
            GgufMetadataType::Float64 => Ok(GgufMetadataValue::Float64(self.read_f64()?)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_v3_header_tensor_info_and_alignment() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, 1);
        push_u64(&mut bytes, 1);

        push_string(&mut bytes, "general.alignment");
        push_u32(&mut bytes, GgufMetadataType::Uint32 as u32);
        push_u32(&mut bytes, 64);

        push_string(&mut bytes, "tok_embeddings.weight");
        push_u32(&mut bytes, 2);
        push_u64(&mut bytes, 32000);
        push_u64(&mut bytes, 4096);
        push_u32(&mut bytes, 0);
        push_u64(&mut bytes, 0);

        while bytes.len() % 64 != 0 {
            bytes.push(0);
        }
        bytes.extend_from_slice(&[1, 2, 3, 4]);

        let parsed = parse_gguf(&bytes).expect("parse succeeds");

        assert_eq!(parsed.version, 3);
        assert_eq!(parsed.tensor_count, 1);
        assert_eq!(parsed.alignment, 64);
        assert_eq!(parsed.data_section_start, 128);
        assert_eq!(parsed.tensor_infos.len(), 1);
        assert_eq!(parsed.tensor_infos[0].name, "tok_embeddings.weight");
        assert_eq!(parsed.tensor_infos[0].dimensions, vec![32000, 4096]);
        assert_eq!(parsed.tensor_infos[0].absolute_offset, 128);
    }

    #[test]
    fn rejects_invalid_magic() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"ABCD");
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, 0);
        push_u64(&mut bytes, 0);

        let err = parse_gguf(&bytes).expect_err("must reject magic");
        assert_eq!(err, GgufParseError::InvalidMagic);
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        push_u32(&mut bytes, 1);
        push_u64(&mut bytes, 0);
        push_u64(&mut bytes, 0);

        let err = parse_gguf(&bytes).expect_err("must reject version");
        assert_eq!(err, GgufParseError::UnsupportedVersion(1));
    }

    #[test]
    fn rejects_invalid_alignment_value() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        push_u32(&mut bytes, 2);
        push_u64(&mut bytes, 0);
        push_u64(&mut bytes, 1);

        push_string(&mut bytes, "general.alignment");
        push_u32(&mut bytes, GgufMetadataType::Uint32 as u32);
        push_u32(&mut bytes, 3);

        let err = parse_gguf(&bytes).expect_err("must reject invalid alignment");
        assert_eq!(err, GgufParseError::InvalidAlignment(3));
    }

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        push_u64(bytes, value.len() as u64);
        bytes.extend_from_slice(value.as_bytes());
    }
}
