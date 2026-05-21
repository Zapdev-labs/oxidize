use crate::tensor::DType;
use memmap2::Mmap;
use safetensors::tensor::{SafeTensors, TensorView};
use std::fs::File;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SafeTensorsError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("SafeTensors parse error: {0}")]
    Parse(String),
    #[error("Unsupported dtype: {0:?}")]
    UnsupportedDtype(safetensors::tensor::Dtype),
}

#[derive(Debug, Clone, PartialEq)]
pub struct SafeTensorsTensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub absolute_offset: usize,
    pub size_bytes: usize,
}

/// A memory-mapped SafeTensors file, similar to `MappedGgufFile`.
pub struct MappedSafeTensorsFile {
    mmap: Mmap,
    tensors: Vec<SafeTensorsTensorInfo>,
}

impl MappedSafeTensorsFile {
    pub fn tensors(&self) -> &[SafeTensorsTensorInfo] {
        &self.tensors
    }

    pub fn bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Get the raw byte slice for a tensor by name.
    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        let info = self.tensors.iter().find(|t| t.name == name)?;
        Some(&self.mmap[info.absolute_offset..info.absolute_offset + info.size_bytes])
    }
}

pub fn load_mapped_safetensors<P: AsRef<Path>>(
    path: P,
) -> Result<MappedSafeTensorsFile, SafeTensorsError> {
    let file = File::open(path)?;
    // SAFETY: The returned mapping is read-only and we keep it alive for as long as
    // the metadata is exposed from MappedSafeTensorsFile.
    let mmap = unsafe { Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)
        .map_err(|e| SafeTensorsError::Parse(format!("{e:?}")))?;

    let header_len = u64::from_le_bytes([
        mmap[0], mmap[1], mmap[2], mmap[3], mmap[4], mmap[5], mmap[6], mmap[7],
    ]) as usize;
    let data_start = 8 + header_len;

    let mut tensors = Vec::with_capacity(st.len());
    for (name, view) in st.tensors() {
        let shape: Vec<usize> = view.shape().iter().map(|&d| d as usize).collect();
        let dtype = convert_dtype(view.dtype())?;
        let size_bytes = view.data().len();

        // Compute absolute offset within the file
        let relative_offset = view.data().as_ptr() as usize - mmap.as_ptr() as usize;

        tensors.push(SafeTensorsTensorInfo {
            name: name.to_string(),
            shape,
            dtype,
            absolute_offset: relative_offset,
            size_bytes,
        });
    }

    Ok(MappedSafeTensorsFile { mmap, tensors })
}

fn convert_dtype(dt: safetensors::tensor::Dtype) -> Result<DType, SafeTensorsError> {
    match dt {
        safetensors::tensor::Dtype::F32 => Ok(DType::F32),
        safetensors::tensor::Dtype::F16 => Ok(DType::F16),
        safetensors::tensor::Dtype::I8 => Ok(DType::I8),
        safetensors::tensor::Dtype::I16 => Ok(DType::I16),
        safetensors::tensor::Dtype::I32 => Ok(DType::I32),
        safetensors::tensor::Dtype::I64 => Ok(DType::I64),
        safetensors::tensor::Dtype::BOOL => Ok(DType::I8), // map bool to i8
        other => Err(SafeTensorsError::UnsupportedDtype(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn create_test_safetensors(path: &std::path::Path) {
        use safetensors::tensor::{Dtype, TensorView};
        use std::collections::HashMap;

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tensor = TensorView::new(Dtype::F32, vec![2, 2], &bytes).unwrap();

        let mut tensors = HashMap::new();
        tensors.insert("weight".to_string(), tensor);

        let st = safetensors::tensor::serialize(&tensors, &None).unwrap();
        let mut file = File::create(path).unwrap();
        file.write_all(&st).unwrap();
    }

    #[test]
    fn loads_mapped_safetensors() {
        let tmp = std::env::temp_dir().join(format!(
            "test-{}.safetensors", std::process::id()));
        create_test_safetensors(&tmp);

        let mapped = load_mapped_safetensors(&tmp).expect("should load safetensors");
        assert_eq!(mapped.tensors().len(), 1);
        assert_eq!(mapped.tensors()[0].name, "weight");
        assert_eq!(mapped.tensors()[0].shape, vec![2, 2]);
        assert_eq!(mapped.tensors()[0].dtype, DType::F32);

        let data = mapped.tensor_data("weight").expect("should find tensor");
        let floats: Vec<f32> = data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        assert_eq!(floats, vec![1.0, 2.0, 3.0, 4.0]);

        let _ = std::fs::remove_file(&tmp);
    }
}
