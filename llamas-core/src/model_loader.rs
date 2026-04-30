use std::path::Path;

use crate::gguf::{GgufParseError, MappedGgufFile, load_mapped_gguf};

pub trait ModelLoader {
    type Model;
    type Error;

    fn load<P: AsRef<Path>>(&self, path: P) -> Result<Self::Model, Self::Error>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GgufModelLoader;

impl ModelLoader for GgufModelLoader {
    type Model = MappedGgufFile;
    type Error = GgufParseError;

    fn load<P: AsRef<Path>>(&self, path: P) -> Result<Self::Model, Self::Error> {
        load_mapped_gguf(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs::{self, File};
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn gguf_model_loader_loads_valid_file() {
        let bytes = valid_minimal_gguf_bytes();
        let path = write_temp_file(&bytes);

        let loader = GgufModelLoader;
        let mapped = loader.load(&path).expect("gguf loader should parse model");

        assert_eq!(mapped.parsed().version, 3);
        assert_eq!(mapped.parsed().tensor_count, 1);
        assert_eq!(mapped.parsed().alignment, 64);
        assert_eq!(mapped.bytes(), bytes.as_slice());

        fs::remove_file(path).expect("temp file removed");
    }

    #[test]
    fn model_loader_trait_supports_custom_loader() {
        #[derive(Debug)]
        struct MockLoader;

        impl ModelLoader for MockLoader {
            type Model = &'static str;
            type Error = &'static str;

            fn load<P: AsRef<Path>>(&self, path: P) -> Result<Self::Model, Self::Error> {
                if path.as_ref().to_string_lossy().contains("ok") {
                    Ok("loaded")
                } else {
                    Err("invalid path")
                }
            }
        }

        let loader = MockLoader;
        assert_eq!(loader.load("model-ok.mock"), Ok("loaded"));
        assert_eq!(loader.load("bad.mock"), Err("invalid path"));
    }

    fn valid_minimal_gguf_bytes() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, 1);
        push_u64(&mut bytes, 1);

        push_string(&mut bytes, "general.alignment");
        push_u32(&mut bytes, 4);
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
        bytes
    }

    fn write_temp_file(bytes: &[u8]) -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock is after epoch")
            .as_nanos();
        let path = env::temp_dir().join(format!("llamas-core-model-loader-{unique}.gguf"));
        let mut file = File::create(&path).expect("temp file created");
        file.write_all(bytes).expect("temp file written");
        path
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
