use std::path::Path;

use crate::gguf::{GgufFile, GgufParseError, MappedGgufFile, load_mapped_gguf, parse_gguf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadProgress {
    pub stage: &'static str,
    pub percent: u8,
    pub bytes_processed: Option<u64>,
    pub total_bytes: Option<u64>,
}

pub trait ModelLoader {
    type Model;
    type Error;

    fn load<P: AsRef<Path>>(&self, path: P) -> Result<Self::Model, Self::Error>;

    fn load_with_progress<P: AsRef<Path>, C: FnMut(LoadProgress)>(
        &self,
        path: P,
        mut on_progress: C,
    ) -> Result<Self::Model, Self::Error> {
        on_progress(LoadProgress {
            stage: "starting",
            percent: 0,
            bytes_processed: None,
            total_bytes: None,
        });
        let model = self.load(path)?;
        on_progress(LoadProgress {
            stage: "complete",
            percent: 100,
            bytes_processed: None,
            total_bytes: None,
        });
        Ok(model)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GgufModelLoader;

#[derive(Debug, Clone, PartialEq)]
pub struct BaselineGgufModel {
    bytes: Vec<u8>,
    parsed: GgufFile,
}

impl BaselineGgufModel {
    pub fn parsed(&self) -> &GgufFile {
        &self.parsed
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

pub fn load_gguf_llama_cpp_baseline<P: AsRef<Path>>(
    path: P,
) -> Result<BaselineGgufModel, GgufParseError> {
    let bytes = std::fs::read(path)?;
    let parsed = parse_gguf(&bytes)?;
    Ok(BaselineGgufModel { bytes, parsed })
}

impl ModelLoader for GgufModelLoader {
    type Model = MappedGgufFile;
    type Error = GgufParseError;

    fn load<P: AsRef<Path>>(&self, path: P) -> Result<Self::Model, Self::Error> {
        load_mapped_gguf(path)
    }

    fn load_with_progress<P: AsRef<Path>, C: FnMut(LoadProgress)>(
        &self,
        path: P,
        mut on_progress: C,
    ) -> Result<Self::Model, Self::Error> {
        let path = path.as_ref();
        let total_bytes = std::fs::metadata(path).ok().map(|metadata| metadata.len());
        on_progress(LoadProgress {
            stage: "starting",
            percent: 0,
            bytes_processed: Some(0),
            total_bytes,
        });
        on_progress(LoadProgress {
            stage: "mapping",
            percent: 35,
            bytes_processed: total_bytes.map(|len| len / 3),
            total_bytes,
        });

        let model = load_mapped_gguf(path)?;

        on_progress(LoadProgress {
            stage: "parsing",
            percent: 85,
            bytes_processed: total_bytes.map(|len| (len / 3) * 2),
            total_bytes,
        });
        on_progress(LoadProgress {
            stage: "complete",
            percent: 100,
            bytes_processed: total_bytes,
            total_bytes,
        });
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join(name)
    }

    #[test]
    fn gguf_model_loader_loads_valid_file() {
        let path = fixture_path("valid-v3.gguf");
        let bytes = fs::read(&path).expect("fixture file exists");

        let loader = GgufModelLoader;
        let mapped = loader.load(&path).expect("gguf loader should parse model");

        assert_eq!(mapped.parsed().version, 3);
        assert_eq!(mapped.parsed().tensor_count, 1);
        assert_eq!(mapped.parsed().alignment, 64);
        assert_eq!(mapped.bytes(), bytes.as_slice());
    }

    #[test]
    fn gguf_model_loader_emits_progress_callbacks() {
        let path = fixture_path("valid-v3.gguf");
        let bytes = fs::read(&path).expect("fixture file exists");
        let loader = GgufModelLoader;
        let mut events = Vec::new();

        let mapped = loader
            .load_with_progress(&path, |progress| events.push(progress))
            .expect("gguf loader should parse model with progress");

        assert_eq!(mapped.parsed().version, 3);
        assert_eq!(events.len(), 4);
        assert_eq!(events[0].stage, "starting");
        assert_eq!(events[0].percent, 0);
        assert_eq!(events[1].stage, "mapping");
        assert_eq!(events[2].stage, "parsing");
        assert_eq!(events[3].stage, "complete");
        assert_eq!(events[3].percent, 100);
        assert_eq!(events[3].bytes_processed, Some(bytes.len() as u64));
        assert_eq!(events[3].total_bytes, Some(bytes.len() as u64));
        assert!(
            events
                .windows(2)
                .all(|pair| pair[0].percent <= pair[1].percent)
        );
    }

    #[test]
    fn llama_cpp_baseline_loader_parses_valid_file() {
        let path = fixture_path("valid-v3.gguf");
        let bytes = fs::read(&path).expect("fixture file exists");

        let baseline =
            load_gguf_llama_cpp_baseline(&path).expect("baseline loader should parse model");

        assert_eq!(baseline.parsed().version, 3);
        assert_eq!(baseline.parsed().tensor_count, 1);
        assert_eq!(baseline.parsed().alignment, 64);
        assert_eq!(baseline.bytes(), bytes.as_slice());
    }

    #[test]
    fn baseline_and_mapped_loader_parse_the_same_header() {
        let path = fixture_path("valid-v3.gguf");
        let loader = GgufModelLoader;

        let mapped = loader
            .load(&path)
            .expect("mapped loader should parse model");
        let baseline =
            load_gguf_llama_cpp_baseline(&path).expect("baseline loader should parse model");

        assert_eq!(mapped.parsed(), baseline.parsed());
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

        let mut events = Vec::new();
        let loaded = loader
            .load_with_progress("model-ok.mock", |progress| events.push(progress))
            .expect("default progress loader should load");
        assert_eq!(loaded, "loaded");
        assert_eq!(
            events,
            vec![
                LoadProgress {
                    stage: "starting",
                    percent: 0,
                    bytes_processed: None,
                    total_bytes: None,
                },
                LoadProgress {
                    stage: "complete",
                    percent: 100,
                    bytes_processed: None,
                    total_bytes: None,
                },
            ]
        );
    }
}
