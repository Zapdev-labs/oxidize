use llamas_core::generation::{GenerationConfig, GenerationStream};
use llamas_core::llama::{LlamaConfig, LlamaModel};
use llamas_core::model::Session;
use llamas_core::sampling::SamplingConfig;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::pin::Pin;
use std::task::{Context, Poll, Wake, Waker};

const PYTHON_PACKAGE_VERSION: &str = env!("CARGO_PKG_VERSION");
const PYTHON_PACKAGE_EXPORTS: [&str; 3] = ["workspace_health", "version", "Llama"];

#[pyclass]
pub struct Llama {
    model_path: String,
    model: LlamaModel,
    session: Session,
}

#[pymethods]
impl Llama {
    #[new]
    #[pyo3(signature = (model_path, vocab_size=32000, context_size=4096, layer_count=32))]
    fn new(
        model_path: String,
        vocab_size: usize,
        context_size: usize,
        layer_count: usize,
    ) -> PyResult<Self> {
        if model_path.trim().is_empty() {
            return Err(PyValueError::new_err("model_path cannot be empty"));
        }
        if vocab_size == 0 || context_size == 0 || layer_count == 0 {
            return Err(PyValueError::new_err(
                "vocab_size, context_size, and layer_count must be greater than zero",
            ));
        }

        Ok(Self {
            model_path,
            model: LlamaModel::new(LlamaConfig::llama3(vocab_size, context_size, layer_count)),
            session: Session::new(),
        })
    }

    #[pyo3(signature = (prompt, max_tokens=16))]
    fn generate(&mut self, prompt: &str, max_tokens: usize) -> PyResult<String> {
        if prompt.is_empty() {
            return Err(PyValueError::new_err("prompt cannot be empty"));
        }
        if max_tokens == 0 {
            return Ok(String::new());
        }

        let prompt_tokens: Vec<u32> = prompt.bytes().map(u32::from).collect();
        let sampled_tokens = self.generate_tokens(&prompt_tokens, max_tokens)?;
        Ok(Self::decode_tokens(&sampled_tokens))
    }

    #[pyo3(signature = (messages, max_tokens=16))]
    fn create_chat_completion(
        &mut self,
        py: Python<'_>,
        messages: Vec<String>,
        max_tokens: usize,
    ) -> PyResult<Py<PyDict>> {
        if messages.is_empty() {
            return Err(PyValueError::new_err("messages cannot be empty"));
        }

        let prompt = messages.join("\n");
        let completion_text = self.generate(&prompt, max_tokens)?;
        let choice = PyDict::new(py);
        let message = PyDict::new(py);
        message.set_item("role", "assistant")?;
        message.set_item("content", completion_text)?;
        choice.set_item("index", 0)?;
        choice.set_item("message", message)?;
        choice.set_item("finish_reason", "length")?;

        let response = PyDict::new(py);
        response.set_item("id", format!("chatcmpl-{}", self.model_path))?;
        response.set_item("object", "chat.completion")?;
        response.set_item("choices", PyList::new(py, &[choice])?)?;
        Ok(response.unbind())
    }

    fn embed(&self, text: &str) -> PyResult<Vec<f32>> {
        if text.is_empty() {
            return Err(PyValueError::new_err("text cannot be empty"));
        }

        let mut buckets = [0.0_f32; 8];
        for (idx, byte) in text.bytes().enumerate() {
            buckets[idx % buckets.len()] += f32::from(byte);
        }

        let norm = buckets.iter().copied().sum::<f32>();
        if norm == 0.0 {
            return Ok(vec![0.0; buckets.len()]);
        }

        Ok(buckets.into_iter().map(|value| value / norm).collect())
    }
}

impl Llama {
    fn generate_tokens(&mut self, prompt_tokens: &[u32], max_tokens: usize) -> PyResult<Vec<u32>> {
        let mut stream = GenerationStream::new(
            &mut self.model,
            &mut self.session,
            prompt_tokens,
            GenerationConfig {
                max_new_tokens: max_tokens,
                sampling: SamplingConfig {
                    temperature: 0.01,
                    ..SamplingConfig::default()
                },
                ..GenerationConfig::default()
            },
            || 0.5,
        );

        let waker: Waker = Waker::from(std::sync::Arc::new(NoopWaker));
        let mut cx = Context::from_waker(&waker);
        let mut pinned = Pin::new(&mut stream);
        let mut tokens = Vec::with_capacity(max_tokens);
        loop {
            match futures_core::Stream::poll_next(pinned.as_mut(), &mut cx) {
                Poll::Ready(Some(Ok(token))) => tokens.push(token),
                Poll::Ready(Some(Err(err))) => {
                    return Err(PyValueError::new_err(format!(
                        "generation failed: {err:?}"
                    )));
                }
                Poll::Ready(None) => break,
                Poll::Pending => {
                    return Err(PyValueError::new_err("generation unexpectedly pending"));
                }
            }
        }

        Ok(tokens)
    }

    fn decode_tokens(tokens: &[u32]) -> String {
        tokens
            .iter()
            .map(|token| char::from_u32(*token).unwrap_or('?'))
            .collect()
    }
}

#[derive(Default)]
struct NoopWaker;

impl Wake for NoopWaker {
    fn wake(self: std::sync::Arc<Self>) {}
}

fn health_status() -> &'static str {
    llamas_core::workspace_health().status
}

#[pyfunction]
fn workspace_health() -> &'static str {
    health_status()
}

#[pyfunction]
fn version() -> &'static str {
    PYTHON_PACKAGE_VERSION
}

#[pymodule]
fn llamas(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Llama>()?;
    module.add_function(wrap_pyfunction!(workspace_health, module)?)?;
    module.add_function(wrap_pyfunction!(version, module)?)?;
    module.add("__version__", PYTHON_PACKAGE_VERSION)?;
    module.add("__all__", PYTHON_PACKAGE_EXPORTS)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    #[test]
    fn health_status_matches_core_workspace_health() {
        assert_eq!(health_status(), "ready");
    }

    #[test]
    fn python_package_has_workspace_version() {
        assert_eq!(version(), env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn python_package_exports_expected_symbols() {
        assert_eq!(PYTHON_PACKAGE_EXPORTS, ["workspace_health", "version", "Llama"]);
    }

    #[test]
    fn llama_rejects_invalid_constructor_arguments() {
        let empty_path = Llama::new(String::new(), 32_000, 4096, 32);
        assert!(empty_path.is_err());

        let invalid_sizes = Llama::new("model.gguf".to_owned(), 0, 4096, 32);
        assert!(invalid_sizes.is_err());
    }

    #[test]
    fn llama_generate_returns_text_for_valid_prompt() {
        let mut llama = Llama::new("model.gguf".to_owned(), 32_000, 4096, 32)
            .expect("llama should initialize");
        let generated = llama.generate("abc", 4).expect("generation should succeed");

        assert_eq!(generated, "cccc");
    }

    #[test]
    fn llama_embed_returns_normalized_vector() {
        let llama = Llama::new("model.gguf".to_owned(), 32_000, 4096, 32)
            .expect("llama should initialize");
        let embedding = llama.embed("hello").expect("embedding should succeed");

        assert_eq!(embedding.len(), 8);
        let sum: f32 = embedding.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn llama_create_chat_completion_returns_openai_shape() {
        Python::initialize();
        Python::attach(|py| {
            let mut llama = Llama::new("model.gguf".to_owned(), 32_000, 4096, 32)
                .expect("llama should initialize");
            let completion = llama
                .create_chat_completion(py, vec!["hi".to_owned()], 2)
                .expect("chat completion should succeed");
            let completion_ref = completion.bind(py);

            let object = completion_ref
                .get_item("object")
                .expect("object key should exist")
                .expect("object should have value");
            let object = object.extract::<String>().expect("object should be string");
            assert_eq!(object, "chat.completion");

            let choices = completion_ref
                .get_item("choices")
                .expect("choices key should exist")
                .expect("choices should have value")
                .extract::<Vec<Bound<'_, PyDict>>>()
                .expect("choices should be list of dicts");
            assert_eq!(choices.len(), 1);
        });
    }
}
