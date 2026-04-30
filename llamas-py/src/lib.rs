use llamas_core::generation::{GenerationConfig, GenerationStream};
use llamas_core::llama::{LlamaConfig, LlamaModel};
use llamas_core::model::Session;
use llamas_core::sampling::SamplingConfig;
use pyo3::exceptions::PyValueError;
use pyo3::ffi::c_str;
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

    #[pyo3(signature = (prompt, max_tokens=16))]
    fn generate_async(
        slf: Py<Self>,
        py: Python<'_>,
        prompt: String,
        max_tokens: usize,
    ) -> PyResult<Py<PyAny>> {
        let to_thread = resolve_to_thread(py)?;
        let generate = slf.bind(py).getattr("generate")?;
        let coroutine = to_thread
            .bind(py)
            .call1((generate, prompt, max_tokens))?;
        Ok(coroutine.unbind())
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

    #[pyo3(signature = (messages, max_tokens=16))]
    fn create_chat_completion_async(
        slf: Py<Self>,
        py: Python<'_>,
        messages: Vec<String>,
        max_tokens: usize,
    ) -> PyResult<Py<PyAny>> {
        let to_thread = resolve_to_thread(py)?;
        let create_chat_completion = slf.bind(py).getattr("create_chat_completion")?;
        let coroutine = to_thread
            .bind(py)
            .call1((create_chat_completion, messages, max_tokens))?;
        Ok(coroutine.unbind())
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

    #[pyo3(signature = (prompt_tokens, max_tokens=16, output_tensor=None))]
    fn generate_from_tokens(
        &mut self,
        py: Python<'_>,
        prompt_tokens: &Bound<'_, PyAny>,
        max_tokens: usize,
        output_tensor: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let prompt_tokens = tensor_like_to_u32_vec(prompt_tokens)?;
        let tokens = self.generate_tokens(&prompt_tokens, max_tokens)?;
        convert_u32_output(py, &tokens, output_tensor)
    }

    #[pyo3(signature = (text, output_tensor=None))]
    fn embed_tensor(
        &self,
        py: Python<'_>,
        text: &str,
        output_tensor: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let embedding = self.embed(text)?;
        convert_f32_output(py, &embedding, output_tensor)
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

fn resolve_to_thread(py: Python<'_>) -> PyResult<Py<PyAny>> {
    if let Ok(asyncio) = py.import("asyncio") {
        return Ok(asyncio.getattr("to_thread")?.unbind());
    }

    let fallback_scope = PyDict::new(py);
    py.run(
        c_str!(
            "async def _llamas_to_thread(fn, *args, **kwargs):\n    return fn(*args, **kwargs)"
        ),
        None,
        Some(&fallback_scope),
    )?;
    let to_thread = fallback_scope
        .get_item("_llamas_to_thread")?
        .ok_or_else(|| PyValueError::new_err("failed to initialize to_thread fallback"))?;
    Ok(to_thread.unbind())
}

fn tensor_like_to_u32_vec(value: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    if let Ok(values) = value.extract::<Vec<u32>>() {
        return Ok(values);
    }

    if value.hasattr("tolist")? {
        let values = value.call_method0("tolist")?;
        return values.extract::<Vec<u32>>().map_err(|_| {
            PyValueError::new_err("prompt_tokens tolist() must return a flat sequence of integers")
        });
    }

    Err(PyValueError::new_err(
        "prompt_tokens must be a sequence of integers, numpy array, or torch tensor",
    ))
}

fn convert_u32_output(
    py: Python<'_>,
    values: &[u32],
    output_tensor: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let output_kind = output_tensor.unwrap_or("list");
    match output_kind {
        "list" => Ok(PyList::new(py, values)?.into_any().unbind()),
        "numpy" => {
            let np = py.import("numpy").map_err(|_| {
                PyValueError::new_err("numpy is not available; install numpy or use output_tensor='list'")
            })?;
            let dtype = np.getattr("uint32")?;
            Ok(np
                .getattr("array")?
                .call1((PyList::new(py, values)?,))?
                .call_method1("astype", (dtype,))?
                .unbind())
        }
        "torch" => {
            let torch = py.import("torch").map_err(|_| {
                PyValueError::new_err("torch is not available; install torch or use output_tensor='list'")
            })?;
            let dtype = torch.getattr("int64")?;
            Ok(torch
                .getattr("tensor")?
                .call1((PyList::new(py, values)?,))?
                .call_method1("to", (dtype,))?
                .unbind())
        }
        _ => Err(PyValueError::new_err(
            "output_tensor must be one of: list, numpy, torch",
        )),
    }
}

fn convert_f32_output(
    py: Python<'_>,
    values: &[f32],
    output_tensor: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let output_kind = output_tensor.unwrap_or("list");
    match output_kind {
        "list" => Ok(PyList::new(py, values)?.into_any().unbind()),
        "numpy" => {
            let np = py.import("numpy").map_err(|_| {
                PyValueError::new_err("numpy is not available; install numpy or use output_tensor='list'")
            })?;
            let dtype = np.getattr("float32")?;
            Ok(np.getattr("array")?.call1((PyList::new(py, values)?, dtype))?.unbind())
        }
        "torch" => {
            let torch = py.import("torch").map_err(|_| {
                PyValueError::new_err("torch is not available; install torch or use output_tensor='list'")
            })?;
            let dtype = torch.getattr("float32")?;
            Ok(torch
                .getattr("tensor")?
                .call1((PyList::new(py, values)?,))?
                .call_method1("to", (dtype,))?
                .unbind())
        }
        _ => Err(PyValueError::new_err(
            "output_tensor must be one of: list, numpy, torch",
        )),
    }
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
    use pyo3::exceptions::PyStopIteration;
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

    #[test]
    fn llama_generate_async_supports_asyncio() {
        Python::initialize();
        Python::attach(|py| {
            let llama = Py::new(
                py,
                Llama::new("model.gguf".to_owned(), 32_000, 4096, 32)
                    .expect("llama should initialize"),
            )
            .expect("python llama instance should initialize");

            let coroutine = Llama::generate_async(llama, py, "abc".to_owned(), 4)
                .expect("async generation should return coroutine");
            let generated = extract_coroutine_result(coroutine.bind(py))
                .expect("coroutine should resolve to a value")
                .extract::<String>()
                .expect("generated response should be string");

            assert_eq!(generated, "cccc");
        });
    }

    #[test]
    fn llama_create_chat_completion_async_supports_asyncio() {
        Python::initialize();
        Python::attach(|py| {
            let llama = Py::new(
                py,
                Llama::new("model.gguf".to_owned(), 32_000, 4096, 32)
                    .expect("llama should initialize"),
            )
            .expect("python llama instance should initialize");

            let coroutine =
                Llama::create_chat_completion_async(llama, py, vec!["hi".to_owned()], 2)
                    .expect("async chat completion should return coroutine");
            let completion = extract_coroutine_result(coroutine.bind(py))
                .expect("coroutine should resolve to a value")
                .extract::<Bound<'_, PyDict>>()
                .expect("chat completion should be dict");

            let object = completion
                .get_item("object")
                .expect("object key lookup should succeed")
                .expect("object should have value")
                .extract::<String>()
                .expect("object should be string");
            assert_eq!(object, "chat.completion");
        });
    }

    fn extract_coroutine_result<'py>(coroutine: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        match coroutine.call_method1("send", (coroutine.py().None(),)) {
            Ok(_) => Err(PyValueError::new_err("coroutine unexpectedly yielded")),
            Err(err) if err.is_instance_of::<PyStopIteration>(coroutine.py()) => Ok(err
                .value(coroutine.py())
                .getattr("value")?
                .to_owned()),
            Err(err) => Err(err),
        }
    }

    #[test]
    fn generate_from_tokens_accepts_tensor_like_input() {
        Python::initialize();
        Python::attach(|py| {
            let mut llama = Llama::new("model.gguf".to_owned(), 32_000, 4096, 32)
                .expect("llama should initialize");
            let locals = PyDict::new(py);
            py.run(
                c_str!(
                    "class FakeTensor:\n    def __init__(self, values):\n        self._values = values\n    def tolist(self):\n        return self._values\nfake = FakeTensor([97, 98, 99])"
                ),
                None,
                Some(&locals),
            )
            .expect("fake tensor type should be defined");
            let fake = locals
                .get_item("fake")
                .expect("fake tensor should exist")
                .expect("fake tensor should have value");
            let generated = llama
                .generate_from_tokens(py, &fake, 4, None)
                .expect("tensor-like generation should succeed");
            let generated = generated
                .bind(py)
                .extract::<Vec<u32>>()
                .expect("default output should be a list");
            assert_eq!(generated, vec![99, 99, 99, 99]);
        });
    }

    #[test]
    fn embed_tensor_supports_numpy_when_available() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                return;
            }
            let llama = Llama::new("model.gguf".to_owned(), 32_000, 4096, 32)
                .expect("llama should initialize");
            let embedding = llama
                .embed_tensor(py, "hello", Some("numpy"))
                .expect("numpy tensor conversion should succeed");
            let class_name = embedding
                .bind(py)
                .getattr("__class__")
                .expect("class lookup should succeed")
                .getattr("__name__")
                .expect("class name should exist")
                .extract::<String>()
                .expect("class name should be string");
            assert_eq!(class_name, "ndarray");
        });
    }

    #[test]
    fn embed_tensor_supports_torch_when_available() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("torch").is_err() {
                return;
            }
            let llama = Llama::new("model.gguf".to_owned(), 32_000, 4096, 32)
                .expect("llama should initialize");
            let embedding = llama
                .embed_tensor(py, "hello", Some("torch"))
                .expect("torch tensor conversion should succeed");
            let class_name = embedding
                .bind(py)
                .getattr("__class__")
                .expect("class lookup should succeed")
                .getattr("__name__")
                .expect("class name should exist")
                .extract::<String>()
                .expect("class name should be string");
            assert_eq!(class_name, "Tensor");
        });
    }

    #[test]
    fn pyproject_configures_maturin_build_backend() {
        let pyproject = include_str!("../pyproject.toml");
        assert!(pyproject.contains("build-backend = \"maturin\""));
        assert!(pyproject.contains("module-name = \"llamas\""));
        assert!(pyproject.contains("include = [\"llamas.pyi\", \"py.typed\"]"));
    }

    #[test]
    fn python_type_stub_defines_public_api() {
        let stub = include_str!("../llamas.pyi");
        assert!(stub.contains("class Llama:"));
        assert!(stub.contains("def workspace_health() -> str: ..."));
        assert!(stub.contains("def version() -> str: ..."));
        assert!(stub.contains("def generate(self, prompt: str, max_tokens: int = 16) -> str: ..."));
    }

    #[test]
    fn py_typed_marker_file_exists() {
        let marker = include_str!("../py.typed");
        assert!(marker.trim().is_empty());
    }
}
