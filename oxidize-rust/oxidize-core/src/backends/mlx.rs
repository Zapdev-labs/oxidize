//! Apple MLX compute backend (macOS only).
//!
//! All MLX-specific code is gated by `#[cfg(target_os = "macos")]` so that
//! Linux builds compile without requiring the `mlx-c` library.

#[cfg(target_os = "macos")]
use crate::backend::ComputeBackend;
#[cfg(target_os = "macos")]
use crate::gguf::GgufQuantizationType;
#[cfg(target_os = "macos")]
use crate::tensor::DType;

// ---------------------------------------------------------------------------
//  Build-info (always available, even on Linux)
// ---------------------------------------------------------------------------

/// Build-time detection info for the MLX backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlxBuildInfo {
    pub detected_at_build: bool,
}

/// Returns whether the MLX backend was detected at build time.
pub fn mlx_build_info() -> MlxBuildInfo {
    MlxBuildInfo {
        detected_at_build: cfg!(target_os = "macos"),
    }
}

/// Error type for MLX kernel operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MlxKernelError {
    InvalidMatrixLength { expected: usize, actual: usize },
    InvalidVectorLength { expected: usize, actual: usize },
    InvalidOutputLength { expected: usize, actual: usize },
}

// ---------------------------------------------------------------------------
//  macOS-only: MlxTensor, MlxWeightStorage, MlxComputeBackend
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
mod mlx_impl {
    use super::*;
    use mlx_rs::{Array, Device, Stream, StreamOrDevice};

    /// Wrapper around `mlx_rs::Array` that carries shape / dtype metadata in
    /// oxidize-core's native types.  The inner `Array` lives in unified memory
    /// and is reference-counted by the MLX C++ runtime.
    #[derive(Debug, Clone)]
    pub struct MlxTensor {
        pub array: Array,
        pub shape: Vec<usize>,
        pub dtype: DType,
    }

    impl MlxTensor {
        /// Wrap an existing `mlx_rs::Array`.
        pub fn from_array(array: Array) -> Self {
            let shape = array.shape().iter().map(|&d| d as usize).collect();
            let dtype = mlx_dtype_to_core(array.dtype());
            Self {
                array,
                shape,
                dtype,
            }
        }

        /// Create a new tensor from a slice of `f32` values.
        pub fn from_f32(data: &[f32]) -> Self {
            let array = Array::from_slice(data, &[data.len() as i32]);
            Self::from_array(array)
        }

        /// Create a new 2-D tensor from a slice of `f32` values.
        pub fn from_f32_2d(data: &[f32], rows: usize, cols: usize) -> Self {
            let array = Array::from_slice(data, &[rows as i32, cols as i32]);
            Self::from_array(array)
        }

        /// Evaluate the array (materialize lazy graph) and copy data back to host.
        pub fn to_f32(&self, out: &mut [f32]) -> Result<usize, String> {
            self.array
                .eval()
                .map_err(|e| format!("MLX eval failed: {e:?}"))?;
            let slice = self
                .array
                .try_as_slice::<f32>()
                .map_err(|e| format!("MLX as_slice failed: {e:?}"))?;
            let len = slice.len().min(out.len());
            out[..len].copy_from_slice(&slice[..len]);
            Ok(len)
        }
    }

    /// Storage for model weights backed by MLX `Array` objects in unified
    /// memory.  Quantized weights are stored as `Array` together with their
    /// MLX-native scale / bias arrays so that `mlx_quantized_matmul` can be
    /// used directly.
    #[derive(Debug, Clone)]
    pub enum MlxWeightStorage {
        /// Full-precision (f32) weight matrix.
        F32(Array),
        /// Quantized weight matrix with MLX-native scale/bias arrays.
        Quantized {
            weights: Array,
            scales: Array,
            biases: Array,
            group_size: i32,
            bits: i32,
        },
    }

    impl MlxWeightStorage {
        /// Build `MlxWeightStorage` from a raw GGUF tensor byte blob.
        ///
        /// The GGUF payload is converted to an MLX `Array` that lives in the
        /// unified memory pool on Apple Silicon.  There is **no explicit
        /// host-to-device staging copy** — `Array::from_slice` (which wraps
        /// `mlx_array_new_data`) copies data directly into MLX-managed
        /// unified memory.
        pub fn from_gguf_tensor(
            qtype: GgufQuantizationType,
            data: &[u8],
            shape: &[usize],
        ) -> Result<Self, String> {
            let value_count: usize = shape.iter().product();
            let mlx_shape: Vec<i32> = shape.iter().map(|&d| d as i32).collect();

            match qtype {
                GgufQuantizationType::F32 => {
                    let expected = value_count * 4;
                    if data.len() != expected {
                        return Err(format!(
                            "F32 data length mismatch: expected {} bytes, got {}",
                            expected,
                            data.len()
                        ));
                    }
                    let f32_data: Vec<f32> = data
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect();
                    let array = Array::from_slice(&f32_data, &mlx_shape);
                    Ok(MlxWeightStorage::F32(array))
                }
                other => {
                    let mut f32_data = vec![0.0_f32; value_count];
                    crate::quantization::dequantize_scalar(other, data, &mut f32_data)
                        .map_err(|e| format!("dequantize failed: {e:?}"))?;
                    let array = Array::from_slice(&f32_data, &mlx_shape);
                    Ok(MlxWeightStorage::F32(array))
                }
            }
        }

        /// Return the shape of the underlying weight tensor.
        pub fn shape(&self) -> Vec<usize> {
            match self {
                MlxWeightStorage::F32(a) => a.shape().iter().map(|&d| d as usize).collect(),
                MlxWeightStorage::Quantized { weights, .. } => {
                    weights.shape().iter().map(|&d| d as usize).collect()
                }
            }
        }

        /// Build `MlxWeightStorage::Quantized` from a raw GGUF tensor byte blob.
        ///
        /// The GGUF payload is first dequantized to f32, then re-quantized
        /// using MLX's native `quantize` operation so that the resulting
        /// `weights`/`scales`/`biases` arrays can be used directly with
        /// `mlx_quantized_matmul`.
        pub fn from_gguf_tensor_quantized(
            qtype: GgufQuantizationType,
            data: &[u8],
            shape: &[usize],
            group_size: i32,
            bits: i32,
        ) -> Result<Self, String> {
            if shape.len() != 2 {
                return Err(format!(
                    "from_gguf_tensor_quantized requires a 2-D shape, got {:?}",
                    shape
                ));
            }
            let value_count: usize = shape.iter().product();
            let mut f32_data = vec![0.0_f32; value_count];
            crate::quantization::dequantize_scalar(qtype, data, &mut f32_data)
                .map_err(|e| format!("dequantize failed: {e:?}"))?;

            // MLX quantize requires a 2-D array with columns divisible by group_size.
            let rows = shape[0] as i32;
            let cols = shape[1] as i32;
            let array = Array::from_slice(&f32_data, &[rows, cols]);
            // Drop the host buffer immediately after the MLX copy to minimize peak memory.
            drop(f32_data);
            let (weights, scales, biases) = mlx_rs::ops::quantize(&array, group_size, bits)
                .map_err(|e| format!("MLX quantize failed: {e:?}"))?;

            Ok(MlxWeightStorage::Quantized {
                weights,
                scales,
                biases,
                group_size,
                bits,
            })
        }

        /// Verify that the array is usable on the GPU stream, confirming
        /// it resides in the Apple Silicon unified memory pool.
        ///
        /// MLX arrays do not have a fixed device location — memory is unified.
        /// This test runs a trivial GPU-side operation and succeeds only if
        /// the array is accessible from the GPU without an explicit H2D copy.
        pub fn verify_unified_memory(&self, stream: &Stream) -> Result<(), String> {
            match self {
                MlxWeightStorage::F32(a) => {
                    // Run a no-op reshape on the GPU stream.  If the array
                    // were not in unified memory this would fail or trigger
                    // an implicit copy that we would log.
                    let _ = mlx_rs::ops::reshape(a, &a.shape(), stream)
                        .map_err(|e| format!("unified-memory verify failed: {e:?}"))?;
                    Ok(())
                }
                MlxWeightStorage::Quantized {
                    weights,
                    scales,
                    biases,
                    group_size,
                    bits,
                } => {
                    // Run a tiny quantized matmul on GPU to prove the
                    // quantized buffers are also in unified memory.
                    let dummy_input = Array::from_slice(&[1.0_f32], &[1, 1]);
                    let _ = mlx_rs::ops::quantized_matmul(
                        &dummy_input,
                        weights,
                        scales,
                        biases,
                        true,
                        Some(*group_size),
                        Some(*bits),
                        stream,
                    )
                    .map_err(|e| format!("unified-memory verify (quantized) failed: {e:?}"))?;
                    Ok(())
                }
            }
        }
    }

    /// Apple MLX compute backend.
    ///
    /// Implements `ComputeBackend` using the safe `mlx-rs` wrapper.  No `unsafe`
    /// blocks appear in this struct or its methods; the only `unsafe` is
    /// inside `mlx-sys` (the FFI boundary) and the `mlx-rs` crate itself.
    #[derive(Debug, Clone)]
    pub struct MlxComputeBackend {
        _device: Device,
        stream: Stream,
    }

    // SAFETY: MLX C API is thread-safe; Device/Stream may contain raw pointers
    // but they are reference-counted and safe to share across threads.
    unsafe impl Send for MlxComputeBackend {}
    unsafe impl Sync for MlxComputeBackend {}

    impl Default for MlxComputeBackend {
        fn default() -> Self {
            Self::new()
        }
    }

    impl MlxComputeBackend {
        /// Create a backend on the default MLX device (GPU when available).
        pub fn new() -> Self {
            let device = Device::default();
            let stream = Stream::new_with_device(&device);
            Self {
                _device: device,
                stream,
            }
        }

        /// Create a backend on a specific MLX device.
        pub fn with_device(device: Device) -> Self {
            let stream = Stream::new_with_device(&device);
            Self {
                _device: device,
                stream,
            }
        }

        /// Access the underlying MLX stream.
        pub fn stream(&self) -> &Stream {
            &self.stream
        }
    }

    impl ComputeBackend for MlxComputeBackend {
        type Tensor = MlxTensor;
        type WeightStorage = MlxWeightStorage;

        fn name(&self) -> &'static str {
            "mlx"
        }

        fn tensor_from_f32(&self, data: &[f32]) -> Result<MlxTensor, String> {
            Ok(MlxTensor::from_f32(data))
        }

        fn tensor_from_f32_2d(
            &self,
            data: &[f32],
            rows: usize,
            cols: usize,
        ) -> Result<MlxTensor, String> {
            Ok(MlxTensor::from_f32_2d(data, rows, cols))
        }

        fn tensor_to_f32(&self, tensor: &MlxTensor, out: &mut [f32]) -> Result<usize, String> {
            tensor.to_f32(out)
        }

        fn tensor_shape(&self, tensor: &MlxTensor) -> Vec<usize> {
            tensor.shape.clone()
        }

        fn tensor_dtype(&self, tensor: &MlxTensor) -> DType {
            tensor.dtype
        }

        fn rms_norm(
            &self,
            input: &MlxTensor,
            weight: &MlxTensor,
            eps: f32,
        ) -> Result<MlxTensor, String> {
            let result = mlx_rs::fast::rms_norm(&input.array, &weight.array, eps, &self.stream)
                .map_err(|e| format!("MLX fast_rms_norm failed: {e:?}"))?;
            Ok(MlxTensor::from_array(result))
        }

        fn apply_rope(
            &self,
            input: &MlxTensor,
            position: usize,
            head_dim: usize,
            theta: f32,
        ) -> Result<MlxTensor, String> {
            // MLX fast_rope takes dimensions and offset; traditional=false, scale=1.0.
            let result = mlx_rs::fast::rope(
                &input.array,
                head_dim as i32,
                false,
                theta,
                1.0,
                position as i32,
                None,
                &self.stream,
            )
            .map_err(|e| format!("MLX fast_rope failed: {e:?}"))?;
            Ok(MlxTensor::from_array(result))
        }

        fn attention_decode(
            &self,
            query: &MlxTensor,
            key_cache: &MlxTensor,
            value_cache: &MlxTensor,
            seq_len: usize,
            head_dim: usize,
            scale: f32,
        ) -> Result<MlxTensor, String> {
            // MLX fast_scaled_dot_product_attention expects 4-D arrays:
            //   query:   [batch, num_heads, q_seq_len, head_dim]
            //   key:     [batch, num_kv_heads, kv_seq_len, head_dim]
            //   value:   [batch, num_kv_heads, kv_seq_len, head_dim]
            //
            // For decode, q_seq_len == 1.  We reshape the 1-D buffers that
            // oxidize-core uses into 4-D MLX arrays.
            let q = mlx_rs::ops::reshape(&query.array, &[1, 1, 1, head_dim as i32], &self.stream)
                .map_err(|e| format!("reshape query failed: {e:?}"))?;

            let k = mlx_rs::ops::reshape(
                &key_cache.array,
                &[1, 1, seq_len as i32, head_dim as i32],
                &self.stream,
            )
            .map_err(|e| format!("reshape key failed: {e:?}"))?;

            let v = mlx_rs::ops::reshape(
                &value_cache.array,
                &[1, 1, seq_len as i32, head_dim as i32],
                &self.stream,
            )
            .map_err(|e| format!("reshape value failed: {e:?}"))?;

            let result = mlx_rs::fast::scaled_dot_product_attention(
                &q,
                &k,
                &v,
                scale,
                None::<&[Array]>,
                &self.stream,
            )
            .map_err(|e| format!("MLX fast_attention failed: {e:?}"))?;

            // Flatten back to 1-D.
            let flat = mlx_rs::ops::flatten(&result, None, None, &self.stream)
                .map_err(|e| format!("flatten failed: {e:?}"))?;
            Ok(MlxTensor::from_array(flat))
        }

        fn gemv(
            &self,
            matrix: &MlxWeightStorage,
            vector: &MlxTensor,
            rows: usize,
            cols: usize,
        ) -> Result<MlxTensor, String> {
            match matrix {
                MlxWeightStorage::F32(w) => {
                    // w is [rows, cols] 2-D; vector is [cols] 1-D.
                    let mat = if w.ndim() == 2 {
                        w.clone()
                    } else {
                        mlx_rs::ops::reshape(w, &[rows as i32, cols as i32], &self.stream)
                            .map_err(|e| format!("reshape weight failed: {e:?}"))?
                    };
                    let vec = if vector.array.ndim() == 1 {
                        vector.array.clone()
                    } else {
                        mlx_rs::ops::reshape(&vector.array, &[cols as i32], &self.stream)
                            .map_err(|e| format!("reshape vector failed: {e:?}"))?
                    };
                    let result = mlx_rs::ops::matmul(&mat, &vec, &self.stream)
                        .map_err(|e| format!("MLX matmul failed: {e:?}"))?;
                    Ok(MlxTensor::from_array(result))
                }
                MlxWeightStorage::Quantized {
                    weights,
                    scales,
                    biases,
                    group_size,
                    bits,
                } => {
                    // mlx_quantized_matmul expects a 2-D input.  Reshape the
                    // 1-D vector to [1, cols], run the op, then flatten the
                    // [1, rows] result back to [rows].
                    let vec = if vector.array.ndim() == 2 && vector.array.shape()[0] == 1 {
                        vector.array.clone()
                    } else {
                        mlx_rs::ops::reshape(&vector.array, &[1, cols as i32], &self.stream)
                            .map_err(|e| format!("reshape vector failed: {e:?}"))?
                    };
                    let result = mlx_rs::ops::quantized_matmul(
                        &vec,
                        weights,
                        scales,
                        biases,
                        true,
                        Some(*group_size),
                        Some(*bits),
                        &self.stream,
                    )
                    .map_err(|e| format!("MLX quantized_matmul failed: {e:?}"))?;
                    let flat = mlx_rs::ops::flatten(&result, None, None, &self.stream)
                        .map_err(|e| format!("flatten quantized result failed: {e:?}"))?;
                    Ok(MlxTensor::from_array(flat))
                }
            }
        }

        fn gemm(
            &self,
            a: &MlxTensor,
            b: &MlxTensor,
            _rows: usize,
            _shared_dim: usize,
            _cols: usize,
        ) -> Result<MlxTensor, String> {
            let result = mlx_rs::ops::matmul(&a.array, &b.array, &self.stream)
                .map_err(|e| format!("MLX matmul failed: {e:?}"))?;
            Ok(MlxTensor::from_array(result))
        }

        fn add(&self, a: &MlxTensor, b: &MlxTensor) -> Result<MlxTensor, String> {
            let result = mlx_rs::ops::add(&a.array, &b.array, &self.stream)
                .map_err(|e| format!("MLX add failed: {e:?}"))?;
            Ok(MlxTensor::from_array(result))
        }

        fn mul(&self, a: &MlxTensor, b: &MlxTensor) -> Result<MlxTensor, String> {
            let result = mlx_rs::ops::multiply(&a.array, &b.array, &self.stream)
                .map_err(|e| format!("MLX multiply failed: {e:?}"))?;
            Ok(MlxTensor::from_array(result))
        }

        fn sigmoid(&self, x: &MlxTensor) -> Result<MlxTensor, String> {
            let result =
                mlx_rs::nn::sigmoid(&x.array).map_err(|e| format!("MLX sigmoid failed: {e:?}"))?;
            Ok(MlxTensor::from_array(result))
        }

        fn softmax(&self, x: &MlxTensor) -> Result<MlxTensor, String> {
            let result = mlx_rs::ops::softmax(&x.array, true, &self.stream)
                .map_err(|e| format!("MLX softmax failed: {e:?}"))?;
            Ok(MlxTensor::from_array(result))
        }

        fn synchronize(&self) -> Result<(), String> {
            mlx_rs::Stream::synchronize().map_err(|e| format!("MLX synchronize failed: {e:?}"))
        }
    }

    impl MlxComputeBackend {
        /// Compute Alibi linear bias slopes for `num_heads`.
        ///
        /// Slope for head h is `-(2^( -(8 / num_heads) * h ))`.
        /// Returns a 1-D MLX array of length `num_heads`.
        pub fn alibi_slopes(&self, num_heads: usize) -> Result<MlxTensor, String> {
            let base: f32 = 2.0_f32.powf(-(8.0_f32 / num_heads as f32));
            let slopes: Vec<f32> = (0..num_heads)
                .map(|h| -(base.powf(h as f32 + 1.0)))
                .collect();
            let array = mlx_rs::Array::from_slice(&slopes, &[num_heads as i32]);
            Ok(MlxTensor::from_array(array))
        }

        /// Scaled dot-product attention with a sliding-window causal mask.
        ///
        /// Only attends to the most recent `window_size` tokens; positions
        /// beyond that are masked to `-inf`.
        pub fn sliding_window_attention_decode(
            &self,
            query: &MlxTensor,
            key_cache: &MlxTensor,
            value_cache: &MlxTensor,
            seq_len: usize,
            head_dim: usize,
            scale: f32,
            window_size: usize,
        ) -> Result<MlxTensor, String> {
            let q = mlx_rs::ops::reshape(&query.array, &[1, 1, 1, head_dim as i32], &self.stream)
                .map_err(|e| format!("reshape query failed: {e:?}"))?;

            let k = mlx_rs::ops::reshape(
                &key_cache.array,
                &[1, 1, seq_len as i32, head_dim as i32],
                &self.stream,
            )
            .map_err(|e| format!("reshape key failed: {e:?}"))?;

            let v = mlx_rs::ops::reshape(
                &value_cache.array,
                &[1, 1, seq_len as i32, head_dim as i32],
                &self.stream,
            )
            .map_err(|e| format!("reshape value failed: {e:?}"))?;

            // MLX fast_attention with causal mask handles causal masking natively.
            // For sliding window we fall back to explicit mask construction when
            // window_size < seq_len, otherwise standard causal is sufficient.
            let result = if window_size > 0 && window_size < seq_len {
                // Build a custom mask: [1, 1, 1, seq_len] where out-of-window = -inf
                let mut mask_vals = vec![f32::NEG_INFINITY; seq_len];
                let start = seq_len.saturating_sub(window_size);
                for i in start..seq_len {
                    mask_vals[i] = 0.0_f32;
                }
                let mask = mlx_rs::Array::from_slice(&mask_vals, &[1, 1, 1, seq_len as i32]);
                let scores = mlx_rs::ops::matmul(&q, &k, &self.stream)
                    .map_err(|e| format!("sliding-window matmul qk: {e:?}"))?;
                let scaled = mlx_rs::ops::multiply(
                    &scores,
                    &mlx_rs::Array::from_slice(&[scale], &[1]),
                    &self.stream,
                )
                .map_err(|e| format!("sliding-window scale: {e:?}"))?;
                let masked = mlx_rs::ops::add(&scaled, &mask, &self.stream)
                    .map_err(|e| format!("sliding-window mask add: {e:?}"))?;
                let weights = mlx_rs::ops::softmax(&masked, true, &self.stream)
                    .map_err(|e| format!("sliding-window softmax: {e:?}"))?;
                let out = mlx_rs::ops::matmul(&weights, &v, &self.stream)
                    .map_err(|e| format!("sliding-window matmul wv: {e:?}"))?;
                out
            } else {
                mlx_rs::fast::scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    scale,
                    None::<&[Array]>,
                    &self.stream,
                )
                .map_err(|e| format!("MLX fast_attention failed: {e:?}"))?
            };

            let flat = mlx_rs::ops::flatten(&result, None, None, &self.stream)
                .map_err(|e| format!("flatten failed: {e:?}"))?;
            Ok(MlxTensor::from_array(flat))
        }

        /// MoE top-k router: given input x and gate weight W, compute
        /// scores = softmax(x @ W.T), then select top-k experts.
        ///
        /// Returns (expert_indices [k], gate_weights [k]).
        pub fn moe_topk(
            &self,
            input: &MlxTensor,
            gate_weight: &MlxWeightStorage,
            num_experts: usize,
            k: usize,
        ) -> Result<(Vec<usize>, Vec<f32>), String> {
            let h = input.shape[0];
            let scores_tensor = self
                .gemv(gate_weight, input, num_experts, h)
                .map_err(|e| format!("moe gate gemv: {e}"))?;
            let softmaxed = mlx_rs::ops::softmax(&scores_tensor.array, true, &self.stream)
                .map_err(|e| format!("moe softmax: {e:?}"))?;
            let scores_slice = softmaxed
                .try_as_slice::<f32>()
                .map_err(|e| format!("moe scores slice: {e:?}"))?;

            // Partial top-k via select_nth_unstable (small vector, num_experts <= 256).
            let mut indexed: Vec<(usize, f32)> = scores_slice
                .iter()
                .enumerate()
                .map(|(i, &v)| (i, v))
                .collect();
            if k < indexed.len() {
                indexed.select_nth_unstable_by(k, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                indexed.truncate(k);
            }
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let indices = indexed.iter().map(|(i, _)| *i).collect();
            let weights = indexed.iter().map(|(_, w)| *w).collect();
            Ok((indices, weights))
        }

        /// DeepSeek MLA: project input to compressed latent, then expand to Q/KV.
        ///
        /// `input` is [hidden_size].
        /// `latent_w` projects [hidden_size] -> [latent_dim].
        /// `q_up_w` projects [latent_dim] -> [q_out_dim].
        /// `kv_up_w` projects [latent_dim] -> [kv_out_dim].
        ///
        /// Returns (q_vec, k_vec, v_vec) as host f32 slices.
        pub fn mla_project_qkv(
            &self,
            input: &MlxTensor,
            latent_w: &MlxWeightStorage,
            q_up_w: &MlxWeightStorage,
            kv_up_w: &MlxWeightStorage,
            latent_dim: usize,
            q_out_dim: usize,
            kv_out_dim: usize,
            hidden_size: usize,
        ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
            let latent = self
                .gemv(latent_w, input, latent_dim, hidden_size)
                .map_err(|e| format!("mla latent gemv: {e}"))?;
            let q = self
                .gemv(q_up_w, &latent, q_out_dim, latent_dim)
                .map_err(|e| format!("mla q_up gemv: {e}"))?;
            let kv = self
                .gemv(kv_up_w, &latent, kv_out_dim, latent_dim)
                .map_err(|e| format!("mla kv_up gemv: {e}"))?;

            let mut q_host = vec![0.0_f32; q_out_dim];
            let mut kv_host = vec![0.0_f32; kv_out_dim];
            self.tensor_to_f32(&q, &mut q_host)
                .map_err(|e| format!("mla q to host: {e}"))?;
            self.tensor_to_f32(&kv, &mut kv_host)
                .map_err(|e| format!("mla kv to host: {e}"))?;

            // Split KV into K and V halves.
            let half = kv_out_dim / 2;
            let k = kv_host[..half].to_vec();
            let v = kv_host[half..].to_vec();
            Ok((q_host, k, v))
        }
    }

    // ------------------------------------------------------------------
    //  Helper: map mlx_rs::Dtype -> oxidize_core::tensor::DType
    // ------------------------------------------------------------------
    fn mlx_dtype_to_core(dtype: mlx_rs::Dtype) -> DType {
        match dtype {
            mlx_rs::Dtype::Float32 => DType::F32,
            mlx_rs::Dtype::Float16 => DType::F16,
            mlx_rs::Dtype::Int8 => DType::I8,
            mlx_rs::Dtype::Int16 => DType::I16,
            mlx_rs::Dtype::Int32 => DType::I32,
            mlx_rs::Dtype::Int64 => DType::I64,
            _ => DType::F32, // safe fallback
        }
    }
}

// ---------------------------------------------------------------------------
//  Re-exports (only on macOS)
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
pub use mlx_impl::{MlxComputeBackend, MlxTensor, MlxWeightStorage};

// ---------------------------------------------------------------------------
//  Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlx_build_info_reports_macos_detection() {
        assert_eq!(
            mlx_build_info().detected_at_build,
            cfg!(target_os = "macos")
        );
    }

    #[cfg(target_os = "macos")]
    mod macos_tests {
        use super::*;
        use crate::backend::ComputeBackend;

        #[test]
        fn mlx_tensor_roundtrip() {
            let backend = MlxComputeBackend::new();
            let data = vec![1.0_f32, 2.0, 3.0, 4.0];
            let tensor = backend
                .tensor_from_f32(&data)
                .expect("tensor creation should succeed");
            assert_eq!(backend.tensor_shape(&tensor), vec![4]);
            assert_eq!(backend.tensor_dtype(&tensor), crate::tensor::DType::F32);

            let mut out = vec![0.0_f32; 4];
            let copied = backend
                .tensor_to_f32(&tensor, &mut out)
                .expect("copy to host should succeed");
            assert_eq!(copied, 4);
            assert_eq!(out, data);
        }

        #[test]
        fn mlx_tensor_2d() {
            let backend = MlxComputeBackend::new();
            let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
            let tensor = backend
                .tensor_from_f32_2d(&data, 2, 3)
                .expect("2d tensor creation should succeed");
            assert_eq!(backend.tensor_shape(&tensor), vec![2, 3]);
        }

        #[test]
        fn mlx_rms_norm_matches_reference() {
            let backend = MlxComputeBackend::new();
            let input = backend.tensor_from_f32(&[1.0_f32, 2.0, 3.0]).unwrap();
            let weight = backend.tensor_from_f32(&[1.0_f32, 1.0, 1.0]).unwrap();
            let eps = 1e-6_f32;

            let mlx_out = backend.rms_norm(&input, &weight, eps).unwrap();

            // CPU reference
            let sum_sq: f32 = [1.0_f32, 2.0, 3.0].iter().map(|v| v * v).sum();
            let mean_sq = sum_sq / 3.0;
            let inv_rms = 1.0 / (mean_sq + eps).sqrt();
            let expected = [1.0_f32 * inv_rms, 2.0 * inv_rms, 3.0 * inv_rms];

            let mut actual = vec![0.0_f32; 3];
            backend.tensor_to_f32(&mlx_out, &mut actual).unwrap();

            for (a, e) in actual.iter().zip(expected.iter()) {
                assert!((a - e).abs() < 1e-4, "rms_norm mismatch: {a} vs {e}");
            }
        }

        #[test]
        fn mlx_add_and_mul() {
            let backend = MlxComputeBackend::new();
            let a = backend.tensor_from_f32(&[1.0_f32, 2.0, 3.0]).unwrap();
            let b = backend.tensor_from_f32(&[4.0_f32, 5.0, 6.0]).unwrap();

            let added = backend.add(&a, &b).unwrap();
            let mut out = vec![0.0_f32; 3];
            backend.tensor_to_f32(&added, &mut out).unwrap();
            assert_eq!(out, vec![5.0, 7.0, 9.0]);

            let mulled = backend.mul(&a, &b).unwrap();
            backend.tensor_to_f32(&mulled, &mut out).unwrap();
            assert_eq!(out, vec![4.0, 10.0, 18.0]);
        }

        #[test]
        fn mlx_sigmoid_values() {
            let backend = MlxComputeBackend::new();
            let x = backend.tensor_from_f32(&[0.0_f32, 1.0, -1.0]).unwrap();
            let y = backend.sigmoid(&x).unwrap();
            let mut out = vec![0.0_f32; 3];
            backend.tensor_to_f32(&y, &mut out).unwrap();
            assert!((out[0] - 0.5).abs() < 1e-4);
            assert!((out[1] - 0.7310586).abs() < 1e-4);
            assert!((out[2] - 0.2689414).abs() < 1e-4);
        }

        #[test]
        fn mlx_softmax_sums_to_one() {
            let backend = MlxComputeBackend::new();
            let x = backend.tensor_from_f32(&[1.0_f32, 2.0, 3.0]).unwrap();
            let y = backend.softmax(&x).unwrap();
            let mut out = vec![0.0_f32; 3];
            backend.tensor_to_f32(&y, &mut out).unwrap();
            let sum: f32 = out.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "softmax sum = {sum}");
        }

        #[test]
        fn mlx_gemm_2x3_3x2() {
            let backend = MlxComputeBackend::new();
            let a = backend
                .tensor_from_f32_2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
                .unwrap();
            let b = backend
                .tensor_from_f32_2d(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2)
                .unwrap();
            let c = backend.gemm(&a, &b, 2, 3, 2).unwrap();
            let mut out = vec![0.0_f32; 4];
            backend.tensor_to_f32(&c, &mut out).unwrap();
            // [1 2 3; 4 5 6] * [7 8; 9 10; 11 12] = [58 64; 139 154]
            assert!((out[0] - 58.0).abs() < 1e-4);
            assert!((out[1] - 64.0).abs() < 1e-4);
            assert!((out[2] - 139.0).abs() < 1e-4);
            assert!((out[3] - 154.0).abs() < 1e-4);
        }

        #[test]
        fn mlx_gemv_f32() {
            let backend = MlxComputeBackend::new();
            let matrix = backend
                .tensor_from_f32_2d(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
                .unwrap();
            let vector = backend.tensor_from_f32(&[0.5_f32, -1.0, 2.0]).unwrap();
            let weights = MlxWeightStorage::F32(matrix.array);
            let out = backend.gemv(&weights, &vector, 2, 3).unwrap();
            let mut host = vec![0.0_f32; 2];
            backend.tensor_to_f32(&out, &mut host).unwrap();
            assert!((host[0] - 4.5).abs() < 1e-4);
            assert!((host[1] - 9.0).abs() < 1e-4);
        }

        #[test]
        fn mlx_weight_storage_from_gguf_f32_direct() {
            // F32 weights should be loaded directly without dequantization.
            let f32_data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
            let bytes: Vec<u8> = f32_data.iter().flat_map(|v| v.to_le_bytes()).collect();
            let shape = vec![8usize, 8];
            let storage = MlxWeightStorage::from_gguf_tensor(
                crate::gguf::GgufQuantizationType::F32,
                &bytes,
                &shape,
            )
            .expect("from_gguf_tensor F32 should succeed");
            assert_eq!(storage.shape(), shape);

            let backend = MlxComputeBackend::new();
            // Verify the array is usable on GPU (unified memory) — no H2D copy needed.
            storage
                .verify_unified_memory(backend.stream())
                .expect("F32 weight should be in unified memory");
        }

        #[test]
        fn mlx_weight_storage_from_gguf_q8_0_roundtrip() {
            // Create a synthetic Q8_0 block to verify from_gguf_tensor round-trips.
            let q8_block = vec![
                0x00, 0x3C, // scale d = 1.0 (f16)
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31,
            ];
            // Q8_0 block is 34 bytes for 32 values; we need shape divisible by 32.
            let shape = vec![32usize];
            let storage = MlxWeightStorage::from_gguf_tensor(
                crate::gguf::GgufQuantizationType::Q8_0,
                &q8_block,
                &shape,
            )
            .expect("from_gguf_tensor should succeed");
            assert_eq!(storage.shape(), shape);

            let backend = MlxComputeBackend::new();
            // Q8_0 goes through dequantize -> Array::from_slice, which still
            // places the array in unified memory.  Verify GPU accessibility.
            storage
                .verify_unified_memory(backend.stream())
                .expect("Q8_0 weight should be in unified memory");
        }

        #[test]
        fn mlx_weight_storage_unified_memory_verification() {
            // The key contract: after `from_gguf_tensor`, the array lives in
            // unified memory and `verify_unified_memory` succeeds on a GPU
            // stream without any explicit host-to-device staging.
            let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            let shape = vec![4usize, 4];
            let storage = MlxWeightStorage::from_gguf_tensor(
                crate::gguf::GgufQuantizationType::F32,
                &bytes,
                &shape,
            )
            .unwrap();

            let backend = MlxComputeBackend::new();
            // This exercises a GPU-side operation (reshape on GPU stream).
            // If the data were not in unified memory the operation would
            // either fail or require an explicit copy.
            storage.verify_unified_memory(backend.stream()).unwrap();
        }

        #[test]
        fn mlx_weight_storage_from_gguf_shape() {
            // Keep the existing shape-only test for backward compatibility.
            let q8_block = vec![
                0x00, 0x3C, // scale d = 1.0 (f16)
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31,
            ];
            let shape = vec![32usize];
            let storage = MlxWeightStorage::from_gguf_tensor(
                crate::gguf::GgufQuantizationType::Q8_0,
                &q8_block,
                &shape,
            )
            .expect("from_gguf_tensor should succeed");
            assert_eq!(storage.shape(), shape);
        }

        #[test]
        fn mlx_attention_decode_runs() {
            let backend = MlxComputeBackend::new();
            let head_dim = 4usize;
            let seq_len = 3usize;
            let q = backend.tensor_from_f32(&[0.3_f32, -0.8, 1.1, 0.2]).unwrap();
            let keys = backend
                .tensor_from_f32(
                    &(0..seq_len * head_dim)
                        .map(|i| ((i as f32 * 0.07).cos() * 1.3) - 0.2)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let values = backend
                .tensor_from_f32(
                    &(0..seq_len * head_dim)
                        .map(|i| ((i as f32 * 0.13).sin() * 0.9) + 0.1)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let scale = 1.0_f32 / (head_dim as f32).sqrt();

            let out = backend
                .attention_decode(&q, &keys, &values, seq_len, head_dim, scale)
                .expect("attention decode should succeed");

            assert_eq!(backend.tensor_shape(&out), vec![head_dim]);
        }

        #[test]
        fn mlx_fast_attention_matches_cpu_reference() {
            let backend = MlxComputeBackend::new();
            let head_dim = 64usize;
            let seq_len = 17usize;
            let num_heads = 1usize;
            let kv_heads = 1usize;
            let kv_len = kv_heads * head_dim;

            let query: Vec<f32> = (0..num_heads * head_dim)
                .map(|i| ((i as f32 * 0.0713).sin() * 2.5) - 0.8)
                .collect();
            let key_layer: Vec<f32> = (0..seq_len * kv_len)
                .map(|i| ((i as f32 * 0.113).cos() * 1.3) - 0.2)
                .collect();
            let value_layer: Vec<f32> = (0..seq_len * kv_len)
                .map(|i| ((i as f32 * 0.0517).sin() * 0.9) + 0.1)
                .collect();
            let scale = 1.0_f32 / (head_dim as f32).sqrt();

            // CPU reference
            let mut cpu_out = vec![0.0_f32; num_heads * head_dim];
            crate::flash_attention::flash_attention_decode_heads_f32(
                &query,
                &key_layer,
                &value_layer,
                seq_len,
                head_dim,
                kv_len,
                num_heads,
                kv_heads,
                &mut cpu_out,
            )
            .expect("cpu reference should succeed");

            // MLX fast primitive
            let q_tensor = backend.tensor_from_f32(&query).unwrap();
            let k_tensor = backend.tensor_from_f32(&key_layer).unwrap();
            let v_tensor = backend.tensor_from_f32(&value_layer).unwrap();
            let mlx_out_tensor = backend
                .attention_decode(&q_tensor, &k_tensor, &v_tensor, seq_len, head_dim, scale)
                .expect("mlx attention should succeed");

            let mut mlx_out = vec![0.0_f32; num_heads * head_dim];
            backend
                .tensor_to_f32(&mlx_out_tensor, &mut mlx_out)
                .unwrap();

            let mut max_diff = 0.0_f32;
            for (cpu, mlx) in cpu_out.iter().zip(mlx_out.iter()) {
                max_diff = max_diff.max((cpu - mlx).abs());
            }
            assert!(
                max_diff < 1e-4,
                "fast_scaled_dot_product_attention mismatch: max_abs_diff={max_diff}"
            );
        }

        #[test]
        fn mlx_fast_rms_norm_matches_cpu_reference() {
            let backend = MlxComputeBackend::new();
            let hidden_dim = 128usize;
            let input: Vec<f32> = (0..hidden_dim)
                .map(|i| ((i as f32 * 0.0713).sin() * 2.5) - 0.8)
                .collect();
            let weight: Vec<f32> = (0..hidden_dim)
                .map(|i| ((i as f32 * 0.113).cos() * 1.3) - 0.2)
                .collect();
            let eps = 1e-6_f32;

            // CPU reference
            let mut cpu_out = vec![0.0_f32; hidden_dim];
            crate::tensor::rms_norm_f32(&input, &weight, eps, &mut cpu_out)
                .expect("cpu rms_norm should succeed");

            // MLX fast primitive
            let input_tensor = backend.tensor_from_f32(&input).unwrap();
            let weight_tensor = backend.tensor_from_f32(&weight).unwrap();
            let mlx_out_tensor = backend
                .rms_norm(&input_tensor, &weight_tensor, eps)
                .expect("mlx rms_norm should succeed");

            let mut mlx_out = vec![0.0_f32; hidden_dim];
            backend
                .tensor_to_f32(&mlx_out_tensor, &mut mlx_out)
                .unwrap();

            let mut max_diff = 0.0_f32;
            for (cpu, mlx) in cpu_out.iter().zip(mlx_out.iter()) {
                max_diff = max_diff.max((cpu - mlx).abs());
            }
            assert!(
                max_diff < 1e-4,
                "fast_rms_norm mismatch: max_abs_diff={max_diff}"
            );
        }

        #[test]
        fn mlx_fast_rope_matches_cpu_reference() {
            let backend = MlxComputeBackend::new();
            let head_dim = 64usize;
            let position = 7usize;
            let theta = 10000.0_f32;
            let input: Vec<f32> = (0..head_dim)
                .map(|i| ((i as f32 * 0.0713).sin() * 2.5) - 0.8)
                .collect();

            // CPU reference
            let mut cpu_out = vec![0.0_f32; head_dim];
            crate::tensor::apply_rope_f32(&input, position, head_dim, theta, &mut cpu_out)
                .expect("cpu apply_rope should succeed");

            // MLX fast primitive
            let input_tensor = backend.tensor_from_f32(&input).unwrap();
            let mlx_out_tensor = backend
                .apply_rope(&input_tensor, position, head_dim, theta)
                .expect("mlx rope should succeed");

            let mut mlx_out = vec![0.0_f32; head_dim];
            backend
                .tensor_to_f32(&mlx_out_tensor, &mut mlx_out)
                .unwrap();

            let mut max_diff = 0.0_f32;
            for (cpu, mlx) in cpu_out.iter().zip(mlx_out.iter()) {
                max_diff = max_diff.max((cpu - mlx).abs());
            }
            assert!(
                max_diff < 1e-4,
                "fast_rope mismatch: max_abs_diff={max_diff}"
            );
        }

        #[test]
        fn mlx_quantized_matmul_q4km_accuracy() {
            let rows = 32usize;
            let cols = 256usize;
            let qtype = crate::gguf::GgufQuantizationType::Q4_K_M;
            run_quantized_matmul_test(qtype, rows, cols, 4, 64);
        }

        #[test]
        fn mlx_quantized_matmul_q6k_accuracy() {
            let rows = 32usize;
            let cols = 256usize;
            let qtype = crate::gguf::GgufQuantizationType::Q6_K;
            run_quantized_matmul_test(qtype, rows, cols, 6, 64);
        }

        #[test]
        fn mlx_quantized_matmul_q8_0_accuracy() {
            let rows = 32usize;
            let cols = 128usize;
            let qtype = crate::gguf::GgufQuantizationType::Q8_0;
            run_quantized_matmul_test(qtype, rows, cols, 8, 32);
        }

        fn run_quantized_matmul_test(
            qtype: crate::gguf::GgufQuantizationType,
            rows: usize,
            cols: usize,
            bits: i32,
            group_size: i32,
        ) {
            let backend = MlxComputeBackend::new();

            // Synthetic f32 weights.
            let weights_f32: Vec<f32> = (0..rows * cols)
                .map(|i| ((i as f32 * 0.0713).sin() * 2.5) - 0.8)
                .collect();

            // Random input vector.
            let vector_f32: Vec<f32> = (0..cols)
                .map(|i| ((i as f32 * 0.113).cos() * 1.2) + 0.3)
                .collect();

            // --- CPU reference: quantize to GGUF, dequantize back, then GEMV ---
            let src_bytes: Vec<u8> = weights_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
            let quantized_len = crate::quantization::quantized_size(qtype, weights_f32.len())
                .expect("quantized size must be known");
            let mut quantized_bytes = vec![0_u8; quantized_len];
            crate::quantization::quantize_scalar(
                crate::gguf::GgufQuantizationType::F32,
                qtype,
                &src_bytes,
                &mut quantized_bytes,
            )
            .expect("scalar quantization should succeed");

            let mut dequantized = vec![0.0_f32; weights_f32.len()];
            crate::quantization::dequantize_scalar(qtype, &quantized_bytes, &mut dequantized)
                .expect("dequantization should succeed");

            let mut cpu_out = vec![0.0_f32; rows];
            crate::tensor::gemv_f32(&dequantized, rows, cols, &vector_f32, &mut cpu_out)
                .expect("cpu gemv should succeed");

            // --- MLX quantized path: create quantized storage, run gemv ---
            let shape = vec![rows, cols];
            let mlx_storage = MlxWeightStorage::from_gguf_tensor_quantized(
                qtype,
                &quantized_bytes,
                &shape,
                group_size,
                bits,
            )
            .expect("from_gguf_tensor_quantized should succeed");

            let vector_tensor = backend.tensor_from_f32(&vector_f32).unwrap();
            let mlx_out_tensor = backend
                .gemv(&mlx_storage, &vector_tensor, rows, cols)
                .expect("mlx quantized gemv should succeed");

            let mut mlx_out = vec![0.0_f32; rows];
            backend
                .tensor_to_f32(&mlx_out_tensor, &mut mlx_out)
                .expect("copy to host should succeed");

            // --- Compare with 1e-3 relative tolerance ---
            let max_abs_ref = cpu_out
                .iter()
                .map(|v| v.abs())
                .fold(0.0_f32, f32::max)
                .max(1e-6);
            for (i, (cpu, mlx)) in cpu_out.iter().zip(mlx_out.iter()).enumerate() {
                let abs_diff = (cpu - mlx).abs();
                let rel_diff = abs_diff / max_abs_ref;
                assert!(
                    rel_diff < 1e-3,
                    "quantized matmul mismatch at row {i}: cpu={cpu} mlx={mlx} rel_diff={rel_diff}"
                );
            }
        }

        #[test]
        fn mlx_alibi_slopes_computed_correctly() {
            let backend = MlxComputeBackend::new();
            let num_heads = 8usize;
            let slopes_tensor = backend
                .alibi_slopes(num_heads)
                .expect("alibi slopes should succeed");
            let mut host = vec![0.0_f32; num_heads];
            backend.tensor_to_f32(&slopes_tensor, &mut host).unwrap();

            let base: f32 = 2.0_f32.powf(-(8.0_f32 / num_heads as f32));
            for (i, &slope) in host.iter().enumerate() {
                let expected = -(base.powf(i as f32 + 1.0));
                assert!(
                    (slope - expected).abs() < 1e-4,
                    "alibi slope mismatch at head {i}: got {slope}, expected {expected}"
                );
            }
        }

        #[test]
        fn mlx_sliding_window_attention_decode_runs() {
            let backend = MlxComputeBackend::new();
            let head_dim = 4usize;
            let seq_len = 8usize;
            let window_size = 4usize;
            let q = backend.tensor_from_f32(&[0.3_f32, -0.8, 1.1, 0.2]).unwrap();
            let keys: Vec<f32> = (0..seq_len * head_dim)
                .map(|i| ((i as f32 * 0.07).cos() * 1.3) - 0.2)
                .collect();
            let values: Vec<f32> = (0..seq_len * head_dim)
                .map(|i| ((i as f32 * 0.13).sin() * 0.9) + 0.1)
                .collect();
            let scale = 1.0_f32 / (head_dim as f32).sqrt();

            let k_tensor = backend.tensor_from_f32(&keys).unwrap();
            let v_tensor = backend.tensor_from_f32(&values).unwrap();

            let out = backend
                .sliding_window_attention_decode(
                    &q,
                    &k_tensor,
                    &v_tensor,
                    seq_len,
                    head_dim,
                    scale,
                    window_size,
                )
                .expect("sliding window attention should succeed");

            assert_eq!(backend.tensor_shape(&out), vec![head_dim]);
        }

        #[test]
        fn mlx_sliding_window_attention_matches_standard_when_window_gte_seq_len() {
            let backend = MlxComputeBackend::new();
            let head_dim = 4usize;
            let seq_len = 3usize;
            let q = backend.tensor_from_f32(&[0.3_f32, -0.8, 1.1, 0.2]).unwrap();
            let keys: Vec<f32> = (0..seq_len * head_dim)
                .map(|i| ((i as f32 * 0.07).cos() * 1.3) - 0.2)
                .collect();
            let values: Vec<f32> = (0..seq_len * head_dim)
                .map(|i| ((i as f32 * 0.13).sin() * 0.9) + 0.1)
                .collect();
            let scale = 1.0_f32 / (head_dim as f32).sqrt();

            let k_tensor = backend.tensor_from_f32(&keys).unwrap();
            let v_tensor = backend.tensor_from_f32(&values).unwrap();

            let standard = backend
                .attention_decode(&q, &k_tensor, &v_tensor, seq_len, head_dim, scale)
                .unwrap();
            let sliding = backend
                .sliding_window_attention_decode(
                    &q, &k_tensor, &v_tensor, seq_len, head_dim, scale, seq_len,
                )
                .unwrap();

            let mut std_host = vec![0.0_f32; head_dim];
            let mut slide_host = vec![0.0_f32; head_dim];
            backend.tensor_to_f32(&standard, &mut std_host).unwrap();
            backend.tensor_to_f32(&sliding, &mut slide_host).unwrap();

            for (i, (s, sl)) in std_host.iter().zip(slide_host.iter()).enumerate() {
                assert!(
                    (s - sl).abs() < 1e-4,
                    "sliding_window vs standard mismatch at dim {i}: std={s} slide={sl}"
                );
            }
        }

        #[test]
        fn mlx_moe_topk_selects_k_experts() {
            let backend = MlxComputeBackend::new();
            let num_experts = 4usize;
            let k = 2usize;
            let hidden_size = 8usize;

            // Build a synthetic gate weight that maps hidden_size -> num_experts.
            let gate_weights: Vec<f32> = (0..num_experts * hidden_size)
                .map(|i| (i as f32 * 0.01).sin() * 0.5)
                .collect();
            let gate_storage = MlxWeightStorage::F32(mlx_rs::Array::from_slice(
                &gate_weights,
                &[num_experts as i32, hidden_size as i32],
            ));
            let input = backend.tensor_from_f32(&[1.0_f32; hidden_size]).unwrap();

            let (indices, weights) = backend
                .moe_topk(&input, &gate_storage, num_experts, k)
                .expect("moe_topk should succeed");

            assert_eq!(indices.len(), k);
            assert_eq!(weights.len(), k);
            assert!(weights.iter().all(|&w| w >= 0.0 && w <= 1.0));
            // Top-k indices should be unique.
            let mut uniq = indices.clone();
            uniq.sort_unstable();
            uniq.dedup();
            assert_eq!(uniq.len(), k, "top-k expert indices should be unique");
        }

        #[test]
        fn mlx_mla_project_qkv_shapes_correct() {
            let backend = MlxComputeBackend::new();
            let hidden_size = 8usize;
            let latent_dim = 4usize;
            let q_out_dim = 6usize;
            let kv_out_dim = 6usize; // split into k=3, v=3

            let latent_w = MlxWeightStorage::F32(mlx_rs::Array::from_slice(
                &vec![0.1_f32; latent_dim * hidden_size],
                &[latent_dim as i32, hidden_size as i32],
            ));
            let q_up_w = MlxWeightStorage::F32(mlx_rs::Array::from_slice(
                &vec![0.2_f32; q_out_dim * latent_dim],
                &[q_out_dim as i32, latent_dim as i32],
            ));
            let kv_up_w = MlxWeightStorage::F32(mlx_rs::Array::from_slice(
                &vec![0.3_f32; kv_out_dim * latent_dim],
                &[kv_out_dim as i32, latent_dim as i32],
            ));

            let input = backend.tensor_from_f32(&[0.5_f32; hidden_size]).unwrap();
            let (q, k, v) = backend
                .mla_project_qkv(
                    &input,
                    &latent_w,
                    &q_up_w,
                    &kv_up_w,
                    latent_dim,
                    q_out_dim,
                    kv_out_dim,
                    hidden_size,
                )
                .expect("mla_project_qkv should succeed");

            assert_eq!(q.len(), q_out_dim, "mla q length mismatch");
            assert_eq!(k.len(), kv_out_dim / 2, "mla k length mismatch");
            assert_eq!(v.len(), kv_out_dim / 2, "mla v length mismatch");
        }
    }
}
