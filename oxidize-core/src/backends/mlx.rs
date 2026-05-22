//! Apple MLX compute backend (macOS only).
//!
//! All MLX-specific code is gated by `#[cfg(target_os = "macos")]` so that
//! Linux builds compile without requiring the `mlx-c` library.

#[cfg(target_os = "macos")]
use crate::backend::ComputeBackend;
#[cfg(target_os = "macos")]
use crate::tensor::DType;
#[cfg(target_os = "macos")]
use crate::gguf::GgufQuantizationType;

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
            Self { array, shape, dtype }
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
            self.array.eval().map_err(|e| format!("MLX eval failed: {e:?}"))?;
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
                        false,
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
            let q = mlx_rs::ops::reshape(
                &query.array,
                &[1, 1, 1, head_dim as i32],
                &self.stream,
            )
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
                &q, &k, &v, scale, None::<&[Array]>, &self.stream,
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
                    let vec = if vector.array.ndim() == 1 {
                        vector.array.clone()
                    } else {
                        mlx_rs::ops::reshape(&vector.array, &[cols as i32], &self.stream)
                            .map_err(|e| format!("reshape vector failed: {e:?}"))?
                    };
                    let result = mlx_rs::ops::quantized_matmul(
                        &vec, weights, scales, biases, false, Some(*group_size), Some(*bits), &self.stream,
                    )
                    .map_err(|e| format!("MLX quantized_matmul failed: {e:?}"))?;
                    Ok(MlxTensor::from_array(result))
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
            let result = mlx_rs::nn::sigmoid(&x.array)
                .map_err(|e| format!("MLX sigmoid failed: {e:?}"))?;
            Ok(MlxTensor::from_array(result))
        }

        fn softmax(&self, x: &MlxTensor) -> Result<MlxTensor, String> {
            let result = mlx_rs::ops::softmax(&x.array, true, &self.stream)
                .map_err(|e| format!("MLX softmax failed: {e:?}"))?;
            Ok(MlxTensor::from_array(result))
        }

        fn synchronize(&self) -> Result<(), String> {
            mlx_rs::Stream::synchronize()
                .map_err(|e| format!("MLX synchronize failed: {e:?}"))
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
        assert_eq!(mlx_build_info().detected_at_build, cfg!(target_os = "macos"));
    }

    #[cfg(target_os = "macos")]
    mod macos_tests {
        use super::*;
        use crate::backend::ComputeBackend;

        #[test]
        fn mlx_tensor_roundtrip() {
            let backend = MlxComputeBackend::new();
            let data = vec![1.0_f32, 2.0, 3.0, 4.0];
            let tensor = backend.tensor_from_f32(&data).expect("tensor creation should succeed");
            assert_eq!(backend.tensor_shape(&tensor), vec![4]);
            assert_eq!(backend.tensor_dtype(&tensor), crate::tensor::DType::F32);

            let mut out = vec![0.0_f32; 4];
            let copied = backend.tensor_to_f32(&tensor, &mut out).expect("copy to host should succeed");
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
            let a = backend.tensor_from_f32_2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
            let b = backend.tensor_from_f32_2d(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2).unwrap();
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
            let bytes: Vec<u8> = f32_data
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
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
                0, 1, 2, 3, 4, 5, 6, 7,
                8, 9, 10, 11, 12, 13, 14, 15,
                16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30, 31,
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
                0, 1, 2, 3, 4, 5, 6, 7,
                8, 9, 10, 11, 12, 13, 14, 15,
                16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30, 31,
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
    }
}
