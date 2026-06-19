use super::*;

#[test]
fn allocates_cpu_buffer_with_requested_size() {
    let buffer = DeviceBuffer::allocate(MemoryDevice::Cpu, 32).expect("buffer allocates");
    assert_eq!(buffer.device(), MemoryDevice::Cpu);
    assert_eq!(buffer.len(), 32);
}

#[test]
fn supports_host_to_device_and_back_for_cpu_buffer() {
    let mut buffer = DeviceBuffer::allocate(MemoryDevice::Cpu, 4).expect("buffer allocates");
    let input = [1_u8, 2, 3, 4];
    buffer
        .copy_from_host(&input)
        .expect("host to device copy succeeds");

    let mut output = [0_u8; 4];
    buffer
        .copy_to_host(&mut output)
        .expect("device to host copy succeeds");
    assert_eq!(output, input);
}

#[test]
fn rejects_mismatched_transfer_lengths() {
    let mut buffer = DeviceBuffer::allocate(MemoryDevice::Cpu, 3).expect("buffer allocates");

    let h2d_error = buffer
        .copy_from_host(&[1_u8, 2])
        .expect_err("h2d mismatch should fail");
    assert_eq!(
        h2d_error,
        MemoryError::SizeMismatch {
            expected: 3,
            actual: 2
        }
    );

    let mut host = [0_u8; 2];
    let d2h_error = buffer
        .copy_to_host(&mut host)
        .expect_err("d2h mismatch should fail");
    assert_eq!(
        d2h_error,
        MemoryError::SizeMismatch {
            expected: 3,
            actual: 2
        }
    );
}

#[test]
fn validates_gemv_cuda_dimensions() {
    let matrix = [1.0_f32, 2.0, 3.0, 4.0];
    let vector = [1.0_f32, 1.0];
    let output = [0.0_f32; 2];
    validate_gemv_dims(&matrix, 2, 2, &vector, &output).expect("dimensions should be valid");
}

#[test]
fn rejects_gemv_cuda_dimension_mismatch() {
    let err = validate_gemv_dims(&[1.0_f32, 2.0, 3.0], 2, 2, &[1.0_f32, 1.0], &[0.0_f32; 2])
        .expect_err("matrix size mismatch should fail");
    assert!(matches!(err, GemvCudaError::InvalidMatrixLength { .. }));
}

#[test]
fn validates_q8_0_gemv_cuda_dimensions() {
    let rows = 2;
    let cols = 32;
    let matrix = vec![0_u8; rows * BLOCK_Q8_0_SIZE];
    let vector = vec![1.0_f32; cols];
    let output = vec![0.0_f32; rows];
    validate_q8_0_gemv_dims(&matrix, rows, cols, &vector, &output)
        .expect("dimensions should be valid");
}

#[test]
fn rejects_q8_0_gemv_cols_not_aligned() {
    let rows = 1;
    let cols = 31;
    let matrix = vec![0_u8; BLOCK_Q8_0_SIZE];
    let vector = vec![1.0_f32; cols];
    let output = vec![0.0_f32; rows];
    let err = validate_q8_0_gemv_dims(&matrix, rows, cols, &vector, &output)
        .expect_err("non-aligned columns should fail");
    assert!(matches!(err, GemvCudaError::InvalidVectorLength { .. }));
}

#[test]
fn validates_gemm_cuda_dimensions() {
    let left = [1.0_f32, 2.0, 3.0, 4.0];
    let right = [1.0_f32, 2.0, 3.0, 4.0];
    let output = [0.0_f32; 4];
    validate_gemm_dims(&left, 2, 2, &right, 2, &output).expect("dimensions should be valid");
}

#[test]
fn rejects_gemm_cuda_dimension_mismatch() {
    let err = validate_gemm_dims(
        &[1.0_f32, 2.0, 3.0],
        2,
        2,
        &[1.0_f32, 2.0, 3.0, 4.0],
        2,
        &[0.0_f32; 4],
    )
    .expect_err("left matrix size mismatch should fail");
    assert!(matches!(err, GemmCudaError::InvalidLeftMatrixLength { .. }));
}

#[test]
#[cfg(feature = "cuda")]
fn gemv_cuda_kernel_name_matches_ptx_entry() {
    assert!(GEMV_F32_PTX.contains(".entry gemv_f32_kernel"));
    assert!(GEMV_F32_PTX.contains(".entry gemv_q4_k_kernel"));
    assert_eq!(GEMV_KERNEL_NAME, "gemv_f32_kernel");
    assert_eq!(GEMV_Q4_K_DIRECT_KERNEL_NAME, "gemv_q4_k_kernel");
}
