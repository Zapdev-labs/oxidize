#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaBuildInfo {
    pub detected_at_build: bool,
    pub cuda_path: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryDevice {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryError {
    SizeMismatch { expected: usize, actual: usize },
    #[cfg(feature = "cuda")]
    Cuda(String),
}

#[cfg(feature = "cuda")]
impl From<cust::error::CudaError> for MemoryError {
    fn from(error: cust::error::CudaError) -> Self {
        Self::Cuda(error.to_string())
    }
}

pub struct DeviceBuffer {
    device: MemoryDevice,
    len: usize,
    host_bytes: Vec<u8>,
    #[cfg(feature = "cuda")]
    cuda_bytes: Option<cust::memory::DeviceBuffer<u8>>,
}

impl DeviceBuffer {
    pub fn allocate(device: MemoryDevice, len: usize) -> Result<Self, MemoryError> {
        let host_bytes = vec![0_u8; len];
        #[cfg(feature = "cuda")]
        let cuda_bytes = match device {
            MemoryDevice::Cpu => None,
            MemoryDevice::Cuda => Some(cust::memory::DeviceBuffer::zeroed(len)?),
        };

        Ok(Self {
            device,
            len,
            host_bytes,
            #[cfg(feature = "cuda")]
            cuda_bytes,
        })
    }

    pub fn device(&self) -> MemoryDevice {
        self.device
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn copy_from_host(&mut self, host: &[u8]) -> Result<(), MemoryError> {
        if host.len() != self.len {
            return Err(MemoryError::SizeMismatch {
                expected: self.len,
                actual: host.len(),
            });
        }

        self.host_bytes.copy_from_slice(host);
        #[cfg(feature = "cuda")]
        if let Some(cuda_buffer) = self.cuda_bytes.as_mut() {
            cuda_buffer.copy_from(host)?;
        }

        Ok(())
    }

    pub fn copy_to_host(&self, host: &mut [u8]) -> Result<(), MemoryError> {
        if host.len() != self.len {
            return Err(MemoryError::SizeMismatch {
                expected: self.len,
                actual: host.len(),
            });
        }

        #[cfg(feature = "cuda")]
        if let Some(cuda_buffer) = self.cuda_bytes.as_ref() {
            cuda_buffer.copy_to(host)?;
            return Ok(());
        }

        host.copy_from_slice(&self.host_bytes);
        Ok(())
    }
}

pub fn cuda_build_info() -> CudaBuildInfo {
    CudaBuildInfo {
        detected_at_build: cfg!(cuda_available),
        cuda_path: option_env!("LLAMAS_CUDA_PATH"),
    }
}

#[cfg(feature = "cuda")]
pub fn initialize_cuda() -> Result<cust::context::Context, cust::error::CudaError> {
    cust::quick_init()
}

#[cfg(test)]
mod tests {
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
}
