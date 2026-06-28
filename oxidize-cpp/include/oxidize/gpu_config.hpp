#pragma once
// GPU backend compile-time flags (CUDA and/or ROCm-HIP).

#if defined(OXIDIZE_CUDA) || defined(OXIDIZE_HIP)
#define OXIDIZE_GPU 1
#endif

#if defined(OXIDIZE_HIP)
#define OXIDIZE_GPU_BACKEND_ROCM 1
#elif defined(OXIDIZE_CUDA)
#define OXIDIZE_GPU_BACKEND_CUDA 1
#endif
