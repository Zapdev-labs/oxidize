#pragma once
// CUDA / ROCm-HIP runtime compatibility shim.
// When built with OXIDIZE_HIP, maps cuda* symbols to hip* so the existing .cu
// sources compile under hipcc without duplicating every kernel file.

#if defined(OXIDIZE_HIP)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
using cudaGraph_t = hipGraph_t;
using cudaGraphExec_t = hipGraphExec_t;

#define cudaSuccess hipSuccess
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpy hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaStreamCaptureModeGlobal hipStreamCaptureModeGlobal
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
using cudaDeviceProp = hipDeviceProp_t;
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraphInstantiate hipGraphInstantiate
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphDestroy hipGraphDestroy
#define cudaMemsetAsync hipMemsetAsync
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError

#elif defined(OXIDIZE_CUDA)

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#else
#error "gpu_common.cuh requires OXIDIZE_CUDA or OXIDIZE_HIP"
#endif
