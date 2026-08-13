cmake_minimum_required(VERSION 3.21)
project(oxidize_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(OXIDIZE_CUDA "Build the CUDA backend (requires nvcc + cuBLAS)" OFF)
option(OXIDIZE_ROCM "Build the ROCm/HIP backend (requires hipcc + hipBLAS)" OFF)
option(OXIDIZE_VULKAN "Build the Vulkan KV-cache backend (any GPU vendor)" OFF)

if(OXIDIZE_CUDA AND OXIDIZE_ROCM)
  message(FATAL_ERROR "OXIDIZE_CUDA and OXIDIZE_ROCM are mutually exclusive")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# ---- core library (CPU; always builds, no nvcc required) -------------------
file(GLOB CORE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(oxidize_core ${CORE_SOURCES})
target_include_directories(oxidize_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

find_package(Threads REQUIRED)
target_link_libraries(oxidize_core PUBLIC Threads::Threads)

# OpenMP for CPU kernels (optional but recommended).
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(oxidize_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# ---- GPU backend (CUDA or ROCm-HIP) ----------------------------------------
if(OXIDIZE_CUDA OR OXIDIZE_ROCM)
  file(GLOB GPU_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cu")
  add_library(oxidize_cuda ${GPU_SOURCES})
  target_include_directories(oxidize_cuda PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
  target_compile_definitions(oxidize_cuda PUBLIC OXIDIZE_GPU=1)
  target_compile_definitions(oxidize_core PUBLIC OXIDIZE_GPU=1)
  target_link_libraries(oxidize_core PUBLIC oxidize_cuda)

  if(OXIDIZE_CUDA)
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 80 90)
    endif()
    set_target_properties(oxidize_cuda PROPERTIES
      CUDA_SEPARABLE_COMPILATION OFF
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CUDA_EXTENSIONS OFF)
    target_compile_definitions(oxidize_cuda PUBLIC OXIDIZE_CUDA=1)
    target_compile_definitions(oxidize_core PUBLIC OXIDIZE_CUDA=1)
    target_compile_options(oxidize_cuda PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(oxidize_cuda PUBLIC CUDA::cudart CUDA::cublas CUDA::cublasLt)
  else()
    enable_language(HIP)
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
      # RDNA3 (7900), RDNA2 (6800 XT), RDNA1 (5700) — override at configure time.
      set(CMAKE_HIP_ARCHITECTURES gfx1100 gfx1030 gfx1012)
    endif()
    set_source_files_properties(${GPU_SOURCES} PROPERTIES LANGUAGE HIP)
    set_target_properties(oxidize_cuda PROPERTIES
      HIP_STANDARD 17
      HIP_STANDARD_REQUIRED ON
      HIP_EXTENSIONS OFF)
    target_compile_definitions(oxidize_cuda PUBLIC OXIDIZE_HIP=1)
    target_compile_definitions(oxidize_core PUBLIC OXIDIZE_HIP=1)

    if(NOT DEFINED ENV{ROCM_PATH} AND NOT EXISTS "/opt/rocm")
      message(WARNING "ROCM_PATH not set and /opt/rocm missing; HIP configure may fail")
    endif()
    if(DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH "$ENV{ROCM_PATH}")
    else()
      set(ROCM_PATH "/opt/rocm")
    endif()
    list(APPEND CMAKE_PREFIX_PATH
      "${ROCM_PATH}" "${ROCM_PATH}/lib/cmake" "${ROCM_PATH}/lib/cmake/hip"
      "${ROCM_PATH}/lib/cmake/hipblas")
    find_package(hip REQUIRED CONFIG)
    find_package(hipblas REQUIRED CONFIG)
    target_link_libraries(oxidize_cuda PUBLIC hip::host hip::device roc::hipblas)
  endif()
endif()

# ---- Vulkan KV-cache backend (vendor-agnostic: NVIDIA/AMD/Intel/iGPU) ------
if(OXIDIZE_VULKAN)
  find_package(Vulkan REQUIRED)
  if(NOT Vulkan_GLSLC_EXECUTABLE)
    message(FATAL_ERROR "OXIDIZE_VULKAN needs glslc (install shaderc / Vulkan SDK)")
  endif()
  set(SHADER_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders_gen")
  file(MAKE_DIRECTORY "${SHADER_GEN_DIR}")
  file(GLOB SHADER_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.comp")
  set(SHADER_HEADERS)
  foreach(shader ${SHADER_SOURCES})
    get_filename_component(shader_name ${shader} NAME_WE)
    set(out "${SHADER_GEN_DIR}/${shader_name}.spv.h")
    add_custom_command(OUTPUT ${out}
      COMMAND ${Vulkan_GLSLC_EXECUTABLE} -O -mfmt=c ${shader} -o ${out}
      DEPENDS ${shader}
      COMMENT "Compiling ${shader_name}.comp to SPIR-V")
    list(APPEND SHADER_HEADERS ${out})
  endforeach()
  add_library(oxidize_vulkan src/vulkan/kv_cache_vk.cpp ${SHADER_HEADERS})
  target_include_directories(oxidize_vulkan
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${SHADER_GEN_DIR}")
  target_compile_definitions(oxidize_vulkan PUBLIC OXIDIZE_VULKAN=1)
  target_link_libraries(oxidize_vulkan PUBLIC Vulkan::Vulkan)
  target_link_libraries(oxidize_core PUBLIC oxidize_vulkan)
endif()

# ---- CLI binary -----------------------------------------------------------
add_executable(oxidize-cpp src/cli/main.cpp)
target_link_libraries(oxidize-cpp PRIVATE oxidize_core)

# ---- OpenAI-compatible HTTP server ----------------------------------------
add_executable(oxidize-cpp-server src/server/main.cpp src/server/json.cpp)
target_link_libraries(oxidize-cpp-server PRIVATE oxidize_core)

# ---- training library (fp32, no CUDA dep) ---------------------------------
file(GLOB TRAIN_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/train/*.cpp")
# Exclude grad_check.cpp (it has its own main()).
list(FILTER TRAIN_SOURCES EXCLUDE REGEX ".*grad_check\\.cpp$")
add_library(oxidize_train ${TRAIN_SOURCES})
target_include_directories(oxidize_train PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(oxidize_train PUBLIC oxidize_core)
if(OpenMP_CXX_FOUND)
  target_link_libraries(oxidize_train PUBLIC OpenMP::OpenMP_CXX)
endif()

# ---- training CLI binary --------------------------------------------------
add_executable(oxidize-cpp-train src/cli/train_main.cpp)
target_link_libraries(oxidize-cpp-train PRIVATE oxidize_train)

# ---- tests ----------------------------------------------------------------
enable_testing()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests")
  file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
  if(NOT OXIDIZE_VULKAN)
    list(FILTER TEST_SOURCES EXCLUDE REGEX ".*kv_cache_vk_test\\.cpp$")
  endif()
  foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE oxidize_core)
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()
endif()

# grad_check test (has its own main, links oxidize_train).
add_executable(grad_check src/train/grad_check.cpp)
target_link_libraries(grad_check PRIVATE oxidize_train)
add_test(NAME grad_check COMMAND grad_check)
