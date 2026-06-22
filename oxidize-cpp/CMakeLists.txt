cmake_minimum_required(VERSION 3.24)
project(oxidize_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(OXIDIZE_CUDA "Build the CUDA backend (requires nvcc + cuBLAS)" OFF)

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

# ---- CUDA backend (Modal / H100 / A100) -----------------------------------
if(OXIDIZE_CUDA)
  enable_language(CUDA)
  # H100 (sm_90), A100 (sm_80). Adjust via -DCMAKE_CUDA_ARCHITECTURES.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 90)
  endif()
  file(GLOB CUDA_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cu")
  add_library(oxidize_cuda ${CUDA_SOURCES})
  target_include_directories(oxidize_cuda PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
  set_target_properties(oxidize_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  # The .cu sources (backend.cu) are themselves guarded by OXIDIZE_CUDA.
  target_compile_definitions(oxidize_cuda PUBLIC OXIDIZE_CUDA=1)
  target_compile_options(oxidize_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(oxidize_cuda PUBLIC CUDA::cudart CUDA::cublas CUDA::cublasLt)
  target_compile_definitions(oxidize_core PUBLIC OXIDIZE_CUDA=1)
  target_link_libraries(oxidize_core PUBLIC oxidize_cuda)
endif()

# ---- CLI binary -----------------------------------------------------------
add_executable(oxidize-cpp src/cli/main.cpp)
target_link_libraries(oxidize-cpp PRIVATE oxidize_core)

# ---- tests ----------------------------------------------------------------
enable_testing()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests")
  file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
  foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE oxidize_core)
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()
endif()
