# =============================================================================
# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

# CUDF_DIR is set from the environment variable, allowing to specify
# a custom cudf repository other than `thirdparty/cudf`.
set(CUDF_DIR
  "$ENV{CUDF_ROOT}"
  CACHE STRING "path to cudf repository"
)
set(CUDF_INSTALL_DIR
  "$ENV{CUDF_INSTALL_DIR}"
  CACHE STRING "path to libcudf install root"
)

# libcudf's kvikio dependency requires this to be set when statically linking CUDA runtime
set(CUDA_STATIC_RUNTIME ON)

include("${CUDF_DIR}/cmake/rapids_config.cmake")
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)
rapids_cpm_init()

rapids_cuda_init_architectures(SPARK_GPU_EXPR)

project(
  SPARK_GPU_EXPR
  VERSION 0.1.0
  LANGUAGES CXX CUDA
)

# ##################################################################################################
# * build options ---------------------------------------------------------------------------------

option(CUDF_USE_PER_THREAD_DEFAULT_STREAM "Build with per-thread default stream" ON)
option(BUILD_TESTS "Configure CMake to build tests" ON)

message(VERBOSE "SPARK_GPU_EXPR: Build with per-thread default stream: ${CUDF_USE_PER_THREAD_DEFAULT_STREAM}")
message(VERBOSE "SPARK_GPU_EXPR: Configure CMake to build tests: ${BUILD_TESTS}")

set(SPARK_GPU_EXPR_BUILD_TESTS ${BUILD_TESTS})

# Set RMM logging level
set(RMM_LOGGING_LEVEL
    "INFO"
    CACHE STRING "Choose the logging level."
)
set_property(
  CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF"
)

# ##################################################################################################
# * compiler options ------------------------------------------------------------------------------

rapids_find_package(CUDAToolkit REQUIRED)

# Set a default build type if none was specified
rapids_cmake_build_type("Release")

# ##################################################################################################
# * dependencies ----------------------------------------------------------------------------------

# Path to search for the pre-installed packages.
set(CMAKE_PREFIX_PATH ${CUDF_INSTALL_DIR} "${CUDF_INSTALL_DIR}/lib64/cmake")

# cudf
if(BUILD_TESTS)
  include(${rapids-cmake-dir}/cpm/gtest.cmake)
  rapids_cpm_gtest(BUILD_STATIC)
  rapids_find_package(cudf REQUIRED COMPONENTS testing)
else()
  rapids_find_package(cudf REQUIRED)
endif()

# Get the compilation flags and definitions from cudf.
get_target_property(CUDF_CXX_FLAGS cudf::cudf CUDF_CXX_FLAGS)
get_target_property(CUDF_CUDA_FLAGS cudf::cudf CUDF_CUDA_FLAGS)
get_target_property(CUDF_CXX_DEFINITIONS cudf::cudf CUDF_CXX_DEFINITIONS)
get_target_property(CUDF_CUDA_DEFINITIONS cudf::cudf CUDF_CUDA_DEFINITIONS)

# spdlog for logging
if(NOT TARGET spdlog::spdlog_header_only)
  message(STATUS "Fetching spdlog")
  include(${rapids-cmake-dir}/cpm/spdlog.cmake)
  rapids_cpm_spdlog()
endif()

# ##################################################################################################
# * library targets -------------------------------------------------------------------------------

add_library(
  spark_gpu_expr SHARED
  src/init.cpp
  src/logical_type.cpp
  src/round_float.cu
  src/round_expression.cpp
  src/date_format_expression.cpp
  src/format_conversion.cpp
)

set_target_properties(
  spark_gpu_expr
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             # set target compile options
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
             CUDA_RUNTIME_LIBRARY Static
)

target_compile_options(
  spark_gpu_expr PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_FLAGS}>"
                         "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>"
)

# Specify include paths for the current target and dependents
target_include_directories(
  spark_gpu_expr
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CUDAToolkit_INCLUDE_DIRS}"
)

target_compile_definitions(
  spark_gpu_expr PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_DEFINITIONS}>"
                        "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_DEFINITIONS}>>"
)

# Per-thread default stream
if(CUDF_USE_PER_THREAD_DEFAULT_STREAM)
  target_compile_definitions(
    spark_gpu_expr PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM CUDF_USE_PER_THREAD_DEFAULT_STREAM
  )
endif()

target_link_libraries(
  spark_gpu_expr
    PUBLIC cudf::cudf
    PRIVATE spdlog::spdlog_header_only
)
rapids_cuda_set_runtime(spark_gpu_expr USE_STATIC ON)

# ##################################################################################################
# * install rules ---------------------------------------------------------------------------------

install(TARGETS spark_gpu_expr
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/spark_gpu_expr
    DESTINATION include
)

# ##################################################################################################
# * add tests -------------------------------------------------------------------------------------

if(SPARK_GPU_EXPR_BUILD_TESTS)
  # include CTest module -- automatically calls enable_testing()
  include(CTest)
  add_subdirectory(tests)
endif()
