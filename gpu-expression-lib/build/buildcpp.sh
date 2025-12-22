#!/bin/bash
#
# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#
# Script to build native code in cudf and spark-gpu-expression-lib
# (Based on spark-rapids-jni/build/buildcpp.sh)
#

set -e

if [[ $FROM_MAVEN == "true" ]]; then
  echo "Building native libraries. To rerun outside Maven enter the build environment via

$ ./build/run-in-docker

then run

$ REUSE_ENV=true $0
"
fi

# Environment variables to control the build
PROJECT_BASE_DIR=${PROJECT_BASE_DIR:-$(realpath $(dirname $0)/..)}
# Use a separate directory for C++ builds (not the Maven target directory)
PROJECT_BUILD_DIR=${PROJECT_BUILD_DIR:-$PROJECT_BASE_DIR/cpp-target}

# Create build directory if it doesn't exist
mkdir -p "$PROJECT_BUILD_DIR"

if [[ "$REUSE_ENV" != "true" ]]; then
  echo "
BUILD_CUDF_BENCHMARKS=${BUILD_CUDF_BENCHMARKS:-OFF}
BUILD_CUDF_TESTS=${BUILD_CUDF_TESTS:-OFF}
BUILD_TESTS=${BUILD_TESTS:-ON}
CMAKE_EXPORT_COMPILE_COMMANDS=${CMAKE_EXPORT_COMPILE_COMMANDS:-ON}
export CMAKE_GENERATOR=${CMAKE_GENERATOR:-Ninja}
CPP_PARALLEL_LEVEL=${CPP_PARALLEL_LEVEL:-10}
CUDF_BUILD_TYPE=${CUDF_BUILD_TYPE:-Release}
# Use the shared cudf submodule from the parent spark-rapids-jni directory
CUDF_PATH=${CUDF_PATH:-$PROJECT_BASE_DIR/../thirdparty/cudf}
CUDF_PIN_PATH=${CUDF_PIN_PATH:-$PROJECT_BASE_DIR/../thirdparty/cudf-pins}
CUDF_USE_PER_THREAD_DEFAULT_STREAM=${CUDF_USE_PER_THREAD_DEFAULT_STREAM:-ON}
GPU_ARCHS=${GPU_ARCHS:-DEPRECATED}
CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES:-RAPIDS}
LIBCUDF_BUILD_CONFIGURE=${LIBCUDF_BUILD_CONFIGURE:-false}
LIBCUDF_BUILD_PATH=${LIBCUDF_BUILD_PATH:-$PROJECT_BUILD_DIR/libcudf/cmake-build}
LIBCUDF_DEPENDENCY_MODE=${LIBCUDF_DEPENDENCY_MODE:-pinned}
LIBCUDF_INSTALL_PATH=${LIBCUDF_INSTALL_PATH:-$PROJECT_BUILD_DIR/libcudf-install}
NATIVE_BUILD_PATH=${NATIVE_BUILD_PATH:-$PROJECT_BUILD_DIR/native/cmake-build}
RMM_LOGGING_LEVEL=${RMM_LOGGING_LEVEL:-OFF}
LIBCUDF_CONFIGURE_ONLY=${LIBCUDF_CONFIGURE_ONLY:-OFF}" > "$PROJECT_BUILD_DIR/buildcpp-env.sh"
fi

source "$PROJECT_BUILD_DIR/buildcpp-env.sh"

if [[ "$GPU_ARCHS" != "DEPRECATED" ]]; then
    CMAKE_CUDA_ARCHITECTURES="$GPU_ARCHS"    
    echo "==========================================================================================
WARNING: CMAKE_CUDA_ARCHITECTURES is overridden by GPU_ARCHS.
         GPU_ARCHS is deprecated. Please use CMAKE_CUDA_ARCHITECTURES instead.
=========================================================================================="
fi

#
# Function to create symlink to compile_commands.json for IDE/clangd discovery
# (similar to NVBenchClangdCompileInfo.cmake)
#
create_compile_commands_symlink() {
  local build_dir=$1
  local source_dir=$2
  local compile_commands_file="$build_dir/compile_commands.json"
  local compile_commands_link="$source_dir/compile_commands.json"
  
  echo "Creating symlink from $compile_commands_link to $compile_commands_file..."
  ln -sf "$compile_commands_file" "$compile_commands_link"
}

#
# libcudf build
#
mkdir -p "$LIBCUDF_INSTALL_PATH" "$LIBCUDF_BUILD_PATH"
cd "$LIBCUDF_BUILD_PATH"

# Skip explicit cudf cmake configuration if it appears it has already configured
if [[ $LIBCUDF_BUILD_CONFIGURE == true || ! -f $LIBCUDF_BUILD_PATH/CMakeCache.txt ]]; then
  echo "Configuring cudf native libs"
  cmake "$CUDF_PATH/cpp" \
    -DBUILD_BENCHMARKS="$BUILD_CUDF_BENCHMARKS" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS="$CMAKE_EXPORT_COMPILE_COMMANDS" \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_TESTS="$BUILD_CUDF_TESTS" \
    -DCMAKE_BUILD_TYPE="$CUDF_BUILD_TYPE" \
    -DCMAKE_CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCHITECTURES" \
    -DCMAKE_INSTALL_PREFIX="$LIBCUDF_INSTALL_PATH" \
    -DCUDF_DEPENDENCY_PIN_MODE="$LIBCUDF_DEPENDENCY_MODE" \
    -DCUDA_STATIC_RUNTIME=ON \
    -DCUDF_USE_PER_THREAD_DEFAULT_STREAM="$CUDF_USE_PER_THREAD_DEFAULT_STREAM" \
    -DCUDF_KVIKIO_REMOTE_IO=OFF \
    -DCUDF_LARGE_STRINGS_DISABLED=ON \
    -DCUDF_EXPORT_NVCOMP=ON \
    -DLIBCUDF_LOGGING_LEVEL="$RMM_LOGGING_LEVEL" \
    -DRMM_LOGGING_LEVEL="$RMM_LOGGING_LEVEL" \
    -C="$CUDF_PIN_PATH/setup.cmake"
fi
if [[ $LIBCUDF_CONFIGURE_ONLY == ON ]]; then # submodule-sync.sh phase 1 call this script with LIBCUDF_CONFIGURE_ONLY=ON
  echo "Skip build..."
  exit 0
fi
echo "Building cudf native libs"
cmake --build "$LIBCUDF_BUILD_PATH" --target install "-j$CPP_PARALLEL_LEVEL"

#
# spark-gpu-expression-lib native build
# (No cudfjni or JNI - just our native library linked to cudf)
#
mkdir -p "$NATIVE_BUILD_PATH"
cd "$NATIVE_BUILD_PATH"
echo "Configuring spark-gpu-expression-lib native libs"
CUDF_ROOT="$CUDF_PATH" \
  CUDF_INSTALL_DIR="$LIBCUDF_INSTALL_PATH" \
  cmake \
    "$PROJECT_BASE_DIR/native" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS="$CMAKE_EXPORT_COMPILE_COMMANDS" \
    -DBUILD_TESTS="$BUILD_TESTS" \
    -DCUDF_DEPENDENCY_PIN_MODE=pinned \
    -DCUDF_USE_PER_THREAD_DEFAULT_STREAM="$CUDF_USE_PER_THREAD_DEFAULT_STREAM" \
    -DCMAKE_CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCHITECTURES" \
    -DRMM_LOGGING_LEVEL="$RMM_LOGGING_LEVEL" \
    -C="$CUDF_PIN_PATH/setup.cmake"

create_compile_commands_symlink "$NATIVE_BUILD_PATH" "$PROJECT_BASE_DIR/native"

echo "Building spark-gpu-expression-lib native libs"
cmake --build "$NATIVE_BUILD_PATH" "-j$CPP_PARALLEL_LEVEL"

# Run tests if enabled
if [[ "$BUILD_TESTS" == "ON" ]]; then
    echo "Running tests..."
    cd "$NATIVE_BUILD_PATH"
    ctest --output-on-failure
fi
