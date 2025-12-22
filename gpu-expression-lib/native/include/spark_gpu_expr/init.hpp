/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "memory.hpp"

namespace spark_gpu_expr {

/**
 * Initialization options for the library.
 */
struct InitOptions {
    /**
     * Custom GPU memory resource. If nullptr, default cudf memory is used.
     * The caller must ensure this remains valid until shutdown().
     */
    MemoryResource* memory_resource = nullptr;
    
    /**
     * Custom host memory resource. If nullptr, default cudf host memory is used.
     * The caller must ensure this remains valid until shutdown().
     */
    HostMemoryResource* host_memory_resource = nullptr;
    
    /**
     * CUDA device ID to use. Default is device 0.
     */
    int device_id = 0;
    
    /**
     * Whether to enable memory pool for cudf operations.
     * This can improve performance by reducing allocation overhead.
     * Ignored if custom memory_resource is provided.
     */
    bool enable_memory_pool = true;
    
    /**
     * Initial pool size in bytes (only used if enable_memory_pool is true).
     * 0 means use cudf's default.
     */
    std::size_t initial_pool_size = 0;
    
    /**
     * Maximum pool size in bytes (only used if enable_memory_pool is true).
     * 0 means no limit.
     */
    std::size_t maximum_pool_size = 0;
};

/**
 * Initialize the spark-gpu-expression library.
 * 
 * Must be called before any expression operations. Can be called
 * multiple times (subsequent calls are no-ops if already initialized).
 * 
 * Thread-safe: Yes (uses internal locking)
 * 
 * @param options Initialization options (optional)
 * @return true if initialization succeeded, false if already initialized
 */
bool init(InitOptions const& options = {});

/**
 * Shutdown the spark-gpu-expression library.
 * 
 * Releases all cached resources and memory pools. After calling this,
 * init() must be called again before using any expression functions.
 * 
 * Thread-safe: Yes (uses internal locking)
 * 
 * Note: This does NOT free memory allocated by custom memory resources.
 * The caller is responsible for managing their custom resources.
 */
void shutdown();

/**
 * Check if the library is initialized.
 * 
 * @return true if init() has been called and shutdown() has not
 */
bool is_initialized();

/**
 * Get library version information.
 * 
 * @return Version string in format "major.minor.patch"
 */
const char* version();

}  // namespace spark_gpu_expr

