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

#include <cstddef>
#include <cstdint>
#include <memory>

namespace spark_gpu_expr {

/**
 * Memory resource interface for GPU memory allocation.
 * 
 * This allows external callers to provide
 * their own memory management. By default, cudf's per-device memory
 * resource is used.
 * 
 * Implementors should provide thread-safe allocate/deallocate methods.
 */
class MemoryResource {
public:
    virtual ~MemoryResource() = default;
    
    /**
     * Allocate GPU memory.
     * 
     * @param bytes Number of bytes to allocate
     * @param alignment Alignment requirement (must be power of 2)
     * @return Pointer to allocated memory
     * @throws std::bad_alloc if allocation fails
     */
    virtual void* allocate(std::size_t bytes, std::size_t alignment = 256) = 0;
    
    /**
     * Deallocate GPU memory.
     * 
     * @param ptr Pointer previously returned by allocate()
     * @param bytes Number of bytes that were allocated
     * @param alignment Alignment that was used
     */
    virtual void deallocate(void* ptr, std::size_t bytes, std::size_t alignment = 256) = 0;
};

/**
 * Host (CPU) memory resource interface.
 * 
 * Used for pinned memory allocations that are efficient for CPU-GPU transfers.
 */
class HostMemoryResource {
public:
    virtual ~HostMemoryResource() = default;
    
    /**
     * Allocate host memory.
     * 
     * @param bytes Number of bytes to allocate
     * @param alignment Alignment requirement (must be power of 2)
     * @return Pointer to allocated memory
     * @throws std::bad_alloc if allocation fails
     */
    virtual void* allocate(std::size_t bytes, std::size_t alignment = 256) = 0;
    
    /**
     * Deallocate host memory.
     * 
     * @param ptr Pointer previously returned by allocate()
     * @param bytes Number of bytes that were allocated
     * @param alignment Alignment that was used
     */
    virtual void deallocate(void* ptr, std::size_t bytes, std::size_t alignment = 256) = 0;
};

/**
 * Get the current GPU memory resource.
 * 
 * @return The current memory resource (never null)
 */
MemoryResource* get_memory_resource();

/**
 * Set a custom GPU memory resource.
 * 
 * The caller is responsible for ensuring the resource remains valid
 * until shutdown() is called or another resource is set.
 * 
 * @param mr The memory resource to use (nullptr to reset to default)
 */
void set_memory_resource(MemoryResource* mr);

/**
 * Get the current host memory resource.
 * 
 * @return The current host memory resource (never null)
 */
HostMemoryResource* get_host_memory_resource();

/**
 * Set a custom host memory resource.
 * 
 * The caller is responsible for ensuring the resource remains valid
 * until shutdown() is called or another resource is set.
 * 
 * @param mr The host memory resource to use (nullptr to reset to default)
 */
void set_host_memory_resource(HostMemoryResource* mr);

}  // namespace spark_gpu_expr

