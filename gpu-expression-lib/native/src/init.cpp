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

#include "spark_gpu_expr/init.hpp"
#include "spark_gpu_expr/memory.hpp"

#include <atomic>
#include <mutex>

// cudf headers
#include <cudf/utilities/default_stream.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

namespace spark_gpu_expr {

namespace {

// Initialization state
std::atomic<bool> g_initialized{false};
std::mutex g_init_mutex;

// Default memory resources
std::unique_ptr<rmm::mr::cuda_memory_resource> g_cuda_mr;
std::unique_ptr<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>> g_pool_mr;

// Custom memory resources (externally owned)
MemoryResource* g_custom_mr = nullptr;
HostMemoryResource* g_custom_host_mr = nullptr;

// Default implementations that wrap RMM
class DefaultMemoryResource : public MemoryResource {
public:
    void* allocate(std::size_t bytes, std::size_t alignment) override {
        auto* mr = rmm::mr::get_current_device_resource();
        return mr->allocate(cudf::get_default_stream(), bytes, alignment);
    }
    
    void deallocate(void* ptr, std::size_t bytes, std::size_t alignment) override {
        auto* mr = rmm::mr::get_current_device_resource();
        mr->deallocate(cudf::get_default_stream(), ptr, bytes, alignment);
    }
};

class DefaultHostMemoryResource : public HostMemoryResource {
public:
    void* allocate(std::size_t bytes, std::size_t /*alignment*/) override {
        void* ptr = nullptr;
        cudaMallocHost(&ptr, bytes);
        if (ptr == nullptr) {
            throw std::bad_alloc();
        }
        return ptr;
    }
    
    void deallocate(void* ptr, std::size_t /*bytes*/, std::size_t /*alignment*/) override {
        cudaFreeHost(ptr);
    }
};

DefaultMemoryResource g_default_mr;
DefaultHostMemoryResource g_default_host_mr;

}  // namespace

bool init(InitOptions const& options) {
    std::lock_guard<std::mutex> lock(g_init_mutex);
    
    if (g_initialized.load()) {
        return false;  // Already initialized
    }
    
    // Set CUDA device
    cudaSetDevice(options.device_id);
    
    // Set up memory resources
    if (options.memory_resource) {
        g_custom_mr = options.memory_resource;
    } else if (options.enable_memory_pool) {
        // Create pooled memory resource
        g_cuda_mr = std::make_unique<rmm::mr::cuda_memory_resource>();
        
        std::optional<std::size_t> max_size = options.maximum_pool_size > 0 
            ? std::optional<std::size_t>(options.maximum_pool_size) 
            : std::nullopt;
        g_pool_mr = std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
            g_cuda_mr.get(),
            options.initial_pool_size,
            max_size);
        
        rmm::mr::set_current_device_resource(g_pool_mr.get());
    } else {
        // Use default cuda memory resource
        g_cuda_mr = std::make_unique<rmm::mr::cuda_memory_resource>();
        rmm::mr::set_current_device_resource(g_cuda_mr.get());
    }
    
    if (options.host_memory_resource) {
        g_custom_host_mr = options.host_memory_resource;
    }
    
    g_initialized.store(true);
    return true;
}

void shutdown() {
    std::lock_guard<std::mutex> lock(g_init_mutex);
    
    if (!g_initialized.load()) {
        return;  // Not initialized
    }
    
    // Release pooled resources
    if (g_pool_mr) {
        rmm::mr::set_current_device_resource(nullptr);
        g_pool_mr.reset();
    }
    g_cuda_mr.reset();
    
    // Clear custom resources (caller is responsible for their lifetime)
    g_custom_mr = nullptr;
    g_custom_host_mr = nullptr;
    
    g_initialized.store(false);
}

bool is_initialized() {
    return g_initialized.load();
}

const char* version() {
    return "0.1.0";
}

MemoryResource* get_memory_resource() {
    if (g_custom_mr) {
        return g_custom_mr;
    }
    return &g_default_mr;
}

void set_memory_resource(MemoryResource* mr) {
    g_custom_mr = mr;
}

HostMemoryResource* get_host_memory_resource() {
    if (g_custom_host_mr) {
        return g_custom_host_mr;
    }
    return &g_default_host_mr;
}

void set_host_memory_resource(HostMemoryResource* mr) {
    g_custom_host_mr = mr;
}

}  // namespace spark_gpu_expr

