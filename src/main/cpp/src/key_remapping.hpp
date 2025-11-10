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

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace spark_rapids_jni {

/**
 * @brief Result of building a key remapping structure.
 *
 * Contains the hash map for remapping.
 */
struct key_remap_build_result {
  void* hash_map_ptr;              // Opaque pointer to the hash map implementation
  cudf::null_equality nulls_equal;  // Whether nulls are considered equal
  bool has_nested_columns;          // Whether the keys contained nested columns

  ~key_remap_build_result();
};

/**
 * @brief Build a key remapping hash map from input keys.
 *
 * Creates a hash map that assigns unique integer IDs to distinct keys.
 * This is a single-pass operation that builds the hash map.
 *
 * @param input_keys The input table containing the keys to remap
 * @param nulls_equal Whether to treat null keys as equal
 * @param stream CUDA stream for device operations
 * @param mr Device memory resource
 * @return A structure containing the hash map
 */
std::unique_ptr<key_remap_build_result> build_key_remap_map(
  cudf::table_view const& input_keys,
  cudf::null_equality nulls_equal,
  rmm::cuda_stream_view stream              = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr         = cudf::get_current_device_resource_ref());

/**
 * @brief Apply key remapping to input keys using a pre-built hash map.
 *
 * This performs a single-pass lookup in the hash map and returns the remapped integer IDs
 * corresponding to build-side row indices.
 * Keys not found in the hash map are assigned a sentinel value (-1).
 *
 * @param build_keys The original build keys used to create the hash map
 * @param input_keys The input table containing the keys to remap
 * @param remap_result The pre-built key remapping structure
 * @param stream CUDA stream for device operations
 * @param mr Device memory resource
 * @return A column of INT32 values with the remapped keys
 */
std::unique_ptr<cudf::column> apply_key_remap(
  cudf::table_view const& build_keys,
  cudf::table_view const& input_keys,
  key_remap_build_result const& remap_result,
  rmm::cuda_stream_view stream              = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr         = cudf::get_current_device_resource_ref());

/**
 * @brief Free the hash map resources held by a key_remap_build_result.
 *
 * @param hash_map_ptr Opaque pointer to the hash map
 */
void free_key_remap_map(void* hash_map_ptr);

}  // namespace spark_rapids_jni

