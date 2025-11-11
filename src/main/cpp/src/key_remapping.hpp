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

// Forward declaration to avoid including .cuh in .hpp
namespace cudf::detail::row::equality {
class preprocessed_table;
}

namespace spark_rapids_jni {

/**
 * @brief Null equality modes for key remapping
 * 
 * Determines how null values are compared at different levels of the data structure.
 */
enum class null_equality_mode : int32_t {
  NULL_EQUAL = 0,      ///< All nulls are equal at every level
  NULL_NOT_EQUAL = 1,  ///< No nulls are equal at any level
  SPARK_EQUALITY = 2   ///< Top-level nulls not equal, nested nulls equal (Spark semantics)
};

// Sentinel values for key remapping
constexpr cudf::size_type NOT_FOUND_SENTINEL = -1;  ///< Probe-side keys not in build table
constexpr cudf::size_type BUILD_NULL_SENTINEL = -2;  ///< Build-side rows with top-level nulls

/**
 * @brief Result of building a key remapping structure.
 *
 * Contains the hash map for remapping and cached preprocessed build table.
 */
struct key_remap_build_result {
  void* hash_map_ptr;              // Opaque pointer to the hash map implementation
  null_equality_mode null_mode;     // Null equality mode for comparisons
  bool has_nested_columns;          // Whether the keys contained nested columns
  
  // Cached preprocessed build table for efficient reuse across multiple probe operations
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> preprocessed_build;

  ~key_remap_build_result();
};

/**
 * @brief Build a key remapping hash map from input keys.
 *
 * Creates a hash map that assigns unique integer IDs to distinct keys.
 * This is a single-pass operation that builds the hash map.
 *
 * @param input_keys The input table containing the keys to remap
 * @param null_mode Null equality mode (NULL_EQUAL, NULL_NOT_EQUAL, or SPARK_EQUALITY)
 * @param stream CUDA stream for device operations
 * @param mr Device memory resource
 * @return A structure containing the hash map
 */
std::unique_ptr<key_remap_build_result> build_key_remap_map(
  cudf::table_view const& input_keys,
  null_equality_mode null_mode,
  rmm::cuda_stream_view stream              = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr         = cudf::get_current_device_resource_ref());

/**
 * @brief Apply key remapping to input keys using a pre-built hash map.
 *
 * This performs a single-pass lookup in the hash map and returns the remapped integer IDs.
 * The behavior depends on the null equality mode and whether this is the build or probe side:
 * - Matching keys: Return non-negative integer (build table row index)
 * - Non-matching probe keys: Return NOT_FOUND_SENTINEL (-1)
 * - Build keys with top-level nulls (when applicable): Return BUILD_NULL_SENTINEL (-2)
 *
 * @param build_keys The original build keys used to create the hash map
 * @param input_keys The input table containing the keys to remap
 * @param remap_result The pre-built key remapping structure
 * @param null_mode Null equality mode (must match the mode used during build)
 * @param is_build_side True if remapping the build side, false for probe side
 * @param stream CUDA stream for device operations
 * @param mr Device memory resource
 * @return A column of INT32 values with the remapped keys
 */
std::unique_ptr<cudf::column> apply_key_remap(
  cudf::table_view const& build_keys,
  cudf::table_view const& input_keys,
  key_remap_build_result const& remap_result,
  null_equality_mode null_mode,
  bool is_build_side,
  rmm::cuda_stream_view stream              = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr         = cudf::get_current_device_resource_ref());

/**
 * @brief Free the hash map resources held by a key_remap_build_result.
 *
 * @param hash_map_ptr Opaque pointer to the hash map
 */
void free_key_remap_map(void* hash_map_ptr);

/**
 * @brief Dump the raw contents of the remapping hash map for debugging.
 *
 * Returns a table with three INT32 columns showing the internal map structure:
 * - Column 0: Hash values (UINT32)
 * - Column 1: Key row indices from the build table (INT32)
 * - Column 2: Mapped values (INT32)
 *
 * @param build_keys The original build keys used to create the hash map (unused, kept for API consistency)
 * @param remap_result The pre-built key remapping structure
 * @param stream CUDA stream for device operations
 * @param mr Device memory resource
 * @return A table containing all entries in the remapping hash map
 */
std::unique_ptr<cudf::table> dump_remap_table(
  cudf::table_view const& build_keys,
  key_remap_build_result const& remap_result,
  rmm::cuda_stream_view stream              = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr         = cudf::get_current_device_resource_ref());

}  // namespace spark_rapids_jni

