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

#include "key_remapping.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <cuco/static_map.cuh>
#include <cuda/atomic>
#include <cuda/stream_ref>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace spark_rapids_jni {

namespace {

/**
 * @brief Custom hasher that extracts hash from pair<hash, index>
 */
struct pair_hasher {
  template <typename T>
  __device__ constexpr cudf::hash_value_type operator()(
    cuco::pair<cudf::hash_value_type, T> const& key) const noexcept
  {
    return key.first;
  }
};

/**
 * @brief Two-table equality comparator adapter for probe operations
 * Compares rows from probe table (lhs.second) against build table (rhs.second)
 */
template <typename TwoTableEqual>
struct two_table_comparator_adapter {
  two_table_comparator_adapter(TwoTableEqual const& d_equal) : _d_equal{d_equal} {}

  __device__ constexpr auto operator()(
    cuco::pair<cudf::hash_value_type, cudf::detail::row::lhs_index_type> const& lhs,
    cuco::pair<cudf::hash_value_type, cudf::detail::row::rhs_index_type> const& rhs) const noexcept
  {
    if (lhs.first != rhs.first) { return false; }  // Hash mismatch
    return _d_equal(lhs.second, rhs.second);        // Compare actual rows
  }

 private:
  TwoTableEqual _d_equal;
};

using key_type = cuco::pair<cudf::hash_value_type, cudf::detail::row::rhs_index_type>;

template <bool has_nested>
using device_row_comparator_type = cudf::detail::row::equality::device_row_comparator<
  has_nested,
  cudf::nullate::DYNAMIC,
  cudf::detail::row::equality::nan_equal_physical_equality_comparator>;

/**
 * @brief Equality comparator for hash map keys that delegates to cudf row comparator.
 */
template <bool has_nested>
struct build_pair_equality {
  device_row_comparator_type<has_nested> row_equal;

  __device__ constexpr bool operator()(key_type const& lhs,
                                       key_type const& rhs) const noexcept
  {
    if (lhs.first != rhs.first) { return false; }
    auto const lhs_index = static_cast<cudf::size_type>(lhs.second);
    auto const rhs_index = static_cast<cudf::size_type>(rhs.second);
    return row_equal(lhs_index, rhs_index);
  }
};

template <bool has_nested>
using key_map_type = cuco::static_map<
  key_type,
  cudf::size_type,
  cuco::extent<int64_t>,
  cuda::thread_scope_device,
  build_pair_equality<has_nested>,
  cuco::linear_probing<1, pair_hasher>,
  rmm::mr::polymorphic_allocator<char>,
  cuco::storage<2>>;

/**
 * @brief Type-erased hash map wrapper for storage
 */
struct hash_map_holder {
  virtual ~hash_map_holder() = default;
};

template <bool has_nested>
struct typed_hash_map_holder : hash_map_holder {
  using map_type = key_map_type<has_nested>;

  map_type map;

  explicit typed_hash_map_holder(map_type&& m) : map(std::move(m)) {}
};

/**
 * @brief Build the hash map from input keys
 */
template <bool has_nested>
std::unique_ptr<hash_map_holder> build_map_impl(
  cudf::table_view const& input,
  cudf::detail::row::hash::row_hasher const& row_hash,
  cudf::detail::row::equality::self_comparator const& row_equal,
  bool has_nulls,
  cudf::null_equality nulls_equal,
  null_equality_mode null_mode,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_rows = input.num_rows();

  auto const d_hasher = row_hash.device_hasher(cudf::nullate::DYNAMIC{has_nulls});

  using rhs_type = cudf::detail::row::rhs_index_type;
  auto const empty_key = key_type{
    std::numeric_limits<cudf::hash_value_type>::max(), rhs_type{cudf::JoinNoMatch}};

  auto key_equal = build_pair_equality<has_nested>{
    row_equal.template equal_to<has_nested>(cudf::nullate::DYNAMIC{has_nulls}, nulls_equal)};

  auto map = typename typed_hash_map_holder<has_nested>::map_type{
    num_rows,
    cuco::empty_key<key_type>{empty_key},
    cuco::empty_value<cudf::size_type>{cudf::detail::CUDF_SIZE_TYPE_SENTINEL},
    std::move(key_equal),
    {},
    {},
    {},
    rmm::mr::polymorphic_allocator<char>{},
    stream.value()};

  // Create device table view for top-level null checking
  auto const d_input = cudf::table_device_view::create(input, stream);
  
  auto map_ref = map.ref(cuco::op::insert_and_find);
  
  // For SPARK_EQUALITY and NULL_NOT_EQUAL, skip rows with top-level nulls during insertion
  bool const skip_top_level_nulls = 
    (null_mode == null_equality_mode::SPARK_EQUALITY) || 
    (null_mode == null_equality_mode::NULL_NOT_EQUAL);
  
  // For SPARK_EQUALITY and NULL_NOT_EQUAL modes, we need to check for top-level nulls.
  // Optimization: if there are no nulls at all, skip to the fast path without null checking.
  // This is a common case in many workloads and avoids device-side null checking overhead.
  if (skip_top_level_nulls && has_nulls) {
    // Need to check for top-level nulls on each row and skip inserting those rows
    auto const d_input_view = *d_input;  // Copy the device view by value
    
    thrust::for_each(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(num_rows),
      [map_ref, d_hasher, d_input_view] __device__(cudf::size_type idx) mutable {
        // Check if this row has any top-level nulls
        bool has_null = false;
        for (cudf::size_type col_idx = 0; col_idx < d_input_view.num_columns(); ++col_idx) {
          if (d_input_view.column(col_idx).is_null(idx)) {
            has_null = true;
            break;
          }
        }
        
        // Skip rows with top-level nulls
        if (has_null) {
          return;
        }
        
        auto const row_hash = d_hasher(idx);
        using rhs_type      = cudf::detail::row::rhs_index_type;
        auto [iter, inserted] =
          map_ref.insert_and_find(cuco::pair{key_type{row_hash, rhs_type{idx}}, idx});

        if (inserted) {
          iter->second = idx;
        } else {
          // If the key already exists, atomically update to the minimum row index
          cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> ref{iter->second};
          ref.fetch_min(idx, cuda::memory_order_relaxed);
        }
      });
  } else {
    // When not skipping nulls, use the simpler version without null checks
    thrust::for_each(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(num_rows),
      [map_ref, d_hasher] __device__(cudf::size_type idx) mutable {
        auto const row_hash = d_hasher(idx);
        using rhs_type      = cudf::detail::row::rhs_index_type;
        auto [iter, inserted] =
          map_ref.insert_and_find(cuco::pair{key_type{row_hash, rhs_type{idx}}, idx});

        if (inserted) {
          iter->second = idx;
        } else {
          // If the key already exists, atomically update to the minimum row index
          cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> ref{iter->second};
          ref.fetch_min(idx, cuda::memory_order_relaxed);
        }
      });
  }

  auto holder =
    std::make_unique<typed_hash_map_holder<has_nested>>(std::move(map));

  return holder;
}

/**
 * @brief Device functor to create probe keys as pair<hash, lhs_index>
 */
struct probe_keys_fn {
  using hasher_type = cudf::detail::row::hash::device_row_hasher<cudf::hashing::detail::default_hash,
                                                                 cudf::nullate::DYNAMIC>;

  probe_keys_fn(hasher_type const& hash) : _hash{hash} {}

  __device__ __forceinline__ auto operator()(cudf::size_type i) const noexcept
  {
    using lhs_type = cudf::detail::row::lhs_index_type;
    return cuco::pair{_hash(i), lhs_type{i}};
  }

 private:
  hasher_type _hash;
};

/**
 * @brief Look up keys from probe table using the build hash map with two-table comparator
 */
template <typename MapType, typename TwoTableEqual>
std::unique_ptr<cudf::column> lookup_keys_two_table(
  cudf::table_view const& probe_keys,
  MapType const& build_map,
  cudf::detail::row::hash::row_hasher const& probe_hasher,
  bool has_nulls,
  TwoTableEqual const& device_comparator,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_probe_rows = probe_keys.num_rows();

  if (num_probe_rows == 0) {
    return cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32}, 0, cudf::mask_state::UNALLOCATED, stream, mr);
  }

  // Output column never has nulls - we use sentinel values instead
  auto output = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 
    num_probe_rows, 
    cudf::mask_state::UNALLOCATED, 
    stream, 
    mr);

  // Get device hasher for probe table
  auto const d_hasher = probe_hasher.device_hasher(cudf::nullate::DYNAMIC{has_nulls});

  // Create iterator over probe keys as pair<hash, lhs_index>
  auto const iter =
    cudf::detail::make_counting_transform_iterator(0, probe_keys_fn{d_hasher});

  auto output_begin = output->mutable_view().begin<cudf::size_type>();
  build_map.find_async(iter,
                       iter + num_probe_rows,
                       two_table_comparator_adapter{device_comparator},
                       pair_hasher{},
                       output_begin,
                       cuda::stream_ref{stream.value()});

  return output;
}

}  // anonymous namespace

// Destructor implementation
key_remap_build_result::~key_remap_build_result()
{
  if (hash_map_ptr) { free_key_remap_map(hash_map_ptr); }
}

std::unique_ptr<key_remap_build_result> build_key_remap_map(cudf::table_view const& input_keys,
                                                             null_equality_mode null_mode,
                                                             rmm::cuda_stream_view stream,
                                                             rmm::device_async_resource_ref mr)
{
  auto const has_nested_columns = cudf::detail::has_nested_columns(input_keys);

  if (input_keys.num_rows() == 0 || input_keys.num_columns() == 0) {
    auto result            = std::make_unique<key_remap_build_result>();
    result->hash_map_ptr   = nullptr;
    result->null_mode      = null_mode;
    result->has_nested_columns = has_nested_columns;
    result->preprocessed_build = nullptr;
    return result;
  }

  // Convert null_equality_mode to CUDF's null_equality
  // For SPARK_EQUALITY, we use EQUAL for nested nulls (CUDF will handle this correctly)
  // and we'll filter out top-level nulls during insertion
  cudf::null_equality cudf_nulls_equal = cudf::null_equality::EQUAL;  // Default initialization
  switch (null_mode) {
    case null_equality_mode::NULL_EQUAL:
      cudf_nulls_equal = cudf::null_equality::EQUAL;
      break;
    case null_equality_mode::NULL_NOT_EQUAL:
      cudf_nulls_equal = cudf::null_equality::UNEQUAL;
      break;
    case null_equality_mode::SPARK_EQUALITY:
      // Use EQUAL for CUDF - this handles nested nulls correctly
      // We'll skip top-level nulls during insertion
      cudf_nulls_equal = cudf::null_equality::EQUAL;
      break;
  }

  // Preprocess the input for hashing
  auto const preprocessed_input =
    cudf::detail::row::hash::preprocessed_table::create(input_keys, stream);
  auto const has_nulls =
    cudf::has_nulls(input_keys) || cudf::has_nested_nulls(input_keys);

  auto const row_hash = cudf::detail::row::hash::row_hasher(preprocessed_input);

  // Preprocess input for equality comparisons (row equality) and cache for reuse
  auto preprocessed_equal =
    cudf::detail::row::equality::preprocessed_table::create(input_keys, stream);
  auto const self_equal =
    cudf::detail::row::equality::self_comparator(preprocessed_equal);

  std::unique_ptr<hash_map_holder> map_holder;

  if (has_nested_columns) {
    map_holder = build_map_impl<true>(
      input_keys, row_hash, self_equal, has_nulls, cudf_nulls_equal, null_mode, stream, mr);
  } else {
    map_holder = build_map_impl<false>(
      input_keys, row_hash, self_equal, has_nulls, cudf_nulls_equal, null_mode, stream, mr);
  }

  // Create the result
  auto result            = std::make_unique<key_remap_build_result>();
  result->hash_map_ptr   = map_holder.release();
  result->null_mode      = null_mode;
  result->has_nested_columns = has_nested_columns;
  
  // Cache the preprocessed build table for reuse across multiple probe operations
  result->preprocessed_build = std::move(preprocessed_equal);

  return result;
}

std::unique_ptr<cudf::column> apply_key_remap(cudf::table_view const& build_keys,
                                               cudf::table_view const& input_keys,
                                               key_remap_build_result const& remap_result,
                                               null_equality_mode null_mode,
                                               bool is_build_side,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  if (input_keys.num_rows() == 0 || input_keys.num_columns() == 0) {
    return cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32}, 0, cudf::mask_state::UNALLOCATED, stream, mr);
  }

  // If build table was empty (hash_map_ptr is nullptr), all keys map to NOT_FOUND_SENTINEL
  if (remap_result.hash_map_ptr == nullptr) {
    auto output = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32}, input_keys.num_rows(), cudf::mask_state::UNALLOCATED, stream, mr);
    thrust::fill(rmm::exec_policy(stream),
                 output->mutable_view().begin<cudf::size_type>(),
                 output->mutable_view().end<cudf::size_type>(),
                 NOT_FOUND_SENTINEL);
    return output;
  }

  // Convert null_equality_mode to CUDF's null_equality
  cudf::null_equality cudf_nulls_equal = cudf::null_equality::EQUAL;  // Default initialization
  switch (null_mode) {
    case null_equality_mode::NULL_EQUAL:
      cudf_nulls_equal = cudf::null_equality::EQUAL;
      break;
    case null_equality_mode::NULL_NOT_EQUAL:
      cudf_nulls_equal = cudf::null_equality::UNEQUAL;
      break;
    case null_equality_mode::SPARK_EQUALITY:
      cudf_nulls_equal = cudf::null_equality::EQUAL;
      break;
  }

  // Only preprocess the probe table (changes with each call)
  // Use cached preprocessed build table (created once during build phase)
  auto preprocessed_probe =
    cudf::detail::row::equality::preprocessed_table::create(input_keys, stream);
  
  auto const has_nulls =
    cudf::has_nulls(build_keys) || cudf::has_nested_nulls(build_keys) || cudf::has_nulls(input_keys) ||
    cudf::has_nested_nulls(input_keys);
  auto const use_nested_comparator = remap_result.has_nested_columns;

  // Create two-table comparator (probe -> build)
  auto const two_table_equal =
    cudf::detail::row::equality::two_table_comparator(preprocessed_probe, remap_result.preprocessed_build);

  // Create probe hasher
  auto const probe_hasher = cudf::detail::row::hash::row_hasher(preprocessed_probe);
  // Call lookup with the appropriate comparator
  std::unique_ptr<cudf::column> result;
  if (use_nested_comparator) {
    auto const device_comparator =
      two_table_equal.equal_to<true>(cudf::nullate::DYNAMIC{has_nulls}, cudf_nulls_equal);
    auto const* holder = static_cast<typed_hash_map_holder<true>*>(remap_result.hash_map_ptr);
    result = lookup_keys_two_table(
      input_keys, holder->map, probe_hasher, has_nulls, device_comparator, stream, mr);
  } else {
    auto const device_comparator =
      two_table_equal.equal_to<false>(cudf::nullate::DYNAMIC{has_nulls}, cudf_nulls_equal);
    auto const* holder = static_cast<typed_hash_map_holder<false>*>(remap_result.hash_map_ptr);
    result = lookup_keys_two_table(
      input_keys, holder->map, probe_hasher, has_nulls, device_comparator, stream, mr);
  }
  
  // For build-side remapping with SPARK_EQUALITY or NULL_NOT_EQUAL:
  // Assign BUILD_NULL_SENTINEL to rows with top-level nulls
  // Optimization: Only check for nulls if the input actually has nulls
  bool const needs_null_processing = is_build_side && 
      (null_mode == null_equality_mode::SPARK_EQUALITY || null_mode == null_equality_mode::NULL_NOT_EQUAL) &&
      (cudf::has_nulls(input_keys) || cudf::has_nested_nulls(input_keys));
  
  if (needs_null_processing) {
    auto const d_input = cudf::table_device_view::create(input_keys, stream);
    auto const d_input_view = *d_input;  // Copy device view by value
    auto result_view = result->mutable_view();
    auto result_data = result_view.begin<cudf::size_type>();
    
    thrust::for_each(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(input_keys.num_rows()),
      [result_data, d_input_view] __device__(cudf::size_type idx) mutable {
        // Check if this row has any top-level nulls
        bool has_null = false;
        for (cudf::size_type col_idx = 0; col_idx < d_input_view.num_columns(); ++col_idx) {
          if (d_input_view.column(col_idx).is_null(idx)) {
            has_null = true;
            break;
          }
        }
        
        if (has_null) {
          result_data[idx] = BUILD_NULL_SENTINEL;
        }
      });
  }

  return result;
}

void free_key_remap_map(void* hash_map_ptr)
{
  if (hash_map_ptr) {
    // We need to know the type to properly delete
    // This is tricky because we've type-erased it
    // The virtual destructor in hash_map_holder should handle this
    delete static_cast<hash_map_holder*>(hash_map_ptr);
  }
}

namespace {

/**
 * @brief Extract all entries from the hash map for debugging
 * Returns the raw map internals: hash values, key row indices, and mapped values
 */
template <typename MapType>
std::unique_ptr<cudf::table> dump_map_impl(
  MapType const& build_map,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // Retrieve all key-value pairs from the map
  rmm::device_uvector<key_type> retrieved_keys(build_map.capacity(), stream);
  rmm::device_uvector<cudf::size_type> retrieved_values(build_map.capacity(), stream);
  
  auto [keys_end, vals_end] = build_map.retrieve_all(
    retrieved_keys.begin(), 
    retrieved_values.begin(), 
    cuda::stream_ref{stream.value()});
  
  auto const num_entries = std::distance(retrieved_keys.begin(), keys_end);
  
  // Resize to actual number of entries
  retrieved_keys.resize(num_entries, stream);
  retrieved_values.resize(num_entries, stream);
  
  // Extract hash values from keys
  auto hash_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::UINT32},
    num_entries,
    cudf::mask_state::UNALLOCATED,
    stream,
    mr);
  auto hash_output = hash_col->mutable_view().template begin<cudf::hash_value_type>();
  thrust::transform(
    rmm::exec_policy(stream),
    retrieved_keys.begin(),
    retrieved_keys.end(),
    hash_output,
    [] __device__(key_type const& k) { 
      return k.first; 
    });
  
  // Extract row indices from keys
  auto row_idx_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32},
    num_entries,
    cudf::mask_state::UNALLOCATED,
    stream,
    mr);
  auto row_idx_output = row_idx_col->mutable_view().template begin<cudf::size_type>();
  thrust::transform(
    rmm::exec_policy(stream),
    retrieved_keys.begin(),
    retrieved_keys.end(),
    row_idx_output,
    [] __device__(key_type const& k) { 
      return static_cast<cudf::size_type>(k.second); 
    });
  
  // Create a column from retrieved values (mapped values)
  auto values_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32},
    num_entries,
    cudf::mask_state::UNALLOCATED,
    stream,
    mr);
  auto values_output = values_col->mutable_view().template begin<cudf::size_type>();
  thrust::copy(
    rmm::exec_policy(stream),
    retrieved_values.begin(),
    retrieved_values.end(),
    values_output);
  
  // Build result table with 3 columns: hash, key_row_index, mapped_value
  std::vector<std::unique_ptr<cudf::column>> result_columns;
  result_columns.push_back(std::move(hash_col));
  result_columns.push_back(std::move(row_idx_col));
  result_columns.push_back(std::move(values_col));
  
  return std::make_unique<cudf::table>(std::move(result_columns));
}

}  // anonymous namespace

std::unique_ptr<cudf::table> dump_remap_table(
  cudf::table_view const& build_keys,
  key_remap_build_result const& remap_result,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (remap_result.hash_map_ptr == nullptr) {
    // Return empty table with 3 columns: hash, key_row_index, mapped_value
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::UINT32}, 0, cudf::mask_state::UNALLOCATED, stream, mr));
    columns.push_back(cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32}, 0, cudf::mask_state::UNALLOCATED, stream, mr));
    columns.push_back(cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32}, 0, cudf::mask_state::UNALLOCATED, stream, mr));
    return std::make_unique<cudf::table>(std::move(columns));
  }

  if (remap_result.has_nested_columns) {
    auto const* holder = static_cast<typed_hash_map_holder<true>*>(remap_result.hash_map_ptr);
    return dump_map_impl(holder->map, stream, mr);
  } else {
    auto const* holder = static_cast<typed_hash_map_holder<false>*>(remap_result.hash_map_ptr);
    return dump_map_impl(holder->map, stream, mr);
  }
}

}  // namespace spark_rapids_jni

