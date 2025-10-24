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

#include "mixed_sort_merge_join.hpp"

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/mixed_join.hpp>
#include <cudf/join/sort_merge_join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda_runtime_api.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <cstdlib>
#include <string>

namespace spark_rapids_jni {

namespace {

// =============================================================================
// HELPER FUNCTIONS (Used by both conditional and non-conditional joins)
// =============================================================================

// Check environment variable to determine which join implementation to use
// Returns true if sort join should be used, false for hash join (default)
bool use_sort_join_for_inner()
{
  static bool use_sort = []() {
    char const* env_val = std::getenv("SPARK_RAPIDS_USE_SORT_JOIN");
    return env_val != nullptr && std::string(env_val) == "1";
  }();
  return use_sort;
}

// Check environment variable to determine whether to use built-in conditional join
// Returns true if cudf's conditional_inner_join should be used (integrates hash join + AST filtering)
// Returns false (default) to use two-step approach: inner join → separate AST filtering
bool use_builtin_conditional_join()
{
  static bool use_builtin = []() {
    char const* env_val = std::getenv("SPARK_RAPIDS_USE_BUILTIN_CONDITIONAL");
    return env_val != nullptr && std::string(env_val) == "1";
  }();
  return use_builtin;
}

/**
 * @brief Perform an inner join using either sort-merge or hash join based on environment variable
 *
 * This wrapper allows switching between sort-merge and hash join implementations at runtime
 * via the SPARK_RAPIDS_USE_SORT_JOIN environment variable (set to "1" to use sort-merge join).
 * Both implementations return the same result format: pairs of indices into the left and right tables.
 *
 * @note When using hash join, the is_left_sorted and is_right_sorted parameters are ignored.
 * @note Hash join optimization: uses the smaller table as the build table for better performance.
 */
std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
inner_join_impl(cudf::table_view const& left_keys,
                cudf::table_view const& right_keys,
                cudf::sorted is_left_sorted,
                cudf::sorted is_right_sorted,
                cudf::null_equality compare_nulls,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr)
{
  if (!use_sort_join_for_inner()) {
    // Use hash join (sorted flags are ignored for hash join)
    // Performance optimization: use the smaller table as the build table
    if (left_keys.num_rows() < right_keys.num_rows()) {
      // Build on left (smaller), probe with right (larger)
      auto hash_join = cudf::hash_join(left_keys, compare_nulls, stream);
      // inner_join returns (probe_indices, build_indices) = (right_indices, left_indices)
      // We need to swap to return (left_indices, right_indices)
      auto [right_indices, left_indices] = hash_join.inner_join(right_keys, std::nullopt, stream, mr);
      return {std::move(left_indices), std::move(right_indices)};
    } else {
      // Build on right (smaller or equal), probe with left (larger)
      auto hash_join = cudf::hash_join(right_keys, compare_nulls, stream);
      // inner_join returns (probe_indices, build_indices) = (left_indices, right_indices)
      return hash_join.inner_join(left_keys, std::nullopt, stream, mr);
    }
  } else {
    // Use sort-merge join
    cudf::sort_merge_join join_obj(right_keys, is_right_sorted, compare_nulls, stream);
    return join_obj.inner_join(left_keys, is_left_sorted, stream, mr);
  }
}

// Forward declaration for filter_by_conditional (defined later in the file)
std::pair<rmm::device_uvector<cudf::size_type>, rmm::device_uvector<cudf::size_type>>
filter_by_conditional(rmm::device_uvector<cudf::size_type>& left_indices,
                      rmm::device_uvector<cudf::size_type>& right_indices,
                      cudf::table_device_view const& left_table,
                      cudf::table_device_view const& right_table,
                      cudf::ast::detail::expression_device_view device_expression_data,
                      cudf::size_type num_intermediates,
                      bool has_nulls,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr);

/**
 * @brief Perform conditional inner join using either built-in or two-step approach
 *
 * This function performs an inner join with both equality and conditional predicates.
 * It can use either:
 * 1. cudf's built-in mixed_inner_join (when SPARK_RAPIDS_USE_BUILTIN_CONDITIONAL=1)
 * 2. Two-step approach: inner_join_impl + filter_by_conditional (default)
 *
 * @return Pair of device vectors containing left and right indices of matching rows
 */
std::pair<rmm::device_uvector<cudf::size_type>, rmm::device_uvector<cudf::size_type>>
conditional_inner_join_impl(cudf::table_view const& left_equality,
                            cudf::table_view const& right_equality,
                            cudf::table_view const& left_conditional,
                            cudf::table_view const& right_conditional,
                            cudf::ast::expression const& binary_predicate,
                            cudf::sorted is_left_sorted,
                            cudf::sorted is_right_sorted,
                            cudf::null_equality compare_nulls,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr)
{
  if (use_builtin_conditional_join()) {
    // Use cudf's built-in mixed_inner_join (hash join + AST filtering integrated)
    // Note: mixed_inner_join uses the full tables (equality + conditional columns combined)
    auto [left_result, right_result] = cudf::mixed_inner_join(
      left_equality, right_equality, left_conditional, right_conditional, 
      binary_predicate, compare_nulls, std::nullopt, stream, mr);
    return {std::move(*left_result), std::move(*right_result)};
  }

  // Otherwise, use two-step approach: inner join → separate AST filtering
  // Check for nulls in conditional columns (top-level and nested)
  auto const has_nulls =
    cudf::has_nested_nulls(left_conditional) || cudf::has_nested_nulls(right_conditional);

  // Parse the AST expression (intermediate)
  auto const parser = cudf::ast::detail::expression_parser{binary_predicate,
                                                           left_conditional,
                                                           right_conditional,
                                                           has_nulls,
                                                           stream,
                                                           cudf::get_current_device_resource_ref()};
  CUDF_EXPECTS(parser.output_type().id() == cudf::type_id::BOOL8,
               "The expression must produce a boolean output.");

  // Step 1: Perform inner join on equality keys (intermediate result)
  auto [equality_left_indices, equality_right_indices] =
    inner_join_impl(left_equality, right_equality, is_left_sorted, is_right_sorted, compare_nulls, stream, cudf::get_current_device_resource_ref());

  // If no equality matches, return empty result
  if (equality_left_indices->size() == 0) {
    return {std::move(*equality_left_indices), std::move(*equality_right_indices)};
  }

  // Create device views of conditional tables
  auto left_table  = cudf::table_device_view::create(left_conditional, stream);
  auto right_table = cudf::table_device_view::create(right_conditional, stream);

  // Step 2: Filter by conditional expression (final result)
  auto result = filter_by_conditional(*equality_left_indices,
                                      *equality_right_indices,
                                      *left_table,
                                      *right_table,
                                      parser.device_expression_data,
                                      parser.device_expression_data.num_intermediates,
                                      has_nulls,
                                      stream,
                                      cudf::get_current_device_resource_ref());
  
  // Transfer ownership from intermediate to final result with correct memory resource
  auto final_left = rmm::device_uvector<cudf::size_type>(result.first.size(), stream, mr);
  auto final_right = rmm::device_uvector<cudf::size_type>(result.second.size(), stream, mr);
  thrust::copy(rmm::exec_policy_nosync(stream), result.first.begin(), result.first.end(), final_left.begin());
  thrust::copy(rmm::exec_policy_nosync(stream), result.second.begin(), result.second.end(), final_right.begin());
  
  return {std::move(final_left), std::move(final_right)};
}

// Type alias for intermediate storage used in AST expression evaluation.
// Using a global alias ensures consistency between kernel launch site and kernel body,
// protecting against discrepancies when doing pointer casts with dynamic shared memory.
template <bool has_nulls>
using intermediate_storage_type = cudf::ast::detail::IntermediateDataType<has_nulls>;

template <bool has_nulls>
__global__ void filter_join_indices_kernel(
  cudf::size_type const* __restrict__ left_indices,
  cudf::size_type const* __restrict__ right_indices,
  cudf::size_type num_pairs,
  cudf::table_device_view left_table,
  cudf::table_device_view right_table,
  cudf::ast::detail::expression_device_view device_expression_data,
  bool* __restrict__ keep_mask)
{
  extern __shared__ char raw_intermediate_storage[];
  auto intermediate_storage =
    reinterpret_cast<intermediate_storage_type<has_nulls>*>(raw_intermediate_storage);
  auto thread_intermediate_storage =
    &intermediate_storage[threadIdx.x * device_expression_data.num_intermediates];

  auto evaluator = cudf::ast::detail::expression_evaluator<has_nulls>(
    left_table, right_table, device_expression_data);

  auto const idx    = cudf::detail::grid_1d::global_thread_id();
  auto const stride = cudf::detail::grid_1d::grid_stride();

  for (cudf::thread_index_type i = idx; i < num_pairs; i += stride) {
    auto output_dest                = cudf::ast::detail::value_expression_result<bool, has_nulls>();
    cudf::size_type const left_idx  = left_indices[i];
    cudf::size_type const right_idx = right_indices[i];

    evaluator.evaluate(output_dest, left_idx, right_idx, 0, thread_intermediate_storage);

    keep_mask[i] = output_dest.is_valid() && output_dest.value();
  }
}

/**
 * @brief Filter join results by evaluating conditional expression
 *
 * This function takes references to the input index vectors (not rvalue references) and uses
 * exec_policy_nosync for all thrust operations. The caller is responsible for synchronizing
 * the stream before using the returned results. This pattern ensures safe memory lifetime
 * management when chaining operations.
 *
 * @note Uses exec_policy_nosync internally - caller must synchronize before using results
 */
template <bool has_nulls>
std::pair<rmm::device_uvector<cudf::size_type>, rmm::device_uvector<cudf::size_type>>
filter_by_conditional_impl(rmm::device_uvector<cudf::size_type>& left_indices,
                           rmm::device_uvector<cudf::size_type>& right_indices,
                           cudf::table_device_view const& left_table,
                           cudf::table_device_view const& right_table,
                           cudf::ast::detail::expression_device_view device_expression_data,
                           cudf::size_type num_intermediates,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  auto const num_pairs = left_indices.size();
  if (num_pairs == 0) {
    return {rmm::device_uvector<cudf::size_type>(0, stream, mr),
            rmm::device_uvector<cudf::size_type>(0, stream, mr)};
  }

  // Create a boolean mask for indices to keep
  auto keep_mask =
    rmm::device_uvector<bool>(num_pairs, stream, cudf::get_current_device_resource_ref());

  // Launch kernel to evaluate expression for each pair with dynamic shared memory sizing
  int current_device = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&current_device));
  int max_shmem_per_block = 0;
  CUDF_CUDA_TRY(cudaDeviceGetAttribute(
    &max_shmem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, current_device));

  // Use the same type alias as in the kernel to ensure consistency
  auto const per_thread_bytes = static_cast<int>(num_intermediates) *
                                static_cast<int>(sizeof(intermediate_storage_type<has_nulls>));
  int block_size = 256;  // default
  if (per_thread_bytes > 0) {
    int const max_by_shmem = max_shmem_per_block / per_thread_bytes;
    if (max_by_shmem > 0) {
      // Prefer multiples of warp size (32)
      int const rounded = (max_by_shmem / 32) * 32;
      block_size        = std::max(32, std::min(block_size, rounded > 0 ? rounded : max_by_shmem));
    } else {
      block_size = 32;  // minimal reasonable size
    }
  }
  cudf::size_type const grid_size = (num_pairs + block_size - 1) / block_size;
  auto const shmem_size = static_cast<size_t>(block_size) * static_cast<size_t>(per_thread_bytes);

  filter_join_indices_kernel<has_nulls>
    <<<grid_size, block_size, shmem_size, stream.value()>>>(left_indices.data(),
                                                            right_indices.data(),
                                                            num_pairs,
                                                            left_table,
                                                            right_table,
                                                            device_expression_data,
                                                            keep_mask.data());

  // Surface any kernel launch errors immediately
  CUDF_CUDA_TRY(cudaPeekAtLastError());

  // Count the number of true values in the mask
  auto const num_matches =
    thrust::count(rmm::exec_policy(stream), keep_mask.begin(), keep_mask.end(), true);

  // Allocate output vectors
  auto out_left_indices  = rmm::device_uvector<cudf::size_type>(num_matches, stream, mr);
  auto out_right_indices = rmm::device_uvector<cudf::size_type>(num_matches, stream, mr);

  // Copy indices where condition is true
  auto input_iter =
    thrust::make_zip_iterator(thrust::make_tuple(left_indices.begin(), right_indices.begin()));
  auto output_iter = thrust::make_zip_iterator(
    thrust::make_tuple(out_left_indices.begin(), out_right_indices.begin()));

  thrust::copy_if(rmm::exec_policy_nosync(stream),
                  input_iter,
                  input_iter + num_pairs,
                  keep_mask.begin(),
                  output_iter,
                  cuda::std::identity{});

  return {std::move(out_left_indices), std::move(out_right_indices)};
}

/**
 * @brief Wrapper to call templated filter function based on has_nulls
 *
 * This function takes references to the input index vectors (not rvalue references) and uses
 * exec_policy_nosync for all thrust operations. The caller is responsible for synchronizing
 * the stream before using the returned results.
 *
 * @note Uses exec_policy_nosync internally - caller must synchronize before using results
 */
std::pair<rmm::device_uvector<cudf::size_type>, rmm::device_uvector<cudf::size_type>>
filter_by_conditional(rmm::device_uvector<cudf::size_type>& left_indices,
                      rmm::device_uvector<cudf::size_type>& right_indices,
                      cudf::table_device_view const& left_table,
                      cudf::table_device_view const& right_table,
                      cudf::ast::detail::expression_device_view device_expression_data,
                      cudf::size_type num_intermediates,
                      bool has_nulls,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr)
{
  if (has_nulls) {
    return filter_by_conditional_impl<true>(left_indices,
                                            right_indices,
                                            left_table,
                                            right_table,
                                            device_expression_data,
                                            num_intermediates,
                                            stream,
                                            mr);
  } else {
    return filter_by_conditional_impl<false>(left_indices,
                                             right_indices,
                                             left_table,
                                             right_table,
                                             device_expression_data,
                                             num_intermediates,
                                             stream,
                                             mr);
  }
}

/**
 * @brief Add back left rows with no matches (for left join)
 *
 * This function takes references to the input index vectors (not rvalue references) and uses
 * exec_policy_nosync for all thrust operations. The caller is responsible for synchronizing
 * the stream before using the returned results. This pattern ensures safe memory lifetime
 * management when the input vectors are allocated in the caller's scope.
 *
 * @note Uses exec_policy_nosync internally - caller must synchronize before using results
 */
std::pair<rmm::device_uvector<cudf::size_type>, rmm::device_uvector<cudf::size_type>>
add_left_unmatched_rows(rmm::device_uvector<cudf::size_type>& left_indices,
                        rmm::device_uvector<cudf::size_type>& right_indices,
                        cudf::size_type left_num_rows,
                        cudf::size_type right_num_rows,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  // Create a boolean mask to track which left rows have matches
  auto left_has_match =
    rmm::device_uvector<bool>(left_num_rows, stream, cudf::get_current_device_resource_ref());
  CUDF_CUDA_TRY(cudaMemsetAsync(left_has_match.data(), 0, left_has_match.size(), stream.value()));

  // Mark left rows that have matches
  thrust::for_each(rmm::exec_policy_nosync(stream),
                   left_indices.begin(),
                   left_indices.end(),
                   [left_has_match = left_has_match.data()] __device__(cudf::size_type idx) {
                     left_has_match[idx] = true;
                   });

  // Count unmatched left rows
  auto const num_unmatched = thrust::count(
    rmm::exec_policy_nosync(stream), left_has_match.begin(), left_has_match.end(), false);

  // Allocate output with space for both matched and unmatched
  auto const total_size  = left_indices.size() + num_unmatched;
  auto out_left_indices  = rmm::device_uvector<cudf::size_type>(total_size, stream, mr);
  auto out_right_indices = rmm::device_uvector<cudf::size_type>(total_size, stream, mr);

  // Copy matched pairs
  thrust::copy(rmm::exec_policy_nosync(stream),
               left_indices.begin(),
               left_indices.end(),
               out_left_indices.begin());
  thrust::copy(rmm::exec_policy_nosync(stream),
               right_indices.begin(),
               right_indices.end(),
               out_right_indices.begin());

  // Add unmatched left rows with null right indices
  auto unmatched_iter = out_left_indices.begin() + left_indices.size();
  thrust::copy_if(rmm::exec_policy_nosync(stream),
                  thrust::counting_iterator<cudf::size_type>(0),
                  thrust::counting_iterator<cudf::size_type>(left_num_rows),
                  left_has_match.begin(),
                  unmatched_iter,
                  cuda::std::logical_not<bool>());

  // Fill corresponding right indices with out-of-bounds values (null markers)
  thrust::fill(rmm::exec_policy_nosync(stream),
               out_right_indices.begin() + left_indices.size(),
               out_right_indices.end(),
               right_num_rows);

  return {std::move(out_left_indices), std::move(out_right_indices)};
}

/**
 * @brief Extract unique left indices for semi/anti joins
 *
 * This function takes a reference to the input index vector (not an rvalue reference) and uses
 * exec_policy_nosync for all thrust operations. The caller is responsible for synchronizing
 * the stream before using the returned result. This pattern ensures safe memory lifetime
 * management when the input vector is allocated in the caller's scope.
 *
 * @note Uses exec_policy_nosync internally - caller must synchronize before using results
 */
rmm::device_uvector<cudf::size_type> extract_unique_left_indices(
  rmm::device_uvector<cudf::size_type>& left_indices,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  if (left_indices.size() == 0) { return rmm::device_uvector<cudf::size_type>(0, stream, mr); }

  // Sort and unique
  thrust::sort(rmm::exec_policy_nosync(stream), left_indices.begin(), left_indices.end());
  auto new_end =
    thrust::unique(rmm::exec_policy_nosync(stream), left_indices.begin(), left_indices.end());
  auto new_size = thrust::distance(left_indices.begin(), new_end);

  auto result = rmm::device_uvector<cudf::size_type>(new_size, stream, mr);
  thrust::copy(rmm::exec_policy_nosync(stream), left_indices.begin(), new_end, result.begin());

  return result;
}

}  // anonymous namespace

// =============================================================================
// EQUALITY-ONLY SORT-MERGE JOIN IMPLEMENTATIONS
// =============================================================================

std::pair<rmm::device_uvector<cudf::size_type>, rmm::device_uvector<cudf::size_type>>
sort_merge_inner_join(cudf::table_view const& left_keys,
                      cudf::table_view const& right_keys,
                      cudf::sorted is_left_sorted,
                      cudf::sorted is_right_sorted,
                      cudf::null_equality compare_nulls,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  // Validate inputs
  CUDF_EXPECTS(left_keys.num_columns() > 0, "Left keys table must have at least one column");
  CUDF_EXPECTS(right_keys.num_columns() > 0, "Right keys table must have at least one column");

  // Handle empty table cases - inner join returns empty if either table is empty
  if (left_keys.num_rows() == 0 || right_keys.num_rows() == 0) {
    return {rmm::device_uvector<cudf::size_type>(0, stream, mr),
            rmm::device_uvector<cudf::size_type>(0, stream, mr)};
  }

  // Perform inner join on equality keys using either sort-merge or hash join
  auto [left_indices, right_indices] = inner_join_impl(left_keys, right_keys, is_left_sorted, is_right_sorted, compare_nulls, stream, mr);

  // Return the raw device_uvectors (RVO will handle the move)
  return {std::move(*left_indices), std::move(*right_indices)};
}

std::pair<rmm::device_uvector<cudf::size_type>, rmm::device_uvector<cudf::size_type>>
sort_merge_left_join(cudf::table_view const& left_keys,
                     cudf::table_view const& right_keys,
                     cudf::sorted is_left_sorted,
                     cudf::sorted is_right_sorted,
                     cudf::null_equality compare_nulls,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  // Validate inputs
  CUDF_EXPECTS(left_keys.num_columns() > 0, "Left keys table must have at least one column");
  CUDF_EXPECTS(right_keys.num_columns() > 0, "Right keys table must have at least one column");

  // Handle empty right table case - all left rows with null right indices
  if (right_keys.num_rows() == 0) {
    auto left_indices  = rmm::device_uvector<cudf::size_type>(left_keys.num_rows(), stream, mr);
    auto right_indices = rmm::device_uvector<cudf::size_type>(left_keys.num_rows(), stream, mr);

    thrust::sequence(rmm::exec_policy_nosync(stream), left_indices.begin(), left_indices.end());
    thrust::fill(rmm::exec_policy_nosync(stream),
                 right_indices.begin(),
                 right_indices.end(),
                 right_keys.num_rows());

    return {std::move(left_indices), std::move(right_indices)};
  }

  // Perform inner join on equality keys (intermediate result) using either sort-merge or hash join
  auto [equality_left_indices, equality_right_indices] =
    inner_join_impl(left_keys, right_keys, is_left_sorted, is_right_sorted, compare_nulls, stream, cudf::get_current_device_resource_ref());

  // Add back left rows with no matches (final result)
  auto result = add_left_unmatched_rows(*equality_left_indices,
                                        *equality_right_indices,
                                        left_keys.num_rows(),
                                        right_keys.num_rows(),
                                        stream,
                                        mr);
  return result;
}

rmm::device_uvector<cudf::size_type> sort_merge_left_semi_join(cudf::table_view const& left_keys,
                                                               cudf::table_view const& right_keys,
                                                               cudf::sorted is_left_sorted,
                                                               cudf::sorted is_right_sorted,
                                                               cudf::null_equality compare_nulls,
                                                               rmm::cuda_stream_view stream,
                                                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  // Validate inputs
  CUDF_EXPECTS(left_keys.num_columns() > 0, "Left keys table must have at least one column");
  CUDF_EXPECTS(right_keys.num_columns() > 0, "Right keys table must have at least one column");

  // Handle empty right table case - no matches
  if (right_keys.num_rows() == 0) { return rmm::device_uvector<cudf::size_type>(0, stream, mr); }

  // Perform inner join on equality keys (intermediate result) using either sort-merge or hash join
  auto [equality_left_indices, equality_right_indices] =
    inner_join_impl(left_keys, right_keys, is_left_sorted, is_right_sorted, compare_nulls, stream, cudf::get_current_device_resource_ref());

  // If no equality matches, return empty result
  if (equality_left_indices->size() == 0) {
    return rmm::device_uvector<cudf::size_type>(0, stream, mr);
  }

  // Extract unique left indices (final result)
  auto result = extract_unique_left_indices(*equality_left_indices, stream, mr);
  return result;
}

rmm::device_uvector<cudf::size_type> sort_merge_left_anti_join(cudf::table_view const& left_keys,
                                                               cudf::table_view const& right_keys,
                                                               cudf::sorted is_left_sorted,
                                                               cudf::sorted is_right_sorted,
                                                               cudf::null_equality compare_nulls,
                                                               rmm::cuda_stream_view stream,
                                                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  // Validate inputs
  CUDF_EXPECTS(left_keys.num_columns() > 0, "Left keys table must have at least one column");
  CUDF_EXPECTS(right_keys.num_columns() > 0, "Right keys table must have at least one column");

  auto const left_num_rows = left_keys.num_rows();

  // Handle empty right table case - all left rows are anti-join matches
  if (right_keys.num_rows() == 0) {
    auto result = rmm::device_uvector<cudf::size_type>(left_num_rows, stream, mr);
    thrust::sequence(rmm::exec_policy_nosync(stream), result.begin(), result.end());
    return result;
  }

  // Get left semi join result (rows that DO match) - intermediate
  auto semi_indices = sort_merge_left_semi_join(left_keys,
                                                right_keys,
                                                is_left_sorted,
                                                is_right_sorted,
                                                compare_nulls,
                                                stream,
                                                cudf::get_current_device_resource_ref());

  // Find complement - rows that DON'T match (intermediate)
  auto left_has_match =
    rmm::device_uvector<bool>(left_num_rows, stream, cudf::get_current_device_resource_ref());
  thrust::fill(
    rmm::exec_policy_nosync(stream), left_has_match.begin(), left_has_match.end(), false);

  // Mark left rows that have matches
  thrust::for_each(rmm::exec_policy_nosync(stream),
                   semi_indices.begin(),
                   semi_indices.end(),
                   [left_has_match = left_has_match.data()] __device__(cudf::size_type idx) {
                     left_has_match[idx] = true;
                   });

  // Count and collect unmatched rows
  auto const num_unmatched = thrust::count(
    rmm::exec_policy_nosync(stream), left_has_match.begin(), left_has_match.end(), false);

  auto result = rmm::device_uvector<cudf::size_type>(num_unmatched, stream, mr);
  thrust::copy_if(rmm::exec_policy_nosync(stream),
                  thrust::counting_iterator<cudf::size_type>(0),
                  thrust::counting_iterator<cudf::size_type>(left_num_rows),
                  left_has_match.begin(),
                  result.begin(),
                  cuda::std::logical_not<bool>());

  return result;
}

// =============================================================================
// CONDITIONAL (MIXED) SORT-MERGE JOIN IMPLEMENTATIONS
// =============================================================================

std::pair<rmm::device_uvector<cudf::size_type>, rmm::device_uvector<cudf::size_type>>
mixed_sort_merge_inner_join(cudf::table_view const& left_equality,
                            cudf::table_view const& right_equality,
                            cudf::table_view const& left_conditional,
                            cudf::table_view const& right_conditional,
                            cudf::ast::expression const& binary_predicate,
                            cudf::sorted is_left_sorted,
                            cudf::sorted is_right_sorted,
                            cudf::null_equality compare_nulls,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  // Validate inputs - only equality tables need columns, conditionals can be empty if expression
  // uses only literals
  CUDF_EXPECTS(left_equality.num_columns() > 0,
               "Left equality table must have at least one column");
  CUDF_EXPECTS(right_equality.num_columns() > 0,
               "Right equality table must have at least one column");

  // Only validate row counts if conditional tables have columns
  // Empty conditional tables (for literal-only expressions) don't have meaningful row counts
  if (left_conditional.num_columns() > 0) {
    CUDF_EXPECTS(left_equality.num_rows() == left_conditional.num_rows(),
                 "Left equality and conditional tables must have the same number of rows");
  }
  if (right_conditional.num_columns() > 0) {
    CUDF_EXPECTS(right_equality.num_rows() == right_conditional.num_rows(),
                 "Right equality and conditional tables must have the same number of rows");
  }

  // Perform conditional inner join (uses either built-in or two-step approach)
  return conditional_inner_join_impl(left_equality,
                                     right_equality,
                                     left_conditional,
                                     right_conditional,
                                     binary_predicate,
                                     is_left_sorted,
                                     is_right_sorted,
                                     compare_nulls,
                                     stream,
                                     mr);
}

std::pair<rmm::device_uvector<cudf::size_type>, rmm::device_uvector<cudf::size_type>>
mixed_sort_merge_left_join(cudf::table_view const& left_equality,
                           cudf::table_view const& right_equality,
                           cudf::table_view const& left_conditional,
                           cudf::table_view const& right_conditional,
                           cudf::ast::expression const& binary_predicate,
                           cudf::sorted is_left_sorted,
                           cudf::sorted is_right_sorted,
                           cudf::null_equality compare_nulls,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  // Validate inputs - only equality tables need columns, conditionals can be empty if expression
  // uses only literals
  CUDF_EXPECTS(left_equality.num_columns() > 0,
               "Left equality table must have at least one column");
  CUDF_EXPECTS(right_equality.num_columns() > 0,
               "Right equality table must have at least one column");

  // Only validate row counts if conditional tables have columns
  // Empty conditional tables (for literal-only expressions) don't have meaningful row counts
  if (left_conditional.num_columns() > 0) {
    CUDF_EXPECTS(left_equality.num_rows() == left_conditional.num_rows(),
                 "Left equality and conditional tables must have the same number of rows");
  }
  if (right_conditional.num_columns() > 0) {
    CUDF_EXPECTS(right_equality.num_rows() == right_conditional.num_rows(),
                 "Right equality and conditional tables must have the same number of rows");
  }

  // Handle empty right table case
  if (right_equality.num_rows() == 0) {
    auto left_indices  = rmm::device_uvector<cudf::size_type>(left_equality.num_rows(), stream, mr);
    auto right_indices = rmm::device_uvector<cudf::size_type>(left_equality.num_rows(), stream, mr);

    thrust::sequence(rmm::exec_policy_nosync(stream), left_indices.begin(), left_indices.end());
    thrust::fill(rmm::exec_policy_nosync(stream),
                 right_indices.begin(),
                 right_indices.end(),
                 right_equality.num_rows());

    return {std::move(left_indices), std::move(right_indices)};
  }

  // Step 1: Perform conditional inner join (uses either built-in or two-step approach)
  auto [filtered_left_indices, filtered_right_indices] =
    conditional_inner_join_impl(left_equality,
                                right_equality,
                                left_conditional,
                                right_conditional,
                                binary_predicate,
                                is_left_sorted,
                                is_right_sorted,
                                compare_nulls,
                                stream,
                                cudf::get_current_device_resource_ref());

  // Step 2: Add back left rows with no matches (final result)
  auto result = add_left_unmatched_rows(filtered_left_indices,
                                        filtered_right_indices,
                                        left_equality.num_rows(),
                                        right_equality.num_rows(),
                                        stream,
                                        mr);
  return result;
}

rmm::device_uvector<cudf::size_type> mixed_sort_merge_left_semi_join(
  cudf::table_view const& left_equality,
  cudf::table_view const& right_equality,
  cudf::table_view const& left_conditional,
  cudf::table_view const& right_conditional,
  cudf::ast::expression const& binary_predicate,
  cudf::sorted is_left_sorted,
  cudf::sorted is_right_sorted,
  cudf::null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  // Validate inputs - only equality tables need columns, conditionals can be empty if expression
  // uses only literals
  CUDF_EXPECTS(left_equality.num_columns() > 0,
               "Left equality table must have at least one column");
  CUDF_EXPECTS(right_equality.num_columns() > 0,
               "Right equality table must have at least one column");

  // Only validate row counts if conditional tables have columns
  // Empty conditional tables (for literal-only expressions) don't have meaningful row counts
  if (left_conditional.num_columns() > 0) {
    CUDF_EXPECTS(left_equality.num_rows() == left_conditional.num_rows(),
                 "Left equality and conditional tables must have the same number of rows");
  }
  if (right_conditional.num_columns() > 0) {
    CUDF_EXPECTS(right_equality.num_rows() == right_conditional.num_rows(),
                 "Right equality and conditional tables must have the same number of rows");
  }

  // Handle empty right table case
  if (right_equality.num_rows() == 0) {
    return rmm::device_uvector<cudf::size_type>(0, stream, mr);
  }

  // Step 1: Perform conditional inner join (uses either built-in or two-step approach)
  auto [filtered_left_indices, filtered_right_indices] =
    conditional_inner_join_impl(left_equality,
                                right_equality,
                                left_conditional,
                                right_conditional,
                                binary_predicate,
                                is_left_sorted,
                                is_right_sorted,
                                compare_nulls,
                                stream,
                                cudf::get_current_device_resource_ref());

  // Step 2: Extract unique left indices (final result)
  auto result = extract_unique_left_indices(filtered_left_indices, stream, mr);
  return result;
}

rmm::device_uvector<cudf::size_type> mixed_sort_merge_left_anti_join(
  cudf::table_view const& left_equality,
  cudf::table_view const& right_equality,
  cudf::table_view const& left_conditional,
  cudf::table_view const& right_conditional,
  cudf::ast::expression const& binary_predicate,
  cudf::sorted is_left_sorted,
  cudf::sorted is_right_sorted,
  cudf::null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  // Validate inputs - only equality tables need columns, conditionals can be empty if expression
  // uses only literals
  CUDF_EXPECTS(left_equality.num_columns() > 0,
               "Left equality table must have at least one column");
  CUDF_EXPECTS(right_equality.num_columns() > 0,
               "Right equality table must have at least one column");

  // Only validate row counts if conditional tables have columns
  // Empty conditional tables (for literal-only expressions) don't have meaningful row counts
  if (left_conditional.num_columns() > 0) {
    CUDF_EXPECTS(left_equality.num_rows() == left_conditional.num_rows(),
                 "Left equality and conditional tables must have the same number of rows");
  }
  if (right_conditional.num_columns() > 0) {
    CUDF_EXPECTS(right_equality.num_rows() == right_conditional.num_rows(),
                 "Right equality and conditional tables must have the same number of rows");
  }

  auto const left_num_rows = left_equality.num_rows();

  // Handle empty right table case - all left rows are anti-join matches
  if (right_equality.num_rows() == 0) {
    auto result = rmm::device_uvector<cudf::size_type>(left_num_rows, stream, mr);
    thrust::sequence(rmm::exec_policy_nosync(stream), result.begin(), result.end());
    return result;
  }

  // Step 1: Get left semi join result (rows that DO match) - intermediate
  auto semi_indices = mixed_sort_merge_left_semi_join(left_equality,
                                                      right_equality,
                                                      left_conditional,
                                                      right_conditional,
                                                      binary_predicate,
                                                      is_left_sorted,
                                                      is_right_sorted,
                                                      compare_nulls,
                                                      stream,
                                                      cudf::get_current_device_resource_ref());

  // Step 2: Find complement - rows that DON'T match (intermediate)
  auto left_has_match =
    rmm::device_uvector<bool>(left_num_rows, stream, cudf::get_current_device_resource_ref());
  thrust::fill(
    rmm::exec_policy_nosync(stream), left_has_match.begin(), left_has_match.end(), false);

  // Mark left rows that have matches
  thrust::for_each(rmm::exec_policy_nosync(stream),
                   semi_indices.begin(),
                   semi_indices.end(),
                   [left_has_match = left_has_match.data()] __device__(cudf::size_type idx) {
                     left_has_match[idx] = true;
                   });

  // Count and collect unmatched rows
  auto const num_unmatched = thrust::count(
    rmm::exec_policy_nosync(stream), left_has_match.begin(), left_has_match.end(), false);

  auto result = rmm::device_uvector<cudf::size_type>(num_unmatched, stream, mr);
  thrust::copy_if(rmm::exec_policy_nosync(stream),
                  thrust::counting_iterator<cudf::size_type>(0),
                  thrust::counting_iterator<cudf::size_type>(left_num_rows),
                  left_has_match.begin(),
                  result.begin(),
                  thrust::logical_not<bool>());

  return result;
}

}  // namespace spark_rapids_jni
