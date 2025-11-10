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
 * @brief Build the hash map from input keys and return the distinct count
 */
template <bool has_nested>
std::pair<std::unique_ptr<hash_map_holder>, cudf::size_type> build_map_impl(
  cudf::table_view const& input,
  cudf::detail::row::hash::row_hasher const& row_hash,
  cudf::detail::row::equality::self_comparator const& row_equal,
  bool has_nulls,
  cudf::null_equality nulls_equal,
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

  rmm::device_scalar<cudf::size_type> distinct_counter(0, stream, mr);

  auto map_ref = map.ref(cuco::op::insert_and_find);
  thrust::for_each(
    rmm::exec_policy_nosync(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(num_rows),
    [map_ref, d_hasher, counter_ptr = distinct_counter.data()] __device__(cudf::size_type idx) mutable {
      auto const row_hash = d_hasher(idx);
      using rhs_type      = cudf::detail::row::rhs_index_type;
      auto [iter, inserted] =
        map_ref.insert_and_find(cuco::pair{key_type{row_hash, rhs_type{idx}}, idx});

      if (inserted) {
        iter->second = idx;
        cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device>{*counter_ptr}.fetch_add(
          1, cuda::memory_order_relaxed);
      }
    });

  stream.synchronize();
  auto const distinct_count = distinct_counter.value(stream);

  auto holder =
    std::make_unique<typed_hash_map_holder<has_nested>>(std::move(map));

  return {std::move(holder), distinct_count};
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

  // Create output column
  auto output = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, num_probe_rows, cudf::mask_state::UNALLOCATED, stream, mr);

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

// No longer needed - we don't use comparator during build

}  // anonymous namespace

// Destructor implementation
key_remap_build_result::~key_remap_build_result()
{
  if (hash_map_ptr) { free_key_remap_map(hash_map_ptr); }
}

std::unique_ptr<key_remap_build_result> build_key_remap_map(cudf::table_view const& input_keys,
                                                             cudf::null_equality nulls_equal,
                                                             rmm::cuda_stream_view stream,
                                                             rmm::device_async_resource_ref mr)
{
  auto const has_nested_columns = cudf::detail::has_nested_columns(input_keys);

  if (input_keys.num_rows() == 0 || input_keys.num_columns() == 0) {
    auto result            = std::make_unique<key_remap_build_result>();
    result->hash_map_ptr   = nullptr;
    result->distinct_count = 0;
    result->nulls_equal    = nulls_equal;
    result->has_nested_columns = has_nested_columns;
    return result;
  }

  // Preprocess the input for hashing
  auto const preprocessed_input =
    cudf::detail::row::hash::preprocessed_table::create(input_keys, stream);
  auto const has_nulls =
    cudf::has_nulls(input_keys) || cudf::has_nested_nulls(input_keys);

  auto const row_hash = cudf::detail::row::hash::row_hasher(preprocessed_input);

  // Preprocess input for equality comparisons (row equality)
  auto const preprocessed_equal =
    cudf::detail::row::equality::preprocessed_table::create(input_keys, stream);
  auto const self_equal =
    cudf::detail::row::equality::self_comparator(preprocessed_equal);

  std::unique_ptr<hash_map_holder> map_holder;
  cudf::size_type distinct_count = 0;

  if (has_nested_columns) {
    std::tie(map_holder, distinct_count) = build_map_impl<true>(
      input_keys, row_hash, self_equal, has_nulls, nulls_equal, stream, mr);
  } else {
    std::tie(map_holder, distinct_count) = build_map_impl<false>(
      input_keys, row_hash, self_equal, has_nulls, nulls_equal, stream, mr);
  }

  // Create the result
  auto result            = std::make_unique<key_remap_build_result>();
  result->hash_map_ptr   = map_holder.release();
  result->distinct_count = distinct_count;
  result->nulls_equal    = nulls_equal;
  result->has_nested_columns = has_nested_columns;

  return result;
}

std::unique_ptr<cudf::column> apply_key_remap(cudf::table_view const& build_keys,
                                               cudf::table_view const& input_keys,
                                               key_remap_build_result const& remap_result,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  if (input_keys.num_rows() == 0 || input_keys.num_columns() == 0) {
    return cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32}, 0, cudf::mask_state::UNALLOCATED, stream, mr);
  }

  // Preprocess both tables for two-table comparison
  auto const preprocessed_build =
    cudf::detail::row::equality::preprocessed_table::create(build_keys, stream);
  auto const preprocessed_probe =
    cudf::detail::row::equality::preprocessed_table::create(input_keys, stream);
    
  auto const has_nulls =
    cudf::has_nulls(build_keys) || cudf::has_nested_nulls(build_keys) || cudf::has_nulls(input_keys) ||
    cudf::has_nested_nulls(input_keys);
  auto const use_nested_comparator = remap_result.has_nested_columns;

  // Create two-table comparator (probe -> build)
  auto const two_table_equal =
    cudf::detail::row::equality::two_table_comparator(preprocessed_probe, preprocessed_build);

  // Create probe hasher
  auto const probe_hasher = cudf::detail::row::hash::row_hasher(preprocessed_probe);

  // Call lookup with the appropriate comparator
  if (use_nested_comparator) {
    auto const device_comparator =
      two_table_equal.equal_to<true>(cudf::nullate::DYNAMIC{has_nulls}, remap_result.nulls_equal);
    auto const* holder = static_cast<typed_hash_map_holder<true>*>(remap_result.hash_map_ptr);
    return lookup_keys_two_table(
      input_keys, holder->map, probe_hasher, has_nulls, device_comparator, stream, mr);
  } else {
    auto const device_comparator =
      two_table_equal.equal_to<false>(cudf::nullate::DYNAMIC{has_nulls}, remap_result.nulls_equal);
    auto const* holder = static_cast<typed_hash_map_holder<false>*>(remap_result.hash_map_ptr);
    return lookup_keys_two_table(
      input_keys, holder->map, probe_hasher, has_nulls, device_comparator, stream, mr);
  }
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

}  // namespace spark_rapids_jni

