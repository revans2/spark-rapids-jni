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

#include "cudf_jni_apis.hpp"

#include <cudf/join/sort_merge_join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/device_uvector.hpp>

namespace {

/**
 * @brief Convert pair of device vectors to Java long array
 * Returns a 5-element array: [size_in_bytes, first_ptr, first_handle, second_ptr, second_handle]
 */
jlongArray gather_maps_to_java(
  JNIEnv* env,
  std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
            std::unique_ptr<rmm::device_uvector<cudf::size_type>>> gather_maps)
{
  // Both gather maps must have the same size for paired results
  CUDF_EXPECTS(gather_maps.first->size() == gather_maps.second->size(),
               "Gather maps must have the same size");

  // Release the underlying device buffers to Java
  auto first_map_buffer  = std::make_unique<rmm::device_buffer>(gather_maps.first->release());
  auto second_map_buffer = std::make_unique<rmm::device_buffer>(gather_maps.second->release());

  cudf::jni::native_jlongArray result(env, 5);
  // Return size in bytes (as expected by DeviceMemoryBuffer.fromRmm)
  result[0] = static_cast<jlong>(first_map_buffer->size());
  result[1] = cudf::jni::ptr_as_jlong(first_map_buffer->data());
  result[2] = cudf::jni::release_as_jlong(first_map_buffer);
  result[3] = cudf::jni::ptr_as_jlong(second_map_buffer->data());
  result[4] = cudf::jni::release_as_jlong(second_map_buffer);
  return result.get_jArray();
}

}  // anonymous namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_SortMergeJoin_createNative(
  JNIEnv* env, jclass, jlong j_build_keys, jboolean j_is_build_sorted, jboolean j_nulls_equal)
{
  JNI_NULL_CHECK(env, j_build_keys, "build keys table is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const build_keys = reinterpret_cast<cudf::table_view const*>(j_build_keys);
    auto const is_build_sorted =
      j_is_build_sorted ? cudf::sorted::YES : cudf::sorted::NO;
    auto const nulls_equal =
      j_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;

    // Create the sort_merge_join object
    // Note: We use new here because sort_merge_join is not copyable/movable
    // CUDF's sort_merge_join constructor takes the "right" table, which is our build table
    auto sort_merge_join_ptr =
      new cudf::sort_merge_join(*build_keys, is_build_sorted, nulls_equal);

    return cudf::jni::ptr_as_jlong(sort_merge_join_ptr);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_SortMergeJoin_innerJoinNative(
  JNIEnv* env, jclass, jlong j_handle, jlong j_probe_keys, jboolean j_is_probe_sorted)
{
  JNI_NULL_CHECK(env, j_handle, "sort merge join handle is null", nullptr);
  JNI_NULL_CHECK(env, j_probe_keys, "probe keys table is null", nullptr);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const sort_merge_join_ptr = reinterpret_cast<cudf::sort_merge_join*>(j_handle);
    auto const probe_keys           = reinterpret_cast<cudf::table_view const*>(j_probe_keys);
    auto const is_probe_sorted = j_is_probe_sorted ? cudf::sorted::YES : cudf::sorted::NO;

    // Perform inner join
    // CUDF's inner_join takes the "left" table, which is our probe table
    // CUDF returns [left_indices, right_indices] = [probe_indices, build_indices]
    // We need to swap to [build_indices, probe_indices]
    auto [probe_indices, build_indices] = sort_merge_join_ptr->inner_join(*probe_keys, is_probe_sorted);

    // Swap to get [build, probe] order
    return gather_maps_to_java(env, {std::move(build_indices), std::move(probe_indices)});
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_SortMergeJoin_closeNative(JNIEnv* env,
                                                                                   jclass,
                                                                                   jlong j_handle)
{
  JNI_TRY { delete reinterpret_cast<cudf::sort_merge_join*>(j_handle); }
  JNI_CATCH(env, );
}

}  // extern "C"

