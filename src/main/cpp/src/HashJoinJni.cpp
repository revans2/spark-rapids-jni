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

#include <cudf/join/hash_join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/device_uvector.hpp>

namespace {

/**
 * @brief Convert pair of device vectors to Java long array
 * Returns a 5-element array: [size_in_bytes, left_ptr, left_handle, right_ptr, right_handle]
 */
jlongArray gather_maps_to_java(
  JNIEnv* env,
  std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
            std::unique_ptr<rmm::device_uvector<cudf::size_type>>> gather_maps)
{
  // Both gather maps must have the same size for paired results
  CUDF_EXPECTS(gather_maps.first->size() == gather_maps.second->size(),
               "Left and right gather maps must have the same size");

  // Release the underlying device buffers to Java
  auto left_map_buffer  = std::make_unique<rmm::device_buffer>(gather_maps.first->release());
  auto right_map_buffer = std::make_unique<rmm::device_buffer>(gather_maps.second->release());

  cudf::jni::native_jlongArray result(env, 5);
  // Return size in bytes (as expected by DeviceMemoryBuffer.fromRmm)
  result[0] = static_cast<jlong>(left_map_buffer->size());
  result[1] = cudf::jni::ptr_as_jlong(left_map_buffer->data());
  result[2] = cudf::jni::release_as_jlong(left_map_buffer);
  result[3] = cudf::jni::ptr_as_jlong(right_map_buffer->data());
  result[4] = cudf::jni::release_as_jlong(right_map_buffer);
  return result.get_jArray();
}

}  // anonymous namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_HashJoin_createNative(
  JNIEnv* env, jclass, jlong j_build_keys, jboolean j_nulls_equal)
{
  JNI_NULL_CHECK(env, j_build_keys, "build keys table is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const build_keys = reinterpret_cast<cudf::table_view const*>(j_build_keys);
    auto const nulls_equal =
      j_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;

    // Create the hash_join object
    // Note: We use new here because hash_join is not copyable/movable
    auto hash_join_ptr =
      new cudf::hash_join(*build_keys, nulls_equal, cudf::get_default_stream());

    return cudf::jni::ptr_as_jlong(hash_join_ptr);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_HashJoin_innerJoinNative(
  JNIEnv* env, jclass, jlong j_handle, jlong j_probe_keys)
{
  JNI_NULL_CHECK(env, j_handle, "hash join handle is null", nullptr);
  JNI_NULL_CHECK(env, j_probe_keys, "probe keys table is null", nullptr);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const hash_join_ptr = reinterpret_cast<cudf::hash_join*>(j_handle);
    auto const probe_keys    = reinterpret_cast<cudf::table_view const*>(j_probe_keys);

    // Perform inner join
    // cudf returns [probe_indices, build_indices], but we want [build_indices, probe_indices]
    // to match the semantic order of our API (build passed to constructor, probe to method)
    auto [probe_indices, build_indices] = hash_join_ptr->inner_join(*probe_keys);
    
    // Swap to get [build, probe] order
    return gather_maps_to_java(env, {std::move(build_indices), std::move(probe_indices)});
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_HashJoin_leftJoinNative(
  JNIEnv* env, jclass, jlong j_handle, jlong j_probe_keys)
{
  JNI_NULL_CHECK(env, j_handle, "hash join handle is null", nullptr);
  JNI_NULL_CHECK(env, j_probe_keys, "probe keys table is null", nullptr);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const hash_join_ptr = reinterpret_cast<cudf::hash_join*>(j_handle);
    auto const probe_keys    = reinterpret_cast<cudf::table_view const*>(j_probe_keys);

    // Perform left join
    // cudf returns [probe_indices, build_indices], but we want [build_indices, probe_indices]
    auto [probe_indices, build_indices] = hash_join_ptr->left_join(*probe_keys);
    
    // Swap to get [build, probe] order
    return gather_maps_to_java(env, {std::move(build_indices), std::move(probe_indices)});
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_HashJoin_fullJoinNative(
  JNIEnv* env, jclass, jlong j_handle, jlong j_probe_keys)
{
  JNI_NULL_CHECK(env, j_handle, "hash join handle is null", nullptr);
  JNI_NULL_CHECK(env, j_probe_keys, "probe keys table is null", nullptr);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const hash_join_ptr = reinterpret_cast<cudf::hash_join*>(j_handle);
    auto const probe_keys    = reinterpret_cast<cudf::table_view const*>(j_probe_keys);

    // Perform full join
    // cudf returns [probe_indices, build_indices], but we want [build_indices, probe_indices]
    auto [probe_indices, build_indices] = hash_join_ptr->full_join(*probe_keys);
    
    // Swap to get [build, probe] order
    return gather_maps_to_java(env, {std::move(build_indices), std::move(probe_indices)});
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_HashJoin_innerJoinSizeNative(
  JNIEnv* env, jclass, jlong j_handle, jlong j_probe_keys)
{
  JNI_NULL_CHECK(env, j_handle, "hash join handle is null", 0);
  JNI_NULL_CHECK(env, j_probe_keys, "probe keys table is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const hash_join_ptr = reinterpret_cast<cudf::hash_join*>(j_handle);
    auto const probe_keys    = reinterpret_cast<cudf::table_view const*>(j_probe_keys);

    return static_cast<jlong>(hash_join_ptr->inner_join_size(*probe_keys));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_HashJoin_leftJoinSizeNative(
  JNIEnv* env, jclass, jlong j_handle, jlong j_probe_keys)
{
  JNI_NULL_CHECK(env, j_handle, "hash join handle is null", 0);
  JNI_NULL_CHECK(env, j_probe_keys, "probe keys table is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const hash_join_ptr = reinterpret_cast<cudf::hash_join*>(j_handle);
    auto const probe_keys    = reinterpret_cast<cudf::table_view const*>(j_probe_keys);

    return static_cast<jlong>(hash_join_ptr->left_join_size(*probe_keys));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_HashJoin_fullJoinSizeNative(
  JNIEnv* env, jclass, jlong j_handle, jlong j_probe_keys)
{
  JNI_NULL_CHECK(env, j_handle, "hash join handle is null", 0);
  JNI_NULL_CHECK(env, j_probe_keys, "probe keys table is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const hash_join_ptr = reinterpret_cast<cudf::hash_join*>(j_handle);
    auto const probe_keys    = reinterpret_cast<cudf::table_view const*>(j_probe_keys);

    return static_cast<jlong>(hash_join_ptr->full_join_size(*probe_keys));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_HashJoin_closeNative(JNIEnv* env,
                                                                              jclass,
                                                                              jlong j_handle)
{
  JNI_TRY { delete reinterpret_cast<cudf::hash_join*>(j_handle); }
  JNI_CATCH(env, );
}

}  // extern "C"

