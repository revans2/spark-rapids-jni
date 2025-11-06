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

#include <cudf/join/filtered_join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/device_uvector.hpp>

namespace {

/**
 * @brief Convert single device vector to Java long array
 * Returns a 3-element array: [size_in_bytes, ptr, handle]
 */
jlongArray gather_single_map_to_java(JNIEnv* env,
                                      std::unique_ptr<rmm::device_uvector<cudf::size_type>> map)
{
  auto map_buffer = std::make_unique<rmm::device_buffer>(map->release());

  cudf::jni::native_jlongArray result(env, 3);
  result[0] = static_cast<jlong>(map_buffer->size());
  result[1] = cudf::jni::ptr_as_jlong(map_buffer->data());
  result[2] = cudf::jni::release_as_jlong(map_buffer);
  return result.get_jArray();
}

}  // anonymous namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_FilteredJoin_createNative(
  JNIEnv* env, jclass, jlong j_build_keys, jboolean j_nulls_equal)
{
  JNI_NULL_CHECK(env, j_build_keys, "build keys table is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const build_keys = reinterpret_cast<cudf::table_view const*>(j_build_keys);
    auto const nulls_equal =
      j_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;
    // Always build on the right table (which is the only currently supported option)
    // In our terminology: build table = right table for filtered join
    auto const reuse_tbl = cudf::set_as_build_table::RIGHT;

    // Create the filtered_join object
    // Note: We use new here because filtered_join is not copyable/movable
    auto filtered_join_ptr =
      new cudf::filtered_join(*build_keys, nulls_equal, reuse_tbl, cudf::get_default_stream());

    return cudf::jni::ptr_as_jlong(filtered_join_ptr);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_FilteredJoin_semiJoinNative(
  JNIEnv* env, jclass, jlong j_handle, jlong j_probe_keys)
{
  JNI_NULL_CHECK(env, j_handle, "filtered join handle is null", nullptr);
  JNI_NULL_CHECK(env, j_probe_keys, "probe keys table is null", nullptr);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const filtered_join_ptr = reinterpret_cast<cudf::filtered_join*>(j_handle);
    auto const probe_keys        = reinterpret_cast<cudf::table_view const*>(j_probe_keys);

    // Perform semi join - returns indices from the probe-side (left) table
    auto result = filtered_join_ptr->semi_join(*probe_keys);

    return gather_single_map_to_java(env, std::move(result));
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_FilteredJoin_antiJoinNative(
  JNIEnv* env, jclass, jlong j_handle, jlong j_probe_keys)
{
  JNI_NULL_CHECK(env, j_handle, "filtered join handle is null", nullptr);
  JNI_NULL_CHECK(env, j_probe_keys, "probe keys table is null", nullptr);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const filtered_join_ptr = reinterpret_cast<cudf::filtered_join*>(j_handle);
    auto const probe_keys        = reinterpret_cast<cudf::table_view const*>(j_probe_keys);

    // Perform anti join - returns indices from the probe-side (left) table
    auto result = filtered_join_ptr->anti_join(*probe_keys);

    return gather_single_map_to_java(env, std::move(result));
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_FilteredJoin_closeNative(JNIEnv* env,
                                                                                  jclass,
                                                                                  jlong j_handle)
{
  JNI_TRY { delete reinterpret_cast<cudf::filtered_join*>(j_handle); }
  JNI_CATCH(env, );
}

}  // extern "C"

