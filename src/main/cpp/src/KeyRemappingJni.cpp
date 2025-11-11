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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/detail/cuco_helpers.hpp>

#include <jni.h>

// Include CUDF JNI utilities from thirdparty
// These provide helper macros and functions for JNI
#include "../../../thirdparty/cudf/java/src/main/native/src/cudf_jni_apis.hpp"

extern "C" {

/**
 * @brief Build a key remapping structure from input keys.
 *
 * Java signature:
 * private static native long buildKeyRemapNative(long inputKeysHandle, int nullEqualityMode);
 *
 * @param env JNI environment
 * @param clazz Java class
 * @param j_input_keys Handle to the input keys table
 * @param j_null_equality_mode Null equality mode (0=NULL_EQUAL, 1=NULL_NOT_EQUAL, 2=SPARK_EQUALITY)
 * @return Handle to the key_remap_build_result
 */
JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_KeyRemapping_buildKeyRemapNative(JNIEnv* env,
                                                                   jclass clazz,
                                                                   jlong j_input_keys,
                                                                   jint j_null_equality_mode)
{
  JNI_NULL_CHECK(env, j_input_keys, "input keys table is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const input_keys = reinterpret_cast<cudf::table_view const*>(j_input_keys);
    auto const null_mode = static_cast<spark_rapids_jni::null_equality_mode>(j_null_equality_mode);

    auto result = spark_rapids_jni::build_key_remap_map(*input_keys, null_mode);
    return cudf::jni::ptr_as_jlong(result.release());
  }
  JNI_CATCH(env, 0);
}

/**
 * @brief Get the sentinel value used for unmatched probe keys.
 *
 * Java signature:
 * private static native int getNotFoundSentinelNative();
 */
JNIEXPORT jint JNICALL
Java_com_nvidia_spark_rapids_jni_KeyRemapping_getNotFoundSentinelNative(JNIEnv* env, jclass clazz)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    return static_cast<jint>(spark_rapids_jni::NOT_FOUND_SENTINEL);
  }
  JNI_CATCH(env, 0);
}

/**
 * @brief Get the sentinel value used for build-side rows with top-level null keys.
 *
 * Java signature:
 * private static native int getBuildNullSentinelNative();
 */
JNIEXPORT jint JNICALL
Java_com_nvidia_spark_rapids_jni_KeyRemapping_getBuildNullSentinelNative(JNIEnv* env, jclass clazz)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    return static_cast<jint>(spark_rapids_jni::BUILD_NULL_SENTINEL);
  }
  JNI_CATCH(env, 0);
}

/**
 * @brief Apply key remapping to input keys.
 *
 * Java signature:
 * private static native long applyKeyRemapNative(long buildKeysHandle, long inputKeysHandle,
 *                                                long remapHandle, int nullEqualityMode, boolean isBuildSide);
 *
 * @param env JNI environment
 * @param clazz Java class
 * @param j_build_keys Handle to the build keys table (original keys used to build the map)
 * @param j_input_keys Handle to the input keys table (keys to remap)
 * @param j_remap_handle Handle to the key_remap_build_result
 * @param j_null_equality_mode Null equality mode (must match mode used during build)
 * @param j_is_build_side Whether this is remapping the build side (true) or probe side (false)
 * @return Column handle for the remapped integer keys
 */
JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_KeyRemapping_applyKeyRemapNative(JNIEnv* env,
                                                                   jclass clazz,
                                                                   jlong j_build_keys,
                                                                   jlong j_input_keys,
                                                                   jlong j_remap_handle,
                                                                   jint j_null_equality_mode,
                                                                   jboolean j_is_build_side)
{
  JNI_NULL_CHECK(env, j_build_keys, "build keys table is null", 0);
  JNI_NULL_CHECK(env, j_input_keys, "input keys table is null", 0);
  JNI_NULL_CHECK(env, j_remap_handle, "remap handle is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const build_keys = reinterpret_cast<cudf::table_view const*>(j_build_keys);
    auto const input_keys = reinterpret_cast<cudf::table_view const*>(j_input_keys);
    auto const* result    = reinterpret_cast<spark_rapids_jni::key_remap_build_result*>(j_remap_handle);
    auto const null_mode = static_cast<spark_rapids_jni::null_equality_mode>(j_null_equality_mode);
    auto const is_build_side = static_cast<bool>(j_is_build_side);

    auto remapped_column = spark_rapids_jni::apply_key_remap(
      *build_keys, *input_keys, *result, null_mode, is_build_side);

    return cudf::jni::release_as_jlong(remapped_column);
  }
  JNI_CATCH(env, 0);
}

/**
 * @brief Free the key remap structure.
 *
 * Java signature:
 * private static native void freeKeyRemapNative(long remapHandle);
 *
 * @param env JNI environment
 * @param clazz Java class
 * @param j_remap_handle Handle to the key_remap_build_result
 */
JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_KeyRemapping_freeKeyRemapNative(
  JNIEnv* env, jclass clazz, jlong j_remap_handle)
{
  JNI_TRY
  {
    delete reinterpret_cast<spark_rapids_jni::key_remap_build_result*>(j_remap_handle);
  }
  JNI_CATCH(env, );
}

/**
 * @brief Dump the raw contents of the remapping hash map for debugging.
 *
 * Returns three columns: hash values (UINT32), key row indices (INT32), and mapped values (INT32).
 *
 * Java signature:
 * private static native long[] dumpRemapTableNative(long buildKeysHandle, long remapHandle);
 *
 * @param env JNI environment
 * @param clazz Java class
 * @param j_build_keys Handle to the build keys table (unused, kept for API consistency)
 * @param j_remap_handle Handle to the key_remap_build_result
 * @return Array of column handles containing all entries in the remapping hash map
 */
JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_KeyRemapping_dumpRemapTableNative(JNIEnv* env,
                                                                     jclass clazz,
                                                                     jlong j_build_keys,
                                                                     jlong j_remap_handle)
{
  JNI_NULL_CHECK(env, j_build_keys, "build keys table is null", 0);
  JNI_NULL_CHECK(env, j_remap_handle, "remap handle is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const build_keys = reinterpret_cast<cudf::table_view const*>(j_build_keys);
    auto const* result    = reinterpret_cast<spark_rapids_jni::key_remap_build_result*>(j_remap_handle);

    // Dump the remapping table
    auto dump_table = spark_rapids_jni::dump_remap_table(*build_keys, *result);

    return cudf::jni::convert_table_for_return(env, dump_table);
  }
  JNI_CATCH(env, 0);
}

}  // extern "C"

