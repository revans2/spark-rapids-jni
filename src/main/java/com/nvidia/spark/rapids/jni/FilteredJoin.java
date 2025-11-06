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

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.GatherMap;
import ai.rapids.cudf.NativeDepsLoader;
import ai.rapids.cudf.Table;

/**
 * Filtered join that builds hash table on the build-side table and probes with multiple
 * probe-side tables for semi/anti join operations.
 * <p>
 * This class enables the filtered join scheme that builds hash table once from the build table,
 * and probes as many times as needed with different probe tables for left semi and left anti joins.
 * </p>
 * <p>
 * <b>Note:</b> For this join type, the build table corresponds to the right table in a left semi/anti
 * join, and probe tables correspond to the left tables. The hash table is built on the right side.
 * </p>
 * <p>
 * <b>IMPORTANT:</b> This class is NOT thread-safe. Each thread should create its own FilteredJoin
 * instance. Do not share a single FilteredJoin object across multiple threads.
 * </p>
 * <p>
 * The filtered join object must not outlive the table viewed by the build table, else behavior is
 * undefined.
 * </p>
 * <p>
 * All NaNs are considered as equal.
 * </p>
 * <p>
 * Example usage:
 * <pre>{@code
 * try (Table rightKeys = ...;
 *      FilteredJoin filteredJoin = FilteredJoin.create(rightKeys, true)) {
 *   // Probe multiple times with different left tables
 *   try (Table leftKeys1 = ...;
 *        GatherMap result1 = filteredJoin.semiJoin(leftKeys1)) {
 *     // Process semi join result1 - contains indices from leftKeys1
 *   }
 *   try (Table leftKeys2 = ...;
 *        GatherMap result2 = filteredJoin.antiJoin(leftKeys2)) {
 *     // Process anti join result2 - contains indices from leftKeys2
 *   }
 * }
 * }</pre>
 * </p>
 */
public class FilteredJoin implements AutoCloseable {
  
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private long nativeHandle;
  private boolean closed = false;

  /**
   * Private constructor - use static create method
   */
  private FilteredJoin(long nativeHandle) {
    this.nativeHandle = nativeHandle;
  }

  /**
   * Create a filtered join object for subsequent semi/anti join calls.
   * <p>
   * This pre-builds the hash table from the build-side table (right table) that can be probed
   * multiple times with different probe-side tables (left tables). The filtered join object must
   * not outlive the build table.
   * </p>
   *
   * @param buildKeys The build-side join keys (right table, hash table will be built from these)
   * @param compareNullsEqual Whether null join-key values should match or not
   * @return A new FilteredJoin object
   * @throws IllegalArgumentException if buildKeys has no columns
   */
  public static FilteredJoin create(Table buildKeys, boolean compareNullsEqual) {
    return new FilteredJoin(createNative(buildKeys.getNativeView(), compareNullsEqual));
  }

  /**
   * Perform a left semi join with the probe table.
   * <p>
   * Returns a gather map containing row indices from the probe-side table (left table) for which
   * there is a matching row in the build-side table (right table provided to create()).
   * </p>
   *
   * @param probeKeys The probe-side join keys (left table) to probe with
   * @return GatherMap containing probe-side (left) indices where a match exists in the build-side (right)
   */
  public GatherMap semiJoin(Table probeKeys) {
    if (closed) {
      throw new IllegalStateException("FilteredJoin object is closed");
    }
    long[] result = semiJoinNative(nativeHandle, probeKeys.getNativeView());
    return JoinPrimitives.gatherMapFromJNI(result);
  }

  /**
   * Perform a left anti join with the probe table.
   * <p>
   * Returns a gather map containing row indices from the probe-side table (left table) for which
   * there are NO matching rows in the build-side table (right table provided to create()).
   * </p>
   *
   * @param probeKeys The probe-side join keys (left table) to probe with
   * @return GatherMap containing probe-side (left) indices where NO match exists in the build-side (right)
   */
  public GatherMap antiJoin(Table probeKeys) {
    if (closed) {
      throw new IllegalStateException("FilteredJoin object is closed");
    }
    long[] result = antiJoinNative(nativeHandle, probeKeys.getNativeView());
    return JoinPrimitives.gatherMapFromJNI(result);
  }

  @Override
  public void close() {
    if (!closed) {
      closeNative(nativeHandle);
      closed = true;
      nativeHandle = 0;
    }
  }

  // Native method declarations
  private static native long createNative(long buildKeysHandle, boolean compareNullsEqual);
  private static native long[] semiJoinNative(long handle, long probeKeysHandle);
  private static native long[] antiJoinNative(long handle, long probeKeysHandle);
  private static native void closeNative(long handle);
}

