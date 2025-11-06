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
 * Sort-merge join that pre-processes the build table on creation and can be used on subsequent
 * join operations with multiple probe tables.
 * <p>
 * This class enables the sort-merge join scheme that caches some preprocessed data on the build side, and
 * merges as many times as needed with different probe tables.
 * </p>
 * <p>
 * <b>IMPORTANT:</b> This class is NOT thread-safe. Each thread should create its own
 * SortMergeJoin instance. Do not share a single SortMergeJoin object across multiple threads.
 * </p>
 * <p>
 * The sort-merge join object must not outlive the table viewed by the build table, else behavior
 * is undefined.
 * </p>
 * <p>
 * Example usage:
 * <pre>{@code
 * try (Table buildKeys = ...;
 *      SortMergeJoin sortMergeJoin = SortMergeJoin.create(buildKeys, false, true)) {
 *   // Probe multiple times with different probe tables
 *   try (Table probeKeys1 = ...;
 *        GatherMap[] result1 = sortMergeJoin.innerJoin(probeKeys1, false)) {
 *     // Process result1
 *   }
 *   try (Table probeKeys2 = ...;
 *        GatherMap[] result2 = sortMergeJoin.innerJoin(probeKeys2, false)) {
 *     // Process result2
 *   }
 * }
 * }</pre>
 * </p>
 */
public class SortMergeJoin implements AutoCloseable {
  
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private long nativeHandle;
  private boolean closed = false;

  /**
   * Private constructor - use static create method
   */
  private SortMergeJoin(long nativeHandle) {
    this.nativeHandle = nativeHandle;
  }

  /**
   * Create a sort-merge join object for subsequent join calls.
   * <p>
   * This pre-processes the build table which can then be merged multiple times with different
   * probe tables.
   * </p>
   *
   * @param buildKeys The build table join keys
   * @param isBuildSorted Whether the build table is already sorted by the join keys
   * @param compareNullsEqual Whether null join-key values should match or not
   * @return A new SortMergeJoin object
   * @throws IllegalArgumentException if buildKeys has no columns
   */
  public static SortMergeJoin create(Table buildKeys, boolean isBuildSorted,
                                      boolean compareNullsEqual) {
    return new SortMergeJoin(createNative(buildKeys.getNativeView(), isBuildSorted,
                                           compareNullsEqual));
  }

  /**
   * Perform an inner join with the specified probe table.
   * <p>
   * Returns gather maps for the join result, containing only rows where keys match between
   * the build table (passed to create()) and the probe table (passed here).
   * </p>
   *
   * @param probeKeys The probe table join keys
   * @param isProbeSorted Whether the probe table is already sorted by the join keys (optimization)
   * @return Array of two GatherMaps [build_indices, probe_indices] for matched rows
   */
  public GatherMap[] innerJoin(Table probeKeys, boolean isProbeSorted) {
    if (closed) {
      throw new IllegalStateException("SortMergeJoin object is closed");
    }
    long[] result = innerJoinNative(nativeHandle, probeKeys.getNativeView(), isProbeSorted);
    return JoinPrimitives.gatherMapsFromJNI(result);
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
  private static native long createNative(long buildKeysHandle, boolean isBuildSorted,
                                           boolean compareNullsEqual);
  private static native long[] innerJoinNative(long handle, long probeKeysHandle,
                                                boolean isProbeSorted);
  private static native void closeNative(long handle);
}

