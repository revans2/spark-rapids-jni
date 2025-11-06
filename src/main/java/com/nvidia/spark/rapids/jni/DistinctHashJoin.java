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
 * Distinct hash join that builds hash table on creation and probes results in subsequent join
 * operations.
 * <p>
 * This class enables the distinct hash join scheme that builds hash table once, and probes as many
 * times as needed.
 * </p>
 * <p>
 * <b>IMPORTANT:</b> This class is NOT thread-safe. Each thread should create its own
 * DistinctHashJoin instance. Do not share a single DistinctHashJoin object across multiple
 * threads.
 * </p>
 * <p>
 * <b>IMPORTANT:</b> Behavior is undefined if the build table contains duplicate keys.
 * All NaNs are considered as equal.
 * </p>
 * <p>
 * The distinct hash join object must not outlive the table viewed by the build table, else
 * behavior is undefined.
 * </p>
 * <p>
 * Example usage:
 * <pre>{@code
 * try (Table buildKeys = ...; // Must have distinct keys!
 *      DistinctHashJoin hashJoin = DistinctHashJoin.create(buildKeys, true)) {
 *   // Probe multiple times with different tables
 *   try (Table probeKeys1 = ...;
 *        GatherMap[] result1 = hashJoin.innerJoin(probeKeys1)) {
 *     // Process result1
 *   }
 *   try (Table probeKeys2 = ...;
 *        GatherMap buildIndices = hashJoin.leftJoin(probeKeys2)) {
 *     // Process result2 - leftJoin returns only build indices
 *   }
 * }
 * }</pre>
 * </p>
 */
public class DistinctHashJoin implements AutoCloseable {
  
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private long nativeHandle;
  private boolean closed = false;

  /**
   * Private constructor - use static create method
   */
  private DistinctHashJoin(long nativeHandle) {
    this.nativeHandle = nativeHandle;
  }

  /**
   * Create a distinct hash join object for subsequent probe calls.
   * <p>
   * This pre-builds the hash table that can be probed multiple times.
   * The hash join object must not outlive the build table.
   * </p>
   * <p>
   * IMPORTANT: The build table must contain distinct keys. Behavior is undefined otherwise.
   * </p>
   *
   * @param buildKeys The build-side join keys (must be distinct, hash table will be built from
   *                  these)
   * @param compareNullsEqual Whether null join-key values should match or not
   * @return A new DistinctHashJoin object
   * @throws IllegalArgumentException if buildKeys has no columns
   */
  public static DistinctHashJoin create(Table buildKeys, boolean compareNullsEqual) {
    return new DistinctHashJoin(createNative(buildKeys.getNativeView(), compareNullsEqual));
  }

  /**
   * Probe the hash table with probe-side keys for inner join.
   * <p>
   * Returns gather maps for the join result, containing only rows where keys match between
   * the build and probe tables. Since the build side has distinct keys, each probe key
   * matches at most one build key.
   * </p>
   *
   * @param probeKeys The probe-side join keys
   * @return Array of two GatherMaps [build_indices, probe_indices] for matched rows
   */
  public GatherMap[] innerJoin(Table probeKeys) {
    if (closed) {
      throw new IllegalStateException("DistinctHashJoin object is closed");
    }
    long[] result = innerJoinNative(nativeHandle, probeKeys.getNativeView());
    return JoinPrimitives.gatherMapsFromJNI(result);
  }

  /**
   * Probe the hash table with probe-side keys for left join.
   * <p>
   * Returns only the build table indices. This is a "left join" in the sense that all probe
   * table rows (right side) are preserved. For a given row index i of the probe table, the
   * resulting build_indices[i] contains the row index of the matched row from the build table
   * if there is a match. Otherwise, contains an out-of-bounds value.
   * </p>
   * <p>
   * This is more efficient than returning both sets of indices since the probe indices are just
   * 0, 1, 2, ..., n-1 and don't need to be materialized.
   * </p>
   * <p>
   * <b>Note:</b> This is semantically a "right join" (all probe/right rows kept), but is called
   * left_join in CUDF to match the convention where build=left. The important thing is that
   * all probe table rows are kept in the result.
   * </p>
   *
   * @param probeKeys The probe-side join keys
   * @return A single GatherMap containing build table indices (one per probe row)
   */
  public GatherMap leftJoin(Table probeKeys) {
    if (closed) {
      throw new IllegalStateException("DistinctHashJoin object is closed");
    }
    long[] result = leftJoinNative(nativeHandle, probeKeys.getNativeView());
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
  private static native long[] innerJoinNative(long handle, long probeKeysHandle);
  private static native long[] leftJoinNative(long handle, long probeKeysHandle);
  private static native void closeNative(long handle);
}

