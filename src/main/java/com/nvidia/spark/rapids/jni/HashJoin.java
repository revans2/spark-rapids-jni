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
 * Hash join that builds hash table on creation and probes results in subsequent join operations.
 * <p>
 * This class enables the hash join scheme that builds hash table once, and probes as many times as
 * needed.
 * </p>
 * <p>
 * <b>IMPORTANT:</b> This class is NOT thread-safe. Each thread should create its own HashJoin
 * instance. Do not share a single HashJoin object across multiple threads.
 * </p>
 * <p>
 * The hash join object must not outlive the table viewed by the build table, else behavior is
 * undefined.
 * </p>
 * <p>
 * Example usage:
 * <pre>{@code
 * try (Table buildKeys = ...;
 *      HashJoin hashJoin = HashJoin.create(buildKeys, true)) {
 *   // Probe multiple times with different tables
 *   try (Table probeKeys1 = ...;
 *        GatherMap[] result1 = hashJoin.innerJoin(probeKeys1)) {
 *     // Process result1
 *   }
 *   try (Table probeKeys2 = ...;
 *        GatherMap[] result2 = hashJoin.innerJoin(probeKeys2)) {
 *     // Process result2
 *   }
 * }
 * }</pre>
 * </p>
 */
public class HashJoin implements AutoCloseable {
  
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private long nativeHandle;
  private boolean closed = false;

  /**
   * Private constructor - use static create method
   */
  private HashJoin(long nativeHandle) {
    this.nativeHandle = nativeHandle;
  }

  /**
   * Create a hash join object for subsequent probe calls.
   * <p>
   * This pre-builds the hash table that can be probed multiple times.
   * The hash join object must not outlive the build table.
   * </p>
   *
   * @param buildKeys The build-side join keys (hash table will be built from these)
   * @param compareNullsEqual Whether null join-key values should match or not
   * @return A new HashJoin object
   * @throws IllegalArgumentException if buildKeys has no columns
   */
  public static HashJoin create(Table buildKeys, boolean compareNullsEqual) {
    return new HashJoin(createNative(buildKeys.getNativeView(), compareNullsEqual));
  }

  /**
   * Probe the hash table with probe-side keys for inner join.
   * <p>
   * Returns gather maps for the join result, containing only rows where keys match between
   * the build and probe tables.
   * </p>
   *
   * @param probeKeys The probe-side join keys
   * @return Array of two GatherMaps [build_indices, probe_indices] for matched rows
   */
  public GatherMap[] innerJoin(Table probeKeys) {
    if (closed) {
      throw new IllegalStateException("HashJoin object is closed");
    }
    long[] result = innerJoinNative(nativeHandle, probeKeys.getNativeView());
    return JoinPrimitives.gatherMapsFromJNI(result);
  }

  /**
   * Probe the hash table with probe-side keys for left join.
   * <p>
   * Returns gather maps for the join result. In a left join, all rows from the build table
   * (left side) are kept. Unmatched build rows will have corresponding probe indices set to
   * out-of-bounds values.
   * </p>
   * <p>
   * <b>Note:</b> This is a "left join" where the build table is the left side and the probe
   * table is the right side. All build table rows are preserved.
   * </p>
   *
   * @param probeKeys The probe-side join keys
   * @return Array of two GatherMaps [build_indices, probe_indices] where unmatched build rows
   *         will have out-of-bounds probe indices
   */
  public GatherMap[] leftJoin(Table probeKeys) {
    if (closed) {
      throw new IllegalStateException("HashJoin object is closed");
    }
    long[] result = leftJoinNative(nativeHandle, probeKeys.getNativeView());
    return JoinPrimitives.gatherMapsFromJNI(result);
  }

  /**
   * Probe the hash table with probe-side keys for full join.
   * <p>
   * Returns gather maps for the join result. In a full join, all rows from both the build
   * and probe tables are kept. Unmatched rows from either side will have their corresponding
   * indices set to out-of-bounds values.
   * </p>
   *
   * @param probeKeys The probe-side join keys
   * @return Array of two GatherMaps [build_indices, probe_indices] where unmatched rows from
   *         either table will have out-of-bounds indices in the corresponding gather map
   */
  public GatherMap[] fullJoin(Table probeKeys) {
    if (closed) {
      throw new IllegalStateException("HashJoin object is closed");
    }
    long[] result = fullJoinNative(nativeHandle, probeKeys.getNativeView());
    return JoinPrimitives.gatherMapsFromJNI(result);
  }

  /**
   * Get the exact output size for an inner join with the specified probe table.
   *
   * @param probeKeys The probe-side join keys
   * @return The exact number of output rows that would result from an inner join
   */
  public long innerJoinSize(Table probeKeys) {
    if (closed) {
      throw new IllegalStateException("HashJoin object is closed");
    }
    return innerJoinSizeNative(nativeHandle, probeKeys.getNativeView());
  }

  /**
   * Get the exact output size for a left join with the specified probe table.
   *
   * @param probeKeys The probe-side join keys
   * @return The exact number of output rows that would result from a left join
   */
  public long leftJoinSize(Table probeKeys) {
    if (closed) {
      throw new IllegalStateException("HashJoin object is closed");
    }
    return leftJoinSizeNative(nativeHandle, probeKeys.getNativeView());
  }

  /**
   * Get the exact output size for a full join with the specified probe table.
   *
   * @param probeKeys The probe-side join keys
   * @return The exact number of output rows that would result from a full join
   */
  public long fullJoinSize(Table probeKeys) {
    if (closed) {
      throw new IllegalStateException("HashJoin object is closed");
    }
    return fullJoinSizeNative(nativeHandle, probeKeys.getNativeView());
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
  private static native long[] fullJoinNative(long handle, long probeKeysHandle);
  private static native long innerJoinSizeNative(long handle, long probeKeysHandle);
  private static native long leftJoinSizeNative(long handle, long probeKeysHandle);
  private static native long fullJoinSizeNative(long handle, long probeKeysHandle);
  private static native void closeNative(long handle);
}

