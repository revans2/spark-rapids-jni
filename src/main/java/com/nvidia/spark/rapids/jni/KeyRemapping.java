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

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.NativeDepsLoader;
import ai.rapids.cudf.Table;

/**
 * Utilities for remapping complex join keys (String, Decimal) to dense integer keys.
 * <p>
 * This optimization can improve join performance by converting complex key types to integers.
 * Each distinct key that appears on the build side is assigned a stable non-negative integer.
 * Keys that do not appear on the build side map to a negative sentinel value (see
 * {@link #getNotFoundSentinel()}).
 * </p>
 * <p>
 * <b>Usage pattern:</b>
 * <pre>{@code
 * // ONE-TIME SETUP (cache these across iterations):
 * try (RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {
 *   // Get distinct count for heuristics if needed
 *   int distinctCount = remap.getDistinctCount();
 *   
 *   // FOR EACH ITERATION:
 *   // 1. Remap build keys
 *   try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap)) {
 *     // 2. Remap probe keys
 *     try (ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap)) {
 *       // 3. Perform join with integer keys
 *       // Keys are mapped to non-negative integers. Unmatched keys return a negative sentinel.
 *     }
 *   }
 * }
 * }</pre>
 * </p>
 */
public class KeyRemapping {

  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Container for remapping structures that can be cached and reused.
   * <p>
   * This provides a high-performance remapping table.
   * Each distinct build-side key is assigned a stable non-negative integer value for the lifetime
   * of the structure.
   * </p>
   * <p>
   * All resources must be closed when no longer needed.
   * </p>
   */
  public static class RemapStructures implements AutoCloseable {
    private long nativeHandle;
    private Table buildKeys;  // Keep the original build keys for two-table lookups
    private boolean closed = false;
    private int distinctCount = -1;  // Cached value

    private RemapStructures(long nativeHandle, Table buildKeys) {
      this.nativeHandle = nativeHandle;
      this.buildKeys = buildKeys;
    }

    /**
     * Get the number of distinct keys in the remapping structure.
     * <p>
     * This is useful for heuristics to decide whether remapping is beneficial,
     * or to choose between different join strategies based on cardinality.
     * </p>
     * 
     * @return The number of distinct keys found during remapping
     */
    public int getDistinctCount() {
      if (closed) {
        throw new IllegalStateException("RemapStructures is already closed");
      }
      if (distinctCount < 0) {
        distinctCount = getDistinctCountNative(nativeHandle);
      }
      return distinctCount;
    }

    /**
     * Get the native handle to the remapping structure.
     * <p>
     * <b>Internal use only.</b>
     * </p>
     */
    long getNativeHandle() {
      if (closed) {
        throw new IllegalStateException("RemapStructures is already closed");
      }
      return nativeHandle;
    }

    @Override
    public void close() {
      if (!closed) {
        if (nativeHandle != 0) {
          freeKeyRemapNative(nativeHandle);
          nativeHandle = 0;
        }
        // Note: We don't close buildKeys here because we don't own it
        // The caller is responsible for managing the buildKeys lifetime
        buildKeys = null;
        closed = true;
      }
    }
  }

  /**
   * Create remapping structures from build-side keys using native implementation.
   * <p>
   * Builds a remapping table where each distinct build-side key is assigned a stable, non-negative
   * integer value for the lifetime of the returned structure.
   * </p>
   * <p>
   * <b>IMPORTANT:</b> The returned RemapStructures retains a reference to the provided buildKeys
   * table. The caller must keep buildKeys alive and unclosed for as long as the RemapStructures
   * instance remains in use (or until it is closed).
   * </p>
   *
   * @param buildKeys The build-side join keys from which to extract distinct values
   * @return RemapStructures containing the native hash map
   */
  public static RemapStructures createRemapStructures(Table buildKeys) {
    return createRemapStructures(buildKeys, true);
  }

  /**
   * Create remapping structures from build-side keys using native implementation.
   * <p>
   * Builds a remapping table where each distinct build-side key is assigned a stable, non-negative
   * integer value for the lifetime of the returned structure.
   * </p>
   * <p>
   * <b>IMPORTANT:</b> The returned RemapStructures retains a reference to the provided buildKeys
   * table. The caller must keep buildKeys alive and unclosed for as long as the RemapStructures
   * instance remains in use (or until it is closed).
   * </p>
   *
   * @param buildKeys The build-side join keys from which to extract distinct values
   * @param compareNullsEqual Whether null key values should be considered equal
   * @return RemapStructures containing the native hash map
   */
  public static RemapStructures createRemapStructures(Table buildKeys,
                                                      boolean compareNullsEqual) {
    long nativeHandle = buildKeyRemapNative(buildKeys.getNativeView(), compareNullsEqual);

    if (nativeHandle == 0) {
      throw new IllegalStateException("Failed to build key remapping structure");
    }

    return new RemapStructures(nativeHandle, buildKeys);
  }

  /**
   * Apply key remapping using the native implementation.
   * <p>
   * Applies the remapping contract to the supplied keys. Values that correspond to build-side keys
   * return the non-negative integer assigned during creation. Keys that were not present in the
   * build table return the negative sentinel reported by {@link #getNotFoundSentinel()}.
   * <b>Note:</b> {@code remapStructures} must still hold a valid, unclosed reference to the original
   * build keys table.
   * </p>
   *
   * @param keys The keys to remap (from either build or probe table)
   * @param remapStructures The cached remapping structures
   * @return ColumnVector containing the remapped integer keys (caller must close)
   */
  public static ColumnVector applyRemapping(Table keys,
                                            RemapStructures remapStructures) {
    long columnHandle = applyKeyRemapNative(remapStructures.buildKeys.getNativeView(),
                                            keys.getNativeView(),
                                            remapStructures.getNativeHandle());
    return new ColumnVector(columnHandle);
  }

  /**
   * Get the sentinel value returned when a key is not present in the build-side map.
   *
   * @return Sentinel value used for unmatched keys.
   */
  public static int getNotFoundSentinel() {
    return getNotFoundSentinelNative();
  }

  // ==================== NATIVE METHOD DECLARATIONS ====================

  /**
   * Build a key remapping structure from input keys.
   *
   * @param inputKeysHandle Native handle to the input keys table
   * @param nullsEqual Whether to treat null keys as equal
   * @return Native handle to the remapping structure
   */
  private static native long buildKeyRemapNative(long inputKeysHandle, boolean nullsEqual);

  /**
   * Get the distinct count from a remapping structure.
   *
   * @param remapHandle Native handle to the remapping structure
   * @return Number of distinct keys
   */
  private static native int getDistinctCountNative(long remapHandle);

  /**
   * Apply key remapping to input keys.
   *
   * @param buildKeysHandle Native handle to the build keys table
   * @param inputKeysHandle Native handle to the input keys table
   * @param remapHandle Native handle to the remapping structure
   * @return Column handle for the remapped integer keys
   */
  private static native long applyKeyRemapNative(long buildKeysHandle,
                                                 long inputKeysHandle,
                                                 long remapHandle);

  /**
   * Get the sentinel value used when a key is not found.
   *
   * @return Sentinel value used by the native implementation.
   */
  private static native int getNotFoundSentinelNative();

  /**
   * Free the native remapping structure.
   *
   * @param remapHandle Native handle to the remapping structure
   */
  private static native void freeKeyRemapNative(long remapHandle);
}

