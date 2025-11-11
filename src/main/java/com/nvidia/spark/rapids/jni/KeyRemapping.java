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
 * <b>Null Handling:</b> The null equality semantics can be controlled via {@link NullEqualityMode}:
 * <ul>
 *   <li>{@link NullEqualityMode#SPARK_EQUALITY} (default): Top-level nulls are not equal to each other,
 *       but nested nulls (inside structs/lists) are equal. Top-level null rows map to
 *       {@link #getBuildNullSentinel()}.</li>
 *   <li>{@link NullEqualityMode#NULL_EQUAL}: All nulls are equal at every level.</li>
 *   <li>{@link NullEqualityMode#NULL_NOT_EQUAL}: No nulls are equal at any level.</li>
 * </ul>
 * </p>
 * <p>
 * <b>Usage pattern:</b>
 * <pre>{@code
 * // ONE-TIME SETUP (cache these across iterations):
 * try (RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys, NullEqualityMode.SPARK_EQUALITY)) {
 *   // FOR EACH ITERATION:
 *   // 1. Remap build keys
 *   try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap, true)) {
 *     // 2. Remap probe keys
 *     try (ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap, false)) {
 *       // 3. Perform join with integer keys
 *       // Keys are mapped to non-negative integers or sentinel values:
 *       //   - Non-negative: valid build table index
 *       //   - getNotFoundSentinel(): probe key not in build table
 *       //   - getBuildNullSentinel(): build key with top-level null (if applicable)
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
   * Null equality modes for key remapping.
   */
  public enum NullEqualityMode {
    /**
     * All nulls are considered equal at every level (top-level and nested).
     * This matches standard CUDF null equality behavior.
     */
    NULL_EQUAL(0),

    /**
     * No nulls are considered equal at any level.
     * This matches CUDF's UNEQUAL null equality behavior.
     */
    NULL_NOT_EQUAL(1),

    /**
     * Spark-specific null equality semantics (default):
     * - Top-level nulls are NOT equal to each other
     * - Nested nulls (inside structs/lists) ARE equal to each other
     * This matches Spark's join behavior.
     */
    SPARK_EQUALITY(2);

    private final int nativeId;

    NullEqualityMode(int nativeId) {
      this.nativeId = nativeId;
    }

    int getNativeId() {
      return nativeId;
    }
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
    private NullEqualityMode nullEqualityMode;
    private boolean closed = false;

    private RemapStructures(long nativeHandle, Table buildKeys, NullEqualityMode nullEqualityMode) {
      this.nativeHandle = nativeHandle;
      this.buildKeys = buildKeys;
      this.nullEqualityMode = nullEqualityMode;
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

    /**
     * Get the null equality mode used for this remapping structure.
     * <p>
     * <b>Internal use only.</b>
     * </p>
     */
    NullEqualityMode getNullEqualityMode() {
      if (closed) {
        throw new IllegalStateException("RemapStructures is already closed");
      }
      return nullEqualityMode;
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
   * Uses {@link NullEqualityMode#SPARK_EQUALITY} as the default null equality mode.
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
    return createRemapStructures(buildKeys, NullEqualityMode.SPARK_EQUALITY);
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
   * @param nullEqualityMode How to handle null equality (top-level vs nested)
   * @return RemapStructures containing the native hash map
   */
  public static RemapStructures createRemapStructures(Table buildKeys,
                                                      NullEqualityMode nullEqualityMode) {
    long nativeHandle = buildKeyRemapNative(buildKeys.getNativeView(), 
                                           nullEqualityMode.getNativeId());

    if (nativeHandle == 0) {
      throw new IllegalStateException("Failed to build key remapping structure");
    }

    return new RemapStructures(nativeHandle, buildKeys, nullEqualityMode);
  }

  /**
   * Apply key remapping using the native implementation.
   * <p>
   * Applies the remapping contract to the supplied keys. The behavior depends on the null equality
   * mode and whether this is the build or probe side:
   * </p>
   * <ul>
   *   <li>Matching keys: Return non-negative integer (build table row index)</li>
   *   <li>Non-matching probe keys: Return {@link #getNotFoundSentinel()}</li>
   *   <li>Build keys with top-level nulls (when applicable): Return {@link #getBuildNullSentinel()}</li>
   * </ul>
   * <p>
   * <b>Note:</b> {@code remapStructures} must still hold a valid, unclosed reference to the original
   * build keys table.
   * </p>
   *
   * @param keys The keys to remap (from either build or probe table)
   * @param remapStructures The cached remapping structures
   * @param isBuildSide True if remapping the build table, false if remapping the probe table
   * @return ColumnVector containing the remapped integer keys (caller must close)
   */
  public static ColumnVector applyRemapping(Table keys,
                                            RemapStructures remapStructures,
                                            boolean isBuildSide) {
    long columnHandle = applyKeyRemapNative(remapStructures.buildKeys.getNativeView(),
                                            keys.getNativeView(),
                                            remapStructures.getNativeHandle(),
                                            remapStructures.getNullEqualityMode().getNativeId(),
                                            isBuildSide);
    return new ColumnVector(columnHandle);
  }

  /**
   * Get the sentinel value returned when a probe-side key is not present in the build-side map.
   *
   * @return Sentinel value used for unmatched probe keys (typically -1).
   */
  public static int getNotFoundSentinel() {
    return getNotFoundSentinelNative();
  }

  /**
   * Get the sentinel value returned for build-side rows with top-level null keys.
   * <p>
   * This sentinel is only used when the null equality mode is {@link NullEqualityMode#SPARK_EQUALITY}
   * or {@link NullEqualityMode#NULL_NOT_EQUAL}, and only for build-side rows where at least one
   * top-level key column contains a null.
   * </p>
   *
   * @return Sentinel value used for build-side null keys (typically -2).
   */
  public static int getBuildNullSentinel() {
    return getBuildNullSentinelNative();
  }

  /**
   * Dump the raw contents of the remapping hash map for debugging.
   * <p>
   * Returns a table with three columns showing the internal map structure:
   * <ul>
   *   <li>Column 0: Hash values (UINT32)</li>
   *   <li>Column 1: Key row indices from the build table (INT32)</li>
   *   <li>Column 2: Mapped values (INT32)</li>
   * </ul>
   * Each row represents one entry in the hash map. The key_row_index shows which row from
   * the build table this entry corresponds to, and mapped_value shows what integer ID was
   * assigned to that key.
   * </p>
   *
   * @param remapStructures The cached remapping structures
   * @return Table containing all entries in the remapping hash map (caller must close)
   */
  public static Table dumpRemapTable(RemapStructures remapStructures) {
    long[] columnHandles = dumpRemapTableNative(remapStructures.buildKeys.getNativeView(),
                                                 remapStructures.getNativeHandle());
    return new Table(columnHandles);
  }

  // ==================== NATIVE METHOD DECLARATIONS ====================

  /**
   * Build a key remapping structure from input keys.
   *
   * @param inputKeysHandle Native handle to the input keys table
   * @param nullEqualityMode Integer representing the null equality mode (0=NULL_EQUAL, 1=NULL_NOT_EQUAL, 2=SPARK_EQUALITY)
   * @return Native handle to the remapping structure
   */
  private static native long buildKeyRemapNative(long inputKeysHandle, int nullEqualityMode);

  /**
   * Apply key remapping to input keys.
   *
   * @param buildKeysHandle Native handle to the build keys table
   * @param inputKeysHandle Native handle to the input keys table
   * @param remapHandle Native handle to the remapping structure
   * @param nullEqualityMode Integer representing the null equality mode
   * @param isBuildSide Whether this is remapping the build side (true) or probe side (false)
   * @return Column handle for the remapped integer keys
   */
  private static native long applyKeyRemapNative(long buildKeysHandle,
                                                 long inputKeysHandle,
                                                 long remapHandle,
                                                 int nullEqualityMode,
                                                 boolean isBuildSide);

  /**
   * Get the sentinel value used when a probe key is not found in the build map.
   *
   * @return Sentinel value for unmatched probe keys (typically -1).
   */
  private static native int getNotFoundSentinelNative();

  /**
   * Get the sentinel value used for build-side rows with top-level null keys.
   *
   * @return Sentinel value for build-side null keys (typically -2).
   */
  private static native int getBuildNullSentinelNative();

  /**
   * Free the native remapping structure.
   *
   * @param remapHandle Native handle to the remapping structure
   */
  private static native void freeKeyRemapNative(long remapHandle);

  /**
   * Dump the contents of the remapping table for debugging.
   *
   * @param buildKeysHandle Native handle to the build keys table
   * @param remapHandle Native handle to the remapping structure
   * @return Array of column handles containing all entries in the remapping table
   */
  private static native long[] dumpRemapTableNative(long buildKeysHandle, long remapHandle);
}

