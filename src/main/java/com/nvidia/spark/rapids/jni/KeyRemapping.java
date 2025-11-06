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
import ai.rapids.cudf.GatherMap;
import ai.rapids.cudf.OutOfBoundsPolicy;
import ai.rapids.cudf.Scalar;
import ai.rapids.cudf.Table;

/**
 * Utilities for remapping complex join keys (String, Decimal) to dense integer keys.
 * <p>
 * This optimization can improve join performance by converting complex key types to integers.
 * The remapping process:
 * <ol>
 *   <li>Extract distinct keys from the build side</li>
 *   <li>Create a dense integer sequence (0, 1, 2, ..., numDistinct-1)</li>
 *   <li>Create a DistinctHashJoin object to perform the remapping</li>
 *   <li>Use left join to remap keys (unmatched keys get sentinel value Integer.MIN_VALUE)</li>
 * </ol>
 * </p>
 * <p>
 * <b>Usage pattern for benchmarking:</b>
 * <pre>{@code
 * // ONE-TIME SETUP (cache these across iterations):
 * try (RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {
 *   
 *   // FOR EACH ITERATION:
 *   // 1. Remap build keys
 *   try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap)) {
 *     // 2. Remap probe keys
 *     try (ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap)) {
 *       // 3. Create join tables with remapped keys
 *       try (Table buildTable = new Table(remappedBuild);
 *            Table probeTable = new Table(remappedProbe)) {
 *         // 4. Perform join with integer keys
 *         // ...
 *       }
 *     }
 *   }
 * }
 * }</pre>
 * </p>
 */
public class KeyRemapping {

  /**
   * Container for remapping structures that can be cached and reused.
   * <p>
   * All resources must be closed when no longer needed.
   * </p>
   */
  public static class RemapStructures implements AutoCloseable {
    private final Table distinctKeys;
    private final Table intSequenceTable;
    private final DistinctHashJoin remapJoinObject;

    private RemapStructures(Table distinctKeys, Table intSequenceTable,
                            DistinctHashJoin remapJoinObject) {
      this.distinctKeys = distinctKeys;
      this.intSequenceTable = intSequenceTable;
      this.remapJoinObject = remapJoinObject;
    }

    /**
     * Get the distinct keys extracted from the build side.
     * <p>
     * <b>Note:</b> This Table is owned by the RemapStructures and will be closed when
     * RemapStructures is closed. Do not close it separately.
     * </p>
     */
    public Table getDistinctKeys() {
      return distinctKeys;
    }

    /**
     * Get the integer sequence table used for remapping.
     * <p>
     * <b>Note:</b> This Table is owned by the RemapStructures and will be closed when
     * RemapStructures is closed. Do not close it separately.
     * </p>
     */
    public Table getIntSequenceTable() {
      return intSequenceTable;
    }

    /**
     * Get the distinct hash join object used for remapping.
     * <p>
     * <b>Note:</b> This DistinctHashJoin is owned by the RemapStructures and will be closed
     * when RemapStructures is closed. Do not close it separately.
     * </p>
     */
    public DistinctHashJoin getRemapJoinObject() {
      return remapJoinObject;
    }

    @Override
    public void close() {
      if (remapJoinObject != null) {
        remapJoinObject.close();
      }
      if (intSequenceTable != null) {
        intSequenceTable.close();
      }
      if (distinctKeys != null) {
        distinctKeys.close();
      }
    }
  }

  /**
   * Create remapping structures from build-side keys.
   * <p>
   * This extracts distinct keys, creates an integer sequence, and builds a DistinctHashJoin
   * object for remapping. These structures can be cached and reused across multiple join
   * operations.
   * </p>
   * <p>
   * The integer sequence uses 0-based integers (0, 1, 2, ..., numDistinct-1), which reserves
   * Integer.MIN_VALUE as a sentinel value for unmatched keys.
   * </p>
   *
   * @param buildKeys The build-side join keys from which to extract distinct values
   * @return RemapStructures containing the distinct keys, integer sequence, and join object
   */
  public static RemapStructures createRemapStructures(Table buildKeys) {
    // 1. Get distinct keys from build side
    // Create array of all column indices
    int[] keyColumns = new int[buildKeys.getNumberOfColumns()];
    for (int i = 0; i < keyColumns.length; i++) {
      keyColumns[i] = i;
    }
    Table distinctKeys = buildKeys.dropDuplicates(keyColumns, Table.DuplicateKeepOption.KEEP_FIRST, true);
    
    try {
      // 2. Create dense integer sequence (0, 1, 2, ..., numDistinct-1)
      // We use 0-based sequence so that Integer.MIN_VALUE can be used as sentinel
      int numDistinct = (int) distinctKeys.getRowCount();
      ColumnVector intSequence = ColumnVector.sequence(
        Scalar.fromInt(0),
        Scalar.fromInt(1),
        numDistinct);
      
      try {
        Table intSequenceTable = new Table(intSequence);
        
        try {
          // 3. Create DistinctHashJoin for the distinct keys
          // This will be used to remap both build and probe keys
          DistinctHashJoin remapJoinObject = DistinctHashJoin.create(
            distinctKeys,
            true  // compareNullsEqual - treat nulls as equal during remapping
          );
          
          try {
            return new RemapStructures(distinctKeys, intSequenceTable, remapJoinObject);
          } catch (Exception e) {
            remapJoinObject.close();
            throw e;
          }
        } catch (Exception e) {
          intSequenceTable.close();
          throw e;
        }
      } catch (Exception e) {
        intSequence.close();
        throw e;
      }
    } catch (Exception e) {
      distinctKeys.close();
      throw e;
    }
  }

  /**
   * Apply key remapping to a set of keys using cached remapping structures.
   * <p>
   * This remaps the input keys to dense integers based on the distinct keys from the build side.
   * Keys that don't exist in the build-side distinct keys are mapped to Integer.MIN_VALUE as a
   * sentinel value.
   * </p>
   * <p>
   * <b>Implementation details:</b>
   * <ul>
   *   <li>Uses left join to ensure every input key gets a mapping</li>
   *   <li>Gathers from integer sequence with NULLIFY policy for unmatched keys</li>
   *   <li>Replaces nulls with Integer.MIN_VALUE sentinel</li>
   * </ul>
   * </p>
   *
   * @param keys The keys to remap (from either build or probe table)
   * @param remapStructures The cached remapping structures
   * @return ColumnVector containing the remapped integer keys (caller must close)
   */
  public static ColumnVector applyRemapping(Table keys, RemapStructures remapStructures) {
    // Use left join to handle keys not in the build-side distinct keys
    // This ensures we get a row for every input key, even if not found
    GatherMap buildIndices = remapStructures.getRemapJoinObject().leftJoin(keys);
    
    try {
      // Gather from integer sequence table with NULLIFY policy
      // For keys not found in distinct keys, this will produce null
      Table gathered = remapStructures.getIntSequenceTable().gather(
        buildIndices.toColumnView(0, (int) buildIndices.getRowCount()),
        OutOfBoundsPolicy.NULLIFY);
      
      try {
        // Replace nulls with Integer.MIN_VALUE sentinel
        // This ensures unmatched keys don't falsely match in the actual join
        ColumnVector result = gathered.getColumn(0).replaceNulls(Scalar.fromInt(Integer.MIN_VALUE));
        return result;
      } finally {
        gathered.close();
      }
    } finally {
      buildIndices.close();
    }
  }
}

