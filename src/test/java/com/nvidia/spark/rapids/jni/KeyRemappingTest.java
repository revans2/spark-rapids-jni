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
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.Table;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

public class KeyRemappingTest {

  private static final int SENTINEL = KeyRemapping.getNotFoundSentinel();

  /**
   * Helper class to represent a key (can be single or multi-column).
   */
  private static class Key {
    private final Object[] values;
    
    public Key(Object... values) {
      this.values = values;
    }
    
    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (o == null || getClass() != o.getClass()) return false;
      Key key = (Key) o;
      return java.util.Arrays.deepEquals(values, key.values);
    }
    
    @Override
    public int hashCode() {
      return java.util.Arrays.deepHashCode(values);
    }
    
    @Override
    public String toString() {
      return java.util.Arrays.toString(values);
    }
  }

  /**
   * Verify that the build side remapping is correct (defaults to SPARK_EQUALITY mode).
   * 
   * @param buildKeys The original build keys (host data)
   * @param remappedBuild The remapped IDs (host data)
   */
  private void verifyBuildRemapping(HostColumnVector[] buildKeys,
                                    HostColumnVector remappedBuild) {
    verifyBuildRemapping(buildKeys, remappedBuild, KeyRemapping.NullEqualityMode.SPARK_EQUALITY);
  }

  /**
   * Verify that the build side remapping is correct.
   * 
   * @param buildKeys The original build keys (host data)
   * @param remappedBuild The remapped IDs (host data)
   * @param nullMode The null equality mode being tested
   */
  private void verifyBuildRemapping(HostColumnVector[] buildKeys,
                                    HostColumnVector remappedBuild,
                                    KeyRemapping.NullEqualityMode nullMode) {
    int rowCount = (int) remappedBuild.getRowCount();
    assertEquals(rowCount, buildKeys[0].getRowCount());

    // Map from key to its assigned ID
    Map<Key, Integer> keyToId = new HashMap<>();
    Set<Integer> assignedIds = new HashSet<>();
    
    for (int i = 0; i < rowCount; i++) {
      // Extract the key for this row
      Object[] keyValues = new Object[buildKeys.length];
      boolean hasTopLevelNull = false;
      for (int col = 0; col < buildKeys.length; col++) {
        if (buildKeys[col].isNull(i)) {
          keyValues[col] = null;
          hasTopLevelNull = true;
        } else if (buildKeys[col].getType().equals(ai.rapids.cudf.DType.STRING)) {
          keyValues[col] = buildKeys[col].getJavaString(i);
        } else if (buildKeys[col].getType().equals(ai.rapids.cudf.DType.INT32)) {
          keyValues[col] = buildKeys[col].getInt(i);
        } else {
          throw new IllegalArgumentException("Unsupported type: " + buildKeys[col].getType());
        }
      }
      Key key = new Key(keyValues);
      
      // Output never has nulls - uses sentinel values instead
      assertFalse(remappedBuild.isNull(i),
          "Build remapping output should never be null (uses sentinels)");
      
      int assignedId = remappedBuild.getInt(i);
      
      // In SPARK_EQUALITY and NULL_NOT_EQUAL: top-level nulls get BUILD_NULL_SENTINEL
      // In NULL_EQUAL: nulls are treated as regular values and get valid IDs
      if (hasTopLevelNull && nullMode != KeyRemapping.NullEqualityMode.NULL_EQUAL) {
        assertEquals(KeyRemapping.getBuildNullSentinel(), assignedId,
            "Null key should get BUILD_NULL_SENTINEL in " + nullMode + " mode");
      } else {
        // Non-null keys (or null keys in NULL_EQUAL mode) get valid IDs
        assertTrue(assignedId >= 0,
            "Valid key should get non-negative ID, got " + assignedId + " for key " + key);
        
        // Check consistency: same key should always get same ID
        if (keyToId.containsKey(key)) {
          assertEquals(keyToId.get(key), assignedId,
              "Key " + key + " should always map to same ID");
        } else {
          keyToId.put(key, assignedId);
        }
        
        assignedIds.add(assignedId);
      }
    }
    
    // Verify different keys got different IDs
    assertEquals(keyToId.size(), assignedIds.size(),
        "Different keys should get different IDs");
  }

  /**
   * Verify that the probe side remapping is correct (defaults to SPARK_EQUALITY mode).
   * 
   * @param probeKeys The original probe keys (host data)
   * @param remappedProbe The remapped IDs (host data)
   * @param buildKeys The original build keys (host data) 
   * @param remappedBuild The build remapped IDs (host data)
   * @param sentinelValue Expected sentinel value for unmatched keys
   */
  private void verifyProbeRemapping(HostColumnVector[] probeKeys,
                                    HostColumnVector remappedProbe,
                                    HostColumnVector[] buildKeys,
                                    HostColumnVector remappedBuild,
                                    int sentinelValue) {
    verifyProbeRemapping(probeKeys, remappedProbe, buildKeys, remappedBuild, sentinelValue, 
                        KeyRemapping.NullEqualityMode.SPARK_EQUALITY);
  }

  /**
   * Verify that the probe side remapping is correct and consistent with build side.
   * 
   * @param probeKeys The original probe keys (host data)
   * @param remappedProbe The remapped IDs (host data)
   * @param buildKeys The original build keys (host data) 
   * @param remappedBuild The build remapped IDs (host data)
   * @param sentinelValue Expected sentinel value for unmatched keys
   * @param nullMode The null equality mode being tested
   */
  private void verifyProbeRemapping(HostColumnVector[] probeKeys,
                                    HostColumnVector remappedProbe,
                                    HostColumnVector[] buildKeys,
                                    HostColumnVector remappedBuild,
                                    int sentinelValue,
                                    KeyRemapping.NullEqualityMode nullMode) {
    int probeRowCount = (int) remappedProbe.getRowCount();
    int buildRowCount = (int) remappedBuild.getRowCount();
    
    // Build a map from build keys to their assigned IDs
    Map<Key, Integer> buildKeyToId = new HashMap<>();
    for (int i = 0; i < buildRowCount; i++) {
      Object[] keyValues = new Object[buildKeys.length];
      boolean hasTopLevelNull = false;
      for (int col = 0; col < buildKeys.length; col++) {
        if (buildKeys[col].isNull(i)) {
          keyValues[col] = null;
          hasTopLevelNull = true;
        } else if (buildKeys[col].getType().equals(ai.rapids.cudf.DType.STRING)) {
          keyValues[col] = buildKeys[col].getJavaString(i);
        } else if (buildKeys[col].getType().equals(ai.rapids.cudf.DType.INT32)) {
          keyValues[col] = buildKeys[col].getInt(i);
        } else {
          throw new IllegalArgumentException("Unsupported type: " + buildKeys[col].getType());
        }
      }
      
      Key key = new Key(keyValues);
      int buildId = remappedBuild.getInt(i);
      
      // Only add to map if it's a valid ID (not a sentinel)
      // In SPARK_EQUALITY/NULL_NOT_EQUAL: null keys get BUILD_NULL_SENTINEL (negative)
      // In NULL_EQUAL: null keys get valid non-negative IDs
      if (buildId >= 0) {
        buildKeyToId.put(key, buildId);
      }
    }
    
    // Verify each probe key
    for (int i = 0; i < probeRowCount; i++) {
      Object[] keyValues = new Object[probeKeys.length];
      boolean hasTopLevelNull = false;
      for (int col = 0; col < probeKeys.length; col++) {
        if (probeKeys[col].isNull(i)) {
          keyValues[col] = null;
          hasTopLevelNull = true;
        } else if (probeKeys[col].getType().equals(ai.rapids.cudf.DType.STRING)) {
          keyValues[col] = probeKeys[col].getJavaString(i);
        } else if (probeKeys[col].getType().equals(ai.rapids.cudf.DType.INT32)) {
          keyValues[col] = probeKeys[col].getInt(i);
        } else {
          throw new IllegalArgumentException("Unsupported type: " + probeKeys[col].getType());
        }
      }
      Key key = new Key(keyValues);
      
      // Output never has nulls - uses sentinel values instead
      assertFalse(remappedProbe.isNull(i),
          "Probe remapping output should never be null (uses sentinels)");
      
      int probeId = remappedProbe.getInt(i);
      
      if (buildKeyToId.containsKey(key)) {
        // Key exists in build side with a valid ID - should match
        assertEquals(buildKeyToId.get(key), probeId,
            "Probe key " + key + " should have same ID as in build side");
      } else {
        // Key doesn't exist in build side (or null when nulls not equal) - should be sentinel
        assertEquals(sentinelValue, probeId,
            "Probe key " + key + " not in build side should have sentinel value");
      }
    }
  }

  @Test
  public void testBasicRemapping() {
    assertTrue(SENTINEL < 0, "Sentinel should be negative");
    // Build: [10, 20, 30]
    // Probe: [20, 30, 40]
    try (ColumnVector buildCol = ColumnVector.fromInts(10, 20, 30);
         ColumnVector probeCol = ColumnVector.fromInts(20, 30, 40);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {

      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap, true);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap, false);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostProbeCol = probeCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {

        // Verify build side mapping
        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild);
        
        // Verify probe side mapping
        verifyProbeRemapping(new HostColumnVector[]{hostProbeCol}, hostProbe,
                            new HostColumnVector[]{hostBuildCol}, hostBuild,
                            SENTINEL);
      }
    }
  }

  @Test
  public void testRemappingMultiColumn() {
    // Test remapping with multi-column keys
    // Build: [(1,10), (2,20), (3,30)]
    // Probe: [(2,20), (3,30), (4,40)]
    try (ColumnVector buildCol1 = ColumnVector.fromInts(1, 2, 3);
         ColumnVector buildCol2 = ColumnVector.fromInts(10, 20, 30);
         ColumnVector probeCol1 = ColumnVector.fromInts(2, 3, 4);
         ColumnVector probeCol2 = ColumnVector.fromInts(20, 30, 40);
         Table buildKeys = new Table(buildCol1, buildCol2);
         Table probeKeys = new Table(probeCol1, probeCol2);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {

      
      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap, true);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap, false);
           HostColumnVector hostBuildCol1 = buildCol1.copyToHost();
           HostColumnVector hostBuildCol2 = buildCol2.copyToHost();
           HostColumnVector hostProbeCol1 = probeCol1.copyToHost();
           HostColumnVector hostProbeCol2 = probeCol2.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {

        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol1, hostBuildCol2}, 
                            hostBuild);
        verifyProbeRemapping(new HostColumnVector[]{hostProbeCol1, hostProbeCol2}, hostProbe,
                            new HostColumnVector[]{hostBuildCol1, hostBuildCol2}, hostBuild,
                            SENTINEL);
      }
    }
  }

  @Test
  public void testRemappingEmpty() {
    // Test with empty probe table
    try (ColumnVector buildCol = ColumnVector.fromInts(1, 2, 3);
         ColumnVector probeCol = ColumnVector.fromInts();
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {


      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap, true);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap, false);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {

        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild);
        assertEquals(0, hostProbe.getRowCount());
      }
    }
  }

  @Test
  public void testRemappingReuseStructures() {
    // Verify that RemapStructures can be reused for multiple probes
    try (ColumnVector buildCol = ColumnVector.fromInts(1, 2, 3);
         Table buildKeys = new Table(buildCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {


      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap, true);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost()) {

        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild);

        // First probe
        try (ColumnVector probeCol1 = ColumnVector.fromInts(1, 2);
             Table probeKeys1 = new Table(probeCol1);
             ColumnVector remappedProbe1 = KeyRemapping.applyRemapping(probeKeys1, remap, false);
             HostColumnVector hostProbeCol1 = probeCol1.copyToHost();
             HostColumnVector hostProbe1 = remappedProbe1.copyToHost()) {
          verifyProbeRemapping(new HostColumnVector[]{hostProbeCol1}, hostProbe1,
                              new HostColumnVector[]{hostBuildCol}, hostBuild,
                              SENTINEL);
        }

        // Second probe with different keys
        try (ColumnVector probeCol2 = ColumnVector.fromInts(2, 3);
             Table probeKeys2 = new Table(probeCol2);
             ColumnVector remappedProbe2 = KeyRemapping.applyRemapping(probeKeys2, remap, false);
             HostColumnVector hostProbeCol2 = probeCol2.copyToHost();
             HostColumnVector hostProbe2 = remappedProbe2.copyToHost()) {
          verifyProbeRemapping(new HostColumnVector[]{hostProbeCol2}, hostProbe2,
                              new HostColumnVector[]{hostBuildCol}, hostBuild,
                              SENTINEL);
        }
      }
    }
  }

  @Test
  public void testRemappingResourceManagement() {
    // Verify that RemapStructures properly manages resources
    KeyRemapping.RemapStructures remap;
    try (ColumnVector buildCol = ColumnVector.fromInts(1, 2, 3);
         Table buildKeys = new Table(buildCol)) {
      remap = KeyRemapping.createRemapStructures(buildKeys);
    }

    // Close and verify cleanup
    assertDoesNotThrow(() -> remap.close());
  }

  // ==================== EXTENDED TESTS ====================

  @Test
  public void testRemappingLargeKeys() {
    // Test with larger dataset
    int[] buildData = new int[1000];
    for (int i = 0; i < 1000; i++) {
      buildData[i] = i * 2;  // Even numbers
    }

    int[] probeData = new int[500];
    for (int i = 0; i < 500; i++) {
      probeData[i] = i * 2 + 1;  // Odd numbers (no matches)
    }

    try (ColumnVector buildCol = ColumnVector.fromInts(buildData);
         ColumnVector probeCol = ColumnVector.fromInts(probeData);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {


      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap, true);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap, false);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostProbeCol = probeCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {

        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild);
        verifyProbeRemapping(new HostColumnVector[]{hostProbeCol}, hostProbe,
                            new HostColumnVector[]{hostBuildCol}, hostBuild,
                            SENTINEL);
      }
    }
  }

  @Test
  public void testRemappingStrings() {
    // Test with string keys
    try (ColumnVector buildCol = ColumnVector.fromStrings("apple", "banana", "cherry");
         ColumnVector probeCol = ColumnVector.fromStrings("banana", "cherry", "durian");
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {


      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap, true);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap, false);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostProbeCol = probeCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {

        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild);
        verifyProbeRemapping(new HostColumnVector[]{hostProbeCol}, hostProbe,
                            new HostColumnVector[]{hostBuildCol}, hostBuild,
                            SENTINEL);
      }
    }
  }

  @Test
  public void testRemappingHighCardinality() {
    // Test with high cardinality (many distinct values)
    int[] buildData = new int[10000];
    for (int i = 0; i < 10000; i++) {
      buildData[i] = i;
    }

    try (ColumnVector buildCol = ColumnVector.fromInts(buildData);
         Table buildKeys = new Table(buildCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {


      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap, true);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost()) {

        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild);
      }
    }
  }

  @Test
  public void testRemappingLowCardinality() {
    // Test with low cardinality (many duplicates)
    int[] buildData = new int[10000];
    for (int i = 0; i < 10000; i++) {
      buildData[i] = i % 10;  // Only 10 distinct values
    }

    try (ColumnVector buildCol = ColumnVector.fromInts(buildData);
         Table buildKeys = new Table(buildCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {


      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap, true);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost()) {

        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild);
      }
    }
  }

  @Test
  public void testRemappingEmptyBuildTable() {
    // Test with empty build table
    try (ColumnVector buildCol = ColumnVector.fromInts();
         ColumnVector probeCol = ColumnVector.fromInts(1, 2, 3);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {


      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap, true);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap, false);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostProbeCol = probeCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {

        // Verify using helper (empty build table)
        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild);
        verifyProbeRemapping(new HostColumnVector[]{hostProbeCol}, hostProbe,
                            new HostColumnVector[]{hostBuildCol}, hostBuild,
                            SENTINEL);
      }
    }
  }

  @Test
  public void testRemappingClosedStructuresError() {
    // Test that using closed RemapStructures throws an error
    KeyRemapping.RemapStructures remap;
    try (ColumnVector buildCol = ColumnVector.fromInts(1, 2, 3);
         Table buildKeys = new Table(buildCol)) {
      remap = KeyRemapping.createRemapStructures(buildKeys);
      remap.close();
    }

    // Trying to use closed RemapStructures should throw
    KeyRemapping.RemapStructures finalRemap = remap;
    assertThrows(IllegalStateException.class, () -> finalRemap.getNativeHandle());
  }

  @Test
  public void testRemappingMultipleClose() {
    // Test that closing RemapStructures multiple times is safe
    try (ColumnVector buildCol = ColumnVector.fromInts(1, 2, 3);
         Table buildKeys = new Table(buildCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {
      
      remap.close();
      // Second close should be safe
      assertDoesNotThrow(() -> remap.close());
    }
  }

  @Test
  public void testRemappingMixedMultiColumn() {
    // Test remapping with mixed-type multi-column keys (int and string)
    try (ColumnVector buildCol1 = ColumnVector.fromInts(1, 2, 2, 3);
         ColumnVector buildCol2 = ColumnVector.fromStrings("a", "b", "b", "c");
         ColumnVector probeCol1 = ColumnVector.fromInts(2, 3, 4);
         ColumnVector probeCol2 = ColumnVector.fromStrings("b", "c", "d");
         Table buildKeys = new Table(buildCol1, buildCol2);
         Table probeKeys = new Table(probeCol1, probeCol2);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {

      // Distinct keys: (1,a), (2,b), (3,c) = 3

      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap, true);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap, false);
           HostColumnVector hostBuildCol1 = buildCol1.copyToHost();
           HostColumnVector hostBuildCol2 = buildCol2.copyToHost();
           HostColumnVector hostProbeCol1 = probeCol1.copyToHost();
           HostColumnVector hostProbeCol2 = probeCol2.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {

        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol1, hostBuildCol2}, 
                            hostBuild);
        verifyProbeRemapping(new HostColumnVector[]{hostProbeCol1, hostProbeCol2}, hostProbe,
                            new HostColumnVector[]{hostBuildCol1, hostBuildCol2}, hostBuild,
                            SENTINEL);
      }
    }
  }

  @Test
  public void testRemappingWithNullsInMultiColumn() {
    // Test multi-column keys with nulls in various columns
    try (ColumnVector buildCol1 = ColumnVector.fromBoxedInts(1, null, 2);
         ColumnVector buildCol2 = ColumnVector.fromBoxedInts(10, 20, null);
         ColumnVector probeCol1 = ColumnVector.fromBoxedInts(null, 2, 3);
         ColumnVector probeCol2 = ColumnVector.fromBoxedInts(20, null, 30);
         Table buildKeys = new Table(buildCol1, buildCol2);
         Table probeKeys = new Table(probeCol1, probeCol2);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {


      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap, true);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap, false);
           HostColumnVector hostBuildCol1 = buildCol1.copyToHost();
           HostColumnVector hostBuildCol2 = buildCol2.copyToHost();
           HostColumnVector hostProbeCol1 = probeCol1.copyToHost();
           HostColumnVector hostProbeCol2 = probeCol2.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {

        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol1, hostBuildCol2}, 
                            hostBuild);
        verifyProbeRemapping(new HostColumnVector[]{hostProbeCol1, hostProbeCol2}, hostProbe,
                            new HostColumnVector[]{hostBuildCol1, hostBuildCol2}, hostBuild,
                            SENTINEL);
      }
    }
  }

  @Test
  public void testRemappingSentinelIsNegative() {
    // Verify sentinel is always negative
    int sentinel = KeyRemapping.getNotFoundSentinel();
    assertTrue(sentinel < 0, "Sentinel value should be negative, got: " + sentinel);
    
    // Verify it's consistent
    assertEquals(sentinel, KeyRemapping.getNotFoundSentinel(),
        "Sentinel should be consistent across calls");
  }

  @Test
  public void testRemappingEmptyStrings() {
    // Test with empty strings
    try (ColumnVector buildCol = ColumnVector.fromStrings("", "a", "");
         ColumnVector probeCol = ColumnVector.fromStrings("", "b");
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {

      // Distinct: "", "a" = 2

      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap, true);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap, false);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostProbeCol = probeCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {

        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild);
        verifyProbeRemapping(new HostColumnVector[]{hostProbeCol}, hostProbe,
                            new HostColumnVector[]{hostBuildCol}, hostBuild,
                            SENTINEL);
      }
    }
  }

  @Test
  public void testDumpRemapTable() {
    // Test the dump functionality with a simple case
    try (ColumnVector buildCol = ColumnVector.fromInts(10, 20, 10, 30, 20);
         Table buildKeys = new Table(buildCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {

      // Build has 5 rows with 3 distinct values: 10, 20, 30
      // Expected distinct entries: 3

      try (Table dumpTable = KeyRemapping.dumpRemapTable(remap)) {
        // Verify table structure
        assertEquals(3, dumpTable.getNumberOfColumns(), 
            "Dump table should have 3 columns: hash, key_row_index, mapped_value");
        
        // Copy to host for inspection
        try (HostColumnVector hashCol = dumpTable.getColumn(0).copyToHost();
             HostColumnVector keyRowIdxCol = dumpTable.getColumn(1).copyToHost();
             HostColumnVector mappedValueCol = dumpTable.getColumn(2).copyToHost()) {

          int numEntries = (int) hashCol.getRowCount();
          
          // Should have 3 distinct entries
          assertEquals(3, numEntries, "Should have 3 distinct keys in the map");

          // Track which row indices we've seen and their mapped values
          Map<Integer, Integer> rowIdxToMappedValue = new HashMap<>();
          Set<Integer> seenHashes = new HashSet<>();
          Set<Integer> seenMappedValues = new HashSet<>();

          for (int i = 0; i < numEntries; i++) {
            // Extract values
            int hash = hashCol.getInt(i);
            int keyRowIdx = keyRowIdxCol.getInt(i);
            int mappedValue = mappedValueCol.getInt(i);

            // Verify hash is non-zero (valid hash)
            assertNotEquals(0, hash, "Hash should not be zero");
            seenHashes.add(hash);

            // Verify key row index is valid (within bounds of build table)
            assertTrue(keyRowIdx >= 0 && keyRowIdx < 5, 
                "Key row index should be within build table bounds [0, 5)");

            // Verify mapped value is non-negative
            assertTrue(mappedValue >= 0, "Mapped value should be non-negative");
            
            // Store mapping
            rowIdxToMappedValue.put(keyRowIdx, mappedValue);
            seenMappedValues.add(mappedValue);
          }

          // Verify we have 3 unique hashes (one per distinct key)
          assertEquals(3, seenHashes.size(), "Should have 3 unique hashes");
          
          // Verify we have 3 unique mapped values
          assertEquals(3, seenMappedValues.size(), "Should have 3 unique mapped values");

          // Verify the key row indices correspond to one example of each distinct value
          // With lowest-index-wins, we expect to see indices: 0 (for 10), 1 (for 20), 3 (for 30)
          assertTrue(rowIdxToMappedValue.containsKey(0), "Should see row index 0 (first 10)");
          assertTrue(rowIdxToMappedValue.containsKey(1), "Should see row index 1 (first 20)");
          assertTrue(rowIdxToMappedValue.containsKey(3), "Should see row index 3 (first 30)");
          
          System.out.println("Dump table contents:");
          for (int i = 0; i < numEntries; i++) {
            System.out.println(String.format("  hash=%d, key_row_index=%d, mapped_value=%d",
                hashCol.getInt(i), keyRowIdxCol.getInt(i), mappedValueCol.getInt(i)));
          }
        }
      }
    }
  }

  @Test
  public void testDumpRemapTableEmpty() {
    // Test dump with empty build table
    try (ColumnVector buildCol = ColumnVector.fromInts();
         Table buildKeys = new Table(buildCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {

      try (Table dumpTable = KeyRemapping.dumpRemapTable(remap)) {
        // Verify table structure
        assertEquals(3, dumpTable.getNumberOfColumns(), 
            "Dump table should have 3 columns even when empty");
        assertEquals(0, dumpTable.getRowCount(), "Dump table should have 0 rows for empty build");
      }
    }
  }

  @Test
  public void testSparkEqualityTopLevelNulls() {
    // Test that top-level nulls are NOT equal in SPARK_EQUALITY mode (default)
    // Build: [1, null, null, 2]
    // Probe: [1, null, 2, 3]
    // Expected: matching keys get same IDs, nulls get sentinels
    try (ColumnVector buildCol = ColumnVector.fromBoxedInts(1, null, null, 2);
         ColumnVector probeCol = ColumnVector.fromBoxedInts(1, null, 2, 3);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {

      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap, true);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap, false);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostProbeCol = probeCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {

        // Verify build side: non-null keys get valid IDs, nulls get BUILD_NULL_SENTINEL
        assertFalse(hostBuild.isNull(0), "Build result should not be null for non-null key");
        assertFalse(hostBuild.isNull(1), "Build result should not be null (uses sentinel instead)");
        assertFalse(hostBuild.isNull(2), "Build result should not be null (uses sentinel instead)");
        assertFalse(hostBuild.isNull(3), "Build result should not be null for non-null key");
        
        int buildKey1Id = hostBuild.getInt(0);
        int buildNull1Sentinel = hostBuild.getInt(1);
        int buildNull2Sentinel = hostBuild.getInt(2);
        int buildKey2Id = hostBuild.getInt(3);
        
        // Build-side nulls should get BUILD_NULL_SENTINEL
        assertEquals(KeyRemapping.getBuildNullSentinel(), buildNull1Sentinel, 
            "Build null should get BUILD_NULL_SENTINEL");
        assertEquals(KeyRemapping.getBuildNullSentinel(), buildNull2Sentinel, 
            "Build null should get BUILD_NULL_SENTINEL");
        
        // Non-null build keys should get non-negative IDs
        assertTrue(buildKey1Id >= 0, "Non-null build key should get non-negative ID");
        assertTrue(buildKey2Id >= 0, "Non-null build key should get non-negative ID");

        // Verify probe side: matching keys get same ID, non-matching get NOT_FOUND_SENTINEL
        assertFalse(hostProbe.isNull(0), "Probe result should not be null");
        assertFalse(hostProbe.isNull(1), "Probe result should not be null (uses sentinel instead)");
        assertFalse(hostProbe.isNull(2), "Probe result should not be null");
        assertFalse(hostProbe.isNull(3), "Probe result should not be null (uses sentinel instead)");
        
        // Probe key 1 should match build key 1
        assertEquals(buildKey1Id, hostProbe.getInt(0), "Probe key 1 should match build");
        
        // Probe null should NOT match (top-level nulls not equal) - gets NOT_FOUND_SENTINEL
        assertEquals(KeyRemapping.getNotFoundSentinel(), hostProbe.getInt(1), 
            "Probe null should not match build (top-level nulls not equal)");
        
        // Probe key 2 should match build key 2
        assertEquals(buildKey2Id, hostProbe.getInt(2), "Probe key 2 should match build");
        
        // Probe key 3 not in build - gets NOT_FOUND_SENTINEL
        assertEquals(KeyRemapping.getNotFoundSentinel(), hostProbe.getInt(3), 
            "Probe key 3 should not be found");
        
        // Also verify using helper methods for overall correctness
        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild);
        verifyProbeRemapping(new HostColumnVector[]{hostProbeCol}, hostProbe,
                            new HostColumnVector[]{hostBuildCol}, hostBuild,
                            SENTINEL);
      }
    }
  }

  @Test
  public void testSparkEqualityNestedNulls() {
    // Test that nested nulls (inside structs) ARE equal in SPARK_EQUALITY mode
    // Build: [struct(1, null), struct(2, 3), struct(1, null)]
    // Expected: struct(1, null) appears twice - should be treated as equal and deduplicate
    // Probe: [struct(1, null), struct(2, 3), struct(3, null)]
    try (ColumnVector child1Build = ColumnVector.fromBoxedInts(1, 2, 1);
         ColumnVector child2Build = ColumnVector.fromBoxedInts(null, 3, null);
         ColumnVector buildCol = ColumnVector.makeStruct(child1Build, child2Build);
         ColumnVector child1Probe = ColumnVector.fromBoxedInts(1, 2, 3);
         ColumnVector child2Probe = ColumnVector.fromBoxedInts(null, 3, null);
         ColumnVector probeCol = ColumnVector.makeStruct(child1Probe, child2Probe);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {

      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap, true);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap, false);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostProbeCol = probeCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {
        
        // Verify: struct(1,null) at rows 0 and 2 should have same ID (nested nulls are equal)
        assertFalse(hostBuild.isNull(0), "Build result should not be null");
        assertFalse(hostBuild.isNull(1), "Build result should not be null");
        assertFalse(hostBuild.isNull(2), "Build result should not be null");
        
        int id0 = hostBuild.getInt(0);  // struct(1, null)
        int id1 = hostBuild.getInt(1);  // struct(2, 3)
        int id2 = hostBuild.getInt(2);  // struct(1, null) again
        
        assertEquals(id0, id2, "struct(1,null) should map to same ID (nested nulls ARE equal)");
        assertNotEquals(id0, id1, "Different structs should have different IDs");
        
        // Verify probe side: struct(1,null) should match, struct(3,null) should not
        assertFalse(hostProbe.isNull(0), "Probe result should not be null");
        assertFalse(hostProbe.isNull(1), "Probe result should not be null");
        assertFalse(hostProbe.isNull(2), "Probe result should not be null");
        
        assertEquals(id0, hostProbe.getInt(0), "Probe struct(1,null) should match build");
        assertEquals(id1, hostProbe.getInt(1), "Probe struct(2,3) should match build");
        assertEquals(KeyRemapping.getNotFoundSentinel(), hostProbe.getInt(2), 
            "Probe struct(3,null) should not be found");
        
        // Note: Helper methods don't support struct types, manual verification is appropriate here
      }
    }
  }

  @Test
  public void testSentinelValuesAreDistinct() {
    // Verify that NOT_FOUND_SENTINEL and BUILD_NULL_SENTINEL are different values
    int notFound = KeyRemapping.getNotFoundSentinel();
    int buildNull = KeyRemapping.getBuildNullSentinel();
    
    assertNotEquals(notFound, buildNull, 
        "NOT_FOUND_SENTINEL and BUILD_NULL_SENTINEL must be distinct");
    assertTrue(notFound < 0, "NOT_FOUND_SENTINEL should be negative");
    assertTrue(buildNull < 0, "BUILD_NULL_SENTINEL should be negative");
  }

  @Test
  public void testSentinelAssignmentBuildVsProbe() {
    // Test that nulls get different sentinels for build vs probe side
    // Build: [null, 1]
    // Probe: [null, 1]
    try (ColumnVector buildCol = ColumnVector.fromBoxedInts(null, 1);
         ColumnVector probeCol = ColumnVector.fromBoxedInts(null, 1);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {

      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap, true);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap, false);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostProbeCol = probeCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {
        
        // Build side: null gets BUILD_NULL_SENTINEL
        assertEquals(KeyRemapping.getBuildNullSentinel(), hostBuild.getInt(0), 
            "Build-side null should get BUILD_NULL_SENTINEL");
        
        // Probe side: null gets NOT_FOUND_SENTINEL (because top-level nulls don't match)
        assertEquals(KeyRemapping.getNotFoundSentinel(), hostProbe.getInt(0), 
            "Probe-side null should get NOT_FOUND_SENTINEL (no match)");
        
        // Also verify using helper methods for overall correctness
        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild);
        verifyProbeRemapping(new HostColumnVector[]{hostProbeCol}, hostProbe,
                            new HostColumnVector[]{hostBuildCol}, hostBuild,
                            SENTINEL);
      }
    }
  }

  @Test
  public void testNullEqualMode() {
    // Test NULL_EQUAL mode: all nulls (including top-level) are equal
    // Build: [null, 1, null]
    // Probe: [null, 1, 2]
    try (ColumnVector buildCol = ColumnVector.fromBoxedInts(null, 1, null);
         ColumnVector probeCol = ColumnVector.fromBoxedInts(null, 1, 2);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys, 
             KeyRemapping.NullEqualityMode.NULL_EQUAL)) {

      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap, true);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap, false);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostProbeCol = probeCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {
        
        // In NULL_EQUAL mode, nulls are treated as equal values
        assertFalse(hostBuild.isNull(0), "Build result should not be null");
        assertFalse(hostBuild.isNull(1), "Build result should not be null");
        assertFalse(hostBuild.isNull(2), "Build result should not be null");
        
        int nullId1 = hostBuild.getInt(0);  // null
        int key1Id = hostBuild.getInt(1);   // 1
        int nullId2 = hostBuild.getInt(2);  // null again
        
        // Null keys at rows 0 and 2 should map to same ID (nulls are equal)
        assertEquals(nullId1, nullId2, "Nulls should map to same ID in NULL_EQUAL mode");
        
        // Probe: null should match build null, key 1 should match, key 2 not found
        assertFalse(hostProbe.isNull(0), "Probe result should not be null");
        assertEquals(nullId1, hostProbe.getInt(0), "Probe null should match build null");
        assertEquals(key1Id, hostProbe.getInt(1), "Probe key 1 should match build");
        assertEquals(KeyRemapping.getNotFoundSentinel(), hostProbe.getInt(2), 
            "Probe key 2 should not be found");
        
        // Also verify using helper methods for overall correctness
        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild, 
                            KeyRemapping.NullEqualityMode.NULL_EQUAL);
        verifyProbeRemapping(new HostColumnVector[]{hostProbeCol}, hostProbe,
                            new HostColumnVector[]{hostBuildCol}, hostBuild,
                            SENTINEL, KeyRemapping.NullEqualityMode.NULL_EQUAL);
      }
    }
  }

  @Test
  public void testNullNotEqualMode() {
    // Test NULL_NOT_EQUAL mode: no nulls are equal (even nested nulls)
    // Behaves the same as SPARK_EQUALITY for top-level nulls
    // Build: [null, 1, null]
    // Probe: [null, 1, 2]
    try (ColumnVector buildCol = ColumnVector.fromBoxedInts(null, 1, null);
         ColumnVector probeCol = ColumnVector.fromBoxedInts(null, 1, 2);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys, 
             KeyRemapping.NullEqualityMode.NULL_NOT_EQUAL)) {

      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap, true);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap, false);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostProbeCol = probeCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {
        
        // Build side: nulls get BUILD_NULL_SENTINEL, non-nulls get valid IDs
        assertFalse(hostBuild.isNull(0), "Build result should not be null");
        assertFalse(hostBuild.isNull(1), "Build result should not be null");
        assertFalse(hostBuild.isNull(2), "Build result should not be null");
        
        assertEquals(KeyRemapping.getBuildNullSentinel(), hostBuild.getInt(0), 
            "Build null should get BUILD_NULL_SENTINEL in NULL_NOT_EQUAL mode");
        assertEquals(KeyRemapping.getBuildNullSentinel(), hostBuild.getInt(2), 
            "Build null should get BUILD_NULL_SENTINEL in NULL_NOT_EQUAL mode");
        
        int key1Id = hostBuild.getInt(1);
        assertTrue(key1Id >= 0, "Non-null build key should get non-negative ID");

        // Probe side: nulls get NOT_FOUND_SENTINEL (nulls are never equal)
        assertFalse(hostProbe.isNull(0), "Probe result should not be null");
        assertFalse(hostProbe.isNull(1), "Probe result should not be null");
        assertFalse(hostProbe.isNull(2), "Probe result should not be null");
        
        assertEquals(KeyRemapping.getNotFoundSentinel(), hostProbe.getInt(0), 
            "Probe null should get NOT_FOUND_SENTINEL (nulls are not equal)");
        assertEquals(key1Id, hostProbe.getInt(1), "Probe key 1 should match build");
        assertEquals(KeyRemapping.getNotFoundSentinel(), hostProbe.getInt(2), 
            "Probe key 2 should not be found");
        
        // Also verify using helper methods for overall correctness
        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild, 
                            KeyRemapping.NullEqualityMode.NULL_NOT_EQUAL);
        verifyProbeRemapping(new HostColumnVector[]{hostProbeCol}, hostProbe,
                            new HostColumnVector[]{hostBuildCol}, hostBuild,
                            SENTINEL, KeyRemapping.NullEqualityMode.NULL_NOT_EQUAL);
      }
    }
  }
}
