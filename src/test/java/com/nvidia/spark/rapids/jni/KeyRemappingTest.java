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
   * Verify that the build side remapping is correct.
   * 
   * @param buildKeys The original build keys (host data)
   * @param remappedBuild The remapped IDs (host data)
   * @param distinctCount Expected number of distinct keys
   */
  private void verifyBuildRemapping(HostColumnVector[] buildKeys,
                                    HostColumnVector remappedBuild,
                                    int distinctCount) {
    int rowCount = (int) remappedBuild.getRowCount();
    assertEquals(rowCount, buildKeys[0].getRowCount());

    // Map from key to its assigned ID
    Map<Key, Integer> keyToId = new HashMap<>();
    Set<Integer> assignedIds = new HashSet<>();
    
    for (int i = 0; i < rowCount; i++) {
      // Extract the key for this row
      Object[] keyValues = new Object[buildKeys.length];
      for (int col = 0; col < buildKeys.length; col++) {
        if (buildKeys[col].isNull(i)) {
          keyValues[col] = null;
        } else if (buildKeys[col].getType().equals(ai.rapids.cudf.DType.STRING)) {
          keyValues[col] = buildKeys[col].getJavaString(i);
        } else if (buildKeys[col].getType().equals(ai.rapids.cudf.DType.INT32)) {
          keyValues[col] = buildKeys[col].getInt(i);
        } else {
          throw new IllegalArgumentException("Unsupported type: " + buildKeys[col].getType());
        }
      }
      Key key = new Key(keyValues);
      
      int assignedId = remappedBuild.getInt(i);
      
      // Verify ID is in range [0, rowCount)
      assertTrue(assignedId >= 0 && assignedId < rowCount,
          "Build key ID " + assignedId + " should be in [0, " + rowCount + ") for key " + key);
      
      // Check consistency: same key should always get same ID
      if (keyToId.containsKey(key)) {
        assertEquals(keyToId.get(key), assignedId,
            "Key " + key + " should always map to same ID");
      } else {
        keyToId.put(key, assignedId);
      }
      
      assignedIds.add(assignedId);
    }
    // Verify we used exactly distinctCount different IDs
    assertEquals(distinctCount, keyToId.size(),
        "Should have exactly " + distinctCount + " distinct keys");
    
    // Verify different keys got different IDs
    assertEquals(keyToId.size(), assignedIds.size(),
        "Different keys should get different IDs");
  }

  /**
   * Verify that the probe side remapping is correct and consistent with build side.
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
    int probeRowCount = (int) remappedProbe.getRowCount();
    int buildRowCount = (int) remappedBuild.getRowCount();
    
    // Build a map from build keys to their assigned IDs
    Map<Key, Integer> buildKeyToId = new HashMap<>();
    for (int i = 0; i < buildRowCount; i++) {
      Object[] keyValues = new Object[buildKeys.length];
      for (int col = 0; col < buildKeys.length; col++) {
        if (buildKeys[col].isNull(i)) {
          keyValues[col] = null;
        } else if (buildKeys[col].getType().equals(ai.rapids.cudf.DType.STRING)) {
          keyValues[col] = buildKeys[col].getJavaString(i);
        } else if (buildKeys[col].getType().equals(ai.rapids.cudf.DType.INT32)) {
          keyValues[col] = buildKeys[col].getInt(i);
        } else {
          throw new IllegalArgumentException("Unsupported type: " + buildKeys[col].getType());
        }
      }
      Key key = new Key(keyValues);
      buildKeyToId.put(key, remappedBuild.getInt(i));
    }
    
    // Verify each probe key
    for (int i = 0; i < probeRowCount; i++) {
      Object[] keyValues = new Object[probeKeys.length];
      for (int col = 0; col < probeKeys.length; col++) {
        if (probeKeys[col].isNull(i)) {
          keyValues[col] = null;
        } else if (probeKeys[col].getType().equals(ai.rapids.cudf.DType.STRING)) {
          keyValues[col] = probeKeys[col].getJavaString(i);
        } else if (probeKeys[col].getType().equals(ai.rapids.cudf.DType.INT32)) {
          keyValues[col] = probeKeys[col].getInt(i);
        } else {
          throw new IllegalArgumentException("Unsupported type: " + probeKeys[col].getType());
        }
      }
      Key key = new Key(keyValues);
      
      int probeId = remappedProbe.getInt(i);
      
      if (buildKeyToId.containsKey(key)) {
        // Key exists in build side - should have same ID
        assertEquals(buildKeyToId.get(key), probeId,
            "Probe key " + key + " should have same ID as in build side");
      } else {
        // Key doesn't exist in build side - should be sentinel
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

      assertEquals(3, remap.getDistinctCount());

      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostProbeCol = probeCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {

        // Verify build side mapping
        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild, 3);
        
        // Verify probe side mapping
        verifyProbeRemapping(new HostColumnVector[]{hostProbeCol}, hostProbe,
                            new HostColumnVector[]{hostBuildCol}, hostBuild,
                            SENTINEL);
      }
    }
  }

  @Test
  public void testRemappingPreservesMatches() {
    // Verify that keys that should match still match after remapping
    // Build: [1, 2, 3, 4], Probe: [2, 3, 5]
    try (ColumnVector buildCol = ColumnVector.fromInts(1, 2, 3, 4);
         ColumnVector probeCol = ColumnVector.fromInts(2, 3, 5);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {

      assertEquals(4, remap.getDistinctCount());

      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostProbeCol = probeCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {

        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild, 4);
        verifyProbeRemapping(new HostColumnVector[]{hostProbeCol}, hostProbe,
                            new HostColumnVector[]{hostBuildCol}, hostBuild,
                            SENTINEL);
      }
    }
  }

  @Test
  public void testRemappingWithDuplicates() {
    // Build has duplicates initially, but distinct keys will have unique values
    // Build: [1, 2, 2, 3], Probe: [2, 2, 4]
    try (ColumnVector buildCol = ColumnVector.fromInts(1, 2, 2, 3);
         ColumnVector probeCol = ColumnVector.fromInts(2, 2, 4);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {

      assertEquals(3, remap.getDistinctCount());

      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostProbeCol = probeCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {

        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild, 3);
        verifyProbeRemapping(new HostColumnVector[]{hostProbeCol}, hostProbe,
                            new HostColumnVector[]{hostBuildCol}, hostBuild,
                            SENTINEL);
      }
    }
  }

  @Test
  public void testRemappingWithNulls() {
    // Test that nulls are handled correctly
    // Build: [1, null, 3], Probe: [null, 3, 4]
    try (ColumnVector buildCol = ColumnVector.fromBoxedInts(1, null, 3);
         ColumnVector probeCol = ColumnVector.fromBoxedInts(null, 3, 4);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {

      assertEquals(3, remap.getDistinctCount());

      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostProbeCol = probeCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {

        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild, 3);
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

      assertEquals(3, remap.getDistinctCount());
      
      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
           HostColumnVector hostBuildCol1 = buildCol1.copyToHost();
           HostColumnVector hostBuildCol2 = buildCol2.copyToHost();
           HostColumnVector hostProbeCol1 = probeCol1.copyToHost();
           HostColumnVector hostProbeCol2 = probeCol2.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {

        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol1, hostBuildCol2}, 
                            hostBuild, 3);
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

      assertEquals(3, remap.getDistinctCount());

      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {

        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild, 3);
        assertEquals(0, hostProbe.getRowCount());
      }
    }
  }

  @Test
  public void testRemappingAllUnmatched() {
    // All probe keys are unmatched
    try (ColumnVector buildCol = ColumnVector.fromInts(1, 2, 3);
         ColumnVector probeCol = ColumnVector.fromInts(4, 5, 6);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {

      assertEquals(3, remap.getDistinctCount());

      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostProbeCol = probeCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {

        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild, 3);
        verifyProbeRemapping(new HostColumnVector[]{hostProbeCol}, hostProbe,
                            new HostColumnVector[]{hostBuildCol}, hostBuild,
                            SENTINEL);
      }
    }
  }

  @Test
  public void testRemappingReuseStructures() {
    // Verify that RemapStructures can be reused for multiple probes
    try (ColumnVector buildCol = ColumnVector.fromInts(1, 2, 3);
         Table buildKeys = new Table(buildCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {

      assertEquals(3, remap.getDistinctCount());

      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost()) {

        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild, 3);

        // First probe
        try (ColumnVector probeCol1 = ColumnVector.fromInts(1, 2);
             Table probeKeys1 = new Table(probeCol1);
             ColumnVector remappedProbe1 = KeyRemapping.applyRemapping(probeKeys1, remap);
             HostColumnVector hostProbeCol1 = probeCol1.copyToHost();
             HostColumnVector hostProbe1 = remappedProbe1.copyToHost()) {
          verifyProbeRemapping(new HostColumnVector[]{hostProbeCol1}, hostProbe1,
                              new HostColumnVector[]{hostBuildCol}, hostBuild,
                              SENTINEL);
        }

        // Second probe with different keys
        try (ColumnVector probeCol2 = ColumnVector.fromInts(2, 3);
             Table probeKeys2 = new Table(probeCol2);
             ColumnVector remappedProbe2 = KeyRemapping.applyRemapping(probeKeys2, remap);
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

    // Verify distinct count is accessible
    assertEquals(3, remap.getDistinctCount());

    // Close and verify cleanup
    assertDoesNotThrow(() -> remap.close());
  }

  // ==================== EXTENDED TESTS ====================

  @Test
  public void testRemappingDenseSequence() {
    // Verify that IDs are assigned as dense sequence 0, 1, 2, ...
    try (ColumnVector buildCol = ColumnVector.fromInts(100, 200, 300, 400);
         Table buildKeys = new Table(buildCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {

      assertEquals(4, remap.getDistinctCount());

      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost()) {

        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild, 4);
      }
    }
  }

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

      assertEquals(1000, remap.getDistinctCount());

      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostProbeCol = probeCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {

        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild, 1000);
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

      assertEquals(3, remap.getDistinctCount());

      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostProbeCol = probeCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {

        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild, 3);
        verifyProbeRemapping(new HostColumnVector[]{hostProbeCol}, hostProbe,
                            new HostColumnVector[]{hostBuildCol}, hostBuild,
                            SENTINEL);
      }
    }
  }

  @Test
  public void testRemappingSentinelValue() {
    // Verify the sentinel value returned for unmatched keys
    try (ColumnVector buildCol = ColumnVector.fromInts(1, 2, 3);
         ColumnVector probeCol = ColumnVector.fromInts(4, 5, 6);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {

      assertEquals(3, remap.getDistinctCount());
      assertEquals(SENTINEL, KeyRemapping.getNotFoundSentinel());

      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostProbeCol = probeCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {

        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild, 3);
        verifyProbeRemapping(new HostColumnVector[]{hostProbeCol}, hostProbe,
                            new HostColumnVector[]{hostBuildCol}, hostBuild,
                            SENTINEL);
        for (int i = 0; i < hostProbe.getRowCount(); i++) {
          assertEquals(SENTINEL, hostProbe.getInt(i));
        }
      }
    }
  }

  @Test
  public void testRemappingNullsEqual() {
    // Test that nullsEqual parameter works correctly
    try (ColumnVector buildCol = ColumnVector.fromBoxedInts(1, null, 3);
         ColumnVector probeCol = ColumnVector.fromBoxedInts(null, null);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys, true)) {

      assertEquals(3, remap.getDistinctCount());

      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostProbeCol = probeCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {

        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild, 3);
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

      assertEquals(10000, remap.getDistinctCount());

      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost()) {

        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild, 10000);
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

      assertEquals(10, remap.getDistinctCount());

      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost()) {

        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild, 10);
      }
    }
  }
}
