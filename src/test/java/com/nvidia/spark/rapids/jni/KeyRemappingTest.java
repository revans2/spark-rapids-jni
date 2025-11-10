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
   * Verify that the build side remapping is correct (defaults to nullsEqual=true).
   * 
   * @param buildKeys The original build keys (host data)
   * @param remappedBuild The remapped IDs (host data)
   */
  private void verifyBuildRemapping(HostColumnVector[] buildKeys,
                                    HostColumnVector remappedBuild) {
    verifyBuildRemapping(buildKeys, remappedBuild, true);
  }

  /**
   * Verify that the build side remapping is correct.
   * 
   * @param buildKeys The original build keys (host data)
   * @param remappedBuild The remapped IDs (host data)
   * @param nullsEqual Whether nulls are considered equal
   */
  private void verifyBuildRemapping(HostColumnVector[] buildKeys,
                                    HostColumnVector remappedBuild,
                                    boolean nullsEqual) {
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
      
      // With nullsEqual=false, null keys cannot be matched (null != null), so they get sentinel
      boolean isNullKey = false;
      for (Object val : keyValues) {
        if (val == null) {
          isNullKey = true;
          break;
        }
      }
      
      if (!nullsEqual && isNullKey) {
        // Null keys should get sentinel when nullsEqual=false
        assertEquals(SENTINEL, assignedId,
            "Null key should get sentinel when nullsEqual=false");
      } else {
        // Non-null keys (or nulls with nullsEqual=true) should get valid IDs
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
    }
    
    // Verify different keys got different IDs
    assertEquals(keyToId.size(), assignedIds.size(),
        "Different keys should get different IDs");
  }

  /**
   * Verify that the probe side remapping is correct (defaults to nullsEqual=true).
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
    verifyProbeRemapping(probeKeys, remappedProbe, buildKeys, remappedBuild, sentinelValue, true);
  }

  /**
   * Verify that the probe side remapping is correct and consistent with build side.
   * 
   * @param probeKeys The original probe keys (host data)
   * @param remappedProbe The remapped IDs (host data)
   * @param buildKeys The original build keys (host data) 
   * @param remappedBuild The build remapped IDs (host data)
   * @param sentinelValue Expected sentinel value for unmatched keys
   * @param nullsEqual Whether nulls are considered equal
   */
  private void verifyProbeRemapping(HostColumnVector[] probeKeys,
                                    HostColumnVector remappedProbe,
                                    HostColumnVector[] buildKeys,
                                    HostColumnVector remappedBuild,
                                    int sentinelValue,
                                    boolean nullsEqual) {
    int probeRowCount = (int) remappedProbe.getRowCount();
    int buildRowCount = (int) remappedBuild.getRowCount();
    
    // Build a map from build keys to their assigned IDs
    Map<Key, Integer> buildKeyToId = new HashMap<>();
    for (int i = 0; i < buildRowCount; i++) {
      Object[] keyValues = new Object[buildKeys.length];
      boolean hasNull = false;
      for (int col = 0; col < buildKeys.length; col++) {
        if (buildKeys[col].isNull(i)) {
          keyValues[col] = null;
          hasNull = true;
        } else if (buildKeys[col].getType().equals(ai.rapids.cudf.DType.STRING)) {
          keyValues[col] = buildKeys[col].getJavaString(i);
        } else if (buildKeys[col].getType().equals(ai.rapids.cudf.DType.INT32)) {
          keyValues[col] = buildKeys[col].getInt(i);
        } else {
          throw new IllegalArgumentException("Unsupported type: " + buildKeys[col].getType());
        }
      }
      Key key = new Key(keyValues);
      
      // Skip null keys when nullsEqual=false (they get sentinel, not valid IDs)
      if (!nullsEqual && hasNull) {
        continue;
      }
      
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
      
      // Check if this is a null key
      boolean isNullKey = false;
      for (Object val : keyValues) {
        if (val == null) {
          isNullKey = true;
          break;
        }
      }
      
      // With nullsEqual=false, null keys always get sentinel (can't match anything)
      if (!nullsEqual && isNullKey) {
        assertEquals(sentinelValue, probeId,
            "Null probe key should get sentinel when nullsEqual=false");
      } else if (buildKeyToId.containsKey(key)) {
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

      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
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
  public void testRemappingPreservesMatches() {
    // Verify that keys that should match still match after remapping
    // Build: [1, 2, 3, 4], Probe: [2, 3, 5]
    try (ColumnVector buildCol = ColumnVector.fromInts(1, 2, 3, 4);
         ColumnVector probeCol = ColumnVector.fromInts(2, 3, 5);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {


      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
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
  public void testRemappingWithDuplicates() {
    // Build has duplicates initially, but distinct keys will have unique values
    // Build: [1, 2, 2, 3], Probe: [2, 2, 4]
    try (ColumnVector buildCol = ColumnVector.fromInts(1, 2, 2, 3);
         ColumnVector probeCol = ColumnVector.fromInts(2, 2, 4);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {


      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
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
  public void testRemappingWithNulls() {
    // Test that nulls are handled correctly
    // Build: [1, null, 3], Probe: [null, 3, 4]
    try (ColumnVector buildCol = ColumnVector.fromBoxedInts(1, null, 3);
         ColumnVector probeCol = ColumnVector.fromBoxedInts(null, 3, 4);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {


      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
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

      
      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
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


      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {

        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild);
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


      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
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
  public void testRemappingReuseStructures() {
    // Verify that RemapStructures can be reused for multiple probes
    try (ColumnVector buildCol = ColumnVector.fromInts(1, 2, 3);
         Table buildKeys = new Table(buildCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {


      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost()) {

        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild);

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


      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost()) {

        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild);
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


      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
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


      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
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
  public void testRemappingSentinelValue() {
    // Verify the sentinel value returned for unmatched keys
    try (ColumnVector buildCol = ColumnVector.fromInts(1, 2, 3);
         ColumnVector probeCol = ColumnVector.fromInts(4, 5, 6);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {

      assertEquals(SENTINEL, KeyRemapping.getNotFoundSentinel());

      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostProbeCol = probeCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {

        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild);
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


      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
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


      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
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


      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost()) {

        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild);
      }
    }
  }

  @Test
  public void testRemappingNullsNotEqual() {
    // Test with nullsEqual=false - nulls cannot match (not even to themselves)
    try (ColumnVector buildCol = ColumnVector.fromBoxedInts(1, null, null, 3);
         ColumnVector probeCol = ColumnVector.fromBoxedInts(null, 3);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys, false)) {

      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostProbeCol = probeCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {

        // Verify build and probe remapping with nullsEqual=false
        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild, false);
        verifyProbeRemapping(new HostColumnVector[]{hostProbeCol}, hostProbe,
                            new HostColumnVector[]{hostBuildCol}, hostBuild,
                            SENTINEL, false);
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


      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {

        // All probe keys should get sentinel since build is empty
        for (int i = 0; i < hostProbe.getRowCount(); i++) {
          assertEquals(SENTINEL, hostProbe.getInt(i),
              "Probe key should get sentinel when build table is empty");
        }
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

      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
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


      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
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
  public void testRemappingStringsWithNulls() {
    // Test string keys with nulls
    try (ColumnVector buildCol = ColumnVector.fromStrings("apple", null, "cherry", null);
         ColumnVector probeCol = ColumnVector.fromStrings(null, "cherry", "durian");
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys, true)) {

      // Distinct: "apple", null, "cherry" = 3

      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
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
  public void testRemappingSentinelIsNegative() {
    // Verify sentinel is always negative
    int sentinel = KeyRemapping.getNotFoundSentinel();
    assertTrue(sentinel < 0, "Sentinel value should be negative, got: " + sentinel);
    
    // Verify it's consistent
    assertEquals(sentinel, KeyRemapping.getNotFoundSentinel(),
        "Sentinel should be consistent across calls");
  }

  @Test
  public void testRemappingBuildKeysAllSame() {
    // Test with all build keys being the same value
    try (ColumnVector buildCol = ColumnVector.fromInts(5, 5, 5, 5, 5);
         ColumnVector probeCol = ColumnVector.fromInts(5, 6);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {


      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
           HostColumnVector hostBuildCol = buildCol.copyToHost();
           HostColumnVector hostProbeCol = probeCol.copyToHost();
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {

        verifyBuildRemapping(new HostColumnVector[]{hostBuildCol}, hostBuild);
        
        // All build keys should have same remapped value
        int firstId = hostBuild.getInt(0);
        for (int i = 1; i < hostBuild.getRowCount(); i++) {
          assertEquals(firstId, hostBuild.getInt(i),
              "All identical keys should have same remapped ID");
        }
        
        verifyProbeRemapping(new HostColumnVector[]{hostProbeCol}, hostProbe,
                            new HostColumnVector[]{hostBuildCol}, hostBuild,
                            SENTINEL);
      }
    }
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

      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
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
}
