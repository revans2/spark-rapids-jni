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

import static org.junit.jupiter.api.Assertions.*;

public class KeyRemappingTest {

  @Test
  public void testBasicRemapping() {
    // Build: [10, 20, 30]
    // Probe: [20, 30, 40]
    // Expected remapping: 10->0, 20->1, 30->2, 40->MIN_VALUE (not in build)
    try (ColumnVector buildCol = ColumnVector.fromInts(10, 20, 30);
         ColumnVector probeCol = ColumnVector.fromInts(20, 30, 40);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {

      // Remap build keys
      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           HostColumnVector hostBuild = remappedBuild.copyToHost()) {
        assertEquals(3, hostBuild.getRowCount());
        // Build keys map to 0, 1, 2
        assertTrue(hostBuild.getInt(0) >= 0 && hostBuild.getInt(0) <= 2);
        assertTrue(hostBuild.getInt(1) >= 0 && hostBuild.getInt(1) <= 2);
        assertTrue(hostBuild.getInt(2) >= 0 && hostBuild.getInt(2) <= 2);
        // All different
        assertTrue(hostBuild.getInt(0) != hostBuild.getInt(1));
        assertTrue(hostBuild.getInt(1) != hostBuild.getInt(2));
        assertTrue(hostBuild.getInt(0) != hostBuild.getInt(2));
      }

      // Remap probe keys
      try (ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {
        assertEquals(3, hostProbe.getRowCount());
        // probe[0]=20, probe[1]=30 should map to valid integers
        assertTrue(hostProbe.getInt(0) >= 0 && hostProbe.getInt(0) <= 2);
        assertTrue(hostProbe.getInt(1) >= 0 && hostProbe.getInt(1) <= 2);
        // probe[2]=40 should map to MIN_VALUE (sentinel)
        assertEquals(Integer.MIN_VALUE, hostProbe.getInt(2));
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
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {

        // build[1]=2 and probe[0]=2 should map to the same integer
        assertEquals(hostBuild.getInt(1), hostProbe.getInt(0));
        
        // build[2]=3 and probe[1]=3 should map to the same integer
        assertEquals(hostBuild.getInt(2), hostProbe.getInt(1));
        
        // probe[2]=5 has no match, should be MIN_VALUE
        assertEquals(Integer.MIN_VALUE, hostProbe.getInt(2));
      }
    }
  }

  @Test
  public void testRemappingWithDuplicates() {
    // Build has duplicates initially, but distinctKeys will have unique values
    // Build: [1, 2, 2, 3], Probe: [2, 2, 4]
    try (ColumnVector buildCol = ColumnVector.fromInts(1, 2, 2, 3);
         ColumnVector probeCol = ColumnVector.fromInts(2, 2, 4);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {

      // Distinct keys should be [1, 2, 3] = 3 distinct values
      assertEquals(3, remap.getDistinctKeys().getRowCount());

      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {

        // build[1]=2 and build[2]=2 should map to the same integer
        assertEquals(hostBuild.getInt(1), hostBuild.getInt(2));
        
        // probe[0]=2 and probe[1]=2 should map to the same integer
        assertEquals(hostProbe.getInt(0), hostProbe.getInt(1));
        
        // All 2's should map to the same value
        assertEquals(hostBuild.getInt(1), hostProbe.getInt(0));
        
        // probe[2]=4 should be MIN_VALUE
        assertEquals(Integer.MIN_VALUE, hostProbe.getInt(2));
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
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {

        // build[1]=null and probe[0]=null should map to the same integer
        assertEquals(hostBuild.getInt(1), hostProbe.getInt(0));
        assertTrue(hostBuild.getInt(1) >= 0 && hostBuild.getInt(1) <= 2);
        
        // build[2]=3 and probe[1]=3 should map to the same integer
        assertEquals(hostBuild.getInt(2), hostProbe.getInt(1));
        
        // probe[2]=4 should be MIN_VALUE
        assertEquals(Integer.MIN_VALUE, hostProbe.getInt(2));
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

      assertEquals(3, remap.getDistinctKeys().getRowCount());
      
      try (ColumnVector remappedBuild = KeyRemapping.applyRemapping(buildKeys, remap);
           ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
           HostColumnVector hostBuild = remappedBuild.copyToHost();
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {

        // build[1]=(2,20) and probe[0]=(2,20) should map to same integer
        assertEquals(hostBuild.getInt(1), hostProbe.getInt(0));
        
        // build[2]=(3,30) and probe[1]=(3,30) should map to same integer
        assertEquals(hostBuild.getInt(2), hostProbe.getInt(1));
        
        // probe[2]=(4,40) should be MIN_VALUE
        assertEquals(Integer.MIN_VALUE, hostProbe.getInt(2));
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

      try (ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap)) {
        assertEquals(0, remappedProbe.getRowCount());
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

      try (ColumnVector remappedProbe = KeyRemapping.applyRemapping(probeKeys, remap);
           HostColumnVector hostProbe = remappedProbe.copyToHost()) {
        // All probe keys should map to MIN_VALUE
        assertEquals(Integer.MIN_VALUE, hostProbe.getInt(0));
        assertEquals(Integer.MIN_VALUE, hostProbe.getInt(1));
        assertEquals(Integer.MIN_VALUE, hostProbe.getInt(2));
      }
    }
  }

  @Test
  public void testRemappingReuseStructures() {
    // Verify that RemapStructures can be reused for multiple probes
    try (ColumnVector buildCol = ColumnVector.fromInts(1, 2, 3);
         Table buildKeys = new Table(buildCol);
         KeyRemapping.RemapStructures remap = KeyRemapping.createRemapStructures(buildKeys)) {

      // First probe
      try (ColumnVector probeCol1 = ColumnVector.fromInts(1, 2);
           Table probeKeys1 = new Table(probeCol1);
           ColumnVector remappedProbe1 = KeyRemapping.applyRemapping(probeKeys1, remap);
           HostColumnVector hostProbe1 = remappedProbe1.copyToHost()) {
        assertTrue(hostProbe1.getInt(0) >= 0 && hostProbe1.getInt(0) <= 2);
        assertTrue(hostProbe1.getInt(1) >= 0 && hostProbe1.getInt(1) <= 2);
      }

      // Second probe with different keys
      try (ColumnVector probeCol2 = ColumnVector.fromInts(2, 3);
           Table probeKeys2 = new Table(probeCol2);
           ColumnVector remappedProbe2 = KeyRemapping.applyRemapping(probeKeys2, remap);
           HostColumnVector hostProbe2 = remappedProbe2.copyToHost()) {
        assertTrue(hostProbe2.getInt(0) >= 0 && hostProbe2.getInt(0) <= 2);
        assertTrue(hostProbe2.getInt(1) >= 0 && hostProbe2.getInt(1) <= 2);
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

    // Verify structures are accessible
    assertNotNull(remap.getDistinctKeys());
    assertNotNull(remap.getIntSequenceTable());
    assertNotNull(remap.getRemapJoinObject());

    // Close and verify cleanup
    assertDoesNotThrow(() -> remap.close());
  }
}

