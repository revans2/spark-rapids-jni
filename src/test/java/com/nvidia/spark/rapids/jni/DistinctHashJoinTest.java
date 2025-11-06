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
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.Table;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class DistinctHashJoinTest {

  @Test
  public void testInnerJoinBasic() {
    // Build side must have distinct keys
    try (ColumnVector buildCol = ColumnVector.fromInts(0, 1, 2, 3, 4);
         ColumnVector probeCol = ColumnVector.fromInts(2, 3, 4, 5, 6);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         DistinctHashJoin hashJoin = DistinctHashJoin.create(buildKeys, true)) {

      // Probe the hash join - expect matches for keys 2, 3, 4
      GatherMap[] result = hashJoin.innerJoin(probeKeys);
      try {
        // Expected matches: build[2,3,4] with probe[0,1,2]
        java.util.Set<java.util.Map.Entry<Integer, Integer>> expected =
          JoinPrimitivesTest.pairSet(
            JoinPrimitivesTest.pair(2, 0),  // build[2]=2 matches probe[0]=2
            JoinPrimitivesTest.pair(3, 1),  // build[3]=3 matches probe[1]=3
            JoinPrimitivesTest.pair(4, 2)   // build[4]=4 matches probe[2]=4
          );
        
        JoinPrimitivesTest.assertGatherMapPairs(
          result,
          buildKeys.getRowCount(),
          probeKeys.getRowCount(),
          expected,
          "Distinct hash inner join basic");
      } finally {
        for (GatherMap gm : result) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  public void testInnerJoinMultipleProbes() {
    try (ColumnVector buildCol = ColumnVector.fromInts(0, 1, 2, 3, 4);
         Table buildKeys = new Table(buildCol);
         DistinctHashJoin hashJoin = DistinctHashJoin.create(buildKeys, true)) {

      // First probe: keys 2, 3
      try (ColumnVector probeCol1 = ColumnVector.fromInts(2, 3);
           Table probeKeys1 = new Table(probeCol1)) {
        GatherMap[] result1 = hashJoin.innerJoin(probeKeys1);
        try {
          java.util.Set<java.util.Map.Entry<Integer, Integer>> expected =
            JoinPrimitivesTest.pairSet(
              JoinPrimitivesTest.pair(2, 0),  // build[2]=2 matches probe[0]=2
              JoinPrimitivesTest.pair(3, 1)   // build[3]=3 matches probe[1]=3
            );
          JoinPrimitivesTest.assertGatherMapPairs(
            result1, buildKeys.getRowCount(), probeKeys1.getRowCount(), expected,
            "First probe");
        } finally {
          for (GatherMap gm : result1) {
            if (gm != null) gm.close();
          }
        }
      }

      // Second probe with different data: keys 0, 1, 4
      try (ColumnVector probeCol2 = ColumnVector.fromInts(0, 1, 4);
           Table probeKeys2 = new Table(probeCol2)) {
        GatherMap[] result2 = hashJoin.innerJoin(probeKeys2);
        try {
          java.util.Set<java.util.Map.Entry<Integer, Integer>> expected =
            JoinPrimitivesTest.pairSet(
              JoinPrimitivesTest.pair(0, 0),  // build[0]=0 matches probe[0]=0
              JoinPrimitivesTest.pair(1, 1),  // build[1]=1 matches probe[1]=1
              JoinPrimitivesTest.pair(4, 2)   // build[4]=4 matches probe[2]=4
            );
          JoinPrimitivesTest.assertGatherMapPairs(
            result2, buildKeys.getRowCount(), probeKeys2.getRowCount(), expected,
            "Second probe");
        } finally {
          for (GatherMap gm : result2) {
            if (gm != null) gm.close();
          }
        }
      }

      // Third probe with no matches: keys 5, 6, 7
      try (ColumnVector probeCol3 = ColumnVector.fromInts(5, 6, 7);
           Table probeKeys3 = new Table(probeCol3)) {
        GatherMap[] result3 = hashJoin.innerJoin(probeKeys3);
        try {
          assertEquals(0, result3[0].getRowCount());
          assertEquals(0, result3[1].getRowCount());
        } finally {
          for (GatherMap gm : result3) {
            if (gm != null) gm.close();
          }
        }
      }
    }
  }

  @Test
  public void testLeftJoinBasic() {
    // Build side must have distinct keys
    try (ColumnVector buildCol = ColumnVector.fromInts(0, 1, 2);
         ColumnVector probeCol = ColumnVector.fromInts(1, 2, 3);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         DistinctHashJoin hashJoin = DistinctHashJoin.create(buildKeys, true)) {

      // leftJoin returns only build indices (one per probe row)
      GatherMap buildIndices = hashJoin.leftJoin(probeKeys);
      try {
        assertNotNull(buildIndices);
        
        // Left join should have same number of rows as probe table (3)
        assertEquals(3, buildIndices.getRowCount());
      } finally {
        buildIndices.close();
      }
    }
  }

  @Test
  public void testLeftJoinVerifyIndices() {
    // Verify that leftJoin returns correct build indices
    try (ColumnVector buildCol = ColumnVector.fromInts(10, 20, 30);
         ColumnVector probeCol = ColumnVector.fromInts(20, 30, 40);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         DistinctHashJoin hashJoin = DistinctHashJoin.create(buildKeys, true)) {

      GatherMap buildIndices = hashJoin.leftJoin(probeKeys);
      try {
        assertEquals(3, buildIndices.getRowCount());
        
        // Check the actual indices
        try (HostColumnVector hostIndices = buildIndices.toColumnView(0, 3).copyToHost()) {
          // probe[0]=20 matches build[1]
          assertEquals(1, hostIndices.getInt(0));
          // probe[1]=30 matches build[2]
          assertEquals(2, hostIndices.getInt(1));
          // probe[2]=40 has no match (should be out-of-bounds/invalid index)
          int unmatchedIndex = hostIndices.getInt(2);
          assertTrue(unmatchedIndex < 0 || unmatchedIndex >= 3, 
              "Unmatched probe row should have invalid build index");
        }
      } finally {
        buildIndices.close();
      }
    }
  }

  @Test
  public void testInnerJoinWithNulls() {
    // Test with nulls when compareNullsEqual = true
    // Build: [0, 1, null, 3], Probe: [1, null, 4]
    try (ColumnVector buildCol = ColumnVector.fromBoxedInts(0, 1, null, 3);
         ColumnVector probeCol = ColumnVector.fromBoxedInts(1, null, 4);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         DistinctHashJoin hashJoin = DistinctHashJoin.create(buildKeys, true)) {

      GatherMap[] result = hashJoin.innerJoin(probeKeys);
      try {
        // Should match: 1 and null (2 matches)
        java.util.Set<java.util.Map.Entry<Integer, Integer>> expected =
          JoinPrimitivesTest.pairSet(
            JoinPrimitivesTest.pair(1, 0),  // build[1]=1 matches probe[0]=1
            JoinPrimitivesTest.pair(2, 1)   // build[2]=null matches probe[1]=null
          );
        JoinPrimitivesTest.assertGatherMapPairs(
          result, buildKeys.getRowCount(), probeKeys.getRowCount(), expected,
          "Inner join with nulls (nulls equal)");
      } finally {
        for (GatherMap gm : result) {
          if (gm != null) gm.close();
        }
      }
    }

    // Test with nulls when compareNullsEqual = false
    try (ColumnVector buildCol = ColumnVector.fromBoxedInts(0, 1, null, 3);
         ColumnVector probeCol = ColumnVector.fromBoxedInts(1, null, 4);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         DistinctHashJoin hashJoin = DistinctHashJoin.create(buildKeys, false)) {

      GatherMap[] result = hashJoin.innerJoin(probeKeys);
      try {
        // Should match: only 1 (null != null when compareNullsEqual = false)
        java.util.Set<java.util.Map.Entry<Integer, Integer>> expected =
          JoinPrimitivesTest.pairSet(
            JoinPrimitivesTest.pair(1, 0)  // build[1]=1 matches probe[0]=1
          );
        JoinPrimitivesTest.assertGatherMapPairs(
          result, buildKeys.getRowCount(), probeKeys.getRowCount(), expected,
          "Inner join with nulls (nulls not equal)");
      } finally {
        for (GatherMap gm : result) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  public void testInnerJoinEmpty() {
    try (ColumnVector buildCol = ColumnVector.fromInts(0, 1, 2);
         ColumnVector probeCol = ColumnVector.fromInts();
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         DistinctHashJoin hashJoin = DistinctHashJoin.create(buildKeys, true)) {

      GatherMap[] result = hashJoin.innerJoin(probeKeys);
      try {
        assertEquals(0, result[0].getRowCount());
        assertEquals(0, result[1].getRowCount());
      } finally {
        for (GatherMap gm : result) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  public void testLeftJoinEmpty() {
    try (ColumnVector buildCol = ColumnVector.fromInts(0, 1, 2);
         ColumnVector probeCol = ColumnVector.fromInts();
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         DistinctHashJoin hashJoin = DistinctHashJoin.create(buildKeys, true)) {

      GatherMap buildIndices = hashJoin.leftJoin(probeKeys);
      try {
        assertEquals(0, buildIndices.getRowCount());
      } finally {
        buildIndices.close();
      }
    }
  }

  @Test
  public void testInnerJoinMultiColumn() {
    // Build side must have distinct key combinations
    // Build keys: [(0,0), (1,1), (2,0)]
    // Probe keys: [(1,1), (2,0), (3,1)]
    try (ColumnVector buildCol1 = ColumnVector.fromInts(0, 1, 2);
         ColumnVector buildCol2 = ColumnVector.fromInts(0, 1, 0);
         ColumnVector probeCol1 = ColumnVector.fromInts(1, 2, 3);
         ColumnVector probeCol2 = ColumnVector.fromInts(1, 0, 1);
         Table buildKeys = new Table(buildCol1, buildCol2);
         Table probeKeys = new Table(probeCol1, probeCol2);
         DistinctHashJoin hashJoin = DistinctHashJoin.create(buildKeys, true)) {

      GatherMap[] result = hashJoin.innerJoin(probeKeys);
      try {
        // Matches: build[1]=(1,1) with probe[0]=(1,1), build[2]=(2,0) with probe[1]=(2,0)
        java.util.Set<java.util.Map.Entry<Integer, Integer>> expected =
          JoinPrimitivesTest.pairSet(
            JoinPrimitivesTest.pair(1, 0),  // build[1]=(1,1) matches probe[0]=(1,1)
            JoinPrimitivesTest.pair(2, 1)   // build[2]=(2,0) matches probe[1]=(2,0)
          );
        JoinPrimitivesTest.assertGatherMapPairs(
          result, buildKeys.getRowCount(), probeKeys.getRowCount(), expected,
          "Inner join multi-column");
      } finally {
        for (GatherMap gm : result) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  public void testCloseMultipleTimes() {
    DistinctHashJoin hashJoin;
    try (ColumnVector buildCol = ColumnVector.fromInts(0, 1, 2);
         Table buildKeys = new Table(buildCol)) {
      hashJoin = DistinctHashJoin.create(buildKeys, true);
    }
    
    // Close multiple times should be safe
    hashJoin.close();
    hashJoin.close();
  }

  @Test
  public void testUseAfterClose() {
    DistinctHashJoin hashJoin;
    try (ColumnVector buildCol = ColumnVector.fromInts(0, 1, 2);
         Table buildKeys = new Table(buildCol)) {
      hashJoin = DistinctHashJoin.create(buildKeys, true);
    }
    
    hashJoin.close();
    
    // Using after close should throw
    try (ColumnVector probeCol = ColumnVector.fromInts(1, 2);
         Table probeKeys = new Table(probeCol)) {
      assertThrows(IllegalStateException.class, () -> {
        hashJoin.innerJoin(probeKeys);
      });
      assertThrows(IllegalStateException.class, () -> {
        hashJoin.leftJoin(probeKeys);
      });
    }
  }

  @Test
  public void testInnerJoinNoDuplicateProbeKeys() {
    // Verify distinct join behavior: each probe key matches at most once
    try (ColumnVector buildCol = ColumnVector.fromInts(1, 2, 3);
         ColumnVector probeCol = ColumnVector.fromInts(1, 1, 2);  // Duplicate probe keys
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         DistinctHashJoin hashJoin = DistinctHashJoin.create(buildKeys, true)) {

      GatherMap[] result = hashJoin.innerJoin(probeKeys);
      try {
        // Each of the 3 probe rows finds exactly one match (no cartesian product)
        assertEquals(3, result[0].getRowCount());
        assertEquals(3, result[1].getRowCount());
        
        // Verify: result[0]=build indices, result[1]=probe indices
        // probe rows 0,1,2 match build rows 0,0,1 respectively
        try (HostColumnVector hostBuildIndices = result[0].toColumnView(0, 3).copyToHost();
             HostColumnVector hostProbeIndices = result[1].toColumnView(0, 3).copyToHost()) {
          assertEquals(0, hostBuildIndices.getInt(0));  // probe[0]=1 matches build[0]=1
          assertEquals(0, hostBuildIndices.getInt(1));  // probe[1]=1 matches build[0]=1
          assertEquals(1, hostBuildIndices.getInt(2));  // probe[2]=2 matches build[1]=2
          assertEquals(0, hostProbeIndices.getInt(0));
          assertEquals(1, hostProbeIndices.getInt(1));
          assertEquals(2, hostProbeIndices.getInt(2));
        }
      } finally {
        for (GatherMap gm : result) {
          if (gm != null) gm.close();
        }
      }
    }
  }
}

