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
import ai.rapids.cudf.Table;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class HashJoinTest {

  @Test
  public void testInnerJoinBasic() {
    try (ColumnVector buildCol = ColumnVector.fromInts(0, 1, 2, 3, 4);
         ColumnVector probeCol = ColumnVector.fromInts(2, 3, 4, 5, 6);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         HashJoin hashJoin = HashJoin.create(buildKeys, true)) {

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
          result, buildKeys.getRowCount(), probeKeys.getRowCount(), expected,
          "Hash join inner join basic");

        // Verify the join output size
        assertEquals(3, hashJoin.innerJoinSize(probeKeys));
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
         HashJoin hashJoin = HashJoin.create(buildKeys, true)) {

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
    try (ColumnVector buildCol = ColumnVector.fromInts(0, 1, 2);
         ColumnVector probeCol = ColumnVector.fromInts(1, 2, 3);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         HashJoin hashJoin = HashJoin.create(buildKeys, true)) {

      GatherMap[] result = hashJoin.leftJoin(probeKeys);
      try {
        assertNotNull(result);
        assertEquals(2, result.length);
        
        // Left join should have same number of rows as build table (3)
        assertEquals(3, result[0].getRowCount());
        assertEquals(3, result[1].getRowCount());

        // Verify the join output size
        assertEquals(3, hashJoin.leftJoinSize(probeKeys));
      } finally {
        for (GatherMap gm : result) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  public void testFullJoinBasic() {
    try (ColumnVector buildCol = ColumnVector.fromInts(0, 1, 2);
         ColumnVector probeCol = ColumnVector.fromInts(1, 2, 3);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         HashJoin hashJoin = HashJoin.create(buildKeys, true)) {

      GatherMap[] result = hashJoin.fullJoin(probeKeys);
      try {
        assertNotNull(result);
        assertEquals(2, result.length);
        
        // Full join should include all rows from both tables
        // Build: 0, 1, 2 (3 rows)
        // Probe: 1, 2, 3 (3 rows)
        // Matches: 1, 2 (2 matches)
        // Unmatched: 0 from build, 3 from probe (2 unmatched)
        // Total: 4 rows
        assertEquals(4, result[0].getRowCount());
        assertEquals(4, result[1].getRowCount());

        // Verify the join output size
        assertEquals(4, hashJoin.fullJoinSize(probeKeys));
      } finally {
        for (GatherMap gm : result) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  public void testInnerJoinWithNulls() {
    // Test with nulls when compareNullsEqual = true
    try (ColumnVector buildCol = ColumnVector.fromBoxedInts(0, 1, null, 3);
         ColumnVector probeCol = ColumnVector.fromBoxedInts(1, null, 4);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         HashJoin hashJoin = HashJoin.create(buildKeys, true)) {

      GatherMap[] result = hashJoin.innerJoin(probeKeys);
      try {
        // Should match: 1 and null (2 matches)
        assertEquals(2, result[0].getRowCount());
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
         HashJoin hashJoin = HashJoin.create(buildKeys, false)) {

      GatherMap[] result = hashJoin.innerJoin(probeKeys);
      try {
        // Should match: only 1 (null != null when compareNullsEqual = false)
        assertEquals(1, result[0].getRowCount());
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
         HashJoin hashJoin = HashJoin.create(buildKeys, true)) {

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
  public void testInnerJoinMultiColumn() {
    try (ColumnVector buildCol1 = ColumnVector.fromInts(0, 1, 2, 2);
         ColumnVector buildCol2 = ColumnVector.fromInts(0, 1, 0, 1);
         ColumnVector probeCol1 = ColumnVector.fromInts(1, 2, 2);
         ColumnVector probeCol2 = ColumnVector.fromInts(1, 0, 1);
         Table buildKeys = new Table(buildCol1, buildCol2);
         Table probeKeys = new Table(probeCol1, probeCol2);
         HashJoin hashJoin = HashJoin.create(buildKeys, true)) {

      GatherMap[] result = hashJoin.innerJoin(probeKeys);
      try {
        // Matches: (1,1), (2,0), (2,1)
        assertEquals(3, result[0].getRowCount());
      } finally {
        for (GatherMap gm : result) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  public void testInnerJoinDuplicates() {
    // Test with duplicate keys on both sides
    try (ColumnVector buildCol = ColumnVector.fromInts(1, 1, 2, 2);
         ColumnVector probeCol = ColumnVector.fromInts(1, 1);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         HashJoin hashJoin = HashJoin.create(buildKeys, true)) {

      GatherMap[] result = hashJoin.innerJoin(probeKeys);
      try {
        // Each probe 1 matches with both build 1s -> 2 probe * 2 build = 4 results
        assertEquals(4, result[0].getRowCount());
      } finally {
        for (GatherMap gm : result) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  public void testCloseMultipleTimes() {
    HashJoin hashJoin;
    try (ColumnVector buildCol = ColumnVector.fromInts(0, 1, 2);
         Table buildKeys = new Table(buildCol)) {
      hashJoin = HashJoin.create(buildKeys, true);
    }
    
    // Close multiple times should be safe
    hashJoin.close();
    hashJoin.close();
  }

  @Test
  public void testUseAfterClose() {
    HashJoin hashJoin;
    try (ColumnVector buildCol = ColumnVector.fromInts(0, 1, 2);
         Table buildKeys = new Table(buildCol)) {
      hashJoin = HashJoin.create(buildKeys, true);
    }
    
    hashJoin.close();
    
    // Using after close should throw
    try (ColumnVector probeCol = ColumnVector.fromInts(1, 2);
         Table probeKeys = new Table(probeCol)) {
      assertThrows(IllegalStateException.class, () -> {
        hashJoin.innerJoin(probeKeys);
      });
    }
  }

  @Test
  public void testGatherResult() {
    // Verify that the gather maps produce the correct results using order-agnostic comparison
    try (ColumnVector buildKeyCol = ColumnVector.fromInts(0, 1, 2, 3);
         ColumnVector probeKeyCol = ColumnVector.fromInts(1, 2, 4);
         Table buildKeys = new Table(buildKeyCol);
         Table probeKeys = new Table(probeKeyCol);
         HashJoin hashJoin = HashJoin.create(buildKeys, true)) {

      GatherMap[] gatherMaps = hashJoin.innerJoin(probeKeys);
      try {
        // Use the order-agnostic comparison helper from JoinPrimitivesTest
        // Expected: probe keys 1, 2 match build keys at indices 1, 2
        // So we expect: (1, 0) and (2, 1) where first is build index, second is probe index
        java.util.Set<java.util.Map.Entry<Integer, Integer>> expected =
          JoinPrimitivesTest.pairSet(
            JoinPrimitivesTest.pair(1, 0),  // build[1]=1 matches probe[0]=1
            JoinPrimitivesTest.pair(2, 1)   // build[2]=2 matches probe[1]=2
          );
        
        JoinPrimitivesTest.assertGatherMapPairs(
          gatherMaps,
          buildKeys.getRowCount(),
          probeKeys.getRowCount(),
          expected,
          "Hash join inner join result");
      } finally {
        for (GatherMap gm : gatherMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }
}
