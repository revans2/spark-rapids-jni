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

public class SortMergeJoinTest {

  @Test
  public void testInnerJoinBasic() {
    // Build: [0, 1, 2, 3, 4], Probe: [2, 3, 4, 5, 6]
    try (ColumnVector buildCol = ColumnVector.fromInts(0, 1, 2, 3, 4);
         ColumnVector probeCol = ColumnVector.fromInts(2, 3, 4, 5, 6);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         SortMergeJoin sortMergeJoin = SortMergeJoin.create(buildKeys, false, true)) {

      GatherMap[] result = sortMergeJoin.innerJoin(probeKeys, false);
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
          "Sort-merge join inner join basic");
      } finally {
        for (GatherMap gm : result) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  public void testInnerJoinBothSorted() {
    // Test with both tables already sorted
    // Build: [0, 1, 2, 3, 4], Probe: [2, 3, 4, 5, 6]
    try (ColumnVector buildCol = ColumnVector.fromInts(0, 1, 2, 3, 4);
         ColumnVector probeCol = ColumnVector.fromInts(2, 3, 4, 5, 6);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         SortMergeJoin sortMergeJoin = SortMergeJoin.create(buildKeys, true, true)) {

      GatherMap[] result = sortMergeJoin.innerJoin(probeKeys, true);
      try {
        // Expected matches: build[2,3,4] with probe[0,1,2]
        java.util.Set<java.util.Map.Entry<Integer, Integer>> expected =
          JoinPrimitivesTest.pairSet(
            JoinPrimitivesTest.pair(2, 0),
            JoinPrimitivesTest.pair(3, 1),
            JoinPrimitivesTest.pair(4, 2)
          );
        
        JoinPrimitivesTest.assertGatherMapPairs(
          result, buildKeys.getRowCount(), probeKeys.getRowCount(), expected,
          "Sort-merge join with both sorted");
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
         SortMergeJoin sortMergeJoin = SortMergeJoin.create(buildKeys, false, true)) {

      // First probe: keys 2, 3
      try (ColumnVector probeCol1 = ColumnVector.fromInts(2, 3);
           Table probeKeys1 = new Table(probeCol1)) {
        GatherMap[] result1 = sortMergeJoin.innerJoin(probeKeys1, false);
        try {
          java.util.Set<java.util.Map.Entry<Integer, Integer>> expected =
            JoinPrimitivesTest.pairSet(
              JoinPrimitivesTest.pair(2, 0),
              JoinPrimitivesTest.pair(3, 1)
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

      // Second probe: keys 0, 1, 4
      try (ColumnVector probeCol2 = ColumnVector.fromInts(0, 1, 4);
           Table probeKeys2 = new Table(probeCol2)) {
        GatherMap[] result2 = sortMergeJoin.innerJoin(probeKeys2, false);
        try {
          java.util.Set<java.util.Map.Entry<Integer, Integer>> expected =
            JoinPrimitivesTest.pairSet(
              JoinPrimitivesTest.pair(0, 0),
              JoinPrimitivesTest.pair(1, 1),
              JoinPrimitivesTest.pair(4, 2)
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

      // Third probe with no matches
      try (ColumnVector probeCol3 = ColumnVector.fromInts(5, 6, 7);
           Table probeKeys3 = new Table(probeCol3)) {
        GatherMap[] result3 = sortMergeJoin.innerJoin(probeKeys3, false);
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
  public void testInnerJoinWithNulls() {
    // Test with nulls when compareNullsEqual = true
    try (ColumnVector buildCol = ColumnVector.fromBoxedInts(0, 1, null, 3);
         ColumnVector probeCol = ColumnVector.fromBoxedInts(1, null, 4);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         SortMergeJoin sortMergeJoin = SortMergeJoin.create(buildKeys, false, true)) {

      GatherMap[] result = sortMergeJoin.innerJoin(probeKeys, false);
      try {
        // Expected: build[1]=1 matches probe[0]=1, build[2]=null matches probe[1]=null
        java.util.Set<java.util.Map.Entry<Integer, Integer>> expected =
          JoinPrimitivesTest.pairSet(
            JoinPrimitivesTest.pair(1, 0),
            JoinPrimitivesTest.pair(2, 1)
          );
        JoinPrimitivesTest.assertGatherMapPairs(
          result, buildKeys.getRowCount(), probeKeys.getRowCount(), expected,
          "Sort-merge join with nulls (nulls equal)");
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
         SortMergeJoin sortMergeJoin = SortMergeJoin.create(buildKeys, false, false)) {

      GatherMap[] result = sortMergeJoin.innerJoin(probeKeys, false);
      try {
        // Expected: only build[1]=1 matches probe[0]=1
        java.util.Set<java.util.Map.Entry<Integer, Integer>> expected =
          JoinPrimitivesTest.pairSet(
            JoinPrimitivesTest.pair(1, 0)
          );
        JoinPrimitivesTest.assertGatherMapPairs(
          result, buildKeys.getRowCount(), probeKeys.getRowCount(), expected,
          "Sort-merge join with nulls (nulls not equal)");
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
         SortMergeJoin sortMergeJoin = SortMergeJoin.create(buildKeys, false, true)) {

      GatherMap[] result = sortMergeJoin.innerJoin(probeKeys, false);
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
    // Build keys: [(0,0), (1,1), (2,0)]
    // Probe keys: [(1,1), (2,0), (3,1)]
    try (ColumnVector buildCol1 = ColumnVector.fromInts(0, 1, 2);
         ColumnVector buildCol2 = ColumnVector.fromInts(0, 1, 0);
         ColumnVector probeCol1 = ColumnVector.fromInts(1, 2, 3);
         ColumnVector probeCol2 = ColumnVector.fromInts(1, 0, 1);
         Table buildKeys = new Table(buildCol1, buildCol2);
         Table probeKeys = new Table(probeCol1, probeCol2);
         SortMergeJoin sortMergeJoin = SortMergeJoin.create(buildKeys, false, true)) {

      GatherMap[] result = sortMergeJoin.innerJoin(probeKeys, false);
      try {
        // Expected: build[1]=(1,1) with probe[0]=(1,1), build[2]=(2,0) with probe[1]=(2,0)
        java.util.Set<java.util.Map.Entry<Integer, Integer>> expected =
          JoinPrimitivesTest.pairSet(
            JoinPrimitivesTest.pair(1, 0),
            JoinPrimitivesTest.pair(2, 1)
          );
        JoinPrimitivesTest.assertGatherMapPairs(
          result, buildKeys.getRowCount(), probeKeys.getRowCount(), expected,
          "Sort-merge join multi-column");
      } finally {
        for (GatherMap gm : result) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  public void testInnerJoinWithDuplicates() {
    // Test case where both tables have duplicate keys
    // Build: [1, 1, 2], Probe: [1, 1, 3]
    try (ColumnVector buildCol = ColumnVector.fromInts(1, 1, 2);
         ColumnVector probeCol = ColumnVector.fromInts(1, 1, 3);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol);
         SortMergeJoin sortMergeJoin = SortMergeJoin.create(buildKeys, false, true)) {

      GatherMap[] result = sortMergeJoin.innerJoin(probeKeys, false);
      try {
        // Expected: cartesian product of matching keys
        // build[0,1]=1 x probe[0,1]=1 = 4 results
        java.util.Set<java.util.Map.Entry<Integer, Integer>> expected =
          JoinPrimitivesTest.pairSet(
            JoinPrimitivesTest.pair(0, 0),  // build[0]=1 x probe[0]=1
            JoinPrimitivesTest.pair(0, 1),  // build[0]=1 x probe[1]=1
            JoinPrimitivesTest.pair(1, 0),  // build[1]=1 x probe[0]=1
            JoinPrimitivesTest.pair(1, 1)   // build[1]=1 x probe[1]=1
          );
        JoinPrimitivesTest.assertGatherMapPairs(
          result, buildKeys.getRowCount(), probeKeys.getRowCount(), expected,
          "Sort-merge join with duplicates");
      } finally {
        for (GatherMap gm : result) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  public void testCloseMultipleTimes() {
    SortMergeJoin sortMergeJoin;
    try (ColumnVector buildCol = ColumnVector.fromInts(0, 1, 2);
         Table buildKeys = new Table(buildCol)) {
      sortMergeJoin = SortMergeJoin.create(buildKeys, false, true);
    }

    // First close
    sortMergeJoin.close();
    
    // Second close should not throw
    assertDoesNotThrow(() -> sortMergeJoin.close());
  }

  @Test
  public void testUseAfterClose() {
    SortMergeJoin sortMergeJoin;
    try (ColumnVector buildCol = ColumnVector.fromInts(0, 1, 2);
         ColumnVector probeCol = ColumnVector.fromInts(1, 2, 3);
         Table buildKeys = new Table(buildCol);
         Table probeKeys = new Table(probeCol)) {
      sortMergeJoin = SortMergeJoin.create(buildKeys, false, true);
      sortMergeJoin.close();

      // Using after close should throw
      assertThrows(IllegalStateException.class, () -> {
        sortMergeJoin.innerJoin(probeKeys, false);
      });
    }
  }
}

