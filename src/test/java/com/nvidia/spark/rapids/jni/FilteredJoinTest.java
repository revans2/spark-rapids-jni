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

public class FilteredJoinTest {

  @Test
  public void testSemiJoinBasic() {
    // Right (build): [0, 1, 2, 3, 4], Left: [2, 3, 4, 5, 6]
    try (ColumnVector rightCol = ColumnVector.fromInts(0, 1, 2, 3, 4);
         ColumnVector leftCol = ColumnVector.fromInts(2, 3, 4, 5, 6);
         Table rightKeys = new Table(rightCol);
         Table leftKeys = new Table(leftCol);
         FilteredJoin filteredJoin = FilteredJoin.create(rightKeys, true)) {

      // Semi join returns left table indices that have matches in right
      GatherMap result = filteredJoin.semiJoin(leftKeys);
      try {
        // Expected: left indices 0, 1, 2 (values 2, 3, 4) have matches in right
        java.util.Set<Integer> expected = new java.util.HashSet<>();
        expected.add(0);
        expected.add(1);
        expected.add(2);
        
        JoinPrimitivesTest.assertGatherMapIndices(
          result, leftKeys.getRowCount(), expected,
          "Semi join basic");
      } finally {
        result.close();
      }
    }
  }

  @Test
  public void testAntiJoinBasic() {
    // Right (build): [0, 1, 2, 3, 4], Left: [2, 3, 4, 5, 6]
    try (ColumnVector rightCol = ColumnVector.fromInts(0, 1, 2, 3, 4);
         ColumnVector leftCol = ColumnVector.fromInts(2, 3, 4, 5, 6);
         Table rightKeys = new Table(rightCol);
         Table leftKeys = new Table(leftCol);
         FilteredJoin filteredJoin = FilteredJoin.create(rightKeys, true)) {

      // Anti join returns left table indices that have NO matches in right
      GatherMap result = filteredJoin.antiJoin(leftKeys);
      try {
        // Expected: left indices 3, 4 (values 5, 6) have NO matches in right
        java.util.Set<Integer> expected = new java.util.HashSet<>();
        expected.add(3);
        expected.add(4);
        
        JoinPrimitivesTest.assertGatherMapIndices(
          result, leftKeys.getRowCount(), expected,
          "Anti join basic");
      } finally {
        result.close();
      }
    }
  }

  @Test
  public void testSemiJoinMultipleProbes() {
    try (ColumnVector rightCol = ColumnVector.fromInts(0, 1, 2, 3, 4);
         Table rightKeys = new Table(rightCol);
         FilteredJoin filteredJoin = FilteredJoin.create(rightKeys, true)) {

      // First probe: keys 2, 3
      try (ColumnVector leftCol1 = ColumnVector.fromInts(2, 3);
           Table leftKeys1 = new Table(leftCol1)) {
        GatherMap result1 = filteredJoin.semiJoin(leftKeys1);
        try {
          java.util.Set<Integer> expected = new java.util.HashSet<>();
          expected.add(0);
          expected.add(1);
          JoinPrimitivesTest.assertGatherMapIndices(
            result1, leftKeys1.getRowCount(), expected, "First probe");
        } finally {
          result1.close();
        }
      }

      // Second probe: keys 0, 1, 4
      try (ColumnVector leftCol2 = ColumnVector.fromInts(0, 1, 4);
           Table leftKeys2 = new Table(leftCol2)) {
        GatherMap result2 = filteredJoin.semiJoin(leftKeys2);
        try {
          java.util.Set<Integer> expected = new java.util.HashSet<>();
          expected.add(0);
          expected.add(1);
          expected.add(2);
          JoinPrimitivesTest.assertGatherMapIndices(
            result2, leftKeys2.getRowCount(), expected, "Second probe");
        } finally {
          result2.close();
        }
      }

      // Third probe: no matches
      try (ColumnVector leftCol3 = ColumnVector.fromInts(5, 6, 7);
           Table leftKeys3 = new Table(leftCol3)) {
        GatherMap result3 = filteredJoin.semiJoin(leftKeys3);
        try {
          assertEquals(0, result3.getRowCount());
        } finally {
          result3.close();
        }
      }
    }
  }

  @Test
  public void testSemiJoinWithNullsEqual() {
    // Test with nulls when compareNullsEqual = true
    try (ColumnVector rightCol = ColumnVector.fromBoxedInts(0, 1, null, 3);
         ColumnVector leftCol = ColumnVector.fromBoxedInts(1, null, 4);
         Table rightKeys = new Table(rightCol);
         Table leftKeys = new Table(leftCol);
         FilteredJoin filteredJoin = FilteredJoin.create(rightKeys, true)) {

      GatherMap result = filteredJoin.semiJoin(leftKeys);
      try {
        // Expected: left indices 0 (value=1) and 1 (value=null) have matches
        java.util.Set<Integer> expected = new java.util.HashSet<>();
        expected.add(0);
        expected.add(1);
        JoinPrimitivesTest.assertGatherMapIndices(
          result, leftKeys.getRowCount(), expected,
          "Semi join with nulls (nulls equal)");
      } finally {
        result.close();
      }
    }
  }

  @Test
  public void testSemiJoinWithNullsNotEqual() {
    // Test with nulls when compareNullsEqual = false
    try (ColumnVector rightCol = ColumnVector.fromBoxedInts(0, 1, null, 3);
         ColumnVector leftCol = ColumnVector.fromBoxedInts(1, null, 4);
         Table rightKeys = new Table(rightCol);
         Table leftKeys = new Table(leftCol);
         FilteredJoin filteredJoin = FilteredJoin.create(rightKeys, false)) {

      GatherMap result = filteredJoin.semiJoin(leftKeys);
      try {
        // Expected: only left index 0 (value=1) has a match
        java.util.Set<Integer> expected = new java.util.HashSet<>();
        expected.add(0);
        JoinPrimitivesTest.assertGatherMapIndices(
          result, leftKeys.getRowCount(), expected,
          "Semi join with nulls (nulls not equal)");
      } finally {
        result.close();
      }
    }
  }

  @Test
  public void testAntiJoinWithNullsEqual() {
    // Test with nulls when compareNullsEqual = true
    try (ColumnVector rightCol = ColumnVector.fromBoxedInts(0, 1, null, 3);
         ColumnVector leftCol = ColumnVector.fromBoxedInts(1, null, 4);
         Table rightKeys = new Table(rightCol);
         Table leftKeys = new Table(leftCol);
         FilteredJoin filteredJoin = FilteredJoin.create(rightKeys, true)) {

      GatherMap result = filteredJoin.antiJoin(leftKeys);
      try {
        // Expected: left index 2 (value=4) has NO matches
        java.util.Set<Integer> expected = new java.util.HashSet<>();
        expected.add(2);
        JoinPrimitivesTest.assertGatherMapIndices(
          result, leftKeys.getRowCount(), expected,
          "Anti join with nulls (nulls equal)");
      } finally {
        result.close();
      }
    }
  }

  @Test
  public void testAntiJoinWithNullsNotEqual() {
    // Test with nulls when compareNullsEqual = false
    try (ColumnVector rightCol = ColumnVector.fromBoxedInts(0, 1, null, 3);
         ColumnVector leftCol = ColumnVector.fromBoxedInts(1, null, 4);
         Table rightKeys = new Table(rightCol);
         Table leftKeys = new Table(leftCol);
         FilteredJoin filteredJoin = FilteredJoin.create(rightKeys, false)) {

      GatherMap result = filteredJoin.antiJoin(leftKeys);
      try {
        // Expected: left indices 1 (null) and 2 (value=4) have NO matches
        java.util.Set<Integer> expected = new java.util.HashSet<>();
        expected.add(1);
        expected.add(2);
        JoinPrimitivesTest.assertGatherMapIndices(
          result, leftKeys.getRowCount(), expected,
          "Anti join with nulls (nulls not equal)");
      } finally {
        result.close();
      }
    }
  }

  @Test
  public void testSemiJoinEmptyLeft() {
    try (ColumnVector rightCol = ColumnVector.fromInts(0, 1, 2);
         ColumnVector leftCol = ColumnVector.fromInts();
         Table rightKeys = new Table(rightCol);
         Table leftKeys = new Table(leftCol);
         FilteredJoin filteredJoin = FilteredJoin.create(rightKeys, true)) {

      GatherMap result = filteredJoin.semiJoin(leftKeys);
      try {
        assertEquals(0, result.getRowCount());
      } finally {
        result.close();
      }
    }
  }

  @Test
  public void testAntiJoinEmptyLeft() {
    try (ColumnVector rightCol = ColumnVector.fromInts(0, 1, 2);
         ColumnVector leftCol = ColumnVector.fromInts();
         Table rightKeys = new Table(rightCol);
         Table leftKeys = new Table(leftCol);
         FilteredJoin filteredJoin = FilteredJoin.create(rightKeys, true)) {

      GatherMap result = filteredJoin.antiJoin(leftKeys);
      try {
        // Empty left table means no rows to return
        assertEquals(0, result.getRowCount());
      } finally {
        result.close();
      }
    }
  }

  @Test
  public void testSemiJoinMultiColumn() {
    // Right keys: [(0,0), (1,1), (2,0)]
    // Left keys: [(1,1), (2,0), (3,1)]
    try (ColumnVector rightCol1 = ColumnVector.fromInts(0, 1, 2);
         ColumnVector rightCol2 = ColumnVector.fromInts(0, 1, 0);
         ColumnVector leftCol1 = ColumnVector.fromInts(1, 2, 3);
         ColumnVector leftCol2 = ColumnVector.fromInts(1, 0, 1);
         Table rightKeys = new Table(rightCol1, rightCol2);
         Table leftKeys = new Table(leftCol1, leftCol2);
         FilteredJoin filteredJoin = FilteredJoin.create(rightKeys, true)) {

      GatherMap result = filteredJoin.semiJoin(leftKeys);
      try {
        // Expected: left indices 0 and 1 have matches
        java.util.Set<Integer> expected = new java.util.HashSet<>();
        expected.add(0);
        expected.add(1);
        JoinPrimitivesTest.assertGatherMapIndices(
          result, leftKeys.getRowCount(), expected,
          "Semi join multi-column");
      } finally {
        result.close();
      }
    }
  }

  @Test
  public void testCloseMultipleTimes() {
    FilteredJoin filteredJoin;
    try (ColumnVector rightCol = ColumnVector.fromInts(0, 1, 2);
         Table rightKeys = new Table(rightCol)) {
      filteredJoin = FilteredJoin.create(rightKeys, true);
    }

    // First close
    filteredJoin.close();
    
    // Second close should not throw
    assertDoesNotThrow(() -> filteredJoin.close());
  }

  @Test
  public void testUseAfterClose() {
    FilteredJoin filteredJoin;
    try (ColumnVector rightCol = ColumnVector.fromInts(0, 1, 2);
         ColumnVector leftCol = ColumnVector.fromInts(1, 2, 3);
         Table rightKeys = new Table(rightCol);
         Table leftKeys = new Table(leftCol)) {
      filteredJoin = FilteredJoin.create(rightKeys, true);
      filteredJoin.close();

      // Using after close should throw
      assertThrows(IllegalStateException.class, () -> {
        filteredJoin.semiJoin(leftKeys);
      });
      assertThrows(IllegalStateException.class, () -> {
        filteredJoin.antiJoin(leftKeys);
      });
    }
  }
}
