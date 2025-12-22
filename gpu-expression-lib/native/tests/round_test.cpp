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

#include <spark_gpu_expr/round_expression.hpp>
#include <gtest/gtest.h>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/column_utilities.hpp>

#include <limits>
#include <cmath>
#include <vector>

namespace spark_gpu_expr {
namespace test {

class RoundTest : public ::testing::Test {};

// =============================================================================
// Basic Float/Double Tests
// =============================================================================

TEST_F(RoundTest, RoundFloatZeroDecimal)
{
    // HALF_UP: rounds away from zero on .5
    cudf::test::fixed_width_column_wrapper<float> input{1.5f, 2.5f, 3.5f, 4.5f, -1.5f, -2.5f};
    
    auto result = expression::round(input, 0);
    
    cudf::test::fixed_width_column_wrapper<float> expected{2.0f, 3.0f, 4.0f, 5.0f, -2.0f, -3.0f};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
}

TEST_F(RoundTest, RoundDoublePositiveDecimal)
{
    // Test positive decimal places rounding
    // Avoid .xx5 boundary values which have floating-point representation issues
    cudf::test::fixed_width_column_wrapper<double> input{1.234, 2.456, 3.567};
    
    auto result = expression::round(input, 2);
    
    // 1.234 -> 1.23 (down), 2.456 -> 2.46 (up), 3.567 -> 3.57 (up)
    cudf::test::fixed_width_column_wrapper<double> expected{1.23, 2.46, 3.57};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
}

TEST_F(RoundTest, RoundDoubleNegativeDecimal)
{
    // Negative decimal places - round to tens, hundreds, etc.
    cudf::test::fixed_width_column_wrapper<double> input{123.0, 456.0, 789.0};
    
    auto result = expression::round(input, -2);
    
    cudf::test::fixed_width_column_wrapper<double> expected{100.0, 500.0, 800.0};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
}

TEST_F(RoundTest, BRoundHalfEven)
{
    // HALF_EVEN (bround): rounds to nearest even on .5
    // This matches Spark's bround() function
    cudf::test::fixed_width_column_wrapper<float> input{0.5f, 1.5f, 2.5f, 3.5f, 4.5f};
    
    auto result = expression::bround(input, 0);
    
    // 0.5 -> 0, 1.5 -> 2, 2.5 -> 2, 3.5 -> 4, 4.5 -> 4
    cudf::test::fixed_width_column_wrapper<float> expected{0.0f, 2.0f, 2.0f, 4.0f, 4.0f};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
}

TEST_F(RoundTest, HandleNulls)
{
    // Verify null values are preserved through rounding
    cudf::test::fixed_width_column_wrapper<double> input{{1.5, 2.5, 3.5}, {true, false, true}};
    
    auto result = expression::round(input, 0);
    
    cudf::test::fixed_width_column_wrapper<double> expected{{2.0, 0.0, 4.0}, {true, false, true}};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
}

// =============================================================================
// Integer Overflow Tests (Spark compatibility)
// =============================================================================

TEST_F(RoundTest, Int32OverflowReturnsZero)
{
    // When -scale >= 10 (INT32 max digits), Spark returns 0
    std::vector<int32_t> values = {
        123456789, 
        -987654321, 
        std::numeric_limits<int32_t>::max(),
        std::numeric_limits<int32_t>::min()
    };
    cudf::test::fixed_width_column_wrapper<int32_t> input(values.begin(), values.end());
    
    auto result = expression::round(input, -10);
    
    // All values become 0 when scale exceeds max digits
    cudf::test::fixed_width_column_wrapper<int32_t> expected{0, 0, 0, 0};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TEST_F(RoundTest, Int32NormalNegativeScale)
{
    // Normal rounding with negative scale (within max digits)
    cudf::test::fixed_width_column_wrapper<int32_t> input{1234, 5678, -1234, -5678};
    
    auto result = expression::round(input, -2);
    
    // Round to nearest 100
    cudf::test::fixed_width_column_wrapper<int32_t> expected{1200, 5700, -1200, -5700};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TEST_F(RoundTest, Int64AtMaxDigitsOverflowBehavior)
{
    // When -scale == 19 (exactly INT64 max digits), Spark has special overflow behavior:
    // - Values with head digit > 4 would round up to 1e19, which overflows to -8446744073709551616
    // - Values with head digit < -4 would round to -1e19, which overflows to 8446744073709551616
    // - Values with |head digit| <= 4 round down to 0
    std::vector<int64_t> values = {
        5000000000000000000L,   // head=5, rounds up -> overflow to -8446744073709551616
        -5000000000000000000L,  // head=-5, rounds down -> overflow to 8446744073709551616
        4000000000000000000L,   // head=4, rounds down -> 0
        -4000000000000000000L,  // head=-4, rounds down -> 0
        9223372036854775807L,   // LONG_MAX, head=9 -> overflow
    };
    cudf::test::fixed_width_column_wrapper<int64_t> input(values.begin(), values.end());
    
    auto result = expression::round(input, -19);
    
    // Expected values match Spark's overflow behavior
    std::vector<int64_t> expected_values = {
        -8446744073709551616L,  // 5e18 -> 1e19 overflow
        8446744073709551616L,   // -5e18 -> -1e19 overflow  
        0,                      // 4e18 -> rounds down
        0,                      // -4e18 -> rounds down
        -8446744073709551616L,  // LONG_MAX -> overflow
    };
    cudf::test::fixed_width_column_wrapper<int64_t> expected(expected_values.begin(), expected_values.end());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TEST_F(RoundTest, Int64BeyondMaxDigitsReturnsZero)
{
    // When -scale > 19 (beyond INT64 max digits), all values become 0
    std::vector<int64_t> values = {
        1234567890123456789L, 
        -1234567890123456789L,
        std::numeric_limits<int64_t>::max(),
        std::numeric_limits<int64_t>::min()
    };
    cudf::test::fixed_width_column_wrapper<int64_t> input(values.begin(), values.end());
    
    auto result = expression::round(input, -20);
    
    cudf::test::fixed_width_column_wrapper<int64_t> expected{0, 0, 0, 0};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TEST_F(RoundTest, Int64NormalNegativeScale)
{
    // Normal rounding with negative scale (within max digits)
    cudf::test::fixed_width_column_wrapper<int64_t> input{
        123456789012L, 
        567890123456L
    };
    
    auto result = expression::round(input, -6);
    
    // Round to nearest million
    cudf::test::fixed_width_column_wrapper<int64_t> expected{
        123457000000L, 
        567890000000L
    };
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TEST_F(RoundTest, Int8OverflowReturnsZero)
{
    // When -scale >= 3 (INT8 max digits), Spark returns 0
    std::vector<int8_t> values = {
        std::numeric_limits<int8_t>::max(),
        std::numeric_limits<int8_t>::min(),
        50,
        -50
    };
    cudf::test::fixed_width_column_wrapper<int8_t> input(values.begin(), values.end());
    
    auto result = expression::round(input, -3);
    
    cudf::test::fixed_width_column_wrapper<int8_t> expected{0, 0, 0, 0};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TEST_F(RoundTest, Int16OverflowReturnsZero)
{
    // When -scale >= 5 (INT16 max digits), Spark returns 0
    std::vector<int16_t> values = {
        std::numeric_limits<int16_t>::max(),
        std::numeric_limits<int16_t>::min(),
        12345,
        -12345
    };
    cudf::test::fixed_width_column_wrapper<int16_t> input(values.begin(), values.end());
    
    auto result = expression::round(input, -5);
    
    cudf::test::fixed_width_column_wrapper<int16_t> expected{0, 0, 0, 0};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TEST_F(RoundTest, IntegerWithNullsOverflow)
{
    // Verify nulls are preserved even with overflow
    cudf::test::fixed_width_column_wrapper<int32_t> input{{123456789, 987654321, 111111111}, {true, false, true}};
    
    auto result = expression::round(input, -10);
    
    cudf::test::fixed_width_column_wrapper<int32_t> expected{{0, 0, 0}, {true, false, true}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

// =============================================================================
// Float/Double Special Values Tests (Spark compatibility)
// =============================================================================

TEST_F(RoundTest, FloatInfinityPreserved)
{
    // When -scale >= 39, normal values become 0 but Inf/-Inf are preserved
    float pos_inf = std::numeric_limits<float>::infinity();
    float neg_inf = -std::numeric_limits<float>::infinity();
    
    cudf::test::fixed_width_column_wrapper<float> input{1.0f, pos_inf, neg_inf, 0.0f};
    
    auto result = expression::round(input, -40);
    
    cudf::test::fixed_width_column_wrapper<float> expected{0.0f, pos_inf, neg_inf, 0.0f};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
}

TEST_F(RoundTest, FloatNanPreserved)
{
    // NaN values are preserved even with extreme scales
    float nan = std::numeric_limits<float>::quiet_NaN();
    float pos_inf = std::numeric_limits<float>::infinity();
    
    cudf::test::fixed_width_column_wrapper<float> input{nan, 1.0f, pos_inf};
    
    auto result = expression::round(input, -40);
    
    // Column equivalence treats NaNs in the same position as equal
    cudf::test::fixed_width_column_wrapper<float> expected{nan, 0.0f, pos_inf};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
}

TEST_F(RoundTest, DoubleNanPreserved)
{
    // Double NaN and infinities preserved at extreme negative scales
    double nan = std::numeric_limits<double>::quiet_NaN();
    double pos_inf = std::numeric_limits<double>::infinity();
    double neg_inf = -std::numeric_limits<double>::infinity();

    cudf::test::fixed_width_column_wrapper<double> input{nan, 1.5, pos_inf, neg_inf};

    auto result = expression::bround(input, -310);

    cudf::test::fixed_width_column_wrapper<double> expected{nan, 0.0, pos_inf, neg_inf};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
}

TEST_F(RoundTest, DoubleInfinityPreserved)
{
    // When -scale >= 309, normal values become 0 but Inf/-Inf are preserved
    double pos_inf = std::numeric_limits<double>::infinity();
    double neg_inf = -std::numeric_limits<double>::infinity();
    
    cudf::test::fixed_width_column_wrapper<double> input{1.0, pos_inf, neg_inf, 1e100};
    
    auto result = expression::round(input, -310);
    
    cudf::test::fixed_width_column_wrapper<double> expected{0.0, pos_inf, neg_inf, 0.0};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
}

TEST_F(RoundTest, FloatLargePositiveScaleUnchanged)
{
    // When scale >= max_digits, values are returned unchanged
    cudf::test::fixed_width_column_wrapper<float> input{1.5f, 2.5f, 3.5f};
    
    auto result = expression::round(input, 40);
    
    cudf::test::fixed_width_column_wrapper<float> expected{1.5f, 2.5f, 3.5f};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
}

// =============================================================================
// BRound (HALF_EVEN) Overflow Tests
// =============================================================================

TEST_F(RoundTest, BRoundInt32Overflow)
{
    // BRound should also handle overflow the same way
    cudf::test::fixed_width_column_wrapper<int32_t> input{123456789, -987654321};
    
    auto result = expression::bround(input, -10);
    
    cudf::test::fixed_width_column_wrapper<int32_t> expected{0, 0};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TEST_F(RoundTest, BRoundFloatInfinity)
{
    float pos_inf = std::numeric_limits<float>::infinity();
    float neg_inf = -std::numeric_limits<float>::infinity();
    
    cudf::test::fixed_width_column_wrapper<float> input{1.0f, pos_inf, neg_inf};
    
    auto result = expression::bround(input, -40);
    
    cudf::test::fixed_width_column_wrapper<float> expected{0.0f, pos_inf, neg_inf};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
}

// =============================================================================
// Decimal Tests (Spark compatibility)
// =============================================================================

TEST_F(RoundTest, RoundDecimalNegativeScale)
{
    // Tests rounding to the left of the decimal point (tens, hundreds, etc.)
    // Input: 123.456 (represented as 123456 with scale -3)
    // round(123.456, -1) -> 120.0
    auto input = cudf::test::fixed_point_column_wrapper<int32_t>({123456, 789012}, {true, true}, numeric::scale_type{-3});
    
    auto result = expression::round(input, -1);
    
    // Spark's decimal rounding to negative places results in scale 0
    // Value 120 with scale 0 is 120.
    auto expected = cudf::test::fixed_point_column_wrapper<int32_t>({120, 790}, {true, true}, numeric::scale_type{0});
    
    EXPECT_EQ(result->type().scale(), 0);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
}

TEST_F(RoundTest, RoundDecimalPositiveScale)
{
    // Tests rounding to fewer decimal places than the input has.
    // Input: 123.456 (scale -3)
    // round(123.456, 1) -> 123.5 (scale -1)
    auto input = cudf::test::fixed_point_column_wrapper<int32_t>({123456}, {true}, numeric::scale_type{-3});
    
    auto result = expression::round(input, 1);
    
    // 123.5 represented as 1235 with scale -1
    auto expected = cudf::test::fixed_point_column_wrapper<int32_t>({1235}, {true}, numeric::scale_type{-1});
    
    EXPECT_EQ(result->type().scale(), -1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
}

TEST_F(RoundTest, RoundDecimalNoOpScale)
{
    // If target decimal places >= input scale, Spark treats it as a no-op
    // Input: 123.456 (scale -3)
    // round(123.456, 5) -> 123.456 (scale -3)
    auto input = cudf::test::fixed_point_column_wrapper<int32_t>({123456}, {true}, numeric::scale_type{-3});
    
    auto result = expression::round(input, 5);
    
    EXPECT_EQ(result->type().scale(), -3);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input, *result);
}

TEST_F(RoundTest, BRoundDecimal)
{
    // banker's rounding on decimals
    // 25.00, 35.00 (scale -2). Round to -1 (tens).
    // 25.00 -> 20 (nearest even ten)
    // 35.00 -> 40 (nearest even ten)
    auto input = cudf::test::fixed_point_column_wrapper<int32_t>({2500, 3500}, {true, true}, numeric::scale_type{-2});
    
    auto result = expression::bround(input, -1);
    
    // Result scale should be 0 for negative decimal_places
    auto expected = cudf::test::fixed_point_column_wrapper<int32_t>({20, 40}, {true, true}, numeric::scale_type{0});
    
    EXPECT_EQ(result->type().scale(), 0);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
}

TEST_F(RoundTest, RoundDecimal128ExtremeNegativeScale)
{
    // round(decimal128(38, 0), -40)
    // Scale is so negative it should correctly return 0
    __int128_t val = 1000000;
    auto input = cudf::test::fixed_point_column_wrapper<__int128_t>({val}, {true}, numeric::scale_type{0});
    
    auto result = expression::round(input, -40);
    
    auto expected = cudf::test::fixed_point_column_wrapper<__int128_t>({0}, {true}, numeric::scale_type{0});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
}

TEST_F(RoundTest, RoundDecimal32ScaleIncreaseOverflow)
{
    // 999,999,999 (scale 0)
    auto input = cudf::test::fixed_point_column_wrapper<int32_t>({999999999}, {true}, numeric::scale_type{0});
    
    // Round to 1 decimal place. 
    // If we increased scale to -1, it would require 9,999,999,990 which overflows int32.
    // Spark Rapids returns original values to avoid this.
    auto result = expression::round(input, 1);
    
    EXPECT_EQ(result->type().scale(), 0);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(input, *result);
}

}  // namespace test
}  // namespace spark_gpu_expr
