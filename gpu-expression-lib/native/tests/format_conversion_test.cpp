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

#include <spark_gpu_expr/date_format_expression.hpp>
#include <gtest/gtest.h>

namespace spark_gpu_expr {
namespace test {

class DateFormatConversionTest : public ::testing::Test {};

TEST_F(DateFormatConversionTest, BasicDate)
{
    auto result = spark_format_to_strftime("yyyy-MM-dd");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ("%Y-%m-%d", result.value());
}

TEST_F(DateFormatConversionTest, DateTime)
{
    auto result = spark_format_to_strftime("yyyy-MM-dd HH:mm:ss");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ("%Y-%m-%d %H:%M:%S", result.value());
}

TEST_F(DateFormatConversionTest, WithMilliseconds)
{
    auto result = spark_format_to_strftime("yyyy-MM-dd HH:mm:ss.SSS");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ("%Y-%m-%d %H:%M:%S.%3f", result.value());
}

TEST_F(DateFormatConversionTest, WithMicroseconds)
{
    auto result = spark_format_to_strftime("yyyy-MM-dd HH:mm:ss.SSSSSS");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ("%Y-%m-%d %H:%M:%S.%6f", result.value());
}

TEST_F(DateFormatConversionTest, IsoFormatWithUnquotedT)
{
    // Unquoted 'T' is allowed as an ISO 8601 separator
    auto result = spark_format_to_strftime("yyyy-MM-ddTHH:mm:ss");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ("%Y-%m-%dT%H:%M:%S", result.value());
}

TEST_F(DateFormatConversionTest, IsoFormatWithQuotedTNotSupported)
{
    // Quoted literals use single quotes which are not supported
    auto result = spark_format_to_strftime("yyyy-MM-dd'T'HH:mm:ss");
    EXPECT_FALSE(result.has_value());
}

TEST_F(DateFormatConversionTest, TwoDigitYear)
{
    auto result = spark_format_to_strftime("yy-MM-dd");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ("%y-%m-%d", result.value());
}

TEST_F(DateFormatConversionTest, SlashSeparator)
{
    auto result = spark_format_to_strftime("yyyy/MM/dd");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ("%Y/%m/%d", result.value());
}

TEST_F(DateFormatConversionTest, IsSupported_ValidFormats)
{
    EXPECT_TRUE(is_format_supported("yyyy-MM-dd"));
    EXPECT_TRUE(is_format_supported("yyyy-MM-dd HH:mm:ss"));
    EXPECT_TRUE(is_format_supported("yyyy/MM/dd"));
    EXPECT_TRUE(is_format_supported("yy-MM-dd"));
}

TEST_F(DateFormatConversionTest, IsSupported_InvalidFormats)
{
    // Timezone name
    EXPECT_FALSE(is_format_supported("yyyy-MM-dd z"));
    // Day of week
    EXPECT_FALSE(is_format_supported("yyyy-MM-dd E"));
    // Week-year (uppercase Y)
    EXPECT_FALSE(is_format_supported("YYYY-MM-dd"));
    // Era
    EXPECT_FALSE(is_format_supported("G yyyy-MM-dd"));
}

TEST_F(DateFormatConversionTest, UnsupportedReturnsNullopt)
{
    auto result = spark_format_to_strftime("yyyy-MM-dd z");
    EXPECT_FALSE(result.has_value());
}

// Tests for unsupported words (patterns of wrong length)
TEST_F(DateFormatConversionTest, UnsupportedWord_SingleY)
{
    // Single 'y' is not supported, must use 'yy' or 'yyyy'
    EXPECT_FALSE(is_format_supported("y-MM-dd"));
}

TEST_F(DateFormatConversionTest, UnsupportedWord_ThreeYs)
{
    // 'yyy' is not supported, must use 'yy' or 'yyyy'
    EXPECT_FALSE(is_format_supported("yyy-MM-dd"));
}

TEST_F(DateFormatConversionTest, UnsupportedWord_SingleM)
{
    // Single 'M' is not supported, must use 'MM'
    EXPECT_FALSE(is_format_supported("yyyy-M-dd"));
}

TEST_F(DateFormatConversionTest, UnsupportedWord_MonthName)
{
    // 'MMM' (month name abbreviation) is not supported
    EXPECT_FALSE(is_format_supported("yyyy-MMM-dd"));
}

TEST_F(DateFormatConversionTest, UnsupportedWord_DayOfYear)
{
    // 'D' and 'DDD' (day of year) are not supported
    EXPECT_FALSE(is_format_supported("yyyy-DDD"));
    EXPECT_FALSE(is_format_supported("D"));
}

TEST_F(DateFormatConversionTest, UnsupportedWord_SingleFracSecond)
{
    // Single 'S' or 'SS' is not supported, must use 'SSS' or 'SSSSSS'
    EXPECT_FALSE(is_format_supported("yyyy-MM-dd HH:mm:ss.S"));
    EXPECT_FALSE(is_format_supported("yyyy-MM-dd HH:mm:ss.SS"));
    EXPECT_FALSE(is_format_supported("yyyy-MM-dd HH:mm:ss.SSSS"));
    EXPECT_FALSE(is_format_supported("yyyy-MM-dd HH:mm:ss.SSSSS"));
}

TEST_F(DateFormatConversionTest, UnsupportedWord_SingleLetterPatterns)
{
    // Single letter patterns (except separators) are not supported
    EXPECT_FALSE(is_format_supported("yyyy-MM-d"));     // single 'd' not supported
    EXPECT_FALSE(is_format_supported("yyyy-MM-dd H:mm:ss")); // single 'H' not supported
    EXPECT_FALSE(is_format_supported("yyyy-MM-dd HH:m:ss")); // single 'm' not supported
    EXPECT_FALSE(is_format_supported("yyyy-MM-dd HH:mm:s")); // single 's' not supported
}

TEST_F(DateFormatConversionTest, ValidateParams_Timezones)
{
    DateFormatParams params;
    params.format = "yyyy-MM-dd";
    
    // UTC should be valid
    params.timezone = "UTC";
    EXPECT_FALSE(validate_date_format_params(params).has_value());
    
    params.timezone = "Etc/UTC";
    EXPECT_FALSE(validate_date_format_params(params).has_value());
    
    params.timezone = "";
    EXPECT_FALSE(validate_date_format_params(params).has_value());
    
    // Non-UTC should be invalid
    params.timezone = "America/Los_Angeles";
    auto error = validate_date_format_params(params);
    ASSERT_TRUE(error.has_value());
    EXPECT_EQ("Unsupported timezone: America/Los_Angeles", error.value());
}

}  // namespace test
}  // namespace spark_gpu_expr

