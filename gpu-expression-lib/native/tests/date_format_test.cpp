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

#include <cudf/column/column_factories.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/column_utilities.hpp>

namespace spark_gpu_expr {
namespace test {

class DateFormatTest : public ::testing::Test {};

TEST_F(DateFormatTest, BasicDateFormat)
{
    // 2024-01-01 12:34:56 UTC
    // represented as microseconds since epoch
    using namespace cudf::test;
    int64_t ts = 1704112496000000L; 
    fixed_width_column_wrapper<cudf::timestamp_us, int64_t> input{{ts}, {true}};
    
    auto result = date_format(input, "yyyy-MM-dd HH:mm:ss");
    
    strings_column_wrapper expected{"2024-01-01 12:34:56"};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *result);
}

TEST_F(DateFormatTest, SupportedTimezones)
{
    using namespace cudf::test;
    int64_t ts = 1704112496000000L;
    fixed_width_column_wrapper<cudf::timestamp_us, int64_t> input{{ts}, {true}};
    
    DateFormatParams params;
    params.format = "yyyy-MM-dd";
    
    // UTC should work
    params.timezone = "UTC";
    EXPECT_NO_THROW(date_format(input, params));
    
    // Etc/UTC should work
    params.timezone = "Etc/UTC";
    EXPECT_NO_THROW(date_format(input, params));
    
    // Empty timezone should work (defaults to UTC)
    params.timezone = "";
    EXPECT_NO_THROW(date_format(input, params));
}

TEST_F(DateFormatTest, UnsupportedTimezoneThrows)
{
    using namespace cudf::test;
    int64_t ts = 1704112496000000L;
    fixed_width_column_wrapper<cudf::timestamp_us, int64_t> input{{ts}, {true}};
    
    DateFormatParams params;
    params.format = "yyyy-MM-dd";
    params.timezone = "America/Los_Angeles";
    
    // This should throw because we don't support it yet
    EXPECT_THROW(date_format(input, params), std::invalid_argument);
}

}  // namespace test
}  // namespace spark_gpu_expr
