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

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

namespace spark_gpu_expr {
namespace expression {

namespace {

// Max digits for each integer type before overflow
// These match the precision values from cudf's DType
constexpr int32_t MAX_DIGITS_INT8  = 3;   // max value 127
constexpr int32_t MAX_DIGITS_INT16 = 5;   // max value 32767
constexpr int32_t MAX_DIGITS_INT32 = 10;  // max value 2147483647
constexpr int32_t MAX_DIGITS_INT64 = 19;  // max value 9223372036854775807

// Max digits for floating point before overflow to zero
constexpr int32_t MAX_DIGITS_FLOAT32 = 39;   // ~3.4e38
constexpr int32_t MAX_DIGITS_FLOAT64 = 309;  // ~1.8e308

/**
 * Get the max digits for an integer type.
 */
int32_t get_max_digits_for_int_type(cudf::type_id type_id)
{
    switch (type_id) {
        case cudf::type_id::INT8:  return MAX_DIGITS_INT8;
        case cudf::type_id::INT16: return MAX_DIGITS_INT16;
        case cudf::type_id::INT32: return MAX_DIGITS_INT32;
        case cudf::type_id::INT64: return MAX_DIGITS_INT64;
        default: return 0;
    }
}

/**
 * Special handling for Int64 at exactly 19 digits (-scale == 19).
 * 
 * Compared to other numeric types, Int64 is special because values with head digit > 4
 * would round UP to ±1e19, which overflows Long. Spark preserves this overflow behavior:
 *   - 1e19 overflows to -8446744073709551616L
 *   - -1e19 overflows to 8446744073709551616L
 * 
 * Translated from Scala: GpuRoundBase.fixUpInt64OnBounds
 */
std::unique_ptr<cudf::column> fix_up_int64_on_bounds(
    cudf::column_view const& input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
{
    // Scalars for computation
    auto const base = cudf::numeric_scalar<int64_t>(1000000000000000000L, true, stream);  // 1e18
    auto const four = cudf::numeric_scalar<int64_t>(4L, true, stream);
    auto const minus_four = cudf::numeric_scalar<int64_t>(-4L, true, stream);
    
    // Replacement values
    auto const zero = cudf::numeric_scalar<int64_t>(0L, true, stream);
    // These are the overflow values that Spark produces
    auto const pos_overflow = cudf::numeric_scalar<int64_t>(8446744073709551616L, true, stream);
    auto const neg_overflow = cudf::numeric_scalar<int64_t>(-8446744073709551616L, true, stream);
    
    // head_digit = input / 1e18
    auto head_digit = cudf::binary_operation(
        input, base, cudf::binary_operator::DIV, input.type(), stream, mr);
    
    // need_pos_replace = head_digit > 4 (values like 5e18..9e18 that round up to 1e19)
    auto need_pos_replace = cudf::binary_operation(
        *head_digit, four, cudf::binary_operator::GREATER, 
        cudf::data_type{cudf::type_id::BOOL8}, stream, mr);
    
    // need_neg_replace = head_digit < -4 (values like -5e18..-9e18 that round to -1e19)
    auto need_neg_replace = cudf::binary_operation(
        *head_digit, minus_four, cudf::binary_operator::LESS,
        cudf::data_type{cudf::type_id::BOOL8}, stream, mr);
    
    // Build result:
    // - If head > 4: return neg_overflow (-8446744073709551616L) - matches Spark's 1e19 overflow
    // - If head < -4: return pos_overflow (8446744073709551616L) - matches Spark's -1e19 overflow
    // - Otherwise: return 0 (rounds down correctly)
    
    // neg_branch = need_neg_replace ? pos_overflow : zero
    auto neg_branch = cudf::copy_if_else(pos_overflow, zero, *need_neg_replace, stream, mr);
    
    // result = need_pos_replace ? neg_overflow : neg_branch
    auto result = cudf::copy_if_else(neg_overflow, *neg_branch, *need_pos_replace, stream, mr);
    
    // Preserve null mask from input
    if (input.has_nulls()) {
        auto null_mask = cudf::detail::copy_bitmask(input, stream, mr);
        result->set_null_mask(std::move(null_mask), input.null_count());
    }
    
    return result;
}

/**
 * Fixes up integral values rounded by a scale exceeding/reaching the max digits.
 * When -scale >= max_digits, cuDF may produce different results than Spark.
 * 
 * For scales exceeding max digits, we return zero values.
 * For Int64 at exactly 19 digits, special handling is needed for round-up overflow.
 * 
 * Translated from Scala: GpuRoundBase.fixUpOverflowInts
 */
template <typename T>
std::unique_ptr<cudf::column> fix_up_overflow_ints(
    cudf::column_view const& input,
    int32_t decimal_places,
    cudf::rounding_method method,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
{
    int32_t max_digits = get_max_digits_for_int_type(input.type().id());
    
    // Special case: Int64 rounding on exactly the 19th digit
    // Values with head > 4 or head < -4 would overflow when rounded up
    // We return the overflow values to match Spark's behavior
    if constexpr (std::is_same_v<T, int64_t>) {
        if (-decimal_places == MAX_DIGITS_INT64) {
            return fix_up_int64_on_bounds(input, stream, mr);
        }
    }
    
    // For scales exceeding max digits, return zero values
    if (-decimal_places >= max_digits) {
        auto const zero = cudf::numeric_scalar<T>(T{0}, true, stream);
        auto result = cudf::make_column_from_scalar(zero, input.size(), stream, mr);
        
        // Preserve null mask from input
        if (input.has_nulls()) {
            auto null_mask = cudf::detail::copy_bitmask(input, stream, mr);
            result->set_null_mask(std::move(null_mask), input.null_count());
        }
        
        return result;
    }
    
    // Normal case: use spark_gpu_expr::round
    return spark_gpu_expr::round(input, decimal_places, method, stream, mr);
}

/**
 * Fixes up floating point values when scale exceeds max digits.
 * Preserves NaN, +Inf, -Inf while replacing normal values with 0.
 * 
 * Translated from Scala: GpuRoundBase.fpZeroReplacement
 */
template <typename T>
std::unique_ptr<cudf::column> fp_zero_replacement(
    cudf::column_view const& input,
    int32_t decimal_places,
    cudf::rounding_method method,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
{
    constexpr int32_t max_digits = std::is_same_v<T, float> ? MAX_DIGITS_FLOAT32 : MAX_DIGITS_FLOAT64;
    
    if (-decimal_places >= max_digits) {
        // Scale exceeds max digits: replace normal values with 0, keep NaN/Inf
        auto const zero = cudf::numeric_scalar<T>(T{0}, true, stream);
        auto const pos_inf = cudf::numeric_scalar<T>(std::numeric_limits<T>::infinity(), true, stream);
        auto const neg_inf = cudf::numeric_scalar<T>(-std::numeric_limits<T>::infinity(), true, stream);
        
        // Use cudf's is_nan function
        auto is_nan_col = cudf::is_nan(input, stream, mr);
        
        // Check for +inf
        auto is_pos_inf = cudf::binary_operation(
            input, pos_inf, cudf::binary_operator::EQUAL,
            cudf::data_type{cudf::type_id::BOOL8}, stream, mr);
        
        // Check for -inf
        auto is_neg_inf = cudf::binary_operation(
            input, neg_inf, cudf::binary_operator::EQUAL,
            cudf::data_type{cudf::type_id::BOOL8}, stream, mr);
        
        // is_special = is_nan OR is_pos_inf OR is_neg_inf
        auto is_nan_or_pos_inf = cudf::binary_operation(
            *is_nan_col, *is_pos_inf, cudf::binary_operator::LOGICAL_OR,
            cudf::data_type{cudf::type_id::BOOL8}, stream, mr);
        
        auto is_special = cudf::binary_operation(
            *is_nan_or_pos_inf, *is_neg_inf, cudf::binary_operator::LOGICAL_OR,
            cudf::data_type{cudf::type_id::BOOL8}, stream, mr);
        
        // result = is_special ? input : 0
        return cudf::copy_if_else(input, zero, *is_special, stream, mr);
    }
    
    if (decimal_places >= max_digits) {
        // Scale is very large positive: just return the original values
        return std::make_unique<cudf::column>(input, stream, mr);
    }
    
    // Normal case: use spark_gpu_expr::round
    return spark_gpu_expr::round(input, decimal_places, method, stream, mr);
}

/**
 * Type dispatcher for Spark-compatible round with overflow handling.
 */
struct spark_round_dispatcher {
    int32_t decimal_places;
    cudf::rounding_method method;
    rmm::cuda_stream_view stream;
    rmm::device_async_resource_ref mr;
    
    template <typename T>
    std::enable_if_t<std::is_integral_v<T> && std::is_signed_v<T>, std::unique_ptr<cudf::column>>
    operator()(cudf::column_view const& input) const
    {
        // Only apply fix-ups to true integral types, not decimals
        if (cudf::is_fixed_point(input.type())) {
            return spark_gpu_expr::round(input, decimal_places, method, stream, mr);
        }
        return fix_up_overflow_ints<T>(input, decimal_places, method, stream, mr);
    }
    
    template <typename T>
    std::enable_if_t<std::is_floating_point_v<T>, std::unique_ptr<cudf::column>>
    operator()(cudf::column_view const& input) const
    {
        return fp_zero_replacement<T>(input, decimal_places, method, stream, mr);
    }
    
    template <typename T>
    std::enable_if_t<!std::is_signed_v<T> || (!std::is_integral_v<T> && !std::is_floating_point_v<T>),
                     std::unique_ptr<cudf::column>>
    operator()(cudf::column_view const& input) const
    {
        // For unsigned types and fixed-point (handled above), just use the base round
        return spark_gpu_expr::round(input, decimal_places, method, stream, mr);
    }
};

/**
 * Spark-compatible decimal rounding logic.
 */
std::unique_ptr<cudf::column> round_decimal_spark(
    cudf::column_view const& input,
    int32_t decimal_places,
    cudf::rounding_method method,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
{
    int32_t input_scale = -input.type().scale();
    
    // 1. If target decimal places >= input scale, it's a no-op in Spark
    if (decimal_places >= input_scale) {
        return std::make_unique<cudf::column>(input, stream, mr);
    }
    
    // 2. Perform the round
    auto rounded = spark_gpu_expr::round(input, decimal_places, method, stream, mr);
    
    // 3. Match Spark's output scale
    // If decimal_places < 0, output scale is 0.
    // If 0 <= decimal_places < input_scale, output scale is decimal_places.
    int32_t target_scale = std::max(0, decimal_places);
    if (-rounded->type().scale() != target_scale) {
        auto output_type = cudf::data_type{input.type().id(), -target_scale};
        return cudf::cast(rounded->view(), output_type, stream, mr);
    }
    
    return rounded;
}

}  // anonymous namespace

std::unique_ptr<cudf::column> round(
    cudf::column_view const& input,
    int32_t decimal_places,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
{
    if (input.is_empty()) {
        return cudf::empty_like(input);
    }
    
    if (cudf::is_fixed_point(input.type())) {
        return round_decimal_spark(input, decimal_places, cudf::rounding_method::HALF_UP, stream, mr);
    }
    
    // Use type dispatcher for Spark-compatible overflow handling
    return cudf::type_dispatcher(
        input.type(),
        spark_round_dispatcher{decimal_places, cudf::rounding_method::HALF_UP, stream, mr},
        input);
}

std::unique_ptr<cudf::column> bround(
    cudf::column_view const& input,
    int32_t decimal_places,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
{
    if (input.is_empty()) {
        return cudf::empty_like(input);
    }
    
    if (cudf::is_fixed_point(input.type())) {
        return round_decimal_spark(input, decimal_places, cudf::rounding_method::HALF_EVEN, stream, mr);
    }
    
    // Use type dispatcher for Spark-compatible overflow handling
    return cudf::type_dispatcher(
        input.type(),
        spark_round_dispatcher{decimal_places, cudf::rounding_method::HALF_EVEN, stream, mr},
        input);
}

}  // namespace expression
}  // namespace spark_gpu_expr
