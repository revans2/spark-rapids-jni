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

#pragma once

#include <spark_gpu_expr/round_float.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/round.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

namespace spark_gpu_expr {
namespace expression {

/**
 * @brief Spark-compatible round expression.
 *
 * Rounds values to the specified number of decimal places using HALF_UP rounding,
 * matching Spark's round() function behavior.
 *
 * Supported types (matching Spark's NumericType):
 * - Integer types: INT8, INT16, INT32, INT64
 * - Floating point: FLOAT32, FLOAT64
 * - Fixed point: DECIMAL32, DECIMAL64, DECIMAL128
 *
 * Spark compatibility features:
 * - Integer overflow handling: returns 0 when -scale >= max_digits for the type
 * - Float/Double special values: preserves NaN, +Inf, -Inf when -scale >= max_digits
 * - Int64 19-digit overflow: handles round-up overflow at the max digit boundary
 *
 * @param input Column of values to round
 * @param decimal_places Number of decimal places (default 0). Negative values round
 *                       to positions left of the decimal point.
 * @param stream CUDA stream
 * @param mr Memory resource
 * @return Rounded column
 */
std::unique_ptr<cudf::column> round(
    cudf::column_view const& input,
    int32_t decimal_places            = 0,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Spark-compatible bround expression.
 *
 * Rounds values to the specified number of decimal places using HALF_EVEN rounding
 * (banker's rounding), matching Spark's bround() function behavior.
 *
 * Uses the same type support and overflow handling as round().
 *
 * @param input Column of values to round
 * @param decimal_places Number of decimal places (default 0). Negative values round
 *                       to positions left of the decimal point.
 * @param stream CUDA stream
 * @param mr Memory resource
 * @return Rounded column
 */
std::unique_ptr<cudf::column> bround(
    cudf::column_view const& input,
    int32_t decimal_places            = 0,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace expression
}  // namespace spark_gpu_expr

