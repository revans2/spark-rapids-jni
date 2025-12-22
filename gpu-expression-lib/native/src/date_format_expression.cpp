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

#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>

namespace spark_gpu_expr {

std::unique_ptr<cudf::column> date_format(
    cudf::column_view const& timestamps,
    std::string const& spark_format,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
{
    // Convert Spark format to strftime format
    auto strf_format = spark_format_to_strftime(spark_format);
    CUDF_EXPECTS(strf_format.has_value(),
                 "Unsupported date format: " + spark_format);
    
    // Create empty names column (not needed for basic formatting)
    auto empty_names = cudf::strings_column_view(cudf::column_view{
        cudf::data_type{cudf::type_id::STRING}, 0, nullptr, nullptr, 0});
    
    // Use cudf's timestamp to string conversion
    return cudf::strings::from_timestamps(timestamps, strf_format.value(), empty_names, stream, mr);
}

std::unique_ptr<cudf::column> date_format(
    cudf::column_view const& timestamps,
    DateFormatParams const& params,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
{
    // Validate parameters
    auto error = validate_date_format_params(params);
    if (error.has_value()) {
        throw std::invalid_argument(error.value());
    }
    
    // TODO: Add timezone conversion before formatting
    // For now, we assume input is already in the desired timezone
    // and delegate to the simpler API
    return date_format(timestamps, params.format, stream, mr);
}

}  // namespace spark_gpu_expr
