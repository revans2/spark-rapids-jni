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

#include <spark_gpu_expr/types.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace spark_gpu_expr {

/**
 * Complete parameters for DateFormatClass expression execution.
 * 
 * All parameters required to produce Spark-compatible results.
 */
struct DateFormatParams {
    /**
     * Format string in Spark/Java DateTimeFormatter format.
     * 
     * Supported patterns:
     *   yyyy - 4-digit year
     *   yy   - 2-digit year
     *   MM   - month (01-12)
     *   dd   - day of month (01-31)
     *   HH   - hour (00-23)
     *   mm   - minute (00-59)
     *   ss   - second (00-59)
     *   SSS  - milliseconds
     *   SSSSSS - microseconds
     * 
     * Examples: "yyyy-MM-dd", "yyyy-MM-dd HH:mm:ss", "yyyy/MM/dd"
     */
    std::string format;
    
    /**
     * Timezone for timestamp interpretation.
     * 
     * REQUIRED for correct results. Timestamps in Spark are stored as UTC
     * and must be converted to the specified timezone before formatting.
     * 
     * Examples: "UTC", "America/Los_Angeles", "Europe/London", "Asia/Tokyo"
     * 
     * Note: Timezone support requires timezone database. If timezone handling
     * is not available, only "UTC" will be supported and others will error.
     */
    std::string timezone;
};

/**
 * Convert a Spark date format string to strftime format.
 * 
 * This is a pure CPU function for format string conversion.
 * 
 * @param spark_format The Spark/Java date format string
 * @return The equivalent strftime format, or nullopt if unsupported
 */
std::optional<std::string> spark_format_to_strftime(std::string const& spark_format);

/**
 * Check if a Spark date format string is supported for GPU execution.
 * 
 * @param spark_format The Spark/Java date format string
 * @return true if the format can be executed on GPU
 */
bool is_format_supported(std::string const& spark_format);

/**
 * Validate DateFormatParams before execution.
 * 
 * Checks:
 * - Format string is supported
 * - Timezone is valid (if timezone database is available)
 * 
 * @param params The parameters to validate
 * @return Error message if invalid, nullopt if valid
 */
std::optional<std::string> validate_date_format_params(DateFormatParams const& params);

/**
 * Check if a timezone is supported.
 * 
 * @param timezone The timezone string (e.g., "UTC", "America/Los_Angeles")
 * @return true if the timezone is supported
 */
bool is_timezone_supported(std::string const& timezone);

/**
 * Get list of supported timezones.
 * 
 * @return Vector of supported timezone names
 */
std::vector<std::string> get_supported_timezones();

/**
 * Format timestamps to strings with full Spark semantics.
 * 
 * This matches the behavior of Spark's date_format() function.
 * 
 * @param timestamps Column of timestamp values (TIMESTAMP_MICROSECONDS)
 * @param params Complete parameters including format and timezone
 * @param stream CUDA stream
 * @param mr Device memory resource
 * @return Column of formatted strings
 * @throws std::invalid_argument if params are invalid
 */
std::unique_ptr<cudf::column> date_format(
    cudf::column_view const& timestamps,
    DateFormatParams const& params,
    rmm::cuda_stream_view stream = rmm::cuda_stream_default,
    rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * Simpler date_format API for basic use cases.
 * 
 * @param timestamps Column of timestamp values
 * @param spark_format Spark format string
 * @param stream CUDA stream
 * @param mr Device memory resource
 * @return Column of formatted strings
 */
std::unique_ptr<cudf::column> date_format(
    cudf::column_view const& timestamps,
    std::string const& spark_format,
    rmm::cuda_stream_view stream = rmm::cuda_stream_default,
    rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

}  // namespace spark_gpu_expr
