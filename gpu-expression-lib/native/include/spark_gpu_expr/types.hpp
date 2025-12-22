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

#include <cstdint>
#include <optional>
#include <string>
#include <variant>

namespace spark_gpu_expr {

/**
 * Overflow handling policy.
 * Corresponds to spark.sql.ansi.enabled behavior.
 */
enum class OverflowPolicy {
    ERROR,    // ANSI mode ON: throw exception on overflow
    NULLIFY   // ANSI mode OFF: return null on overflow
};

/**
 * Spark DecimalType representation.
 * precision: Total number of digits (1-38)
 * scale: Number of digits after decimal point (0 to precision)
 */
struct DecimalType {
    int32_t precision;
    int32_t scale;
    
    bool operator==(DecimalType const& other) const {
        return precision == other.precision && scale == other.scale;
    }
};

/**
 * Spark numeric type representation for expression output types.
 * Covers all numeric types that expressions can produce.
 */
enum class NumericTypeId {
    INT8,       // ByteType
    INT16,      // ShortType
    INT32,      // IntegerType
    INT64,      // LongType
    FLOAT32,    // FloatType
    FLOAT64,    // DoubleType
    DECIMAL     // DecimalType (use DecimalType struct for precision/scale)
};

/**
 * Complete output type specification for numeric expressions.
 */
struct OutputType {
    NumericTypeId type_id;
    
    // Only valid when type_id == DECIMAL
    std::optional<DecimalType> decimal_type;
    
    // Factory methods for clarity
    static OutputType int8() { return {NumericTypeId::INT8, std::nullopt}; }
    static OutputType int16() { return {NumericTypeId::INT16, std::nullopt}; }
    static OutputType int32() { return {NumericTypeId::INT32, std::nullopt}; }
    static OutputType int64() { return {NumericTypeId::INT64, std::nullopt}; }
    static OutputType float32() { return {NumericTypeId::FLOAT32, std::nullopt}; }
    static OutputType float64() { return {NumericTypeId::FLOAT64, std::nullopt}; }
    static OutputType decimal(int32_t precision, int32_t scale) {
        return {NumericTypeId::DECIMAL, DecimalType{precision, scale}};
    }
};

}  // namespace spark_gpu_expr

