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
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace spark_gpu_expr {

/**
 * Logical type identifiers matching Spark's DataType hierarchy.
 * 
 * This enum represents the kind of data, separate from physical storage.
 * For example, DECIMAL can be stored as int32, int64, or int128 depending
 * on precision, but logically it's always DECIMAL.
 */
enum class LogicalTypeId {
    // Numeric types
    BYTE,           // Spark ByteType (signed 8-bit integer)
    SHORT,          // Spark ShortType (signed 16-bit integer)  
    INT,            // Spark IntegerType (signed 32-bit integer)
    LONG,           // Spark LongType (signed 64-bit integer)
    FLOAT,          // Spark FloatType (32-bit IEEE 754)
    DOUBLE,         // Spark DoubleType (64-bit IEEE 754)
    DECIMAL,        // Spark DecimalType (precision/scale)
    
    // Boolean
    BOOLEAN,        // Spark BooleanType
    
    // String/Binary
    STRING,         // Spark StringType (UTF-8)
    BINARY,         // Spark BinaryType (raw bytes)
    
    // Date/Time
    DATE,           // Spark DateType (days since epoch)
    TIMESTAMP,      // Spark TimestampType (microseconds since epoch, UTC)
    TIMESTAMP_NTZ,  // Spark TimestampNTZType (microseconds, no timezone)
    
    // Complex types
    ARRAY,          // Spark ArrayType
    MAP,            // Spark MapType
    STRUCT,         // Spark StructType
    
    // Null type
    NULL_TYPE,      // Spark NullType
    
    // Calendar interval (not commonly used in expressions)
    CALENDAR_INTERVAL,
    
    // Year-Month and Day-Time intervals (Spark 3.2+)
    YEAR_MONTH_INTERVAL,
    DAY_TIME_INTERVAL
};

// Forward declarations
class LogicalType;
using LogicalTypePtr = std::shared_ptr<LogicalType>;

/**
 * Struct field definition for StructType.
 */
struct StructField {
    std::string name;
    LogicalTypePtr type;
    bool nullable = true;
    
    bool operator==(StructField const& other) const;
};

/**
 * Logical type representation for Spark data types.
 * 
 * Provides a type system independent of both Spark's JVM types and cudf's
 * physical types. This allows:
 * 1. No coupling to Java/JVM
 * 2. Expression of logical properties (decimal precision/scale, struct fields)
 * 3. Easy mapping to either Spark DataType or cudf data_type
 */
class LogicalType {
public:
    LogicalType(LogicalTypeId id) : type_id_(id) {}
    virtual ~LogicalType() = default;
    
    LogicalTypeId type_id() const { return type_id_; }
    
    /**
     * Human-readable type name (e.g., "int", "decimal(10,2)", "array<string>")
     */
    virtual std::string to_string() const;
    
    /**
     * Check if this type equals another.
     */
    virtual bool equals(LogicalType const& other) const;
    
    bool operator==(LogicalType const& other) const { return equals(other); }
    bool operator!=(LogicalType const& other) const { return !equals(other); }
    
    // Convenience type checks
    bool is_numeric() const;
    bool is_integral() const;
    bool is_floating_point() const;
    bool is_decimal() const { return type_id_ == LogicalTypeId::DECIMAL; }
    bool is_string() const { return type_id_ == LogicalTypeId::STRING; }
    bool is_temporal() const;
    bool is_complex() const;
    bool is_nullable() const { return nullable_; }
    
    void set_nullable(bool nullable) { nullable_ = nullable; }
    
    // Factory methods for simple types
    static LogicalTypePtr byte_type();
    static LogicalTypePtr short_type();
    static LogicalTypePtr int_type();
    static LogicalTypePtr long_type();
    static LogicalTypePtr float_type();
    static LogicalTypePtr double_type();
    static LogicalTypePtr boolean_type();
    static LogicalTypePtr string_type();
    static LogicalTypePtr binary_type();
    static LogicalTypePtr date_type();
    static LogicalTypePtr timestamp_type();
    static LogicalTypePtr timestamp_ntz_type();
    static LogicalTypePtr null_type();
    
    // Factory methods for parameterized types
    static LogicalTypePtr decimal_type(int32_t precision, int32_t scale);
    static LogicalTypePtr array_type(LogicalTypePtr element_type, bool contains_null = true);
    static LogicalTypePtr map_type(LogicalTypePtr key_type, LogicalTypePtr value_type, bool value_contains_null = true);
    static LogicalTypePtr struct_type(std::vector<StructField> fields);

protected:
    LogicalTypeId type_id_;
    bool nullable_ = true;
};

/**
 * Decimal type with precision and scale.
 */
class DecimalLogicalType : public LogicalType {
public:
    DecimalLogicalType(int32_t precision, int32_t scale);
    
    int32_t precision() const { return precision_; }
    int32_t scale() const { return scale_; }
    
    std::string to_string() const override;
    bool equals(LogicalType const& other) const override;
    
private:
    int32_t precision_;
    int32_t scale_;
};

/**
 * Array type with element type.
 */
class ArrayLogicalType : public LogicalType {
public:
    ArrayLogicalType(LogicalTypePtr element_type, bool contains_null = true);
    
    LogicalTypePtr element_type() const { return element_type_; }
    bool contains_null() const { return contains_null_; }
    
    std::string to_string() const override;
    bool equals(LogicalType const& other) const override;
    
private:
    LogicalTypePtr element_type_;
    bool contains_null_;
};

/**
 * Map type with key and value types.
 */
class MapLogicalType : public LogicalType {
public:
    MapLogicalType(LogicalTypePtr key_type, LogicalTypePtr value_type, bool value_contains_null = true);
    
    LogicalTypePtr key_type() const { return key_type_; }
    LogicalTypePtr value_type() const { return value_type_; }
    bool value_contains_null() const { return value_contains_null_; }
    
    std::string to_string() const override;
    bool equals(LogicalType const& other) const override;
    
private:
    LogicalTypePtr key_type_;
    LogicalTypePtr value_type_;
    bool value_contains_null_;
};

/**
 * Struct type with named fields.
 */
class StructLogicalType : public LogicalType {
public:
    explicit StructLogicalType(std::vector<StructField> fields);
    
    std::vector<StructField> const& fields() const { return fields_; }
    std::size_t num_fields() const { return fields_.size(); }
    
    /**
     * Get field by index.
     */
    StructField const& field(std::size_t index) const { return fields_.at(index); }
    
    /**
     * Get field by name (returns nullptr if not found).
     */
    StructField const* field(std::string const& name) const;
    
    std::string to_string() const override;
    bool equals(LogicalType const& other) const override;
    
private:
    std::vector<StructField> fields_;
};

}  // namespace spark_gpu_expr

