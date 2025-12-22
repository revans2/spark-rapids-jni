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

#include "spark_gpu_expr/logical_type.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace spark_gpu_expr {

// StructField

bool StructField::operator==(StructField const& other) const {
    return name == other.name && 
           nullable == other.nullable &&
           type && other.type && type->equals(*other.type);
}

// LogicalType

std::string LogicalType::to_string() const {
    switch (type_id_) {
        case LogicalTypeId::BYTE: return "byte";
        case LogicalTypeId::SHORT: return "short";
        case LogicalTypeId::INT: return "int";
        case LogicalTypeId::LONG: return "long";
        case LogicalTypeId::FLOAT: return "float";
        case LogicalTypeId::DOUBLE: return "double";
        case LogicalTypeId::DECIMAL: return "decimal";
        case LogicalTypeId::BOOLEAN: return "boolean";
        case LogicalTypeId::STRING: return "string";
        case LogicalTypeId::BINARY: return "binary";
        case LogicalTypeId::DATE: return "date";
        case LogicalTypeId::TIMESTAMP: return "timestamp";
        case LogicalTypeId::TIMESTAMP_NTZ: return "timestamp_ntz";
        case LogicalTypeId::ARRAY: return "array";
        case LogicalTypeId::MAP: return "map";
        case LogicalTypeId::STRUCT: return "struct";
        case LogicalTypeId::NULL_TYPE: return "null";
        case LogicalTypeId::CALENDAR_INTERVAL: return "calendarinterval";
        case LogicalTypeId::YEAR_MONTH_INTERVAL: return "yearmonthinterval";
        case LogicalTypeId::DAY_TIME_INTERVAL: return "daytimeinterval";
        default: return "unknown";
    }
}

bool LogicalType::equals(LogicalType const& other) const {
    return type_id_ == other.type_id_;
}

bool LogicalType::is_numeric() const {
    switch (type_id_) {
        case LogicalTypeId::BYTE:
        case LogicalTypeId::SHORT:
        case LogicalTypeId::INT:
        case LogicalTypeId::LONG:
        case LogicalTypeId::FLOAT:
        case LogicalTypeId::DOUBLE:
        case LogicalTypeId::DECIMAL:
            return true;
        default:
            return false;
    }
}

bool LogicalType::is_integral() const {
    switch (type_id_) {
        case LogicalTypeId::BYTE:
        case LogicalTypeId::SHORT:
        case LogicalTypeId::INT:
        case LogicalTypeId::LONG:
            return true;
        default:
            return false;
    }
}

bool LogicalType::is_floating_point() const {
    return type_id_ == LogicalTypeId::FLOAT || type_id_ == LogicalTypeId::DOUBLE;
}

bool LogicalType::is_temporal() const {
    switch (type_id_) {
        case LogicalTypeId::DATE:
        case LogicalTypeId::TIMESTAMP:
        case LogicalTypeId::TIMESTAMP_NTZ:
        case LogicalTypeId::CALENDAR_INTERVAL:
        case LogicalTypeId::YEAR_MONTH_INTERVAL:
        case LogicalTypeId::DAY_TIME_INTERVAL:
            return true;
        default:
            return false;
    }
}

bool LogicalType::is_complex() const {
    switch (type_id_) {
        case LogicalTypeId::ARRAY:
        case LogicalTypeId::MAP:
        case LogicalTypeId::STRUCT:
            return true;
        default:
            return false;
    }
}

// Factory methods

LogicalTypePtr LogicalType::byte_type() {
    return std::make_shared<LogicalType>(LogicalTypeId::BYTE);
}

LogicalTypePtr LogicalType::short_type() {
    return std::make_shared<LogicalType>(LogicalTypeId::SHORT);
}

LogicalTypePtr LogicalType::int_type() {
    return std::make_shared<LogicalType>(LogicalTypeId::INT);
}

LogicalTypePtr LogicalType::long_type() {
    return std::make_shared<LogicalType>(LogicalTypeId::LONG);
}

LogicalTypePtr LogicalType::float_type() {
    return std::make_shared<LogicalType>(LogicalTypeId::FLOAT);
}

LogicalTypePtr LogicalType::double_type() {
    return std::make_shared<LogicalType>(LogicalTypeId::DOUBLE);
}

LogicalTypePtr LogicalType::boolean_type() {
    return std::make_shared<LogicalType>(LogicalTypeId::BOOLEAN);
}

LogicalTypePtr LogicalType::string_type() {
    return std::make_shared<LogicalType>(LogicalTypeId::STRING);
}

LogicalTypePtr LogicalType::binary_type() {
    return std::make_shared<LogicalType>(LogicalTypeId::BINARY);
}

LogicalTypePtr LogicalType::date_type() {
    return std::make_shared<LogicalType>(LogicalTypeId::DATE);
}

LogicalTypePtr LogicalType::timestamp_type() {
    return std::make_shared<LogicalType>(LogicalTypeId::TIMESTAMP);
}

LogicalTypePtr LogicalType::timestamp_ntz_type() {
    return std::make_shared<LogicalType>(LogicalTypeId::TIMESTAMP_NTZ);
}

LogicalTypePtr LogicalType::null_type() {
    return std::make_shared<LogicalType>(LogicalTypeId::NULL_TYPE);
}

LogicalTypePtr LogicalType::decimal_type(int32_t precision, int32_t scale) {
    return std::make_shared<DecimalLogicalType>(precision, scale);
}

LogicalTypePtr LogicalType::array_type(LogicalTypePtr element_type, bool contains_null) {
    return std::make_shared<ArrayLogicalType>(std::move(element_type), contains_null);
}

LogicalTypePtr LogicalType::map_type(LogicalTypePtr key_type, LogicalTypePtr value_type, bool value_contains_null) {
    return std::make_shared<MapLogicalType>(std::move(key_type), std::move(value_type), value_contains_null);
}

LogicalTypePtr LogicalType::struct_type(std::vector<StructField> fields) {
    return std::make_shared<StructLogicalType>(std::move(fields));
}

// DecimalLogicalType

DecimalLogicalType::DecimalLogicalType(int32_t precision, int32_t scale)
    : LogicalType(LogicalTypeId::DECIMAL)
    , precision_(precision)
    , scale_(scale) {
    if (precision < 1 || precision > 38) {
        throw std::invalid_argument("Decimal precision must be between 1 and 38");
    }
    if (scale < 0 || scale > precision) {
        throw std::invalid_argument("Decimal scale must be between 0 and precision");
    }
}

std::string DecimalLogicalType::to_string() const {
    std::ostringstream oss;
    oss << "decimal(" << precision_ << "," << scale_ << ")";
    return oss.str();
}

bool DecimalLogicalType::equals(LogicalType const& other) const {
    if (other.type_id() != LogicalTypeId::DECIMAL) {
        return false;
    }
    auto const& dec = static_cast<DecimalLogicalType const&>(other);
    return precision_ == dec.precision_ && scale_ == dec.scale_;
}

// ArrayLogicalType

ArrayLogicalType::ArrayLogicalType(LogicalTypePtr element_type, bool contains_null)
    : LogicalType(LogicalTypeId::ARRAY)
    , element_type_(std::move(element_type))
    , contains_null_(contains_null) {
    if (!element_type_) {
        throw std::invalid_argument("Array element type cannot be null");
    }
}

std::string ArrayLogicalType::to_string() const {
    return "array<" + element_type_->to_string() + ">";
}

bool ArrayLogicalType::equals(LogicalType const& other) const {
    if (other.type_id() != LogicalTypeId::ARRAY) {
        return false;
    }
    auto const& arr = static_cast<ArrayLogicalType const&>(other);
    return contains_null_ == arr.contains_null_ && element_type_->equals(*arr.element_type_);
}

// MapLogicalType

MapLogicalType::MapLogicalType(LogicalTypePtr key_type, LogicalTypePtr value_type, bool value_contains_null)
    : LogicalType(LogicalTypeId::MAP)
    , key_type_(std::move(key_type))
    , value_type_(std::move(value_type))
    , value_contains_null_(value_contains_null) {
    if (!key_type_ || !value_type_) {
        throw std::invalid_argument("Map key and value types cannot be null");
    }
}

std::string MapLogicalType::to_string() const {
    return "map<" + key_type_->to_string() + "," + value_type_->to_string() + ">";
}

bool MapLogicalType::equals(LogicalType const& other) const {
    if (other.type_id() != LogicalTypeId::MAP) {
        return false;
    }
    auto const& map = static_cast<MapLogicalType const&>(other);
    return value_contains_null_ == map.value_contains_null_ &&
           key_type_->equals(*map.key_type_) &&
           value_type_->equals(*map.value_type_);
}

// StructLogicalType

StructLogicalType::StructLogicalType(std::vector<StructField> fields)
    : LogicalType(LogicalTypeId::STRUCT)
    , fields_(std::move(fields)) {}

StructField const* StructLogicalType::field(std::string const& name) const {
    auto it = std::find_if(fields_.begin(), fields_.end(),
        [&name](StructField const& f) { return f.name == name; });
    return it != fields_.end() ? &(*it) : nullptr;
}

std::string StructLogicalType::to_string() const {
    std::ostringstream oss;
    oss << "struct<";
    for (std::size_t i = 0; i < fields_.size(); ++i) {
        if (i > 0) oss << ",";
        oss << fields_[i].name << ":" << fields_[i].type->to_string();
    }
    oss << ">";
    return oss.str();
}

bool StructLogicalType::equals(LogicalType const& other) const {
    if (other.type_id() != LogicalTypeId::STRUCT) {
        return false;
    }
    auto const& st = static_cast<StructLogicalType const&>(other);
    if (fields_.size() != st.fields_.size()) {
        return false;
    }
    for (std::size_t i = 0; i < fields_.size(); ++i) {
        if (!(fields_[i] == st.fields_[i])) {
            return false;
        }
    }
    return true;
}

}  // namespace spark_gpu_expr

