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

package com.nvidia.spark.expression;

import org.apache.spark.sql.types.*;

/**
 * Utility class for checking Spark SQL data types for GPU compatibility.
 * 
 * This mirrors the type signature checking in spark-rapids ExprChecks.
 */
public final class TypeSignature {
    
    private TypeSignature() {
        // Utility class - no instantiation
    }
    
    /**
     * Check if a data type is numeric (supported for Round expression).
     * Includes: byte, short, int, long, float, double, and decimal.
     */
    public static boolean isNumeric(DataType dt) {
        return dt instanceof ByteType ||
               dt instanceof ShortType ||
               dt instanceof IntegerType ||
               dt instanceof LongType ||
               dt instanceof FloatType ||
               dt instanceof DoubleType ||
               dt instanceof DecimalType;
    }
    
    /**
     * Check if a data type is supported for Round operation.
     * This is currently the same as isNumeric but could diverge in the future.
     */
    public static boolean isRoundSupported(DataType dt) {
        return isNumeric(dt);
    }
    
    /**
     * Check if a data type is a timestamp type.
     */
    public static boolean isTimestamp(DataType dt) {
        return dt instanceof TimestampType;
    }
    
    /**
     * Check if a data type is a string type.
     */
    public static boolean isString(DataType dt) {
        return dt instanceof StringType;
    }
    
    /**
     * Check if a data type is a date type.
     */
    public static boolean isDate(DataType dt) {
        return dt instanceof DateType;
    }
    
    /**
     * Check if a data type is integral (whole numbers only).
     */
    public static boolean isIntegral(DataType dt) {
        return dt instanceof ByteType ||
               dt instanceof ShortType ||
               dt instanceof IntegerType ||
               dt instanceof LongType;
    }
    
    /**
     * Check if a data type is floating point.
     */
    public static boolean isFloatingPoint(DataType dt) {
        return dt instanceof FloatType ||
               dt instanceof DoubleType;
    }
}

