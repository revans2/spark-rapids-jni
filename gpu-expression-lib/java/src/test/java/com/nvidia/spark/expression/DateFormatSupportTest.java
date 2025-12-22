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

import org.apache.spark.sql.catalyst.expressions.*;
import org.apache.spark.sql.types.*;
import org.apache.spark.unsafe.types.UTF8String;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import scala.Option;
import scala.collection.immutable.Seq$;

public class DateFormatSupportTest {
    
    private AttributeReference createTimestampAttr() {
        return new AttributeReference(
            "ts",
            DataTypes.TimestampType,
            true,
            Metadata.empty(),
            ExprId.apply(1L),
            Seq$.MODULE$.<String>empty());
    }
    
    private Literal createStringLiteral(String value) {
        return Literal.create(UTF8String.fromString(value), DataTypes.StringType);
    }
    
    @Test
    public void testSupportedFormat_BasicDate() {
        DateFormatClass dateFormat = new DateFormatClass(
            createTimestampAttr(),
            createStringLiteral("yyyy-MM-dd"),
            Option.apply("UTC"));
        
        assertTrue(DateFormatSupport.canTranslate(dateFormat));
    }
    
    @Test
    public void testSupportedFormat_DateTime() {
        DateFormatClass dateFormat = new DateFormatClass(
            createTimestampAttr(),
            createStringLiteral("yyyy-MM-dd HH:mm:ss"),
            Option.apply("UTC"));
        
        assertTrue(DateFormatSupport.canTranslate(dateFormat));
    }
    
    @Test
    public void testSupportedFormat_WithMillis() {
        DateFormatClass dateFormat = new DateFormatClass(
            createTimestampAttr(),
            createStringLiteral("yyyy-MM-dd HH:mm:ss.SSS"),
            Option.apply("UTC"));
        
        assertTrue(DateFormatSupport.canTranslate(dateFormat));
    }
    
    @Test
    public void testSupportedFormat_SlashSeparator() {
        DateFormatClass dateFormat = new DateFormatClass(
            createTimestampAttr(),
            createStringLiteral("yyyy/MM/dd"),
            Option.apply("UTC"));
        
        assertTrue(DateFormatSupport.canTranslate(dateFormat));
    }
    
    @Test
    public void testUnsupportedFormat_TimezoneChar() {
        // 'z' is timezone name - not supported
        DateFormatClass dateFormat = new DateFormatClass(
            createTimestampAttr(),
            createStringLiteral("yyyy-MM-dd z"),
            Option.apply("UTC"));
        
        assertFalse(DateFormatSupport.canTranslate(dateFormat));
    }
    
    @Test
    public void testUnsupportedFormat_DayOfWeek() {
        // 'E' is day of week - not supported
        DateFormatClass dateFormat = new DateFormatClass(
            createTimestampAttr(),
            createStringLiteral("yyyy-MM-dd E"),
            Option.apply("UTC"));
        
        assertFalse(DateFormatSupport.canTranslate(dateFormat));
    }
    
    @Test
    public void testNonLiteralFormatNotSupported() {
        AttributeReference format = new AttributeReference(
            "fmt",
            DataTypes.StringType,
            true,
            Metadata.empty(),
            ExprId.apply(2L),
            Seq$.MODULE$.<String>empty());
        
        DateFormatClass dateFormat = new DateFormatClass(
            createTimestampAttr(),
            format,
            Option.apply("UTC"));
        
        assertFalse(DateFormatSupport.canTranslate(dateFormat));
    }
    
    @Test
    public void testNonTimestampInputNotSupported() {
        AttributeReference dateAttr = new AttributeReference(
            "d",
            DataTypes.DateType,
            true,
            Metadata.empty(),
            ExprId.apply(1L),
            Seq$.MODULE$.<String>empty());
        
        DateFormatClass dateFormat = new DateFormatClass(
            dateAttr,
            createStringLiteral("yyyy-MM-dd"),
            Option.apply("UTC"));
        
        assertFalse(DateFormatSupport.canTranslate(dateFormat));
    }
    
    @Test
    public void testUnsupportedWord_SingleY() {
        // Single 'y' is not supported, must use 'yy' or 'yyyy'
        DateFormatClass dateFormat = new DateFormatClass(
            createTimestampAttr(),
            createStringLiteral("y-MM-dd"),
            Option.apply("UTC"));
        
        assertFalse(DateFormatSupport.canTranslate(dateFormat));
    }
    
    @Test
    public void testUnsupportedWord_ThreeYs() {
        // 'yyy' is not supported, must use 'yy' or 'yyyy'
        DateFormatClass dateFormat = new DateFormatClass(
            createTimestampAttr(),
            createStringLiteral("yyy-MM-dd"),
            Option.apply("UTC"));
        
        assertFalse(DateFormatSupport.canTranslate(dateFormat));
    }
    
    @Test
    public void testUnsupportedWord_SingleM() {
        // Single 'M' is not supported, must use 'MM'
        DateFormatClass dateFormat = new DateFormatClass(
            createTimestampAttr(),
            createStringLiteral("yyyy-M-dd"),
            Option.apply("UTC"));
        
        assertFalse(DateFormatSupport.canTranslate(dateFormat));
    }
    
    @Test
    public void testUnsupportedWord_MonthName() {
        // 'MMM' (month name) is not supported
        DateFormatClass dateFormat = new DateFormatClass(
            createTimestampAttr(),
            createStringLiteral("yyyy-MMM-dd"),
            Option.apply("UTC"));
        
        assertFalse(DateFormatSupport.canTranslate(dateFormat));
    }
    
    @Test
    public void testUnsupportedWord_DayOfYear() {
        // 'D' (day of year) is not supported
        DateFormatClass dateFormat = new DateFormatClass(
            createTimestampAttr(),
            createStringLiteral("yyyy-DDD"),
            Option.apply("UTC"));
        
        assertFalse(DateFormatSupport.canTranslate(dateFormat));
    }
    
    @Test
    public void testUnsupportedWord_SingleFracSecond() {
        // Single 'S' is not supported, must use 'SSS' or 'SSSSSS'
        DateFormatClass dateFormat = new DateFormatClass(
            createTimestampAttr(),
            createStringLiteral("yyyy-MM-dd HH:mm:ss.S"),
            Option.apply("UTC"));
        
        assertFalse(DateFormatSupport.canTranslate(dateFormat));
    }
    
    @Test
    public void testSupportedFormat_Microseconds() {
        // 'SSSSSS' (microseconds) is supported
        DateFormatClass dateFormat = new DateFormatClass(
            createTimestampAttr(),
            createStringLiteral("yyyy-MM-dd HH:mm:ss.SSSSSS"),
            Option.apply("UTC"));
        
        assertTrue(DateFormatSupport.canTranslate(dateFormat));
    }
    
    @Test
    public void testSupportedFormat_TwoDigitYear() {
        // 'yy' (two-digit year) is supported
        DateFormatClass dateFormat = new DateFormatClass(
            createTimestampAttr(),
            createStringLiteral("yy-MM-dd"),
            Option.apply("UTC"));
        
        assertTrue(DateFormatSupport.canTranslate(dateFormat));
    }

    @Test
    public void testNonUtcTimezoneNotSupported() {
        // Non-UTC timezone should be rejected until native adds timezone DB support
        DateFormatClass dateFormat = new DateFormatClass(
            createTimestampAttr(),
            createStringLiteral("yyyy-MM-dd"),
            Option.apply("America/Los_Angeles"));

        assertFalse(DateFormatSupport.canTranslate(dateFormat));
    }
}
