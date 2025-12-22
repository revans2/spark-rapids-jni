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

public class ExpressionSupportTest {
    
    private AttributeReference createAttr(String name, DataType dataType) {
        return new AttributeReference(
            name,
            dataType,
            true,
            Metadata.empty(),
            ExprId.apply(1L),
            Seq$.MODULE$.<String>empty());
    }
    
    @Test
    public void testCanTranslateRound() {
        AttributeReference value = createAttr("value", DataTypes.DoubleType);
        Literal scale = Literal.create(2, DataTypes.IntegerType);
        Round round = new Round(value, scale, false);  // ansiEnabled = false
        
        assertTrue(ExpressionSupport.canTranslate(round));
    }
    
    @Test
    public void testCanTranslateDateFormat() {
        AttributeReference ts = createAttr("ts", DataTypes.TimestampType);
        Literal format = Literal.create(UTF8String.fromString("yyyy-MM-dd"), DataTypes.StringType);
        
        DateFormatClass dateFormat = new DateFormatClass(
            ts,
            format,
            Option.apply("UTC"));
        
        assertTrue(ExpressionSupport.canTranslate(dateFormat));
    }
    
    @Test
    public void testUnknownExpressionNotSupported() {
        // Create an expression we don't have a checker for
        Literal lit = Literal.create(UTF8String.fromString("hello"), DataTypes.StringType);
        
        assertFalse(ExpressionSupport.canTranslate(lit));
    }
    
    @Test
    public void testNullExpressionNotSupported() {
        assertFalse(ExpressionSupport.canTranslate(null));
    }
    
    @Test
    public void testDirectApiUsage() {
        // Test using individual support classes directly (recommended for flexibility)
        AttributeReference value = createAttr("value", DataTypes.DoubleType);
        Literal scale = Literal.create(2, DataTypes.IntegerType);
        Round round = new Round(value, scale, false);  // ansiEnabled = false
        
        // Direct API - allows intermixing with other GPU expression implementations
        assertTrue(RoundSupport.canTranslate(round));
    }
}
