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
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import scala.collection.immutable.Seq$;

public class RoundSupportTest {
    
    private AttributeReference createAttr(String name, DataType dataType) {
        return new AttributeReference(
            name,
            dataType,
            true,
            Metadata.empty(),
            ExprId.apply(1L),
            Seq$.MODULE$.<String>empty());
    }
    
    private Literal createIntLiteral(int value) {
        return Literal.create(value, DataTypes.IntegerType);
    }
    
    // Helper to create Round with ANSI mode disabled
    private Round createRound(Expression child, Expression scale) {
        return new Round(child, scale, false);  // ansiEnabled = false
    }
    
    // Helper to create BRound with ANSI mode disabled
    private BRound createBRound(Expression child, Expression scale) {
        return new BRound(child, scale, false);  // ansiEnabled = false
    }
    
    @Test
    public void testRoundWithLiteralScaleSupported() {
        AttributeReference value = createAttr("value", DataTypes.DoubleType);
        Literal scale = createIntLiteral(2);
        Round round = createRound(value, scale);
        
        assertTrue(RoundSupport.canTranslate(round));
    }
    
    @Test
    public void testRoundWithNonLiteralScaleNotSupported() {
        AttributeReference value = createAttr("value", DataTypes.DoubleType);
        AttributeReference scale = createAttr("scale", DataTypes.IntegerType);
        Round round = createRound(value, scale);
        
        assertFalse(RoundSupport.canTranslate(round));
    }
    
    @Test
    public void testBRoundSupported() {
        AttributeReference value = createAttr("value", DataTypes.FloatType);
        Literal scale = createIntLiteral(0);
        BRound bround = createBRound(value, scale);
        
        assertTrue(RoundSupport.canTranslate(bround));
    }
    
    @Test
    public void testIntegerTypeSupported() {
        AttributeReference value = createAttr("value", DataTypes.IntegerType);
        Literal scale = createIntLiteral(0);
        Round round = createRound(value, scale);
        
        assertTrue(RoundSupport.canTranslate(round));
    }
    
    @Test
    public void testLongTypeSupported() {
        AttributeReference value = createAttr("value", DataTypes.LongType);
        Literal scale = createIntLiteral(0);
        Round round = createRound(value, scale);
        
        assertTrue(RoundSupport.canTranslate(round));
    }
    
    @Test
    public void testDecimalTypeSupported() {
        DataType decimalType = DataTypes.createDecimalType(10, 2);
        AttributeReference value = createAttr("value", decimalType);
        Literal scale = createIntLiteral(0);
        Round round = createRound(value, scale);
        
        assertTrue(RoundSupport.canTranslate(round));
    }
    
    @Test
    public void testStringTypeNotSupported() {
        AttributeReference value = createAttr("value", DataTypes.StringType);
        Literal scale = createIntLiteral(0);
        Round round = createRound(value, scale);
        
        assertFalse(RoundSupport.canTranslate(round));
    }
    
    @Test
    public void testAnsiModeNotSupported() {
        AttributeReference value = createAttr("value", DataTypes.DoubleType);
        Literal scale = createIntLiteral(2);
        // Create with ANSI mode enabled
        Round round = new Round(value, scale, true);
        
        assertFalse(RoundSupport.canTranslate(round));
    }
    
    @Test
    public void testBRoundAnsiModeNotSupported() {
        AttributeReference value = createAttr("value", DataTypes.FloatType);
        Literal scale = createIntLiteral(0);
        // Create with ANSI mode enabled
        BRound bround = new BRound(value, scale, true);
        
        assertFalse(RoundSupport.canTranslate(bround));
    }
}
