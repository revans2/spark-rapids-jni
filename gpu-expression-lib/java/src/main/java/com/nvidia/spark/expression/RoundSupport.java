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

/**
 * GPU support checker for Round and BRound expressions.
 * 
 * Provides a simple predicate API that takes an Expression and returns
 * true/false for whether it can be executed on GPU.
 * 
 * Validation rules (matching spark-rapids):
 * - Expression must be Round or BRound
 * - Scale must be a literal integer
 * - Value type must be numeric (byte, short, int, long, float, double, decimal)
 * - ANSI mode must be disabled (we don't support ANSI overflow checking)
 */
public final class RoundSupport {
    
    private RoundSupport() {
        // Static utility class
    }
    
    /**
     * Check if a Round or BRound expression can be executed on GPU.
     * 
     * @param expr The Spark SQL expression to check (should be Round or BRound)
     * @return true if this expression can run on GPU, false otherwise
     */
    public static boolean canTranslate(Expression expr) {
        Expression scaleExpr;
        DataType valueType;
        boolean ansiEnabled;
        
        if (expr instanceof Round) {
            Round round = (Round) expr;
            scaleExpr = round.scale();
            valueType = round.child().dataType();
            ansiEnabled = round.ansiEnabled();
        } else if (expr instanceof BRound) {
            BRound bround = (BRound) expr;
            scaleExpr = bround.scale();
            valueType = bround.child().dataType();
            ansiEnabled = bround.ansiEnabled();
        } else {
            return false;
        }
        
        // ANSI mode is not supported - we don't have overflow checking
        if (ansiEnabled) {
            return false;
        }
        
        // Scale must be a literal integer
        if (!isLiteralInt(scaleExpr)) {
            return false;
        }
        
        // Value type must be numeric
        return TypeSignature.isRoundSupported(valueType);
    }
    
    private static boolean isLiteralInt(Expression expr) {
        if (!(expr instanceof Literal)) {
            return false;
        }
        Literal lit = (Literal) expr;
        return lit.dataType() instanceof IntegerType;
    }
}
