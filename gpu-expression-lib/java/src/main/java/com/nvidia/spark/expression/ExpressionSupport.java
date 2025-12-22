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

/**
 * Aggregated entry point for GPU expression support checking.
 * 
 * This class provides convenience methods that delegate to the individual
 * per-expression support classes. For maximum flexibility in intermixing
 * CUDF expressions with other GPU expression implementations, use the
 * individual support classes directly:
 * 
 * <ul>
 *   <li>{@link RoundSupport#canTranslate(Expression)} - Round and BRound</li>
 *   <li>{@link DateFormatSupport#canTranslate(Expression)} - DateFormatClass</li>
 * </ul>
 * 
 * Usage:
 * <pre>
 * // Using individual APIs (recommended for flexibility)
 * if (expr instanceof Round || expr instanceof BRound) {
 *     return RoundSupport.canTranslate(expr);
 * }
 * 
 * // Using aggregated API (convenience)
 * boolean supported = ExpressionSupport.canTranslate(expr);
 * </pre>
 */
public final class ExpressionSupport {
    
    private ExpressionSupport() {
        // Utility class - no instantiation
    }
    
    /**
     * Check if an expression can be translated to run on GPU.
     * 
     * Convenience method that dispatches to the appropriate per-expression
     * support class based on the expression type.
     * 
     * @param expr The Spark SQL expression to check
     * @return true if the expression can run on GPU, false otherwise
     */
    public static boolean canTranslate(Expression expr) {
        if (expr == null) {
            return false;
        }
        
        // Dispatch to per-expression support classes
        if (expr instanceof Round || expr instanceof BRound) {
            return RoundSupport.canTranslate(expr);
        }
        
        if (expr instanceof DateFormatClass) {
            return DateFormatSupport.canTranslate(expr);
        }
        
        // Unknown expression type
        return false;
    }
}
