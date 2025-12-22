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

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.apache.spark.sql.catalyst.expressions.*;
import org.apache.spark.sql.types.*;
import scala.Option;

/**
 * GPU support checker for DateFormatClass expressions.
 * 
 * Provides a simple predicate API that takes an Expression and returns
 * true/false for whether it can be executed on GPU.
 * 
 * Validation rules (matching spark-rapids):
 * - Expression must be DateFormatClass
 * - Input must be timestamp type
 * - Format must be a string literal
 * - Format string must use only supported patterns
 */
public final class DateFormatSupport {
    
    private DateFormatSupport() {
        // Static utility class
    }
    
    /**
     * Characters that are NOT supported in format strings.
     * From spark-rapids DateUtils.unsupportedCharacter
     */
    private static final Set<Character> UNSUPPORTED_CHARS = new HashSet<>();
    
    /**
     * Words/patterns that are NOT supported in format strings.
     * From spark-rapids DateUtils.unsupportedWord
     */
    private static final Set<String> UNSUPPORTED_WORDS = new HashSet<>();
    
    private static final Set<String> SUPPORTED_TIMEZONES = new HashSet<>();

    static {
        char[] unsupportedChars = {
            'k', 'K', 'z', 'V', 'c', 'F', 'W', 'Q', 'q',
            'G', 'A', 'n', 'N', 'O', 'X', 'p', '\'',
            '[', ']', '#', '{', '}', 'Z', 'w', 'e', 'E', 'x', 'Y'
        };
        for (char c : unsupportedChars) {
            UNSUPPORTED_CHARS.add(c);
        }
        
        String[] unsupportedWords = {
            "u", "uu", "uuu", "uuuu", "uuuuu", "uuuuuu", 
            "uuuuuuu", "uuuuuuuu", "uuuuuuuuu", "uuuuuuuuuu",
            "y", "yyy", "yyyyy", "yyyyyy", "yyyyyyy", 
            "yyyyyyyy", "yyyyyyyyy", "yyyyyyyyyy",
            "D", "DD", "DDD",
            "s", "m", "H", "h", "M", "d",
            "MMM", "MMMM", "MMMMM",
            "L", "LLL", "LLLL", "LLLLL",
            "S", "SS", "SSSS", "SSSSS", "SSSSSSS", "SSSSSSSS", "SSSSSSSSS"
        };
        for (String word : unsupportedWords) {
            UNSUPPORTED_WORDS.add(word);
        }

        String[] supportedTimezones = {"UTC", "Etc/UTC"};
        for (String tz : supportedTimezones) {
            SUPPORTED_TIMEZONES.add(tz.toUpperCase());
        }
    }
    
    /**
     * Check if a DateFormatClass expression can be executed on GPU.
     * 
     * @param expr The Spark SQL expression to check (should be DateFormatClass)
     * @return true if this expression can run on GPU, false otherwise
     */
    public static boolean canTranslate(Expression expr) {
        if (!(expr instanceof DateFormatClass)) {
            return false;
        }
        
        DateFormatClass dateFormat = (DateFormatClass) expr;
        
        // Input must be timestamp type
        if (!TypeSignature.isTimestamp(dateFormat.left().dataType())) {
            return false;
        }
        
        // Format must be a string literal
        Expression formatExpr = dateFormat.right();
        if (!(formatExpr instanceof Literal)) {
            return false;
        }
        
        Literal formatLit = (Literal) formatExpr;
        if (!TypeSignature.isString(formatLit.dataType())) {
            return false;
        }
        
        Object value = formatLit.value();
        if (value == null) {
            return false;
        }
        
        if (!isTimezoneSupported(dateFormat.timeZoneId())) {
            return false;
        }

        return isFormatSupported(value.toString());
    }
    
    private static boolean isFormatSupported(String format) {
        List<String> words = parseFormatIntoWords(format);
        
        for (String word : words) {
            char firstChar = word.charAt(0);
            if (UNSUPPORTED_CHARS.contains(firstChar)) {
                return false;
            }
            if (UNSUPPORTED_WORDS.contains(word)) {
                return false;
            }
        }
        
        return true;
    }

    private static boolean isTimezoneSupported(Option<String> tzOpt) {
        String tz = tzOpt == null ? null : tzOpt.getOrElse(null);
        if (tz == null || tz.isEmpty()) {
            // treat unspecified as UTC (matches native validation)
            return true;
        }
        return SUPPORTED_TIMEZONES.contains(tz.toUpperCase());
    }
    
    private static List<String> parseFormatIntoWords(String format) {
        List<String> words = new ArrayList<>();
        if (format == null || format.isEmpty()) {
            return words;
        }
        
        StringBuilder currentWord = new StringBuilder();
        currentWord.append(format.charAt(0));
        
        for (int i = 1; i < format.length(); i++) {
            char c = format.charAt(i);
            char lastChar = currentWord.charAt(currentWord.length() - 1);
            
            if (c == lastChar) {
                currentWord.append(c);
            } else {
                words.add(currentWord.toString());
                currentWord = new StringBuilder();
                currentWord.append(c);
            }
        }
        
        if (currentWord.length() > 0) {
            words.add(currentWord.toString());
        }
        
        return words;
    }
}