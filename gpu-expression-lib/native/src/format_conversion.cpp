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

#include <spark_gpu_expr/date_format_expression.hpp>

#include <unordered_set>
#include <vector>

namespace spark_gpu_expr {

namespace {

// Characters not supported in Spark format strings
// Based on spark-rapids DateUtils.unsupportedCharacter
const std::unordered_set<char> UNSUPPORTED_CHARS = {
    'k', 'K', 'z', 'V', 'c', 'F', 'W', 'Q', 'q',
    'G', 'A', 'n', 'N', 'O', 'X', 'p', '\'',
    '[', ']', '#', '{', '}', 'Z', 'w', 'e', 'E', 'x', 'Y'
};

// Words/patterns not supported in Spark format strings
// Based on spark-rapids DateUtils.unsupportedWord
const std::unordered_set<std::string> UNSUPPORTED_WORDS = {
    // Week-based year patterns
    "u", "uu", "uuu", "uuuu", "uuuuu", "uuuuuu",
    "uuuuuuu", "uuuuuuuu", "uuuuuuuuu", "uuuuuuuuuu",
    // Unsupported year patterns (we only support yy and yyyy)
    "y", "yyy", "yyyyy", "yyyyyy", "yyyyyyy",
    "yyyyyyyy", "yyyyyyyyy", "yyyyyyyyyy",
    // Day of year
    "D", "DD", "DDD",
    // Single-letter patterns that need two letters
    "s", "m", "H", "h", "M", "d",
    // Month name patterns (not supported)
    "MMM", "MMMM", "MMMMM",
    // Standalone month patterns (only LL supported)
    "L", "LLL", "LLLL", "LLLLL",
    // Fractional seconds (only SSS and SSSSSS supported)
    "S", "SS", "SSSS", "SSSSS", "SSSSSSS", "SSSSSSSS", "SSSSSSSSS"
};

// Mapping from Spark patterns to strftime patterns
// Ordered by length (longest first) to avoid partial matches
const std::vector<std::pair<std::string, std::string>> FORMAT_CONVERSIONS = {
    {"SSSSSS", "%6f"},
    {"SSS", "%3f"},
    {"yyyy", "%Y"},
    {"yy", "%y"},
    {"MM", "%m"},
    {"LL", "%m"},
    {"dd", "%d"},
    {"HH", "%H"},
    {"mm", "%M"},
    {"ss", "%S"}
};

// Parse format string into words (groups of consecutive identical characters)
std::vector<std::string> parse_format_into_words(std::string const& format) {
    std::vector<std::string> words;
    if (format.empty()) return words;
    
    std::string current_word(1, format[0]);
    
    for (size_t i = 1; i < format.size(); ++i) {
        if (format[i] == current_word.back()) {
            current_word += format[i];
        } else {
            words.push_back(current_word);
            current_word = std::string(1, format[i]);
        }
    }
    
    if (!current_word.empty()) {
        words.push_back(current_word);
    }
    
    return words;
}

}  // anonymous namespace

std::optional<std::string> spark_format_to_strftime(std::string const& spark_format)
{
    // Parse format string into words
    auto words = parse_format_into_words(spark_format);
    
    // Validate all words
    for (const auto& word : words) {
        // Check first character for unsupported chars
        char first_char = word[0];
        if (UNSUPPORTED_CHARS.count(first_char) > 0) {
            return std::nullopt;
        }
        
        // Check for unsupported words
        if (UNSUPPORTED_WORDS.count(word) > 0) {
            return std::nullopt;
        }
    }
    
    // Build result by converting each word
    std::string result;
    for (const auto& word : words) {
        bool converted = false;
        for (const auto& [spark_pat, strf_pat] : FORMAT_CONVERSIONS) {
            if (word == spark_pat) {
                result += strf_pat;
                converted = true;
                break;
            }
        }
        if (!converted) {
            // Pass through as-is (separators, unknown patterns, etc.)
            result += word;
        }
    }
    
    return result;
}

bool is_format_supported(std::string const& spark_format)
{
    return spark_format_to_strftime(spark_format).has_value();
}

std::optional<std::string> validate_date_format_params(DateFormatParams const& params)
{
    if (!is_format_supported(params.format)) {
        return "Unsupported date format: " + params.format;
    }
    
    if (!is_timezone_supported(params.timezone)) {
        return "Unsupported timezone: " + params.timezone;
    }
    
    return std::nullopt;  // Valid
}

bool is_timezone_supported(std::string const& timezone)
{
    // For now, only UTC is fully supported without timezone database
    // TODO: Add full timezone support via cudf's timezone utilities
    return timezone == "UTC" || 
           timezone == "Etc/UTC" ||
           timezone.empty();
}

std::vector<std::string> get_supported_timezones()
{
    // Return the currently supported timezones
    return {"UTC", "Etc/UTC"};
}

}  // namespace spark_gpu_expr

