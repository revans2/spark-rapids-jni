/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "json_utils.hpp"
#include "map_utils_debug.cuh"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/detail/json.hpp>
#include <cudf/io/detail/tokenize_json.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/strip.hpp>
#include <cudf/unary.hpp>

#include <stdexcept>

namespace spark_rapids_jni {

std::unique_ptr<cudf::column> is_empty_or_null(
    cudf::column_view const& input, 
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr) {

  auto byte_count = cudf::strings::count_bytes(cudf::strings_column_view{input}, mr); // stream not exposed yet...
  using IntScalarType = cudf::scalar_type_t<int32_t>;
  auto zero = cudf::make_numeric_scalar(cudf::data_type{cudf::type_id::INT32}, stream, mr);
  reinterpret_cast<IntScalarType *>(zero.get())->set_value(0, stream);
  zero->set_valid_async(true, stream);
  auto is_empty = cudf::binary_operation(*byte_count, *zero, cudf::binary_operator::LESS_EQUAL, cudf::data_type{cudf::type_id::BOOL8}, stream, mr);
  auto is_null = cudf::is_null(input, stream, mr);
  auto mostly_empty_or_null = cudf::binary_operation(*is_empty, *is_null, cudf::binary_operator::NULL_LOGICAL_OR, cudf::data_type{cudf::type_id::BOOL8}, stream, mr);
  is_empty.reset();
  is_null.reset();
  zero.reset();
  auto null_lit = cudf::make_string_scalar("null", stream, mr);
  auto is_lit_null = cudf::binary_operation(*null_lit, input, cudf::binary_operator::EQUAL, cudf::data_type{cudf::type_id::BOOL8}, stream, mr);
  return cudf::binary_operation(*is_lit_null, *mostly_empty_or_null, cudf::binary_operator::NULL_LOGICAL_OR, cudf::data_type{cudf::type_id::BOOL8}, stream, mr);
}

bool contains_char(
    cudf::column_view const& input,
    std::string const& needle,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr) {

  cudf::string_scalar s(needle, stream, mr);
  auto has_s = cudf::strings::contains(cudf::strings_column_view(input), s);
  auto any = cudf::make_any_aggregation<cudf::reduce_aggregation>();
  auto ret = cudf::reduce(*has_s, *any, cudf::data_type{cudf::type_id::BOOL8}, mr); // no stream is supported for reduce yet
  using BoolScalarType = cudf::scalar_type_t<bool>;
  return ret->is_valid(stream) && reinterpret_cast<BoolScalarType *>(ret.get())->value(stream);
}

std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>> clean(
    cudf::column_view const& input,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr) {
  auto const input_scv  = cudf::strings_column_view{input};
  auto stripped = cudf::strings::strip(input_scv, cudf::strings::side_type::BOTH, cudf::string_scalar(""), stream, mr);
  auto is_n_or_e = is_empty_or_null(*stripped, stream, mr);
  auto empty_row = cudf::make_string_scalar("{}", stream, mr);
  auto cleaned = cudf::copy_if_else(*empty_row, *stripped, *is_n_or_e, stream, mr);
  stripped.reset();
  empty_row.reset();
  if (contains_char(*cleaned, "\n", stream, mr)) {
    throw std::logic_error("line separator is not currently supported in a JSON string");
  }
  if (contains_char(*cleaned, "\r", stream, mr)) {
    throw std::logic_error("carriage return is not currently supported in a JSON string");
  }



  // TODO probably want to have/use a data source instead of a concat buffer.

  /*
  auto const d_strings  = cudf::column_device_view::create(input, stream);
  auto const chars_size = input_scv.chars_size(stream);
  auto const output_size =
    static_cast<int64_t>(chars_size) +
    static_cast<int64_t>(input.size() - 1) +        // append `\n` character between input rows
    static_cast<int64_t>(input.null_count()) * 2l;  // replace null with "{}" (we probably want to deal with empty strings too)
  // TODO: This assertion eventually needs to be removed.
  // See https://github.com/NVIDIA/spark-rapids-jni/issues/1707
  CUDF_EXPECTS(output_size <= static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max()),
               "The input json column is too large and causes overflow.");

  auto const joined_input = cudf::strings::detail::join_strings(
    input_scv,
    cudf::string_scalar("\n"),   // append `,` character between the input rows
    cudf::string_scalar("{}"),  // replacement for null rows
    stream,
    mr);
  auto const joined_input_scv        = cudf::strings_column_view{*joined_input};
  auto const joined_input_size_bytes = joined_input_scv.chars_size(stream);
  // TODO: This assertion requires a stream synchronization, may want to remove at some point.
  // See https://github.com/NVIDIA/spark-rapids-jni/issues/1707
  CUDF_EXPECTS(joined_input_size_bytes + 2 == output_size, "Incorrect output size computation.");

  // We want to concatenate 3 strings: "[" + joined_input + "]".
  // For efficiency, let's use memcpy instead of `cudf::strings::detail::concatenate`.
  auto output = rmm::device_uvector<char>(joined_input_size_bytes + 2, stream);
  CUDF_CUDA_TRY(cudaMemsetAsync(output.data(), static_cast<int>('['), 1, stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(output.data() + 1,
                                joined_input_scv.chars_begin(stream),
                                joined_input_size_bytes,
                                cudaMemcpyDefault,
                                stream.value()));
  CUDF_CUDA_TRY(cudaMemsetAsync(
    output.data() + joined_input_size_bytes + 1, static_cast<int>(']'), 1, stream.value()));

#ifdef DEBUG_FROM_JSON
  print_debug<char, char>(output, "Processed json string", "", stream);
#endif
  return output;

  if (input.data
  */
  throw std::runtime_error("NOT DONE YET");
}

std::unique_ptr<cudf::column> tokenize_json(
  cudf::column_view const& input,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr) {

  CUDF_EXPECTS(input.type().id() == cudf::type_id::STRING, "Invalid input format");

  if (input.is_empty()) {
    auto tok_out = cudf::make_empty_column(cudf::type_id::INT8);
    auto offset_out = cudf::make_empty_column(cudf::type_id::UINT32);
    std::vector<std::unique_ptr<cudf::column>> tok_off_children;
    tok_off_children.push_back(std::move(tok_out));
    tok_off_children.push_back(std::move(offset_out));
    auto tok_off_out = cudf::make_structs_column(0, std::move(tok_off_children), 0, rmm::device_buffer{}, stream, mr);
    auto empty_offsets = cudf::make_empty_column(cudf::type_id::INT32);
    auto tokens_out = cudf::make_lists_column(0, std::move(empty_offsets), std::move(tok_off_out), 0, rmm::device_buffer{}, stream, mr);
    auto buffer_out = cudf::make_empty_column(cudf::type_id::STRING);
    std::vector<std::unique_ptr<cudf::column>> children;
    children.push_back(std::move(buffer_out));
    children.push_back(std::move(tokens_out));
    return cudf::make_structs_column(0, std::move(children), 0, rmm::device_buffer{}, stream, mr);
  }

  auto [cleaned, was_empty] = clean(input, stream, mr);

  // TODO we probably want a JSON options to pass in at some point. For now we are
  // just going to hard code thigns...

  // First off we need to get all of the data into a single buffer.  In the future
  // This will use \0 nul as the separator, but for now we are going to use \n
  // and check that it is not in there...

  throw std::runtime_error("NOT IMPLEMENTED YET");
}

}  // namespace spark_rapids_jni
