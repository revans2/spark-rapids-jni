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

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/detail/json.hpp>
#include <cudf/io/detail/tokenize_json.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/strip.hpp>
#include <cudf/unary.hpp>

#include <sstream>
#include <stdexcept>

namespace spark_rapids_jni {

// Print the content of the input device vector.
template <typename T, typename U = int>
void print_debug(rmm::device_uvector<T> const& input,
                 std::string const& name,
                 std::string const& separator,
                 rmm::cuda_stream_view stream)
{
  auto const h_input = cudf::detail::make_host_vector_sync(
    cudf::device_span<T const>{input.data(), input.size()}, stream);
  std::stringstream ss;
  ss << name << ":\n";
  for (size_t i = 0; i < h_input.size(); ++i) {
    ss << static_cast<U>(h_input[i]);
    if (separator.size() > 0 && i + 1 < h_input.size()) { ss << separator; }
  }
  std::cerr << ss.str() << std::endl;
}

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

  cudf::string_scalar s(needle, true, stream, mr);
  auto has_s = cudf::strings::contains(cudf::strings_column_view(input), s);
  auto any = cudf::make_any_aggregation<cudf::reduce_aggregation>();
  auto ret = cudf::reduce(*has_s, *any, cudf::data_type{cudf::type_id::BOOL8}, mr); // no stream is supported for reduce yet
  using BoolScalarType = cudf::scalar_type_t<bool>;
  return ret->is_valid(stream) && reinterpret_cast<BoolScalarType *>(ret.get())->value(stream);
}

std::unique_ptr<rmm::device_uvector<cudf::io::json::SymbolT>> extract_character_buffer(cudf::column_view const& input,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr) {  
  // Sadly there is no good way around this. We have to make a copy of the data...
  cudf::strings_column_view scv(input);
  auto data_length = scv.chars_size(stream);
  auto ret = std::make_unique<rmm::device_uvector<cudf::io::json::SymbolT>>(data_length, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(ret->data(),
                                scv.chars_begin(stream),
                                data_length,
                                cudaMemcpyDefault,
                                stream.value()));
  return ret;
}

std::pair<std::unique_ptr<rmm::device_uvector<cudf::io::json::SymbolT>>, std::unique_ptr<cudf::column>> clean_and_concat(
    cudf::column_view const& input,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr) {
  auto const input_scv  = cudf::strings_column_view{input};
  auto stripped = cudf::strings::strip(input_scv, cudf::strings::side_type::BOTH, cudf::string_scalar("", true, stream, mr), stream, mr);
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
  // Eventually we want to use null, but for now...
  auto all_done = cudf::strings::join_strings(cudf::strings_column_view(*cleaned),
      cudf::string_scalar("\n", true, stream, mr),
      cudf::string_scalar("{}", true, stream, mr), // This should be ignored
      stream,
      mr);
  return std::make_pair(extract_character_buffer(*all_done, stream, mr), std::move(is_n_or_e));
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

  auto [cleaned, was_empty] = clean_and_concat(input, stream, mr);
  print_debug<char, char>(*cleaned, "CLEANED INPUT", "", stream);


  // TODO we probably want a JSON options to pass in at some point. For now we are
  // just going to hard code thigns...

  // First off we need to get all of the data into a single buffer.  In the future
  // This will use \0 nul as the separator, but for now we are going to use \n
  // and check that it is not in there...

  throw std::runtime_error("NOT IMPLEMENTED YET");
}

}  // namespace spark_rapids_jni
