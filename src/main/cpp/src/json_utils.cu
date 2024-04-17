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

#include <cudf/io/detail/json.hpp>

#include <stdexcept>

namespace spark_rapids_jni {

// TODO is this really the right way to do this???
std::pair<cudf::column, cudf::column> clean_and_concat(cudf::column_view const & input) {
  /*
  if (input.is_empty()) {
    // make an empty buffer
    return cudf::detail::make_device_uvector_async<char>(
      std::vector<char>{'[', ']'}, stream, rmm::mr::get_current_device_resource());
  }

  auto const d_strings  = cudf::column_device_view::create(input, stream);
  auto const input_scv  = cudf::strings_column_view{input};
  auto const chars_size = input_scv.chars_size(stream);
  auto const output_size =
    2l +                                            // two extra bracket characters '[' and ']'
    static_cast<int64_t>(chars_size) +
    static_cast<int64_t>(input.size() - 1) +        // append `,` character between input rows
    static_cast<int64_t>(input.null_count()) * 2l;  // replace null with "{}"
  // TODO: This assertion eventually needs to be removed.
  // See https://github.com/NVIDIA/spark-rapids-jni/issues/1707
  CUDF_EXPECTS(output_size <= static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max()),
               "The input json column is too large and causes overflow.");

  auto const joined_input = cudf::strings::detail::join_strings(
    input_scv,
    cudf::string_scalar(","),   // append `,` character between the input rows
    cudf::string_scalar("{}"),  // replacement for null rows
    stream,
    rmm::mr::get_current_device_resource());
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

  // TODO we probably want a JSON options to pass in at some point. For now we are
  // just going to hard code thigns...

  // First off we need to get all of the data into a single buffer.  In the future
  // This will use \0 nul as the separator, but for now we are going to use \n
  // and check that it is not in there...

  auto [concated, was_empty] = clean_and_concat(input);

  throw std::runtime_error("NOT IMPLEMENTED YET");
}

}  // namespace spark_rapids_jni
