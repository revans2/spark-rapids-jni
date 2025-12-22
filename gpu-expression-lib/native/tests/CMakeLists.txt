# =============================================================================
# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

# We need to include the source code defined in cudftestutil_impl, but it is an interface library
# that does not compile, we need to define an object library for the compilation. By doing so, we
# only compile such source files once instead of doing so for every test executable.
add_library(spark_gpu_expr_test_common OBJECT)

target_compile_options(spark_gpu_expr_test_common
                PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_FLAGS}>"
                        "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>")

target_link_libraries(
    spark_gpu_expr_test_common
    PUBLIC cudf::cudftestutil GTest::gmock GTest::gmock_main GTest::gtest GTest::gtest_main
    PRIVATE cudf::cudftestutil_impl
)

# Function to configure test executables
function(ConfigureTest CMAKE_TEST_NAME)
    add_executable(${CMAKE_TEST_NAME} ${ARGN})
    target_compile_options(${CMAKE_TEST_NAME}
                PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_FLAGS}>"
                        "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>")
    target_include_directories(${CMAKE_TEST_NAME}
                PRIVATE "$<BUILD_INTERFACE:${CUDF_DIR}/cpp>"
                        "$<BUILD_INTERFACE:${CUDF_DIR}/cpp/include>"
                        "$<BUILD_INTERFACE:${CUDF_DIR}/cpp/tests>")
    set_target_properties(
        ${CMAKE_TEST_NAME}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/gtests>"
                   CXX_STANDARD 20
                   CXX_STANDARD_REQUIRED ON
                   CUDA_STANDARD 20
                   CUDA_STANDARD_REQUIRED ON
    )
    target_link_libraries(${CMAKE_TEST_NAME} 
        PRIVATE 
            spark_gpu_expr_test_common
            spark_gpu_expr
    )
    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
endfunction(ConfigureTest)

# Format conversion tests (CPU-only, doesn't require cudf test utilities)
add_executable(format_conversion_tests
    format_conversion_test.cpp
)

set_target_properties(
  format_conversion_tests
  PROPERTIES CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
)

target_link_libraries(format_conversion_tests
    PRIVATE
        spark_gpu_expr
        GTest::gtest
        GTest::gtest_main
)

add_test(NAME format_conversion_tests COMMAND format_conversion_tests)

# Round tests (requires GPU and cudf test utilities)
ConfigureTest(round_tests round_test.cpp)

# Date format tests (requires GPU and cudf test utilities)
ConfigureTest(date_format_tests date_format_test.cpp)
