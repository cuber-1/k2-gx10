cmake_minimum_required(VERSION 3.22)
project(k2_q6k_microbenchmark LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(LLAMA_CPP_ROOT "/home/dvijraicha/llama.cpp" CACHE PATH "Pinned llama.cpp source tree")
if(NOT EXISTS "${LLAMA_CPP_ROOT}/ggml/CMakeLists.txt")
    message(FATAL_ERROR "LLAMA_CPP_ROOT does not identify the expected llama.cpp tree")
endif()

set(GGML_CUDA ON CACHE BOOL "Build the isolated CUDA backend" FORCE)
set(GGML_CUDA_GRAPHS OFF CACHE BOOL "One launch must remain visible to profilers" FORCE)
set(GGML_CUDA_NCCL OFF CACHE BOOL "NCCL is unnecessary for a one-GPU microbenchmark" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "Keep the isolated build reproducible" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "Do not build broad backend test suites" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "Do not build unrelated examples" FORCE)

add_subdirectory("${LLAMA_CPP_ROOT}/ggml" "${CMAKE_BINARY_DIR}/llama.cpp-ggml" EXCLUDE_FROM_ALL)

add_executable(q6k-microbench src/q6k-microbench.cpp)
target_include_directories(q6k-microbench PRIVATE "${LLAMA_CPP_ROOT}/ggml/src")
target_compile_definitions(q6k-microbench PRIVATE
    K2_GGML_COMMON_HEADER="${LLAMA_CPP_ROOT}/ggml/src/ggml-common.h")
target_link_libraries(q6k-microbench PRIVATE ggml)
target_compile_options(q6k-microbench PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(q6k-ffn-microbench src/q6k-ffn-microbench.cpp)
target_include_directories(q6k-ffn-microbench PRIVATE "${LLAMA_CPP_ROOT}/ggml/src")
target_compile_definitions(q6k-ffn-microbench PRIVATE
    K2_GGML_COMMON_HEADER="${LLAMA_CPP_ROOT}/ggml/src/ggml-common.h")
target_link_libraries(q6k-ffn-microbench PRIVATE ggml)
target_compile_options(q6k-ffn-microbench PRIVATE -Wall -Wextra -Wpedantic -Werror)
