cmake_minimum_required(VERSION 3.14)
project(nexterabert.cpp VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(NEXTERA_BUILD_SHARED "Build libnexterabert as a shared library (needed by the Python binding)" ON)
set(NEXTERA_GGML_TAG "v0.9.9" CACHE STRING "ggml git tag / commit to build against")
set(NEXTERA_GGML_DIR "" CACHE PATH "Use an existing ggml checkout instead of fetching one")

option(NEXTERA_NATIVE "Optimize for the CPU this is built on" ON)
option(NEXTERA_OPENMP "Use OpenMP instead of ggml's own thread pool" OFF)
option(NEXTERA_NO_VNNI "Build the int8 GEMM without VNNI even if this CPU has it (testing)" OFF)

# ggml (the tensor library under llama.cpp) supplies the GGUF reader, the
# Q4_0 / Q4_K / Q8_0 ... block formats and their SIMD kernels.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE ${NEXTERA_NATIVE} CACHE BOOL "" FORCE)
set(GGML_OPENMP ${NEXTERA_OPENMP} CACHE BOOL "" FORCE)
# tinyBLAS: blocked GEMM kernels for F32 / F16 / Q8_0 / Q4_0 weights.
set(GGML_LLAMAFILE ON CACHE BOOL "" FORCE)

# MSVC has no -march=native: probe the two x86 extensions that matter here.
# (GCC / Clang get both from -march=native, which ggml adds under GGML_NATIVE.)
set(NEXTERA_HAVE_AVX2 OFF)
if(MSVC AND NEXTERA_NATIVE AND CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
    include(CheckCXXSourceRuns)
    check_cxx_source_runs("
        #include <intrin.h>
        int main() { int r[4]; __cpuidex(r, 7, 0); return (r[1] & (1 << 5)) ? 0 : 1; }"
        NEXTERA_CPU_AVX2)
    check_cxx_source_runs("
        #include <intrin.h>
        int main() { int r[4]; __cpuidex(r, 7, 1); return (r[0] & (1 << 4)) ? 0 : 1; }"
        NEXTERA_CPU_AVX_VNNI)
    if(NEXTERA_CPU_AVX2)
        set(NEXTERA_HAVE_AVX2 ON)
    endif()
    if(NEXTERA_CPU_AVX_VNNI)
        # int8 dot products in one instruction (vpdpbusd) for the quantized kernels
        set(GGML_AVX_VNNI ON CACHE BOOL "" FORCE)
    endif()
endif()
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
if(NEXTERA_GGML_DIR)
    add_subdirectory(${NEXTERA_GGML_DIR} ggml EXCLUDE_FROM_ALL)
else()
    include(FetchContent)
    FetchContent_Declare(ggml
        GIT_REPOSITORY https://github.com/ggml-org/ggml.git
        GIT_TAG        ${NEXTERA_GGML_TAG}
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(ggml)
endif()

if(NEXTERA_BUILD_SHARED)
    add_library(nexterabert SHARED src/nexterabert.cpp)
    target_compile_definitions(nexterabert PRIVATE NEXTERA_BUILD PUBLIC NEXTERA_SHARED)
    set_target_properties(ggml ggml-base ggml-cpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
else()
    add_library(nexterabert STATIC src/nexterabert.cpp)
endif()
target_include_directories(nexterabert PUBLIC src)
if(MSVC)
    if(NEXTERA_HAVE_AVX2)
        target_compile_options(nexterabert PRIVATE /arch:AVX2)
    endif()
    if(NEXTERA_CPU_AVX_VNNI AND NOT NEXTERA_NO_VNNI)
        # MSVC never defines this itself
        target_compile_definitions(nexterabert PRIVATE __AVXVNNI__)
    endif()
    target_compile_options(nexterabert PRIVATE /fp:fast)
elseif(NEXTERA_NATIVE AND NOT CMAKE_CROSSCOMPILING)
    target_compile_options(nexterabert PRIVATE -march=native)
endif()
target_link_libraries(nexterabert PRIVATE ggml)

add_executable(nexterabert-cli tools/cli.cpp)
target_link_libraries(nexterabert-cli PRIVATE nexterabert)

add_executable(nexterabert-quantize tools/quantize.cpp)
target_link_libraries(nexterabert-quantize PRIVATE ggml)

# One output directory for every config, so the Python binding finds the DLL.
foreach(t nexterabert nexterabert-cli nexterabert-quantize)
    set_target_properties(${t} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    foreach(cfg ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${cfg} CFG)
        set_target_properties(${t} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY_${CFG} ${CMAKE_BINARY_DIR}/bin
            LIBRARY_OUTPUT_DIRECTORY_${CFG} ${CMAKE_BINARY_DIR}/bin)
    endforeach()
endforeach()
