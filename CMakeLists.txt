cmake_minimum_required(VERSION 3.20)

project(exrelaxer 
    VERSION 1.0.0
    DESCRIPTION "Biologically inspired neural network core"
    LANGUAGES CXX
)

# Default to an optimized build: without a build type there is no
# optimization, and the tests run ~8x slower. Pass -DCMAKE_BUILD_TYPE=Debug
# for debugging.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -----------------------------------------------------------------------------
# C++20 standard requirements
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler options (warning flags)
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# -----------------------------------------------------------------------------
# Core library
# -----------------------------------------------------------------------------
add_library(exrelaxer_core STATIC
    core/random.cpp
    core/random.hpp
    core/kernels.cpp
    core/kernels.hpp
    core/parallel.hpp
    core/filters.cpp
    core/filters.hpp
    core/binary_io.hpp
    core/neuron.cpp
    core/neuron.hpp
    core/layers/layer.cpp
    core/layers/layer.hpp
    core/layers/neuron_layer.cpp
    core/layers/neuron_layer.hpp
    core/layers/dense.cpp
    core/layers/dense.hpp
    core/layers/spatial.cpp
    core/layers/spatial.hpp
    core/layers/spatial_neuron_layer.cpp
    core/layers/spatial_neuron_layer.hpp
    core/layers/conv2d.cpp
    core/layers/conv2d.hpp
    core/layers/locally_connected2d.cpp
    core/layers/locally_connected2d.hpp
    core/layers/pool2d.cpp
    core/layers/pool2d.hpp
    core/layers/retina.cpp
    core/layers/retina.hpp
    core/layers/layer_factory.cpp
    core/layers/layer_factory.hpp
    core/network.cpp
    core/network.hpp
)

# Expose the core directory as an include directory, so both test files and
# code inside layers can write either #include "neuron.hpp"
# or #include "layers/dense.hpp"
target_include_directories(exrelaxer_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/core
)

# Results must not depend on the compiler's choice to fuse a multiply and an
# add into one FMA instruction (different rounding), so contraction is off.
if(NOT MSVC)
    target_compile_options(exrelaxer_core PRIVATE -ffp-contract=off)
endif()

# Optional: build for this machine's CPU (e.g. AVX2). Results are the same;
# the binary may not run on older CPUs.
option(EXRELAXER_NATIVE "Optimize for the CPU of the build machine" OFF)
if(EXRELAXER_NATIVE AND NOT MSVC)
    target_compile_options(exrelaxer_core PRIVATE -march=native)
endif()

# Optional: large groups of neurons are stepped (and learn) in parallel with
# OpenMP. Without it the same code runs serially.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(exrelaxer_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Position-independent, so the core also links into shared libraries such as
# the Python extension (EXrelaxer.py/).
set_target_properties(exrelaxer_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Tests and the benchmark harness are built by default only when this is the
# top-level project, so a project that adds exrelaxer with add_subdirectory
# gets just the library (and no GoogleTest download).
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(EXRELAXER_TOP_LEVEL ON)
else()
    set(EXRELAXER_TOP_LEVEL OFF)
endif()
option(EXRELAXER_BUILD_TESTS "Build the GoogleTest unit tests" ${EXRELAXER_TOP_LEVEL})
option(EXRELAXER_BUILD_NNTESTING "Build the nntest benchmark harness" ${EXRELAXER_TOP_LEVEL})
option(EXRELAXER_BUILD_PYTHON "Build the Python extension (EXrelaxer.py/, needs nanobind)" OFF)

if(EXRELAXER_BUILD_TESTS OR EXRELAXER_BUILD_NNTESTING)
    enable_testing()
endif()

if(EXRELAXER_BUILD_TESTS)
# -----------------------------------------------------------------------------
# GoogleTest setup via FetchContent
# -----------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# Prevents GTest from overriding the global MSVC CRT settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# -----------------------------------------------------------------------------
# Test executable
# -----------------------------------------------------------------------------

add_executable(exrelaxer_tests
    tests/neuron.cpp
    tests/dense.cpp
    tests/network.cpp
    tests/regressions.cpp
    tests/kernels.cpp
    tests/spatial.cpp
    tests/filters.cpp
)

target_link_libraries(exrelaxer_tests PRIVATE
    exrelaxer_core
    GTest::gtest_main
)

# Task code shared with the benchmark harness (NNtesting/tasks).
target_include_directories(exrelaxer_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/NNtesting/tasks)

# Automatic test discovery for CTest
include(GoogleTest)
gtest_discover_tests(exrelaxer_tests)
endif()

# -----------------------------------------------------------------------------
# Benchmark harness (nntest): experiments on networks, one file each
# -----------------------------------------------------------------------------
if(EXRELAXER_BUILD_NNTESTING)
    add_subdirectory(NNtesting)
endif()

# -----------------------------------------------------------------------------
# Python extension (normally built by pip from EXrelaxer.py/; this option
# builds it in this tree too, e.g. to run its tests next to the unit tests)
# -----------------------------------------------------------------------------
if(EXRELAXER_BUILD_PYTHON)
    add_subdirectory(EXrelaxer.py)
endif()
