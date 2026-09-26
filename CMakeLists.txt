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

include(GNUInstallDirs)

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
    core/layers/audio.hpp
    core/layers/cochlea.cpp
    core/layers/cochlea.hpp
    core/layers/history.cpp
    core/layers/history.hpp
    core/layers/layer_factory.cpp
    core/layers/layer_factory.hpp
    core/network.cpp
    core/network.hpp
)

# Expose the core directory as an include directory, so both test files and
# code inside layers can write either #include "neuron.hpp"
# or #include "layers/dense.hpp". Installed, the same headers are in
# include/exrelaxer, so a program includes them the same way.
target_include_directories(exrelaxer_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/core>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/exrelaxer>
)
# The name other projects use, with add_subdirectory or find_package alike.
add_library(exrelaxer::core ALIAS exrelaxer_core)
set_target_properties(exrelaxer_core PROPERTIES EXPORT_NAME core)

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
option(EXRELAXER_INSTALL "Install the library as a CMake package (find_package(exrelaxer))" ${EXRELAXER_TOP_LEVEL})

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
# GoogleTest is only for our tests: not part of an install.
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

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
    tests/audio.cpp
    tests/snake.cpp
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

# -----------------------------------------------------------------------------
# Install: the library, its headers and a CMake package, so another program
# can use it with
#   find_package(exrelaxer REQUIRED)
#   target_link_libraries(my_app PRIVATE exrelaxer::core)
# (cmake --install build --prefix DIR; then -DCMAKE_PREFIX_PATH=DIR there).
# nntest is installed too, when built.
# -----------------------------------------------------------------------------
if(EXRELAXER_INSTALL)
    include(CMakePackageConfigHelpers)
    set(EXRELAXER_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/exrelaxer)

    install(TARGETS exrelaxer_core EXPORT exrelaxerTargets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/exrelaxer)
    install(DIRECTORY core/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/exrelaxer
            FILES_MATCHING PATTERN "*.hpp")
    install(EXPORT exrelaxerTargets NAMESPACE exrelaxer:: DESTINATION ${EXRELAXER_CMAKE_DIR})

    set(EXRELAXER_WITH_OPENMP ${OpenMP_CXX_FOUND})
    configure_package_config_file(cmake/exrelaxerConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/exrelaxerConfig.cmake
        INSTALL_DESTINATION ${EXRELAXER_CMAKE_DIR})
    write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/exrelaxerConfigVersion.cmake
        COMPATIBILITY SameMajorVersion)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/exrelaxerConfig.cmake
                  ${CMAKE_CURRENT_BINARY_DIR}/exrelaxerConfigVersion.cmake
            DESTINATION ${EXRELAXER_CMAKE_DIR})

    if(EXRELAXER_BUILD_NNTESTING)
        install(TARGETS nntest RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()

    # A separate program (examples/consumer) built against an install of
    # this build: checks that the installed package works.
    if(EXRELAXER_BUILD_TESTS)
        add_test(NAME installed_package_example
                 COMMAND ${CMAKE_COMMAND}
                     -DBUILD_DIR=${CMAKE_BINARY_DIR}
                     -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/examples/consumer
                     -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/installed_package_example
                     -DCONFIG=$<CONFIG>
                     -DGENERATOR=${CMAKE_GENERATOR}
                     -DCXX_COMPILER=${CMAKE_CXX_COMPILER}
                     -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_installed_package.cmake)
    endif()
endif()
