# The Python extension exrelaxer._core (nanobind). Built two ways:
#   - by pip from this directory (pyproject.toml, scikit-build-core): this is
#     then the top-level project and pulls in the library from the parent
#     directory, without its tests;
#   - from the main build with -DEXRELAXER_BUILD_PYTHON=ON: the package is
#     staged in the build tree and its pytest suite runs as the CTest test
#     python_tests.
cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(exrelaxer_python LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(EXRELAXER_PYTHON_STANDALONE ON)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. exrelaxer)
else()
    set(EXRELAXER_PYTHON_STANDALONE OFF)
endif()

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
if(NOT nanobind_ROOT)
    execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
endif()
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core NB_STATIC src/bindings.cpp)
target_link_libraries(_core PRIVATE exrelaxer_core)
if(EXRELAXER_NATIVE)
    set(exr_native 1)
else()
    set(exr_native 0)
endif()
target_compile_definitions(_core PRIVATE
    EXRELAXER_BUILD_TYPE="$<IF:$<CONFIG:>,unspecified,$<CONFIG>>"
    EXRELAXER_NATIVE_BUILD=${exr_native})

if(EXRELAXER_PYTHON_STANDALONE)
    install(TARGETS _core LIBRARY DESTINATION exrelaxer)
else()
    # A package in the build tree: the Python files copied next to _core.
    set(stage ${CMAKE_CURRENT_BINARY_DIR}/package/exrelaxer)
    set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${stage})
    file(GLOB py_files CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/exrelaxer/*.py)
    add_custom_target(exrelaxer_py_stage ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory ${stage}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${py_files} ${stage}
        COMMENT "Staging the exrelaxer Python package")
    add_dependencies(_core exrelaxer_py_stage)
    add_test(NAME python_tests
             COMMAND ${Python_EXECUTABLE} -m pytest -q ${CMAKE_CURRENT_SOURCE_DIR}/tests
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(python_tests PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/package")
    # The quick Python experiments, like nntest_quick for the C++ ones.
    add_test(NAME nntest_py_quick
             COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../NNtesting/nntest.py run tag:quick)
    set_tests_properties(nntest_py_quick PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/package")
endif()
