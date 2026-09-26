# nntest: the benchmark harness for exRelaxer networks (see README.md here).
# Every .cpp file in experiments/ is compiled in: adding an experiment is
# adding a file (re-run cmake so the glob picks it up).
file(GLOB NNT_EXPERIMENTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/experiments/*.cpp)

add_executable(nntest
    harness/main.cpp
    harness/experiment.cpp
    harness/experiment.hpp
    harness/results.cpp
    harness/results.hpp
    ${NNT_EXPERIMENTS}
)
target_include_directories(nntest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/harness
    ${CMAKE_CURRENT_SOURCE_DIR}/tasks
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)
target_link_libraries(nntest PRIVATE exrelaxer_core)
target_compile_options(nntest PRIVATE ${EXRELAXER_WARNINGS})

# Git commit and build settings, regenerated on every build so each result
# records exactly which code produced it.
add_custom_target(nnt_build_info
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
        -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/generated/nnt_build_info.hpp
        -DBUILD_TYPE=$<CONFIG>
        -DNATIVE=${EXRELAXER_NATIVE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_info.cmake
    BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/generated/nnt_build_info.hpp
    COMMENT "nntest build info"
)
add_dependencies(nntest nnt_build_info)

# The quick experiments (tag "quick") as one CTest test: their checks guard
# learning quality alongside the unit tests. Full runs are done by hand.
add_test(NAME nntest_quick COMMAND nntest run tag:quick)

# spiral.py (tools/) must sample exactly like the C++ Spiral retina: this
# runs both on the same images and compares every value bit for bit.
add_executable(retina_dump tools/retina_dump.cpp)
target_link_libraries(retina_dump PRIVATE exrelaxer_core)
target_compile_options(retina_dump PRIVATE ${EXRELAXER_WARNINGS})
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy"
                    RESULT_VARIABLE NNT_NUMPY_MISSING OUTPUT_QUIET ERROR_QUIET)
    if(NNT_NUMPY_MISSING EQUAL 0)
        add_test(NAME spiral_matches_retina
                 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_spiral.py $<TARGET_FILE:retina_dump>)
    endif()
endif()
