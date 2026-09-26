# CTest test installed_package_example: installs the build in BUILD_DIR into
# WORK_DIR/prefix, then configures, builds and runs the program in
# SOURCE_DIR (examples/consumer) against it with find_package(exrelaxer).
foreach(var BUILD_DIR SOURCE_DIR WORK_DIR)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is not set")
    endif()
endforeach()
if(NOT CONFIG)
    set(CONFIG Release)
endif()
file(REMOVE_RECURSE ${WORK_DIR})

function(run)
    execute_process(COMMAND ${ARGN} RESULT_VARIABLE status)
    if(NOT status EQUAL 0)
        list(JOIN ARGN " " command)
        message(FATAL_ERROR "failed (${status}): ${command}")
    endif()
endfunction()

run(${CMAKE_COMMAND} --install ${BUILD_DIR} --prefix ${WORK_DIR}/prefix --config ${CONFIG})
run(${CMAKE_COMMAND} -S ${SOURCE_DIR} -B ${WORK_DIR}/build -G ${GENERATOR}
    -DCMAKE_BUILD_TYPE=${CONFIG} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_PREFIX_PATH=${WORK_DIR}/prefix)
run(${CMAKE_COMMAND} --build ${WORK_DIR}/build --config ${CONFIG})

find_program(example exrelaxer_example PATHS ${WORK_DIR}/build ${WORK_DIR}/build/${CONFIG} NO_DEFAULT_PATH)
if(NOT example)
    message(FATAL_ERROR "the example program was not built")
endif()
run(${example})
