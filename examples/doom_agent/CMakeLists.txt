# Loads an evolved Doom agent (best.exr) with the installed library and runs
# one game step on a blank screen and silence (see doc/doom_guide.md):
#
#   ./build.sh                                   # installs into ./install
#   cmake -S examples/doom_agent -B doom-agent-build -DCMAKE_PREFIX_PATH=$PWD/install
#   cmake --build doom-agent-build && ./doom-agent-build/doom_agent results/dynamic/doom_agent/untildeath_d1/best.exr
cmake_minimum_required(VERSION 3.20)
project(doom_agent LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(exrelaxer 1.0 REQUIRED)

add_executable(doom_agent main.cpp)
target_link_libraries(doom_agent PRIVATE exrelaxer::core)
