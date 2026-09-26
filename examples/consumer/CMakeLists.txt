# A separate program using an installed exrelaxer:
#
#   ./build.sh --prefix ~/opt/exrelaxer          # or: cmake --install build --prefix ~/opt/exrelaxer
#   cmake -S examples/consumer -B consumer-build -DCMAKE_PREFIX_PATH=~/opt/exrelaxer
#   cmake --build consumer-build && ./consumer-build/exrelaxer_example
cmake_minimum_required(VERSION 3.20)
project(exrelaxer_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(exrelaxer 1.0 REQUIRED)

add_executable(exrelaxer_example main.cpp)
target_link_libraries(exrelaxer_example PRIVATE exrelaxer::core)
