cmake_minimum_required(VERSION 3.20)

project(glcore)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)

add_subdirectory(extern)
add_subdirectory(src)
