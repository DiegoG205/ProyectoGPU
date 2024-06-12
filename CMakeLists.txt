cmake_minimum_required(VERSION 3.20)
project(cuda-opengl C CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)

add_definitions(-D_USE_MATH_DEFINES)

add_subdirectory(glcore-master)
add_subdirectory(src)