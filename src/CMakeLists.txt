# CUDA with OpenGL Test

# function(add_example name)
#   file(GLOB SOURCES ${name}/*.cpp ${name}/*.h)
#   add_executable(${name} ${SOURCES})
#   target_link_libraries(${name} PRIVATE glcore)
#   target_compile_definitions(${name} PRIVATE SHADER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/${name}/shaders/")
# endfunction()

# add_example(collision)

add_executable(fluid_sim main.cu)

target_link_libraries(fluid_sim PRIVATE glcore)
target_include_directories(fluid_sim PRIVATE "${PROJECT_SOURCE_DIR}/glcore-master/include" ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           "${PROJECT_SOURCE_DIR}/src")
target_compile_definitions(fluid_sim PRIVATE SHADER_PATH="${CMAKE_CURRENT_SOURCE_DIR}/shaders/")

