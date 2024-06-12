add_subdirectory(glfw)
add_subdirectory(glad)
add_subdirectory(glm)


# ImGui
file(GLOB IMGUI_SOURCES imgui/*.cpp imgui/*.h)
add_library(imgui STATIC
        ${IMGUI_SOURCES}
        imgui/backends/imgui_impl_glfw.cpp
        imgui/backends/imgui_impl_opengl3.cpp
)

target_include_directories(imgui PUBLIC imgui imgui/backends)

# STB
file(GLOB STB_SOURCES stb/*.c stb/*.h)
add_library(stb ${STB_SOURCES})
target_include_directories(stb PUBLIC stb)
