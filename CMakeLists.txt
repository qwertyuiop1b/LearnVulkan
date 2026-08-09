cmake_minimum_required(VERSION 3.20)
project(vk-engine)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(VK_ENGINE_BUILD_EXAMPLES "Build vk-engine examples" ON)

# vulkan
find_package(Vulkan REQUIRED)

# vk-boostrap
add_subdirectory(external/vk-booststrap)

# glfw 
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "")
set(GLFW_BUILD_TESTS OFF CACHE BOOL "")
set(GLFW_BUILD_DOCS OFF CACHE BOOL "")
add_subdirectory(external/glfw)

# glm
add_subdirectory(external/glm)

# vma
add_subdirectory(external/VulkanMemoryAllocator)

# imgui (core + glfw platform backend + vulkan renderer backend)
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui)
add_library(imgui-lib STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
)
target_include_directories(
    imgui-lib PUBLIC
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)
target_link_libraries(imgui-lib PUBLIC glfw Vulkan::Vulkan)
set_target_properties(imgui-lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

file(GLOB_RECURSE ENGINE_SOURCES CONFIGURE_DEPENDS src/*.cpp src/*.h src/*.hpp)

add_library(${PROJECT_NAME} STATIC ${ENGINE_SOURCES})
target_include_directories(
    ${PROJECT_NAME} PUBLIC
    ${Vulkan_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    external/glfw/include
    external/glm
    external/vk-booststrap/
    external/VulkanMemoryAllocator/include
    external/stb
)
target_link_libraries(${PROJECT_NAME} PUBLIC glfw Vulkan::Vulkan vk-bootstrap::vk-bootstrap imgui-lib)

if(VK_ENGINE_BUILD_EXAMPLES)
    find_program(GLSLC_EXECUTABLE NAMES glslc REQUIRED)
    add_subdirectory(examples/texture)
    add_subdirectory(examples/triangle)
    add_subdirectory(examples/compute_clearcolor)
    add_subdirectory(examples/imgui)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(vk-utils-tests tests/vk_utils_tests.cpp)
    target_include_directories(vk-utils-tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${Vulkan_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/external/glfw/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/glm
    )
    target_link_libraries(vk-utils-tests PRIVATE ${PROJECT_NAME})
    add_test(NAME vk-utils-tests COMMAND vk-utils-tests)

    if(VK_ENGINE_BUILD_EXAMPLES)
        add_executable(vk-texture-tests tests/texture_vertex_tests.cpp)
        target_include_directories(vk-texture-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples/texture)
        target_link_libraries(vk-texture-tests PRIVATE ${PROJECT_NAME})
        add_test(NAME vk-texture-tests COMMAND vk-texture-tests)

        add_executable(vk-triangle-tests tests/triangle_vertex_tests.cpp)
        target_include_directories(vk-triangle-tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/examples/triangle
        )
        target_link_libraries(vk-triangle-tests PRIVATE ${PROJECT_NAME})
        add_test(NAME vk-triangle-tests COMMAND vk-triangle-tests)
    endif()
endif()
