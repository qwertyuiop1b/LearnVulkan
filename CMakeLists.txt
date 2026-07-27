cmake_minimum_required(VERSION 3.20)
project(vk-engine)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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


file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.h src/*.hpp)

add_executable(${PROJECT_NAME} ${SRC_FILES})
target_include_directories(
    ${PROJECT_NAME} PRIVATE 
    ${Vulkan_INCLUDE_DIRS}
    external/glfw/include
    external/glm
    external/vk-booststrap/
)
target_link_libraries(${PROJECT_NAME} PRIVATE glfw Vulkan::Vulkan vk-bootstrap::vk-bootstrap)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/shaders
    $<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders
)
