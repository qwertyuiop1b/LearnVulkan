cmake_minimum_required(VERSION 3.20)
project(LearnVulkan VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Vulkan SDK
find_package(Vulkan REQUIRED)
message(STATUS "Vulkan_INCLUDE_DIRS: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "Vulkan_LIBRARIES: ${Vulkan_LIBRARIES}")
message(STATUS "Vulkan_FOUND: ${Vulkan_FOUND}")
message(STATUS "Vulkan_VERSION_STRING: ${Vulkan_VERSION_STRING}")
message(STATUS "Vulkan_TOOLS_DIR: ${Vulkan_TOOLS_DIR}")

add_subdirectory(external/glfw)

FILE(GLOB_RECURSE SOURCE_FILES src/*.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC 
    include
    ${Vulkan_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${Vulkan_LIBRARIES}
    glfw
)

# Copy shaders directory to build directory
# add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_directory
#     ${CMAKE_SOURCE_DIR}/shaders
#     $<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders
#     COMMENT "Copying shaders to build directory"
# )

