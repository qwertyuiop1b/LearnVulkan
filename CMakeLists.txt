cmake_minimum_required(VERSION 3.20)
project(vk-engine)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# vulkan
find_package(Vulkan REQUIRED)
find_program(GLSLC_EXECUTABLE NAMES glslc REQUIRED)

# vk-boostrap
add_subdirectory(external/vk-booststrap)

# glfw 
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "")
set(GLFW_BUILD_TESTS OFF CACHE BOOL "")
set(GLFW_BUILD_DOCS OFF CACHE BOOL "")
add_subdirectory(external/glfw)

# glm
add_subdirectory(external/glm)


file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS src/*.cpp src/*.h src/*.hpp)

set(SHADER_SOURCES
    ${CMAKE_SOURCE_DIR}/shaders/simple.vert
    ${CMAKE_SOURCE_DIR}/shaders/simple.frag
)
set(SHADER_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
set(SHADER_BINARIES)
foreach(SHADER_SOURCE IN LISTS SHADER_SOURCES)
    get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME)
    set(SHADER_BINARY ${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv)
    add_custom_command(
        OUTPUT ${SHADER_BINARY}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUTPUT_DIR}
        COMMAND ${GLSLC_EXECUTABLE} ${SHADER_SOURCE} -o ${SHADER_BINARY}
        DEPENDS ${SHADER_SOURCE}
        VERBATIM
    )
    list(APPEND SHADER_BINARIES ${SHADER_BINARY})
endforeach()
add_custom_target(vk-engine-shaders DEPENDS ${SHADER_BINARIES})

add_executable(${PROJECT_NAME} ${SRC_FILES})
add_dependencies(${PROJECT_NAME} vk-engine-shaders)
target_compile_definitions(${PROJECT_NAME} PRIVATE VK_ENGINE_SHADER_DIR="${SHADER_OUTPUT_DIR}")
target_include_directories(
    ${PROJECT_NAME} PRIVATE 
    ${Vulkan_INCLUDE_DIRS}
    external/glfw/include
    external/glm
    external/vk-booststrap/
)
target_link_libraries(${PROJECT_NAME} PRIVATE glfw Vulkan::Vulkan vk-bootstrap::vk-bootstrap)

include(CTest)
if(BUILD_TESTING)
    add_executable(vk-utils-tests tests/vk_utils_tests.cpp)
    target_sources(vk-utils-tests PRIVATE src/vk_shader.cpp)
    target_include_directories(vk-utils-tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${Vulkan_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/external/glfw/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external/glm
    )
    target_link_libraries(vk-utils-tests PRIVATE Vulkan::Vulkan)
    add_test(NAME vk-utils-tests COMMAND vk-utils-tests)
endif()
