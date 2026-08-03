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


file(GLOB_RECURSE ENGINE_SOURCES CONFIGURE_DEPENDS src/*.cpp src/*.h src/*.hpp)

add_library(${PROJECT_NAME} STATIC ${ENGINE_SOURCES})
target_include_directories(
    ${PROJECT_NAME} PUBLIC
    ${Vulkan_INCLUDE_DIRS}
    external/glfw/include
    external/glm
    external/vk-booststrap/
)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/external/stb)
target_link_libraries(${PROJECT_NAME} PUBLIC glfw Vulkan::Vulkan vk-bootstrap::vk-bootstrap)

if(VK_ENGINE_BUILD_EXAMPLES)
    find_program(GLSLC_EXECUTABLE NAMES glslc REQUIRED)

    set(TEXTURE_SHADER_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/texture/shaders/texture.vert
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/texture/shaders/texture.frag
    )
    set(TEXTURE_SHADER_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/examples/texture/shaders)
    set(TEXTURE_SHADER_BINARIES)
    foreach(SHADER_SOURCE IN LISTS TEXTURE_SHADER_SOURCES)
        get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME)
        set(SHADER_BINARY ${TEXTURE_SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv)
        add_custom_command(
            OUTPUT ${SHADER_BINARY}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${TEXTURE_SHADER_OUTPUT_DIR}
            COMMAND ${GLSLC_EXECUTABLE} ${SHADER_SOURCE} -o ${SHADER_BINARY}
            DEPENDS ${SHADER_SOURCE}
            VERBATIM
        )
        list(APPEND TEXTURE_SHADER_BINARIES ${SHADER_BINARY})
    endforeach()
    add_custom_target(vk-engine-texture-shaders DEPENDS ${TEXTURE_SHADER_BINARIES})

    add_executable(vk-engine-texture examples/texture/main.cpp examples/texture/texture_vertex.h)
    add_dependencies(vk-engine-texture vk-engine-texture-shaders)
    target_compile_definitions(vk-engine-texture PRIVATE
        VK_ENGINE_TEXTURE_SHADER_DIR="${TEXTURE_SHADER_OUTPUT_DIR}"
        VK_ENGINE_TEXTURE_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/examples/texture/assets"
    )
    target_include_directories(vk-engine-texture PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples/texture)
    target_link_libraries(vk-engine-texture PRIVATE ${PROJECT_NAME})


    set(TRIANGLE_SHADER_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/triangle/shaders/simple.vert
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/triangle/shaders/simple.frag
    )
    set(TRIANGLE_SHADER_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/examples/triangle/shaders)
    set(TRIANGLE_SHADER_BINARIES)
    foreach(SHADER_SOURCE IN LISTS TRIANGLE_SHADER_SOURCES)
        get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME)
        set(SHADER_BINARY ${TRIANGLE_SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv)
        add_custom_command(
            OUTPUT ${SHADER_BINARY}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${TRIANGLE_SHADER_OUTPUT_DIR}
            COMMAND ${GLSLC_EXECUTABLE} ${SHADER_SOURCE} -o ${SHADER_BINARY}
            DEPENDS ${SHADER_SOURCE}
            VERBATIM
        )
        list(APPEND TRIANGLE_SHADER_BINARIES ${SHADER_BINARY})
    endforeach()
    add_custom_target(vk-engine-triangle-shaders DEPENDS ${TRIANGLE_SHADER_BINARIES})

    add_executable(vk-engine-triangle
        examples/triangle/main.cpp
        examples/triangle/triangle_vertex.h
    )
    add_dependencies(vk-engine-triangle vk-engine-triangle-shaders)
    target_compile_definitions(vk-engine-triangle PRIVATE
        VK_ENGINE_TRIANGLE_SHADER_DIR="${TRIANGLE_SHADER_OUTPUT_DIR}"
    )
    target_include_directories(vk-engine-triangle PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples/triangle)
    target_link_libraries(vk-engine-triangle PRIVATE ${PROJECT_NAME})
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
