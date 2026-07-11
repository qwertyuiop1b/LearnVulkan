cmake_minimum_required(VERSION 3.20)

# Prefer clang++ for this project (designated init order is warning in clang,
# but hard error in GCC for some chapter sources).
if(NOT DEFINED CMAKE_CXX_COMPILER)
    find_program(CLANGXX_EXECUTABLE clang++)
    if(CLANGXX_EXECUTABLE)
        set(CMAKE_CXX_COMPILER "${CLANGXX_EXECUTABLE}" CACHE FILEPATH "C++ compiler" FORCE)
    endif()
endif()

project(VulkanTutorial VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Clang only: suppress designated-initializer order warnings.
add_compile_options($<$<COMPILE_LANG_AND_ID:CXX,Clang>:-Wno-reorder-init-list>)

# ─── Vulkan SDK ───────────────────────────────────────────────────────────────
find_package(Vulkan REQUIRED)

# ─── Dependencies (git submodules) ────────────────────────────────────────────
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/external/glfw/CMakeLists.txt"
   OR NOT EXISTS "${CMAKE_SOURCE_DIR}/external/glm/CMakeLists.txt"
   OR NOT EXISTS "${CMAKE_SOURCE_DIR}/external/imgui/imgui.cpp")
    message(FATAL_ERROR "Git submodules not initialized. Run: git submodule update --init --recursive")
endif()

# ─── GLFW ─────────────────────────────────────────────────────────────────────
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(external/glfw EXCLUDE_FROM_ALL)

add_subdirectory(external/fmt EXCLUDE_FROM_ALL)

# ─── GLM ──────────────────────────────────────────────────────────────────────
add_subdirectory(external/glm EXCLUDE_FROM_ALL)

# ─── Shader compiler (prefer glslc, fallback to glslangValidator) ────────────
find_program(GLSLC_EXECUTABLE glslc)
if(GLSLC_EXECUTABLE)
    set(SHADER_COMPILER_EXECUTABLE "${GLSLC_EXECUTABLE}")
    set(SHADER_COMPILER_ARGS --target-env=vulkan1.3)
    set(SHADER_COMPILER_NAME "glslc")
elseif(DEFINED Vulkan_GLSLANG_VALIDATOR_EXECUTABLE
       AND EXISTS "${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}")
    set(SHADER_COMPILER_EXECUTABLE "${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}")
    set(SHADER_COMPILER_ARGS -V --target-env vulkan1.3)
    set(SHADER_COMPILER_NAME "glslangValidator")
else()
    find_program(GLSLANG_VALIDATOR_EXECUTABLE glslangValidator)
    if(GLSLANG_VALIDATOR_EXECUTABLE)
        set(SHADER_COMPILER_EXECUTABLE "${GLSLANG_VALIDATOR_EXECUTABLE}")
        set(SHADER_COMPILER_ARGS -V --target-env vulkan1.3)
        set(SHADER_COMPILER_NAME "glslangValidator")
    else()
        message(FATAL_ERROR
            "Could not find shader compiler: install glslc or glslangValidator")
    endif()
endif()
message(STATUS "Shader compiler: ${SHADER_COMPILER_NAME} (${SHADER_COMPILER_EXECUTABLE})")

set(SHADER_COMPILER_SUPPORTS_MESH_SHADERS ON)
if(EXISTS "${CMAKE_SOURCE_DIR}/shaders/mesh_basic.mesh")
    execute_process(
        COMMAND ${SHADER_COMPILER_EXECUTABLE} ${SHADER_COMPILER_ARGS}
                ${CMAKE_SOURCE_DIR}/shaders/mesh_basic.mesh
                -o ${CMAKE_BINARY_DIR}/CMakeFiles/mesh_shader_probe.spv
        RESULT_VARIABLE MESH_SHADER_PROBE_RESULT
        OUTPUT_QUIET
        ERROR_QUIET)
    if(NOT MESH_SHADER_PROBE_RESULT EQUAL 0)
        set(SHADER_COMPILER_SUPPORTS_MESH_SHADERS OFF)
        message(WARNING
            "Shader compiler '${SHADER_COMPILER_NAME}' does not support mesh/task shaders. "
            "Skipping .mesh/.task shaders and chapter 21.")
    endif()
endif()

# ─── Copy runtime assets once ────────────────────────────────────────────────
file(GLOB_RECURSE ASSET_FILES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/assets/*")
set(ASSET_STAMP "${CMAKE_BINARY_DIR}/CMakeFiles/assets.stamp")
add_custom_command(
    OUTPUT "${ASSET_STAMP}"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/assets"
            "${CMAKE_BINARY_DIR}/assets"
    COMMAND ${CMAKE_COMMAND} -E touch "${ASSET_STAMP}"
    DEPENDS ${ASSET_FILES}
    COMMENT "Copying runtime assets")
add_custom_target(vulkan_tutorial_copy_assets ALL DEPENDS "${ASSET_STAMP}")

# ─── Compile shaders helper ───────────────────────────────────────────────────
function(compile_shaders TARGET)
    set(SHADER_DIR "${CMAKE_SOURCE_DIR}/shaders")
    set(SPIRV_DIR  "${CMAKE_BINARY_DIR}/shaders")
    file(MAKE_DIRECTORY ${SPIRV_DIR})

    file(GLOB SHADER_SOURCES
        "${SHADER_DIR}/*.vert"
        "${SHADER_DIR}/*.frag"
        "${SHADER_DIR}/*.comp"
        "${SHADER_DIR}/*.tesc"
        "${SHADER_DIR}/*.tese"
        "${SHADER_DIR}/*.geom"
        "${SHADER_DIR}/*.rgen"
        "${SHADER_DIR}/*.rmiss"
        "${SHADER_DIR}/*.rchit")

    if(SHADER_COMPILER_SUPPORTS_MESH_SHADERS)
        file(GLOB MESH_SHADER_SOURCES
            "${SHADER_DIR}/*.mesh"
            "${SHADER_DIR}/*.task")
        list(APPEND SHADER_SOURCES ${MESH_SHADER_SOURCES})
    endif()

    set(SPIRV_FILES)
    foreach(SHADER ${SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER} NAME)
        set(SPIRV "${SPIRV_DIR}/${SHADER_NAME}.spv")
        add_custom_command(
            OUTPUT  ${SPIRV}
            COMMAND ${SHADER_COMPILER_EXECUTABLE} ${SHADER_COMPILER_ARGS} ${SHADER} -o ${SPIRV}
            DEPENDS ${SHADER}
            COMMENT "Compiling shader: ${SHADER_NAME}")
        list(APPEND SPIRV_FILES ${SPIRV})
    endforeach()

    add_custom_target(${TARGET}_shaders ALL DEPENDS ${SPIRV_FILES})
    add_dependencies(${TARGET} ${TARGET}_shaders)
endfunction()

# ─── Common include interface library ─────────────────────────────────────────
add_library(vulkan_tutorial_common INTERFACE)
target_include_directories(vulkan_tutorial_common INTERFACE
    ${CMAKE_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS})
target_link_libraries(vulkan_tutorial_common INTERFACE
    Vulkan::Vulkan
    glfw
    glm::glm
    fmt::fmt)
target_compile_definitions(vulkan_tutorial_common INTERFACE
    GLFW_INCLUDE_NONE          # Do not let GLFW include GL headers
    GLM_FORCE_RADIANS
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    GLM_ENABLE_EXPERIMENTAL    # 允许 GLM 实验性扩展（gtx/euler_angles 等）
    SHADER_DIR="${CMAKE_BINARY_DIR}/shaders"
    ASSET_DIR="${CMAKE_SOURCE_DIR}/assets")

# ─── Asset loading library (stb_image + cgltf) ───────────────────────────────
add_library(vulkan_tutorial_assets STATIC
    src/common/stb_image_impl.cpp
    src/common/cgltf_impl.cpp
    src/common/gltf_loader.cpp)
target_include_directories(vulkan_tutorial_assets PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external/stb
    ${CMAKE_SOURCE_DIR}/external/cgltf)
target_link_libraries(vulkan_tutorial_assets PUBLIC vulkan_tutorial_common glm::glm)

add_library(vulkan_tutorial_interactive STATIC
    external/imgui/imgui.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_tables.cpp
    external/imgui/imgui_widgets.cpp
    external/imgui/backends/imgui_impl_glfw.cpp
    external/imgui/backends/imgui_impl_vulkan.cpp
    src/common/interactive_chapter.cpp)
target_include_directories(vulkan_tutorial_interactive PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external/imgui
    ${CMAKE_SOURCE_DIR}/external/imgui/backends)
target_link_libraries(vulkan_tutorial_interactive PUBLIC
    vulkan_tutorial_common
    vulkan_tutorial_assets
    glm::glm)

# ─── Render Graph library (ch51+) ────────────────────────────────────────────
add_library(vulkan_tutorial_render_graph STATIC
    src/common/render_graph.cpp)
target_include_directories(vulkan_tutorial_render_graph PUBLIC
    ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(vulkan_tutorial_render_graph PUBLIC
    vulkan_tutorial_common)

# ─── Engine library (ch61+) ──────────────────────────────────────────────────
# 收集 src/engine/ 下所有 .cpp（按需添加）
set(ENGINE_SOURCES
    src/engine/rhi_device.cpp
    src/engine/rhi_buffer.cpp
    src/engine/rhi_texture.cpp
    src/engine/rhi_shader.cpp
    src/engine/pipeline_builder.cpp
    src/engine/descriptor_manager.cpp
    src/engine/command_recorder.cpp
    src/engine/scene_ecs.cpp
    src/engine/material_system.cpp)

# mini_engine.cpp 由子 Agent 生成，条件包含
if(EXISTS ${CMAKE_SOURCE_DIR}/src/engine/mini_engine.cpp)
    list(APPEND ENGINE_SOURCES src/engine/mini_engine.cpp)
endif()

add_library(vulkan_tutorial_engine STATIC ${ENGINE_SOURCES})
target_include_directories(vulkan_tutorial_engine PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external/stb
    ${CMAKE_SOURCE_DIR}/external/cgltf)
target_link_libraries(vulkan_tutorial_engine PUBLIC
    vulkan_tutorial_common
    vulkan_tutorial_assets
    vulkan_tutorial_render_graph
    glm::glm)

# ─── Chapters ─────────────────────────────────────────────────────────────────
add_subdirectory(src)
