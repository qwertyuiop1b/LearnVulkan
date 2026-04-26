cmake_minimum_required(VERSION 3.20)
project(VulkanTutorial VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ─── Vulkan SDK ───────────────────────────────────────────────────────────────
set(VULKAN_SDK "$ENV{HOME}/VulkanSDK/1.4.328.1/macOS")
find_package(Vulkan QUIET HINTS "${VULKAN_SDK}")
if (NOT Vulkan_FOUND)
    # Fallback: set manually
    set(Vulkan_INCLUDE_DIRS "${VULKAN_SDK}/include")
    set(Vulkan_LIBRARIES    "${VULKAN_SDK}/lib/libvulkan.dylib")
    add_library(Vulkan::Vulkan UNKNOWN IMPORTED)
    set_target_properties(Vulkan::Vulkan PROPERTIES
        IMPORTED_LOCATION "${Vulkan_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${Vulkan_INCLUDE_DIRS}")
    message(STATUS "Vulkan: using manual path ${VULKAN_SDK}")
else()
    message(STATUS "Vulkan: found ${Vulkan_VERSION}")
endif()

# ─── GLFW ─────────────────────────────────────────────────────────────────────
find_package(glfw3 REQUIRED HINTS "/opt/homebrew/lib/cmake/glfw3")

# ─── GLM ──────────────────────────────────────────────────────────────────────
find_package(glm REQUIRED HINTS "/opt/homebrew/lib/cmake/glm")

# ─── glslc shader compiler ────────────────────────────────────────────────────
find_program(GLSLC_EXECUTABLE glslc
    HINTS "${VULKAN_SDK}/bin" "$ENV{HOME}/VulkanSDK/1.4.328.1/macOS/bin"
    REQUIRED)
message(STATUS "glslc: ${GLSLC_EXECUTABLE}")

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
        "${SHADER_DIR}/*.mesh"
        "${SHADER_DIR}/*.task"
        "${SHADER_DIR}/*.rgen"
        "${SHADER_DIR}/*.rmiss"
        "${SHADER_DIR}/*.rchit")

    set(SPIRV_FILES)
    foreach(SHADER ${SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER} NAME)
        set(SPIRV "${SPIRV_DIR}/${SHADER_NAME}.spv")
        add_custom_command(
            OUTPUT  ${SPIRV}
            COMMAND ${GLSLC_EXECUTABLE} --target-env=vulkan1.3 ${SHADER} -o ${SPIRV}
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
    glm::glm)
target_compile_definitions(vulkan_tutorial_common INTERFACE
    GLFW_INCLUDE_NONE          # Do not let GLFW include GL headers
    GLM_FORCE_RADIANS
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    SHADER_DIR="${CMAKE_BINARY_DIR}/shaders")

# ─── Chapters ─────────────────────────────────────────────────────────────────
add_subdirectory(src)
