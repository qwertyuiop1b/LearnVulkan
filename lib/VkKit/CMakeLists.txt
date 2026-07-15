cmake_minimum_required(VERSION 3.20)

project(VkKit VERSION 0.1.0 LANGUAGES CXX)

find_package(Vulkan REQUIRED)

find_path(VKKIT_VMA_INCLUDE_DIR
    NAMES vk_mem_alloc.h
    HINTS
        "${CMAKE_SOURCE_DIR}/external/VulkanMemoryAllocator/include"
        ${Vulkan_INCLUDE_DIRS}
    PATH_SUFFIXES vma)
if(NOT VKKIT_VMA_INCLUDE_DIR)
    message(FATAL_ERROR "VkKit requires vk_mem_alloc.h from Vulkan Memory Allocator (VMA)")
endif()

add_library(VkKitVma INTERFACE)
target_include_directories(VkKitVma SYSTEM INTERFACE ${VKKIT_VMA_INCLUDE_DIR})

if(NOT TARGET glfw AND NOT TARGET glfw::glfw AND NOT TARGET GLFW::glfw)
    find_package(glfw3 3.3 CONFIG REQUIRED)
endif()

if(TARGET glfw)
    set(VKKIT_GLFW_TARGET glfw)
elseif(TARGET glfw::glfw)
    set(VKKIT_GLFW_TARGET glfw::glfw)
elseif(TARGET GLFW::glfw)
    set(VKKIT_GLFW_TARGET GLFW::glfw)
else()
    message(FATAL_ERROR "VkKit requires a GLFW CMake target (glfw, glfw::glfw, or GLFW::glfw)")
endif()

add_library(VkKit
    src/graphics/vulkan_context.cpp
    src/graphics/vulkan_allocator.cpp
    src/graphics/buffer.cpp
    src/graphics/command_pool.cpp
    src/graphics/descriptor_pool.cpp
    src/graphics/descriptor_set.cpp
    src/graphics/descriptor_set_layout.cpp
    src/graphics/frame_context.cpp
    src/graphics/frame_scheduler.cpp
    src/graphics/graphics_pipeline.cpp
    src/graphics/image.cpp
    src/graphics/pipeline_layout.cpp
    src/graphics/sampler.cpp
    src/graphics/shader_module.cpp
    src/graphics/swapchain.cpp
    src/graphics/texture.cpp
    src/graphics/upload_context.cpp
    src/vma_impl.cpp)
add_library(VkKit::VkKit ALIAS VkKit)

target_include_directories(VkKit
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_compile_features(VkKit PUBLIC cxx_std_17)
target_compile_definitions(VkKit PRIVATE GLFW_INCLUDE_NONE)
target_link_libraries(VkKit PUBLIC
    Vulkan::Vulkan
    $<BUILD_INTERFACE:VkKitVma>
    $<BUILD_INTERFACE:${VKKIT_GLFW_TARGET}>
    $<INSTALL_INTERFACE:VkKit::Vma>
    $<INSTALL_INTERFACE:VkKit::Glfw>)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(VKKIT_INSTALL_DEFAULT ON)
else()
    set(VKKIT_INSTALL_DEFAULT OFF)
endif()
option(VKKIT_INSTALL "Generate installation and CMake package targets" ${VKKIT_INSTALL_DEFAULT})

if(VKKIT_INSTALL)
    include(CMakePackageConfigHelpers)
    include(GNUInstallDirs)

    install(TARGETS VkKit
        EXPORT VkKitTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(EXPORT VkKitTargets
        FILE VkKitTargets.cmake
        NAMESPACE VkKit::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VkKit)

    configure_package_config_file(
        cmake/VkKitConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/VkKitConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VkKit)
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/VkKitConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion)
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/VkKitConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/VkKitConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VkKit)
endif()
