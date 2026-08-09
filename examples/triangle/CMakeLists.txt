set(SHADER_SOURCES
    shaders/simple.vert
    shaders/simple.frag
)
set(SHADER_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
set(SHADER_BINARIES)
foreach(SHADER_SOURCE IN LISTS SHADER_SOURCES)
    get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME)
    set(SHADER_BINARY ${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv)
    add_custom_command(
        OUTPUT ${SHADER_BINARY}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUTPUT_DIR}
        COMMAND ${GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SOURCE} -o ${SHADER_BINARY}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SOURCE}
        VERBATIM
    )
    list(APPEND SHADER_BINARIES ${SHADER_BINARY})
endforeach()

add_custom_target(vk-engine-triangle-shaders DEPENDS ${SHADER_BINARIES})

add_executable(vk-engine-triangle main.cpp)
add_dependencies(vk-engine-triangle vk-engine-triangle-shaders)
set_target_properties(vk-engine-triangle PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_compile_definitions(vk-engine-triangle PRIVATE VK_ENGINE_SHADER_DIR="${SHADER_OUTPUT_DIR}")
target_include_directories(vk-engine-triangle PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vk-engine-triangle PRIVATE vk-engine)
