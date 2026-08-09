set(SHADER_SOURCES
    shaders/texture.vert
    shaders/texture.frag
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

add_custom_target(vk-engine-texture-shaders DEPENDS ${SHADER_BINARIES})

add_executable(vk-engine-texture main.cpp)
add_dependencies(vk-engine-texture vk-engine-texture-shaders)
set_target_properties(vk-engine-texture PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_compile_definitions(vk-engine-texture PRIVATE
    VK_ENGINE_SHADER_DIR="${SHADER_OUTPUT_DIR}"
    VK_ENGINE_TEXTURE_ASSET_DIR="${CMAKE_SOURCE_DIR}/assets"
)
target_include_directories(vk-engine-texture PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vk-engine-texture PRIVATE vk-engine)
