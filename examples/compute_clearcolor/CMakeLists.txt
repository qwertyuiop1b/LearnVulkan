set(SHADER_SOURCE shaders/clear_color.comp)
set(SHADER_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
set(SHADER_BINARY ${SHADER_OUTPUT_DIR}/clear_color.comp.spv)
add_custom_command(
    OUTPUT ${SHADER_BINARY}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUTPUT_DIR}
    COMMAND ${GLSLC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SOURCE} -o ${SHADER_BINARY}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SOURCE}
    VERBATIM
)
add_custom_target(vk-engine-compute-clearcolor-shaders DEPENDS ${SHADER_BINARY})

add_executable(vk-engine-compute-clearcolor main.cpp)
add_dependencies(vk-engine-compute-clearcolor vk-engine-compute-clearcolor-shaders)
set_target_properties(vk-engine-compute-clearcolor PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_compile_definitions(vk-engine-compute-clearcolor PRIVATE VK_ENGINE_SHADER_DIR="${SHADER_OUTPUT_DIR}")
target_link_libraries(vk-engine-compute-clearcolor PRIVATE vk-engine)
