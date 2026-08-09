add_executable(vk-engine-imgui main.cpp)
set_target_properties(vk-engine-imgui PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_link_libraries(vk-engine-imgui PRIVATE vk-engine)
