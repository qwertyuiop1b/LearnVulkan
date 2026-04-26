# ─── Chapter 01: Instance + Validation Layers ────────────────────────────────
add_executable(ch01_instance chapters/ch01_instance.cpp)
target_link_libraries(ch01_instance PRIVATE vulkan_tutorial_common)

# ─── Chapter 02: Physical Device Selection ────────────────────────────────────
add_executable(ch02_physical_device chapters/ch02_physical_device.cpp)
target_link_libraries(ch02_physical_device PRIVATE vulkan_tutorial_common)

# ─── Chapter 03: Logical Device & Queues ──────────────────────────────────────
add_executable(ch03_logical_device chapters/ch03_logical_device.cpp)
target_link_libraries(ch03_logical_device PRIVATE vulkan_tutorial_common)

# ─── Chapter 04: Window Surface & Swap Chain ──────────────────────────────────
add_executable(ch04_swapchain chapters/ch04_swapchain.cpp)
target_link_libraries(ch04_swapchain PRIVATE vulkan_tutorial_common)

# ─── Chapter 05: Image Views & Render Pass ────────────────────────────────────
add_executable(ch05_renderpass chapters/ch05_renderpass.cpp)
target_link_libraries(ch05_renderpass PRIVATE vulkan_tutorial_common)

# ─── Chapter 06: Graphics Pipeline ───────────────────────────────────────────
add_executable(ch06_pipeline chapters/ch06_pipeline.cpp)
target_link_libraries(ch06_pipeline PRIVATE vulkan_tutorial_common)

# ─── Chapter 07: Framebuffers & Command Buffers ───────────────────────────────
add_executable(ch07_commands chapters/ch07_commands.cpp)
target_link_libraries(ch07_commands PRIVATE vulkan_tutorial_common)

# ─── Chapter 08: Render Loop (DRAW TRIANGLE!) ────────────────────────────────
add_executable(ch08_triangle chapters/ch08_triangle.cpp)
target_link_libraries(ch08_triangle PRIVATE vulkan_tutorial_common)
compile_shaders(ch08_triangle)

# ─── Chapter 09: Vertex Buffers & Memory ──────────────────────────────────────
add_executable(ch09_vertex_buffer chapters/ch09_vertex_buffer.cpp)
target_link_libraries(ch09_vertex_buffer PRIVATE vulkan_tutorial_common)
compile_shaders(ch09_vertex_buffer)

# ─── Chapter 10: Uniform Buffers & Descriptors ────────────────────────────────
add_executable(ch10_uniforms chapters/ch10_uniforms.cpp)
target_link_libraries(ch10_uniforms PRIVATE vulkan_tutorial_common)
compile_shaders(ch10_uniforms)

# ─── Chapter 11: Texture Mapping ─────────────────────────────────────────────
add_executable(ch11_texture chapters/ch11_texture.cpp)
target_link_libraries(ch11_texture PRIVATE vulkan_tutorial_common)
compile_shaders(ch11_texture)

# ─── Chapter 12: Depth Buffering ─────────────────────────────────────────────
add_executable(ch12_depth chapters/ch12_depth.cpp)
target_link_libraries(ch12_depth PRIVATE vulkan_tutorial_common)
compile_shaders(ch12_depth)

# ─── Chapter 13: Generating Mipmaps ──────────────────────────────────────────
add_executable(ch13_mipmaps chapters/ch13_mipmaps.cpp)
target_link_libraries(ch13_mipmaps PRIVATE vulkan_tutorial_common)
compile_shaders(ch13_mipmaps)

# ─── Chapter 14: Multisampling (MSAA) ────────────────────────────────────────
add_executable(ch14_msaa chapters/ch14_msaa.cpp)
target_link_libraries(ch14_msaa PRIVATE vulkan_tutorial_common)
compile_shaders(ch14_msaa)

# ─── Chapter 15: Compute Shader (Particle System) ────────────────────────────
add_executable(ch15_compute chapters/ch15_compute.cpp)
target_link_libraries(ch15_compute PRIVATE vulkan_tutorial_common)
compile_shaders(ch15_compute)

# ─── Chapter 16: Dynamic Rendering (No RenderPass) ───────────────────────────
add_executable(ch16_dynamic_rendering chapters/ch16_dynamic_rendering.cpp)
target_link_libraries(ch16_dynamic_rendering PRIVATE vulkan_tutorial_common)
compile_shaders(ch16_dynamic_rendering)

# ─── Chapter 17: Instanced Rendering ─────────────────────────────────────────
add_executable(ch17_instancing chapters/ch17_instancing.cpp)
target_link_libraries(ch17_instancing PRIVATE vulkan_tutorial_common)
compile_shaders(ch17_instancing)

# ─── Chapter 18: Shadow Mapping ──────────────────────────────────────────────
add_executable(ch18_shadow chapters/ch18_shadow.cpp)
target_link_libraries(ch18_shadow PRIVATE vulkan_tutorial_common)
compile_shaders(ch18_shadow)

# ─── Chapter 19: Deferred Rendering (G-Buffer) ───────────────────────────────
add_executable(ch19_deferred chapters/ch19_deferred.cpp)
target_link_libraries(ch19_deferred PRIVATE vulkan_tutorial_common)
compile_shaders(ch19_deferred)

# ─── Chapter 20: Ray Tracing Pipeline ────────────────────────────────────────
add_executable(ch20_raytracing chapters/ch20_raytracing.cpp)
target_link_libraries(ch20_raytracing PRIVATE vulkan_tutorial_common)
compile_shaders(ch20_raytracing)

# ─── Chapter 21: Mesh Shaders ────────────────────────────────────────────────
add_executable(ch21_mesh_shader chapters/ch21_mesh_shader.cpp)
target_link_libraries(ch21_mesh_shader PRIVATE vulkan_tutorial_common)
compile_shaders(ch21_mesh_shader)

# ─── Chapter 22: Bindless Rendering ──────────────────────────────────────────
add_executable(ch22_bindless chapters/ch22_bindless.cpp)
target_link_libraries(ch22_bindless PRIVATE vulkan_tutorial_common)
compile_shaders(ch22_bindless)

# ─── Chapter 23: GPU-Driven Rendering ────────────────────────────────────────
add_executable(ch23_gpu_driven chapters/ch23_gpu_driven.cpp)
target_link_libraries(ch23_gpu_driven PRIVATE vulkan_tutorial_common)
compile_shaders(ch23_gpu_driven)

# ─── Chapter 24: Post-Processing (HDR + Bloom) ────────────────────────────────
add_executable(ch24_postprocess chapters/ch24_postprocess.cpp)
target_link_libraries(ch24_postprocess PRIVATE vulkan_tutorial_common)
compile_shaders(ch24_postprocess)

# ─── Chapter 25: Timeline Semaphores + Multi-threaded Commands ────────────────
add_executable(ch25_advanced_sync chapters/ch25_advanced_sync.cpp)
target_link_libraries(ch25_advanced_sync PRIVATE vulkan_tutorial_common)
compile_shaders(ch25_advanced_sync)

# ─── Chapter 26: Query Pool (GPU Profiling) ───────────────────────────────────
add_executable(ch26_query_pool chapters/ch26_query_pool.cpp)
target_link_libraries(ch26_query_pool PRIVATE vulkan_tutorial_common)
compile_shaders(ch26_query_pool)

# ─── Chapter 27: Tessellation Shaders ────────────────────────────────────────
add_executable(ch27_tessellation chapters/ch27_tessellation.cpp)
target_link_libraries(ch27_tessellation PRIVATE vulkan_tutorial_common)
compile_shaders(ch27_tessellation)

# ─── Chapter 28: Subgroup Operations (Wave Intrinsics) ───────────────────────
add_executable(ch28_subgroup chapters/ch28_subgroup.cpp)
target_link_libraries(ch28_subgroup PRIVATE vulkan_tutorial_common)
compile_shaders(ch28_subgroup)

# ─── Chapter 29: Variable Rate Shading (VRS) ─────────────────────────────────
add_executable(ch29_vrs chapters/ch29_vrs.cpp)
target_link_libraries(ch29_vrs PRIVATE vulkan_tutorial_common)
compile_shaders(ch29_vrs)

# ─── Chapter 30: Vulkan Memory Allocator (VMA) ───────────────────────────────
add_executable(ch30_vma chapters/ch30_vma.cpp)
target_link_libraries(ch30_vma PRIVATE vulkan_tutorial_common)
compile_shaders(ch30_vma)

# ─── Chapter 31: Shader Object (VK_EXT_shader_object) ────────────────────────
add_executable(ch31_shader_object chapters/ch31_shader_object.cpp)
target_link_libraries(ch31_shader_object PRIVATE vulkan_tutorial_common)
compile_shaders(ch31_shader_object)

# ─── Chapter 32: Sparse Resources (Virtual Texturing) ────────────────────────
add_executable(ch32_sparse chapters/ch32_sparse.cpp)
target_link_libraries(ch32_sparse PRIVATE vulkan_tutorial_common)
compile_shaders(ch32_sparse)

# ─── Chapter 33: Push Descriptors ────────────────────────────────────────────
add_executable(ch33_push_descriptors chapters/ch33_push_descriptors.cpp)
target_link_libraries(ch33_push_descriptors PRIVATE vulkan_tutorial_common)
compile_shaders(ch33_push_descriptors)

# ─── Chapter 34: Geometry Shaders + Transform Feedback ───────────────────────
add_executable(ch34_geometry chapters/ch34_geometry.cpp)
target_link_libraries(ch34_geometry PRIVATE vulkan_tutorial_common)
compile_shaders(ch34_geometry)

# ─── Chapter 35: Multiview (VR Stereo Rendering) ─────────────────────────────
add_executable(ch35_multiview chapters/ch35_multiview.cpp)
target_link_libraries(ch35_multiview PRIVATE vulkan_tutorial_common)
compile_shaders(ch35_multiview)

# ─── Chapter 36: Graphics Pipeline Library (GPL) ─────────────────────────────
add_executable(ch36_gpl chapters/ch36_gpl.cpp)
target_link_libraries(ch36_gpl PRIVATE vulkan_tutorial_common)
compile_shaders(ch36_gpl)

# ─── Chapter 37: Conditional Rendering ───────────────────────────────────────
add_executable(ch37_conditional_rendering chapters/ch37_conditional_rendering.cpp)
target_link_libraries(ch37_conditional_rendering PRIVATE vulkan_tutorial_common)
compile_shaders(ch37_conditional_rendering)
