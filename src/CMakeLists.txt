add_executable(ch_example chapters/ch_example.cpp)
target_link_libraries(ch_example PRIVATE vulkan_tutorial_common vulkan_tutorial_assets)

add_executable(ch00_start chapters/ch00_start.cpp)
target_link_libraries(ch00_start PRIVATE vulkan_tutorial_common)

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
compile_shaders(ch06_pipeline)

# ─── Chapter 07: Framebuffers & Command Buffers ───────────────────────────────
add_executable(ch07_commands chapters/ch07_commands.cpp)
target_link_libraries(ch07_commands PRIVATE vulkan_tutorial_common)
compile_shaders(ch07_commands)

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
include(CheckSymbolExists)
check_symbol_exists(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT
                    "vulkan/vulkan.h"
                    VULKAN_HEADERS_HAVE_MESH_SHADER_EXT)

if(SHADER_COMPILER_SUPPORTS_MESH_SHADERS AND VULKAN_HEADERS_HAVE_MESH_SHADER_EXT)
    add_executable(ch21_mesh_shader chapters/ch21_mesh_shader.cpp)
    target_link_libraries(ch21_mesh_shader PRIVATE vulkan_tutorial_common)
    compile_shaders(ch21_mesh_shader)
else()
    message(STATUS "Skipping ch21_mesh_shader: shader compiler or Vulkan headers lack VK_EXT_mesh_shader support")
endif()

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
include(CheckSymbolExists)
check_symbol_exists(VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT "vulkan/vulkan.h"
                    VULKAN_HEADERS_HAVE_SHADER_OBJECT)

if(VULKAN_HEADERS_HAVE_SHADER_OBJECT)
    add_executable(ch31_shader_object chapters/ch31_shader_object.cpp)
    target_link_libraries(ch31_shader_object PRIVATE vulkan_tutorial_common)
    compile_shaders(ch31_shader_object)
else()
    message(STATUS "Skipping ch31_shader_object: Vulkan headers lack VK_EXT_shader_object types")
endif()

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
check_symbol_exists(VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT
                    "vulkan/vulkan.h"
                    VULKAN_HEADERS_HAVE_GPL)
if(VULKAN_HEADERS_HAVE_GPL)
    add_executable(ch36_gpl chapters/ch36_gpl.cpp)
    target_link_libraries(ch36_gpl PRIVATE vulkan_tutorial_common)
    compile_shaders(ch36_gpl)
else()
    message(STATUS "Skipping ch36_gpl: Vulkan headers lack VK_EXT_graphics_pipeline_library types")
endif()

# ─── Chapter 37: Conditional Rendering ───────────────────────────────────────
add_executable(ch37_conditional_rendering chapters/ch37_conditional_rendering.cpp)
target_link_libraries(ch37_conditional_rendering PRIVATE vulkan_tutorial_common)
compile_shaders(ch37_conditional_rendering)

# ─── Chapter 38: Alpha Blending ──────────────────────────────────────────────
add_executable(ch38_alpha_blend chapters/ch38_alpha_blend.cpp)
target_link_libraries(ch38_alpha_blend PRIVATE vulkan_tutorial_common vulkan_tutorial_assets vulkan_tutorial_interactive)
compile_shaders(ch38_alpha_blend)

# ─── Chapter 39: Stencil Buffer ──────────────────────────────────────────────
add_executable(ch39_stencil chapters/ch39_stencil.cpp)
target_link_libraries(ch39_stencil PRIVATE vulkan_tutorial_common vulkan_tutorial_assets vulkan_tutorial_interactive)
compile_shaders(ch39_stencil)

# ─── Chapter 40: Normal Mapping ──────────────────────────────────────────────
add_executable(ch40_normal_mapping chapters/ch40_normal_mapping.cpp)
target_link_libraries(ch40_normal_mapping PRIVATE vulkan_tutorial_common vulkan_tutorial_assets vulkan_tutorial_interactive)
compile_shaders(ch40_normal_mapping)

# ─── Chapter 41: Cubemap Skybox ──────────────────────────────────────────────
add_executable(ch41_skybox chapters/ch41_skybox.cpp)
target_link_libraries(ch41_skybox PRIVATE vulkan_tutorial_common vulkan_tutorial_assets vulkan_tutorial_interactive)
compile_shaders(ch41_skybox)

# ─── Chapter 42: Image-Based Lighting ────────────────────────────────────────
add_executable(ch42_ibl chapters/ch42_ibl.cpp)
target_link_libraries(ch42_ibl PRIVATE vulkan_tutorial_common vulkan_tutorial_assets vulkan_tutorial_interactive)
compile_shaders(ch42_ibl)

# ─── Chapter 43: glTF Loading ────────────────────────────────────────────────
add_executable(ch43_gltf chapters/ch43_gltf.cpp)
target_link_libraries(ch43_gltf PRIVATE vulkan_tutorial_common vulkan_tutorial_assets vulkan_tutorial_interactive)
compile_shaders(ch43_gltf)

# ─── Chapter 44: PBR Rendering ───────────────────────────────────────────────
add_executable(ch44_pbr chapters/ch44_pbr.cpp)
target_link_libraries(ch44_pbr PRIVATE vulkan_tutorial_common vulkan_tutorial_assets vulkan_tutorial_interactive)
compile_shaders(ch44_pbr)

# ─── Chapter 45: Cascaded Shadow Maps ────────────────────────────────────────
add_executable(ch45_csm chapters/ch45_csm.cpp)
target_link_libraries(ch45_csm PRIVATE vulkan_tutorial_common vulkan_tutorial_assets vulkan_tutorial_interactive)
compile_shaders(ch45_csm)

# ─── Chapter 46: SSAO ────────────────────────────────────────────────────────
add_executable(ch46_ssao chapters/ch46_ssao.cpp)
target_link_libraries(ch46_ssao PRIVATE vulkan_tutorial_common vulkan_tutorial_assets vulkan_tutorial_interactive)
compile_shaders(ch46_ssao)

# ─── Chapter 47: Screen Space Reflections ────────────────────────────────────
add_executable(ch47_ssr chapters/ch47_ssr.cpp)
target_link_libraries(ch47_ssr PRIVATE vulkan_tutorial_common vulkan_tutorial_assets vulkan_tutorial_interactive)
compile_shaders(ch47_ssr)

# ─── Chapter 48: Weighted Blended OIT ────────────────────────────────────────
add_executable(ch48_oit chapters/ch48_oit.cpp)
target_link_libraries(ch48_oit PRIVATE vulkan_tutorial_common vulkan_tutorial_assets vulkan_tutorial_interactive)
compile_shaders(ch48_oit)

# ─── Chapter 49: Temporal Anti-Aliasing ──────────────────────────────────────
add_executable(ch49_taa chapters/ch49_taa.cpp)
target_link_libraries(ch49_taa PRIVATE vulkan_tutorial_common vulkan_tutorial_assets vulkan_tutorial_interactive)
compile_shaders(ch49_taa)

# ─── Chapter 50: Skeletal Animation ──────────────────────────────────────────
add_executable(ch50_skinning chapters/ch50_skinning.cpp)
target_link_libraries(ch50_skinning PRIVATE vulkan_tutorial_common vulkan_tutorial_assets vulkan_tutorial_interactive)
compile_shaders(ch50_skinning)

# ─── Chapter 51: Render Graph ────────────────────────────────────────────────
add_executable(ch51_render_graph chapters/ch51_render_graph.cpp)
target_link_libraries(ch51_render_graph PRIVATE vulkan_tutorial_common vulkan_tutorial_assets
    vulkan_tutorial_interactive vulkan_tutorial_render_graph)
compile_shaders(ch51_render_graph)

# ─── Chapter 52: GPU Particles ───────────────────────────────────────────────
add_executable(ch52_particles chapters/ch52_particles.cpp)
target_link_libraries(ch52_particles PRIVATE vulkan_tutorial_common vulkan_tutorial_assets
    vulkan_tutorial_interactive)
compile_shaders(ch52_particles)

# ─── Chapter 53: Water Rendering ─────────────────────────────────────────────
add_executable(ch53_water chapters/ch53_water.cpp)
target_link_libraries(ch53_water PRIVATE vulkan_tutorial_common vulkan_tutorial_assets
    vulkan_tutorial_interactive)
compile_shaders(ch53_water)

# ─── Chapter 54: Volumetric Fog ──────────────────────────────────────────────
add_executable(ch54_volumetric_fog chapters/ch54_volumetric_fog.cpp)
target_link_libraries(ch54_volumetric_fog PRIVATE vulkan_tutorial_common vulkan_tutorial_assets
    vulkan_tutorial_interactive)
compile_shaders(ch54_volumetric_fog)

# ─── Chapter 55: Screen-Space Decal (source generated by agent) ──────────────
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/chapters/ch55_decal.cpp)
    add_executable(ch55_decal chapters/ch55_decal.cpp)
    target_link_libraries(ch55_decal PRIVATE vulkan_tutorial_common vulkan_tutorial_assets vulkan_tutorial_interactive)
    compile_shaders(ch55_decal)
endif()

# ─── Chapter 56: Async Texture Loading ───────────────────────────────────────
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/chapters/ch56_async_loading.cpp)
    add_executable(ch56_async_loading chapters/ch56_async_loading.cpp)
    target_link_libraries(ch56_async_loading PRIVATE vulkan_tutorial_common vulkan_tutorial_assets vulkan_tutorial_interactive)
    compile_shaders(ch56_async_loading)
endif()

# ─── Chapter 57: Parallel Command Recording ──────────────────────────────────
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/chapters/ch57_parallel_record.cpp)
    add_executable(ch57_parallel_record chapters/ch57_parallel_record.cpp)
    target_link_libraries(ch57_parallel_record PRIVATE vulkan_tutorial_common vulkan_tutorial_assets vulkan_tutorial_interactive)
    compile_shaders(ch57_parallel_record)
endif()

# ─── Chapter 58: PCSS Soft Shadows ───────────────────────────────────────────
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/chapters/ch58_pcss.cpp)
    add_executable(ch58_pcss chapters/ch58_pcss.cpp)
    target_link_libraries(ch58_pcss PRIVATE vulkan_tutorial_common vulkan_tutorial_assets vulkan_tutorial_interactive)
    compile_shaders(ch58_pcss)
endif()

# ─── Chapter 59: Reflection Probe ────────────────────────────────────────────
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/chapters/ch59_reflection_probe.cpp)
    add_executable(ch59_reflection_probe chapters/ch59_reflection_probe.cpp)
    target_link_libraries(ch59_reflection_probe PRIVATE vulkan_tutorial_common vulkan_tutorial_assets vulkan_tutorial_interactive)
    compile_shaders(ch59_reflection_probe)
endif()

# ─── Chapter 60: Outdoor Demo ────────────────────────────────────────────────
add_executable(ch60_outdoor_demo chapters/ch60_outdoor_demo.cpp)
target_link_libraries(ch60_outdoor_demo PRIVATE vulkan_tutorial_common vulkan_tutorial_assets
    vulkan_tutorial_interactive vulkan_tutorial_render_graph)
compile_shaders(ch60_outdoor_demo)

# ─── ch71-ch83 新章节 ────────────────────────────────────────────────────────
macro(add_advanced_chapter num name)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/chapters/${name}.cpp)
        add_executable(${name} chapters/${name}.cpp)
        target_link_libraries(${name} PRIVATE
            vulkan_tutorial_common
            vulkan_tutorial_assets
            vulkan_tutorial_interactive
            vulkan_tutorial_engine)
        compile_shaders(${name})
    endif()
endmacro()

add_advanced_chapter(71 ch71_depth_of_field)
add_advanced_chapter(72 ch72_motion_blur)
add_advanced_chapter(73 ch73_atmospheric_scattering)
add_advanced_chapter(74 ch74_ssgi)
add_advanced_chapter(75 ch75_lut_color_grading)
add_advanced_chapter(76 ch76_clustered_forward)
add_advanced_chapter(77 ch77_virtual_texture)
add_advanced_chapter(78 ch78_async_compute)
add_advanced_chapter(79 ch79_terrain_lod)
add_advanced_chapter(80 ch80_vegetation)
add_advanced_chapter(81 ch81_descriptor_buffer)
add_advanced_chapter(82 ch82_device_generated_commands)
add_advanced_chapter(83 ch83_restir)
add_advanced_chapter(84 ch84_toon_rendering)
add_advanced_chapter(85 ch85_lens_flare)
add_advanced_chapter(86 ch86_subsurface_scattering)
add_advanced_chapter(87 ch87_parallax_occlusion)
add_advanced_chapter(88 ch88_dynamic_weather)
# Engine algorithm labs: these use DemoApp for visualization but do not claim
# an independent Vulkan rendering technique.
add_advanced_chapter(89 ch89_navmesh_pathfinding)
add_advanced_chapter(92 ch92_network_sync)

# Engine/GPU integration demos with real compute and graphics data paths.
add_advanced_chapter(90 ch90_scene_streaming)
add_advanced_chapter(91 ch91_physics_integration)
add_advanced_chapter(93 ch93_procedural_animation)
add_advanced_chapter(94 ch94_minigame_demo)
add_advanced_chapter(95 ch95_sync2_barriers)

# Standalone GPU micro-tutorials. Each target owns its pipeline variant and shaders.
add_executable(ch96_push_constants chapters/ch96_push_constants.cpp)
target_link_libraries(ch96_push_constants PRIVATE vulkan_tutorial_common)
compile_shaders(ch96_push_constants)

add_executable(ch97_multi_draw chapters/ch97_multi_draw.cpp)
target_link_libraries(ch97_multi_draw PRIVATE vulkan_tutorial_common)
compile_shaders(ch97_multi_draw)

add_executable(ch98_specialization_constants chapters/ch98_specialization_constants.cpp)
target_link_libraries(ch98_specialization_constants PRIVATE vulkan_tutorial_common)
compile_shaders(ch98_specialization_constants)

add_executable(ch99_attack_slash chapters/ch99_attack_slash.cpp)
target_link_libraries(ch99_attack_slash PRIVATE vulkan_tutorial_common)
compile_shaders(ch99_attack_slash)

add_executable(ch100_gpu_particle_pool chapters/ch100_gpu_particle_pool.cpp)
target_link_libraries(ch100_gpu_particle_pool PRIVATE vulkan_tutorial_common vulkan_tutorial_assets
    vulkan_tutorial_interactive)
compile_shaders(ch100_gpu_particle_pool)

add_executable(ch101_procedural_heightmap chapters/ch101_procedural_heightmap.cpp)
target_link_libraries(ch101_procedural_heightmap PRIVATE vulkan_tutorial_common vulkan_tutorial_assets
    vulkan_tutorial_interactive vulkan_tutorial_engine)
compile_shaders(ch101_procedural_heightmap)

add_executable(ch102_hydraulic_erosion chapters/ch102_hydraulic_erosion.cpp)
target_link_libraries(ch102_hydraulic_erosion PRIVATE vulkan_tutorial_common vulkan_tutorial_assets
    vulkan_tutorial_interactive vulkan_tutorial_engine)
compile_shaders(ch102_hydraulic_erosion)

add_executable(ch103_gpu_clipmap_terrain chapters/ch103_gpu_clipmap_terrain.cpp)
target_link_libraries(ch103_gpu_clipmap_terrain PRIVATE vulkan_tutorial_common vulkan_tutorial_assets
    vulkan_tutorial_interactive vulkan_tutorial_engine)
compile_shaders(ch103_gpu_clipmap_terrain)

add_executable(ch104_procedural_biomes chapters/ch104_procedural_biomes.cpp)
target_link_libraries(ch104_procedural_biomes PRIVATE vulkan_tutorial_common vulkan_tutorial_assets
    vulkan_tutorial_interactive vulkan_tutorial_engine)
compile_shaders(ch104_procedural_biomes)

add_executable(ch105_procedural_vegetation chapters/ch105_procedural_vegetation.cpp)
target_link_libraries(ch105_procedural_vegetation PRIVATE vulkan_tutorial_common vulkan_tutorial_assets
    vulkan_tutorial_interactive vulkan_tutorial_engine)
compile_shaders(ch105_procedural_vegetation)

add_executable(ch106_marching_cubes chapters/ch106_marching_cubes.cpp)
target_link_libraries(ch106_marching_cubes PRIVATE vulkan_tutorial_common vulkan_tutorial_assets
    vulkan_tutorial_interactive vulkan_tutorial_engine)
compile_shaders(ch106_marching_cubes)

add_executable(ch107_sdf_caves chapters/ch107_sdf_caves.cpp)
target_link_libraries(ch107_sdf_caves PRIVATE vulkan_tutorial_common vulkan_tutorial_assets
    vulkan_tutorial_interactive vulkan_tutorial_engine)
compile_shaders(ch107_sdf_caves)

add_executable(ch108_procedural_rivers_roads chapters/ch108_procedural_rivers_roads.cpp)
target_link_libraries(ch108_procedural_rivers_roads PRIVATE vulkan_tutorial_common vulkan_tutorial_assets
    vulkan_tutorial_interactive vulkan_tutorial_engine)
compile_shaders(ch108_procedural_rivers_roads)

# Production Vulkan engineering chapters.
foreach(PRODUCTION_CHAPTER
        ch109_sync2_multiqueue
        ch110_frame_context
        ch111_async_upload
        ch112_swapchain_pacing
        ch113_capability_profile
        ch114_debug_diagnostics)
    add_executable(${PRODUCTION_CHAPTER} chapters/${PRODUCTION_CHAPTER}.cpp)
    target_link_libraries(${PRODUCTION_CHAPTER} PRIVATE vulkan_tutorial_common vulkan_tutorial_assets
        vulkan_tutorial_interactive vulkan_tutorial_engine)
    compile_shaders(${PRODUCTION_CHAPTER})
endforeach()

add_executable(ch115_headless_tests chapters/ch115_headless_tests.cpp)
target_link_libraries(ch115_headless_tests PRIVATE vulkan_tutorial_common)
compile_shaders(ch115_headless_tests)
if(BUILD_TESTING)
    add_test(NAME ch115_headless COMMAND ch115_headless_tests)
    set_tests_properties(ch115_headless PROPERTIES SKIP_RETURN_CODE 77)
endif()

foreach(PRODUCTION_GRAPH_CHAPTER
        ch116_render_graph2
        ch117_shader_pipeline_toolchain
        ch118_gpu_profiling2)
    add_executable(${PRODUCTION_GRAPH_CHAPTER} chapters/${PRODUCTION_GRAPH_CHAPTER}.cpp)
    target_link_libraries(${PRODUCTION_GRAPH_CHAPTER} PRIVATE vulkan_tutorial_common vulkan_tutorial_assets
        vulkan_tutorial_interactive vulkan_tutorial_engine)
    compile_shaders(${PRODUCTION_GRAPH_CHAPTER})
endforeach()

# ─── Engine chapters macro ────────────────────────────────────────────────────
# 使用宏简化重复注册
macro(add_engine_chapter num name)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/chapters/${name}.cpp)
        add_executable(${name} chapters/${name}.cpp)
        target_link_libraries(${name} PRIVATE
            vulkan_tutorial_common
            vulkan_tutorial_assets
            vulkan_tutorial_interactive
            vulkan_tutorial_engine)
        compile_shaders(${name})
    endif()
endmacro()

# ─── Chapter 61-70: Engine Chapters ──────────────────────────────────────────
add_engine_chapter(61 ch61_rhi_device)
add_engine_chapter(62 ch62_rhi_buffer)
add_engine_chapter(63 ch63_rhi_texture)
add_engine_chapter(64 ch64_rhi_shader)
add_engine_chapter(65 ch65_pipeline_builder)
add_engine_chapter(66 ch66_descriptor_manager)
add_engine_chapter(67 ch67_command_recorder)
add_engine_chapter(68 ch68_scene_ecs)
add_engine_chapter(69 ch69_material_system)

# ch70 需要 render_graph
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/chapters/ch70_mini_engine.cpp)
    add_executable(ch70_mini_engine chapters/ch70_mini_engine.cpp)
    target_link_libraries(ch70_mini_engine PRIVATE
        vulkan_tutorial_common
        vulkan_tutorial_assets
        vulkan_tutorial_interactive
        vulkan_tutorial_engine
        vulkan_tutorial_render_graph)
    compile_shaders(ch70_mini_engine)
endif()
