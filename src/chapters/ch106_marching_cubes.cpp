/**
 * @file ch106_marching_cubes.cpp
 * @brief GPU marching-tetrahedra isosurface generation with indirect drawing.
 *
 * The compute pass evaluates a procedural 3D density field, splits each voxel
 * into six tetrahedra, and appends generated triangles to a device-local SSBO.
 * A second compute pass converts the atomic triangle count into a Vulkan
 * indirect draw command. No surface vertices or draw counts are produced by
 * the CPU.
 */

#include <vulkan_tutorial/engine/demo_app.hpp>

#include <imgui.h>
#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

using namespace vulkan_tutorial;

namespace {
constexpr uint32_t GRID_CELLS = 28;
constexpr uint32_t TETRAHEDRA_PER_CELL = 6;
constexpr uint32_t MAX_TRIANGLES_PER_TETRAHEDRON = 2;
constexpr uint32_t MAX_TRIANGLES =
    GRID_CELLS * GRID_CELLS * GRID_CELLS * TETRAHEDRA_PER_CELL * MAX_TRIANGLES_PER_TETRAHEDRON;
constexpr uint32_t MAX_VERTICES = MAX_TRIANGLES * 3;

struct alignas(16) SurfaceVertex {
    glm::vec4 position;
    glm::vec4 normal;
};

struct alignas(16) SurfaceCounters {
    uint32_t triangleCount;
    uint32_t overflowCount;
    uint32_t activeCellCount;
    uint32_t padding;
};

struct alignas(16) GenerationPush {
    glm::vec4 volume;
    glm::vec4 shape;
    glm::uvec4 grid;
};

struct alignas(16) CameraPush {
    glm::mat4 viewProjection;
    glm::vec4 cameraPosition;
    glm::vec4 renderParams;
};

static_assert(sizeof(SurfaceVertex) == 32);
static_assert(sizeof(SurfaceCounters) == 16);
static_assert(sizeof(GenerationPush) == 48);
static_assert(sizeof(CameraPush) == 96);
} // namespace

class Ch106App final : public DemoApp {
  protected:
    bool needsDepthBuffer() const override {
        return true;
    }

    void onInit() override {
        bgColor_ = {0.012f, 0.026f, 0.052f};
        interactive_.camera().setTarget({0.0f, 0.0f, 0.0f});
        interactive_.camera().setDistance(12.5f);
        interactive_.camera().setAngles(42.0f, 21.0f);
        createGpuBuffers();
        createDescriptors();
        createComputePipelines();
        createGraphicsPipeline();
    }

    void onUpdate() override {
        if (animate_) {
            elapsed_ += 0.016f;
            geometryDirty_ = true;
        }
    }

    void buildUi() override {
        interactive_.buildDebugPanel("Chapter 106: GPU Marching Cubes");
        ImGui::SetNextWindowPos(ImVec2(12.0f, 315.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(330.0f, 0.0f), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Isosurface Generator", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            geometryDirty_ |= ImGui::SliderFloat("Iso level", &isoLevel_, -0.65f, 0.65f, "%.3f");
            geometryDirty_ |=
                ImGui::SliderFloat("Noise frequency", &noiseFrequency_, 0.25f, 1.75f, "%.2f");
            geometryDirty_ |=
                ImGui::SliderFloat("Noise strength", &noiseStrength_, 0.0f, 1.15f, "%.2f");
            geometryDirty_ |=
                ImGui::SliderFloat("Cave strength", &caveStrength_, 0.0f, 1.0f, "%.2f");
            geometryDirty_ |=
                ImGui::SliderFloat("Lobe blending", &lobeBlend_, 0.25f, 1.35f, "%.2f");
            geometryDirty_ |= ImGui::SliderInt("Seed", &seed_, 1, 2048);
            geometryDirty_ |= ImGui::Checkbox("Animate density field", &animate_);
            ImGui::Separator();
            ImGui::SliderFloat("Exposure", &exposure_, 0.65f, 2.4f, "%.2f");
            ImGui::SliderFloat("Rim light", &rimStrength_, 0.0f, 1.2f, "%.2f");
            ImGui::Text("Grid: %u x %u x %u", GRID_CELLS, GRID_CELLS, GRID_CELLS);
            ImGui::Text("Capacity: %u triangles", MAX_TRIANGLES);
            ImGui::TextUnformatted("Compute -> compact SSBO -> indirect draw");
        }
        ImGui::End();
    }

    void onRecordPreRender(VkCommandBuffer commandBuffer, uint32_t) override {
        if (generatedOnce_ && !geometryDirty_)
            return;

        if (generatedOnce_) {
            std::array<VkBufferMemoryBarrier, 3> reuseBarriers{};
            const VkBuffer buffers[] = {surfaceBuffer_, counterBuffer_, indirectBuffer_};
            for (size_t index = 0; index < reuseBarriers.size(); ++index) {
                auto& barrier = reuseBarriers[index];
                barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.buffer = buffers[index];
                barrier.offset = 0;
                barrier.size = VK_WHOLE_SIZE;
            }
            reuseBarriers[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            reuseBarriers[0].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            reuseBarriers[1].srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            reuseBarriers[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            reuseBarriers[2].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
            reuseBarriers[2].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(commandBuffer,
                                 VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                     VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                                 static_cast<uint32_t>(reuseBarriers.size()), reuseBarriers.data(), 0, nullptr);
        }

        vkCmdFillBuffer(commandBuffer, counterBuffer_, 0, sizeof(SurfaceCounters), 0);
        vkCmdFillBuffer(commandBuffer, indirectBuffer_, 0, sizeof(VkDrawIndirectCommand), 0);
        std::array<VkBufferMemoryBarrier, 2> resetBarriers{};
        const VkBuffer resetBuffers[] = {counterBuffer_, indirectBuffer_};
        for (size_t index = 0; index < resetBarriers.size(); ++index) {
            auto& barrier = resetBarriers[index];
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = resetBuffers[index];
            barrier.offset = 0;
            barrier.size = VK_WHOLE_SIZE;
        }
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             0, nullptr, static_cast<uint32_t>(resetBarriers.size()), resetBarriers.data(), 0, nullptr);

        GenerationPush generation{};
        generation.volume = glm::vec4(VOLUME_HALF_EXTENT, isoLevel_, elapsed_, noiseFrequency_);
        generation.shape = glm::vec4(noiseStrength_, caveStrength_, lobeBlend_, 0.0f);
        generation.grid = glm::uvec4(GRID_CELLS, MAX_TRIANGLES, static_cast<uint32_t>(seed_), 0u);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, generationPipeline_);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computeLayout_, 0, 1, &descriptorSet_,
                                0, nullptr);
        vkCmdPushConstants(commandBuffer, computeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(generation),
                           &generation);
        constexpr uint32_t groups = (GRID_CELLS + 3u) / 4u;
        vkCmdDispatch(commandBuffer, groups, groups, groups);

        VkBufferMemoryBarrier counterBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        counterBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        counterBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        counterBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        counterBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        counterBarrier.buffer = counterBuffer_;
        counterBarrier.offset = 0;
        counterBarrier.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &counterBarrier, 0, nullptr);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, finalizePipeline_);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computeLayout_, 0, 1, &descriptorSet_,
                                0, nullptr);
        vkCmdPushConstants(commandBuffer, computeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(generation),
                           &generation);
        vkCmdDispatch(commandBuffer, 1, 1, 1);

        std::array<VkBufferMemoryBarrier, 2> drawBarriers{};
        drawBarriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        drawBarriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        drawBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        drawBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        drawBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        drawBarriers[0].buffer = surfaceBuffer_;
        drawBarriers[0].size = VK_WHOLE_SIZE;
        drawBarriers[1] = drawBarriers[0];
        drawBarriers[1].dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        drawBarriers[1].buffer = indirectBuffer_;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 0, nullptr,
                             static_cast<uint32_t>(drawBarriers.size()), drawBarriers.data(), 0, nullptr);
        generatedOnce_ = true;
        geometryDirty_ = false;
    }

    void onRecordRender(VkCommandBuffer commandBuffer, uint32_t) override {
        VkViewport viewport{0.0f, 0.0f, static_cast<float>(extent_.width), static_cast<float>(extent_.height), 0.0f,
                            1.0f};
        VkRect2D scissor{{0, 0}, extent_};
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline_);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsLayout_, 0, 1,
                                &descriptorSet_, 0, nullptr);

        CameraPush camera{};
        const float aspect = static_cast<float>(extent_.width) / static_cast<float>(std::max(1u, extent_.height));
        camera.viewProjection = interactive_.camera().projectionMatrix(aspect, 51.0f, 0.1f, 80.0f) *
                                interactive_.camera().viewMatrix();
        camera.cameraPosition = glm::vec4(interactive_.camera().eyePosition(), 1.0f);
        camera.renderParams = glm::vec4(elapsed_, exposure_, rimStrength_, 0.0f);
        vkCmdPushConstants(commandBuffer, graphicsLayout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(camera), &camera);
        vkCmdDrawIndirect(commandBuffer, indirectBuffer_, 0, 1, sizeof(VkDrawIndirectCommand));
    }

    void onShutdown() override {
        vkDestroyPipeline(device_, graphicsPipeline_, nullptr);
        vkDestroyPipelineLayout(device_, graphicsLayout_, nullptr);
        vkDestroyPipeline(device_, finalizePipeline_, nullptr);
        vkDestroyPipeline(device_, generationPipeline_, nullptr);
        vkDestroyPipelineLayout(device_, computeLayout_, nullptr);
        vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
        vkDestroyBuffer(device_, indirectBuffer_, nullptr);
        vkFreeMemory(device_, indirectMemory_, nullptr);
        vkDestroyBuffer(device_, counterBuffer_, nullptr);
        vkFreeMemory(device_, counterMemory_, nullptr);
        vkDestroyBuffer(device_, surfaceBuffer_, nullptr);
        vkFreeMemory(device_, surfaceMemory_, nullptr);
    }

  private:
    static constexpr float VOLUME_HALF_EXTENT = 5.2f;

    VkBuffer surfaceBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory surfaceMemory_ = VK_NULL_HANDLE;
    VkBuffer counterBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory counterMemory_ = VK_NULL_HANDLE;
    VkBuffer indirectBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory indirectMemory_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;
    VkPipelineLayout computeLayout_ = VK_NULL_HANDLE;
    VkPipeline generationPipeline_ = VK_NULL_HANDLE;
    VkPipeline finalizePipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout graphicsLayout_ = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline_ = VK_NULL_HANDLE;

    bool generatedOnce_ = false;
    bool geometryDirty_ = true;
    bool animate_ = true;
    float elapsed_ = 0.0f;
    float isoLevel_ = 0.0f;
    float noiseFrequency_ = 0.82f;
    float noiseStrength_ = 0.48f;
    float caveStrength_ = 1.0f;
    float lobeBlend_ = 1.0f;
    float exposure_ = 1.45f;
    float rimStrength_ = 0.46f;
    int seed_ = 317;

    void createGpuBuffers() {
        constexpr VkDeviceSize surfaceSize = VkDeviceSize(MAX_VERTICES) * sizeof(SurfaceVertex);
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physDev_, &properties);
        if (surfaceSize > properties.limits.maxStorageBufferRange)
            throw std::runtime_error("Marching-cubes surface buffer exceeds maxStorageBufferRange");

        createBuffer(physDev_, device_, surfaceSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, surfaceBuffer_, surfaceMemory_);
        createBuffer(physDev_, device_, sizeof(SurfaceCounters),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, counterBuffer_, counterMemory_);
        createBuffer(physDev_, device_, sizeof(VkDrawIndirectCommand),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indirectBuffer_, indirectMemory_);
    }

    void createDescriptors() {
        std::array<VkDescriptorSetLayoutBinding, 3> bindings = {{
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT,
             nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        }};
        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &descriptorSetLayout_));

        VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        VK_CHECK(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_));

        VkDescriptorSetAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocateInfo.descriptorPool = descriptorPool_;
        allocateInfo.descriptorSetCount = 1;
        allocateInfo.pSetLayouts = &descriptorSetLayout_;
        VK_CHECK(vkAllocateDescriptorSets(device_, &allocateInfo, &descriptorSet_));

        constexpr VkDeviceSize surfaceSize = VkDeviceSize(MAX_VERTICES) * sizeof(SurfaceVertex);
        VkDescriptorBufferInfo surfaceInfo{surfaceBuffer_, 0, surfaceSize};
        VkDescriptorBufferInfo counterInfo{counterBuffer_, 0, sizeof(SurfaceCounters)};
        VkDescriptorBufferInfo indirectInfo{indirectBuffer_, 0, sizeof(VkDrawIndirectCommand)};
        const VkDescriptorBufferInfo* infos[] = {&surfaceInfo, &counterInfo, &indirectInfo};
        std::array<VkWriteDescriptorSet, 3> writes{};
        for (uint32_t index = 0; index < writes.size(); ++index) {
            writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[index].dstSet = descriptorSet_;
            writes[index].dstBinding = index;
            writes[index].descriptorCount = 1;
            writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[index].pBufferInfo = infos[index];
        }
        vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    void createComputePipelines() {
        VkPushConstantRange pushRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GenerationPush)};
        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &descriptorSetLayout_;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushRange;
        VK_CHECK(vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &computeLayout_));

        VkShaderModule generationModule =
            createShaderModuleFromFile(device_, "marching_cubes_generate.comp.spv");
        VkShaderModule finalizeModule = createShaderModuleFromFile(device_, "marching_cubes_finalize.comp.spv");
        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.layout = computeLayout_;
        pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                              VK_SHADER_STAGE_COMPUTE_BIT, generationModule, "main"};
        VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &generationPipeline_));
        pipelineInfo.stage.module = finalizeModule;
        VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &finalizePipeline_));
        vkDestroyShaderModule(device_, finalizeModule, nullptr);
        vkDestroyShaderModule(device_, generationModule, nullptr);
    }

    void createGraphicsPipeline() {
        VkPushConstantRange pushRange{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                      sizeof(CameraPush)};
        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &descriptorSetLayout_;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushRange;
        VK_CHECK(vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &graphicsLayout_));

        VkShaderModule vertexModule = createShaderModuleFromFile(device_, "marching_cubes_surface.vert.spv");
        VkShaderModule fragmentModule = createShaderModuleFromFile(device_, "marching_cubes_surface.frag.spv");
        std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {{
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT,
             vertexModule, "main"},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT,
             fragmentModule, "main"},
        }};
        VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport.viewportCount = 1;
        viewport.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rasterization{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rasterization.polygonMode = VK_POLYGON_MODE_FILL;
        rasterization.cullMode = VK_CULL_MODE_NONE;
        rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterization.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo multisampling{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depth.depthTestEnable = VK_TRUE;
        depth.depthWriteEnable = VK_TRUE;
        depth.depthCompareOp = VK_COMPARE_OP_LESS;
        VkPipelineColorBlendAttachmentState blendAttachment{};
        blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo blending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blending.attachmentCount = 1;
        blending.pAttachments = &blendAttachment;
        const std::array<VkDynamicState, 2> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamic.pDynamicStates = dynamicStates.data();

        VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineInfo.pStages = shaderStages.data();
        pipelineInfo.pVertexInputState = &vertexInput;
        pipelineInfo.pInputAssemblyState = &assembly;
        pipelineInfo.pViewportState = &viewport;
        pipelineInfo.pRasterizationState = &rasterization;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depth;
        pipelineInfo.pColorBlendState = &blending;
        pipelineInfo.pDynamicState = &dynamic;
        pipelineInfo.layout = graphicsLayout_;
        pipelineInfo.renderPass = renderPass_;
        pipelineInfo.subpass = 0;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline_));
        vkDestroyShaderModule(device_, fragmentModule, nullptr);
        vkDestroyShaderModule(device_, vertexModule, nullptr);
    }
};

int main() {
    try {
        Ch106App app;
        app.run("ch106 - GPU Marching Cubes", 1280, 720);
    } catch (const std::exception& error) {
        std::cerr << "ch106 failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
