/**
 * @file ch105_procedural_vegetation.cpp
 * @brief GPU-generated vegetation with compute culling and indirect drawing.
 */

#include <vulkan_tutorial/engine/demo_app.hpp>

#include <imgui.h>
#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

using namespace vulkan_tutorial;

namespace {
constexpr uint32_t VEGETATION_GRID_WIDTH = 160;
constexpr uint32_t VEGETATION_GRID_HEIGHT = 160;
constexpr uint32_t MAX_VEGETATION_INSTANCES = VEGETATION_GRID_WIDTH * VEGETATION_GRID_HEIGHT;
constexpr uint32_t TERRAIN_GRID_SIZE = 144;
constexpr uint32_t TERRAIN_VERTEX_COUNT =
    (TERRAIN_GRID_SIZE - 1) * (TERRAIN_GRID_SIZE - 1) * 6;

struct alignas(16) GpuVegetationInstance {
    glm::vec4 positionHeight;
    glm::vec4 attributes;
};

struct alignas(16) ScatterPush {
    glm::vec4 cameraTime;
    glm::vec4 cameraForwardFov;
    glm::vec4 fieldClimate;
    glm::vec4 windCullSlope;
    glm::uvec4 gridLimits;
};

struct alignas(16) ScenePush {
    glm::mat4 viewProjection;
    glm::vec4 cameraTime;
    glm::vec4 terrainWind;
};

static_assert(sizeof(GpuVegetationInstance) == 32);
static_assert(sizeof(ScatterPush) == 80);
static_assert(sizeof(ScenePush) == 96);
static_assert(sizeof(VkDrawIndirectCommand) == 16);
} // namespace

class Ch105App final : public DemoApp {
  protected:
    bool needsDepthBuffer() const override { return true; }

    void onInit() override {
        bgColor_ = {0.48f, 0.62f, 0.72f};
        interactive_.camera().setTarget({0.0f, 2.8f, 0.0f});
        interactive_.camera().setDistance(38.0f);
        interactive_.camera().setAngles(42.0f, 21.0f);

        createGpuBuffers();
        createDescriptors();
        createPipelineLayouts();
        createComputePipelines();
        createGraphicsPipelines();
    }

    void onUpdate() override {
        elapsed_ += 0.016f;
    }

    void buildUi() override {
        interactive_.buildDebugPanel("第105章：GPU 程序化植被");
        ImGui::SetNextWindowPos(ImVec2(12, 315), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(340, 0), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("GPU Vegetation", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::SliderFloat("Density", &density_, 0.05f, 1.0f, "%.2f");
            ImGui::SliderFloat("Wind", &windStrength_, 0.0f, 1.8f, "%.2f");
            ImGui::SliderFloat("Field Radius", &fieldHalfExtent_, 35.0f, 95.0f, "%.0f m");
            ImGui::SliderFloat("Cull Radius", &cullRadius_, 25.0f, 110.0f, "%.0f m");
            ImGui::SliderFloat("Minimum Flatness", &minimumNormalY_, 0.72f, 0.97f, "%.2f");
            ImGui::SliderFloat("Terrain Height", &heightScale_, 5.0f, 18.0f, "%.1f m");
            ImGui::SliderFloat("Seed", &seed_, 0.0f, 1000.0f, "%.1f");
            ImGui::Separator();
            ImGui::Text("Candidates: %u", MAX_VEGETATION_INSTANCES);
            ImGui::Text("Blade vertices: 36 per instance");
            ImGui::Text("Compute -> SSBO -> vkCmdDrawIndirect");
            ImGui::TextDisabled("The live instance count remains on the GPU.");
        }
        ImGui::End();
    }

    void onRecordPreRender(VkCommandBuffer commandBuffer, uint32_t) override {
        if (hasRecordedFrame_) {
            std::array<VkBufferMemoryBarrier, 2> previousFrameBarriers{};
            previousFrameBarriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            previousFrameBarriers[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            previousFrameBarriers[0].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            previousFrameBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            previousFrameBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            previousFrameBarriers[0].buffer = instanceBuffer_;
            previousFrameBarriers[0].offset = 0;
            previousFrameBarriers[0].size = VK_WHOLE_SIZE;

            previousFrameBarriers[1] = previousFrameBarriers[0];
            previousFrameBarriers[1].srcAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
            previousFrameBarriers[1].buffer = indirectBuffer_;

            vkCmdPipelineBarrier(commandBuffer,
                                 VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                                 static_cast<uint32_t>(previousFrameBarriers.size()),
                                 previousFrameBarriers.data(), 0, nullptr);
        }

        ScatterPush scatter{};
        const glm::vec3 cameraPosition = interactive_.camera().eyePosition();
        const glm::vec3 cameraForward =
            glm::normalize(interactive_.camera().target() - cameraPosition);
        scatter.cameraTime = glm::vec4(cameraPosition, elapsed_);
        scatter.cameraForwardFov =
            glm::vec4(cameraForward, std::cos(glm::radians(55.0f)));
        scatter.fieldClimate = glm::vec4(fieldHalfExtent_, density_, heightScale_, seed_);
        scatter.windCullSlope = glm::vec4(windStrength_, cullRadius_, minimumNormalY_, 0.0f);
        scatter.gridLimits = glm::uvec4(VEGETATION_GRID_WIDTH, VEGETATION_GRID_HEIGHT,
                                        MAX_VEGETATION_INSTANCES, 0);

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computeLayout_, 0, 1,
                                &descriptorSet_, 0, nullptr);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, preparePipeline_);
        vkCmdPushConstants(commandBuffer, computeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(scatter), &scatter);
        vkCmdDispatch(commandBuffer, 1, 1, 1);

        VkBufferMemoryBarrier prepareBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        prepareBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        prepareBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        prepareBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        prepareBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        prepareBarrier.buffer = indirectBuffer_;
        prepareBarrier.offset = 0;
        prepareBarrier.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1,
                             &prepareBarrier, 0, nullptr);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scatterPipeline_);
        vkCmdDispatch(commandBuffer, (VEGETATION_GRID_WIDTH + 7) / 8,
                      (VEGETATION_GRID_HEIGHT + 7) / 8, 1);

        std::array<VkBufferMemoryBarrier, 2> renderBarriers{};
        renderBarriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        renderBarriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        renderBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        renderBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        renderBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        renderBarriers[0].buffer = instanceBuffer_;
        renderBarriers[0].offset = 0;
        renderBarriers[0].size = VK_WHOLE_SIZE;

        renderBarriers[1] = renderBarriers[0];
        renderBarriers[1].dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        renderBarriers[1].buffer = indirectBuffer_;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                             0, 0, nullptr, static_cast<uint32_t>(renderBarriers.size()),
                             renderBarriers.data(), 0, nullptr);
        hasRecordedFrame_ = true;
    }

    void onRecordRender(VkCommandBuffer commandBuffer, uint32_t) override {
        VkViewport viewport{0.0f, 0.0f, static_cast<float>(extent_.width),
                            static_cast<float>(extent_.height), 0.0f, 1.0f};
        VkRect2D scissor{{0, 0}, extent_};
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        const ScenePush scene = makeScenePush();
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, terrainPipeline_);
        vkCmdPushConstants(commandBuffer, terrainLayout_,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(scene), &scene);
        vkCmdDraw(commandBuffer, TERRAIN_VERTEX_COUNT, 1, 0, 0);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vegetationPipeline_);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vegetationLayout_,
                                0, 1, &descriptorSet_, 0, nullptr);
        vkCmdPushConstants(commandBuffer, vegetationLayout_,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(scene), &scene);
        vkCmdDrawIndirect(commandBuffer, indirectBuffer_, 0, 1, sizeof(VkDrawIndirectCommand));
    }

    void onShutdown() override {
        vkDestroyPipeline(device_, vegetationPipeline_, nullptr);
        vkDestroyPipeline(device_, terrainPipeline_, nullptr);
        vkDestroyPipeline(device_, scatterPipeline_, nullptr);
        vkDestroyPipeline(device_, preparePipeline_, nullptr);
        vkDestroyPipelineLayout(device_, vegetationLayout_, nullptr);
        vkDestroyPipelineLayout(device_, terrainLayout_, nullptr);
        vkDestroyPipelineLayout(device_, computeLayout_, nullptr);
        vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
        vkDestroyBuffer(device_, indirectBuffer_, nullptr);
        vkFreeMemory(device_, indirectMemory_, nullptr);
        vkDestroyBuffer(device_, instanceBuffer_, nullptr);
        vkFreeMemory(device_, instanceMemory_, nullptr);
    }

  private:
    VkBuffer instanceBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory instanceMemory_ = VK_NULL_HANDLE;
    VkBuffer indirectBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory indirectMemory_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;
    VkPipelineLayout computeLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout terrainLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout vegetationLayout_ = VK_NULL_HANDLE;
    VkPipeline preparePipeline_ = VK_NULL_HANDLE;
    VkPipeline scatterPipeline_ = VK_NULL_HANDLE;
    VkPipeline terrainPipeline_ = VK_NULL_HANDLE;
    VkPipeline vegetationPipeline_ = VK_NULL_HANDLE;

    float elapsed_ = 0.0f;
    float density_ = 0.78f;
    float windStrength_ = 0.72f;
    float fieldHalfExtent_ = 72.0f;
    float cullRadius_ = 82.0f;
    float minimumNormalY_ = 0.82f;
    float heightScale_ = 11.0f;
    float seed_ = 73.0f;
    bool hasRecordedFrame_ = false;

    ScenePush makeScenePush() const {
        ScenePush scene{};
        const float aspect = static_cast<float>(extent_.width) /
                             static_cast<float>(std::max(1u, extent_.height));
        scene.viewProjection = interactive_.camera().projectionMatrix(aspect, 55.0f, 0.1f, 420.0f) *
                               interactive_.camera().viewMatrix();
        scene.cameraTime = glm::vec4(interactive_.camera().eyePosition(), elapsed_);
        scene.terrainWind = glm::vec4(fieldHalfExtent_ + 32.0f, heightScale_, seed_, windStrength_);
        return scene;
    }

    void createGpuBuffers() {
        const VkDeviceSize instanceBufferSize =
            sizeof(GpuVegetationInstance) * MAX_VEGETATION_INSTANCES;
        createBuffer(physDev_, device_, instanceBufferSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     instanceBuffer_, instanceMemory_);
        createBuffer(physDev_, device_, sizeof(VkDrawIndirectCommand),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     indirectBuffer_, indirectMemory_);
    }

    void createDescriptors() {
        std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT;
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &descriptorSetLayout_));

        VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
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

        VkDescriptorBufferInfo instanceInfo{
            instanceBuffer_, 0, sizeof(GpuVegetationInstance) * MAX_VEGETATION_INSTANCES};
        VkDescriptorBufferInfo indirectInfo{indirectBuffer_, 0, sizeof(VkDrawIndirectCommand)};
        std::array<VkWriteDescriptorSet, 2> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = descriptorSet_;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &instanceInfo;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = descriptorSet_;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &indirectInfo;
        vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    void createPipelineLayouts() {
        VkPushConstantRange computeRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ScatterPush)};
        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &descriptorSetLayout_;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &computeRange;
        VK_CHECK(vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &computeLayout_));

        VkPushConstantRange sceneRange{
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ScenePush)};
        layoutInfo.setLayoutCount = 0;
        layoutInfo.pSetLayouts = nullptr;
        layoutInfo.pPushConstantRanges = &sceneRange;
        VK_CHECK(vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &terrainLayout_));

        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &descriptorSetLayout_;
        VK_CHECK(vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &vegetationLayout_));
    }

    void createComputePipelines() {
        preparePipeline_ = createComputePipeline("vegetation_gpu_prepare.comp.spv");
        scatterPipeline_ = createComputePipeline("vegetation_gpu_scatter.comp.spv");
    }

    VkPipeline createComputePipeline(const char* shaderName) {
        VkShaderModule module = createShaderModuleFromFile(device_, shaderName);
        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                              VK_SHADER_STAGE_COMPUTE_BIT, module, "main"};
        pipelineInfo.layout = computeLayout_;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline));
        vkDestroyShaderModule(device_, module, nullptr);
        return pipeline;
    }

    void createGraphicsPipelines() {
        terrainPipeline_ = createGraphicsPipeline("vegetation_gpu_terrain.vert.spv",
                                                   "vegetation_gpu_terrain.frag.spv",
                                                   terrainLayout_);
        vegetationPipeline_ = createGraphicsPipeline("vegetation_gpu_blade.vert.spv",
                                                      "vegetation_gpu_blade.frag.spv",
                                                      vegetationLayout_);
    }

    VkPipeline createGraphicsPipeline(const char* vertexShaderName,
                                      const char* fragmentShaderName,
                                      VkPipelineLayout pipelineLayout) {
        VkShaderModule vertexModule = createShaderModuleFromFile(device_, vertexShaderName);
        VkShaderModule fragmentModule = createShaderModuleFromFile(device_, fragmentShaderName);
        std::array<VkPipelineShaderStageCreateInfo, 2> stages = {{
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
             VK_SHADER_STAGE_VERTEX_BIT, vertexModule, "main"},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
             VK_SHADER_STAGE_FRAGMENT_BIT, fragmentModule, "main"},
        }};

        VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rasterization{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rasterization.polygonMode = VK_POLYGON_MODE_FILL;
        rasterization.cullMode = VK_CULL_MODE_NONE;
        rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterization.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo multisampling{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        VkPipelineColorBlendAttachmentState colorAttachment{};
        colorAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo colorBlending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorAttachment;
        std::array<VkDynamicState, 2> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamicState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
        pipelineInfo.pStages = stages.data();
        pipelineInfo.pVertexInputState = &vertexInput;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterization;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass_;
        pipelineInfo.subpass = 0;

        VkPipeline pipeline = VK_NULL_HANDLE;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline));
        vkDestroyShaderModule(device_, fragmentModule, nullptr);
        vkDestroyShaderModule(device_, vertexModule, nullptr);
        return pipeline;
    }
};

int main() {
    try {
        Ch105App app;
        app.run("ch105 - GPU Procedural Vegetation", 1280, 720);
    } catch (const std::exception& error) {
        std::cerr << "ch105 failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
