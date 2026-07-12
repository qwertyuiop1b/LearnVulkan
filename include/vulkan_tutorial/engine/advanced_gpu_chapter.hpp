#pragma once

#include <vulkan_tutorial/engine/demo_app.hpp>

#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

struct alignas(16) AdvancedGpuPush {
    std::array<glm::vec4, 8> values{};
};

static_assert(sizeof(AdvancedGpuPush) == 128);

class AdvancedGpuChapterApp : public DemoApp {
  protected:
    static constexpr uint32_t STORAGE_BUFFER_COUNT = 4;

    virtual const char* chapterTitle() const = 0;
    virtual const char* fragmentShaderName() const = 0;
    virtual const char* vertexShaderName() const {
        return "advanced_fullscreen.vert.spv";
    }
    virtual const char* computeShaderName() const {
        return nullptr;
    }
    virtual std::array<VkDeviceSize, STORAGE_BUFFER_COUNT> storageBufferSizes() const {
        return {16, 16, 16, 16};
    }
    virtual std::array<VkBufferUsageFlags, STORAGE_BUFFER_COUNT> extraBufferUsage() const {
        return {};
    }
    virtual void initializeStorageBuffer(uint32_t, void*, VkDeviceSize) {}
    virtual void configureChapter() {}
    virtual void updateChapter() {}
    virtual void buildChapterUi() = 0;
    virtual void fillPushConstants(AdvancedGpuPush& push) const = 0;
    virtual glm::uvec3 computeDispatch() const {
        return {1, 1, 1};
    }
    virtual bool dispatchComputeThisFrame() const {
        return computeShaderName() != nullptr;
    }
    virtual uint32_t drawVertexCount() const {
        return 3;
    }
    virtual uint32_t drawInstanceCount() const {
        return 1;
    }
    virtual int32_t indirectDrawBufferIndex() const {
        return -1;
    }
    virtual int32_t indirectDrawCountBufferIndex() const { return -1; }
    virtual uint32_t indirectMaxDrawCount() const { return 1; }
    virtual VkPrimitiveTopology primitiveTopology() const {
        return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }
    virtual bool chapterNeedsDepth() const {
        return false;
    }
    virtual bool chapterUsesAlphaBlend() const {
        return false;
    }
    virtual bool chapterUsesSynchronization2() const {
        return false;
    }
    virtual void beforeAdvancedCompute(VkCommandBuffer, uint32_t) {}
    virtual void afterAdvancedCompute(VkCommandBuffer, uint32_t) {}
    virtual void beforeAdvancedDraw(VkCommandBuffer, uint32_t) {}
    virtual void afterAdvancedDraw(VkCommandBuffer, uint32_t) {}
    virtual void onAdvancedInit() {}
    virtual void onAdvancedShutdown() {}

    bool needsDepthBuffer() const final {
        return chapterNeedsDepth();
    }

    void onInit() final {
        configureChapter();
        createStorageBuffers();
        createDescriptorResources();
        createPipelineLayout();
        createGraphicsPipeline();
        if (computeShaderName() != nullptr)
            createComputePipeline();
        onAdvancedInit();
    }

    void onUpdate() final {
        elapsed_ += 0.016f;
        updateChapter();
    }

    void buildUi() final {
        interactive_.buildDebugPanel(chapterTitle());
        buildChapterUi();
    }

    void onRecordPreRender(VkCommandBuffer commandBuffer, uint32_t frameIndex) final {
        currentFrameIndex_ = frameIndex;
        if (computePipeline_ == VK_NULL_HANDLE || !dispatchComputeThisFrame())
            return;

        beforeAdvancedCompute(commandBuffer, frameIndex);

        if (chapterUsesSynchronization2()) {
            recordReuseBarriers2(commandBuffer);
        } else {
            std::array<VkBufferMemoryBarrier, STORAGE_BUFFER_COUNT> reuseBarriers{};
            for (uint32_t index = 0; index < STORAGE_BUFFER_COUNT; ++index) {
                auto& barrier = reuseBarriers[index];
                barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                barrier.srcAccessMask = hasDispatched_ ?
                    (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                     VK_ACCESS_INDIRECT_COMMAND_READ_BIT) : VK_ACCESS_HOST_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.buffer = storageBuffers_[index];
                barrier.offset = 0;
                barrier.size = storageSizes_[index];
            }
            const VkPipelineStageFlags sourceStages = hasDispatched_ ?
                (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT) :
                VK_PIPELINE_STAGE_HOST_BIT;
            vkCmdPipelineBarrier(commandBuffer, sourceStages, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                 0, nullptr, STORAGE_BUFFER_COUNT, reuseBarriers.data(), 0, nullptr);
        }

        AdvancedGpuPush push{};
        fillPushConstants(push);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline_);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_,
                                0, 1, &descriptorSet_, 0, nullptr);
        vkCmdPushConstants(commandBuffer, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(push), &push);
        const glm::uvec3 groups = computeDispatch();
        vkCmdDispatch(commandBuffer, std::max(groups.x, 1u), std::max(groups.y, 1u),
                      std::max(groups.z, 1u));
        afterAdvancedCompute(commandBuffer, frameIndex);

        if (chapterUsesSynchronization2()) {
            recordRenderBarriers2(commandBuffer);
        } else {
            std::array<VkBufferMemoryBarrier, STORAGE_BUFFER_COUNT> renderBarriers{};
            for (uint32_t index = 0; index < STORAGE_BUFFER_COUNT; ++index) {
                auto& barrier = renderBarriers[index];
                barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.buffer = storageBuffers_[index];
                barrier.offset = 0;
                barrier.size = storageSizes_[index];
            }
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                     VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                                 0, 0, nullptr, STORAGE_BUFFER_COUNT, renderBarriers.data(), 0, nullptr);
        }
        hasDispatched_ = true;
    }

    void onRecordRender(VkCommandBuffer commandBuffer, uint32_t frameIndex) final {
        currentFrameIndex_ = frameIndex;
        beforeAdvancedDraw(commandBuffer, frameIndex);
        const VkViewport viewport{0.0f, 0.0f, static_cast<float>(extent_.width),
                                  static_cast<float>(extent_.height), 0.0f, 1.0f};
        const VkRect2D scissor{{0, 0}, extent_};
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline_);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_,
                                0, 1, &descriptorSet_, 0, nullptr);

        AdvancedGpuPush push{};
        fillPushConstants(push);
        vkCmdPushConstants(commandBuffer, pipelineLayout_,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(push), &push);
        const int32_t indirectIndex = indirectDrawBufferIndex();
        const int32_t countIndex = indirectDrawCountBufferIndex();
        if (indirectIndex >= 0 && countIndex >= 0) {
            vkCmdDrawIndirectCount(commandBuffer, storageBuffers_[static_cast<uint32_t>(indirectIndex)], 0,
                                   storageBuffers_[static_cast<uint32_t>(countIndex)], 0,
                                   indirectMaxDrawCount(), sizeof(VkDrawIndirectCommand));
        } else if (indirectIndex >= 0) {
            vkCmdDrawIndirect(commandBuffer, storageBuffers_[static_cast<uint32_t>(indirectIndex)],
                              0, 1, sizeof(VkDrawIndirectCommand));
        } else {
            vkCmdDraw(commandBuffer, drawVertexCount(), drawInstanceCount(), 0, 0);
        }
        ++frameNumber_;
        afterAdvancedDraw(commandBuffer, frameIndex);
    }

    void onShutdown() final {
        onAdvancedShutdown();
        vkDestroyPipeline(device_, computePipeline_, nullptr);
        vkDestroyPipeline(device_, graphicsPipeline_, nullptr);
        vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
        vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
        for (uint32_t index = 0; index < STORAGE_BUFFER_COUNT; ++index) {
            vkDestroyBuffer(device_, storageBuffers_[index], nullptr);
            vkFreeMemory(device_, storageMemories_[index], nullptr);
        }
    }

    float elapsed_ = 0.0f;
    uint64_t frameNumber_ = 0;
    uint32_t currentFrameIndex_ = 0;

    void fillCameraFrame(AdvancedGpuPush& push, uint32_t firstValue = 1,
                         float verticalFovDegrees = 55.0f) const {
        const glm::vec3 eye = interactive_.camera().eyePosition();
        const glm::vec3 forward = glm::normalize(interactive_.camera().target() - eye);
        glm::vec3 right = glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f));
        if (glm::dot(right, right) < 1.0e-6f)
            right = glm::vec3(1.0f, 0.0f, 0.0f);
        else
            right = glm::normalize(right);
        const glm::vec3 up = glm::normalize(glm::cross(right, forward));
        push.values[firstValue + 0] = glm::vec4(eye, 1.0f);
        push.values[firstValue + 1] =
            glm::vec4(forward, std::tan(glm::radians(verticalFovDegrees) * 0.5f));
        push.values[firstValue + 2] = glm::vec4(right, 0.0f);
        push.values[firstValue + 3] = glm::vec4(up, 0.0f);
    }

  private:
    std::array<VkBuffer, STORAGE_BUFFER_COUNT> storageBuffers_{};
    std::array<VkDeviceMemory, STORAGE_BUFFER_COUNT> storageMemories_{};
    std::array<VkDeviceSize, STORAGE_BUFFER_COUNT> storageSizes_{};
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline_ = VK_NULL_HANDLE;
    VkPipeline computePipeline_ = VK_NULL_HANDLE;
    bool hasDispatched_ = false;

    void recordReuseBarriers2(VkCommandBuffer commandBuffer) {
        std::array<VkBufferMemoryBarrier2, STORAGE_BUFFER_COUNT> barriers{};
        for (uint32_t index = 0; index < STORAGE_BUFFER_COUNT; ++index) {
            auto& barrier = barriers[index];
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            barrier.srcStageMask = hasDispatched_ ?
                (VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT) :
                VK_PIPELINE_STAGE_2_HOST_BIT;
            barrier.srcAccessMask = hasDispatched_ ?
                (VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT |
                 VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT) : VK_ACCESS_2_HOST_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = storageBuffers_[index];
            barrier.offset = 0;
            barrier.size = storageSizes_[index];
        }
        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.bufferMemoryBarrierCount = STORAGE_BUFFER_COUNT;
        dependency.pBufferMemoryBarriers = barriers.data();
        vkCmdPipelineBarrier2(commandBuffer, &dependency);
    }

    void recordRenderBarriers2(VkCommandBuffer commandBuffer) {
        std::array<VkBufferMemoryBarrier2, STORAGE_BUFFER_COUNT> barriers{};
        for (uint32_t index = 0; index < STORAGE_BUFFER_COUNT; ++index) {
            auto& barrier = barriers[index];
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                   VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = storageBuffers_[index];
            barrier.offset = 0;
            barrier.size = storageSizes_[index];
        }
        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.bufferMemoryBarrierCount = STORAGE_BUFFER_COUNT;
        dependency.pBufferMemoryBarriers = barriers.data();
        vkCmdPipelineBarrier2(commandBuffer, &dependency);
    }

    void createStorageBuffers() {
        storageSizes_ = storageBufferSizes();
        const auto extraUsage = extraBufferUsage();
        for (uint32_t index = 0; index < STORAGE_BUFFER_COUNT; ++index) {
            storageSizes_[index] = std::max<VkDeviceSize>(storageSizes_[index], 16);
            createBuffer(physDev_, device_, storageSizes_[index],
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | extraUsage[index],
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         storageBuffers_[index], storageMemories_[index]);
            void* mapped = nullptr;
            VK_CHECK(vkMapMemory(device_, storageMemories_[index], 0, storageSizes_[index], 0, &mapped));
            std::memset(mapped, 0, static_cast<size_t>(storageSizes_[index]));
            initializeStorageBuffer(index, mapped, storageSizes_[index]);
            vkUnmapMemory(device_, storageMemories_[index]);
        }
    }

    void createDescriptorResources() {
        std::array<VkDescriptorSetLayoutBinding, STORAGE_BUFFER_COUNT> bindings{};
        for (uint32_t index = 0; index < STORAGE_BUFFER_COUNT; ++index) {
            bindings[index].binding = index;
            bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[index].descriptorCount = 1;
            bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT |
                                         VK_SHADER_STAGE_VERTEX_BIT |
                                         VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = STORAGE_BUFFER_COUNT;
        layoutInfo.pBindings = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &descriptorSetLayout_));

        const VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, STORAGE_BUFFER_COUNT};
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

        std::array<VkDescriptorBufferInfo, STORAGE_BUFFER_COUNT> bufferInfos{};
        std::array<VkWriteDescriptorSet, STORAGE_BUFFER_COUNT> writes{};
        for (uint32_t index = 0; index < STORAGE_BUFFER_COUNT; ++index) {
            bufferInfos[index] = {storageBuffers_[index], 0, storageSizes_[index]};
            writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[index].dstSet = descriptorSet_;
            writes[index].dstBinding = index;
            writes[index].descriptorCount = 1;
            writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[index].pBufferInfo = &bufferInfos[index];
        }
        vkUpdateDescriptorSets(device_, STORAGE_BUFFER_COUNT, writes.data(), 0, nullptr);
    }

    void createPipelineLayout() {
        const VkPushConstantRange pushRange{
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(AdvancedGpuPush)};
        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &descriptorSetLayout_;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushRange;
        VK_CHECK(vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &pipelineLayout_));
    }

    void createGraphicsPipeline() {
        const VkShaderModule vertexModule = createShaderModuleFromFile(device_, vertexShaderName());
        const VkShaderModule fragmentModule = createShaderModuleFromFile(device_, fragmentShaderName());
        const std::array<VkPipelineShaderStageCreateInfo, 2> stages{{
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
             VK_SHADER_STAGE_VERTEX_BIT, vertexModule, "main"},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
             VK_SHADER_STAGE_FRAGMENT_BIT, fragmentModule, "main"},
        }};
        const VkPipelineVertexInputStateCreateInfo vertexInput{
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo assembly{
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        assembly.topology = primitiveTopology();
        VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport.viewportCount = 1;
        viewport.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depth.depthTestEnable = chapterNeedsDepth() ? VK_TRUE : VK_FALSE;
        depth.depthWriteEnable = chapterNeedsDepth() ? VK_TRUE : VK_FALSE;
        depth.depthCompareOp = VK_COMPARE_OP_LESS;
        VkPipelineColorBlendAttachmentState colorAttachment{};
        colorAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        if (chapterUsesAlphaBlend()) {
            colorAttachment.blendEnable = VK_TRUE;
            colorAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            colorAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            colorAttachment.colorBlendOp = VK_BLEND_OP_ADD;
            colorAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            colorAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            colorAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
        }
        VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blend.attachmentCount = 1;
        blend.pAttachments = &colorAttachment;
        const std::array<VkDynamicState, 2> dynamicStates{
            VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamic.pDynamicStates = dynamicStates.data();

        VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
        pipelineInfo.pStages = stages.data();
        pipelineInfo.pVertexInputState = &vertexInput;
        pipelineInfo.pInputAssemblyState = &assembly;
        pipelineInfo.pViewportState = &viewport;
        pipelineInfo.pRasterizationState = &raster;
        pipelineInfo.pMultisampleState = &multisample;
        pipelineInfo.pDepthStencilState = &depth;
        pipelineInfo.pColorBlendState = &blend;
        pipelineInfo.pDynamicState = &dynamic;
        pipelineInfo.layout = pipelineLayout_;
        pipelineInfo.renderPass = renderPass_;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo,
                                           nullptr, &graphicsPipeline_));
        vkDestroyShaderModule(device_, fragmentModule, nullptr);
        vkDestroyShaderModule(device_, vertexModule, nullptr);
    }

    void createComputePipeline() {
        const VkShaderModule module = createShaderModuleFromFile(device_, computeShaderName());
        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                              VK_SHADER_STAGE_COMPUTE_BIT, module, "main"};
        pipelineInfo.layout = pipelineLayout_;
        VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo,
                                          nullptr, &computePipeline_));
        vkDestroyShaderModule(device_, module, nullptr);
    }
};
