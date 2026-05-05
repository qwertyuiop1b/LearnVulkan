#include "first_app.hpp"

#include <sys/types.h>

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>

#include "Qpipeline.hpp"
#include "vulkan/vulkan_core.h"

namespace q_vulkan {
FirstApp::FirstApp() {
    createPipelineLayout();
    createPipeline();
    createCommandBuffer();
}

FirstApp::~FirstApp() { vkDestroyPipelineLayout(device.device(), pipelineLayout, nullptr); }

void FirstApp::createPipelineLayout() {
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};

    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 0;
    pipelineLayoutInfo.pSetLayouts = 0;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;

    if (vkCreatePipelineLayout(device.device(), &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout");
    }
}

void FirstApp::createPipeline() {
    auto pipelineConfig = QPipeline::defaultPipelineConfigInfo(swapchain.width(), swapchain.height());
    pipelineConfig.renderPass = swapchain.getRenderPass();
    pipelineConfig.pipelineLayout = pipelineLayout;
    pipeline =
        std::make_unique<QPipeline>(device, "shaders/simple.vert.spv", "shaders/simple.frag.spv", pipelineConfig);
}

void FirstApp::createCommandBuffer() {
    commmandBuffers.resize(swapchain.imageCount());
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = device.getCommandPool();
    allocInfo.commandBufferCount = static_cast<uint32_t>(commmandBuffers.size());

    if (vkAllocateCommandBuffers(device.device(), &allocInfo, commmandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffers");
    }

    for (int i = 0; i < commmandBuffers.size(); i++) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        if (vkBeginCommandBuffer(commmandBuffers[i], &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording commmand buffer");
        }

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = swapchain.getRenderPass();
        renderPassInfo.framebuffer = swapchain.getFrameBuffer(i);
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapchain.getSwapChainExtent();

        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = {0.1f, 0.1f, 0.1f, 1.0f};
        clearValues[1].depthStencil = {1.0f, 0};
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(commmandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        pipeline->bind(commmandBuffers[i]);
        vkCmdDraw(commmandBuffers[i], 3, 1, 0, 0);

        vkCmdEndRenderPass(commmandBuffers[i]);
        if (vkEndCommandBuffer(commmandBuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to record command buffer");
        }
    }
}

void FirstApp::drawFrame() {
    uint32_t imageIndex;
    auto result = swapchain.acquireNextImage(&imageIndex);

    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("failed to acquire swap chain image");
    }

    result = swapchain.submitCommandBuffers(&commmandBuffers[imageIndex], &imageIndex);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit commmand buffer");
    }
}
}  // namespace q_vulkan