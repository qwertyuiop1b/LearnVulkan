#include <graphics/command/frame_context.hpp>

#include <stdexcept>

namespace vulkan_graphics {

FrameContext::FrameContext(const VulkanContext& context)
    : context_(&context),
      commandPool_(context, context.graphicsQueue().familyIndex, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT) {
    try {
        VkCommandBufferAllocateInfo commandBufferInfo{};
        commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferInfo.commandPool = commandPool_.nativeHandle();
        commandBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferInfo.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(static_cast<VkDevice>(context.device()), &commandBufferInfo, &commandBuffer_) !=
            VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate Vulkan frame command buffer");
        }

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        if (vkCreateSemaphore(static_cast<VkDevice>(context.device()), &semaphoreInfo, nullptr, &imageAvailableSemaphore_) !=
            VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan image-available semaphore");
        }
        if (vkCreateSemaphore(static_cast<VkDevice>(context.device()), &semaphoreInfo, nullptr, &renderFinishedSemaphore_) !=
            VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan render-finished semaphore");
        }

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        if (vkCreateFence(static_cast<VkDevice>(context.device()), &fenceInfo, nullptr, &inFlightFence_) != VK_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan frame fence");
    } catch (...) {
        destroy();
        throw;
    }
}

FrameContext::~FrameContext() noexcept {
    destroy();
}

VkCommandBuffer FrameContext::commandBuffer() const noexcept {
    return commandBuffer_;
}

vk::CommandBuffer FrameContext::commandBufferHandle() const noexcept {
    return vk::CommandBuffer{commandBuffer_};
}

VkSemaphore FrameContext::imageAvailableSemaphore() const noexcept {
    return imageAvailableSemaphore_;
}

VkSemaphore FrameContext::renderFinishedSemaphore() const noexcept {
    return renderFinishedSemaphore_;
}

VkFence FrameContext::inFlightFence() const noexcept {
    return inFlightFence_;
}

void FrameContext::waitForCompletion() const {
    if (vkWaitForFences(static_cast<VkDevice>(context_->device()), 1, &inFlightFence_, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        throw std::runtime_error("Failed while waiting for Vulkan frame fence");
}

void FrameContext::beginCommandBuffer() {
    if (vkResetCommandBuffer(commandBuffer_, 0) != VK_SUCCESS)
        throw std::runtime_error("Failed to reset Vulkan frame command buffer");

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(commandBuffer_, &beginInfo) != VK_SUCCESS)
        throw std::runtime_error("Failed to begin Vulkan frame command buffer");
}

void FrameContext::endCommandBuffer() {
    if (vkEndCommandBuffer(commandBuffer_) != VK_SUCCESS)
        throw std::runtime_error("Failed to end Vulkan frame command buffer");
}

void FrameContext::resetFence() {
    if (vkResetFences(static_cast<VkDevice>(context_->device()), 1, &inFlightFence_) != VK_SUCCESS)
        throw std::runtime_error("Failed to reset Vulkan frame fence");
}

void FrameContext::recoverFence() {
    if (inFlightFence_ != VK_NULL_HANDLE)
        vkDestroyFence(static_cast<VkDevice>(context_->device()), inFlightFence_, nullptr);
    inFlightFence_ = VK_NULL_HANDLE;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (vkCreateFence(static_cast<VkDevice>(context_->device()), &fenceInfo, nullptr, &inFlightFence_) != VK_SUCCESS)
        throw std::runtime_error("Failed to recover Vulkan frame fence after queue submission failure");
}

void FrameContext::destroy() noexcept {
    if (context_ != nullptr) {
        const VkDevice device = static_cast<VkDevice>(context_->device());
        if (inFlightFence_ != VK_NULL_HANDLE)
            vkDestroyFence(device, inFlightFence_, nullptr);
        if (renderFinishedSemaphore_ != VK_NULL_HANDLE)
            vkDestroySemaphore(device, renderFinishedSemaphore_, nullptr);
        if (imageAvailableSemaphore_ != VK_NULL_HANDLE)
            vkDestroySemaphore(device, imageAvailableSemaphore_, nullptr);
    }

    commandBuffer_ = VK_NULL_HANDLE;
    imageAvailableSemaphore_ = VK_NULL_HANDLE;
    renderFinishedSemaphore_ = VK_NULL_HANDLE;
    inFlightFence_ = VK_NULL_HANDLE;
    context_ = nullptr;
}

} // namespace vulkan_graphics
