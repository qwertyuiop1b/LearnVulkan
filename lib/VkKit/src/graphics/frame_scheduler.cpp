#include <graphics/frame_scheduler.hpp>

#include <stdexcept>

namespace vulkan_graphics {

bool FrameBeginResult::isReady() const noexcept {
    return frame != nullptr && renderTarget != nullptr && status != SwapchainStatus::eOutOfDate;
}

FrameScheduler::FrameScheduler(const VulkanContext& context, const FrameSchedulerCreateInfo& createInfo)
    : context_(&context), swapchain_(createInfo.swapchain) {
    if (!context.dynamicRenderingEnabled())
        throw std::invalid_argument("FrameScheduler requires Dynamic Rendering enabled on VulkanContext");
    if (swapchain_ == nullptr || !swapchain_->isValid())
        throw std::invalid_argument("FrameScheduler requires a valid Swapchain");
    if (createInfo.framesInFlight == 0)
        throw std::invalid_argument("FrameScheduler requires at least one frame in flight");

    frames_.reserve(createInfo.framesInFlight);
    for (uint32_t index = 0; index < createInfo.framesInFlight; ++index)
        frames_.emplace_back(new FrameContext(context));

    resetSwapchainTracking();
}

FrameBeginResult FrameScheduler::beginFrame() {
    if (activeFrame_ != nullptr)
        throw std::logic_error("Cannot begin a frame while another frame is in progress");
    ensureSwapchainHasNotChanged();

    FrameContext& frame = *frames_[currentFrameIndex_];
    frame.waitForCompletion();

    const SwapchainAcquireResult acquisition = swapchain_->acquireNextImage(frame.imageAvailableSemaphore());
    if (acquisition.status == SwapchainStatus::eOutOfDate)
        return {acquisition.status};
    if (acquisition.imageIndex >= imageInFlightFences_.size())
        throw std::runtime_error("Swapchain returned an image index outside the tracked range");

    const VkFence imageFence = imageInFlightFences_[acquisition.imageIndex];
    if (imageFence != VK_NULL_HANDLE && imageFence != frame.inFlightFence()) {
        if (vkWaitForFences(static_cast<VkDevice>(context_->device()), 1, &imageFence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
            throw std::runtime_error("Failed while waiting for a Vulkan swapchain image fence");
    }

    frame.beginCommandBuffer();
    currentImageIndex_ = acquisition.imageIndex;
    activeFrame_ = &frame;
    dynamicRenderingActive_ = false;
    return {acquisition.status, currentImageIndex_, activeFrame_, &swapchain_->renderTarget(currentImageIndex_)};
}

void FrameScheduler::beginDynamicRendering(const DynamicRenderingInfo& renderingInfo) {
    if (activeFrame_ == nullptr)
        throw std::logic_error("Cannot begin Dynamic Rendering without an active frame");
    if (dynamicRenderingActive_)
        throw std::logic_error("Dynamic Rendering is already active for the current frame");

    transitionSwapchainImage(vk::ImageLayout::eColorAttachmentOptimal);

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = swapchain_->renderTarget(currentImageIndex_).nativeView();
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = renderingInfo.colorLoadOp;
    colorAttachment.storeOp = renderingInfo.colorStoreOp;
    colorAttachment.clearValue.color = renderingInfo.colorClearValue;

    VkRenderingAttachmentInfo depthAttachment{};
    if (renderingInfo.depthAttachment != VK_NULL_HANDLE) {
        depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depthAttachment.imageView = renderingInfo.depthAttachment;
        depthAttachment.imageLayout = renderingInfo.depthAttachmentLayout;
        depthAttachment.loadOp = renderingInfo.depthLoadOp;
        depthAttachment.storeOp = renderingInfo.depthStoreOp;
        depthAttachment.clearValue.depthStencil = {renderingInfo.depthClearValue, renderingInfo.stencilClearValue};
    }

    VkRenderingAttachmentInfo stencilAttachment{};
    if (renderingInfo.stencilAttachment != VK_NULL_HANDLE) {
        stencilAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        stencilAttachment.imageView = renderingInfo.stencilAttachment;
        stencilAttachment.imageLayout = renderingInfo.stencilAttachmentLayout;
        stencilAttachment.loadOp = renderingInfo.stencilLoadOp;
        stencilAttachment.storeOp = renderingInfo.stencilStoreOp;
        stencilAttachment.clearValue.depthStencil = {renderingInfo.depthClearValue, renderingInfo.stencilClearValue};
    }

    VkRenderingInfo nativeRenderingInfo{};
    nativeRenderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    nativeRenderingInfo.renderArea.offset = {0, 0};
    nativeRenderingInfo.renderArea.extent = swapchain_->extent();
    nativeRenderingInfo.layerCount = 1;
    nativeRenderingInfo.colorAttachmentCount = 1;
    nativeRenderingInfo.pColorAttachments = &colorAttachment;
    nativeRenderingInfo.pDepthAttachment = renderingInfo.depthAttachment == VK_NULL_HANDLE ? nullptr : &depthAttachment;
    nativeRenderingInfo.pStencilAttachment =
        renderingInfo.stencilAttachment == VK_NULL_HANDLE ? nullptr : &stencilAttachment;
    vkCmdBeginRendering(activeFrame_->commandBuffer(), &nativeRenderingInfo);
    dynamicRenderingActive_ = true;
}

void FrameScheduler::endDynamicRendering() {
    if (activeFrame_ == nullptr || !dynamicRenderingActive_)
        throw std::logic_error("Dynamic Rendering is not active for the current frame");

    vkCmdEndRendering(activeFrame_->commandBuffer());
    dynamicRenderingActive_ = false;
}

SwapchainStatus FrameScheduler::endFrame(vk::PipelineStageFlags imageAvailableWaitStage) {
    if (activeFrame_ == nullptr)
        throw std::logic_error("Cannot end a frame when no frame is in progress");
    if (dynamicRenderingActive_)
        throw std::logic_error("End Dynamic Rendering before submitting the frame");
    if (!imageAvailableWaitStage)
        throw std::invalid_argument("Frame submission requires a non-empty image-available wait stage");

    transitionSwapchainImage(vk::ImageLayout::ePresentSrcKHR);
    activeFrame_->endCommandBuffer();
    activeFrame_->resetFence();

    const VkSemaphore imageAvailableSemaphore = activeFrame_->imageAvailableSemaphore();
    const VkSemaphore renderFinishedSemaphore = activeFrame_->renderFinishedSemaphore();
    const VkPipelineStageFlags waitStage = static_cast<VkPipelineStageFlags>(imageAvailableWaitStage);
    const VkCommandBuffer commandBuffer = activeFrame_->commandBuffer();
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &imageAvailableSemaphore;
    submitInfo.pWaitDstStageMask = &waitStage;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &renderFinishedSemaphore;

    if (vkQueueSubmit(context_->graphicsQueue().handle, 1, &submitInfo, activeFrame_->inFlightFence()) != VK_SUCCESS) {
        activeFrame_->recoverFence();
        activeFrame_ = nullptr;
        throw std::runtime_error("Failed to submit Vulkan frame command buffer");
    }

    imageInFlightFences_[currentImageIndex_] = activeFrame_->inFlightFence();
    currentFrameIndex_ = (currentFrameIndex_ + 1) % static_cast<uint32_t>(frames_.size());
    activeFrame_ = nullptr;
    return swapchain_->present(currentImageIndex_, renderFinishedSemaphore);
}

void FrameScheduler::recreateSwapchain(VkExtent2D desiredExtent) {
    if (activeFrame_ != nullptr)
        throw std::logic_error("Cannot recreate the swapchain while a frame is in progress");

    swapchain_->recreate(desiredExtent);
    resetSwapchainTracking();
}

bool FrameScheduler::frameInProgress() const noexcept {
    return activeFrame_ != nullptr;
}

FrameContext& FrameScheduler::currentFrame() {
    if (activeFrame_ == nullptr)
        throw std::logic_error("No frame is currently in progress");

    return *activeFrame_;
}

const FrameContext& FrameScheduler::currentFrame() const {
    if (activeFrame_ == nullptr)
        throw std::logic_error("No frame is currently in progress");

    return *activeFrame_;
}

uint32_t FrameScheduler::currentImageIndex() const {
    if (activeFrame_ == nullptr)
        throw std::logic_error("No frame is currently in progress");

    return currentImageIndex_;
}

void FrameScheduler::transitionSwapchainImage(vk::ImageLayout newLayout) {
    const vk::ImageLayout oldLayout = imageLayouts_[currentImageIndex_];
    if (oldLayout == newLayout)
        return;

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = static_cast<VkImageLayout>(oldLayout);
    barrier.newLayout = static_cast<VkImageLayout>(newLayout);
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = swapchain_->renderTarget(currentImageIndex_).nativeImage();
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkPipelineStageFlags destinationStage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    if (newLayout == vk::ImageLayout::eColorAttachmentOptimal) {
        if (oldLayout != vk::ImageLayout::eUndefined && oldLayout != vk::ImageLayout::ePresentSrcKHR)
            throw std::logic_error("Swapchain image has an unsupported layout before Dynamic Rendering");
        barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        destinationStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    } else if (newLayout == vk::ImageLayout::ePresentSrcKHR) {
        if (oldLayout != vk::ImageLayout::eUndefined && oldLayout != vk::ImageLayout::eColorAttachmentOptimal)
            throw std::logic_error("Swapchain image has an unsupported layout before presentation");
        barrier.srcAccessMask = oldLayout == vk::ImageLayout::eColorAttachmentOptimal ? VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT : 0;
        sourceStage = oldLayout == vk::ImageLayout::eColorAttachmentOptimal ? VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                                                                              : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    } else {
        throw std::invalid_argument("FrameScheduler supports only color-attachment and present swapchain layouts");
    }

    vkCmdPipelineBarrier(activeFrame_->commandBuffer(), sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1,
                         &barrier);
    imageLayouts_[currentImageIndex_] = newLayout;
}

void FrameScheduler::resetSwapchainTracking() noexcept {
    trackedSwapchain_ = swapchain_->nativeHandle();
    imageInFlightFences_.assign(swapchain_->imageCount(), VK_NULL_HANDLE);
    imageLayouts_.assign(swapchain_->imageCount(), vk::ImageLayout::eUndefined);
}

void FrameScheduler::ensureSwapchainHasNotChanged() const {
    if (swapchain_->nativeHandle() != trackedSwapchain_ || swapchain_->imageCount() != imageInFlightFences_.size()) {
        throw std::logic_error("Swapchain changed outside FrameScheduler; use FrameScheduler::recreateSwapchain");
    }
}

} // namespace vulkan_graphics
