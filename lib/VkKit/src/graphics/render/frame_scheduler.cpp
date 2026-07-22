#include <graphics/render/frame_scheduler.hpp>

#include <stdexcept>

namespace vulkan_graphics {

bool FrameBeginResult::isReady() const noexcept {
    return frame != nullptr && renderTarget != nullptr && status != SwapchainStatus::eOutOfDate;
}

FrameScheduler::FrameScheduler(const VulkanContext& context, const FrameSchedulerCreateInfo& createInfo)
    : context_(&context), swapchain_(createInfo.swapchain) {
    if (!context.dynamicRenderingEnabled())
        throw std::invalid_argument("FrameScheduler requires Dynamic Rendering enabled on VulkanContext");
    if (!context.synchronization2Enabled())
        throw std::invalid_argument("FrameScheduler requires Synchronization2 enabled on VulkanContext");
    if (swapchain_ == nullptr || !swapchain_->isValid())
        throw std::invalid_argument("FrameScheduler requires a valid Swapchain");
    if (createInfo.framesInFlight == 0)
        throw std::invalid_argument("FrameScheduler requires at least one frame in flight");

    frames_.reserve(createInfo.framesInFlight);
    for (uint32_t index = 0; index < createInfo.framesInFlight; ++index)
        frames_.emplace_back(new FrameContext(context));

    resetSwapchainTracking();
}

FrameScheduler::~FrameScheduler() noexcept {
    destroyImageSemaphores();
}

FrameBeginResult FrameScheduler::beginFrame() {
    if (activeFrame_ != nullptr)
        throw std::logic_error("Cannot begin a frame while another frame is in progress");
    ensureSwapchainHasNotChanged();

    FrameContext& frame = *frames_[currentFrameIndex_];
    frame.waitForCompletion();

    // 用当前帧的 imageAvailable 信号量触发 acquire
    // 这个信号量与 frameIndex 绑定，framesInFlight 个帧循环使用
    // 关键：acquire 完成后立即知道 imageIndex，再用 imageIndex 对应的信号量做后续 submit/present
    const VkSemaphore acquireSemaphore = imageSemaphores_[currentFrameIndex_].imageAvailable;

    const SwapchainAcquireResult acquisition = swapchain_->acquireNextImage(acquireSemaphore);
    if (acquisition.status == SwapchainStatus::eOutOfDate)
        return {acquisition.status};
    if (acquisition.imageIndex >= imageInFlightFences_.size())
        throw std::runtime_error("Swapchain returned an image index outside the tracked range");

    currentImageIndex_ = acquisition.imageIndex;

    // 等待该图像上一次渲染完成（防止 write-after-write）
    const VkFence imageFence = imageInFlightFences_[currentImageIndex_];
    if (imageFence != VK_NULL_HANDLE && imageFence != frame.inFlightFence()) {
        if (vkWaitForFences(static_cast<VkDevice>(context_->device()), 1, &imageFence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
            throw std::runtime_error("Failed while waiting for a Vulkan swapchain image fence");
    }

    frame.beginCommandBuffer();
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

SwapchainStatus FrameScheduler::endFrame(VkPipelineStageFlags2 imageAvailableWaitStage) {
    if (activeFrame_ == nullptr)
        throw std::logic_error("Cannot end a frame when no frame is in progress");
    if (dynamicRenderingActive_)
        throw std::logic_error("End Dynamic Rendering before submitting the frame");
    if (!imageAvailableWaitStage)
        throw std::invalid_argument("Frame submission requires a non-empty image-available wait stage");

    transitionSwapchainImage(vk::ImageLayout::ePresentSrcKHR);
    activeFrame_->endCommandBuffer();
    activeFrame_->resetFence();

    const VkSemaphore imageAvailableSemaphore = imageSemaphores_[currentFrameIndex_].imageAvailable;
    const VkSemaphore renderFinishedSemaphore = imageSemaphores_[currentImageIndex_].renderFinished;
    const VkCommandBuffer commandBuffer = activeFrame_->commandBuffer();
    VkSemaphoreSubmitInfo waitInfo{};
    waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    waitInfo.semaphore = imageAvailableSemaphore;
    waitInfo.stageMask = imageAvailableWaitStage;

    VkCommandBufferSubmitInfo commandBufferInfo{};
    commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    commandBufferInfo.commandBuffer = commandBuffer;

    VkSemaphoreSubmitInfo signalInfo{};
    signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    signalInfo.semaphore = renderFinishedSemaphore;
    signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

    VkSubmitInfo2 submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submitInfo.waitSemaphoreInfoCount = 1;
    submitInfo.pWaitSemaphoreInfos = &waitInfo;
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &commandBufferInfo;
    submitInfo.signalSemaphoreInfoCount = 1;
    submitInfo.pSignalSemaphoreInfos = &signalInfo;

    if (vkQueueSubmit2(context_->graphicsQueue().handle, 1, &submitInfo, activeFrame_->inFlightFence()) != VK_SUCCESS) {
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
    const ImageSubresourceState oldState = imageStateTracker_.state(0, currentImageIndex_);
    const vk::ImageLayout oldLayout = static_cast<vk::ImageLayout>(oldState.layout);
    if (oldLayout == newLayout)
        return;

    ImageSubresourceState newState{};
    newState.layout = static_cast<VkImageLayout>(newLayout);

    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = oldState.stageMask;
    barrier.srcAccessMask = oldState.accessMask;
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

    if (newLayout == vk::ImageLayout::eColorAttachmentOptimal) {
        if (oldLayout != vk::ImageLayout::eUndefined && oldLayout != vk::ImageLayout::ePresentSrcKHR)
            throw std::logic_error("Swapchain image has an unsupported layout before Dynamic Rendering");
        newState.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        newState.accessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    } else if (newLayout == vk::ImageLayout::ePresentSrcKHR) {
        if (oldLayout != vk::ImageLayout::eUndefined && oldLayout != vk::ImageLayout::eColorAttachmentOptimal)
            throw std::logic_error("Swapchain image has an unsupported layout before presentation");
    } else {
        throw std::invalid_argument("FrameScheduler supports only color-attachment and present swapchain layouts");
    }

    barrier.dstStageMask = newState.stageMask;
    barrier.dstAccessMask = newState.accessMask;
    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.imageMemoryBarrierCount = 1;
    dependencyInfo.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(activeFrame_->commandBuffer(), &dependencyInfo);
    imageStateTracker_.setState(0, currentImageIndex_, newState);
}

void FrameScheduler::resetSwapchainTracking() {
    trackedSwapchain_ = swapchain_->nativeHandle();
    imageInFlightFences_.assign(swapchain_->imageCount(), VK_NULL_HANDLE);
    imageStateTracker_ = ImageStateTracker{1, swapchain_->imageCount()};

    // 销毁旧信号量，重建与交换链图像数对应的信号量组
    destroyImageSemaphores();

    const uint32_t imageCount = swapchain_->imageCount();
    imageSemaphores_.resize(imageCount);
    const VkDevice device = static_cast<VkDevice>(context_->device());
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    for (uint32_t i = 0; i < imageCount; ++i) {
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageSemaphores_[i].imageAvailable) != VK_SUCCESS)
            throw std::runtime_error("Failed to create imageAvailable semaphore");
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageSemaphores_[i].renderFinished) != VK_SUCCESS)
            throw std::runtime_error("Failed to create renderFinished semaphore");
    }
}

void FrameScheduler::destroyImageSemaphores() noexcept {
    if (!context_)
        return;
    const VkDevice device = static_cast<VkDevice>(context_->device());
    for (auto& s : imageSemaphores_) {
        if (s.imageAvailable != VK_NULL_HANDLE)
            vkDestroySemaphore(device, s.imageAvailable, nullptr);
        if (s.renderFinished != VK_NULL_HANDLE)
            vkDestroySemaphore(device, s.renderFinished, nullptr);
    }
    imageSemaphores_.clear();
}

void FrameScheduler::ensureSwapchainHasNotChanged() const {
    if (swapchain_->nativeHandle() != trackedSwapchain_ || swapchain_->imageCount() != imageInFlightFences_.size() ||
        imageStateTracker_.arrayLayers() != swapchain_->imageCount()) {
        throw std::logic_error("Swapchain changed outside FrameScheduler; use FrameScheduler::recreateSwapchain");
    }
}

} // namespace vulkan_graphics
