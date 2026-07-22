#pragma once

#include <graphics/command/frame_context.hpp>
#include <graphics/memory/image_state_tracker.hpp>
#include <graphics/render/swapchain.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace vulkan_graphics {

struct FrameSchedulerCreateInfo {
    Swapchain* swapchain = nullptr;
    uint32_t framesInFlight = 2;
};

struct FrameBeginResult {
    SwapchainStatus status = SwapchainStatus::eOutOfDate;
    uint32_t imageIndex = 0;
    FrameContext* frame = nullptr;
    const RenderTarget* renderTarget = nullptr;

    [[nodiscard]] bool isReady() const noexcept;
};

struct DynamicRenderingInfo {
    VkAttachmentLoadOp colorLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    VkAttachmentStoreOp colorStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkClearColorValue colorClearValue = {{0.0f, 0.0f, 0.0f, 1.0f}};
    VkImageView depthAttachment = VK_NULL_HANDLE;
    VkImageLayout depthAttachmentLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    VkAttachmentLoadOp depthLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    VkAttachmentStoreOp depthStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    float depthClearValue = 1.0f;
    uint32_t stencilClearValue = 0;
    VkImageView stencilAttachment = VK_NULL_HANDLE;
    VkImageLayout stencilAttachmentLayout = VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentLoadOp stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    VkAttachmentStoreOp stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
};

class FrameScheduler final {
  public:
    FrameScheduler(const VulkanContext& context, const FrameSchedulerCreateInfo& createInfo);
    ~FrameScheduler() noexcept;

    FrameScheduler(const FrameScheduler&) = delete;
    FrameScheduler& operator=(const FrameScheduler&) = delete;
    FrameScheduler(FrameScheduler&&) = delete;
    FrameScheduler& operator=(FrameScheduler&&) = delete;

    [[nodiscard]] FrameBeginResult beginFrame();
    void beginDynamicRendering(const DynamicRenderingInfo& renderingInfo = {});
    void endDynamicRendering();
    [[nodiscard]] SwapchainStatus endFrame(
        VkPipelineStageFlags2 imageAvailableWaitStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
    void recreateSwapchain(VkExtent2D desiredExtent);

    [[nodiscard]] bool frameInProgress() const noexcept;
    [[nodiscard]] FrameContext& currentFrame();
    [[nodiscard]] const FrameContext& currentFrame() const;
    [[nodiscard]] uint32_t currentImageIndex() const;

  private:
    void transitionSwapchainImage(vk::ImageLayout newLayout);
    void resetSwapchainTracking();
    void ensureSwapchainHasNotChanged() const;
    void destroyImageSemaphores() noexcept;

    // 每张交换链图像对应一组信号量：imageAvailable 用于 acquire，renderFinished 用于 present
    struct ImageSemaphores {
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        VkSemaphore renderFinished = VK_NULL_HANDLE;
    };

    const VulkanContext* context_ = nullptr;
    Swapchain* swapchain_ = nullptr;
    std::vector<std::unique_ptr<FrameContext>> frames_;
    std::vector<VkFence> imageInFlightFences_;
    std::vector<ImageSemaphores> imageSemaphores_;  // 按 imageIndex 索引
    ImageStateTracker imageStateTracker_;
    VkSwapchainKHR trackedSwapchain_ = VK_NULL_HANDLE;
    uint32_t currentFrameIndex_ = 0;
    uint32_t currentImageIndex_ = 0;
    FrameContext* activeFrame_ = nullptr;
    bool dynamicRenderingActive_ = false;
};

} // namespace vulkan_graphics
