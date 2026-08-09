#pragma once

#include "vk_context.h"
#include "vk_image.h"
#include "vk_swapchain.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace vk_engine
{
/**
 * @brief Per-frame helper for transitioning and querying the offscreen draw image.
 */
class RenderHelper
{
public:
    static constexpr size_t kFramesInFlight = 2;

    RenderHelper(vk::CommandBuffer commandBuffer,
                 vk::Image image,
                 vk::ImageView imageView,
                 vk::Extent2D extent,
                 vk::Format format,
                 vk::ImageLayout& currentLayout,
                 size_t frameIndex);

    void TransitionToCompute();
    void TransitionToGraphics();

    vk::ImageView GetDrawImageView() const noexcept
    {
        return imageView;
    }

    vk::Extent2D GetDrawExtent() const noexcept
    {
        return extent;
    }

    vk::Format GetDrawImageFormat() const noexcept
    {
        return format;
    }

    size_t GetFrameIndex() const noexcept
    {
        return frameIndex;
    }

private:
    void TransitionTo(vk::ImageLayout newLayout, vk::AccessFlags dstAccess, vk::PipelineStageFlags dstStage);

    vk::CommandBuffer commandBuffer;
    vk::Image image;
    vk::ImageView imageView;
    vk::Extent2D extent;
    vk::Format format;
    vk::ImageLayout& currentLayout;
    size_t frameIndex;
};

class VkFrameContext
{
public:
    VkFrameContext(const vk::raii::Device& device, uint32_t queueFamilyIndex);
    ~VkFrameContext() = default;

    VkFrameContext(const VkFrameContext&) = delete;
    VkFrameContext& operator=(const VkFrameContext&) = delete;
    VkFrameContext(VkFrameContext&&) noexcept = default;
    VkFrameContext& operator=(VkFrameContext&&) noexcept = default;

    // CommandBuffer must be destroyed before the CommandPool declared above it.
    vk::raii::CommandPool commandPool{nullptr};
    vk::raii::CommandBuffer commandBuffer{nullptr};
    vk::raii::Semaphore imageAvailable{nullptr};
    vk::raii::Fence inFlight{nullptr};
};

class VkRenderer
{
public:
    using RenderCallback = std::function<void(vk::CommandBuffer commandBuffer, RenderHelper& helper)>;

    VkRenderer(const VkContext& context, VkSwapchain& swapchain);
    ~VkRenderer() noexcept;

    VkRenderer(const VkRenderer&) = delete;
    VkRenderer& operator=(const VkRenderer&) = delete;
    VkRenderer(VkRenderer&&) = delete;
    VkRenderer& operator=(VkRenderer&&) = delete;

    void DrawFrame(const RenderCallback& callback);
    void WaitIdle() const;

private:
    void CreateDrawImage();
    void RecreateSwapchain();
    static std::vector<vk::raii::Semaphore> CreateRenderFinishedSemaphores(const vk::raii::Device& device,
                                                                           uint32_t imageCount);
    void TransitionImage(vk::CommandBuffer commandBuffer,
                         vk::Image image,
                         vk::ImageLayout oldLayout,
                         vk::ImageLayout newLayout,
                         vk::AccessFlags sourceAccess,
                         vk::AccessFlags destinationAccess,
                         vk::PipelineStageFlags sourceStage,
                         vk::PipelineStageFlags destinationStage) const;
    void BlitDrawImageToSwapchain(vk::CommandBuffer commandBuffer, vk::Image swapchainImage) const;

    const VkContext& context;
    VkSwapchain& swapchain;
    Image drawImage;
    vk::ImageLayout drawImageLayout{vk::ImageLayout::eUndefined};
    std::array<VkFrameContext, RenderHelper::kFramesInFlight> frames;
    std::vector<vk::raii::Semaphore> renderFinished;
    std::vector<vk::Fence> imagesInFlight;
    std::vector<vk::ImageLayout> swapchainImageLayouts;
    size_t currentFrame = 0;
};
} // namespace vk_engine
