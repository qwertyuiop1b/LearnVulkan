#pragma once

#include "vk_context.h"
#include "vk_swapchain.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace vk_engine
{
struct RenderTarget
{
    vk::Image image;
    vk::ImageView imageView;
    vk::Extent2D extent;
    vk::Format colorFormat;
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
    using RecordCallback = std::function<void(vk::CommandBuffer commandBuffer, const RenderTarget& target)>;

    VkRenderer(const VkContext& context, VkSwapchain& swapchain);
    ~VkRenderer() noexcept;

    VkRenderer(const VkRenderer&) = delete;
    VkRenderer& operator=(const VkRenderer&) = delete;
    VkRenderer(VkRenderer&&) = delete;
    VkRenderer& operator=(VkRenderer&&) = delete;

    void DrawFrame(const RecordCallback& record);
    void WaitIdle() const;

private:
    static constexpr size_t kMaxFramesInFlight = 2;

    void RecreateSwapchain();
    static std::vector<vk::raii::Semaphore> CreateRenderFinishedSemaphores(const vk::raii::Device& device,
                                                                           uint32_t imageCount);
    void TransitionToColorAttachment(vk::CommandBuffer commandBuffer, vk::Image image, vk::ImageLayout oldLayout) const;
    void TransitionToPresent(vk::CommandBuffer commandBuffer, vk::Image image) const;

    const VkContext& context;
    VkSwapchain& swapchain;
    std::array<VkFrameContext, kMaxFramesInFlight> frames;
    std::vector<vk::raii::Semaphore> renderFinished;
    std::vector<vk::Fence> imagesInFlight;
    std::vector<vk::ImageLayout> imageLayouts;
    size_t currentFrame = 0;
};
} // namespace vk_engine
