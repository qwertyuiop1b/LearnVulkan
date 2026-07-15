#pragma once

#include <graphics/command_pool.hpp>

namespace vulkan_graphics {

class FrameScheduler;

class FrameContext final {
  public:
    ~FrameContext() noexcept;

    FrameContext(const FrameContext&) = delete;
    FrameContext& operator=(const FrameContext&) = delete;
    FrameContext(FrameContext&&) = delete;
    FrameContext& operator=(FrameContext&&) = delete;

    [[nodiscard]] VkCommandBuffer commandBuffer() const noexcept;
    [[nodiscard]] vk::CommandBuffer commandBufferHandle() const noexcept;
    [[nodiscard]] VkSemaphore imageAvailableSemaphore() const noexcept;
    [[nodiscard]] VkSemaphore renderFinishedSemaphore() const noexcept;
    [[nodiscard]] VkFence inFlightFence() const noexcept;

  private:
    friend class FrameScheduler;

    explicit FrameContext(const VulkanContext& context);

    void waitForCompletion() const;
    void beginCommandBuffer();
    void endCommandBuffer();
    void resetFence();
    void recoverFence();
    void destroy() noexcept;

    const VulkanContext* context_ = nullptr;
    CommandPool commandPool_;
    VkCommandBuffer commandBuffer_ = VK_NULL_HANDLE;
    VkSemaphore imageAvailableSemaphore_ = VK_NULL_HANDLE;
    VkSemaphore renderFinishedSemaphore_ = VK_NULL_HANDLE;
    VkFence inFlightFence_ = VK_NULL_HANDLE;
};

} // namespace vulkan_graphics
