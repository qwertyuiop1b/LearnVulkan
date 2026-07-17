#pragma once

#include <graphics/command/command_pool.hpp>
#include <graphics/memory/buffer.hpp>
#include <graphics/memory/image.hpp>

#include <functional>
#include <mutex>

namespace vulkan_graphics {

class UploadContext final {
  public:
    explicit UploadContext(const VulkanContext& context);
    ~UploadContext() noexcept;

    UploadContext(const UploadContext&) = delete;
    UploadContext& operator=(const UploadContext&) = delete;
    UploadContext(UploadContext&&) = delete;
    UploadContext& operator=(UploadContext&&) = delete;

    void copyBuffer(const Buffer& source,
                    Buffer& destination,
                    VkDeviceSize bytes = VK_WHOLE_SIZE,
                    VkDeviceSize sourceOffset = 0,
                    VkDeviceSize destinationOffset = 0);
    void transitionImageLayout(Image& image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout);
    void copyBufferToImage(const Buffer& source, Image& destination);

  private:
    void executeAndWait(const std::function<void(VkCommandBuffer)>& record);
    [[nodiscard]] VkCommandBuffer allocateCommandBuffer();
    void submitAndWait(VkCommandBuffer commandBuffer, bool& submitted);
    void freeCommandBuffer(VkCommandBuffer commandBuffer) noexcept;

    const VulkanContext* context_ = nullptr;
    CommandPool commandPool_;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkFence completionFence_ = VK_NULL_HANDLE;
    std::mutex mutex_;
};

} // namespace vulkan_graphics
