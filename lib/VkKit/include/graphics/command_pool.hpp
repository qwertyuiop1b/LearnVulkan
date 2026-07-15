#pragma once

#include <graphics/vulkan_context.hpp>

#include <cstdint>

namespace vulkan_graphics {

class CommandPool final {
  public:
    CommandPool() = default;
    CommandPool(const VulkanContext& context,
                uint32_t queueFamilyIndex,
                VkCommandPoolCreateFlags flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
    ~CommandPool() noexcept;

    CommandPool(const CommandPool&) = delete;
    CommandPool& operator=(const CommandPool&) = delete;
    CommandPool(CommandPool&& other) noexcept;
    CommandPool& operator=(CommandPool&& other) noexcept;

    [[nodiscard]] bool isValid() const noexcept;
    [[nodiscard]] vk::CommandPool handle() const noexcept;
    [[nodiscard]] VkCommandPool nativeHandle() const noexcept;
    [[nodiscard]] uint32_t queueFamilyIndex() const noexcept;

  private:
    void destroy() noexcept;

    const VulkanContext* context_ = nullptr;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex_ = VK_QUEUE_FAMILY_IGNORED;
};

} // namespace vulkan_graphics
