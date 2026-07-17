#pragma once

#include <vulkan/vulkan.hpp>

namespace vulkan_graphics {

class Swapchain;

class RenderTarget final {
  public:
    RenderTarget() = default;

    [[nodiscard]] bool isValid() const noexcept;
    [[nodiscard]] vk::Image image() const noexcept;
    [[nodiscard]] VkImage nativeImage() const noexcept;
    [[nodiscard]] vk::ImageView view() const noexcept;
    [[nodiscard]] VkImageView nativeView() const noexcept;
    [[nodiscard]] VkExtent2D extent() const noexcept;
    [[nodiscard]] vk::Format format() const noexcept;

  private:
    friend class Swapchain;

    VkImage image_ = VK_NULL_HANDLE;
    VkImageView imageView_ = VK_NULL_HANDLE;
    VkExtent2D extent_{};
    vk::Format format_ = vk::Format::eUndefined;
};

} // namespace vulkan_graphics
