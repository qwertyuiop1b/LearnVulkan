#pragma once

#include "vk_buffer.h"
#include <span>
#include <vulkan/vulkan_raii.hpp>

namespace vk_engine
{
class VkImage2D
{
public:
    VkImage2D() = default;
    VkImage2D(const VkContext& context, vk::Extent2D extent, std::span<const std::byte> pixels);
    ~VkImage2D() = default;
    VkImage2D(const VkImage2D&) = delete;
    VkImage2D& operator=(const VkImage2D&) = delete;
    VkImage2D(VkImage2D&&) noexcept = default;
    VkImage2D& operator=(VkImage2D&&) noexcept = default;
    vk::ImageView GetImageView() const noexcept
    {
        return *imageView;
    }

private:
    vk::raii::Image image{nullptr};
    vk::raii::DeviceMemory memory{nullptr};
    vk::raii::ImageView imageView{nullptr};
};
} // namespace vk_engine
