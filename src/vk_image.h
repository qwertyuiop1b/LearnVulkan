#pragma once

#include "vk_buffer.h"

#include <span>

#include <vulkan/vulkan_raii.hpp>

namespace vk_engine
{
class Image
{
public:
    Image() = default;
    Image(const VkContext& context, vk::Extent2D extent, std::span<const std::byte> pixels);
    ~Image();

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    Image(Image&& other) noexcept;
    Image& operator=(Image&& other) noexcept;

    vk::ImageView GetImageView() const noexcept
    {
        return *imageView;
    }

private:
    void Destroy();

    const VkContext* context{nullptr};
    VkImage image{VK_NULL_HANDLE};
    VmaAllocation allocation{nullptr};
    vk::raii::ImageView imageView{nullptr};
};
} // namespace vk_engine
