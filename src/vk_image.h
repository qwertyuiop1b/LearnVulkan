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
    Image(const VkContext& context, vk::Extent2D extent, vk::Format format, vk::ImageUsageFlags usage);
    Image(const VkContext& context, vk::Extent2D extent, std::span<const std::byte> pixels);
    ~Image();

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    Image(Image&& other) noexcept;
    Image& operator=(Image&& other) noexcept;

    vk::Image GetHandle() const noexcept
    {
        return image;
    }

    vk::ImageView GetImageView() const noexcept
    {
        return *imageView;
    }

    vk::Extent2D GetExtent() const noexcept
    {
        return extent;
    }

    vk::Format GetFormat() const noexcept
    {
        return format;
    }

private:
    void Create(const VkContext& context, vk::Extent2D extent, vk::Format format, vk::ImageUsageFlags usage);
    void Destroy();
    void ReleaseOwnership() noexcept;

    const VkContext* context{nullptr};
    VkImage image{VK_NULL_HANDLE};
    VmaAllocation allocation{nullptr};
    vk::raii::ImageView imageView{nullptr};
    vk::Extent2D extent{};
    vk::Format format{vk::Format::eUndefined};
};
} // namespace vk_engine
