#pragma once

#include <graphics/image_state_tracker.hpp>
#include <graphics/vulkan_allocator.hpp>

#include <cstdint>
#include <vector>

namespace vulkan_graphics {

class UploadContext;

struct ImageCreateInfo {
    VkExtent2D extent{};
    vk::Format format = vk::Format::eUndefined;
    vk::ImageUsageFlags usage{};
    vk::ImageAspectFlags aspectMask{};
    uint32_t mipLevels = 1;
    uint32_t arrayLayers = 1;
    vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1;
    vk::ImageTiling tiling = vk::ImageTiling::eOptimal;
    vk::SharingMode sharingMode = vk::SharingMode::eExclusive;
    std::vector<uint32_t> queueFamilyIndices;
};

class Image final {
  public:
    Image() = default;
    Image(VulkanAllocator& allocator, const ImageCreateInfo& createInfo);
    ~Image() noexcept;

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    Image(Image&& other) noexcept;
    Image& operator=(Image&& other) noexcept;

    [[nodiscard]] bool isValid() const noexcept;
    [[nodiscard]] vk::Image handle() const noexcept;
    [[nodiscard]] VkImage nativeHandle() const noexcept;
    [[nodiscard]] vk::ImageView view() const noexcept;
    [[nodiscard]] VkImageView nativeView() const noexcept;
    [[nodiscard]] VkExtent2D extent() const noexcept;
    [[nodiscard]] vk::Format format() const noexcept;
    [[nodiscard]] vk::ImageUsageFlags usage() const noexcept;
    [[nodiscard]] vk::ImageAspectFlags aspectMask() const noexcept;
    [[nodiscard]] uint32_t mipLevels() const noexcept;
    [[nodiscard]] uint32_t arrayLayers() const noexcept;
    [[nodiscard]] vk::ImageLayout layout() const noexcept;
    [[nodiscard]] const ImageSubresourceState& subresourceState(uint32_t mipLevel = 0,
                                                                  uint32_t arrayLayer = 0) const;

  private:
    friend class UploadContext;

    void destroy() noexcept;
    void setState(const ImageSubresourceState& state);

    VulkanAllocator* allocator_ = nullptr;
    VkImage image_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VkImageView imageView_ = VK_NULL_HANDLE;
    VkExtent2D extent_{};
    vk::Format format_ = vk::Format::eUndefined;
    vk::ImageUsageFlags usage_{};
    vk::ImageAspectFlags aspectMask_{};
    uint32_t mipLevels_ = 0;
    uint32_t arrayLayers_ = 0;
    ImageStateTracker stateTracker_;
};

} // namespace vulkan_graphics
