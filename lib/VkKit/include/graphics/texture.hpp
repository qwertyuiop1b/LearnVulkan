#pragma once

#include <graphics/image.hpp>
#include <graphics/sampler.hpp>
#include <graphics/upload_context.hpp>

namespace vulkan_graphics {

struct TextureCreateInfo {
    VkExtent2D extent{};
    const void* pixelData = nullptr;
    VkDeviceSize dataSize = 0;
    vk::Format format = vk::Format::eR8G8B8A8Srgb;
    SamplerCreateInfo sampler{};
};

class Texture final {
  public:
    Texture() = default;
    Texture(VulkanAllocator& allocator, UploadContext& uploads, const TextureCreateInfo& createInfo);
    ~Texture() = default;

    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;
    Texture(Texture&&) noexcept = default;
    Texture& operator=(Texture&&) noexcept = default;

    [[nodiscard]] bool isValid() const noexcept;
    [[nodiscard]] const Image& image() const noexcept;
    [[nodiscard]] const Sampler& sampler() const noexcept;

  private:
    Image image_;
    Sampler sampler_;
};

} // namespace vulkan_graphics
