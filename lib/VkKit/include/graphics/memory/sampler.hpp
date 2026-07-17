#pragma once

#include <graphics/core/vulkan_context.hpp>

namespace vulkan_graphics {

struct SamplerCreateInfo {
    vk::Filter magFilter = vk::Filter::eLinear;
    vk::Filter minFilter = vk::Filter::eLinear;
    vk::SamplerMipmapMode mipmapMode = vk::SamplerMipmapMode::eLinear;
    vk::SamplerAddressMode addressModeU = vk::SamplerAddressMode::eRepeat;
    vk::SamplerAddressMode addressModeV = vk::SamplerAddressMode::eRepeat;
    vk::SamplerAddressMode addressModeW = vk::SamplerAddressMode::eRepeat;
    float mipLodBias = 0.0f;
    float minLod = 0.0f;
    float maxLod = VK_LOD_CLAMP_NONE;
    bool enableAnisotropy = false;
    float maxAnisotropy = 1.0f;
    bool enableCompare = false;
    vk::CompareOp compareOp = vk::CompareOp::eAlways;
    vk::BorderColor borderColor = vk::BorderColor::eIntOpaqueBlack;
    bool unnormalizedCoordinates = false;
};

class Sampler final {
  public:
    Sampler() = default;
    Sampler(const VulkanContext& context, const SamplerCreateInfo& createInfo = {});
    ~Sampler() noexcept;

    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;
    Sampler(Sampler&& other) noexcept;
    Sampler& operator=(Sampler&& other) noexcept;

    [[nodiscard]] bool isValid() const noexcept;
    [[nodiscard]] vk::Sampler handle() const noexcept;
    [[nodiscard]] VkSampler nativeHandle() const noexcept;

  private:
    void destroy() noexcept;

    const VulkanContext* context_ = nullptr;
    VkSampler sampler_ = VK_NULL_HANDLE;
};

} // namespace vulkan_graphics
