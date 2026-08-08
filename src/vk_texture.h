#pragma once
#include "vk_image.h"
#include <filesystem>
namespace vk_engine
{
class VkTexture
{
public:
    VkTexture(const VkContext& context, const std::filesystem::path& path);
    ~VkTexture() = default;
    VkTexture(const VkTexture&) = delete;
    VkTexture& operator=(const VkTexture&) = delete;
    vk::DescriptorImageInfo GetDescriptorInfo() const noexcept;

private:
    const VkContext& context;
    Image image;
    vk::raii::Sampler sampler{nullptr};
};
} // namespace vk_engine
