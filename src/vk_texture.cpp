#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include "vk_texture.h"
#include <stdexcept>
#include <span>
namespace vk_engine
{
VkTexture::VkTexture(const VkContext& inContext, const std::filesystem::path& path) : context(inContext)
{
    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc* loaded = stbi_load(path.string().c_str(), &width, &height, &channels, STBI_rgb_alpha);
    if (loaded == nullptr)
        throw std::runtime_error("failed to load texture: " + path.string());
    const std::size_t byteCount = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4;
    const std::span<const std::byte> pixels{reinterpret_cast<const std::byte*>(loaded), byteCount};
    image = VkImage2D(context, vk::Extent2D{static_cast<uint32_t>(width), static_cast<uint32_t>(height)}, pixels);
    stbi_image_free(loaded);
    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo.setMagFilter(vk::Filter::eLinear)
        .setMinFilter(vk::Filter::eLinear)
        .setMipmapMode(vk::SamplerMipmapMode::eNearest)
        .setAddressModeU(vk::SamplerAddressMode::eRepeat)
        .setAddressModeV(vk::SamplerAddressMode::eRepeat)
        .setAddressModeW(vk::SamplerAddressMode::eRepeat)
        .setMaxLod(0.0F);
    sampler = vk::raii::Sampler(context.GetDevice(), samplerInfo);
}
vk::DescriptorImageInfo VkTexture::GetDescriptorInfo() const noexcept
{
    return vk::DescriptorImageInfo{*sampler, image.GetImageView(), vk::ImageLayout::eShaderReadOnlyOptimal};
}
} // namespace vk_engine
