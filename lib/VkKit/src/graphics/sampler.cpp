#include <graphics/sampler.hpp>

#include <stdexcept>
#include <utility>

namespace vulkan_graphics {

Sampler::Sampler(const VulkanContext& context, const SamplerCreateInfo& createInfo) : context_(&context) {
    if (static_cast<VkDevice>(context.device()) == VK_NULL_HANDLE)
        throw std::invalid_argument("Sampler requires an initialized VulkanContext");
    if (createInfo.minLod > createInfo.maxLod)
        throw std::invalid_argument("Sampler minimum LOD must not exceed maximum LOD");
    if (createInfo.enableAnisotropy) {
        if (!context.samplerAnisotropyEnabled())
            throw std::invalid_argument("Sampler anisotropy was not enabled in VulkanContext");
        if (createInfo.maxAnisotropy < 1.0f || createInfo.maxAnisotropy > context.properties().limits.maxSamplerAnisotropy)
            throw std::out_of_range("Sampler anisotropy exceeds device limits");
    }

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = static_cast<VkFilter>(createInfo.magFilter);
    samplerInfo.minFilter = static_cast<VkFilter>(createInfo.minFilter);
    samplerInfo.mipmapMode = static_cast<VkSamplerMipmapMode>(createInfo.mipmapMode);
    samplerInfo.addressModeU = static_cast<VkSamplerAddressMode>(createInfo.addressModeU);
    samplerInfo.addressModeV = static_cast<VkSamplerAddressMode>(createInfo.addressModeV);
    samplerInfo.addressModeW = static_cast<VkSamplerAddressMode>(createInfo.addressModeW);
    samplerInfo.mipLodBias = createInfo.mipLodBias;
    samplerInfo.anisotropyEnable = createInfo.enableAnisotropy;
    samplerInfo.maxAnisotropy = createInfo.enableAnisotropy ? createInfo.maxAnisotropy : 1.0f;
    samplerInfo.compareEnable = createInfo.enableCompare;
    samplerInfo.compareOp = static_cast<VkCompareOp>(createInfo.compareOp);
    samplerInfo.minLod = createInfo.minLod;
    samplerInfo.maxLod = createInfo.maxLod;
    samplerInfo.borderColor = static_cast<VkBorderColor>(createInfo.borderColor);
    samplerInfo.unnormalizedCoordinates = createInfo.unnormalizedCoordinates;

    if (vkCreateSampler(static_cast<VkDevice>(context.device()), &samplerInfo, nullptr, &sampler_) != VK_SUCCESS) {
        context_ = nullptr;
        throw std::runtime_error("Failed to create Vulkan sampler");
    }
}

Sampler::~Sampler() noexcept {
    destroy();
}

Sampler::Sampler(Sampler&& other) noexcept
    : context_(std::exchange(other.context_, nullptr)), sampler_(std::exchange(other.sampler_, VK_NULL_HANDLE)) {}

Sampler& Sampler::operator=(Sampler&& other) noexcept {
    if (this == &other)
        return *this;

    destroy();
    context_ = std::exchange(other.context_, nullptr);
    sampler_ = std::exchange(other.sampler_, VK_NULL_HANDLE);
    return *this;
}

bool Sampler::isValid() const noexcept {
    return sampler_ != VK_NULL_HANDLE;
}

vk::Sampler Sampler::handle() const noexcept {
    return vk::Sampler{sampler_};
}

VkSampler Sampler::nativeHandle() const noexcept {
    return sampler_;
}

void Sampler::destroy() noexcept {
    if (sampler_ != VK_NULL_HANDLE)
        vkDestroySampler(static_cast<VkDevice>(context_->device()), sampler_, nullptr);

    context_ = nullptr;
    sampler_ = VK_NULL_HANDLE;
}

} // namespace vulkan_graphics
