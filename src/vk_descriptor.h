#pragma once
#include "vk_texture.h"
namespace vk_engine
{
class VkTextureDescriptor
{
public:
    VkTextureDescriptor(const VkContext& context, const VkTexture& texture);
    vk::DescriptorSetLayout GetLayout() const noexcept
    {
        return *layout;
    }
    vk::DescriptorSet GetSet() const noexcept
    {
        return set;
    }

private:
    vk::raii::DescriptorSetLayout layout{nullptr};
    vk::raii::DescriptorPool pool{nullptr};
    vk::raii::DescriptorSets sets{nullptr};
    vk::DescriptorSet set{nullptr};
};
} // namespace vk_engine
