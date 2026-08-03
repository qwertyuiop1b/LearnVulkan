#include "vk_descriptor.h"
namespace vk_engine
{
VkTextureDescriptor::VkTextureDescriptor(const VkContext& context, const VkTexture& texture)
{
    const vk::DescriptorSetLayoutBinding binding{
        0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment};
    layout =
        vk::raii::DescriptorSetLayout(context.GetDevice(), vk::DescriptorSetLayoutCreateInfo{}.setBindings(binding));
    const vk::DescriptorPoolSize poolSize{vk::DescriptorType::eCombinedImageSampler, 1};
    pool = vk::raii::DescriptorPool(context.GetDevice(),
                                    vk::DescriptorPoolCreateInfo{}
                                        .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet)
                                        .setMaxSets(1)
                                        .setPoolSizes(poolSize));
    sets = vk::raii::DescriptorSets(context.GetDevice(),
                                    vk::DescriptorSetAllocateInfo{}.setDescriptorPool(*pool).setSetLayouts(*layout));
    set = *sets.front();
    const vk::DescriptorImageInfo imageInfo = texture.GetDescriptorInfo();
    context.GetDevice().updateDescriptorSets(vk::WriteDescriptorSet{}
                                                 .setDstSet(set)
                                                 .setDstBinding(0)
                                                 .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
                                                 .setImageInfo(imageInfo),
                                             {});
}
} // namespace vk_engine
