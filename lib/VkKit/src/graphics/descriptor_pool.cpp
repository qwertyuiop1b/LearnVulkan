#include <graphics/descriptor_pool.hpp>

#include "descriptor_pool_state.hpp"

#include <set>
#include <stdexcept>

namespace vulkan_graphics {

DescriptorPool::DescriptorPool(const VulkanContext& context, const DescriptorPoolCreateInfo& createInfo) {
    if (static_cast<VkDevice>(context.device()) == VK_NULL_HANDLE)
        throw std::invalid_argument("DescriptorPool requires an initialized VulkanContext");
    if (createInfo.maxSets == 0)
        throw std::invalid_argument("DescriptorPool max set count must be greater than zero");
    if (createInfo.sizes.empty())
        throw std::invalid_argument("DescriptorPool requires at least one descriptor pool size");

    std::vector<VkDescriptorPoolSize> nativeSizes;
    nativeSizes.reserve(createInfo.sizes.size());
    std::set<VkDescriptorType> descriptorTypes;
    for (const DescriptorPoolSize& size : createInfo.sizes) {
        if (size.descriptorCount == 0)
            throw std::invalid_argument("DescriptorPool descriptor counts must be greater than zero");
        const VkDescriptorType descriptorType = static_cast<VkDescriptorType>(size.descriptorType);
        if (!descriptorTypes.insert(descriptorType).second)
            throw std::invalid_argument("DescriptorPool descriptor types must be unique");
        nativeSizes.push_back({descriptorType, size.descriptorCount});
    }

    auto state = std::make_shared<DescriptorPoolState>();
    state->context = &context;

    VkDescriptorPoolCreateInfo nativeCreateInfo{};
    nativeCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    nativeCreateInfo.flags = static_cast<VkDescriptorPoolCreateFlags>(createInfo.flags) |
                             VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    nativeCreateInfo.maxSets = createInfo.maxSets;
    nativeCreateInfo.poolSizeCount = static_cast<uint32_t>(nativeSizes.size());
    nativeCreateInfo.pPoolSizes = nativeSizes.data();
    if (vkCreateDescriptorPool(static_cast<VkDevice>(context.device()), &nativeCreateInfo, nullptr, &state->pool) !=
        VK_SUCCESS)
        throw std::runtime_error("Failed to create Vulkan descriptor pool");

    state_ = std::move(state);
}

bool DescriptorPool::isValid() const noexcept {
    return state_ && state_->pool != VK_NULL_HANDLE;
}

vk::DescriptorPool DescriptorPool::handle() const noexcept {
    return vk::DescriptorPool{nativeHandle()};
}

VkDescriptorPool DescriptorPool::nativeHandle() const noexcept {
    return state_ ? state_->pool : VK_NULL_HANDLE;
}

DescriptorSet DescriptorPool::allocate(const DescriptorSetLayout& layout) const {
    if (!isValid())
        throw std::logic_error("DescriptorPool is not valid");
    if (!layout.isValid())
        throw std::invalid_argument("Descriptor set allocation requires a valid layout");

    const VkDescriptorSetLayout nativeLayout = layout.nativeHandle();
    VkDescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = state_->pool;
    allocateInfo.descriptorSetCount = 1;
    allocateInfo.pSetLayouts = &nativeLayout;

    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(static_cast<VkDevice>(state_->context->device()), &allocateInfo, &descriptorSet) !=
        VK_SUCCESS)
        throw std::runtime_error("Failed to allocate Vulkan descriptor set");

    return DescriptorSet{state_, descriptorSet, layout.bindings()};
}

} // namespace vulkan_graphics
