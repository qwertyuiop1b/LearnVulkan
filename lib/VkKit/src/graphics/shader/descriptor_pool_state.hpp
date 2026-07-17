#pragma once

#include <graphics/core/vulkan_context.hpp>

namespace vulkan_graphics {

struct DescriptorPoolState {
    const VulkanContext* context = nullptr;
    VkDescriptorPool pool = VK_NULL_HANDLE;

    ~DescriptorPoolState() {
        if (pool != VK_NULL_HANDLE)
            vkDestroyDescriptorPool(static_cast<VkDevice>(context->device()), pool, nullptr);
    }
};

} // namespace vulkan_graphics
