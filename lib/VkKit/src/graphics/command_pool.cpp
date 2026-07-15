#include <graphics/command_pool.hpp>

#include <stdexcept>
#include <utility>

namespace vulkan_graphics {

CommandPool::CommandPool(const VulkanContext& context, uint32_t queueFamilyIndex, VkCommandPoolCreateFlags flags)
    : context_(&context), queueFamilyIndex_(queueFamilyIndex) {
    if (static_cast<VkDevice>(context.device()) == VK_NULL_HANDLE)
        throw std::invalid_argument("CommandPool requires an initialized VulkanContext");
    if (queueFamilyIndex == VK_QUEUE_FAMILY_IGNORED)
        throw std::invalid_argument("CommandPool requires a valid queue family index");

    VkCommandPoolCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    createInfo.flags = flags;
    createInfo.queueFamilyIndex = queueFamilyIndex;

    if (vkCreateCommandPool(static_cast<VkDevice>(context.device()), &createInfo, nullptr, &commandPool_) != VK_SUCCESS) {
        context_ = nullptr;
        queueFamilyIndex_ = VK_QUEUE_FAMILY_IGNORED;
        throw std::runtime_error("Failed to create Vulkan command pool");
    }
}

CommandPool::~CommandPool() noexcept {
    destroy();
}

CommandPool::CommandPool(CommandPool&& other) noexcept
    : context_(std::exchange(other.context_, nullptr)),
      commandPool_(std::exchange(other.commandPool_, VK_NULL_HANDLE)),
      queueFamilyIndex_(std::exchange(other.queueFamilyIndex_, VK_QUEUE_FAMILY_IGNORED)) {}

CommandPool& CommandPool::operator=(CommandPool&& other) noexcept {
    if (this == &other)
        return *this;

    destroy();
    context_ = std::exchange(other.context_, nullptr);
    commandPool_ = std::exchange(other.commandPool_, VK_NULL_HANDLE);
    queueFamilyIndex_ = std::exchange(other.queueFamilyIndex_, VK_QUEUE_FAMILY_IGNORED);
    return *this;
}

bool CommandPool::isValid() const noexcept {
    return commandPool_ != VK_NULL_HANDLE;
}

vk::CommandPool CommandPool::handle() const noexcept {
    return vk::CommandPool{commandPool_};
}

VkCommandPool CommandPool::nativeHandle() const noexcept {
    return commandPool_;
}

uint32_t CommandPool::queueFamilyIndex() const noexcept {
    return queueFamilyIndex_;
}

void CommandPool::destroy() noexcept {
    if (commandPool_ != VK_NULL_HANDLE)
        vkDestroyCommandPool(static_cast<VkDevice>(context_->device()), commandPool_, nullptr);

    context_ = nullptr;
    commandPool_ = VK_NULL_HANDLE;
    queueFamilyIndex_ = VK_QUEUE_FAMILY_IGNORED;
}

} // namespace vulkan_graphics
