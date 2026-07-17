#include <graphics/shader/descriptor_set_layout.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace vulkan_graphics {

DescriptorSetLayout::DescriptorSetLayout(const VulkanContext& context, const DescriptorSetLayoutCreateInfo& createInfo)
    : context_(&context), bindings_(createInfo.bindings) {
    if (static_cast<VkDevice>(context.device()) == VK_NULL_HANDLE)
        throw std::invalid_argument("DescriptorSetLayout requires an initialized VulkanContext");

    std::sort(bindings_.begin(), bindings_.end(), [](const DescriptorBinding& left, const DescriptorBinding& right) {
        return left.binding < right.binding;
    });

    std::vector<VkDescriptorSetLayoutBinding> nativeBindings;
    nativeBindings.reserve(bindings_.size());
    bool hasPreviousBinding = false;
    uint32_t previousBinding = 0;
    for (const DescriptorBinding& binding : bindings_) {
        if (binding.descriptorCount == 0)
            throw std::invalid_argument("Descriptor binding count must be greater than zero");
        if (binding.stageFlags == vk::ShaderStageFlags{})
            throw std::invalid_argument("Descriptor binding shader stages must not be empty");
        if (hasPreviousBinding && previousBinding == binding.binding)
            throw std::invalid_argument("Descriptor bindings must be unique");

        VkDescriptorSetLayoutBinding nativeBinding{};
        nativeBinding.binding = binding.binding;
        nativeBinding.descriptorType = static_cast<VkDescriptorType>(binding.descriptorType);
        nativeBinding.descriptorCount = binding.descriptorCount;
        nativeBinding.stageFlags = static_cast<VkShaderStageFlags>(binding.stageFlags);
        nativeBinding.pImmutableSamplers = nullptr;
        nativeBindings.push_back(nativeBinding);
        previousBinding = binding.binding;
        hasPreviousBinding = true;
    }

    VkDescriptorSetLayoutCreateInfo nativeCreateInfo{};
    nativeCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    nativeCreateInfo.flags = static_cast<VkDescriptorSetLayoutCreateFlags>(createInfo.flags);
    nativeCreateInfo.bindingCount = static_cast<uint32_t>(nativeBindings.size());
    nativeCreateInfo.pBindings = nativeBindings.data();
    if (vkCreateDescriptorSetLayout(static_cast<VkDevice>(context.device()), &nativeCreateInfo, nullptr, &layout_) !=
        VK_SUCCESS) {
        context_ = nullptr;
        bindings_.clear();
        throw std::runtime_error("Failed to create Vulkan descriptor set layout");
    }
}

DescriptorSetLayout::~DescriptorSetLayout() noexcept {
    destroy();
}

DescriptorSetLayout::DescriptorSetLayout(DescriptorSetLayout&& other) noexcept
    : context_(std::exchange(other.context_, nullptr)),
      layout_(std::exchange(other.layout_, VK_NULL_HANDLE)),
      bindings_(std::move(other.bindings_)) {}

DescriptorSetLayout& DescriptorSetLayout::operator=(DescriptorSetLayout&& other) noexcept {
    if (this == &other)
        return *this;

    destroy();
    context_ = std::exchange(other.context_, nullptr);
    layout_ = std::exchange(other.layout_, VK_NULL_HANDLE);
    bindings_ = std::move(other.bindings_);
    return *this;
}

bool DescriptorSetLayout::isValid() const noexcept {
    return layout_ != VK_NULL_HANDLE;
}

vk::DescriptorSetLayout DescriptorSetLayout::handle() const noexcept {
    return vk::DescriptorSetLayout{layout_};
}

VkDescriptorSetLayout DescriptorSetLayout::nativeHandle() const noexcept {
    return layout_;
}

const std::vector<DescriptorBinding>& DescriptorSetLayout::bindings() const noexcept {
    return bindings_;
}

void DescriptorSetLayout::destroy() noexcept {
    if (layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(static_cast<VkDevice>(context_->device()), layout_, nullptr);

    context_ = nullptr;
    layout_ = VK_NULL_HANDLE;
    bindings_.clear();
}

} // namespace vulkan_graphics
