#include <graphics/pipeline_layout.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace vulkan_graphics {
namespace {

bool rangesOverlap(const PushConstantRange& left, const PushConstantRange& right) {
    const uint64_t leftEnd = static_cast<uint64_t>(left.offset) + left.size;
    const uint64_t rightEnd = static_cast<uint64_t>(right.offset) + right.size;
    return left.offset < rightEnd && right.offset < leftEnd;
}

} // namespace

PipelineLayout::PipelineLayout(const VulkanContext& context, const PipelineLayoutCreateInfo& createInfo)
    : context_(&context), pushConstantRanges_(createInfo.pushConstantRanges) {
    if (static_cast<VkDevice>(context.device()) == VK_NULL_HANDLE)
        throw std::invalid_argument("PipelineLayout requires an initialized VulkanContext");
    if (createInfo.setLayouts.size() > context.properties().limits.maxBoundDescriptorSets)
        throw std::invalid_argument("Pipeline layout exceeds the device descriptor set layout limit");

    setLayouts_.reserve(createInfo.setLayouts.size());
    for (const DescriptorSetLayout* layout : createInfo.setLayouts) {
        if (layout == nullptr || !layout->isValid())
            throw std::invalid_argument("Pipeline layout requires valid descriptor set layouts");
        setLayouts_.push_back(layout->nativeHandle());
    }

    const uint32_t maxPushConstantsSize = context.properties().limits.maxPushConstantsSize;
    for (const PushConstantRange& range : pushConstantRanges_) {
        if (range.stageFlags == vk::ShaderStageFlags{})
            throw std::invalid_argument("Push constant shader stages must not be empty");
        if (range.size == 0 || range.offset % 4 != 0 || range.size % 4 != 0)
            throw std::invalid_argument("Push constant ranges require non-zero four-byte aligned offset and size");
        if (range.offset >= maxPushConstantsSize || range.size > maxPushConstantsSize - range.offset)
            throw std::out_of_range("Push constant range exceeds the device limit");
    }

    for (size_t leftIndex = 0; leftIndex < pushConstantRanges_.size(); ++leftIndex) {
        for (size_t rightIndex = leftIndex + 1; rightIndex < pushConstantRanges_.size(); ++rightIndex) {
            const PushConstantRange& left = pushConstantRanges_[leftIndex];
            const PushConstantRange& right = pushConstantRanges_[rightIndex];
            if (rangesOverlap(left, right) && (left.stageFlags & right.stageFlags))
                throw std::invalid_argument("Overlapping push constant ranges must use disjoint shader stages");
        }
    }

    std::vector<VkPushConstantRange> nativePushConstantRanges;
    nativePushConstantRanges.reserve(pushConstantRanges_.size());
    for (const PushConstantRange& range : pushConstantRanges_) {
        nativePushConstantRanges.push_back(
            {static_cast<VkShaderStageFlags>(range.stageFlags), range.offset, range.size});
    }

    VkPipelineLayoutCreateInfo nativeCreateInfo{};
    nativeCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    nativeCreateInfo.flags = static_cast<VkPipelineLayoutCreateFlags>(createInfo.flags);
    nativeCreateInfo.setLayoutCount = static_cast<uint32_t>(setLayouts_.size());
    nativeCreateInfo.pSetLayouts = setLayouts_.data();
    nativeCreateInfo.pushConstantRangeCount = static_cast<uint32_t>(nativePushConstantRanges.size());
    nativeCreateInfo.pPushConstantRanges = nativePushConstantRanges.data();
    if (vkCreatePipelineLayout(static_cast<VkDevice>(context.device()), &nativeCreateInfo, nullptr, &pipelineLayout_) !=
        VK_SUCCESS) {
        context_ = nullptr;
        setLayouts_.clear();
        pushConstantRanges_.clear();
        throw std::runtime_error("Failed to create Vulkan pipeline layout");
    }
}

PipelineLayout::~PipelineLayout() noexcept {
    destroy();
}

PipelineLayout::PipelineLayout(PipelineLayout&& other) noexcept
    : context_(std::exchange(other.context_, nullptr)),
      pipelineLayout_(std::exchange(other.pipelineLayout_, VK_NULL_HANDLE)),
      setLayouts_(std::move(other.setLayouts_)),
      pushConstantRanges_(std::move(other.pushConstantRanges_)) {}

PipelineLayout& PipelineLayout::operator=(PipelineLayout&& other) noexcept {
    if (this == &other)
        return *this;

    destroy();
    context_ = std::exchange(other.context_, nullptr);
    pipelineLayout_ = std::exchange(other.pipelineLayout_, VK_NULL_HANDLE);
    setLayouts_ = std::move(other.setLayouts_);
    pushConstantRanges_ = std::move(other.pushConstantRanges_);
    return *this;
}

bool PipelineLayout::isValid() const noexcept {
    return pipelineLayout_ != VK_NULL_HANDLE;
}

vk::PipelineLayout PipelineLayout::handle() const noexcept {
    return vk::PipelineLayout{pipelineLayout_};
}

VkPipelineLayout PipelineLayout::nativeHandle() const noexcept {
    return pipelineLayout_;
}

const std::vector<VkDescriptorSetLayout>& PipelineLayout::setLayouts() const noexcept {
    return setLayouts_;
}

const std::vector<PushConstantRange>& PipelineLayout::pushConstantRanges() const noexcept {
    return pushConstantRanges_;
}

void PipelineLayout::destroy() noexcept {
    if (pipelineLayout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(static_cast<VkDevice>(context_->device()), pipelineLayout_, nullptr);

    context_ = nullptr;
    pipelineLayout_ = VK_NULL_HANDLE;
    setLayouts_.clear();
    pushConstantRanges_.clear();
}

} // namespace vulkan_graphics
