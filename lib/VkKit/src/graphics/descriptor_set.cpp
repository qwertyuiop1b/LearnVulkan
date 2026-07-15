#include <graphics/descriptor_set.hpp>

#include "descriptor_pool_state.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace vulkan_graphics {
namespace {

bool isBufferDescriptor(vk::DescriptorType type) {
    return type == vk::DescriptorType::eUniformBuffer || type == vk::DescriptorType::eStorageBuffer ||
           type == vk::DescriptorType::eUniformBufferDynamic || type == vk::DescriptorType::eStorageBufferDynamic;
}

} // namespace

DescriptorSet::DescriptorSet(std::shared_ptr<DescriptorPoolState> poolState,
                             VkDescriptorSet descriptorSet,
                             std::vector<DescriptorBinding> bindings)
    : poolState_(std::move(poolState)), descriptorSet_(descriptorSet), bindings_(std::move(bindings)) {}

DescriptorSet::~DescriptorSet() noexcept {
    release();
}

DescriptorSet::DescriptorSet(DescriptorSet&& other) noexcept
    : poolState_(std::move(other.poolState_)),
      descriptorSet_(std::exchange(other.descriptorSet_, VK_NULL_HANDLE)),
      bindings_(std::move(other.bindings_)) {}

DescriptorSet& DescriptorSet::operator=(DescriptorSet&& other) noexcept {
    if (this == &other)
        return *this;

    release();
    poolState_ = std::move(other.poolState_);
    descriptorSet_ = std::exchange(other.descriptorSet_, VK_NULL_HANDLE);
    bindings_ = std::move(other.bindings_);
    return *this;
}

bool DescriptorSet::isValid() const noexcept {
    return poolState_ && descriptorSet_ != VK_NULL_HANDLE;
}

vk::DescriptorSet DescriptorSet::handle() const noexcept {
    return vk::DescriptorSet{descriptorSet_};
}

VkDescriptorSet DescriptorSet::nativeHandle() const noexcept {
    return descriptorSet_;
}

void DescriptorSet::writeBuffer(uint32_t binding,
                                const Buffer& buffer,
                                VkDeviceSize offset,
                                VkDeviceSize range,
                                uint32_t arrayElement) {
    if (!isValid())
        throw std::logic_error("DescriptorSet is not valid");
    if (!buffer.isValid())
        throw std::invalid_argument("Descriptor buffer write requires a valid buffer");

    const DescriptorBinding& bindingInfo = findBinding(binding, arrayElement);
    if (!isBufferDescriptor(bindingInfo.descriptorType))
        throw std::invalid_argument("Descriptor binding is not a buffer descriptor");
    if (offset >= buffer.size())
        throw std::out_of_range("Descriptor buffer offset exceeds buffer size");
    if (range != VK_WHOLE_SIZE && (range == 0 || range > buffer.size() - offset))
        throw std::out_of_range("Descriptor buffer range exceeds buffer size");

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer.nativeHandle();
    bufferInfo.offset = offset;
    bufferInfo.range = range;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSet_;
    write.dstBinding = binding;
    write.dstArrayElement = arrayElement;
    write.descriptorCount = 1;
    write.descriptorType = static_cast<VkDescriptorType>(bindingInfo.descriptorType);
    write.pBufferInfo = &bufferInfo;
    vkUpdateDescriptorSets(static_cast<VkDevice>(poolState_->context->device()), 1, &write, 0, nullptr);
}

void DescriptorSet::writeTexture(uint32_t binding, const Texture& texture, uint32_t arrayElement) {
    if (!isValid())
        throw std::logic_error("DescriptorSet is not valid");
    if (!texture.isValid())
        throw std::invalid_argument("Descriptor texture write requires a valid texture");

    const DescriptorBinding& bindingInfo = findBinding(binding, arrayElement);
    if (bindingInfo.descriptorType != vk::DescriptorType::eCombinedImageSampler)
        throw std::invalid_argument("Descriptor binding is not a combined image sampler");
    if (texture.image().layout() != vk::ImageLayout::eShaderReadOnlyOptimal)
        throw std::logic_error("Texture image must use shader-read-only layout before descriptor update");

    VkDescriptorImageInfo imageInfo{};
    imageInfo.sampler = texture.sampler().nativeHandle();
    imageInfo.imageView = texture.image().nativeView();
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSet_;
    write.dstBinding = binding;
    write.dstArrayElement = arrayElement;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &imageInfo;
    vkUpdateDescriptorSets(static_cast<VkDevice>(poolState_->context->device()), 1, &write, 0, nullptr);
}

const DescriptorBinding& DescriptorSet::findBinding(uint32_t binding, uint32_t arrayElement) const {
    const auto found = std::lower_bound(bindings_.begin(), bindings_.end(), binding, [](const DescriptorBinding& item,
                                                                                          uint32_t bindingNumber) {
        return item.binding < bindingNumber;
    });
    if (found == bindings_.end() || found->binding != binding)
        throw std::out_of_range("Descriptor binding does not exist in the layout");
    if (arrayElement >= found->descriptorCount)
        throw std::out_of_range("Descriptor array element exceeds the binding descriptor count");
    return *found;
}

void DescriptorSet::release() noexcept {
    if (descriptorSet_ != VK_NULL_HANDLE && poolState_ && poolState_->pool != VK_NULL_HANDLE) {
        vkFreeDescriptorSets(static_cast<VkDevice>(poolState_->context->device()), poolState_->pool, 1,
                             &descriptorSet_);
    }

    descriptorSet_ = VK_NULL_HANDLE;
    bindings_.clear();
    poolState_.reset();
}

} // namespace vulkan_graphics
