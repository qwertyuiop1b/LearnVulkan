#include <graphics/memory/image_state_tracker.hpp>

#include <stdexcept>

namespace vulkan_graphics {

ImageStateTracker::ImageStateTracker(uint32_t mipLevels, uint32_t arrayLayers, ImageSubresourceState initialState)
    : mipLevels_(mipLevels), arrayLayers_(arrayLayers) {
    if (mipLevels == 0 || arrayLayers == 0)
        throw std::invalid_argument("ImageStateTracker requires non-zero mip levels and array layers");

    states_.assign(static_cast<size_t>(mipLevels) * arrayLayers, initialState);
}

bool ImageStateTracker::isValid() const noexcept {
    return mipLevels_ != 0 && arrayLayers_ != 0 && states_.size() == static_cast<size_t>(mipLevels_) * arrayLayers_;
}

uint32_t ImageStateTracker::mipLevels() const noexcept {
    return mipLevels_;
}

uint32_t ImageStateTracker::arrayLayers() const noexcept {
    return arrayLayers_;
}

const ImageSubresourceState& ImageStateTracker::state(uint32_t mipLevel, uint32_t arrayLayer) const {
    return states_.at(index(mipLevel, arrayLayer));
}

bool ImageStateTracker::hasUniformLayout(VkImageLayout layout) const noexcept {
    for (const ImageSubresourceState& state : states_) {
        if (state.layout != layout)
            return false;
    }
    return !states_.empty();
}

bool ImageStateTracker::hasUniformState(const ImageSubresourceState& state) const noexcept {
    for (const ImageSubresourceState& currentState : states_) {
        if (currentState.layout != state.layout || currentState.stageMask != state.stageMask ||
            currentState.accessMask != state.accessMask || currentState.queueFamilyIndex != state.queueFamilyIndex) {
            return false;
        }
    }
    return !states_.empty();
}

void ImageStateTracker::setState(uint32_t mipLevel, uint32_t arrayLayer, const ImageSubresourceState& state) {
    states_.at(index(mipLevel, arrayLayer)) = state;
}

void ImageStateTracker::setRange(uint32_t baseMipLevel,
                                 uint32_t levelCount,
                                 uint32_t baseArrayLayer,
                                 uint32_t layerCount,
                                 const ImageSubresourceState& state) {
    validateRange(baseMipLevel, levelCount, baseArrayLayer, layerCount);
    for (uint32_t arrayLayer = baseArrayLayer; arrayLayer < baseArrayLayer + layerCount; ++arrayLayer) {
        for (uint32_t mipLevel = baseMipLevel; mipLevel < baseMipLevel + levelCount; ++mipLevel)
            setState(mipLevel, arrayLayer, state);
    }
}

void ImageStateTracker::reset() noexcept {
    mipLevels_ = 0;
    arrayLayers_ = 0;
    states_.clear();
}

size_t ImageStateTracker::index(uint32_t mipLevel, uint32_t arrayLayer) const {
    if (mipLevel >= mipLevels_ || arrayLayer >= arrayLayers_)
        throw std::out_of_range("Image subresource index is out of range");

    return static_cast<size_t>(arrayLayer) * mipLevels_ + mipLevel;
}

void ImageStateTracker::validateRange(uint32_t baseMipLevel,
                                      uint32_t levelCount,
                                      uint32_t baseArrayLayer,
                                      uint32_t layerCount) const {
    if (levelCount == 0 || layerCount == 0 || baseMipLevel >= mipLevels_ || baseArrayLayer >= arrayLayers_ ||
        levelCount > mipLevels_ - baseMipLevel || layerCount > arrayLayers_ - baseArrayLayer) {
        throw std::out_of_range("Image subresource range is out of bounds");
    }
}

} // namespace vulkan_graphics
