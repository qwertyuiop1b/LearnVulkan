#pragma once

#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vulkan_graphics {

struct ImageSubresourceState {
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkPipelineStageFlags2 stageMask = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 accessMask = VK_ACCESS_2_NONE;
    uint32_t queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
};

class ImageStateTracker final {
  public:
    ImageStateTracker() = default;
    ImageStateTracker(uint32_t mipLevels, uint32_t arrayLayers, ImageSubresourceState initialState = {});

    [[nodiscard]] bool isValid() const noexcept;
    [[nodiscard]] uint32_t mipLevels() const noexcept;
    [[nodiscard]] uint32_t arrayLayers() const noexcept;
    [[nodiscard]] const ImageSubresourceState& state(uint32_t mipLevel = 0, uint32_t arrayLayer = 0) const;
    [[nodiscard]] bool hasUniformLayout(VkImageLayout layout) const noexcept;
    [[nodiscard]] bool hasUniformState(const ImageSubresourceState& state) const noexcept;

    void setState(uint32_t mipLevel, uint32_t arrayLayer, const ImageSubresourceState& state);
    void setRange(uint32_t baseMipLevel,
                  uint32_t levelCount,
                  uint32_t baseArrayLayer,
                  uint32_t layerCount,
                  const ImageSubresourceState& state);
    void reset() noexcept;

  private:
    [[nodiscard]] size_t index(uint32_t mipLevel, uint32_t arrayLayer) const;
    void validateRange(uint32_t baseMipLevel, uint32_t levelCount, uint32_t baseArrayLayer, uint32_t layerCount) const;

    uint32_t mipLevels_ = 0;
    uint32_t arrayLayers_ = 0;
    std::vector<ImageSubresourceState> states_;
};

} // namespace vulkan_graphics
