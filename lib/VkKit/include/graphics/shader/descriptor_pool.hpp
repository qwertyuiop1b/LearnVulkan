#pragma once

#include <graphics/shader/descriptor_set.hpp>
#include <graphics/shader/descriptor_set_layout.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace vulkan_graphics {

struct DescriptorPoolState;

struct DescriptorPoolSize {
    vk::DescriptorType descriptorType = vk::DescriptorType::eUniformBuffer;
    uint32_t descriptorCount = 0;
};

struct DescriptorPoolCreateInfo {
    uint32_t maxSets = 0;
    std::vector<DescriptorPoolSize> sizes;
    vk::DescriptorPoolCreateFlags flags{};
};

class DescriptorPool final {
  public:
    DescriptorPool() = default;
    DescriptorPool(const VulkanContext& context, const DescriptorPoolCreateInfo& createInfo);
    ~DescriptorPool() = default;

    DescriptorPool(const DescriptorPool&) = delete;
    DescriptorPool& operator=(const DescriptorPool&) = delete;
    DescriptorPool(DescriptorPool&&) noexcept = default;
    DescriptorPool& operator=(DescriptorPool&&) noexcept = default;

    [[nodiscard]] bool isValid() const noexcept;
    [[nodiscard]] vk::DescriptorPool handle() const noexcept;
    [[nodiscard]] VkDescriptorPool nativeHandle() const noexcept;
    [[nodiscard]] DescriptorSet allocate(const DescriptorSetLayout& layout) const;

  private:
    std::shared_ptr<DescriptorPoolState> state_;
};

} // namespace vulkan_graphics
