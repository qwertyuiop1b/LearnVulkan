#pragma once

#include <graphics/vulkan_context.hpp>

#include <cstdint>
#include <vector>

namespace vulkan_graphics {

struct DescriptorBinding {
    uint32_t binding = 0;
    vk::DescriptorType descriptorType = vk::DescriptorType::eUniformBuffer;
    uint32_t descriptorCount = 1;
    vk::ShaderStageFlags stageFlags{};
};

struct DescriptorSetLayoutCreateInfo {
    std::vector<DescriptorBinding> bindings;
    vk::DescriptorSetLayoutCreateFlags flags{};
};

class DescriptorSetLayout final {
  public:
    DescriptorSetLayout() = default;
    DescriptorSetLayout(const VulkanContext& context, const DescriptorSetLayoutCreateInfo& createInfo);
    ~DescriptorSetLayout() noexcept;

    DescriptorSetLayout(const DescriptorSetLayout&) = delete;
    DescriptorSetLayout& operator=(const DescriptorSetLayout&) = delete;
    DescriptorSetLayout(DescriptorSetLayout&& other) noexcept;
    DescriptorSetLayout& operator=(DescriptorSetLayout&& other) noexcept;

    [[nodiscard]] bool isValid() const noexcept;
    [[nodiscard]] vk::DescriptorSetLayout handle() const noexcept;
    [[nodiscard]] VkDescriptorSetLayout nativeHandle() const noexcept;
    [[nodiscard]] const std::vector<DescriptorBinding>& bindings() const noexcept;

  private:
    void destroy() noexcept;

    const VulkanContext* context_ = nullptr;
    VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;
    std::vector<DescriptorBinding> bindings_;
};

} // namespace vulkan_graphics
