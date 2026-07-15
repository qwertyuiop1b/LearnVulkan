#pragma once

#include <graphics/descriptor_set_layout.hpp>

#include <cstdint>
#include <vector>

namespace vulkan_graphics {

struct PushConstantRange {
    vk::ShaderStageFlags stageFlags{};
    uint32_t offset = 0;
    uint32_t size = 0;
};

struct PipelineLayoutCreateInfo {
    std::vector<const DescriptorSetLayout*> setLayouts;
    std::vector<PushConstantRange> pushConstantRanges;
    vk::PipelineLayoutCreateFlags flags{};
};

class PipelineLayout final {
  public:
    PipelineLayout() = default;
    PipelineLayout(const VulkanContext& context, const PipelineLayoutCreateInfo& createInfo);
    ~PipelineLayout() noexcept;

    PipelineLayout(const PipelineLayout&) = delete;
    PipelineLayout& operator=(const PipelineLayout&) = delete;
    PipelineLayout(PipelineLayout&& other) noexcept;
    PipelineLayout& operator=(PipelineLayout&& other) noexcept;

    [[nodiscard]] bool isValid() const noexcept;
    [[nodiscard]] vk::PipelineLayout handle() const noexcept;
    [[nodiscard]] VkPipelineLayout nativeHandle() const noexcept;
    [[nodiscard]] const std::vector<VkDescriptorSetLayout>& setLayouts() const noexcept;
    [[nodiscard]] const std::vector<PushConstantRange>& pushConstantRanges() const noexcept;

  private:
    void destroy() noexcept;

    const VulkanContext* context_ = nullptr;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSetLayout> setLayouts_;
    std::vector<PushConstantRange> pushConstantRanges_;
};

} // namespace vulkan_graphics
