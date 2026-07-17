#pragma once

#include <graphics/core/vulkan_context.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace vulkan_graphics {

class ShaderModule final {
  public:
    ShaderModule() = default;
    ShaderModule(const VulkanContext& context, const uint32_t* spirvWords, size_t wordCount);
    ShaderModule(const VulkanContext& context, const std::vector<uint32_t>& spirvWords);
    ~ShaderModule() noexcept;

    ShaderModule(const ShaderModule&) = delete;
    ShaderModule& operator=(const ShaderModule&) = delete;
    ShaderModule(ShaderModule&& other) noexcept;
    ShaderModule& operator=(ShaderModule&& other) noexcept;

    [[nodiscard]] static ShaderModule fromFile(const VulkanContext& context, const std::filesystem::path& path);
    [[nodiscard]] bool isValid() const noexcept;
    [[nodiscard]] vk::ShaderModule handle() const noexcept;
    [[nodiscard]] VkShaderModule nativeHandle() const noexcept;

  private:
    void destroy() noexcept;

    const VulkanContext* context_ = nullptr;
    VkShaderModule shaderModule_ = VK_NULL_HANDLE;
};

} // namespace vulkan_graphics
