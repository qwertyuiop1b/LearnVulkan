#pragma once

#include <utility>
#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct GLFWwindow;

namespace vulkan_graphics {

struct VulkanContextCreateInfo {
    GLFWwindow* window = nullptr;
    std::string applicationName = "VulkanGraphics";
    uint32_t applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    uint32_t apiVersion = VK_API_VERSION_1_3;
    bool enableValidation = true;
    bool preferDiscreteGpu = true;
    bool requireDynamicRendering = true;
    bool requireSynchronization2 = true;
    bool requireSamplerAnisotropy = false;
    std::vector<const char*> requiredDeviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
};

struct Queue {
    vk::Queue handle{};
    uint32_t familyIndex = VK_QUEUE_FAMILY_IGNORED;
    bool supportsTimestamps = false;
};

class VulkanContext final {
  public:
    explicit VulkanContext(const VulkanContextCreateInfo& createInfo);
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;
    VulkanContext(VulkanContext&&) noexcept;
    VulkanContext& operator=(VulkanContext&&) noexcept;

    [[nodiscard]] vk::Instance instance() const noexcept;
    [[nodiscard]] vk::SurfaceKHR surface() const noexcept;
    [[nodiscard]] vk::PhysicalDevice physicalDevice() const noexcept;
    [[nodiscard]] vk::Device device() const noexcept;
    [[nodiscard]] const Queue& graphicsQueue() const noexcept;
    [[nodiscard]] const Queue& presentQueue() const noexcept;
    [[nodiscard]] const Queue& computeQueue() const noexcept;
    [[nodiscard]] const vk::PhysicalDeviceProperties& properties() const noexcept;
    [[nodiscard]] const vk::PhysicalDeviceMemoryProperties& memoryProperties() const noexcept;
    [[nodiscard]] uint32_t apiVersion() const noexcept;
    [[nodiscard]] VkFormat depthFormat() const noexcept;
    [[nodiscard]] bool validationEnabled() const noexcept;
    [[nodiscard]] bool dynamicRenderingEnabled() const noexcept;
    [[nodiscard]] bool synchronization2Enabled() const noexcept;
    [[nodiscard]] bool samplerAnisotropyEnabled() const noexcept;

    void waitIdle() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vulkan_graphics
