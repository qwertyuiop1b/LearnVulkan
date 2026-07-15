#pragma once

#include <graphics/render_target.hpp>
#include <graphics/vulkan_context.hpp>

#include <cstdint>
#include <vector>

namespace vulkan_graphics {

enum class SwapchainStatus {
    eSuccess,
    eSuboptimal,
    eOutOfDate,
};

struct SwapchainAcquireResult {
    SwapchainStatus status = SwapchainStatus::eOutOfDate;
    uint32_t imageIndex = 0;
};

struct SwapchainCreateInfo {
    VkExtent2D desiredExtent{};
    uint32_t desiredImageCount = 0;
    std::vector<vk::SurfaceFormatKHR> preferredFormats = {
        {vk::Format::eB8G8R8A8Srgb, vk::ColorSpaceKHR::eSrgbNonlinear},
        {vk::Format::eR8G8B8A8Srgb, vk::ColorSpaceKHR::eSrgbNonlinear},
    };
    std::vector<vk::PresentModeKHR> preferredPresentModes = {
        vk::PresentModeKHR::eMailbox,
        vk::PresentModeKHR::eFifo,
    };
    vk::ImageUsageFlags imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
};

class Swapchain final {
  public:
    Swapchain() = default;
    Swapchain(const VulkanContext& context, const SwapchainCreateInfo& createInfo);
    ~Swapchain() noexcept;

    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;
    Swapchain(Swapchain&& other) noexcept;
    Swapchain& operator=(Swapchain&& other) noexcept;

    void recreate(VkExtent2D desiredExtent);
    [[nodiscard]] SwapchainAcquireResult acquireNextImage(VkSemaphore imageAvailableSemaphore,
                                                           uint64_t timeout = UINT64_MAX) const;
    [[nodiscard]] SwapchainStatus present(uint32_t imageIndex,
                                          VkSemaphore renderFinishedSemaphore = VK_NULL_HANDLE) const;

    [[nodiscard]] bool isValid() const noexcept;
    [[nodiscard]] vk::SwapchainKHR handle() const noexcept;
    [[nodiscard]] VkSwapchainKHR nativeHandle() const noexcept;
    [[nodiscard]] vk::Format format() const noexcept;
    [[nodiscard]] vk::ColorSpaceKHR colorSpace() const noexcept;
    [[nodiscard]] vk::PresentModeKHR presentMode() const noexcept;
    [[nodiscard]] VkExtent2D extent() const noexcept;
    [[nodiscard]] uint32_t imageCount() const noexcept;
    [[nodiscard]] const RenderTarget& renderTarget(uint32_t imageIndex) const;

  private:
    struct Resources;

    static Resources createResources(const VulkanContext& context,
                                     const SwapchainCreateInfo& createInfo,
                                     VkSwapchainKHR oldSwapchain);
    void assign(Resources&& resources) noexcept;
    void destroy() noexcept;

    const VulkanContext* context_ = nullptr;
    SwapchainCreateInfo createInfo_{};
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    vk::Format format_ = vk::Format::eUndefined;
    vk::ColorSpaceKHR colorSpace_ = vk::ColorSpaceKHR::eSrgbNonlinear;
    vk::PresentModeKHR presentMode_ = vk::PresentModeKHR::eFifo;
    VkExtent2D extent_{};
    std::vector<RenderTarget> renderTargets_;
};

} // namespace vulkan_graphics
