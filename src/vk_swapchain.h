#pragma once

#include "vk_context.h"
#include "vk_window.h"

#include <cstdint>
#include <vector>

#include <vulkan/vulkan_raii.hpp>

namespace vk_engine
{
class VkSwapchain
{
public:
    VkSwapchain(const VkContext& inContext, const VkWindow& inWindow);
    ~VkSwapchain() = default;

    VkSwapchain(const VkSwapchain&) = delete;
    VkSwapchain& operator=(const VkSwapchain&) = delete;
    VkSwapchain(VkSwapchain&&) = delete;
    VkSwapchain& operator=(VkSwapchain&&) = delete;

    void Recreate();

    const vk::raii::SwapchainKHR& GetHandle() const noexcept
    {
        return swapchain;
    }

    vk::Extent2D GetExtent() const noexcept
    {
        return extent;
    }

    vk::Format GetImageFormat() const noexcept
    {
        return format;
    }

    uint32_t GetImageCount() const noexcept
    {
        return static_cast<uint32_t>(swapchainImages.size());
    }

    const std::vector<vk::Image>& GetImages() const noexcept
    {
        return swapchainImages;
    }

    vk::Image GetImage(uint32_t index) const
    {
        return swapchainImages.at(index);
    }

    bool HasValidFramebufferExtent() const noexcept;

private:
    void Create(VkSwapchainKHR oldSwapchain);

    const VkContext& context;
    const VkWindow& window;

    vk::raii::SwapchainKHR swapchain{nullptr};
    std::vector<vk::Image> swapchainImages;
    vk::Format format{vk::Format::eUndefined};
    vk::Extent2D extent{};
};
} // namespace vk_engine
