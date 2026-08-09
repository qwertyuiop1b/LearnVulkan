#pragma once

#include "vk_window.h"
#include <cstdint>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan_raii.hpp>
#include "vk_vma.h"

namespace vk_engine
{
class VkContext
{
public:
    VkContext(const VkWindow& inWindow);
    ~VkContext();

    VkContext(const VkContext&) = delete;
    VkContext& operator=(const VkContext&) = delete;

    VkContext(VkContext&&) = delete;
    VkContext& operator=(VkContext&&) = delete;

    const vk::raii::PhysicalDevice& GetPhysicalDevice() const noexcept
    {
        return physicalDevice;
    }

    const vk::raii::Device& GetDevice() const noexcept
    {
        return device;
    }

    const vk::raii::SurfaceKHR& GetSurface() const noexcept
    {
        return surface;
    }

    const vk::raii::Instance& GetInstance() const noexcept
    {
        return instance;
    }

    const vk::raii::Queue& GetGraphicQueue() const noexcept
    {
        return graphicQueue;
    }

    const vk::raii::Queue& GetPresentQueue() const noexcept
    {
        return presentQueue;
    }

    uint32_t GetGraphicQueueFamilyIndex() const noexcept
    {
        return graphicQueueIndex;
    }

    uint32_t GetPresentQueueFamilyIndex() const noexcept
    {
        return presentQueueIndex;
    }

    VmaAllocator GetAllocator() const noexcept
    {
        return allocator;
    }

private:
    const VkWindow& vkWindow;

    vk::raii::Instance instance{nullptr};
    vk::raii::DebugUtilsMessengerEXT debugMessenger{nullptr};
    vk::raii::SurfaceKHR surface{nullptr};
    vk::raii::PhysicalDevice physicalDevice{nullptr};
    vk::raii::Device device{nullptr};

    vk::raii::Queue graphicQueue{nullptr};
    vk::raii::Queue presentQueue{nullptr};

    uint32_t graphicQueueIndex{};
    uint32_t presentQueueIndex{};

    VmaAllocator allocator{nullptr};
};
} // namespace vk_engine
