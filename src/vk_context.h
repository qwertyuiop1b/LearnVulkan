#pragma once

#include "vk_window.h"
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan_raii.hpp>

namespace vk_engine
{
class VkContext
{
public:
    VkContext(const VkWindow& inWindow);
    ~VkContext();

private:
    const VkWindow& vkWindow;

    vk::raii::Instance instance{nullptr};
    vk::raii::DebugUtilsMessengerEXT debugMessenger{nullptr};
    vk::raii::SurfaceKHR surface{nullptr};
    vk::raii::PhysicalDevice physicalDevice{nullptr};
    vk::raii::Device device{nullptr};
};
} // namespace vk_engine