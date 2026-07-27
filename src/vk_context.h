#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan_raii.hpp>
namespace vk_engine {
class VkContext {
public:
    VkContext(GLFWwindow* window);
    ~VkContext();

private:
    GLFWwindow *window {nullptr};

    vk::raii::Instance instance                     {nullptr};
    vk::raii::SurfaceKHR surface                    {nullptr};
    // vk::raii::DebugUtilsMessengerEXT debugMessenger {nullptr};
    vk::raii::PhysicalDevice physicalDevice         {nullptr};
    vk::raii::Device device                         {nullptr};

};
}