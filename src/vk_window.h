#pragma once

#include <cstdint>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace vk_engine
{
class VkWindow
{
public:
    VkWindow(uint32_t inWidth, uint32_t inHeight);
    ~VkWindow();

    VkWindow(const VkWindow&) = delete;
    VkWindow& operator=(const VkWindow&) = delete;

    inline uint32_t GetWidth() const
    {
        return width;
    }

    inline uint32_t GetHeight() const
    {
        return height;
    }

    inline GLFWwindow& GetWindow() const
    {
        return *window;
    }

    inline bool ShouldClose() const
    {
        return glfwWindowShouldClose(window);
    }

    void ProcessPendingEvents() const
    {
        glfwPollEvents();
    }

private:
    uint32_t width;
    uint32_t height;
    GLFWwindow* window{nullptr};

    void CreateWindow();
};
} // namespace vk_engine
