#pragma once

#include <cstdint>
#include <string>

#include "vulkan/vulkan_core.h"
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

namespace q_vulkan {

class QWindow {
public:
    QWindow(uint32_t w, uint32_t h, const std::string& title);

    ~QWindow();

    QWindow(const QWindow&) = delete;

    void operator=(const QWindow&) = delete;

    void createWindowSurface(VkInstance instance, VkSurfaceKHR* surface);

    VkExtent2D getExtent() const { return {width, height}; }

    inline bool shouldClose() const { return glfwWindowShouldClose(window); };

    inline void pollEvents() const { glfwPollEvents(); };

private:
    GLFWwindow* window;
    const uint32_t width;
    const uint32_t height;
    const std::string title;

    void initWindow();
};

};  // namespace q_vulkan