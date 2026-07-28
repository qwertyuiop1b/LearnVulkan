#include "vk_window.h"
#include <iostream>

namespace vk_engine
{
VkWindow::VkWindow(uint32_t inWidth, uint32_t inHeight) : width(inWidth), height(inHeight)
{
    CreateWindow();
}

VkWindow::~VkWindow()
{
    if (window != nullptr)
    {
        glfwDestroyWindow(window);
    }
    glfwTerminate();
}

void VkWindow::CreateWindow()
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window = glfwCreateWindow(width, height, "No Title", nullptr, nullptr);
    if (window == nullptr)
    {
        std::cerr << "failed to create window" << std::endl;
        return;
    }
    glfwSetWindowUserPointer(window, this);
}
} // namespace vk_engine