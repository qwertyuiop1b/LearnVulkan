#include "Qwindow.hpp"
#include <stdexcept>

namespace q_vulkan {
    QWindow::QWindow(uint32_t w, uint32_t h, const std::string& title)
    : width(w)
    , height(h)
    , title(title) {
        initWindow();
    }

    QWindow::~QWindow() {
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void QWindow::createWindowSurface(VkInstance instance, VkSurfaceKHR* surface) {
        if (glfwCreateWindowSurface(instance, window, nullptr, surface) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create window surface");
        } 
    }

    void QWindow::initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
        if (window == nullptr) {
            throw std::runtime_error("Failed to create window!");
        }
    }
};