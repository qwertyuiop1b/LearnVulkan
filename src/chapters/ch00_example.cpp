#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fmt/base.h>
#include <string>
#include <vulkan/vulkan_core.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

static void framebufferResizeCallback(GLFWwindow* window, int height, int width) {
    fmt::println("resize: {},{}", width, height);
}

class Example {
  public:
    Example(VkExtent2D extent, const std::string& title) : extent(extent), title(title) {
        initWindow();
        initVulkan();
    }

    ~Example() {
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void run() {}

  private:
    VkExtent2D extent;
    const std::string title;
    GLFWwindow* window{nullptr};

    void initWindow() {
        glfwInit();
        if (glfwVulkanSupported()) {
            fmt::println("Vulkan not supported");
            return;
        }
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        window = glfwCreateWindow(extent.width, extent.height, title.c_str(), nullptr, nullptr);
        if (window == nullptr) {
            fmt::println("Failed to create glfw window");
            return;
        }
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }
    void initVulkan() {}
};

int main() {
    try {
        Example app{VkExtent2D{800, 600}, "Learn Vulkan"};
        app.run();
    } catch (const std::exception& e) {
        fmt::println(e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
};