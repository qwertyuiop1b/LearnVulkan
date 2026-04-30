#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string.h>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE



const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;
const char* TITLE = "LearnVulkan";

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

bool checkValidationLayerSupport() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
    for (const auto& layer: availableLayers) {
        std::cout << "Available layer: \n";
        std::cout << "\t" << layer.layerName << "\n";
    }
    for (const char* layerName: validationLayers) {
        bool layerFound = false;
        for (const auto& layer: availableLayers) {
            if (strcmp(layerName, layer.layerName) == 0) {
                layerFound = true;
                break;
            }
        }

        if (!layerFound) {
            std::cout << "Validation layer not found: " << layerName << std::endl;
            return false;
        }
    }
    return true;
}



class HelloTriagleApplication {
public:
    void run() {
        initVulkan();
        mainLoop();
        cleanup();
    }
private:
    GLFWwindow *window = nullptr;
    VkInstance instance {};

    void createInstance() {
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }
        VkApplicationInfo appInfo {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = TITLE;
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_3;

        VkInstanceCreateInfo createInfo {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        uint32_t extensionCount = 0;
        const char** extensions = nullptr;
        extensions = glfwGetRequiredInstanceExtensions(&extensionCount);
        std::cout << "Required extensions: \n";
        for (uint32_t i = 0; i < extensionCount; i++) {
            std::cout << "\t" << extensions[i] << "\n";
        }

        createInfo.enabledExtensionCount = extensionCount;
        createInfo.ppEnabledExtensionNames = extensions;
        createInfo.enabledLayerCount = 0;

        // check for extension support
        uint32_t extensionCountAvailable = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCountAvailable, nullptr);
        std::vector<VkExtensionProperties> extensionsAvailable(extensionCountAvailable);
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCountAvailable, extensionsAvailable.data());
        std::cout << "Available extensions: \n";
        for (const auto& extension: extensionsAvailable) {
            std::cout << "\t" << extension.extensionName << "\n";
        }

        VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }
    }

    void initVulkan() {
        createInstance();
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(WIDTH, HEIGHT, TITLE, nullptr, nullptr);
        if (window == nullptr) {
            std::cerr << "Failed to create window";
            glfwTerminate();
            throw std::runtime_error("Failed to create window");
        }
    }

    void mainLoop() {
        while(!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }
    }

    void cleanup() {
        vkDestroyInstance(instance, nullptr);
        glfwDestroyWindow(window);
        glfwTerminate();
    }

};


int main() {
    HelloTriagleApplication app;
    try {
        app.run();
    } catch(const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}