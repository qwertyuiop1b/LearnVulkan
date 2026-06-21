#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fmt/base.h>
#include <string>
#include <vector>
#include <vulkan/vk_platform.h>
#include <vulkan/vulkan_core.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

static void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    fmt::println("resize: {},{}", width, height);
}

static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    switch (key) {
    case GLFW_KEY_ESCAPE:
        if (action == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }
        break;
    default:
        fmt::println("keyCode: {}", key);
    }
}

#ifdef NDEBUG
const bool enable_validation_layers = false;
#else
const bool enable_validation_layers = true;
#endif

inline std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"
#ifdef __APPLE__
                                                    "VK_KHR_portability_subset"
#endif
};

VKAPI_ATTR VkBool32 VKAPI_CALL debugUtilsCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                  VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                                                  const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                  void* pUserData) {
    fmt::println("validation layer: {}", pCallbackData->pMessage);
    return VK_FALSE;
}

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
};

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT messenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, messenger, pAllocator);
    }
}

class Example {
  public:
    Example(const VkExtent2D& extent, const std::string& title) : extent(extent), title(title) {
        initWindow();
        initVulkan();
    }
    ~Example() {
        DestroyDebugUtilsMessengerEXT(instance, debugUtilsMessenger, nullptr);
        vkDestroyInstance(instance, nullptr);
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void run() {
        while (!glfwWindowShouldClose(window)) {
            draw();
            glfwPollEvents();
        }
    }

  private:
    VkExtent2D extent;
    const std::string& title;
    GLFWwindow* window{nullptr};

    VkInstance instance{VK_NULL_HANDLE};
    VkDebugUtilsMessengerEXT debugUtilsMessenger{VK_NULL_HANDLE};

    void initWindow() {
        glfwInit();
        if (!glfwVulkanSupported()) {
            fmt::println("Vulkan not supported!");
            return;
        }
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        window = glfwCreateWindow(extent.width, extent.height, title.c_str(), nullptr, nullptr);
        if (window == nullptr) {
            fmt::println("Failed to create window!");
            return;
        }
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
        glfwSetKeyCallback(window, keyCallback);
    }

    static std::vector<const char*> getInstanceExtensions() {
        uint32_t extensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&extensionCount);
        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + extensionCount);
        if (enable_validation_layers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
#ifdef __APPLE__
        extensions.push_back(VK_KHR_PROTABILITY_ENUMERATE_EXTENSION_NAME);
        extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#endif
        return extensions;
    }

    void createInstance() {
        VkApplicationInfo appInfo{
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = "LearnVulkan",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "No Engine",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = VK_API_VERSION_1_4,
        };

        auto extensions = getInstanceExtensions();
        VkInstanceCreateInfo instanceCI{
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &appInfo,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = nullptr,
            .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data(),
        };
#ifdef __APPLE__
        instanceCI.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
        auto debugCreateInfo = getDebugUtilsCreateInfo();
        if (enable_validation_layers) {
            instanceCI.enabledLayerCount = validationLayers.size();
            instanceCI.ppEnabledLayerNames = validationLayers.data();
            instanceCI.pNext = &debugCreateInfo;
        }
        if (vkCreateInstance(&instanceCI, nullptr, &instance) != VK_SUCCESS) {
            fmt::println("Failed to create instance!");
        }
    }

    static VkDebugUtilsMessengerCreateInfoEXT getDebugUtilsCreateInfo() {
        VkDebugUtilsMessengerCreateInfoEXT createInfo{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT,
            .messageType =
                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pUserData = nullptr,
            .pfnUserCallback = debugUtilsCallback,
        };
        return createInfo;
    }

    void createDebugMessenger() {
        if (!enable_validation_layers)
            return;
        auto createInfo = getDebugUtilsCreateInfo();
        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugUtilsMessenger) != VK_SUCCESS) {
            fmt::println("Failed to create debug utils messenger!");
        }
    }

    void pickPhysicalDevice() {
        
    }

    void initVulkan() {
        createInstance();
        createDebugMessenger();
        pickPhysicalDevice();
    }

    void draw() {}
};

int main() {
    try {
        Example app{VkExtent2D{800, 600}, "LearnVulkan"};
        app.run();
    } catch (const std::exception& e) {
        fmt::println(e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}