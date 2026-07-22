#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace custom {
struct ContextInfo {
    std::string appName = "LearnVulkan";
    uint32_t appVersion = VK_MAKE_VERSION(0, 1, 0);
    uint32_t vkVersion = VK_API_VERSION_1_0;

    // The platform layer supplies WSI extensions here. Context does not know about windows.
    std::vector<std::string> instanceExtensions;
    bool enableValidation = true;
    std::vector<std::string> validationLayers = {"VK_LAYER_KHRONOS_validation"};
};

class Context {
  public:
    Context(const ContextInfo& info);
    ~Context();

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    bool init();

    [[nodiscard]] bool isValid() const {
        return initialized;
    }

    [[nodiscard]] VkInstance instanceHandle() const {
        return instance;
    }

    [[nodiscard]] VkResult lastResult() const {
        return lastResult_;
    }

  private:
    void destroy();
    bool checkRequestedApiVersion();
    bool checkInstanceExtensionSupport() const;
    bool checkValidationLayerSupport() const;
    void addRequiredExtension(const char* extensionName);

    VkResult createInstance();
    VkResult loadDebugUtilsFunctions();
    VkResult createDebugUtilsMessengerEXT();

  private:
    bool initialized = false;
    bool validationEnabled_ = false;
    std::string appName;
    uint32_t appVersion;
    uint32_t vkVersion;
    std::vector<std::string> instanceExtensions;
    std::vector<std::string> validationLayers;

    VkInstance instance{VK_NULL_HANDLE};
    VkDebugUtilsMessengerEXT debugUtilsMessenger{VK_NULL_HANDLE};
    PFN_vkCreateDebugUtilsMessengerEXT createDebugUtilsMessenger_ = nullptr;
    PFN_vkDestroyDebugUtilsMessengerEXT destroyDebugUtilsMessenger_ = nullptr;
    VkResult lastResult_ = VK_SUCCESS;
};
} // namespace custom
