#include "context.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace custom {
namespace {

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT,
                                             const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
                                             void*) {
    const char* label = "info";
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        label = "error";
    else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        label = "warning";
    else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT)
        label = "verbose";

    std::cerr << "[Vulkan " << label << "] " << callbackData->pMessage << '\n';
    return VK_FALSE;
}

VkDebugUtilsMessengerCreateInfoEXT makeDebugMessengerCreateInfo() {
    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
    return createInfo;
}

std::vector<const char*> makeNamePointers(const std::vector<std::string>& names) {
    std::vector<const char*> pointers;
    pointers.reserve(names.size());
    for (const std::string& name : names)
        pointers.push_back(name.c_str());
    return pointers;
}

} // namespace

Context::Context(const ContextInfo& info)
    : validationEnabled_(info.enableValidation), appName(info.appName), appVersion(info.appVersion),
      vkVersion(info.vkVersion), instanceExtensions(info.instanceExtensions), validationLayers(info.validationLayers) {}

Context::~Context() {
    destroy();
}

bool Context::init() {
    if (initialized)
        return true;

    if (validationEnabled_)
        addRequiredExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    if (!checkRequestedApiVersion())
        return false;
    if (!checkInstanceExtensionSupport()) {
        lastResult_ = VK_ERROR_EXTENSION_NOT_PRESENT;
        return false;
    }
    if (validationEnabled_ && !checkValidationLayerSupport()) {
        lastResult_ = VK_ERROR_LAYER_NOT_PRESENT;
        return false;
    }

    lastResult_ = createInstance();
    if (lastResult_ != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan instance: " << lastResult_ << '\n';
        return false;
    }

    lastResult_ = loadDebugUtilsFunctions();
    if (lastResult_ == VK_SUCCESS)
        lastResult_ = createDebugUtilsMessengerEXT();
    if (lastResult_ != VK_SUCCESS) {
        std::cerr << "Failed to initialize debug utils: " << lastResult_ << '\n';
        destroy();
        return false;
    }

    initialized = true;
    return true;
}

void Context::destroy() {
    if (debugUtilsMessenger != VK_NULL_HANDLE && destroyDebugUtilsMessenger_ != nullptr)
        destroyDebugUtilsMessenger_(instance, debugUtilsMessenger, nullptr);
    debugUtilsMessenger = VK_NULL_HANDLE;

    if (instance != VK_NULL_HANDLE)
        vkDestroyInstance(instance, nullptr);
    instance = VK_NULL_HANDLE;

    createDebugUtilsMessenger_ = nullptr;
    destroyDebugUtilsMessenger_ = nullptr;
    initialized = false;
}

bool Context::checkRequestedApiVersion() {
    uint32_t supportedVersion = VK_API_VERSION_1_0;
    lastResult_ = vkEnumerateInstanceVersion(&supportedVersion);
    if (lastResult_ != VK_SUCCESS) {
        std::cerr << "Failed to query the Vulkan loader version: " << lastResult_ << '\n';
        return false;
    }
    if (vkVersion > supportedVersion) {
        std::cerr << "Requested Vulkan API version is not supported by the loader\n";
        lastResult_ = VK_ERROR_INCOMPATIBLE_DRIVER;
        return false;
    }
    return true;
}

bool Context::checkInstanceExtensionSupport() const {
    uint32_t count = 0;
    if (vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr) != VK_SUCCESS)
        return false;

    std::vector<VkExtensionProperties> availableExtensions(count);
    if (vkEnumerateInstanceExtensionProperties(nullptr, &count, availableExtensions.data()) != VK_SUCCESS)
        return false;

    for (const std::string& requested : instanceExtensions) {
        const bool found = std::any_of(availableExtensions.begin(), availableExtensions.end(),
                                       [&requested](const VkExtensionProperties& available) {
                                           return requested == available.extensionName;
                                       });
        if (!found) {
            std::cerr << "Required instance extension is unavailable: " << requested << '\n';
            return false;
        }
    }
    return true;
}

bool Context::checkValidationLayerSupport() const {
    uint32_t count = 0;
    if (vkEnumerateInstanceLayerProperties(&count, nullptr) != VK_SUCCESS)
        return false;

    std::vector<VkLayerProperties> availableLayers(count);
    if (vkEnumerateInstanceLayerProperties(&count, availableLayers.data()) != VK_SUCCESS)
        return false;

    for (const std::string& requested : validationLayers) {
        const bool found = std::any_of(availableLayers.begin(), availableLayers.end(),
                                       [&requested](const VkLayerProperties& available) {
                                           return requested == available.layerName;
                                       });
        if (!found) {
            std::cerr << "Required validation layer is unavailable: " << requested << '\n';
            return false;
        }
    }
    return true;
}

void Context::addRequiredExtension(const char* extensionName) {
    const std::string requiredName(extensionName);
    const auto found = std::find(instanceExtensions.begin(), instanceExtensions.end(), requiredName);
    if (found == instanceExtensions.end())
        instanceExtensions.emplace_back(extensionName);
}

VkResult Context::createInstance() {
    const std::vector<const char*> extensionNames = makeNamePointers(instanceExtensions);
    const std::vector<const char*> layerNames = makeNamePointers(validationLayers);

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = appName.c_str();
    appInfo.applicationVersion = appVersion;
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(0, 0, 1);
    appInfo.apiVersion = vkVersion;

    VkInstanceCreateInfo instanceCI{};
    instanceCI.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceCI.pApplicationInfo = &appInfo;
    instanceCI.enabledLayerCount = validationEnabled_ ? static_cast<uint32_t>(layerNames.size()) : 0;
    instanceCI.ppEnabledLayerNames = validationEnabled_ ? layerNames.data() : nullptr;
    instanceCI.enabledExtensionCount = static_cast<uint32_t>(extensionNames.size());
    instanceCI.ppEnabledExtensionNames = extensionNames.data();

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if (validationEnabled_) {
        debugCreateInfo = makeDebugMessengerCreateInfo();
        instanceCI.pNext = &debugCreateInfo;
    }

    return vkCreateInstance(&instanceCI, nullptr, &instance);
}

VkResult Context::loadDebugUtilsFunctions() {
    if (!validationEnabled_)
        return VK_SUCCESS;

    createDebugUtilsMessenger_ = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    destroyDebugUtilsMessenger_ = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
    return createDebugUtilsMessenger_ != nullptr && destroyDebugUtilsMessenger_ != nullptr
               ? VK_SUCCESS
               : VK_ERROR_EXTENSION_NOT_PRESENT;
}

VkResult Context::createDebugUtilsMessengerEXT() {
    if (!validationEnabled_)
        return VK_SUCCESS;

    VkDebugUtilsMessengerCreateInfoEXT createInfo = makeDebugMessengerCreateInfo();
    return createDebugUtilsMessenger_(instance, &createInfo, nullptr, &debugUtilsMessenger);
}
} // namespace custom
