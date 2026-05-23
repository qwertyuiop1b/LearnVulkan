#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#include <vulkan/vk_platform.h>
#include <vulkan/vulkan_core.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define VK_CHECK(call)                                            \
    do {                                                          \
        VkResult result = (call);                                 \
        if (result != VK_SUCCESS) {                               \
            std::cerr << "Vulkan Error: " << result               \
                << " in " << __FUNCTION__                         \
                << " at " << __FILE__ << ":" << __LINE__          \
                << " - Call: " << #call << "\n";                  \
            assert(false && ("Vulkan call failed"));              \
        }                                                         \
    } while(0)                                                    


#ifdef NDEBUG 
    const bool ENABLE_VALIDATION_LAYERS = false;
#else
    const bool ENABLE_VALIDATION_LAYERS = true;
#endif


const std::vector<const char*> VALIDATION_LAYERS = 
{
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> DEVICE_EXTENSIONS = 
{
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

static std::vector<const char*> getRequiredDeviceExtensions(VkPhysicalDevice device)
{
    std::vector<const char*> extensions = DEVICE_EXTENSIONS;
    
    // Check for VK_KHR_portability_subset support (required on macOS)
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());
    
    for (const auto& ext : availableExtensions)
    {
        if (strcmp(ext.extensionName, "VK_KHR_portability_subset") == 0)
        {
            extensions.push_back("VK_KHR_portability_subset");
            break;
        }
    }
    
    return extensions;
}

static bool checkValidationLayerSupport() 
{
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> layers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, layers.data());

    for (const auto& validation : VALIDATION_LAYERS) 
    {
        bool supported = false;
        for (const auto& layer: layers) 
        {
            if (strcmp(validation, layer.layerName) == 0) 
            {
                supported = true;
                break;
            }
        }
        if (!supported) 
        {
            return false;
        }
    } 

    return true;
}

static std::vector<const char*> getRequiredExtensions() 
{
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    std::cout << "Required extensions:" << std::endl;
    for (auto& extension : extensions) 
    {
        std::cout << "\t" << extension << std::endl;
    }
#ifdef __APPLE__
    extensions.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif 
    if (ENABLE_VALIDATION_LAYERS) {
        extensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

static bool checkDeviceExtensionSupport(VkPhysicalDevice device)
{
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, extensions.data());

    std::set<std::string> requiredExtensions(DEVICE_EXTENSIONS.begin(), DEVICE_EXTENSIONS.end());
    
    // Check which available extensions we have
    std::set<std::string> availableExtensionNames;
    bool hasPortabilitySubset = false;
    for (const auto& extension : extensions) 
    {
        availableExtensionNames.insert(extension.extensionName);
        if (strcmp(extension.extensionName, "VK_KHR_portability_subset") == 0)
        {
            hasPortabilitySubset = true;
        }
    }
    
    // Check required extensions are available
    for (const auto& requiredExt : DEVICE_EXTENSIONS)
    {
        if (availableExtensionNames.find(requiredExt) == availableExtensionNames.end())
        {
            return false;
        }
    }
    
    // If portability subset is available, it must be enabled
    // This is enforced in getRequiredDeviceExtensions
    
    return true;
}

struct SwapchainSupportDetails 
{
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;

    VkExtent2D chooseSwapExtent(GLFWwindow* window) const 
    {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
        {
            return capabilities.currentExtent;
        }
        else 
        {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height),
            };
            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
            return actualExtent;
        }
    }

    VkSurfaceFormatKHR chooseSwapchainFormat() const 
    {
        assert(!formats.empty() && "formats is empty!");
        for (const auto& format : formats)
        {
            if (format.format == VK_FORMAT_B8G8R8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            {
                return format;
            }
        }
        return formats[0];
    }

    VkPresentModeKHR choosePresentMode() const
    {
        for (const auto& presentMode : presentModes)
        {
            if (presentMode == VK_PRESENT_MODE_MAILBOX_KHR)
            {
                return VK_PRESENT_MODE_MAILBOX_KHR;
            }
        }
        return VK_PRESENT_MODE_FIFO_KHR;
    }
};


struct QueueFamilyIndices 
{
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() 
    {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};



static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageServerity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pData
) 
{
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
}

static VkResult createDebugUtilsMessengerEXT(
    VkInstance instance, 
    const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, 
    const VkAllocationCallbacks* pAllocator, 
    VkDebugUtilsMessengerEXT* pDebugMessenger
) 
{
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) 
    {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } 
    else 
    {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

static void destroyDebugUtilsMessengerEXT(
    VkInstance instance,
    VkDebugUtilsMessengerEXT debugMessenger,
    const VkAllocationCallbacks* pAllocator
) 
{
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) 
    {
        func(instance, debugMessenger, nullptr);
    }
}

static std::vector<char> readFile(const std::string& filename) 
{
    const std::filesystem::path inputPath(filename);
    const std::filesystem::path currentPath = std::filesystem::current_path();
    const std::filesystem::path candidates[] =
    {
        inputPath,
        currentPath / inputPath,
        currentPath.parent_path() / inputPath,
    };

    std::filesystem::path resolvedPath;
    for (const auto& candidate : candidates)
    {
        if (std::filesystem::exists(candidate))
        {
            resolvedPath = candidate;
            break;
        }
    }

    if (resolvedPath.empty())
    {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::ifstream file(resolvedPath, std::ios::ate | std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + resolvedPath.string());
    }

    const size_t size = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(size);
    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(size));
    file.close();
    return buffer;
}
