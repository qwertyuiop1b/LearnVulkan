#pragma once
#include <iostream>
#include <vulkan/vulkan_core.h>
#include <array>
#include <stdexcept>

namespace custom {

#ifdef NDEBUG
const bool ENABLE_VALIDATION_LAYER = false;
#else
const bool ENABLE_VALIDATION_LAYER = true;
#endif

#define VK_CHECK_RESULT(func)                                                                                          \
    do {                                                                                                               \
        VkResult result = (func);                                                                                      \
        if (result != VK_SUCCESS) {                                                                                    \
            std::cout << "[VK_CHECK FAILED] " << __FILE__ << " at line " << __LINE__ << std::endl;                     \                                                                    
        }                                                                                                              \
    } while (false);

inline VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,
                                             const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                             const VkAllocationCallbacks* pAllocator,
                                             VkDebugUtilsMessengerEXT* pMessenger) {
    auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    if (func == nullptr) {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
    return func(instance, pCreateInfo, pAllocator, pMessenger);
}

inline void DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                          VkDebugUtilsMessengerEXT messenger,
                                          const VkAllocationCallbacks* pAllocator) {
    auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
    if (func != nullptr) {
        func(instance, messenger, pAllocator);
    }
}

inline VkBool32 DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                              VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                              const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                              void* pUserData) {
    std::cerr << "【VK_DEBUG_INFO】"
              << " " << pCallbackData->pMessage << "\n";
    return VK_FALSE;
}

}; // namespace custom