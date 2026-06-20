#pragma once

/**
 * @file features.hpp
 * @brief 跨平台 Vulkan 特性与格式检测，支持降级路径
 */

#include <iostream>
#include <vector>
#include <vulkan/vulkan.h>

namespace vulkan_tutorial {

inline bool
isFormatSupported(VkPhysicalDevice device, VkFormat format, VkImageTiling tiling, VkFormatFeatureFlags features) {
    VkFormatProperties props{};
    vkGetPhysicalDeviceFormatProperties(device, format, &props);
    if (tiling == VK_IMAGE_TILING_LINEAR)
        return (props.linearTilingFeatures & features) == features;
    return (props.optimalTilingFeatures & features) == features;
}

inline VkFormat findDepthFormat(VkPhysicalDevice device) {
    const std::vector<VkFormat> candidates = {
        VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT};
    for (VkFormat format : candidates) {
        if (isFormatSupported(device, format, VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT))
            return format;
    }
    return VK_FORMAT_UNDEFINED;
}

inline VkFormat findDepthStencilFormat(VkPhysicalDevice device) {
    const std::vector<VkFormat> candidates = {
        VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT};
    for (VkFormat format : candidates) {
        if (isFormatSupported(
                device, format, VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
            VkFormatProperties props{};
            vkGetPhysicalDeviceFormatProperties(device, format, &props);
            if (props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)
                return format;
        }
    }
    return findDepthFormat(device);
}

inline bool hasStencilComponent(VkFormat format) {
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

inline float getMaxAnisotropy(VkPhysicalDevice device) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(device, &props);
    return props.limits.maxSamplerAnisotropy;
}

inline bool supportsGeometryShader(VkPhysicalDevice device) {
    VkPhysicalDeviceFeatures features{};
    vkGetPhysicalDeviceFeatures(device, &features);
    return features.geometryShader == VK_TRUE;
}

inline bool supportsSamplerAnisotropy(VkPhysicalDevice device) {
    VkPhysicalDeviceFeatures features{};
    vkGetPhysicalDeviceFeatures(device, &features);
    return features.samplerAnisotropy == VK_TRUE;
}

inline bool supportsWideLines(VkPhysicalDevice device) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(device, &props);
    return props.limits.lineWidthRange[1] > 1.0f;
}

inline void logFeatureFallback(const char* featureName, const char* fallback) {
    std::cerr << "[Feature] " << featureName << " 不可用，降级为: " << fallback << "\n";
}

} // namespace vulkan_tutorial
