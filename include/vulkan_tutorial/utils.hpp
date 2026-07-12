#pragma once

/**
 * @file utils.hpp
 * @brief 通用工具：调试回调、文件读取、错误检查宏
 *
 * 所有章节共用的基础设施。
 */

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// ─── 错误检查宏 ───────────────────────────────────────────────────────────────

/// 检查 VkResult，失败则抛出带描述的异常
#define VK_CHECK(call)                                                                                                 \
    do {                                                                                                               \
        VkResult _result = (call);                                                                                     \
        if (_result != VK_SUCCESS) {                                                                                   \
            throw std::runtime_error(std::string("[VK_CHECK] ") + #call + " failed with code " +                       \
                                     std::to_string(_result) + " at " __FILE__ ":" + std::to_string(__LINE__));        \
        }                                                                                                              \
    } while (0)

// ─── 验证层配置 ───────────────────────────────────────────────────────────────

#ifdef NDEBUG
constexpr bool ENABLE_VALIDATION_LAYERS = false;
#else
constexpr bool ENABLE_VALIDATION_LAYERS = true;
#endif

inline const std::vector<const char*> VALIDATION_LAYERS = {"VK_LAYER_KHRONOS_validation"};

inline const std::vector<const char*> DEVICE_EXTENSIONS = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
// macOS 上 MoltenVK 需要以下扩展
#ifdef __APPLE__
    "VK_KHR_portability_subset",
#endif
};

// ─── 调试信使 ──────────────────────────────────────────────────────────────────

/**
 * @brief Vulkan 验证层调试回调函数
 *
 * 当验证层检测到问题时，此函数被调用，将信息打印到 stderr。
 */
inline VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                    VkDebugUtilsMessageTypeFlagsEXT /*type*/,
                                                    const VkDebugUtilsMessengerCallbackDataEXT* data,
                                                    void* /*userData*/) {
    const char* prefix = "[VK]";
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        prefix = "\033[31m[VK ERROR]\033[0m";
    else if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        prefix = "\033[33m[VK WARN]\033[0m";
    else if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
        prefix = "\033[36m[VK INFO]\033[0m";

    std::cerr << prefix << " " << data->pMessage << "\n";
    return VK_FALSE; // 不中止调用
}

/// 填充 VkDebugUtilsMessengerCreateInfoEXT 结构体
inline void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& info) {
    info = {};
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    info.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    info.pfnUserCallback = debugCallback;
}

// ─── 验证层支持检测 ────────────────────────────────────────────────────────────

/// 检查所需验证层是否被系统支持
inline bool checkValidationLayerSupport() {
    uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> available(count);
    vkEnumerateInstanceLayerProperties(&count, available.data());

    for (const char* name : VALIDATION_LAYERS) {
        bool found = std::any_of(available.begin(), available.end(), [name](const VkLayerProperties& p) {
            return std::strcmp(p.layerName, name) == 0;
        });
        if (!found)
            return false;
    }
    return true;
}

// ─── Instance 扩展枚举 ────────────────────────────────────────────────────────

/// 获取创建 VkInstance 所需的扩展列表（含调试扩展和 macOS 兼容性扩展）
inline std::vector<const char*> getRequiredInstanceExtensions() {
    uint32_t glfwCount = 0;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwCount);
    std::vector<const char*> extensions(glfwExts, glfwExts + glfwCount);

    if (ENABLE_VALIDATION_LAYERS)
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    // macOS / MoltenVK 需要的扩展
#ifdef __APPLE__
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#endif

    return extensions;
}

// ─── 文件读取（用于 SPIR-V 着色器） ───────────────────────────────────────────

/**
 * @brief 从已编译的 SPIR-V 文件创建着色器模块
 *
 * 这是在实际项目中加载着色器的正确方式。
 * CMake 将 GLSL 编译为 SPIR-V，存放在 build/shaders/ 目录。
 * SHADER_DIR 宏由 CMake 在编译时注入，指向 build/shaders/ 的绝对路径。
 *
 * @param device   逻辑设备
 * @param filename SPIR-V 文件名（不含路径，如 "triangle.vert.spv"）
 */
inline VkShaderModule createShaderModuleFromFile(VkDevice device, const std::string& filename) {
#ifndef SHADER_DIR
    const std::string path = filename;
#else
    const std::string path = std::string(SHADER_DIR) + "/" + filename;
#endif
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("无法打开着色器文件: " + path +
                                 "\n  请确保在 build 目录下运行程序，"
                                 "或检查 CMake 的 SHADER_DIR 定义是否正确。");

    const size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> code(fileSize);
    file.seekg(0);
    file.read(code.data(), static_cast<std::streamsize>(fileSize));

    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = code.size();
    ci.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkResult result = vkCreateShaderModule(device, &ci, nullptr, &shaderModule);
    if (result != VK_SUCCESS)
        throw std::runtime_error("创建着色器模块失败: " + filename + " (code=" + std::to_string(result) + ")");
    return shaderModule;
}

/// 读取二进制文件（SPIR-V shader 字节码）
inline std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Failed to open file: " + filename);

    const size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(fileSize));
    return buffer;
}

// ─── 队列族索引 ───────────────────────────────────────────────────────────────

/**
 * @brief 存储物理设备队列族的索引
 *
 * Vulkan 中，不同类型的操作（图形、呈现）由不同的队列族处理。
 * 使用 std::optional 表示"尚未找到"的状态。
 */
struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily; ///< 支持图形命令的队列族
    std::optional<uint32_t> presentFamily;  ///< 支持窗口呈现的队列族
    std::optional<uint32_t> computeFamily;  ///< 优先选择不含 graphics 的独立计算队列族

    /// 两个队列族都已找到则返回 true
    [[nodiscard]] bool isComplete() const {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

/// 查找物理设备上满足需求的队列族索引
inline QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface) {
    QueueFamilyIndices indices;

    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());

    for (uint32_t i = 0; i < count; ++i) {
        if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && !indices.graphicsFamily)
            indices.graphicsFamily = i;

        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            const bool dedicated = (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0;
            if (!indices.computeFamily || dedicated)
                indices.computeFamily = i;
        }

        VkBool32 presentSupport = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
        if (presentSupport)
            indices.presentFamily = i;

    }
    if (!indices.computeFamily)
        indices.computeFamily = indices.graphicsFamily;
    return indices;
}

// ─── 交换链支持信息 ────────────────────────────────────────────────────────────

/**
 * @brief 查询物理设备对交换链的支持细节
 *
 * - capabilities: 交换链能力（图像数量范围、分辨率范围等）
 * - formats:      支持的表面格式（颜色空间、像素格式）
 * - presentModes: 支持的呈现模式（FIFO/Mailbox/Immediate 等）
 */
struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

inline SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface) {
    SwapChainSupportDetails details;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
    if (formatCount) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
    }

    uint32_t modeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &modeCount, nullptr);
    if (modeCount) {
        details.presentModes.resize(modeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &modeCount, details.presentModes.data());
    }
    return details;
}

/// 选择最佳表面格式（优先 B8G8R8A8_SRGB + SRGB_NONLINEAR）
inline VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& available) {
    for (const auto& fmt : available) {
        if (fmt.format == VK_FORMAT_B8G8R8A8_SRGB && fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            return fmt;
    }
    return available[0];
}

/// 选择最佳呈现模式（优先三缓冲 Mailbox，无则 FIFO）
inline VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& available) {
    for (auto mode : available) {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
            return mode;
    }
    return VK_PRESENT_MODE_FIFO_KHR; // FIFO 是 Vulkan 规范保证必须支持的
}

/// 选择交换链图像分辨率（尽量匹配窗口实际像素大小）
inline VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& caps, GLFWwindow* window) {
    if (caps.currentExtent.width != UINT32_MAX)
        return caps.currentExtent;

    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window, &width, &height);

    VkExtent2D actual = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    actual.width = std::clamp(actual.width, caps.minImageExtent.width, caps.maxImageExtent.width);
    actual.height = std::clamp(actual.height, caps.minImageExtent.height, caps.maxImageExtent.height);
    return actual;
}

/// 检查物理设备是否支持所需的设备扩展
inline bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> available(count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, available.data());

    std::set<std::string> required(DEVICE_EXTENSIONS.begin(), DEVICE_EXTENSIONS.end());
    for (const auto& ext : available)
        required.erase(ext.extensionName);

    return required.empty();
}
