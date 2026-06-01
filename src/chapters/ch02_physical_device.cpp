/**
 * @file ch02_physical_device.cpp
 * @brief 第02章：物理设备选择与队列族
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【核心概念】
 *
 * VkPhysicalDevice（物理设备）代表系统中一个实际的 GPU 硬件。
 * 一台机器可能有多个 GPU（如笔记本的集显+独显）。
 *
 * 【选择标准】
 *
 *  1. 设备类型偏好：独立 GPU > 集成 GPU > 虚拟 GPU
 *  2. 功能支持：几何着色器、曲面细分、各向异性过滤等
 *  3. 队列族支持：必须有图形队列族
 *  4. 扩展支持：必须支持 VK_KHR_swapchain
 *  5. 交换链支持：有可用格式和呈现模式
 *
 * 【队列族（Queue Families）】
 *
 *  Vulkan 中，命令通过"队列（Queue）"提交给 GPU。
 *  不同类型的命令需要不同类型的队列：
 *
 *  ┌─────────────────┬──────────────────────────────────────┐
 *  │ 队列类型        │ 用途                                  │
 *  ├─────────────────┼──────────────────────────────────────┤
 *  │ Graphics        │ 绘制命令（draw calls）                │
 *  │ Compute         │ 计算着色器                            │
 *  │ Transfer        │ 内存拷贝                              │
 *  │ Present         │ 将图像显示到屏幕（不一定独立存在）     │
 *  └─────────────────┴──────────────────────────────────────┘
 *
 * 【设备评分系统】
 *
 *  我们实现一个评分机制，自动选出最合适的 GPU：
 *  - 离散 GPU: +1000 分
 *  - 最大纹理尺寸: 加相应分数
 *  - 必要功能缺失: 0 分（直接淘汰）
 * ═══════════════════════════════════════════════════════════════════════════
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan_tutorial/utils.hpp>

#include <iostream>
#include <map>
#include <stdexcept>

// ─── 辅助函数 ──────────────────────────────────────────────────────────────────

/// 将设备类型枚举转换为可读字符串
static std::string deviceTypeToString(VkPhysicalDeviceType type) {
    switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        return "集成显卡 (Integrated GPU)";
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        return "独立显卡 (Discrete GPU)";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        return "虚拟显卡 (Virtual GPU)";
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
        return "CPU 软件渲染";
    default:
        return "其他/未知";
    }
}

/// 打印详细的物理设备信息
static void printDeviceInfo(VkPhysicalDevice device) {
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceProperties(device, &props);
    vkGetPhysicalDeviceFeatures(device, &features);

    std::cout << "  📌 设备名称   : " << props.deviceName << "\n";
    std::cout << "  🖥️  设备类型   : " << deviceTypeToString(props.deviceType) << "\n";
    std::cout << "  🔢 API 版本   : " << VK_VERSION_MAJOR(props.apiVersion) << "." << VK_VERSION_MINOR(props.apiVersion)
              << "." << VK_VERSION_PATCH(props.apiVersion) << "\n";
    std::cout << "  🔢 Driver版本 : " << props.driverVersion << "\n";
    std::cout << "  🏭 Vendor ID  : 0x" << std::hex << props.vendorID << std::dec << "\n";

    // 打印重要的 Limits
    const auto& lim = props.limits;
    std::cout << "  📐 最大纹理   : " << lim.maxImageDimension2D << "x" << lim.maxImageDimension2D << "\n";
    std::cout << "  📦 最大UBO    : " << lim.maxUniformBufferRange << " bytes\n";
    std::cout << "  🔧 最大SSBO   : " << lim.maxStorageBufferRange << " bytes\n";

    // 打印重要功能
    std::cout << "  ✨ 功能支持   :\n";
    std::cout << "    几何着色器  : " << (features.geometryShader ? "✅" : "❌") << "\n";
    std::cout << "    曲面细分    : " << (features.tessellationShader ? "✅" : "❌") << "\n";
    std::cout << "    各向异性过滤: " << (features.samplerAnisotropy ? "✅" : "❌") << "\n";
    std::cout << "    多重采样    : " << (features.sampleRateShading ? "✅" : "❌") << "\n";
    std::cout << "    64位浮点    : " << (features.shaderFloat64 ? "✅" : "❌") << "\n";
}

/// 打印设备队列族信息
static void printQueueFamilies(VkPhysicalDevice device) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());

    std::cout << "  🚀 队列族（共 " << count << " 个）：\n";
    for (uint32_t i = 0; i < count; ++i) {
        const auto& qf = families[i];
        std::cout << "    [" << i << "] 数量=" << qf.queueCount << " 支持: ";
        if (qf.queueFlags & VK_QUEUE_GRAPHICS_BIT)
            std::cout << "Graphics ";
        if (qf.queueFlags & VK_QUEUE_COMPUTE_BIT)
            std::cout << "Compute ";
        if (qf.queueFlags & VK_QUEUE_TRANSFER_BIT)
            std::cout << "Transfer ";
        if (qf.queueFlags & VK_QUEUE_SPARSE_BINDING_BIT)
            std::cout << "Sparse ";
        std::cout << "\n";
    }
}

// ─── 设备评分 ──────────────────────────────────────────────────────────────────

/**
 * @brief 对物理设备进行评分
 *
 * 不满足基本要求的设备得 0 分（被淘汰）。
 * 满足要求的设备根据特性获得更高分数。
 *
 * @return 设备评分，0 表示不可用
 */
static uint32_t scoreDevice(VkPhysicalDevice device) {
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceProperties(device, &props);
    vkGetPhysicalDeviceFeatures(device, &features);

    uint32_t score = 0;

    // ─── 独立 GPU 有极大优势 ───────────────────────────────────────────────
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        score += 1000;
    else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
        score += 100;

    // ─── 纹理越大越好 ──────────────────────────────────────────────────────
    score += props.limits.maxImageDimension2D;

    // ─── 各向异性过滤是我们需要的功能 ────────────────────────────────────
    if (features.samplerAnisotropy)
        score += 100;

    return score;
}

// ─── 应用程序类 ────────────────────────────────────────────────────────────────

class Ch02App {
  public:
    void run() {
        initVulkan();
        selectPhysicalDevice();
        cleanup();
    }

  private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;

    void initVulkan() {
        // 初始化 GLFW（仅用于获取需要的扩展）
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        if (ENABLE_VALIDATION_LAYERS && !checkValidationLayerSupport())
            throw std::runtime_error("验证层不可用");

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.apiVersion = VK_API_VERSION_1_3;

        auto extensions = getRequiredInstanceExtensions();

        VkInstanceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo = &appInfo;
        ci.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        ci.ppEnabledExtensionNames = extensions.data();
#ifdef __APPLE__
#ifdef __APPLE__
        ci.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

#endif

        if (ENABLE_VALIDATION_LAYERS) {
            ci.enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
            ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();
        }

        VK_CHECK(vkCreateInstance(&ci, nullptr, &instance_));
    }

    void selectPhysicalDevice() {
        // ① 枚举所有物理设备
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);

        if (deviceCount == 0)
            throw std::runtime_error("没有找到支持 Vulkan 的 GPU！");

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

        std::cout << "🔍 找到 " << deviceCount << " 个支持 Vulkan 的设备：\n\n";

        // ② 打印所有设备信息并评分
        std::multimap<uint32_t, VkPhysicalDevice, std::greater<uint32_t>> candidates;

        for (size_t i = 0; i < devices.size(); ++i) {
            std::cout << "── 设备 #" << i << " ──────────────────────────────\n";
            printDeviceInfo(devices[i]);
            printQueueFamilies(devices[i]);

            uint32_t score = scoreDevice(devices[i]);
            candidates.insert({score, devices[i]});
            std::cout << "  🏆 评分: " << score << "\n\n";
        }

        // ③ 选择评分最高且满足基本要求的设备
        for (auto& [score, device] : candidates) {
            if (score > 0) {
                physicalDevice_ = device;
                break;
            }
        }

        if (physicalDevice_ == VK_NULL_HANDLE)
            throw std::runtime_error("没有找到合适的 GPU！");

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice_, &props);
        std::cout << "✅ 已选择设备：" << props.deviceName << "\n";
    }

    void cleanup() {
        // 注意：VkPhysicalDevice 不需要手动销毁
        // 它随 VkInstance 的销毁自动释放
        vkDestroyInstance(instance_, nullptr);
        glfwTerminate();
        std::cout << "✅ 清理完成。\n";
    }
};

int main() {
    std::cout << "═══════════════════════════════════════\n";
    std::cout << " 第02章：物理设备选择与队列族\n";
    std::cout << "═══════════════════════════════════════\n\n";

    Ch02App app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ 错误：" << e.what() << "\n";
        return 1;
    }
    return 0;
}
