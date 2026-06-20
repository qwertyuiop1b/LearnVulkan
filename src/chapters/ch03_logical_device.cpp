/**
 * @file ch03_logical_device.cpp
 * @brief 第03章：逻辑设备与队列创建
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【核心概念】
 *
 * VkDevice（逻辑设备）是应用程序与物理设备交互的接口。
 * 一个物理设备可以创建多个逻辑设备（各自有独立的资源和状态）。
 *
 * 【物理设备 vs 逻辑设备】
 *
 *  VkPhysicalDevice ─── 代表硬件本身（只读，用于查询属性）
 *  VkDevice         ─── 我们的"工作空间"（用于创建资源、提交命令）
 *
 * 【Queue（队列）】
 *
 *  逻辑设备创建时，我们指定需要哪些队列。
 *  创建后通过 vkGetDeviceQueue() 获取队列句柄。
 *  队列是提交命令缓冲区（CommandBuffer）的入口。
 *
 * 【创建流程】
 *
 *  ┌──────────────────────────────────────────────────────────┐
 *  │  ① 找到队列族索引（graphicsFamily, presentFamily）        │
 *  │  ② VkDeviceQueueCreateInfo  ─→  描述要创建的队列         │
 *  │  ③ VkDeviceCreateInfo       ─→  描述逻辑设备配置         │
 *  │  ④ vkCreateDevice()         ─→  创建逻辑设备             │
 *  │  ⑤ vkGetDeviceQueue()       ─→  获取队列句柄             │
 *  └──────────────────────────────────────────────────────────┘
 * ═══════════════════════════════════════════════════════════════════════════
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan_tutorial/utils.hpp>

#include <iostream>
#include <set>
#include <stdexcept>

class Ch03App {
  public:
    void run() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        createInstance();
        pickPhysicalDevice();
        createLogicalDevice();
        printQueueInfo();
        cleanup();
    }

  private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue graphicsQueue_ = VK_NULL_HANDLE;
    VkQueue presentQueue_ = VK_NULL_HANDLE;

    // 用一个假的 surface 来查询 presentFamily（真实代码需要真实窗口）
    // 本章先用 graphicsFamily 代替 presentFamily 演示
    uint32_t graphicsFamilyIndex_ = 0;

    void createInstance() {
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

    void pickPhysicalDevice() {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance_, &count, nullptr);
        if (count == 0)
            throw std::runtime_error("没有支持 Vulkan 的 GPU");

        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance_, &count, devices.data());

        for (auto& d : devices) {
            // 找到支持图形操作的队列族
            uint32_t qfCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(d, &qfCount, nullptr);
            std::vector<VkQueueFamilyProperties> qfs(qfCount);
            vkGetPhysicalDeviceQueueFamilyProperties(d, &qfCount, qfs.data());

            for (uint32_t i = 0; i < qfCount; ++i) {
                if (qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                    physicalDevice_ = d;
                    graphicsFamilyIndex_ = i;
                    break;
                }
            }
            if (physicalDevice_ != VK_NULL_HANDLE)
                break;
        }

        if (physicalDevice_ == VK_NULL_HANDLE)
            throw std::runtime_error("没有找到合适的 GPU");

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice_, &props);
        std::cout << "✅ 使用设备：" << props.deviceName << "  (图形队列族索引=" << graphicsFamilyIndex_ << ")\n\n";
    }

    void createLogicalDevice() {
        // ─── ① 指定需要创建的队列 ────────────────────────────────────────────
        //
        // 即使我们需要多个不同类型的队列族（如图形+传输），
        // 如果它们属于同一个队列族，只需创建一次。
        // 用 set 去重，避免同一队列族被重复请求。
        std::set<uint32_t> uniqueFamilies = {graphicsFamilyIndex_};

        // 队列优先级：0.0f ~ 1.0f，影响 GPU 调度，同一设备的多个队列间有效
        const float queuePriority = 1.0f;

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        for (uint32_t family : uniqueFamilies) {
            VkDeviceQueueCreateInfo qci{};
            qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            qci.queueFamilyIndex = family;
            qci.queueCount = 1; // 通常每族创建 1 个队列就够
            qci.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(qci);
        }

        // ─── ② 指定需要的设备功能 ────────────────────────────────────────────
        VkPhysicalDeviceFeatures deviceFeatures{};
        deviceFeatures.samplerAnisotropy = VK_TRUE; // 启用各向异性过滤

        // ─── ③ 创建逻辑设备 ──────────────────────────────────────────────────
        VkDeviceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        ci.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        ci.pQueueCreateInfos = queueCreateInfos.data();
        ci.pEnabledFeatures = &deviceFeatures;

        // 设备扩展（如交换链扩展）
        // 注意：DEVICE_EXTENSIONS 包含 VK_KHR_portability_subset，
        //       macOS 上 MoltenVK 需要这个扩展
        ci.enabledExtensionCount = static_cast<uint32_t>(DEVICE_EXTENSIONS.size());
        ci.ppEnabledExtensionNames = DEVICE_EXTENSIONS.data();

        // 旧版 Vulkan 区分 instance 层和 device 层，现代 Vulkan 忽略设备层，
        // 但为了兼容旧驱动，仍然设置验证层
        if (ENABLE_VALIDATION_LAYERS) {
            ci.enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
            ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();
        }

        VK_CHECK(vkCreateDevice(physicalDevice_, &ci, nullptr, &device_));

        // ─── ④ 获取队列句柄 ──────────────────────────────────────────────────
        // 参数：(逻辑设备, 队列族索引, 队列索引（第几个）, 输出句柄)
        vkGetDeviceQueue(device_, graphicsFamilyIndex_, 0, &graphicsQueue_);
        presentQueue_ = graphicsQueue_; // 本章简化：复用同一队列

        std::cout << "✅ 逻辑设备创建成功！\n";
        std::cout << "✅ 图形队列句柄已获取：" << graphicsQueue_ << "\n";
    }

    void printQueueInfo() {
        std::cout << "\n📋 逻辑设备信息：\n";

        // 枚举设备扩展
        uint32_t extCount = 0;
        vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> exts(extCount);
        vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &extCount, exts.data());

        std::cout << "  支持的设备扩展（共 " << extCount << " 个）：\n";
        for (const auto& e : exts)
            if (std::string(e.extensionName).find("KHR") != std::string::npos)
                std::cout << "    - " << e.extensionName << "\n";

        // 查询内存属性
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProps);

        std::cout << "\n  💾 内存堆信息（共 " << memProps.memoryHeapCount << " 个堆）：\n";
        for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i) {
            const auto& heap = memProps.memoryHeaps[i];
            std::cout << "    堆[" << i << "]: " << (heap.size / 1024 / 1024) << " MB";
            if (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                std::cout << " [GPU本地，高速]";
            std::cout << "\n";
        }

        std::cout << "\n  🗂️  内存类型（共 " << memProps.memoryTypeCount << " 种）：\n";
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            const auto& mt = memProps.memoryTypes[i];
            std::cout << "    类型[" << i << "] 堆=" << mt.heapIndex << " 属性: ";
            if (mt.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                std::cout << "DeviceLocal ";
            if (mt.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
                std::cout << "HostVisible ";
            if (mt.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                std::cout << "HostCoherent ";
            if (mt.propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
                std::cout << "HostCached ";
            std::cout << "\n";
        }
    }

    void cleanup() {
        vkDestroyDevice(device_, nullptr); // 销毁逻辑设备（队列随之自动销毁）
        vkDestroyInstance(instance_, nullptr);
        glfwTerminate();
        std::cout << "\n✅ 清理完成。\n";
    }
};

int main() {
    std::cout << "═══════════════════════════════════════\n";
    std::cout << " 第03章：逻辑设备与队列创建\n";
    std::cout << "═══════════════════════════════════════\n\n";

    Ch03App app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ 错误：" << e.what() << "\n";
        return 1;
    }
    return 0;
}
