/**
 * @file rhi_device.cpp
 * @brief 第61章：RHIDevice 实现 — Vulkan 设备抽象层
 *
 * 封装 ch01-ch03 的设备创建全流程：
 *   Instance → Surface → Physical Device → Logical Device → Queues → Command Pool
 *
 * 设计原则：
 *   - 复用 utils.hpp / vk_helpers.hpp 中已有的工具函数
 *   - RHIDevice 只负责生命周期管理，不重复实现底层逻辑
 *   - 通过 enabledFeatureMask_ 位域记录已启用特性
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan_tutorial/engine/rhi_device.hpp>
#include <vulkan_tutorial/utils.hpp>
#include <vulkan_tutorial/vk_helpers.hpp>
#include <vulkan_tutorial/features.hpp>

#include <set>
#include <stdexcept>
#include <string>

namespace engine {

// ─── Move 语义 ───────────────────────────────────────────────────────────────

RHIDevice::RHIDevice(RHIDevice&& o) noexcept
    : instance_(o.instance_), surface_(o.surface_), physDev_(o.physDev_), device_(o.device_), cmdPool_(o.cmdPool_),
      graphics_(o.graphics_), compute_(o.compute_), transfer_(o.transfer_), depthFmt_(o.depthFmt_),
      deviceName_(std::move(o.deviceName_)), isDiscrete_(o.isDiscrete_), videoMemBytes_(o.videoMemBytes_),
      enabledFeatureMask_(o.enabledFeatureMask_), debugMsg_(o.debugMsg_) {
    o.instance_ = VK_NULL_HANDLE;
    o.surface_ = VK_NULL_HANDLE;
    o.physDev_ = VK_NULL_HANDLE;
    o.device_ = VK_NULL_HANDLE;
    o.cmdPool_ = VK_NULL_HANDLE;
    o.debugMsg_ = VK_NULL_HANDLE;
}

RHIDevice& RHIDevice::operator=(RHIDevice&& o) noexcept {
    if (this != &o) {
        destroy();
        instance_ = o.instance_;
        surface_ = o.surface_;
        physDev_ = o.physDev_;
        device_ = o.device_;
        cmdPool_ = o.cmdPool_;
        graphics_ = o.graphics_;
        compute_ = o.compute_;
        transfer_ = o.transfer_;
        depthFmt_ = o.depthFmt_;
        deviceName_ = std::move(o.deviceName_);
        isDiscrete_ = o.isDiscrete_;
        videoMemBytes_ = o.videoMemBytes_;
        enabledFeatureMask_ = o.enabledFeatureMask_;
        debugMsg_ = o.debugMsg_;
        o.instance_ = VK_NULL_HANDLE;
        o.surface_ = VK_NULL_HANDLE;
        o.physDev_ = VK_NULL_HANDLE;
        o.device_ = VK_NULL_HANDLE;
        o.cmdPool_ = VK_NULL_HANDLE;
        o.debugMsg_ = VK_NULL_HANDLE;
    }
    return *this;
}

// ─── init ───────────────────────────────────────────────────────────────────

void RHIDevice::init(const DeviceCreateInfo& ci) {
    // 1. 创建 VkInstance（复用工具函数）
    vulkan_tutorial::createInstance(instance_);

    // 2. 设置验证层调试信使
    if (ci.enableValidation && ENABLE_VALIDATION_LAYERS) {
        VkDebugUtilsMessengerCreateInfoEXT msgCI{};
        populateDebugMessengerCreateInfo(msgCI);
        auto createFn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
        if (createFn)
            createFn(instance_, &msgCI, nullptr, &debugMsg_);
    }

    // 3. 创建 Surface（GLFWwindow* → VkSurfaceKHR）
    auto* window = static_cast<GLFWwindow*>(ci.windowHandle);
    if (window)
        VK_CHECK(glfwCreateWindowSurface(instance_, window, nullptr, &surface_));

    // 4. 选择物理设备
    pickPhysicalDevice(ci.preferDiscreteGpu);

    // 5. 创建逻辑设备（开启 required + optional 特性）
    createLogicalDevice(ci.requiredFeatures, ci.optionalFeatures);

    // 6. 创建命令池（允许单条命令 Reset）
    QueueFamilyIndices qfi = findQueueFamilies(physDev_, surface_);
    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.queueFamilyIndex = qfi.graphicsFamily.value();
    poolCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(device_, &poolCI, nullptr, &cmdPool_));

    // 7. 检测并缓存设备能力
    detectCapabilities();

    // 8. 缓存深度格式
    cacheDepthFormat();
}

// ─── destroy ────────────────────────────────────────────────────────────────

void RHIDevice::destroy() {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
        if (cmdPool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device_, cmdPool_, nullptr);
            cmdPool_ = VK_NULL_HANDLE;
        }
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }
    if (surface_ != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        surface_ = VK_NULL_HANDLE;
    }
    if (debugMsg_ != VK_NULL_HANDLE) {
        auto destroyFn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroyFn)
            destroyFn(instance_, debugMsg_, nullptr);
        debugMsg_ = VK_NULL_HANDLE;
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
    physDev_ = VK_NULL_HANDLE;
    graphics_ = {};
    compute_ = {};
    transfer_ = {};
}

// ─── pickPhysicalDevice ──────────────────────────────────────────────────────

void RHIDevice::pickPhysicalDevice(bool preferDiscrete) {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (count == 0)
        throw std::runtime_error("[RHIDevice] 未找到支持 Vulkan 的 GPU");

    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance_, &count, devices.data());

    VkPhysicalDevice fallback = VK_NULL_HANDLE;

    for (VkPhysicalDevice dev : devices) {
        if (!findQueueFamilies(dev, surface_).isComplete())
            continue;
        if (!checkDeviceExtensionSupport(dev))
            continue;

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(dev, &props);

        if (preferDiscrete && props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            physDev_ = dev;
            return;
        }
        if (fallback == VK_NULL_HANDLE)
            fallback = dev;
    }

    if (fallback != VK_NULL_HANDLE)
        physDev_ = fallback;
    else
        throw std::runtime_error("[RHIDevice] 找不到满足要求的 GPU");
}

// ─── createLogicalDevice ────────────────────────────────────────────────────

void RHIDevice::createLogicalDevice(const std::vector<DeviceFeature>& req, const std::vector<DeviceFeature>& opt) {
    VkPhysicalDeviceFeatures supported{};
    vkGetPhysicalDeviceFeatures(physDev_, &supported);

    VkPhysicalDeviceFeatures enabled{};

    // 根据 DeviceFeature 枚举开启对应的 VkPhysicalDeviceFeatures 字段
    auto tryEnable = [&](DeviceFeature f, bool required) -> bool {
        bool available = false;
        switch (f) {
        case DeviceFeature::GeometryShader:
            available = supported.geometryShader == VK_TRUE;
            if (available)
                enabled.geometryShader = VK_TRUE;
            break;
        case DeviceFeature::TessellationShader:
            available = supported.tessellationShader == VK_TRUE;
            if (available)
                enabled.tessellationShader = VK_TRUE;
            break;
        case DeviceFeature::WideLines:
            available = supported.wideLines == VK_TRUE;
            if (available)
                enabled.wideLines = VK_TRUE;
            break;
        case DeviceFeature::FillModeNonSolid:
            available = supported.fillModeNonSolid == VK_TRUE;
            if (available)
                enabled.fillModeNonSolid = VK_TRUE;
            break;
        case DeviceFeature::SamplerAnisotropy:
            available = supported.samplerAnisotropy == VK_TRUE;
            if (available)
                enabled.samplerAnisotropy = VK_TRUE;
            break;
        case DeviceFeature::MultiDrawIndirect:
            available = supported.multiDrawIndirect == VK_TRUE;
            if (available)
                enabled.multiDrawIndirect = VK_TRUE;
            break;
        case DeviceFeature::DrawIndirectFirstInstance:
            available = supported.drawIndirectFirstInstance == VK_TRUE;
            if (available)
                enabled.drawIndirectFirstInstance = VK_TRUE;
            break;
        case DeviceFeature::ShaderClipDistance:
            available = supported.shaderClipDistance == VK_TRUE;
            if (available)
                enabled.shaderClipDistance = VK_TRUE;
            break;
        case DeviceFeature::TimestampQueries:
            available = true; // 通过 QueryPool 支持，不影响 features
            break;
        case DeviceFeature::MeshShader:
        case DeviceFeature::RayTracing:
            available = false; // 需要扩展，此处简化跳过
            break;
        }
        if (available)
            enabledFeatureMask_ |= (1u << static_cast<uint32_t>(f));
        else if (required)
            throw std::runtime_error("[RHIDevice] 必需特性不支持: " + std::to_string(static_cast<uint32_t>(f)));
        return available;
    };

    for (auto f : req)
        tryEnable(f, true);
    for (auto f : opt)
        tryEnable(f, false);

    // 构建队列创建信息
    QueueFamilyIndices qfi = findQueueFamilies(physDev_, surface_);
    std::set<uint32_t> uniqueFamilies = {qfi.graphicsFamily.value(), qfi.presentFamily.value()};
    const float priority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueCIs;
    for (uint32_t family : uniqueFamilies) {
        VkDeviceQueueCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &priority;
        queueCIs.push_back(qci);
    }

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = static_cast<uint32_t>(queueCIs.size());
    dci.pQueueCreateInfos = queueCIs.data();
    dci.pEnabledFeatures = &enabled;
    dci.enabledExtensionCount = static_cast<uint32_t>(DEVICE_EXTENSIONS.size());
    dci.ppEnabledExtensionNames = DEVICE_EXTENSIONS.data();
    if (ENABLE_VALIDATION_LAYERS) {
        dci.enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
        dci.ppEnabledLayerNames = VALIDATION_LAYERS.data();
    }
    VK_CHECK(vkCreateDevice(physDev_, &dci, nullptr, &device_));

    // 获取队列句柄
    vkGetDeviceQueue(device_, qfi.graphicsFamily.value(), 0, &graphics_.handle);
    graphics_.familyIndex = qfi.graphicsFamily.value();

    // 检查时间戳支持
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physDev_, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfProps(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physDev_, &qfCount, qfProps.data());
    if (qfi.graphicsFamily.value() < qfCount) {
        graphics_.supportsTimestamps = qfProps[qfi.graphicsFamily.value()].timestampValidBits > 0;
    }

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physDev_, &props);
    graphics_.timestampPeriodNs = props.limits.timestampPeriod;

    // 大多数硬件：图形队列同时支持计算和传输
    compute_ = graphics_;
    transfer_ = graphics_;
}

// ─── detectCapabilities ─────────────────────────────────────────────────────

void RHIDevice::detectCapabilities() {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physDev_, &props);
    deviceName_ = props.deviceName;
    isDiscrete_ = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);

    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(physDev_, &memProps);

    videoMemBytes_ = 0;
    for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i) {
        if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            videoMemBytes_ += memProps.memoryHeaps[i].size;
    }
}

// ─── cacheDepthFormat ────────────────────────────────────────────────────────

void RHIDevice::cacheDepthFormat() {
    depthFmt_ = findSupportedFormat({VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
                                    VK_IMAGE_TILING_OPTIMAL,
                                    VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

// ─── 公共工具方法 ────────────────────────────────────────────────────────────

bool RHIDevice::supportsFeature(DeviceFeature f) const {
    return (enabledFeatureMask_ & (1u << static_cast<uint32_t>(f))) != 0;
}

uint32_t RHIDevice::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) const {
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(physDev_, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    throw std::runtime_error("[RHIDevice] 找不到合适的内存类型");
}

VkFormat RHIDevice::findSupportedFormat(const std::vector<VkFormat>& candidates,
                                        VkImageTiling tiling,
                                        VkFormatFeatureFlags features) const {
    for (VkFormat fmt : candidates) {
        VkFormatProperties p{};
        vkGetPhysicalDeviceFormatProperties(physDev_, fmt, &p);
        bool ok = (tiling == VK_IMAGE_TILING_LINEAR) ? (p.linearTilingFeatures & features) == features
                                                     : (p.optimalTilingFeatures & features) == features;
        if (ok)
            return fmt;
    }
    throw std::runtime_error("[RHIDevice] 找不到支持的图像格式");
}

// ─── 单次命令缓冲区 ──────────────────────────────────────────────────────────

VkCommandBuffer RHIDevice::beginOneShot() {
    VkCommandBufferAllocateInfo allocCI{};
    allocCI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocCI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocCI.commandPool = cmdPool_;
    allocCI.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(device_, &allocCI, &cmd));

    VkCommandBufferBeginInfo beginCI{};
    beginCI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginCI.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginCI));
    return cmd;
}

void RHIDevice::endOneShot(VkCommandBuffer cmd) {
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo submitCI{};
    submitCI.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitCI.commandBufferCount = 1;
    submitCI.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(graphics_.handle, 1, &submitCI, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(graphics_.handle));

    vkFreeCommandBuffers(device_, cmdPool_, 1, &cmd);
}

} // namespace engine
