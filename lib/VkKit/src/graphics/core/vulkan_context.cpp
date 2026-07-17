#include <graphics/core/vulkan_context.hpp>

#ifndef GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_NONE
#endif
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace vulkan_graphics {
namespace {

constexpr std::array<const char*, 1> kValidationLayers = {"VK_LAYER_KHRONOS_validation"};

bool hasValidationLayerSupport() {
    uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    return std::all_of(kValidationLayers.begin(), kValidationLayers.end(), [&](const char* requiredLayer) {
        return std::any_of(availableLayers.begin(), availableLayers.end(), [&](const VkLayerProperties& availableLayer) {
            return std::strcmp(requiredLayer, availableLayer.layerName) == 0;
        });
    });
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT,
                                             const VkDebugUtilsMessengerCallbackDataEXT* data,
                                             void*) {
    const char* prefix = "[VK]";
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        prefix = "[VK ERROR]";
    else if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        prefix = "[VK WARN]";

    std::cerr << prefix << ' ' << data->pMessage << '\n';
    return VK_FALSE;
}

struct QueueFamilies {
    uint32_t graphics = VK_QUEUE_FAMILY_IGNORED;
    uint32_t present = VK_QUEUE_FAMILY_IGNORED;
    uint32_t compute = VK_QUEUE_FAMILY_IGNORED;

    [[nodiscard]] bool isComplete() const noexcept {
        return graphics != VK_QUEUE_FAMILY_IGNORED && present != VK_QUEUE_FAMILY_IGNORED;
    }
};

QueueFamilies findQueueFamilies(const vk::raii::PhysicalDevice& physicalDevice, vk::SurfaceKHR surface) {
    QueueFamilies result{};
    const auto families = physicalDevice.getQueueFamilyProperties();

    for (uint32_t index = 0; index < families.size(); ++index) {
        const auto flags = families[index].queueFlags;
        if (result.graphics == VK_QUEUE_FAMILY_IGNORED && (flags & vk::QueueFlagBits::eGraphics))
            result.graphics = index;

        if (flags & vk::QueueFlagBits::eCompute) {
            const bool isDedicated = !(flags & vk::QueueFlagBits::eGraphics);
            if (result.compute == VK_QUEUE_FAMILY_IGNORED || isDedicated)
                result.compute = index;
        }

        if (physicalDevice.getSurfaceSupportKHR(index, surface))
            result.present = index;
    }

    if (result.compute == VK_QUEUE_FAMILY_IGNORED)
        result.compute = result.graphics;
    return result;
}

bool supportsDeviceExtensions(const vk::raii::PhysicalDevice& physicalDevice,
                              const std::vector<const char*>& requiredExtensions) {
    std::set<std::string> missing(requiredExtensions.begin(), requiredExtensions.end());
    for (const auto& extension : physicalDevice.enumerateDeviceExtensionProperties())
        missing.erase(extension.extensionName);
    return missing.empty();
}

bool supportsSwapchain(const vk::raii::PhysicalDevice& physicalDevice, vk::SurfaceKHR surface) {
    return !physicalDevice.getSurfaceFormatsKHR(surface).empty() &&
           !physicalDevice.getSurfacePresentModesKHR(surface).empty();
}

VkFormat chooseDepthFormat(const vk::raii::PhysicalDevice& physicalDevice) {
    constexpr VkFormat candidates[] = {
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_FORMAT_D24_UNORM_S8_UINT,
    };

    for (const VkFormat candidate : candidates) {
        const auto properties = physicalDevice.getFormatProperties(static_cast<vk::Format>(candidate));
        if ((properties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment) ==
            vk::FormatFeatureFlagBits::eDepthStencilAttachment)
            return candidate;
    }

    throw std::runtime_error("No supported depth-stencil attachment format");
}

int scoreDevice(const vk::raii::PhysicalDevice& physicalDevice, bool preferDiscreteGpu) {
    const auto properties = physicalDevice.getProperties();
    int score = static_cast<int>(properties.limits.maxImageDimension2D);
    if (preferDiscreteGpu && properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
        score += 100000;
    return score;
}

bool supportsRequiredVulkan13Features(const vk::raii::PhysicalDevice& physicalDevice,
                                      bool requireDynamicRendering,
                                      bool requireSynchronization2) {
    VkPhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeatures{};
    dynamicRenderingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
    VkPhysicalDeviceSynchronization2Features synchronization2Features{};
    synchronization2Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;

    VkPhysicalDeviceFeatures2 features{};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    if (requireDynamicRendering)
        features.pNext = &dynamicRenderingFeatures;
    if (requireSynchronization2) {
        synchronization2Features.pNext = features.pNext;
        features.pNext = &synchronization2Features;
    }
    vkGetPhysicalDeviceFeatures2(static_cast<VkPhysicalDevice>(*physicalDevice), &features);
    return (!requireDynamicRendering || dynamicRenderingFeatures.dynamicRendering == VK_TRUE) &&
           (!requireSynchronization2 || synchronization2Features.synchronization2 == VK_TRUE);
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallbackHpp(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
                                                vk::DebugUtilsMessageTypeFlagsEXT type,
                                                const vk::DebugUtilsMessengerCallbackDataEXT* data,
                                                void* userData) {
    return debugCallback(static_cast<VkDebugUtilsMessageSeverityFlagBitsEXT>(severity),
                         static_cast<VkDebugUtilsMessageTypeFlagsEXT>(type),
                         reinterpret_cast<const VkDebugUtilsMessengerCallbackDataEXT*>(data), userData);
}

} // namespace

struct VulkanContext::Impl {
    vk::raii::Context context{};
    vk::raii::Instance instance{nullptr};
    vk::raii::DebugUtilsMessengerEXT debugMessenger{nullptr};
    vk::raii::SurfaceKHR surface{nullptr};
    vk::raii::PhysicalDevice physicalDevice{nullptr};
    vk::raii::Device device{nullptr};

    Queue graphicsQueue{};
    Queue presentQueue{};
    Queue computeQueue{};
    vk::PhysicalDeviceProperties properties{};
    vk::PhysicalDeviceMemoryProperties memoryProperties{};
    VkFormat depthFormat = VK_FORMAT_UNDEFINED;
    uint32_t apiVersion = VK_API_VERSION_1_0;
    bool validationEnabled = false;
    bool dynamicRenderingEnabled = false;
    bool synchronization2Enabled = false;
    bool samplerAnisotropyEnabled = false;
};

VulkanContext::VulkanContext(const VulkanContextCreateInfo& createInfo) {
    if (createInfo.window == nullptr)
        throw std::invalid_argument("VulkanContext requires a GLFW window");

    auto impl = std::make_unique<Impl>();
    impl->validationEnabled = createInfo.enableValidation;
    if (impl->validationEnabled && !hasValidationLayerSupport())
        throw std::runtime_error("Requested Vulkan validation layers are unavailable");
    if (impl->context.enumerateInstanceVersion() < createInfo.apiVersion)
        throw std::runtime_error("Requested Vulkan API version is unavailable");
    if ((createInfo.requireDynamicRendering || createInfo.requireSynchronization2) &&
        createInfo.apiVersion < VK_API_VERSION_1_3) {
        throw std::invalid_argument("Dynamic rendering and Synchronization2 require Vulkan API version 1.3 or later");
    }
    impl->apiVersion = createInfo.apiVersion;

    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    if (glfwExtensions == nullptr)
        throw std::runtime_error("GLFW did not report required Vulkan instance extensions");

    std::vector<const char*> instanceExtensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    if (impl->validationEnabled)
        instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#ifdef __APPLE__
    instanceExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    instanceExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#endif

    vk::ApplicationInfo applicationInfo{};
    applicationInfo.pApplicationName = createInfo.applicationName.c_str();
    applicationInfo.applicationVersion = createInfo.applicationVersion;
    applicationInfo.pEngineName = "LearnVulkan";
    applicationInfo.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    applicationInfo.apiVersion = createInfo.apiVersion;

    vk::InstanceCreateInfo instanceInfo{};
    instanceInfo.pApplicationInfo = &applicationInfo;
    instanceInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size());
    instanceInfo.ppEnabledExtensionNames = instanceExtensions.data();
    if (impl->validationEnabled) {
        instanceInfo.enabledLayerCount = static_cast<uint32_t>(kValidationLayers.size());
        instanceInfo.ppEnabledLayerNames = kValidationLayers.data();
    }
#ifdef __APPLE__
    instanceInfo.flags = vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
#endif
    impl->instance = vk::raii::Instance{impl->context, instanceInfo};

    if (impl->validationEnabled) {
        vk::DebugUtilsMessengerCreateInfoEXT debugInfo{};
        debugInfo.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                                    vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
        debugInfo.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                                vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                                vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
        debugInfo.pfnUserCallback = debugCallbackHpp;
        impl->debugMessenger = vk::raii::DebugUtilsMessengerEXT{impl->instance, debugInfo};
    }

    VkSurfaceKHR rawSurface = VK_NULL_HANDLE;
    const VkResult surfaceResult =
        glfwCreateWindowSurface(static_cast<VkInstance>(*impl->instance), createInfo.window, nullptr, &rawSurface);
    if (surfaceResult != VK_SUCCESS)
        throw std::runtime_error("GLFW failed to create the Vulkan surface");
    impl->surface = vk::raii::SurfaceKHR{impl->instance, rawSurface};

    int bestScore = std::numeric_limits<int>::min();
    for (auto& candidate : impl->instance.enumeratePhysicalDevices()) {
        const QueueFamilies families = findQueueFamilies(candidate, *impl->surface);
        if (!families.isComplete() || !supportsDeviceExtensions(candidate, createInfo.requiredDeviceExtensions) ||
            !supportsSwapchain(candidate, *impl->surface))
            continue;

        const auto supportedFeatures = candidate.getFeatures();
        if (createInfo.requireSamplerAnisotropy && !supportedFeatures.samplerAnisotropy)
            continue;
        if (!supportsRequiredVulkan13Features(candidate,
                                              createInfo.requireDynamicRendering,
                                              createInfo.requireSynchronization2)) {
            continue;
        }

        const int score = scoreDevice(candidate, createInfo.preferDiscreteGpu);
        if (score > bestScore) {
            bestScore = score;
            impl->physicalDevice = std::move(candidate);
        }
    }

    if (static_cast<VkPhysicalDevice>(*impl->physicalDevice) == VK_NULL_HANDLE)
        throw std::runtime_error("No suitable Vulkan physical device was found");

    const QueueFamilies families = findQueueFamilies(impl->physicalDevice, *impl->surface);
    std::set<uint32_t> uniqueFamilies = {families.graphics, families.present, families.compute};
    constexpr float queuePriority = 1.0f;
    std::vector<vk::DeviceQueueCreateInfo> queueInfos;
    queueInfos.reserve(uniqueFamilies.size());
    for (const uint32_t family : uniqueFamilies) {
        vk::DeviceQueueCreateInfo queueInfo{};
        queueInfo.queueFamilyIndex = family;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &queuePriority;
        queueInfos.push_back(queueInfo);
    }

    const auto supportedFeatures = impl->physicalDevice.getFeatures();
    vk::PhysicalDeviceFeatures enabledFeatures{};
    if (createInfo.requireSamplerAnisotropy)
        enabledFeatures.samplerAnisotropy = supportedFeatures.samplerAnisotropy;
    impl->samplerAnisotropyEnabled = createInfo.requireSamplerAnisotropy;

    VkPhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeatures{};
    dynamicRenderingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
    dynamicRenderingFeatures.dynamicRendering = createInfo.requireDynamicRendering;
    impl->dynamicRenderingEnabled = createInfo.requireDynamicRendering;

    VkPhysicalDeviceSynchronization2Features synchronization2Features{};
    synchronization2Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;
    synchronization2Features.synchronization2 = createInfo.requireSynchronization2;
    synchronization2Features.pNext = createInfo.requireDynamicRendering ? &dynamicRenderingFeatures : nullptr;
    impl->synchronization2Enabled = createInfo.requireSynchronization2;

    vk::DeviceCreateInfo deviceInfo{};
    deviceInfo.queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size());
    deviceInfo.pQueueCreateInfos = queueInfos.data();
    deviceInfo.enabledExtensionCount = static_cast<uint32_t>(createInfo.requiredDeviceExtensions.size());
    deviceInfo.ppEnabledExtensionNames = createInfo.requiredDeviceExtensions.data();
    deviceInfo.pEnabledFeatures = &enabledFeatures;
    void* enabledFeatureChain = nullptr;
    if (createInfo.requireDynamicRendering)
        enabledFeatureChain = &dynamicRenderingFeatures;
    if (createInfo.requireSynchronization2)
        enabledFeatureChain = &synchronization2Features;
    deviceInfo.pNext = enabledFeatureChain;
    impl->device = vk::raii::Device{impl->physicalDevice, deviceInfo};

    const auto queueProperties = impl->physicalDevice.getQueueFamilyProperties();
    impl->graphicsQueue = {*impl->device.getQueue(families.graphics, 0), families.graphics,
                            queueProperties[families.graphics].timestampValidBits > 0};
    impl->presentQueue = {*impl->device.getQueue(families.present, 0), families.present,
                           queueProperties[families.present].timestampValidBits > 0};
    impl->computeQueue = {*impl->device.getQueue(families.compute, 0), families.compute,
                           queueProperties[families.compute].timestampValidBits > 0};
    impl->properties = impl->physicalDevice.getProperties();
    impl->memoryProperties = impl->physicalDevice.getMemoryProperties();
    impl->depthFormat = chooseDepthFormat(impl->physicalDevice);

    impl_ = std::move(impl);
}

VulkanContext::~VulkanContext() = default;
VulkanContext::VulkanContext(VulkanContext&&) noexcept = default;
VulkanContext& VulkanContext::operator=(VulkanContext&&) noexcept = default;

vk::Instance VulkanContext::instance() const noexcept {
    return impl_ ? *impl_->instance : vk::Instance{};
}

vk::SurfaceKHR VulkanContext::surface() const noexcept {
    return impl_ ? *impl_->surface : vk::SurfaceKHR{};
}

vk::PhysicalDevice VulkanContext::physicalDevice() const noexcept {
    return impl_ ? *impl_->physicalDevice : vk::PhysicalDevice{};
}

vk::Device VulkanContext::device() const noexcept {
    return impl_ ? *impl_->device : vk::Device{};
}

const Queue& VulkanContext::graphicsQueue() const noexcept {
    static const Queue emptyQueue{};
    return impl_ ? impl_->graphicsQueue : emptyQueue;
}

const Queue& VulkanContext::presentQueue() const noexcept {
    static const Queue emptyQueue{};
    return impl_ ? impl_->presentQueue : emptyQueue;
}

const Queue& VulkanContext::computeQueue() const noexcept {
    static const Queue emptyQueue{};
    return impl_ ? impl_->computeQueue : emptyQueue;
}

const vk::PhysicalDeviceProperties& VulkanContext::properties() const noexcept {
    static const vk::PhysicalDeviceProperties emptyProperties{};
    return impl_ ? impl_->properties : emptyProperties;
}

const vk::PhysicalDeviceMemoryProperties& VulkanContext::memoryProperties() const noexcept {
    static const vk::PhysicalDeviceMemoryProperties emptyProperties{};
    return impl_ ? impl_->memoryProperties : emptyProperties;
}

uint32_t VulkanContext::apiVersion() const noexcept {
    return impl_ ? impl_->apiVersion : VK_API_VERSION_1_0;
}

VkFormat VulkanContext::depthFormat() const noexcept {
    return impl_ ? impl_->depthFormat : VK_FORMAT_UNDEFINED;
}

bool VulkanContext::validationEnabled() const noexcept {
    return impl_ && impl_->validationEnabled;
}

bool VulkanContext::dynamicRenderingEnabled() const noexcept {
    return impl_ && impl_->dynamicRenderingEnabled;
}

bool VulkanContext::synchronization2Enabled() const noexcept {
    return impl_ && impl_->synchronization2Enabled;
}

bool VulkanContext::samplerAnisotropyEnabled() const noexcept {
    return impl_ && impl_->samplerAnisotropyEnabled;
}

void VulkanContext::waitIdle() const {
    if (impl_ && static_cast<VkDevice>(*impl_->device) != VK_NULL_HANDLE)
        impl_->device.waitIdle();
}

} // namespace vulkan_graphics
