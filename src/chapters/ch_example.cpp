#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fmt/base.h>
#include <fstream>
#include <ios>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vk_platform.h>
#include <vulkan/vulkan_core.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

static void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    fmt::println("resize: {},{}", width, height);
}

static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    switch (key) {
    case GLFW_KEY_ESCAPE:
        if (action == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }
        break;
    default:
        fmt::println("keyCode: {}", key);
    }
}

#ifdef NDEBUG
const bool enable_validation_layers = false;
#else
const bool enable_validation_layers = true;
#endif

inline std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};
inline std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
#ifdef __APPLE__
    "VK_KHR_portability_subset",
#endif
};

#include <vulkan/vk_enum_string_helper.h>

VKAPI_ATTR VkBool32 VKAPI_CALL debugUtilsCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                  VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                                                  const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                  void* pUserData) {
    fmt::println("validation layer: {}", pCallbackData->pMessage);
    return VK_FALSE;
}

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,
                                      const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                      const VkAllocationCallbacks* pAllocator,
                                      VkDebugUtilsMessengerEXT* pMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
};

void DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT messenger,
                                   const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, messenger, pAllocator);
    }
}

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicFamily;
    std::optional<uint32_t> presentFamily;

    bool isCompleted() const {
        return graphicFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapchainSupportedDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;

    VkExtent2D chooseSurfaceExtent(GLFWwindow* window) const {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        return VkExtent2D{
            std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
            std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
        };
    }

    VkSurfaceFormatKHR chooseSurfaceFormat() const {
        assert(!formats.empty() && "formats is empty!");
        for (const auto& format : formats) {
            if (format.format == VK_FORMAT_B8G8R8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return format;
            }
        }
        return formats[0];
    }

    VkPresentModeKHR choosePresentMode() const {
        for (const auto& presentMode : presentModes) {
            if (presentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return presentMode;
            }
        }
        return VK_PRESENT_MODE_FIFO_KHR;
    }
};

class Example {
  public:
    Example(const VkExtent2D& extent, const std::string& title) : extent(extent), title(title) {
        initWindow();
        initVulkan();
    }
    ~Example() {
        vkDestroyCommandPool(device, commandPool, nullptr);
        for (const auto& framebuffer : framebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }
        vkDestroyPipeline(device, graphicPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);
        for (const auto& swapchainImageView : swapchainImageViews) {
            vkDestroyImageView(device, swapchainImageView, nullptr);
        }
        vkDestroySwapchainKHR(device, swapchain, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroySurfaceKHR(instance, surface, nullptr);
        if (debugUtilsMessenger != VK_NULL_HANDLE) {
            DestroyDebugUtilsMessengerEXT(instance, debugUtilsMessenger, nullptr);
        }
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
        }
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void run() {
        while (!glfwWindowShouldClose(window)) {
            draw();
            glfwPollEvents();
        }
    }

  private:
    VkExtent2D extent;
    const std::string& title;
    GLFWwindow* window{nullptr};

    VkInstance instance{VK_NULL_HANDLE};
    VkDebugUtilsMessengerEXT debugUtilsMessenger{VK_NULL_HANDLE};
    VkSurfaceKHR surface{VK_NULL_HANDLE};
    VkPhysicalDevice physicalDevice{VK_NULL_HANDLE};
    VkDevice device{VK_NULL_HANDLE};
    VkQueue graphicQueue{VK_NULL_HANDLE};
    VkQueue presentQueue{VK_NULL_HANDLE};

    QueueFamilyIndices queueFamilyIndices;
    VkSwapchainKHR swapchain {VK_NULL_HANDLE};
    std::vector<VkImage> swapchainImages {};
    VkFormat imageFormat {};
    VkExtent2D imageExtent {};
    std::vector<VkImageView> swapchainImageViews {};

    VkRenderPass renderPass {VK_NULL_HANDLE};
    VkPipelineLayout pipelineLayout {VK_NULL_HANDLE};
    VkPipeline graphicPipeline {VK_NULL_HANDLE};

    std::vector<VkFramebuffer> framebuffers {};
    VkCommandPool commandPool {};
    VkCommandBuffer commandBuffer {};
    

    void initWindow() {
        glfwInit();
        if (!glfwVulkanSupported()) {
            fmt::println("Vulkan not supported!");
            return;
        }
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        window = glfwCreateWindow(extent.width, extent.height, title.c_str(), nullptr, nullptr);
        if (window == nullptr) {
            fmt::println("Failed to create window!");
            return;
        }
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
        glfwSetKeyCallback(window, keyCallback);
    }

    static std::vector<const char*> getInstanceExtensions() {
        uint32_t extensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&extensionCount);
        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + extensionCount);
        if (enable_validation_layers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
#ifdef __APPLE__
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#endif
        return extensions;
    }

    void createInstance() {
        VkApplicationInfo appInfo{
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = "LearnVulkan",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "No Engine",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = VK_API_VERSION_1_3,
        };

        auto extensions = getInstanceExtensions();
        VkInstanceCreateInfo instanceCI{
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &appInfo,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = nullptr,
            .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data(),
        };
#ifdef __APPLE__
        instanceCI.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
        auto debugCreateInfo = getDebugUtilsCreateInfo();
        if (enable_validation_layers) {
            instanceCI.enabledLayerCount = validationLayers.size();
            instanceCI.ppEnabledLayerNames = validationLayers.data();
            instanceCI.pNext = &debugCreateInfo;
        }
        if (vkCreateInstance(&instanceCI, nullptr, &instance) != VK_SUCCESS) {
            fmt::println("Failed to create instance!");
        }
    }

    static VkDebugUtilsMessengerCreateInfoEXT getDebugUtilsCreateInfo() {
        VkDebugUtilsMessengerCreateInfoEXT createInfo{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT,
            .messageType =
                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pUserData = nullptr,
            .pfnUserCallback = debugUtilsCallback,
        };
        return createInfo;
    }

    void createDebugMessenger() {
        if (!enable_validation_layers || instance == VK_NULL_HANDLE)
            return;
        auto createInfo = getDebugUtilsCreateInfo();
        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugUtilsMessenger) != VK_SUCCESS) {
            fmt::println("Failed to create debug utils messenger!");
        }
    }

    void createSurface() {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create window surface");
        }
    }

    QueueFamilyIndices findQueueFamily(VkPhysicalDevice device, VkSurfaceKHR surface) {
        uint32_t familyCount;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> familyProps(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, familyProps.data());

        QueueFamilyIndices familyIndices;
        for (uint32_t i = 0; i < familyCount; i++) {
            if (!familyIndices.graphicFamily.has_value() && (familyProps[i].queueFlags | VK_QUEUE_GRAPHICS_BIT)) {
                familyIndices.graphicFamily = i;
            }

            VkBool32 supported = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &supported);
            if (!familyIndices.presentFamily.has_value() && supported) {
                familyIndices.presentFamily = i;
            }

            if (familyIndices.isCompleted())
                break;
        }
        return familyIndices;
    }

    bool isSuitablePhysicalDevice(VkPhysicalDevice device, VkSurfaceKHR surface) {
        return findQueueFamily(device, surface).isCompleted();
    }

    void pickPhysicalDevice() {
        uint32_t deviceCount;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0) {
            throw std::runtime_error("Failed to find a physical device");
        }
        std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices.data());
        for (const auto& physical : physicalDevices) {
            if (isSuitablePhysicalDevice(physical, surface)) {
                physicalDevice = physical;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("Failed to find a suitable physical device");
        }
    }

    void createLogicalDevice() {
        queueFamilyIndices = findQueueFamily(physicalDevice, surface);
        std::set<uint32_t> uniqueIndices{queueFamilyIndices.graphicFamily.value(), queueFamilyIndices.presentFamily.value()};

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        float priority = 1.0f;
        for (const auto& index : uniqueIndices) {
            VkDeviceQueueCreateInfo createInfo{
                .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .queueFamilyIndex = index,
                .queueCount = 1,
                .pQueuePriorities = &priority,
            };
            queueCreateInfos.push_back(createInfo);
        }

        VkPhysicalDeviceFeatures features{
            .samplerAnisotropy = VK_TRUE,
        };
        VkDeviceCreateInfo deviceInfo{
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
            .pQueueCreateInfos = queueCreateInfos.data(),
            .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
            .ppEnabledExtensionNames = deviceExtensions.data(),
            .pEnabledFeatures = &features,
        };

        if (vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create device!");
        }

        vkGetDeviceQueue(device, queueFamilyIndices.graphicFamily.value(), 0, &graphicQueue);
        vkGetDeviceQueue(device, queueFamilyIndices.presentFamily.value(), 0, &presentQueue);
    }

    SwapchainSupportedDetails querySwapchainSupportedDetails() const {
        SwapchainSupportedDetails detailInfos;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &detailInfos.capabilities);

        uint32_t formatCount {};
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);
        if (formatCount != 0) {
            detailInfos.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, detailInfos.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);
        if (presentModeCount != 0) {
            detailInfos.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, detailInfos.presentModes.data());
        }

        return detailInfos;
    }

    void createSwapchain() {
        SwapchainSupportedDetails supportedDetails = querySwapchainSupportedDetails();
        VkExtent2D surfaceExtent = supportedDetails.chooseSurfaceExtent(window);
        VkSurfaceFormatKHR surfaceFormat = supportedDetails.chooseSurfaceFormat();
        VkPresentModeKHR presentMode = supportedDetails.choosePresentMode();
        fmt::println("format: {}, {}; presentMode: {}",
                     string_VkFormat(surfaceFormat.format),
                     string_VkColorSpaceKHR(surfaceFormat.colorSpace),
                     string_VkPresentModeKHR(presentMode));

        uint32_t imageCount = supportedDetails.capabilities.minImageCount + 1;
        if (imageCount > supportedDetails.capabilities.maxImageCount) {
            imageCount = supportedDetails.capabilities.maxImageCount;
        }        

        VkSwapchainCreateInfoKHR createInfo {
            .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .surface = surface,
            .minImageCount = imageCount,
            .imageFormat = surfaceFormat.format,
            .imageColorSpace = surfaceFormat.colorSpace,
            .imageExtent = surfaceExtent,
            .imageArrayLayers = 1,
            .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .preTransform = supportedDetails.capabilities.currentTransform,
            .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = presentMode,
            .clipped = VK_TRUE,
            .oldSwapchain = VK_NULL_HANDLE,
        };

        bool uniqueIndicesFlag = queueFamilyIndices.graphicFamily.value() != queueFamilyIndices.presentFamily.value();
        std::array<uint32_t, 2> queueIndicesArray;
        if (uniqueIndicesFlag) {
            queueIndicesArray = {queueFamilyIndices.graphicFamily.value(), queueFamilyIndices.presentFamily.value()};
            createInfo.queueFamilyIndexCount = queueIndicesArray.size();
            createInfo.pQueueFamilyIndices = queueIndicesArray.data();
        }

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create swapchain!");
        }

        vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
        swapchainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchainImages.data());

        imageFormat = surfaceFormat.format;
        imageExtent = surfaceExtent;
    }

    void createImageViews() {
        swapchainImageViews.resize(swapchainImages.size());
        for (uint32_t i = 0; i < swapchainImages.size(); i++) {
            VkImageViewCreateInfo createInfo {
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = swapchainImages[i],
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = imageFormat,
                .components = {
                    VK_COMPONENT_SWIZZLE_IDENTITY,
                    VK_COMPONENT_SWIZZLE_IDENTITY,
                    VK_COMPONENT_SWIZZLE_IDENTITY,
                    VK_COMPONENT_SWIZZLE_IDENTITY,
                },
                .subresourceRange = {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            };
            if (vkCreateImageView(device, &createInfo, nullptr, &swapchainImageViews[i]) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create image views!");
            }
        }
    }

    void createRenderPass() {
        VkAttachmentDescription attachmentDesc {
            .format = imageFormat,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        };

        VkAttachmentReference attachmentRef {
            .attachment = 0,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };

        VkSubpassDescription subpass {
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1,
            .pColorAttachments = &attachmentRef,
        };

        VkRenderPassCreateInfo createInfo {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .attachmentCount = 1,
            .pAttachments = &attachmentDesc,
        };
        if (vkCreateRenderPass(device, &createInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create renderpass!");
        }

    }

    static std::vector<char> readFile(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            throw std::runtime_error("Failed. to open file: " + filepath);
        }
        size_t size = file.tellg();
        file.seekg(0);
        std::vector<char> buffer(size);
        file.read(buffer.data(), size);
        file.close();
        return buffer;
    }

    VkShaderModule createShaderModule(const std::string& shaderPath) {
        auto shaderCode = readFile(shaderPath);
        VkShaderModule shaderModule {VK_NULL_HANDLE};
        VkShaderModuleCreateInfo createInfo {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = shaderCode.size(),
            .pCode = reinterpret_cast<uint32_t*>(shaderCode.data()),
        };
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create shader module!");
        }

        return shaderModule;
    }

    void createGraphicPipeline() {
        VkShaderModule vertexShaderModule = createShaderModule("shaders/triangle.vert.spv");
        VkShaderModule fragmentShaderModule = createShaderModule("shaders/triangle.frag.spv");

        VkPipelineShaderStageCreateInfo vertexStageCreateInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertexShaderModule,
            .pName = "main",
        };

        VkPipelineShaderStageCreateInfo fragmentStageCreateInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragmentShaderModule,
            .pName = "main",
        };
        
        VkPipelineShaderStageCreateInfo shaderStages[] = {vertexStageCreateInfo, fragmentStageCreateInfo};

        VkPipelineVertexInputStateCreateInfo vertexInputState {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 0,
            .pVertexBindingDescriptions = nullptr,
            .vertexAttributeDescriptionCount = 0,
            .pVertexAttributeDescriptions = nullptr,
        };

        VkPipelineInputAssemblyStateCreateInfo inputAssemblyState {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE,
        };

        VkViewport viewport {
            .x = 0,
            .y = 0,
            .width = (float)imageExtent.width,
            .height = (float)imageExtent.height,
            .minDepth = 0.f,
            .maxDepth = 1.f
        };
        VkRect2D scissor {
            .offset = {0, 0,},
            .extent = imageExtent,
        };
        std::vector<VkDynamicState> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamicState {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data(),
        };

        VkPipelineViewportStateCreateInfo viewportState {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .scissorCount = 1,
        };

        VkPipelineRasterizationStateCreateInfo rasterState {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .depthClampEnable = VK_FALSE,
            .cullMode = VK_CULL_MODE_BACK_BIT,
            .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .lineWidth = 1.0,
            .depthBiasEnable = VK_FALSE,
            .depthBiasClamp = 0.f,
            .depthBiasConstantFactor = 0.f,
            .depthBiasSlopeFactor = 0.f,
        };

        VkPipelineMultisampleStateCreateInfo multisampleState {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable = VK_FALSE,
            .minSampleShading = 1.f,
            .alphaToCoverageEnable = VK_FALSE,
            .alphaToOneEnable = VK_FALSE,
        };

        VkPipelineColorBlendAttachmentState colorAttachmentState {
            .blendEnable = VK_FALSE,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
            .colorBlendOp = VK_BLEND_OP_ADD,
            .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        };

        VkPipelineColorBlendStateCreateInfo colorBlendState {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = VK_FALSE,
            .attachmentCount = 1,
            .pAttachments = &colorAttachmentState,
            .blendConstants = { 0.f, 0.f, 0.f, 0.f},
        };

        VkPipelineLayoutCreateInfo layoutCreateInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 0,
            .pSetLayouts = nullptr,
            .pushConstantRangeCount = 0,
            .pPushConstantRanges = nullptr,
        };

        if (vkCreatePipelineLayout(device, &layoutCreateInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed ot create pipeline layout!");
        }

        VkGraphicsPipelineCreateInfo pipelineCreateInfo {
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2,
            .pStages = shaderStages,
            .pVertexInputState = &vertexInputState,
            .pInputAssemblyState = &inputAssemblyState,
            .pRasterizationState = &rasterState,
            .pDynamicState = &dynamicState,
            .pViewportState = &viewportState,
            .pMultisampleState = &multisampleState,
            .pColorBlendState = &colorBlendState,
            .renderPass = renderPass,
            .subpass = 0,
            .layout = pipelineLayout,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1,
        };
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &graphicPipeline) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create graphic pipeline!");
        }

        vkDestroyShaderModule(device, vertexShaderModule, nullptr);
        vkDestroyShaderModule(device, fragmentShaderModule, nullptr);
    }

    void createFramebuffers() {
        framebuffers.resize(swapchainImages.size());
        for (uint32_t i = 0; i < framebuffers.size(); i++) {
            VkFramebufferCreateInfo createInfo {
                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .attachmentCount = 1,
                .pAttachments = &swapchainImageViews[i],
                .width = imageExtent.width,
                .height = imageExtent.height,
                .layers = 1,
                .renderPass = renderPass,
            
            };
            if (vkCreateFramebuffer(device, &createInfo, nullptr, &framebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create framebuffer!");
            }
        }
    }

    void createCommandPool() {
        VkCommandPoolCreateInfo createInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = queueFamilyIndices.graphicFamily.value(),
        };
        if (vkCreateCommandPool(device, &createInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create command pool!");
        }
    }

    void createCommandBuffers() {
        VkCommandBufferAllocateInfo allocInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = commandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        if (vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate command buffer!");
        }
    }

    void recordCommandBuffers(VkCommandBuffer cmdBuf, uint32_t index) {
        VkCommandBufferBeginInfo beginInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = 0,
            .pInheritanceInfo = nullptr,
        };
        if (vkBeginCommandBuffer(cmdBuf, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("Failed to begin command buffer!");
        }
        VkClearValue clearColor = {{0.f, 0.f, 0.f, 1.0f}};
        VkRenderPassBeginInfo renderPassInfo {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = renderPass,
            .renderArea = {
                .offset = {0, 0},
                .extent = imageExtent,
            },
            .framebuffer = framebuffers[index],
            .clearValueCount = 1,
            .pClearValues = &clearColor,
        };
        vkCmdBeginRenderPass(cmdBuf, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicPipeline);
        VkViewport viewport {
            .x = 0,
            .y = 0,
            .width = static_cast<float>(imageExtent.width),
            .height = static_cast<float>(imageExtent.height),
            .minDepth = 0.f,
            .maxDepth = 1.0f,
        };
        VkRect2D scissor = {
            .offset = {0, 0},
            .extent = imageExtent,
        };
        vkCmdSetViewport(cmdBuf, 0, 1, &viewport);
        vkCmdSetScissor(cmdBuf, 0, 1, &scissor);
        vkCmdDraw(cmdBuf, 3, 1, 0, 0);
        vkCmdEndRenderPass(cmdBuf);
        if (vkEndCommandBuffer(cmdBuf) != VK_SUCCESS) {
            throw std::runtime_error("Failed to end command buffer!");
        }
    }

    void initVulkan() {
        createInstance();
        createDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapchain();
        createImageViews();
        createRenderPass();
        createGraphicPipeline();
        createFramebuffers();
        createCommandPool();
        createCommandBuffers();
    }

    void draw() {}
};

int main() {
    try {
        Example app{VkExtent2D{800, 600}, "LearnVulkan"};
        app.run();
    } catch (const std::exception& e) {
        fmt::println(e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
