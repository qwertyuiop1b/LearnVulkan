#include "vulkan_tutorial/utils.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <ios>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <vector>
#include <vulkan/vk_platform.h>
#include <vulkan/vulkan_core.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace hhq {

#ifdef NDEBUG
    const bool ENABLE_LAYER_VALIDATION = false;
#else
    const bool ENABLE_LAYER_VALIDATION = true;
#endif 

inline const std::vector<const char*> validationLayers = 
{
    "VK_LAYER_KHRONOS_validation"
};

inline const std::vector<const char*> deviceExtensions = 
{
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
#ifdef __APPLE__
    "VK_KHR_portability_subset",
#endif 
};
 

inline VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT *data, void *)
{
    std::cerr << data->pMessage << std::endl;
    return VK_FALSE;
}

inline VkResult createDebugUtilsMessenger(
    VkInstance instance, 
    VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, 
    VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pDebugMessenger
) 
{
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr)
    {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    return VK_ERROR_EXTENSION_NOT_PRESENT;
}

inline void destroyDebugUtilsMessenger
(
    VkInstance instance, 
    VkDebugUtilsMessengerEXT debugMessenger,
    VkAllocationCallbacks* pAllocator
)
{
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr)
    {
        func(instance, debugMessenger, pAllocator);
    }

}

struct QFamilyIndices 
{
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete()
    {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapchainDetailSupport 
{
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;

    VkSurfaceFormatKHR chooseSurfaceFromat() const 
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

    VkPresentModeKHR choosePresetMode() const 
    {
        for (const auto& presetMode : presentModes)
        {
            if (presetMode == VK_PRESENT_MODE_MAILBOX_KHR)
            {
                return presetMode;
            }
        }
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseExtent(GLFWwindow* window) const 
    {
        if (capabilities.currentExtent.width != UINT32_MAX)
        {
            return capabilities.currentExtent;
        }
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        VkExtent2D actual = {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height),
        };
        actual.width = std::clamp(actual.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actual.height = std::clamp(actual.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
        return actual;
    }
};


class App {
public:
    App(uint32_t w, uint32_t h, const std::string& tilte)
    : width(w)
    , height(h)
    , title(tilte)
    {
        initWindow();
        initVulkan();
    }

    void run()
    {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }
    }

    ~App() 
    {
        vkDestroyCommandPool(device, commandPool, nullptr);
        for (const auto& framebuffer: framebuffers)
        {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);
        for (const auto& imageView : swapchainImageViews)
        {
            vkDestroyImageView(device, imageView, nullptr);
        }
        vkDestroySwapchainKHR(device, swapchain, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroySurfaceKHR(instance, surface, nullptr);
        if (ENABLE_LAYER_VALIDATION)
        {
            destroyDebugUtilsMessenger(instance, debugMessenger, nullptr);
        }
        vkDestroyInstance(instance, nullptr);
        glfwDestroyWindow(window);
        glfwTerminate();
    }



private:
    void initWindow() 
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
        if (window == nullptr)
        {
            throw std::runtime_error("Failed to create glfw window!");
        }
    }

    static std::vector<const char*> getInstanceRequireExtensions() 
    {
        uint32_t count;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&count);
        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + count);
        if (ENABLE_LAYER_VALIDATION)
        {
            extensions.push_back("VK_EXT_debug_utils");
        }
#ifdef __APPLE__
        // macOS / MoltenVK 需要的扩展
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#endif
        return extensions;
    }

    bool checkDeviceExtensionSupported(VkPhysicalDevice candidate) 
    {
        uint32_t count = 0;
        vkEnumerateDeviceExtensionProperties(candidate, nullptr, &count, nullptr);

        std::vector<VkExtensionProperties> available(count);
        vkEnumerateDeviceExtensionProperties(candidate, nullptr, &count, available.data());

        std::set<std::string> required(deviceExtensions.begin(), deviceExtensions.end());
        for (const auto& extension : available) {
            required.erase(extension.extensionName);
        }

        return required.empty();
    }

    bool checkValidationSupported() 
    {
        uint32_t count;
        vkEnumerateInstanceLayerProperties(&count, nullptr);
        std::vector<VkLayerProperties> props(count);
        vkEnumerateInstanceLayerProperties(&count, props.data());
        std::cout << "count: " << count << std::endl;
        for (const auto&  layer: validationLayers) 
        {
            bool found = false;
            for (const auto& prop : props) 
            {
                if (strcmp(prop.layerName, layer) == 0)
                {
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }

        return true;
    }

    void createInstance()
    {
        if (ENABLE_LAYER_VALIDATION && !checkValidationSupported()) 
        {
            throw std::runtime_error("Required enable layer validation, but validation not supported!");
        }
        VkApplicationInfo appInfo 
        {
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = "Learn Vulkan",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "No Engine",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = VK_API_VERSION_1_3,
        };

        const auto extensions = getInstanceRequireExtensions();

        VkInstanceCreateInfo createInfo 
        {
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &appInfo,
            .ppEnabledExtensionNames = extensions.data(),
            .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
            .ppEnabledLayerNames = nullptr,
            .enabledLayerCount = 0,
        };
#ifdef  __APPLE__
        createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif 
        auto debugMessengerInfo = populateDebugMessengerCreateInfo();
        if (ENABLE_LAYER_VALIDATION)
        {
            createInfo.ppEnabledLayerNames = validationLayers.data();
            createInfo.enabledLayerCount = validationLayers.size();
            createInfo.pNext = &debugMessengerInfo;
        }

        VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
        if (result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create instance");
        }

    }

    inline static VkDebugUtilsMessengerCreateInfoEXT populateDebugMessengerCreateInfo() 
    {
        VkDebugUtilsMessengerCreateInfoEXT debugMessengerInfo 
        {
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT
                | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT, 
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
                | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                | VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT,
            .pfnUserCallback = debugCallback,
            
        };
        return debugMessengerInfo;
    }

    void createDebugMessenger()
    {
        if (!ENABLE_LAYER_VALIDATION) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo = populateDebugMessengerCreateInfo();
        VkResult result = createDebugUtilsMessenger(instance, &createInfo, nullptr, &debugMessenger);
        if (result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create debug utils messenger ext!");
        }
    }

    void createSurface()
    {
        VkResult result = glfwCreateWindowSurface(instance, window, nullptr, &surface);
        if (result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create window surface");
        }
    }

    QFamilyIndices findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface) 
    {
        uint32_t count;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
        std::vector<VkQueueFamilyProperties> properties(count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &count, properties.data());
        QFamilyIndices indices {};
        uint32_t index = 0;
        for (const auto& prop : properties)
        {
            if (!indices.graphicsFamily.has_value() && (prop.queueFlags & VK_QUEUE_GRAPHICS_BIT))
            {
                indices.graphicsFamily = index;
            }
            VkBool32 supported;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, index, surface, &supported);
            if (supported)
            {
                indices.presentFamily = index;
            }
            if (indices.isComplete()) break;
            index++;
        }
        return indices;
    }

    bool isSuitableDevice(VkPhysicalDevice candidate)
    {
        QFamilyIndices indices = findQueueFamilies(candidate, surface);
        bool extensionsSupported = checkDeviceExtensionSupported(candidate);
        SwapchainDetailSupport detail = querySwapchainSupport(candidate);

        return indices.isComplete()
            && extensionsSupported
            && !detail.formats.empty()
            && !detail.presentModes.empty();
    }

    void pickPhysicalDevice()
    {
        uint32_t count;
        vkEnumeratePhysicalDevices(instance, &count, nullptr);
        std::vector<VkPhysicalDevice> physicalDevices(count);
        vkEnumeratePhysicalDevices(instance, &count, physicalDevices.data());

        for (const auto& device : physicalDevices)
        {
            if (isSuitableDevice(device))
            {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE)
        {
            throw std::runtime_error("Failed to find a suitable physical device!");
        }
    }

    void createLogicalDevice()
    {
        queueFamilyIndices = findQueueFamilies(physicalDevice, surface);
        std::set<uint32_t> indices = {queueFamilyIndices.graphicsFamily.value(), queueFamilyIndices.presentFamily.value()};
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        float priority = 1.0f;
        for (const auto& idx : indices)
        {
            VkDeviceQueueCreateInfo queueInfo {
                .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .queueCount = 1,
                .queueFamilyIndex = idx,
                .pQueuePriorities = &priority,
            };
            queueCreateInfos.push_back(queueInfo);
        }

        VkPhysicalDeviceFeatures feature {};

        VkDeviceCreateInfo createInfo {
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
            .pQueueCreateInfos = queueCreateInfos.data(),
            .pEnabledFeatures = &feature,
            .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
            .ppEnabledExtensionNames = deviceExtensions.data(),
        };
        if (ENABLE_LAYER_VALIDATION)
        {
            createInfo.enabledLayerCount = validationLayers.size();
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        VkResult result = vkCreateDevice(physicalDevice, &createInfo, nullptr, &device);
        if (result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create device!");
        }
        vkGetDeviceQueue(device, queueFamilyIndices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, queueFamilyIndices.presentFamily.value(), 0, &presentQueue);
    }

    SwapchainDetailSupport querySwapchainSupport(VkPhysicalDevice candidate) 
    {
        SwapchainDetailSupport support {};
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(candidate, surface, &support.capabilities);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(candidate, surface, &formatCount, nullptr);
        if (formatCount != 0) 
        {
            support.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(candidate, surface, &formatCount, support.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(candidate, surface, &presentModeCount, nullptr);
        if (presentModeCount != 0)
        {
            support.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(candidate, surface, &presentModeCount, support.presentModes.data());
        }
        return support;
    }

    void createSwapchain() 
    {
        SwapchainDetailSupport detail = querySwapchainSupport(physicalDevice);
        auto format = detail.chooseSurfaceFromat();
        auto extent = detail.chooseExtent(window);
        auto presentMode = detail.choosePresetMode();
        uint32_t indices[] = {
            queueFamilyIndices.graphicsFamily.value(),
            queueFamilyIndices.presentFamily.value(),
        };

        uint32_t imageCount = detail.capabilities.minImageCount + 1;
        if (detail.capabilities.maxImageCount > 0 && imageCount > detail.capabilities.maxImageCount)
        {
            imageCount = detail.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo {
            .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .surface = surface,
            .imageFormat = format.format,
            .imageColorSpace = format.colorSpace,
            .imageExtent = extent,
            .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .imageArrayLayers = 1,
            .minImageCount = imageCount,
            .presentMode = presentMode,
            .preTransform = detail.capabilities.currentTransform,
            .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .clipped = VK_TRUE,
            .oldSwapchain = VK_NULL_HANDLE,
            .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        if (queueFamilyIndices.graphicsFamily != queueFamilyIndices.presentFamily)
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = indices;
        } 

        VkResult result = vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain);
        if (result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create swapchain");
        }
        
        vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
        swapchainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchainImages.data());

        swapchainFormat = format.format;
        swapchainExtent = extent;
    }

    void createImageViews()
    {
        swapchainImageViews.resize(swapchainImages.size());
        for (uint32_t i = 0; i < swapchainImages.size(); i++)
        {
            VkImageViewCreateInfo createInfo {
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = swapchainImages[i],
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .components = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A},
                .format = swapchainFormat,
                .subresourceRange = {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                }
            };
            VkResult result = vkCreateImageView(device, &createInfo, nullptr, &swapchainImageViews[i]);
            if (result != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create image view");
            }
        }
    }

    void createRenderPass()
    {
        VkAttachmentDescription colorAttachment 
        {
            .format = swapchainFormat,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        };

        VkAttachmentReference colorAttachmentref 
        {
            .attachment = 0,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };

        VkSubpassDescription subpass
        {
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorAttachmentref,
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        };

        VkSubpassDependency dependency 
        {
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        };

        VkRenderPassCreateInfo createInfo {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments = &colorAttachment,
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = 1,
            .pDependencies = &dependency,
        };

        VkResult result = vkCreateRenderPass(device, &createInfo, nullptr, &renderPass);
        if (result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create render pass!");
        }
    }

    VkShaderModule createShaderModuleFromFile(VkDevice device, const std::string& filename)
    {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error("failed to open file: " + filename);
        }
        size_t size = file.tellg();
        file.seekg(0);
        std::vector<char> buffer(size);
        file.read(buffer.data(), size);
        file.close();

        VkShaderModuleCreateInfo createInfo 
        {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = size,
            .pCode = reinterpret_cast<const uint32_t *>(buffer.data())
        };

        VkShaderModule shaderModule {VK_NULL_HANDLE};
        VkResult result = vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);
        if (result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create shader module!");
        }
        return shaderModule;
    }

    void createGraphicsPipeline()
    {
        VkShaderModule vertShaderModule = createShaderModuleFromFile(device, "shaders/triangle.vert.spv");
        VkShaderModule fragShaderModule = createShaderModuleFromFile(device, "shaders/triangle.frag.spv");

        VkPipelineShaderStageCreateInfo vertStage 
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertShaderModule,
            .pName = "main",
        };

        VkPipelineShaderStageCreateInfo fragStage 
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragShaderModule,
            .pName = "main",
        };
        VkPipelineShaderStageCreateInfo shaderStages[] = { vertStage, fragStage};

        VkPipelineVertexInputStateCreateInfo verteInputInfo 
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .pVertexBindingDescriptions = nullptr,
            .vertexBindingDescriptionCount = 0,
            .pVertexAttributeDescriptions = nullptr,
            .vertexAttributeDescriptionCount = 0,
        };

        VkPipelineInputAssemblyStateCreateInfo inputAssembly 
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .primitiveRestartEnable = VK_FALSE,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        };

        VkViewport viewport 
        {
            .x = 0,
            .y = 0,
            .width = static_cast<float>(swapchainExtent.width),
            .height = static_cast<float>(swapchainExtent.height),
            .minDepth = 0.f,
            .maxDepth = 1.f,
        };
        VkRect2D scissor 
        {
            .offset = {0, 0},
            .extent = swapchainExtent,
        };
        VkPipelineViewportStateCreateInfo viewportState
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .pViewports = &viewport,
            .scissorCount = 1,
            .pScissors = &scissor,
        };

        VkPipelineRasterizationStateCreateInfo rasterizer
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .cullMode = VK_CULL_MODE_BACK_BIT,
            .frontFace = VK_FRONT_FACE_CLOCKWISE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .lineWidth = 1.f,

            .depthBiasClamp = 0.f,
            .rasterizerDiscardEnable = VK_FALSE,
            .depthClampEnable = VK_FALSE,
            .depthBiasEnable = VK_FALSE,
        };

        VkPipelineMultisampleStateCreateInfo multisampling 
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .sampleShadingEnable = VK_FALSE,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        };

        VkPipelineColorBlendAttachmentState colorBlendAttachment
        {
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
            .blendEnable = VK_FALSE,
        };

        VkPipelineColorBlendStateCreateInfo colorBlending
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = VK_FALSE,
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment,
            .blendConstants = {0.f, 0.f, 0.f, 0.f},
        };

        std::vector<VkDynamicState> dynamicStates
        {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR,
        };
        VkPipelineDynamicStateCreateInfo dynamicState
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data(),
        };
        VkPipelineLayoutCreateInfo pipelineLayoutInfo 
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 0,
            .pushConstantRangeCount = 0,
        };
        VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout);
        if (result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create pipeline layout");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo 
        {
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2,
            .pStages = shaderStages,
            .pVertexInputState = &verteInputInfo,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = nullptr,
            .pColorBlendState = &colorBlending,
            .pDynamicState = &dynamicState,
            .layout = pipelineLayout,
            .renderPass = renderPass,
            .subpass = 0,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1
        };
        result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline);
        if (result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create graphic pipeline");
        }

        vkDestroyShaderModule(device, vertShaderModule, nullptr);
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
    }

    void createFrameBuffers()
    {
        framebuffers.resize(swapchainImageViews.size());
        for (size_t i = 0; i < swapchainImageViews.size(); i++)
        {
            VkImageView attachments[] = {swapchainImageViews[i]};
            VkFramebufferCreateInfo createInfo 
            {
                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .renderPass = renderPass,
                .attachmentCount = 1,
                .pAttachments = attachments,
                .width = swapchainExtent.width,
                .height = swapchainExtent.height,
                .layers = 1,
            };
            VkResult result = vkCreateFramebuffer(device, &createInfo, nullptr, &framebuffers[i]);
            if (result != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create framebuffer");
            }

        }

    }

    void createCommandPool()
    {
        VkCommandPoolCreateInfo createInfo 
        {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .queueFamilyIndex = queueFamilyIndices.graphicsFamily.value(),
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        };
        vkCreateCommandPool(device, &createInfo, nullptr, &commandPool);
    }

    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t index)
    {
        VkCommandBufferBeginInfo beginInfo 
        {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = 0
        };
        VkResult result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
        if (result != VK_SUCCESS)
        {
            throw std::runtime_error("Cannt begin command buffer!");
        }

        VkClearValue clearColor 
        {
            .color = {1.0, 1.0, 1.0, 1.0f},
        };
        VkRenderPassBeginInfo rpInfo 
        {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = renderPass,
            .framebuffer = framebuffers[index],
            .clearValueCount = 1,
            .pClearValues = &clearColor,
            .renderArea = {
                .offset = {0, 0},
                .extent = swapchainExtent,
            }
        };
        vkCmdBeginRenderPass(commandBuffer, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
        VkViewport viewport{};
        viewport.x        = 0.0f;
        viewport.y        = 0.0f;
        viewport.width    = static_cast<float>(swapchainExtent.width);
        viewport.height   = static_cast<float>(swapchainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapchainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
        vkCmdEndRenderPass(commandBuffer);
        result = vkEndCommandBuffer(commandBuffer);
        if (result != VK_SUCCESS)
        {
            throw std::runtime_error("end command buffer failed!");
        }
    }

    void createCommandBuffers()
    {
        commandBuffers.resize(framebuffers.size());
        VkCommandBufferAllocateInfo allocInfo 
        {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = commandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = static_cast<uint32_t>(commandBuffers.size()),
        };
        VkResult result = vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data());
        if (result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate command buffers");
        }

        for (size_t i = 0; i < commandBuffers.size(); i++)
        {
            recordCommandBuffer(commandBuffers[i], static_cast<uint32_t>(i));
        }
    }

    void initVulkan() 
    {
        createInstance();
        createDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapchain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFrameBuffers();
        createCommandPool();
        createCommandBuffers();
    }

private:
    uint32_t width;
    uint32_t height;
    const std::string title;

    GLFWwindow* window {nullptr};
    VkInstance instance {VK_NULL_HANDLE};
    VkDebugUtilsMessengerEXT debugMessenger {VK_NULL_HANDLE};
    VkSurfaceKHR surface {VK_NULL_HANDLE};
    VkPhysicalDevice physicalDevice {VK_NULL_HANDLE};

    QFamilyIndices queueFamilyIndices;

    VkDevice device;
    VkQueue graphicsQueue;
    VkQueue presentQueue;

    VkSwapchainKHR swapchain;
    std::vector<VkImage> swapchainImages;
    VkFormat swapchainFormat;
    VkExtent2D swapchainExtent;
    std::vector<VkImageView> swapchainImageViews;

    VkRenderPass renderPass;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    std::vector<VkFramebuffer> framebuffers;
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
};




};


int main() 
{
    try 
    {
        hhq::App app{800, 600, "Learn Vulkan"};
        app.run();
    } 
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
