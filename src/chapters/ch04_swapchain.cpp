/**
 * @file ch04_swapchain.cpp
 * @brief 第04章：窗口表面（Surface）与交换链（Swap Chain）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【核心概念】
 *
 * 【VkSurfaceKHR — 窗口表面】
 *
 *  Vulkan 是平台无关的，不直接与窗口系统交互。
 *  VkSurfaceKHR 是 Vulkan 与平台窗口系统之间的桥梁（KHR = Khronos 扩展）。
 *  GLFW 帮我们封装了各平台（Windows/Linux/macOS）的 Surface 创建。
 *
 * 【VkSwapchainKHR — 交换链】
 *
 *  交换链本质上是一组图像缓冲区，GPU 渲染到其中一个，
 *  显示系统从另一个读取显示，避免画面撕裂。
 *
 *  ┌────────────────────────────────────────────────────────┐
 *  │  交换链图像队列示意图：                                  │
 *  │                                                        │
 *  │  [图像0: 正在显示] ← 显示器读取                        │
 *  │  [图像1: GPU 渲染中]                                   │
 *  │  [图像2: 等待渲染]  (三缓冲模式)                       │
 *  └────────────────────────────────────────────────────────┘
 *
 * 【三个关键选择】
 *
 *  1. Surface Format（表面格式）
 *     - 颜色格式：B8G8R8A8_SRGB（8位每通道，SRGB颜色空间）
 *     - 色彩空间：SRGB_NONLINEAR（标准显示器色彩空间）
 *
 *  2. Present Mode（呈现模式）
 *     ┌─────────────┬───────────────────────────────────────┐
 *     │ IMMEDIATE   │ 立即替换，可能撕裂                     │
 *     │ FIFO        │ 垂直同步，最大帧率=屏幕刷新率，必须支持 │
 *     │ FIFO_RELAXED│ 宽松垂直同步，延迟低时可能撕裂          │
 *     │ MAILBOX     │ 三缓冲，帧率不受限，低延迟，最推荐      │
 *     └─────────────┴───────────────────────────────────────┘
 *
 *  3. Swap Extent（交换范围）
 *     - 交换链图像的分辨率，通常等于窗口分辨率
 *     - 注意：Retina 屏幕的像素数量 ≠ 屏幕坐标数量
 * ═══════════════════════════════════════════════════════════════════════════
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan_tutorial/utils.hpp>

#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>

constexpr uint32_t WIDTH  = 800;
constexpr uint32_t HEIGHT = 600;

class Ch04App {
public:
    void run()
    {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow*      window_         = nullptr;
    VkInstance       instance_       = VK_NULL_HANDLE;
    VkSurfaceKHR     surface_        = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice         device_         = VK_NULL_HANDLE;
    VkQueue          graphicsQueue_  = VK_NULL_HANDLE;
    VkQueue          presentQueue_   = VK_NULL_HANDLE;
    VkSwapchainKHR   swapchain_      = VK_NULL_HANDLE;

    std::vector<VkImage>     swapchainImages_;
    VkFormat                 swapchainImageFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D               swapchainExtent_{};

    void initWindow()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);   // 不创建 OpenGL 上下文
        glfwWindowHint(GLFW_RESIZABLE,  GLFW_FALSE);    // 暂时禁止窗口缩放
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "Ch04 - Swap Chain", nullptr, nullptr);
        std::cout << "✅ 窗口已创建 " << WIDTH << "x" << HEIGHT << "\n";
    }

    void initVulkan()
    {
        createInstance();
        createSurface();       // Surface 必须在选择物理设备之前创建！
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapchain();
    }

    void createInstance()
    {
        VkApplicationInfo appInfo{};
        appInfo.sType      = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.apiVersion = VK_API_VERSION_1_3;

        auto exts = getRequiredInstanceExtensions();

        VkInstanceCreateInfo ci{};
        ci.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo        = &appInfo;
        ci.enabledExtensionCount   = static_cast<uint32_t>(exts.size());
        ci.ppEnabledExtensionNames = exts.data();
        ci.flags                  |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;

        if (ENABLE_VALIDATION_LAYERS) {
            ci.enabledLayerCount   = static_cast<uint32_t>(VALIDATION_LAYERS.size());
            ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();
        }
        VK_CHECK(vkCreateInstance(&ci, nullptr, &instance_));
    }

    // ─── 创建 Surface ─────────────────────────────────────────────────────────

    void createSurface()
    {
        // GLFW 封装了跨平台 Surface 创建：
        //   macOS → VK_MVK_macos_surface / VK_EXT_metal_surface
        //   Windows → VK_KHR_win32_surface
        //   Linux → VK_KHR_xcb_surface / VK_KHR_xlib_surface
        VK_CHECK(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_));
        std::cout << "✅ VkSurfaceKHR 创建成功！\n";
    }

    void pickPhysicalDevice()
    {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance_, &count, nullptr);
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance_, &count, devices.data());

        for (auto& d : devices) {
            if (isDeviceSuitable(d)) {
                physicalDevice_ = d;
                break;
            }
        }
        if (physicalDevice_ == VK_NULL_HANDLE)
            throw std::runtime_error("没有找到合适的 GPU");

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice_, &props);
        std::cout << "✅ 选择设备：" << props.deviceName << "\n";
    }

    bool isDeviceSuitable(VkPhysicalDevice device)
    {
        QueueFamilyIndices indices = findQueueFamilies(device, surface_);
        bool extsOk = checkDeviceExtensionSupport(device);

        bool swapchainOk = false;
        if (extsOk) {
            SwapChainSupportDetails sc = querySwapChainSupport(device, surface_);
            swapchainOk = !sc.formats.empty() && !sc.presentModes.empty();
        }
        return indices.isComplete() && extsOk && swapchainOk;
    }

    void createLogicalDevice()
    {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice_, surface_);
        std::set<uint32_t> uniqueFamilies = {
            indices.graphicsFamily.value(),
            indices.presentFamily.value()
        };

        const float priority = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> qcis;
        for (uint32_t family : uniqueFamilies) {
            VkDeviceQueueCreateInfo qci{};
            qci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            qci.queueFamilyIndex = family;
            qci.queueCount       = 1;
            qci.pQueuePriorities = &priority;
            qcis.push_back(qci);
        }

        VkPhysicalDeviceFeatures features{};
        features.samplerAnisotropy = VK_TRUE;

        VkDeviceCreateInfo ci{};
        ci.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        ci.queueCreateInfoCount    = static_cast<uint32_t>(qcis.size());
        ci.pQueueCreateInfos       = qcis.data();
        ci.pEnabledFeatures        = &features;
        ci.enabledExtensionCount   = static_cast<uint32_t>(DEVICE_EXTENSIONS.size());
        ci.ppEnabledExtensionNames = DEVICE_EXTENSIONS.data();

        if (ENABLE_VALIDATION_LAYERS) {
            ci.enabledLayerCount   = static_cast<uint32_t>(VALIDATION_LAYERS.size());
            ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();
        }
        VK_CHECK(vkCreateDevice(physicalDevice_, &ci, nullptr, &device_));

        vkGetDeviceQueue(device_, indices.graphicsFamily.value(), 0, &graphicsQueue_);
        vkGetDeviceQueue(device_, indices.presentFamily.value(),  0, &presentQueue_);
        std::cout << "✅ 逻辑设备创建成功！\n";
    }

    // ─── 创建交换链（核心！） ──────────────────────────────────────────────────

    void createSwapchain()
    {
        SwapChainSupportDetails sc = querySwapChainSupport(physicalDevice_, surface_);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(sc.formats);
        VkPresentModeKHR   presentMode   = chooseSwapPresentMode(sc.presentModes);
        VkExtent2D         extent        = chooseSwapExtent(sc.capabilities, window_);

        // 交换链图像数量：比最小数量多1（实现三缓冲）
        uint32_t imageCount = sc.capabilities.minImageCount + 1;
        if (sc.capabilities.maxImageCount > 0)
            imageCount = std::min(imageCount, sc.capabilities.maxImageCount);

        VkSwapchainCreateInfoKHR ci{};
        ci.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        ci.surface          = surface_;
        ci.minImageCount    = imageCount;
        ci.imageFormat      = surfaceFormat.format;
        ci.imageColorSpace  = surfaceFormat.colorSpace;
        ci.imageExtent      = extent;
        ci.imageArrayLayers = 1;                              // 非 VR 应用始终为 1
        ci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;  // 作为颜色附件

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice_, surface_);
        uint32_t queueFamilyIndices[] = {
            indices.graphicsFamily.value(),
            indices.presentFamily.value()
        };

        if (indices.graphicsFamily != indices.presentFamily) {
            // 不同队列族：使用并发模式（性能较低，但实现简单）
            ci.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
            ci.queueFamilyIndexCount = 2;
            ci.pQueueFamilyIndices   = queueFamilyIndices;
        } else {
            // 同一队列族：使用独占模式（最高性能）
            ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        // 图像变换（如旋转90度）：使用当前变换（不额外旋转）
        ci.preTransform   = sc.capabilities.currentTransform;
        // 与其他窗口合成时是否使用 Alpha 通道（通常忽略）
        ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        ci.presentMode    = presentMode;
        ci.clipped        = VK_TRUE;   // 裁剪被其他窗口遮挡的像素，提升性能
        ci.oldSwapchain   = VK_NULL_HANDLE;  // 窗口大小改变时需要重建交换链

        VK_CHECK(vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_));

        // 获取交换链图像句柄（数量可能多于我们请求的最小值）
        vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, nullptr);
        swapchainImages_.resize(imageCount);
        vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, swapchainImages_.data());

        swapchainImageFormat_ = surfaceFormat.format;
        swapchainExtent_      = extent;

        std::cout << "✅ 交换链创建成功！\n";
        std::cout << "   图像数量：" << imageCount << "\n";
        std::cout << "   图像格式：" << surfaceFormat.format << "\n";
        std::cout << "   分辨率  ：" << extent.width << "x" << extent.height << "\n";
        std::cout << "   呈现模式：" << (presentMode == VK_PRESENT_MODE_MAILBOX_KHR
                                          ? "Mailbox (三缓冲)"
                                          : "FIFO (垂直同步)") << "\n";
    }

    void mainLoop()
    {
        std::cout << "\n🎮 窗口已打开，按 ESC 或关闭窗口退出...\n";
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
                glfwSetWindowShouldClose(window_, GLFW_TRUE);
        }
    }

    void cleanup()
    {
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        vkDestroyDevice(device_, nullptr);
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        vkDestroyInstance(instance_, nullptr);
        glfwDestroyWindow(window_);
        glfwTerminate();
        std::cout << "✅ 清理完成。\n";
    }
};

int main()
{
    std::cout << "═══════════════════════════════════════\n";
    std::cout << " 第04章：窗口表面与交换链\n";
    std::cout << "═══════════════════════════════════════\n\n";

    Ch04App app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ 错误：" << e.what() << "\n";
        return 1;
    }
    return 0;
}
