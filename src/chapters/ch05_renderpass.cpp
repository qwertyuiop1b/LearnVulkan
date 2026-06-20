/**
 * @file ch05_renderpass.cpp
 * @brief 第05章：图像视图（ImageView）与渲染流程（RenderPass）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【VkImageView — 图像视图】
 *
 *  VkImage 是原始的图像数据（像素数组）。
 *  VkImageView 告诉 Vulkan "如何解读"这个图像数据：
 *    - 这是 2D 图像还是立方体贴图？
 *    - 使用哪个 mip 层级范围？
 *    - 使用哪个数组层？
 *    - 各颜色通道如何映射（swizzle）？
 *
 *  交换链的每个 VkImage 都需要对应一个 VkImageView。
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【VkRenderPass — 渲染流程】
 *
 *  RenderPass 描述了一次渲染操作的"框架"：
 *    - 使用哪些附件（颜色、深度、模板）？
 *    - 每个附件渲染前后如何处理（加载/清除/存储）？
 *    - Subpass（子流程）之间的依赖关系？
 *
 *  ┌─────────────────────────────────────────────────────────┐
 *  │  RenderPass 结构：                                       │
 *  │                                                         │
 *  │  Attachment Description（附件描述）                      │
 *  │    ├─ 格式（Format）                                    │
 *  │    ├─ 多重采样（Samples）                               │
 *  │    ├─ loadOp:  渲染前 → CLEAR（清除）/ LOAD / DONT_CARE │
 *  │    └─ storeOp: 渲染后 → STORE（保存）/ DONT_CARE        │
 *  │                                                         │
 *  │  Subpass（子流程）                                       │
 *  │    └─ 引用上面定义的附件                                 │
 *  │                                                         │
 *  │  Subpass Dependency（子流程依赖）                        │
 *  │    └─ 控制图像布局转换的同步时机                         │
 *  └─────────────────────────────────────────────────────────┘
 *
 * 【图像布局（Image Layout）】
 *
 *  GPU 对图像有不同的访问模式，需要显式指定布局：
 *    UNDEFINED            → 初始状态，内容不保证
 *    COLOR_ATTACHMENT     → 作为颜色输出
 *    PRESENT_SRC          → 准备显示给用户
 *    SHADER_READ_ONLY     → 在着色器中读取（纹理）
 *    TRANSFER_SRC/DST     → 内存传输源/目标
 * ═══════════════════════════════════════════════════════════════════════════
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan_tutorial/utils.hpp>

#include <iostream>
#include <set>
#include <stdexcept>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;

class Ch05App {
  public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

  private:
    GLFWwindow* window_ = nullptr;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue graphicsQueue_ = VK_NULL_HANDLE;
    VkQueue presentQueue_ = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkRenderPass renderPass_ = VK_NULL_HANDLE;

    std::vector<VkImage> swapchainImages_;
    std::vector<VkImageView> swapchainImageViews_;
    VkFormat swapchainImageFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchainExtent_{};

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "Ch05 - RenderPass", nullptr, nullptr);
    }

    void initVulkan() {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapchain();
        createImageViews(); // ← 新增
        createRenderPass(); // ← 新增
        std::cout << "\n✅ 所有 Vulkan 对象初始化完成！\n";
    }

    // ─── 复用前几章的代码（省略重复注释） ────────────────────────────────────

    void createInstance() {
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.apiVersion = VK_API_VERSION_1_3;

        auto exts = getRequiredInstanceExtensions();
        VkInstanceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo = &appInfo;
        ci.enabledExtensionCount = static_cast<uint32_t>(exts.size());
        ci.ppEnabledExtensionNames = exts.data();
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

    void createSurface() {
        VK_CHECK(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_));
    }

    void pickPhysicalDevice() {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance_, &count, nullptr);
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance_, &count, devices.data());
        for (auto& d : devices) {
            QueueFamilyIndices idx = findQueueFamilies(d, surface_);
            if (idx.isComplete() && checkDeviceExtensionSupport(d)) {
                physicalDevice_ = d;
                break;
            }
        }
        if (physicalDevice_ == VK_NULL_HANDLE)
            throw std::runtime_error("无合适GPU");
    }

    void createLogicalDevice() {
        QueueFamilyIndices idx = findQueueFamilies(physicalDevice_, surface_);
        std::set<uint32_t> families = {idx.graphicsFamily.value(), idx.presentFamily.value()};
        const float pri = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> qcis;
        for (uint32_t f : families) {
            VkDeviceQueueCreateInfo qci{};
            qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            qci.queueFamilyIndex = f;
            qci.queueCount = 1;
            qci.pQueuePriorities = &pri;
            qcis.push_back(qci);
        }
        VkPhysicalDeviceFeatures feat{};
        feat.samplerAnisotropy = VK_TRUE;
        VkDeviceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        ci.queueCreateInfoCount = static_cast<uint32_t>(qcis.size());
        ci.pQueueCreateInfos = qcis.data();
        ci.pEnabledFeatures = &feat;
        ci.enabledExtensionCount = static_cast<uint32_t>(DEVICE_EXTENSIONS.size());
        ci.ppEnabledExtensionNames = DEVICE_EXTENSIONS.data();
        if (ENABLE_VALIDATION_LAYERS) {
            ci.enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
            ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();
        }
        VK_CHECK(vkCreateDevice(physicalDevice_, &ci, nullptr, &device_));
        vkGetDeviceQueue(device_, idx.graphicsFamily.value(), 0, &graphicsQueue_);
        vkGetDeviceQueue(device_, idx.presentFamily.value(), 0, &presentQueue_);
    }

    void createSwapchain() {
        SwapChainSupportDetails sc = querySwapChainSupport(physicalDevice_, surface_);
        VkSurfaceFormatKHR fmt = chooseSwapSurfaceFormat(sc.formats);
        VkPresentModeKHR mode = chooseSwapPresentMode(sc.presentModes);
        VkExtent2D ext = chooseSwapExtent(sc.capabilities, window_);

        uint32_t imgCount = sc.capabilities.minImageCount + 1;
        if (sc.capabilities.maxImageCount > 0)
            imgCount = std::min(imgCount, sc.capabilities.maxImageCount);

        VkSwapchainCreateInfoKHR ci{};
        ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        ci.surface = surface_;
        ci.minImageCount = imgCount;
        ci.imageFormat = fmt.format;
        ci.imageColorSpace = fmt.colorSpace;
        ci.imageExtent = ext;
        ci.imageArrayLayers = 1;
        ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ci.preTransform = sc.capabilities.currentTransform;
        ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        ci.presentMode = mode;
        ci.clipped = VK_TRUE;
        VK_CHECK(vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_));

        vkGetSwapchainImagesKHR(device_, swapchain_, &imgCount, nullptr);
        swapchainImages_.resize(imgCount);
        vkGetSwapchainImagesKHR(device_, swapchain_, &imgCount, swapchainImages_.data());
        swapchainImageFormat_ = fmt.format;
        swapchainExtent_ = ext;
        std::cout << "✅ 交换链创建成功（" << imgCount << " 张图像）\n";
    }

    // ─── 核心新增：创建图像视图 ───────────────────────────────────────────────

    void createImageViews() {
        swapchainImageViews_.resize(swapchainImages_.size());

        for (size_t i = 0; i < swapchainImages_.size(); ++i) {
            VkImageViewCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            ci.image = swapchainImages_[i];
            ci.viewType = VK_IMAGE_VIEW_TYPE_2D; // 将图像视作 2D 纹理
            ci.format = swapchainImageFormat_;

            // components.swizzle：通道重映射（IDENTITY = 不重映射）
            // 例如可以将所有通道映射到 R 通道，实现灰度效果
            ci.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            ci.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            ci.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            ci.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

            // subresourceRange：图像的哪个部分被此视图访问
            ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            ci.subresourceRange.baseMipLevel = 0;   // 从第 0 个 mip 开始
            ci.subresourceRange.levelCount = 1;     // 共 1 个 mip 层级
            ci.subresourceRange.baseArrayLayer = 0; // 从第 0 个数组层开始
            ci.subresourceRange.layerCount = 1;     // 共 1 层（非 VR）

            VK_CHECK(vkCreateImageView(device_, &ci, nullptr, &swapchainImageViews_[i]));
        }
        std::cout << "✅ " << swapchainImageViews_.size() << " 个 ImageView 已创建\n";
    }

    // ─── 核心新增：创建渲染流程 ───────────────────────────────────────────────

    void createRenderPass() {
        // ① 附件描述（我们只有一个颜色附件）
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapchainImageFormat_;  // 必须匹配交换链格式
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT; // 不多重采样
        // loadOp：渲染前对附件做什么
        //   CLEAR     → 用清除色清空（我们想要干净的画布）
        //   LOAD      → 保留上一帧内容
        //   DONT_CARE → 不关心，可能是任意值（最快）
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        // storeOp：渲染后对附件做什么
        //   STORE     → 保存结果（我们需要显示到屏幕）
        //   DONT_CARE → 不保存（用于中间附件）
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        // 模板缓冲（本章不使用）
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        // 图像布局转换：
        //   initialLayout: 渲染开始时图像的布局（UNDEFINED = 不关心初始内容）
        //   finalLayout:   渲染结束时图像自动转换到的布局
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        // ② 附件引用（Subpass 通过引用使用附件）
        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0; // 引用上面 index=0 的附件
        // 此附件在 Subpass 执行期间使用的布局
        // GPU 会在进入 Subpass 前自动将图像转换到此布局
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        // ③ Subpass（子流程）描述
        // 一个 RenderPass 可以有多个 Subpass（用于移动端的 Tile-Based 优化）
        // 后续 Subpass 可以读取前一个 Subpass 的输出（避免写回内存）
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; // 图形管线
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        // 其他附件引用（本章不用）:
        // pInputAttachments        → 从之前 subpass 读取
        // pResolveAttachments      → 多重采样解析目标
        // pDepthStencilAttachment  → 深度/模板附件

        // ④ Subpass 依赖（同步控制）
        // 告诉 Vulkan：本 Subpass 必须等待交换链图像可用才能开始
        VkSubpassDependency dependency{};
        // VK_SUBPASS_EXTERNAL = RenderPass 外部（之前的所有操作）
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0; // 我们的第一个（也是唯一的）subpass
        // 在哪个管线阶段需要等待/被等待
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        // 等待写操作完成，才允许写操作开始
        dependency.srcAccessMask = 0;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        // ⑤ 创建 RenderPass
        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        VK_CHECK(vkCreateRenderPass(device_, &renderPassInfo, nullptr, &renderPass_));
        std::cout << "✅ RenderPass 创建成功！\n";
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
                glfwSetWindowShouldClose(window_, GLFW_TRUE);
        }
    }

    void cleanup() {
        vkDestroyRenderPass(device_, renderPass_, nullptr);
        for (auto& iv : swapchainImageViews_)
            vkDestroyImageView(device_, iv, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        vkDestroyDevice(device_, nullptr);
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        vkDestroyInstance(instance_, nullptr);
        glfwDestroyWindow(window_);
        glfwTerminate();
        std::cout << "✅ 清理完成。\n";
    }
};

int main() {
    std::cout << "═══════════════════════════════════════\n";
    std::cout << " 第05章：图像视图与渲染流程\n";
    std::cout << "═══════════════════════════════════════\n\n";

    Ch05App app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ 错误：" << e.what() << "\n";
        return 1;
    }
    return 0;
}
