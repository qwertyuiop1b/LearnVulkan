/**
 * @file ch07_commands.cpp
 * @brief 第07章：帧缓冲（Framebuffer）与命令缓冲（Command Buffer）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【VkFramebuffer — 帧缓冲】
 *
 *  RenderPass 描述了附件的抽象格式，Framebuffer 将具体的 ImageView 与之绑定。
 *  每个交换链图像都需要一个对应的 Framebuffer。
 *
 *  ┌───────────────────────────────────────────────────┐
 *  │  RenderPass（抽象描述）                            │
 *  │    ↓                                              │
 *  │  Framebuffer[0] → ImageView[0] → SwapImage[0]    │
 *  │  Framebuffer[1] → ImageView[1] → SwapImage[1]    │
 *  │  Framebuffer[2] → ImageView[2] → SwapImage[2]    │
 *  └───────────────────────────────────────────────────┘
 *
 * 【VkCommandPool & VkCommandBuffer — 命令池与命令缓冲】
 *
 *  Vulkan 中 CPU 不直接调用绘制函数，而是：
 *    1. 将命令"录制"到命令缓冲
 *    2. 一次性提交整个命令缓冲给 GPU
 *
 *  好处：
 *    - 多线程录制命令（不同线程录制不同 CommandBuffer）
 *    - 预录制固定命令（Static scene），直接重用
 *    - 驱动程序可以优化整批命令
 *
 *  CommandPool → 为特定队列族分配 CommandBuffer 的内存池
 *
 * 【命令录制流程】
 *
 *  vkBeginCommandBuffer()
 *    vkCmdBeginRenderPass()
 *      vkCmdBindPipeline()
 *      vkCmdSetViewport()
 *      vkCmdSetScissor()
 *      vkCmdDraw()           ← 核心绘制命令
 *    vkCmdEndRenderPass()
 *  vkEndCommandBuffer()
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

class Ch07App {
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
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;

    std::vector<VkImage> swapchainImages_;
    std::vector<VkImageView> swapchainImageViews_;
    std::vector<VkFramebuffer> framebuffers_;
    std::vector<VkCommandBuffer> commandBuffers_;
    VkFormat swapchainImageFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchainExtent_{};
    QueueFamilyIndices queueIndices_;

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "Ch07 - Framebuffers & Commands", nullptr, nullptr);
    }

    void initVulkan() {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapchain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();   // ← 新增
        createCommandPool();    // ← 新增
        createCommandBuffers(); // ← 新增
        std::cout << "\n✅ 所有 Vulkan 对象初始化完成！\n";
    }

    // ─── 核心新增：创建帧缓冲 ─────────────────────────────────────────────────

    void createFramebuffers() {
        framebuffers_.resize(swapchainImageViews_.size());

        for (size_t i = 0; i < swapchainImageViews_.size(); ++i) {
            // 帧缓冲的附件数组（与 RenderPass 中附件描述一一对应）
            VkImageView attachments[] = {swapchainImageViews_[i]};

            VkFramebufferCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            ci.renderPass = renderPass_; // 必须与渲染时使用的 RenderPass 兼容
            ci.attachmentCount = 1;
            ci.pAttachments = attachments;
            ci.width = swapchainExtent_.width;
            ci.height = swapchainExtent_.height;
            ci.layers = 1; // 非 VR 始终为 1

            VK_CHECK(vkCreateFramebuffer(device_, &ci, nullptr, &framebuffers_[i]));
        }
        std::cout << "✅ " << framebuffers_.size() << " 个帧缓冲已创建\n";
    }

    // ─── 核心新增：创建命令池 ─────────────────────────────────────────────────

    void createCommandPool() {
        VkCommandPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        // flags 可选：
        //   TRANSIENT_BIT            → 命令缓冲会频繁重录（可优化内存分配）
        //   RESET_COMMAND_BUFFER_BIT → 允许单独重置每个命令缓冲（否则只能整体重置命令池）
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        // 命令池绑定到特定队列族，从它分配的命令缓冲只能提交到该族的队列
        ci.queueFamilyIndex = queueIndices_.graphicsFamily.value();

        VK_CHECK(vkCreateCommandPool(device_, &ci, nullptr, &commandPool_));
        std::cout << "✅ 命令池已创建\n";
    }

    // ─── 核心新增：分配并录制命令缓冲 ────────────────────────────────────────

    void createCommandBuffers() {
        // 为每个帧缓冲分配一个对应的命令缓冲
        commandBuffers_.resize(framebuffers_.size());

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool_;
        // level:
        //   PRIMARY   → 可直接提交到队列
        //   SECONDARY → 不能直接提交，被 PRIMARY 通过 vkCmdExecuteCommands 调用
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers_.size());

        VK_CHECK(vkAllocateCommandBuffers(device_, &allocInfo, commandBuffers_.data()));

        // 录制每个命令缓冲
        for (size_t i = 0; i < commandBuffers_.size(); ++i)
            recordCommandBuffer(commandBuffers_[i], static_cast<uint32_t>(i));

        std::cout << "✅ " << commandBuffers_.size() << " 个命令缓冲已录制\n";
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
        // ── 开始录制 ──────────────────────────────────────────────────────────
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        // flags:
        //   ONE_TIME_SUBMIT_BIT      → 只提交一次，之后重录（一次性命令）
        //   RENDER_PASS_CONTINUE_BIT → Secondary 命令缓冲专用
        //   SIMULTANEOUS_USE_BIT     → 可以在仍处于 Pending 状态时再次提交
        beginInfo.flags = 0;

        VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

        // ── 开始 RenderPass ───────────────────────────────────────────────────
        VkClearValue clearColor = {{{0.02f, 0.02f, 0.05f, 1.0f}}}; // 深蓝黑色背景

        VkRenderPassBeginInfo rpInfo{};
        rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpInfo.renderPass = renderPass_;
        rpInfo.framebuffer = framebuffers_[imageIndex]; // 绑定对应的帧缓冲
        rpInfo.renderArea.offset = {0, 0};
        rpInfo.renderArea.extent = swapchainExtent_; // 渲染范围 = 整个帧缓冲
        rpInfo.clearValueCount = 1;
        rpInfo.pClearValues = &clearColor;

        // subpassContents:
        //   INLINE              → 命令直接内联在 Primary 中
        //   SECONDARY_COMMAND_BUFFERS → 使用 Secondary 命令缓冲
        vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

        // ── 绑定图形管线 ──────────────────────────────────────────────────────
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

        // ── 设置动态视口和裁剪（因为管线中声明了 Dynamic State） ──────────────
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(swapchainExtent_.width);
        viewport.height = static_cast<float>(swapchainExtent_.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapchainExtent_;
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        // ── 发出绘制命令 ──────────────────────────────────────────────────────
        // vkCmdDraw(commandBuffer, vertexCount, instanceCount,
        //           firstVertex, firstInstance)
        //   vertexCount  = 3（三角形的3个顶点）
        //   instanceCount = 1（画1个实例，instanced rendering 时>1）
        //   firstVertex  = 0（从顶点索引0开始）
        //   firstInstance = 0（从实例0开始）
        vkCmdDraw(cmd, 3, 1, 0, 0);

        // ── 结束 RenderPass ───────────────────────────────────────────────────
        vkCmdEndRenderPass(cmd);

        // ── 完成录制 ──────────────────────────────────────────────────────────
        VK_CHECK(vkEndCommandBuffer(cmd));
    }

    // ─── 复用代码（前几章已详解） ────────────────────────────────────────────

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
            if (findQueueFamilies(d, surface_).isComplete() && checkDeviceExtensionSupport(d)) {
                physicalDevice_ = d;
                break;
            }
        }
        if (physicalDevice_ == VK_NULL_HANDLE)
            throw std::runtime_error("无合适GPU");
    }

    void createLogicalDevice() {
        queueIndices_ = findQueueFamilies(physicalDevice_, surface_);
        std::set<uint32_t> families = {queueIndices_.graphicsFamily.value(), queueIndices_.presentFamily.value()};
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
        vkGetDeviceQueue(device_, queueIndices_.graphicsFamily.value(), 0, &graphicsQueue_);
        vkGetDeviceQueue(device_, queueIndices_.presentFamily.value(), 0, &presentQueue_);
    }

    void createSwapchain() {
        SwapChainSupportDetails sc = querySwapChainSupport(physicalDevice_, surface_);
        VkSurfaceFormatKHR fmt = chooseSwapSurfaceFormat(sc.formats);
        VkPresentModeKHR mode = chooseSwapPresentMode(sc.presentModes);
        swapchainExtent_ = chooseSwapExtent(sc.capabilities, window_);
        uint32_t imgCount = sc.capabilities.minImageCount + 1;
        if (sc.capabilities.maxImageCount > 0)
            imgCount = std::min(imgCount, sc.capabilities.maxImageCount);
        VkSwapchainCreateInfoKHR ci{};
        ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        ci.surface = surface_;
        ci.minImageCount = imgCount;
        ci.imageFormat = fmt.format;
        ci.imageColorSpace = fmt.colorSpace;
        ci.imageExtent = swapchainExtent_;
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
    }

    void createImageViews() {
        swapchainImageViews_.resize(swapchainImages_.size());
        for (size_t i = 0; i < swapchainImages_.size(); ++i) {
            VkImageViewCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            ci.image = swapchainImages_[i];
            ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            ci.format = swapchainImageFormat_;
            ci.components = {VK_COMPONENT_SWIZZLE_IDENTITY,
                             VK_COMPONENT_SWIZZLE_IDENTITY,
                             VK_COMPONENT_SWIZZLE_IDENTITY,
                             VK_COMPONENT_SWIZZLE_IDENTITY};
            ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VK_CHECK(vkCreateImageView(device_, &ci, nullptr, &swapchainImageViews_[i]));
        }
    }

    void createRenderPass() {
        VkAttachmentDescription ca{};
        ca.format = swapchainImageFormat_;
        ca.samples = VK_SAMPLE_COUNT_1_BIT;
        ca.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        ca.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        ca.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        ca.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ca.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkAttachmentReference cr{};
        cr.attachment = 0;
        cr.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkSubpassDescription sp{};
        sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sp.colorAttachmentCount = 1;
        sp.pColorAttachments = &cr;
        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.srcAccessMask = 0;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        VkRenderPassCreateInfo rpi{};
        rpi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpi.attachmentCount = 1;
        rpi.pAttachments = &ca;
        rpi.subpassCount = 1;
        rpi.pSubpasses = &sp;
        rpi.dependencyCount = 1;
        rpi.pDependencies = &dep;
        VK_CHECK(vkCreateRenderPass(device_, &rpi, nullptr, &renderPass_));
    }

    void createGraphicsPipeline() {
        VkShaderModule vert = createShaderModuleFromFile(device_, "triangle.vert.spv");
        VkShaderModule frag = createShaderModuleFromFile(device_, "triangle.frag.spv");
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vert;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = frag;
        stages[1].pName = "main";
        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vs{};
        vs.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vs.viewportCount = 1;
        vs.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.lineWidth = 1.0f;
        rs.cullMode = VK_CULL_MODE_BACK_BIT;
        rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = 0xf;
        cba.blendEnable = VK_FALSE;
        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1;
        cb.pAttachments = &cba;
        std::vector<VkDynamicState> dyn = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        ds.dynamicStateCount = static_cast<uint32_t>(dyn.size());
        ds.pDynamicStates = dyn.data();
        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        VK_CHECK(vkCreatePipelineLayout(device_, &pli, nullptr, &pipelineLayout_));
        VkGraphicsPipelineCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pi.stageCount = 2;
        pi.pStages = stages;
        pi.pVertexInputState = &vi;
        pi.pInputAssemblyState = &ia;
        pi.pViewportState = &vs;
        pi.pRasterizationState = &rs;
        pi.pMultisampleState = &ms;
        pi.pColorBlendState = &cb;
        pi.pDynamicState = &ds;
        pi.layout = pipelineLayout_;
        pi.renderPass = renderPass_;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline_));
        vkDestroyShaderModule(device_, frag, nullptr);
        vkDestroyShaderModule(device_, vert, nullptr);
    }

    void mainLoop() {
        std::cout << "\n🎮 命令已录制，第08章起正式渲染...\n";
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
                glfwSetWindowShouldClose(window_, GLFW_TRUE);
        }
    }

    void cleanup() {
        vkDestroyCommandPool(device_, commandPool_, nullptr); // 命令缓冲随命令池自动释放
        for (auto& fb : framebuffers_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        vkDestroyPipeline(device_, pipeline_, nullptr);
        vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
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
    std::cout << " 第07章：帧缓冲与命令缓冲\n";
    std::cout << "═══════════════════════════════════════\n\n";
    Ch07App app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ 错误：" << e.what() << "\n";
        return 1;
    }
    return 0;
}
