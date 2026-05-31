/**
 * @file ch06_pipeline.cpp
 * @brief 第06章：图形管线（Graphics Pipeline）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【图形管线总览】
 *
 *  Vulkan 图形管线是将几何数据转换为屏幕像素的完整流程：
 *
 *  CPU 端数据
 *    │
 *    ▼
 *  ┌─────────────────┐   可编程
 *  │ Vertex Input    │  ← 顶点格式描述（绑定、属性）
 *  └────────┬────────┘
 *           │
 *  ┌────────▼────────┐   可编程 ★
 *  │ Vertex Shader   │  ← 顶点位置变换（MVP矩阵）
 *  └────────┬────────┘
 *           │
 *  ┌────────▼────────┐   固定
 *  │ Input Assembly  │  ← 如何组装图元（三角形、线条）
 *  └────────┬────────┘
 *           │
 *  ┌────────▼────────┐   固定（可配置）
 *  │ Tessellation    │  ← 曲面细分（可选）
 *  └────────┬────────┘
 *           │
 *  ┌────────▼────────┐   可编程（可选）
 *  │ Geometry Shader │  ← 几何着色器（可选）
 *  └────────┬────────┘
 *           │
 *  ┌────────▼────────┐   固定
 *  │ Rasterization   │  ← 三角形 → 片段（像素）
 *  └────────┬────────┘
 *           │
 *  ┌────────▼────────┐   可编程 ★
 *  │ Fragment Shader │  ← 计算每个像素的颜色
 *  └────────┬────────┘
 *           │
 *  ┌────────▼────────┐   固定（可配置）
 *  │ Color Blending  │  ← 与帧缓冲混合（透明度）
 *  └────────┬────────┘
 *           │
 *    屏幕像素输出
 *
 * 【与 OpenGL 的对比】
 *
 *  OpenGL：可以随时修改管线状态（glEnable/glDisable）
 *  Vulkan：整个管线状态固化为一个对象，切换管线有明确开销
 *          → 鼓励提前创建所有管线，避免运行时状态变化
 *
 * 【着色器模块（Shader Module）】
 *
 *  Vulkan 使用 SPIR-V 字节码格式（不是 GLSL 源码）。
 *  glslc 将 shaders/pipeline.vert 与 pipeline.frag 编译为 SPIR-V，
 *  运行时通过 createShaderModuleFromFile 加载。
 * ═══════════════════════════════════════════════════════════════════════════
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan_tutorial/utils.hpp>

#include <iostream>
#include <set>
#include <stdexcept>

constexpr uint32_t WIDTH  = 800;
constexpr uint32_t HEIGHT = 600;

class Ch06App {
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
    VkRenderPass     renderPass_     = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline       pipeline_       = VK_NULL_HANDLE;

    std::vector<VkImage>     swapchainImages_;
    std::vector<VkImageView> swapchainImageViews_;
    VkFormat                 swapchainImageFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D               swapchainExtent_{};

    void initWindow()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE,  GLFW_FALSE);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "Ch06 - Graphics Pipeline", nullptr, nullptr);
    }

    void initVulkan()
    {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapchain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();   // ← 本章核心
        std::cout << "\n✅ 图形管线创建完成！\n";
    }

    // ─── 核心：创建图形管线 ───────────────────────────────────────────────────

    void createGraphicsPipeline()
    {
        // ═══════════════════════════════════════════════════════════════════
        // ① 可编程阶段：着色器
        // ═══════════════════════════════════════════════════════════════════

        VkShaderModule vertModule = createShaderModuleFromFile(device_, "pipeline.vert.spv");
        VkShaderModule fragModule = createShaderModuleFromFile(device_, "pipeline.frag.spv");

        // 着色器阶段创建信息
        VkPipelineShaderStageCreateInfo vertStage{};
        vertStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertStage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
        vertStage.module = vertModule;
        vertStage.pName  = "main";   // 着色器入口函数名
        // pSpecializationInfo：可以在创建管线时为着色器常量赋值（优化机会）

        VkPipelineShaderStageCreateInfo fragStage{};
        fragStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragStage.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragStage.module = fragModule;
        fragStage.pName  = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = { vertStage, fragStage };

        // ═══════════════════════════════════════════════════════════════════
        // ② 顶点输入：描述顶点数据格式
        // ═══════════════════════════════════════════════════════════════════
        //
        // 顶点坐标在 pipeline.vert 中硬编码（gl_VertexIndex），
        // 所以这里顶点输入为空。第09章会在这里描述顶点缓冲布局。
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount   = 0;   // 无顶点缓冲绑定
        vertexInputInfo.vertexAttributeDescriptionCount = 0;   // 无顶点属性

        // ═══════════════════════════════════════════════════════════════════
        // ③ 图元装配（Input Assembly）
        // ═══════════════════════════════════════════════════════════════════
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        // topology 决定顶点如何组合成几何体：
        //   TRIANGLE_LIST  → 每3个顶点组成一个独立三角形（最常用）
        //   TRIANGLE_STRIP → 相邻三角形共享边（省顶点）
        //   LINE_LIST      → 每2个顶点一条线
        //   POINT_LIST     → 每个顶点一个点
        inputAssembly.topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        // ═══════════════════════════════════════════════════════════════════
        // ④ 视口与裁剪矩形
        // ═══════════════════════════════════════════════════════════════════
        //
        // 视口（Viewport）：NDC 坐标 → 屏幕坐标的映射
        //   (0,0) 到 (width, height) 是视口范围
        //   minDepth/maxDepth：深度范围，通常 0.0~1.0
        VkViewport viewport{};
        viewport.x        = 0.0f;
        viewport.y        = 0.0f;
        viewport.width    = static_cast<float>(swapchainExtent_.width);
        viewport.height   = static_cast<float>(swapchainExtent_.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        // 裁剪矩形（Scissor）：超出此矩形的片段被丢弃
        // 通常等于视口大小
        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapchainExtent_;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports    = &viewport;
        viewportState.scissorCount  = 1;
        viewportState.pScissors     = &scissor;

        // ═══════════════════════════════════════════════════════════════════
        // ⑤ 光栅化（Rasterization）
        // ═══════════════════════════════════════════════════════════════════
        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        // depthClampEnable：超出近/远裁剪面的片段是否截断（而非丢弃）
        // 需要开启 GPU 特性，阴影贴图时有用
        rasterizer.depthClampEnable        = VK_FALSE;
        // rasterizerDiscardEnable：是否禁止几何体通过光栅化阶段（用于只用 Transform Feedback）
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        // polygonMode：几何体如何渲染
        //   FILL  → 填充多边形内部（正常渲染）
        //   LINE  → 只渲染边线（线框模式，需 fillModeNonSolid 特性）
        //   POINT → 只渲染顶点（需 fillModeNonSolid 特性）
        rasterizer.polygonMode             = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth               = 1.0f;   // 线宽（> 1.0 需要 wideLines 特性）
        // cullMode：背面剔除，减少不必要的片段处理
        //   NONE      → 不剔除
        //   BACK_BIT  → 剔除背面（最常用）
        //   FRONT_BIT → 剔除正面
        rasterizer.cullMode                = VK_CULL_MODE_BACK_BIT;
        // frontFace：如何判断哪面是正面（顶点顺序）
        //   CLOCKWISE         → 顺时针为正面
        //   COUNTER_CLOCKWISE → 逆时针为正面（OpenGL 默认）
        // 注意：Vulkan Y 轴朝下，与 OpenGL 相反！
        rasterizer.frontFace               = VK_FRONT_FACE_CLOCKWISE;
        // 深度偏移（shadow mapping 中避免 shadow acne）
        rasterizer.depthBiasEnable         = VK_FALSE;

        // ═══════════════════════════════════════════════════════════════════
        // ⑥ 多重采样（MSAA 抗锯齿）
        // ═══════════════════════════════════════════════════════════════════
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable  = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;   // 暂不使用 MSAA

        // ═══════════════════════════════════════════════════════════════════
        // ⑦ 深度/模板测试（本章暂不使用，第12章启用）
        // ═══════════════════════════════════════════════════════════════════
        // pDepthStencilState = nullptr;

        // ═══════════════════════════════════════════════════════════════════
        // ⑧ 颜色混合（Color Blending）
        // ═══════════════════════════════════════════════════════════════════
        //
        // 片段着色器输出的颜色如何与帧缓冲中已有的颜色混合：
        //
        // 【不混合（覆盖写入）】finalColor = newColor
        // 【Alpha 混合（透明效果）】
        //   finalColor.rgb = newAlpha * newColor + (1-newAlpha) * oldColor
        //   finalColor.a   = newAlpha

        // 每个附件的混合配置
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;   // 禁用混合：直接覆盖
        // 如果启用混合（VK_TRUE），需要配置以下参数实现 Alpha 混合：
        // colorBlendAttachment.blendEnable         = VK_TRUE;
        // colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        // colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        // colorBlendAttachment.colorBlendOp        = VK_BLEND_OP_ADD;
        // colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        // colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        // colorBlendAttachment.alphaBlendOp        = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType             = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable     = VK_FALSE;   // 位操作混合（与 blendEnable 互斥）
        colorBlending.attachmentCount   = 1;
        colorBlending.pAttachments      = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        // ═══════════════════════════════════════════════════════════════════
        // ⑨ 动态状态（Dynamic State）
        // ═══════════════════════════════════════════════════════════════════
        //
        // 部分管线状态可以设置为"动态的"，在绘制命令时再指定
        // 这避免了每次窗口大小变化都重建管线
        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates    = dynamicStates.data();

        // ═══════════════════════════════════════════════════════════════════
        // ⑩ 管线布局（Pipeline Layout）
        // ═══════════════════════════════════════════════════════════════════
        //
        // 描述着色器使用的资源布局：
        //   setLayouts     → 描述符集布局（Uniform Buffer, 纹理等）
        //   pushConstants  → 推送常量（小量频繁更新的数据）
        // 本章着色器不需要外部资源，布局为空
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount         = 0;   // 无描述符集
        pipelineLayoutInfo.pushConstantRangeCount = 0;   // 无推送常量

        VK_CHECK(vkCreatePipelineLayout(device_, &pipelineLayoutInfo,
                                        nullptr, &pipelineLayout_));

        // ═══════════════════════════════════════════════════════════════════
        // ⑪ 最终：创建图形管线！
        // ═══════════════════════════════════════════════════════════════════
        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount          = 2;
        pipelineInfo.pStages             = shaderStages;
        pipelineInfo.pVertexInputState   = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState      = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState   = &multisampling;
        pipelineInfo.pDepthStencilState  = nullptr;
        pipelineInfo.pColorBlendState    = &colorBlending;
        pipelineInfo.pDynamicState       = &dynamicState;
        pipelineInfo.layout              = pipelineLayout_;
        pipelineInfo.renderPass          = renderPass_;
        pipelineInfo.subpass             = 0;   // 使用 RenderPass 的第0个 Subpass
        // basePipelineHandle：从已有管线派生（减少创建时间，本章不使用）
        pipelineInfo.basePipelineHandle  = VK_NULL_HANDLE;
        pipelineInfo.basePipelineIndex   = -1;

        // 第二个参数是 PipelineCache（可以跨多次运行缓存管线，加速启动）
        VK_CHECK(vkCreateGraphicsPipelines(
            device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline_));

        // 着色器模块在管线创建后可以立即销毁
        // 管线已经包含了 SPIR-V 字节码的编译结果
        vkDestroyShaderModule(device_, fragModule, nullptr);
        vkDestroyShaderModule(device_, vertModule, nullptr);

        std::cout << "✅ 图形管线创建成功！\n";
        std::cout << "   管线包含：\n";
        std::cout << "   - 顶点着色器（pipeline.vert，硬编码三角形坐标）\n";
        std::cout << "   - 片段着色器（pipeline.frag，输出红色）\n";
        std::cout << "   - 光栅化：FILL 模式，背面剔除\n";
        std::cout << "   - 混合：禁用（直接覆盖写入）\n";
    }

    // ─── 复用代码（前几章已详解，此处省略注释） ──────────────────────────────

    void createInstance()
    {
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
        ci.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
        if (ENABLE_VALIDATION_LAYERS) {
            ci.enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
            ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();
        }
        VK_CHECK(vkCreateInstance(&ci, nullptr, &instance_));
    }

    void createSurface()
    {
        VK_CHECK(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_));
    }

    void pickPhysicalDevice()
    {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance_, &count, nullptr);
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance_, &count, devices.data());
        for (auto& d : devices) {
            if (findQueueFamilies(d, surface_).isComplete() &&
                checkDeviceExtensionSupport(d)) {
                physicalDevice_ = d;
                break;
            }
        }
        if (physicalDevice_ == VK_NULL_HANDLE) throw std::runtime_error("无合适GPU");
    }

    void createLogicalDevice()
    {
        QueueFamilyIndices idx = findQueueFamilies(physicalDevice_, surface_);
        std::set<uint32_t> families = {idx.graphicsFamily.value(), idx.presentFamily.value()};
        const float pri = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> qcis;
        for (uint32_t f : families) {
            VkDeviceQueueCreateInfo qci{};
            qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            qci.queueFamilyIndex = f; qci.queueCount = 1; qci.pQueuePriorities = &pri;
            qcis.push_back(qci);
        }
        VkPhysicalDeviceFeatures feat{}; feat.samplerAnisotropy = VK_TRUE;
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

    void createSwapchain()
    {
        SwapChainSupportDetails sc = querySwapChainSupport(physicalDevice_, surface_);
        VkSurfaceFormatKHR fmt = chooseSwapSurfaceFormat(sc.formats);
        VkPresentModeKHR mode = chooseSwapPresentMode(sc.presentModes);
        VkExtent2D ext = chooseSwapExtent(sc.capabilities, window_);
        uint32_t imgCount = sc.capabilities.minImageCount + 1;
        if (sc.capabilities.maxImageCount > 0)
            imgCount = std::min(imgCount, sc.capabilities.maxImageCount);
        VkSwapchainCreateInfoKHR ci{};
        ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        ci.surface = surface_; ci.minImageCount = imgCount;
        ci.imageFormat = fmt.format; ci.imageColorSpace = fmt.colorSpace;
        ci.imageExtent = ext; ci.imageArrayLayers = 1;
        ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ci.preTransform = sc.capabilities.currentTransform;
        ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        ci.presentMode = mode; ci.clipped = VK_TRUE;
        VK_CHECK(vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_));
        vkGetSwapchainImagesKHR(device_, swapchain_, &imgCount, nullptr);
        swapchainImages_.resize(imgCount);
        vkGetSwapchainImagesKHR(device_, swapchain_, &imgCount, swapchainImages_.data());
        swapchainImageFormat_ = fmt.format; swapchainExtent_ = ext;
    }

    void createImageViews()
    {
        swapchainImageViews_.resize(swapchainImages_.size());
        for (size_t i = 0; i < swapchainImages_.size(); ++i) {
            VkImageViewCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            ci.image = swapchainImages_[i]; ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            ci.format = swapchainImageFormat_;
            ci.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                             VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
            ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VK_CHECK(vkCreateImageView(device_, &ci, nullptr, &swapchainImageViews_[i]));
        }
    }

    void createRenderPass()
    {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapchainImageFormat_;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorRef{};
        colorRef.attachment = 0;
        colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorRef;

        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL; dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.srcAccessMask = 0;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo rpInfo{};
        rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpInfo.attachmentCount = 1; rpInfo.pAttachments = &colorAttachment;
        rpInfo.subpassCount = 1; rpInfo.pSubpasses = &subpass;
        rpInfo.dependencyCount = 1; rpInfo.pDependencies = &dep;
        VK_CHECK(vkCreateRenderPass(device_, &rpInfo, nullptr, &renderPass_));
    }

    void mainLoop()
    {
        std::cout << "\n🎮 窗口已打开（管线已就绪，第08章才会真正渲染）...\n";
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
                glfwSetWindowShouldClose(window_, GLFW_TRUE);
        }
    }

    void cleanup()
    {
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

int main()
{
    std::cout << "═══════════════════════════════════════\n";
    std::cout << " 第06章：图形管线\n";
    std::cout << "═══════════════════════════════════════\n\n";

    Ch06App app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ 错误：" << e.what() << "\n";
        return 1;
    }
    return 0;
}
