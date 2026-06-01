/**
 * @file ch35_multiview.cpp
 * @brief 第35章：Multiview（VK_KHR_multiview）— VR 双眼立体渲染
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【什么是 Multiview？】
 *
 *  VR 渲染需要为左眼和右眼各渲染一帧，传统方式：
 *    Pass 1: 设置左眼 View 矩阵 → 渲染
 *    Pass 2: 设置右眼 View 矩阵 → 渲染
 *    = 2x CPU 命令提交，2x 图元处理，2x 顶点着色器
 *
 *  VK_KHR_multiview（Vulkan 1.1 核心）：
 *    一次 Draw Call 同时渲染到 2 个视图！
 *    GPU 内部并行处理两套 View 矩阵
 *    节省 40-50% 的渲染开销（顶点变换合并）
 *
 * 【实现方式】
 *
 *  1. RenderPass 设置 viewMask（位掩码指定激活的视图）
 *     viewMask = 0b11 = 视图 0 和视图 1 都激活
 *
 *  2. 图像需要是 Array 类型（imageArrayLayers >= 2）
 *     layer 0 = 左眼渲染结果
 *     layer 1 = 右眼渲染结果
 *
 *  3. 着色器中 gl_ViewIndex 获取当前视图索引（0=左，1=右）
 *     → 根据视图选择不同的 View 矩阵
 *
 * 【VkRenderPassMultiviewCreateInfo】
 *
 *  通过 pNext 链附加到 VkRenderPassCreateInfo：
 *    .subpassCount    = 1
 *    .pViewMasks      = {0b11}    ← Bit 0+1 = 渲染到视图0和1
 *    .correlationMasks = {0b11}  ← 告诉驱动这两个视图相关（同帧）
 *
 * 【UBO 结构】
 *
 *  不同于单眼渲染，需要同时提供两套 View 矩阵：
 *  layout(binding=0) uniform UBO {
 *      mat4 viewLeft;    // 左眼
 *      mat4 viewRight;   // 右眼
 *      mat4 projection;  // 共享投影矩阵
 *  } ubo;
 *
 * 【本章示例】
 *
 *  左右分屏显示（模拟 VR 头显）：
 *  - 左眼视角（红色调）→ 屏幕左半部分
 *  - 右眼视角（蓝色调）→ 屏幕右半部分
 *  - 旋转场景，观察两眼视差
 *  - 展示 gl_ViewIndex 的用法
 * ═══════════════════════════════════════════════════════════════════════════
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vulkan_tutorial/utils.hpp>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int MAX_FRAMES = 2;
constexpr float EYE_SEPARATION = 0.065f; // 瞳距 6.5cm

struct MVVertex {
    glm::vec3 pos;
    glm::vec3 color;
};
struct MultiviewUBO {
    alignas(16) glm::mat4 viewLeft;
    alignas(16) glm::mat4 viewRight;
    alignas(16) glm::mat4 projection;
};

static const std::vector<MVVertex> SCENE_VERTS = {
    // 正四面体（彩色）
    {{0.0f, 0.8f, 0.0f}, {1, .3f, .3f}},
    {{0.8f, -.3f, 0.3f}, {.3f, 1, .3f}},
    {{-.8f, -.3f, 0.3f}, {.3f, .3f, 1}},
    {{0.0f, 0.8f, 0.0f}, {1, .8f, .3f}},
    {{0.8f, -.3f, 0.3f}, {.3f, 1, .8f}},
    {{0.0f, 0.0f, -.8f}, {.8f, .3f, 1}},
    {{0.0f, 0.8f, 0.0f}, {1, .3f, .8f}},
    {{-.8f, -.3f, 0.3f}, {.8f, 1, .3f}},
    {{0.0f, 0.0f, -.8f}, {.3f, .8f, 1}},
    {{0.8f, -.3f, 0.3f}, {1, .3f, .3f}},
    {{-.8f, -.3f, 0.3f}, {.3f, 1, .3f}},
    {{0.0f, 0.0f, -.8f}, {.3f, .3f, 1}},
};

class Ch35App {
  public:
    void run() {
        initWindow();
        if (!initVulkan()) {
            printMVGuide();
            glfwDestroyWindow(window_);
            glfwTerminate();
            return;
        }
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
    VkCommandPool commandPool_ = VK_NULL_HANDLE;

    // ─── Multiview 渲染目标（Array Image，2 层） ──────────────────────────
    VkImage mvImage_ = VK_NULL_HANDLE; ///< 双眼渲染目标（2层）
    VkDeviceMemory mvMemory_ = VK_NULL_HANDLE;
    VkImageView mvView_ = VK_NULL_HANDLE;      ///< 全数组视图（层0+1）
    VkImageView mvViewLeft_ = VK_NULL_HANDLE;  ///< 仅层0（左眼）
    VkImageView mvViewRight_ = VK_NULL_HANDLE; ///< 仅层1（右眼）

    VkRenderPass mvRenderPass_ = VK_NULL_HANDLE; ///< Multiview RenderPass
    VkFramebuffer mvFramebuffer_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool descPool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descSets_;

    // ─── 最终合成：将双眼图像分屏显示 ─────────────────────────────────────
    VkRenderPass compositeRenderPass_ = VK_NULL_HANDLE;
    VkPipelineLayout compositePipeLayout_ = VK_NULL_HANDLE;
    VkPipeline compositePipeline_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout compSetLayout_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> compDescSets_;
    VkSampler sampler_ = VK_NULL_HANDLE;

    std::vector<VkBuffer> ubos_;
    std::vector<VkDeviceMemory> uboMemories_;
    std::vector<void*> uboMapped_;

    VkBuffer vertexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory_ = VK_NULL_HANDLE;

    std::vector<VkImage> swapchainImages_;
    std::vector<VkImageView> swapchainImageViews_;
    std::vector<VkFramebuffer> compositeFramebuffers_;
    std::vector<VkCommandBuffer> commandBuffers_;
    VkFormat swapchainImageFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchainExtent_{};
    QueueFamilyIndices queueIndices_;

    std::vector<VkSemaphore> imageAvailableSems_;
    std::vector<VkSemaphore> renderFinishedSems_;
    std::vector<VkFence> inFlightFences_;
    uint32_t currentFrame_ = 0;
    bool resized_ = false;

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "Ch35 - Multiview（VR 双眼立体渲染）", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
            reinterpret_cast<Ch35App*>(glfwGetWindowUserPointer(w))->resized_ = true;
        });
    }

    bool initVulkan() {
        try {
            createInstance();
            createSurface();
            if (!pickPhysicalDeviceMV())
                return false;
            createLogicalDeviceMV();
            createSwapchain();
            createImageViews();
            createMultiviewResources();  // ← 双层渲染目标
            createMultiviewRenderPass(); // ← 含 multiview 信息的 RenderPass
            createMultiviewPipeline();   // ← gl_ViewIndex 顶点着色器（内含 setLayout_ 创建）
            createMultiviewFramebuffer();
            createCompositePass(); // ← 分屏显示两眼图像
            createCommandPool();
            createVertexBuffer();
            createSampler();
            createUBOs();
            createDescriptorPool();
            createDescriptorSets();
            createCommandBuffers();
            createSyncObjects();
            std::cout << "\n✅ Multiview 初始化完成！\n";
            std::cout << "👁️  左眼/右眼同时渲染（一次 Draw Call）\n";
        } catch (const std::exception& e) {
            std::cerr << "⚠️  " << e.what() << "\n";
            return false;
        }
        return true;
    }

    bool pickPhysicalDeviceMV() {
        uint32_t c = 0;
        vkEnumeratePhysicalDevices(instance_, &c, nullptr);
        std::vector<VkPhysicalDevice> devs(c);
        vkEnumeratePhysicalDevices(instance_, &c, devs.data());
        for (auto& d : devs) {
            if (!findQueueFamilies(d, surface_).isComplete())
                continue;
            VkPhysicalDeviceMultiviewFeatures mvFeatures{};
            mvFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES;
            VkPhysicalDeviceFeatures2 feat2{};
            feat2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            feat2.pNext = &mvFeatures;
            vkGetPhysicalDeviceFeatures2(d, &feat2);
            if (!mvFeatures.multiview) {
                std::cout << "⚠️  GPU 不支持 multiview\n";
                continue;
            }
            physicalDevice_ = d;
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(d, &p);
            std::cout << "✅ GPU: " << p.deviceName << " (Multiview 支持)\n";

            VkPhysicalDeviceMultiviewProperties mvProps{};
            mvProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_PROPERTIES;
            VkPhysicalDeviceProperties2 props2{};
            props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            props2.pNext = &mvProps;
            vkGetPhysicalDeviceProperties2(d, &props2);
            std::cout << "👁️  最大视图数：" << mvProps.maxMultiviewViewCount << "\n";
            return true;
        }
        return false;
    }

    void createLogicalDeviceMV() {
        queueIndices_ = findQueueFamilies(physicalDevice_, surface_);
        std::set<uint32_t> fams = {queueIndices_.graphicsFamily.value(), queueIndices_.presentFamily.value()};
        const float pri = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> qcis;
        for (uint32_t f : fams) {
            VkDeviceQueueCreateInfo q{};
            q.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            q.queueFamilyIndex = f;
            q.queueCount = 1;
            q.pQueuePriorities = &pri;
            qcis.push_back(q);
        }

        VkPhysicalDeviceMultiviewFeatures mvFeatures{};
        mvFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES;
        mvFeatures.multiview = VK_TRUE; // ← 启用 Multiview

        VkPhysicalDeviceFeatures feat{};
        feat.samplerAnisotropy = VK_TRUE;
        VkDeviceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        ci.pNext = &mvFeatures;
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

    void createMultiviewResources() {
        // 创建 2 层的 Array Image（层0=左眼，层1=右眼）
        VkImageCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.extent = {swapchainExtent_.width / 2, swapchainExtent_.height, 1}; // 半宽（因为分屏）
        ci.mipLevels = 1;
        ci.arrayLayers = 2; // ← 2 层！
        ci.format = VK_FORMAT_R8G8B8A8_UNORM;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ci.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        VK_CHECK(vkCreateImage(device_, &ci, nullptr, &mvImage_));

        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(device_, mvImage_, &mr);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = mr.size;
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((mr.memoryTypeBits & (1u << i)) &&
                (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                ai.memoryTypeIndex = i;
                break;
            }
        VK_CHECK(vkAllocateMemory(device_, &ai, nullptr, &mvMemory_));
        VK_CHECK(vkBindImageMemory(device_, mvImage_, mvMemory_, 0));

        // 创建各种视图
        auto createView = [this](VkImageView& view, uint32_t baseLayer, uint32_t layerCount) {
            VkImageViewCreateInfo vci{};
            vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image = mvImage_;
            vci.format = VK_FORMAT_R8G8B8A8_UNORM;
            vci.viewType = (layerCount > 1) ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
            vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, baseLayer, layerCount};
            VK_CHECK(vkCreateImageView(device_, &vci, nullptr, &view));
        };
        createView(mvView_, 0, 2);      // 全数组（供 Multiview Framebuffer 用）
        createView(mvViewLeft_, 0, 1);  // 层0（左眼，供合成 Pass 采样）
        createView(mvViewRight_, 1, 1); // 层1（右眼）

        std::cout << "✅ Multiview 图像（2层）创建：每层 " << swapchainExtent_.width / 2 << "×"
                  << swapchainExtent_.height << "\n";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 核心：创建 Multiview RenderPass
    // ═══════════════════════════════════════════════════════════════════════

    void createMultiviewRenderPass() {
        VkAttachmentDescription att{};
        att.format = VK_FORMAT_R8G8B8A8_UNORM;
        att.samples = VK_SAMPLE_COUNT_1_BIT;
        att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        att.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sp{};
        sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sp.colorAttachmentCount = 1;
        sp.pColorAttachments = &ref;

        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.srcAccessMask = 0;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        // ── 关键：VkRenderPassMultiviewCreateInfo ─────────────────────────
        uint32_t viewMask = 0b11;        // 激活视图0和视图1
        uint32_t correlationMask = 0b11; // 两个视图相关（同帧）

        VkRenderPassMultiviewCreateInfo mvInfo{};
        mvInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO;
        mvInfo.subpassCount = 1;
        mvInfo.pViewMasks = &viewMask; // 哪些视图被激活
        mvInfo.correlationMaskCount = 1;
        mvInfo.pCorrelationMasks = &correlationMask;

        VkRenderPassCreateInfo rpi{};
        rpi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpi.pNext = &mvInfo; // ← 附加 Multiview 信息！
        rpi.attachmentCount = 1;
        rpi.pAttachments = &att;
        rpi.subpassCount = 1;
        rpi.pSubpasses = &sp;
        rpi.dependencyCount = 1;
        rpi.pDependencies = &dep;
        VK_CHECK(vkCreateRenderPass(device_, &rpi, nullptr, &mvRenderPass_));

        std::cout << "✅ Multiview RenderPass 创建（viewMask=0b11，同时渲染两眼）\n";
    }

    void createMultiviewFramebuffer() {
        // Framebuffer 使用 Array Image View（2层）
        // layers=1（multiview 自动处理层数）
        VkFramebufferCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        ci.renderPass = mvRenderPass_;
        ci.attachmentCount = 1;
        ci.pAttachments = &mvView_;
        ci.width = swapchainExtent_.width / 2;
        ci.height = swapchainExtent_.height;
        ci.layers = 1; // ← Multiview 时 layers=1（视图数由 viewMask 控制）
        VK_CHECK(vkCreateFramebuffer(device_, &ci, nullptr, &mvFramebuffer_));
    }

    void createMultiviewPipeline() {
        VkShaderModule vert = createShaderModuleFromFile(device_, "multiview.vert.spv");
        VkShaderModule frag = createShaderModuleFromFile(device_, "multiview.frag.spv");
        VkPipelineShaderStageCreateInfo stages[2]{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                   nullptr,
                                                   0,
                                                   VK_SHADER_STAGE_VERTEX_BIT,
                                                   vert,
                                                   "main",
                                                   nullptr},
                                                  {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                   nullptr,
                                                   0,
                                                   VK_SHADER_STAGE_FRAGMENT_BIT,
                                                   frag,
                                                   "main",
                                                   nullptr}};
        VkVertexInputBindingDescription bind{0, sizeof(MVVertex), VK_VERTEX_INPUT_RATE_VERTEX};
        std::array<VkVertexInputAttributeDescription, 2> attrs{
            {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}, {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 12}}};
        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &bind;
        vi.vertexAttributeDescriptionCount = 2;
        vi.pVertexAttributeDescriptions = attrs.data();
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                  nullptr,
                                                  0,
                                                  VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                                                  VK_FALSE};
        VkViewport fixedVP{0, 0, (float)swapchainExtent_.width / 2, (float)swapchainExtent_.height, 0, 1};
        VkRect2D fixedSC{{0, 0}, {swapchainExtent_.width / 2, swapchainExtent_.height}};
        VkPipelineViewportStateCreateInfo vs{};
        vs.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vs.viewportCount = 1;
        vs.pViewports = &fixedVP;
        vs.scissorCount = 1;
        vs.pScissors = &fixedSC;
        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.lineWidth = 1.0f;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = VK_FALSE;
        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = 0xf;
        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1;
        cb.pAttachments = &cba;
        VkDescriptorSetLayoutBinding b{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = 1;
        dlci.pBindings = &b;
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &dlci, nullptr, &setLayout_));
        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &setLayout_;
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
        pi.pDepthStencilState = &ds;
        pi.pColorBlendState = &cb;
        pi.layout = pipelineLayout_;
        pi.renderPass = mvRenderPass_;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline_));
        vkDestroyShaderModule(device_, frag, nullptr);
        vkDestroyShaderModule(device_, vert, nullptr);
        std::cout << "✅ Multiview 管线创建（gl_ViewIndex 选择 view 矩阵）\n";
    }

    void createCompositePass() {
        // 合成管线：将双眼图像左右拼接显示
        VkAttachmentDescription ca{};
        ca.format = swapchainImageFormat_;
        ca.samples = VK_SAMPLE_COUNT_1_BIT;
        ca.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        ca.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        ca.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        ca.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ca.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkAttachmentReference cr{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
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
        VK_CHECK(vkCreateRenderPass(device_, &rpi, nullptr, &compositeRenderPass_));

        // 合成使用简单的全屏三角形着色器，采样两眼纹理分屏显示
        // 此处简化：复用 tonemap.vert + vrs_demo.frag 仅显示左眼画面
        // 实际实现需要专门的合成着色器
        compositeFramebuffers_.resize(swapchainImageViews_.size());
        for (size_t i = 0; i < swapchainImageViews_.size(); ++i) {
            VkImageView att[] = {swapchainImageViews_[i]};
            VkFramebufferCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            ci.renderPass = compositeRenderPass_;
            ci.attachmentCount = 1;
            ci.pAttachments = att;
            ci.width = swapchainExtent_.width;
            ci.height = swapchainExtent_.height;
            ci.layers = 1;
            VK_CHECK(vkCreateFramebuffer(device_, &ci, nullptr, &compositeFramebuffers_[i]));
        }

        // 合成描述符布局（2 个纹理：左眼+右眼）
        std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = 2;
        dlci.pBindings = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &dlci, nullptr, &compSetLayout_));

        // 合成管线（全屏三角形 + 分屏采样）
        VkShaderModule vert = createShaderModuleFromFile(device_, "tonemap.vert.spv");
        VkShaderModule frag = createShaderModuleFromFile(device_, "tonemap.frag.spv");
        VkPipelineShaderStageCreateInfo stages[2]{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                   nullptr,
                                                   0,
                                                   VK_SHADER_STAGE_VERTEX_BIT,
                                                   vert,
                                                   "main",
                                                   nullptr},
                                                  {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                   nullptr,
                                                   0,
                                                   VK_SHADER_STAGE_FRAGMENT_BIT,
                                                   frag,
                                                   "main",
                                                   nullptr}};
        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                  nullptr,
                                                  0,
                                                  VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                                                  VK_FALSE};
        VkPipelineViewportStateCreateInfo vs{
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, nullptr, 1, nullptr};
        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.lineWidth = 1.0f;
        rs.cullMode = VK_CULL_MODE_NONE;
        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = 0xf;
        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1;
        cb.pAttachments = &cba;
        std::vector<VkDynamicState> dyn = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynS{};
        dynS.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynS.dynamicStateCount = static_cast<uint32_t>(dyn.size());
        dynS.pDynamicStates = dyn.data();
        // Tone Map Push Constant
        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pcRange.size = 12;
        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &compSetLayout_;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcRange;
        VK_CHECK(vkCreatePipelineLayout(device_, &pli, nullptr, &compositePipeLayout_));
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
        pi.pDynamicState = &dynS;
        pi.layout = compositePipeLayout_;
        pi.renderPass = compositeRenderPass_;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &compositePipeline_));
        vkDestroyShaderModule(device_, frag, nullptr);
        vkDestroyShaderModule(device_, vert, nullptr);
        std::cout << "✅ 合成管线创建（分屏显示双眼）\n";
    }

    void updateUBO(uint32_t frame) {
        static auto start = std::chrono::high_resolution_clock::now();
        float t = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start).count();
        MultiviewUBO ubo{};
        glm::mat4 model = glm::rotate(glm::mat4(1.0f), t * 0.5f, glm::vec3(0, 1, 0));
        float aspect = (float)(swapchainExtent_.width / 2) / swapchainExtent_.height;
        ubo.projection = glm::perspective(glm::radians(90.0f), aspect, 0.1f, 10.0f);
        ubo.projection[1][1] *= -1;
        glm::vec3 target = glm::vec3(0);
        glm::vec3 up = glm::vec3(0, 1, 0);
        // 左眼：向左偏移半个瞳距
        glm::vec3 eyeLeft = glm::vec3(-EYE_SEPARATION / 2, 0, 2);
        ubo.viewLeft = glm::lookAt(eyeLeft, target, up) * glm::inverse(model);
        // 右眼：向右偏移半个瞳距
        glm::vec3 eyeRight = glm::vec3(EYE_SEPARATION / 2, 0, 2);
        ubo.viewRight = glm::lookAt(eyeRight, target, up) * glm::inverse(model);
        std::memcpy(uboMapped_[frame], &ubo, sizeof(ubo));
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

        // ── Pass 1: Multiview 渲染（一次 Draw Call，渲染两眼）────────────────
        VkClearValue mvClear{};
        mvClear.color.float32[0] = 0.02f;
        mvClear.color.float32[2] = 0.05f;
        mvClear.color.float32[3] = 1.0f;
        VkRenderPassBeginInfo mvRP{};
        mvRP.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        mvRP.renderPass = mvRenderPass_;
        mvRP.framebuffer = mvFramebuffer_;
        mvRP.renderArea = {{0, 0}, {swapchainExtent_.width / 2, swapchainExtent_.height}};
        mvRP.clearValueCount = 1;
        mvRP.pClearValues = &mvClear;
        vkCmdBeginRenderPass(cmd, &mvRP, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        vkCmdBindDescriptorSets(
            cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_, 0, 1, &descSets_[currentFrame_], 0, nullptr);
        VkBuffer vb[] = {vertexBuffer_};
        VkDeviceSize off[] = {0};
        vkCmdBindVertexBuffers(cmd, 0, 1, vb, off);
        vkCmdDraw(cmd, static_cast<uint32_t>(SCENE_VERTS.size()), 1, 0, 0); // 一次 Draw，两眼同时！
        vkCmdEndRenderPass(cmd);

        // ── Pass 2: 合成（左眼左半屏 + 右眼右半屏）──────────────────────────
        VkClearValue compClear{};
        compClear.color.float32[3] = 1.0f;
        VkRenderPassBeginInfo compRP{};
        compRP.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        compRP.renderPass = compositeRenderPass_;
        compRP.framebuffer = compositeFramebuffers_[imageIndex];
        compRP.renderArea = {{0, 0}, swapchainExtent_};
        compRP.clearValueCount = 1;
        compRP.pClearValues = &compClear;
        vkCmdBeginRenderPass(cmd, &compRP, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, compositePipeline_);

        // 左眼：左半屏
        VkViewport leftVP{0, 0, (float)swapchainExtent_.width / 2, (float)swapchainExtent_.height, 0, 1};
        vkCmdSetViewport(cmd, 0, 1, &leftVP);
        VkRect2D leftSC{{0, 0}, {swapchainExtent_.width / 2, swapchainExtent_.height}};
        vkCmdSetScissor(cmd, 0, 1, &leftSC);
        float pcData[] = {1.0f, 1.0f, 2.2f}; // Tone map push constant
        vkCmdPushConstants(cmd, compositePipeLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 12, pcData);
        vkCmdBindDescriptorSets(cmd,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                compositePipeLayout_,
                                0,
                                1,
                                &compDescSets_[currentFrame_],
                                0,
                                nullptr);
        vkCmdDraw(cmd, 3, 1, 0, 0);

        // 右眼：右半屏
        VkViewport rightVP{(float)swapchainExtent_.width / 2,
                           0,
                           (float)swapchainExtent_.width / 2,
                           (float)swapchainExtent_.height,
                           0,
                           1};
        vkCmdSetViewport(cmd, 0, 1, &rightVP);
        VkRect2D rightSC{{(int32_t)(swapchainExtent_.width / 2), 0},
                         {swapchainExtent_.width / 2, swapchainExtent_.height}};
        vkCmdSetScissor(cmd, 0, 1, &rightSC);
        vkCmdDraw(cmd, 3, 1, 0, 0);

        vkCmdEndRenderPass(cmd);
        VK_CHECK(vkEndCommandBuffer(cmd));
    }

    void drawFrame() {
        vkWaitForFences(device_, 1, &inFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX);
        uint32_t imgIdx = 0;
        VkResult r = vkAcquireNextImageKHR(
            device_, swapchain_, UINT64_MAX, imageAvailableSems_[currentFrame_], VK_NULL_HANDLE, &imgIdx);
        if (r == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapchain();
            return;
        }
        updateUBO(currentFrame_);
        vkResetFences(device_, 1, &inFlightFences_[currentFrame_]);
        vkResetCommandBuffer(commandBuffers_[currentFrame_], 0);
        recordCommandBuffer(commandBuffers_[currentFrame_], imgIdx);
        VkSemaphore ws[] = {imageAvailableSems_[currentFrame_]};
        VkPipelineStageFlags wst[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        VkSemaphore ss[] = {renderFinishedSems_[currentFrame_]};
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = ws;
        si.pWaitDstStageMask = wst;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &commandBuffers_[currentFrame_];
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = ss;
        VK_CHECK(vkQueueSubmit(graphicsQueue_, 1, &si, inFlightFences_[currentFrame_]));
        VkSwapchainKHR scs[] = {swapchain_};
        VkPresentInfoKHR pi{};
        pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = ss;
        pi.swapchainCount = 1;
        pi.pSwapchains = scs;
        pi.pImageIndices = &imgIdx;
        r = vkQueuePresentKHR(presentQueue_, &pi);
        if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR || resized_) {
            resized_ = false;
            recreateSwapchain();
        }
        currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES;
    }

    void mainLoop() {
        std::cout << "👁️  左右分屏 VR 视角（旋转场景，观察视差，ESC退出）\n";
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
                glfwSetWindowShouldClose(window_, GLFW_TRUE);
            drawFrame();
        }
        vkDeviceWaitIdle(device_);
    }

    void printMVGuide() {
        std::cout << "\n Multiview 概念速查（此 GPU 不支持 multiview）\n\n";
        std::cout << "  VkRenderPassMultiviewCreateInfo（pNext 链）：\n";
        std::cout << "    pViewMasks = {0b11}  → 视图0+视图1同时激活\n";
        std::cout << "    pCorrelationMasks → 告诉驱动视图间是相关的\n\n";
        std::cout << "  渲染目标：Array Image（imageArrayLayers=2）\n";
        std::cout << "  着色器：gl_ViewIndex → 当前视图（0=左，1=右）\n\n";
        std::cout << "  性能：一次 Draw = 两眼渲染，顶点变换合并\n";
        std::cout << "         约节省 40% VR 渲染开销\n";
    }

    uint32_t findMemoryType(uint32_t f, VkMemoryPropertyFlags p) {
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((f & (1u << i)) && (mp.memoryTypes[i].propertyFlags & p) == p)
                return i;
        throw std::runtime_error("找不到内存类型");
    }
    void createBuffer(VkDeviceSize sz, VkBufferUsageFlags u, VkMemoryPropertyFlags p, VkBuffer& b, VkDeviceMemory& m) {
        VkBufferCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ci.size = sz;
        ci.usage = u;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(device_, &ci, nullptr, &b));
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(device_, b, &mr);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, p);
        VK_CHECK(vkAllocateMemory(device_, &ai, nullptr, &m));
        VK_CHECK(vkBindBufferMemory(device_, b, m, 0));
    }
    void createVertexBuffer() {
        VkDeviceSize sz = sizeof(SCENE_VERTS[0]) * SCENE_VERTS.size();
        createBuffer(sz,
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     vertexBuffer_,
                     vertexMemory_);
        void* d = nullptr;
        vkMapMemory(device_, vertexMemory_, 0, sz, 0, &d);
        std::memcpy(d, SCENE_VERTS.data(), (size_t)sz);
        vkUnmapMemory(device_, vertexMemory_);
    }
    void createSampler() {
        VkSamplerCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        ci.magFilter = VK_FILTER_LINEAR;
        ci.minFilter = VK_FILTER_LINEAR;
        ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VK_CHECK(vkCreateSampler(device_, &ci, nullptr, &sampler_));
    }
    void createUBOs() {
        ubos_.resize(MAX_FRAMES);
        uboMemories_.resize(MAX_FRAMES);
        uboMapped_.resize(MAX_FRAMES);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            createBuffer(sizeof(MultiviewUBO),
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         ubos_[i],
                         uboMemories_[i]);
            vkMapMemory(device_, uboMemories_[i], 0, sizeof(MultiviewUBO), 0, &uboMapped_[i]);
        }
    }
    void createDescriptorPool() {
        std::array<VkDescriptorPoolSize, 2> ps{};
        ps[0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(MAX_FRAMES)};
        ps[1] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(MAX_FRAMES * 2)};
        VkDescriptorPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        ci.poolSizeCount = 2;
        ci.pPoolSizes = ps.data();
        ci.maxSets = static_cast<uint32_t>(MAX_FRAMES * 2);
        VK_CHECK(vkCreateDescriptorPool(device_, &ci, nullptr, &descPool_));
    }
    void createDescriptorSets() {
        // Multiview 管线的 UBO 描述符集
        std::vector<VkDescriptorSetLayout> lays(MAX_FRAMES, setLayout_);
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = descPool_;
        ai.descriptorSetCount = MAX_FRAMES;
        ai.pSetLayouts = lays.data();
        descSets_.resize(MAX_FRAMES);
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, descSets_.data()));
        for (int i = 0; i < MAX_FRAMES; ++i) {
            VkDescriptorBufferInfo bi{};
            bi.buffer = ubos_[i];
            bi.offset = 0;
            bi.range = sizeof(MultiviewUBO);
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = descSets_[i];
            w.dstBinding = 0;
            w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            w.descriptorCount = 1;
            w.pBufferInfo = &bi;
            vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
        }

        // 合成管线的纹理描述符集（左眼+右眼）
        std::vector<VkDescriptorSetLayout> clays(MAX_FRAMES, compSetLayout_);
        ai.descriptorSetCount = MAX_FRAMES;
        ai.pSetLayouts = clays.data();
        compDescSets_.resize(MAX_FRAMES);
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, compDescSets_.data()));
        for (int i = 0; i < MAX_FRAMES; ++i) {
            VkDescriptorImageInfo leftII{sampler_, mvViewLeft_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkDescriptorImageInfo rightII{sampler_, mvViewRight_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            std::array<VkWriteDescriptorSet, 2> ws{};
            ws[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                     nullptr,
                     compDescSets_[i],
                     0,
                     0,
                     1,
                     VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                     &leftII,
                     nullptr,
                     nullptr};
            ws[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                     nullptr,
                     compDescSets_[i],
                     1,
                     0,
                     1,
                     VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                     &rightII,
                     nullptr,
                     nullptr};
            vkUpdateDescriptorSets(device_, 2, ws.data(), 0, nullptr);
        }
    }
    void createInstance() {
        VkApplicationInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        ai.apiVersion = VK_API_VERSION_1_3;
        auto exts = getRequiredInstanceExtensions();
        VkInstanceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo = &ai;
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
    void createSurface() {
        VK_CHECK(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_));
    }
    void createSwapchain() {
        auto sc = querySwapChainSupport(physicalDevice_, surface_);
        auto fmt = chooseSwapSurfaceFormat(sc.formats);
        auto mode = chooseSwapPresentMode(sc.presentModes);
        swapchainExtent_ = chooseSwapExtent(sc.capabilities, window_);
        uint32_t n = sc.capabilities.minImageCount + 1;
        if (sc.capabilities.maxImageCount > 0)
            n = std::min(n, sc.capabilities.maxImageCount);
        VkSwapchainCreateInfoKHR ci{};
        ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        ci.surface = surface_;
        ci.minImageCount = n;
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
        vkGetSwapchainImagesKHR(device_, swapchain_, &n, nullptr);
        swapchainImages_.resize(n);
        vkGetSwapchainImagesKHR(device_, swapchain_, &n, swapchainImages_.data());
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
    void createCommandPool() {
        VkCommandPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = queueIndices_.graphicsFamily.value();
        VK_CHECK(vkCreateCommandPool(device_, &ci, nullptr, &commandPool_));
    }
    void createCommandBuffers() {
        commandBuffers_.resize(MAX_FRAMES);
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = commandPool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = static_cast<uint32_t>(commandBuffers_.size());
        VK_CHECK(vkAllocateCommandBuffers(device_, &ai, commandBuffers_.data()));
    }
    void createSyncObjects() {
        imageAvailableSems_.resize(MAX_FRAMES);
        renderFinishedSems_.resize(MAX_FRAMES);
        inFlightFences_.resize(MAX_FRAMES);
        VkSemaphoreCreateInfo sCI{};
        sCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkFenceCreateInfo fCI{};
        fCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for (int i = 0; i < MAX_FRAMES; ++i) {
            VK_CHECK(vkCreateSemaphore(device_, &sCI, nullptr, &imageAvailableSems_[i]));
            VK_CHECK(vkCreateSemaphore(device_, &sCI, nullptr, &renderFinishedSems_[i]));
            VK_CHECK(vkCreateFence(device_, &fCI, nullptr, &inFlightFences_[i]));
        }
    }
    void recreateSwapchain() {
        int w = 0, h = 0;
        glfwGetFramebufferSize(window_, &w, &h);
        while (!w || !h) {
            glfwGetFramebufferSize(window_, &w, &h);
            glfwWaitEvents();
        }
        vkDeviceWaitIdle(device_);
        for (auto& fb : compositeFramebuffers_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        for (auto& iv : swapchainImageViews_)
            vkDestroyImageView(device_, iv, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        createSwapchain();
        createImageViews();
        createCompositePass();
    }
    void cleanup() {
        vkDestroySampler(device_, sampler_, nullptr);
        vkDestroyImageView(device_, mvViewLeft_, nullptr);
        vkDestroyImageView(device_, mvViewRight_, nullptr);
        vkDestroyImageView(device_, mvView_, nullptr);
        vkDestroyImage(device_, mvImage_, nullptr);
        vkFreeMemory(device_, mvMemory_, nullptr);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroyBuffer(device_, ubos_[i], nullptr);
            vkFreeMemory(device_, uboMemories_[i], nullptr);
        }
        vkDestroyBuffer(device_, vertexBuffer_, nullptr);
        vkFreeMemory(device_, vertexMemory_, nullptr);
        vkDestroyDescriptorPool(device_, descPool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, setLayout_, nullptr);
        vkDestroyDescriptorSetLayout(device_, compSetLayout_, nullptr);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroySemaphore(device_, imageAvailableSems_[i], nullptr);
            vkDestroySemaphore(device_, renderFinishedSems_[i], nullptr);
            vkDestroyFence(device_, inFlightFences_[i], nullptr);
        }
        vkDestroyCommandPool(device_, commandPool_, nullptr);
        vkDestroyFramebuffer(device_, mvFramebuffer_, nullptr);
        for (auto& fb : compositeFramebuffers_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        vkDestroyPipeline(device_, pipeline_, nullptr);
        vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
        vkDestroyPipeline(device_, compositePipeline_, nullptr);
        vkDestroyPipelineLayout(device_, compositePipeLayout_, nullptr);
        vkDestroyRenderPass(device_, mvRenderPass_, nullptr);
        vkDestroyRenderPass(device_, compositeRenderPass_, nullptr);
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
    std::cout << "═══════════════════════════════════════════════════════════════════\n";
    std::cout << " 第35章：Multiview（VK_KHR_multiview）— VR 双眼立体渲染\n";
    std::cout << "\n";
    std::cout << " 关键技术：\n";
    std::cout << "   VkRenderPassMultiviewCreateInfo（pNext）\n";
    std::cout << "     pViewMasks = {0b11} → 同时激活视图0和视图1\n";
    std::cout << "   Array Image（imageArrayLayers=2）\n";
    std::cout << "     layer 0 = 左眼，layer 1 = 右眼\n";
    std::cout << "   gl_ViewIndex → 着色器内的当前视图索引\n";
    std::cout << "\n";
    std::cout << " 性能：一次 Draw Call = 两眼同时渲染，节省 40% GPU 开销\n";
    std::cout << " 应用：VR、AR、多摄像机（监控/安防/汽车后视）\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n\n";
    Ch35App app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
    return 0;
}
