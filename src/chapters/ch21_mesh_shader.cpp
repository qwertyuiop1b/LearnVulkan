/**
 * @file ch21_mesh_shader.cpp
 * @brief 第21章：Mesh Shader（网格着色器）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【传统顶点管线 vs Mesh Shader 管线】
 *
 *  传统图形管线：
 *    CPU:  vkCmdBindVertexBuffers()  → 绑定顶点缓冲
 *          vkCmdDraw(N, 1, 0, 0)     → 触发 N 个顶点
 *    GPU:  Vertex Fetch → Vertex Shader → Primitive Assembly → Rasterization
 *
 *  Mesh Shader 管线：
 *    CPU:  vkCmdDrawMeshTasksEXT(X, Y, Z)  → 触发 X×Y×Z 个 task workgroup
 *    GPU:  Task Shader (可选) → Mesh Shader → Rasterization
 *          ↑ 完全在 GPU 上生成顶点和图元！
 *
 * 【Mesh Shader 的优势】
 *
 *  1. 无顶点缓冲绑定：顶点数据存在 SSBO，着色器直接读取
 *  2. 灵活图元生成：可以生成任意拓扑（不限于三角形）
 *  3. Meshlet 剔除：在 Task Shader 中做 GPU 端背面剔除/视锥体剔除
 *  4. 更好的 LOD：Task Shader 根据距离选择不同精细度
 *
 * 【Meshlet】
 *
 *  大型 3D 模型被切分为小块（通常 64-128 顶点，~126 三角形），
 *  每个 meshlet 对应一个 Mesh Shader workgroup。
 *  这使得 GPU 可以高效地以 meshlet 粒度做剔除和 LOD 选择。
 *
 *  ┌────────────────────────────────────────────────────────────┐
 *  │  Meshlet 化的兔子模型（~100k 三角形）                       │
 *  │                                                            │
 *  │  [M0][M1][M2]  ← 背面 meshlet，Task Shader 直接剔除        │
 *  │  [M3][M4][M5]  ← 可见 meshlet，启动 Mesh Shader workgroup  │
 *  └────────────────────────────────────────────────────────────┘
 *
 * 【所需扩展】
 *  VK_EXT_mesh_shader（Vulkan 1.3 或扩展）
 *  着色器：.mesh / .task（新的 GLSL 扩展 GL_EXT_mesh_shader）
 *
 * 【本章示例】
 *  在 Mesh Shader 中直接生成一个旋转的彩色三角形，
 *  无需任何顶点缓冲，演示完整的 Mesh Shader 管线。
 * ═══════════════════════════════════════════════════════════════════════════
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan_tutorial/utils.hpp>

#include <chrono>
#include <iostream>
#include <set>
#include <stdexcept>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int MAX_FRAMES = 2;

struct PushConstants {
    float time;
    float padding[3];
};

class Ch21App {
  public:
    void run() {
        initWindow();
        if (!initVulkan()) {
            printMeshShaderGuide();
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

    std::vector<VkSemaphore> imageAvailableSems_;
    std::vector<VkSemaphore> renderFinishedSems_;
    std::vector<VkFence> inFlightFences_;
    uint32_t currentFrame_ = 0;
    bool resized_ = false;

    // Mesh Shader 函数指针
    PFN_vkCmdDrawMeshTasksEXT fpCmdDrawMeshTasks_ = nullptr;

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "Ch21 - Mesh Shader（网格着色器）", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
            reinterpret_cast<Ch21App*>(glfwGetWindowUserPointer(w))->resized_ = true;
        });
    }

    bool initVulkan() {
        try {
            createInstance();
            createSurface();
            if (!pickPhysicalDeviceMS())
                return false;
            createLogicalDeviceMS();
            loadMSFunctionPointers();
            createSwapchain();
            createImageViews();
            createRenderPass();
            createMeshShaderPipeline();
            createFramebuffers();
            createCommandPool();
            createCommandBuffers();
            createSyncObjects();
            std::cout << "\n✅ Mesh Shader 管线初始化完成！\n";
        } catch (const std::exception& e) {
            std::cerr << "⚠️  Mesh Shader 初始化失败：" << e.what() << "\n";
            return false;
        }
        return true;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 检查 Mesh Shader 支持
    // ═══════════════════════════════════════════════════════════════════════

    bool pickPhysicalDeviceMS() {
        uint32_t c = 0;
        vkEnumeratePhysicalDevices(instance_, &c, nullptr);
        std::vector<VkPhysicalDevice> devs(c);
        vkEnumeratePhysicalDevices(instance_, &c, devs.data());

        for (auto& d : devs) {
            if (!findQueueFamilies(d, surface_).isComplete())
                continue;

            uint32_t extCount = 0;
            vkEnumerateDeviceExtensionProperties(d, nullptr, &extCount, nullptr);
            std::vector<VkExtensionProperties> exts(extCount);
            vkEnumerateDeviceExtensionProperties(d, nullptr, &extCount, exts.data());

            bool hasMeshExt = false;
            for (auto& ext : exts)
                if (std::string(ext.extensionName) == "VK_EXT_mesh_shader") {
                    hasMeshExt = true;
                    break;
                }

            if (!hasMeshExt) {
                std::cout << "⚠️  GPU 不支持 VK_EXT_mesh_shader\n";
                continue;
            }

            // 查询 Mesh Shader 特性
            VkPhysicalDeviceMeshShaderFeaturesEXT msFeatures{};
            msFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
            VkPhysicalDeviceFeatures2 feat2{};
            feat2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            feat2.pNext = &msFeatures;
            vkGetPhysicalDeviceFeatures2(d, &feat2);

            if (!msFeatures.meshShader)
                continue;

            physicalDevice_ = d;
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(d, &p);

            // 打印 Mesh Shader 属性
            VkPhysicalDeviceMeshShaderPropertiesEXT msProps{};
            msProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT;
            VkPhysicalDeviceProperties2 props2{};
            props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            props2.pNext = &msProps;
            vkGetPhysicalDeviceProperties2(d, &props2);

            std::cout << "✅ GPU: " << p.deviceName << "\n";
            std::cout << "🔷 最大 Mesh Output 顶点：" << msProps.maxMeshOutputVertices << "\n";
            std::cout << "🔷 最大 Mesh Output 图元：" << msProps.maxMeshOutputPrimitives << "\n";
            std::cout << "🔷 最大 Task Workgroup 大小：(" << msProps.maxTaskWorkGroupSize[0] << ","
                      << msProps.maxTaskWorkGroupSize[1] << "," << msProps.maxTaskWorkGroupSize[2] << ")\n";
            return true;
        }
        return false;
    }

    void createLogicalDeviceMS() {
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

        // 启用 Mesh Shader 特性
        VkPhysicalDeviceMeshShaderFeaturesEXT msFeatures{};
        msFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
        msFeatures.meshShader = VK_TRUE;
        msFeatures.taskShader = VK_TRUE; // 可选，本章使用

        // 同时启用维护特性（Mesh Shader 依赖）
        VkPhysicalDeviceMaintenance4Features maint4{};
        maint4.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES;
        maint4.maintenance4 = VK_TRUE;
        maint4.pNext = &msFeatures;

        // 扩展列表中加入 mesh shader 扩展
        static const std::vector<const char*> ms_exts = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            "VK_KHR_portability_subset",
            "VK_EXT_mesh_shader",
            "VK_KHR_spirv_1_4",
            "VK_KHR_shader_float_controls",
        };

        VkPhysicalDeviceFeatures feat{};
        feat.samplerAnisotropy = VK_TRUE;
        VkDeviceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        ci.pNext = &maint4;
        ci.queueCreateInfoCount = static_cast<uint32_t>(qcis.size());
        ci.pQueueCreateInfos = qcis.data();
        ci.pEnabledFeatures = &feat;
        ci.enabledExtensionCount = static_cast<uint32_t>(ms_exts.size());
        ci.ppEnabledExtensionNames = ms_exts.data();
        if (ENABLE_VALIDATION_LAYERS) {
            ci.enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
            ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();
        }
        VK_CHECK(vkCreateDevice(physicalDevice_, &ci, nullptr, &device_));
        vkGetDeviceQueue(device_, queueIndices_.graphicsFamily.value(), 0, &graphicsQueue_);
        vkGetDeviceQueue(device_, queueIndices_.presentFamily.value(), 0, &presentQueue_);
    }

    void loadMSFunctionPointers() {
        fpCmdDrawMeshTasks_ =
            reinterpret_cast<PFN_vkCmdDrawMeshTasksEXT>(vkGetDeviceProcAddr(device_, "vkCmdDrawMeshTasksEXT"));
        if (!fpCmdDrawMeshTasks_)
            throw std::runtime_error("无法加载 vkCmdDrawMeshTasksEXT");
        std::cout << "✅ Mesh Shader 函数指针加载完成\n";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Mesh Shader 管线创建（关键：用 pMeshShaderStage 替代 pVertexInputState）
    // ═══════════════════════════════════════════════════════════════════════

    void createMeshShaderPipeline() {
        // 编译 Mesh Shader：需要 glslc 支持 --target-env=vulkan1.2 + SPIR-V 1.4
        // mesh_basic.mesh → mesh_basic.mesh.spv
        // mesh_basic.frag → mesh_basic.frag.spv
        VkShaderModule meshModule = createShaderModuleFromFile(device_, "mesh_basic.mesh.spv");
        VkShaderModule fragModule = createShaderModuleFromFile(device_, "mesh_basic.frag.spv");

        // ── Mesh Shader 管线只有 2 个着色器阶段（无 vertex stage！） ─────────
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_MESH_BIT_EXT; // ← 新阶段类型！
        stages[0].module = meshModule;
        stages[0].pName = "main";

        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fragModule;
        stages[1].pName = "main";

        // ── 无顶点输入状态！（Mesh Shader 自行生成顶点） ────────────────────
        // 注意：pVertexInputState = nullptr（不设置）

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vs{
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, nullptr, 1, nullptr};
        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.lineWidth = 1.0f;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
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

        // Push constant：传递时间（旋转动画）
        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT;
        pcRange.offset = 0;
        pcRange.size = sizeof(PushConstants);

        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcRange;
        VK_CHECK(vkCreatePipelineLayout(device_, &pli, nullptr, &pipelineLayout_));

        VkGraphicsPipelineCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pi.stageCount = 2;
        pi.pStages = stages;
        pi.pVertexInputState = nullptr;   // ← Mesh Shader 管线：不设置！
        pi.pInputAssemblyState = nullptr; // ← 同上
        pi.pViewportState = &vs;
        pi.pRasterizationState = &rs;
        pi.pMultisampleState = &ms;
        pi.pColorBlendState = &cb;
        pi.pDynamicState = &dynS;
        pi.layout = pipelineLayout_;
        pi.renderPass = renderPass_;

        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline_));
        vkDestroyShaderModule(device_, fragModule, nullptr);
        vkDestroyShaderModule(device_, meshModule, nullptr);
        std::cout << "✅ Mesh Shader 管线创建成功！\n";
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
        static auto start = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start).count();

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

        VkClearValue clearColor{};
        clearColor.color.float32[0] = 0.02f;
        clearColor.color.float32[2] = 0.05f;
        clearColor.color.float32[3] = 1.0f;

        VkRenderPassBeginInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp.renderPass = renderPass_;
        rp.framebuffer = framebuffers_[imageIndex];
        rp.renderArea = {{0, 0}, swapchainExtent_};
        rp.clearValueCount = 1;
        rp.pClearValues = &clearColor;
        vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        VkViewport vp{0, 0, (float)swapchainExtent_.width, (float)swapchainExtent_.height, 0, 1};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{{0, 0}, swapchainExtent_};
        vkCmdSetScissor(cmd, 0, 1, &sc);

        // Push Constants：传递时间
        PushConstants pc{time};
        vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_MESH_BIT_EXT, 0, sizeof(pc), &pc);

        // ── 关键：vkCmdDrawMeshTasksEXT 替代 vkCmdDraw ──────────────────────
        // 参数：groupCountX=1, Y=1, Z=1
        // 启动 1×1×1 = 1 个 mesh workgroup
        // 每个 workgroup 的 local_size_x=3（3 个线程处理 3 个顶点）
        fpCmdDrawMeshTasks_(cmd, 1, 1, 1);

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
        std::cout << "🔷 Mesh Shader 三角形旋转中（ESC 退出）...\n";
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
                glfwSetWindowShouldClose(window_, GLFW_TRUE);
            drawFrame();
        }
        vkDeviceWaitIdle(device_);
    }

    void printMeshShaderGuide() {
        std::cout << "\n════════════════════════════════════════════\n";
        std::cout << " Mesh Shader API 概念速查\n";
        std::cout << "════════════════════════════════════════════\n\n";
        std::cout << "1. 所需扩展：VK_EXT_mesh_shader\n\n";
        std::cout << "2. 新着色器类型（GLSL 扩展：GL_EXT_mesh_shader）\n";
        std::cout << "   .mesh  → 生成顶点和图元\n";
        std::cout << "   .task  → 控制 mesh workgroup 数量（可选）\n\n";
        std::cout << "3. 关键 API：\n";
        std::cout << "   vkCmdDrawMeshTasksEXT(X, Y, Z) 替代 vkCmdDraw\n";
        std::cout << "   pVertexInputState = nullptr（管线中不需要！）\n\n";
        std::cout << "4. Mesh Shader 中的关键内置变量：\n";
        std::cout << "   SetMeshOutputsEXT(vertCount, primCount)\n";
        std::cout << "   gl_MeshVerticesEXT[i].gl_Position = vec4(...)\n";
        std::cout << "   gl_PrimitiveTriangleIndicesEXT[i] = uvec3(a,b,c)\n\n";
        std::cout << "5. 适用场景：\n";
        std::cout << "   - GPU 端 LOD 选择（Task Shader 决定精细度）\n";
        std::cout << "   - Meshlet 级别视锥体剔除（Task Shader 条件不启动 Mesh）\n";
        std::cout << "   - 程序化几何（直接在 GPU 上生成地形、草等）\n";
        std::cout << "   - 粒子系统的几何生成（替代 Geometry Shader）\n";
    }

    // ─── 辅助代码 ─────────────────────────────────────────────────────────────

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
        VK_CHECK(vkCreateRenderPass(device_, &rpi, nullptr, &renderPass_));
    }
    void createFramebuffers() {
        framebuffers_.resize(swapchainImageViews_.size());
        for (size_t i = 0; i < swapchainImageViews_.size(); ++i) {
            VkImageView att[] = {swapchainImageViews_[i]};
            VkFramebufferCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            ci.renderPass = renderPass_;
            ci.attachmentCount = 1;
            ci.pAttachments = att;
            ci.width = swapchainExtent_.width;
            ci.height = swapchainExtent_.height;
            ci.layers = 1;
            VK_CHECK(vkCreateFramebuffer(device_, &ci, nullptr, &framebuffers_[i]));
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
        for (auto& fb : framebuffers_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        for (auto& iv : swapchainImageViews_)
            vkDestroyImageView(device_, iv, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        createSwapchain();
        createImageViews();
        createFramebuffers();
    }
    void cleanup() {
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroySemaphore(device_, imageAvailableSems_[i], nullptr);
            vkDestroySemaphore(device_, renderFinishedSems_[i], nullptr);
            vkDestroyFence(device_, inFlightFences_[i], nullptr);
        }
        vkDestroyCommandPool(device_, commandPool_, nullptr);
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
    std::cout << "═══════════════════════════════════════════════════════════════════\n";
    std::cout << " 第21章：Mesh Shader（网格着色器）\n";
    std::cout << "\n";
    std::cout << " 核心改变：\n";
    std::cout << "   • 无 VkVertexInputAttributeDescription（Mesh Shader 自生成顶点）\n";
    std::cout << "   • 无 vkCmdBindVertexBuffers\n";
    std::cout << "   • vkCmdDrawMeshTasksEXT(1,1,1) 替代 vkCmdDraw\n";
    std::cout << "   • 新着色器阶段：VK_SHADER_STAGE_MESH_BIT_EXT\n";
    std::cout << "   • 扩展：VK_EXT_mesh_shader\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n\n";
    Ch21App app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
    return 0;
}
