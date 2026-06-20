/**
 * @file ch19_deferred.cpp
 * @brief 第19章：延迟渲染（Deferred Rendering / G-Buffer）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【延迟渲染 vs 前向渲染】
 *
 *  前向渲染（Forward Rendering）：
 *    对每个片段，遍历所有光源计算光照
 *    复杂度：O(片段数 × 光源数)
 *    100个光源 × 100万片段 = 1亿次光照计算
 *    问题：过度绘制（overdraw）浪费，被遮挡的片段也被计算
 *
 *  延迟渲染（Deferred Rendering）：
 *    Pass 1：几何通道 → 将场景属性写入 G-Buffer（MRT）
 *    Pass 2：光照通道 → 读 G-Buffer，每像素只计算一次光照
 *    复杂度：O(分辨率 × 光源数)，不受场景复杂度影响
 *    适合：大量光源（数百个），复杂场景
 *
 * 【G-Buffer 布局】
 *
 *  ┌─────────────────────────────────────────────────────────┐
 *  │  G-Buffer（4个颜色附件 + 1个深度附件）                   │
 *  │                                                         │
 *  │  Attachment 0: RGBA16F  → 世界坐标 (Position.xyz)      │
 *  │  Attachment 1: RGBA16F  → 法线向量 (Normal.xyz)        │
 *  │  Attachment 2: RGBA8    → 漫反射颜色 (Albedo.rgb)      │
 *  │  Attachment 3 (depth): D32F → 深度缓冲                 │
 *  └─────────────────────────────────────────────────────────┘
 *
 * 【Subpass（子流程）的使用】
 *
 *  延迟渲染的两个 Pass 可以合并为一个 RenderPass 的两个 Subpass：
 *
 *  Subpass 0（几何）：写入 G-Buffer（Attachment 0,1,2）
 *  Subpass 1（光照）：读取 Subpass 0 的输出（Input Attachments）
 *
 *  好处（移动端 Tile-Based 架构）：
 *    G-Buffer 数据保留在 GPU Tile 缓存中，避免写出显存再读入
 *    → 带宽减少 80%+，功耗大幅降低
 *
 * 【MRT（Multiple Render Targets）】
 *
 *  一个 RenderPass 写多个颜色附件：
 *    GLSL: layout(location=0) out vec4 outPosition;
 *          layout(location=1) out vec4 outNormal;
 *          layout(location=2) out vec4 outAlbedo;
 *  每个 location 对应 Framebuffer 中的不同颜色附件。
 *
 * 【Input Attachments】
 *
 *  Subpass 1 读取 Subpass 0 的输出（不经过显存，仅在同一 RenderPass）：
 *    GLSL: layout(input_attachment_index=0, binding=0) uniform subpassInput inPosition;
 *          subpassLoad(inPosition)  ← 当前像素坐标，无法随机访问
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
#include <random>
#include <set>
#include <stdexcept>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int MAX_FRAMES = 2;
constexpr int N_LIGHTS = 4;

struct GVertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec3 color;
    glm::vec2 uv;
};

struct GeometryUBO {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 projection;
};

struct Light {
    alignas(16) glm::vec4 position; // xyz=位置, w=半径
    alignas(16) glm::vec4 color;    // rgb=颜色, a=强度
};

struct LightingUBO {
    Light lights[N_LIGHTS];
    alignas(16) glm::vec4 cameraPos;
};

// 简单场景：地面 + 一个立方体
static const std::vector<GVertex> SCENE_VERTICES = {
    // 地面
    {{-3, -0.5f, -3}, {0, 1, 0}, {0.5f, 0.5f, 0.5f}, {0, 0}},
    {{3, -0.5f, -3}, {0, 1, 0}, {0.5f, 0.5f, 0.5f}, {1, 0}},
    {{3, -0.5f, 3}, {0, 1, 0}, {0.5f, 0.5f, 0.5f}, {1, 1}},
    {{-3, -0.5f, -3}, {0, 1, 0}, {0.5f, 0.5f, 0.5f}, {0, 0}},
    {{3, -0.5f, 3}, {0, 1, 0}, {0.5f, 0.5f, 0.5f}, {1, 1}},
    {{-3, -0.5f, 3}, {0, 1, 0}, {0.5f, 0.5f, 0.5f}, {0, 1}},
    // 立方体前面
    // 立方体前面（底部顶点抬高 0.02，避免与地面 y=-0.5 Z-Fighting）
    {{-0.5f, -0.48f, -0.5f}, {0, 0, -1}, {0.8f, 0.3f, 0.3f}, {0, 0}},
    {{0.5f, -0.48f, -0.5f}, {0, 0, -1}, {0.8f, 0.3f, 0.3f}, {1, 0}},
    {{0.5f, 1.0f, -0.5f}, {0, 0, -1}, {0.8f, 0.3f, 0.3f}, {1, 1}},
    {{-0.5f, -0.48f, -0.5f}, {0, 0, -1}, {0.8f, 0.3f, 0.3f}, {0, 0}},
    {{0.5f, 1.0f, -0.5f}, {0, 0, -1}, {0.8f, 0.3f, 0.3f}, {1, 1}},
    {{-0.5f, 1.0f, -0.5f}, {0, 0, -1}, {0.8f, 0.3f, 0.3f}, {0, 1}},
};

class Ch19App {
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
    VkCommandPool commandPool_ = VK_NULL_HANDLE;

    // ── G-Buffer 附件 ───────────────────────────────────────────────────────
    VkImage posImage_ = VK_NULL_HANDLE; ///< G-Buffer 位置
    VkDeviceMemory posMemory_ = VK_NULL_HANDLE;
    VkImageView posView_ = VK_NULL_HANDLE;
    VkImage normalImage_ = VK_NULL_HANDLE; ///< G-Buffer 法线
    VkDeviceMemory normalMemory_ = VK_NULL_HANDLE;
    VkImageView normalView_ = VK_NULL_HANDLE;
    VkImage albedoImage_ = VK_NULL_HANDLE; ///< G-Buffer 颜色
    VkDeviceMemory albedoMemory_ = VK_NULL_HANDLE;
    VkImageView albedoView_ = VK_NULL_HANDLE;
    VkImage depthImage_ = VK_NULL_HANDLE;
    VkDeviceMemory depthMemory_ = VK_NULL_HANDLE;
    VkImageView depthView_ = VK_NULL_HANDLE;
    VkFormat depthFormat_ = VK_FORMAT_UNDEFINED;

    // ── RenderPass（含两个 Subpass） ─────────────────────────────────────────
    VkRenderPass renderPass_ = VK_NULL_HANDLE;

    // ── Subpass 0：几何管线 ────────────────────────────────────────────────
    VkDescriptorSetLayout geomSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout geomPipeLayout_ = VK_NULL_HANDLE;
    VkPipeline geomPipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> geomDescSets_;
    std::vector<VkBuffer> geomUBOs_;
    std::vector<VkDeviceMemory> geomUBOMemories_;
    std::vector<void*> geomUBOMapped_;

    // ── Subpass 1：光照管线 ────────────────────────────────────────────────
    VkDescriptorSetLayout lightSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout lightPipeLayout_ = VK_NULL_HANDLE;
    VkPipeline lightPipeline_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> lightDescSets_;
    std::vector<VkBuffer> lightUBOs_;
    std::vector<VkDeviceMemory> lightUBOMemories_;
    std::vector<void*> lightUBOMapped_;

    VkBuffer vertexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory_ = VK_NULL_HANDLE;

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

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "Ch19 - 延迟渲染（G-Buffer + 4 点光源）", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
            reinterpret_cast<Ch19App*>(glfwGetWindowUserPointer(w))->resized_ = true;
        });
    }

    void initVulkan() {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapchain();
        createImageViews();
        depthFormat_ = findDepthFormat();
        createGBufferImages();      // ← 创建 G-Buffer 附件图像
        createDeferredRenderPass(); // ← 含两个 Subpass 的 RenderPass
        createDescriptorSetLayouts();
        createGeometryPipeline(); // ← Subpass 0：写入 G-Buffer
        createLightingPipeline(); // ← Subpass 1：读取 G-Buffer，光照计算
        createFramebuffers();
        createCommandPool();
        createVertexBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
        std::cout << "\n✅ 延迟渲染初始化完成！\n";
        std::cout << "📐 G-Buffer：3 个颜色附件（Position/Normal/Albedo）\n";
        std::cout << "💡 光源数量：" << N_LIGHTS << " 个点光源\n";
        std::cout << "🔄 Subpass：几何通道 → 光照通道（Input Attachment 零带宽）\n";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 创建 G-Buffer 图像（3个颜色缓冲 + 1个深度缓冲）
    // ═══════════════════════════════════════════════════════════════════════

    void createGBufferImages() {
        // 创建辅助函数
        auto createGImage =
            [this](VkFormat fmt, VkImageUsageFlags usage, VkImage& img, VkDeviceMemory& mem, VkImageView& view) {
                VkImageCreateInfo ci{};
                ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
                ci.imageType = VK_IMAGE_TYPE_2D;
                ci.extent = {swapchainExtent_.width, swapchainExtent_.height, 1};
                ci.mipLevels = 1;
                ci.arrayLayers = 1;
                ci.format = fmt;
                ci.tiling = VK_IMAGE_TILING_OPTIMAL;
                ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                ci.usage = usage;
                ci.samples = VK_SAMPLE_COUNT_1_BIT;
                VK_CHECK(vkCreateImage(device_, &ci, nullptr, &img));
                VkMemoryRequirements mr;
                vkGetImageMemoryRequirements(device_, img, &mr);
                VkMemoryAllocateInfo ai{};
                ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                ai.allocationSize = mr.size;
                ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                VK_CHECK(vkAllocateMemory(device_, &ai, nullptr, &mem));
                VK_CHECK(vkBindImageMemory(device_, img, mem, 0));
                VkImageViewCreateInfo vci{};
                vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                vci.image = img;
                vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
                vci.format = fmt;
                VkImageAspectFlags aspect =
                    (fmt == depthFormat_) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
                vci.subresourceRange = {aspect, 0, 1, 0, 1};
                VK_CHECK(vkCreateImageView(device_, &vci, nullptr, &view));
            };

        // G-Buffer 使用 RGBA16F 格式存储浮点数（位置、法线需要精度）
        VkImageUsageFlags colorUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | // 作为颜色附件写入
                                       VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;  // 作为 Input Attachment 读取

        createGImage(VK_FORMAT_R16G16B16A16_SFLOAT, colorUsage, posImage_, posMemory_, posView_);
        createGImage(VK_FORMAT_R16G16B16A16_SFLOAT, colorUsage, normalImage_, normalMemory_, normalView_);
        createGImage(VK_FORMAT_R8G8B8A8_UNORM, colorUsage, albedoImage_, albedoMemory_, albedoView_);
        createGImage(depthFormat_, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, depthImage_, depthMemory_, depthView_);

        std::cout << "✅ G-Buffer 附件已创建（Position/Normal/Albedo/Depth）\n";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 创建 Deferred RenderPass（含 Subpass 依赖和 Input Attachments）
    // ═══════════════════════════════════════════════════════════════════════

    void createDeferredRenderPass() {
        // 附件描述（共 5 个）
        // [0] G-Buffer 位置
        // [1] G-Buffer 法线
        // [2] G-Buffer 颜色
        // [3] 深度
        // [4] 最终颜色输出（交换链）

        VkAttachmentDescription posAtt{};
        posAtt.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        posAtt.samples = VK_SAMPLE_COUNT_1_BIT;
        posAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        posAtt.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; // 不需要保存到显存
        posAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        posAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        posAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        posAtt.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription normalAtt = posAtt;
        normalAtt.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkAttachmentDescription albedoAtt = posAtt;
        albedoAtt.format = VK_FORMAT_R8G8B8A8_UNORM;
        VkAttachmentDescription depthAtt{};
        depthAtt.format = depthFormat_;
        depthAtt.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAtt.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        VkAttachmentDescription colorAtt{};
        colorAtt.format = swapchainImageFormat_;
        colorAtt.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAtt.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        // ── Subpass 0：几何通道 ────────────────────────────────────────────
        // 输出到 G-Buffer 的 3 个颜色附件
        std::array<VkAttachmentReference, 3> gColorRefs{};
        gColorRefs[0] = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        gColorRefs[1] = {1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        gColorRefs[2] = {2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference gDepthRef{3, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

        VkSubpassDescription geomSubpass{};
        geomSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        geomSubpass.colorAttachmentCount = static_cast<uint32_t>(gColorRefs.size());
        geomSubpass.pColorAttachments = gColorRefs.data();
        geomSubpass.pDepthStencilAttachment = &gDepthRef;

        // ── Subpass 1：光照通道 ────────────────────────────────────────────
        // Input Attachments：读取 Subpass 0 的 G-Buffer 输出
        std::array<VkAttachmentReference, 3> inputRefs{};
        inputRefs[0] = {0, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}; // 位置
        inputRefs[1] = {1, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}; // 法线
        inputRefs[2] = {2, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}; // 颜色
        // 输出最终颜色到交换链
        VkAttachmentReference finalColorRef{4, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

        VkSubpassDescription lightSubpass{};
        lightSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        lightSubpass.inputAttachmentCount = static_cast<uint32_t>(inputRefs.size());
        lightSubpass.pInputAttachments = inputRefs.data(); // ← Input Attachments!
        lightSubpass.colorAttachmentCount = 1;
        lightSubpass.pColorAttachments = &finalColorRef;

        // ── Subpass 依赖 ──────────────────────────────────────────────────
        std::array<VkSubpassDependency, 3> deps{};
        // 外部 → Subpass 0
        deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        deps[0].dstSubpass = 0;
        deps[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        deps[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        deps[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        // Subpass 0 → Subpass 1（关键：G-Buffer 写完才能读）
        deps[1].srcSubpass = 0;
        deps[1].dstSubpass = 1;
        deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        deps[1].dstAccessMask = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT; // ← Input Attachment 读
        deps[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        // Subpass 1 → 外部
        deps[2].srcSubpass = 1;
        deps[2].dstSubpass = VK_SUBPASS_EXTERNAL;
        deps[2].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        deps[2].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        deps[2].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        deps[2].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        deps[2].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        std::array<VkAttachmentDescription, 5> attachments = {posAtt, normalAtt, albedoAtt, depthAtt, colorAtt};
        std::array<VkSubpassDescription, 2> subpasses = {geomSubpass, lightSubpass};

        VkRenderPassCreateInfo rpi{};
        rpi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpi.attachmentCount = static_cast<uint32_t>(attachments.size());
        rpi.pAttachments = attachments.data();
        rpi.subpassCount = static_cast<uint32_t>(subpasses.size());
        rpi.pSubpasses = subpasses.data();
        rpi.dependencyCount = static_cast<uint32_t>(deps.size());
        rpi.pDependencies = deps.data();
        VK_CHECK(vkCreateRenderPass(device_, &rpi, nullptr, &renderPass_));

        std::cout << "✅ Deferred RenderPass 创建（2 个 Subpass，3 个 G-Buffer 附件）\n";
    }

    void createDescriptorSetLayouts() {
        // 几何通道布局：binding=0 UBO
        {
            VkDescriptorSetLayoutBinding b{
                0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            ci.bindingCount = 1;
            ci.pBindings = &b;
            VK_CHECK(vkCreateDescriptorSetLayout(device_, &ci, nullptr, &geomSetLayout_));
        }
        // 光照通道布局：binding=0,1,2 Input Attachments + binding=3 LightUBO
        {
            std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
            for (int i = 0; i < 3; ++i)
                bindings[i] = {static_cast<uint32_t>(i),
                               VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
                               1,
                               VK_SHADER_STAGE_FRAGMENT_BIT,
                               nullptr};
            bindings[3] = {3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            ci.bindingCount = static_cast<uint32_t>(bindings.size());
            ci.pBindings = bindings.data();
            VK_CHECK(vkCreateDescriptorSetLayout(device_, &ci, nullptr, &lightSetLayout_));
        }
    }

    void createGeometryPipeline() {
        VkShaderModule vert = createShaderModuleFromFile(device_, "gbuffer.vert.spv");
        VkShaderModule frag = createShaderModuleFromFile(device_, "gbuffer.frag.spv");
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                     nullptr,
                     0,
                     VK_SHADER_STAGE_VERTEX_BIT,
                     vert,
                     "main",
                     nullptr};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                     nullptr,
                     0,
                     VK_SHADER_STAGE_FRAGMENT_BIT,
                     frag,
                     "main",
                     nullptr};

        VkVertexInputBindingDescription bind{0, sizeof(GVertex), VK_VERTEX_INPUT_RATE_VERTEX};
        std::array<VkVertexInputAttributeDescription, 4> attrs{};
        attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(GVertex, pos)};
        attrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(GVertex, normal)};
        attrs[2] = {2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(GVertex, color)};
        attrs[3] = {3, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(GVertex, uv)};
        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &bind;
        vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
        vi.pVertexAttributeDescriptions = attrs.data();

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
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS;

        // MRT：3 个颜色附件各自的混合配置
        std::array<VkPipelineColorBlendAttachmentState, 3> cbas{};
        for (auto& c : cbas)
            c.colorWriteMask = 0xf;
        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = static_cast<uint32_t>(cbas.size());
        cb.pAttachments = cbas.data();

        std::vector<VkDynamicState> dyn = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynS{};
        dynS.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynS.dynamicStateCount = static_cast<uint32_t>(dyn.size());
        dynS.pDynamicStates = dyn.data();
        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &geomSetLayout_;
        VK_CHECK(vkCreatePipelineLayout(device_, &pli, nullptr, &geomPipeLayout_));
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
        pi.pDynamicState = &dynS;
        pi.layout = geomPipeLayout_;
        pi.renderPass = renderPass_;
        pi.subpass = 0;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &geomPipeline_));
        vkDestroyShaderModule(device_, frag, nullptr);
        vkDestroyShaderModule(device_, vert, nullptr);
        std::cout << "✅ 几何管线创建（MRT 输出 3 个 G-Buffer）\n";
    }

    void createLightingPipeline() {
        VkShaderModule vert = createShaderModuleFromFile(device_, "deferred_lighting.vert.spv");
        VkShaderModule frag = createShaderModuleFromFile(device_, "deferred_lighting.frag.spv");
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                     nullptr,
                     0,
                     VK_SHADER_STAGE_VERTEX_BIT,
                     vert,
                     "main",
                     nullptr};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                     nullptr,
                     0,
                     VK_SHADER_STAGE_FRAGMENT_BIT,
                     frag,
                     "main",
                     nullptr};
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
        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &lightSetLayout_;
        VK_CHECK(vkCreatePipelineLayout(device_, &pli, nullptr, &lightPipeLayout_));
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
        pi.layout = lightPipeLayout_;
        pi.renderPass = renderPass_;
        pi.subpass = 1; // Subpass 1!
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &lightPipeline_));
        vkDestroyShaderModule(device_, frag, nullptr);
        vkDestroyShaderModule(device_, vert, nullptr);
        std::cout << "✅ 光照管线创建（读取 Input Attachments，4 点光源 Phong）\n";
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

        // 5个清除值：position, normal, albedo, depth, finalColor
        std::array<VkClearValue, 5> clears{};
        clears[0].color.float32[0] = 0.0f;  // G-Buffer position (黑色)
        clears[1].color.float32[0] = 0.0f;  // G-Buffer normal
        clears[2].color.float32[0] = 0.0f;  // G-Buffer albedo
        clears[3].depthStencil = {1.0f, 0}; // 深度
        clears[4].color.float32[0] = 0.02f;
        clears[4].color.float32[2] = 0.05f;
        clears[4].color.float32[3] = 1.0f; // 最终背景色

        VkRenderPassBeginInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp.renderPass = renderPass_;
        rp.framebuffer = framebuffers_[imageIndex];
        rp.renderArea = {{0, 0}, swapchainExtent_};
        rp.clearValueCount = static_cast<uint32_t>(clears.size());
        rp.pClearValues = clears.data();

        vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

        // ── Subpass 0：几何通道 ────────────────────────────────────────────
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, geomPipeline_);
        VkViewport vp{0, 0, (float)swapchainExtent_.width, (float)swapchainExtent_.height, 0, 1};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{{0, 0}, swapchainExtent_};
        vkCmdSetScissor(cmd, 0, 1, &sc);
        VkBuffer vb[] = {vertexBuffer_};
        VkDeviceSize off[] = {0};
        vkCmdBindVertexBuffers(cmd, 0, 1, vb, off);
        vkCmdBindDescriptorSets(
            cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, geomPipeLayout_, 0, 1, &geomDescSets_[currentFrame_], 0, nullptr);
        vkCmdDraw(cmd, static_cast<uint32_t>(SCENE_VERTICES.size()), 1, 0, 0);

        // ── 切换到 Subpass 1 ──────────────────────────────────────────────
        vkCmdNextSubpass(cmd, VK_SUBPASS_CONTENTS_INLINE);

        // ── Subpass 1：光照通道 ────────────────────────────────────────────
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lightPipeline_);
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindDescriptorSets(
            cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lightPipeLayout_, 0, 1, &lightDescSets_[currentFrame_], 0, nullptr);
        // 全屏三角形（顶点在着色器中硬编码）
        vkCmdDraw(cmd, 3, 1, 0, 0);

        vkCmdEndRenderPass(cmd);
        VK_CHECK(vkEndCommandBuffer(cmd));
    }

    void updateUniformBuffers(uint32_t frame) {
        static auto start = std::chrono::high_resolution_clock::now();
        float t = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start).count();
        GeometryUBO gubo{};
        gubo.model = glm::mat4(1.0f);
        glm::vec3 camPos = glm::vec3(3 * std::cos(t * 0.3f), 2.5f, 3 * std::sin(t * 0.3f));
        gubo.view = glm::lookAt(camPos, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
        gubo.projection =
            glm::perspective(glm::radians(60.0f), (float)swapchainExtent_.width / swapchainExtent_.height, 0.1f, 20.0f);
        gubo.projection[1][1] *= -1;
        std::memcpy(geomUBOMapped_[frame], &gubo, sizeof(gubo));

        LightingUBO lubo{};
        lubo.cameraPos = glm::vec4(camPos, 1.0f);
        // 4个动态彩色点光源
        const glm::vec3 lcolors[] = {{1, 0.3f, 0.3f}, {0.3f, 1, 0.3f}, {0.3f, 0.3f, 1}, {1, 1, 0.3f}};
        const float angles[] = {0, 1.57f, 3.14f, 4.71f};
        for (int i = 0; i < N_LIGHTS; ++i) {
            float angle = angles[i] + t * 0.5f;
            lubo.lights[i].position = glm::vec4(2 * std::cos(angle), 0.8f, 2 * std::sin(angle), 3.0f);
            lubo.lights[i].color = glm::vec4(lcolors[i], 1.5f);
        }
        std::memcpy(lightUBOMapped_[frame], &lubo, sizeof(lubo));
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
        updateUniformBuffers(currentFrame_);
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
        std::cout << "🌈 延迟渲染：4 个动态点光源（ESC 退出）...\n";
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
                glfwSetWindowShouldClose(window_, GLFW_TRUE);
            drawFrame();
        }
        vkDeviceWaitIdle(device_);
    }

    // ─── 辅助函数 ─────────────────────────────────────────────────────────────

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
    VkFormat findSupportedFormat(const std::vector<VkFormat>& c, VkImageTiling t, VkFormatFeatureFlags f) {
        for (VkFormat fmt : c) {
            VkFormatProperties p;
            vkGetPhysicalDeviceFormatProperties(physicalDevice_, fmt, &p);
            if (t == VK_IMAGE_TILING_OPTIMAL && (p.optimalTilingFeatures & f) == f)
                return fmt;
        }
        throw std::runtime_error("找不到格式");
    }
    VkFormat findDepthFormat() {
        return findSupportedFormat({VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
                                   VK_IMAGE_TILING_OPTIMAL,
                                   VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    }
    void createVertexBuffer() {
        VkDeviceSize sz = sizeof(SCENE_VERTICES[0]) * SCENE_VERTICES.size();
        createBuffer(sz,
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     vertexBuffer_,
                     vertexMemory_);
        void* d = nullptr;
        vkMapMemory(device_, vertexMemory_, 0, sz, 0, &d);
        std::memcpy(d, SCENE_VERTICES.data(), (size_t)sz);
        vkUnmapMemory(device_, vertexMemory_);
    }
    void createUniformBuffers() {
        geomUBOs_.resize(MAX_FRAMES);
        geomUBOMemories_.resize(MAX_FRAMES);
        geomUBOMapped_.resize(MAX_FRAMES);
        lightUBOs_.resize(MAX_FRAMES);
        lightUBOMemories_.resize(MAX_FRAMES);
        lightUBOMapped_.resize(MAX_FRAMES);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            createBuffer(sizeof(GeometryUBO),
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         geomUBOs_[i],
                         geomUBOMemories_[i]);
            vkMapMemory(device_, geomUBOMemories_[i], 0, sizeof(GeometryUBO), 0, &geomUBOMapped_[i]);
            createBuffer(sizeof(LightingUBO),
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         lightUBOs_[i],
                         lightUBOMemories_[i]);
            vkMapMemory(device_, lightUBOMemories_[i], 0, sizeof(LightingUBO), 0, &lightUBOMapped_[i]);
        }
    }
    void createDescriptorPool() {
        std::array<VkDescriptorPoolSize, 2> ps{};
        ps[0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(MAX_FRAMES * 2)};
        ps[1] = {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, static_cast<uint32_t>(MAX_FRAMES * 3)};
        VkDescriptorPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        ci.poolSizeCount = static_cast<uint32_t>(ps.size());
        ci.pPoolSizes = ps.data();
        ci.maxSets = static_cast<uint32_t>(MAX_FRAMES * 2);
        VK_CHECK(vkCreateDescriptorPool(device_, &ci, nullptr, &descriptorPool_));
    }
    void createDescriptorSets() {
        // 几何通道描述符集
        {
            std::vector<VkDescriptorSetLayout> lays(MAX_FRAMES, geomSetLayout_);
            VkDescriptorSetAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool = descriptorPool_;
            ai.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES);
            ai.pSetLayouts = lays.data();
            geomDescSets_.resize(MAX_FRAMES);
            VK_CHECK(vkAllocateDescriptorSets(device_, &ai, geomDescSets_.data()));
            for (int i = 0; i < MAX_FRAMES; ++i) {
                VkDescriptorBufferInfo bi{};
                bi.buffer = geomUBOs_[i];
                bi.offset = 0;
                bi.range = sizeof(GeometryUBO);
                VkWriteDescriptorSet w{};
                w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w.dstSet = geomDescSets_[i];
                w.dstBinding = 0;
                w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                w.descriptorCount = 1;
                w.pBufferInfo = &bi;
                vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
            }
        }
        // 光照通道描述符集（含 Input Attachments）
        {
            std::vector<VkDescriptorSetLayout> lays(MAX_FRAMES, lightSetLayout_);
            VkDescriptorSetAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool = descriptorPool_;
            ai.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES);
            ai.pSetLayouts = lays.data();
            lightDescSets_.resize(MAX_FRAMES);
            VK_CHECK(vkAllocateDescriptorSets(device_, &ai, lightDescSets_.data()));
            for (int i = 0; i < MAX_FRAMES; ++i) {
                // Input Attachments（G-Buffer）
                VkDescriptorImageInfo posII{VK_NULL_HANDLE, posView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                VkDescriptorImageInfo nrmII{VK_NULL_HANDLE, normalView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                VkDescriptorImageInfo albII{VK_NULL_HANDLE, albedoView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                VkDescriptorBufferInfo luboBI{};
                luboBI.buffer = lightUBOs_[i];
                luboBI.offset = 0;
                luboBI.range = sizeof(LightingUBO);
                std::array<VkWriteDescriptorSet, 4> ws{};
                ws[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                         nullptr,
                         lightDescSets_[i],
                         0,
                         0,
                         1,
                         VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
                         &posII,
                         nullptr,
                         nullptr};
                ws[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                         nullptr,
                         lightDescSets_[i],
                         1,
                         0,
                         1,
                         VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
                         &nrmII,
                         nullptr,
                         nullptr};
                ws[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                         nullptr,
                         lightDescSets_[i],
                         2,
                         0,
                         1,
                         VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
                         &albII,
                         nullptr,
                         nullptr};
                ws[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                         nullptr,
                         lightDescSets_[i],
                         3,
                         0,
                         1,
                         VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                         nullptr,
                         &luboBI,
                         nullptr};
                vkUpdateDescriptorSets(device_, static_cast<uint32_t>(ws.size()), ws.data(), 0, nullptr);
            }
        }
    }
    void createFramebuffers() {
        framebuffers_.resize(swapchainImageViews_.size());
        for (size_t i = 0; i < swapchainImageViews_.size(); ++i) {
            // 5个附件顺序与 RenderPass 定义一致
            std::array<VkImageView, 5> att = {posView_, normalView_, albedoView_, depthView_, swapchainImageViews_[i]};
            VkFramebufferCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            ci.renderPass = renderPass_;
            ci.attachmentCount = static_cast<uint32_t>(att.size());
            ci.pAttachments = att.data();
            ci.width = swapchainExtent_.width;
            ci.height = swapchainExtent_.height;
            ci.layers = 1;
            VK_CHECK(vkCreateFramebuffer(device_, &ci, nullptr, &framebuffers_[i]));
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
    void pickPhysicalDevice() {
        uint32_t c = 0;
        vkEnumeratePhysicalDevices(instance_, &c, nullptr);
        std::vector<VkPhysicalDevice> devs(c);
        vkEnumeratePhysicalDevices(instance_, &c, devs.data());
        for (auto& d : devs)
            if (findQueueFamilies(d, surface_).isComplete() && checkDeviceExtensionSupport(d)) {
                physicalDevice_ = d;
                break;
            }
        if (!physicalDevice_)
            throw std::runtime_error("无合适GPU");
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(physicalDevice_, &p);
        std::cout << "✅ GPU: " << p.deviceName << "\n";
    }
    void createLogicalDevice() {
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
        for (auto& fb : framebuffers_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        // 重建 G-Buffer 图像（与交换链大小相同）
        auto destroyImage = [this](VkImage& i, VkDeviceMemory& m, VkImageView& v) {
            vkDestroyImageView(device_, v, nullptr);
            vkDestroyImage(device_, i, nullptr);
            vkFreeMemory(device_, m, nullptr);
        };
        destroyImage(posImage_, posMemory_, posView_);
        destroyImage(normalImage_, normalMemory_, normalView_);
        destroyImage(albedoImage_, albedoMemory_, albedoView_);
        destroyImage(depthImage_, depthMemory_, depthView_);
        for (auto& iv : swapchainImageViews_)
            vkDestroyImageView(device_, iv, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        createSwapchain();
        createImageViews();
        createGBufferImages();
        createFramebuffers();
        // 重新更新描述符集（Input Attachment 指向新 G-Buffer）
        createDescriptorSets();
    }
    void cleanup() {
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroyBuffer(device_, geomUBOs_[i], nullptr);
            vkFreeMemory(device_, geomUBOMemories_[i], nullptr);
            vkDestroyBuffer(device_, lightUBOs_[i], nullptr);
            vkFreeMemory(device_, lightUBOMemories_[i], nullptr);
        }
        vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, geomSetLayout_, nullptr);
        vkDestroyDescriptorSetLayout(device_, lightSetLayout_, nullptr);
        vkDestroyBuffer(device_, vertexBuffer_, nullptr);
        vkFreeMemory(device_, vertexMemory_, nullptr);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroySemaphore(device_, imageAvailableSems_[i], nullptr);
            vkDestroySemaphore(device_, renderFinishedSems_[i], nullptr);
            vkDestroyFence(device_, inFlightFences_[i], nullptr);
        }
        vkDestroyCommandPool(device_, commandPool_, nullptr);
        for (auto& fb : framebuffers_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        vkDestroyPipeline(device_, geomPipeline_, nullptr);
        vkDestroyPipelineLayout(device_, geomPipeLayout_, nullptr);
        vkDestroyPipeline(device_, lightPipeline_, nullptr);
        vkDestroyPipelineLayout(device_, lightPipeLayout_, nullptr);
        vkDestroyRenderPass(device_, renderPass_, nullptr);
        auto destroyImage = [this](VkImage& i, VkDeviceMemory& m, VkImageView& v) {
            vkDestroyImageView(device_, v, nullptr);
            vkDestroyImage(device_, i, nullptr);
            vkFreeMemory(device_, m, nullptr);
        };
        destroyImage(posImage_, posMemory_, posView_);
        destroyImage(normalImage_, normalMemory_, normalView_);
        destroyImage(albedoImage_, albedoMemory_, albedoView_);
        destroyImage(depthImage_, depthMemory_, depthView_);
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
    std::cout << " 第19章：延迟渲染（Deferred Rendering / G-Buffer）\n";
    std::cout << "\n";
    std::cout << " 架构：\n";
    std::cout << "   Subpass 0（几何）：MRT → G-Buffer(Position/Normal/Albedo)\n";
    std::cout << "   Subpass 1（光照）：读 Input Attachments → 4点光源 Phong\n";
    std::cout << "\n";
    std::cout << " 关键技术：\n";
    std::cout << "   • MRT（多渲染目标）：layout(location=N) out\n";
    std::cout << "   • Input Attachments：subpassLoad（移动端零带宽）\n";
    std::cout << "   • vkCmdNextSubpass()：切换 Subpass\n";
    std::cout << "   • 全屏三角形：顶点在 vert 着色器中硬编码\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n\n";
    Ch19App app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
    return 0;
}
