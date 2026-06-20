/**
 * @file ch17_instancing.cpp
 * @brief 第17章：实例化渲染（Instanced Rendering）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【什么是实例化渲染？】
 *
 *  用一次 draw call 渲染同一网格的 N 个副本，每个副本有不同的
 *  位置、旋转、颜色等属性。
 *
 *  【朴素方法 vs 实例化】
 *
 *  朴素方法（N 次 draw call）：
 *    for (int i = 0; i < N; ++i) {
 *        updateUniform(transforms[i]);   // CPU 上传数据
 *        vkCmdDraw(...);                 // GPU 执行
 *    }
 *    → CPU-GPU 频繁切换，N=10000 时极慢
 *
 *  实例化（1 次 draw call）：
 *    vkCmdDrawIndexed(indexCount, N, 0, 0, 0);
 *    → GPU 内部循环 N 次，gl_InstanceIndex 标识当前实例
 *    → CPU 只发一次命令，极快！
 *
 * 【Vulkan 实现方式：实例顶点缓冲】
 *
 *  顶点缓冲0（VBO）：网格几何数据（VERTEX_RATE）
 *    - pos, normal, texcoord
 *    - 每个顶点读一次
 *
 *  顶点缓冲1（IBO = Instance Buffer）：实例数据（INSTANCE_RATE）
 *    - instancePos, instanceColor, instanceScale
 *    - 每个实例读一次（不是每个顶点）
 *
 *  两个缓冲绑定到不同的 binding 点，GLSL 通过 location 访问。
 *
 * 【本章示例：1000 个随机大小、颜色、位置的立方体】
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
constexpr uint32_t N_INSTANCES = 1000; // 实例数量

// ─── 顶点数据结构（网格几何） ─────────────────────────────────────────────────

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
};

// ─── 实例数据结构（每个实例独有） ────────────────────────────────────────────

/**
 * @brief 实例数据（存放在独立的实例缓冲）
 *
 * VkVertexInputBindingDescription.inputRate = INSTANCE_RATE
 * → 每个实例读取一次（而不是每个顶点）
 */
struct InstanceData {
    glm::vec3 instancePos;   ///< 世界坐标偏移
    glm::vec3 instanceColor; ///< 颜色乘数
    float instanceScale;     ///< 缩放
    float instanceRotY;      ///< 绕 Y 轴旋转（弧度）
};

// ─── 三角形网格 ───────────────────────────────────────────────────────────────
static const std::vector<Vertex> VERTICES = {
    {{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}},
    {{0.0f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}},
};

// ─── 统一缓冲（共享相机矩阵） ─────────────────────────────────────────────────
struct UBO {
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 projection;
    float time;
};

class Ch17App {
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
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;

    // ─── 深度缓冲 ─────────────────────────────────────────────────────────
    VkImage depthImage_ = VK_NULL_HANDLE;
    VkDeviceMemory depthImageMemory_ = VK_NULL_HANDLE;
    VkImageView depthImageView_ = VK_NULL_HANDLE;
    VkFormat depthFormat_ = VK_FORMAT_UNDEFINED;

    // ─── 缓冲 ─────────────────────────────────────────────────────────────
    VkBuffer vertexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory_ = VK_NULL_HANDLE;
    VkBuffer instanceBuffer_ = VK_NULL_HANDLE; // ← 实例数据缓冲（新增）
    VkDeviceMemory instanceMemory_ = VK_NULL_HANDLE;

    std::vector<VkBuffer> uniformBuffers_;
    std::vector<VkDeviceMemory> uniformMemories_;
    std::vector<void*> uniformMapped_;

    std::vector<VkDescriptorSet> descriptorSets_;
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
        window_ =
            glfwCreateWindow(WIDTH, HEIGHT, "Ch17 - 实例化渲染（1000 个三角形，1 次 DrawCall）", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
            reinterpret_cast<Ch17App*>(glfwGetWindowUserPointer(w))->resized_ = true;
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
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createDepthResources();
        createFramebuffers();
        createCommandPool();
        createVertexBuffer();
        createInstanceBuffer(); // ← 新增：创建实例数据缓冲
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
        std::cout << "\n✅ 实例化渲染初始化完成！\n";
        std::cout << "🔢 实例数量：" << N_INSTANCES << "\n";
        std::cout << "📦 顶点缓冲 binding=0（" << VERTICES.size() << " 个顶点）\n";
        std::cout << "📦 实例缓冲 binding=1（" << N_INSTANCES << " 个实例，INSTANCE_RATE）\n";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 核心：创建实例数据缓冲（随机初始化 N_INSTANCES 个实例）
    // ═══════════════════════════════════════════════════════════════════════

    void createInstanceBuffer() {
        std::mt19937 rng(123);
        std::uniform_real_distribution<float> distXZ(-3.0f, 3.0f);
        std::uniform_real_distribution<float> distY(-2.0f, 2.0f);
        std::uniform_real_distribution<float> distScale(0.05f, 0.25f);
        std::uniform_real_distribution<float> distColor(0.3f, 1.0f);
        std::uniform_real_distribution<float> distRot(0, 6.28f);

        std::vector<InstanceData> instances(N_INSTANCES);
        for (auto& inst : instances) {
            inst.instancePos = {distXZ(rng), distY(rng), distXZ(rng)};
            inst.instanceColor = {distColor(rng), distColor(rng), distColor(rng)};
            inst.instanceScale = distScale(rng);
            inst.instanceRotY = distRot(rng);
        }

        VkDeviceSize sz = sizeof(InstanceData) * N_INSTANCES;

        // 使用 HOST_VISIBLE 以便可以动态更新（动画）
        // 生产项目中可以用 Staging + DEVICE_LOCAL 获得更好性能
        createBuffer(sz,
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     instanceBuffer_,
                     instanceMemory_);

        void* data = nullptr;
        vkMapMemory(device_, instanceMemory_, 0, sz, 0, &data);
        std::memcpy(data, instances.data(), static_cast<size_t>(sz));
        vkUnmapMemory(device_, instanceMemory_);

        std::cout << "✅ 实例缓冲已创建（" << sz / 1024 << " KB，" << N_INSTANCES << " 个实例）\n";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 核心：管线创建（两个顶点缓冲绑定：VERTEX_RATE + INSTANCE_RATE）
    // ═══════════════════════════════════════════════════════════════════════

    void createGraphicsPipeline() {
        VkShaderModule vert = createShaderModuleFromFile(device_, "instanced.vert.spv");
        VkShaderModule frag = createShaderModuleFromFile(device_, "triangle.frag.spv");

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

        // ── 顶点输入：两个 binding ────────────────────────────────────────────
        // binding 0：逐顶点数据（普通网格顶点）
        // binding 1：逐实例数据（每个实例读一次）
        std::array<VkVertexInputBindingDescription, 2> bindings{};
        bindings[0].binding = 0;
        bindings[0].stride = sizeof(Vertex);
        bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX; // 每顶点一次

        bindings[1].binding = 1;
        bindings[1].stride = sizeof(InstanceData);
        bindings[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE; // ← 每实例一次！

        // 属性描述（binding=0: pos, color；binding=1: instancePos, color, scale, rotY）
        std::array<VkVertexInputAttributeDescription, 6> attrs{};
        attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)};
        attrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)};
        attrs[2] = {2, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(InstanceData, instancePos)};
        attrs[3] = {3, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(InstanceData, instanceColor)};
        attrs[4] = {4, 1, VK_FORMAT_R32_SFLOAT, offsetof(InstanceData, instanceScale)};
        attrs[5] = {5, 1, VK_FORMAT_R32_SFLOAT, offsetof(InstanceData, instanceRotY)};

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = static_cast<uint32_t>(bindings.size());
        vi.pVertexBindingDescriptions = bindings.data();
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
        pli.pSetLayouts = &descriptorSetLayout_;
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
        pi.pDynamicState = &dynS;
        pi.layout = pipelineLayout_;
        pi.renderPass = renderPass_;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline_));
        vkDestroyShaderModule(device_, frag, nullptr);
        vkDestroyShaderModule(device_, vert, nullptr);
        std::cout << "✅ 管线创建成功（双 binding 顶点输入）\n";
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
        std::array<VkClearValue, 2> clears{};
        clears[0].color.float32[0] = 0.02f;
        clears[0].color.float32[1] = 0.02f;
        clears[0].color.float32[2] = 0.05f;
        clears[0].color.float32[3] = 1.0f;
        clears[1].depthStencil = {1.0f, 0};
        VkRenderPassBeginInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp.renderPass = renderPass_;
        rp.framebuffer = framebuffers_[imageIndex];
        rp.renderArea = {{0, 0}, swapchainExtent_};
        rp.clearValueCount = static_cast<uint32_t>(clears.size());
        rp.pClearValues = clears.data();
        vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        VkViewport vp{0, 0, (float)swapchainExtent_.width, (float)swapchainExtent_.height, 0, 1};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{{0, 0}, swapchainExtent_};
        vkCmdSetScissor(cmd, 0, 1, &sc);

        // ── 绑定两个顶点缓冲！ ────────────────────────────────────────────────
        // binding 0 = 网格顶点，binding 1 = 实例数据
        std::array<VkBuffer, 2> vbs = {vertexBuffer_, instanceBuffer_};
        std::array<VkDeviceSize, 2> offs = {0, 0};
        vkCmdBindVertexBuffers(cmd, 0, static_cast<uint32_t>(vbs.size()), vbs.data(), offs.data());

        vkCmdBindDescriptorSets(
            cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_, 0, 1, &descriptorSets_[currentFrame_], 0, nullptr);

        // ── 一次 draw call 渲染所有实例！ ─────────────────────────────────────
        // vkCmdDraw(cmd, vertexCount, instanceCount, firstVertex, firstInstance)
        // instanceCount = N_INSTANCES → GPU 内部循环 N_INSTANCES 次
        vkCmdDraw(cmd,
                  static_cast<uint32_t>(VERTICES.size()), // 3 个顶点/三角形
                  N_INSTANCES,                            // N_INSTANCES 个实例
                  0,
                  0);

        vkCmdEndRenderPass(cmd);
        VK_CHECK(vkEndCommandBuffer(cmd));
    }

    void updateUniformBuffer(uint32_t frame) {
        static auto start = std::chrono::high_resolution_clock::now();
        float t = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start).count();
        UBO ubo{};
        ubo.view =
            glm::lookAt(glm::vec3(4 * std::sin(t * 0.3f), 2, 4 * std::cos(t * 0.3f)), glm::vec3(0), glm::vec3(0, 1, 0));
        ubo.projection =
            glm::perspective(glm::radians(60.0f), (float)swapchainExtent_.width / swapchainExtent_.height, 0.1f, 20.0f);
        ubo.projection[1][1] *= -1;
        ubo.time = t;
        std::memcpy(uniformMapped_[frame], &ubo, sizeof(ubo));
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
        updateUniformBuffer(currentFrame_);
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
        std::cout << "🎨 " << N_INSTANCES << " 个三角形，1 次 DrawCall！（ESC 退出）\n";
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
        throw std::runtime_error("找不到合适内存");
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
        VkDeviceSize sz = sizeof(VERTICES[0]) * VERTICES.size();
        createBuffer(sz,
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     vertexBuffer_,
                     vertexMemory_);
        void* d = nullptr;
        vkMapMemory(device_, vertexMemory_, 0, sz, 0, &d);
        std::memcpy(d, VERTICES.data(), (size_t)sz);
        vkUnmapMemory(device_, vertexMemory_);
    }
    void createUniformBuffers() {
        VkDeviceSize sz = sizeof(UBO);
        uniformBuffers_.resize(MAX_FRAMES);
        uniformMemories_.resize(MAX_FRAMES);
        uniformMapped_.resize(MAX_FRAMES);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            createBuffer(sz,
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         uniformBuffers_[i],
                         uniformMemories_[i]);
            vkMapMemory(device_, uniformMemories_[i], 0, sz, 0, &uniformMapped_[i]);
        }
    }

    VkFormat findSupportedFormat(const std::vector<VkFormat>& cands, VkImageTiling tiling, VkFormatFeatureFlags feats) {
        for (VkFormat f : cands) {
            VkFormatProperties p;
            vkGetPhysicalDeviceFormatProperties(physicalDevice_, f, &p);
            if (tiling == VK_IMAGE_TILING_OPTIMAL && (p.optimalTilingFeatures & feats) == feats)
                return f;
        }
        throw std::runtime_error("找不到支持格式");
    }
    VkFormat findDepthFormat() {
        return findSupportedFormat({VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
                                   VK_IMAGE_TILING_OPTIMAL,
                                   VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    }
    void createDepthResources() {
        VkImageCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.extent = {swapchainExtent_.width, swapchainExtent_.height, 1};
        ci.mipLevels = 1;
        ci.arrayLayers = 1;
        ci.format = depthFormat_;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ci.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        VK_CHECK(vkCreateImage(device_, &ci, nullptr, &depthImage_));
        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(device_, depthImage_, &mr);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK(vkAllocateMemory(device_, &ai, nullptr, &depthImageMemory_));
        VK_CHECK(vkBindImageMemory(device_, depthImage_, depthImageMemory_, 0));
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = depthImage_;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = depthFormat_;
        vci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(device_, &vci, nullptr, &depthImageView_));
    }
    void createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0;
        b.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 1;
        ci.pBindings = &b;
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &ci, nullptr, &descriptorSetLayout_));
    }
    void createDescriptorPool() {
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, static_cast<uint32_t>(MAX_FRAMES)};
        VkDescriptorPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        ci.poolSizeCount = 1;
        ci.pPoolSizes = &ps;
        ci.maxSets = static_cast<uint32_t>(MAX_FRAMES);
        VK_CHECK(vkCreateDescriptorPool(device_, &ci, nullptr, &descriptorPool_));
    }
    void createDescriptorSets() {
        std::vector<VkDescriptorSetLayout> lays(MAX_FRAMES, descriptorSetLayout_);
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = descriptorPool_;
        ai.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES);
        ai.pSetLayouts = lays.data();
        descriptorSets_.resize(MAX_FRAMES);
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, descriptorSets_.data()));
        for (int i = 0; i < MAX_FRAMES; ++i) {
            VkDescriptorBufferInfo bi{};
            bi.buffer = uniformBuffers_[i];
            bi.offset = 0;
            bi.range = sizeof(UBO);
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = descriptorSets_[i];
            w.dstBinding = 0;
            w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            w.descriptorCount = 1;
            w.pBufferInfo = &bi;
            vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
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
        VkAttachmentDescription da{};
        da.format = depthFormat_;
        da.samples = VK_SAMPLE_COUNT_1_BIT;
        da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        da.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        da.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        da.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        da.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        da.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        VkAttachmentReference cr{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference dr{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sp{};
        sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sp.colorAttachmentCount = 1;
        sp.pColorAttachments = &cr;
        sp.pDepthStencilAttachment = &dr;
        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.srcAccessMask = 0;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        std::array<VkAttachmentDescription, 2> atts = {ca, da};
        VkRenderPassCreateInfo rpi{};
        rpi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpi.attachmentCount = static_cast<uint32_t>(atts.size());
        rpi.pAttachments = atts.data();
        rpi.subpassCount = 1;
        rpi.pSubpasses = &sp;
        rpi.dependencyCount = 1;
        rpi.pDependencies = &dep;
        VK_CHECK(vkCreateRenderPass(device_, &rpi, nullptr, &renderPass_));
    }
    void createFramebuffers() {
        framebuffers_.resize(swapchainImageViews_.size());
        for (size_t i = 0; i < swapchainImageViews_.size(); ++i) {
            std::array<VkImageView, 2> att = {swapchainImageViews_[i], depthImageView_};
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
        vkDestroyImageView(device_, depthImageView_, nullptr);
        vkDestroyImage(device_, depthImage_, nullptr);
        vkFreeMemory(device_, depthImageMemory_, nullptr);
        for (auto& iv : swapchainImageViews_)
            vkDestroyImageView(device_, iv, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        createSwapchain();
        createImageViews();
        createDepthResources();
        createFramebuffers();
    }
    void cleanup() {
        vkDestroyBuffer(device_, instanceBuffer_, nullptr);
        vkFreeMemory(device_, instanceMemory_, nullptr);
        vkDestroyBuffer(device_, vertexBuffer_, nullptr);
        vkFreeMemory(device_, vertexMemory_, nullptr);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroyBuffer(device_, uniformBuffers_[i], nullptr);
            vkFreeMemory(device_, uniformMemories_[i], nullptr);
        }
        vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroySemaphore(device_, imageAvailableSems_[i], nullptr);
            vkDestroySemaphore(device_, renderFinishedSems_[i], nullptr);
            vkDestroyFence(device_, inFlightFences_[i], nullptr);
        }
        vkDestroyCommandPool(device_, commandPool_, nullptr);
        for (auto& fb : framebuffers_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        vkDestroyImageView(device_, depthImageView_, nullptr);
        vkDestroyImage(device_, depthImage_, nullptr);
        vkFreeMemory(device_, depthImageMemory_, nullptr);
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
    std::cout << " 第17章：实例化渲染（Instanced Rendering）\n";
    std::cout << "\n";
    std::cout << " 关键点：\n";
    std::cout << "   • 两个 VkVertexInputBindingDescription（VERTEX + INSTANCE）\n";
    std::cout << "   • vkCmdBindVertexBuffers(binding=0,1) 同时绑定两个缓冲\n";
    std::cout << "   • vkCmdDraw(3, N_INSTANCES, ...) 一次 DrawCall 渲染所有实例\n";
    std::cout << "   • gl_InstanceIndex 在着色器中标识当前实例编号\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n\n";
    Ch17App app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
    return 0;
}
