/**
 * @file ch15_compute.cpp
 * @brief 第15章：计算着色器（Compute Shader）— GPU 粒子系统
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【什么是 Compute Shader？】
 *
 *  计算着色器是完全运行在 GPU 上的通用计算程序，不经过图形管线。
 *  适用场景：
 *    - 物理模拟（粒子系统、流体、布料）
 *    - 图像处理（滤镜、后处理效果）
 *    - 光线追踪（BVH 遍历）
 *    - 机器学习推理
 *
 * 【与图形着色器的对比】
 *
 *  ┌─────────────────┬─────────────────┬───────────────────────────────┐
 *  │                 │ 图形着色器       │ 计算着色器                     │
 *  ├─────────────────┼─────────────────┼───────────────────────────────┤
 *  │ 触发方式         │ vkCmdDraw*      │ vkCmdDispatch                │
 *  │ 管线类型         │ VkGraphicsPipeline│ VkComputePipeline           │
 *  │ 输入/输出       │ 顶点/帧缓冲     │ 任意缓冲/图像（SSBO/Image）   │
 *  │ 内置变量         │ gl_VertexIndex等│ gl_GlobalInvocationID 等     │
 *  │ 线程组织         │ GPU 自动分配    │ 手动指定 workgroup 大小       │
 *  └─────────────────┴─────────────────┴───────────────────────────────┘
 *
 * 【工作组（Workgroup）】
 *
 *  计算着色器以"工作组"为单位执行。每个工作组是一个线程块：
 *
 *  vkCmdDispatch(32, 1, 1)       → 32 个工作组
 *    每个工作组 local_size = (256, 1, 1) → 256 个线程
 *    总线程数 = 32 × 256 = 8192 个粒子，每线程处理 1 个粒子
 *
 *  工作组内的线程可以共享 shared memory（局部高速缓存）。
 *  不同工作组之间无法直接通信（需要 barrier + SSBO）。
 *
 * 【SSBO（Shader Storage Buffer Object）】
 *
 *  与 UBO（Uniform Buffer）的对比：
 *    UBO：只读，较小（通常 < 64KB），访问快
 *    SSBO：可读写，几乎无大小限制，可以动态大小数组
 *
 * 【本章示例：8192 个粒子的弹跳系统】
 *
 *  CPU 初始化粒子（随机位置、速度、颜色）
 *    ↓
 *  每帧：
 *    ① 计算着色器（GPU）：并行更新 8192 个粒子的位置（8 微秒！）
 *    ② Pipeline Barrier：等待计算完成，让顶点着色器可以读取
 *    ③ 图形管线：将粒子渲染为彩色圆点
 *
 * 【同步：计算 → 图形】
 *
 *  计算着色器写 SSBO，图形管线（顶点着色器）读 SSBO 作为顶点缓冲。
 *  必须插入 pipeline barrier 确保数据一致性：
 *
 *  srcStageMask  = COMPUTE_SHADER           （计算写完后）
 *  dstStageMask  = VERTEX_INPUT             （顶点读取前）
 *  srcAccessMask = SHADER_WRITE             （写操作完成）
 *  dstAccessMask = VERTEX_ATTRIBUTE_READ    （读操作开始）
 *
 * 【Push Constants（推送常量）】
 *
 *  小量数据（< 128 字节）可通过 vkCmdPushConstants 直接推送到着色器，
 *  比 Uniform Buffer 更快（无需缓冲更新和 descriptor 绑定）。
 *  本章用于传递 deltaTime。
 * ═══════════════════════════════════════════════════════════════════════════
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan_tutorial/utils.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int MAX_FRAMES = 2;
constexpr uint32_t N_PARTICLES = 8192; // 必须是 local_size_x(256) 的倍数

#ifdef CH15_USE_SYNCHRONIZATION2
constexpr const char* WINDOW_TITLE = "Ch95 - Synchronization 2: GPU Particles";
#else
constexpr const char* WINDOW_TITLE = "Ch15 - GPU Particles (Compute Shader)";
#endif

// ─── 粒子数据结构 ─────────────────────────────────────────────────────────────

/**
 * @brief GPU 粒子（SSBO 元素）
 *
 * 内存布局与 GLSL struct Particle 完全对应：
 *   vec2 position  → 8 bytes
 *   vec2 velocity  → 8 bytes
 *   vec4 color     → 16 bytes
 *   Total          → 32 bytes/粒子
 */
struct Particle {
    float position[2]; ///< NDC 坐标
    float velocity[2]; ///< NDC/秒
    float color[4];    ///< RGBA

    /// @brief 顶点缓冲绑定描述（SSBO 同时作为顶点缓冲使用）
    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription d{};
        d.binding = 0;
        d.stride = sizeof(Particle);
        d.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return d;
    }

    /// @brief 顶点属性描述（position / velocity / color）
    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 3> a{};
        a[0] = {0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Particle, position)};
        a[1] = {1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Particle, velocity)};
        a[2] = {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Particle, color)};
        return a;
    }
};

// ─── Push Constant 结构体（与着色器的 layout(push_constant) 对应）────────────

struct PushConstants {
    float deltaTime; ///< 帧时间（秒），用于物理积分
};

// ─── 应用程序 ─────────────────────────────────────────────────────────────────

class Ch15App {
  public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

  private:
    // ─── Vulkan 基础对象 ───────────────────────────────────────────────────────
    GLFWwindow* window_ = nullptr;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue graphicsQueue_ = VK_NULL_HANDLE;
    VkQueue presentQueue_ = VK_NULL_HANDLE;
    VkQueue computeQueue_ = VK_NULL_HANDLE; // ← 新增计算队列
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;

    // ─── 图形管线（渲染粒子） ─────────────────────────────────────────────────
    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    VkPipelineLayout graphicsPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline_ = VK_NULL_HANDLE;

    // ─── 计算管线（更新粒子物理） ─────────────────────────────────────────────
    VkDescriptorSetLayout computeSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout computePipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline computePipeline_ = VK_NULL_HANDLE;

    // ─── 描述符 ───────────────────────────────────────────────────────────────
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> computeDescSets_; // 每帧一份

    // ─── 命令池与命令缓冲 ─────────────────────────────────────────────────────
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> graphicsCmdBufs_;
    std::vector<VkCommandBuffer> computeCmdBufs_; // ← 独立的计算命令缓冲

    // ─── 粒子 SSBO（每帧一份，双缓冲） ──────────────────────────────────────
    std::vector<VkBuffer> particleBuffers_;
    std::vector<VkDeviceMemory> particleMemories_;

    // ─── 交换链辅助 ───────────────────────────────────────────────────────────
    std::vector<VkImage> swapchainImages_;
    std::vector<VkImageView> swapchainImageViews_;
    std::vector<VkFramebuffer> framebuffers_;
    VkFormat swapchainImageFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchainExtent_{};
    QueueFamilyIndices queueIndices_;

    // ─── 同步原语 ──────────────────────────────────────────────────────────────
    std::vector<VkSemaphore> imageAvailableSems_;
    std::vector<VkSemaphore> renderFinishedSems_;
    std::vector<VkSemaphore> computeFinishedSems_; // ← 计算完成信号量
    std::vector<VkFence> inFlightFences_;
    std::vector<VkFence> computeInFlightFences_; // ← 计算帧栅栏

    uint32_t currentFrame_ = 0;
    bool resized_ = false;

    // ─── 时间统计 ─────────────────────────────────────────────────────────────
    std::chrono::steady_clock::time_point lastTime_;
    uint64_t frameCount_ = 0;
    double totalParticleMs_ = 0.0;

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, WINDOW_TITLE, nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
            reinterpret_cast<Ch15App*>(glfwGetWindowUserPointer(w))->resized_ = true;
        });
    }

    void initVulkan() {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapchain();
        createImageViews();
        createRenderPass();
        createComputeDescriptorSetLayout(); // ← 计算描述符布局
        createGraphicsPipeline();
        createComputePipeline(); // ← 计算管线
        createFramebuffers();
        createCommandPool();
        createParticleSSBO(); // ← 初始化粒子数据
        createDescriptorPool();
        createComputeDescriptorSets(); // ← 绑定 SSBO 到计算着色器
        createCommandBuffers();
        createSyncObjects();

        lastTime_ = std::chrono::steady_clock::now();
        std::cout << "\n✅ 所有 Vulkan 对象初始化完成！\n";
        std::cout << "🎯 粒子数量：" << N_PARTICLES << "\n";
        std::cout << "💻 计算工作组：" << N_PARTICLES / 256 << " 组 × 256 线程\n";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 核心新增 1：初始化粒子 SSBO
    // ═══════════════════════════════════════════════════════════════════════

    void createParticleSSBO() {
        // ① CPU 端随机初始化粒子数据
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> distPos(-0.9f, 0.9f);
        std::uniform_real_distribution<float> distVel(-0.3f, 0.3f);
        std::uniform_real_distribution<float> distCol(0.3f, 1.0f);

        std::vector<Particle> particles(N_PARTICLES);
        for (auto& p : particles) {
            p.position[0] = distPos(rng);
            p.position[1] = distPos(rng);
            // 随机速度，但确保不为零
            do {
                p.velocity[0] = distVel(rng);
            } while (std::abs(p.velocity[0]) < 0.05f);
            do {
                p.velocity[1] = distVel(rng);
            } while (std::abs(p.velocity[1]) < 0.05f);
            p.color[0] = distCol(rng);
            p.color[1] = distCol(rng);
            p.color[2] = distCol(rng);
            p.color[3] = 1.0f;
        }

        VkDeviceSize bufferSize = sizeof(Particle) * N_PARTICLES;
        std::cout << "💾 粒子缓冲大小：" << (bufferSize / 1024) << " KB × " << MAX_FRAMES << "\n";

        // ② 用 Staging Buffer 上传到 GPU（高速 DEVICE_LOCAL 内存）
        VkBuffer stagingBuf;
        VkDeviceMemory stagingMem;
        createBuffer(bufferSize,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuf,
                     stagingMem);

        void* data = nullptr;
        vkMapMemory(device_, stagingMem, 0, bufferSize, 0, &data);
        std::memcpy(data, particles.data(), static_cast<size_t>(bufferSize));
        vkUnmapMemory(device_, stagingMem);

        // ③ 为每个飞行帧创建独立的 SSBO
        //   这样计算着色器修改第 N 帧的缓冲时，第 N-1 帧仍在图形管线中使用
        particleBuffers_.resize(MAX_FRAMES);
        particleMemories_.resize(MAX_FRAMES);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            createBuffer(bufferSize,
                         // STORAGE_BUFFER_BIT：用于 SSBO（计算着色器读写）
                         // VERTEX_BUFFER_BIT ：同时用作顶点缓冲（图形管线读取位置）
                         // TRANSFER_DST_BIT  ：接受从 Staging Buffer 的拷贝
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, // GPU 专用内存，最快
                         particleBuffers_[i],
                         particleMemories_[i]);

            // 用同一份初始数据填充所有帧的缓冲
            copyBuffer(stagingBuf, particleBuffers_[i], bufferSize);
        }
        vkDestroyBuffer(device_, stagingBuf, nullptr);
        vkFreeMemory(device_, stagingMem, nullptr);

        std::cout << "✅ " << MAX_FRAMES << " 个粒子 SSBO 已创建（" << N_PARTICLES << " 粒子）\n";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 核心新增 2：计算描述符集布局
    // ═══════════════════════════════════════════════════════════════════════

    void createComputeDescriptorSetLayout() {
        // 计算着色器绑定：binding=0 → SSBO（可读写）
        VkDescriptorSetLayoutBinding ssboBinding{};
        ssboBinding.binding = 0;
        ssboBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ssboBinding.descriptorCount = 1;
        ssboBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; // 只在计算着色器使用
        ssboBinding.pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 1;
        ci.pBindings = &ssboBinding;
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &ci, nullptr, &computeSetLayout_));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 核心新增 3：创建计算管线
    // ═══════════════════════════════════════════════════════════════════════

    void createComputePipeline() {
        // ─── 管线布局（含 Push Constant 范围） ───────────────────────────────
        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset = 0;
        pcRange.size = sizeof(PushConstants);

        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &computeSetLayout_;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcRange;
        VK_CHECK(vkCreatePipelineLayout(device_, &pli, nullptr, &computePipelineLayout_));

        // ─── 计算着色器阶段（只有一个着色器，无顶点/片段）───────────────────
        VkShaderModule compModule = createShaderModuleFromFile(device_, "particle.comp.spv");

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; // ← 计算阶段
        stage.module = compModule;
        stage.pName = "main";

        // ─── 创建计算管线 ─────────────────────────────────────────────────────
        // 注意：VkComputePipelineCreateInfo 远比图形管线简单
        // 只需指定一个着色器阶段和管线布局
        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = computePipelineLayout_;
        pipelineInfo.stage = stage;
        // basePipelineHandle：可从已有管线派生（减少创建时间）
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        VK_CHECK(vkCreateComputePipelines(device_,
                                          VK_NULL_HANDLE, // Pipeline Cache（可用于跨运行缓存）
                                          1,
                                          &pipelineInfo,
                                          nullptr,
                                          &computePipeline_));

        vkDestroyShaderModule(device_, compModule, nullptr);

        std::cout << "✅ 计算管线创建成功！\n";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 核心新增 4：绑定 SSBO 到计算描述符集
    // ═══════════════════════════════════════════════════════════════════════

    void createComputeDescriptorSets() {
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES, computeSetLayout_);
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = descriptorPool_;
        ai.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES);
        ai.pSetLayouts = layouts.data();
        computeDescSets_.resize(MAX_FRAMES);
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, computeDescSets_.data()));

        for (int i = 0; i < MAX_FRAMES; ++i) {
            // 每帧的计算着色器读写当前帧的粒子缓冲
            VkDescriptorBufferInfo bufInfo{};
            bufInfo.buffer = particleBuffers_[i];
            bufInfo.offset = 0;
            bufInfo.range = VK_WHOLE_SIZE; // 使用整个缓冲

            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = computeDescSets_[i];
            write.dstBinding = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &bufInfo;
            vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
        }
        std::cout << "✅ 计算描述符集已绑定\n";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 核心新增 5：录制计算命令缓冲
    // ═══════════════════════════════════════════════════════════════════════

    void recordComputeCommandBuffer(VkCommandBuffer cmd, float deltaTime) {
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

        // ── 绑定计算管线 ──────────────────────────────────────────────────────
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline_);

        // ── 绑定描述符集（SSBO） ──────────────────────────────────────────────
        vkCmdBindDescriptorSets(cmd,
                                VK_PIPELINE_BIND_POINT_COMPUTE, // 计算绑定点（非图形！）
                                computePipelineLayout_,
                                0,
                                1,
                                &computeDescSets_[currentFrame_],
                                0,
                                nullptr);

        // ── 推送 Push Constant（deltaTime） ──────────────────────────────────
        PushConstants pc{deltaTime};
        vkCmdPushConstants(cmd, computePipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc);

        // ── 派发计算任务！ ────────────────────────────────────────────────────
        // vkCmdDispatch(cmd, groupCountX, groupCountY, groupCountZ)
        //
        // 每个工作组有 local_size_x=256 个线程
        // 总线程 = N_PARTICLES → 需要 N_PARTICLES/256 个工作组
        //
        // GPU 自动并行执行所有工作组（真正的 GPU 并行）
        vkCmdDispatch(cmd, N_PARTICLES / 256, 1, 1);

        VK_CHECK(vkEndCommandBuffer(cmd));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 核心新增 6：录制图形命令缓冲（含计算→图形同步屏障）
    // ═══════════════════════════════════════════════════════════════════════

    void recordGraphicsCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

        // ── Pipeline Barrier：等待计算写完，再让顶点着色器读 ────────────────
        //
        // 这是关键同步点！如果没有这个 barrier：
        //   - 顶点着色器可能读到上一帧（甚至更旧）的粒子数据
        //   - 出现视觉撕裂或数据竞争
#ifdef CH15_USE_SYNCHRONIZATION2
        VkBufferMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = particleBuffers_[currentFrame_];
        barrier.offset = 0;
        barrier.size = VK_WHOLE_SIZE;
        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.bufferMemoryBarrierCount = 1;
        dependency.pBufferMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dependency);
#else
        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = particleBuffers_[currentFrame_];
        barrier.offset = 0;
        barrier.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 0, nullptr,
                             1, &barrier, 0, nullptr);
#endif

        // ── 开始 RenderPass ───────────────────────────────────────────────────
        VkClearValue clearColor{};
        clearColor.color.float32[0] = 0.0f;
        clearColor.color.float32[1] = 0.0f;
        clearColor.color.float32[2] = 0.02f; // 深蓝黑色背景
        clearColor.color.float32[3] = 1.0f;

        VkRenderPassBeginInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp.renderPass = renderPass_;
        rp.framebuffer = framebuffers_[imageIndex];
        rp.renderArea = {{0, 0}, swapchainExtent_};
        rp.clearValueCount = 1;
        rp.pClearValues = &clearColor;
        vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline_);

        VkViewport vp{0, 0, (float)swapchainExtent_.width, (float)swapchainExtent_.height, 0, 1};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{{0, 0}, swapchainExtent_};
        vkCmdSetScissor(cmd, 0, 1, &sc);

        // ── 绑定粒子 SSBO 作为顶点缓冲 ───────────────────────────────────────
        // 同一个缓冲：计算着色器写，图形管线读（通过 barrier 同步）
        VkBuffer vb[] = {particleBuffers_[currentFrame_]};
        VkDeviceSize offs[] = {0};
        vkCmdBindVertexBuffers(cmd, 0, 1, vb, offs);

        // ── 绘制所有粒子为点 ──────────────────────────────────────────────────
        // 每个粒子 = 一个顶点 = 一个 POINT
        vkCmdDraw(cmd, N_PARTICLES, 1, 0, 0);

        vkCmdEndRenderPass(cmd);
        VK_CHECK(vkEndCommandBuffer(cmd));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 核心新增 7：渲染一帧（含计算提交）
    // ═══════════════════════════════════════════════════════════════════════

    void drawFrame() {
        // ─── 计算阶段 ─────────────────────────────────────────────────────────

        // 等待上一轮计算完成
        vkWaitForFences(device_, 1, &computeInFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX);
        vkResetFences(device_, 1, &computeInFlightFences_[currentFrame_]);

        // 计算 deltaTime
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - lastTime_).count();
        lastTime_ = now;
        dt = std::min(dt, 0.05f); // 限制最大步长（窗口最小化恢复时防止穿墙）

        // 重置并录制计算命令缓冲
        vkResetCommandBuffer(computeCmdBufs_[currentFrame_], 0);
        recordComputeCommandBuffer(computeCmdBufs_[currentFrame_], dt);

        // 提交计算命令到计算队列
#ifdef CH15_USE_SYNCHRONIZATION2
        VkCommandBufferSubmitInfo computeCmd{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        computeCmd.commandBuffer = computeCmdBufs_[currentFrame_];
        VkSemaphoreSubmitInfo computeSignal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        computeSignal.semaphore = computeFinishedSems_[currentFrame_];
        computeSignal.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        VkSubmitInfo2 computeSubmit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        computeSubmit.commandBufferInfoCount = 1;
        computeSubmit.pCommandBufferInfos = &computeCmd;
        computeSubmit.signalSemaphoreInfoCount = 1;
        computeSubmit.pSignalSemaphoreInfos = &computeSignal;
        VK_CHECK(vkQueueSubmit2(computeQueue_, 1, &computeSubmit, computeInFlightFences_[currentFrame_]));
#else
        VkSubmitInfo computeSubmit{};
        computeSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        computeSubmit.commandBufferCount = 1;
        computeSubmit.pCommandBuffers = &computeCmdBufs_[currentFrame_];
        computeSubmit.signalSemaphoreCount = 1;
        computeSubmit.pSignalSemaphores = &computeFinishedSems_[currentFrame_];
        VK_CHECK(vkQueueSubmit(computeQueue_, 1, &computeSubmit, computeInFlightFences_[currentFrame_]));
#endif

        // ─── 图形阶段 ─────────────────────────────────────────────────────────

        vkWaitForFences(device_, 1, &inFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex = 0;
        VkResult r = vkAcquireNextImageKHR(
            device_, swapchain_, UINT64_MAX, imageAvailableSems_[currentFrame_], VK_NULL_HANDLE, &imageIndex);
        if (r == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapchain();
            return;
        }

        vkResetFences(device_, 1, &inFlightFences_[currentFrame_]);
        vkResetCommandBuffer(graphicsCmdBufs_[currentFrame_], 0);
        recordGraphicsCommandBuffer(graphicsCmdBufs_[currentFrame_], imageIndex);

        // 图形提交：等待两个信号量：
        //   ① imageAvailableSems_[currentFrame_]  — 交换链图像已获取
        //   ② computeFinishedSems_[currentFrame_] — 计算已完成
        std::array<VkSemaphore, 2> waitSems = {
            imageAvailableSems_[currentFrame_],
            computeFinishedSems_[currentFrame_] // ← 等待计算完成！
        };
        std::array<VkPipelineStageFlags, 2> waitStages = {
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_VERTEX_INPUT_BIT // 在顶点输入阶段等计算
        };

        // Present can retain a binary semaphore after its submitting fence signals.
        // Therefore the render-finished semaphore is owned by the acquired image,
        // not by the CPU frame slot.
        VkSemaphore signalSems[] = {renderFinishedSems_[imageIndex]};
#ifdef CH15_USE_SYNCHRONIZATION2
        std::array<VkSemaphoreSubmitInfo, 2> waitInfos{};
        waitInfos[0].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        waitInfos[0].semaphore = waitSems[0];
        waitInfos[0].stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        waitInfos[1].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        waitInfos[1].semaphore = waitSems[1];
        waitInfos[1].stageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
        VkCommandBufferSubmitInfo graphicsCmd{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        graphicsCmd.commandBuffer = graphicsCmdBufs_[currentFrame_];
        VkSemaphoreSubmitInfo graphicsSignal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        graphicsSignal.semaphore = signalSems[0];
        graphicsSignal.stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;
        VkSubmitInfo2 graphicsSubmit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        graphicsSubmit.waitSemaphoreInfoCount = static_cast<uint32_t>(waitInfos.size());
        graphicsSubmit.pWaitSemaphoreInfos = waitInfos.data();
        graphicsSubmit.commandBufferInfoCount = 1;
        graphicsSubmit.pCommandBufferInfos = &graphicsCmd;
        graphicsSubmit.signalSemaphoreInfoCount = 1;
        graphicsSubmit.pSignalSemaphoreInfos = &graphicsSignal;
        VK_CHECK(vkQueueSubmit2(graphicsQueue_, 1, &graphicsSubmit, inFlightFences_[currentFrame_]));
#else
        VkSubmitInfo graphicsSubmit{};
        graphicsSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        graphicsSubmit.waitSemaphoreCount = static_cast<uint32_t>(waitSems.size());
        graphicsSubmit.pWaitSemaphores = waitSems.data();
        graphicsSubmit.pWaitDstStageMask = waitStages.data();
        graphicsSubmit.commandBufferCount = 1;
        graphicsSubmit.pCommandBuffers = &graphicsCmdBufs_[currentFrame_];
        graphicsSubmit.signalSemaphoreCount = 1;
        graphicsSubmit.pSignalSemaphores = signalSems;
        VK_CHECK(vkQueueSubmit(graphicsQueue_, 1, &graphicsSubmit, inFlightFences_[currentFrame_]));
#endif

        VkSwapchainKHR scs[] = {swapchain_};
        VkPresentInfoKHR pi{};
        pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = signalSems;
        pi.swapchainCount = 1;
        pi.pSwapchains = scs;
        pi.pImageIndices = &imageIndex;
        r = vkQueuePresentKHR(presentQueue_, &pi);
        if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR || resized_) {
            resized_ = false;
            recreateSwapchain();
        }

        currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES;
        ++frameCount_;
    }

    void mainLoop() {
        std::cout << "\n🎆 " << N_PARTICLES << " 个粒子正在 GPU 上跳动！（ESC 退出）\n\n";

        auto fpsTimer = std::chrono::steady_clock::now();

        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
                glfwSetWindowShouldClose(window_, GLFW_TRUE);

            drawFrame();

            auto now = std::chrono::steady_clock::now();
            float elapsed = std::chrono::duration<float>(now - fpsTimer).count();
            if (elapsed >= 3.0f) {
                float fps = frameCount_ / elapsed;
                std::cout << "📊 FPS: " << static_cast<int>(fps) << "  粒子数: " << N_PARTICLES << "  Compute: GPU\n";
                frameCount_ = 0;
                fpsTimer = now;
            }
        }
        vkDeviceWaitIdle(device_);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 图形管线（渲染粒子为彩色圆点）
    // ═══════════════════════════════════════════════════════════════════════

    void createGraphicsPipeline() {
        VkShaderModule vert = createShaderModuleFromFile(device_, "particle.vert.spv");
        VkShaderModule frag = createShaderModuleFromFile(device_, "particle.frag.spv");

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

        auto bd = Particle::getBindingDescription();
        auto ad = Particle::getAttributeDescriptions();
        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &bd;
        vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(ad.size());
        vi.pVertexAttributeDescriptions = ad.data();

        // 拓扑类型：POINT_LIST — 每个顶点渲染为一个点
        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST; // ← 点云！

        VkPipelineViewportStateCreateInfo vs{
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, nullptr, 1, nullptr};

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.lineWidth = 1.0f;
        rs.cullMode = VK_CULL_MODE_NONE; // 点不需要背面剔除

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        // Alpha 混合：粒子边缘渐变
        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        cba.blendEnable = VK_TRUE;
        cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE; // 加法混合（粒子叠加更亮）
        cba.colorBlendOp = VK_BLEND_OP_ADD;
        cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        cba.alphaBlendOp = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1;
        cb.pAttachments = &cba;

        std::vector<VkDynamicState> dyn = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynS{};
        dynS.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynS.dynamicStateCount = static_cast<uint32_t>(dyn.size());
        dynS.pDynamicStates = dyn.data();

        // 图形管线布局（无描述符，无 push constant）
        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        VK_CHECK(vkCreatePipelineLayout(device_, &pli, nullptr, &graphicsPipelineLayout_));

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
        pi.layout = graphicsPipelineLayout_;
        pi.renderPass = renderPass_;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &graphicsPipeline_));

        vkDestroyShaderModule(device_, frag, nullptr);
        vkDestroyShaderModule(device_, vert, nullptr);
        std::cout << "✅ 图形管线创建成功（点云渲染 + Alpha 加法混合）\n";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 同步对象（含计算专用的信号量和栅栏）
    // ═══════════════════════════════════════════════════════════════════════

    void createSyncObjects() {
        imageAvailableSems_.resize(MAX_FRAMES);
        renderFinishedSems_.resize(swapchainImages_.size());
        computeFinishedSems_.resize(MAX_FRAMES);
        inFlightFences_.resize(MAX_FRAMES);
        computeInFlightFences_.resize(MAX_FRAMES);

        VkSemaphoreCreateInfo sCI{};
        sCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkFenceCreateInfo fCI{};
        fCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (int i = 0; i < MAX_FRAMES; ++i) {
            VK_CHECK(vkCreateSemaphore(device_, &sCI, nullptr, &imageAvailableSems_[i]));
            VK_CHECK(vkCreateSemaphore(device_, &sCI, nullptr, &computeFinishedSems_[i]));
            VK_CHECK(vkCreateFence(device_, &fCI, nullptr, &inFlightFences_[i]));
            VK_CHECK(vkCreateFence(device_, &fCI, nullptr, &computeInFlightFences_[i]));
        }
        for (auto& semaphore : renderFinishedSems_)
            VK_CHECK(vkCreateSemaphore(device_, &sCI, nullptr, &semaphore));
        std::cout << "✅ 同步对象已创建（含计算专用信号量和栅栏）\n";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Vulkan 初始化（复用前几章，重点是队列族查找加入 Compute）
    // ═══════════════════════════════════════════════════════════════════════

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
        enablePortabilityBit(ci);

        if (ENABLE_VALIDATION_LAYERS && checkValidationLayerSupport()) {
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
        std::vector<VkPhysicalDevice> devs(count);
        vkEnumeratePhysicalDevices(instance_, &count, devs.data());
        for (auto& d : devs) {
            if (findQueueFamilies(d, surface_).isComplete() && checkDeviceExtensionSupport(d)) {
                physicalDevice_ = d;
                break;
            }
        }
        if (!physicalDevice_)
            throw std::runtime_error("无合适GPU");
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(physicalDevice_, &p);
        std::cout << "✅ GPU: " << p.deviceName << "\n";
    }

    void createLogicalDevice() {
        queueIndices_ = findQueueFamilies(physicalDevice_, surface_);

        // ── 查找支持 Compute 的队列族 ─────────────────────────────────────────
        //
        // 大多数现代 GPU 的图形队列族同时支持 Compute。
        // 某些 GPU 有专用的 Compute 队列族（更高并行效率）。
        // 这里优先使用图形+计算同一队列族（简化同步），
        // 实际项目可以使用专用 Compute 队列族进行异步计算。
        uint32_t computeFamily = queueIndices_.graphicsFamily.value();

        // 检查是否有专用计算队列族
        uint32_t qfCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &qfCount, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(qfCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &qfCount, qfs.data());

        for (uint32_t i = 0; i < qfCount; ++i) {
            if ((qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                computeFamily = i; // 找到专用 Compute 队列族
                std::cout << "🚀 发现专用 Compute 队列族（索引=" << i << "）\n";
                break;
            }
        }
        std::cout << "🔧 使用计算队列族：" << computeFamily << "\n";

        std::set<uint32_t> uniqueFamilies = {
            queueIndices_.graphicsFamily.value(), queueIndices_.presentFamily.value(), computeFamily};
        const float pri = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> qcis;
        for (uint32_t f : uniqueFamilies) {
            VkDeviceQueueCreateInfo q{};
            q.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            q.queueFamilyIndex = f;
            q.queueCount = 1;
            q.pQueuePriorities = &pri;
            qcis.push_back(q);
        }

        VkPhysicalDeviceFeatures feat{};
        feat.samplerAnisotropy = VK_TRUE;

#ifdef CH15_USE_SYNCHRONIZATION2
        VkPhysicalDeviceVulkan13Features supportedVulkan13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceFeatures2 supportedFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        supportedFeatures.pNext = &supportedVulkan13;
        vkGetPhysicalDeviceFeatures2(physicalDevice_, &supportedFeatures);
        if (!supportedVulkan13.synchronization2)
            throw std::runtime_error("当前设备不支持 Vulkan Synchronization 2");
        if (!supportedVulkan13.shaderDemoteToHelperInvocation)
            throw std::runtime_error("当前设备不支持 particle.frag 所需的 shaderDemoteToHelperInvocation");
        VkPhysicalDeviceVulkan13Features vulkan13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        vulkan13.synchronization2 = VK_TRUE;
        vulkan13.shaderDemoteToHelperInvocation = VK_TRUE;
#endif

        VkDeviceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
#ifdef CH15_USE_SYNCHRONIZATION2
        ci.pNext = &vulkan13;
#endif
        ci.queueCreateInfoCount = static_cast<uint32_t>(qcis.size());
        ci.pQueueCreateInfos = qcis.data();
        ci.pEnabledFeatures = &feat;
        ci.enabledExtensionCount = static_cast<uint32_t>(DEVICE_EXTENSIONS.size());
        ci.ppEnabledExtensionNames = DEVICE_EXTENSIONS.data();
        if (ENABLE_VALIDATION_LAYERS && checkValidationLayerSupport()) {
            ci.enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
            ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();
        }
        VK_CHECK(vkCreateDevice(physicalDevice_, &ci, nullptr, &device_));

        vkGetDeviceQueue(device_, queueIndices_.graphicsFamily.value(), 0, &graphicsQueue_);
        vkGetDeviceQueue(device_, queueIndices_.presentFamily.value(), 0, &presentQueue_);
        vkGetDeviceQueue(device_, computeFamily, 0, &computeQueue_);
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

        // 注意：srcStageMask 需要包含 COMPUTE_SHADER，确保计算完成后图形才开始
        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
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

    void createDescriptorPool() {
        // 需要 MAX_FRAMES 个 STORAGE_BUFFER 描述符（给计算着色器的 SSBO）
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, static_cast<uint32_t>(MAX_FRAMES)};
        VkDescriptorPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        ci.poolSizeCount = 1;
        ci.pPoolSizes = &ps;
        ci.maxSets = static_cast<uint32_t>(MAX_FRAMES);
        VK_CHECK(vkCreateDescriptorPool(device_, &ci, nullptr, &descriptorPool_));
    }

    void createCommandBuffers() {
        graphicsCmdBufs_.resize(MAX_FRAMES);
        computeCmdBufs_.resize(MAX_FRAMES);

        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = commandPool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = static_cast<uint32_t>(graphicsCmdBufs_.size());
        VK_CHECK(vkAllocateCommandBuffers(device_, &ai, graphicsCmdBufs_.data()));

        ai.commandBufferCount = static_cast<uint32_t>(computeCmdBufs_.size());
        VK_CHECK(vkAllocateCommandBuffers(device_, &ai, computeCmdBufs_.data()));
    }

    // ─── 辅助函数 ─────────────────────────────────────────────────────────────

    uint32_t findMemoryType(uint32_t f, VkMemoryPropertyFlags p) {
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((f & (1u << i)) && (mp.memoryTypes[i].propertyFlags & p) == p)
                return i;
        throw std::runtime_error("找不到合适内存类型");
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

    void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size) {
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandPool = commandPool_;
        ai.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        vkAllocateCommandBuffers(device_, &ai, &cmd);
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bi);
        VkBufferCopy region{};
        region.size = size;
        vkCmdCopyBuffer(cmd, src, dst, 1, &region);
        vkEndCommandBuffer(cmd);
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        vkQueueSubmit(graphicsQueue_, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue_);
        vkFreeCommandBuffers(device_, commandPool_, 1, &cmd);
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
            vkDestroyBuffer(device_, particleBuffers_[i], nullptr);
            vkFreeMemory(device_, particleMemories_[i], nullptr);
            vkDestroySemaphore(device_, imageAvailableSems_[i], nullptr);
            vkDestroySemaphore(device_, computeFinishedSems_[i], nullptr);
            vkDestroyFence(device_, inFlightFences_[i], nullptr);
            vkDestroyFence(device_, computeInFlightFences_[i], nullptr);
        }
        for (auto semaphore : renderFinishedSems_)
            vkDestroySemaphore(device_, semaphore, nullptr);
        vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, computeSetLayout_, nullptr);
        vkDestroyCommandPool(device_, commandPool_, nullptr);
        for (auto& fb : framebuffers_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        vkDestroyPipeline(device_, computePipeline_, nullptr);
        vkDestroyPipelineLayout(device_, computePipelineLayout_, nullptr);
        vkDestroyPipeline(device_, graphicsPipeline_, nullptr);
        vkDestroyPipelineLayout(device_, graphicsPipelineLayout_, nullptr);
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
#ifdef CH15_USE_SYNCHRONIZATION2
    std::cout << " 第95章：Synchronization 2 — GPU 粒子系统\n";
#else
    std::cout << " 第15章：Compute Shader — GPU 粒子系统\n";
#endif
    std::cout << "\n";
    std::cout << " 核心概念：\n";
    std::cout << "   • VkComputePipeline       — 独立于图形管线的计算管线\n";
    std::cout << "   • vkCmdDispatch(32,1,1)   — 派发 32×256=8192 个 GPU 线程\n";
    std::cout << "   • SSBO                    — 计算着色器读写，同时作为顶点缓冲\n";
    std::cout << "   • Pipeline Barrier        — 计算→图形的同步屏障\n";
    std::cout << "   • Push Constants          — 快速传递 deltaTime 给计算着色器\n";
    std::cout << "   • Compute Semaphore       — 计算完成信号给图形队列\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n\n";

    Ch15App app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ 错误：" << e.what() << "\n";
        return 1;
    }
    return 0;
}
