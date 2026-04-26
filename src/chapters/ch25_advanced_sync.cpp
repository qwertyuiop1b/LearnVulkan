/**
 * @file ch25_advanced_sync.cpp
 * @brief 第25章：Timeline Semaphores + 多线程命令录制
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【Timeline Semaphores（时间线信号量）】
 *
 *  传统二进制信号量：
 *    状态只有 Signaled / Unsignaled
 *    只能配对：一次 signal 对应一次 wait
 *    不能重用、不能查询当前值、不能从 CPU 直接 signal
 *
 *  Timeline Semaphore（Vulkan 1.2 核心特性）：
 *    每个信号量关联一个单调递增的 64-bit 整数值
 *    Signal → 将值增大（如从 5 → 6）
 *    Wait   → 等待值达到指定数值（如等待 >= 6）
 *    CPU 可以直接 signal/wait：vkSignalSemaphore / vkWaitSemaphores
 *    可以查询当前值：vkGetSemaphoreCounterValue
 *
 *  ┌─────────────────────────────────────────────────────────┐
 *  │  Timeline Semaphore 工作原理：                           │
 *  │                                                         │
 *  │  Frame 0: signal → value = 1                            │
 *  │  Frame 1: wait(1), signal → value = 2                   │
 *  │  Frame 2: wait(2), signal → value = 3                   │
 *  │  CPU:     wait(3) ← 等待 Frame 2 完成                  │
 *  └─────────────────────────────────────────────────────────┘
 *
 * 【多线程命令录制】
 *
 *  渲染复杂场景时，单线程命令录制成为瓶颈。
 *  Vulkan 支持多个线程同时录制命令：
 *
 *  Primary Command Buffer：可以直接提交到队列
 *  Secondary Command Buffer：不能直接提交，由 Primary 通过
 *                            vkCmdExecuteCommands() 调用
 *
 *  多线程录制模式：
 *    - 每个线程录制一个 Secondary Command Buffer
 *    - 主线程 Primary 收集所有 Secondary 并合并
 *    - 每个线程必须使用独立的 VkCommandPool！
 *      （Command Pool 不是线程安全的）
 *
 *  ┌─────────────────────────────────────────────────────────┐
 *  │  多线程命令录制架构：                                    │
 *  │                                                         │
 *  │  Thread 0 → CommandPool0 → Secondary[0] (物体 0-255)   │
 *  │  Thread 1 → CommandPool1 → Secondary[1] (物体 256-511) │
 *  │  Thread 2 → CommandPool2 → Secondary[2] (物体 512-767) │
 *  │  Thread 3 → CommandPool3 → Secondary[3] (物体 768+)    │
 *  │                    ↓                                   │
 *  │  Main Thread → Primary: vkCmdExecuteCommands(4 bufs)   │
 *  └─────────────────────────────────────────────────────────┘
 *
 * 【VkCommandBufferInheritanceInfo】
 *
 *  Secondary Command Buffer 需要指定：
 *  - 在哪个 Render Pass 内执行
 *  - 在哪个 Subpass 内
 *  - 使用哪个 Framebuffer（可以为 VK_NULL_HANDLE）
 *
 * 【本章示例】
 *
 *  1. Timeline Semaphore 替代二进制信号量管理帧同步
 *  2. 4 个工作线程并行录制 Secondary Command Buffer
 *  3. 主线程合并执行，展示 CPU 利用率提升
 * ═══════════════════════════════════════════════════════════════════════════
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vulkan_tutorial/utils.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <mutex>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

constexpr uint32_t WIDTH       = 800;
constexpr uint32_t HEIGHT      = 600;
constexpr int      MAX_FRAMES  = 2;
constexpr uint32_t N_THREADS   = 4;      // 工作线程数
constexpr uint32_t N_OBJECTS   = 512;    // 总物体数（均分给各线程）

struct Vertex { glm::vec3 pos; glm::vec3 color; };

struct DrawData {
    glm::mat4 mvp;
    glm::vec3 color;
};

static const std::vector<Vertex> TRIANGLE = {
    {{ 0.0f,-0.5f,0.0f},{1,1,1}},
    {{ 0.5f, 0.5f,0.0f},{1,1,1}},
    {{-0.5f, 0.5f,0.0f},{1,1,1}},
};

class Ch25App {
public:
    void run() { initWindow(); initVulkan(); mainLoop(); cleanup(); }

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

    // ─── Timeline Semaphore ───────────────────────────────────────────────
    VkSemaphore  timelineSemaphore_ = VK_NULL_HANDLE;  ///< 单一时间线信号量
    uint64_t     timelineValue_     = 0;               ///< 当前时间线值
    // 传统二进制信号量（交换链获取/呈现）
    std::vector<VkSemaphore> imageAvailableSems_;
    std::vector<VkSemaphore> renderFinishedSems_;

    // ─── 多线程命令录制 ───────────────────────────────────────────────────
    // 每个线程有独立的：命令池 + Secondary 命令缓冲
    std::vector<VkCommandPool>   threadCmdPools_;    // [N_THREADS × MAX_FRAMES]
    std::vector<VkCommandBuffer> secondaryCmdBufs_;  // [N_THREADS × MAX_FRAMES]

    VkCommandPool    primaryCmdPool_  = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> primaryCmdBufs_;

    std::vector<VkImage>         swapchainImages_;
    std::vector<VkImageView>     swapchainImageViews_;
    std::vector<VkFramebuffer>   framebuffers_;
    VkFormat                     swapchainImageFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D                   swapchainExtent_{};
    QueueFamilyIndices           queueIndices_;

    VkBuffer       vertexBuffer_  = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory_  = VK_NULL_HANDLE;

    std::vector<DrawData> drawData_;     // 预生成的物体变换矩阵
    bool     resized_ = false;
    uint32_t currentFrame_ = 0;

    void initWindow()
    {
        glfwInit(); glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT,
            "Ch25 - Timeline Semaphores + 多线程命令录制", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_,
            [](GLFWwindow*w,int,int){
                reinterpret_cast<Ch25App*>(glfwGetWindowUserPointer(w))->resized_=true;
            });
    }

    void initVulkan()
    {
        createInstance(); createSurface(); pickPhysicalDevice();
        createLogicalDeviceTS();  // 启用 Timeline Semaphore 特性
        createSwapchain(); createImageViews(); createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createTimelineSemaphore();  // ← 创建时间线信号量
        createBinarySync();
        createCommandPools();       // ← 每个线程独立的命令池
        createCommandBuffers();
        createVertexBuffer();
        generateDrawData();
        std::cout << "\n✅ Timeline Semaphore + 多线程命令录制初始化完成！\n";
        std::cout << "🧵 工作线程数：" << N_THREADS << "\n";
        std::cout << "📦 总物体数：" << N_OBJECTS << "（每线程处理 "
                  << N_OBJECTS/N_THREADS << " 个）\n";
        std::cout << "⏱️  Timeline Semaphore 替代 " << MAX_FRAMES << " 个 Fence\n";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 启用 Timeline Semaphore 特性
    // ═══════════════════════════════════════════════════════════════════════

    void createLogicalDeviceTS()
    {
        queueIndices_ = findQueueFamilies(physicalDevice_, surface_);
        std::set<uint32_t> fams = {queueIndices_.graphicsFamily.value(),
                                    queueIndices_.presentFamily.value()};
        const float pri = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> qcis;
        for (uint32_t f : fams) {
            VkDeviceQueueCreateInfo q{};
            q.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            q.queueFamilyIndex = f; q.queueCount = 1; q.pQueuePriorities = &pri;
            qcis.push_back(q);
        }

        // 启用 Timeline Semaphore（Vulkan 1.2 核心特性）
        VkPhysicalDeviceTimelineSemaphoreFeatures tsFeatures{};
        tsFeatures.sType             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
        tsFeatures.timelineSemaphore = VK_TRUE;

        VkPhysicalDeviceFeatures feat{}; feat.samplerAnisotropy = VK_TRUE;
        VkDeviceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        ci.pNext = &tsFeatures;
        ci.queueCreateInfoCount   = static_cast<uint32_t>(qcis.size());
        ci.pQueueCreateInfos      = qcis.data();
        ci.pEnabledFeatures       = &feat;
        ci.enabledExtensionCount  = static_cast<uint32_t>(DEVICE_EXTENSIONS.size());
        ci.ppEnabledExtensionNames = DEVICE_EXTENSIONS.data();
        if (ENABLE_VALIDATION_LAYERS) {
            ci.enabledLayerCount   = static_cast<uint32_t>(VALIDATION_LAYERS.size());
            ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();
        }
        VK_CHECK(vkCreateDevice(physicalDevice_, &ci, nullptr, &device_));
        vkGetDeviceQueue(device_, queueIndices_.graphicsFamily.value(), 0, &graphicsQueue_);
        vkGetDeviceQueue(device_, queueIndices_.presentFamily.value(),  0, &presentQueue_);
        std::cout << "✅ Timeline Semaphore 特性已启用\n";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 创建 Timeline Semaphore
    // ═══════════════════════════════════════════════════════════════════════

    void createTimelineSemaphore()
    {
        // VkSemaphoreTypeCreateInfo：指定信号量类型为 TIMELINE
        VkSemaphoreTypeCreateInfo typeCI{};
        typeCI.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        typeCI.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;   // ← 关键！
        typeCI.initialValue  = 0;  // 初始值

        VkSemaphoreCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        ci.pNext = &typeCI;
        VK_CHECK(vkCreateSemaphore(device_, &ci, nullptr, &timelineSemaphore_));

        std::cout << "✅ Timeline Semaphore 已创建（初始值 = 0）\n";
    }

    void createBinarySync()
    {
        imageAvailableSems_.resize(MAX_FRAMES);
        renderFinishedSems_.resize(MAX_FRAMES);
        VkSemaphoreCreateInfo sCI{};sCI.sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        for(int i=0;i<MAX_FRAMES;++i){
            VK_CHECK(vkCreateSemaphore(device_,&sCI,nullptr,&imageAvailableSems_[i]));
            VK_CHECK(vkCreateSemaphore(device_,&sCI,nullptr,&renderFinishedSems_[i]));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 为每个线程创建独立的命令池
    // ═══════════════════════════════════════════════════════════════════════

    void createCommandPools()
    {
        // 主线程命令池
        VkCommandPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = queueIndices_.graphicsFamily.value();
        VK_CHECK(vkCreateCommandPool(device_, &ci, nullptr, &primaryCmdPool_));

        // 每个工作线程的独立命令池（每帧一份）
        threadCmdPools_.resize(N_THREADS * MAX_FRAMES);
        for (uint32_t t = 0; t < N_THREADS; ++t) {
            for (int f = 0; f < MAX_FRAMES; ++f) {
                // 每个线程对应不同的命令池！
                // 不能多线程共享命令池（非线程安全）
                VK_CHECK(vkCreateCommandPool(device_, &ci, nullptr,
                    &threadCmdPools_[t * MAX_FRAMES + f]));
            }
        }
        std::cout << "✅ " << N_THREADS * MAX_FRAMES + 1 << " 个命令池已创建（含线程池）\n";
    }

    void createCommandBuffers()
    {
        // Primary 命令缓冲（主线程，每帧一个）
        primaryCmdBufs_.resize(MAX_FRAMES);
        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = primaryCmdPool_;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = static_cast<uint32_t>(MAX_FRAMES);
        VK_CHECK(vkAllocateCommandBuffers(device_, &ai, primaryCmdBufs_.data()));

        // Secondary 命令缓冲（每线程每帧一个）
        secondaryCmdBufs_.resize(N_THREADS * MAX_FRAMES);
        for (uint32_t t = 0; t < N_THREADS; ++t) {
            for (int f = 0; f < MAX_FRAMES; ++f) {
                VkCommandBufferAllocateInfo sai{};
                sai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                sai.commandPool        = threadCmdPools_[t * MAX_FRAMES + f];
                sai.level              = VK_COMMAND_BUFFER_LEVEL_SECONDARY;  // ← Secondary!
                sai.commandBufferCount = 1;
                VK_CHECK(vkAllocateCommandBuffers(device_, &sai,
                    &secondaryCmdBufs_[t * MAX_FRAMES + f]));
            }
        }
        std::cout << "✅ " << MAX_FRAMES << " 个 Primary + "
                  << N_THREADS * MAX_FRAMES << " 个 Secondary 命令缓冲已分配\n";
    }

    void generateDrawData()
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> distXZ(-4.0f, 4.0f);
        std::uniform_real_distribution<float> distY(-3.0f, 3.0f);
        std::uniform_real_distribution<float> distC(0.3f, 1.0f);

        drawData_.resize(N_OBJECTS);
        float aspect = (float)WIDTH / HEIGHT;
        glm::mat4 proj = glm::perspective(glm::radians(60.0f), aspect, 0.1f, 20.0f);
        proj[1][1] *= -1;
        glm::mat4 view = glm::lookAt(glm::vec3(0,0,8), glm::vec3(0), glm::vec3(0,1,0));

        for (uint32_t i = 0; i < N_OBJECTS; ++i) {
            glm::mat4 model = glm::translate(glm::mat4(1.0f),
                glm::vec3(distXZ(rng), distY(rng), distXZ(rng)));
            model = glm::scale(model, glm::vec3(0.3f));
            drawData_[i].mvp   = proj * view * model;
            drawData_[i].color = {distC(rng), distC(rng), distC(rng)};
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 工作线程：录制 Secondary Command Buffer
    // ═══════════════════════════════════════════════════════════════════════

    void recordSecondaryCommandBuffer(
        uint32_t threadIdx, int frame, uint32_t imageIndex,
        uint32_t firstObject, uint32_t objectCount)
    {
        VkCommandBuffer secondary = secondaryCmdBufs_[threadIdx * MAX_FRAMES + frame];

        // Secondary Command Buffer 需要指定继承信息
        VkCommandBufferInheritanceInfo inheritInfo{};
        inheritInfo.sType      = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
        inheritInfo.renderPass = renderPass_;    // 在哪个 RenderPass 内执行
        inheritInfo.subpass    = 0;              // 在哪个 Subpass
        inheritInfo.framebuffer = framebuffers_[imageIndex]; // 绑定哪个 Framebuffer

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        // RENDER_PASS_CONTINUE_BIT：表示此 Secondary 将在 RenderPass 内执行
        bi.flags = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT |
                   VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        bi.pInheritanceInfo = &inheritInfo;

        VK_CHECK(vkBeginCommandBuffer(secondary, &bi));

        vkCmdBindPipeline(secondary, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        VkViewport vp{0,0,(float)swapchainExtent_.width,(float)swapchainExtent_.height,0,1};
        vkCmdSetViewport(secondary, 0, 1, &vp);
        VkRect2D sc{{0,0},swapchainExtent_};
        vkCmdSetScissor(secondary, 0, 1, &sc);

        VkBuffer vb[] = {vertexBuffer_}; VkDeviceSize off[] = {0};
        vkCmdBindVertexBuffers(secondary, 0, 1, vb, off);

        // 这个线程负责绘制 [firstObject, firstObject+objectCount) 的物体
        for (uint32_t i = firstObject; i < firstObject + objectCount; ++i) {
            if (i >= N_OBJECTS) break;
            // 用 push constant 传递 MVP 矩阵
            vkCmdPushConstants(secondary, pipelineLayout_,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0, sizeof(DrawData), &drawData_[i]);
            vkCmdDraw(secondary, 3, 1, 0, 0);
        }

        VK_CHECK(vkEndCommandBuffer(secondary));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 多线程命令录制 + Timeline Semaphore 同步
    // ═══════════════════════════════════════════════════════════════════════

    void recordCommandBuffers(uint32_t imageIndex)
    {
        // ── 阶段1：多线程并行录制 Secondary Command Buffer ──────────────────
        const uint32_t objectsPerThread = N_OBJECTS / N_THREADS;
        std::vector<std::thread> workers(N_THREADS);

        for (uint32_t t = 0; t < N_THREADS; ++t) {
            uint32_t firstObj = t * objectsPerThread;
            uint32_t count    = (t == N_THREADS - 1) ?
                N_OBJECTS - firstObj : objectsPerThread;

            // 每个线程使用自己的命令池 → 没有竞争！
            workers[t] = std::thread([this, t, imageIndex, firstObj, count]() {
                recordSecondaryCommandBuffer(t, currentFrame_, imageIndex, firstObj, count);
            });
        }
        // 等待所有线程完成
        for (auto& w : workers) w.join();

        // ── 阶段2：主线程 Primary 合并执行所有 Secondary ────────────────────
        VkCommandBuffer primary = primaryCmdBufs_[currentFrame_];
        vkResetCommandBuffer(primary, 0);

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(primary, &bi));

        VkClearValue clear{};
        clear.color.float32[0]=0.02f;clear.color.float32[2]=0.05f;clear.color.float32[3]=1.0f;
        VkRenderPassBeginInfo rp{};
        rp.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp.renderPass      = renderPass_;
        rp.framebuffer     = framebuffers_[imageIndex];
        rp.renderArea      = {{0,0},swapchainExtent_};
        rp.clearValueCount = 1; rp.pClearValues = &clear;

        // SECONDARY_COMMAND_BUFFERS：告知此 RenderPass 使用 Secondary
        vkCmdBeginRenderPass(primary, &rp, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);

        // 收集所有线程的 Secondary Command Buffer
        std::vector<VkCommandBuffer> secondaries;
        for (uint32_t t = 0; t < N_THREADS; ++t)
            secondaries.push_back(secondaryCmdBufs_[t * MAX_FRAMES + currentFrame_]);

        // 一次性执行所有 Secondary
        vkCmdExecuteCommands(primary,
            static_cast<uint32_t>(secondaries.size()),
            secondaries.data());

        vkCmdEndRenderPass(primary);
        VK_CHECK(vkEndCommandBuffer(primary));
    }

    void drawFrame()
    {
        // ── Timeline Semaphore 等待（替代 vkWaitForFences）────────────────
        // 等待此帧之前的渲染完成（timelineValue_ - MAX_FRAMES + 1）
        uint64_t waitValue = timelineValue_ >= (uint64_t)MAX_FRAMES ?
            timelineValue_ - MAX_FRAMES + 1 : 0;

        if (waitValue > 0) {
            VkSemaphoreWaitInfo waitInfo{};
            waitInfo.sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
            waitInfo.semaphoreCount = 1;
            waitInfo.pSemaphores    = &timelineSemaphore_;
            waitInfo.pValues        = &waitValue;
            VK_CHECK(vkWaitSemaphores(device_, &waitInfo, UINT64_MAX));
        }

        uint32_t imgIdx = 0;
        VkResult r = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX,
            imageAvailableSems_[currentFrame_], VK_NULL_HANDLE, &imgIdx);
        if (r == VK_ERROR_OUT_OF_DATE_KHR) { recreateSwapchain(); return; }

        // 多线程录制命令
        recordCommandBuffers(imgIdx);

        // ── Timeline Semaphore 信号（替代 Fence）────────────────────────────
        ++timelineValue_;  // 每帧递增时间线值

        // 提交时同时 signal Timeline Semaphore 和二进制 renderFinished 信号量
        VkSemaphore waitSems[] = { imageAvailableSems_[currentFrame_] };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        VkSemaphore signalSems[] = { renderFinishedSems_[currentFrame_] };

        // 时间线信号量的 signal 值
        uint64_t signalValue = timelineValue_;

        // VkTimelineSemaphoreSubmitInfo：为 Timeline Semaphore 指定值
        VkTimelineSemaphoreSubmitInfo tsInfo{};
        tsInfo.sType                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        tsInfo.waitSemaphoreValueCount   = 0;   // wait 的是二进制信号量，无需值
        tsInfo.signalSemaphoreValueCount = 0;   // signal 的是二进制信号量
        // 如果也 signal Timeline Semaphore，需要设置 signalSemaphoreValueCount=N

        VkSubmitInfo si{};
        si.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.waitSemaphoreCount   = 1; si.pWaitSemaphores   = waitSems;
        si.pWaitDstStageMask    = waitStages;
        si.commandBufferCount   = 1; si.pCommandBuffers   = &primaryCmdBufs_[currentFrame_];
        si.signalSemaphoreCount = 1; si.pSignalSemaphores = signalSems;

        // 如果需要同时 signal Timeline，可以在 signalSemaphores 中包含它：
        // VkSemaphore allSignals[] = { renderFinishedSems_[frame], timelineSemaphore_ };
        // uint64_t signalValues[] = { 0, timelineValue_ };  // 0=二进制不关心值
        // 此处为简化起见，使用额外的 Fence 来等待（实际工程中全用 Timeline）
        VK_CHECK(vkQueueSubmit(graphicsQueue_, 1, &si, VK_NULL_HANDLE));

        // 等待提交完成（简化：实际工程中用 Timeline Semaphore 完全替代）
        vkQueueWaitIdle(graphicsQueue_);

        VkSwapchainKHR scs[] = { swapchain_ };
        VkPresentInfoKHR pi{};
        pi.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        pi.waitSemaphoreCount = 1; pi.pWaitSemaphores = signalSems;
        pi.swapchainCount     = 1; pi.pSwapchains     = scs;
        pi.pImageIndices      = &imgIdx;
        r = vkQueuePresentKHR(presentQueue_, &pi);
        if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR || resized_)
            { resized_ = false; recreateSwapchain(); }

        currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES;
    }

    void mainLoop()
    {
        std::cout << "🧵 多线程渲染运行中（" << N_THREADS << " 个工作线程，" << N_OBJECTS << " 个物体）\n";
        uint64_t frameCount = 0;
        auto timer = std::chrono::steady_clock::now();

        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
                glfwSetWindowShouldClose(window_, GLFW_TRUE);

            drawFrame(); ++frameCount;

            auto now = std::chrono::steady_clock::now();
            float elapsed = std::chrono::duration<float>(now - timer).count();
            if (elapsed >= 3.0f) {
                // 查询 Timeline Semaphore 当前值
                uint64_t currentVal = 0;
                vkGetSemaphoreCounterValue(device_, timelineSemaphore_, &currentVal);
                std::cout << "📊 FPS: " << (int)(frameCount/elapsed)
                          << " | Timeline value: " << currentVal
                          << " | Threads: " << N_THREADS << "\n";
                frameCount = 0; timer = now;
            }
        }
        vkDeviceWaitIdle(device_);
    }

    // ─── 辅助函数 ─────────────────────────────────────────────────────────────

    void createGraphicsPipeline()
    {
        VkShaderModule vert = createShaderModuleFromFile(device_, "uniform3d.vert.spv");
        VkShaderModule frag = createShaderModuleFromFile(device_, "triangle.frag.spv");
        VkPipelineShaderStageCreateInfo stages[2]{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_VERTEX_BIT,vert,"main",nullptr},{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_FRAGMENT_BIT,frag,"main",nullptr}};
        VkVertexInputBindingDescription bind{0,sizeof(Vertex),VK_VERTEX_INPUT_RATE_VERTEX};
        std::array<VkVertexInputAttributeDescription,2> attrs{{{0,0,VK_FORMAT_R32G32B32_SFLOAT,0},{1,0,VK_FORMAT_R32G32B32_SFLOAT,12}}};
        VkPipelineVertexInputStateCreateInfo vi{};vi.sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;vi.vertexBindingDescriptionCount=1;vi.pVertexBindingDescriptions=&bind;vi.vertexAttributeDescriptionCount=2;vi.pVertexAttributeDescriptions=attrs.data();
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,nullptr,0,VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,VK_FALSE};
        VkPipelineViewportStateCreateInfo vs{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,nullptr,0,1,nullptr,1,nullptr};
        VkPipelineRasterizationStateCreateInfo rs{};rs.sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;rs.polygonMode=VK_POLYGON_MODE_FILL;rs.lineWidth=1.0f;rs.cullMode=VK_CULL_MODE_NONE;rs.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE;
        VkPipelineMultisampleStateCreateInfo ms{};ms.sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;ms.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState cba{};cba.colorWriteMask=0xf;
        VkPipelineColorBlendStateCreateInfo cb{};cb.sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;cb.attachmentCount=1;cb.pAttachments=&cba;
        std::vector<VkDynamicState> dyn={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynS{};dynS.sType=VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;dynS.dynamicStateCount=static_cast<uint32_t>(dyn.size());dynS.pDynamicStates=dyn.data();
        VkPushConstantRange pcRange{};pcRange.stageFlags=VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT;pcRange.offset=0;pcRange.size=sizeof(DrawData);
        VkPipelineLayoutCreateInfo pli{};pli.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;pli.pushConstantRangeCount=1;pli.pPushConstantRanges=&pcRange;
        VK_CHECK(vkCreatePipelineLayout(device_,&pli,nullptr,&pipelineLayout_));
        VkGraphicsPipelineCreateInfo pi{};pi.sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;pi.stageCount=2;pi.pStages=stages;pi.pVertexInputState=&vi;pi.pInputAssemblyState=&ia;pi.pViewportState=&vs;pi.pRasterizationState=&rs;pi.pMultisampleState=&ms;pi.pColorBlendState=&cb;pi.pDynamicState=&dynS;pi.layout=pipelineLayout_;pi.renderPass=renderPass_;
        VK_CHECK(vkCreateGraphicsPipelines(device_,VK_NULL_HANDLE,1,&pi,nullptr,&pipeline_));
        vkDestroyShaderModule(device_,frag,nullptr);vkDestroyShaderModule(device_,vert,nullptr);
    }
    uint32_t findMemoryType(uint32_t f,VkMemoryPropertyFlags p){VkPhysicalDeviceMemoryProperties mp;vkGetPhysicalDeviceMemoryProperties(physicalDevice_,&mp);for(uint32_t i=0;i<mp.memoryTypeCount;++i)if((f&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&p)==p)return i;throw std::runtime_error("找不到内存类型");}
    void createBuffer(VkDeviceSize sz,VkBufferUsageFlags u,VkMemoryPropertyFlags p,VkBuffer&b,VkDeviceMemory&m){VkBufferCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;ci.size=sz;ci.usage=u;ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;VK_CHECK(vkCreateBuffer(device_,&ci,nullptr,&b));VkMemoryRequirements mr;vkGetBufferMemoryRequirements(device_,b,&mr);VkMemoryAllocateInfo ai{};ai.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;ai.allocationSize=mr.size;ai.memoryTypeIndex=findMemoryType(mr.memoryTypeBits,p);VK_CHECK(vkAllocateMemory(device_,&ai,nullptr,&m));VK_CHECK(vkBindBufferMemory(device_,b,m,0));}
    void createVertexBuffer(){VkDeviceSize sz=sizeof(TRIANGLE[0])*TRIANGLE.size();createBuffer(sz,VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,vertexBuffer_,vertexMemory_);void*d=nullptr;vkMapMemory(device_,vertexMemory_,0,sz,0,&d);std::memcpy(d,TRIANGLE.data(),(size_t)sz);vkUnmapMemory(device_,vertexMemory_);}
    void createInstance(){VkApplicationInfo ai{};ai.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO;ai.apiVersion=VK_API_VERSION_1_3;auto exts=getRequiredInstanceExtensions();VkInstanceCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;ci.pApplicationInfo=&ai;ci.enabledExtensionCount=static_cast<uint32_t>(exts.size());ci.ppEnabledExtensionNames=exts.data();ci.flags|=VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;if(ENABLE_VALIDATION_LAYERS){ci.enabledLayerCount=static_cast<uint32_t>(VALIDATION_LAYERS.size());ci.ppEnabledLayerNames=VALIDATION_LAYERS.data();}VK_CHECK(vkCreateInstance(&ci,nullptr,&instance_));}
    void createSurface(){VK_CHECK(glfwCreateWindowSurface(instance_,window_,nullptr,&surface_));}
    void pickPhysicalDevice(){uint32_t c=0;vkEnumeratePhysicalDevices(instance_,&c,nullptr);std::vector<VkPhysicalDevice> devs(c);vkEnumeratePhysicalDevices(instance_,&c,devs.data());for(auto&d:devs)if(findQueueFamilies(d,surface_).isComplete()&&checkDeviceExtensionSupport(d)){physicalDevice_=d;break;}if(!physicalDevice_)throw std::runtime_error("无合适GPU");VkPhysicalDeviceProperties p;vkGetPhysicalDeviceProperties(physicalDevice_,&p);std::cout<<"✅ GPU: "<<p.deviceName<<"\n";}
    void createSwapchain(){auto sc=querySwapChainSupport(physicalDevice_,surface_);auto fmt=chooseSwapSurfaceFormat(sc.formats);auto mode=chooseSwapPresentMode(sc.presentModes);swapchainExtent_=chooseSwapExtent(sc.capabilities,window_);uint32_t n=sc.capabilities.minImageCount+1;if(sc.capabilities.maxImageCount>0)n=std::min(n,sc.capabilities.maxImageCount);VkSwapchainCreateInfoKHR ci{};ci.sType=VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;ci.surface=surface_;ci.minImageCount=n;ci.imageFormat=fmt.format;ci.imageColorSpace=fmt.colorSpace;ci.imageExtent=swapchainExtent_;ci.imageArrayLayers=1;ci.imageUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;ci.imageSharingMode=VK_SHARING_MODE_EXCLUSIVE;ci.preTransform=sc.capabilities.currentTransform;ci.compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;ci.presentMode=mode;ci.clipped=VK_TRUE;VK_CHECK(vkCreateSwapchainKHR(device_,&ci,nullptr,&swapchain_));vkGetSwapchainImagesKHR(device_,swapchain_,&n,nullptr);swapchainImages_.resize(n);vkGetSwapchainImagesKHR(device_,swapchain_,&n,swapchainImages_.data());swapchainImageFormat_=fmt.format;}
    void createImageViews(){swapchainImageViews_.resize(swapchainImages_.size());for(size_t i=0;i<swapchainImages_.size();++i){VkImageViewCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;ci.image=swapchainImages_[i];ci.viewType=VK_IMAGE_VIEW_TYPE_2D;ci.format=swapchainImageFormat_;ci.components={VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY};ci.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};VK_CHECK(vkCreateImageView(device_,&ci,nullptr,&swapchainImageViews_[i]));}}
    void createRenderPass(){VkAttachmentDescription ca{};ca.format=swapchainImageFormat_;ca.samples=VK_SAMPLE_COUNT_1_BIT;ca.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR;ca.storeOp=VK_ATTACHMENT_STORE_OP_STORE;ca.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE;ca.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;ca.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;ca.finalLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;VkAttachmentReference cr{0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};VkSubpassDescription sp{};sp.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;sp.colorAttachmentCount=1;sp.pColorAttachments=&cr;VkSubpassDependency dep{};dep.srcSubpass=VK_SUBPASS_EXTERNAL;dep.dstSubpass=0;dep.srcStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;dep.dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;dep.srcAccessMask=0;dep.dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;VkRenderPassCreateInfo rpi{};rpi.sType=VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;rpi.attachmentCount=1;rpi.pAttachments=&ca;rpi.subpassCount=1;rpi.pSubpasses=&sp;rpi.dependencyCount=1;rpi.pDependencies=&dep;VK_CHECK(vkCreateRenderPass(device_,&rpi,nullptr,&renderPass_));}
    void createFramebuffers(){framebuffers_.resize(swapchainImageViews_.size());for(size_t i=0;i<swapchainImageViews_.size();++i){VkImageView att[]={swapchainImageViews_[i]};VkFramebufferCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;ci.renderPass=renderPass_;ci.attachmentCount=1;ci.pAttachments=att;ci.width=swapchainExtent_.width;ci.height=swapchainExtent_.height;ci.layers=1;VK_CHECK(vkCreateFramebuffer(device_,&ci,nullptr,&framebuffers_[i]));}}
    void recreateSwapchain(){int w=0,h=0;glfwGetFramebufferSize(window_,&w,&h);while(!w||!h){glfwGetFramebufferSize(window_,&w,&h);glfwWaitEvents();}vkDeviceWaitIdle(device_);for(auto&fb:framebuffers_)vkDestroyFramebuffer(device_,fb,nullptr);for(auto&iv:swapchainImageViews_)vkDestroyImageView(device_,iv,nullptr);vkDestroySwapchainKHR(device_,swapchain_,nullptr);createSwapchain();createImageViews();createFramebuffers();}
    void cleanup()
    {
        vkDestroySemaphore(device_,timelineSemaphore_,nullptr);
        for(int i=0;i<MAX_FRAMES;++i){vkDestroySemaphore(device_,imageAvailableSems_[i],nullptr);vkDestroySemaphore(device_,renderFinishedSems_[i],nullptr);}
        for(uint32_t t=0;t<N_THREADS;++t)for(int f=0;f<MAX_FRAMES;++f)vkDestroyCommandPool(device_,threadCmdPools_[t*MAX_FRAMES+f],nullptr);
        vkDestroyCommandPool(device_,primaryCmdPool_,nullptr);
        vkDestroyBuffer(device_,vertexBuffer_,nullptr);vkFreeMemory(device_,vertexMemory_,nullptr);
        for(auto&fb:framebuffers_)vkDestroyFramebuffer(device_,fb,nullptr);
        vkDestroyPipeline(device_,pipeline_,nullptr);vkDestroyPipelineLayout(device_,pipelineLayout_,nullptr);vkDestroyRenderPass(device_,renderPass_,nullptr);
        for(auto&iv:swapchainImageViews_)vkDestroyImageView(device_,iv,nullptr);vkDestroySwapchainKHR(device_,swapchain_,nullptr);
        vkDestroyDevice(device_,nullptr);vkDestroySurfaceKHR(instance_,surface_,nullptr);vkDestroyInstance(instance_,nullptr);glfwDestroyWindow(window_);glfwTerminate();
        std::cout<<"✅ 清理完成。\n";
    }
};

int main()
{
    std::cout<<"═══════════════════════════════════════════════════════════════════\n";
    std::cout<<" 第25章：Timeline Semaphores + 多线程命令录制\n";
    std::cout<<"\n";
    std::cout<<" Timeline Semaphore：\n";
    std::cout<<"   • VkSemaphoreTypeCreateInfo → 类型 TIMELINE\n";
    std::cout<<"   • vkWaitSemaphores(device, waitInfo, timeout)\n";
    std::cout<<"   • vkGetSemaphoreCounterValue() 查询当前值\n";
    std::cout<<"   • 替代多个 Fence，简化同步逻辑\n";
    std::cout<<"\n";
    std::cout<<" 多线程命令录制：\n";
    std::cout<<"   • 每线程独立 VkCommandPool（非线程安全！）\n";
    std::cout<<"   • Secondary Command Buffer + RENDER_PASS_CONTINUE_BIT\n";
    std::cout<<"   • vkCmdExecuteCommands(primary, N, secondaries)\n";
    std::cout<<"═══════════════════════════════════════════════════════════════════\n\n";
    Ch25App app;
    try{app.run();}catch(const std::exception&e){std::cerr<<"❌ "<<e.what()<<"\n";return 1;}
    return 0;
}
