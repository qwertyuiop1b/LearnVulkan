/**
 * @file ch26_query_pool.cpp
 * @brief 第26章：Query Pool — GPU 性能分析（时间戳 / 遮挡查询 / 管线统计）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【Query Pool 概述】
 *
 *  VkQueryPool 允许 GPU 在执行命令时记录各种度量值，
 *  CPU 通过 vkGetQueryPoolResults 读取结果。
 *
 * 【三种主要查询类型】
 *
 *  ┌────────────────────────────────────────────────────────────────────┐
 *  │ 1. TIMESTAMP（时间戳查询）                                          │
 *  │    - vkCmdWriteTimestamp(stage) → 记录 GPU 时钟                    │
 *  │    - 两个时间戳相减 × timestampPeriod(ns) = GPU 执行时间            │
 *  │    - 用途：精确测量渲染各阶段的 GPU 耗时                            │
 *  │                                                                    │
 *  │ 2. OCCLUSION（遮挡查询）                                            │
 *  │    - vkCmdBeginQuery / vkCmdEndQuery 包裹一组绘制命令               │
 *  │    - 结果 = 通过深度/模板测试的片段数量                              │
 *  │    - 用途：软件遮挡剔除、LOD 决策                                    │
 *  │                                                                    │
 *  │ 3. PIPELINE_STATISTICS（管线统计）                                  │
 *  │    - 收集图形管线各阶段的调用次数                                    │
 *  │    - 顶点着色器调用次数、图元数量、片段数量等                         │
 *  │    - 用途：性能瓶颈分析（CPU-bound vs GPU-bound？哪个着色器最贵？）   │
 *  └────────────────────────────────────────────────────────────────────┘
 *
 * 【timestampPeriod】
 *
 *  不同 GPU 的时钟精度不同，通过：
 *    VkPhysicalDeviceLimits.timestampPeriod
 *  获取每个时钟周期对应多少纳秒（通常 1.0ns ~ 40ns）
 *
 * 【注意事项】
 *
 *  - Query 结果在 GPU 执行到该点后才可用
 *  - 需要 vkCmdResetQueryPool 在每帧开始重置
 *  - vkGetQueryPoolResults 的 VK_QUERY_RESULT_WAIT_BIT：等待结果可用
 *  - Pipeline Statistics 查询需要启用 pipelineStatisticsQuery 特性
 * ═══════════════════════════════════════════════════════════════════════════
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan_tutorial/utils.hpp>

#include <array>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

constexpr uint32_t WIDTH  = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int      MAX_FRAMES = 2;

// ─── Query Pool 配置 ──────────────────────────────────────────────────────────

constexpr uint32_t TS_COUNT    = 8;   // 时间戳数量（4对：帧开始/结束，渲染开始/结束等）
constexpr uint32_t OCC_COUNT   = 4;   // 遮挡查询数量（测试4个物体可见性）
// 管线统计标志：收集哪些统计
constexpr VkQueryPipelineStatisticFlags STAT_FLAGS =
    VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT  |   // 顶点着色器调用次数
    VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT|   // 片段着色器调用次数
    VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT  |   // 输入图元数
    VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT;           // 裁剪后图元数
constexpr uint32_t STAT_COUNT = 4;    // 与标志位数量一致

struct TsVertex { float pos[3]; float col[3]; };

static const std::vector<TsVertex> QUADS = {
    // 4 个不同大小的四边形（用于遮挡测试）
    {{-0.9f,-0.9f,0},{1,0.2f,0.2f}},{{-0.1f,-0.9f,0},{1,0.2f,0.2f}},
    {{-0.1f,-0.1f,0},{1,0.2f,0.2f}},{{-0.9f,-0.9f,0},{1,0.2f,0.2f}},
    {{-0.1f,-0.1f,0},{1,0.2f,0.2f}},{{-0.9f,-0.1f,0},{1,0.2f,0.2f}},

    {{ 0.1f,-0.9f,0},{0.2f,1,0.2f}},{{ 0.9f,-0.9f,0},{0.2f,1,0.2f}},
    {{ 0.9f,-0.1f,0},{0.2f,1,0.2f}},{{ 0.1f,-0.9f,0},{0.2f,1,0.2f}},
    {{ 0.9f,-0.1f,0},{0.2f,1,0.2f}},{{ 0.1f,-0.1f,0},{0.2f,1,0.2f}},

    {{-0.9f, 0.1f,0},{0.2f,0.2f,1}},{{-0.1f, 0.1f,0},{0.2f,0.2f,1}},
    {{-0.1f, 0.9f,0},{0.2f,0.2f,1}},{{-0.9f, 0.1f,0},{0.2f,0.2f,1}},
    {{-0.1f, 0.9f,0},{0.2f,0.2f,1}},{{-0.9f, 0.9f,0},{0.2f,0.2f,1}},

    {{ 0.1f, 0.1f,0},{1,1,0.2f}},{{ 0.9f, 0.1f,0},{1,1,0.2f}},
    {{ 0.9f, 0.9f,0},{1,1,0.2f}},{{ 0.1f, 0.1f,0},{1,1,0.2f}},
    {{ 0.9f, 0.9f,0},{1,1,0.2f}},{{ 0.1f, 0.9f,0},{1,1,0.2f}},
};

class Ch26App {
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
    VkCommandPool    commandPool_    = VK_NULL_HANDLE;
    VkBuffer         vertexBuffer_   = VK_NULL_HANDLE;
    VkDeviceMemory   vertexMemory_   = VK_NULL_HANDLE;

    // ─── Query Pools ─────────────────────────────────────────────────────
    VkQueryPool timestampPool_  = VK_NULL_HANDLE;   ///< 时间戳查询池
    VkQueryPool occlusionPool_  = VK_NULL_HANDLE;   ///< 遮挡查询池
    VkQueryPool statisticsPool_ = VK_NULL_HANDLE;   ///< 管线统计池

    float timestampPeriodNs_ = 1.0f;  ///< 每个时钟周期的纳秒数

    std::vector<VkImage>         swapchainImages_;
    std::vector<VkImageView>     swapchainImageViews_;
    std::vector<VkFramebuffer>   framebuffers_;
    std::vector<VkCommandBuffer> commandBuffers_;
    VkFormat                     swapchainImageFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D                   swapchainExtent_{};
    QueueFamilyIndices           queueIndices_;

    std::vector<VkSemaphore> imageAvailableSems_;
    std::vector<VkSemaphore> renderFinishedSems_;
    std::vector<VkFence>     inFlightFences_;
    uint32_t currentFrame_ = 0;
    bool     resized_      = false;

    uint64_t frameCount_ = 0;

    void initWindow()
    {
        glfwInit(); glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT,
            "Ch26 - Query Pool（时间戳/遮挡/管线统计）", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_,
            [](GLFWwindow*w,int,int){
                reinterpret_cast<Ch26App*>(glfwGetWindowUserPointer(w))->resized_=true;
            });
    }

    void initVulkan()
    {
        createInstance(); createSurface(); pickPhysicalDevice();
        createLogicalDeviceWithStats();  // 需要启用 pipelineStatisticsQuery 特性
        createSwapchain(); createImageViews(); createRenderPass();
        createGraphicsPipeline();
        createFramebuffers(); createCommandPool();
        createQueryPools();     // ← 创建 3 种查询池
        createVertexBuffer();
        createCommandBuffers(); createSyncObjects();
        std::cout << "\n✅ Query Pool 演示初始化完成！\n";
        std::cout << "⏱️  Timestamp 精度：" << timestampPeriodNs_ << " ns/tick\n";
        std::cout << "📊 管线统计：顶点调用/片段调用/输入图元/裁剪图元\n";
        std::cout << "👁️  遮挡查询：4 个四边形（检测可见片段数）\n\n";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 创建三种 Query Pool
    // ═══════════════════════════════════════════════════════════════════════

    void createQueryPools()
    {
        // ① Timestamp Query Pool
        {
            VkQueryPoolCreateInfo ci{};
            ci.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            ci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
            ci.queryCount = TS_COUNT * MAX_FRAMES;   // 每帧 TS_COUNT 个时间戳
            VK_CHECK(vkCreateQueryPool(device_, &ci, nullptr, &timestampPool_));
            std::cout << "✅ Timestamp Pool：" << TS_COUNT * MAX_FRAMES << " 个时间戳槽\n";
        }

        // ② Occlusion Query Pool
        {
            VkQueryPoolCreateInfo ci{};
            ci.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            ci.queryType  = VK_QUERY_TYPE_OCCLUSION;
            ci.queryCount = OCC_COUNT * MAX_FRAMES;
            VK_CHECK(vkCreateQueryPool(device_, &ci, nullptr, &occlusionPool_));
            std::cout << "✅ Occlusion Pool：" << OCC_COUNT * MAX_FRAMES << " 个遮挡查询槽\n";
        }

        // ③ Pipeline Statistics Query Pool
        {
            VkQueryPoolCreateInfo ci{};
            ci.sType                    = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            ci.queryType                = VK_QUERY_TYPE_PIPELINE_STATISTICS;
            ci.queryCount               = MAX_FRAMES;   // 每帧一个统计查询
            ci.pipelineStatistics       = STAT_FLAGS;   // 指定收集哪些统计
            VK_CHECK(vkCreateQueryPool(device_, &ci, nullptr, &statisticsPool_));
            std::cout << "✅ Statistics Pool：收集 " << STAT_COUNT << " 项管线统计\n";
        }

        // 查询 timestamp 时钟精度
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice_, &props);
        timestampPeriodNs_ = props.limits.timestampPeriod;
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex)
    {
        const uint32_t f = currentFrame_;   // 当前帧索引
        const uint32_t tsBase  = f * TS_COUNT;
        const uint32_t occBase = f * OCC_COUNT;

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

        // ── 重置查询（每帧开始必须重置！否则上帧结果仍然存在）─────────────
        // vkCmdResetQueryPool 在命令缓冲中重置（不同于 vkResetQueryPool）
        vkCmdResetQueryPool(cmd, timestampPool_,  tsBase,  TS_COUNT);
        vkCmdResetQueryPool(cmd, occlusionPool_,  occBase, OCC_COUNT);
        vkCmdResetQueryPool(cmd, statisticsPool_, f, 1);

        // ── 时间戳 0：帧开始（TOP_OF_PIPE = 管线最早阶段）────────────────
        vkCmdWriteTimestamp(cmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            timestampPool_, tsBase + 0);

        VkClearValue clear{};
        clear.color.float32[0]=0.02f;clear.color.float32[2]=0.05f;clear.color.float32[3]=1.0f;
        VkRenderPassBeginInfo rp{};
        rp.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp.renderPass=renderPass_;rp.framebuffer=framebuffers_[imageIndex];
        rp.renderArea={{0,0},swapchainExtent_};
        rp.clearValueCount=1;rp.pClearValues=&clear;
        vkCmdBeginRenderPass(cmd,&rp,VK_SUBPASS_CONTENTS_INLINE);

        // ── 时间戳 1：RenderPass 开始 ─────────────────────────────────────
        vkCmdWriteTimestamp(cmd,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            timestampPool_, tsBase + 1);

        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline_);
        VkViewport vp{0,0,(float)swapchainExtent_.width,(float)swapchainExtent_.height,0,1};
        vkCmdSetViewport(cmd,0,1,&vp);
        VkRect2D sc{{0,0},swapchainExtent_};vkCmdSetScissor(cmd,0,1,&sc);
        VkBuffer vb[]={vertexBuffer_};VkDeviceSize off[]={0};
        vkCmdBindVertexBuffers(cmd,0,1,vb,off);

        // ── 开始管线统计查询 ─────────────────────────────────────────────
        // 收集整个帧的管线统计（顶点/片段着色器调用次数等）
        vkCmdBeginQuery(cmd, statisticsPool_, f, 0);

        // ── 绘制 4 个四边形，每个各自有遮挡查询 ──────────────────────────
        for (uint32_t q = 0; q < OCC_COUNT; ++q) {
            // 开始遮挡查询（将计数通过深度测试的片段数）
            vkCmdBeginQuery(cmd, occlusionPool_, occBase + q, 0);

            vkCmdDraw(cmd, 6, 1, q * 6, 0);   // 每个四边形 6 个顶点

            // 结束遮挡查询
            vkCmdEndQuery(cmd, occlusionPool_, occBase + q);
        }

        // ── 结束管线统计查询 ─────────────────────────────────────────────
        vkCmdEndQuery(cmd, statisticsPool_, f);

        // ── 时间戳 2：绘制结束（FRAGMENT_SHADER_BIT = 片段着色器完成后）──
        vkCmdWriteTimestamp(cmd,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            timestampPool_, tsBase + 2);

        vkCmdEndRenderPass(cmd);

        // ── 时间戳 3：RenderPass 结束 ─────────────────────────────────────
        vkCmdWriteTimestamp(cmd,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            timestampPool_, tsBase + 3);

        VK_CHECK(vkEndCommandBuffer(cmd));
    }

    void readAndPrintQueryResults(uint32_t frame)
    {
        if (frame < 2) return;   // 等第一帧结果稳定
        const uint32_t f       = frame % MAX_FRAMES;
        const uint32_t tsBase  = f * TS_COUNT;
        const uint32_t occBase = f * OCC_COUNT;

        // ── 读取时间戳 ────────────────────────────────────────────────────
        std::vector<uint64_t> timestamps(TS_COUNT, 0);
        vkGetQueryPoolResults(
            device_, timestampPool_,
            tsBase, TS_COUNT,            // 从 tsBase 开始，读 TS_COUNT 个
            TS_COUNT * sizeof(uint64_t), // 结果缓冲大小
            timestamps.data(),
            sizeof(uint64_t),            // 步长（每个结果占 8 字节）
            VK_QUERY_RESULT_64_BIT |     // 64-bit 精度
            VK_QUERY_RESULT_WAIT_BIT);   // 等待结果可用

        // 计算各阶段耗时
        double renderPassMs = (timestamps[2] - timestamps[1]) * timestampPeriodNs_ / 1e6;
        double totalFrameMs = (timestamps[3] - timestamps[0]) * timestampPeriodNs_ / 1e6;

        // ── 读取遮挡查询 ──────────────────────────────────────────────────
        std::vector<uint64_t> occResults(OCC_COUNT, 0);
        vkGetQueryPoolResults(
            device_, occlusionPool_,
            occBase, OCC_COUNT,
            OCC_COUNT * sizeof(uint64_t),
            occResults.data(),
            sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

        // ── 读取管线统计 ──────────────────────────────────────────────────
        std::vector<uint64_t> stats(STAT_COUNT, 0);
        vkGetQueryPoolResults(
            device_, statisticsPool_,
            f, 1,
            STAT_COUNT * sizeof(uint64_t),
            stats.data(),
            sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

        // ── 打印结果 ──────────────────────────────────────────────────────
        std::cout << "\n╔══════════════════════════════════════════════════╗\n";
        std::cout << "║  GPU Performance - Frame " << std::setw(8) << frame << "          ║\n";
        std::cout << "╠══════════════════════════════════════════════════╣\n";

        std::cout << "║  ⏱️  RenderPass 时间 : "
                  << std::fixed << std::setprecision(3) << renderPassMs << " ms        ║\n";
        std::cout << "║  ⏱️  总帧 GPU 时间  : "
                  << totalFrameMs << " ms        ║\n";

        std::cout << "╠══════════════════════════════════════════════════╣\n";
        std::cout << "║  👁️  遮挡查询（通过测试的片段数）：               ║\n";
        const char* colors[] = {"🔴 红色", "🟢 绿色", "🔵 蓝色", "🟡 黄色"};
        for (uint32_t q = 0; q < OCC_COUNT; ++q) {
            std::cout << "║    " << colors[q] << "四边形: "
                      << std::setw(12) << occResults[q] << " 个片段    ║\n";
        }

        std::cout << "╠══════════════════════════════════════════════════╣\n";
        std::cout << "║  📊 管线统计：                                    ║\n";
        std::cout << "║    顶点着色器调用  : " << std::setw(12) << stats[0] << "           ║\n";
        std::cout << "║    片段着色器调用  : " << std::setw(12) << stats[1] << "           ║\n";
        std::cout << "║    输入图元数      : " << std::setw(12) << stats[2] << "           ║\n";
        std::cout << "║    裁剪后图元数    : " << std::setw(12) << stats[3] << "           ║\n";
        std::cout << "╚══════════════════════════════════════════════════╝\n";
    }

    void drawFrame()
    {
        vkWaitForFences(device_,1,&inFlightFences_[currentFrame_],VK_TRUE,UINT64_MAX);
        uint32_t imgIdx=0;
        VkResult r=vkAcquireNextImageKHR(device_,swapchain_,UINT64_MAX,
            imageAvailableSems_[currentFrame_],VK_NULL_HANDLE,&imgIdx);
        if(r==VK_ERROR_OUT_OF_DATE_KHR){recreateSwapchain();return;}
        vkResetFences(device_,1,&inFlightFences_[currentFrame_]);
        vkResetCommandBuffer(commandBuffers_[currentFrame_],0);
        recordCommandBuffer(commandBuffers_[currentFrame_],imgIdx);

        VkSemaphore ws[]={imageAvailableSems_[currentFrame_]};
        VkPipelineStageFlags wst[]={VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        VkSemaphore ss[]={renderFinishedSems_[currentFrame_]};
        VkSubmitInfo si{};si.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.waitSemaphoreCount=1;si.pWaitSemaphores=ws;si.pWaitDstStageMask=wst;
        si.commandBufferCount=1;si.pCommandBuffers=&commandBuffers_[currentFrame_];
        si.signalSemaphoreCount=1;si.pSignalSemaphores=ss;
        VK_CHECK(vkQueueSubmit(graphicsQueue_,1,&si,inFlightFences_[currentFrame_]));

        VkSwapchainKHR scs[]={swapchain_};
        VkPresentInfoKHR pi{};pi.sType=VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        pi.waitSemaphoreCount=1;pi.pWaitSemaphores=ss;pi.swapchainCount=1;
        pi.pSwapchains=scs;pi.pImageIndices=&imgIdx;
        r=vkQueuePresentKHR(presentQueue_,&pi);
        if(r==VK_ERROR_OUT_OF_DATE_KHR||r==VK_SUBOPTIMAL_KHR||resized_)
            {resized_=false;recreateSwapchain();}
        currentFrame_=(currentFrame_+1)%MAX_FRAMES;
        ++frameCount_;
    }

    void mainLoop()
    {
        std::cout<<"📊 每 60 帧打印一次 GPU 性能数据（ESC 退出）...\n";
        while(!glfwWindowShouldClose(window_)){
            glfwPollEvents();
            if(glfwGetKey(window_,GLFW_KEY_ESCAPE)==GLFW_PRESS)
                glfwSetWindowShouldClose(window_,GLFW_TRUE);
            drawFrame();
            if(frameCount_ % 60 == 0)
                readAndPrintQueryResults(static_cast<uint32_t>(frameCount_));
        }
        vkDeviceWaitIdle(device_);
    }

    // ─── 辅助代码 ─────────────────────────────────────────────────────────────

    void createLogicalDeviceWithStats()
    {
        queueIndices_=findQueueFamilies(physicalDevice_,surface_);
        std::set<uint32_t> fams={queueIndices_.graphicsFamily.value(),queueIndices_.presentFamily.value()};
        const float pri=1.0f;std::vector<VkDeviceQueueCreateInfo> qcis;
        for(uint32_t f:fams){VkDeviceQueueCreateInfo q{};q.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;q.queueFamilyIndex=f;q.queueCount=1;q.pQueuePriorities=&pri;qcis.push_back(q);}
        VkPhysicalDeviceFeatures feat{};
        feat.samplerAnisotropy=VK_TRUE;
        feat.pipelineStatisticsQuery=VK_TRUE;  // ← 必须启用此特性！
        VkDeviceCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        ci.queueCreateInfoCount=static_cast<uint32_t>(qcis.size());ci.pQueueCreateInfos=qcis.data();
        ci.pEnabledFeatures=&feat;
        ci.enabledExtensionCount=static_cast<uint32_t>(DEVICE_EXTENSIONS.size());
        ci.ppEnabledExtensionNames=DEVICE_EXTENSIONS.data();
        if(ENABLE_VALIDATION_LAYERS){ci.enabledLayerCount=static_cast<uint32_t>(VALIDATION_LAYERS.size());ci.ppEnabledLayerNames=VALIDATION_LAYERS.data();}
        VK_CHECK(vkCreateDevice(physicalDevice_,&ci,nullptr,&device_));
        vkGetDeviceQueue(device_,queueIndices_.graphicsFamily.value(),0,&graphicsQueue_);
        vkGetDeviceQueue(device_,queueIndices_.presentFamily.value(),0,&presentQueue_);
        std::cout<<"✅ 逻辑设备已创建（pipelineStatisticsQuery=true）\n";
    }
    uint32_t findMemoryType(uint32_t f,VkMemoryPropertyFlags p){VkPhysicalDeviceMemoryProperties mp;vkGetPhysicalDeviceMemoryProperties(physicalDevice_,&mp);for(uint32_t i=0;i<mp.memoryTypeCount;++i)if((f&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&p)==p)return i;throw std::runtime_error("找不到内存类型");}
    void createBuffer(VkDeviceSize sz,VkBufferUsageFlags u,VkMemoryPropertyFlags p,VkBuffer&b,VkDeviceMemory&m){VkBufferCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;ci.size=sz;ci.usage=u;ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;VK_CHECK(vkCreateBuffer(device_,&ci,nullptr,&b));VkMemoryRequirements mr;vkGetBufferMemoryRequirements(device_,b,&mr);VkMemoryAllocateInfo ai{};ai.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;ai.allocationSize=mr.size;ai.memoryTypeIndex=findMemoryType(mr.memoryTypeBits,p);VK_CHECK(vkAllocateMemory(device_,&ai,nullptr,&m));VK_CHECK(vkBindBufferMemory(device_,b,m,0));}
    void createVertexBuffer(){VkDeviceSize sz=sizeof(QUADS[0])*QUADS.size();createBuffer(sz,VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,vertexBuffer_,vertexMemory_);void*d=nullptr;vkMapMemory(device_,vertexMemory_,0,sz,0,&d);std::memcpy(d,QUADS.data(),(size_t)sz);vkUnmapMemory(device_,vertexMemory_);}
    void createGraphicsPipeline()
    {
        VkShaderModule vert=createShaderModuleFromFile(device_,"uniform3d.vert.spv");
        VkShaderModule frag=createShaderModuleFromFile(device_,"triangle.frag.spv");
        VkPipelineShaderStageCreateInfo stages[2]{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_VERTEX_BIT,vert,"main",nullptr},{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_FRAGMENT_BIT,frag,"main",nullptr}};
        VkVertexInputBindingDescription bind{0,sizeof(TsVertex),VK_VERTEX_INPUT_RATE_VERTEX};
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
        VkPipelineLayoutCreateInfo pli{};pli.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        VK_CHECK(vkCreatePipelineLayout(device_,&pli,nullptr,&pipelineLayout_));
        VkGraphicsPipelineCreateInfo pi{};pi.sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;pi.stageCount=2;pi.pStages=stages;pi.pVertexInputState=&vi;pi.pInputAssemblyState=&ia;pi.pViewportState=&vs;pi.pRasterizationState=&rs;pi.pMultisampleState=&ms;pi.pColorBlendState=&cb;pi.pDynamicState=&dynS;pi.layout=pipelineLayout_;pi.renderPass=renderPass_;
        VK_CHECK(vkCreateGraphicsPipelines(device_,VK_NULL_HANDLE,1,&pi,nullptr,&pipeline_));
        vkDestroyShaderModule(device_,frag,nullptr);vkDestroyShaderModule(device_,vert,nullptr);
    }
    void createInstance(){VkApplicationInfo ai{};ai.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO;ai.apiVersion=VK_API_VERSION_1_3;auto exts=getRequiredInstanceExtensions();VkInstanceCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;ci.pApplicationInfo=&ai;ci.enabledExtensionCount=static_cast<uint32_t>(exts.size());ci.ppEnabledExtensionNames=exts.data();ci.flags|=VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;if(ENABLE_VALIDATION_LAYERS){ci.enabledLayerCount=static_cast<uint32_t>(VALIDATION_LAYERS.size());ci.ppEnabledLayerNames=VALIDATION_LAYERS.data();}VK_CHECK(vkCreateInstance(&ci,nullptr,&instance_));}
    void createSurface(){VK_CHECK(glfwCreateWindowSurface(instance_,window_,nullptr,&surface_));}
    void pickPhysicalDevice(){uint32_t c=0;vkEnumeratePhysicalDevices(instance_,&c,nullptr);std::vector<VkPhysicalDevice> devs(c);vkEnumeratePhysicalDevices(instance_,&c,devs.data());for(auto&d:devs)if(findQueueFamilies(d,surface_).isComplete()&&checkDeviceExtensionSupport(d)){physicalDevice_=d;break;}if(!physicalDevice_)throw std::runtime_error("无合适GPU");VkPhysicalDeviceProperties p;vkGetPhysicalDeviceProperties(physicalDevice_,&p);std::cout<<"✅ GPU: "<<p.deviceName<<"\n";}
    void createSwapchain(){auto sc=querySwapChainSupport(physicalDevice_,surface_);auto fmt=chooseSwapSurfaceFormat(sc.formats);auto mode=chooseSwapPresentMode(sc.presentModes);swapchainExtent_=chooseSwapExtent(sc.capabilities,window_);uint32_t n=sc.capabilities.minImageCount+1;if(sc.capabilities.maxImageCount>0)n=std::min(n,sc.capabilities.maxImageCount);VkSwapchainCreateInfoKHR ci{};ci.sType=VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;ci.surface=surface_;ci.minImageCount=n;ci.imageFormat=fmt.format;ci.imageColorSpace=fmt.colorSpace;ci.imageExtent=swapchainExtent_;ci.imageArrayLayers=1;ci.imageUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;ci.imageSharingMode=VK_SHARING_MODE_EXCLUSIVE;ci.preTransform=sc.capabilities.currentTransform;ci.compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;ci.presentMode=mode;ci.clipped=VK_TRUE;VK_CHECK(vkCreateSwapchainKHR(device_,&ci,nullptr,&swapchain_));vkGetSwapchainImagesKHR(device_,swapchain_,&n,nullptr);swapchainImages_.resize(n);vkGetSwapchainImagesKHR(device_,swapchain_,&n,swapchainImages_.data());swapchainImageFormat_=fmt.format;}
    void createImageViews(){swapchainImageViews_.resize(swapchainImages_.size());for(size_t i=0;i<swapchainImages_.size();++i){VkImageViewCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;ci.image=swapchainImages_[i];ci.viewType=VK_IMAGE_VIEW_TYPE_2D;ci.format=swapchainImageFormat_;ci.components={VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY};ci.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};VK_CHECK(vkCreateImageView(device_,&ci,nullptr,&swapchainImageViews_[i]));}}
    void createRenderPass(){VkAttachmentDescription ca{};ca.format=swapchainImageFormat_;ca.samples=VK_SAMPLE_COUNT_1_BIT;ca.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR;ca.storeOp=VK_ATTACHMENT_STORE_OP_STORE;ca.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE;ca.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;ca.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;ca.finalLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;VkAttachmentReference cr{0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};VkSubpassDescription sp{};sp.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;sp.colorAttachmentCount=1;sp.pColorAttachments=&cr;VkSubpassDependency dep{};dep.srcSubpass=VK_SUBPASS_EXTERNAL;dep.dstSubpass=0;dep.srcStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;dep.dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;dep.srcAccessMask=0;dep.dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;VkRenderPassCreateInfo rpi{};rpi.sType=VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;rpi.attachmentCount=1;rpi.pAttachments=&ca;rpi.subpassCount=1;rpi.pSubpasses=&sp;rpi.dependencyCount=1;rpi.pDependencies=&dep;VK_CHECK(vkCreateRenderPass(device_,&rpi,nullptr,&renderPass_));}
    void createFramebuffers(){framebuffers_.resize(swapchainImageViews_.size());for(size_t i=0;i<swapchainImageViews_.size();++i){VkImageView att[]={swapchainImageViews_[i]};VkFramebufferCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;ci.renderPass=renderPass_;ci.attachmentCount=1;ci.pAttachments=att;ci.width=swapchainExtent_.width;ci.height=swapchainExtent_.height;ci.layers=1;VK_CHECK(vkCreateFramebuffer(device_,&ci,nullptr,&framebuffers_[i]));}}
    void createCommandPool(){VkCommandPoolCreateInfo ci{};ci.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;ci.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;ci.queueFamilyIndex=queueIndices_.graphicsFamily.value();VK_CHECK(vkCreateCommandPool(device_,&ci,nullptr,&commandPool_));}
    void createCommandBuffers(){commandBuffers_.resize(MAX_FRAMES);VkCommandBufferAllocateInfo ai{};ai.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;ai.commandPool=commandPool_;ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ai.commandBufferCount=static_cast<uint32_t>(commandBuffers_.size());VK_CHECK(vkAllocateCommandBuffers(device_,&ai,commandBuffers_.data()));}
    void createSyncObjects(){imageAvailableSems_.resize(MAX_FRAMES);renderFinishedSems_.resize(MAX_FRAMES);inFlightFences_.resize(MAX_FRAMES);VkSemaphoreCreateInfo sCI{};sCI.sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;VkFenceCreateInfo fCI{};fCI.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;fCI.flags=VK_FENCE_CREATE_SIGNALED_BIT;for(int i=0;i<MAX_FRAMES;++i){VK_CHECK(vkCreateSemaphore(device_,&sCI,nullptr,&imageAvailableSems_[i]));VK_CHECK(vkCreateSemaphore(device_,&sCI,nullptr,&renderFinishedSems_[i]));VK_CHECK(vkCreateFence(device_,&fCI,nullptr,&inFlightFences_[i]));}}
    void recreateSwapchain(){int w=0,h=0;glfwGetFramebufferSize(window_,&w,&h);while(!w||!h){glfwGetFramebufferSize(window_,&w,&h);glfwWaitEvents();}vkDeviceWaitIdle(device_);for(auto&fb:framebuffers_)vkDestroyFramebuffer(device_,fb,nullptr);for(auto&iv:swapchainImageViews_)vkDestroyImageView(device_,iv,nullptr);vkDestroySwapchainKHR(device_,swapchain_,nullptr);createSwapchain();createImageViews();createFramebuffers();}
    void cleanup()
    {
        vkDestroyQueryPool(device_,timestampPool_,nullptr);
        vkDestroyQueryPool(device_,occlusionPool_,nullptr);
        vkDestroyQueryPool(device_,statisticsPool_,nullptr);
        vkDestroyBuffer(device_,vertexBuffer_,nullptr);vkFreeMemory(device_,vertexMemory_,nullptr);
        for(int i=0;i<MAX_FRAMES;++i){vkDestroySemaphore(device_,imageAvailableSems_[i],nullptr);vkDestroySemaphore(device_,renderFinishedSems_[i],nullptr);vkDestroyFence(device_,inFlightFences_[i],nullptr);}
        vkDestroyCommandPool(device_,commandPool_,nullptr);for(auto&fb:framebuffers_)vkDestroyFramebuffer(device_,fb,nullptr);
        vkDestroyPipeline(device_,pipeline_,nullptr);vkDestroyPipelineLayout(device_,pipelineLayout_,nullptr);vkDestroyRenderPass(device_,renderPass_,nullptr);
        for(auto&iv:swapchainImageViews_)vkDestroyImageView(device_,iv,nullptr);vkDestroySwapchainKHR(device_,swapchain_,nullptr);
        vkDestroyDevice(device_,nullptr);vkDestroySurfaceKHR(instance_,surface_,nullptr);vkDestroyInstance(instance_,nullptr);glfwDestroyWindow(window_);glfwTerminate();
        std::cout<<"✅ 清理完成。\n";
    }
};

int main()
{
    std::cout<<"═══════════════════════════════════════════════════════════════════\n";
    std::cout<<" 第26章：Query Pool — GPU 性能分析\n";
    std::cout<<"\n";
    std::cout<<" 三种查询类型：\n";
    std::cout<<"   • Timestamp：vkCmdWriteTimestamp → GPU 精确计时\n";
    std::cout<<"   • Occlusion：vkCmdBeginQuery(OCCLUSION) → 可见片段计数\n";
    std::cout<<"   • Pipeline Statistics：顶点/片段着色器调用、图元数\n";
    std::cout<<"\n";
    std::cout<<" 关键 API：\n";
    std::cout<<"   vkCreateQueryPool / vkCmdResetQueryPool\n";
    std::cout<<"   vkGetQueryPoolResults(WAIT_BIT | 64_BIT)\n";
    std::cout<<"   pipelineStatisticsQuery = VK_TRUE（设备特性）\n";
    std::cout<<"═══════════════════════════════════════════════════════════════════\n\n";
    Ch26App app;
    try{app.run();}catch(const std::exception&e){std::cerr<<"❌ "<<e.what()<<"\n";return 1;}
    return 0;
}
