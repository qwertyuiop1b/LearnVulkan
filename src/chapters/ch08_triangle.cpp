/**
 * @file ch08_triangle.cpp
 * @brief 第08章：完整渲染循环 ★ 画出彩色三角形 ★
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【渲染循环（Render Loop）】
 *
 * 这是 Vulkan 最复杂的部分：同步机制。
 * Vulkan 将 GPU-CPU 同步完全交给开发者管理。
 *
 * 【每帧的工作】
 *
 *  ① 等待上一帧完成（CPU fence）
 *  ② 从交换链获取下一张可用图像
 *  ③ 录制（或重用）命令缓冲
 *  ④ 提交命令缓冲到图形队列
 *  ⑤ 将渲染好的图像提交到呈现队列显示
 *
 * 【三种同步原语】
 *
 *  ┌─────────────────────────────────────────────────────────┐
 *  │ VkSemaphore（信号量）                                   │
 *  │   GPU-GPU 同步。等待/通知 GPU 上的操作顺序。            │
 *  │   例："图像获取完成" → 才能 → "开始渲染"               │
 *  │                                                         │
 *  │ VkFence（栅栏）                                         │
 *  │   GPU-CPU 同步。CPU 等待 GPU 完成某项工作。             │
 *  │   例：CPU 等待上一帧渲染完成再修改资源                  │
 *  │                                                         │
 *  │ vkDeviceWaitIdle()                                      │
 *  │   等待设备上所有操作完成（清理时使用）                   │
 *  └─────────────────────────────────────────────────────────┘
 *
 * 【双缓冲/多缓冲飞行帧（Frames in Flight）】
 *
 *  每帧渲染时，允许 CPU 提前准备下 N 帧的命令，
 *  而无需等待 GPU 完成当前帧。这提高了 CPU 利用率。
 *
 *  同步结构：
 *    - imageAvailableSemaphore: 图像已获取，可以渲染
 *    - renderFinishedSemaphore: 渲染已完成，可以显示（每个交换链图像一份）
 *    - inFlightFence:           此帧的命令已执行完毕
 * ═══════════════════════════════════════════════════════════════════════════
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan_tutorial/utils.hpp>

#include <chrono>
#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int MAX_FRAMES_IN_FLIGHT = 2; // 最多允许 2 帧同时在飞行

#if defined(CH96_PUSH_CONSTANTS) || defined(CH97_MULTI_DRAW) || defined(CH99_ATTACK_SLASH)
struct DrawPushConstants {
    float time;
    float offsetX;
    float offsetY;
    float scale;
    float hue;
};
#endif

class Ch08App {
  public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop(); // ← 包含真正的渲染
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

    // ─── 同步对象 ────────────────────────────────────────────────────────────
    // imageAvailableSemaphores_ 和 inFlightFences_ 保护“飞行帧”资源。
    // renderFinishedSemaphores_ 会交给 vkQueuePresentKHR 使用，必须按交换链图像分配。
    std::vector<VkSemaphore> imageAvailableSemaphores_;
    std::vector<VkSemaphore> renderFinishedSemaphores_;
    std::vector<VkFence> inFlightFences_;
    uint32_t currentFrame_ = 0;

    bool framebufferResized_ = false;
    float animationTime_ = 0.0f;

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
#ifdef CH96_PUSH_CONSTANTS
        const char* title = "Ch96 - Push Constants: Animated Triangle";
#elif defined(CH97_MULTI_DRAW)
        const char* title = "Ch97 - Per-Draw Push Constants";
#elif defined(CH98_SPECIALIZATION_CONSTANTS)
        const char* title = "Ch98 - Specialization Constants";
#elif defined(CH99_ATTACK_SLASH)
        const char* title = "Ch99 - GPU Attack Slash Effect";
#else
        const char* title = "Ch08 - 彩色三角形 ★";
#endif
        window_ = glfwCreateWindow(WIDTH, HEIGHT, title, nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, framebufferResizeCallback);
    }

    static void framebufferResizeCallback(GLFWwindow* win, int /*w*/, int /*h*/) {
        auto* app = reinterpret_cast<Ch08App*>(glfwGetWindowUserPointer(win));
        app->framebufferResized_ = true;
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
        createFramebuffers();
        createCommandPool();
        createCommandBuffers();
        createSyncObjects(); // ← 新增：创建同步原语
    }

    // ─── 核心新增：创建同步对象 ───────────────────────────────────────────────

    void createSyncObjects() {
        imageAvailableSemaphores_.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences_.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semCI{};
        semCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceCI{};
        fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        // SIGNALED_BIT：初始化为已触发状态
        // 否则第一帧等待 Fence 时会永远阻塞（因为 Fence 从未被 signal 过）
        fenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            VK_CHECK(vkCreateSemaphore(device_, &semCI, nullptr, &imageAvailableSemaphores_[i]));
            VK_CHECK(vkCreateFence(device_, &fenceCI, nullptr, &inFlightFences_[i]));
        }
        createRenderFinishedSemaphores();
        std::cout << "✅ 同步对象已创建（" << MAX_FRAMES_IN_FLIGHT << " 个飞行帧，" << renderFinishedSemaphores_.size()
                  << " 个呈现信号量）\n";
    }

    void createRenderFinishedSemaphores() {
        renderFinishedSemaphores_.resize(swapchainImages_.size());

        VkSemaphoreCreateInfo semCI{};
        semCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        for (auto& semaphore : renderFinishedSemaphores_) {
            VK_CHECK(vkCreateSemaphore(device_, &semCI, nullptr, &semaphore));
        }
    }

    void cleanupRenderFinishedSemaphores() {
        for (auto semaphore : renderFinishedSemaphores_) {
            vkDestroySemaphore(device_, semaphore, nullptr);
        }
        renderFinishedSemaphores_.clear();
    }

    // ─── 核心：渲染一帧 ───────────────────────────────────────────────────────

    void drawFrame() {
        // ① 等待当前帧的 Fence（确保上一轮此帧的 GPU 工作已完成）
        //    UINT64_MAX = 无限等待
        vkWaitForFences(device_, 1, &inFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX);

        // ② 从交换链获取下一张可用图像
        uint32_t imageIndex = 0;
        VkResult result = vkAcquireNextImageKHR(device_,
                                                swapchain_,
                                                UINT64_MAX,                               // 超时时间
                                                imageAvailableSemaphores_[currentFrame_], // 获取完成时 signal 此信号量
                                                VK_NULL_HANDLE,                           // 不使用 Fence
                                                &imageIndex);

        // 交换链过期（窗口大小改变）→ 重建交换链
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapchain();
            return;
        }
        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
            throw std::runtime_error("获取交换链图像失败");

        // ③ 重置 Fence（只有确认要提交工作时才重置，避免死锁）
        vkResetFences(device_, 1, &inFlightFences_[currentFrame_]);

        // ④ 重置并录制命令缓冲
        vkResetCommandBuffer(commandBuffers_[currentFrame_], 0);
        recordCommandBuffer(commandBuffers_[currentFrame_], imageIndex);

        // ⑤ 提交命令缓冲到图形队列
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        // 等待此信号量 signal 后才开始执行（图像获取完成才能开始渲染）
        VkSemaphore waitSemaphores[] = {imageAvailableSemaphores_[currentFrame_]};
        // 在管线的颜色输出阶段等待（更细粒度，允许顶点着色器先跑）
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers_[currentFrame_];

        // 命令执行完成后 signal 此信号量（通知呈现队列可以显示了）。
        // 这个信号量会被 presentation engine 消费，所以按交换链图像索引复用；
        // 再次 acquire 到同一个 imageIndex 时，表示它上一次的 present 已经不再使用该信号量。
        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores_[imageIndex]};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        // inFlightFences_ 在所有提交的命令完成后被 signal（让 CPU 知道此帧完成）
        VK_CHECK(vkQueueSubmit(graphicsQueue_, 1, &submitInfo, inFlightFences_[currentFrame_]));

        // ⑥ 呈现：将渲染好的图像显示到屏幕
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores; // 等待渲染完成

        VkSwapchainKHR swapchains[] = {swapchain_};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapchains;
        presentInfo.pImageIndices = &imageIndex;

        result = vkQueuePresentKHR(presentQueue_, &presentInfo);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized_) {
            framebufferResized_ = false;
            recreateSwapchain();
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("呈现图像失败");
        }

        // 进入下一帧（循环使用 MAX_FRAMES_IN_FLIGHT 个帧资源）
        currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    // ─── 重建交换链（窗口大小改变时） ────────────────────────────────────────

    void recreateSwapchain() {
        // 处理窗口最小化（帧缓冲大小为 0）
        int width = 0, height = 0;
        glfwGetFramebufferSize(window_, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window_, &width, &height);
            glfwWaitEvents();
        }

        // 等待设备空闲后才能重建
        vkDeviceWaitIdle(device_);

        cleanupSwapchain();
        createSwapchain();
        createImageViews();
        createFramebuffers();
        createRenderFinishedSemaphores();
        std::cout << "🔄 交换链已重建（" << width << "x" << height << "）\n";
    }

    void cleanupSwapchain() {
        for (auto& fb : framebuffers_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        for (auto& iv : swapchainImageViews_)
            vkDestroyImageView(device_, iv, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        cleanupRenderFinishedSemaphores();
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

#ifdef CH99_ATTACK_SLASH
        VkClearValue clearColor = {{{0.012f, 0.016f, 0.028f, 1.0f}}};
#else
        VkClearValue clearColor = {{{0.02f, 0.02f, 0.05f, 1.0f}}};
#endif
        VkRenderPassBeginInfo rpBI{};
        rpBI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpBI.renderPass = renderPass_;
        rpBI.framebuffer = framebuffers_[imageIndex];
        rpBI.renderArea = {{0, 0}, swapchainExtent_};
        rpBI.clearValueCount = 1;
        rpBI.pClearValues = &clearColor;

        vkCmdBeginRenderPass(cmd, &rpBI, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

        VkViewport vp{0.0f, 0.0f, (float)swapchainExtent_.width, (float)swapchainExtent_.height, 0.0f, 1.0f};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{{0, 0}, swapchainExtent_};
        vkCmdSetScissor(cmd, 0, 1, &sc);

#ifdef CH96_PUSH_CONSTANTS
        DrawPushConstants push{animationTime_, 0.0f, 0.0f, 0.72f, 0.5f + 0.5f * std::sin(animationTime_)};
        vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(push), &push);
        vkCmdDraw(cmd, 3, 1, 0, 0);
#elif defined(CH97_MULTI_DRAW)
        for (int y = -2; y <= 2; ++y) {
            for (int x = -3; x <= 3; ++x) {
                DrawPushConstants push{animationTime_, float(x) * 0.27f, float(y) * 0.34f, 0.16f,
                                       float((x + 3) + (y + 2) * 7) / 35.0f};
                vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   0, sizeof(push), &push);
                vkCmdDraw(cmd, 3, 1, 0, 0);
            }
        }
#elif defined(CH99_ATTACK_SLASH)
        DrawPushConstants push{animationTime_, 0.0f, 0.0f, float(swapchainExtent_.width),
                               float(swapchainExtent_.height)};
        vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(push), &push);
        // Two procedural 3D ribbons, 64 segments each, 2 triangles per segment.
        vkCmdDraw(cmd, 2 * 64 * 6, 1, 0, 0);
#else
        vkCmdDraw(cmd, 3, 1, 0, 0); // 3 顶点，1 实例
#endif
        vkCmdEndRenderPass(cmd);
        VK_CHECK(vkEndCommandBuffer(cmd));
    }

    void mainLoop() {
        std::cout << "\n🎨 开始渲染！你应该看到一个彩色三角形...\n";
        std::cout << "   按 ESC 或关闭窗口退出\n\n";

        uint64_t frameCount = 0;
        auto startTime = std::chrono::steady_clock::now();

        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();

            animationTime_ = std::chrono::duration<float>(std::chrono::steady_clock::now() - startTime).count();

            if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
                glfwSetWindowShouldClose(window_, GLFW_TRUE);

            drawFrame();
            ++frameCount;

            // 每5秒打印一次帧率
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration<float>(now - startTime).count();
            if (elapsed >= 5.0f) {
                std::cout << "📊 FPS: " << static_cast<uint64_t>(frameCount / elapsed) << "\n";
                frameCount = 0;
                startTime = now;
            }
        }

        // 等待 GPU 完成所有工作后再清理
        vkDeviceWaitIdle(device_);
    }

    // ─── 以下为前几章的初始化代码 ────────────────────────────────────────────

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
        if (ENABLE_VALIDATION_LAYERS && checkValidationLayerSupport()) {
            ci.enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
            ci.ppEnabledLayerNames = VALIDATION_LAYERS.data();
        } else if (ENABLE_VALIDATION_LAYERS) {
            std::cerr << "[Vulkan] validation layer unavailable; continuing without it.\n";
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
        SwapChainSupportDetails sc = querySwapChainSupport(physicalDevice_, surface_);
        VkSurfaceFormatKHR fmt = chooseSwapSurfaceFormat(sc.formats);
        VkPresentModeKHR mode = chooseSwapPresentMode(sc.presentModes);
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
        // triangle.vert: 顶点坐标硬编码在着色器内（无顶点缓冲输入）
        // triangle.frag: 输出光栅化插值得到的颜色
#if defined(CH96_PUSH_CONSTANTS) || defined(CH97_MULTI_DRAW)
        VkShaderModule vert = createShaderModuleFromFile(device_, "push_constants.vert.spv");
        VkShaderModule frag = createShaderModuleFromFile(device_, "push_constants.frag.spv");
#elif defined(CH99_ATTACK_SLASH)
        VkShaderModule vert = createShaderModuleFromFile(device_, "attack_slash.vert.spv");
        VkShaderModule frag = createShaderModuleFromFile(device_, "attack_slash.frag.spv");
#elif defined(CH98_SPECIALIZATION_CONSTANTS)
        VkShaderModule vert = createShaderModuleFromFile(device_, "triangle.vert.spv");
        VkShaderModule frag = createShaderModuleFromFile(device_, "specialization.frag.spv");
#else
        VkShaderModule vert = createShaderModuleFromFile(device_, "triangle.vert.spv");
        VkShaderModule frag = createShaderModuleFromFile(device_, "triangle.frag.spv");
#endif
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
#ifdef CH98_SPECIALIZATION_CONSTANTS
        const uint32_t colorMode = 2;
        const float stripeFrequency = 18.0f;
        VkSpecializationMapEntry specEntries[] = {{0, 0, sizeof(colorMode)}, {1, sizeof(colorMode), sizeof(stripeFrequency)}};
        struct SpecializationData {
            uint32_t mode;
            float frequency;
        } specData{colorMode, stripeFrequency};
        VkSpecializationInfo specInfo{};
        specInfo.mapEntryCount = 2;
        specInfo.pMapEntries = specEntries;
        specInfo.dataSize = sizeof(specData);
        specInfo.pData = &specData;
        stages[1].pSpecializationInfo = &specInfo;
#endif
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
#ifdef CH99_ATTACK_SLASH
        rs.cullMode = VK_CULL_MODE_NONE;
#else
        rs.cullMode = VK_CULL_MODE_BACK_BIT;
#endif
        rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = 0xf;
#ifdef CH99_ATTACK_SLASH
        cba.blendEnable = VK_TRUE;
        cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        cba.colorBlendOp = VK_BLEND_OP_ADD;
        cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        cba.alphaBlendOp = VK_BLEND_OP_ADD;
#endif
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
#if defined(CH96_PUSH_CONSTANTS) || defined(CH97_MULTI_DRAW) || defined(CH99_ATTACK_SLASH)
        VkPushConstantRange pushRange{};
#ifdef CH99_ATTACK_SLASH
        pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
#else
        pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
#endif
        pushRange.offset = 0;
        pushRange.size = sizeof(DrawPushConstants);
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pushRange;
#endif
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
        commandBuffers_.resize(MAX_FRAMES_IN_FLIGHT);
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = commandPool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = static_cast<uint32_t>(commandBuffers_.size());
        VK_CHECK(vkAllocateCommandBuffers(device_, &ai, commandBuffers_.data()));
    }

    void cleanup() {
        cleanupSwapchain();
        vkDestroyPipeline(device_, pipeline_, nullptr);
        vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
        vkDestroyRenderPass(device_, renderPass_, nullptr);
        vkDestroyCommandPool(device_, commandPool_, nullptr);
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            vkDestroySemaphore(device_, imageAvailableSemaphores_[i], nullptr);
            vkDestroyFence(device_, inFlightFences_[i], nullptr);
        }
        vkDestroyDevice(device_, nullptr);
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        vkDestroyInstance(instance_, nullptr);
        glfwDestroyWindow(window_);
        glfwTerminate();
        std::cout << "✅ 清理完成。\n";
    }
};

int main() {
    std::cout << "═══════════════════════════════════════════════════\n";
    std::cout << " 第08章：完整渲染循环 ★ 彩色三角形 ★\n";
    std::cout << "═══════════════════════════════════════════════════\n\n";
    Ch08App app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ 错误：" << e.what() << "\n";
        return 1;
    }
    return 0;
}
