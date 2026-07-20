/**
 * @file vk_renderer_refactored.cpp
 * @brief Vulkan 渲染器 - 重构版（分层封装示例）
 *
 * 这个例子展示如何将 Vulkan 对象按照分层架构进行封装
 * 便于代码复用、维护和扩展
 */

#pragma once
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <vulkan_tutorial/portability.hpp>
#include <vector>
#include <memory>
#include <stdexcept>

constexpr int MAX_FRAMES_IN_FLIGHT = 2;

// ═══════════════════════════════════════════════════════════════════════════
// 层 1：VkContext - Vulkan 基础上下文
// ═══════════════════════════════════════════════════════════════════════════

class VkContext {
  public:
    /**
     * 初始化 Vulkan 上下文
     * 创建：Instance, PhysicalDevice, Device, Queue, CommandPool
     */
    void init(GLFWwindow* window, const std::string& appName = "VkApp") {
        createInstance(appName);
        createSurface(window);
        pickPhysicalDevice();
        createLogicalDevice();
        createCommandPool();
    }

    void cleanup() {
        if (device_) {
            vkDestroyCommandPool(device_, commandPool_, nullptr);
            vkDestroyDevice(device_, nullptr);
        }
        if (instance_) {
            vkDestroySurfaceKHR(instance_, surface_, nullptr);
            vkDestroyInstance(instance_, nullptr);
        }
    }

    // ─── 访问器 ──────────────────────────────────────────────────────────
    VkInstance instance() const {
        return instance_;
    }
    VkPhysicalDevice physicalDevice() const {
        return physicalDevice_;
    }
    VkDevice device() const {
        return device_;
    }
    VkQueue graphicsQueue() const {
        return graphicsQueue_;
    }
    VkQueue presentQueue() const {
        return presentQueue_;
    }
    VkSurfaceKHR surface() const {
        return surface_;
    }
    VkCommandPool commandPool() const {
        return commandPool_;
    }

    // 求队列族索引
    struct QueueFamilyIndices {
        uint32_t graphicsFamily = -1;
        uint32_t presentFamily = -1;
        bool isComplete() const {
            return graphicsFamily != (uint32_t)-1 && presentFamily != (uint32_t)-1;
        }
    };

    QueueFamilyIndices getQueueFamilyIndices() const {
        return queueIndices_;
    }

  private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkQueue graphicsQueue_ = VK_NULL_HANDLE;
    VkQueue presentQueue_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    QueueFamilyIndices queueIndices_;

    void createInstance(const std::string& appName) {
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = appName.c_str();
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_3;

        // 获取所需扩展
        uint32_t glfwExtCount = 0;
        const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
        std::vector<const char*> exts(glfwExts, glfwExts + glfwExtCount);

#ifdef __APPLE__
        exts.push_back("VK_KHR_portability_enumeration");
#endif

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(exts.size());
        createInfo.ppEnabledExtensionNames = exts.data();
        enablePortabilityBit(createInfo);

        if (vkCreateInstance(&createInfo, nullptr, &instance_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create VkInstance");
        }
    }

    void createSurface(GLFWwindow* window) {
        if (glfwCreateWindowSurface(instance_, window, nullptr, &surface_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create window surface");
        }
    }

    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);

        if (deviceCount == 0) {
            throw std::runtime_error("Failed to find GPUs with Vulkan support!");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

        for (const auto& device : devices) {
            if (isSuitableDevice(device)) {
                physicalDevice_ = device;
                return;
            }
        }

        throw std::runtime_error("Failed to find a suitable GPU!");
    }

    bool isSuitableDevice(VkPhysicalDevice device) {
        // 检查队列族
        queueIndices_ = findQueueFamilies(device);
        if (!queueIndices_.isComplete())
            return false;

        // 检查扩展支持
        // 💡 这里需要检查 VK_KHR_swapchain 和 VK_KHR_portability_subset

        return true;
    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }

            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface_, &presentSupport);
            if (presentSupport) {
                indices.presentFamily = i;
            }

            if (indices.isComplete())
                break;
            i++;
        }

        return indices;
    }

    void createLogicalDevice() {
        queueIndices_ = findQueueFamilies(physicalDevice_);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        float queuePriority = 1.0f;

        // 创建图形队列
        VkDeviceQueueCreateInfo queueInfo{};
        queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.queueFamilyIndex = queueIndices_.graphicsFamily;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueInfo);

        // 如果呈现队列不同，也创建它
        if (queueIndices_.presentFamily != queueIndices_.graphicsFamily) {
            queueInfo.queueFamilyIndex = queueIndices_.presentFamily;
            queueCreateInfos.push_back(queueInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures{};
        deviceFeatures.samplerAnisotropy = VK_TRUE;

        const std::vector<const char*> deviceExtensions = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            "VK_KHR_portability_subset" // macOS 必需
        };

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.pEnabledFeatures = &deviceFeatures;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (vkCreateDevice(physicalDevice_, &createInfo, nullptr, &device_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create logical device!");
        }

        vkGetDeviceQueue(device_, queueIndices_.graphicsFamily, 0, &graphicsQueue_);
        vkGetDeviceQueue(device_, queueIndices_.presentFamily, 0, &presentQueue_);
    }

    void createCommandPool() {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queueIndices_.graphicsFamily;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        if (vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create command pool!");
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// 层 2：VkRenderFramework - 渲染框架（RenderPass + Pipeline）
// ═══════════════════════════════════════════════════════════════════════════

class VkRenderFramework {
  public:
    void init(std::shared_ptr<VkContext> context, VkFormat swapchainFormat) {
        context_ = context;
        swapchainFormat_ = swapchainFormat;

        createRenderPass();
        createGraphicsPipeline();
    }

    void cleanup() {
        vkDestroyPipeline(context_->device(), pipeline_, nullptr);
        vkDestroyPipelineLayout(context_->device(), pipelineLayout_, nullptr);
        vkDestroyRenderPass(context_->device(), renderPass_, nullptr);
    }

    VkRenderPass renderPass() const {
        return renderPass_;
    }
    VkPipeline pipeline() const {
        return pipeline_;
    }

  private:
    std::shared_ptr<VkContext> context_;
    VkFormat swapchainFormat_;
    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;

    void createRenderPass() {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapchainFormat_;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(context_->device(), &renderPassInfo, nullptr, &renderPass_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create render pass!");
        }
    }

    void createGraphicsPipeline() {
        // 💡 这里省略了着色器模块创建的细节
        // 实际实现需要：createShaderModule() 等

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

        if (vkCreatePipelineLayout(context_->device(), &pipelineLayoutInfo, nullptr, &pipelineLayout_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create pipeline layout!");
        }

        // 创建图形管线
        // ... (省略详细代码，参考 ch08_triangle.cpp)
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// 层 3：VkSwapChainManager - 交换链管理（支持重建）
// ═══════════════════════════════════════════════════════════════════════════

class VkSwapChainManager {
  public:
    void init(std::shared_ptr<VkContext> context, std::shared_ptr<VkRenderFramework> framework, GLFWwindow* window) {
        context_ = context;
        framework_ = framework;
        window_ = window;

        recreate();
    }

    void cleanup() {
        destroySwapchain();
    }

    /**
     * 重建交换链（窗口大小改变时调用）
     */
    void recreate() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window_, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window_, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(context_->device());

        destroySwapchain();
        createSwapchain();
        createImageViews();
        createFramebuffers();
    }

    VkSwapchainKHR swapchain() const {
        return swapchain_;
    }
    const std::vector<VkFramebuffer>& framebuffers() const {
        return framebuffers_;
    }
    VkExtent2D extent() const {
        return extent_;
    }
    VkFormat format() const {
        return format_;
    }

  private:
    std::shared_ptr<VkContext> context_;
    std::shared_ptr<VkRenderFramework> framework_;
    GLFWwindow* window_;

    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    std::vector<VkImage> images_;
    std::vector<VkImageView> imageViews_;
    std::vector<VkFramebuffer> framebuffers_;
    VkExtent2D extent_{};
    VkFormat format_;

    void createSwapchain() { /* 参考 ch08_triangle.cpp */
    }
    void createImageViews() { /* 参考 ch08_triangle.cpp */
    }
    void createFramebuffers() { /* 参考 ch08_triangle.cpp */
    }

    void destroySwapchain() {
        for (auto fb : framebuffers_) {
            vkDestroyFramebuffer(context_->device(), fb, nullptr);
        }
        for (auto iv : imageViews_) {
            vkDestroyImageView(context_->device(), iv, nullptr);
        }
        if (swapchain_) {
            vkDestroySwapchainKHR(context_->device(), swapchain_, nullptr);
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// 层 4：VkFrameResource - 单帧资源
// ═══════════════════════════════════════════════════════════════════════════

struct VkFrameResource {
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkSemaphore imageAvailableSem = VK_NULL_HANDLE;
    VkSemaphore renderFinishedSem = VK_NULL_HANDLE;
    VkFence inFlightFence = VK_NULL_HANDLE;

    void init(VkDevice device, VkCommandPool cmdPool) {
        // 分配命令缓冲
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = cmdPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        // 创建信号量
        VkSemaphoreCreateInfo semInfo{};
        semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        vkCreateSemaphore(device, &semInfo, nullptr, &imageAvailableSem);
        vkCreateSemaphore(device, &semInfo, nullptr, &renderFinishedSem);

        // 创建栅栏（初始化为 SIGNALED，否则第一帧会永远等待）
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        vkCreateFence(device, &fenceInfo, nullptr, &inFlightFence);
    }

    void cleanup(VkDevice device) {
        vkDestroySemaphore(device, imageAvailableSem, nullptr);
        vkDestroySemaphore(device, renderFinishedSem, nullptr);
        vkDestroyFence(device, inFlightFence, nullptr);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// 顶层：VkRenderer - 应用程序接口
// ═══════════════════════════════════════════════════════════════════════════

class VkRenderer {
  public:
    void init(GLFWwindow* window) {
        window_ = window;

        // 初始化各层
        context_ = std::make_shared<VkContext>();
        context_->init(window);

        framework_ = std::make_shared<VkRenderFramework>();
        framework_->init(context_, VK_FORMAT_B8G8R8A8_SRGB);

        swapchain_ = std::make_shared<VkSwapChainManager>();
        swapchain_->init(context_, framework_, window);

        // 初始化帧资源
        frameResources_.resize(MAX_FRAMES_IN_FLIGHT);
        for (auto& res : frameResources_) {
            res.init(context_->device(), context_->commandPool());
        }
    }

    void cleanup() {
        vkDeviceWaitIdle(context_->device());

        for (auto& res : frameResources_) {
            res.cleanup(context_->device());
        }

        swapchain_->cleanup();
        framework_->cleanup();
        context_->cleanup();
    }

    /**
     * 渲染一帧
     */
    void render() {
        auto& frame = frameResources_[currentFrame_];

        // ① 等待上一帧完成
        vkWaitForFences(context_->device(), 1, &frame.inFlightFence, VK_TRUE, UINT64_MAX);

        // ② 获取下一张图像
        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(context_->device(),
                                                swapchain_->swapchain(),
                                                UINT64_MAX,
                                                frame.imageAvailableSem,
                                                VK_NULL_HANDLE,
                                                &imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            swapchain_->recreate();
            return;
        }

        // ③ 重置栅栏
        vkResetFences(context_->device(), 1, &frame.inFlightFence);

        // ④ 录制命令
        recordCommandBuffer(frame.commandBuffer, imageIndex);

        // ⑤ 提交命令
        submitFrame(frame);

        // ⑥ 呈现
        presentFrame(imageIndex, frame);

        currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void onWindowResized() {
        swapchain_->recreate();
    }

  private:
    GLFWwindow* window_;
    std::shared_ptr<VkContext> context_;
    std::shared_ptr<VkRenderFramework> framework_;
    std::shared_ptr<VkSwapChainManager> swapchain_;
    std::vector<VkFrameResource> frameResources_;
    uint32_t currentFrame_ = 0;

    void recordCommandBuffer(VkCommandBuffer cmdBuffer, uint32_t imageIndex) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmdBuffer, &beginInfo);

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = framework_->renderPass();
        renderPassInfo.framebuffer = swapchain_->framebuffers()[imageIndex];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapchain_->extent();

        VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;

        vkCmdBeginRenderPass(cmdBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, framework_->pipeline());
        vkCmdDraw(cmdBuffer, 3, 1, 0, 0);
        vkCmdEndRenderPass(cmdBuffer);

        vkEndCommandBuffer(cmdBuffer);
    }

    void submitFrame(VkFrameResource& frame) {
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &frame.imageAvailableSem;

        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        submitInfo.pWaitDstStageMask = &waitStage;

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &frame.commandBuffer;

        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &frame.renderFinishedSem;

        vkQueueSubmit(context_->graphicsQueue(), 1, &submitInfo, frame.inFlightFence);
    }

    void presentFrame(uint32_t imageIndex, VkFrameResource& frame) {
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &frame.renderFinishedSem;

        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain_->swapchain();
        presentInfo.pImageIndices = &imageIndex;

        VkResult result = vkQueuePresentKHR(context_->presentQueue(), &presentInfo);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            swapchain_->recreate();
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// 使用示例
// ═══════════════════════════════════════════════════════════════════════════

/*
int main() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(800, 600, "Refactored Vulkan", nullptr, nullptr);

    VkRenderer renderer;
    renderer.init(window);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        renderer.render();
    }

    renderer.cleanup();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
*/
