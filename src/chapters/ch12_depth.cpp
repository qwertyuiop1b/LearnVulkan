/**
 * @file ch12_depth.cpp
 * @brief 第12章：深度缓冲（Depth Buffering）与 3D 渲染
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【为什么需要深度缓冲？】
 *
 *  没有深度缓冲时，后绘制的几何体会覆盖先绘制的（画家算法）。
 *  深度缓冲（Z-Buffer）让 GPU 自动处理遮挡关系：
 *    - 每个片段有一个深度值（0.0 近 ~ 1.0 远）
 *    - 只有深度值比已存储值更近的片段才会通过测试（被写入）
 *
 * 【深度缓冲实现步骤】
 *
 *  1. 找到合适的深度格式（D32_SFLOAT 或 D24_UNORM_S8_UINT）
 *  2. 创建深度 VkImage（与交换链同样大小）
 *  3. 创建深度 VkImageView
 *  4. 修改 RenderPass，添加深度附件
 *  5. 修改 Framebuffer，绑定深度 ImageView
 *  6. 修改管线，启用深度测试（VkPipelineDepthStencilStateCreateInfo）
 *
 * 【深度格式选择】
 *
 *  VK_FORMAT_D32_SFLOAT           → 32位深度，无模板
 *  VK_FORMAT_D32_SFLOAT_S8_UINT   → 32位深度 + 8位模板
 *  VK_FORMAT_D24_UNORM_S8_UINT    → 24位深度 + 8位模板（最常见）
 *
 * 【3D 示例：两个交叠的彩色正方形】
 *
 *  ┌────────────────────────────────────────────────────┐
 *  │  黄色正方形（z = 0.0，正前方）                      │
 *  │  橙色正方形（z = 0.5，较远处）                      │
 *  │                                                    │
 *  │  深度测试自动处理遮挡：黄色遮住橙色的重叠部分       │
 *  └────────────────────────────────────────────────────┘
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

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 projection;
};

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription d{};
        d.binding = 0;
        d.stride = sizeof(Vertex);
        d.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return d;
    }
    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> a{};
        a[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)};
        a[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)};
        return a;
    }
};

// 两个 3D 正方形，不同 Z 深度，验证深度缓冲效果
static const std::vector<Vertex> VERTICES = {
    // 前方正方形（z=0.0）：黄色
    {{-0.5f, -0.5f, 0.0f}, {1.0f, 1.0f, 0.0f}},
    {{0.5f, -0.5f, 0.0f}, {1.0f, 1.0f, 0.0f}},
    {{0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 0.0f}},
    {{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 0.0f}},

    // 后方正方形（z=-0.5）：橙色（部分被遮挡）
    {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.5f, 0.0f}},
    {{0.5f, -0.5f, -0.5f}, {1.0f, 0.5f, 0.0f}},
    {{0.5f, 0.5f, -0.5f}, {1.0f, 0.5f, 0.0f}},
    {{-0.5f, 0.5f, -0.5f}, {1.0f, 0.5f, 0.0f}},
};
static const std::vector<uint16_t> INDICES = {
    0,
    1,
    2,
    2,
    3,
    0, // 前方正方形
    4,
    5,
    6,
    6,
    7,
    4 // 后方正方形
};

class Ch12App {
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

    // ─── 深度缓冲（新增） ──────────────────────────────────────────────────
    VkImage depthImage_ = VK_NULL_HANDLE;
    VkDeviceMemory depthImageMemory_ = VK_NULL_HANDLE;
    VkImageView depthImageView_ = VK_NULL_HANDLE;
    VkFormat depthFormat_ = VK_FORMAT_UNDEFINED;

    std::vector<VkDescriptorSet> descriptorSets_;
    std::vector<VkBuffer> uniformBuffers_;
    std::vector<VkDeviceMemory> uniformBuffersMemory_;
    std::vector<void*> uniformBuffersMapped_;
    std::vector<VkImage> swapchainImages_;
    std::vector<VkImageView> swapchainImageViews_;
    std::vector<VkFramebuffer> framebuffers_;
    std::vector<VkCommandBuffer> commandBuffers_;
    VkFormat swapchainImageFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchainExtent_{};
    QueueFamilyIndices queueIndices_;

    VkBuffer vertexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory vertexBufferMemory_ = VK_NULL_HANDLE;
    VkBuffer indexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory indexBufferMemory_ = VK_NULL_HANDLE;

    std::vector<VkSemaphore> imageAvailableSems_;
    std::vector<VkSemaphore> renderFinishedSems_;
    std::vector<VkFence> inFlightFences_;
    uint32_t currentFrame_ = 0;
    bool resized_ = false;

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "Ch12 - 深度缓冲（3D两个正方形）", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
            reinterpret_cast<Ch12App*>(glfwGetWindowUserPointer(w))->resized_ = true;
        });
    }

    void initVulkan() {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapchain();
        createImageViews();
        depthFormat_ = findDepthFormat(); // ← 新增
        createRenderPass();               // 包含深度附件
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createDepthResources(); // ← 新增
        createFramebuffers();   // 包含深度 ImageView
        createCommandPool();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
        std::cout << "\n✅ 深度缓冲初始化完成！\n";
    }

    // ─── 核心新增：查找深度格式 ───────────────────────────────────────────────

    VkFormat
    findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
        for (VkFormat format : candidates) {
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(physicalDevice_, format, &props);
            if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features)
                return format;
            if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features)
                return format;
        }
        throw std::runtime_error("找不到支持的图像格式");
    }

    VkFormat findDepthFormat() {
        return findSupportedFormat({VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
                                   VK_IMAGE_TILING_OPTIMAL,
                                   VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    }

    bool hasStencilComponent(VkFormat format) {
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
    }

    // ─── 核心新增：创建深度资源 ───────────────────────────────────────────────

    void createDepthResources() {
        createImage(swapchainExtent_.width,
                    swapchainExtent_.height,
                    depthFormat_,
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, // 用作深度/模板附件
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    depthImage_,
                    depthImageMemory_);

        // 创建深度图像视图（aspect = DEPTH_BIT）
        VkImageViewCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ci.image = depthImage_;
        ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ci.format = depthFormat_;
        ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        ci.subresourceRange.baseMipLevel = 0;
        ci.subresourceRange.levelCount = 1;
        ci.subresourceRange.baseArrayLayer = 0;
        ci.subresourceRange.layerCount = 1;
        VK_CHECK(vkCreateImageView(device_, &ci, nullptr, &depthImageView_));

        std::cout << "✅ 深度缓冲已创建（格式：" << depthFormat_ << "）\n";
    }

    // ─── 修改 RenderPass：增加深度附件 ───────────────────────────────────────

    void createRenderPass() {
        // 颜色附件（与之前相同）
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapchainImageFormat_;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        // ── 深度附件（新增） ──────────────────────────────────────────────────
        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = depthFormat_;
        depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;       // 每帧清除为最远值
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; // 渲染后不需要保存
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorRef{};
        colorRef.attachment = 0; // 附件索引0
        colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthRef{};
        depthRef.attachment = 1; // 附件索引1
        depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorRef;
        subpass.pDepthStencilAttachment = &depthRef; // 绑定深度附件

        // Subpass 依赖：确保深度也同步
        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.srcAccessMask = 0;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        std::array<VkAttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};
        VkRenderPassCreateInfo rpi{};
        rpi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpi.attachmentCount = static_cast<uint32_t>(attachments.size());
        rpi.pAttachments = attachments.data();
        rpi.subpassCount = 1;
        rpi.pSubpasses = &subpass;
        rpi.dependencyCount = 1;
        rpi.pDependencies = &dep;

        VK_CHECK(vkCreateRenderPass(device_, &rpi, nullptr, &renderPass_));
    }

    // ─── 修改 Framebuffer：添加深度 ImageView ─────────────────────────────────

    void createFramebuffers() {
        framebuffers_.resize(swapchainImageViews_.size());
        for (size_t i = 0; i < swapchainImageViews_.size(); ++i) {
            std::array<VkImageView, 2> attachments = {
                swapchainImageViews_[i],
                depthImageView_ // 所有帧缓冲共用同一个深度缓冲
            };
            VkFramebufferCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            ci.renderPass = renderPass_;
            ci.attachmentCount = static_cast<uint32_t>(attachments.size());
            ci.pAttachments = attachments.data();
            ci.width = swapchainExtent_.width;
            ci.height = swapchainExtent_.height;
            ci.layers = 1;
            VK_CHECK(vkCreateFramebuffer(device_, &ci, nullptr, &framebuffers_[i]));
        }
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t idx) {
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

        // 清除值：颜色 + 深度
        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color.float32[0] = 0.02f;
        clearValues[0].color.float32[1] = 0.02f;
        clearValues[0].color.float32[2] = 0.05f;
        clearValues[0].color.float32[3] = 1.0f;
        clearValues[1].depthStencil = {1.0f, 0}; // 深度初始化为最远（1.0）

        VkRenderPassBeginInfo rpBI{};
        rpBI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpBI.renderPass = renderPass_;
        rpBI.framebuffer = framebuffers_[idx];
        rpBI.renderArea = {{0, 0}, swapchainExtent_};
        rpBI.clearValueCount = static_cast<uint32_t>(clearValues.size());
        rpBI.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(cmd, &rpBI, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        VkViewport vp{0, 0, (float)swapchainExtent_.width, (float)swapchainExtent_.height, 0, 1};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{{0, 0}, swapchainExtent_};
        vkCmdSetScissor(cmd, 0, 1, &sc);
        VkBuffer vb[] = {vertexBuffer_};
        VkDeviceSize off[] = {0};
        vkCmdBindVertexBuffers(cmd, 0, 1, vb, off);
        vkCmdBindIndexBuffer(cmd, indexBuffer_, 0, VK_INDEX_TYPE_UINT16);
        vkCmdBindDescriptorSets(
            cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_, 0, 1, &descriptorSets_[currentFrame_], 0, nullptr);
        vkCmdDrawIndexed(cmd, static_cast<uint32_t>(INDICES.size()), 1, 0, 0, 0);
        vkCmdEndRenderPass(cmd);
        VK_CHECK(vkEndCommandBuffer(cmd));
    }

    void updateUniformBuffer(uint32_t frame) {
        static auto start = std::chrono::high_resolution_clock::now();
        float t = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start).count();
        UniformBufferObject ubo{};
        ubo.model = glm::rotate(glm::mat4(1.0f), t * glm::radians(30.0f), glm::vec3(0, 0, 1));
        ubo.view = glm::lookAt(glm::vec3(2, 2, 2), glm::vec3(0, 0, 0), glm::vec3(0, 0, 1));
        ubo.projection =
            glm::perspective(glm::radians(45.0f), (float)swapchainExtent_.width / swapchainExtent_.height, 0.1f, 10.0f);
        ubo.projection[1][1] *= -1;
        std::memcpy(uniformBuffersMapped_[frame], &ubo, sizeof(ubo));
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
        VkSwapchainKHR sc2[] = {swapchain_};
        VkPresentInfoKHR pi{};
        pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = ss;
        pi.swapchainCount = 1;
        pi.pSwapchains = sc2;
        pi.pImageIndices = &imgIdx;
        r = vkQueuePresentKHR(presentQueue_, &pi);
        if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR || resized_) {
            resized_ = false;
            recreateSwapchain();
        }
        currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES;
    }

    void mainLoop() {
        std::cout << "🎨 3D两个正方形（深度测试中，黄色在前，橙色在后）...\n";
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

    void createImage(uint32_t w,
                     uint32_t h,
                     VkFormat fmt,
                     VkImageTiling til,
                     VkImageUsageFlags u,
                     VkMemoryPropertyFlags p,
                     VkImage& img,
                     VkDeviceMemory& mem) {
        VkImageCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.extent = {w, h, 1};
        ci.mipLevels = 1;
        ci.arrayLayers = 1;
        ci.format = fmt;
        ci.tiling = til;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ci.usage = u;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateImage(device_, &ci, nullptr, &img));
        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(device_, img, &mr);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, p);
        VK_CHECK(vkAllocateMemory(device_, &ai, nullptr, &mem));
        VK_CHECK(vkBindImageMemory(device_, img, mem, 0));
    }

    VkCommandBuffer beginSingleTimeCommands() {
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
        return cmd;
    }
    void endSingleTimeCommands(VkCommandBuffer cmd) {
        vkEndCommandBuffer(cmd);
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        vkQueueSubmit(graphicsQueue_, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue_);
        vkFreeCommandBuffers(device_, commandPool_, 1, &cmd);
    }
    void copyBuffer(VkBuffer s, VkBuffer d, VkDeviceSize sz) {
        auto cmd = beginSingleTimeCommands();
        VkBufferCopy r{};
        r.size = sz;
        vkCmdCopyBuffer(cmd, s, d, 1, &r);
        endSingleTimeCommands(cmd);
    }

    void createVertexBuffer() {
        VkDeviceSize sz = sizeof(VERTICES[0]) * VERTICES.size();
        VkBuffer sb;
        VkDeviceMemory sm;
        createBuffer(sz,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     sb,
                     sm);
        void* d = nullptr;
        vkMapMemory(device_, sm, 0, sz, 0, &d);
        std::memcpy(d, VERTICES.data(), (size_t)sz);
        vkUnmapMemory(device_, sm);
        createBuffer(sz,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     vertexBuffer_,
                     vertexBufferMemory_);
        copyBuffer(sb, vertexBuffer_, sz);
        vkDestroyBuffer(device_, sb, nullptr);
        vkFreeMemory(device_, sm, nullptr);
    }
    void createIndexBuffer() {
        VkDeviceSize sz = sizeof(INDICES[0]) * INDICES.size();
        VkBuffer sb;
        VkDeviceMemory sm;
        createBuffer(sz,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     sb,
                     sm);
        void* d = nullptr;
        vkMapMemory(device_, sm, 0, sz, 0, &d);
        std::memcpy(d, INDICES.data(), (size_t)sz);
        vkUnmapMemory(device_, sm);
        createBuffer(sz,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     indexBuffer_,
                     indexBufferMemory_);
        copyBuffer(sb, indexBuffer_, sz);
        vkDestroyBuffer(device_, sb, nullptr);
        vkFreeMemory(device_, sm, nullptr);
    }
    void createUniformBuffers() {
        VkDeviceSize sz = sizeof(UniformBufferObject);
        uniformBuffers_.resize(MAX_FRAMES);
        uniformBuffersMemory_.resize(MAX_FRAMES);
        uniformBuffersMapped_.resize(MAX_FRAMES);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            createBuffer(sz,
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         uniformBuffers_[i],
                         uniformBuffersMemory_[i]);
            vkMapMemory(device_, uniformBuffersMemory_[i], 0, sz, 0, &uniformBuffersMapped_[i]);
        }
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
            bi.range = sizeof(UniformBufferObject);
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
    VkShaderModule createShaderModule(const uint32_t* c, size_t s) {
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = s;
        ci.pCode = c;
        VkShaderModule m = VK_NULL_HANDLE;
        VK_CHECK(vkCreateShaderModule(device_, &ci, nullptr, &m));
        return m;
    }
    void createGraphicsPipeline() {
        // uniform3d.vert: vec3 pos + vec3 color + UBO; triangle.frag: 插值颜色
        VkShaderModule vert = createShaderModuleFromFile(device_, "uniform3d.vert.spv");
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
        auto bd = Vertex::getBindingDescription();
        auto ad = Vertex::getAttributeDescriptions();
        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &bd;
        vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(ad.size());
        vi.pVertexAttributeDescriptions = ad.data();
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
        rs.cullMode = VK_CULL_MODE_BACK_BIT;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        // ── 深度/模板测试配置（新增）────────────────────────────────────────
        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;  // 启用深度测试
        depthStencil.depthWriteEnable = VK_TRUE; // 通过测试的片段写入深度缓冲
        // compareOp：通过测试的条件
        //   LESS：新片段深度 < 已有深度才通过（近处覆盖远处）
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE; // 深度范围限制（不使用）
        depthStencil.stencilTestEnable = VK_FALSE;     // 模板测试（不使用）

        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = 0xf;
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
        pi.pDepthStencilState = &depthStencil; // ← 启用深度测试
        pi.pColorBlendState = &cb;
        pi.pDynamicState = &ds;
        pi.layout = pipelineLayout_;
        pi.renderPass = renderPass_;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline_));
        vkDestroyShaderModule(device_, frag, nullptr);
        vkDestroyShaderModule(device_, vert, nullptr);
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
    void cleanup() {
        vkDestroyImageView(device_, depthImageView_, nullptr);
        vkDestroyImage(device_, depthImage_, nullptr);
        vkFreeMemory(device_, depthImageMemory_, nullptr);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroyBuffer(device_, uniformBuffers_[i], nullptr);
            vkFreeMemory(device_, uniformBuffersMemory_[i], nullptr);
        }
        vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
        vkDestroyBuffer(device_, indexBuffer_, nullptr);
        vkFreeMemory(device_, indexBufferMemory_, nullptr);
        vkDestroyBuffer(device_, vertexBuffer_, nullptr);
        vkFreeMemory(device_, vertexBufferMemory_, nullptr);
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
    std::cout << "═══════════════════════════════════════════════════\n";
    std::cout << " 第12章：深度缓冲（3D两个正方形，正确遮挡）\n";
    std::cout << "═══════════════════════════════════════════════════\n\n";
    Ch12App app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ 错误：" << e.what() << "\n";
        return 1;
    }
    return 0;
}
