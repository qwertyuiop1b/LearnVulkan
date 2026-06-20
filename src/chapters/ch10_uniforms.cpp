/**
 * @file ch10_uniforms.cpp
 * @brief 第10章：Uniform Buffer Object（UBO）与描述符集
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【Uniform Buffer Object（UBO）】
 *
 *  UBO 是每帧从 CPU 更新到 GPU 的常量数据：
 *    - MVP 矩阵（模型/视图/投影变换）
 *    - 光照参数
 *    - 时间、分辨率等全局参数
 *
 * 【描述符（Descriptor）系统】
 *
 *  Vulkan 通过描述符将 GPU 资源（缓冲、纹理）绑定到着色器：
 *
 *  ┌──────────────────────────────────────────────────────────┐
 *  │  描述符集布局 (DescriptorSetLayout)                       │
 *  │    → 声明：此集合中有哪些绑定（UBO? 纹理?）              │
 *  │                                                          │
 *  │  描述符池 (DescriptorPool)                               │
 *  │    → 为描述符集分配内存                                   │
 *  │                                                          │
 *  │  描述符集 (DescriptorSet)                                │
 *  │    → 将实际资源（VkBuffer、VkImageView）绑定到着色器槽位  │
 *  └──────────────────────────────────────────────────────────┘
 *
 * 【旋转正方形示例】
 *
 *  使用 GLM 计算 MVP 矩阵，实现正方形旋转动画：
 *    - model：随时间旋转的矩阵
 *    - view：从上方45度俯视的相机
 *    - projection：透视投影（45度视角）
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

// ─── UBO 数据结构（与 GLSL uniform 布局一致） ─────────────────────────────────

/**
 * @brief MVP 矩阵 Uniform Buffer
 *
 * 注意：GLSL 中 mat4 对齐到 16 字节，C++ 中 glm::mat4 默认满足。
 * 如果有 vec3 等较小类型，要小心 std140 内存布局规则。
 */
struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 projection;
};

// ─── 带 UV 坐标的顶点 ─────────────────────────────────────────────────────────

struct Vertex {
    glm::vec2 pos;
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
        a[0] = {0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, pos)};
        a[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)};
        return a;
    }
};

// 正方形顶点
static const std::vector<Vertex> VERTICES = {
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}}, // 左上 红
    {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},  // 右上 绿
    {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},   // 右下 蓝
    {{-0.5f, 0.5f}, {1.0f, 1.0f, 0.0f}},  // 左下 黄
};

static const std::vector<uint16_t> INDICES = {0, 1, 2, 2, 3, 0};

// ─── 应用程序 ─────────────────────────────────────────────────────────────────

class Ch10App {
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
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;

    // ─── 描述符相关对象（新增） ─────────────────────────────────────────────
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets_;

    // ─── UBO（每帧一份，避免 GPU 读取时 CPU 覆写） ──────────────────────────
    std::vector<VkBuffer> uniformBuffers_;
    std::vector<VkDeviceMemory> uniformBuffersMemory_;
    std::vector<void*> uniformBuffersMapped_; // 持久化映射指针

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
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "Ch10 - UBO 旋转正方形", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
            reinterpret_cast<Ch10App*>(glfwGetWindowUserPointer(w))->resized_ = true;
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
        createDescriptorSetLayout(); // ← 新增
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers(); // ← 新增
        createDescriptorPool(); // ← 新增
        createDescriptorSets(); // ← 新增
        createCommandBuffers();
        createSyncObjects();
        std::cout << "\n✅ 旋转正方形初始化完成！\n";
    }

    // ─── 核心新增：描述符集布局 ───────────────────────────────────────────────

    void createDescriptorSetLayout() {
        // 描述绑定点0：Uniform Buffer Object
        VkDescriptorSetLayoutBinding uboBinding{};
        uboBinding.binding = 0; // binding=0 对应 GLSL: layout(binding=0)
        uboBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboBinding.descriptorCount = 1;                     // 不是数组，只有1个
        uboBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT; // 只在顶点着色器使用
        uboBinding.pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 1;
        ci.pBindings = &uboBinding;

        VK_CHECK(vkCreateDescriptorSetLayout(device_, &ci, nullptr, &descriptorSetLayout_));
        std::cout << "✅ 描述符集布局已创建\n";
    }

    // ─── 核心新增：创建 UBO（每个飞行帧一份） ────────────────────────────────

    void createUniformBuffers() {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);

        uniformBuffers_.resize(MAX_FRAMES);
        uniformBuffersMemory_.resize(MAX_FRAMES);
        uniformBuffersMapped_.resize(MAX_FRAMES);

        for (int i = 0; i < MAX_FRAMES; ++i) {
            createBuffer(bufferSize,
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         // HOST_VISIBLE + HOST_COHERENT：CPU 可以直接写入
                         // 持久化映射（Persistent Mapping）：一次 Map，永远保持映射
                         // 比每帧 Map/Unmap 更高效
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         uniformBuffers_[i],
                         uniformBuffersMemory_[i]);

            // 持久化映射：获取指针，后续直接 memcpy
            vkMapMemory(device_, uniformBuffersMemory_[i], 0, bufferSize, 0, &uniformBuffersMapped_[i]);
        }
        std::cout << "✅ UBO 已创建（" << MAX_FRAMES << " 份，持久化映射）\n";
    }

    // ─── 核心新增：描述符池 ───────────────────────────────────────────────────

    void createDescriptorPool() {
        // 池的容量：能容纳多少个各类描述符
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSize.descriptorCount = static_cast<uint32_t>(MAX_FRAMES);

        VkDescriptorPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        ci.poolSizeCount = 1;
        ci.pPoolSizes = &poolSize;
        ci.maxSets = static_cast<uint32_t>(MAX_FRAMES); // 最多分配多少个描述符集

        VK_CHECK(vkCreateDescriptorPool(device_, &ci, nullptr, &descriptorPool_));
        std::cout << "✅ 描述符池已创建\n";
    }

    // ─── 核心新增：分配并更新描述符集 ────────────────────────────────────────

    void createDescriptorSets() {
        // 每个飞行帧需要一个描述符集（各自引用独立的 UBO）
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES, descriptorSetLayout_);

        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = descriptorPool_;
        ai.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES);
        ai.pSetLayouts = layouts.data();

        descriptorSets_.resize(MAX_FRAMES);
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, descriptorSets_.data()));

        // 更新描述符集：将实际的 UBO 缓冲绑定到描述符
        for (int i = 0; i < MAX_FRAMES; ++i) {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers_[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject); // VK_WHOLE_SIZE 也可以

            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = descriptorSets_[i];
            write.dstBinding = 0;      // 写入 binding=0
            write.dstArrayElement = 0; // 不是数组，从索引0开始
            write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &bufferInfo;

            vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
        }
        std::cout << "✅ 描述符集已分配并更新\n";
    }

    // ─── 每帧更新 UBO ─────────────────────────────────────────────────────────

    void updateUniformBuffer(uint32_t currentImage) {
        static auto startTime = std::chrono::high_resolution_clock::now();
        auto now = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float>(now - startTime).count();

        UniformBufferObject ubo{};
        // 绕 Z 轴旋转（角速度：90度/秒）
        ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));

        // 从 (2,2,2) 位置，看向原点，Y 轴朝上
        ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));

        // 透视投影：45度视角，宽高比，近裁面0.1，远裁面10
        ubo.projection = glm::perspective(
            glm::radians(45.0f), static_cast<float>(swapchainExtent_.width) / swapchainExtent_.height, 0.1f, 10.0f);

        // Vulkan NDC Y 轴朝下，GLM 设计给 OpenGL（Y 朝上），需要翻转
        ubo.projection[1][1] *= -1;

        // 持久化映射：直接写入，HOST_COHERENT 确保 GPU 看到最新数据
        std::memcpy(uniformBuffersMapped_[currentImage], &ubo, sizeof(ubo));
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
        VkClearValue clear = {{{0.02f, 0.02f, 0.05f, 1.0f}}};
        VkRenderPassBeginInfo rpBI{};
        rpBI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpBI.renderPass = renderPass_;
        rpBI.framebuffer = framebuffers_[imageIndex];
        rpBI.renderArea = {{0, 0}, swapchainExtent_};
        rpBI.clearValueCount = 1;
        rpBI.pClearValues = &clear;
        vkCmdBeginRenderPass(cmd, &rpBI, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        VkViewport vp{0, 0, (float)swapchainExtent_.width, (float)swapchainExtent_.height, 0, 1};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{{0, 0}, swapchainExtent_};
        vkCmdSetScissor(cmd, 0, 1, &sc);
        VkBuffer vbufs[] = {vertexBuffer_};
        VkDeviceSize offs[] = {0};
        vkCmdBindVertexBuffers(cmd, 0, 1, vbufs, offs);
        vkCmdBindIndexBuffer(cmd, indexBuffer_, 0, VK_INDEX_TYPE_UINT16);

        // ── 绑定描述符集（关联 UBO） ─────────────────────────────────────────
        // pipelineBindPoint: 图形管线还是计算管线
        // pipelineLayout: 与管线创建时使用的布局一致
        // firstSet: 从第几个描述符集开始绑定（这里从 set=0 开始）
        vkCmdBindDescriptorSets(cmd,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayout_,
                                0, // firstSet
                                1,
                                &descriptorSets_[currentFrame_], // 当前帧的描述符集
                                0,
                                nullptr); // 动态偏移（不使用）

        vkCmdDrawIndexed(cmd, static_cast<uint32_t>(INDICES.size()), 1, 0, 0, 0);
        vkCmdEndRenderPass(cmd);
        VK_CHECK(vkEndCommandBuffer(cmd));
    }

    void drawFrame() {
        vkWaitForFences(device_, 1, &inFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX);
        uint32_t imageIndex = 0;
        VkResult res = vkAcquireNextImageKHR(
            device_, swapchain_, UINT64_MAX, imageAvailableSems_[currentFrame_], VK_NULL_HANDLE, &imageIndex);
        if (res == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapchain();
            return;
        }
        updateUniformBuffer(currentFrame_); // ← 每帧更新 UBO
        vkResetFences(device_, 1, &inFlightFences_[currentFrame_]);
        vkResetCommandBuffer(commandBuffers_[currentFrame_], 0);
        recordCommandBuffer(commandBuffers_[currentFrame_], imageIndex);
        VkSemaphore waitSems[] = {imageAvailableSems_[currentFrame_]};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        VkSemaphore signalSems[] = {renderFinishedSems_[currentFrame_]};
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = waitSems;
        si.pWaitDstStageMask = waitStages;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &commandBuffers_[currentFrame_];
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = signalSems;
        VK_CHECK(vkQueueSubmit(graphicsQueue_, 1, &si, inFlightFences_[currentFrame_]));
        VkSwapchainKHR swaps[] = {swapchain_};
        VkPresentInfoKHR pi{};
        pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = signalSems;
        pi.swapchainCount = 1;
        pi.pSwapchains = swaps;
        pi.pImageIndices = &imageIndex;
        res = vkQueuePresentKHR(presentQueue_, &pi);
        if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR || resized_) {
            resized_ = false;
            recreateSwapchain();
        }
        currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES;
    }

    void mainLoop() {
        std::cout << "🎨 正方形正在旋转！（按 ESC 退出）\n";
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
                glfwSetWindowShouldClose(window_, GLFW_TRUE);
            drawFrame();
        }
        vkDeviceWaitIdle(device_);
    }

    // ─── 辅助函数 ─────────────────────────────────────────────────────────────

    uint32_t findMemoryType(uint32_t filter, VkMemoryPropertyFlags props) {
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((filter & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
                return i;
        throw std::runtime_error("找不到合适内存");
    }

    void createBuffer(
        VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, VkBuffer& buf, VkDeviceMemory& mem) {
        VkBufferCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ci.size = size;
        ci.usage = usage;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(device_, &ci, nullptr, &buf));
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(device_, buf, &mr);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, props);
        VK_CHECK(vkAllocateMemory(device_, &ai, nullptr, &mem));
        VK_CHECK(vkBindBufferMemory(device_, buf, mem, 0));
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

    void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size) {
        VkCommandBuffer cmd = beginSingleTimeCommands();
        VkBufferCopy r{};
        r.size = size;
        vkCmdCopyBuffer(cmd, src, dst, 1, &r);
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
        for (auto& d : devs) {
            if (findQueueFamilies(d, surface_).isComplete() && checkDeviceExtensionSupport(d)) {
                physicalDevice_ = d;
                break;
            }
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

    // 保留兼容旧接口（未使用，可删除）
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
        // uniform.vert: vec2 pos + vec3 color + UBO MVP 矩阵
        // triangle.frag: 输出插值颜色
        VkShaderModule vert = createShaderModuleFromFile(device_, "uniform.vert.spv");
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

        // ── 管线布局中包含描述符集布局 ───────────────────────────────────────
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
    std::cout << " 第10章：Uniform Buffer + 描述符（旋转正方形）\n";
    std::cout << "═══════════════════════════════════════════════════\n\n";
    Ch10App app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ 错误：" << e.what() << "\n";
        return 1;
    }
    return 0;
}
