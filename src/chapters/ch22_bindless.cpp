/**
 * @file ch22_bindless.cpp
 * @brief 第22章：Bindless Rendering + Buffer Device Address
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【传统描述符绑定的痛点】
 *
 *  传统 Vulkan 渲染：
 *    绘制 1000 个不同纹理的物体：
 *    for (obj : objects) {
 *        vkCmdBindDescriptorSets(... obj.textureSet ...);   // 切换描述符！
 *        vkCmdDraw(...);
 *    }
 *    问题：1000 次描述符切换 = 驱动验证 + GPU 状态切换 = 性能瓶颈
 *
 *  Bindless 渲染：
 *    一次绑定包含所有纹理的超大描述符集
 *    在着色器里通过运行时索引选择纹理：
 *    texture(textures[materialID.textureIndex], uv)
 *    → 零描述符切换，性能提升 5-10 倍！
 *
 * 【核心技术：Descriptor Indexing（VK_EXT_descriptor_indexing）】
 *
 *  已在 Vulkan 1.2 成为核心特性。关键特性：
 *
 *  1. 变长数组（Runtime Array）：
 *     layout(binding=0) uniform sampler2D textures[];  // 无固定大小
 *
 *  2. 更新后继续使用：
 *     PARTIALLY_BOUND_BIT：数组中可以有未绑定的槽位
 *     UPDATE_AFTER_BIND_BIT：绑定到命令缓冲后仍可更新
 *
 *  3. nonuniformEXT：
 *     告诉驱动同一 wave 内的线程可能访问不同索引
 *
 * 【Buffer Device Address（VK_KHR_buffer_device_address）】
 *
 *  GPU 端直接使用 64-bit 指针访问缓冲数据：
 *    VkDeviceAddress addr = vkGetBufferDeviceAddress(device, &info);
 *    // 通过 push constant 传给着色器
 *    // 着色器中：layout(buffer_reference) buffer MyBuf { ... };
 *    //            MyBuf data = MyBuf(addr);  ← GPU 指针！
 *
 *  优势：
 *    - 无需描述符集管理
 *    - 随时切换数据源（只需更新 push constant）
 *    - 支持 GPU 端计算指针偏移
 *
 * 【本章示例】
 *
 *  - 创建包含程序化纹理的大型纹理数组（32 个纹理）
 *  - 每个物体通过 per-draw push constant 指定使用哪个纹理
 *  - 顶点数据通过 Buffer Device Address 传递（无顶点绑定！）
 *  - 演示 descriptor_count=1024 的变长描述符数组
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
#include <vector>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int MAX_FRAMES = 2;
constexpr uint32_t N_TEXTURES = 8; // 纹理数组大小
constexpr uint32_t N_OBJECTS = 32; // 绘制的物体数量

// ─── 数据结构 ──────────────────────────────────────────────────────────────────

struct alignas(8) Vertex {
    glm::vec3 pos;
    glm::vec2 uv;
};

// Per-draw Push Constant（每次绘制变一次）
struct DrawPushConstants {
    glm::mat4 mvp;
    uint64_t vertexBufferAddress; // Buffer Device Address（GPU 指针）
    uint32_t textureIndex;        // 运行时纹理索引
    float time;
};

static const std::vector<Vertex> QUAD_VERTICES = {
    {{-0.5f, -0.5f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, -0.5f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, 0.5f, 0.0f}, {1.0f, 1.0f}},
    {{-0.5f, -0.5f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, 0.5f, 0.0f}, {1.0f, 1.0f}},
    {{-0.5f, 0.5f, 0.0f}, {0.0f, 1.0f}},
};

class Ch22App {
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

    // ─── Bindless 纹理数组描述符 ────────────────────────────────────────────
    VkDescriptorSetLayout bindlessSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool bindlessPool_ = VK_NULL_HANDLE;
    VkDescriptorSet bindlessSet_ = VK_NULL_HANDLE; // 一个集合包含所有纹理！

    // ─── 纹理数组 ──────────────────────────────────────────────────────────
    std::vector<VkImage> textureImages_;
    std::vector<VkDeviceMemory> textureMemories_;
    std::vector<VkImageView> textureViews_;
    VkSampler textureSampler_ = VK_NULL_HANDLE;

    // ─── 顶点缓冲（Buffer Device Address） ────────────────────────────────
    VkBuffer vertexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory_ = VK_NULL_HANDLE;
    VkDeviceAddress vertexBDA_ = 0; // GPU 端指针！

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
            glfwCreateWindow(WIDTH, HEIGHT, "Ch22 - Bindless Rendering（32个物体，0次描述符切换）", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
            reinterpret_cast<Ch22App*>(glfwGetWindowUserPointer(w))->resized_ = true;
        });
    }

    void initVulkan() {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice(); // 需要启用 descriptor_indexing + buffer_device_address
        createSwapchain();
        createImageViews();
        createRenderPass();
        createBindlessDescriptorSetLayout(); // ← 变长数组布局
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createTextureArray();          // ← 创建 N_TEXTURES 个程序化纹理
        createBindlessDescriptorSet(); // ← 将所有纹理绑定到一个描述符集
        createVertexBufferWithBDA();   // ← 获取 Buffer Device Address
        createCommandBuffers();
        createSyncObjects();
        std::cout << "\n✅ Bindless 渲染初始化完成！\n";
        std::cout << "🖼️  纹理数组大小：" << N_TEXTURES << " 个纹理\n";
        std::cout << "📦 物体数量：" << N_OBJECTS << " 个（0 次描述符切换！）\n";
        std::cout << "🔗 顶点缓冲 GPU 地址：0x" << std::hex << vertexBDA_ << std::dec << "\n";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Bindless 描述符集布局（变长数组）
    // ═══════════════════════════════════════════════════════════════════════

    void createBindlessDescriptorSetLayout() {
        // 关键：bindingFlags 控制 Bindless 行为
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binding.descriptorCount = N_TEXTURES; // 数组大小（运行时可以小于此值）
        binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        // ── Bindless 关键 flags ───────────────────────────────────────────
        VkDescriptorBindingFlags bindingFlags =
            // 数组中可以有未绑定的槽位（不需要填满 N_TEXTURES 个）
            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
            // 即使已经提交到命令缓冲，仍然可以更新描述符（动态加载纹理）
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;

        VkDescriptorSetLayoutBindingFlagsCreateInfo flagsCI{};
        flagsCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
        flagsCI.bindingCount = 1;
        flagsCI.pBindingFlags = &bindingFlags;

        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.pNext = &flagsCI;
        // UPDATE_AFTER_BIND_POOL_BIT：池也需要支持 update_after_bind
        ci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        ci.bindingCount = 1;
        ci.pBindings = &binding;
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &ci, nullptr, &bindlessSetLayout_));

        std::cout << "✅ Bindless 描述符布局：binding=0，" << N_TEXTURES << " 个纹理（PARTIALLY_BOUND）\n";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 创建程序化纹理数组（每个纹理不同颜色）
    // ═══════════════════════════════════════════════════════════════════════

    void createTextureArray() {
        constexpr uint32_t TEX_W = 64, TEX_H = 64;

        // 创建采样器（所有纹理共享同一采样器）
        VkSamplerCreateInfo sampCI{};
        sampCI.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampCI.magFilter = VK_FILTER_LINEAR;
        sampCI.minFilter = VK_FILTER_LINEAR;
        sampCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        VK_CHECK(vkCreateSampler(device_, &sampCI, nullptr, &textureSampler_));

        textureImages_.resize(N_TEXTURES);
        textureMemories_.resize(N_TEXTURES);
        textureViews_.resize(N_TEXTURES);

        for (uint32_t t = 0; t < N_TEXTURES; ++t) {
            // 生成不同颜色的程序化纹理
            std::vector<uint8_t> pixels(TEX_W * TEX_H * 4);
            float hue = (float)t / N_TEXTURES; // 0..1 的色相值
            // 简单 HSV→RGB 转换
            float r = 0.5f + 0.5f * std::cos(2 * 3.14159f * (hue + 0.0f / 3));
            float g = 0.5f + 0.5f * std::cos(2 * 3.14159f * (hue + 1.0f / 3));
            float b = 0.5f + 0.5f * std::cos(2 * 3.14159f * (hue + 2.0f / 3));
            for (uint32_t y = 0; y < TEX_H; ++y) {
                for (uint32_t x = 0; x < TEX_W; ++x) {
                    bool checker = ((x / 8) + (y / 8)) % 2 == 0;
                    size_t idx = (y * TEX_W + x) * 4;
                    pixels[idx + 0] = static_cast<uint8_t>((checker ? r : r * 0.5f) * 255);
                    pixels[idx + 1] = static_cast<uint8_t>((checker ? g : g * 0.5f) * 255);
                    pixels[idx + 2] = static_cast<uint8_t>((checker ? b : b * 0.5f) * 255);
                    pixels[idx + 3] = 255;
                }
            }

            // 创建并上传纹理
            createAndUploadTexture(t, TEX_W, TEX_H, pixels);
        }
        std::cout << "✅ " << N_TEXTURES << " 个程序化纹理已创建\n";
    }

    void createAndUploadTexture(uint32_t idx, uint32_t w, uint32_t h, const std::vector<uint8_t>& pixels) {
        VkDeviceSize sz = w * h * 4;
        VkBuffer sb;
        VkDeviceMemory sm;
        createBuffer(sz,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     sb,
                     sm);
        void* d = nullptr;
        vkMapMemory(device_, sm, 0, sz, 0, &d);
        std::memcpy(d, pixels.data(), (size_t)sz);
        vkUnmapMemory(device_, sm);

        VkImageCreateInfo imgCI{};
        imgCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgCI.imageType = VK_IMAGE_TYPE_2D;
        imgCI.extent = {w, h, 1};
        imgCI.mipLevels = 1;
        imgCI.arrayLayers = 1;
        imgCI.format = VK_FORMAT_R8G8B8A8_SRGB;
        imgCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        imgCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imgCI.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imgCI.samples = VK_SAMPLE_COUNT_1_BIT;
        VK_CHECK(vkCreateImage(device_, &imgCI, nullptr, &textureImages_[idx]));

        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(device_, textureImages_[idx], &mr);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK(vkAllocateMemory(device_, &ai, nullptr, &textureMemories_[idx]));
        VK_CHECK(vkBindImageMemory(device_, textureImages_[idx], textureMemories_[idx], 0));

        // 布局转换 + 上传
        transitionImage(textureImages_[idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        copyBufferToImage(sb, textureImages_[idx], w, h);
        transitionImage(
            textureImages_[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        vkDestroyBuffer(device_, sb, nullptr);
        vkFreeMemory(device_, sm, nullptr);

        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = textureImages_[idx];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = VK_FORMAT_R8G8B8A8_SRGB;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(device_, &vci, nullptr, &textureViews_[idx]));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 将所有纹理写入一个描述符集（Bindless 核心）
    // ═══════════════════════════════════════════════════════════════════════

    void createBindlessDescriptorSet() {
        // 池需要包含 UPDATE_AFTER_BIND_BIT
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, N_TEXTURES};
        VkDescriptorPoolCreateInfo poolCI{};
        poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolCI.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT; // ← 必须！
        poolCI.maxSets = 1;
        poolCI.poolSizeCount = 1;
        poolCI.pPoolSizes = &ps;
        VK_CHECK(vkCreateDescriptorPool(device_, &poolCI, nullptr, &bindlessPool_));

        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = bindlessPool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &bindlessSetLayout_;
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, &bindlessSet_));

        // ── 一次性将所有纹理写入描述符集 ────────────────────────────────────
        // 这是 Bindless 的关键：一次 vkUpdateDescriptorSets 写入 N_TEXTURES 个纹理！
        std::vector<VkDescriptorImageInfo> imgInfos(N_TEXTURES);
        for (uint32_t i = 0; i < N_TEXTURES; ++i) {
            imgInfos[i].sampler = textureSampler_;
            imgInfos[i].imageView = textureViews_[i];
            imgInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = bindlessSet_;
        write.dstBinding = 0;
        write.dstArrayElement = 0;          // 从索引 0 开始写
        write.descriptorCount = N_TEXTURES; // 一次写入所有 N_TEXTURES 个！
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = imgInfos.data();
        vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);

        std::cout << "✅ Bindless 描述符集：" << N_TEXTURES << " 个纹理一次绑定完成\n";
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 创建顶点缓冲并获取 Buffer Device Address（GPU 端指针）
    // ═══════════════════════════════════════════════════════════════════════

    void createVertexBufferWithBDA() {
        VkDeviceSize sz = sizeof(QUAD_VERTICES[0]) * QUAD_VERTICES.size();

        // 创建缓冲时必须加 SHADER_DEVICE_ADDRESS_BIT
        createBuffer(sz,
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, // ← 必须！
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     vertexBuffer_,
                     vertexMemory_);

        void* d = nullptr;
        vkMapMemory(device_, vertexMemory_, 0, sz, 0, &d);
        std::memcpy(d, QUAD_VERTICES.data(), (size_t)sz);
        vkUnmapMemory(device_, vertexMemory_);

        // 获取 GPU 端地址（64-bit 指针）
        VkBufferDeviceAddressInfo addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addrInfo.buffer = vertexBuffer_;
        vertexBDA_ = vkGetBufferDeviceAddress(device_, &addrInfo);
        // 这个地址可以通过 push constant 传给着色器，着色器直接用它访问数据！

        std::cout << "✅ 顶点缓冲 BDA（Buffer Device Address）已获取\n";
    }

    void createGraphicsPipeline() {
        // bindless.vert：通过 Buffer Device Address 访问顶点，通过描述符数组访问纹理
        // bindless.frag：通过 nonuniformEXT 索引访问纹理数组
        VkShaderModule vert = createShaderModuleFromFile(device_, "bindless.vert.spv");
        VkShaderModule frag = createShaderModuleFromFile(device_, "bindless.frag.spv");

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

        // 顶点输入：空（顶点通过 BDA 在着色器内读取）
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

        // Push constant：含 Buffer Device Address（8 字节对齐！）
        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pcRange.offset = 0;
        pcRange.size = sizeof(DrawPushConstants);

        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &bindlessSetLayout_;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcRange;
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
        pi.pDynamicState = &dynS;
        pi.layout = pipelineLayout_;
        pi.renderPass = renderPass_;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline_));
        vkDestroyShaderModule(device_, frag, nullptr);
        vkDestroyShaderModule(device_, vert, nullptr);
        std::cout << "✅ Bindless 管线创建完成（push constants 含 64-bit BDA）\n";
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
        static auto start = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start).count();

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

        VkClearValue clear{};
        clear.color.float32[0] = 0.02f;
        clear.color.float32[2] = 0.05f;
        clear.color.float32[3] = 1.0f;
        VkRenderPassBeginInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp.renderPass = renderPass_;
        rp.framebuffer = framebuffers_[imageIndex];
        rp.renderArea = {{0, 0}, swapchainExtent_};
        rp.clearValueCount = 1;
        rp.pClearValues = &clear;
        vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        VkViewport vp{0, 0, (float)swapchainExtent_.width, (float)swapchainExtent_.height, 0, 1};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{{0, 0}, swapchainExtent_};
        vkCmdSetScissor(cmd, 0, 1, &sc);

        // ── 一次绑定 Bindless 描述符集（整个帧只绑定一次！） ─────────────────
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_, 0, 1, &bindlessSet_, 0, nullptr);

        // ── 绘制 N_OBJECTS 个物体，每个使用不同纹理 ─────────────────────────
        // 每次绘制只需更新 push constant（无描述符切换！）
        float aspect = (float)swapchainExtent_.width / swapchainExtent_.height;
        glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 50.0f);
        proj[1][1] *= -1;
        glm::mat4 view = glm::lookAt(glm::vec3(0, 0, 10), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));

        for (uint32_t i = 0; i < N_OBJECTS; ++i) {
            float col = (float)(i % 8) - 3.5f;
            float row = (float)(i / 8) - 2.0f;
            float angle = time * (1.0f + i * 0.1f);
            glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(col * 1.3f, row * 1.3f, 0));
            model = glm::rotate(model, angle, glm::vec3(0, 0, 1));
            model = glm::scale(model, glm::vec3(0.5f));

            DrawPushConstants pc{};
            pc.mvp = proj * view * model;
            pc.vertexBufferAddress = vertexBDA_; // GPU 指针
            pc.textureIndex = i % N_TEXTURES;    // 循环使用纹理
            pc.time = time;

            vkCmdPushConstants(
                cmd, pipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

            // 不需要 vkCmdBindVertexBuffers！顶点通过 BDA 在着色器内读取
            vkCmdDraw(cmd, 6, 1, 0, 0); // 6 顶点/四边形
        }

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
        std::cout << "🎨 " << N_OBJECTS << " 个四边形，" << N_TEXTURES << " 个纹理，0 次描述符切换！（ESC 退出）\n";
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
        // Buffer Device Address 需要特殊的分配标志
        VkMemoryAllocateFlagsInfo flagsInfo{};
        if (u & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
            flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
            flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
            ai.pNext = &flagsInfo;
        }
        ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, p);
        VK_CHECK(vkAllocateMemory(device_, &ai, nullptr, &m));
        VK_CHECK(vkBindBufferMemory(device_, b, m, 0));
    }
    void transitionImage(VkImage img, VkImageLayout from, VkImageLayout to) {
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
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = from;
        barrier.newLayout = to;
        barrier.image = img;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        VkPipelineStageFlags src = 0, dst = 0;
        if (from == VK_IMAGE_LAYOUT_UNDEFINED && to == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            src = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            dst = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (from == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && to == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            src = VK_PIPELINE_STAGE_TRANSFER_BIT;
            dst = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        } else
            throw std::invalid_argument("不支持的布局转换");
        vkCmdPipelineBarrier(cmd, src, dst, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        vkEndCommandBuffer(cmd);
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        vkQueueSubmit(graphicsQueue_, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue_);
        vkFreeCommandBuffers(device_, commandPool_, 1, &cmd);
    }
    void copyBufferToImage(VkBuffer buf, VkImage img, uint32_t w, uint32_t h) {
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
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {w, h, 1};
        vkCmdCopyBufferToImage(cmd, buf, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        vkEndCommandBuffer(cmd);
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        vkQueueSubmit(graphicsQueue_, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue_);
        vkFreeCommandBuffers(device_, commandPool_, 1, &cmd);
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

        // 启用 Descriptor Indexing + Buffer Device Address（Vulkan 1.2 核心）
        VkPhysicalDeviceDescriptorIndexingFeatures diFeatures{};
        diFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
        diFeatures.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
        diFeatures.descriptorBindingPartiallyBound = VK_TRUE;
        diFeatures.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
        diFeatures.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
        diFeatures.runtimeDescriptorArray = VK_TRUE; // 变长数组

        VkPhysicalDeviceBufferDeviceAddressFeatures bdaFeatures{};
        bdaFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
        bdaFeatures.bufferDeviceAddress = VK_TRUE;
        bdaFeatures.pNext = &diFeatures;

        VkPhysicalDeviceFeatures feat{};
        feat.samplerAnisotropy = VK_TRUE;
        VkDeviceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        ci.pNext = &bdaFeatures;
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
        vkDestroySampler(device_, textureSampler_, nullptr);
        for (uint32_t i = 0; i < N_TEXTURES; ++i) {
            vkDestroyImageView(device_, textureViews_[i], nullptr);
            vkDestroyImage(device_, textureImages_[i], nullptr);
            vkFreeMemory(device_, textureMemories_[i], nullptr);
        }
        vkDestroyBuffer(device_, vertexBuffer_, nullptr);
        vkFreeMemory(device_, vertexMemory_, nullptr);
        vkDestroyDescriptorPool(device_, bindlessPool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, bindlessSetLayout_, nullptr);
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
    std::cout << " 第22章：Bindless Rendering + Buffer Device Address\n";
    std::cout << "\n";
    std::cout << " 关键技术：\n";
    std::cout << "   • 变长描述符数组（PARTIALLY_BOUND + runtimeDescriptorArray）\n";
    std::cout << "   • 一次 vkCmdBindDescriptorSets 绑定所有纹理\n";
    std::cout << "   • per-draw Push Constants 传递 textureIndex\n";
    std::cout << "   • Buffer Device Address（64-bit GPU 指针，无需顶点绑定）\n";
    std::cout << "   • nonuniformEXT（着色器中非均匀索引）\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n\n";
    Ch22App app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
    return 0;
}
