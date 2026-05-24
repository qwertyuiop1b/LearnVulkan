/**
 * @file ch52_particles.cpp
 * @brief 第52章：游戏向 GPU 粒子系统
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【设计目标】
 *
 *  ch15 是"Compute Shader 入门"，粒子行为简单。
 *  本章是"游戏中如何用粒子"，关注：
 *    ① 发射器（Emitter）抽象：位置、方向、类型（点/锥形）、速率、颜色曲线
 *    ② 重生机制：死亡粒子在 Compute 中就地重生，无 CPU-GPU 往返
 *    ③ Billboard 展开：SSBO → 6 顶点/粒子，无需顶点缓冲区
 *    ④ 程序化软圆精灵：不依赖贴图文件，片段着色器生成
 *    ⑤ ImGui：实时调节每个发射器的参数
 *
 * 【管线】
 *   每帧：
 *     Compute Pass（particle2_update.comp）→ 更新 SSBO
 *     SSBO → Vertex stage barrier
 *     Graphics Pass → Billboard 渲染（Alpha 混合）
 *
 * 【三个发射器】
 *   0. 篝火（橙色锥形，向上）     — 位置 (0, 0, 0)
 *   1. 魔法（蓝色球形散射）        — 位置 (-2, 1, 0)
 *   2. 烟（灰色慢速向上漂移）      — 位置 (0, 0.5, 0)
 * ═══════════════════════════════════════════════════════════════════════════
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vulkan_tutorial/utils.hpp>
#include <vulkan_tutorial/vk_helpers.hpp>
#include <vulkan_tutorial/interactive_chapter.hpp>

#include <imgui.h>

#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using namespace vulkan_tutorial;

constexpr uint32_t WIDTH      = 800;
constexpr uint32_t HEIGHT     = 600;
constexpr int      MAX_FRAMES = 2;
constexpr uint32_t N_PARTICLES = 16384;     // 必须是 local_size_x(256) 的倍数
constexpr int      N_EMITTERS  = 3;

// ─── 数据结构（与着色器 GLSL 对齐）────────────────────────────────────────

struct GpuParticle {
    glm::vec4 position;    // xyz=pos, w=lifetime(当前)
    glm::vec4 velocity;    // xyz=vel, w=maxLifetime
    glm::vec4 color;       // rgba
    float     size;
    float     pad[3];
};

struct GpuEmitter {
    glm::vec4 position;     // xyz=pos, w=emitType(0=点,1=锥)
    glm::vec4 direction;    // xyz=朝向, w=spread(弧度)
    glm::vec4 colorMin;
    glm::vec4 colorMax;
    float speed;
    float speedVariance;
    float lifetime;
    float lifetimeVariance;
    float size;
    float emitRate;
    float pad[2];
};

struct EmitterUBO {
    GpuEmitter emitters[4];
    int        emitterCount;
    float      deltaTime;
    float      time;
    uint32_t   randomSeed;
};

struct CameraUBO {
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec3 cameraRight; float pad0;
    glm::vec3 cameraUp;    float pad1;
};

class Ch52App {
public:
    void run() { initWindow(); initVulkan(); mainLoop(); cleanup(); }

private:
    GLFWwindow*      window_  = nullptr;
    VkInstance       instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR     surface_  = VK_NULL_HANDLE;
    VkPhysicalDevice physDev_  = VK_NULL_HANDLE;
    VkDevice         device_   = VK_NULL_HANDLE;
    VkQueue          gQueue_   = VK_NULL_HANDLE;
    VkQueue          pQueue_   = VK_NULL_HANDLE;
    QueueFamilyIndices queueIdx_{};
    VkSwapchainKHR   swapchain_ = VK_NULL_HANDLE;
    std::vector<VkImage>       swapImages_;
    std::vector<VkImageView>   swapViews_;
    std::vector<VkFramebuffer> swapFBs_;
    VkFormat   swapFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D extent_{};
    DepthResources depth_{};
    VkCommandPool  cmdPool_ = VK_NULL_HANDLE;

    // 管线
    VkRenderPass     renderPass_      = VK_NULL_HANDLE;
    VkPipeline       computePipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout computeLayout_   = VK_NULL_HANDLE;
    VkDescriptorSetLayout computeDSL_ = VK_NULL_HANDLE;
    VkPipeline       drawPipeline_    = VK_NULL_HANDLE;
    VkPipelineLayout drawLayout_      = VK_NULL_HANDLE;
    VkDescriptorSetLayout drawDSL_    = VK_NULL_HANDLE;

    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> computeSets_;
    std::vector<VkDescriptorSet> drawSets_;

    // 粒子 SSBO（每帧一份，避免 in-flight 竞争）
    std::vector<VkBuffer>       particleBuffers_;
    std::vector<VkDeviceMemory> particleMem_;

    // UBO
    std::vector<VkBuffer>       emitterUBOs_;
    std::vector<VkDeviceMemory> emitterUBOMem_;
    std::vector<void*>          emitterMapped_;
    std::vector<VkBuffer>       cameraUBOs_;
    std::vector<VkDeviceMemory> cameraUBOMem_;
    std::vector<void*>          cameraMapped_;

    std::vector<VkCommandBuffer> cmdBuffers_;
    std::vector<VkSemaphore>     imageAvail_;
    std::vector<VkSemaphore>     renderDone_;
    std::vector<VkFence>         inFlight_;
    uint32_t currentFrame_ = 0;
    bool     resized_      = false;

    InteractiveChapterTools interactive_;

    // 发射器参数（ImGui 可调）
    GpuEmitter emitters_[N_EMITTERS];
    float      totalTime_ = 0.0f;
    float      deltaTime_ = 0.016f;

    // ─── 初始化 ─────────────────────────────────────────────────────────────

    void initWindow()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "第52章：GPU 粒子系统", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
            reinterpret_cast<Ch52App*>(glfwGetWindowUserPointer(w))->resized_ = true;
        });
        interactive_.attachInput(window_);
        initEmitters();
    }

    void initEmitters()
    {
        // 0: 篝火（橙黄锥形，向上）
        emitters_[0].position    = glm::vec4(0, 0, 0, 1);   // w=1: 锥形
        emitters_[0].direction   = glm::vec4(0, 1, 0, 0.4f); // spread=0.4 rad
        emitters_[0].colorMin    = glm::vec4(1.0f, 0.3f, 0.0f, 1.0f);
        emitters_[0].colorMax    = glm::vec4(1.0f, 0.8f, 0.1f, 1.0f);
        emitters_[0].speed       = 2.5f;
        emitters_[0].speedVariance = 0.8f;
        emitters_[0].lifetime    = 1.0f;
        emitters_[0].lifetimeVariance = 0.3f;
        emitters_[0].size        = 0.18f;

        // 1: 魔法球（蓝紫，球形散射）
        emitters_[1].position    = glm::vec4(-2, 1, 0, 1);
        emitters_[1].direction   = glm::vec4(0, 1, 0, 3.14159f); // 全向
        emitters_[1].colorMin    = glm::vec4(0.2f, 0.4f, 1.0f, 1.0f);
        emitters_[1].colorMax    = glm::vec4(0.8f, 0.2f, 1.0f, 1.0f);
        emitters_[1].speed       = 1.5f;
        emitters_[1].speedVariance = 1.0f;
        emitters_[1].lifetime    = 1.2f;
        emitters_[1].lifetimeVariance = 0.5f;
        emitters_[1].size        = 0.12f;

        // 2: 烟（灰白，慢速向上漂移）
        emitters_[2].position    = glm::vec4(0, 0.3f, 0, 1);
        emitters_[2].direction   = glm::vec4(0, 1, 0, 0.2f);
        emitters_[2].colorMin    = glm::vec4(0.5f, 0.5f, 0.5f, 0.6f);
        emitters_[2].colorMax    = glm::vec4(0.8f, 0.8f, 0.8f, 0.3f);
        emitters_[2].speed       = 0.5f;
        emitters_[2].speedVariance = 0.2f;
        emitters_[2].lifetime    = 2.5f;
        emitters_[2].lifetimeVariance = 0.5f;
        emitters_[2].size        = 0.3f;
    }

    void initVulkan()
    {
        createInstance(instance_);
        VK_CHECK(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_));
        pickPhysicalDevice(instance_, surface_, physDev_);
        createLogicalDevice(physDev_, surface_, device_, gQueue_, pQueue_, queueIdx_);
        createSwapchain();
        createImageViews();
        depth_ = createDepthResources(physDev_, device_, extent_);
        createCommandPool();
        createRenderPass();
        createFramebuffers();
        createParticleBuffers();
        createUniformBuffers();
        createDescriptorLayouts();
        createDescriptorPool();
        createDescriptorSets();
        createPipelines();
        createCommandBuffers();
        createSyncObjects();

        InteractiveInitInfo ii{};
        ii.window = window_; ii.instance = instance_;
        ii.physicalDevice = physDev_; ii.device = device_;
        ii.graphicsQueue = gQueue_;
        ii.queueFamily = queueIdx_.graphicsFamily.value();
        ii.renderPass = renderPass_;
        ii.swapchainFormat = swapFormat_;
        ii.imageCount = static_cast<uint32_t>(swapImages_.size());
        ii.maxFramesInFlight = MAX_FRAMES;
        interactive_.initVulkan(ii);
        interactive_.camera().setDistance(8.0f);
        interactive_.camera().setAngles(45.0f, 20.0f);
    }

    void createSwapchain()
    {
        auto details = querySwapChainSupport(physDev_, surface_);
        auto fmt     = chooseSwapSurfaceFormat(details.formats);
        auto mode    = chooseSwapPresentMode(details.presentModes);
        extent_      = chooseSwapExtent(details.capabilities, window_);
        swapFormat_  = fmt.format;
        uint32_t cnt = details.capabilities.minImageCount + 1;
        if (details.capabilities.maxImageCount > 0)
            cnt = std::min(cnt, details.capabilities.maxImageCount);
        VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        ci.surface          = surface_;
        ci.minImageCount    = cnt;
        ci.imageFormat      = fmt.format;
        ci.imageColorSpace  = fmt.colorSpace;
        ci.imageExtent      = extent_;
        ci.imageArrayLayers = 1;
        ci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        uint32_t qf[2]      = {queueIdx_.graphicsFamily.value(), queueIdx_.presentFamily.value()};
        if (qf[0] != qf[1]) {
            ci.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
            ci.queueFamilyIndexCount = 2;
            ci.pQueueFamilyIndices   = qf;
        } else { ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE; }
        ci.preTransform   = details.capabilities.currentTransform;
        ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        ci.presentMode    = mode;
        ci.clipped        = VK_TRUE;
        VK_CHECK(vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_));
        uint32_t imgCnt = 0;
        vkGetSwapchainImagesKHR(device_, swapchain_, &imgCnt, nullptr);
        swapImages_.resize(imgCnt);
        vkGetSwapchainImagesKHR(device_, swapchain_, &imgCnt, swapImages_.data());
    }

    void createImageViews()
    {
        swapViews_.resize(swapImages_.size());
        for (size_t i = 0; i < swapImages_.size(); ++i) {
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image    = swapImages_[i];
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format   = swapFormat_;
            vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VK_CHECK(vkCreateImageView(device_, &vi, nullptr, &swapViews_[i]));
        }
    }

    void createCommandPool()
    {
        VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        ci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = queueIdx_.graphicsFamily.value();
        VK_CHECK(vkCreateCommandPool(device_, &ci, nullptr, &cmdPool_));
    }

    void createRenderPass()
    {
        std::array<VkAttachmentDescription, 2> atts{};
        atts[0].format         = swapFormat_;
        atts[0].samples        = VK_SAMPLE_COUNT_1_BIT;
        atts[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        atts[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        atts[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        atts[0].finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        atts[1].format         = depth_.format;
        atts[1].samples        = VK_SAMPLE_COUNT_1_BIT;
        atts[1].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        atts[1].storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        atts[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        atts[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        atts[1].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        atts[1].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorRef = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depthRef = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription  sub{};
        sub.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.colorAttachmentCount    = 1;
        sub.pColorAttachments       = &colorRef;
        sub.pDepthStencilAttachment = &depthRef;

        VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        rpci.attachmentCount = static_cast<uint32_t>(atts.size());
        rpci.pAttachments    = atts.data();
        rpci.subpassCount    = 1;
        rpci.pSubpasses      = &sub;
        VK_CHECK(vkCreateRenderPass(device_, &rpci, nullptr, &renderPass_));
    }

    void createFramebuffers()
    {
        swapFBs_.resize(swapImages_.size());
        for (size_t i = 0; i < swapImages_.size(); ++i) {
            std::array<VkImageView, 2> atts = {swapViews_[i], depth_.view};
            VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            fci.renderPass      = renderPass_;
            fci.attachmentCount = static_cast<uint32_t>(atts.size());
            fci.pAttachments    = atts.data();
            fci.width  = extent_.width;
            fci.height = extent_.height;
            fci.layers = 1;
            VK_CHECK(vkCreateFramebuffer(device_, &fci, nullptr, &swapFBs_[i]));
        }
    }

    void createParticleBuffers()
    {
        const VkDeviceSize size = sizeof(GpuParticle) * N_PARTICLES;
        particleBuffers_.resize(MAX_FRAMES);
        particleMem_.resize(MAX_FRAMES);

        // 初始化粒子（全部死亡，Compute Shader 会在首帧重生它们）
        std::vector<GpuParticle> init(N_PARTICLES);
        for (auto& p : init)
            p.position.w = -1.0f;   // lifetime < 0 → 死亡，触发重生

        VkBuffer stagingBuf; VkDeviceMemory stagingMem;
        createBuffer(physDev_, device_, size,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuf, stagingMem);
        void* data;
        vkMapMemory(device_, stagingMem, 0, size, 0, &data);
        std::memcpy(data, init.data(), size);
        vkUnmapMemory(device_, stagingMem);

        for (int i = 0; i < MAX_FRAMES; ++i) {
            createBuffer(physDev_, device_, size,
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT
                         | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                         | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                         particleBuffers_[i], particleMem_[i]);
            copyBuffer(device_, cmdPool_, gQueue_, stagingBuf, particleBuffers_[i], size);
        }
        vkDestroyBuffer(device_, stagingBuf, nullptr);
        vkFreeMemory(device_, stagingMem, nullptr);
    }

    void createUniformBuffers()
    {
        emitterUBOs_.resize(MAX_FRAMES); emitterUBOMem_.resize(MAX_FRAMES); emitterMapped_.resize(MAX_FRAMES);
        cameraUBOs_.resize(MAX_FRAMES);  cameraUBOMem_.resize(MAX_FRAMES);  cameraMapped_.resize(MAX_FRAMES);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            createBuffer(physDev_, device_, sizeof(EmitterUBO),
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         emitterUBOs_[i], emitterUBOMem_[i]);
            vkMapMemory(device_, emitterUBOMem_[i], 0, sizeof(EmitterUBO), 0, &emitterMapped_[i]);
            createBuffer(physDev_, device_, sizeof(CameraUBO),
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         cameraUBOs_[i], cameraUBOMem_[i]);
            vkMapMemory(device_, cameraUBOMem_[i], 0, sizeof(CameraUBO), 0, &cameraMapped_[i]);
        }
    }

    void createDescriptorLayouts()
    {
        // Compute DSL：SSBO + EmitterUBO
        std::array<VkDescriptorSetLayoutBinding, 2> compBs = {{
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
            {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
        }};
        VkDescriptorSetLayoutCreateInfo dci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dci.bindingCount = 2; dci.pBindings = compBs.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &dci, nullptr, &computeDSL_));

        // Draw DSL：SSBO + CameraUBO
        std::array<VkDescriptorSetLayoutBinding, 2> drawBs = {{
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT},
            {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT},
        }};
        dci.pBindings = drawBs.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &dci, nullptr, &drawDSL_));
    }

    void createDescriptorPool()
    {
        std::array<VkDescriptorPoolSize, 2> sizes = {{
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, MAX_FRAMES * 2},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, MAX_FRAMES * 2},
        }};
        VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        ci.poolSizeCount = static_cast<uint32_t>(sizes.size());
        ci.pPoolSizes    = sizes.data();
        ci.maxSets       = MAX_FRAMES * 2;
        VK_CHECK(vkCreateDescriptorPool(device_, &ci, nullptr, &pool_));
    }

    void createDescriptorSets()
    {
        auto allocSets = [&](VkDescriptorSetLayout dsl, std::vector<VkDescriptorSet>& sets) {
            std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES, dsl);
            VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ai.descriptorPool     = pool_;
            ai.descriptorSetCount = MAX_FRAMES;
            ai.pSetLayouts        = layouts.data();
            sets.resize(MAX_FRAMES);
            VK_CHECK(vkAllocateDescriptorSets(device_, &ai, sets.data()));
        };
        allocSets(computeDSL_, computeSets_);
        allocSets(drawDSL_, drawSets_);

        for (int i = 0; i < MAX_FRAMES; ++i) {
            VkDescriptorBufferInfo ssboInfo{particleBuffers_[i], 0, VK_WHOLE_SIZE};
            VkDescriptorBufferInfo emitInfo{emitterUBOs_[i], 0, sizeof(EmitterUBO)};
            VkDescriptorBufferInfo camInfo {cameraUBOs_[i],  0, sizeof(CameraUBO)};

            std::array<VkWriteDescriptorSet, 2> compW = {{
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, computeSets_[i], 0, 0, 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &ssboInfo},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, computeSets_[i], 1, 0, 1,
                 VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &emitInfo},
            }};
            vkUpdateDescriptorSets(device_, 2, compW.data(), 0, nullptr);

            std::array<VkWriteDescriptorSet, 2> drawW = {{
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, drawSets_[i], 0, 0, 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &ssboInfo},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, drawSets_[i], 1, 0, 1,
                 VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &camInfo},
            }};
            vkUpdateDescriptorSets(device_, 2, drawW.data(), 0, nullptr);
        }
    }

    void createPipelines()
    {
        // ── Compute pipeline（粒子更新）──
        {
            VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            lci.setLayoutCount = 1;
            lci.pSetLayouts    = &computeDSL_;
            VK_CHECK(vkCreatePipelineLayout(device_, &lci, nullptr, &computeLayout_));

            auto comp = createShaderModuleFromFile(device_, "particle2_update.comp.spv");
            VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, comp, "main"};
            VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            ci.stage  = stage;
            ci.layout = computeLayout_;
            VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &ci, nullptr, &computePipeline_));
            vkDestroyShaderModule(device_, comp, nullptr);
        }

        // ── Draw pipeline（Billboard 粒子）──
        {
            VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            lci.setLayoutCount = 1;
            lci.pSetLayouts    = &drawDSL_;
            VK_CHECK(vkCreatePipelineLayout(device_, &lci, nullptr, &drawLayout_));

            auto vert = createShaderModuleFromFile(device_, "particle2.vert.spv");
            auto frag = createShaderModuleFromFile(device_, "particle2.frag.spv");

            VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            vps.viewportCount = 1; vps.scissorCount = 1;

            VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rs.polygonMode = VK_POLYGON_MODE_FILL;
            rs.cullMode    = VK_CULL_MODE_NONE;   // Billboard 双面
            rs.lineWidth   = 1.0f;

            VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            ds.depthTestEnable  = VK_TRUE;
            ds.depthWriteEnable = VK_FALSE;   // 粒子不写深度（半透明叠加）
            ds.depthCompareOp   = VK_COMPARE_OP_LESS;

            // Alpha 加法混合（适合火焰/魔法）
            VkPipelineColorBlendAttachmentState blend{};
            blend.blendEnable         = VK_TRUE;
            blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;   // 加法 → 叠加发光感
            blend.colorBlendOp        = VK_BLEND_OP_ADD;
            blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            blend.alphaBlendOp        = VK_BLEND_OP_ADD;
            blend.colorWriteMask      = 0xF;

            VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            cbs.attachmentCount = 1; cbs.pAttachments = &blend;

            std::array<VkDynamicState, 2> dyn = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dyns{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dyns.dynamicStateCount = 2; dyns.pDynamicStates = dyn.data();

            std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
            stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                         nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vert, "main"};
            stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                         nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, frag, "main"};

            VkGraphicsPipelineCreateInfo gci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            gci.stageCount = 2; gci.pStages = stages.data();
            gci.pVertexInputState = &vi; gci.pInputAssemblyState = &ia;
            gci.pViewportState = &vps; gci.pRasterizationState = &rs;
            gci.pMultisampleState = &ms; gci.pDepthStencilState = &ds;
            gci.pColorBlendState = &cbs; gci.pDynamicState = &dyns;
            gci.layout = drawLayout_; gci.renderPass = renderPass_;
            VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gci, nullptr, &drawPipeline_));
            vkDestroyShaderModule(device_, vert, nullptr);
            vkDestroyShaderModule(device_, frag, nullptr);
        }
    }

    void createCommandBuffers()
    {
        cmdBuffers_.resize(MAX_FRAMES);
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool        = cmdPool_;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = MAX_FRAMES;
        VK_CHECK(vkAllocateCommandBuffers(device_, &ai, cmdBuffers_.data()));
    }

    void createSyncObjects()
    {
        imageAvail_.resize(MAX_FRAMES); renderDone_.resize(MAX_FRAMES); inFlight_.resize(MAX_FRAMES);
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VkFenceCreateInfo     fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for (int i = 0; i < MAX_FRAMES; ++i) {
            VK_CHECK(vkCreateSemaphore(device_, &si, nullptr, &imageAvail_[i]));
            VK_CHECK(vkCreateSemaphore(device_, &si, nullptr, &renderDone_[i]));
            VK_CHECK(vkCreateFence(device_, &fi, nullptr, &inFlight_[i]));
        }
    }

    // ─── 主循环 ──────────────────────────────────────────────────────────────

    void mainLoop()
    {
        auto lastTime = std::chrono::high_resolution_clock::now();
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            auto now  = std::chrono::high_resolution_clock::now();
            deltaTime_ = std::chrono::duration<float>(now - lastTime).count();
            lastTime   = now;
            totalTime_ += deltaTime_;

            interactive_.beginFrame(deltaTime_);
            buildUi();
            drawFrame();
            interactive_.updateWindowTitle();
        }
        vkDeviceWaitIdle(device_);
    }

    void buildUi()
    {
        interactive_.buildDebugPanel("第52章：GPU 粒子系统");
        ImGui::Separator();
        ImGui::Text("粒子总数：%u", N_PARTICLES);
        ImGui::Separator();
        const char* emitterNames[] = {"篝火（锥形）", "魔法（全向）", "烟（慢漂）"};
        for (int i = 0; i < N_EMITTERS; ++i) {
            if (ImGui::TreeNode(emitterNames[i])) {
                auto& e = emitters_[i];
                glm::vec3 pos = glm::vec3(e.position);
                if (ImGui::DragFloat3("位置", &pos.x, 0.05f))
                    e.position = glm::vec4(pos, e.position.w);
                ImGui::SliderFloat("速度",   &e.speed,     0.1f, 5.0f);
                ImGui::SliderFloat("寿命",   &e.lifetime,  0.2f, 5.0f);
                ImGui::SliderFloat("大小",   &e.size,      0.02f, 0.5f);
                float spread = e.direction.w;
                if (ImGui::SliderFloat("扩散角(rad)", &spread, 0.01f, 3.14159f))
                    e.direction.w = spread;
                glm::vec3 cMin = glm::vec3(e.colorMin);
                glm::vec3 cMax = glm::vec3(e.colorMax);
                if (ImGui::ColorEdit3("最小颜色", &cMin.x))
                    e.colorMin = glm::vec4(cMin, e.colorMin.w);
                if (ImGui::ColorEdit3("最大颜色", &cMax.x))
                    e.colorMax = glm::vec4(cMax, e.colorMax.w);
                ImGui::TreePop();
            }
        }
    }

    void drawFrame()
    {
        vkWaitForFences(device_, 1, &inFlight_[currentFrame_], VK_TRUE, UINT64_MAX);
        uint32_t imageIdx = 0;
        VkResult result = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX,
                                                imageAvail_[currentFrame_],
                                                VK_NULL_HANDLE, &imageIdx);
        if (result == VK_ERROR_OUT_OF_DATE_KHR) { recreateSwapchain(); return; }
        vkResetFences(device_, 1, &inFlight_[currentFrame_]);
        updateUBOs(currentFrame_);

        VkCommandBuffer cmd = cmdBuffers_[currentFrame_];
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
        interactive_.beginGpuSection(cmd, currentFrame_);

        // ── Compute Pass ──
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                computeLayout_, 0, 1, &computeSets_[currentFrame_], 0, nullptr);
        vkCmdDispatch(cmd, N_PARTICLES / 256, 1, 1);

        // SSBO：Compute write → Vertex read 屏障
        VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        bmb.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
        bmb.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;   // SSBO read in vertex stage
        bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb.buffer              = particleBuffers_[currentFrame_];
        bmb.offset              = 0;
        bmb.size                = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
            0, 0, nullptr, 1, &bmb, 0, nullptr);

        // ── Graphics Pass ──
        VkClearValue clears[2]{};
        clears[0].color.float32[0] = 0.02f; clears[0].color.float32[1] = 0.02f;
        clears[0].color.float32[2] = 0.05f; clears[0].color.float32[3] = 1.0f;
        clears[1].depthStencil.depth = 1.0f;
        VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rbi.renderPass        = renderPass_;
        rbi.framebuffer       = swapFBs_[imageIdx];
        rbi.renderArea.extent = extent_;
        rbi.clearValueCount   = 2;
        rbi.pClearValues      = clears;
        vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport vp{0,0,float(extent_.width),float(extent_.height),0,1};
        VkRect2D   sc{{0,0},extent_};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, drawPipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                drawLayout_, 0, 1, &drawSets_[currentFrame_], 0, nullptr);
        vkCmdDraw(cmd, N_PARTICLES * 6, 1, 0, 0);

        interactive_.renderUi(cmd);
        vkCmdEndRenderPass(cmd);
        interactive_.endGpuSection(cmd, currentFrame_);
        VK_CHECK(vkEndCommandBuffer(cmd));

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        VkPipelineStageFlags wait = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        si.waitSemaphoreCount   = 1;
        si.pWaitSemaphores      = &imageAvail_[currentFrame_];
        si.pWaitDstStageMask    = &wait;
        si.commandBufferCount   = 1;
        si.pCommandBuffers      = &cmd;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores    = &renderDone_[currentFrame_];
        VK_CHECK(vkQueueSubmit(gQueue_, 1, &si, inFlight_[currentFrame_]));

        VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores    = &renderDone_[currentFrame_];
        pi.swapchainCount     = 1;
        pi.pSwapchains        = &swapchain_;
        pi.pImageIndices      = &imageIdx;
        result = vkQueuePresentKHR(pQueue_, &pi);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || resized_) {
            resized_ = false; recreateSwapchain();
        }
        interactive_.endFrame(currentFrame_);
        currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES;
    }

    void updateUBOs(uint32_t fi)
    {
        EmitterUBO eu{};
        for (int i = 0; i < N_EMITTERS; ++i)
            eu.emitters[i] = emitters_[i];
        eu.emitterCount = N_EMITTERS;
        eu.deltaTime    = deltaTime_;
        eu.time         = totalTime_;
        eu.randomSeed   = static_cast<uint32_t>(totalTime_ * 1000.0f);
        std::memcpy(emitterMapped_[fi], &eu, sizeof(eu));

        glm::mat4 view = interactive_.camera().viewMatrix();
        glm::mat4 proj = glm::perspective(glm::radians(45.0f),
                                          float(extent_.width) / float(extent_.height), 0.1f, 100.0f);
        proj[1][1] *= -1;
        CameraUBO cu{};
        cu.view        = view;
        cu.proj        = proj;
        // 从 view 矩阵提取相机右向量和上向量（用于 Billboard）
        cu.cameraRight = glm::vec3(view[0][0], view[1][0], view[2][0]);
        cu.cameraUp    = glm::vec3(view[0][1], view[1][1], view[2][1]);
        std::memcpy(cameraMapped_[fi], &cu, sizeof(cu));
    }

    void recreateSwapchain()
    {
        int w = 0, h = 0;
        glfwGetFramebufferSize(window_, &w, &h);
        while (w == 0 || h == 0) { glfwGetFramebufferSize(window_, &w, &h); glfwWaitEvents(); }
        vkDeviceWaitIdle(device_);
        for (auto fb : swapFBs_) vkDestroyFramebuffer(device_, fb, nullptr);
        swapFBs_.clear();
        destroyDepthResources(device_, depth_);
        for (auto v : swapViews_) vkDestroyImageView(device_, v, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        createSwapchain();
        createImageViews();
        depth_ = createDepthResources(physDev_, device_, extent_);
        createFramebuffers();
        interactive_.onSwapchainRecreated(renderPass_, swapFormat_,
                                          static_cast<uint32_t>(swapImages_.size()));
    }

    void cleanup()
    {
        vkDestroyDescriptorPool(device_, pool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, computeDSL_, nullptr);
        vkDestroyDescriptorSetLayout(device_, drawDSL_, nullptr);
        vkDestroyPipeline(device_, computePipeline_, nullptr);
        vkDestroyPipeline(device_, drawPipeline_, nullptr);
        vkDestroyPipelineLayout(device_, computeLayout_, nullptr);
        vkDestroyPipelineLayout(device_, drawLayout_, nullptr);
        vkDestroyRenderPass(device_, renderPass_, nullptr);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroyBuffer(device_, particleBuffers_[i], nullptr);
            vkFreeMemory(device_, particleMem_[i], nullptr);
            vkDestroyBuffer(device_, emitterUBOs_[i], nullptr);
            vkFreeMemory(device_, emitterUBOMem_[i], nullptr);
            vkDestroyBuffer(device_, cameraUBOs_[i], nullptr);
            vkFreeMemory(device_, cameraUBOMem_[i], nullptr);
            vkDestroySemaphore(device_, imageAvail_[i], nullptr);
            vkDestroySemaphore(device_, renderDone_[i], nullptr);
            vkDestroyFence(device_, inFlight_[i], nullptr);
        }
        vkDestroyCommandPool(device_, cmdPool_, nullptr);
        destroyDepthResources(device_, depth_);
        for (auto fb : swapFBs_) vkDestroyFramebuffer(device_, fb, nullptr);
        for (auto v : swapViews_) vkDestroyImageView(device_, v, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        interactive_.shutdown(device_);
        vkDestroyDevice(device_, nullptr);
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        vkDestroyInstance(instance_, nullptr);
        glfwDestroyWindow(window_);
        glfwTerminate();
    }
};

int main()
{
    std::cout << "══════════════════════════════════════════════════\n";
    std::cout << " 第52章：游戏向 GPU 粒子系统\n";
    std::cout << "══════════════════════════════════════════════════\n\n";
    std::cout << "粒子数量：" << N_PARTICLES << "\n";
    std::cout << "发射器：篝火 + 魔法球 + 烟\n";
    std::cout << "控制：LMB 旋转 | RMB 平移 | 滚轮缩放 | ESC 退出\n\n";
    try {
        Ch52App app;
        app.run();
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
