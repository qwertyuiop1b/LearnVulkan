/**
 * @file ch54_volumetric_fog.cpp
 * @brief 第54章：体积雾（高度指数雾 + God Rays）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【管线 · 2 个 Pass】
 *
 *  Pass 1: Scene → HDR 颜色 + 深度（共享深度附件）
 *  Pass 2: Fullscreen Fog（vol_fog.frag）
 *    - 从深度重建世界坐标
 *    - 指数高度雾：density ∝ exp(-falloff * y) * dist
 *    - 可选 God Rays：Radial Blur 沿太阳方向叠加散射光
 *
 * 【ImGui 控制】
 *   雾密度、高度衰减系数、雾颜色、God Rays 强度开关
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
#include <vector>

using namespace vulkan_tutorial;

constexpr uint32_t WIDTH     = 800;
constexpr uint32_t HEIGHT    = 600;
constexpr int      MAX_FRAMES = 2;

struct SceneVertex { glm::vec3 pos; glm::vec3 normal; glm::vec3 color; };
struct SceneUBO    { alignas(16) glm::mat4 model, view, proj; };

static const std::vector<SceneVertex> SCENE = {
    // 地面
    {{-8,-0.5f,-8},{0,1,0},{0.4f,0.38f,0.3f}}, {{ 8,-0.5f,-8},{0,1,0},{0.4f,0.38f,0.3f}},
    {{ 8,-0.5f, 8},{0,1,0},{0.4f,0.38f,0.3f}}, {{-8,-0.5f,-8},{0,1,0},{0.4f,0.38f,0.3f}},
    {{ 8,-0.5f, 8},{0,1,0},{0.4f,0.38f,0.3f}}, {{-8,-0.5f, 8},{0,1,0},{0.4f,0.38f,0.3f}},
    // 前排树桩
    {{-2,-0.5f,-2},{0,0,-1},{0.6f,0.35f,0.2f}}, {{ 2,-0.5f,-2},{0,0,-1},{0.6f,0.35f,0.2f}},
    {{ 2, 2.5f,-2},{0,0,-1},{0.6f,0.35f,0.2f}}, {{-2,-0.5f,-2},{0,0,-1},{0.6f,0.35f,0.2f}},
    {{ 2, 2.5f,-2},{0,0,-1},{0.6f,0.35f,0.2f}}, {{-2, 2.5f,-2},{0,0,-1},{0.6f,0.35f,0.2f}},
    // 远处柱（雾中若隐若现）
    {{-5,-0.5f,-6},{0,0,-1},{0.5f,0.5f,0.55f}}, {{-4,-0.5f,-6},{0,0,-1},{0.5f,0.5f,0.55f}},
    {{-4, 3.5f,-6},{0,0,-1},{0.5f,0.5f,0.55f}}, {{-5,-0.5f,-6},{0,0,-1},{0.5f,0.5f,0.55f}},
    {{-4, 3.5f,-6},{0,0,-1},{0.5f,0.5f,0.55f}}, {{-5, 3.5f,-6},{0,0,-1},{0.5f,0.5f,0.55f}},
    {{ 4,-0.5f,-7},{0,0,-1},{0.5f,0.5f,0.55f}}, {{ 5,-0.5f,-7},{0,0,-1},{0.5f,0.5f,0.55f}},
    {{ 5, 4.0f,-7},{0,0,-1},{0.5f,0.5f,0.55f}}, {{ 4,-0.5f,-7},{0,0,-1},{0.5f,0.5f,0.55f}},
    {{ 5, 4.0f,-7},{0,0,-1},{0.5f,0.5f,0.55f}}, {{ 4, 4.0f,-7},{0,0,-1},{0.5f,0.5f,0.55f}},
};

// Fog push constant（与 vol_fog.frag 对齐）
struct FogPC {
    glm::mat4 invProjView;
    glm::vec4 cameraPos;
    glm::vec4 fogColor;
    glm::vec4 sunDir;
    float     fogDensity;
    float     fogHeightFalloff;
    float     fogStart;
    int       enableGodRays;
};

class Ch54App {
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

    // HDR offscreen（场景先渲染到这里）
    VkImage        hdrImage_ = VK_NULL_HANDLE;
    VkDeviceMemory hdrMem_   = VK_NULL_HANDLE;
    VkImageView    hdrView_  = VK_NULL_HANDLE;
    VkFramebuffer  sceneFB_  = VK_NULL_HANDLE;

    VkRenderPass  sceneRP_   = VK_NULL_HANDLE;
    VkRenderPass  fogRP_     = VK_NULL_HANDLE;

    VkPipeline       scenePipeline_  = VK_NULL_HANDLE;
    VkPipelineLayout sceneLayout_    = VK_NULL_HANDLE;
    VkDescriptorSetLayout sceneDSL_  = VK_NULL_HANDLE;
    VkPipeline       fogPipeline_    = VK_NULL_HANDLE;
    VkPipelineLayout fogLayout_      = VK_NULL_HANDLE;
    VkDescriptorSetLayout fogDSL_    = VK_NULL_HANDLE;

    VkDescriptorPool             pool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> sceneSets_;
    std::vector<VkDescriptorSet> fogSets_;
    VkSampler                    sampler_ = VK_NULL_HANDLE;

    VkBuffer       vb_  = VK_NULL_HANDLE;
    VkDeviceMemory vbMem_ = VK_NULL_HANDLE;
    std::vector<VkBuffer>       ubos_;
    std::vector<VkDeviceMemory> uboMem_;
    std::vector<void*>          uboMapped_;

    std::vector<VkCommandBuffer> cmds_;
    std::vector<VkSemaphore>     imgAvail_, renderDone_;
    std::vector<VkFence>         inFlight_;
    uint32_t frame_ = 0;
    bool resized_   = false;

    InteractiveChapterTools interactive_;

    // 调节参数
    float fogDensity_      = 0.18f;
    float fogFalloff_      = 0.4f;
    glm::vec3 fogColor_    = {0.7f, 0.75f, 0.85f};
    float godRayStrength_  = 0.4f;
    bool  enableGodRays_   = true;

    // ─── 初始化 ─────────────────────────────────────────────────────────────

    void initWindow()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "第54章：体积雾", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
            reinterpret_cast<Ch54App*>(glfwGetWindowUserPointer(w))->resized_ = true;
        });
        interactive_.attachInput(window_);
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
        createSampler();
        createOffscreenImage();
        createRenderPasses();
        createFramebuffers();
        createVertexBuffer();
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
        ii.renderPass = fogRP_; ii.swapchainFormat = swapFormat_;
        ii.imageCount = static_cast<uint32_t>(swapImages_.size());
        ii.maxFramesInFlight = MAX_FRAMES;
        interactive_.initVulkan(ii);
        interactive_.camera().setDistance(8.0f);
        interactive_.camera().setAngles(0.0f, 15.0f);
    }

    void createSwapchain()
    {
        auto details = querySwapChainSupport(physDev_, surface_);
        auto fmt = chooseSwapSurfaceFormat(details.formats);
        auto mode = chooseSwapPresentMode(details.presentModes);
        extent_ = chooseSwapExtent(details.capabilities, window_);
        swapFormat_ = fmt.format;
        uint32_t cnt = std::min(details.capabilities.minImageCount + 1,
                                details.capabilities.maxImageCount > 0
                                    ? details.capabilities.maxImageCount : UINT32_MAX);
        VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        ci.surface = surface_; ci.minImageCount = cnt;
        ci.imageFormat = fmt.format; ci.imageColorSpace = fmt.colorSpace;
        ci.imageExtent = extent_; ci.imageArrayLayers = 1;
        ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        uint32_t qf[2] = {queueIdx_.graphicsFamily.value(), queueIdx_.presentFamily.value()};
        if (qf[0] != qf[1]) { ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            ci.queueFamilyIndexCount = 2; ci.pQueueFamilyIndices = qf; }
        else ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ci.preTransform = details.capabilities.currentTransform;
        ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        ci.presentMode = mode; ci.clipped = VK_TRUE;
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
            vi.image = swapImages_[i]; vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format = swapFormat_; vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
            VK_CHECK(vkCreateImageView(device_, &vi, nullptr, &swapViews_[i]));
        }
    }

    void createCommandPool()
    {
        VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = queueIdx_.graphicsFamily.value();
        VK_CHECK(vkCreateCommandPool(device_, &ci, nullptr, &cmdPool_));
    }

    void createSampler()
    {
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter = VK_FILTER_LINEAR; si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = 1.0f;
        VK_CHECK(vkCreateSampler(device_, &si, nullptr, &sampler_));
    }

    void createOffscreenImage()
    {
        createImage(physDev_, device_, extent_.width, extent_.height, swapFormat_,
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, hdrImage_, hdrMem_);
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = hdrImage_; vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = swapFormat_; vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
        VK_CHECK(vkCreateImageView(device_, &vi, nullptr, &hdrView_));

        // 初始 layout 转换
        VkCommandBuffer cmd = beginSingleTimeCommands(device_, cmdPool_);
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = hdrImage_;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,nullptr,0,nullptr,1,&b);
        endSingleTimeCommands(device_, cmdPool_, gQueue_, cmd);
    }

    void createRenderPasses()
    {
        // Scene pass（输出到 HDR 离屏 + 共享深度）
        {
            std::array<VkAttachmentDescription, 2> atts{};
            atts[0].format = swapFormat_; atts[0].samples = VK_SAMPLE_COUNT_1_BIT;
            atts[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            atts[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            atts[0].initialLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            atts[0].finalLayout   = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            atts[1].format = depth_.format; atts[1].samples = VK_SAMPLE_COUNT_1_BIT;
            atts[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            atts[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;  // 深度需要给雾 pass 采样
            atts[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            atts[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            atts[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            atts[1].finalLayout   = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            VkAttachmentReference cr = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
            VkAttachmentReference dr = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
            VkSubpassDescription sub{};
            sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            sub.colorAttachmentCount = 1; sub.pColorAttachments = &cr;
            sub.pDepthStencilAttachment = &dr;
            VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
            rpci.attachmentCount = 2; rpci.pAttachments = atts.data();
            rpci.subpassCount = 1; rpci.pSubpasses = &sub;
            VK_CHECK(vkCreateRenderPass(device_, &rpci, nullptr, &sceneRP_));
        }
        // Fog pass（输出到交换链）
        {
            VkAttachmentDescription att{};
            att.format = swapFormat_; att.samples = VK_SAMPLE_COUNT_1_BIT;
            att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            att.finalLayout   = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            VkAttachmentReference cr = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
            VkSubpassDescription sub{};
            sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            sub.colorAttachmentCount = 1; sub.pColorAttachments = &cr;
            VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
            rpci.attachmentCount = 1; rpci.pAttachments = &att;
            rpci.subpassCount = 1; rpci.pSubpasses = &sub;
            VK_CHECK(vkCreateRenderPass(device_, &rpci, nullptr, &fogRP_));
        }
    }

    void createFramebuffers()
    {
        // Scene FB（HDR + 深度）
        std::array<VkImageView, 2> sfAtts = {hdrView_, depth_.view};
        VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fci.renderPass = sceneRP_; fci.attachmentCount = 2; fci.pAttachments = sfAtts.data();
        fci.width = extent_.width; fci.height = extent_.height; fci.layers = 1;
        VK_CHECK(vkCreateFramebuffer(device_, &fci, nullptr, &sceneFB_));

        // Swap FBs（Fog Pass 输出）
        swapFBs_.resize(swapImages_.size());
        for (size_t i = 0; i < swapImages_.size(); ++i) {
            fci.renderPass = fogRP_; fci.attachmentCount = 1; fci.pAttachments = &swapViews_[i];
            VK_CHECK(vkCreateFramebuffer(device_, &fci, nullptr, &swapFBs_[i]));
        }
    }

    void createVertexBuffer()
    {
        VkDeviceSize size = sizeof(SceneVertex) * SCENE.size();
        VkBuffer staging; VkDeviceMemory stagingMem;
        createBuffer(physDev_, device_, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     staging, stagingMem);
        void* data; vkMapMemory(device_, stagingMem, 0, size, 0, &data);
        std::memcpy(data, SCENE.data(), size); vkUnmapMemory(device_, stagingMem);
        createBuffer(physDev_, device_, size,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vb_, vbMem_);
        copyBuffer(device_, cmdPool_, gQueue_, staging, vb_, size);
        vkDestroyBuffer(device_, staging, nullptr); vkFreeMemory(device_, stagingMem, nullptr);
    }

    void createUniformBuffers()
    {
        ubos_.resize(MAX_FRAMES); uboMem_.resize(MAX_FRAMES); uboMapped_.resize(MAX_FRAMES);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            createBuffer(physDev_, device_, sizeof(SceneUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         ubos_[i], uboMem_[i]);
            vkMapMemory(device_, uboMem_[i], 0, sizeof(SceneUBO), 0, &uboMapped_[i]);
        }
    }

    void createDescriptorLayouts()
    {
        VkDescriptorSetLayoutBinding sb{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT};
        VkDescriptorSetLayoutCreateInfo dci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dci.bindingCount = 1; dci.pBindings = &sb;
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &dci, nullptr, &sceneDSL_));

        std::array<VkDescriptorSetLayoutBinding, 2> fogBs = {{
            {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT},
            {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT},
        }};
        dci.bindingCount = 2; dci.pBindings = fogBs.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &dci, nullptr, &fogDSL_));
    }

    void createDescriptorPool()
    {
        std::array<VkDescriptorPoolSize, 2> sizes = {{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         MAX_FRAMES},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_FRAMES * 2},
        }};
        VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        ci.poolSizeCount = 2; ci.pPoolSizes = sizes.data();
        ci.maxSets = MAX_FRAMES * 2;
        VK_CHECK(vkCreateDescriptorPool(device_, &ci, nullptr, &pool_));
    }

    void createDescriptorSets()
    {
        auto alloc = [&](VkDescriptorSetLayout dsl, std::vector<VkDescriptorSet>& sets) {
            std::vector<VkDescriptorSetLayout> ls(MAX_FRAMES, dsl);
            VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ai.descriptorPool = pool_; ai.descriptorSetCount = MAX_FRAMES; ai.pSetLayouts = ls.data();
            sets.resize(MAX_FRAMES);
            VK_CHECK(vkAllocateDescriptorSets(device_, &ai, sets.data()));
        };
        alloc(sceneDSL_, sceneSets_);
        alloc(fogDSL_, fogSets_);

        for (int i = 0; i < MAX_FRAMES; ++i) {
            VkDescriptorBufferInfo bi{ubos_[i], 0, sizeof(SceneUBO)};
            VkWriteDescriptorSet sw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                sceneSets_[i], 0,0,1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &bi};
            vkUpdateDescriptorSets(device_, 1, &sw, 0, nullptr);

            VkDescriptorImageInfo hdrImg {sampler_, hdrView_,    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkDescriptorImageInfo depthImg{sampler_, depth_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            std::array<VkWriteDescriptorSet, 2> fw = {{
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, fogSets_[i],0,0,1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &hdrImg, nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, fogSets_[i],1,0,1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &depthImg, nullptr},
            }};
            vkUpdateDescriptorSets(device_, 2, fw.data(), 0, nullptr);
        }
    }

    void createPipelines()
    {
        // ── Scene pipeline ──
        {
            VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            lci.setLayoutCount = 1; lci.pSetLayouts = &sceneDSL_;
            VK_CHECK(vkCreatePipelineLayout(device_, &lci, nullptr, &sceneLayout_));

            auto vert = createShaderModuleFromFile(device_, "rg_scene.vert.spv");
            auto frag = createShaderModuleFromFile(device_, "rg_scene.frag.spv");

            VkVertexInputBindingDescription bind{0, sizeof(SceneVertex), VK_VERTEX_INPUT_RATE_VERTEX};
            std::array<VkVertexInputAttributeDescription, 3> attrs = {{
                {0,0,VK_FORMAT_R32G32B32_SFLOAT,0},{1,0,VK_FORMAT_R32G32B32_SFLOAT,12},{2,0,VK_FORMAT_R32G32B32_SFLOAT,24}
            }};
            VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            vi.vertexBindingDescriptionCount=1; vi.pVertexBindingDescriptions=&bind;
            vi.vertexAttributeDescriptionCount=3; vi.pVertexAttributeDescriptions=attrs.data();
            VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            ia.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            vps.viewportCount=1; vps.scissorCount=1;
            VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rs.polygonMode=VK_POLYGON_MODE_FILL; rs.cullMode=VK_CULL_MODE_BACK_BIT;
            rs.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth=1.0f;
            VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            ms.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
            VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            ds.depthTestEnable=VK_TRUE; ds.depthWriteEnable=VK_TRUE; ds.depthCompareOp=VK_COMPARE_OP_LESS;
            // 场景 frag 输出两个 location；我们只用一个 attachment，只需 1 个 blend state
            // （rg_scene.frag 有 dual output，但 renderpass 只有 1 个颜色 attachment，第2个会被丢弃）
            VkPipelineColorBlendAttachmentState blend{};
            blend.colorWriteMask = 0xF;
            VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            cbs.attachmentCount=1; cbs.pAttachments=&blend;
            std::array<VkDynamicState,2> dyn={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dyns{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dyns.dynamicStateCount=2; dyns.pDynamicStates=dyn.data();

            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage=VK_SHADER_STAGE_VERTEX_BIT; stages[0].module=vert; stages[0].pName="main";
            stages[1].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage=VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module=frag; stages[1].pName="main";

            VkGraphicsPipelineCreateInfo gci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            gci.stageCount=2; gci.pStages=stages;
            gci.pVertexInputState=&vi; gci.pInputAssemblyState=&ia;
            gci.pViewportState=&vps; gci.pRasterizationState=&rs;
            gci.pMultisampleState=&ms; gci.pDepthStencilState=&ds;
            gci.pColorBlendState=&cbs; gci.pDynamicState=&dyns;
            gci.layout=sceneLayout_; gci.renderPass=sceneRP_;
            VK_CHECK(vkCreateGraphicsPipelines(device_,VK_NULL_HANDLE,1,&gci,nullptr,&scenePipeline_));
            vkDestroyShaderModule(device_, vert, nullptr);
            vkDestroyShaderModule(device_, frag, nullptr);
        }

        // ── Fog pipeline（fullscreen）──
        {
            VkPushConstantRange pcr{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(FogPC)};
            VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            lci.setLayoutCount=1; lci.pSetLayouts=&fogDSL_;
            lci.pushConstantRangeCount=1; lci.pPushConstantRanges=&pcr;
            VK_CHECK(vkCreatePipelineLayout(device_, &lci, nullptr, &fogLayout_));

            auto vert = createShaderModuleFromFile(device_, "rg_fullscreen.vert.spv");
            auto frag = createShaderModuleFromFile(device_, "vol_fog.frag.spv");

            VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            ia.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            vps.viewportCount=1; vps.scissorCount=1;
            VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rs.polygonMode=VK_POLYGON_MODE_FILL; rs.lineWidth=1.0f;
            VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            ms.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
            VkPipelineColorBlendAttachmentState blend{};
            blend.colorWriteMask=0xF;
            VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            cbs.attachmentCount=1; cbs.pAttachments=&blend;
            std::array<VkDynamicState,2> dyn={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dyns{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dyns.dynamicStateCount=2; dyns.pDynamicStates=dyn.data();

            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage=VK_SHADER_STAGE_VERTEX_BIT; stages[0].module=vert; stages[0].pName="main";
            stages[1].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage=VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module=frag; stages[1].pName="main";

            VkGraphicsPipelineCreateInfo gci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            gci.stageCount=2; gci.pStages=stages;
            gci.pVertexInputState=&vi; gci.pInputAssemblyState=&ia;
            gci.pViewportState=&vps; gci.pRasterizationState=&rs;
            gci.pMultisampleState=&ms; gci.pColorBlendState=&cbs; gci.pDynamicState=&dyns;
            gci.layout=fogLayout_; gci.renderPass=fogRP_;
            VK_CHECK(vkCreateGraphicsPipelines(device_,VK_NULL_HANDLE,1,&gci,nullptr,&fogPipeline_));
            vkDestroyShaderModule(device_, vert, nullptr);
            vkDestroyShaderModule(device_, frag, nullptr);
        }
    }

    void createCommandBuffers()
    {
        cmds_.resize(MAX_FRAMES);
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool=cmdPool_; ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount=MAX_FRAMES;
        VK_CHECK(vkAllocateCommandBuffers(device_, &ai, cmds_.data()));
    }

    void createSyncObjects()
    {
        imgAvail_.resize(MAX_FRAMES); renderDone_.resize(MAX_FRAMES); inFlight_.resize(MAX_FRAMES);
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fi.flags=VK_FENCE_CREATE_SIGNALED_BIT;
        for (int i=0;i<MAX_FRAMES;++i) {
            VK_CHECK(vkCreateSemaphore(device_,&si,nullptr,&imgAvail_[i]));
            VK_CHECK(vkCreateSemaphore(device_,&si,nullptr,&renderDone_[i]));
            VK_CHECK(vkCreateFence(device_,&fi,nullptr,&inFlight_[i]));
        }
    }

    // ─── 主循环 ──────────────────────────────────────────────────────────────

    void mainLoop()
    {
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            interactive_.beginFrame(0.016f);
            buildUi();
            drawFrame();
            interactive_.updateWindowTitle();
        }
        vkDeviceWaitIdle(device_);
    }

    void buildUi()
    {
        interactive_.buildDebugPanel("第54章：体积雾");
        ImGui::Separator();
        ImGui::SliderFloat("雾密度",    &fogDensity_,    0.01f, 0.5f);
        ImGui::SliderFloat("高度衰减",  &fogFalloff_,    0.05f, 2.0f);
        ImGui::ColorEdit3("雾颜色",     &fogColor_.x);
        ImGui::Separator();
        ImGui::Checkbox("启用 God Rays", &enableGodRays_);
        if (enableGodRays_)
            ImGui::SliderFloat("光柱强度", &godRayStrength_, 0.0f, 1.5f);
    }

    void drawFrame()
    {
        vkWaitForFences(device_, 1, &inFlight_[frame_], VK_TRUE, UINT64_MAX);
        uint32_t imgIdx = 0;
        VkResult res = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX,
                                             imgAvail_[frame_], VK_NULL_HANDLE, &imgIdx);
        if (res == VK_ERROR_OUT_OF_DATE_KHR) { recreateSwapchain(); return; }
        vkResetFences(device_, 1, &inFlight_[frame_]);
        updateUBO(frame_);

        VkCommandBuffer cmd = cmds_[frame_];
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
        interactive_.beginGpuSection(cmd, frame_);

        // ── Scene Pass ──
        VkClearValue clears[2]{};
        clears[0].color.float32[0]=0.5f; clears[0].color.float32[1]=0.6f;
        clears[0].color.float32[2]=0.7f; clears[0].color.float32[3]=1.0f;
        clears[1].depthStencil.depth=1.0f;
        VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rbi.renderPass=sceneRP_; rbi.framebuffer=sceneFB_;
        rbi.renderArea.extent=extent_; rbi.clearValueCount=2; rbi.pClearValues=clears;
        vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
        VkViewport vp{0,0,float(extent_.width),float(extent_.height),0,1};
        VkRect2D sc{{0,0},extent_};
        vkCmdSetViewport(cmd,0,1,&vp); vkCmdSetScissor(cmd,0,1,&sc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scenePipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sceneLayout_,
                                0,1,&sceneSets_[frame_],0,nullptr);
        VkDeviceSize zero=0;
        vkCmdBindVertexBuffers(cmd,0,1,&vb_,&zero);
        vkCmdDraw(cmd, static_cast<uint32_t>(SCENE.size()), 1, 0, 0);
        vkCmdEndRenderPass(cmd);

        // ── Fog Pass ──
        VkClearValue fogClear{};
        fogClear.color.float32[3]=1.0f;
        rbi.renderPass=fogRP_; rbi.framebuffer=swapFBs_[imgIdx];
        rbi.renderArea.extent=extent_; rbi.clearValueCount=1; rbi.pClearValues=&fogClear;
        vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdSetViewport(cmd,0,1,&vp); vkCmdSetScissor(cmd,0,1,&sc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, fogPipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, fogLayout_,
                                0,1,&fogSets_[frame_],0,nullptr);
        FogPC fogpc{};
        glm::mat4 view = interactive_.camera().viewMatrix();
        glm::mat4 proj = glm::perspective(glm::radians(45.0f),
                                          float(extent_.width)/float(extent_.height), 0.1f, 100.0f);
        proj[1][1]*=-1;
        fogpc.invProjView     = glm::inverse(proj * view);
        fogpc.cameraPos       = glm::vec4(interactive_.camera().eyePosition(), 1);
        fogpc.fogColor        = glm::vec4(fogColor_, 1);
        fogpc.sunDir          = glm::vec4(glm::normalize(glm::vec3(1,2,1)), godRayStrength_);
        fogpc.fogDensity      = fogDensity_;
        fogpc.fogHeightFalloff= fogFalloff_;
        fogpc.fogStart        = 1.0f;
        fogpc.enableGodRays   = enableGodRays_ ? 1 : 0;
        vkCmdPushConstants(cmd, fogLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(FogPC), &fogpc);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        interactive_.renderUi(cmd);
        vkCmdEndRenderPass(cmd);

        interactive_.endGpuSection(cmd, frame_);
        VK_CHECK(vkEndCommandBuffer(cmd));

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        VkPipelineStageFlags wait=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        si.waitSemaphoreCount=1; si.pWaitSemaphores=&imgAvail_[frame_]; si.pWaitDstStageMask=&wait;
        si.commandBufferCount=1; si.pCommandBuffers=&cmd;
        si.signalSemaphoreCount=1; si.pSignalSemaphores=&renderDone_[frame_];
        VK_CHECK(vkQueueSubmit(gQueue_,1,&si,inFlight_[frame_]));

        VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pi.waitSemaphoreCount=1; pi.pWaitSemaphores=&renderDone_[frame_];
        pi.swapchainCount=1; pi.pSwapchains=&swapchain_; pi.pImageIndices=&imgIdx;
        res = vkQueuePresentKHR(pQueue_, &pi);
        if (res==VK_ERROR_OUT_OF_DATE_KHR || res==VK_SUBOPTIMAL_KHR || resized_) {
            resized_=false; recreateSwapchain();
        }
        interactive_.endFrame(frame_);
        frame_ = (frame_+1) % MAX_FRAMES;
    }

    void updateUBO(uint32_t fi)
    {
        SceneUBO ubo{};
        ubo.model = glm::mat4(1.0f);
        ubo.view  = interactive_.camera().viewMatrix();
        ubo.proj  = glm::perspective(glm::radians(45.0f),
                                     float(extent_.width)/float(extent_.height), 0.1f, 100.0f);
        ubo.proj[1][1] *= -1;
        std::memcpy(uboMapped_[fi], &ubo, sizeof(ubo));
    }

    void recreateSwapchain()
    {
        int w=0, h=0;
        glfwGetFramebufferSize(window_,&w,&h);
        while(w==0||h==0){glfwGetFramebufferSize(window_,&w,&h);glfwWaitEvents();}
        vkDeviceWaitIdle(device_);
        vkDestroyFramebuffer(device_, sceneFB_, nullptr);
        for (auto fb : swapFBs_) vkDestroyFramebuffer(device_, fb, nullptr); swapFBs_.clear();
        vkDestroyImageView(device_, hdrView_, nullptr);
        vkDestroyImage(device_, hdrImage_, nullptr); vkFreeMemory(device_, hdrMem_, nullptr);
        destroyDepthResources(device_, depth_);
        for (auto v : swapViews_) vkDestroyImageView(device_, v, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        createSwapchain(); createImageViews();
        depth_ = createDepthResources(physDev_, device_, extent_);
        createOffscreenImage(); createFramebuffers();
        VK_CHECK(vkResetDescriptorPool(device_, pool_, 0));
        sceneSets_.clear(); fogSets_.clear();
        createDescriptorSets();
        interactive_.onSwapchainRecreated(fogRP_, swapFormat_, static_cast<uint32_t>(swapImages_.size()));
    }

    void cleanup()
    {
        vkDestroyDescriptorPool(device_, pool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, sceneDSL_, nullptr);
        vkDestroyDescriptorSetLayout(device_, fogDSL_, nullptr);
        vkDestroyPipeline(device_, scenePipeline_, nullptr);
        vkDestroyPipeline(device_, fogPipeline_, nullptr);
        vkDestroyPipelineLayout(device_, sceneLayout_, nullptr);
        vkDestroyPipelineLayout(device_, fogLayout_, nullptr);
        vkDestroyRenderPass(device_, sceneRP_, nullptr);
        vkDestroyRenderPass(device_, fogRP_, nullptr);
        vkDestroySampler(device_, sampler_, nullptr);
        vkDestroyImageView(device_, hdrView_, nullptr);
        vkDestroyImage(device_, hdrImage_, nullptr); vkFreeMemory(device_, hdrMem_, nullptr);
        vkDestroyFramebuffer(device_, sceneFB_, nullptr);
        for (auto fb : swapFBs_) vkDestroyFramebuffer(device_, fb, nullptr);
        for (int i=0;i<MAX_FRAMES;++i) {
            vkDestroyBuffer(device_,ubos_[i],nullptr); vkFreeMemory(device_,uboMem_[i],nullptr);
            vkDestroySemaphore(device_,imgAvail_[i],nullptr);
            vkDestroySemaphore(device_,renderDone_[i],nullptr);
            vkDestroyFence(device_,inFlight_[i],nullptr);
        }
        vkDestroyBuffer(device_,vb_,nullptr); vkFreeMemory(device_,vbMem_,nullptr);
        vkDestroyCommandPool(device_,cmdPool_,nullptr);
        destroyDepthResources(device_,depth_);
        for (auto v : swapViews_) vkDestroyImageView(device_,v,nullptr);
        vkDestroySwapchainKHR(device_,swapchain_,nullptr);
        interactive_.shutdown(device_);
        vkDestroyDevice(device_,nullptr);
        vkDestroySurfaceKHR(instance_,surface_,nullptr);
        vkDestroyInstance(instance_,nullptr);
        glfwDestroyWindow(window_); glfwTerminate();
    }
};

int main()
{
    std::cout << "══════════════════════════════════════════════════\n";
    std::cout << " 第54章：体积雾（高度指数雾 + God Rays）\n";
    std::cout << "══════════════════════════════════════════════════\n\n";
    std::cout << "控制：LMB 旋转 | RMB 平移 | 滚轮缩放 | ESC 退出\n\n";
    try { Ch54App app; app.run(); std::cout<<"✅ 正常退出\n"; }
    catch(const std::exception& e) { std::cerr<<"❌ "<<e.what()<<"\n"; return 1; }
}
