/**
 * @file ch51_render_graph.cpp
 * @brief 第51章：渲染图（Render Graph）基础
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【为什么需要 Render Graph？】
 *
 *  传统做法（ch24 风格）：
 *    手动在代码中写 vkCmdPipelineBarrier → 容易出错、顺序耦合、难以复用。
 *
 *  Render Graph（本章）：
 *    ① 声明资源：hdrTex = graph.declareTexture(...)
 *    ② 声明 Pass 及其输入/输出：graph.addGraphicsPass("scene", {}, {hdrTex, brightTex}, ...)
 *    ③ Graph 自动推导并插入 Image Layout 屏障
 *    ④ 每帧 graph.execute(cmd, fi) 录制全部命令
 *
 *  好处：
 *    - 屏障自动化，不再手写每个 Layout 转换
 *    - Pass 顺序与屏障逻辑集中管理，易于调试
 *    - 任意插拔 Pass（如 debug 显示中间纹理）
 *
 * 【本章管线】
 *  Scene Pass  ──┬─→ HDR Color ──────────────────→ Composite Pass → 屏幕
 *                └─→ Bright Buffer → BlurH → BlurV ─→ Composite Pass
 *
 * 【ImGui 面板】
 *  - Debug View：完整 / 仅HDR / 仅Bloom / 亮区提取
 *  - Bloom Strength、Exposure 实时调节
 * ═══════════════════════════════════════════════════════════════════════════
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vulkan_tutorial/render_graph.hpp>
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

// 高亮场景：地面 + 多个超亮立方体面 → 产生明显 Bloom
static const std::vector<SceneVertex> SCENE_VERTS = {
    // 地面（普通亮度）
    {{-5,-0.5f,-5},{0,1,0},{0.3f,0.3f,0.4f}}, {{ 5,-0.5f,-5},{0,1,0},{0.3f,0.3f,0.4f}},
    {{ 5,-0.5f, 5},{0,1,0},{0.3f,0.3f,0.4f}}, {{-5,-0.5f,-5},{0,1,0},{0.3f,0.3f,0.4f}},
    {{ 5,-0.5f, 5},{0,1,0},{0.3f,0.3f,0.4f}}, {{-5,-0.5f, 5},{0,1,0},{0.3f,0.3f,0.4f}},
    // 超亮黄色灯柱面（亮度 > 1.0 → 产生 Bloom）
    {{-0.3f,-0.5f,0},{0,0,-1},{3.5f,2.8f,0.1f}}, {{ 0.3f,-0.5f,0},{0,0,-1},{3.5f,2.8f,0.1f}},
    {{ 0.3f, 1.5f,0},{0,0,-1},{3.5f,2.8f,0.1f}}, {{-0.3f,-0.5f,0},{0,0,-1},{3.5f,2.8f,0.1f}},
    {{ 0.3f, 1.5f,0},{0,0,-1},{3.5f,2.8f,0.1f}}, {{-0.3f, 1.5f,0},{0,0,-1},{3.5f,2.8f,0.1f}},
    // 超亮蓝色（左）
    {{-2.5f,-0.5f,-0.3f},{1,0,0},{0.1f,0.5f,4.5f}}, {{-2.5f,-0.5f, 0.3f},{1,0,0},{0.1f,0.5f,4.5f}},
    {{-2.5f, 1.0f, 0.3f},{1,0,0},{0.1f,0.5f,4.5f}}, {{-2.5f,-0.5f,-0.3f},{1,0,0},{0.1f,0.5f,4.5f}},
    {{-2.5f, 1.0f, 0.3f},{1,0,0},{0.1f,0.5f,4.5f}}, {{-2.5f, 1.0f,-0.3f},{1,0,0},{0.1f,0.5f,4.5f}},
    // 超亮红色（右）
    {{ 2.5f,-0.5f,-0.3f},{-1,0,0},{4.0f,0.2f,0.2f}}, {{ 2.5f,-0.5f, 0.3f},{-1,0,0},{4.0f,0.2f,0.2f}},
    {{ 2.5f, 1.0f, 0.3f},{-1,0,0},{4.0f,0.2f,0.2f}}, {{ 2.5f,-0.5f,-0.3f},{-1,0,0},{4.0f,0.2f,0.2f}},
    {{ 2.5f, 1.0f, 0.3f},{-1,0,0},{4.0f,0.2f,0.2f}}, {{ 2.5f, 1.0f,-0.3f},{-1,0,0},{4.0f,0.2f,0.2f}},
};

struct BlurPC       { int horizontal; float blurStrength; };
struct CompositePC  { float bloomStrength; float exposure; int debugView; float pad; };

class Ch51App {
public:
    void run() { initWindow(); initVulkan(); mainLoop(); cleanup(); }

private:
    GLFWwindow*      window_   = nullptr;
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
    VkFormat     swapFormat_  = VK_FORMAT_UNDEFINED;
    VkExtent2D   extent_{};
    DepthResources depth_{};
    VkCommandPool cmdPool_ = VK_NULL_HANDLE;

    // Render Graph 及其管理的瞬态纹理
    RenderGraph      graph_;
    RgTextureHandle  hdrHandle_{};
    RgTextureHandle  brightHandle_{};
    RgTextureHandle  blurHHandle_{};
    RgTextureHandle  blurVHandle_{};

    // RenderPass / Framebuffer（Scene pass 内部管理）
    VkRenderPass              sceneRP_   = VK_NULL_HANDLE;
    VkFramebuffer             sceneFB_   = VK_NULL_HANDLE;
    VkRenderPass              compositeRP_ = VK_NULL_HANDLE;

    // Pipelines & Layouts
    VkPipeline       scenePipeline_     = VK_NULL_HANDLE;
    VkPipelineLayout sceneLayout_       = VK_NULL_HANDLE;
    VkDescriptorSetLayout sceneDSL_     = VK_NULL_HANDLE;

    VkPipeline       blurPipeline_      = VK_NULL_HANDLE;
    VkPipelineLayout blurLayout_        = VK_NULL_HANDLE;
    VkDescriptorSetLayout blurDSL_      = VK_NULL_HANDLE;

    VkPipeline       compositePipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout compositeLayout_   = VK_NULL_HANDLE;
    VkDescriptorSetLayout compositeDSL_ = VK_NULL_HANDLE;

    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> sceneSets_;
    VkDescriptorSet blurHSet_ = VK_NULL_HANDLE;
    VkDescriptorSet blurVSet_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> compositeSets_;

    VkBuffer       vertexBuffer_  = VK_NULL_HANDLE;
    VkDeviceMemory vertexMem_     = VK_NULL_HANDLE;
    std::vector<VkBuffer>       sceneUBOs_;
    std::vector<VkDeviceMemory> sceneUBOMem_;
    std::vector<void*>          sceneMapped_;
    VkSampler                   linearSampler_ = VK_NULL_HANDLE;

    std::vector<VkCommandBuffer> cmdBuffers_;
    std::vector<VkSemaphore> imageAvail_;
    std::vector<VkSemaphore> renderDone_;
    std::vector<VkFence>     inFlight_;
    uint32_t currentFrame_ = 0;
    bool     resized_      = false;

    // ImGui / Camera
    InteractiveChapterTools interactive_;

    // 调节参数
    float bloomStrength_  = 0.8f;
    float exposure_       = 1.0f;
    float blurStrength_   = 1.0f;
    int   debugView_      = 0;

    // ─── 初始化 ─────────────────────────────────────────────────────────────

    void initWindow()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "第51章：Render Graph", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
            reinterpret_cast<Ch51App*>(glfwGetWindowUserPointer(w))->resized_ = true;
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
        createRenderPasses();
        declareGraphResources();
        graph_.build(device_, physDev_, cmdPool_, gQueue_, extent_);
        createDescriptorLayouts();
        createPipelines();
        createFramebuffers();
        createVertexBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        buildGraphPasses();
        createCommandBuffers();
        createSyncObjects();

        InteractiveInitInfo ii{};
        ii.window = window_; ii.instance = instance_;
        ii.physicalDevice = physDev_; ii.device = device_;
        ii.graphicsQueue = gQueue_;
        ii.queueFamily = queueIdx_.graphicsFamily.value();
        ii.renderPass = compositeRP_;
        ii.swapchainFormat = swapFormat_;
        ii.imageCount = static_cast<uint32_t>(swapImages_.size());
        ii.maxFramesInFlight = MAX_FRAMES;
        interactive_.initVulkan(ii);
        interactive_.camera().setDistance(6.0f);
    }

    void createSwapchain()
    {
        auto details = querySwapChainSupport(physDev_, surface_);
        auto fmt     = chooseSwapSurfaceFormat(details.formats);
        auto mode    = chooseSwapPresentMode(details.presentModes);
        extent_      = chooseSwapExtent(details.capabilities, window_);
        swapFormat_  = fmt.format;

        uint32_t imgCount = details.capabilities.minImageCount + 1;
        if (details.capabilities.maxImageCount > 0)
            imgCount = std::min(imgCount, details.capabilities.maxImageCount);

        VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        ci.surface          = surface_;
        ci.minImageCount    = imgCount;
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

        uint32_t cnt = 0;
        vkGetSwapchainImagesKHR(device_, swapchain_, &cnt, nullptr);
        swapImages_.resize(cnt);
        vkGetSwapchainImagesKHR(device_, swapchain_, &cnt, swapImages_.data());
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

    void createSampler()
    {
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter  = VK_FILTER_LINEAR;
        si.minFilter  = VK_FILTER_LINEAR;
        si.addressModeU = si.addressModeV = si.addressModeW =
            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod     = 1.0f;
        VK_CHECK(vkCreateSampler(device_, &si, nullptr, &linearSampler_));
    }

    void createRenderPasses()
    {
        // ── Scene Pass（双颜色 Attachment：HDR + Bright）──
        std::array<VkAttachmentDescription, 3> sceneAtts{};
        // HDR color — 图 Barrier 已确保进入 COLOR_ATTACHMENT_OPTIMAL
        sceneAtts[0].format         = VK_FORMAT_R16G16B16A16_SFLOAT;
        sceneAtts[0].samples        = VK_SAMPLE_COUNT_1_BIT;
        sceneAtts[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        sceneAtts[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        sceneAtts[0].initialLayout  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        sceneAtts[0].finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        // Bright buffer
        sceneAtts[1] = sceneAtts[0];
        // Depth
        sceneAtts[2].format         = depth_.format;
        sceneAtts[2].samples        = VK_SAMPLE_COUNT_1_BIT;
        sceneAtts[2].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        sceneAtts[2].storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        sceneAtts[2].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        sceneAtts[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        sceneAtts[2].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;     // 每帧清空，不关心前帧布局
        sceneAtts[2].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorRefs[2] = {
            {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
            {1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}
        };
        VkAttachmentReference depthRef = {2, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 2;
        subpass.pColorAttachments    = colorRefs;
        subpass.pDepthStencilAttachment = &depthRef;

        VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        rpci.attachmentCount = static_cast<uint32_t>(sceneAtts.size());
        rpci.pAttachments    = sceneAtts.data();
        rpci.subpassCount    = 1;
        rpci.pSubpasses      = &subpass;
        VK_CHECK(vkCreateRenderPass(device_, &rpci, nullptr, &sceneRP_));

        // ── Composite Pass（输出到交换链）──
        VkAttachmentDescription compAtt{};
        compAtt.format         = swapFormat_;
        compAtt.samples        = VK_SAMPLE_COUNT_1_BIT;
        compAtt.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        compAtt.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        compAtt.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        compAtt.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkAttachmentReference compRef = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription  compSub{};
        compSub.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
        compSub.colorAttachmentCount = 1;
        compSub.pColorAttachments    = &compRef;
        rpci.attachmentCount = 1;
        rpci.pAttachments    = &compAtt;
        rpci.pSubpasses      = &compSub;
        VK_CHECK(vkCreateRenderPass(device_, &rpci, nullptr, &compositeRP_));
    }

    void declareGraphResources()
    {
        constexpr VkImageUsageFlags HDR_USAGE =
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        constexpr VkImageUsageFlags STORE_USAGE =
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
            | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        hdrHandle_    = graph_.declareTexture({"hdr",    VK_FORMAT_R16G16B16A16_SFLOAT, {}, HDR_USAGE});
        brightHandle_ = graph_.declareTexture({"bright", VK_FORMAT_R16G16B16A16_SFLOAT, {}, STORE_USAGE});
        blurHHandle_  = graph_.declareTexture({"blur_h", VK_FORMAT_R16G16B16A16_SFLOAT, {}, STORE_USAGE});
        blurVHandle_  = graph_.declareTexture({"blur_v", VK_FORMAT_R16G16B16A16_SFLOAT, {}, STORE_USAGE});
    }

    void buildGraphPasses()
    {
        // Pass 1: Scene → HDR + Bright
        graph_.addGraphicsPass("scene",
            {},
            {hdrHandle_, brightHandle_},
            {},
            [this](VkCommandBuffer cmd, uint32_t fi) {
                recordScenePass(cmd, fi);
            });

        // Pass 2: Blur H（Compute）
        graph_.addComputePass("blur_h",
            {brightHandle_},
            {blurHHandle_},
            [this](VkCommandBuffer cmd, uint32_t /*fi*/) {
                BlurPC pc{1, blurStrength_};
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, blurPipeline_);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        blurLayout_, 0, 1, &blurHSet_, 0, nullptr);
                vkCmdPushConstants(cmd, blurLayout_,
                                   VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BlurPC), &pc);
                vkCmdDispatch(cmd, (extent_.width + 15) / 16, (extent_.height + 15) / 16, 1);
            });

        // Pass 3: Blur V（Compute）
        graph_.addComputePass("blur_v",
            {blurHHandle_},
            {blurVHandle_},
            [this](VkCommandBuffer cmd, uint32_t /*fi*/) {
                BlurPC pc{0, blurStrength_};
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, blurPipeline_);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        blurLayout_, 0, 1, &blurVSet_, 0, nullptr);
                vkCmdPushConstants(cmd, blurLayout_,
                                   VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BlurPC), &pc);
                vkCmdDispatch(cmd, (extent_.width + 15) / 16, (extent_.height + 15) / 16, 1);
            });
    }

    void recordScenePass(VkCommandBuffer cmd, uint32_t fi)
    {
        VkClearValue clears[3]{};
        clears[0].color.float32[0] = 0.05f; clears[0].color.float32[1] = 0.05f;
        clears[0].color.float32[2] = 0.08f; clears[0].color.float32[3] = 1.0f;
        clears[2].depthStencil.depth = 1.0f; clears[2].depthStencil.stencil = 0;

        VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rbi.renderPass        = sceneRP_;
        rbi.framebuffer       = sceneFB_;
        rbi.renderArea.extent = extent_;
        rbi.clearValueCount   = 3;
        rbi.pClearValues      = clears;
        vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport vp{0,0,float(extent_.width),float(extent_.height),0,1};
        VkRect2D sc{{0,0},extent_};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scenePipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                sceneLayout_, 0, 1, &sceneSets_[fi], 0, nullptr);
        VkDeviceSize zero = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer_, &zero);
        vkCmdDraw(cmd, static_cast<uint32_t>(SCENE_VERTS.size()), 1, 0, 0);
        vkCmdEndRenderPass(cmd);
    }

    void createDescriptorLayouts()
    {
        // Scene DSL：1 UBO
        VkDescriptorSetLayoutBinding sceneB{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                                            VK_SHADER_STAGE_VERTEX_BIT};
        VkDescriptorSetLayoutCreateInfo dci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dci.bindingCount = 1;
        dci.pBindings    = &sceneB;
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &dci, nullptr, &sceneDSL_));

        // Blur DSL：2 storage images
        std::array<VkDescriptorSetLayoutBinding, 2> blurBs = {{
            {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT},
        }};
        dci.bindingCount = 2;
        dci.pBindings    = blurBs.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &dci, nullptr, &blurDSL_));

        // Composite DSL：2 combined image samplers（hdr + bloom）
        std::array<VkDescriptorSetLayoutBinding, 2> compBs = {{
            {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT},
            {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT},
        }};
        dci.bindingCount = 2;
        dci.pBindings    = compBs.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &dci, nullptr, &compositeDSL_));
    }

    void createPipelines()
    {
        // ── Scene pipeline ──
        {
            VkPushConstantRange pcr{};
            VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            lci.setLayoutCount = 1;
            lci.pSetLayouts    = &sceneDSL_;
            VK_CHECK(vkCreatePipelineLayout(device_, &lci, nullptr, &sceneLayout_));

            auto vert = createShaderModuleFromFile(device_, "rg_scene.vert.spv");
            auto frag = createShaderModuleFromFile(device_, "rg_scene.frag.spv");

            VkVertexInputBindingDescription bind{0, sizeof(SceneVertex), VK_VERTEX_INPUT_RATE_VERTEX};
            std::array<VkVertexInputAttributeDescription, 3> attrs = {{
                {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
                {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 12},
                {2, 0, VK_FORMAT_R32G32B32_SFLOAT, 24},
            }};

            VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            vi.vertexBindingDescriptionCount   = 1;
            vi.pVertexBindingDescriptions      = &bind;
            vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
            vi.pVertexAttributeDescriptions    = attrs.data();

            VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            vps.viewportCount = 1; vps.scissorCount = 1;

            VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rs.polygonMode = VK_POLYGON_MODE_FILL;
            rs.cullMode    = VK_CULL_MODE_BACK_BIT;
            rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rs.lineWidth   = 1.0f;

            VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            ds.depthTestEnable  = VK_TRUE;
            ds.depthWriteEnable = VK_TRUE;
            ds.depthCompareOp   = VK_COMPARE_OP_LESS;

            std::array<VkPipelineColorBlendAttachmentState, 2> blends{};
            blends[0].colorWriteMask = 0xF;
            blends[1].colorWriteMask = 0xF;
            VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            cbs.attachmentCount = 2;
            cbs.pAttachments    = blends.data();

            std::array<VkDynamicState, 2> dyn = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dyns{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dyns.dynamicStateCount = static_cast<uint32_t>(dyn.size());
            dyns.pDynamicStates    = dyn.data();

            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vert; stages[0].pName = "main";
            stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = frag; stages[1].pName = "main";

            VkGraphicsPipelineCreateInfo gci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            gci.stageCount          = 2;
            gci.pStages             = stages;
            gci.pVertexInputState   = &vi;
            gci.pInputAssemblyState = &ia;
            gci.pViewportState      = &vps;
            gci.pRasterizationState = &rs;
            gci.pMultisampleState   = &ms;
            gci.pDepthStencilState  = &ds;
            gci.pColorBlendState    = &cbs;
            gci.pDynamicState       = &dyns;
            gci.layout              = sceneLayout_;
            gci.renderPass          = sceneRP_;
            VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gci, nullptr, &scenePipeline_));
            vkDestroyShaderModule(device_, vert, nullptr);
            vkDestroyShaderModule(device_, frag, nullptr);
        }

        // ── Blur compute pipeline ──
        {
            VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BlurPC)};
            VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            lci.setLayoutCount      = 1;
            lci.pSetLayouts         = &blurDSL_;
            lci.pushConstantRangeCount = 1;
            lci.pPushConstantRanges = &pcr;
            VK_CHECK(vkCreatePipelineLayout(device_, &lci, nullptr, &blurLayout_));

            auto comp = createShaderModuleFromFile(device_, "rg_blur.comp.spv");
            VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, comp, "main"};
            VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            ci.stage  = stage;
            ci.layout = blurLayout_;
            VK_CHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &ci, nullptr, &blurPipeline_));
            vkDestroyShaderModule(device_, comp, nullptr);
        }

        // ── Composite pipeline ──
        {
            VkPushConstantRange pcr{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(CompositePC)};
            VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            lci.setLayoutCount         = 1;
            lci.pSetLayouts            = &compositeDSL_;
            lci.pushConstantRangeCount = 1;
            lci.pPushConstantRanges    = &pcr;
            VK_CHECK(vkCreatePipelineLayout(device_, &lci, nullptr, &compositeLayout_));

            auto vert = createShaderModuleFromFile(device_, "rg_fullscreen.vert.spv");
            auto frag = createShaderModuleFromFile(device_, "rg_composite.frag.spv");

            VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            vps.viewportCount = 1; vps.scissorCount = 1;
            VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rs.polygonMode = VK_POLYGON_MODE_FILL; rs.lineWidth = 1.0f;
            VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            VkPipelineColorBlendAttachmentState blend{};
            blend.colorWriteMask = 0xF;
            VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            cbs.attachmentCount = 1; cbs.pAttachments = &blend;
            std::array<VkDynamicState, 2> dyn = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dyns{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dyns.dynamicStateCount = 2; dyns.pDynamicStates = dyn.data();

            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vert; stages[0].pName = "main";
            stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = frag; stages[1].pName = "main";

            VkGraphicsPipelineCreateInfo gci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            gci.stageCount = 2; gci.pStages = stages;
            gci.pVertexInputState = &vi; gci.pInputAssemblyState = &ia;
            gci.pViewportState = &vps; gci.pRasterizationState = &rs;
            gci.pMultisampleState = &ms; gci.pColorBlendState = &cbs;
            gci.pDynamicState = &dyns;
            gci.layout = compositeLayout_; gci.renderPass = compositeRP_;
            VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gci, nullptr, &compositePipeline_));
            vkDestroyShaderModule(device_, vert, nullptr);
            vkDestroyShaderModule(device_, frag, nullptr);
        }
    }

    void createFramebuffers()
    {
        // Scene framebuffer：使用渲染图的瞬态纹理
        std::array<VkImageView, 3> sfAtts = {
            graph_.getView(hdrHandle_),
            graph_.getView(brightHandle_),
            depth_.view
        };
        VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fci.renderPass      = sceneRP_;
        fci.attachmentCount = static_cast<uint32_t>(sfAtts.size());
        fci.pAttachments    = sfAtts.data();
        fci.width  = extent_.width;
        fci.height = extent_.height;
        fci.layers = 1;
        VK_CHECK(vkCreateFramebuffer(device_, &fci, nullptr, &sceneFB_));

        // 交换链 Framebuffer（Composite Pass 输出）
        swapFBs_.resize(swapImages_.size());
        for (size_t i = 0; i < swapImages_.size(); ++i) {
            fci.renderPass      = compositeRP_;
            fci.attachmentCount = 1;
            fci.pAttachments    = &swapViews_[i];
            VK_CHECK(vkCreateFramebuffer(device_, &fci, nullptr, &swapFBs_[i]));
        }
    }

    void createVertexBuffer()
    {
        VkDeviceSize size = sizeof(SceneVertex) * SCENE_VERTS.size();
        VkBuffer staging; VkDeviceMemory stagingMem;
        createBuffer(physDev_, device_, size,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     staging, stagingMem);
        void* data;
        vkMapMemory(device_, stagingMem, 0, size, 0, &data);
        std::memcpy(data, SCENE_VERTS.data(), size);
        vkUnmapMemory(device_, stagingMem);
        createBuffer(physDev_, device_, size,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer_, vertexMem_);
        copyBuffer(device_, cmdPool_, gQueue_, staging, vertexBuffer_, size);
        vkDestroyBuffer(device_, staging, nullptr);
        vkFreeMemory(device_, stagingMem, nullptr);
    }

    void createUniformBuffers()
    {
        sceneUBOs_.resize(MAX_FRAMES); sceneUBOMem_.resize(MAX_FRAMES); sceneMapped_.resize(MAX_FRAMES);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            createBuffer(physDev_, device_, sizeof(SceneUBO),
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         sceneUBOs_[i], sceneUBOMem_[i]);
            vkMapMemory(device_, sceneUBOMem_[i], 0, sizeof(SceneUBO), 0, &sceneMapped_[i]);
        }
    }

    void createDescriptorPool()
    {
        std::array<VkDescriptorPoolSize, 3> sizes = {{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         MAX_FRAMES},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_FRAMES * 2},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          4},
        }};
        VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        ci.poolSizeCount = static_cast<uint32_t>(sizes.size());
        ci.pPoolSizes    = sizes.data();
        ci.maxSets       = MAX_FRAMES * 2 + 2;
        VK_CHECK(vkCreateDescriptorPool(device_, &ci, nullptr, &pool_));
    }

    void createDescriptorSets()
    {
        // Scene sets
        std::vector<VkDescriptorSetLayout> sLayouts(MAX_FRAMES, sceneDSL_);
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool     = pool_;
        ai.descriptorSetCount = MAX_FRAMES;
        ai.pSetLayouts        = sLayouts.data();
        sceneSets_.resize(MAX_FRAMES);
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, sceneSets_.data()));
        for (int i = 0; i < MAX_FRAMES; ++i) {
            VkDescriptorBufferInfo bi{sceneUBOs_[i], 0, sizeof(SceneUBO)};
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                sceneSets_[i], 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &bi};
            vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
        }

        // Blur sets（BlurH: bright→blurH, BlurV: blurH→blurV）
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &blurDSL_;
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, &blurHSet_));
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, &blurVSet_));

        auto writeBlurSet = [&](VkDescriptorSet set, RgTextureHandle inH, RgTextureHandle outH) {
            VkDescriptorImageInfo inImg {VK_NULL_HANDLE, graph_.getView(inH),  VK_IMAGE_LAYOUT_GENERAL};
            VkDescriptorImageInfo outImg{VK_NULL_HANDLE, graph_.getView(outH), VK_IMAGE_LAYOUT_GENERAL};
            std::array<VkWriteDescriptorSet, 2> ws = {{
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 0, 0, 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &inImg, nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 1, 0, 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outImg, nullptr},
            }};
            vkUpdateDescriptorSets(device_, 2, ws.data(), 0, nullptr);
        };
        writeBlurSet(blurHSet_, brightHandle_, blurHHandle_);
        writeBlurSet(blurVSet_, blurHHandle_,  blurVHandle_);

        // Composite sets
        std::vector<VkDescriptorSetLayout> cLayouts(MAX_FRAMES, compositeDSL_);
        ai.descriptorSetCount = MAX_FRAMES;
        ai.pSetLayouts        = cLayouts.data();
        compositeSets_.resize(MAX_FRAMES);
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, compositeSets_.data()));
        for (int i = 0; i < MAX_FRAMES; ++i) {
            VkDescriptorImageInfo hdrImg  {linearSampler_, graph_.getView(hdrHandle_),  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkDescriptorImageInfo bloomImg{linearSampler_, graph_.getView(blurVHandle_), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            std::array<VkWriteDescriptorSet, 2> ws = {{
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, compositeSets_[i], 0, 0, 1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &hdrImg, nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, compositeSets_[i], 1, 0, 1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &bloomImg, nullptr},
            }};
            vkUpdateDescriptorSets(device_, 2, ws.data(), 0, nullptr);
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
        interactive_.buildDebugPanel("第51章：Render Graph");
        ImGui::Separator();
        ImGui::TextUnformatted("Bloom 控制");
        ImGui::SliderFloat("Bloom Strength", &bloomStrength_, 0.0f, 2.0f);
        ImGui::SliderFloat("Blur Strength",  &blurStrength_,  0.5f, 2.0f);
        ImGui::SliderFloat("Exposure",       &exposure_,      0.5f, 3.0f);
        ImGui::Separator();
        ImGui::TextUnformatted("调试视图（Debug View）");
        const char* views[] = {"完整合成", "仅HDR（未ToneMap）", "仅Bloom层", "亮区提取"};
        ImGui::Combo("View Mode", &debugView_, views, 4);
        ImGui::Separator();
        ImGui::TextUnformatted("Pass 顺序（自动屏障）");
        ImGui::BulletText("Scene   → HDR + Bright");
        ImGui::BulletText("BlurH   → bright→blurH（Compute）");
        ImGui::BulletText("BlurV   → blurH→blurV（Compute）");
        ImGui::BulletText("Composite→ 屏幕（ToneMap + Bloom）");
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

        // 渲染图执行前 3 个 Pass（Scene + BlurH + BlurV）
        graph_.execute(cmd, currentFrame_);

        // Composite Pass（需要 HDR + BlurV 都处于 SHADER_READ_ONLY）
        // 渲染图已把 hdrHandle_ 和 blurVHandle_ 过渡到正确 Layout
        // 但 hdrHandle_ 在 Scene Pass 后是 COLOR_ATTACHMENT，需要转换
        {
            // 手动把 HDR 从 COLOR_ATTACHMENT → SHADER_READ_ONLY
            VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            b.oldLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            b.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b.image         = graph_.getImage(hdrHandle_);
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.subresourceRange  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &b);

            // blurV 在 GENERAL 后 → SHADER_READ_ONLY
            b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            b.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
            b.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b.image         = graph_.getImage(blurVHandle_);
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &b);

            // 同步 RenderGraph 的 Layout 追踪（手动 Barrier 之后必须更新）
            graph_.setLayout(hdrHandle_,   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            graph_.setLayout(blurVHandle_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }

        VkClearValue compClear{};
        compClear.color.float32[3] = 1.0f;
        VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rbi.renderPass        = compositeRP_;
        rbi.framebuffer       = swapFBs_[imageIdx];
        rbi.renderArea.extent = extent_;
        rbi.clearValueCount   = 1;
        rbi.pClearValues      = &compClear;
        vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport vp{0,0,float(extent_.width),float(extent_.height),0,1};
        VkRect2D sc{{0,0},extent_};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, compositePipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                compositeLayout_, 0, 1, &compositeSets_[currentFrame_], 0, nullptr);
        CompositePC pc{bloomStrength_, exposure_, debugView_, 0.0f};
        vkCmdPushConstants(cmd, compositeLayout_,
                           VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(CompositePC), &pc);
        vkCmdDraw(cmd, 3, 1, 0, 0);

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
        static auto startTime = std::chrono::high_resolution_clock::now();
        auto now  = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration<float>(now - startTime).count();
        (void)elapsed;

        SceneUBO ubo{};
        ubo.model = glm::mat4(1.0f);
        ubo.view  = interactive_.camera().viewMatrix();
        ubo.proj  = glm::perspective(glm::radians(45.0f),
                                     float(extent_.width) / float(extent_.height), 0.1f, 50.0f);
        ubo.proj[1][1] *= -1;
        std::memcpy(sceneMapped_[fi], &ubo, sizeof(ubo));
    }

    void recreateSwapchain()
    {
        int w = 0, h = 0;
        glfwGetFramebufferSize(window_, &w, &h);
        while (w == 0 || h == 0) { glfwGetFramebufferSize(window_, &w, &h); glfwWaitEvents(); }
        vkDeviceWaitIdle(device_);

        vkDestroyFramebuffer(device_, sceneFB_, nullptr);
        for (auto fb : swapFBs_) vkDestroyFramebuffer(device_, fb, nullptr);
        swapFBs_.clear();
        destroyDepthResources(device_, depth_);
        for (auto v : swapViews_) vkDestroyImageView(device_, v, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);

        createSwapchain();
        createImageViews();
        depth_ = createDepthResources(physDev_, device_, extent_);
        graph_.resize(device_, physDev_, cmdPool_, gQueue_, extent_);
        createFramebuffers();
        // 重新写 blur/composite descriptor（view 地址可能变化）
        VK_CHECK(vkResetDescriptorPool(device_, pool_, 0));
        sceneSets_.clear(); compositeSets_.clear();
        blurHSet_ = VK_NULL_HANDLE; blurVSet_ = VK_NULL_HANDLE;
        createDescriptorSets();
        interactive_.onSwapchainRecreated(compositeRP_, swapFormat_,
                                          static_cast<uint32_t>(swapImages_.size()));
    }

    void cleanup()
    {
        vkDestroyDescriptorPool(device_, pool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, sceneDSL_, nullptr);
        vkDestroyDescriptorSetLayout(device_, blurDSL_, nullptr);
        vkDestroyDescriptorSetLayout(device_, compositeDSL_, nullptr);
        vkDestroyPipeline(device_, scenePipeline_, nullptr);
        vkDestroyPipeline(device_, blurPipeline_, nullptr);
        vkDestroyPipeline(device_, compositePipeline_, nullptr);
        vkDestroyPipelineLayout(device_, sceneLayout_, nullptr);
        vkDestroyPipelineLayout(device_, blurLayout_, nullptr);
        vkDestroyPipelineLayout(device_, compositeLayout_, nullptr);
        vkDestroyFramebuffer(device_, sceneFB_, nullptr);
        for (auto fb : swapFBs_) vkDestroyFramebuffer(device_, fb, nullptr);
        vkDestroyRenderPass(device_, sceneRP_, nullptr);
        vkDestroyRenderPass(device_, compositeRP_, nullptr);
        graph_.destroy(device_);
        vkDestroySampler(device_, linearSampler_, nullptr);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroyBuffer(device_, sceneUBOs_[i], nullptr);
            vkFreeMemory(device_, sceneUBOMem_[i], nullptr);
            vkDestroySemaphore(device_, imageAvail_[i], nullptr);
            vkDestroySemaphore(device_, renderDone_[i], nullptr);
            vkDestroyFence(device_, inFlight_[i], nullptr);
        }
        vkDestroyBuffer(device_, vertexBuffer_, nullptr);
        vkFreeMemory(device_, vertexMem_, nullptr);
        vkDestroyCommandPool(device_, cmdPool_, nullptr);
        destroyDepthResources(device_, depth_);
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
    std::cout << " 第51章：Render Graph（HDR + Bloom 自动屏障）\n";
    std::cout << "══════════════════════════════════════════════════\n\n";
    std::cout << "控制：LMB 旋转 | RMB 平移 | 滚轮缩放 | ESC 退出\n";
    std::cout << "ImGui：调节 Bloom/Exposure/调试视图\n\n";
    try {
        Ch51App app;
        app.run();
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
