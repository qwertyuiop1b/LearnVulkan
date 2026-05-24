/**
 * @file ch53_water.cpp
 * @brief 第53章：水面渲染（平面反射 + 折射 + Fresnel）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【渲染管线 · 3 个 Pass】
 *
 *   Pass 1: Reflection（反射）
 *     - 相机翻转到水面以下（关于 y=0 做镜像）
 *     - 开启 gl_ClipDistance[0] 裁剪水面以下的几何
 *     - 渲染结果存入 reflectRT_
 *
 *   Pass 2: Refraction（折射）
 *     - 正常相机，裁剪水面以上的几何（负方向裁剪平面）
 *     - 渲染结果存入 refractRT_
 *
 *   Pass 3: Water + Scene（合成）
 *     - 先绘制不透明场景（天空、柱子、地面）
 *     - 再绘制水面：采样 reflectRT_ + refractRT_，计算 Fresnel + 法线扰动
 *
 * 【无需法线贴图文件】
 *     水面法线在 water.frag 中通过 sin/cos 程序化生成（双层叠加）
 *
 * 【Gerstner 波】
 *     water.vert 对顶点做 Y 轴位移，模拟真实水面形态
 *
 * 【ImGui 控制】
 *     反射强度、折射强度、扰动强度、浑浊度、波高、波速
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

constexpr uint32_t WIDTH      = 800;
constexpr uint32_t HEIGHT     = 600;
constexpr int      MAX_FRAMES = 2;

// RTT 分辨率（比屏幕小一些节省带宽）
constexpr uint32_t RTT_W = 512;
constexpr uint32_t RTT_H = 384;

struct SceneVertex { glm::vec3 pos; glm::vec3 normal; glm::vec3 color; };

// 场景：地面（不在水面）、柱子、上方平台
static const std::vector<SceneVertex> SCENE_VERTS = {
    // 地面（y=-0.5）
    {{-6,-0.5f,-6},{0,1,0},{0.35f,0.3f,0.25f}}, {{ 6,-0.5f,-6},{0,1,0},{0.35f,0.3f,0.25f}},
    {{ 6,-0.5f, 6},{0,1,0},{0.35f,0.3f,0.25f}}, {{-6,-0.5f,-6},{0,1,0},{0.35f,0.3f,0.25f}},
    {{ 6,-0.5f, 6},{0,1,0},{0.35f,0.3f,0.25f}}, {{-6,-0.5f, 6},{0,1,0},{0.35f,0.3f,0.25f}},
    // 石柱（y=−0.5 → +2，反射中可见）
    {{-0.2f,-0.5f,-0.2f},{0,0,-1},{0.6f,0.55f,0.5f}}, {{ 0.2f,-0.5f,-0.2f},{0,0,-1},{0.6f,0.55f,0.5f}},
    {{ 0.2f, 2.0f,-0.2f},{0,0,-1},{0.6f,0.55f,0.5f}}, {{-0.2f,-0.5f,-0.2f},{0,0,-1},{0.6f,0.55f,0.5f}},
    {{ 0.2f, 2.0f,-0.2f},{0,0,-1},{0.6f,0.55f,0.5f}}, {{-0.2f, 2.0f,-0.2f},{0,0,-1},{0.6f,0.55f,0.5f}},
    // 左侧彩色立面（鲜艳色便于观察反射效果）
    {{-3, 0.0f,-0.3f},{1,0,0},{0.2f,0.6f,1.0f}}, {{-3, 0.0f, 0.3f},{1,0,0},{0.2f,0.6f,1.0f}},
    {{-3, 2.5f, 0.3f},{1,0,0},{0.2f,0.6f,1.0f}}, {{-3, 0.0f,-0.3f},{1,0,0},{0.2f,0.6f,1.0f}},
    {{-3, 2.5f, 0.3f},{1,0,0},{0.2f,0.6f,1.0f}}, {{-3, 2.5f,-0.3f},{1,0,0},{0.2f,0.6f,1.0f}},
    // 右侧橙色立面
    {{ 3, 0.0f,-0.3f},{-1,0,0},{1.0f,0.5f,0.1f}}, {{ 3, 0.0f, 0.3f},{-1,0,0},{1.0f,0.5f,0.1f}},
    {{ 3, 2.0f, 0.3f},{-1,0,0},{1.0f,0.5f,0.1f}}, {{ 3, 0.0f,-0.3f},{-1,0,0},{1.0f,0.5f,0.1f}},
    {{ 3, 2.0f, 0.3f},{-1,0,0},{1.0f,0.5f,0.1f}}, {{ 3, 2.0f,-0.3f},{-1,0,0},{1.0f,0.5f,0.1f}},
};

// 水面网格（y=0 平面，细分增加波浪精度）
static std::vector<SceneVertex> buildWaterMesh()
{
    std::vector<SceneVertex> verts;
    constexpr int N = 20;
    constexpr float HALF = 5.0f;
    for (int z = 0; z < N; ++z) {
        for (int x = 0; x < N; ++x) {
            float x0 = -HALF + HALF * 2.0f * x / N;
            float x1 = -HALF + HALF * 2.0f * (x+1) / N;
            float z0 = -HALF + HALF * 2.0f * z / N;
            float z1 = -HALF + HALF * 2.0f * (z+1) / N;
            verts.push_back({{x0,0,z0},{0,1,0},{0,0,0}});
            verts.push_back({{x1,0,z0},{0,1,0},{0,0,0}});
            verts.push_back({{x1,0,z1},{0,1,0},{0,0,0}});
            verts.push_back({{x0,0,z0},{0,1,0},{0,0,0}});
            verts.push_back({{x1,0,z1},{0,1,0},{0,0,0}});
            verts.push_back({{x0,0,z1},{0,1,0},{0,0,0}});
        }
    }
    return verts;
}

struct SceneUBO {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

struct WaterUBO {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::mat4 reflectView;  // 镜像相机
    alignas(16) glm::vec4 cameraPos;
    float time;
    float waveHeight;
    float waveSpeed;
    float tiling;
};

struct ClipPC   { glm::vec4 clipPlane; };
struct WaterPC  { float reflectStrength; float refractStrength; float distortStrength; float murkiness; };

class Ch53App {
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
    DepthResources screenDepth_{};
    VkCommandPool  cmdPool_ = VK_NULL_HANDLE;

    // RTT 资源（固定分辨率，不随交换链变化）
    VkImage        reflectImage_  = VK_NULL_HANDLE;
    VkDeviceMemory reflectMem_    = VK_NULL_HANDLE;
    VkImageView    reflectView_   = VK_NULL_HANDLE;
    VkImage        refractImage_  = VK_NULL_HANDLE;
    VkDeviceMemory refractMem_    = VK_NULL_HANDLE;
    VkImageView    refractView_   = VK_NULL_HANDLE;
    VkImage        rttDepthImage_ = VK_NULL_HANDLE;
    VkDeviceMemory rttDepthMem_   = VK_NULL_HANDLE;
    VkImageView    rttDepthView_  = VK_NULL_HANDLE;
    VkFormat       rttDepthFmt_   = VK_FORMAT_UNDEFINED;
    VkFramebuffer  reflectFB_     = VK_NULL_HANDLE;
    VkFramebuffer  refractFB_     = VK_NULL_HANDLE;

    // RenderPasses
    VkRenderPass rttRP_   = VK_NULL_HANDLE;  // 用于 reflect + refract（颜色+深度）
    VkRenderPass finalRP_ = VK_NULL_HANDLE;  // 用于最终渲染

    // Pipelines
    VkPipeline       scenePipeline_  = VK_NULL_HANDLE;
    VkPipelineLayout sceneLayout_    = VK_NULL_HANDLE;
    VkDescriptorSetLayout sceneDSL_  = VK_NULL_HANDLE;
    VkPipeline       waterPipeline_  = VK_NULL_HANDLE;
    VkPipelineLayout waterLayout_    = VK_NULL_HANDLE;
    VkDescriptorSetLayout waterDSL_  = VK_NULL_HANDLE;

    VkDescriptorPool             pool_          = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> sceneSets_;
    std::vector<VkDescriptorSet> waterSets_;

    VkSampler linearSampler_ = VK_NULL_HANDLE;

    // Buffers & UBOs
    VkBuffer       sceneVB_    = VK_NULL_HANDLE;
    VkDeviceMemory sceneVBMem_ = VK_NULL_HANDLE;
    VkBuffer       waterVB_    = VK_NULL_HANDLE;
    VkDeviceMemory waterVBMem_ = VK_NULL_HANDLE;
    uint32_t       waterVertCount_ = 0;

    std::vector<VkBuffer>       sceneUBOs_, waterUBOs_;
    std::vector<VkDeviceMemory> sceneUBOMem_, waterUBOMem_;
    std::vector<void*>          sceneMapped_, waterMapped_;

    std::vector<VkCommandBuffer> cmdBuffers_;
    std::vector<VkSemaphore>     imageAvail_, renderDone_;
    std::vector<VkFence>         inFlight_;
    uint32_t currentFrame_ = 0;
    bool     resized_      = false;

    InteractiveChapterTools interactive_;

    float totalTime_  = 0.0f;
    // 可调参数
    float reflectStrength_ = 0.7f;
    float refractStrength_ = 1.0f;
    float distortStrength_ = 0.02f;
    float murkiness_       = 0.3f;
    float waveHeight_      = 0.08f;
    float waveSpeed_       = 0.6f;
    float tiling_          = 0.5f;

    // ─── 初始化 ─────────────────────────────────────────────────────────────

    void initWindow()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "第53章：水面渲染", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
            reinterpret_cast<Ch53App*>(glfwGetWindowUserPointer(w))->resized_ = true;
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
        screenDepth_ = createDepthResources(physDev_, device_, extent_);
        createCommandPool();
        createRttResources();
        createRenderPasses();
        createSampler();
        createVertexBuffers();
        createUniformBuffers();
        createDescriptorLayouts();
        createDescriptorPool();
        createDescriptorSets();
        createPipelines();
        createFramebuffers();
        createCommandBuffers();
        createSyncObjects();

        InteractiveInitInfo ii{};
        ii.window = window_; ii.instance = instance_;
        ii.physicalDevice = physDev_; ii.device = device_;
        ii.graphicsQueue = gQueue_;
        ii.queueFamily = queueIdx_.graphicsFamily.value();
        ii.renderPass = finalRP_;
        ii.swapchainFormat = swapFormat_;
        ii.imageCount = static_cast<uint32_t>(swapImages_.size());
        ii.maxFramesInFlight = MAX_FRAMES;
        interactive_.initVulkan(ii);
        interactive_.camera().setDistance(8.0f);
        interactive_.camera().setAngles(45.0f, 25.0f);
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
        ci.surface = surface_; ci.minImageCount = cnt;
        ci.imageFormat = fmt.format; ci.imageColorSpace = fmt.colorSpace;
        ci.imageExtent = extent_; ci.imageArrayLayers = 1;
        ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        uint32_t qf[2] = {queueIdx_.graphicsFamily.value(), queueIdx_.presentFamily.value()};
        if (qf[0] != qf[1]) {
            ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            ci.queueFamilyIndexCount = 2; ci.pQueueFamilyIndices = qf;
        } else { ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE; }
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
            vi.format = swapFormat_;
            vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
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

    void createRttResources()
    {
        rttDepthFmt_ = findDepthFormat(physDev_);
        constexpr VkImageUsageFlags COLOR_USAGE =
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        constexpr VkImageUsageFlags DEPTH_USAGE =
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

        auto makeColor = [&](VkImage& img, VkDeviceMemory& mem, VkImageView& view) {
            createImage(physDev_, device_, RTT_W, RTT_H, swapFormat_,
                        VK_IMAGE_TILING_OPTIMAL, COLOR_USAGE,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, img, mem);
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format = swapFormat_;
            vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VK_CHECK(vkCreateImageView(device_, &vi, nullptr, &view));
        };
        makeColor(reflectImage_, reflectMem_, reflectView_);
        makeColor(refractImage_, refractMem_, refractView_);

        createImage(physDev_, device_, RTT_W, RTT_H, rttDepthFmt_,
                    VK_IMAGE_TILING_OPTIMAL, DEPTH_USAGE,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, rttDepthImage_, rttDepthMem_);
        VkImageViewCreateInfo dvi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        dvi.image = rttDepthImage_; dvi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        dvi.format = rttDepthFmt_;
        dvi.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(device_, &dvi, nullptr, &rttDepthView_));

        // 初始 layout 转换（UNDEFINED → SHADER_READ_ONLY）
        auto transition = [&](VkImage img, VkImageAspectFlags asp,
                               VkImageLayout oldL, VkImageLayout newL) {
            VkCommandBuffer cmd = beginSingleTimeCommands(device_, cmdPool_);
            VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b.oldLayout = oldL; b.newLayout = newL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = img; b.subresourceRange = {asp, 0, 1, 0, 1};
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &b);
            endSingleTimeCommands(device_, cmdPool_, gQueue_, cmd);
        };
        transition(reflectImage_, VK_IMAGE_ASPECT_COLOR_BIT,
                   VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        transition(refractImage_, VK_IMAGE_ASPECT_COLOR_BIT,
                   VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    void createRenderPasses()
    {
        // RTT RenderPass（Reflection 和 Refraction 共用）
        {
            std::array<VkAttachmentDescription, 2> atts{};
            atts[0].format = swapFormat_; atts[0].samples = VK_SAMPLE_COUNT_1_BIT;
            atts[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            atts[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            atts[0].initialLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            atts[0].finalLayout   = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            atts[1].format = rttDepthFmt_; atts[1].samples = VK_SAMPLE_COUNT_1_BIT;
            atts[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            atts[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            atts[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            atts[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            atts[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            atts[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            VkAttachmentReference colorRef = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
            VkAttachmentReference depthRef = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
            VkSubpassDescription sub{};
            sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            sub.colorAttachmentCount = 1; sub.pColorAttachments = &colorRef;
            sub.pDepthStencilAttachment = &depthRef;
            // 不添加 subpass dependency（用显式 pipeline barrier 同步，
            // 同时保证与 finalRP_ 兼容，两者 dependencyCount 都为 0）
            VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
            rpci.attachmentCount = 2; rpci.pAttachments = atts.data();
            rpci.subpassCount = 1; rpci.pSubpasses = &sub;
            VK_CHECK(vkCreateRenderPass(device_, &rpci, nullptr, &rttRP_));
        }

        // Final RenderPass
        {
            std::array<VkAttachmentDescription, 2> atts{};
            atts[0].format = swapFormat_; atts[0].samples = VK_SAMPLE_COUNT_1_BIT;
            atts[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            atts[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            atts[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            atts[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            atts[1].format = screenDepth_.format; atts[1].samples = VK_SAMPLE_COUNT_1_BIT;
            atts[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            atts[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            atts[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            atts[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            atts[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            atts[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            VkAttachmentReference colorRef = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
            VkAttachmentReference depthRef = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
            VkSubpassDescription sub{};
            sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            sub.colorAttachmentCount = 1; sub.pColorAttachments = &colorRef;
            sub.pDepthStencilAttachment = &depthRef;
            VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
            rpci.attachmentCount = 2; rpci.pAttachments = atts.data();
            rpci.subpassCount = 1; rpci.pSubpasses = &sub;
            VK_CHECK(vkCreateRenderPass(device_, &rpci, nullptr, &finalRP_));
        }
    }

    void createSampler()
    {
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter = VK_FILTER_LINEAR; si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = 1.0f;
        VK_CHECK(vkCreateSampler(device_, &si, nullptr, &linearSampler_));
    }

    void createVertexBuffers()
    {
        auto upload = [&](const std::vector<SceneVertex>& v, VkBuffer& buf, VkDeviceMemory& mem) {
            VkDeviceSize size = sizeof(SceneVertex) * v.size();
            VkBuffer staging; VkDeviceMemory stagingMem;
            createBuffer(physDev_, device_, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         staging, stagingMem);
            void* data; vkMapMemory(device_, stagingMem, 0, size, 0, &data);
            std::memcpy(data, v.data(), size);
            vkUnmapMemory(device_, stagingMem);
            createBuffer(physDev_, device_, size,
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buf, mem);
            copyBuffer(device_, cmdPool_, gQueue_, staging, buf, size);
            vkDestroyBuffer(device_, staging, nullptr);
            vkFreeMemory(device_, stagingMem, nullptr);
        };
        upload(SCENE_VERTS, sceneVB_, sceneVBMem_);
        auto waterVerts = buildWaterMesh();
        waterVertCount_ = static_cast<uint32_t>(waterVerts.size());
        upload(waterVerts, waterVB_, waterVBMem_);
    }

    void createUniformBuffers()
    {
        sceneUBOs_.resize(MAX_FRAMES); sceneUBOMem_.resize(MAX_FRAMES); sceneMapped_.resize(MAX_FRAMES);
        waterUBOs_.resize(MAX_FRAMES); waterUBOMem_.resize(MAX_FRAMES); waterMapped_.resize(MAX_FRAMES);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            createBuffer(physDev_, device_, sizeof(SceneUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         sceneUBOs_[i], sceneUBOMem_[i]);
            vkMapMemory(device_, sceneUBOMem_[i], 0, sizeof(SceneUBO), 0, &sceneMapped_[i]);
            createBuffer(physDev_, device_, sizeof(WaterUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         waterUBOs_[i], waterUBOMem_[i]);
            vkMapMemory(device_, waterUBOMem_[i], 0, sizeof(WaterUBO), 0, &waterMapped_[i]);
        }
    }

    void createDescriptorLayouts()
    {
        // Scene DSL：1 UBO
        VkDescriptorSetLayoutBinding sb{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                                        VK_SHADER_STAGE_VERTEX_BIT};
        VkDescriptorSetLayoutCreateInfo dci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dci.bindingCount = 1; dci.pBindings = &sb;
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &dci, nullptr, &sceneDSL_));

        // Water DSL：1 UBO + 2 samplers（reflect + refract）
        std::array<VkDescriptorSetLayoutBinding, 3> wb = {{
            {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
            {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT},
            {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT},
        }};
        dci.bindingCount = 3; dci.pBindings = wb.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &dci, nullptr, &waterDSL_));
    }

    void createDescriptorPool()
    {
        std::array<VkDescriptorPoolSize, 2> sizes = {{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         MAX_FRAMES * 2},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_FRAMES * 2},
        }};
        VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        ci.poolSizeCount = static_cast<uint32_t>(sizes.size());
        ci.pPoolSizes = sizes.data();
        ci.maxSets    = MAX_FRAMES * 2;
        VK_CHECK(vkCreateDescriptorPool(device_, &ci, nullptr, &pool_));
    }

    void createDescriptorSets()
    {
        auto alloc = [&](VkDescriptorSetLayout dsl, std::vector<VkDescriptorSet>& sets) {
            std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES, dsl);
            VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ai.descriptorPool = pool_;
            ai.descriptorSetCount = MAX_FRAMES;
            ai.pSetLayouts = layouts.data();
            sets.resize(MAX_FRAMES);
            VK_CHECK(vkAllocateDescriptorSets(device_, &ai, sets.data()));
        };
        alloc(sceneDSL_, sceneSets_);
        alloc(waterDSL_, waterSets_);

        for (int i = 0; i < MAX_FRAMES; ++i) {
            VkDescriptorBufferInfo sbi{sceneUBOs_[i], 0, sizeof(SceneUBO)};
            VkWriteDescriptorSet sw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                sceneSets_[i], 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &sbi};
            vkUpdateDescriptorSets(device_, 1, &sw, 0, nullptr);

            VkDescriptorBufferInfo wbi{waterUBOs_[i], 0, sizeof(WaterUBO)};
            VkDescriptorImageInfo  reflImg{linearSampler_, reflectView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkDescriptorImageInfo  refrImg{linearSampler_, refractView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            std::array<VkWriteDescriptorSet, 3> ww = {{
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, waterSets_[i], 0, 0, 1,
                 VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &wbi},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, waterSets_[i], 1, 0, 1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &reflImg, nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, waterSets_[i], 2, 0, 1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &refrImg, nullptr},
            }};
            vkUpdateDescriptorSets(device_, 3, ww.data(), 0, nullptr);
        }
    }

    void createPipelines()
    {
        auto makeVertexInput = []() -> std::pair<VkVertexInputBindingDescription,
                                                 std::array<VkVertexInputAttributeDescription, 3>> {
            VkVertexInputBindingDescription bind{0, sizeof(SceneVertex), VK_VERTEX_INPUT_RATE_VERTEX};
            std::array<VkVertexInputAttributeDescription, 3> attrs = {{
                {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
                {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 12},
                {2, 0, VK_FORMAT_R32G32B32_SFLOAT, 24},
            }};
            return {bind, attrs};
        };

        // ── Scene pipeline（RTT 和 Final 两个 RenderPass 共用布局）──
        {
            VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ClipPC)};
            VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            lci.setLayoutCount = 1; lci.pSetLayouts = &sceneDSL_;
            lci.pushConstantRangeCount = 1; lci.pPushConstantRanges = &pcr;
            VK_CHECK(vkCreatePipelineLayout(device_, &lci, nullptr, &sceneLayout_));

            auto vert = createShaderModuleFromFile(device_, "water_scene.vert.spv");
            auto frag = createShaderModuleFromFile(device_, "water_scene.frag.spv");
            auto [bind, attrs] = makeVertexInput();

            VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &bind;
            vi.vertexAttributeDescriptionCount = 3; vi.pVertexAttributeDescriptions = attrs.data();

            VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            vps.viewportCount = 1; vps.scissorCount = 1;
            VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1.0f;
            VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE; ds.depthCompareOp = VK_COMPARE_OP_LESS;
            VkPipelineColorBlendAttachmentState blend{}; blend.colorWriteMask = 0xF;
            VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            cbs.attachmentCount = 1; cbs.pAttachments = &blend;
            std::array<VkDynamicState, 2> dyn = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dyns{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dyns.dynamicStateCount = 2; dyns.pDynamicStates = dyn.data();

            std::array<VkPipelineShaderStageCreateInfo, 2> stages{{
                {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vert, "main"},
                {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, frag, "main"},
            }};
            VkGraphicsPipelineCreateInfo gci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            gci.stageCount = 2; gci.pStages = stages.data();
            gci.pVertexInputState = &vi; gci.pInputAssemblyState = &ia;
            gci.pViewportState = &vps; gci.pRasterizationState = &rs;
            gci.pMultisampleState = &ms; gci.pDepthStencilState = &ds;
            gci.pColorBlendState = &cbs; gci.pDynamicState = &dyns;
            gci.layout = sceneLayout_; gci.renderPass = rttRP_;
            VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gci, nullptr, &scenePipeline_));
            vkDestroyShaderModule(device_, vert, nullptr);
            vkDestroyShaderModule(device_, frag, nullptr);
        }

        // ── Water pipeline ──
        {
            VkPushConstantRange pcr{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(WaterPC)};
            VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            lci.setLayoutCount = 1; lci.pSetLayouts = &waterDSL_;
            lci.pushConstantRangeCount = 1; lci.pPushConstantRanges = &pcr;
            VK_CHECK(vkCreatePipelineLayout(device_, &lci, nullptr, &waterLayout_));

            auto vert = createShaderModuleFromFile(device_, "water.vert.spv");
            auto frag = createShaderModuleFromFile(device_, "water.frag.spv");
            auto [bind, attrs] = makeVertexInput();

            VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &bind;
            vi.vertexAttributeDescriptionCount = 3; vi.pVertexAttributeDescriptions = attrs.data();
            VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            vps.viewportCount = 1; vps.scissorCount = 1;
            VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1.0f;
            VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE; ds.depthCompareOp = VK_COMPARE_OP_LESS;
            VkPipelineColorBlendAttachmentState blend{}; blend.colorWriteMask = 0xF;
            VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            cbs.attachmentCount = 1; cbs.pAttachments = &blend;
            std::array<VkDynamicState, 2> dyn = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dyns{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dyns.dynamicStateCount = 2; dyns.pDynamicStates = dyn.data();

            std::array<VkPipelineShaderStageCreateInfo, 2> stages{{
                {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vert, "main"},
                {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, frag, "main"},
            }};
            VkGraphicsPipelineCreateInfo gci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            gci.stageCount = 2; gci.pStages = stages.data();
            gci.pVertexInputState = &vi; gci.pInputAssemblyState = &ia;
            gci.pViewportState = &vps; gci.pRasterizationState = &rs;
            gci.pMultisampleState = &ms; gci.pDepthStencilState = &ds;
            gci.pColorBlendState = &cbs; gci.pDynamicState = &dyns;
            gci.layout = waterLayout_; gci.renderPass = finalRP_;
            VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gci, nullptr, &waterPipeline_));
            vkDestroyShaderModule(device_, vert, nullptr);
            vkDestroyShaderModule(device_, frag, nullptr);
        }
    }

    void createFramebuffers()
    {
        // RTT framebuffers
        auto makeRttFB = [&](VkImageView colorView, VkFramebuffer& fb) {
            std::array<VkImageView, 2> atts = {colorView, rttDepthView_};
            VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            fci.renderPass = rttRP_; fci.attachmentCount = 2; fci.pAttachments = atts.data();
            fci.width = RTT_W; fci.height = RTT_H; fci.layers = 1;
            VK_CHECK(vkCreateFramebuffer(device_, &fci, nullptr, &fb));
        };
        makeRttFB(reflectView_, reflectFB_);
        makeRttFB(refractView_, refractFB_);

        // Final framebuffers（交换链）
        swapFBs_.resize(swapImages_.size());
        for (size_t i = 0; i < swapImages_.size(); ++i) {
            std::array<VkImageView, 2> atts = {swapViews_[i], screenDepth_.view};
            VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            fci.renderPass = finalRP_; fci.attachmentCount = 2; fci.pAttachments = atts.data();
            fci.width = extent_.width; fci.height = extent_.height; fci.layers = 1;
            VK_CHECK(vkCreateFramebuffer(device_, &fci, nullptr, &swapFBs_[i]));
        }
    }

    void createCommandBuffers()
    {
        cmdBuffers_.resize(MAX_FRAMES);
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = cmdPool_; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
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
            auto now = std::chrono::high_resolution_clock::now();
            float dt = std::chrono::duration<float>(now - lastTime).count();
            lastTime = now;
            totalTime_ += dt;
            interactive_.beginFrame(dt);
            buildUi();
            drawFrame();
            interactive_.updateWindowTitle();
        }
        vkDeviceWaitIdle(device_);
    }

    void buildUi()
    {
        interactive_.buildDebugPanel("第53章：水面渲染");
        ImGui::Separator();
        ImGui::TextUnformatted("水面参数");
        ImGui::SliderFloat("反射强度", &reflectStrength_, 0.0f, 1.0f);
        ImGui::SliderFloat("折射强度", &refractStrength_, 0.0f, 1.0f);
        ImGui::SliderFloat("扰动强度", &distortStrength_, 0.0f, 0.05f, "%.4f");
        ImGui::SliderFloat("浑浊度",   &murkiness_,       0.0f, 1.0f);
        ImGui::Separator();
        ImGui::TextUnformatted("波浪参数");
        ImGui::SliderFloat("波高",   &waveHeight_, 0.0f, 0.3f);
        ImGui::SliderFloat("波速",   &waveSpeed_,  0.1f, 2.0f);
        ImGui::SliderFloat("平铺度", &tiling_,     0.1f, 2.0f);
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

        glm::mat4 normalView  = interactive_.camera().viewMatrix();
        glm::mat4 reflectView = makeReflectView(normalView);

        // ── Pass 1: Reflection ──
        recordRttPass(cmd, reflectFB_, reflectView,
                      glm::vec4(0, 1, 0, 0),      // 裁剪水面以下（y > 0）
                      {RTT_W, RTT_H});

        // ── Pass 2: Refraction ──
        recordRttPass(cmd, refractFB_, normalView,
                      glm::vec4(0, -1, 0, 0),     // 裁剪水面以上（y < 0）
                      {RTT_W, RTT_H});

        // ── Pass 3: Final（场景 + 水面）──
        {
            VkClearValue clears[2]{};
            clears[0].color.float32[0] = 0.52f; clears[0].color.float32[1] = 0.80f;
            clears[0].color.float32[2] = 0.92f; clears[0].color.float32[3] = 1.0f;
            clears[1].depthStencil.depth = 1.0f;
            VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
            rbi.renderPass = finalRP_; rbi.framebuffer = swapFBs_[imageIdx];
            rbi.renderArea.extent = extent_;
            rbi.clearValueCount = 2; rbi.pClearValues = clears;
            vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);

            VkViewport vp{0,0,float(extent_.width),float(extent_.height),0,1};
            VkRect2D   sc{{0,0},extent_};
            vkCmdSetViewport(cmd, 0, 1, &vp);
            vkCmdSetScissor(cmd, 0, 1, &sc);

            // 场景（无裁剪平面）
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scenePipeline_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    sceneLayout_, 0, 1, &sceneSets_[currentFrame_], 0, nullptr);
            ClipPC noClip{glm::vec4(0, 0, 0, 0)};
            vkCmdPushConstants(cmd, sceneLayout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ClipPC), &noClip);
            VkDeviceSize zero = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &sceneVB_, &zero);
            vkCmdDraw(cmd, static_cast<uint32_t>(SCENE_VERTS.size()), 1, 0, 0);

            // 水面
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, waterPipeline_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    waterLayout_, 0, 1, &waterSets_[currentFrame_], 0, nullptr);
            WaterPC wpc{reflectStrength_, refractStrength_, distortStrength_, murkiness_};
            vkCmdPushConstants(cmd, waterLayout_, VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(WaterPC), &wpc);
            vkCmdBindVertexBuffers(cmd, 0, 1, &waterVB_, &zero);
            vkCmdDraw(cmd, waterVertCount_, 1, 0, 0);

            interactive_.renderUi(cmd);
            vkCmdEndRenderPass(cmd);
        }

        interactive_.endGpuSection(cmd, currentFrame_);
        VK_CHECK(vkEndCommandBuffer(cmd));

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        VkPipelineStageFlags wait = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        si.waitSemaphoreCount = 1; si.pWaitSemaphores = &imageAvail_[currentFrame_];
        si.pWaitDstStageMask = &wait;
        si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        si.signalSemaphoreCount = 1; si.pSignalSemaphores = &renderDone_[currentFrame_];
        VK_CHECK(vkQueueSubmit(gQueue_, 1, &si, inFlight_[currentFrame_]));

        VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pi.waitSemaphoreCount = 1; pi.pWaitSemaphores = &renderDone_[currentFrame_];
        pi.swapchainCount = 1; pi.pSwapchains = &swapchain_; pi.pImageIndices = &imageIdx;
        result = vkQueuePresentKHR(pQueue_, &pi);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || resized_) {
            resized_ = false; recreateSwapchain();
        }
        interactive_.endFrame(currentFrame_);
        currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES;
    }

    /// 录制 RTT Pass（反射或折射）
    void recordRttPass(VkCommandBuffer cmd, VkFramebuffer fb,
                       const glm::mat4& view, glm::vec4 clipPlane, VkExtent2D ext)
    {
        VkClearValue clears[2]{};
        clears[0].color.float32[0] = 0.52f; clears[0].color.float32[1] = 0.80f;
        clears[0].color.float32[2] = 0.92f; clears[0].color.float32[3] = 1.0f;
        clears[1].depthStencil.depth = 1.0f;
        VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rbi.renderPass = rttRP_; rbi.framebuffer = fb;
        rbi.renderArea.extent = ext;
        rbi.clearValueCount = 2; rbi.pClearValues = clears;
        vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport vp{0,0,float(ext.width),float(ext.height),0,1};
        VkRect2D   sc{{0,0},ext};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);

        // 临时覆盖 SceneUBO 的 view 矩阵
        glm::mat4 proj = glm::perspective(glm::radians(45.0f),
                                          float(ext.width) / float(ext.height), 0.1f, 100.0f);
        proj[1][1] *= -1;
        SceneUBO tmpUBO{glm::mat4(1.0f), view, proj};
        // 使用当前帧的 UBO（临时写入）
        std::memcpy(sceneMapped_[currentFrame_], &tmpUBO, sizeof(tmpUBO));

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scenePipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                sceneLayout_, 0, 1, &sceneSets_[currentFrame_], 0, nullptr);
        ClipPC cp{clipPlane};
        vkCmdPushConstants(cmd, sceneLayout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ClipPC), &cp);
        VkDeviceSize zero = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &sceneVB_, &zero);
        vkCmdDraw(cmd, static_cast<uint32_t>(SCENE_VERTS.size()), 1, 0, 0);

        vkCmdEndRenderPass(cmd);
    }

    /// 构造关于 y=0 的镜像相机 view 矩阵
    glm::mat4 makeReflectView(const glm::mat4& view) const
    {
        // 从 view 矩阵还原相机位置和目标
        glm::mat4 inv = glm::inverse(view);
        glm::vec3 pos    = glm::vec3(inv[3]);
        glm::vec3 front  = -glm::vec3(inv[2]);
        glm::vec3 up     =  glm::vec3(inv[1]);
        // 关于 y=0 翻转
        pos.y   = -pos.y;
        front.y = -front.y;
        up.y    = -up.y;
        return glm::lookAt(pos, pos + front, up);
    }

    void updateUBOs(uint32_t fi)
    {
        glm::mat4 view = interactive_.camera().viewMatrix();
        glm::mat4 proj = glm::perspective(glm::radians(45.0f),
                                          float(extent_.width) / float(extent_.height), 0.1f, 100.0f);
        proj[1][1] *= -1;
        // Scene UBO（正常相机）
        SceneUBO subo{glm::mat4(1.0f), view, proj};
        std::memcpy(sceneMapped_[fi], &subo, sizeof(subo));

        // Water UBO
        WaterUBO wubo{};
        wubo.model       = glm::mat4(1.0f);
        wubo.view        = view;
        wubo.proj        = proj;
        wubo.reflectView = makeReflectView(view);
        wubo.cameraPos   = glm::vec4(interactive_.camera().eyePosition(), 1.0f);
        wubo.time        = totalTime_;
        wubo.waveHeight  = waveHeight_;
        wubo.waveSpeed   = waveSpeed_;
        wubo.tiling      = tiling_;
        std::memcpy(waterMapped_[fi], &wubo, sizeof(wubo));
    }

    void recreateSwapchain()
    {
        int w = 0, h = 0;
        glfwGetFramebufferSize(window_, &w, &h);
        while (w == 0 || h == 0) { glfwGetFramebufferSize(window_, &w, &h); glfwWaitEvents(); }
        vkDeviceWaitIdle(device_);
        for (auto fb : swapFBs_) vkDestroyFramebuffer(device_, fb, nullptr);
        swapFBs_.clear();
        destroyDepthResources(device_, screenDepth_);
        for (auto v : swapViews_) vkDestroyImageView(device_, v, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        createSwapchain();
        createImageViews();
        screenDepth_ = createDepthResources(physDev_, device_, extent_);
        for (size_t i = 0; i < swapImages_.size(); ++i) {
            std::array<VkImageView, 2> atts = {swapViews_[i], screenDepth_.view};
            VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            fci.renderPass = finalRP_; fci.attachmentCount = 2; fci.pAttachments = atts.data();
            fci.width = extent_.width; fci.height = extent_.height; fci.layers = 1;
            swapFBs_.push_back(VK_NULL_HANDLE);
            VK_CHECK(vkCreateFramebuffer(device_, &fci, nullptr, &swapFBs_.back()));
        }
        interactive_.onSwapchainRecreated(finalRP_, swapFormat_,
                                          static_cast<uint32_t>(swapImages_.size()));
    }

    void cleanup()
    {
        vkDestroyDescriptorPool(device_, pool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, sceneDSL_, nullptr);
        vkDestroyDescriptorSetLayout(device_, waterDSL_, nullptr);
        vkDestroyPipeline(device_, scenePipeline_, nullptr);
        vkDestroyPipeline(device_, waterPipeline_, nullptr);
        vkDestroyPipelineLayout(device_, sceneLayout_, nullptr);
        vkDestroyPipelineLayout(device_, waterLayout_, nullptr);
        vkDestroyRenderPass(device_, rttRP_, nullptr);
        vkDestroyRenderPass(device_, finalRP_, nullptr);
        vkDestroySampler(device_, linearSampler_, nullptr);
        vkDestroyImageView(device_, reflectView_, nullptr);
        vkDestroyImage(device_, reflectImage_, nullptr);
        vkFreeMemory(device_, reflectMem_, nullptr);
        vkDestroyImageView(device_, refractView_, nullptr);
        vkDestroyImage(device_, refractImage_, nullptr);
        vkFreeMemory(device_, refractMem_, nullptr);
        vkDestroyImageView(device_, rttDepthView_, nullptr);
        vkDestroyImage(device_, rttDepthImage_, nullptr);
        vkFreeMemory(device_, rttDepthMem_, nullptr);
        vkDestroyFramebuffer(device_, reflectFB_, nullptr);
        vkDestroyFramebuffer(device_, refractFB_, nullptr);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroyBuffer(device_, sceneUBOs_[i], nullptr); vkFreeMemory(device_, sceneUBOMem_[i], nullptr);
            vkDestroyBuffer(device_, waterUBOs_[i], nullptr); vkFreeMemory(device_, waterUBOMem_[i], nullptr);
            vkDestroySemaphore(device_, imageAvail_[i], nullptr);
            vkDestroySemaphore(device_, renderDone_[i], nullptr);
            vkDestroyFence(device_, inFlight_[i], nullptr);
        }
        vkDestroyBuffer(device_, sceneVB_, nullptr); vkFreeMemory(device_, sceneVBMem_, nullptr);
        vkDestroyBuffer(device_, waterVB_, nullptr); vkFreeMemory(device_, waterVBMem_, nullptr);
        vkDestroyCommandPool(device_, cmdPool_, nullptr);
        destroyDepthResources(device_, screenDepth_);
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
    std::cout << " 第53章：水面渲染（反射 + 折射 + Fresnel + Gerstner 波）\n";
    std::cout << "══════════════════════════════════════════════════\n\n";
    std::cout << "管线：Reflection RTT → Refraction RTT → Final（水面）\n";
    std::cout << "控制：LMB 旋转 | RMB 平移 | 滚轮缩放 | ESC 退出\n\n";
    try {
        Ch53App app;
        app.run();
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
