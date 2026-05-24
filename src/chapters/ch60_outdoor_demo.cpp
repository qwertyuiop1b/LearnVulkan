/**
 * @file ch60_outdoor_demo.cpp
 * @brief 第60章：户外小关卡综合 Demo
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【综合展示的技术】
 *  1. Render Graph（ch51）       — 自动屏障，Pass 管理
 *  2. Shadow Map（ch18/ch45）    — 3×3 PCF 软化阴影
 *  3. 水面（ch53）               — 反射 RTT + 折射 RTT + Fresnel
 *  4. GPU 粒子（ch52）           — 篝火 + 火花
 *  5. 体积雾（ch54）             — 高度指数雾
 *  6. HDR + Bloom（ch51/ch24）   — ACES + 高亮提取
 *
 * 【Pass 顺序（通过 Render Graph 管理屏障）】
 *  Shadow Pass  → 深度图
 *  Reflect Pass → 反射 RTT（翻转相机，裁剪水面以下）
 *  Refract Pass → 折射 RTT（正常相机，裁剪水面以上）
 *  Scene Pass   → HDR 颜色 + Bright Buffer（含水面+场景+阴影）
 *  Particle Pass→ 叠加粒子到 HDR（Alpha 加法）
 *  BlurH/BlurV  → Bloom
 *  Fog Pass     → 高度雾
 *  Composite    → ACES ToneMap → 屏幕
 *
 * 【场景内容】
 *  - 程序化地形（20×20 网格 + 噪声高度）
 *  - 水池（y=0 平面，5×5 网格）
 *  - 彩色石柱（供反射可见）
 *  - 篝火粒子发射器
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
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

using namespace vulkan_tutorial;

constexpr uint32_t WIDTH      = 1024;
constexpr uint32_t HEIGHT     = 768;
constexpr int      MAX_FRAMES = 2;
constexpr uint32_t SHADOW_DIM = 1024;
constexpr uint32_t RTT_W      = 512;
constexpr uint32_t RTT_H      = 384;
constexpr uint32_t N_PARTS    = 8192;

// ─── 数据结构 ────────────────────────────────────────────────────────────────

struct SceneVertex { glm::vec3 pos; glm::vec3 normal; glm::vec3 color; };

struct SceneUBO {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::mat4 lightMVP;
    alignas(16) glm::vec4 lightDir;
};

struct WaterUBO {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::mat4 reflectView;
    alignas(16) glm::vec4 cameraPos;
    float time;
    float waveHeight;
    float waveSpeed;
    float tiling;
};

struct ParticleGPU {
    glm::vec4 position;  // w=lifetime
    glm::vec4 velocity;  // w=maxLifetime
    glm::vec4 color;
    float     size;
    float     pad[3];
};

struct EmitterUBO {
    // 只用 1 个发射器（篝火）
    glm::vec4 position;   // w=type
    glm::vec4 direction;  // w=spread
    glm::vec4 colorMin;
    glm::vec4 colorMax;
    float speed; float speedVar; float lifetime; float lifetimeVar;
    float size; float emitRate; float pad[2];
    // 公共参数
    int   emitterCount;
    float deltaTime;
    float time;
    uint32_t randomSeed;
};

struct CameraUBO {
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::vec4 cameraRight;
    alignas(16) glm::vec4 cameraUp;
};

struct BlurPC      { int horizontal; float strength; };
struct CompositePC { float bloomStrength; float exposure; int debugView; float pad; };
struct FogPC {
    alignas(16) glm::mat4 invProjView;
    alignas(16) glm::vec4 cameraPos;
    alignas(16) glm::vec4 fogColor;
    alignas(16) glm::vec4 sunDir;
    float fogDensity; float fogFalloff; float fogStart; int enableGodRays;
};
struct WaterPC { float reflectStr; float refractStr; float distortStr; float murkiness; };
struct ShadowPC { glm::mat4 lightMVP; };
struct ClipPC { glm::vec4 clipPlane; };

// ─── 场景几何生成 ────────────────────────────────────────────────────────────

static float simpleNoise(float x, float z)
{
    return (std::sin(x * 1.3f + z * 0.7f) + std::cos(x * 0.9f - z * 1.1f)) * 0.25f;
}

static std::vector<SceneVertex> buildTerrain()
{
    std::vector<SceneVertex> v;
    constexpr int N = 20;
    constexpr float HALF = 7.0f;
    for (int zi = 0; zi < N; ++zi) {
        for (int xi = 0; xi < N; ++xi) {
            float x0 = -HALF + HALF * 2.0f * xi / N;
            float x1 = -HALF + HALF * 2.0f * (xi+1) / N;
            float z0 = -HALF + HALF * 2.0f * zi / N;
            float z1 = -HALF + HALF * 2.0f * (zi+1) / N;
            // 水池区域（中心附近）不生成地形（高度在水面以下）
            auto h = [](float x, float z) -> float {
                float waterZone = std::max(std::abs(x) - 2.5f, std::abs(z) - 2.5f);
                if (waterZone < 0.0f) return -0.3f;  // 水下
                return simpleNoise(x, z) - 0.5f;
            };
            float y00=h(x0,z0), y10=h(x1,z0), y11=h(x1,z1), y01=h(x0,z1);
            // 简单法线（近似）
            auto n = [](float ya, float yb, float yc, float step) -> glm::vec3 {
                return glm::normalize(glm::vec3(-(yb-ya)/step, 1.0f, -(yc-ya)/step));
            };
            float step = HALF * 2.0f / N;
            glm::vec3 col{0.35f + simpleNoise(x0,z0)*0.1f, 0.32f, 0.25f};
            glm::vec3 n0 = n(y00,y10,y01,step);
            glm::vec3 n1 = n(y10,y11,y01,step);
            v.push_back({{x0,y00,z0},n0,col}); v.push_back({{x1,y10,z0},n0,col});
            v.push_back({{x1,y11,z1},n1,col}); v.push_back({{x0,y00,z0},n0,col});
            v.push_back({{x1,y11,z1},n1,col}); v.push_back({{x0,y01,z1},n1,col});
        }
    }
    return v;
}

static std::vector<SceneVertex> buildPillars()
{
    std::vector<SceneVertex> v;
    // 4 根彩色石柱（在反射/折射中可见）
    struct PillarDef { glm::vec3 pos; glm::vec3 col; };
    PillarDef pillars[] = {
        {{-4,0,-4}, {0.2f,0.6f,1.0f}},
        {{ 4,0,-4}, {1.0f,0.5f,0.1f}},
        {{-4,0, 4}, {0.2f,0.9f,0.3f}},
        {{ 4,0, 4}, {0.9f,0.2f,0.7f}},
    };
    for (auto& p : pillars) {
        float x0=p.pos.x-0.2f, x1=p.pos.x+0.2f;
        float z0=p.pos.z-0.2f, z1=p.pos.z+0.2f;
        float y0=p.pos.y-0.5f, y1=p.pos.y+2.5f;
        glm::vec3 n{0,0,-1};
        v.push_back({{x0,y0,z0},n,p.col}); v.push_back({{x1,y0,z0},n,p.col});
        v.push_back({{x1,y1,z0},n,p.col}); v.push_back({{x0,y0,z0},n,p.col});
        v.push_back({{x1,y1,z0},n,p.col}); v.push_back({{x0,y1,z0},n,p.col});
    }
    return v;
}

static std::vector<SceneVertex> buildWaterMesh()
{
    std::vector<SceneVertex> v;
    constexpr int N = 16;
    constexpr float HALF = 2.6f;
    for (int zi = 0; zi < N; ++zi) {
        for (int xi = 0; xi < N; ++xi) {
            float x0 = -HALF + HALF*2.0f*xi/N, x1 = -HALF + HALF*2.0f*(xi+1)/N;
            float z0 = -HALF + HALF*2.0f*zi/N, z1 = -HALF + HALF*2.0f*(zi+1)/N;
            glm::vec3 n{0,1,0}, c{0,0,0};
            v.push_back({{x0,0,z0},n,c}); v.push_back({{x1,0,z0},n,c});
            v.push_back({{x1,0,z1},n,c}); v.push_back({{x0,0,z0},n,c});
            v.push_back({{x1,0,z1},n,c}); v.push_back({{x0,0,z1},n,c});
        }
    }
    return v;
}

// ─── 主应用 ──────────────────────────────────────────────────────────────────

class Ch60App {
public:
    void run() { initWindow(); initVulkan(); mainLoop(); cleanup(); }

private:
    GLFWwindow*        window_   = nullptr;
    VkInstance         instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR       surface_  = VK_NULL_HANDLE;
    VkPhysicalDevice   physDev_  = VK_NULL_HANDLE;
    VkDevice           device_   = VK_NULL_HANDLE;
    VkQueue            gQueue_   = VK_NULL_HANDLE;
    VkQueue            pQueue_   = VK_NULL_HANDLE;
    QueueFamilyIndices queueIdx_{};
    VkSwapchainKHR     swapchain_ = VK_NULL_HANDLE;
    std::vector<VkImage>       swapImages_;
    std::vector<VkImageView>   swapViews_;
    std::vector<VkFramebuffer> swapFBs_;
    VkFormat   swapFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D extent_{};
    DepthResources screenDepth_{};
    VkCommandPool  cmdPool_ = VK_NULL_HANDLE;

    // Render Graph（管理 HDR、Bright、BlurH、BlurV）
    RenderGraph      graph_;
    RgTextureHandle  hdrH_{}, brightH_{}, blurHH_{}, blurVH_{};

    // Shadow Map
    VkImage        shadowImage_  = VK_NULL_HANDLE;
    VkDeviceMemory shadowMem_    = VK_NULL_HANDLE;
    VkImageView    shadowView_   = VK_NULL_HANDLE;
    VkImageView    shadowSampleView_ = VK_NULL_HANDLE;
    VkFramebuffer  shadowFB_     = VK_NULL_HANDLE;
    VkRenderPass   shadowRP_     = VK_NULL_HANDLE;
    DepthResources shadowDepth_{};

    // Water RTT
    VkImage        reflectImage_ = VK_NULL_HANDLE, refractImage_ = VK_NULL_HANDLE;
    VkDeviceMemory reflectMem_   = VK_NULL_HANDLE, refractMem_   = VK_NULL_HANDLE;
    VkImageView    reflectView_  = VK_NULL_HANDLE, refractView_  = VK_NULL_HANDLE;
    VkImage        rttDepthImage_= VK_NULL_HANDLE;
    VkDeviceMemory rttDepthMem_  = VK_NULL_HANDLE;
    VkImageView    rttDepthView_ = VK_NULL_HANDLE;
    VkFramebuffer  reflectFB_    = VK_NULL_HANDLE, refractFB_    = VK_NULL_HANDLE;
    VkRenderPass   rttRP_        = VK_NULL_HANDLE;

    // Scene RenderPass（HDR + Bright + Depth）
    VkRenderPass  sceneRP_    = VK_NULL_HANDLE;
    VkFramebuffer sceneFB_    = VK_NULL_HANDLE;
    DepthResources sceneDepth_{};

    // Composite RenderPass（→ swap）
    VkRenderPass  compositeRP_ = VK_NULL_HANDLE;

    // ─ Pipelines ─
    VkPipeline       shadowPipeline_     = VK_NULL_HANDLE;
    VkPipelineLayout shadowLayout_       = VK_NULL_HANDLE;
    VkPipeline       scenePipeline_      = VK_NULL_HANDLE;
    VkPipelineLayout sceneLayout_        = VK_NULL_HANDLE;
    VkDescriptorSetLayout sceneDSL_      = VK_NULL_HANDLE;
    VkPipeline       waterPipeline_      = VK_NULL_HANDLE;
    VkPipelineLayout waterLayout_        = VK_NULL_HANDLE;
    VkDescriptorSetLayout waterDSL_      = VK_NULL_HANDLE;
    VkPipeline       particlePipeline_   = VK_NULL_HANDLE;
    VkPipelineLayout particleLayout_     = VK_NULL_HANDLE;
    VkDescriptorSetLayout particleDSL_   = VK_NULL_HANDLE;
    VkPipeline       computePipeline_    = VK_NULL_HANDLE;
    VkPipelineLayout computeLayout_      = VK_NULL_HANDLE;
    VkDescriptorSetLayout computeDSL_    = VK_NULL_HANDLE;
    VkPipeline       blurPipeline_       = VK_NULL_HANDLE;
    VkPipelineLayout blurLayout_         = VK_NULL_HANDLE;
    VkDescriptorSetLayout blurDSL_       = VK_NULL_HANDLE;
    VkPipeline       fogPipeline_        = VK_NULL_HANDLE;
    VkPipelineLayout fogLayout_          = VK_NULL_HANDLE;
    VkDescriptorSetLayout fogDSL_        = VK_NULL_HANDLE;
    VkPipeline       compositePipeline_  = VK_NULL_HANDLE;
    VkPipelineLayout compositeLayout_    = VK_NULL_HANDLE;
    VkDescriptorSetLayout compositeDSL_  = VK_NULL_HANDLE;

    // Descriptor Pool & Sets
    VkDescriptorPool             pool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> sceneSets_;
    std::vector<VkDescriptorSet> waterSets_;
    std::vector<VkDescriptorSet> particleSets_;
    std::vector<VkDescriptorSet> computeSets_;
    std::vector<VkDescriptorSet> fogSets_;
    std::vector<VkDescriptorSet> compositeSets_;
    VkDescriptorSet blurHSet_ = VK_NULL_HANDLE, blurVSet_ = VK_NULL_HANDLE;

    VkSampler linearSampler_  = VK_NULL_HANDLE;
    VkSampler shadowSampler_  = VK_NULL_HANDLE;

    // Vertex Buffers
    VkBuffer       terrainVB_    = VK_NULL_HANDLE; VkDeviceMemory terrainVBMem_    = VK_NULL_HANDLE;
    VkBuffer       pillarVB_     = VK_NULL_HANDLE; VkDeviceMemory pillarVBMem_     = VK_NULL_HANDLE;
    VkBuffer       waterVB_      = VK_NULL_HANDLE; VkDeviceMemory waterVBMem_      = VK_NULL_HANDLE;
    uint32_t terrainVertCount_ = 0, pillarVertCount_ = 0, waterVertCount_ = 0;

    // UBOs
    std::vector<VkBuffer>       sceneUBOs_, waterUBOs_, emitterUBOs_, cameraUBOs_;
    std::vector<VkDeviceMemory> sceneUBOMem_, waterUBOMem_, emitterUBOMem_, cameraUBOMem_;
    std::vector<void*>          sceneMapped_, waterMapped_, emitterMapped_, cameraMapped_;

    // Particle SSBO
    std::vector<VkBuffer>       particleBuffers_;
    std::vector<VkDeviceMemory> particleMem_;

    std::vector<VkCommandBuffer> cmdBuffers_;
    std::vector<VkSemaphore>     imgAvail_, renderDone_;
    std::vector<VkFence>         inFlight_;
    uint32_t currentFrame_ = 0;
    bool     resized_      = false;

    InteractiveChapterTools interactive_;

    // 时间
    float totalTime_  = 0.0f;
    float deltaTime_  = 0.016f;

    // ImGui 调节参数
    float bloomStr_    = 0.6f;
    float exposure_    = 1.2f;
    float fogDensity_  = 0.12f;
    float fogFalloff_  = 0.5f;
    glm::vec3 fogColor_ = {0.75f, 0.78f, 0.9f};
    float reflectStr_  = 0.7f;
    float refractStr_  = 1.0f;
    float distortStr_  = 0.02f;
    float waveHeight_  = 0.06f;
    bool  showShadow_  = true;
    bool  showWater_   = true;
    bool  showParticle_= true;
    bool  showFog_     = true;
    bool  showBloom_   = true;
    int   debugView_   = 0;

    // 光源
    glm::vec3 lightDir_{-1.0f, -2.0f, -1.0f};

    // ─── 初始化 ─────────────────────────────────────────────────────────────

    void initWindow()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "第60章：户外综合 Demo", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
            reinterpret_cast<Ch60App*>(glfwGetWindowUserPointer(w))->resized_ = true;
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
        createSamplers();
        createShadowResources();
        createRttResources();
        declareGraphResources();
        graph_.build(device_, physDev_, cmdPool_, gQueue_, extent_);
        createSceneDepth();
        createRenderPasses();
        createFramebuffers();
        uploadGeometry();
        createUniformBuffers();
        createParticleBuffers();
        createDescriptorLayouts();
        createDescriptorPool();
        createDescriptorSets();
        createPipelines();
        buildGraphPasses();
        createCommandBuffers();
        createSyncObjects();

        InteractiveInitInfo ii{};
        ii.window = window_; ii.instance = instance_;
        ii.physicalDevice = physDev_; ii.device = device_;
        ii.graphicsQueue = gQueue_;
        ii.queueFamily = queueIdx_.graphicsFamily.value();
        ii.renderPass = compositeRP_; ii.swapchainFormat = swapFormat_;
        ii.imageCount = static_cast<uint32_t>(swapImages_.size());
        ii.maxFramesInFlight = MAX_FRAMES;
        interactive_.initVulkan(ii);
        interactive_.camera().setDistance(12.0f);
        interactive_.camera().setAngles(30.0f, 25.0f);

        std::cout << "✅ 初始化完成\n";
        std::cout << "   地形: " << terrainVertCount_ << " 顶点\n";
        std::cout << "   粒子: " << N_PARTS << "\n";
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
        if (qf[0] != qf[1]) { ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            ci.queueFamilyIndexCount = 2; ci.pQueueFamilyIndices = qf; }
        else ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ci.preTransform = details.capabilities.currentTransform;
        ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        ci.presentMode = mode; ci.clipped = VK_TRUE;
        VK_CHECK(vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_));
        uint32_t n = 0;
        vkGetSwapchainImagesKHR(device_, swapchain_, &n, nullptr);
        swapImages_.resize(n);
        vkGetSwapchainImagesKHR(device_, swapchain_, &n, swapImages_.data());
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

    void createSamplers()
    {
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter = VK_FILTER_LINEAR; si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = 1.0f;
        VK_CHECK(vkCreateSampler(device_, &si, nullptr, &linearSampler_));
        si.magFilter = VK_FILTER_NEAREST; si.minFilter = VK_FILTER_NEAREST;
        VK_CHECK(vkCreateSampler(device_, &si, nullptr, &shadowSampler_));
    }

    void createShadowResources()
    {
        VkFormat depthFmt = findDepthFormat(physDev_);
        createImage(physDev_, device_, SHADOW_DIM, SHADOW_DIM, depthFmt,
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                    | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, shadowImage_, shadowMem_);
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = shadowImage_; vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = depthFmt; vi.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT,0,1,0,1};
        VK_CHECK(vkCreateImageView(device_, &vi, nullptr, &shadowView_));
        shadowSampleView_ = shadowView_;  // 共用 view（仅采样深度）

        // 初始 layout
        VkCommandBuffer cmd = beginSingleTimeCommands(device_, cmdPool_);
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = shadowImage_; b.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT,0,1,0,1};
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,0,nullptr,0,nullptr,1,&b);
        endSingleTimeCommands(device_, cmdPool_, gQueue_, cmd);
    }

    void createRttResources()
    {
        VkFormat depthFmt = findDepthFormat(physDev_);
        auto makeColor = [&](VkImage& img, VkDeviceMemory& mem, VkImageView& view) {
            createImage(physDev_, device_, RTT_W, RTT_H, swapFormat_,
                        VK_IMAGE_TILING_OPTIMAL,
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, img, mem);
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format = swapFormat_; vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
            VK_CHECK(vkCreateImageView(device_, &vi, nullptr, &view));
        };
        makeColor(reflectImage_, reflectMem_, reflectView_);
        makeColor(refractImage_, refractMem_, refractView_);
        createImage(physDev_, device_, RTT_W, RTT_H, depthFmt,
                    VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, rttDepthImage_, rttDepthMem_);
        VkImageViewCreateInfo dvi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        dvi.image = rttDepthImage_; dvi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        dvi.format = depthFmt; dvi.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT,0,1,0,1};
        VK_CHECK(vkCreateImageView(device_, &dvi, nullptr, &rttDepthView_));

        auto initLayout = [&](VkImage img, VkImageAspectFlags asp) {
            VkCommandBuffer cmd = beginSingleTimeCommands(device_, cmdPool_);
            VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = img; b.subresourceRange = {asp,0,1,0,1};
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,0,nullptr,0,nullptr,1,&b);
            endSingleTimeCommands(device_, cmdPool_, gQueue_, cmd);
        };
        initLayout(reflectImage_, VK_IMAGE_ASPECT_COLOR_BIT);
        initLayout(refractImage_, VK_IMAGE_ASPECT_COLOR_BIT);
    }

    void declareGraphResources()
    {
        constexpr VkImageUsageFlags HDR_USE =
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        constexpr VkImageUsageFlags STORE_USE =
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        hdrH_    = graph_.declareTexture({"hdr",    VK_FORMAT_R16G16B16A16_SFLOAT, {}, HDR_USE});
        brightH_ = graph_.declareTexture({"bright", VK_FORMAT_R16G16B16A16_SFLOAT, {}, STORE_USE});
        blurHH_  = graph_.declareTexture({"blur_h", VK_FORMAT_R16G16B16A16_SFLOAT, {}, STORE_USE});
        blurVH_  = graph_.declareTexture({"blur_v", VK_FORMAT_R16G16B16A16_SFLOAT, {}, STORE_USE});
    }

    void createSceneDepth()
    {
        VkFormat fmt = findDepthFormat(physDev_);
        sceneDepth_.format = fmt;
        createImage(physDev_, device_, extent_.width, extent_.height, fmt,
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, sceneDepth_.image, sceneDepth_.memory);
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = sceneDepth_.image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = fmt; vi.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(device_, &vi, nullptr, &sceneDepth_.view));
    }

    void createRenderPasses()
    {
        // Shadow RP（depth only）
        {
            VkAttachmentDescription att{};
            att.format = findDepthFormat(physDev_); att.samples = VK_SAMPLE_COUNT_1_BIT;
            att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            att.initialLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            att.finalLayout   = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            VkAttachmentReference dr{0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
            VkSubpassDescription sub{}; sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            sub.pDepthStencilAttachment = &dr;
            VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
            rpci.attachmentCount=1; rpci.pAttachments=&att; rpci.subpassCount=1; rpci.pSubpasses=&sub;
            VK_CHECK(vkCreateRenderPass(device_, &rpci, nullptr, &shadowRP_));
        }
        // RTT RP（color + depth → SHADER_READ_ONLY）
        {
            std::array<VkAttachmentDescription, 2> atts{};
            atts[0].format = swapFormat_; atts[0].samples = VK_SAMPLE_COUNT_1_BIT;
            atts[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; atts[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            atts[0].initialLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            atts[0].finalLayout   = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            atts[1].format = findDepthFormat(physDev_); atts[1].samples = VK_SAMPLE_COUNT_1_BIT;
            atts[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; atts[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            atts[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            atts[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            atts[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            atts[1].finalLayout   = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            VkAttachmentReference cr{0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
            VkAttachmentReference dr{1,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
            VkSubpassDescription sub{}; sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            sub.colorAttachmentCount=1; sub.pColorAttachments=&cr; sub.pDepthStencilAttachment=&dr;
            VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
            rpci.attachmentCount=2; rpci.pAttachments=atts.data(); rpci.subpassCount=1; rpci.pSubpasses=&sub;
            VK_CHECK(vkCreateRenderPass(device_, &rpci, nullptr, &rttRP_));
        }
        // Scene RP（HDR dual output + depth）
        {
            std::array<VkAttachmentDescription, 3> atts{};
            atts[0].format = VK_FORMAT_R16G16B16A16_SFLOAT; atts[0].samples = VK_SAMPLE_COUNT_1_BIT;
            atts[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; atts[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            atts[0].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            atts[0].finalLayout   = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            atts[1] = atts[0]; // bright
            atts[2].format = sceneDepth_.format; atts[2].samples = VK_SAMPLE_COUNT_1_BIT;
            atts[2].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; atts[2].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            atts[2].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            atts[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            atts[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            atts[2].finalLayout   = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            VkAttachmentReference cr[2]={{0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
                                         {1,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}};
            VkAttachmentReference dr{2,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
            VkSubpassDescription sub{}; sub.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;
            sub.colorAttachmentCount=2; sub.pColorAttachments=cr; sub.pDepthStencilAttachment=&dr;
            VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
            rpci.attachmentCount=3; rpci.pAttachments=atts.data(); rpci.subpassCount=1; rpci.pSubpasses=&sub;
            VK_CHECK(vkCreateRenderPass(device_, &rpci, nullptr, &sceneRP_));
        }
        // Composite RP（→ swap）
        {
            VkAttachmentDescription att{};
            att.format=swapFormat_; att.samples=VK_SAMPLE_COUNT_1_BIT;
            att.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR; att.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
            att.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED; att.finalLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            VkAttachmentReference cr{0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
            VkSubpassDescription sub{}; sub.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;
            sub.colorAttachmentCount=1; sub.pColorAttachments=&cr;
            VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
            rpci.attachmentCount=1; rpci.pAttachments=&att; rpci.subpassCount=1; rpci.pSubpasses=&sub;
            VK_CHECK(vkCreateRenderPass(device_, &rpci, nullptr, &compositeRP_));
        }
    }

    void createFramebuffers()
    {
        // Shadow FB
        VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fci.renderPass=shadowRP_; fci.attachmentCount=1; fci.pAttachments=&shadowView_;
        fci.width=SHADOW_DIM; fci.height=SHADOW_DIM; fci.layers=1;
        VK_CHECK(vkCreateFramebuffer(device_, &fci, nullptr, &shadowFB_));

        // RTT FBs
        auto makeRttFB = [&](VkImageView colorV, VkFramebuffer& fb) {
            std::array<VkImageView,2> atts={colorV, rttDepthView_};
            fci.renderPass=rttRP_; fci.attachmentCount=2; fci.pAttachments=atts.data();
            fci.width=RTT_W; fci.height=RTT_H; fci.layers=1;
            VK_CHECK(vkCreateFramebuffer(device_, &fci, nullptr, &fb));
        };
        makeRttFB(reflectView_, reflectFB_);
        makeRttFB(refractView_, refractFB_);

        // Scene FB（HDR + Bright + Depth）
        std::array<VkImageView,3> sfAtts={graph_.getView(hdrH_), graph_.getView(brightH_), sceneDepth_.view};
        fci.renderPass=sceneRP_; fci.attachmentCount=3; fci.pAttachments=sfAtts.data();
        fci.width=extent_.width; fci.height=extent_.height; fci.layers=1;
        VK_CHECK(vkCreateFramebuffer(device_, &fci, nullptr, &sceneFB_));

        // Swap FBs
        swapFBs_.resize(swapImages_.size());
        for (size_t i=0;i<swapImages_.size();++i) {
            fci.renderPass=compositeRP_; fci.attachmentCount=1; fci.pAttachments=&swapViews_[i];
            fci.width=extent_.width; fci.height=extent_.height; fci.layers=1;
            VK_CHECK(vkCreateFramebuffer(device_, &fci, nullptr, &swapFBs_[i]));
        }
    }

    void uploadGeometry()
    {
        auto upload = [&](const std::vector<SceneVertex>& v, VkBuffer& buf, VkDeviceMemory& mem, uint32_t& cnt) {
            cnt = static_cast<uint32_t>(v.size());
            VkDeviceSize sz = sizeof(SceneVertex)*v.size();
            VkBuffer st; VkDeviceMemory stm;
            createBuffer(physDev_,device_,sz,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,st,stm);
            void* data; vkMapMemory(device_,stm,0,sz,0,&data); std::memcpy(data,v.data(),sz);
            vkUnmapMemory(device_,stm);
            createBuffer(physDev_,device_,sz,
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,buf,mem);
            copyBuffer(device_,cmdPool_,gQueue_,st,buf,sz);
            vkDestroyBuffer(device_,st,nullptr); vkFreeMemory(device_,stm,nullptr);
        };
        upload(buildTerrain(), terrainVB_, terrainVBMem_, terrainVertCount_);
        upload(buildPillars(), pillarVB_,  pillarVBMem_,  pillarVertCount_);
        upload(buildWaterMesh(), waterVB_, waterVBMem_,   waterVertCount_);
    }

    void createUniformBuffers()
    {
        auto makeUBOs = [&](size_t sz, std::vector<VkBuffer>& bufs, std::vector<VkDeviceMemory>& mems, std::vector<void*>& maps) {
            bufs.resize(MAX_FRAMES); mems.resize(MAX_FRAMES); maps.resize(MAX_FRAMES);
            for (int i=0;i<MAX_FRAMES;++i) {
                createBuffer(physDev_,device_,sz,VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                             bufs[i],mems[i]);
                vkMapMemory(device_,mems[i],0,sz,0,&maps[i]);
            }
        };
        makeUBOs(sizeof(SceneUBO),   sceneUBOs_,   sceneUBOMem_,   sceneMapped_);
        makeUBOs(sizeof(WaterUBO),   waterUBOs_,   waterUBOMem_,   waterMapped_);
        makeUBOs(sizeof(EmitterUBO), emitterUBOs_, emitterUBOMem_, emitterMapped_);
        makeUBOs(sizeof(CameraUBO),  cameraUBOs_,  cameraUBOMem_,  cameraMapped_);
    }

    void createParticleBuffers()
    {
        const VkDeviceSize sz = sizeof(ParticleGPU)*N_PARTS;
        particleBuffers_.resize(MAX_FRAMES); particleMem_.resize(MAX_FRAMES);
        std::vector<ParticleGPU> init(N_PARTS);
        for (auto& p : init) p.position.w = -1.0f;
        VkBuffer st; VkDeviceMemory stm;
        createBuffer(physDev_,device_,sz,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,st,stm);
        void* data; vkMapMemory(device_,stm,0,sz,0,&data); std::memcpy(data,init.data(),sz);
        vkUnmapMemory(device_,stm);
        for (int i=0;i<MAX_FRAMES;++i) {
            createBuffer(physDev_,device_,sz,
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,particleBuffers_[i],particleMem_[i]);
            copyBuffer(device_,cmdPool_,gQueue_,st,particleBuffers_[i],sz);
        }
        vkDestroyBuffer(device_,st,nullptr); vkFreeMemory(device_,stm,nullptr);
    }

    void createDescriptorLayouts()
    {
        auto makeLayout = [&](std::vector<VkDescriptorSetLayoutBinding> bindings, VkDescriptorSetLayout& dsl) {
            VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            ci.bindingCount=static_cast<uint32_t>(bindings.size()); ci.pBindings=bindings.data();
            VK_CHECK(vkCreateDescriptorSetLayout(device_,&ci,nullptr,&dsl));
        };
        makeLayout({{0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
                    {1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr}}, sceneDSL_);
        makeLayout({{0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
                    {1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
                    {2,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr}}, waterDSL_);
        makeLayout({{0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_VERTEX_BIT,nullptr},
                    {1,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_VERTEX_BIT,nullptr}}, particleDSL_);
        makeLayout({{0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},
                    {1,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr}}, computeDSL_);
        makeLayout({{0,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},
                    {1,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr}}, blurDSL_);
        makeLayout({{0,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
                    {1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr}}, fogDSL_);
        makeLayout({{0,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
                    {1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr}}, compositeDSL_);
    }

    void createDescriptorPool()
    {
        std::array<VkDescriptorPoolSize,3> sizes = {{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         MAX_FRAMES * 6},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_FRAMES * 8 + 4},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          4},
        }};
        // Also storage buffers
        VkDescriptorPoolSize sbSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, MAX_FRAMES * 2};
        VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        std::array<VkDescriptorPoolSize,4> allSizes = {sizes[0],sizes[1],sizes[2],sbSize};
        ci.poolSizeCount=4; ci.pPoolSizes=allSizes.data();
        ci.maxSets = MAX_FRAMES * 6 + 2;
        VK_CHECK(vkCreateDescriptorPool(device_,&ci,nullptr,&pool_));
    }

    void createDescriptorSets()
    {
        auto allocN = [&](VkDescriptorSetLayout dsl, std::vector<VkDescriptorSet>& sets) {
            std::vector<VkDescriptorSetLayout> ls(MAX_FRAMES, dsl);
            VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ai.descriptorPool=pool_; ai.descriptorSetCount=MAX_FRAMES; ai.pSetLayouts=ls.data();
            sets.resize(MAX_FRAMES);
            VK_CHECK(vkAllocateDescriptorSets(device_,&ai,sets.data()));
        };
        allocN(sceneDSL_,     sceneSets_);
        allocN(waterDSL_,     waterSets_);
        allocN(particleDSL_,  particleSets_);
        allocN(computeDSL_,   computeSets_);
        allocN(fogDSL_,       fogSets_);
        allocN(compositeDSL_, compositeSets_);

        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool=pool_; ai.descriptorSetCount=1; ai.pSetLayouts=&blurDSL_;
        VK_CHECK(vkAllocateDescriptorSets(device_,&ai,&blurHSet_));
        VK_CHECK(vkAllocateDescriptorSets(device_,&ai,&blurVSet_));

        auto w = [&](VkDescriptorSet set, uint32_t binding, VkDescriptorType type,
                     VkBuffer buf, VkDeviceSize sz) {
            VkDescriptorBufferInfo bi{buf,0,sz};
            VkWriteDescriptorSet ws{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,set,binding,0,1,type,nullptr,&bi};
            vkUpdateDescriptorSets(device_,1,&ws,0,nullptr);
        };
        auto wi = [&](VkDescriptorSet set, uint32_t binding, VkSampler samp, VkImageView view, VkImageLayout layout) {
            VkDescriptorImageInfo ii{samp,view,layout};
            VkWriteDescriptorSet ws{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,set,binding,0,1,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,&ii,nullptr};
            vkUpdateDescriptorSets(device_,1,&ws,0,nullptr);
        };
        auto wsi = [&](VkDescriptorSet set, uint32_t binding, VkImageView view) {
            VkDescriptorImageInfo ii{VK_NULL_HANDLE,view,VK_IMAGE_LAYOUT_GENERAL};
            VkWriteDescriptorSet ws{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,set,binding,0,1,
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,&ii,nullptr};
            vkUpdateDescriptorSets(device_,1,&ws,0,nullptr);
        };
        auto ws = [&](VkDescriptorSet set, uint32_t binding, VkBuffer buf) {
            VkDescriptorBufferInfo bi{buf,0,VK_WHOLE_SIZE};
            VkWriteDescriptorSet wss{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,set,binding,0,1,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,nullptr,&bi};
            vkUpdateDescriptorSets(device_,1,&wss,0,nullptr);
        };

        for (int i=0;i<MAX_FRAMES;++i) {
            w(sceneSets_[i],   0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, sceneUBOs_[i],   sizeof(SceneUBO));
            wi(sceneSets_[i],  1, shadowSampler_, shadowSampleView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            w(waterSets_[i],   0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, waterUBOs_[i],   sizeof(WaterUBO));
            wi(waterSets_[i],  1, linearSampler_, reflectView_,      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            wi(waterSets_[i],  2, linearSampler_, refractView_,      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            ws(computeSets_[i],0, particleBuffers_[i]);
            w(computeSets_[i], 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, emitterUBOs_[i], sizeof(EmitterUBO));
            ws(particleSets_[i],0, particleBuffers_[i]);
            w(particleSets_[i], 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, cameraUBOs_[i], sizeof(CameraUBO));
            wi(fogSets_[i],    0, linearSampler_, graph_.getView(hdrH_),   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            wi(fogSets_[i],    1, linearSampler_, sceneDepth_.view,         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            wi(compositeSets_[i],0, linearSampler_, graph_.getView(hdrH_),   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            wi(compositeSets_[i],1, linearSampler_, graph_.getView(blurVH_),  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        wsi(blurHSet_,0,graph_.getView(brightH_)); wsi(blurHSet_,1,graph_.getView(blurHH_));
        wsi(blurVSet_,0,graph_.getView(blurHH_));  wsi(blurVSet_,1,graph_.getView(blurVH_));
    }

    void buildGraphPasses()
    {
        // Blur H
        graph_.addComputePass("blur_h", {brightH_}, {blurHH_},
            [this](VkCommandBuffer cmd, uint32_t) {
                if (!showBloom_) return;
                BlurPC pc{1, 1.0f};
                vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,blurPipeline_);
                vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,blurLayout_,0,1,&blurHSet_,0,nullptr);
                vkCmdPushConstants(cmd,blurLayout_,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(BlurPC),&pc);
                vkCmdDispatch(cmd,(extent_.width+15)/16,(extent_.height+15)/16,1);
            });
        // Blur V
        graph_.addComputePass("blur_v", {blurHH_}, {blurVH_},
            [this](VkCommandBuffer cmd, uint32_t) {
                if (!showBloom_) return;
                BlurPC pc{0, 1.0f};
                vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,blurPipeline_);
                vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,blurLayout_,0,1,&blurVSet_,0,nullptr);
                vkCmdPushConstants(cmd,blurLayout_,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(BlurPC),&pc);
                vkCmdDispatch(cmd,(extent_.width+15)/16,(extent_.height+15)/16,1);
            });
    }

    // Pipeline 创建辅助
    void makePipelineLayout(VkDescriptorSetLayout dsl, VkPipelineLayout& layout,
                            uint32_t pcSize=0, VkShaderStageFlags pcStage=0)
    {
        VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        lci.setLayoutCount=1; lci.pSetLayouts=&dsl;
        VkPushConstantRange pcr{pcStage,0,pcSize};
        if (pcSize>0) { lci.pushConstantRangeCount=1; lci.pPushConstantRanges=&pcr; }
        VK_CHECK(vkCreatePipelineLayout(device_,&lci,nullptr,&layout));
    }

    VkGraphicsPipelineCreateInfo buildGraphicsPCI(
        VkPipelineVertexInputStateCreateInfo& vi, VkPipelineInputAssemblyStateCreateInfo& ia,
        VkPipelineViewportStateCreateInfo& vps, VkPipelineRasterizationStateCreateInfo& rs,
        VkPipelineMultisampleStateCreateInfo& ms, VkPipelineDepthStencilStateCreateInfo& ds,
        VkPipelineColorBlendStateCreateInfo& cbs, VkPipelineDynamicStateCreateInfo& dyns,
        VkPipelineShaderStageCreateInfo* stages, uint32_t stageCount,
        VkPipelineLayout layout, VkRenderPass rp, uint32_t subpass=0)
    {
        VkGraphicsPipelineCreateInfo gci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gci.stageCount=stageCount; gci.pStages=stages;
        gci.pVertexInputState=&vi; gci.pInputAssemblyState=&ia;
        gci.pViewportState=&vps; gci.pRasterizationState=&rs;
        gci.pMultisampleState=&ms; gci.pDepthStencilState=&ds;
        gci.pColorBlendState=&cbs; gci.pDynamicState=&dyns;
        gci.layout=layout; gci.renderPass=rp; gci.subpass=subpass;
        return gci;
    }

    void createPipelines()
    {
        // Common state
        VkVertexInputBindingDescription bind{0,sizeof(SceneVertex),VK_VERTEX_INPUT_RATE_VERTEX};
        std::array<VkVertexInputAttributeDescription,3> attrs = {{
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
        rs.polygonMode=VK_POLYGON_MODE_FILL; rs.cullMode=VK_CULL_MODE_NONE; rs.lineWidth=1.0f;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable=VK_TRUE; ds.depthWriteEnable=VK_TRUE; ds.depthCompareOp=VK_COMPARE_OP_LESS;
        VkPipelineColorBlendAttachmentState blend{}; blend.colorWriteMask=0xF;
        VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cbs.attachmentCount=1; cbs.pAttachments=&blend;
        std::array<VkDynamicState,2> dyn={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyns{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dyns.dynamicStateCount=2; dyns.pDynamicStates=dyn.data();

        // ── Shadow pipeline ──
        {
            VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT,0,sizeof(ShadowPC)};
            VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            lci.pushConstantRangeCount=1; lci.pPushConstantRanges=&pcr;
            VK_CHECK(vkCreatePipelineLayout(device_,&lci,nullptr,&shadowLayout_));

            auto vert=createShaderModuleFromFile(device_,"pcss_shadow.vert.spv");
            VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_VERTEX_BIT,vert,"main"};
            VkPipelineColorBlendStateCreateInfo noCbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            noCbs.attachmentCount=0;
            auto gci=buildGraphicsPCI(vi,ia,vps,rs,ms,ds,noCbs,dyns,&stage,1,shadowLayout_,shadowRP_);
            VK_CHECK(vkCreateGraphicsPipelines(device_,VK_NULL_HANDLE,1,&gci,nullptr,&shadowPipeline_));
            vkDestroyShaderModule(device_,vert,nullptr);
        }

        // ── Scene pipeline（HDR dual output）──
        {
            makePipelineLayout(sceneDSL_, sceneLayout_, sizeof(ClipPC), VK_SHADER_STAGE_VERTEX_BIT);
            auto vert=createShaderModuleFromFile(device_,"outdoor_scene.vert.spv");
            auto frag=createShaderModuleFromFile(device_,"outdoor_scene.frag.spv");
            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage=VK_SHADER_STAGE_VERTEX_BIT; stages[0].module=vert; stages[0].pName="main";
            stages[1].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage=VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module=frag; stages[1].pName="main";
            std::array<VkPipelineColorBlendAttachmentState,2> dualBlend{};
            dualBlend[0].colorWriteMask=0xF; dualBlend[1].colorWriteMask=0xF;
            VkPipelineColorBlendStateCreateInfo dualCbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            dualCbs.attachmentCount=2; dualCbs.pAttachments=dualBlend.data();
            auto gci=buildGraphicsPCI(vi,ia,vps,rs,ms,ds,dualCbs,dyns,stages,2,sceneLayout_,sceneRP_);
            VK_CHECK(vkCreateGraphicsPipelines(device_,VK_NULL_HANDLE,1,&gci,nullptr,&scenePipeline_));
            vkDestroyShaderModule(device_,vert,nullptr); vkDestroyShaderModule(device_,frag,nullptr);
        }

        // ── Water pipeline（scene pass，双颜色输出）──
        {
            makePipelineLayout(waterDSL_, waterLayout_, sizeof(WaterPC), VK_SHADER_STAGE_FRAGMENT_BIT);
            auto vert=createShaderModuleFromFile(device_,"water.vert.spv");
            auto frag=createShaderModuleFromFile(device_,"water.frag.spv");
            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage=VK_SHADER_STAGE_VERTEX_BIT; stages[0].module=vert; stages[0].pName="main";
            stages[1].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage=VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module=frag; stages[1].pName="main";
            std::array<VkPipelineColorBlendAttachmentState,2> dualBlend2{};
            dualBlend2[0].colorWriteMask=0xF; dualBlend2[1].colorWriteMask=0xF;
            VkPipelineColorBlendStateCreateInfo dualCbs2{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            dualCbs2.attachmentCount=2; dualCbs2.pAttachments=dualBlend2.data();
            auto gci=buildGraphicsPCI(vi,ia,vps,rs,ms,ds,dualCbs2,dyns,stages,2,waterLayout_,sceneRP_);
            VK_CHECK(vkCreateGraphicsPipelines(device_,VK_NULL_HANDLE,1,&gci,nullptr,&waterPipeline_));
            vkDestroyShaderModule(device_,vert,nullptr); vkDestroyShaderModule(device_,frag,nullptr);
        }

        // ── RTT scene pipeline（without dual output，用于 reflect/refract pass）──
        // 复用 sceneLayout_，渲染到 rttRP_（单颜色输出）
        VkPipeline rttScenePipeline = VK_NULL_HANDLE;
        {
            auto vert=createShaderModuleFromFile(device_,"water_scene.vert.spv");
            auto frag=createShaderModuleFromFile(device_,"water_scene.frag.spv");
            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage=VK_SHADER_STAGE_VERTEX_BIT; stages[0].module=vert; stages[0].pName="main";
            stages[1].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage=VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module=frag; stages[1].pName="main";
            // 需要 ClipPC push constant
            VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(ClipPC)};
            VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            lci.setLayoutCount=1; lci.pSetLayouts=&sceneDSL_;
            lci.pushConstantRangeCount=1; lci.pPushConstantRanges=&pcr;
            VkPipelineLayout rttLayout=VK_NULL_HANDLE;
            VK_CHECK(vkCreatePipelineLayout(device_,&lci,nullptr,&rttLayout));
            auto gci=buildGraphicsPCI(vi,ia,vps,rs,ms,ds,cbs,dyns,stages,2,rttLayout,rttRP_);
            VK_CHECK(vkCreateGraphicsPipelines(device_,VK_NULL_HANDLE,1,&gci,nullptr,&rttScenePipeline));
            // 注：为简化，rttLayout 和 rttScenePipeline 临时存在，在 drawFrame 后销毁
            // 实际工程应管理这些资源；本章将它存入 scenePipeline_ 变量复用
            // 为教学简洁：把 rttLayout/rttScenePipeline 存入 fogLayout_/fogPipeline_（临时借用）
            fogLayout_   = rttLayout;
            fogPipeline_ = rttScenePipeline;
            vkDestroyShaderModule(device_,vert,nullptr); vkDestroyShaderModule(device_,frag,nullptr);
        }

        // ── Particle compute ──
        {
            VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            lci.setLayoutCount=1; lci.pSetLayouts=&computeDSL_;
            VK_CHECK(vkCreatePipelineLayout(device_,&lci,nullptr,&computeLayout_));
            auto comp=createShaderModuleFromFile(device_,"particle2_update.comp.spv");
            VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,comp,"main"};
            VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            ci.stage=stage; ci.layout=computeLayout_;
            VK_CHECK(vkCreateComputePipelines(device_,VK_NULL_HANDLE,1,&ci,nullptr,&computePipeline_));
            vkDestroyShaderModule(device_,comp,nullptr);
        }

        // ── Particle draw ──
        {
            VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            lci.setLayoutCount=1; lci.pSetLayouts=&particleDSL_;
            VK_CHECK(vkCreatePipelineLayout(device_,&lci,nullptr,&particleLayout_));
            auto vert=createShaderModuleFromFile(device_,"particle2.vert.spv");
            auto frag=createShaderModuleFromFile(device_,"particle2.frag.spv");
            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage=VK_SHADER_STAGE_VERTEX_BIT; stages[0].module=vert; stages[0].pName="main";
            stages[1].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage=VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module=frag; stages[1].pName="main";
            VkPipelineVertexInputStateCreateInfo emptyVI{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            VkPipelineDepthStencilStateCreateInfo pDs{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            pDs.depthTestEnable=VK_TRUE; pDs.depthWriteEnable=VK_FALSE; pDs.depthCompareOp=VK_COMPARE_OP_LESS;
            VkPipelineColorBlendAttachmentState pBlend{};
            pBlend.blendEnable=VK_TRUE;
            pBlend.srcColorBlendFactor=VK_BLEND_FACTOR_SRC_ALPHA;
            pBlend.dstColorBlendFactor=VK_BLEND_FACTOR_ONE;
            pBlend.colorBlendOp=VK_BLEND_OP_ADD;
            pBlend.srcAlphaBlendFactor=VK_BLEND_FACTOR_ONE;
            pBlend.dstAlphaBlendFactor=VK_BLEND_FACTOR_ZERO;
            pBlend.alphaBlendOp=VK_BLEND_OP_ADD;
            pBlend.colorWriteMask=0xF;
            // Dual output blend for scene pass
            std::array<VkPipelineColorBlendAttachmentState,2> dualPBlend{pBlend,pBlend};
            VkPipelineColorBlendStateCreateInfo pCbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            pCbs.attachmentCount=2; pCbs.pAttachments=dualPBlend.data();
            auto gci=buildGraphicsPCI(emptyVI,ia,vps,rs,ms,pDs,pCbs,dyns,stages,2,particleLayout_,sceneRP_);
            VK_CHECK(vkCreateGraphicsPipelines(device_,VK_NULL_HANDLE,1,&gci,nullptr,&particlePipeline_));
            vkDestroyShaderModule(device_,vert,nullptr); vkDestroyShaderModule(device_,frag,nullptr);
        }

        // ── Blur compute ──
        {
            VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(BlurPC)};
            VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            lci.setLayoutCount=1; lci.pSetLayouts=&blurDSL_;
            lci.pushConstantRangeCount=1; lci.pPushConstantRanges=&pcr;
            VK_CHECK(vkCreatePipelineLayout(device_,&lci,nullptr,&blurLayout_));
            auto comp=createShaderModuleFromFile(device_,"rg_blur.comp.spv");
            VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,comp,"main"};
            VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            ci.stage=stage; ci.layout=blurLayout_;
            VK_CHECK(vkCreateComputePipelines(device_,VK_NULL_HANDLE,1,&ci,nullptr,&blurPipeline_));
            vkDestroyShaderModule(device_,comp,nullptr);
        }

        // ── Composite pipeline（fullscreen，ACES + Bloom）──
        {
            VkPushConstantRange pcr{VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(CompositePC)};
            VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            lci.setLayoutCount=1; lci.pSetLayouts=&compositeDSL_;
            lci.pushConstantRangeCount=1; lci.pPushConstantRanges=&pcr;
            VK_CHECK(vkCreatePipelineLayout(device_,&lci,nullptr,&compositeLayout_));
            auto vert=createShaderModuleFromFile(device_,"rg_fullscreen.vert.spv");
            auto frag=createShaderModuleFromFile(device_,"rg_composite.frag.spv");
            VkPipelineVertexInputStateCreateInfo emptyVI{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            VkPipelineDepthStencilStateCreateInfo nods{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage=VK_SHADER_STAGE_VERTEX_BIT; stages[0].module=vert; stages[0].pName="main";
            stages[1].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage=VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module=frag; stages[1].pName="main";
            auto gci=buildGraphicsPCI(emptyVI,ia,vps,rs,ms,nods,cbs,dyns,stages,2,compositeLayout_,compositeRP_);
            VK_CHECK(vkCreateGraphicsPipelines(device_,VK_NULL_HANDLE,1,&gci,nullptr,&compositePipeline_));
            vkDestroyShaderModule(device_,vert,nullptr); vkDestroyShaderModule(device_,frag,nullptr);
        }
    }

    void createCommandBuffers()
    {
        cmdBuffers_.resize(MAX_FRAMES);
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool=cmdPool_; ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount=MAX_FRAMES;
        VK_CHECK(vkAllocateCommandBuffers(device_,&ai,cmdBuffers_.data()));
    }

    void createSyncObjects()
    {
        imgAvail_.resize(MAX_FRAMES); renderDone_.resize(MAX_FRAMES); inFlight_.resize(MAX_FRAMES);
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; fi.flags=VK_FENCE_CREATE_SIGNALED_BIT;
        for (int i=0;i<MAX_FRAMES;++i) {
            VK_CHECK(vkCreateSemaphore(device_,&si,nullptr,&imgAvail_[i]));
            VK_CHECK(vkCreateSemaphore(device_,&si,nullptr,&renderDone_[i]));
            VK_CHECK(vkCreateFence(device_,&fi,nullptr,&inFlight_[i]));
        }
    }

    // ─── 主循环 ──────────────────────────────────────────────────────────────

    void mainLoop()
    {
        auto lastTime = std::chrono::high_resolution_clock::now();
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            auto now = std::chrono::high_resolution_clock::now();
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
        interactive_.buildDebugPanel("第60章：户外综合 Demo");
        ImGui::Separator();
        ImGui::TextUnformatted("功能开关");
        ImGui::Checkbox("阴影",   &showShadow_);
        ImGui::SameLine();
        ImGui::Checkbox("水面",   &showWater_);
        ImGui::SameLine();
        ImGui::Checkbox("粒子",   &showParticle_);
        ImGui::Checkbox("体积雾", &showFog_);
        ImGui::SameLine();
        ImGui::Checkbox("Bloom",  &showBloom_);
        ImGui::Separator();
        ImGui::TextUnformatted("渲染参数");
        ImGui::SliderFloat("曝光度", &exposure_,  0.5f, 3.0f);
        ImGui::SliderFloat("Bloom",  &bloomStr_,   0.0f, 2.0f);
        ImGui::Separator();
        ImGui::TextUnformatted("雾");
        ImGui::SliderFloat("雾密度",   &fogDensity_, 0.0f, 0.4f);
        ImGui::SliderFloat("高度衰减", &fogFalloff_, 0.05f, 2.0f);
        ImGui::ColorEdit3("雾颜色",   &fogColor_.x);
        ImGui::Separator();
        ImGui::TextUnformatted("水面");
        ImGui::SliderFloat("反射强度", &reflectStr_, 0.0f, 1.0f);
        ImGui::SliderFloat("折射强度", &refractStr_, 0.0f, 1.0f);
        ImGui::SliderFloat("波高",     &waveHeight_,  0.0f, 0.2f);
        ImGui::Separator();
        const char* views[]={"完整合成","仅HDR","仅Bloom","亮区"};
        ImGui::Combo("调试视图", &debugView_, views, 4);
    }

    void drawFrame()
    {
        vkWaitForFences(device_,1,&inFlight_[currentFrame_],VK_TRUE,UINT64_MAX);
        uint32_t imgIdx=0;
        VkResult res=vkAcquireNextImageKHR(device_,swapchain_,UINT64_MAX,
                                           imgAvail_[currentFrame_],VK_NULL_HANDLE,&imgIdx);
        if (res==VK_ERROR_OUT_OF_DATE_KHR) { recreateSwapchain(); return; }
        vkResetFences(device_,1,&inFlight_[currentFrame_]);
        updateUBOs(currentFrame_);

        VkCommandBuffer cmd=cmdBuffers_[currentFrame_];
        vkResetCommandBuffer(cmd,0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(cmd,&bi));
        interactive_.beginGpuSection(cmd,currentFrame_);

        glm::mat4 view=interactive_.camera().viewMatrix();
        glm::mat4 proj=glm::perspective(glm::radians(45.0f),
                                        float(extent_.width)/float(extent_.height),0.1f,100.0f);
        proj[1][1]*=-1;

        // ── Pass 0: Particle Compute ──
        if (showParticle_) {
            vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,computePipeline_);
            vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,computeLayout_,
                                    0,1,&computeSets_[currentFrame_],0,nullptr);
            vkCmdDispatch(cmd,N_PARTS/256,1,1);
            VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            bmb.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
            bmb.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
            bmb.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            bmb.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            bmb.buffer=particleBuffers_[currentFrame_]; bmb.offset=0; bmb.size=VK_WHOLE_SIZE;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,0,0,nullptr,1,&bmb,0,nullptr);
        }

        // ── Pass 1: Shadow ──
        if (showShadow_) {
            VkClearValue shadowClear{}; shadowClear.depthStencil.depth=1.0f;
            VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
            rbi.renderPass=shadowRP_; rbi.framebuffer=shadowFB_;
            rbi.renderArea={0,0,SHADOW_DIM,SHADOW_DIM};
            rbi.clearValueCount=1; rbi.pClearValues=&shadowClear;
            vkCmdBeginRenderPass(cmd,&rbi,VK_SUBPASS_CONTENTS_INLINE);
            VkViewport svp{0,0,float(SHADOW_DIM),float(SHADOW_DIM),0,1};
            VkRect2D ssc{{0,0},{SHADOW_DIM,SHADOW_DIM}};
            vkCmdSetViewport(cmd,0,1,&svp); vkCmdSetScissor(cmd,0,1,&ssc);
            vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,shadowPipeline_);
            ShadowPC spc{makeLightMVP()};
            vkCmdPushConstants(cmd,shadowLayout_,VK_SHADER_STAGE_VERTEX_BIT,0,sizeof(ShadowPC),&spc);
            VkDeviceSize zero=0;
            vkCmdBindVertexBuffers(cmd,0,1,&terrainVB_,&zero);
            vkCmdDraw(cmd,terrainVertCount_,1,0,0);
            vkCmdBindVertexBuffers(cmd,0,1,&pillarVB_,&zero);
            vkCmdDraw(cmd,pillarVertCount_,1,0,0);
            vkCmdEndRenderPass(cmd);
        }

        // ── Pass 2&3: Water RTT（Reflect + Refract）──
        if (showWater_) {
            // RTT scene pass 使用 water_scene.vert/frag (存在 fogLayout_/fogPipeline_)
            auto drawRttScene = [&](VkFramebuffer fb, const glm::mat4& rttView, glm::vec4 clip) {
                VkClearValue clears[2]{};
                clears[0].color.float32[0]=0.5f; clears[0].color.float32[1]=0.6f;
                clears[0].color.float32[2]=0.8f; clears[0].color.float32[3]=1.0f;
                clears[1].depthStencil.depth=1.0f;
                VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
                rbi.renderPass=rttRP_; rbi.framebuffer=fb;
                rbi.renderArea={0,0,RTT_W,RTT_H};
                rbi.clearValueCount=2; rbi.pClearValues=clears;
                vkCmdBeginRenderPass(cmd,&rbi,VK_SUBPASS_CONTENTS_INLINE);
                VkViewport rvp{0,0,float(RTT_W),float(RTT_H),0,1};
                VkRect2D rsc{{0,0},{RTT_W,RTT_H}};
                vkCmdSetViewport(cmd,0,1,&rvp); vkCmdSetScissor(cmd,0,1,&rsc);
                vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,fogPipeline_);
                // 更新 scene UBO 到 rttView
                SceneUBO rttUBO{glm::mat4(1.0f),rttView,proj,makeLightMVP(),glm::vec4(lightDir_,0)};
                std::memcpy(sceneMapped_[currentFrame_],&rttUBO,sizeof(rttUBO));
                vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,fogLayout_,
                                        0,1,&sceneSets_[currentFrame_],0,nullptr);
                ClipPC cp{clip};
                vkCmdPushConstants(cmd,fogLayout_,
                                   VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,
                                   0,sizeof(ClipPC),&cp);
                VkDeviceSize zero=0;
                vkCmdBindVertexBuffers(cmd,0,1,&terrainVB_,&zero);
                vkCmdDraw(cmd,terrainVertCount_,1,0,0);
                vkCmdBindVertexBuffers(cmd,0,1,&pillarVB_,&zero);
                vkCmdDraw(cmd,pillarVertCount_,1,0,0);
                vkCmdEndRenderPass(cmd);
            };

            glm::mat4 reflectV = makeReflectView(view);
            drawRttScene(reflectFB_, reflectV, glm::vec4(0,1,0,0));   // reflect：裁剪 y<0
            drawRttScene(refractFB_, view,     glm::vec4(0,-1,0,0));  // refract：裁剪 y>0

            // 还原正常 scene UBO
            SceneUBO subo{glm::mat4(1.0f),view,proj,makeLightMVP(),glm::vec4(lightDir_,0)};
            std::memcpy(sceneMapped_[currentFrame_],&subo,sizeof(subo));
        }

        // ── Pass 4: Scene（HDR + Bright，含水面+粒子）——使用 Render Graph 管理屏障 ──
        // 先让 graph 把 HDR 和 Bright 转换到 COLOR_ATTACHMENT
        // (graph.execute 管理 blurH/V，但我们手动管理 scene pass)
        {
            auto barrierToColor = [&](RgTextureHandle h) {
                VkImageLayout oldLayout = graph_.getLayout(h);
                if (oldLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) return;
                VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                b.srcAccessMask = (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) ? 0
                                  : VK_ACCESS_SHADER_READ_BIT;
                b.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                b.oldLayout = oldLayout;
                b.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image = graph_.getImage(h);
                b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
                VkPipelineStageFlags srcStage = (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED)
                    ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                    : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
                vkCmdPipelineBarrier(cmd, srcStage,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    0,0,nullptr,0,nullptr,1,&b);
                graph_.setLayout(h, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            };
            barrierToColor(hdrH_);
            barrierToColor(brightH_);
        }

        VkClearValue sClears[3]{};
        sClears[0].color.float32[0]=0.5f; sClears[0].color.float32[1]=0.6f;
        sClears[0].color.float32[2]=0.8f; sClears[0].color.float32[3]=1.0f;
        sClears[2].depthStencil.depth=1.0f;
        VkRenderPassBeginInfo srbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        srbi.renderPass=sceneRP_; srbi.framebuffer=sceneFB_;
        srbi.renderArea.extent=extent_; srbi.clearValueCount=3; srbi.pClearValues=sClears;
        vkCmdBeginRenderPass(cmd,&srbi,VK_SUBPASS_CONTENTS_INLINE);
        VkViewport vp{0,0,float(extent_.width),float(extent_.height),0,1};
        VkRect2D sc{{0,0},extent_};
        vkCmdSetViewport(cmd,0,1,&vp); vkCmdSetScissor(cmd,0,1,&sc);

        // 地形 + 柱子
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,scenePipeline_);
        vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,sceneLayout_,
                                0,1,&sceneSets_[currentFrame_],0,nullptr);
        ClipPC noClip{glm::vec4(0)};
        vkCmdPushConstants(cmd,sceneLayout_,VK_SHADER_STAGE_VERTEX_BIT,0,sizeof(ClipPC),&noClip);
        VkDeviceSize zero=0;
        vkCmdBindVertexBuffers(cmd,0,1,&terrainVB_,&zero);
        vkCmdDraw(cmd,terrainVertCount_,1,0,0);
        vkCmdBindVertexBuffers(cmd,0,1,&pillarVB_,&zero);
        vkCmdDraw(cmd,pillarVertCount_,1,0,0);

        // 水面
        if (showWater_) {
            vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,waterPipeline_);
            vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,waterLayout_,
                                    0,1,&waterSets_[currentFrame_],0,nullptr);
            WaterPC wpc{reflectStr_,refractStr_,distortStr_,0.3f};
            vkCmdPushConstants(cmd,waterLayout_,VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(WaterPC),&wpc);
            vkCmdBindVertexBuffers(cmd,0,1,&waterVB_,&zero);
            vkCmdDraw(cmd,waterVertCount_,1,0,0);
        }

        // 粒子
        if (showParticle_) {
            vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,particlePipeline_);
            vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,particleLayout_,
                                    0,1,&particleSets_[currentFrame_],0,nullptr);
            vkCmdDraw(cmd,N_PARTS*6,1,0,0);
        }

        vkCmdEndRenderPass(cmd);

        // 更新 graph layout（HDR+Bright 现在是 COLOR_ATTACHMENT 状态）
        // 转换到 SHADER_READ_ONLY 供后续 pass 读取
        {
            auto barrierToShaderRead = [&](RgTextureHandle h) {
                VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                b.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                b.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image = graph_.getImage(h);
                b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
                vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,0,nullptr,0,nullptr,1,&b);
                graph_.setLayout(h, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            };
            barrierToShaderRead(hdrH_);
            barrierToShaderRead(brightH_);
        }

        // ── Pass 5&6: BlurH + BlurV（通过 Render Graph）──
        if (showBloom_)
            graph_.execute(cmd, currentFrame_);

        // blurV → SHADER_READ_ONLY（供 composite 采样，无论 Bloom 开关如何）
        {
            VkImageLayout curLayout = graph_.getLayout(blurVH_);
            if (curLayout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                b.oldLayout = curLayout;
                b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image = graph_.getImage(blurVH_);
                b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
                VkPipelineStageFlags srcStage = (curLayout == VK_IMAGE_LAYOUT_UNDEFINED)
                    ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                    : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
                vkCmdPipelineBarrier(cmd, srcStage,
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,0,0,nullptr,0,nullptr,1,&b);
                graph_.setLayout(blurVH_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }
        }

        // ── Pass 7: Composite（Fog + ACES + Bloom → 屏幕）──
        VkClearValue compClear{}; compClear.color.float32[3]=1.0f;
        VkRenderPassBeginInfo crbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        crbi.renderPass=compositeRP_; crbi.framebuffer=swapFBs_[imgIdx];
        crbi.renderArea.extent=extent_; crbi.clearValueCount=1; crbi.pClearValues=&compClear;
        vkCmdBeginRenderPass(cmd,&crbi,VK_SUBPASS_CONTENTS_INLINE);
        vkCmdSetViewport(cmd,0,1,&vp); vkCmdSetScissor(cmd,0,1,&sc);
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,compositePipeline_);
        vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,compositeLayout_,
                                0,1,&compositeSets_[currentFrame_],0,nullptr);
        CompositePC cpc{bloomStr_,exposure_,debugView_,0.0f};
        vkCmdPushConstants(cmd,compositeLayout_,VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(CompositePC),&cpc);
        vkCmdDraw(cmd,3,1,0,0);
        interactive_.renderUi(cmd);
        vkCmdEndRenderPass(cmd);

        // 通知 graph HDR 现在是 SHADER_READ_ONLY（已被 composite 读取）
        graph_.setLayout(hdrH_,   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        graph_.setLayout(blurVH_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        interactive_.endGpuSection(cmd,currentFrame_);
        VK_CHECK(vkEndCommandBuffer(cmd));

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        VkPipelineStageFlags wait=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        si.waitSemaphoreCount=1; si.pWaitSemaphores=&imgAvail_[currentFrame_]; si.pWaitDstStageMask=&wait;
        si.commandBufferCount=1; si.pCommandBuffers=&cmd;
        si.signalSemaphoreCount=1; si.pSignalSemaphores=&renderDone_[currentFrame_];
        VK_CHECK(vkQueueSubmit(gQueue_,1,&si,inFlight_[currentFrame_]));

        VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pi.waitSemaphoreCount=1; pi.pWaitSemaphores=&renderDone_[currentFrame_];
        pi.swapchainCount=1; pi.pSwapchains=&swapchain_; pi.pImageIndices=&imgIdx;
        res=vkQueuePresentKHR(pQueue_,&pi);
        if (res==VK_ERROR_OUT_OF_DATE_KHR||res==VK_SUBOPTIMAL_KHR||resized_) {
            resized_=false; recreateSwapchain();
        }
        interactive_.endFrame(currentFrame_);
        currentFrame_=(currentFrame_+1)%MAX_FRAMES;
    }

    glm::mat4 makeLightMVP() const
    {
        glm::mat4 lightView = glm::lookAt(-glm::normalize(lightDir_) * 20.0f,
                                          glm::vec3(0), glm::vec3(0,1,0));
        glm::mat4 lightProj = glm::ortho(-12.0f,12.0f,-12.0f,12.0f, 0.1f, 50.0f);
        lightProj[1][1] *= -1;
        return lightProj * lightView;
    }

    glm::mat4 makeReflectView(const glm::mat4& view) const
    {
        glm::mat4 inv = glm::inverse(view);
        glm::vec3 pos    = glm::vec3(inv[3]);
        glm::vec3 front  = -glm::vec3(inv[2]);
        glm::vec3 up     =  glm::vec3(inv[1]);
        pos.y   = -pos.y; front.y = -front.y; up.y = -up.y;
        return glm::lookAt(pos, pos+front, up);
    }

    void updateUBOs(uint32_t fi)
    {
        glm::mat4 view = interactive_.camera().viewMatrix();
        glm::mat4 proj = glm::perspective(glm::radians(45.0f),
                                          float(extent_.width)/float(extent_.height),0.1f,100.0f);
        proj[1][1] *= -1;
        glm::mat4 lmvp  = makeLightMVP();

        SceneUBO subo{glm::mat4(1.0f), view, proj, lmvp, glm::vec4(lightDir_,0)};
        std::memcpy(sceneMapped_[fi], &subo, sizeof(subo));

        WaterUBO wubo{};
        wubo.model=glm::mat4(1.0f); wubo.view=view; wubo.proj=proj;
        wubo.reflectView=makeReflectView(view);
        wubo.cameraPos=glm::vec4(interactive_.camera().eyePosition(),1);
        wubo.time=totalTime_; wubo.waveHeight=waveHeight_; wubo.waveSpeed=0.6f; wubo.tiling=0.5f;
        std::memcpy(waterMapped_[fi], &wubo, sizeof(wubo));

        // 篝火发射器
        EmitterUBO eu{};
        eu.position  = glm::vec4(0, 0, 0, 1);
        eu.direction = glm::vec4(0, 1, 0, 0.4f);
        eu.colorMin  = glm::vec4(1,0.3f,0,1); eu.colorMax=glm::vec4(1,0.8f,0.1f,1);
        eu.speed=2.5f; eu.speedVar=0.8f; eu.lifetime=1.0f; eu.lifetimeVar=0.3f;
        eu.size=0.2f; eu.emitRate=0;
        eu.emitterCount=1; eu.deltaTime=deltaTime_; eu.time=totalTime_;
        eu.randomSeed=static_cast<uint32_t>(totalTime_*1000);
        std::memcpy(emitterMapped_[fi], &eu, sizeof(eu));

        glm::mat4 v=view;
        CameraUBO cu{};
        cu.view=v; cu.proj=proj;
        cu.cameraRight=glm::vec4(v[0][0],v[1][0],v[2][0],0);
        cu.cameraUp   =glm::vec4(v[0][1],v[1][1],v[2][1],0);
        std::memcpy(cameraMapped_[fi], &cu, sizeof(cu));
    }

    void recreateSwapchain()
    {
        int w=0,h=0;
        glfwGetFramebufferSize(window_,&w,&h);
        while(w==0||h==0){glfwGetFramebufferSize(window_,&w,&h);glfwWaitEvents();}
        vkDeviceWaitIdle(device_);
        vkDestroyFramebuffer(device_,sceneFB_,nullptr);
        for (auto fb:swapFBs_) vkDestroyFramebuffer(device_,fb,nullptr); swapFBs_.clear();
        destroyDepthResources(device_,sceneDepth_);
        destroyDepthResources(device_,screenDepth_);
        for (auto v:swapViews_) vkDestroyImageView(device_,v,nullptr);
        vkDestroySwapchainKHR(device_,swapchain_,nullptr);
        createSwapchain(); createImageViews();
        screenDepth_=createDepthResources(physDev_,device_,extent_);
        graph_.resize(device_,physDev_,cmdPool_,gQueue_,extent_);
        createSceneDepth();
        // 重建 sceneFB
        std::array<VkImageView,3> sfAtts={graph_.getView(hdrH_),graph_.getView(brightH_),sceneDepth_.view};
        VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fci.renderPass=sceneRP_; fci.attachmentCount=3; fci.pAttachments=sfAtts.data();
        fci.width=extent_.width; fci.height=extent_.height; fci.layers=1;
        VK_CHECK(vkCreateFramebuffer(device_,&fci,nullptr,&sceneFB_));
        swapFBs_.resize(swapImages_.size());
        for (size_t i=0;i<swapImages_.size();++i) {
            fci.renderPass=compositeRP_; fci.attachmentCount=1; fci.pAttachments=&swapViews_[i];
            VK_CHECK(vkCreateFramebuffer(device_,&fci,nullptr,&swapFBs_[i]));
        }
        // 重建描述符（view 地址可能变化）
        VK_CHECK(vkResetDescriptorPool(device_,pool_,0));
        sceneSets_.clear(); waterSets_.clear(); particleSets_.clear();
        computeSets_.clear(); fogSets_.clear(); compositeSets_.clear();
        blurHSet_=VK_NULL_HANDLE; blurVSet_=VK_NULL_HANDLE;
        createDescriptorSets();
        interactive_.onSwapchainRecreated(compositeRP_,swapFormat_,static_cast<uint32_t>(swapImages_.size()));
    }

    void cleanup()
    {
        vkDestroyDescriptorPool(device_,pool_,nullptr);
        auto destroyDSL=[&](VkDescriptorSetLayout& dsl){ vkDestroyDescriptorSetLayout(device_,dsl,nullptr); };
        destroyDSL(sceneDSL_); destroyDSL(waterDSL_); destroyDSL(particleDSL_);
        destroyDSL(computeDSL_); destroyDSL(blurDSL_); destroyDSL(fogDSL_); destroyDSL(compositeDSL_);
        auto destroyPP=[&](VkPipeline& p, VkPipelineLayout& l){
            vkDestroyPipeline(device_,p,nullptr); vkDestroyPipelineLayout(device_,l,nullptr); };
        destroyPP(shadowPipeline_,shadowLayout_); destroyPP(scenePipeline_,sceneLayout_);
        destroyPP(waterPipeline_,waterLayout_);   destroyPP(fogPipeline_,fogLayout_);
        destroyPP(particlePipeline_,particleLayout_); destroyPP(computePipeline_,computeLayout_);
        destroyPP(blurPipeline_,blurLayout_);     destroyPP(compositePipeline_,compositeLayout_);
        vkDestroyRenderPass(device_,shadowRP_,nullptr);
        vkDestroyRenderPass(device_,rttRP_,nullptr);
        vkDestroyRenderPass(device_,sceneRP_,nullptr);
        vkDestroyRenderPass(device_,compositeRP_,nullptr);
        vkDestroySampler(device_,linearSampler_,nullptr);
        vkDestroySampler(device_,shadowSampler_,nullptr);
        vkDestroyFramebuffer(device_,shadowFB_,nullptr);
        vkDestroyFramebuffer(device_,reflectFB_,nullptr);
        vkDestroyFramebuffer(device_,refractFB_,nullptr);
        vkDestroyFramebuffer(device_,sceneFB_,nullptr);
        for (auto fb:swapFBs_) vkDestroyFramebuffer(device_,fb,nullptr);
        vkDestroyImageView(device_,shadowView_,nullptr);
        vkDestroyImage(device_,shadowImage_,nullptr); vkFreeMemory(device_,shadowMem_,nullptr);
        vkDestroyImageView(device_,reflectView_,nullptr);
        vkDestroyImage(device_,reflectImage_,nullptr); vkFreeMemory(device_,reflectMem_,nullptr);
        vkDestroyImageView(device_,refractView_,nullptr);
        vkDestroyImage(device_,refractImage_,nullptr); vkFreeMemory(device_,refractMem_,nullptr);
        vkDestroyImageView(device_,rttDepthView_,nullptr);
        vkDestroyImage(device_,rttDepthImage_,nullptr); vkFreeMemory(device_,rttDepthMem_,nullptr);
        graph_.destroy(device_);
        destroyDepthResources(device_,sceneDepth_);
        for (int i=0;i<MAX_FRAMES;++i) {
            vkDestroyBuffer(device_,sceneUBOs_[i],nullptr); vkFreeMemory(device_,sceneUBOMem_[i],nullptr);
            vkDestroyBuffer(device_,waterUBOs_[i],nullptr); vkFreeMemory(device_,waterUBOMem_[i],nullptr);
            vkDestroyBuffer(device_,emitterUBOs_[i],nullptr); vkFreeMemory(device_,emitterUBOMem_[i],nullptr);
            vkDestroyBuffer(device_,cameraUBOs_[i],nullptr); vkFreeMemory(device_,cameraUBOMem_[i],nullptr);
            vkDestroyBuffer(device_,particleBuffers_[i],nullptr); vkFreeMemory(device_,particleMem_[i],nullptr);
            vkDestroySemaphore(device_,imgAvail_[i],nullptr);
            vkDestroySemaphore(device_,renderDone_[i],nullptr);
            vkDestroyFence(device_,inFlight_[i],nullptr);
        }
        vkDestroyBuffer(device_,terrainVB_,nullptr); vkFreeMemory(device_,terrainVBMem_,nullptr);
        vkDestroyBuffer(device_,pillarVB_,nullptr);  vkFreeMemory(device_,pillarVBMem_,nullptr);
        vkDestroyBuffer(device_,waterVB_,nullptr);   vkFreeMemory(device_,waterVBMem_,nullptr);
        vkDestroyCommandPool(device_,cmdPool_,nullptr);
        destroyDepthResources(device_,screenDepth_);
        for (auto v:swapViews_) vkDestroyImageView(device_,v,nullptr);
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
    std::cout << "══════════════════════════════════════════════════════════════\n";
    std::cout << " 第60章：户外小关卡综合 Demo\n";
    std::cout << " 技术：Shadow + Water RTT + Particles + Bloom + Volumetric Fog\n";
    std::cout << "══════════════════════════════════════════════════════════════\n\n";
    std::cout << "控制：LMB 旋转 | RMB 平移 | 滚轮缩放 | WASD 移动 | ESC 退出\n\n";
    try { Ch60App app; app.run(); std::cout<<"✅ 正常退出\n"; }
    catch(const std::exception& e) { std::cerr<<"❌ "<<e.what()<<"\n"; return 1; }
}
