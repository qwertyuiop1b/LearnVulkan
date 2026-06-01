/**
 * @file ch84_toon_rendering.cpp
 * @brief 第84章：卡通渲染（升级版）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【三 Pass 管线】
 *
 *  Pass 1 — Outline（Inverted Hull 轮廓线）
 *    绘制背面并沿法线外扩，产生物体外轮廓。
 *
 *  Pass 2 — Toon Shading（写入颜色 + 法线）
 *    量化漫反射 + 交叉线条阴影 + 高光 + 边缘光。
 *    同时输出法线到第二颜色 Attachment（供 Pass 3 使用）。
 *
 *  Pass 3 — Sobel Edge Detection（后处理）
 *    对 Pass 2 的法线缓冲区 + 深度缓冲区做 Sobel 算子，
 *    检测法线不连续（内部折痕）和深度不连续（轮廓），
 *    生成手绘风格的"内部线条"叠加到场景上。
 *
 * 【四种风格预设】（ImGui 一键切换）
 *   Anime  — 3 色阶，饱和度提升，高对比
 *   Manga  — 2 色阶，单色 + 交叉线条
 *   Comic  — 4 色阶，高饱和度，类漫画本风格
 *   Sketch — 3 色阶，低饱和，密集交叉线条
 *
 * 【ImGui 控制】
 *   风格预设 / 色阶数 / 阴影阈值 / 高光大小 / 边缘光
 *   轮廓线宽度+颜色 / 内部线条阈值+强度 / 交叉线条密度
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
#include <cmath>

using namespace vulkan_tutorial;

constexpr uint32_t WIDTH = 1000;
constexpr uint32_t HEIGHT = 750;
constexpr int MAX_FRAMES = 2;

// ─── 数据结构 ────────────────────────────────────────────────────────────────

struct SceneVertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec3 color;
};

struct ToonUBO {
    alignas(16) glm::mat4 model, view, proj;
    alignas(16) glm::vec4 lightDir;
    alignas(16) glm::vec4 lightColor;
    alignas(16) glm::vec4 cameraPos;
    float specularSize;
    float rimPower;
    float pad[2];
};

struct ToonPC {
    int shadingBands;
    float shadowThreshold;
    float shadowSmooth;
    int enableRim;
    int enableSpecular;
    int stylePreset;
    float hatchDensity;
    int enableHatching;
};

struct OutlinePC {
    float outlineWidth;
    float outlineWidthNDC;
    int useNDCWidth;
    float pad1;
    glm::vec4 outlineColor;
};

struct EdgePC {
    glm::vec2 texelSize;
    float normalThreshold;
    float depthThreshold;
    float edgeStrength;
    glm::vec4 edgeColor;
    int onlyEdges;
    float pad[3];
};

// ─── 场景几何 ─────────────────────────────────────────────────────────────────

static std::vector<SceneVertex>
buildSphere(glm::vec3 center, float r, glm::vec3 color, int stacks = 20, int slices = 30) {
    std::vector<SceneVertex> v;
    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            auto vtx = [&](int ii, int jj) {
                float phi = float(ii) / stacks * glm::pi<float>();
                float th = float(jj) / slices * 2.0f * glm::pi<float>();
                glm::vec3 n{sinf(phi) * cosf(th), cosf(phi), sinf(phi) * sinf(th)};
                return SceneVertex{center + n * r, n, color};
            };
            v.push_back(vtx(i, j));
            v.push_back(vtx(i + 1, j));
            v.push_back(vtx(i + 1, j + 1));
            v.push_back(vtx(i, j));
            v.push_back(vtx(i + 1, j + 1));
            v.push_back(vtx(i, j + 1));
        }
    }
    return v;
}

static std::vector<SceneVertex> buildCube(glm::vec3 center, float size, glm::vec3 color) {
    float h = size * 0.5f;
    std::vector<SceneVertex> v;
    auto face = [&](glm::vec3 n, std::array<glm::vec3, 4> pts) {
        v.push_back({center + pts[0], n, color});
        v.push_back({center + pts[1], n, color});
        v.push_back({center + pts[2], n, color});
        v.push_back({center + pts[0], n, color});
        v.push_back({center + pts[2], n, color});
        v.push_back({center + pts[3], n, color});
    };
    face({0, 0, -1}, {{{-h, -h, -h}, {h, -h, -h}, {h, h, -h}, {-h, h, -h}}});
    face({0, 0, 1}, {{{h, -h, h}, {-h, -h, h}, {-h, h, h}, {h, h, h}}});
    face({-1, 0, 0}, {{{-h, -h, h}, {-h, -h, -h}, {-h, h, -h}, {-h, h, h}}});
    face({1, 0, 0}, {{{h, -h, -h}, {h, -h, h}, {h, h, h}, {h, h, -h}}});
    face({0, -1, 0}, {{{-h, -h, h}, {h, -h, h}, {h, -h, -h}, {-h, -h, -h}}});
    face({0, 1, 0}, {{{-h, h, -h}, {h, h, -h}, {h, h, h}, {-h, h, h}}});
    return v;
}

static std::vector<SceneVertex> buildGround() {
    return {
        {{-5, -1, -5}, {0, 1, 0}, {0.6f, 0.58f, 0.5f}},
        {{5, -1, -5}, {0, 1, 0}, {0.6f, 0.58f, 0.5f}},
        {{5, -1, 5}, {0, 1, 0}, {0.6f, 0.58f, 0.5f}},
        {{-5, -1, -5}, {0, 1, 0}, {0.6f, 0.58f, 0.5f}},
        {{5, -1, 5}, {0, 1, 0}, {0.6f, 0.58f, 0.5f}},
        {{-5, -1, 5}, {0, 1, 0}, {0.6f, 0.58f, 0.5f}},
    };
}

// ─── App ─────────────────────────────────────────────────────────────────────

class Ch84App {
  public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

  private:
    // Vulkan 基础资源
    GLFWwindow* window_ = nullptr;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice physDev_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue gQueue_ = VK_NULL_HANDLE;
    VkQueue pQueue_ = VK_NULL_HANDLE;
    QueueFamilyIndices qIdx_{};
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    std::vector<VkImage> swapImages_;
    std::vector<VkImageView> swapViews_;
    std::vector<VkFramebuffer> swapFBs_;
    VkFormat swapFmt_ = VK_FORMAT_UNDEFINED;
    VkExtent2D extent_{};
    DepthResources depth_{};
    VkCommandPool cmdPool_ = VK_NULL_HANDLE;

    // 离屏资源（Pass 2 的颜色 + 法线输出）
    VkImage colorImage_ = VK_NULL_HANDLE;
    VkDeviceMemory colorMem_ = VK_NULL_HANDLE;
    VkImageView colorView_ = VK_NULL_HANDLE;
    VkImage normalImage_ = VK_NULL_HANDLE;
    VkDeviceMemory normalMem_ = VK_NULL_HANDLE;
    VkImageView normalView_ = VK_NULL_HANDLE;
    VkFramebuffer offscreenFB_ = VK_NULL_HANDLE;

    // RenderPasses
    VkRenderPass toonRP_ = VK_NULL_HANDLE; // Pass 1&2（轮廓 + 卡通着色，输出到离屏）
    VkRenderPass edgeRP_ = VK_NULL_HANDLE; // Pass 3（边缘检测，输出到交换链）

    // Pipelines
    VkPipeline outlinePipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout outlineLayout_ = VK_NULL_HANDLE;
    VkPipeline toonPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout toonLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout toonDSL_ = VK_NULL_HANDLE;
    VkPipeline edgePipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout edgeLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout edgeDSL_ = VK_NULL_HANDLE;

    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> toonSets_;
    std::vector<VkDescriptorSet> edgeSets_;
    VkSampler sampler_ = VK_NULL_HANDLE;

    // 场景几何
    VkBuffer sceneVB_ = VK_NULL_HANDLE;
    VkDeviceMemory sceneVBMem_ = VK_NULL_HANDLE;
    uint32_t sceneVC_ = 0;
    std::vector<VkBuffer> ubos_;
    std::vector<VkDeviceMemory> uboMem_;
    std::vector<void*> uboMapped_;

    std::vector<VkCommandBuffer> cmds_;
    std::vector<VkSemaphore> imgAvail_, renderDone_;
    std::vector<VkFence> inFlight_;
    uint32_t frame_ = 0;
    bool resized_ = false;

    InteractiveChapterTools interactive_;
    float totalTime_ = 0.0f;

    // ── 卡通参数 ────────────────────────────────────────────────────────────
    int stylePreset_ = 0; // 0=Anime,1=Manga,2=Comic,3=Sketch
    int shadingBands_ = 3;
    float shadowThreshold_ = 0.3f;
    float shadowSmooth_ = 0.04f;
    bool enableRim_ = true;
    float rimPower_ = 3.5f;
    bool enableSpecular_ = true;
    float specularSize_ = 0.4f;
    bool enableHatching_ = false;
    float hatchDensity_ = 1.0f;
    float outlineWidthNDC_ = 0.003f;
    glm::vec4 outlineColor_{0.05f, 0.05f, 0.05f, 1.0f};
    float normalThreshold_ = 0.3f;
    float depthThreshold_ = 0.001f;
    float edgeStrength_ = 0.9f;
    glm::vec4 edgeColor_{0.05f, 0.05f, 0.05f, 1.0f};
    bool enableSobelEdge_ = true;
    bool debugEdgeOnly_ = false;
    glm::vec3 lightDir_ = glm::normalize(glm::vec3(1, 2, 1));
    float lightElev_ = 55.0f;
    float lightAzim_ = 40.0f;
    glm::vec3 lightColor_{1.0f, 0.95f, 0.85f};

    // ─── 初始化 ─────────────────────────────────────────────────────────────

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "第84章：卡通渲染（升级版）", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
            reinterpret_cast<Ch84App*>(glfwGetWindowUserPointer(w))->resized_ = true;
        });
        interactive_.attachInput(window_);
    }

    void initVulkan() {
        createInstance(instance_);
        VK_CHECK(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_));
        pickPhysicalDevice(instance_, surface_, physDev_);
        createLogicalDevice(physDev_, surface_, device_, gQueue_, pQueue_, qIdx_);
        createSwapchain();
        createImageViews();
        createDepthWithSampling();
        createCommandPool();
        createOffscreenResources();
        createSampler();
        createRenderPasses();
        createFramebuffers();
        buildGeometry();
        createUniformBuffers();
        createDescriptorLayouts();
        createDescriptorPool();
        createDescriptorSets();
        createPipelines();
        createCommandBuffers();
        createSyncObjects();

        InteractiveInitInfo ii{};
        ii.window = window_;
        ii.instance = instance_;
        ii.physicalDevice = physDev_;
        ii.device = device_;
        ii.graphicsQueue = gQueue_;
        ii.queueFamily = qIdx_.graphicsFamily.value();
        ii.renderPass = edgeRP_;
        ii.swapchainFormat = swapFmt_;
        ii.imageCount = static_cast<uint32_t>(swapImages_.size());
        ii.maxFramesInFlight = MAX_FRAMES;
        interactive_.initVulkan(ii);
        interactive_.camera().setDistance(7.0f);
        interactive_.camera().setAngles(30.0f, 20.0f);
    }

    void createSwapchain() {
        auto d = querySwapChainSupport(physDev_, surface_);
        auto f = chooseSwapSurfaceFormat(d.formats);
        auto m = chooseSwapPresentMode(d.presentModes);
        extent_ = chooseSwapExtent(d.capabilities, window_);
        swapFmt_ = f.format;
        uint32_t cnt = d.capabilities.minImageCount + 1;
        if (d.capabilities.maxImageCount)
            cnt = std::min(cnt, d.capabilities.maxImageCount);
        VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        sci.surface = surface_;
        sci.minImageCount = cnt;
        sci.imageFormat = f.format;
        sci.imageColorSpace = f.colorSpace;
        sci.imageExtent = extent_;
        sci.imageArrayLayers = 1;
        sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        uint32_t qf[2] = {qIdx_.graphicsFamily.value(), qIdx_.presentFamily.value()};
        if (qf[0] != qf[1]) {
            sci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            sci.queueFamilyIndexCount = 2;
            sci.pQueueFamilyIndices = qf;
        } else
            sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        sci.preTransform = d.capabilities.currentTransform;
        sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        sci.presentMode = m;
        sci.clipped = VK_TRUE;
        VK_CHECK(vkCreateSwapchainKHR(device_, &sci, nullptr, &swapchain_));
        uint32_t n = 0;
        vkGetSwapchainImagesKHR(device_, swapchain_, &n, nullptr);
        swapImages_.resize(n);
        vkGetSwapchainImagesKHR(device_, swapchain_, &n, swapImages_.data());
    }

    void createImageViews() {
        swapViews_.resize(swapImages_.size());
        for (size_t i = 0; i < swapImages_.size(); ++i) {
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = swapImages_[i];
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format = swapFmt_;
            vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VK_CHECK(vkCreateImageView(device_, &vi, nullptr, &swapViews_[i]));
        }
    }

    void createDepthWithSampling() {
        // createDepthResources 只用 DEPTH_STENCIL_ATTACHMENT，不含 SAMPLED_BIT
        // 这里手动创建带 SAMPLED_BIT 的深度缓冲区
        VkFormat fmt = findDepthFormat(physDev_);
        depth_.format = fmt;
        createImage(physDev_,
                    device_,
                    extent_.width,
                    extent_.height,
                    fmt,
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    depth_.image,
                    depth_.memory);
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = depth_.image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = fmt;
        vi.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(device_, &vi, nullptr, &depth_.view));
    }

    void createCommandPool() {
        VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = qIdx_.graphicsFamily.value();
        VK_CHECK(vkCreateCommandPool(device_, &ci, nullptr, &cmdPool_));
    }

    void createOffscreenResources() {
        auto mkColorRT = [&](VkImage& img, VkDeviceMemory& mem, VkImageView& view) {
            createImage(physDev_,
                        device_,
                        extent_.width,
                        extent_.height,
                        swapFmt_,
                        VK_IMAGE_TILING_OPTIMAL,
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        img,
                        mem);
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = img;
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format = swapFmt_;
            vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VK_CHECK(vkCreateImageView(device_, &vi, nullptr, &view));

            VkCommandBuffer cmd = beginSingleTimeCommands(device_, cmdPool_);
            VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = img;
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0,
                                 0,
                                 nullptr,
                                 0,
                                 nullptr,
                                 1,
                                 &b);
            endSingleTimeCommands(device_, cmdPool_, gQueue_, cmd);
        };
        mkColorRT(colorImage_, colorMem_, colorView_);
        mkColorRT(normalImage_, normalMem_, normalView_);
    }

    void createSampler() {
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter = VK_FILTER_NEAREST;
        si.minFilter = VK_FILTER_NEAREST;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = 1.0f;
        VK_CHECK(vkCreateSampler(device_, &si, nullptr, &sampler_));
    }

    void createRenderPasses() {
        // Toon RP（Pass1+2 输出到离屏 color + normal + depth）
        {
            std::array<VkAttachmentDescription, 3> atts{};
            // color
            atts[0].format = swapFmt_;
            atts[0].samples = VK_SAMPLE_COUNT_1_BIT;
            atts[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            atts[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            atts[0].initialLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            atts[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            // normal
            atts[1] = atts[0];
            // depth
            atts[2].format = depth_.format;
            atts[2].samples = VK_SAMPLE_COUNT_1_BIT;
            atts[2].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            atts[2].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            atts[2].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            atts[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            atts[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            atts[2].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            VkAttachmentReference cr[2] = {{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
                                           {1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}};
            VkAttachmentReference dr{2, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
            VkSubpassDescription sub{};
            sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            sub.colorAttachmentCount = 2;
            sub.pColorAttachments = cr;
            sub.pDepthStencilAttachment = &dr;
            VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
            rpci.attachmentCount = 3;
            rpci.pAttachments = atts.data();
            rpci.subpassCount = 1;
            rpci.pSubpasses = &sub;
            VK_CHECK(vkCreateRenderPass(device_, &rpci, nullptr, &toonRP_));
        }
        // Edge RP（Pass3 输出到交换链）
        {
            VkAttachmentDescription att{};
            att.format = swapFmt_;
            att.samples = VK_SAMPLE_COUNT_1_BIT;
            att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            att.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            VkAttachmentReference cr{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
            VkSubpassDescription sub{};
            sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            sub.colorAttachmentCount = 1;
            sub.pColorAttachments = &cr;
            VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
            rpci.attachmentCount = 1;
            rpci.pAttachments = &att;
            rpci.subpassCount = 1;
            rpci.pSubpasses = &sub;
            VK_CHECK(vkCreateRenderPass(device_, &rpci, nullptr, &edgeRP_));
        }
    }

    void createFramebuffers() {
        // Toon FB（离屏：颜色 + 法线 + 深度）
        std::array<VkImageView, 3> ofAtts = {colorView_, normalView_, depth_.view};
        VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fci.renderPass = toonRP_;
        fci.attachmentCount = 3;
        fci.pAttachments = ofAtts.data();
        fci.width = extent_.width;
        fci.height = extent_.height;
        fci.layers = 1;
        VK_CHECK(vkCreateFramebuffer(device_, &fci, nullptr, &offscreenFB_));

        // Edge FBs（交换链）
        swapFBs_.resize(swapImages_.size());
        fci.renderPass = edgeRP_;
        fci.attachmentCount = 1;
        for (size_t i = 0; i < swapImages_.size(); ++i) {
            fci.pAttachments = &swapViews_[i];
            VK_CHECK(vkCreateFramebuffer(device_, &fci, nullptr, &swapFBs_[i]));
        }
    }

    void buildGeometry() {
        std::vector<SceneVertex> all;
        auto add = [&](const std::vector<SceneVertex>& v) { all.insert(all.end(), v.begin(), v.end()); };
        add(buildSphere({0, 0, 0}, 1.0f, {0.95f, 0.30f, 0.20f}));
        add(buildSphere({-2.5f, 0, 0}, 0.7f, {0.20f, 0.50f, 0.90f}));
        add(buildSphere({2.5f, 0, 0}, 0.7f, {0.20f, 0.80f, 0.30f}));
        add(buildCube({0, 0, -2.5f}, 1.0f, {0.90f, 0.70f, 0.10f}));
        add(buildGround());
        sceneVC_ = static_cast<uint32_t>(all.size());

        VkDeviceSize sz = sizeof(SceneVertex) * all.size();
        VkBuffer st;
        VkDeviceMemory stm;
        createBuffer(physDev_,
                     device_,
                     sz,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     st,
                     stm);
        void* d;
        vkMapMemory(device_, stm, 0, sz, 0, &d);
        std::memcpy(d, all.data(), sz);
        vkUnmapMemory(device_, stm);
        createBuffer(physDev_,
                     device_,
                     sz,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     sceneVB_,
                     sceneVBMem_);
        copyBuffer(device_, cmdPool_, gQueue_, st, sceneVB_, sz);
        vkDestroyBuffer(device_, st, nullptr);
        vkFreeMemory(device_, stm, nullptr);
    }

    void createUniformBuffers() {
        ubos_.resize(MAX_FRAMES);
        uboMem_.resize(MAX_FRAMES);
        uboMapped_.resize(MAX_FRAMES);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            createBuffer(physDev_,
                         device_,
                         sizeof(ToonUBO),
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         ubos_[i],
                         uboMem_[i]);
            vkMapMemory(device_, uboMem_[i], 0, sizeof(ToonUBO), 0, &uboMapped_[i]);
        }
    }

    void createDescriptorLayouts() {
        // Toon DSL：1 UBO
        VkDescriptorSetLayoutBinding lb0{
            0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT};
        VkDescriptorSetLayoutCreateInfo dci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dci.bindingCount = 1;
        dci.pBindings = &lb0;
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &dci, nullptr, &toonDSL_));

        // Edge DSL：3 samplers（color, normal, depth）
        std::array<VkDescriptorSetLayoutBinding, 3> edgeBs = {{
            {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        }};
        dci.bindingCount = 3;
        dci.pBindings = edgeBs.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &dci, nullptr, &edgeDSL_));
    }

    void createDescriptorPool() {
        std::array<VkDescriptorPoolSize, 2> ps = {{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, MAX_FRAMES},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_FRAMES * 3},
        }};
        VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pci.poolSizeCount = 2;
        pci.pPoolSizes = ps.data();
        pci.maxSets = MAX_FRAMES * 2;
        VK_CHECK(vkCreateDescriptorPool(device_, &pci, nullptr, &pool_));
    }

    void createDescriptorSets() {
        auto alloc = [&](VkDescriptorSetLayout dsl, std::vector<VkDescriptorSet>& sets) {
            std::vector<VkDescriptorSetLayout> ls(MAX_FRAMES, dsl);
            VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ai.descriptorPool = pool_;
            ai.descriptorSetCount = MAX_FRAMES;
            ai.pSetLayouts = ls.data();
            sets.resize(MAX_FRAMES);
            VK_CHECK(vkAllocateDescriptorSets(device_, &ai, sets.data()));
        };
        alloc(toonDSL_, toonSets_);
        alloc(edgeDSL_, edgeSets_);

        for (int i = 0; i < MAX_FRAMES; ++i) {
            VkDescriptorBufferInfo bi{ubos_[i], 0, sizeof(ToonUBO)};
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                   nullptr,
                                   toonSets_[i],
                                   0,
                                   0,
                                   1,
                                   VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                   nullptr,
                                   &bi};
            vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);

            VkDescriptorImageInfo ci{sampler_, colorView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkDescriptorImageInfo ni{sampler_, normalView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkDescriptorImageInfo di{sampler_, depth_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            std::array<VkWriteDescriptorSet, 3> ews = {{
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                 nullptr,
                 edgeSets_[i],
                 0,
                 0,
                 1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                 &ci,
                 nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                 nullptr,
                 edgeSets_[i],
                 1,
                 0,
                 1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                 &ni,
                 nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                 nullptr,
                 edgeSets_[i],
                 2,
                 0,
                 1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                 &di,
                 nullptr},
            }};
            vkUpdateDescriptorSets(device_, 3, ews.data(), 0, nullptr);
        }
    }

    void createPipelines() {
        VkVertexInputBindingDescription bind{0, sizeof(SceneVertex), VK_VERTEX_INPUT_RATE_VERTEX};
        std::array<VkVertexInputAttributeDescription, 3> attrs = {{
            {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
            {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 12},
            {2, 0, VK_FORMAT_R32G32B32_SFLOAT, 24},
        }};
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &bind;
        vi.vertexAttributeDescriptionCount = 3;
        vi.pVertexAttributeDescriptions = attrs.data();
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vps.viewportCount = 1;
        vps.scissorCount = 1;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS;
        std::array<VkDynamicState, 2> dyn = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyns{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dyns.dynamicStateCount = 2;
        dyns.pDynamicStates = dyn.data();

        // ── Outline pipeline ──────────────────────────────────────────────
        {
            VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(OutlinePC)};
            VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            lci.setLayoutCount = 1;
            lci.pSetLayouts = &toonDSL_;
            lci.pushConstantRangeCount = 1;
            lci.pPushConstantRanges = &pcr;
            VK_CHECK(vkCreatePipelineLayout(device_, &lci, nullptr, &outlineLayout_));

            auto vert = createShaderModuleFromFile(device_, "toon_outline.vert.spv");
            auto frag = createShaderModuleFromFile(device_, "toon_outline.frag.spv");
            VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rs.polygonMode = VK_POLYGON_MODE_FILL;
            rs.cullMode = VK_CULL_MODE_FRONT_BIT;
            rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rs.lineWidth = 1.0f;
            VkPipelineColorBlendAttachmentState blends[2]{};
            blends[0].colorWriteMask = 0xF;
            blends[1].colorWriteMask = 0xF;
            VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            cbs.attachmentCount = 2;
            cbs.pAttachments = blends;
            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vert;
            stages[0].pName = "main";
            stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = frag;
            stages[1].pName = "main";
            VkGraphicsPipelineCreateInfo gci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            gci.stageCount = 2;
            gci.pStages = stages;
            gci.pVertexInputState = &vi;
            gci.pInputAssemblyState = &ia;
            gci.pViewportState = &vps;
            gci.pRasterizationState = &rs;
            gci.pMultisampleState = &ms;
            gci.pDepthStencilState = &ds;
            gci.pColorBlendState = &cbs;
            gci.pDynamicState = &dyns;
            gci.layout = outlineLayout_;
            gci.renderPass = toonRP_;
            VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gci, nullptr, &outlinePipeline_));
            vkDestroyShaderModule(device_, vert, nullptr);
            vkDestroyShaderModule(device_, frag, nullptr);
        }

        // ── Toon pipeline（dual output: color + normal）──────────────────
        {
            VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ToonPC)};
            VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            lci.setLayoutCount = 1;
            lci.pSetLayouts = &toonDSL_;
            lci.pushConstantRangeCount = 1;
            lci.pPushConstantRanges = &pcr;
            VK_CHECK(vkCreatePipelineLayout(device_, &lci, nullptr, &toonLayout_));

            auto vert = createShaderModuleFromFile(device_, "toon.vert.spv");
            auto frag = createShaderModuleFromFile(device_, "toon.frag.spv");
            VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rs.polygonMode = VK_POLYGON_MODE_FILL;
            rs.cullMode = VK_CULL_MODE_BACK_BIT;
            rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rs.lineWidth = 1.0f;
            VkPipelineColorBlendAttachmentState blends[2]{};
            blends[0].colorWriteMask = 0xF;
            blends[1].colorWriteMask = 0xF;
            VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            cbs.attachmentCount = 2;
            cbs.pAttachments = blends;
            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vert;
            stages[0].pName = "main";
            stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = frag;
            stages[1].pName = "main";
            VkGraphicsPipelineCreateInfo gci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            gci.stageCount = 2;
            gci.pStages = stages;
            gci.pVertexInputState = &vi;
            gci.pInputAssemblyState = &ia;
            gci.pViewportState = &vps;
            gci.pRasterizationState = &rs;
            gci.pMultisampleState = &ms;
            gci.pDepthStencilState = &ds;
            gci.pColorBlendState = &cbs;
            gci.pDynamicState = &dyns;
            gci.layout = toonLayout_;
            gci.renderPass = toonRP_;
            VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gci, nullptr, &toonPipeline_));
            vkDestroyShaderModule(device_, vert, nullptr);
            vkDestroyShaderModule(device_, frag, nullptr);
        }

        // ── Edge pipeline（fullscreen，Sobel）────────────────────────────
        {
            VkPushConstantRange pcr{VK_SHADER_STAGE_FRAGMENT_BIT, 0, 64}; // EdgePC + 对齐填充
            VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            lci.setLayoutCount = 1;
            lci.pSetLayouts = &edgeDSL_;
            lci.pushConstantRangeCount = 1;
            lci.pPushConstantRanges = &pcr;
            VK_CHECK(vkCreatePipelineLayout(device_, &lci, nullptr, &edgeLayout_));

            auto vert = createShaderModuleFromFile(device_, "rg_fullscreen.vert.spv");
            auto frag = createShaderModuleFromFile(device_, "toon_edge.frag.spv");
            VkPipelineVertexInputStateCreateInfo emptyVI{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rs.polygonMode = VK_POLYGON_MODE_FILL;
            rs.lineWidth = 1.0f;
            VkPipelineDepthStencilStateCreateInfo nods{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            VkPipelineColorBlendAttachmentState blend{};
            blend.colorWriteMask = 0xF;
            VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            cbs.attachmentCount = 1;
            cbs.pAttachments = &blend;
            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vert;
            stages[0].pName = "main";
            stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = frag;
            stages[1].pName = "main";
            VkGraphicsPipelineCreateInfo gci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            gci.stageCount = 2;
            gci.pStages = stages;
            gci.pVertexInputState = &emptyVI;
            gci.pInputAssemblyState = &ia;
            gci.pViewportState = &vps;
            gci.pRasterizationState = &rs;
            gci.pMultisampleState = &ms;
            gci.pDepthStencilState = &nods;
            gci.pColorBlendState = &cbs;
            gci.pDynamicState = &dyns;
            gci.layout = edgeLayout_;
            gci.renderPass = edgeRP_;
            VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gci, nullptr, &edgePipeline_));
            vkDestroyShaderModule(device_, vert, nullptr);
            vkDestroyShaderModule(device_, frag, nullptr);
        }
    }

    void createCommandBuffers() {
        cmds_.resize(MAX_FRAMES);
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = cmdPool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = MAX_FRAMES;
        VK_CHECK(vkAllocateCommandBuffers(device_, &ai, cmds_.data()));
    }

    void createSyncObjects() {
        imgAvail_.resize(MAX_FRAMES);
        renderDone_.resize(MAX_FRAMES);
        inFlight_.resize(MAX_FRAMES);
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for (int i = 0; i < MAX_FRAMES; ++i) {
            VK_CHECK(vkCreateSemaphore(device_, &si, nullptr, &imgAvail_[i]));
            VK_CHECK(vkCreateSemaphore(device_, &si, nullptr, &renderDone_[i]));
            VK_CHECK(vkCreateFence(device_, &fi, nullptr, &inFlight_[i]));
        }
    }

    // ─── 主循环 ──────────────────────────────────────────────────────────────

    void mainLoop() {
        auto last = std::chrono::high_resolution_clock::now();
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            float dt = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - last).count();
            last = std::chrono::high_resolution_clock::now();
            totalTime_ += dt;
            interactive_.beginFrame(dt);
            buildUi();
            drawFrame();
            interactive_.updateWindowTitle();
        }
        vkDeviceWaitIdle(device_);
    }

    void applyStylePreset() {
        switch (stylePreset_) {
        case 0: // Anime
            shadingBands_ = 3;
            shadowThreshold_ = 0.3f;
            shadowSmooth_ = 0.04f;
            enableSpecular_ = true;
            specularSize_ = 0.4f;
            enableRim_ = true;
            rimPower_ = 3.5f;
            enableHatching_ = false;
            outlineWidthNDC_ = 0.003f;
            outlineColor_ = {0.05f, 0.05f, 0.05f, 1};
            enableSobelEdge_ = true;
            normalThreshold_ = 0.3f;
            edgeStrength_ = 0.8f;
            break;
        case 1: // Manga
            shadingBands_ = 2;
            shadowThreshold_ = 0.4f;
            shadowSmooth_ = 0.02f;
            enableSpecular_ = false;
            enableRim_ = false;
            enableHatching_ = true;
            hatchDensity_ = 1.5f;
            outlineWidthNDC_ = 0.004f;
            outlineColor_ = {0, 0, 0, 1};
            enableSobelEdge_ = true;
            normalThreshold_ = 0.2f;
            edgeStrength_ = 1.0f;
            break;
        case 2: // Comic
            shadingBands_ = 4;
            shadowThreshold_ = 0.25f;
            shadowSmooth_ = 0.03f;
            enableSpecular_ = true;
            specularSize_ = 0.5f;
            enableRim_ = true;
            rimPower_ = 2.5f;
            enableHatching_ = false;
            outlineWidthNDC_ = 0.005f;
            outlineColor_ = {0, 0, 0, 1};
            enableSobelEdge_ = true;
            normalThreshold_ = 0.35f;
            edgeStrength_ = 0.9f;
            break;
        case 3: // Sketch
            shadingBands_ = 3;
            shadowThreshold_ = 0.4f;
            shadowSmooth_ = 0.06f;
            enableSpecular_ = false;
            enableRim_ = false;
            enableHatching_ = true;
            hatchDensity_ = 2.0f;
            outlineWidthNDC_ = 0.002f;
            outlineColor_ = {0.1f, 0.1f, 0.1f, 1};
            enableSobelEdge_ = true;
            normalThreshold_ = 0.25f;
            edgeStrength_ = 0.7f;
            break;
        }
    }

    void buildUi() {
        interactive_.buildDebugPanel("第84章：卡通渲染（升级版）");
        ImGui::Separator();

        // 风格预设按钮行
        ImGui::TextColored(ImVec4(1, 0.9f, 0.2f, 1), "风格预设（一键切换）");
        const char* presetNames[] = {"🎌 Anime", "📖 Manga", "💥 Comic", "✏️ Sketch"};
        for (int i = 0; i < 4; ++i) {
            if (i > 0)
                ImGui::SameLine();
            bool active = (stylePreset_ == i);
            if (active)
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.5f, 0.9f, 1));
            if (ImGui::Button(presetNames[i], ImVec2(120, 28))) {
                stylePreset_ = i;
                applyStylePreset();
            }
            if (active)
                ImGui::PopStyleColor();
        }

        ImGui::Separator();
        if (ImGui::CollapsingHeader("卡通光照设置", ImGuiTreeNodeFlags_DefaultOpen)) {
            const char* bandNames[] = {"平滑（Phong）", "2色阶", "3色阶", "4色阶", "5色阶"};
            int bidx = shadingBands_ <= 0 ? 0 : std::min(shadingBands_ - 1, 3);
            if (ImGui::Combo("色阶数", &bidx, bandNames, 5))
                shadingBands_ = bidx == 0 ? 0 : bidx + 1;
            ImGui::SliderFloat("阴影阈值", &shadowThreshold_, 0.1f, 0.7f);
            ImGui::SliderFloat("边缘柔化", &shadowSmooth_, 0.0f, 0.12f);
            ImGui::Checkbox("启用高光", &enableSpecular_);
            if (enableSpecular_)
                ImGui::SliderFloat("高光大小", &specularSize_, 0.05f, 0.95f);
            ImGui::Checkbox("启用边缘光", &enableRim_);
            if (enableRim_)
                ImGui::SliderFloat("边缘光强度", &rimPower_, 1.0f, 8.0f);
        }

        if (ImGui::CollapsingHeader("交叉线条（Cross-Hatching）")) {
            ImGui::Checkbox("启用交叉线条", &enableHatching_);
            if (enableHatching_)
                ImGui::SliderFloat("线条密度", &hatchDensity_, 0.5f, 3.0f);
            ImGui::TextDisabled("交叉线条会在暗部区域显示倾斜/交叉线条，产生手绘素描感");
        }

        if (ImGui::CollapsingHeader("轮廓线（Inverted Hull）", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("轮廓宽度（屏幕空间）", &outlineWidthNDC_, 0.001f, 0.01f, "%.4f");
            ImGui::ColorEdit4("轮廓颜色", &outlineColor_.x);
        }

        if (ImGui::CollapsingHeader("内部线条（Sobel 边缘检测）", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("启用 Sobel 内轮廓", &enableSobelEdge_);
            ImGui::TextDisabled("法线不连续 → 折叠边；深度不连续 → 轮廓边");
            if (enableSobelEdge_) {
                ImGui::SliderFloat("法线阈值", &normalThreshold_, 0.1f, 0.8f);
                ImGui::SliderFloat("深度阈值", &depthThreshold_, 0.0001f, 0.01f, "%.5f");
                ImGui::SliderFloat("线条强度", &edgeStrength_, 0.0f, 1.5f);
                ImGui::ColorEdit4("线条颜色", &edgeColor_.x);
                ImGui::Checkbox("调试：只显示边缘", &debugEdgeOnly_);
            }
        }

        if (ImGui::CollapsingHeader("光源设置")) {
            bool lc = false;
            lc |= ImGui::SliderFloat("仰角", &lightElev_, -90.0f, 90.0f);
            lc |= ImGui::SliderFloat("方位角", &lightAzim_, 0.0f, 360.0f);
            if (lc) {
                float el = glm::radians(lightElev_), az = glm::radians(lightAzim_);
                lightDir_ = glm::normalize(glm::vec3(cosf(el) * sinf(az), sinf(el), cosf(el) * cosf(az)));
            }
            ImGui::ColorEdit3("光源颜色", &lightColor_.x);
        }

        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.5f, 0.8f, 1, 1), "管线：Outline → Toon+法线 → Sobel边缘检测");
    }

    void drawFrame() {
        vkWaitForFences(device_, 1, &inFlight_[frame_], VK_TRUE, UINT64_MAX);
        uint32_t idx = 0;
        VkResult r = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, imgAvail_[frame_], VK_NULL_HANDLE, &idx);
        if (r == VK_ERROR_OUT_OF_DATE_KHR) {
            recreate();
            return;
        }
        vkResetFences(device_, 1, &inFlight_[frame_]);
        updateUBO(frame_);

        VkCommandBuffer cmd = cmds_[frame_];
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
        interactive_.beginGpuSection(cmd, frame_);

        // ── Pass 1+2: Toon RP（离屏，颜色+法线+深度）─────────────────────────
        {
            VkClearValue clears[3]{};
            clears[0].color.float32[0] = 0.88f;
            clears[0].color.float32[1] = 0.90f;
            clears[0].color.float32[2] = 0.95f;
            clears[0].color.float32[3] = 1.0f;
            clears[1].color.float32[2] = 0.5f;
            clears[1].color.float32[3] = 1.0f; // 法线初始值
            clears[2].depthStencil.depth = 1.0f;
            VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
            rbi.renderPass = toonRP_;
            rbi.framebuffer = offscreenFB_;
            rbi.renderArea.extent = extent_;
            rbi.clearValueCount = 3;
            rbi.pClearValues = clears;
            vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
            VkViewport vp{0, 0, float(extent_.width), float(extent_.height), 0, 1};
            VkRect2D sc{{0, 0}, extent_};
            vkCmdSetViewport(cmd, 0, 1, &vp);
            vkCmdSetScissor(cmd, 0, 1, &sc);

            // Pass 1: Outline
            {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, outlinePipeline_);
                vkCmdBindDescriptorSets(
                    cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, outlineLayout_, 0, 1, &toonSets_[frame_], 0, nullptr);
                OutlinePC opc{};
                opc.outlineWidth = 0.03f;
                opc.outlineWidthNDC = outlineWidthNDC_;
                opc.useNDCWidth = 1;
                opc.outlineColor = outlineColor_;
                vkCmdPushConstants(cmd,
                                   outlineLayout_,
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   0,
                                   sizeof(OutlinePC),
                                   &opc);
                VkDeviceSize z = 0;
                vkCmdBindVertexBuffers(cmd, 0, 1, &sceneVB_, &z);
                vkCmdDraw(cmd, sceneVC_, 1, 0, 0);
            }

            // Pass 2: Toon Shading
            {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, toonPipeline_);
                vkCmdBindDescriptorSets(
                    cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, toonLayout_, 0, 1, &toonSets_[frame_], 0, nullptr);
                ToonPC tpc{};
                tpc.shadingBands = shadingBands_;
                tpc.shadowThreshold = shadowThreshold_;
                tpc.shadowSmooth = shadowSmooth_;
                tpc.enableRim = enableRim_ ? 1 : 0;
                tpc.enableSpecular = enableSpecular_ ? 1 : 0;
                tpc.stylePreset = stylePreset_;
                tpc.hatchDensity = hatchDensity_;
                tpc.enableHatching = enableHatching_ ? 1 : 0;
                vkCmdPushConstants(cmd,
                                   toonLayout_,
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   0,
                                   sizeof(ToonPC),
                                   &tpc);
                VkDeviceSize z = 0;
                vkCmdBindVertexBuffers(cmd, 0, 1, &sceneVB_, &z);
                vkCmdDraw(cmd, sceneVC_, 1, 0, 0);
            }
            vkCmdEndRenderPass(cmd);
        }

        // ── Pass 3: Edge（Sobel，输出到交换链）────────────────────────────────
        {
            VkClearValue cv{};
            cv.color.float32[3] = 1.0f;
            VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
            rbi.renderPass = edgeRP_;
            rbi.framebuffer = swapFBs_[idx];
            rbi.renderArea.extent = extent_;
            rbi.clearValueCount = 1;
            rbi.pClearValues = &cv;
            vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
            VkViewport vp{0, 0, float(extent_.width), float(extent_.height), 0, 1};
            VkRect2D sc{{0, 0}, extent_};
            vkCmdSetViewport(cmd, 0, 1, &vp);
            vkCmdSetScissor(cmd, 0, 1, &sc);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, edgePipeline_);
            vkCmdBindDescriptorSets(
                cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, edgeLayout_, 0, 1, &edgeSets_[frame_], 0, nullptr);
            EdgePC epc{};
            epc.texelSize = {1.0f / extent_.width, 1.0f / extent_.height};
            epc.normalThreshold = normalThreshold_;
            epc.depthThreshold = depthThreshold_;
            epc.edgeStrength = enableSobelEdge_ ? edgeStrength_ : 0.0f;
            epc.edgeColor = edgeColor_;
            epc.onlyEdges = debugEdgeOnly_ ? 1 : 0;
            vkCmdPushConstants(cmd, edgeLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(EdgePC), &epc);
            vkCmdDraw(cmd, 3, 1, 0, 0);
            interactive_.renderUi(cmd);
            vkCmdEndRenderPass(cmd);
        }

        interactive_.endGpuSection(cmd, frame_);
        VK_CHECK(vkEndCommandBuffer(cmd));

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        VkPipelineStageFlags wait = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &imgAvail_[frame_];
        si.pWaitDstStageMask = &wait;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &renderDone_[frame_];
        VK_CHECK(vkQueueSubmit(gQueue_, 1, &si, inFlight_[frame_]));
        VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &renderDone_[frame_];
        pi.swapchainCount = 1;
        pi.pSwapchains = &swapchain_;
        pi.pImageIndices = &idx;
        r = vkQueuePresentKHR(pQueue_, &pi);
        if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR || resized_) {
            resized_ = false;
            recreate();
        }
        interactive_.endFrame(frame_);
        frame_ = (frame_ + 1) % MAX_FRAMES;
    }

    void updateUBO(uint32_t fi) {
        ToonUBO u{};
        u.model = glm::mat4(1.0f);
        u.view = interactive_.camera().viewMatrix();
        u.proj = glm::perspective(glm::radians(45.0f), float(extent_.width) / float(extent_.height), 0.1f, 100.0f);
        u.proj[1][1] *= -1;
        u.lightDir = glm::vec4(glm::normalize(lightDir_), 0);
        u.lightColor = glm::vec4(lightColor_, 1);
        u.cameraPos = glm::vec4(interactive_.camera().eyePosition(), 1);
        u.specularSize = specularSize_;
        u.rimPower = rimPower_;
        std::memcpy(uboMapped_[fi], &u, sizeof(u));
    }

    void recreate() {
        int w = 0, h = 0;
        glfwGetFramebufferSize(window_, &w, &h);
        while (w == 0 || h == 0) {
            glfwGetFramebufferSize(window_, &w, &h);
            glfwWaitEvents();
        }
        vkDeviceWaitIdle(device_);
        vkDestroyFramebuffer(device_, offscreenFB_, nullptr);
        for (auto fb : swapFBs_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        swapFBs_.clear();
        vkDestroyImageView(device_, colorView_, nullptr);
        vkDestroyImage(device_, colorImage_, nullptr);
        vkFreeMemory(device_, colorMem_, nullptr);
        vkDestroyImageView(device_, normalView_, nullptr);
        vkDestroyImage(device_, normalImage_, nullptr);
        vkFreeMemory(device_, normalMem_, nullptr);
        destroyDepthResources(device_, depth_);
        for (auto v : swapViews_)
            vkDestroyImageView(device_, v, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        createSwapchain();
        createImageViews();
        createDepthWithSampling();
        createOffscreenResources();
        createFramebuffers();
        VK_CHECK(vkResetDescriptorPool(device_, pool_, 0));
        toonSets_.clear();
        edgeSets_.clear();
        createDescriptorSets();
        interactive_.onSwapchainRecreated(edgeRP_, swapFmt_, static_cast<uint32_t>(swapImages_.size()));
    }

    void cleanup() {
        vkDestroyDescriptorPool(device_, pool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, toonDSL_, nullptr);
        vkDestroyDescriptorSetLayout(device_, edgeDSL_, nullptr);
        vkDestroyPipeline(device_, outlinePipeline_, nullptr);
        vkDestroyPipelineLayout(device_, outlineLayout_, nullptr);
        vkDestroyPipeline(device_, toonPipeline_, nullptr);
        vkDestroyPipelineLayout(device_, toonLayout_, nullptr);
        vkDestroyPipeline(device_, edgePipeline_, nullptr);
        vkDestroyPipelineLayout(device_, edgeLayout_, nullptr);
        vkDestroyRenderPass(device_, toonRP_, nullptr);
        vkDestroyRenderPass(device_, edgeRP_, nullptr);
        vkDestroySampler(device_, sampler_, nullptr);
        vkDestroyImageView(device_, colorView_, nullptr);
        vkDestroyImage(device_, colorImage_, nullptr);
        vkFreeMemory(device_, colorMem_, nullptr);
        vkDestroyImageView(device_, normalView_, nullptr);
        vkDestroyImage(device_, normalImage_, nullptr);
        vkFreeMemory(device_, normalMem_, nullptr);
        vkDestroyFramebuffer(device_, offscreenFB_, nullptr);
        for (auto fb : swapFBs_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        destroyDepthResources(device_, depth_);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroyBuffer(device_, ubos_[i], nullptr);
            vkFreeMemory(device_, uboMem_[i], nullptr);
            vkDestroySemaphore(device_, imgAvail_[i], nullptr);
            vkDestroySemaphore(device_, renderDone_[i], nullptr);
            vkDestroyFence(device_, inFlight_[i], nullptr);
        }
        vkDestroyBuffer(device_, sceneVB_, nullptr);
        vkFreeMemory(device_, sceneVBMem_, nullptr);
        for (auto v : swapViews_)
            vkDestroyImageView(device_, v, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        vkDestroyCommandPool(device_, cmdPool_, nullptr);
        interactive_.shutdown(device_);
        vkDestroyDevice(device_, nullptr);
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        vkDestroyInstance(instance_, nullptr);
        glfwDestroyWindow(window_);
        glfwTerminate();
    }
};

int main() {
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << " 第84章：卡通渲染（升级版）\n";
    std::cout << " Pass 1: Inverted Hull 轮廓线\n";
    std::cout << " Pass 2: 量化光照 + 交叉线条 + 法线输出\n";
    std::cout << " Pass 3: Sobel 边缘检测（内部折叠线条）\n";
    std::cout << "══════════════════════════════════════════════════════════\n\n";
    std::cout << "风格预设：Anime / Manga / Comic / Sketch（ImGui 一键切换）\n";
    std::cout << "控制：LMB 旋转 | RMB 平移 | 滚轮缩放 | ESC 退出\n\n";
    try {
        Ch84App app;
        app.run();
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
