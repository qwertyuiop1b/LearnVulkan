/**
 * @file ch55_decal.cpp
 * @brief 第55章：屏幕空间贴花（Screen-Space Decal）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【原理】
 *  贴花系统无需修改场景几何，实现步骤：
 *  1. 场景正常渲染到离屏 HDR RT + 深度缓冲
 *  2. 全屏 Decal Overlay Pass（decal_overlay.frag）：
 *     - 对每个像素从深度重建世界坐标
 *     - 遍历 UBO 中最多 8 个贴花
 *     - 若世界坐标在贴花 box 内（局部坐标 [-0.5,0.5]³）
 *       → 以边缘渐变 Alpha 混合贴花颜色
 *
 * 【管线 · 2 个 Pass】
 *  Pass 1: Scene → sceneFB（HDR 颜色 + 可采样深度）
 *  Pass 2: Fullscreen Decal Overlay → 交换链
 *          （同时复制场景颜色并叠加贴花）
 *
 * 【ImGui 控制】
 *  - 点击按钮在随机位置添加贴花（地面或墙面）
 *  - 调节颜色、大小、透明度
 *  - 清除所有贴花
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
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

using namespace vulkan_tutorial;

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int MAX_FRAMES = 2;
constexpr int MAX_DECALS = 8;

struct SceneVertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec3 color;
};
struct SceneUBO {
    alignas(16) glm::mat4 model, view, proj;
};

// 单个贴花的 GPU 表示（与 decal_overlay.frag 对齐）
struct DecalDataGPU {
    glm::mat4 invWorld;
    glm::vec4 color;
};

// 贴花 UBO（binding=2）
struct DecalUBO {
    alignas(16) glm::mat4 invProjView;
    int decalCount;
    float pad[3];
    DecalDataGPU decals[MAX_DECALS];
};

// CPU 端贴花描述
struct Decal {
    glm::vec3 position{0};
    float yaw = 0.0f;
    glm::vec3 scale{1, 0.25f, 1};
    glm::vec4 color{1, 0.2f, 0.1f, 0.9f};
};

// 场景几何（地面 + 墙面）
static const std::vector<SceneVertex> SCENE = {
    {{-5, -0.5f, -5}, {0, 1, 0}, {0.45f, 0.4f, 0.35f}},
    {{5, -0.5f, -5}, {0, 1, 0}, {0.45f, 0.4f, 0.35f}},
    {{5, -0.5f, 5}, {0, 1, 0}, {0.45f, 0.4f, 0.35f}},
    {{-5, -0.5f, -5}, {0, 1, 0}, {0.45f, 0.4f, 0.35f}},
    {{5, -0.5f, 5}, {0, 1, 0}, {0.45f, 0.4f, 0.35f}},
    {{-5, -0.5f, 5}, {0, 1, 0}, {0.45f, 0.4f, 0.35f}},
    {{-2, -0.5f, -2}, {0, 0, 1}, {0.6f, 0.55f, 0.5f}},
    {{2, -0.5f, -2}, {0, 0, 1}, {0.6f, 0.55f, 0.5f}},
    {{2, 2.5f, -2}, {0, 0, 1}, {0.6f, 0.55f, 0.5f}},
    {{-2, -0.5f, -2}, {0, 0, 1}, {0.6f, 0.55f, 0.5f}},
    {{2, 2.5f, -2}, {0, 0, 1}, {0.6f, 0.55f, 0.5f}},
    {{-2, 2.5f, -2}, {0, 0, 1}, {0.6f, 0.55f, 0.5f}},
};

class Ch55App {
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
    VkPhysicalDevice physDev_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue gQueue_ = VK_NULL_HANDLE;
    VkQueue pQueue_ = VK_NULL_HANDLE;
    QueueFamilyIndices queueIdx_{};
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    std::vector<VkImage> swapImages_;
    std::vector<VkImageView> swapViews_;
    std::vector<VkFramebuffer> swapFBs_;
    VkFormat swapFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D extent_{};
    VkCommandPool cmdPool_ = VK_NULL_HANDLE;

    // Offscreen resources
    VkImage hdrImage_ = VK_NULL_HANDLE;
    VkDeviceMemory hdrMem_ = VK_NULL_HANDLE;
    VkImageView hdrView_ = VK_NULL_HANDLE;
    VkImage depthImage_ = VK_NULL_HANDLE;
    VkDeviceMemory depthMem_ = VK_NULL_HANDLE;
    VkImageView depthView_ = VK_NULL_HANDLE;
    VkFramebuffer sceneFB_ = VK_NULL_HANDLE;

    VkRenderPass sceneRP_ = VK_NULL_HANDLE;
    VkRenderPass overlayRP_ = VK_NULL_HANDLE;

    // Pipelines
    VkPipeline scenePipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout sceneLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout sceneDSL_ = VK_NULL_HANDLE;
    VkPipeline overlayPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout overlayLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout overlayDSL_ = VK_NULL_HANDLE;

    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> sceneSets_;
    std::vector<VkDescriptorSet> overlaySets_;
    VkSampler sampler_ = VK_NULL_HANDLE;

    VkBuffer sceneVB_ = VK_NULL_HANDLE;
    VkDeviceMemory sceneVBMem_ = VK_NULL_HANDLE;
    std::vector<VkBuffer> sceneUBOs_, decalUBOs_;
    std::vector<VkDeviceMemory> sceneUBOMem_, decalUBOMem_;
    std::vector<void*> sceneMapped_, decalMapped_;

    std::vector<VkCommandBuffer> cmds_;
    std::vector<VkSemaphore> imgAvail_, renderDone_;
    std::vector<VkFence> inFlight_;
    uint32_t frame_ = 0;
    bool resized_ = false;

    InteractiveChapterTools interactive_;

    std::vector<Decal> decals_;
    glm::vec4 newColor_{1.0f, 0.2f, 0.1f, 0.9f};
    float newSize_ = 1.5f;
    float newHeight_ = 0.2f;

    // ─── 初始化 ─────────────────────────────────────────────────────────────

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "第55章：屏幕空间贴花", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
            reinterpret_cast<Ch55App*>(glfwGetWindowUserPointer(w))->resized_ = true;
        });
        interactive_.attachInput(window_);
        // 默认贴花
        decals_.push_back({{0, -0.49f, 0}, 0.0f, {1.5f, 0.02f, 1.5f}, {1, 0.1f, 0.1f, 0.9f}});
        decals_.push_back({{-1.5f, 0.8f, -2}, 1.57f, {1.0f, 0.02f, 1.0f}, {0.1f, 0.4f, 1, 0.85f}});
        decals_.push_back({{1.5f, 0.3f, -2}, 1.57f, {0.8f, 0.02f, 0.8f}, {0.2f, 1, 0.2f, 0.85f}});
    }

    void initVulkan() {
        createInstance(instance_);
        VK_CHECK(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_));
        pickPhysicalDevice(instance_, surface_, physDev_);
        createLogicalDevice(physDev_, surface_, device_, gQueue_, pQueue_, queueIdx_);
        createSwapchain();
        createImageViews();
        createCommandPool();
        createSampler();
        createOffscreen();
        createRenderPasses();
        createFramebuffers();
        uploadGeometry();
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
        ii.queueFamily = queueIdx_.graphicsFamily.value();
        ii.renderPass = overlayRP_;
        ii.swapchainFormat = swapFormat_;
        ii.imageCount = static_cast<uint32_t>(swapImages_.size());
        ii.maxFramesInFlight = MAX_FRAMES;
        interactive_.initVulkan(ii);
        interactive_.camera().setDistance(7.0f);
        interactive_.camera().setAngles(30.0f, 30.0f);
    }

    void createSwapchain() {
        auto d = querySwapChainSupport(physDev_, surface_);
        auto f = chooseSwapSurfaceFormat(d.formats);
        auto m = chooseSwapPresentMode(d.presentModes);
        extent_ = chooseSwapExtent(d.capabilities, window_);
        swapFormat_ = f.format;
        uint32_t cnt = d.capabilities.minImageCount + 1;
        if (d.capabilities.maxImageCount > 0)
            cnt = std::min(cnt, d.capabilities.maxImageCount);
        VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        ci.surface = surface_;
        ci.minImageCount = cnt;
        ci.imageFormat = f.format;
        ci.imageColorSpace = f.colorSpace;
        ci.imageExtent = extent_;
        ci.imageArrayLayers = 1;
        ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        uint32_t qf[2] = {queueIdx_.graphicsFamily.value(), queueIdx_.presentFamily.value()};
        if (qf[0] != qf[1]) {
            ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            ci.queueFamilyIndexCount = 2;
            ci.pQueueFamilyIndices = qf;
        } else
            ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ci.preTransform = d.capabilities.currentTransform;
        ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        ci.presentMode = m;
        ci.clipped = VK_TRUE;
        VK_CHECK(vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_));
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
            vi.format = swapFormat_;
            vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VK_CHECK(vkCreateImageView(device_, &vi, nullptr, &swapViews_[i]));
        }
    }

    void createCommandPool() {
        VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = queueIdx_.graphicsFamily.value();
        VK_CHECK(vkCreateCommandPool(device_, &ci, nullptr, &cmdPool_));
    }

    void createSampler() {
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter = VK_FILTER_LINEAR;
        si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = 1.0f;
        VK_CHECK(vkCreateSampler(device_, &si, nullptr, &sampler_));
    }

    void createOffscreen() {
        VkFormat depthFmt = findDepthFormat(physDev_);
        auto mkImg = [&](uint32_t w,
                         uint32_t h,
                         VkFormat fmt,
                         VkImageUsageFlags usage,
                         VkImage& img,
                         VkDeviceMemory& mem,
                         VkImageView& view,
                         VkImageAspectFlags asp) {
            createImage(physDev_,
                        device_,
                        w,
                        h,
                        fmt,
                        VK_IMAGE_TILING_OPTIMAL,
                        usage,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        img,
                        mem);
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = img;
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format = fmt;
            vi.subresourceRange = {asp, 0, 1, 0, 1};
            VK_CHECK(vkCreateImageView(device_, &vi, nullptr, &view));
        };
        mkImg(extent_.width,
              extent_.height,
              swapFormat_,
              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
              hdrImage_,
              hdrMem_,
              hdrView_,
              VK_IMAGE_ASPECT_COLOR_BIT);
        mkImg(extent_.width,
              extent_.height,
              depthFmt,
              VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
              depthImage_,
              depthMem_,
              depthView_,
              VK_IMAGE_ASPECT_DEPTH_BIT);
        // Initial layouts
        VkCommandBuffer cmd = beginSingleTimeCommands(device_, cmdPool_);
        auto trans =
            [&](VkImage img, VkImageAspectFlags asp, VkImageLayout newL, VkPipelineStageFlags dst, VkAccessFlags da) {
                VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                b.newLayout = newL;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image = img;
                b.subresourceRange = {asp, 0, 1, 0, 1};
                b.dstAccessMask = da;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, dst, 0, 0, nullptr, 0, nullptr, 1, &b);
            };
        trans(hdrImage_,
              VK_IMAGE_ASPECT_COLOR_BIT,
              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
              VK_ACCESS_SHADER_READ_BIT);
        trans(depthImage_,
              VK_IMAGE_ASPECT_DEPTH_BIT,
              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
              VK_ACCESS_SHADER_READ_BIT);
        endSingleTimeCommands(device_, cmdPool_, gQueue_, cmd);
    }

    void createRenderPasses() {
        // Scene RP
        {
            std::array<VkAttachmentDescription, 2> a{};
            a[0].format = swapFormat_;
            a[0].samples = VK_SAMPLE_COUNT_1_BIT;
            a[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            a[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            a[0].initialLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            a[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            a[1].format = findDepthFormat(physDev_);
            a[1].samples = VK_SAMPLE_COUNT_1_BIT;
            a[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            a[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            a[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            a[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            a[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            a[1].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            VkAttachmentReference cr{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
            VkAttachmentReference dr{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
            VkSubpassDescription sub{};
            sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            sub.colorAttachmentCount = 1;
            sub.pColorAttachments = &cr;
            sub.pDepthStencilAttachment = &dr;
            VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
            rpci.attachmentCount = 2;
            rpci.pAttachments = a.data();
            rpci.subpassCount = 1;
            rpci.pSubpasses = &sub;
            VK_CHECK(vkCreateRenderPass(device_, &rpci, nullptr, &sceneRP_));
        }
        // Overlay RP（→ swap）
        {
            VkAttachmentDescription a{};
            a.format = swapFormat_;
            a.samples = VK_SAMPLE_COUNT_1_BIT;
            a.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            a.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            a.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            VkAttachmentReference cr{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
            VkSubpassDescription sub{};
            sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            sub.colorAttachmentCount = 1;
            sub.pColorAttachments = &cr;
            VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
            rpci.attachmentCount = 1;
            rpci.pAttachments = &a;
            rpci.subpassCount = 1;
            rpci.pSubpasses = &sub;
            VK_CHECK(vkCreateRenderPass(device_, &rpci, nullptr, &overlayRP_));
        }
    }

    void createFramebuffers() {
        std::array<VkImageView, 2> sa = {hdrView_, depthView_};
        VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fci.renderPass = sceneRP_;
        fci.attachmentCount = 2;
        fci.pAttachments = sa.data();
        fci.width = extent_.width;
        fci.height = extent_.height;
        fci.layers = 1;
        VK_CHECK(vkCreateFramebuffer(device_, &fci, nullptr, &sceneFB_));
        swapFBs_.resize(swapImages_.size());
        for (size_t i = 0; i < swapImages_.size(); ++i) {
            fci.renderPass = overlayRP_;
            fci.attachmentCount = 1;
            fci.pAttachments = &swapViews_[i];
            VK_CHECK(vkCreateFramebuffer(device_, &fci, nullptr, &swapFBs_[i]));
        }
    }

    void uploadGeometry() {
        VkDeviceSize sz = sizeof(SceneVertex) * SCENE.size();
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
        std::memcpy(d, SCENE.data(), sz);
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
        auto mk = [&](size_t sz, std::vector<VkBuffer>& b, std::vector<VkDeviceMemory>& m, std::vector<void*>& p) {
            b.resize(MAX_FRAMES);
            m.resize(MAX_FRAMES);
            p.resize(MAX_FRAMES);
            for (int i = 0; i < MAX_FRAMES; ++i) {
                createBuffer(physDev_,
                             device_,
                             sz,
                             VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                             b[i],
                             m[i]);
                vkMapMemory(device_, m[i], 0, sz, 0, &p[i]);
            }
        };
        mk(sizeof(SceneUBO), sceneUBOs_, sceneUBOMem_, sceneMapped_);
        mk(sizeof(DecalUBO), decalUBOs_, decalUBOMem_, decalMapped_);
    }

    void createDescriptorLayouts() {
        // Scene DSL：1 UBO
        VkDescriptorSetLayoutBinding sb{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT};
        VkDescriptorSetLayoutCreateInfo dci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dci.bindingCount = 1;
        dci.pBindings = &sb;
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &dci, nullptr, &sceneDSL_));

        // Overlay DSL：scene sampler + depth sampler + decal UBO
        std::array<VkDescriptorSetLayoutBinding, 3> ob = {{
            {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        }};
        dci.bindingCount = 3;
        dci.pBindings = ob.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &dci, nullptr, &overlayDSL_));
    }

    void createDescriptorPool() {
        std::array<VkDescriptorPoolSize, 2> sz = {{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, MAX_FRAMES * 2},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_FRAMES * 2},
        }};
        VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        ci.poolSizeCount = 2;
        ci.pPoolSizes = sz.data();
        ci.maxSets = MAX_FRAMES * 2;
        VK_CHECK(vkCreateDescriptorPool(device_, &ci, nullptr, &pool_));
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
        alloc(sceneDSL_, sceneSets_);
        alloc(overlayDSL_, overlaySets_);

        for (int i = 0; i < MAX_FRAMES; ++i) {
            VkDescriptorBufferInfo sbi{sceneUBOs_[i], 0, sizeof(SceneUBO)};
            VkWriteDescriptorSet sw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                    nullptr,
                                    sceneSets_[i],
                                    0,
                                    0,
                                    1,
                                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                    nullptr,
                                    &sbi};
            vkUpdateDescriptorSets(device_, 1, &sw, 0, nullptr);

            VkDescriptorImageInfo hdrI{sampler_, hdrView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkDescriptorImageInfo depI{sampler_, depthView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkDescriptorBufferInfo dbi{decalUBOs_[i], 0, sizeof(DecalUBO)};
            std::array<VkWriteDescriptorSet, 3> ow = {{
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                 nullptr,
                 overlaySets_[i],
                 0,
                 0,
                 1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                 &hdrI,
                 nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                 nullptr,
                 overlaySets_[i],
                 1,
                 0,
                 1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                 &depI,
                 nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                 nullptr,
                 overlaySets_[i],
                 2,
                 0,
                 1,
                 VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                 nullptr,
                 &dbi},
            }};
            vkUpdateDescriptorSets(device_, 3, ow.data(), 0, nullptr);
        }
    }

    void createPipelines() {
        // 公共状态辅助
        auto makeDyn = [&]() -> VkPipelineDynamicStateCreateInfo {
            static std::array<VkDynamicState, 2> dyn = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo d{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            d.dynamicStateCount = 2;
            d.pDynamicStates = dyn.data();
            return d;
        };

        // ── Scene pipeline ──
        {
            VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            lci.setLayoutCount = 1;
            lci.pSetLayouts = &sceneDSL_;
            VK_CHECK(vkCreatePipelineLayout(device_, &lci, nullptr, &sceneLayout_));
            auto vert = createShaderModuleFromFile(device_, "rg_scene.vert.spv");
            auto frag = createShaderModuleFromFile(device_, "rg_scene.frag.spv");
            VkVertexInputBindingDescription bind{0, sizeof(SceneVertex), VK_VERTEX_INPUT_RATE_VERTEX};
            std::array<VkVertexInputAttributeDescription, 3> attrs = {{{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
                                                                       {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 12},
                                                                       {2, 0, VK_FORMAT_R32G32B32_SFLOAT, 24}}};
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
            VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rs.polygonMode = VK_POLYGON_MODE_FILL;
            rs.cullMode = VK_CULL_MODE_BACK_BIT;
            rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rs.lineWidth = 1.0f;
            VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            ds.depthTestEnable = VK_TRUE;
            ds.depthWriteEnable = VK_TRUE;
            ds.depthCompareOp = VK_COMPARE_OP_LESS;
            VkPipelineColorBlendAttachmentState blend{};
            blend.colorWriteMask = 0xF;
            VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            cbs.attachmentCount = 1;
            cbs.pAttachments = &blend;
            auto dyns = makeDyn();
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
            gci.layout = sceneLayout_;
            gci.renderPass = sceneRP_;
            VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gci, nullptr, &scenePipeline_));
            vkDestroyShaderModule(device_, vert, nullptr);
            vkDestroyShaderModule(device_, frag, nullptr);
        }

        // ── Overlay pipeline（rg_fullscreen.vert + decal_overlay.frag）──
        {
            VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            lci.setLayoutCount = 1;
            lci.pSetLayouts = &overlayDSL_;
            VK_CHECK(vkCreatePipelineLayout(device_, &lci, nullptr, &overlayLayout_));
            auto vert = createShaderModuleFromFile(device_, "rg_fullscreen.vert.spv");
            auto frag = createShaderModuleFromFile(device_, "decal_overlay.frag.spv");
            VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            vps.viewportCount = 1;
            vps.scissorCount = 1;
            VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rs.polygonMode = VK_POLYGON_MODE_FILL;
            rs.lineWidth = 1.0f;
            VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            VkPipelineDepthStencilStateCreateInfo nods{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            VkPipelineColorBlendAttachmentState blend{};
            blend.colorWriteMask = 0xF;
            VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            cbs.attachmentCount = 1;
            cbs.pAttachments = &blend;
            auto dyns = makeDyn();
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
            gci.pDepthStencilState = &nods;
            gci.pColorBlendState = &cbs;
            gci.pDynamicState = &dyns;
            gci.layout = overlayLayout_;
            gci.renderPass = overlayRP_;
            VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gci, nullptr, &overlayPipeline_));
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
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            interactive_.beginFrame(0.016f);
            buildUi();
            drawFrame();
            interactive_.updateWindowTitle();
        }
        vkDeviceWaitIdle(device_);
    }

    glm::mat4 decalWorldMatrix(const Decal& d) const {
        glm::mat4 m = glm::translate(glm::mat4(1.0f), d.position);
        m = glm::rotate(m, d.yaw, glm::vec3(0, 1, 0));
        m = glm::scale(m, d.scale);
        return m;
    }

    void buildUi() {
        interactive_.buildDebugPanel("第55章：屏幕空间贴花");
        ImGui::Separator();
        ImGui::Text("已放置贴花：%d / %d", static_cast<int>(decals_.size()), MAX_DECALS);
        ImGui::ColorEdit4("新贴花颜色", &newColor_.x);
        ImGui::SliderFloat("半径", &newSize_, 0.3f, 3.0f);
        ImGui::SliderFloat("厚度", &newHeight_, 0.05f, 1.0f);
        if (ImGui::Button("放置（地面）") && static_cast<int>(decals_.size()) < MAX_DECALS) {
            Decal d;
            d.position = {float(rand() % 60 - 30) * 0.1f, -0.49f, float(rand() % 60 - 30) * 0.1f};
            d.yaw = float(rand() % 314) * 0.01f;
            d.scale = {newSize_, newHeight_, newSize_};
            d.color = newColor_;
            decals_.push_back(d);
        }
        ImGui::SameLine();
        if (ImGui::Button("放置（墙面）") && static_cast<int>(decals_.size()) < MAX_DECALS) {
            Decal d;
            d.position = {float(rand() % 30 - 15) * 0.1f, float(rand() % 20) * 0.1f, -2.0f};
            d.yaw = 1.57f;
            d.scale = {newSize_ * 0.8f, newSize_ * 0.8f, newHeight_};
            d.color = newColor_;
            decals_.push_back(d);
        }
        ImGui::SameLine();
        if (ImGui::Button("清除"))
            decals_.clear();
        for (int i = 0; i < static_cast<int>(decals_.size()); ++i) {
            ImGui::PushID(i);
            ImGui::Text(
                "  [%d] (%.1f,%.1f,%.1f)", i, decals_[i].position.x, decals_[i].position.y, decals_[i].position.z);
            ImGui::SameLine();
            ImGui::ColorEdit4("##c", &decals_[i].color.x);
            ImGui::PopID();
        }
    }

    void drawFrame() {
        vkWaitForFences(device_, 1, &inFlight_[frame_], VK_TRUE, UINT64_MAX);
        uint32_t imgIdx = 0;
        VkResult res =
            vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, imgAvail_[frame_], VK_NULL_HANDLE, &imgIdx);
        if (res == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapchain();
            return;
        }
        vkResetFences(device_, 1, &inFlight_[frame_]);
        updateUBOs(frame_);

        VkCommandBuffer cmd = cmds_[frame_];
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
        interactive_.beginGpuSection(cmd, frame_);

        // ── Scene Pass ──
        VkClearValue sc[2]{};
        sc[0].color.float32[0] = 0.2f;
        sc[0].color.float32[1] = 0.25f;
        sc[0].color.float32[2] = 0.3f;
        sc[0].color.float32[3] = 1.0f;
        sc[1].depthStencil.depth = 1.0f;
        VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rbi.renderPass = sceneRP_;
        rbi.framebuffer = sceneFB_;
        rbi.renderArea.extent = extent_;
        rbi.clearValueCount = 2;
        rbi.pClearValues = sc;
        vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
        VkViewport vp{0, 0, float(extent_.width), float(extent_.height), 0, 1};
        VkRect2D scc{{0, 0}, extent_};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &scc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scenePipeline_);
        vkCmdBindDescriptorSets(
            cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sceneLayout_, 0, 1, &sceneSets_[frame_], 0, nullptr);
        VkDeviceSize z = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &sceneVB_, &z);
        vkCmdDraw(cmd, static_cast<uint32_t>(SCENE.size()), 1, 0, 0);
        vkCmdEndRenderPass(cmd);

        // ── Overlay Pass（scene copy + decals）──
        VkClearValue oc{};
        oc.color.float32[3] = 1.0f;
        rbi.renderPass = overlayRP_;
        rbi.framebuffer = swapFBs_[imgIdx];
        rbi.renderArea.extent = extent_;
        rbi.clearValueCount = 1;
        rbi.pClearValues = &oc;
        vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &scc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, overlayPipeline_);
        vkCmdBindDescriptorSets(
            cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, overlayLayout_, 0, 1, &overlaySets_[frame_], 0, nullptr);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        interactive_.renderUi(cmd);
        vkCmdEndRenderPass(cmd);

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
        pi.pImageIndices = &imgIdx;
        res = vkQueuePresentKHR(pQueue_, &pi);
        if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR || resized_) {
            resized_ = false;
            recreateSwapchain();
        }
        interactive_.endFrame(frame_);
        frame_ = (frame_ + 1) % MAX_FRAMES;
    }

    void updateUBOs(uint32_t fi) {
        SceneUBO s{};
        s.model = glm::mat4(1.0f);
        s.view = interactive_.camera().viewMatrix();
        s.proj = glm::perspective(glm::radians(45.0f), float(extent_.width) / float(extent_.height), 0.1f, 100.0f);
        s.proj[1][1] *= -1;
        std::memcpy(sceneMapped_[fi], &s, sizeof(s));

        DecalUBO du{};
        du.invProjView = glm::inverse(s.proj * s.view);
        du.decalCount = static_cast<int>(std::min((int)decals_.size(), MAX_DECALS));
        for (int i = 0; i < du.decalCount; ++i) {
            du.decals[i].invWorld = glm::inverse(decalWorldMatrix(decals_[i]));
            du.decals[i].color = decals_[i].color;
        }
        std::memcpy(decalMapped_[fi], &du, sizeof(du));
    }

    void recreateSwapchain() {
        int w = 0, h = 0;
        glfwGetFramebufferSize(window_, &w, &h);
        while (w == 0 || h == 0) {
            glfwGetFramebufferSize(window_, &w, &h);
            glfwWaitEvents();
        }
        vkDeviceWaitIdle(device_);
        vkDestroyFramebuffer(device_, sceneFB_, nullptr);
        for (auto fb : swapFBs_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        swapFBs_.clear();
        vkDestroyImageView(device_, hdrView_, nullptr);
        vkDestroyImage(device_, hdrImage_, nullptr);
        vkFreeMemory(device_, hdrMem_, nullptr);
        vkDestroyImageView(device_, depthView_, nullptr);
        vkDestroyImage(device_, depthImage_, nullptr);
        vkFreeMemory(device_, depthMem_, nullptr);
        for (auto v : swapViews_)
            vkDestroyImageView(device_, v, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        createSwapchain();
        createImageViews();
        createOffscreen();
        createFramebuffers();
        VK_CHECK(vkResetDescriptorPool(device_, pool_, 0));
        sceneSets_.clear();
        overlaySets_.clear();
        createDescriptorSets();
        interactive_.onSwapchainRecreated(overlayRP_, swapFormat_, static_cast<uint32_t>(swapImages_.size()));
    }

    void cleanup() {
        vkDestroyDescriptorPool(device_, pool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, sceneDSL_, nullptr);
        vkDestroyDescriptorSetLayout(device_, overlayDSL_, nullptr);
        vkDestroyPipeline(device_, scenePipeline_, nullptr);
        vkDestroyPipelineLayout(device_, sceneLayout_, nullptr);
        vkDestroyPipeline(device_, overlayPipeline_, nullptr);
        vkDestroyPipelineLayout(device_, overlayLayout_, nullptr);
        vkDestroyRenderPass(device_, sceneRP_, nullptr);
        vkDestroyRenderPass(device_, overlayRP_, nullptr);
        vkDestroySampler(device_, sampler_, nullptr);
        vkDestroyImageView(device_, hdrView_, nullptr);
        vkDestroyImage(device_, hdrImage_, nullptr);
        vkFreeMemory(device_, hdrMem_, nullptr);
        vkDestroyImageView(device_, depthView_, nullptr);
        vkDestroyImage(device_, depthImage_, nullptr);
        vkFreeMemory(device_, depthMem_, nullptr);
        vkDestroyFramebuffer(device_, sceneFB_, nullptr);
        for (auto fb : swapFBs_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroyBuffer(device_, sceneUBOs_[i], nullptr);
            vkFreeMemory(device_, sceneUBOMem_[i], nullptr);
            vkDestroyBuffer(device_, decalUBOs_[i], nullptr);
            vkFreeMemory(device_, decalUBOMem_[i], nullptr);
            vkDestroySemaphore(device_, imgAvail_[i], nullptr);
            vkDestroySemaphore(device_, renderDone_[i], nullptr);
            vkDestroyFence(device_, inFlight_[i], nullptr);
        }
        vkDestroyBuffer(device_, sceneVB_, nullptr);
        vkFreeMemory(device_, sceneVBMem_, nullptr);
        vkDestroyCommandPool(device_, cmdPool_, nullptr);
        for (auto v : swapViews_)
            vkDestroyImageView(device_, v, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        interactive_.shutdown(device_);
        vkDestroyDevice(device_, nullptr);
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        vkDestroyInstance(instance_, nullptr);
        glfwDestroyWindow(window_);
        glfwTerminate();
    }
};

int main() {
    std::cout << "══════════════════════════════════════════════════\n";
    std::cout << " 第55章：屏幕空间贴花（Screen-Space Decal）\n";
    std::cout << "══════════════════════════════════════════════════\n\n";
    std::cout << "原理：深度重建世界坐标 → box 检测 → Alpha 混合叠加\n";
    std::cout << "控制：LMB 旋转 | RMB 平移 | 滚轮缩放 | ImGui 添加贴花 | ESC 退出\n\n";
    try {
        Ch55App app;
        app.run();
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
