/**
 * @file ch57_parallel_record.cpp
 * @brief 第57章：多线程命令录制
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【教学目标】
 *  在有大量 draw call 时，串行录制会成为 CPU 瓶颈。
 *  本章演示：
 *  ① 将场景分成 N 组，每组分配给一个工作线程
 *  ② 每个工作线程录制 Secondary Command Buffer（VkCommandBuffer）
 *  ③ 主线程将所有 Secondary CB 通过 vkCmdExecuteCommands 提交
 *  ④ ImGui 显示：单线程 vs 多线程录制耗时对比
 *
 * 【Secondary Command Buffer 要点】
 *  - 在 render pass 内部使用：VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT
 *  - 需要设置 VkCommandBufferInheritanceInfo（renderPass, subpass, framebuffer）
 *  - 每线程持有独立的 VkCommandPool（避免竞争）
 *
 * 【场景】12 个彩色立方体面（分成 N_THREADS 组并行录制）
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
#include <thread>
#include <vector>

using namespace vulkan_tutorial;

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int MAX_FRAMES = 2;
constexpr int N_THREADS = 3;  // 工作线程数
constexpr int N_OBJECTS = 12; // 总绘制对象数（每线程 N_OBJECTS/N_THREADS 个）

struct SceneVertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec3 color;
};
struct SceneUBO {
    alignas(16) glm::mat4 model, view, proj;
};

// 12 个彩色面片（4 列 × 3 行排列）
static std::vector<SceneVertex> buildScene() {
    std::vector<SceneVertex> v;
    const glm::vec3 colors[] = {
        {1.0f, 0.2f, 0.2f},
        {0.2f, 1.0f, 0.2f},
        {0.2f, 0.2f, 1.0f},
        {1.0f, 1.0f, 0.2f},
        {1.0f, 0.2f, 1.0f},
        {0.2f, 1.0f, 1.0f},
        {1.0f, 0.6f, 0.1f},
        {0.6f, 0.1f, 1.0f},
        {0.1f, 0.8f, 0.5f},
        {0.9f, 0.5f, 0.3f},
        {0.3f, 0.9f, 0.5f},
        {0.5f, 0.3f, 0.9f},
    };
    for (int i = 0; i < N_OBJECTS; ++i) {
        float x = (i % 4) * 2.0f - 3.0f;
        float y = (i / 4) * 2.0f - 1.5f;
        glm::vec3 c = colors[i];
        glm::vec3 n{0, 0, 1};
        // 正方形面片（2 个三角形）
        v.push_back({{x - 0.7f, y - 0.7f, 0}, n, c});
        v.push_back({{x + 0.7f, y - 0.7f, 0}, n, c});
        v.push_back({{x + 0.7f, y + 0.7f, 0}, n, c});
        v.push_back({{x - 0.7f, y - 0.7f, 0}, n, c});
        v.push_back({{x + 0.7f, y + 0.7f, 0}, n, c});
        v.push_back({{x - 0.7f, y + 0.7f, 0}, n, c});
    }
    return v;
}

class Ch57App {
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
    DepthResources depth_{};
    VkCommandPool cmdPool_ = VK_NULL_HANDLE;

    VkRenderPass renderPass_ = VK_NULL_HANDLE;

    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> sets_;

    VkBuffer vb_ = VK_NULL_HANDLE;
    VkDeviceMemory vbMem_ = VK_NULL_HANDLE;
    std::vector<VkBuffer> ubos_;
    std::vector<VkDeviceMemory> uboMem_;
    std::vector<void*> uboMapped_;

    // 每帧的主 Command Buffer
    std::vector<VkCommandBuffer> primaryCBs_;
    // 每线程 × 每帧的 Secondary Command Buffer + 独立 Pool
    std::array<VkCommandPool, N_THREADS> threadPools_{};
    std::array<std::array<VkCommandBuffer, MAX_FRAMES>, N_THREADS> secondaryCBs_{};

    std::vector<VkSemaphore> imgAvail_, renderDone_;
    std::vector<VkFence> inFlight_;
    uint32_t frame_ = 0;
    bool resized_ = false;

    InteractiveChapterTools interactive_;

    // 性能统计
    double singleThreadMs_ = 0.0;
    double multiThreadMs_ = 0.0;
    bool useMultiThread_ = true;
    int sceneVertCount_ = 0;

    // ─── 初始化 ─────────────────────────────────────────────────────────────

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "第57章：多线程命令录制", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
            reinterpret_cast<Ch57App*>(glfwGetWindowUserPointer(w))->resized_ = true;
        });
        interactive_.attachInput(window_);
    }

    void initVulkan() {
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
        createThreadPools();
        uploadGeometry();
        createUniformBuffers();
        createDescriptorLayout();
        createDescriptorPool();
        createDescriptorSets();
        createPipeline();
        createPrimaryCommandBuffers();
        createSecondaryCommandBuffers();
        createSyncObjects();

        InteractiveInitInfo ii{};
        ii.window = window_;
        ii.instance = instance_;
        ii.physicalDevice = physDev_;
        ii.device = device_;
        ii.graphicsQueue = gQueue_;
        ii.queueFamily = queueIdx_.graphicsFamily.value();
        ii.renderPass = renderPass_;
        ii.swapchainFormat = swapFormat_;
        ii.imageCount = static_cast<uint32_t>(swapImages_.size());
        ii.maxFramesInFlight = MAX_FRAMES;
        interactive_.initVulkan(ii);
        interactive_.camera().setDistance(10.0f);
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

    void createRenderPass() {
        std::array<VkAttachmentDescription, 2> a{};
        a[0].format = swapFormat_;
        a[0].samples = VK_SAMPLE_COUNT_1_BIT;
        a[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        a[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        a[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        a[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        a[1].format = depth_.format;
        a[1].samples = VK_SAMPLE_COUNT_1_BIT;
        a[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        a[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        a[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        a[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        a[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        a[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
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
        VK_CHECK(vkCreateRenderPass(device_, &rpci, nullptr, &renderPass_));
    }

    void createFramebuffers() {
        swapFBs_.resize(swapImages_.size());
        VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fci.renderPass = renderPass_;
        fci.attachmentCount = 2;
        fci.width = extent_.width;
        fci.height = extent_.height;
        fci.layers = 1;
        for (size_t i = 0; i < swapImages_.size(); ++i) {
            std::array<VkImageView, 2> atts = {swapViews_[i], depth_.view};
            fci.pAttachments = atts.data();
            VK_CHECK(vkCreateFramebuffer(device_, &fci, nullptr, &swapFBs_[i]));
        }
    }

    void createThreadPools() {
        VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = queueIdx_.graphicsFamily.value();
        for (int t = 0; t < N_THREADS; ++t)
            VK_CHECK(vkCreateCommandPool(device_, &ci, nullptr, &threadPools_[t]));
    }

    void uploadGeometry() {
        auto scene = buildScene();
        sceneVertCount_ = static_cast<int>(scene.size());
        VkDeviceSize sz = sizeof(SceneVertex) * scene.size();
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
        std::memcpy(d, scene.data(), sz);
        vkUnmapMemory(device_, stm);
        createBuffer(physDev_,
                     device_,
                     sz,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     vb_,
                     vbMem_);
        copyBuffer(device_, cmdPool_, gQueue_, st, vb_, sz);
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
                         sizeof(SceneUBO),
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         ubos_[i],
                         uboMem_[i]);
            vkMapMemory(device_, uboMem_[i], 0, sizeof(SceneUBO), 0, &uboMapped_[i]);
        }
    }

    void createDescriptorLayout() {
        VkDescriptorSetLayoutBinding b{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT};
        VkDescriptorSetLayoutCreateInfo dci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dci.bindingCount = 1;
        dci.pBindings = &b;
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &dci, nullptr, &dsl_));
    }

    void createDescriptorPool() {
        VkDescriptorPoolSize sz{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, MAX_FRAMES};
        VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        ci.poolSizeCount = 1;
        ci.pPoolSizes = &sz;
        ci.maxSets = MAX_FRAMES;
        VK_CHECK(vkCreateDescriptorPool(device_, &ci, nullptr, &pool_));
    }

    void createDescriptorSets() {
        std::vector<VkDescriptorSetLayout> ls(MAX_FRAMES, dsl_);
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = pool_;
        ai.descriptorSetCount = MAX_FRAMES;
        ai.pSetLayouts = ls.data();
        sets_.resize(MAX_FRAMES);
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, sets_.data()));
        for (int i = 0; i < MAX_FRAMES; ++i) {
            VkDescriptorBufferInfo bi{ubos_[i], 0, sizeof(SceneUBO)};
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                   nullptr,
                                   sets_[i],
                                   0,
                                   0,
                                   1,
                                   VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                   nullptr,
                                   &bi};
            vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
        }
    }

    void createPipeline() {
        VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        lci.setLayoutCount = 1;
        lci.pSetLayouts = &dsl_;
        VK_CHECK(vkCreatePipelineLayout(device_, &lci, nullptr, &layout_));

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
        std::array<VkDynamicState, 2> dyn = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyns{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dyns.dynamicStateCount = 2;
        dyns.pDynamicStates = dyn.data();
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
        gci.layout = layout_;
        gci.renderPass = renderPass_;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gci, nullptr, &pipeline_));
        vkDestroyShaderModule(device_, vert, nullptr);
        vkDestroyShaderModule(device_, frag, nullptr);
    }

    void createPrimaryCommandBuffers() {
        primaryCBs_.resize(MAX_FRAMES);
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = cmdPool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = MAX_FRAMES;
        VK_CHECK(vkAllocateCommandBuffers(device_, &ai, primaryCBs_.data()));
    }

    void createSecondaryCommandBuffers() {
        for (int t = 0; t < N_THREADS; ++t) {
            VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            ai.commandPool = threadPools_[t];
            ai.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
            ai.commandBufferCount = MAX_FRAMES;
            VK_CHECK(vkAllocateCommandBuffers(device_, &ai, secondaryCBs_[t].data()));
        }
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

    // ─── 命令录制 ────────────────────────────────────────────────────────────

    /// 在单线程中串行录制所有 draw call（基准对照）
    double recordSingleThread(VkCommandBuffer primaryCB, uint32_t imgIdx) {
        auto t0 = std::chrono::high_resolution_clock::now();

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(primaryCB, &bi));
        beginRenderPass(primaryCB, imgIdx);

        vkCmdBindPipeline(primaryCB, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        vkCmdBindDescriptorSets(primaryCB, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1, &sets_[frame_], 0, nullptr);
        VkViewport vp{0, 0, float(extent_.width), float(extent_.height), 0, 1};
        VkRect2D sc{{0, 0}, extent_};
        vkCmdSetViewport(primaryCB, 0, 1, &vp);
        vkCmdSetScissor(primaryCB, 0, 1, &sc);
        VkDeviceSize z = 0;
        vkCmdBindVertexBuffers(primaryCB, 0, 1, &vb_, &z);
        // 全部对象串行绘制
        for (int i = 0; i < N_OBJECTS; ++i) {
            uint32_t firstVert = i * 6;
            vkCmdDraw(primaryCB, 6, 1, firstVert, 0);
        }

        vkCmdEndRenderPass(primaryCB);
        VK_CHECK(vkEndCommandBuffer(primaryCB));

        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    /// 多线程并行录制 Secondary CB，主线程通过 vkCmdExecuteCommands 提交
    double recordMultiThread(VkCommandBuffer primaryCB, uint32_t imgIdx) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // 为每个线程准备 inheritance info（共享 render pass + framebuffer）
        VkCommandBufferInheritanceInfo inherit{VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
        inherit.renderPass = renderPass_;
        inherit.subpass = 0;
        inherit.framebuffer = swapFBs_[imgIdx];

        // 工作线程：各自录制一部分对象
        const int objPerThread = N_OBJECTS / N_THREADS;
        std::vector<std::thread> threads;
        threads.reserve(N_THREADS);

        for (int t = 0; t < N_THREADS; ++t) {
            threads.push_back(std::thread([this, t, objPerThread, &inherit]() {
                VkCommandBuffer scb = secondaryCBs_[t][frame_];
                vkResetCommandBuffer(scb, 0);

                VkCommandBufferBeginInfo sbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                sbi.flags =
                    VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT | VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                sbi.pInheritanceInfo = &inherit;
                VK_CHECK(vkBeginCommandBuffer(scb, &sbi));

                vkCmdBindPipeline(scb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
                vkCmdBindDescriptorSets(
                    scb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1, &sets_[frame_], 0, nullptr);
                VkViewport vp{0, 0, float(extent_.width), float(extent_.height), 0, 1};
                VkRect2D sc{{0, 0}, extent_};
                vkCmdSetViewport(scb, 0, 1, &vp);
                vkCmdSetScissor(scb, 0, 1, &sc);
                VkDeviceSize z = 0;
                vkCmdBindVertexBuffers(scb, 0, 1, &vb_, &z);

                int start = t * objPerThread;
                int end = (t == N_THREADS - 1) ? N_OBJECTS : start + objPerThread;
                for (int i = start; i < end; ++i) {
                    vkCmdDraw(scb, 6, 1, static_cast<uint32_t>(i * 6), 0);
                }
                VK_CHECK(vkEndCommandBuffer(scb));
            }));
        }
        for (auto& th : threads)
            th.join();

        auto t1 = std::chrono::high_resolution_clock::now();

        // 主线程录制 Primary CB，执行所有 Secondary CB
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(primaryCB, &bi));
        beginRenderPass(primaryCB, imgIdx, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);

        std::array<VkCommandBuffer, N_THREADS> scbs;
        for (int t = 0; t < N_THREADS; ++t)
            scbs[t] = secondaryCBs_[t][frame_];
        vkCmdExecuteCommands(primaryCB, N_THREADS, scbs.data());

        vkCmdEndRenderPass(primaryCB);
        VK_CHECK(vkEndCommandBuffer(primaryCB));

        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    void
    beginRenderPass(VkCommandBuffer cmd, uint32_t imgIdx, VkSubpassContents contents = VK_SUBPASS_CONTENTS_INLINE) {
        VkClearValue clears[2]{};
        clears[0].color.float32[0] = 0.08f;
        clears[0].color.float32[1] = 0.08f;
        clears[0].color.float32[2] = 0.12f;
        clears[0].color.float32[3] = 1.0f;
        clears[1].depthStencil.depth = 1.0f;
        VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rbi.renderPass = renderPass_;
        rbi.framebuffer = swapFBs_[imgIdx];
        rbi.renderArea.extent = extent_;
        rbi.clearValueCount = 2;
        rbi.pClearValues = clears;
        vkCmdBeginRenderPass(cmd, &rbi, contents);
    }

    // ─── 主循环 ──────────────────────────────────────────────────────────────

    void mainLoop() {
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            // 检测 M 键切换模式
            if (glfwGetKey(window_, GLFW_KEY_M) == GLFW_PRESS)
                useMultiThread_ = !useMultiThread_;
            drawFrame();
            updateWindowTitle();
        }
        vkDeviceWaitIdle(device_);
    }

    void updateWindowTitle() {
        char buf[256];
        snprintf(buf,
                 sizeof(buf),
                 "第57章 多线程命令录制 | %s | 单线程:%.3fms 多线程:%.3fms | [M]切换",
                 useMultiThread_ ? "多线程" : "单线程",
                 singleThreadMs_,
                 multiThreadMs_);
        glfwSetWindowTitle(window_, buf);
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
        updateUBO(frame_);

        VkCommandBuffer cmd = primaryCBs_[frame_];
        vkResetCommandBuffer(cmd, 0);

        // 每帧两种方式都计时（供对比，交替使用）
        if (useMultiThread_) {
            // 先跑一次单线程计时（不提交，仅计时）
            {
                VkCommandBuffer dummy = primaryCBs_[(frame_ + 1) % MAX_FRAMES];
                vkResetCommandBuffer(dummy, 0);
                singleThreadMs_ = recordSingleThread(dummy, imgIdx);
                vkResetCommandBuffer(dummy, 0);
            }
            multiThreadMs_ = recordMultiThread(cmd, imgIdx);
        } else {
            singleThreadMs_ = recordSingleThread(cmd, imgIdx);
            // 多线程计时用 dummy
            {
                for (int t = 0; t < N_THREADS; ++t)
                    vkResetCommandBuffer(secondaryCBs_[t][frame_], 0);
                VkCommandBuffer dummy = secondaryCBs_[0][frame_];
                vkResetCommandBuffer(dummy, 0);
                auto t0 = std::chrono::high_resolution_clock::now();
                // 仅计时录制，不提交
                std::vector<std::thread> ts;
                for (int t = 0; t < N_THREADS; ++t) {
                    ts.push_back(std::thread([&, t]() {
                        VkCommandBuffer scb = secondaryCBs_[t][frame_];
                        VkCommandBufferInheritanceInfo inh{VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
                        inh.renderPass = renderPass_;
                        inh.framebuffer = swapFBs_[imgIdx];
                        VkCommandBufferBeginInfo sbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                        sbi.flags = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT |
                                    VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                        sbi.pInheritanceInfo = &inh;
                        vkBeginCommandBuffer(scb, &sbi);
                        int obj = N_OBJECTS / N_THREADS;
                        for (int i = t * obj; i < (t + 1) * obj; ++i)
                            vkCmdDraw(scb, 6, 1, i * 6, 0);
                        vkEndCommandBuffer(scb);
                    }));
                }
                for (auto& th : ts)
                    th.join();
                auto t1 = std::chrono::high_resolution_clock::now();
                multiThreadMs_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
                for (int t = 0; t < N_THREADS; ++t)
                    vkResetCommandBuffer(secondaryCBs_[t][frame_], 0);
            }
        }

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
        frame_ = (frame_ + 1) % MAX_FRAMES;
    }

    void updateUBO(uint32_t fi) {
        SceneUBO u{};
        u.model = glm::mat4(1.0f);
        u.view = interactive_.camera().viewMatrix();
        u.proj = glm::perspective(glm::radians(45.0f), float(extent_.width) / float(extent_.height), 0.1f, 50.0f);
        u.proj[1][1] *= -1;
        std::memcpy(uboMapped_[fi], &u, sizeof(u));
    }

    void recreateSwapchain() {
        int w = 0, h = 0;
        glfwGetFramebufferSize(window_, &w, &h);
        while (w == 0 || h == 0) {
            glfwGetFramebufferSize(window_, &w, &h);
            glfwWaitEvents();
        }
        vkDeviceWaitIdle(device_);
        for (auto fb : swapFBs_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        swapFBs_.clear();
        destroyDepthResources(device_, depth_);
        for (auto v : swapViews_)
            vkDestroyImageView(device_, v, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        createSwapchain();
        createImageViews();
        depth_ = createDepthResources(physDev_, device_, extent_);
        createFramebuffers();
        interactive_.onSwapchainRecreated(renderPass_, swapFormat_, static_cast<uint32_t>(swapImages_.size()));
    }

    void cleanup() {
        vkDestroyDescriptorPool(device_, pool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, dsl_, nullptr);
        vkDestroyPipeline(device_, pipeline_, nullptr);
        vkDestroyPipelineLayout(device_, layout_, nullptr);
        vkDestroyRenderPass(device_, renderPass_, nullptr);
        for (int t = 0; t < N_THREADS; ++t)
            vkDestroyCommandPool(device_, threadPools_[t], nullptr);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroyBuffer(device_, ubos_[i], nullptr);
            vkFreeMemory(device_, uboMem_[i], nullptr);
            vkDestroySemaphore(device_, imgAvail_[i], nullptr);
            vkDestroySemaphore(device_, renderDone_[i], nullptr);
            vkDestroyFence(device_, inFlight_[i], nullptr);
        }
        vkDestroyBuffer(device_, vb_, nullptr);
        vkFreeMemory(device_, vbMem_, nullptr);
        for (auto fb : swapFBs_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        destroyDepthResources(device_, depth_);
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
    std::cout << "══════════════════════════════════════════════════\n";
    std::cout << " 第57章：多线程命令录制\n";
    std::cout << "══════════════════════════════════════════════════\n\n";
    std::cout << "工作线程数：" << N_THREADS << "，绘制对象：" << N_OBJECTS << "\n";
    std::cout << "控制：LMB 旋转 | 滚轮缩放 | ImGui 切换模式 | ESC 退出\n\n";
    try {
        Ch57App app;
        app.run();
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
