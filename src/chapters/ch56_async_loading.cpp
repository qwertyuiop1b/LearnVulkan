/**
 * @file ch56_async_loading.cpp
 * @brief 第56章：异步纹理加载
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【教学目标】
 *  在游戏/引擎中，纹理加载不能阻塞主线程（否则会卡帧）。
 *  本章演示：
 *  ① 主线程正常渲染（4 个纹理 quad，初始显示棋盘格占位）
 *  ② 点击"开始加载"后，工作线程用程序化方式生成纹理数据（模拟 I/O）
 *  ③ 主线程每帧检查完成队列，通过 Staging Buffer + Fence 安全上传 GPU
 *  ④ 上传完成后纹理切换为真实内容，ImGui 显示耗时
 *
 * 【同步机制】
 *  - std::thread  : 工作线程生成像素数据
 *  - std::mutex   : 保护完成队列
 *  - VkFence      : 等待 GPU upload 命令完成
 *  - std::atomic  : 进度报告
 *
 * 【着色器】async_tex.vert / async_tex.frag（已存在）
 * ═══════════════════════════════════════════════════════════════════════════
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <vulkan_tutorial/utils.hpp>
#include <vulkan_tutorial/vk_helpers.hpp>
#include <vulkan_tutorial/interactive_chapter.hpp>

#include <imgui.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

using namespace vulkan_tutorial;

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr int MAX_FRAMES = 2;
constexpr int N_TEXTURES = 4;
constexpr uint32_t TEX_SIZE = 256;

// 每张纹理的加载状态
enum class TexState { Pending, Loading, Uploading, Ready };

struct AsyncTexture {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE; // upload 完成信号
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMem = VK_NULL_HANDLE;
    TexState state = TexState::Pending;
    float progress = 0.0f;          // 0→1，加载进度
    float loadMs = 0.0f;            // CPU 加载耗时（毫秒）
    std::vector<uint8_t> pixelData; // CPU 端像素数据
};

// 完成队列：工作线程填充，主线程消费
struct UploadRequest {
    int index;
    std::vector<uint8_t> pixels;
    float loadMs;
};

// Push constant（async_tex.frag 中的 loadProgress 字段）
struct QuadPC {
    glm::vec2 pos;
    glm::vec2 size;
    float loadProgress;
    float pad[3];
};

class Ch56App {
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
    VkCommandPool uploadPool_ = VK_NULL_HANDLE; // 上传专用 command pool

    VkRenderPass renderPass_ = VK_NULL_HANDLE;

    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkSampler sampler_ = VK_NULL_HANDLE;

    // 纹理数组（N_TEXTURES 个）
    std::array<AsyncTexture, N_TEXTURES> textures_;
    std::array<VkDescriptorSet, N_TEXTURES> texSets_{};

    // 占位纹理（棋盘格，始终可用）
    VkImage placeholderImage_ = VK_NULL_HANDLE;
    VkDeviceMemory placeholderMem_ = VK_NULL_HANDLE;
    VkImageView placeholderView_ = VK_NULL_HANDLE;

    std::vector<VkCommandBuffer> cmds_;
    std::vector<VkSemaphore> imgAvail_, renderDone_;
    std::vector<VkFence> inFlight_;
    uint32_t frame_ = 0;
    bool resized_ = false;

    InteractiveChapterTools interactive_;

    // 异步加载
    std::mutex uploadMutex_;
    std::vector<UploadRequest> uploadQueue_;
    std::vector<std::thread> workers_;
    bool loadingStarted_ = false;

    // 四种不同颜色方案（供程序化生成）
    struct ColorScheme {
        uint8_t r, g, b;
        const char* name;
    };
    static constexpr ColorScheme SCHEMES[N_TEXTURES] = {
        {220, 80, 60, "红色棋盘"},
        {60, 150, 220, "蓝色渐变"},
        {80, 200, 100, "绿色噪声"},
        {200, 160, 40, "金色图案"},
    };

    // ─── 初始化 ─────────────────────────────────────────────────────────────

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "第56章：异步纹理加载", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
            reinterpret_cast<Ch56App*>(glfwGetWindowUserPointer(w))->resized_ = true;
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
        createCommandPool();
        createRenderPass();
        createFramebuffers();
        createSampler();
        createPlaceholderTexture();
        createDescriptorLayout();
        createDescriptorPool();
        allocateDescriptorSets();
        createPipeline();
        createCommandBuffers();
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
        VK_CHECK(vkCreateCommandPool(device_, &ci, nullptr, &uploadPool_));
    }

    void createRenderPass() {
        VkAttachmentDescription att{};
        att.format = swapFormat_;
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
        VK_CHECK(vkCreateRenderPass(device_, &rpci, nullptr, &renderPass_));
    }

    void createFramebuffers() {
        swapFBs_.resize(swapImages_.size());
        VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fci.renderPass = renderPass_;
        fci.attachmentCount = 1;
        fci.width = extent_.width;
        fci.height = extent_.height;
        fci.layers = 1;
        for (size_t i = 0; i < swapImages_.size(); ++i) {
            fci.pAttachments = &swapViews_[i];
            VK_CHECK(vkCreateFramebuffer(device_, &fci, nullptr, &swapFBs_[i]));
        }
    }

    void createSampler() {
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter = VK_FILTER_LINEAR;
        si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = 1.0f;
        VK_CHECK(vkCreateSampler(device_, &si, nullptr, &sampler_));
    }

    std::vector<uint8_t> generateCheckerboard(uint8_t r, uint8_t g, uint8_t b) {
        std::vector<uint8_t> px(TEX_SIZE * TEX_SIZE * 4);
        for (uint32_t y = 0; y < TEX_SIZE; ++y) {
            for (uint32_t x = 0; x < TEX_SIZE; ++x) {
                bool c = ((x / 16 + y / 16) % 2) == 0;
                uint32_t idx = (y * TEX_SIZE + x) * 4;
                px[idx + 0] = c ? r : r / 2;
                px[idx + 1] = c ? g : g / 2;
                px[idx + 2] = c ? b : b / 2;
                px[idx + 3] = 255;
            }
        }
        return px;
    }

    void createPlaceholderTexture() {
        auto pixels = generateCheckerboard(80, 80, 80);
        uploadTextureImmediate(pixels, placeholderImage_, placeholderMem_, placeholderView_);
    }

    void
    uploadTextureImmediate(const std::vector<uint8_t>& pixels, VkImage& img, VkDeviceMemory& mem, VkImageView& view) {
        VkDeviceSize sz = TEX_SIZE * TEX_SIZE * 4;
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
        std::memcpy(d, pixels.data(), sz);
        vkUnmapMemory(device_, stm);

        createImage(physDev_,
                    device_,
                    TEX_SIZE,
                    TEX_SIZE,
                    VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    img,
                    mem);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_, uploadPool_);
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = img;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(
            cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        VkBufferImageCopy bic{};
        bic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        bic.imageExtent = {TEX_SIZE, TEX_SIZE, 1};
        vkCmdCopyBufferToImage(cmd, st, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bic);
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &b);
        endSingleTimeCommands(device_, uploadPool_, gQueue_, cmd);
        vkDestroyBuffer(device_, st, nullptr);
        vkFreeMemory(device_, stm, nullptr);

        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = img;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = VK_FORMAT_R8G8B8A8_UNORM;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(device_, &vi, nullptr, &view));
    }

    void createDescriptorLayout() {
        VkDescriptorSetLayoutBinding sb{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT};
        VkDescriptorSetLayoutCreateInfo dci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dci.bindingCount = 1;
        dci.pBindings = &sb;
        VK_CHECK(vkCreateDescriptorSetLayout(device_, &dci, nullptr, &dsl_));
    }

    void createDescriptorPool() {
        VkDescriptorPoolSize sz{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, N_TEXTURES};
        VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        ci.poolSizeCount = 1;
        ci.pPoolSizes = &sz;
        ci.maxSets = N_TEXTURES;
        VK_CHECK(vkCreateDescriptorPool(device_, &ci, nullptr, &pool_));
    }

    void allocateDescriptorSets() {
        std::array<VkDescriptorSetLayout, N_TEXTURES> ls;
        ls.fill(dsl_);
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = pool_;
        ai.descriptorSetCount = N_TEXTURES;
        ai.pSetLayouts = ls.data();
        VK_CHECK(vkAllocateDescriptorSets(device_, &ai, texSets_.data()));

        // 所有 set 初始指向占位纹理
        for (int i = 0; i < N_TEXTURES; ++i)
            writeDescriptor(i, placeholderView_);
    }

    void writeDescriptor(int idx, VkImageView view) {
        VkDescriptorImageInfo ii{sampler_, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                               nullptr,
                               texSets_[idx],
                               0,
                               0,
                               1,
                               VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                               &ii,
                               nullptr};
        vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    }

    void createPipeline() {
        VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(QuadPC)};
        VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        lci.setLayoutCount = 1;
        lci.pSetLayouts = &dsl_;
        lci.pushConstantRangeCount = 1;
        lci.pPushConstantRanges = &pcr;
        VK_CHECK(vkCreatePipelineLayout(device_, &lci, nullptr, &layout_));

        auto vert = createShaderModuleFromFile(device_, "async_tex.vert.spv");
        auto frag = createShaderModuleFromFile(device_, "async_tex.frag.spv");
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
        gci.pDepthStencilState = &nods;
        gci.pColorBlendState = &cbs;
        gci.pDynamicState = &dyns;
        gci.layout = layout_;
        gci.renderPass = renderPass_;
        VK_CHECK(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gci, nullptr, &pipeline_));
        vkDestroyShaderModule(device_, vert, nullptr);
        vkDestroyShaderModule(device_, frag, nullptr);
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
            processUploads();
            drawFrame();
            interactive_.updateWindowTitle();
        }
        // 等待所有工作线程结束
        for (auto& t : workers_)
            if (t.joinable())
                t.join();
        vkDeviceWaitIdle(device_);
    }

    void buildUi() {
        interactive_.buildDebugPanel("第56章：异步纹理加载");
        ImGui::Separator();
        ImGui::TextUnformatted("纹理加载状态");
        for (int i = 0; i < N_TEXTURES; ++i) {
            const char* state = "等待";
            if (textures_[i].state == TexState::Loading)
                state = "加载中...";
            if (textures_[i].state == TexState::Uploading)
                state = "GPU 上传中";
            if (textures_[i].state == TexState::Ready)
                state = "✅ 就绪";
            ImGui::Text("[%d] %s  %s", i, SCHEMES[i].name, state);
            if (textures_[i].state == TexState::Loading) {
                ImGui::SameLine();
                ImGui::ProgressBar(textures_[i].progress, ImVec2(-1, 0));
            }
            if (textures_[i].state == TexState::Ready) {
                ImGui::SameLine();
                ImGui::Text("(%.1f ms)", textures_[i].loadMs);
            }
        }
        ImGui::Separator();
        if (!loadingStarted_) {
            if (ImGui::Button("开始异步加载全部纹理")) {
                loadingStarted_ = true;
                for (int i = 0; i < N_TEXTURES; ++i)
                    startAsyncLoad(i);
            }
        } else {
            if (ImGui::Button("重置（清除所有纹理）")) {
                for (auto& t : workers_)
                    if (t.joinable())
                        t.join();
                workers_.clear();
                resetTextures();
                loadingStarted_ = false;
            }
        }
    }

    void startAsyncLoad(int idx) {
        textures_[idx].state = TexState::Loading;
        textures_[idx].progress = 0.0f;
        auto cs = SCHEMES[idx];
        workers_.push_back(std::thread([this, idx, cs]() {
            auto t0 = std::chrono::high_resolution_clock::now();
            // 模拟耗时工作（分批生成，更新进度）
            const int ROWS = TEX_SIZE;
            std::vector<uint8_t> px(TEX_SIZE * TEX_SIZE * 4);
            for (uint32_t y = 0; y < (uint32_t)ROWS; ++y) {
                // 模拟每行需要一点 CPU 时间
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                for (uint32_t x = 0; x < TEX_SIZE; ++x) {
                    uint32_t off = (y * TEX_SIZE + x) * 4;
                    // 生成不同图案
                    float fx = float(x) / float(TEX_SIZE);
                    float fy = float(y) / float(TEX_SIZE);
                    float pat = 0;
                    switch (idx % 4) {
                    case 0:
                        pat = ((x / 16 + y / 16) % 2) ? 1.0f : 0.5f;
                        break; // 棋盘
                    case 1:
                        pat = fx;
                        break; // 水平渐变
                    case 2:
                        pat = 0.5f + 0.5f * sinf((fx + fy) * 12.0f);
                        break; // 噪声
                    case 3:
                        pat = (sinf(fx * 6.28f * 4) + cosf(fy * 6.28f * 4) + 2) * 0.25f;
                        break; // 波纹
                    }
                    px[off + 0] = static_cast<uint8_t>(cs.r * pat);
                    px[off + 1] = static_cast<uint8_t>(cs.g * pat);
                    px[off + 2] = static_cast<uint8_t>(cs.b * pat);
                    px[off + 3] = 255;
                }
                textures_[idx].progress = float(y + 1) / ROWS;
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            // 推入完成队列
            std::lock_guard<std::mutex> lock(uploadMutex_);
            uploadQueue_.push_back({idx, std::move(px), ms});
        }));
    }

    void processUploads() {
        std::vector<UploadRequest> pending;
        {
            std::lock_guard<std::mutex> lock(uploadMutex_);
            pending.swap(uploadQueue_);
        }
        for (auto& req : pending) {
            int i = req.index;
            textures_[i].state = TexState::Uploading;
            textures_[i].loadMs = req.loadMs;
            // 上传到 GPU（同步，但在主循环中一次一个，不影响太多帧时）
            uploadTextureImmediate(req.pixels, textures_[i].image, textures_[i].memory, textures_[i].view);
            textures_[i].state = TexState::Ready;
            textures_[i].progress = 1.0f;
            // 更新描述符集指向真实纹理
            writeDescriptor(i, textures_[i].view);
        }
    }

    void resetTextures() {
        vkDeviceWaitIdle(device_);
        for (int i = 0; i < N_TEXTURES; ++i) {
            auto& t = textures_[i];
            if (t.view)
                vkDestroyImageView(device_, t.view, nullptr);
            if (t.image)
                vkDestroyImage(device_, t.image, nullptr);
            if (t.memory)
                vkFreeMemory(device_, t.memory, nullptr);
            t = {};
            writeDescriptor(i, placeholderView_);
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

        VkCommandBuffer cmd = cmds_[frame_];
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
        interactive_.beginGpuSection(cmd, frame_);

        VkClearValue clear{};
        clear.color.float32[0] = 0.1f;
        clear.color.float32[1] = 0.1f;
        clear.color.float32[2] = 0.1f;
        clear.color.float32[3] = 1.0f;
        VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rbi.renderPass = renderPass_;
        rbi.framebuffer = swapFBs_[imgIdx];
        rbi.renderArea.extent = extent_;
        rbi.clearValueCount = 1;
        rbi.pClearValues = &clear;
        vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport vp{0, 0, float(extent_.width), float(extent_.height), 0, 1};
        VkRect2D sc{{0, 0}, extent_};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

        // 2×2 网格显示 4 张纹理
        for (int i = 0; i < N_TEXTURES; ++i) {
            float col = float(i % 2), row = float(i / 2);
            QuadPC pc{};
            pc.pos = {col * 2.0f - 1.0f + 0.05f, row * 2.0f - 1.0f + 0.05f};
            pc.size = {2.0f - 0.1f, 2.0f - 0.1f};
            pc.loadProgress = textures_[i].progress;
            vkCmdPushConstants(
                cmd, layout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(QuadPC), &pc);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1, &texSets_[i], 0, nullptr);
            vkCmdDraw(cmd, 6, 1, 0, 0);
        }

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
        for (auto v : swapViews_)
            vkDestroyImageView(device_, v, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        createSwapchain();
        createImageViews();
        createFramebuffers();
        interactive_.onSwapchainRecreated(renderPass_, swapFormat_, static_cast<uint32_t>(swapImages_.size()));
    }

    void cleanup() {
        for (auto& t : workers_)
            if (t.joinable())
                t.join();
        vkDestroyDescriptorPool(device_, pool_, nullptr);
        vkDestroyDescriptorSetLayout(device_, dsl_, nullptr);
        vkDestroyPipeline(device_, pipeline_, nullptr);
        vkDestroyPipelineLayout(device_, layout_, nullptr);
        vkDestroyRenderPass(device_, renderPass_, nullptr);
        vkDestroySampler(device_, sampler_, nullptr);
        vkDestroyImageView(device_, placeholderView_, nullptr);
        vkDestroyImage(device_, placeholderImage_, nullptr);
        vkFreeMemory(device_, placeholderMem_, nullptr);
        for (int i = 0; i < N_TEXTURES; ++i) {
            auto& t = textures_[i];
            if (t.view)
                vkDestroyImageView(device_, t.view, nullptr);
            if (t.image)
                vkDestroyImage(device_, t.image, nullptr);
            if (t.memory)
                vkFreeMemory(device_, t.memory, nullptr);
        }
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroySemaphore(device_, imgAvail_[i], nullptr);
            vkDestroySemaphore(device_, renderDone_[i], nullptr);
            vkDestroyFence(device_, inFlight_[i], nullptr);
        }
        for (auto fb : swapFBs_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        for (auto v : swapViews_)
            vkDestroyImageView(device_, v, nullptr);
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        vkDestroyCommandPool(device_, cmdPool_, nullptr);
        vkDestroyCommandPool(device_, uploadPool_, nullptr);
        interactive_.shutdown(device_);
        vkDestroyDevice(device_, nullptr);
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        vkDestroyInstance(instance_, nullptr);
        glfwDestroyWindow(window_);
        glfwTerminate();
    }
};

constexpr Ch56App::ColorScheme Ch56App::SCHEMES[N_TEXTURES];

int main() {
    std::cout << "══════════════════════════════════════════════════\n";
    std::cout << " 第56章：异步纹理加载\n";
    std::cout << "══════════════════════════════════════════════════\n\n";
    std::cout << "原理：工作线程生成像素数据 → 主线程通过 Staging Buffer 上传 GPU\n";
    std::cout << "点击 ImGui「开始异步加载」观察加载进度和耗时\n\n";
    try {
        Ch56App app;
        app.run();
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
