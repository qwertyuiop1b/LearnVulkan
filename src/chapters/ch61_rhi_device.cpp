/**
 * @file ch61_rhi_device.cpp
 * @brief 第61章：设备抽象层（RHIDevice）
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * 【为什么需要设备抽象？】
 *
 *  ch01–ch03 的传统做法：分散在 3 章、约 450 行代码，
 *  createInstance / pickPhysicalDevice / createLogicalDevice
 *  每次写新章节都要复制一遍。
 *
 *  RHIDevice 封装后（本章演示的 API）：
 *    DeviceCreateInfo ci{};
 *    ci.windowHandle     = window;
 *    ci.requiredFeatures = { DeviceFeature::SamplerAnisotropy };
 *    RHIDevice dev;
 *    dev.init(ci);        // ← 5 行完成 ch01~ch03 所有工作
 *
 * 【RHIDevice 的核心设计】
 *   ① 工厂方法 init()：自动选择最佳设备（独显优先）
 *   ② 能力查询：supportsFeature(), isDiscreteGpu(), totalVideoMemoryBytes()
 *   ③ 队列封装：graphicsQueue / computeQueue / transferQueue（含 familyIndex）
 *   ④ 便利工具：findMemoryType / depthFormat / beginOneShot / endOneShot
 *
 * 【本章 Demo】
 *   ImGui 面板显示当前 GPU 的完整信息，展示 RHIDevice 的查询 API
 * ═══════════════════════════════════════════════════════════════════════════
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan_tutorial/utils.hpp>
#include <vulkan_tutorial/vk_helpers.hpp>
#include <vulkan_tutorial/interactive_chapter.hpp>

#include <imgui.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace vulkan_tutorial;

constexpr uint32_t WIDTH = 900;
constexpr uint32_t HEIGHT = 700;
constexpr int MAX_FRAMES = 2;

// ─── 模拟 RHIDevice 的查询结果（展示封装后可获取的信息）─────────────────────

struct GpuCapabilities {
    // 来自 VkPhysicalDeviceProperties
    std::string name;
    std::string vendorName;
    uint32_t apiVersion = 0;
    uint32_t driverVersion = 0;
    bool isDiscrete = false;
    float timestampPeriodNs = 0.0f;
    uint32_t maxColorAttachments = 0;
    uint32_t maxDescriptorSets = 0;
    uint32_t maxPushConstantSize = 0;
    // 来自 VkPhysicalDeviceMemoryProperties
    uint64_t deviceLocalBytes = 0;
    uint64_t hostVisibleBytes = 0;
    uint32_t heapCount = 0;
    // 来自 VkPhysicalDeviceFeatures
    bool samplerAnisotropy = false;
    bool geometryShader = false;
    bool tessellationShader = false;
    bool wideLines = false;
    bool fillModeNonSolid = false;
    bool multiDrawIndirect = false;
    bool drawIndirectFirstInst = false;
    bool shaderClipDistance = false;
    // 队列信息（来自 QueueFamilyProperties）
    uint32_t graphicsQueueFamily = UINT32_MAX;
    uint32_t computeQueueFamily = UINT32_MAX;
    uint32_t transferQueueFamily = UINT32_MAX;
    bool graphicsSupportsTimestamp = false;
};

static GpuCapabilities queryCapabilities(VkPhysicalDevice physDev, VkSurfaceKHR surface) {
    GpuCapabilities caps;

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physDev, &props);
    caps.name = props.deviceName;
    caps.apiVersion = props.apiVersion;
    caps.driverVersion = props.driverVersion;
    caps.isDiscrete = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    caps.timestampPeriodNs = props.limits.timestampPeriod;
    caps.maxColorAttachments = props.limits.maxColorAttachments;
    caps.maxDescriptorSets = props.limits.maxBoundDescriptorSets;
    caps.maxPushConstantSize = props.limits.maxPushConstantsSize;
    caps.graphicsSupportsTimestamp = props.limits.timestampComputeAndGraphics != 0;

    // Vendor mapping
    switch (props.vendorID) {
    case 0x10DE:
        caps.vendorName = "NVIDIA";
        break;
    case 0x1002:
        caps.vendorName = "AMD";
        break;
    case 0x8086:
        caps.vendorName = "Intel";
        break;
    case 0x106B:
        caps.vendorName = "Apple";
        break;
    default:
        caps.vendorName = "Unknown";
        break;
    }

    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(physDev, &memProps);
    caps.heapCount = memProps.memoryHeapCount;
    for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i) {
        if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            caps.deviceLocalBytes = std::max(caps.deviceLocalBytes, memProps.memoryHeaps[i].size);
        else
            caps.hostVisibleBytes += memProps.memoryHeaps[i].size;
    }

    VkPhysicalDeviceFeatures feats{};
    vkGetPhysicalDeviceFeatures(physDev, &feats);
    caps.samplerAnisotropy = feats.samplerAnisotropy;
    caps.geometryShader = feats.geometryShader;
    caps.tessellationShader = feats.tessellationShader;
    caps.wideLines = feats.wideLines;
    caps.fillModeNonSolid = feats.fillModeNonSolid;
    caps.multiDrawIndirect = feats.multiDrawIndirect;
    caps.drawIndirectFirstInst = feats.drawIndirectFirstInstance;
    caps.shaderClipDistance = feats.shaderClipDistance;

    // Queue families
    uint32_t qCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physDev, &qCount, nullptr);
    std::vector<VkQueueFamilyProperties> qProps(qCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physDev, &qCount, qProps.data());
    for (uint32_t i = 0; i < qCount; ++i) {
        if ((qProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && caps.graphicsQueueFamily == UINT32_MAX)
            caps.graphicsQueueFamily = i;
        if ((qProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
            caps.computeQueueFamily == UINT32_MAX)
            caps.computeQueueFamily = i;
        if ((qProps[i].queueFlags & VK_QUEUE_TRANSFER_BIT) && !(qProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
            !(qProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && caps.transferQueueFamily == UINT32_MAX)
            caps.transferQueueFamily = i;
    }
    if (caps.computeQueueFamily == UINT32_MAX)
        caps.computeQueueFamily = caps.graphicsQueueFamily;
    if (caps.transferQueueFamily == UINT32_MAX)
        caps.transferQueueFamily = caps.graphicsQueueFamily;

    return caps;
}

// ─── App ─────────────────────────────────────────────────────────────────────

class Ch61App {
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
    QueueFamilyIndices qIdx_{};
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    std::vector<VkImage> swapImages_;
    std::vector<VkImageView> swapViews_;
    std::vector<VkFramebuffer> swapFBs_;
    VkFormat swapFmt_ = VK_FORMAT_UNDEFINED;
    VkExtent2D extent_{};
    VkCommandPool cmdPool_ = VK_NULL_HANDLE;
    VkRenderPass renderPass_ = VK_NULL_HANDLE;

    std::vector<VkCommandBuffer> cmds_;
    std::vector<VkSemaphore> imgAvail_, renderDone_;
    std::vector<VkFence> inFlight_;
    uint32_t frame_ = 0;
    bool resized_ = false;

    InteractiveChapterTools interactive_;
    GpuCapabilities caps_;

    // ImGui 当前选中的标签页
    int tabIdx_ = 0;

    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window_ = glfwCreateWindow(WIDTH, HEIGHT, "第61章：RHIDevice 设备抽象", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
            reinterpret_cast<Ch61App*>(glfwGetWindowUserPointer(w))->resized_ = true;
        });
        interactive_.attachInput(window_);
    }

    void initVulkan() {
        createInstance(instance_);
        VK_CHECK(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_));
        pickPhysicalDevice(instance_, surface_, physDev_);
        createLogicalDevice(physDev_, surface_, device_, gQueue_, pQueue_, qIdx_);
        caps_ = queryCapabilities(physDev_, surface_);

        createSwapchain();
        createImageViews();
        createCommandPool();
        createRenderPass();
        createFramebuffers();
        createCommandBuffers();
        createSyncObjects();

        InteractiveInitInfo ii{};
        ii.window = window_;
        ii.instance = instance_;
        ii.physicalDevice = physDev_;
        ii.device = device_;
        ii.graphicsQueue = gQueue_;
        ii.queueFamily = qIdx_.graphicsFamily.value();
        ii.renderPass = renderPass_;
        ii.swapchainFormat = swapFmt_;
        ii.imageCount = static_cast<uint32_t>(swapImages_.size());
        ii.maxFramesInFlight = MAX_FRAMES;
        interactive_.initVulkan(ii);
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

    void createCommandPool() {
        VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = qIdx_.graphicsFamily.value();
        VK_CHECK(vkCreateCommandPool(device_, &ci, nullptr, &cmdPool_));
    }

    void createRenderPass() {
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

    void buildUi() {
        interactive_.buildDebugPanel("第61章：RHIDevice 设备抽象");
        ImGui::Separator();

        // 大型信息面板
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(880, 660), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("RHIDevice — 设备抽象层", nullptr, ImGuiWindowFlags_NoCollapse)) {

            if (ImGui::BeginTabBar("DeviceTabs")) {

                // ── 设备概览 ──────────────────────────────────────────────
                if (ImGui::BeginTabItem("概览")) {
                    ImGui::TextColored(ImVec4(1, 0.85f, 0.2f, 1), "GPU 基本信息");
                    ImGui::Separator();
                    ImGui::Text("设备名称 : %s", caps_.name.c_str());
                    ImGui::Text("GPU 厂商 : %s", caps_.vendorName.c_str());
                    ImGui::Text("设备类型 : %s",
                                caps_.isDiscrete ? "独立显卡 (Discrete GPU)" : "集成显卡 (Integrated GPU)");
                    ImGui::Text("API 版本 : %d.%d.%d",
                                VK_API_VERSION_MAJOR(caps_.apiVersion),
                                VK_API_VERSION_MINOR(caps_.apiVersion),
                                VK_API_VERSION_PATCH(caps_.apiVersion));
                    ImGui::Spacing();
                    ImGui::TextColored(ImVec4(0.5f, 1, 0.5f, 1), "显存");
                    ImGui::Text("设备本地内存 : %.2f GB", caps_.deviceLocalBytes / (1024.0 * 1024.0 * 1024.0));
                    ImGui::Text("系统共享内存 : %.2f GB", caps_.hostVisibleBytes / (1024.0 * 1024.0 * 1024.0));
                    ImGui::Text("内存堆数量   : %u", caps_.heapCount);
                    ImGui::EndTabItem();
                }

                // ── 特性支持 ──────────────────────────────────────────────
                if (ImGui::BeginTabItem("特性支持")) {
                    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1, 1), "VkPhysicalDeviceFeatures 查询结果");
                    ImGui::Separator();
                    auto feat = [](const char* name, bool supported) {
                        if (supported)
                            ImGui::TextColored(ImVec4(0.2f, 1, 0.3f, 1), "  ✅ %s", name);
                        else
                            ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "  ❌ %s", name);
                    };
                    feat("samplerAnisotropy（各向异性过滤）", caps_.samplerAnisotropy);
                    feat("geometryShader（几何着色器）", caps_.geometryShader);
                    feat("tessellationShader（曲面细分）", caps_.tessellationShader);
                    feat("wideLines（宽线段）", caps_.wideLines);
                    feat("fillModeNonSolid（线框/点模式）", caps_.fillModeNonSolid);
                    feat("multiDrawIndirect（多重间接绘制）", caps_.multiDrawIndirect);
                    feat("drawIndirectFirstInstance", caps_.drawIndirectFirstInst);
                    feat("shaderClipDistance（裁剪距离）", caps_.shaderClipDistance);
                    ImGui::Spacing();
                    ImGui::Text("Timestamp 精度 : %.2f ns / tick", caps_.timestampPeriodNs);
                    feat("timestampComputeAndGraphics", caps_.graphicsSupportsTimestamp);
                    ImGui::EndTabItem();
                }

                // ── 限制参数 ──────────────────────────────────────────────
                if (ImGui::BeginTabItem("资源限制")) {
                    ImGui::TextColored(ImVec4(1, 0.6f, 0.2f, 1), "VkPhysicalDeviceLimits 关键参数");
                    ImGui::Separator();
                    ImGui::Text("maxBoundDescriptorSets : %u", caps_.maxDescriptorSets);
                    ImGui::Text("maxColorAttachments    : %u", caps_.maxColorAttachments);
                    ImGui::Text("maxPushConstantsSize   : %u bytes", caps_.maxPushConstantSize);
                    ImGui::EndTabItem();
                }

                // ── 队列族 ────────────────────────────────────────────────
                if (ImGui::BeginTabItem("队列族")) {
                    ImGui::TextColored(ImVec4(0.8f, 0.5f, 1, 1), "Queue Family Indices");
                    ImGui::Separator();
                    ImGui::Text("Graphics Queue Family  : %u", caps_.graphicsQueueFamily);
                    ImGui::Text("Compute  Queue Family  : %u", caps_.computeQueueFamily);
                    ImGui::Text("Transfer Queue Family  : %u", caps_.transferQueueFamily);
                    bool sep = caps_.graphicsQueueFamily != caps_.computeQueueFamily;
                    ImGui::TextWrapped(sep ? "独立 Compute Queue → 可以 Compute 和 Graphics 并行提交"
                                           : "共用 Queue → Compute 和 Graphics 共享同一个队列族");
                    ImGui::EndTabItem();
                }

                // ── 封装对比 ──────────────────────────────────────────────
                if (ImGui::BeginTabItem("封装对比")) {
                    ImGui::TextColored(ImVec4(1, 1, 0.3f, 1), "代码量对比：传统写法 vs RHIDevice");
                    ImGui::Separator();
                    ImGui::TextColored(ImVec4(1, 0.5f, 0.5f, 1), "传统写法（ch01–ch03）：");
                    ImGui::TextWrapped("  ch01: createInstance()          ~80 行\n"
                                       "  ch02: pickPhysicalDevice()      ~100 行\n"
                                       "  ch03: createLogicalDevice()     ~80 行\n"
                                       "  各章节队列获取、错误处理、debug callback ...\n"
                                       "  每次新章节都要复制 ~250 行代码\n");
                    ImGui::Spacing();
                    ImGui::TextColored(ImVec4(0.3f, 1, 0.5f, 1), "RHIDevice 封装：");
                    ImGui::TextWrapped("  #include <vulkan_tutorial/engine/rhi_device.hpp>\n\n"
                                       "  DeviceCreateInfo ci{};\n"
                                       "  ci.windowHandle     = window;     // GLFW window\n"
                                       "  ci.preferDiscreteGpu = true;\n"
                                       "  ci.requiredFeatures = {\n"
                                       "      DeviceFeature::SamplerAnisotropy,\n"
                                       "  };\n"
                                       "  RHIDevice dev;\n"
                                       "  dev.init(ci);    // ← 5 行完成所有工作\n\n"
                                       "  // 使用：\n"
                                       "  dev.device()           // VkDevice\n"
                                       "  dev.graphicsQueue()    // QueueInfo\n"
                                       "  dev.supportsFeature(DeviceFeature::RayTracing)\n"
                                       "  dev.totalVideoMemoryBytes()\n"
                                       "  dev.beginOneShot()     // one-shot CB\n");
                    ImGui::EndTabItem();
                }

                ImGui::EndTabBar();
            }
        }
        ImGui::End();
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

        VkCommandBuffer cmd = cmds_[frame_];
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
        interactive_.beginGpuSection(cmd, frame_);

        VkClearValue cv{};
        cv.color.float32[0] = 0.05f;
        cv.color.float32[1] = 0.07f;
        cv.color.float32[2] = 0.12f;
        cv.color.float32[3] = 1.0f;
        VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rbi.renderPass = renderPass_;
        rbi.framebuffer = swapFBs_[idx];
        rbi.renderArea.extent = extent_;
        rbi.clearValueCount = 1;
        rbi.pClearValues = &cv;
        vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
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
        pi.pImageIndices = &idx;
        r = vkQueuePresentKHR(pQueue_, &pi);
        if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR || resized_) {
            resized_ = false;
            recreate();
        }
        interactive_.endFrame(frame_);
        frame_ = (frame_ + 1) % MAX_FRAMES;
    }

    void recreate() {
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
        interactive_.onSwapchainRecreated(renderPass_, swapFmt_, static_cast<uint32_t>(swapImages_.size()));
    }

    void cleanup() {
        vkDestroyRenderPass(device_, renderPass_, nullptr);
        for (auto fb : swapFBs_)
            vkDestroyFramebuffer(device_, fb, nullptr);
        for (int i = 0; i < MAX_FRAMES; ++i) {
            vkDestroySemaphore(device_, imgAvail_[i], nullptr);
            vkDestroySemaphore(device_, renderDone_[i], nullptr);
            vkDestroyFence(device_, inFlight_[i], nullptr);
        }
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
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << " 第61章：设备抽象层（RHIDevice）\n";
    std::cout << " 引擎封装系列 — ch61/10\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    std::cout << "ImGui 面板展示 RHIDevice 的查询 API：\n";
    std::cout << "  概览 / 特性支持 / 资源限制 / 队列族 / 封装对比\n\n";
    try {
        Ch61App app;
        app.run();
        std::cout << "✅ 正常退出\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << "\n";
        return 1;
    }
}
